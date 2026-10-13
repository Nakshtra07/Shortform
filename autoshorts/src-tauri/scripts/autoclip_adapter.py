"""AutoClip Studio Data Translation Adapter for AutoShorts 11.0 (Presentation Mode 2).

Translates AutoShorts candidate data, selection boundaries, and transcript words
into AutoClip Draft and Scene models, builds subtitle clause SRT cues, and executes
AutoClip's framing and rendering pipeline offline with zero external network or LLM calls.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any

# Ensure UTF-8 streams on Windows
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")

# Ensure autoclip_studio and script directory are importable
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

STUDIO_DIR = SCRIPT_DIR / "autoclip_studio"
if str(STUDIO_DIR) not in sys.path:
    sys.path.insert(0, str(STUDIO_DIR))

# AutoClip studio imports
from autoclip_studio import (
    Draft,
    Scene,
    CropPoint,
    Packaging,
    PackagingCue,
    platform_strategy,
    auto_frame,
    render_draft,
    TITLE_STYLES,
)
from autoclip_studio.publish_export import _probe, to_srt_time
from autoclip_studio.project_cancellation import project_directory

logger = logging.getLogger("autoclip_adapter")

PUNCTUATION_CHARS = set(".,!?。！？;；，")


def is_cjk(ch: str) -> bool:
    """Check if character is within common CJK unicode blocks."""
    if not ch:
        return False
    cp = ord(ch)
    return (
        0x4E00 <= cp <= 0x9FFF
        or 0x3400 <= cp <= 0x4DBF
        or 0x20000 <= cp <= 0x2A6DF
        or 0x3000 <= cp <= 0x303F
        or 0xFF00 <= cp <= 0xFFEF
    )


def join_words_text(words: list[dict[str, Any]]) -> str:
    """Join words naturally, inserting spaces for Latin and concatenating for CJK."""
    parts: list[str] = []
    for w in words:
        txt = str(w.get("text", "")).strip()
        if not txt:
            continue
        if not parts:
            parts.append(txt)
            continue
        prev = parts[-1]
        if is_cjk(prev[-1]) and is_cjk(txt[0]):
            parts.append(txt)
        else:
            parts.append(" " + txt)
    return "".join(parts).strip()


def has_clause_punctuation(text: str) -> bool:
    """Check if text contains any clause-splitting punctuation mark."""
    return any(ch in PUNCTUATION_CHARS for ch in text)


def _make_cue(clause_words: list[dict[str, Any]]) -> dict[str, Any]:
    st = float(clause_words[0]["start"])
    en = float(clause_words[-1]["end"])
    if en <= st:
        en = st + 0.1
    return {
        "start": round(st, 3),
        "end": round(en, 3),
        "text": join_words_text(clause_words),
        "speaker": clause_words[0].get("speaker"),
    }


def group_words_into_cues(
    words: list[dict[str, Any]],
    scene_start: float | None = None,
    scene_end: float | None = None,
) -> list[dict[str, Any]]:
    """Group transcript words into natural clauses.

    Splits on punctuation [.,!?。！？;；], speaker transitions, silence gaps >= 0.4s,
    or maximum 8 words per clause.
    """
    valid_words: list[dict[str, Any]] = []
    for w in words:
        if not isinstance(w, dict):
            continue
        txt = str(w.get("text", "")).strip()
        if not txt:
            continue
        try:
            st = float(w.get("start", 0.0))
            en = float(w.get("end", st + 0.1))
        except (ValueError, TypeError):
            continue
        if en <= st:
            en = st + 0.05

        if scene_start is not None and scene_end is not None:
            if en <= scene_start or st >= scene_end:
                continue

        valid_words.append({
            "text": txt,
            "start": st,
            "end": en,
            "speaker": w.get("speaker"),
        })

    valid_words.sort(key=lambda x: x["start"])
    if not valid_words:
        return []

    cues: list[dict[str, Any]] = []
    current_clause: list[dict[str, Any]] = []

    for word in valid_words:
        if not current_clause:
            current_clause.append(word)
            if has_clause_punctuation(word["text"]):
                cues.append(_make_cue(current_clause))
                current_clause = []
            continue

        prev_word = current_clause[-1]

        # Split conditions:
        gap = word["start"] - prev_word["end"]
        silence_split = gap >= 0.4
        speaker_split = (
            word.get("speaker") is not None
            and prev_word.get("speaker") is not None
            and word.get("speaker") != prev_word.get("speaker")
        )
        max_words_split = len(current_clause) >= 8

        if silence_split or speaker_split or max_words_split:
            cues.append(_make_cue(current_clause))
            current_clause = [word]
        else:
            current_clause.append(word)

        if has_clause_punctuation(word["text"]):
            cues.append(_make_cue(current_clause))
            current_clause = []

    if current_clause:
        cues.append(_make_cue(current_clause))

    return cues


def generate_srt_string(cues: list[dict[str, Any]]) -> str:
    """Format cues into standard SRT string."""
    lines: list[str] = []
    for idx, cue in enumerate(cues, 1):
        s_time = to_srt_time(cue["start"])
        e_time = to_srt_time(cue["end"])
        txt = cue["text"].strip()
        lines.append(f"{idx}\n{s_time} --> {e_time}\n{txt}\n")
    return "\n".join(lines)


def setup_data_directory(project_id: str) -> Path:
    """Ensure data directory and project directories exist and are configured."""
    if "AUTOCLIP_DATA_DIR" not in os.environ:
        if os.getenv("AUTOSHORTS_DATA_DIR"):
            os.environ["AUTOCLIP_DATA_DIR"] = os.environ["AUTOSHORTS_DATA_DIR"]
        elif os.getenv("LOCALAPPDATA"):
            os.environ["AUTOCLIP_DATA_DIR"] = str((Path(os.environ["LOCALAPPDATA"]) / "autoshorts" / "autoclip_data").resolve())
        else:
            import tempfile
            os.environ["AUTOCLIP_DATA_DIR"] = str((Path(tempfile.gettempdir()) / "autoshorts" / "autoclip_data").resolve())

    proj_dir = project_directory(project_id)
    proj_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = proj_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    return proj_dir


def write_srt_file(project_id: str, srt_content: str) -> Path:
    """Write input.srt to project metadata directory."""
    proj_dir = setup_data_directory(project_id)
    srt_path = proj_dir / "metadata" / "input.srt"
    srt_path.write_text(srt_content, encoding="utf-8")
    return srt_path


def validate_payload(payload: dict[str, Any]) -> None:
    """Validate structure and required values of input JSON payload."""
    if not isinstance(payload, dict):
        raise ValueError("Payload must be a JSON object")

    project_id = payload.get("project_id")
    if not project_id or not isinstance(project_id, str):
        raise ValueError("Missing or invalid 'project_id' in payload")

    video_path_str = payload.get("video_path")
    if not video_path_str or not isinstance(video_path_str, str):
        raise ValueError("Missing or invalid 'video_path' in payload")
    video_path = Path(video_path_str).resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Source video file not found: {video_path}")

    output_path = payload.get("output_path")
    if not output_path or not isinstance(output_path, str):
        raise ValueError("Missing or invalid 'output_path' in payload")

    cand = payload.get("candidate")
    if not cand or not isinstance(cand, dict):
        raise ValueError("Missing or invalid 'candidate' in payload")
    if "id" not in cand:
        raise ValueError("Candidate object missing required field 'id'")
    if "start_sec" not in cand:
        raise ValueError("Candidate object missing required field 'start_sec'")
    if "end_sec" not in cand:
        raise ValueError("Candidate object missing required field 'end_sec'")

    start_sec = float(cand["start_sec"])
    end_sec = float(cand["end_sec"])
    if start_sec < 0:
        raise ValueError(f"Candidate start_sec cannot be negative: {start_sec}")
    if end_sec <= start_sec:
        raise ValueError(
            f"Candidate end_sec ({end_sec}) must be greater than start_sec ({start_sec})"
        )
    if end_sec - start_sec < 0.1:
        raise ValueError(
            f"Candidate duration too short ({end_sec - start_sec:.3f}s), minimum 0.1s required"
        )


def build_draft_and_scene(
    payload: dict[str, Any], cues: list[dict[str, Any]]
) -> tuple[Draft, Scene, Any]:
    """Translate payload into AutoClip Draft, Scene, and PlatformStrategy."""
    cand = payload["candidate"]
    config = payload.get("presentation_config") or {}

    strategy_id = config.get("platform_strategy") or "douyin"
    strat = platform_strategy(strategy_id)

    subtitle_style = config.get("subtitle_style") or strat.subtitle_style
    title_style = config.get("title_style") or strat.title_style
    title_motion = (
        config.get("title_motion")
        if config.get("title_motion") is not None
        else strat.title_motion
    )
    template_name = config.get("template") or strat.template

    start_sec = float(cand["start_sec"])
    end_sec = float(cand["end_sec"])
    if end_sec - start_sec < 0.1:
        raise ValueError(
            f"Candidate interval too short ({end_sec - start_sec:.3f}s), minimum 0.1s required"
        )

    cand_id = re.sub(r"[^a-zA-Z0-9_-]", "_", str(cand["id"]))
    if not cand_id:
        cand_id = "candidate_01"
    scene_id = f"scene_{cand_id}"

    scene = Scene(
        id=scene_id,
        start=start_sec,
        end=end_sec,
        label="片段",
        crop_x=0.5,
    )

    title_version = 6 if title_style in TITLE_STYLES else 1

    packaging_obj = None
    if template_name in ("interview_zh", "podcast_en"):
        hook_text = str(cand.get("hook") or "")
        packaging_cues = [
            PackagingCue(start=c["start"], end=c["end"], text=c["text"][:600])
            for c in cues
            if c["end"] > start_sec and c["start"] < end_sec
        ][:600]
        packaging_obj = Packaging(
            template=template_name,
            audience_language=strat.audience_language,
            title_lines=[hook_text[:40]] if hook_text else [],
            cues=packaging_cues,
            fallback=True,
        )

    title_scale = float(config.get("title_scale", 1.0))
    title_scale = max(0.75, min(1.2, title_scale))

    title_y = float(config.get("title_y", 0.12))
    title_y = max(0.06, min(0.70, title_y))

    hook_str = str(cand.get("hook") or "")[:120]
    title_str = (cand.get("hook") or "Untitled")[:200].strip()
    if not title_str:
        title_str = "Untitled"

    layout_val = config.get("layout") or strat.layout
    if layout_val not in ("fit", "crop", "blur", "window"):
        layout_val = "fit"

    draft = Draft(
        id=cand_id,
        title=title_str,
        hook=hook_str,
        scenes=[scene],
        language="source",  # Guarantees 100% offline local execution by bypassing LLM translation
        aspect=strat.aspect,
        layout=layout_val,
        crop_x=0.5,
        title_style=title_style,
        title_template_version=title_version,
        title_motion=bool(title_motion),
        title_scale=title_scale,
        title_y=title_y,
        subtitles=True,
        subtitle_style=subtitle_style,
        original_audio=True,
        packaging=packaging_obj,
    )

    return draft, scene, strat


def run_adapter(payload: dict[str, Any]) -> dict[str, Any]:
    """Execute complete data translation and rendering pipeline for AutoClip presentation mode."""
    # 1. Validate input payload
    validate_payload(payload)

    project_id = payload["project_id"]
    video_path = Path(payload["video_path"]).resolve()
    target_output_path = Path(payload["output_path"]).resolve()
    cand = payload["candidate"]
    words = payload.get("transcript_words") or []

    # 2. Setup project metadata & SRT
    setup_data_directory(project_id)
    scene_start = float(cand["start_sec"])
    scene_end = float(cand["end_sec"])
    cues = group_words_into_cues(words, scene_start=scene_start, scene_end=scene_end)
    srt_content = generate_srt_string(cues)
    write_srt_file(project_id, srt_content)

    # 3. Build Draft & Scene
    draft, scene, strat = build_draft_and_scene(payload, cues)

    # 4. Probe video & execute shot-aligned auto_frame
    probe_info = _probe(video_path)
    src_w = int(probe_info.get("width") or 1920)
    src_h = int(probe_info.get("height") or 1080)
    if src_w <= 0 or src_h <= 0:
        raise ValueError(f"Invalid video dimensions probed from {video_path}: {src_w}x{src_h}")

    framing_res = auto_frame(video_path, draft, src_w, src_h)

    # Assign computed crop points to scene
    for scn, s_info in zip(draft.scenes, framing_res.get("scenes", [])):
        if s_info.get("crop_x") is not None:
            scn.crop_x = float(s_info["crop_x"])
        if s_info.get("crop_track"):
            scn.crop_track = [CropPoint(**p) for p in s_info["crop_track"]]

    # 5. Execute render_draft
    job_id = f"render_{draft.id}"

    def progress_cb(pct: int):
        sys.stderr.write(f"[AutoClip Progress] {pct}%\n")
        sys.stderr.flush()

    render_res = render_draft(
        project_id=project_id,
        video=video_path,
        draft=draft,
        job_id=job_id,
        progress=progress_cb,
        brand_outro=False,
    )

    # 6. Move rendered MP4 to requested output_path
    proj_dir = project_directory(project_id)
    rendered_file = proj_dir / "output" / "studio" / f"{job_id}.mp4"
    if not rendered_file.is_file():
        raise FileNotFoundError(f"Rendered video file was not created: {rendered_file}")

    target_output_path.parent.mkdir(parents=True, exist_ok=True)
    if rendered_file.resolve() != target_output_path.resolve():
        if target_output_path.exists():
            target_output_path.unlink()
        shutil.move(str(rendered_file), str(target_output_path))

    return {
        "status": "success",
        "output_path": str(target_output_path),
        "duration": render_res.get("duration"),
        "width": render_res.get("width"),
        "height": render_res.get("height"),
        "strategy_id": strat.id,
        "framing_info": framing_res,
        "warnings": render_res.get("warnings", []),
        "errors": [],
    }


class AdapterArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ValueError(f"CLI argument error: {message}")


def main(argv: list[str] | None = None) -> int:
    """CLI Entrypoint for autoclip_adapter.py."""
    parser = AdapterArgumentParser(
        description="AutoClip Studio Data Translation Adapter for AutoShorts 11.0"
    )
    parser.add_argument("--payload", type=str, help="Path to input JSON payload file")
    parser.add_argument("--payload-json", type=str, help="Raw JSON string input payload")
    parser.add_argument("--stdin", action="store_true", help="Read input JSON payload from stdin")
    parser.add_argument("--output", type=str, help="Optional output path to write result JSON")

    output_file: str | None = None
    try:
        args = parser.parse_args(argv)
        output_file = args.output

        raw_json: str | None = None
        if args.payload:
            payload_path = Path(args.payload)
            if not payload_path.is_file():
                raise FileNotFoundError(f"Payload file not found: {args.payload}")
            raw_json = payload_path.read_text(encoding="utf-8")
        elif args.payload_json:
            raw_json = args.payload_json
        elif args.stdin:
            raw_json = sys.stdin.read()
        else:
            raise ValueError(
                "No input payload provided. Must specify one of --payload, --payload-json, or --stdin."
            )

        if not raw_json or not raw_json.strip():
            raise ValueError("Input payload JSON is empty")

        payload = json.loads(raw_json)
        result = run_adapter(payload)
        res_json = json.dumps(result, ensure_ascii=False, indent=2)
        if output_file:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            Path(output_file).write_text(res_json, encoding="utf-8")
        print(res_json)
        return 0

    except Exception as exc:
        err_dict = {
            "status": "error",
            "output_path": None,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        res_json = json.dumps(err_dict, ensure_ascii=False, indent=2)
        if output_file:
            try:
                Path(output_file).parent.mkdir(parents=True, exist_ok=True)
                Path(output_file).write_text(res_json, encoding="utf-8")
            except Exception:
                pass
        print(res_json)
        return 1


if __name__ == "__main__":
    sys.exit(main())
