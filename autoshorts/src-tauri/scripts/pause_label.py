#!/usr/bin/env python3
"""
AutoShorts 11.0 Phase 3 — Pause Intelligence Dataset Labeling Pipeline

Extracts acoustic + structural features (23 dimensions) for detected speech gaps
and labels them using human annotations (when provided) or deterministic
weak-supervision heuristics from the Smart Pacing engine.

Supported 6 classes:
  1. breath_pause
  2. waiting_pause
  3. sentence_pause
  4. normal_word_gap
  5. nonspeech_unknown
  6. speaker_transition

CLI usage:
  python scripts/pause_label.py <source_path> <words_json> [options]
  python scripts/pause_label.py --corpus <corpus_dir> [options]
"""

import argparse
from datetime import datetime, timezone
import json
import math
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

# Robust UTF-8 console output on Windows
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    if hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

try:
    import pause_intelligence as pi
except ImportError:
    pi = None

try:
    import breath_detect
except ImportError:
    breath_detect = None

VALID_CLASSES = {
    "breath_pause",
    "waiting_pause",
    "sentence_pause",
    "normal_word_gap",
    "nonspeech_unknown",
    "speaker_transition",
}

FEATURE_ORDER = [
    "gapRmsDb", "preSpeechRmsDb", "postSpeechRmsDb", "hfRatio", "flatness",
    "zcr", "dipDb", "levelDropDb", "durationSec", "speakerChange",
    "endsSentence", "endsClause", "startsFiller", "endsFiller",
    "preSpeechRate", "postSpeechRate", "relPosInClip", "vadSpeechProb",
    "vadSilenceFrac", "evHfRatio", "evAboveFloorDb", "evContrastDb", "evFlatness",
]

MEDIA_EXTENSIONS = {".mp4", ".mov", ".mkv", ".wav", ".m4a", ".mp3", ".webm"}


def log(msg: str) -> None:
    sys.stderr.write(f"[PauseLabel] {msg}\n")
    sys.stderr.flush()


def ensure_data_dir() -> str:
    """Ensure scripts/data directory exists and has a .gitkeep file."""
    data_dir = os.path.join(SCRIPT_DIR, "data")
    os.makedirs(data_dir, exist_ok=True)
    gitkeep = os.path.join(data_dir, ".gitkeep")
    if not os.path.exists(gitkeep):
        try:
            with open(gitkeep, "w", encoding="utf-8") as f:
                f.write("")
        except OSError:
            pass
    return data_dir


def load_words(words_json_path: str) -> Optional[List[Dict[str, Any]]]:
    """Load and normalize word timestamps from JSON transcript.

    Supports:
      - Array of word dicts: [{"text": ..., "start": ..., "end": ...}]
      - Dict with words: {"words": [...]}
      - Dict with segments: {"segments": [{"words": [...]}]}
    """
    if not os.path.isfile(words_json_path):
        log(f"Words JSON not found: {words_json_path}")
        return None

    try:
        with open(words_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        log(f"Failed to parse words JSON ({words_json_path}): {e}")
        return None

    if isinstance(data, list):
        raw_words = data
    elif isinstance(data, dict):
        if "words" in data and isinstance(data["words"], list):
            raw_words = data["words"]
        elif "segments" in data and isinstance(data["segments"], list):
            raw_words = []
            for seg in data["segments"]:
                if isinstance(seg, dict) and "words" in seg and isinstance(seg["words"], list):
                    raw_words.extend(seg["words"])
        else:
            raw_words = []
    else:
        log(f"Unrecognized transcript structure in {words_json_path}")
        return None

    words = []
    for w in raw_words:
        if not isinstance(w, dict):
            continue
        if "start" not in w or "end" not in w:
            continue
        try:
            s = float(w["start"])
            e = float(w["end"])
        except (ValueError, TypeError):
            continue
        if e <= s:
            continue
        w_norm = dict(w)
        w_norm["start"] = s
        w_norm["end"] = e
        if "text" not in w_norm and "word" in w_norm:
            w_norm["text"] = str(w_norm["word"])
        words.append(w_norm)

    words.sort(key=lambda x: x["start"])
    return words


def probe_media_duration(source_path: str) -> Optional[float]:
    """Probe audio/video duration in seconds via ffprobe."""
    try:
        proc = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", source_path
            ],
            capture_output=True, text=True, timeout=30, errors="replace"
        )
        if proc.returncode == 0 and proc.stdout:
            val = float(proc.stdout.strip())
            if val > 0:
                return val
    except Exception:
        pass
    return None


def load_human_labels(labels_path: Optional[str]) -> Optional[Dict[str, Any]]:
    """Load human annotation records from a JSONL file.

    Supports indexing by:
      - (sourceHash, gapIdx)
      - sourceHash + [srcStartSec, srcEndSec]
      - fallback by gapIdx or timestamp
    """
    if not labels_path:
        return None
    if not os.path.isfile(labels_path):
        log(f"Human labels file not found: {labels_path}")
        return None

    by_hash_idx = {}
    by_hash_spans = {}
    by_idx = {}
    by_spans = []

    count = 0
    try:
        with open(labels_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    rec = json.loads(line)
                except Exception as e:
                    log(f"Corrupt JSON on line {line_no} of {labels_path}: {e}")
                    continue

                if not isinstance(rec, dict):
                    continue

                sh = rec.get("sourceHash") or rec.get("source_hash") or ""
                cls = rec.get("class") or rec.get("label") or rec.get("category")
                if not cls:
                    continue

                gap_idx = rec.get("gapIdx") if rec.get("gapIdx") is not None else rec.get("gap_idx", rec.get("idx"))
                try:
                    gap_idx = int(gap_idx) if gap_idx is not None else None
                except (ValueError, TypeError):
                    gap_idx = None

                s_val = rec.get("srcStartSec", rec.get("start", rec.get("start_sec")))
                e_val = rec.get("srcEndSec", rec.get("end", rec.get("end_sec")))
                try:
                    s_sec = float(s_val) if s_val is not None else None
                    e_sec = float(e_val) if e_val is not None else None
                except (ValueError, TypeError):
                    s_sec, e_sec = None, None

                clean_rec = {
                    "sourceHash": sh,
                    "class": cls,
                    "gapIdx": gap_idx,
                    "srcStartSec": s_sec,
                    "srcEndSec": e_sec,
                }

                if sh:
                    if gap_idx is not None:
                        by_hash_idx[(sh, gap_idx)] = clean_rec
                    if s_sec is not None and e_sec is not None:
                        by_hash_spans.setdefault(sh, []).append((s_sec, e_sec, clean_rec))
                else:
                    if gap_idx is not None:
                        by_idx[gap_idx] = clean_rec
                    if s_sec is not None and e_sec is not None:
                        by_spans.append((s_sec, e_sec, clean_rec))

                count += 1
        log(f"Loaded {count} human annotation(s) from {labels_path}")
        return {
            "by_hash_idx": by_hash_idx,
            "by_hash_spans": by_hash_spans,
            "by_idx": by_idx,
            "by_spans": by_spans,
        }
    except Exception as e:
        log(f"Failed to read human labels {labels_path}: {e}")
        return None


def find_matching_human_label(
    labels_data: Optional[Dict[str, Any]],
    source_hash: str,
    gap_idx: Optional[int],
    src_start: float,
    src_end: float
) -> Optional[Dict[str, Any]]:
    """Resolve whether a candidate gap has a matching human annotation."""
    if not labels_data:
        return None

    # 1. Match by (sourceHash, gapIdx)
    if source_hash and gap_idx is not None:
        rec = labels_data["by_hash_idx"].get((source_hash, gap_idx))
        if rec:
            return rec

    # 2. Match by sourceHash and span overlap
    if source_hash and source_hash in labels_data["by_hash_spans"]:
        dur = max(0.001, src_end - src_start)
        for s_sec, e_sec, rec in labels_data["by_hash_spans"][source_hash]:
            ov = min(src_end, e_sec) - max(src_start, s_sec)
            if ov > 0:
                l_dur = max(0.001, e_sec - s_sec)
                if ov / min(dur, l_dur) >= 0.5 or (abs(src_start - s_sec) < 0.1 and abs(src_end - e_sec) < 0.1):
                    return rec

    # 3. Match without hash by gapIdx
    if gap_idx is not None and gap_idx in labels_data["by_idx"]:
        return labels_data["by_idx"][gap_idx]

    # 4. Match without hash by span overlap
    dur = max(0.001, src_end - src_start)
    for s_sec, e_sec, rec in labels_data["by_spans"]:
        ov = min(src_end, e_sec) - max(src_start, s_sec)
        if ov > 0:
            l_dur = max(0.001, e_sec - s_sec)
            if ov / min(dur, l_dur) >= 0.5 or (abs(src_start - s_sec) < 0.1 and abs(src_end - e_sec) < 0.1):
                return rec

    return None


def attach_words_to_gap(gap: Dict[str, Any], sorted_words: List[Dict[str, Any]]) -> None:
    """Ensure gap has prevWord and nextWord context populated for structural features."""
    if gap.get("prevWord") is not None and gap.get("nextWord") is not None:
        return

    idx = gap.get("idx")
    if idx is not None and 0 <= idx < len(sorted_words) - 1:
        pw = sorted_words[idx]
        nw = sorted_words[idx + 1]
        gs = float(gap.get("srcStartSec", 0.0))
        ge = float(gap.get("srcEndSec", 0.0))
        if abs(float(pw["end"]) - gs) < 0.1 and abs(float(nw["start"]) - ge) < 0.1:
            gap["prevWord"] = pw
            gap["nextWord"] = nw
            return

    # Fallback search by timestamp
    gs = float(gap.get("srcStartSec", 0.0))
    ge = float(gap.get("srcEndSec", 0.0))
    prev_w = None
    next_w = None
    for w in sorted_words:
        if float(w["end"]) <= gs + 1e-3:
            prev_w = w
        elif float(w["start"]) >= ge - 1e-3:
            if next_w is None:
                next_w = w
                break

    gap["prevWord"] = prev_w or {}
    gap["nextWord"] = next_w or {}


def resolve_gap_label(
    gap: Dict[str, Any],
    feat_dict: Dict[str, float],
    human_rec: Optional[Dict[str, Any]]
) -> Tuple[str, str, bool]:
    """Resolve gap label using human annotations or weak-supervision heuristics.

    Returns:
      (label, labelSource, weakSupervision)
    """
    # 1. Human annotation resolution
    if human_rec:
        cls = str(human_rec.get("class", "")).strip().lower()
        if cls in VALID_CLASSES:
            return cls, "human_annotation", False
        log(f"Warning: human annotation class '{cls}' not in valid 6 classes; falling back to heuristics")

    # 2. Weak supervision heuristics
    speaker_change = (
        float(feat_dict.get("speakerChange", 0.0)) == 1.0 or
        gap.get("category") == "speaker_transition"
    )
    if speaker_change:
        return "speaker_transition", "weak_supervision_engine_decision", True

    cat = gap.get("category")
    conf = float(gap.get("confidence", 0.0))
    engine_removable = (cat in ("breath_pause", "waiting_pause") and conf >= 0.70)
    engine_breath = (cat == "breath_pause")
    vad_quiet = (
        float(feat_dict.get("vadSilenceFrac", 0.0)) >= 0.60 and
        float(feat_dict.get("levelDropDb", 0.0)) < -6.0
    )

    if engine_removable or vad_quiet or engine_breath:
        hf_ratio = max(
            float(feat_dict.get("hfRatio", 0.0)),
            float(feat_dict.get("evHfRatio", 0.0))
        )
        if hf_ratio >= 0.15:
            return "breath_pause", "weak_supervision_engine_decision", True
        else:
            return "waiting_pause", "weak_supervision_engine_decision", True

    if cat == "nonspeech_unknown":
        return "nonspeech_unknown", "weak_supervision_engine_decision", True

    ends_sentence = (
        float(feat_dict.get("endsSentence", 0.0)) == 1.0 or
        cat == "sentence_pause"
    )
    if ends_sentence:
        return "sentence_pause", "weak_supervision_engine_decision", True

    return "normal_word_gap", "weak_supervision_engine_decision", True


def process_media_transcript(
    source_path: str,
    words_json: str,
    labels_data: Optional[Dict[str, Any]] = None,
    clip_start: float = 0.0,
    clip_end: Optional[float] = None
) -> List[Dict[str, Any]]:
    """Process a single media and words transcript pair.

    Returns a list of JSONL-compliant record dicts.
    """
    if pi is None or breath_detect is None:
        log("Error: pause_intelligence or breath_detect failed to import")
        return []

    if not os.path.isfile(source_path):
        log(f"Media file not found: {source_path}")
        return []

    words = load_words(words_json)
    if words is None or len(words) < 2:
        log(f"Transcript has insufficient words (<2) in {words_json}")
        return []

    # Determine clip boundaries
    if clip_end is None:
        dur = probe_media_duration(source_path)
        if dur is not None and dur > clip_start:
            clip_end = dur
        else:
            clip_end = max(float(w["end"]) for w in words) + 0.5

    if clip_end <= clip_start:
        log(f"Degenerate clip range [{clip_start}, {clip_end}] for {source_path}")
        return []

    # 1. Gap detection
    try:
        det_res = breath_detect.detect_gaps(source_path, clip_start, clip_end, words)
    except Exception as e:
        log(f"Exception during detect_gaps for {source_path}: {e}")
        return []

    if not det_res or det_res.get("status") != "ok":
        log(f"detect_gaps failed ({det_res.get('reason') if det_res else 'unknown'}) for {source_path}")
        return []

    raw_gaps = det_res.get("gaps", [])
    if not raw_gaps:
        return []

    # 2. Strict filtering: reject gaps with duration < 0.05s or duration > 5.0s
    filtered_gaps = []
    for g in raw_gaps:
        try:
            gs = float(g["srcStartSec"])
            ge = float(g["srcEndSec"])
        except (KeyError, TypeError, ValueError):
            continue
        dur = float(g.get("durationSec", ge - gs))
        calc_dur = ge - gs
        if dur < 0.05 or dur > 5.0 or calc_dur < 0.05 or calc_dur > 5.0:
            continue
        attach_words_to_gap(g, words)
        filtered_gaps.append(g)

    if not filtered_gaps:
        return []

    # 3. Feature extraction (23 dimensions)
    try:
        feats = pi.build_gap_features(source_path, filtered_gaps, words, clip_start, clip_end)
    except Exception as e:
        log(f"Exception during build_gap_features for {source_path}: {e}")
        return []

    if feats is None:
        log(f"Audio decoding or VAD failure for {source_path}; no fake rows fabricated")
        return []

    src_hash = pi.source_hash(source_path)
    now_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    records = []
    for i, (gap, f_dict) in enumerate(zip(filtered_gaps, feats)):
        gap_idx = int(gap.get("idx", i))
        gs = round(float(gap["srcStartSec"]), 3)
        ge = round(float(gap["srcEndSec"]), 3)

        # Build feature vector
        row = pi.features_to_row(f_dict)
        if len(row) != len(FEATURE_ORDER):
            # Ensure row length matches 23 dimensions exactly
            row = [float(f_dict.get(k, 0.0)) for k in FEATURE_ORDER]

        # Check human label
        human_rec = find_matching_human_label(labels_data, src_hash, gap_idx, gs, ge)
        label, label_source, is_weak = resolve_gap_label(gap, f_dict, human_rec)

        rec = {
            "sourceHash": src_hash,
            "gapIdx": gap_idx,
            "srcStartSec": gs,
            "srcEndSec": ge,
            "features": [round(float(v), 5) for v in row],
            "label": label,
            "labelSource": label_source,
            "weakSupervision": is_weak,
            "timestamp": now_ts,
        }
        records.append(rec)

    return records


def crawl_corpus(corpus_dir: str) -> List[Tuple[str, str]]:
    """Crawl a directory for media files paired with word transcripts."""
    if not os.path.isdir(corpus_dir):
        log(f"Corpus directory not found: {corpus_dir}")
        return []

    pairs = []
    for root, _, files in os.walk(corpus_dir):
        file_set = set(files)
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in MEDIA_EXTENSIONS:
                media_path = os.path.join(root, f)
                base = os.path.splitext(f)[0]

                # Check candidate transcript names
                candidates = [
                    f"{base}.words.json",
                    f"{base}.json",
                    f"{base}_words.json",
                    f"{base}.transcript.json",
                ]
                matched_words = None
                for cand in candidates:
                    if cand in file_set:
                        words_path = os.path.join(root, cand)
                        # Verify candidate looks like transcript
                        if os.path.isfile(words_path):
                            matched_words = words_path
                            break

                if matched_words:
                    pairs.append((media_path, matched_words))

    return pairs


def write_records(records: List[Dict[str, Any]], out_path: str, overwrite: bool = False) -> int:
    """Write records as JSONL to out_path."""
    if not records:
        return 0

    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)

    mode = "w" if overwrite else "a"
    count = 0
    with open(out_path, mode, encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    return count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pause Intelligence Dataset Labeling Pipeline (`scripts/pause_label.py`)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/pause_label.py video.mp4 video.words.json
  python scripts/pause_label.py video.mp4 video.words.json --labels human.jsonl --out scripts/data/pause_features_v1.jsonl
  python scripts/pause_label.py --corpus ./corpus_dir --out scripts/data/pause_features_v1.jsonl
        """
    )
    parser.add_argument("source_path", nargs="?", default=None, help="Path to audio/video media file")
    parser.add_argument("words_json", nargs="?", default=None, help="Path to JSON transcript with word-level timestamps")
    parser.add_argument("--labels", default=None, help="Path to human labels JSONL")
    parser.add_argument(
        "--out",
        default=os.path.join(SCRIPT_DIR, "data", "pause_features_v1.jsonl"),
        help="Target output JSONL file (default: scripts/data/pause_features_v1.jsonl)"
    )
    parser.add_argument("--corpus", default=None, help="Directory to crawl for media + transcript pairs")
    parser.add_argument("--clip-start", type=float, default=0.0, help="Optional clip start boundary in seconds (default: 0.0)")
    parser.add_argument("--clip-end", type=float, default=None, help="Optional clip end boundary in seconds")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output JSONL instead of appending")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose informational logs")
    return parser


def main(argv=None) -> int:
    ensure_data_dir()
    parser = build_parser()
    args = parser.parse_args(argv)

    if not (args.source_path and args.words_json) and not args.corpus:
        parser.print_help(sys.stderr)
        return 1

    labels_data = load_human_labels(args.labels) if args.labels else None

    # Discover jobs
    jobs: List[Tuple[str, str]] = []
    if args.source_path and args.words_json:
        jobs.append((args.source_path, args.words_json))
    if args.corpus:
        corpus_pairs = crawl_corpus(args.corpus)
        if not args.quiet:
            log(f"Corpus crawl found {len(corpus_pairs)} media/transcript pair(s) in {args.corpus}")
        jobs.extend(corpus_pairs)

    if not jobs:
        log("No valid media and transcript pairs to process.")
        return 0

    total_records = 0
    first_job = True
    for media_file, words_file in jobs:
        if not args.quiet:
            log(f"Processing: {media_file} with {words_file}")
        recs = process_media_transcript(
            media_file,
            words_file,
            labels_data=labels_data,
            clip_start=args.clip_start,
            clip_end=args.clip_end,
        )
        if recs:
            overwrite_mode = args.overwrite and first_job
            written = write_records(recs, args.out, overwrite=overwrite_mode)
            total_records += written
            first_job = False
            if not args.quiet:
                log(f"Extracted and wrote {written} gap records for {os.path.basename(media_file)}")

    if not args.quiet:
        log(f"Pipeline complete. Total records written: {total_records} -> {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
