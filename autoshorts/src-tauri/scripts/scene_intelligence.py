#!/usr/bin/env python3
"""
AutoShorts 11.0 Phase 3 — Scene Intelligence (source-level shot boundaries)

Runs PyTorch CUDA AdaptiveDetector ONCE per source video and caches the result
per (source hash, detector, config). The boundary list is a SOURCE-LEVEL METADATA
LAYER consumed by framing (speaker_tracker.py) and available to re-ID / caption systems.

Failure contract: any failure (missing file, decode error, package missing)
returns a single scene covering the whole video — rendering continues with
the existing deterministic behavior. Scenes are ADVISORY: the framing
engine's own cut detection and safety logic remain authoritative.
"""

import os
import sys
import json
import time
import hashlib
import argparse
import subprocess
import threading
from collections import deque
from typing import List, Dict, Optional

import numpy as np
import torch

SCENE_DETECTOR_NAME = "PyTorch-CUDA-AdaptiveDetector"
SCENE_DETECTOR_VERSION = "1.0.0"
SCENE_CONFIG_VERSION = "gpu-cuda-v1"

# Adaptive threshold and minimum debounce distance (frames)
ADAPTIVE_THRESHOLD = 3.0
MIN_SCENE_LEN_FRAMES = 12
BATCH_SIZE = 128


def _env_frame_skip() -> int:
    raw = os.environ.get("AUTOSHORTS_SCENE_FRAME_SKIP", "").strip()
    if not raw:
        return 2
    try:
        value = int(raw)
    except ValueError:
        sys.stderr.write(f"[SceneIntel] invalid AUTOSHORTS_SCENE_FRAME_SKIP={raw!r}; using 2\n")
        return 2
    return value if value >= 0 else 2


FRAME_SKIP = _env_frame_skip()


def compute_source_hash(path: str) -> str:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""


def _compute_luma_diffs_pytorch(chunk: List[np.ndarray], device: torch.device) -> Optional[np.ndarray]:
    """Calculate frame-to-frame L1 luminance diffs on GPU."""
    if len(chunk) < 2:
        return None
    try:
        tensor_frames = torch.from_numpy(np.stack(chunk)).to(device=device, dtype=torch.float32)
        if tensor_frames.ndim == 4 and tensor_frames.shape[-1] == 3:
            # Convert to grayscale luminance: 0.299 R + 0.587 G + 0.114 B
            # Input frames from OpenCV / numpy are BGR (channel 0: B, 1: G, 2: R)
            luma = 0.299 * tensor_frames[:, :, :, 2] + 0.587 * tensor_frames[:, :, :, 1] + 0.114 * tensor_frames[:, :, :, 0]
        elif tensor_frames.ndim == 4 and tensor_frames.shape[-1] == 1:
            luma = tensor_frames[:, :, :, 0]
        elif tensor_frames.ndim == 3:
            # Grayscale frames directly from rawvideo gray pipe: shape (batch, H, W)
            luma = tensor_frames
        else:
            luma = tensor_frames.squeeze()
        diffs = torch.abs(luma[1:] - luma[:-1]).mean(dim=(1, 2))
        diffs_cpu = diffs.cpu().numpy()
        del tensor_frames, luma, diffs
        return diffs_cpu
    except Exception as e:
        sys.stderr.write(f"[SceneIntel] PyTorch tensor error: {e}\n")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None


def _detect_cuts_from_diffs(
    diffs_cpu: np.ndarray,
    fps: float = 30.0,
    adaptive_threshold: float = ADAPTIVE_THRESHOLD,
    min_scene_len_frames: int = MIN_SCENE_LEN_FRAMES,
    min_content_val: float = 15.0,
) -> List[float]:
    """Rolling adaptive mean over window w=2 (frames t-2..t+2)."""
    w = 2
    n = len(diffs_cpu)
    cuts = []
    last_cut_frame = -min_scene_len_frames

    for i in range(n):
        score = float(diffs_cpu[i])
        if score < min_content_val:
            continue
        start_w = max(0, i - w)
        end_w = min(n, i + w + 1)
        window_mean = float(np.mean(diffs_cpu[start_w:end_w]))
        ratio = score / (window_mean + 1e-5)

        if ratio >= adaptive_threshold and (i - last_cut_frame) >= min_scene_len_frames:
            cut_time = round((i + 1) / fps, 3)
            cuts.append(cut_time)
            last_cut_frame = i

    return cuts


def detect_cuts_from_frames_pytorch(
    frames: List[np.ndarray],
    fps: float = 30.0,
    adaptive_threshold: float = ADAPTIVE_THRESHOLD,
    min_scene_len_frames: int = MIN_SCENE_LEN_FRAMES,
    min_content_val: float = 15.0,
) -> List[float]:
    """Calculate cut timestamps using PyTorch CUDA tensor operations with rolling mean."""
    if len(frames) < 2:
        return []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        batch_size = BATCH_SIZE
        diffs_cpu_list = []
        for i in range(0, len(frames) - 1, batch_size):
            chunk = frames[i : min(len(frames), i + batch_size + 1)]
            if len(chunk) < 2:
                break
            batch_diffs = _compute_luma_diffs_pytorch(chunk, device)
            if batch_diffs is not None and len(batch_diffs) > 0:
                diffs_cpu_list.append(batch_diffs)

        if not diffs_cpu_list:
            return []
        diffs_cpu = np.concatenate(diffs_cpu_list)
        return _detect_cuts_from_diffs(
            diffs_cpu,
            fps=fps,
            adaptive_threshold=adaptive_threshold,
            min_scene_len_frames=min_scene_len_frames,
            min_content_val=min_content_val,
        )
    except Exception as e:
        sys.stderr.write(f"[SceneIntel] PyTorch detection error: {e}\n")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return []


def _probe_video_stream(source_path: str) -> Dict:
    """Probes video stream info using ffprobe, returning codec, duration, and fps."""
    info = {
        "codec": "",
        "duration": 0.0,
        "fps": 30.0,
    }
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,r_frame_rate,avg_frame_rate,duration:format=duration",
            "-of", "json",
            source_path,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if res.returncode == 0 and res.stdout:
            data = json.loads(res.stdout)
            streams = data.get("streams", [])
            if streams:
                v = streams[0]
                info["codec"] = v.get("codec_name", "").lower()
                for r_key in ["avg_frame_rate", "r_frame_rate"]:
                    val = v.get(r_key, "")
                    if "/" in val:
                        num, den = val.split("/", 1)
                        try:
                            f_num, f_den = float(num), float(den)
                            if f_den > 0:
                                parsed_fps = f_num / f_den
                                if 1.0 <= parsed_fps <= 240.0:
                                    info["fps"] = parsed_fps
                                    break
                        except Exception:
                            pass
                try:
                    if v.get("duration"):
                        info["duration"] = float(v["duration"])
                except Exception:
                    pass
            if info["duration"] <= 0.0 and "format" in data:
                try:
                    info["duration"] = float(data["format"].get("duration", 0.0))
                except Exception:
                    pass
    except Exception as e:
        sys.stderr.write(f"[SceneIntel] ffprobe probe error: {e}\n")
    return info


def _build_scenes_from_cuts(cuts: List[float], duration_sec: float) -> Optional[List[Dict]]:
    """Helper to construct standard scene boundaries dictionary list from detected cut timestamps."""
    scenes = []
    dur_r = round(duration_sec, 3) if duration_sec > 0.0 else 0.0
    prev_time = 0.0
    scene_id = 0

    for cut in cuts:
        cut_r = round(cut, 3)
        prev_r = round(prev_time, 3)
        if cut_r > prev_r and (dur_r <= 0.0 or cut_r < dur_r):
            scenes.append({
                "sceneId": scene_id,
                "start": prev_r,
                "end": cut_r,
            })
            prev_time = cut_r
            scene_id += 1

    prev_r = round(prev_time, 3)
    final_end = dur_r if dur_r > prev_r else round(prev_r + 0.1, 3)
    if final_end > prev_r:
        scenes.append({
            "sceneId": scene_id,
            "start": prev_r,
            "end": final_end,
        })

    if not scenes:
        return None
    return scenes


def _detect_scenes_ffmpeg_pipe(
    source_path: str,
    frame_skip: int = FRAME_SKIP,
    adaptive_threshold: float = ADAPTIVE_THRESHOLD,
    min_scene_len_frames: int = MIN_SCENE_LEN_FRAMES,
    allow_hwaccel: bool = True,
) -> Optional[List[Dict]]:
    """Decode video via FFmpeg subprocess pipe directly into 640x360 grayscale frames."""
    info = _probe_video_stream(source_path)
    fps = info.get("fps", 30.0)
    if fps <= 0.0 or not np.isfinite(fps):
        fps = 30.0

    step = frame_skip + 1 if frame_skip > 0 else 1
    effective_fps = fps / step
    effective_min_scene_len = max(1, min_scene_len_frames // step)
    duration_sec = info.get("duration", 0.0)

    target_w, target_h = 640, 360
    frame_bytes = target_w * target_h

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    codec = info.get("codec", "").lower()
    has_cuda = torch.cuda.is_available()

    # Hardware acceleration / multi-threaded decoder selection
    is_h264_hevc_vp9 = any(c in codec for c in ("h264", "avc", "hevc", "h265", "vp9"))
    if allow_hwaccel and is_h264_hevc_vp9 and has_cuda:
        cmd.extend(["-hwaccel", "cuda"])
    elif "av1" in codec or "av01" in codec:
        cmd.extend(["-c:v", "libdav1d", "-threads", "0"])
    else:
        cmd.extend(["-threads", "0"])

    cmd.extend([
        "-i", source_path,
        "-vf", f"fps={effective_fps:.4f},scale={target_w}:{target_h}",
        "-f", "rawvideo",
        "-pix_fmt", "gray",
        "pipe:1"
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = BATCH_SIZE
    diffs_cpu_list = []
    current_chunk = []
    frames_read = 0
    last_pct = 0
    expected_total_frames = int(duration_sec * effective_fps) if duration_sec > 0 else 0

    stderr_tail = deque(maxlen=20)

    def _drain_stderr(pipe):
        try:
            for line in iter(pipe.readline, b""):
                stderr_tail.append(line.decode("utf-8", errors="replace"))
        except Exception:
            pass
        finally:
            try:
                pipe.close()
            except Exception:
                pass

    proc = None
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        t_stderr = threading.Thread(target=_drain_stderr, args=(proc.stderr,), daemon=True)
        t_stderr.start()

        sys.stderr.write("[SceneIntel] PROGRESS 0%\n")
        sys.stderr.flush()

        while True:
            raw = proc.stdout.read(frame_bytes)
            if not raw or len(raw) < frame_bytes:
                break
            frames_read += 1
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((target_h, target_w))
            current_chunk.append(frame)

            if expected_total_frames > 0:
                pct = int(min(99, max(0, (frames_read / expected_total_frames) * 100)))
                if pct > last_pct:
                    sys.stderr.write(f"[SceneIntel] PROGRESS {pct}%\n")
                    sys.stderr.flush()
                    last_pct = pct

            if len(current_chunk) >= batch_size + 1:
                batch_diffs = _compute_luma_diffs_pytorch(current_chunk, device)
                if batch_diffs is not None and len(batch_diffs) > 0:
                    diffs_cpu_list.append(batch_diffs)
                # Keep last frame for continuity with the next batch
                current_chunk = [current_chunk[-1]]

        if len(current_chunk) >= 2:
            batch_diffs = _compute_luma_diffs_pytorch(current_chunk, device)
            if batch_diffs is not None and len(batch_diffs) > 0:
                diffs_cpu_list.append(batch_diffs)
            current_chunk = []

        sys.stderr.write("[SceneIntel] PROGRESS 100%\n")
        sys.stderr.flush()

    except Exception as e:
        sys.stderr.write(f"[SceneIntel] FFmpeg pipe stream error: {e}\n")
        return None
    finally:
        if proc is not None:
            try:
                if proc.stdout:
                    proc.stdout.close()
                proc.terminate()
                proc.wait(timeout=2.0)
            except Exception:
                pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not diffs_cpu_list or frames_read < 2:
        err_msg = "".join(stderr_tail).strip()
        if err_msg:
            sys.stderr.write(f"[SceneIntel] FFmpeg error output: {err_msg}\n")
        sys.stderr.write(f"[SceneIntel] FFmpeg pipe decoded insufficient frames ({frames_read})\n")
        return None

    diffs_cpu = np.concatenate(diffs_cpu_list)
    cuts = _detect_cuts_from_diffs(
        diffs_cpu,
        fps=effective_fps,
        adaptive_threshold=adaptive_threshold,
        min_scene_len_frames=effective_min_scene_len,
    )

    if duration_sec <= 0.0:
        duration_sec = _probe_duration(source_path)
    if duration_sec <= 0.0 and len(diffs_cpu) > 0:
        duration_sec = round((len(diffs_cpu) + 1) / effective_fps, 3)

    return _build_scenes_from_cuts(cuts, duration_sec)


def _detect_scenes_opencv(
    source_path: str,
    frame_skip: int = FRAME_SKIP,
    adaptive_threshold: float = ADAPTIVE_THRESHOLD,
    min_scene_len_frames: int = MIN_SCENE_LEN_FRAMES,
) -> Optional[List[Dict]]:
    """OpenCV VideoCapture fallback for scene detection."""
    try:
        import cv2
    except ImportError as e:
        sys.stderr.write(f"[SceneIntel] OpenCV not available ({e}); fallback\n")
        return None

    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        sys.stderr.write(f"[SceneIntel] OpenCV failed to open video: {source_path}\n")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_w, target_h = 640, 360
    batch_size = BATCH_SIZE
    step = frame_skip + 1 if frame_skip > 0 else 1

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0.0 or not np.isfinite(fps):
            fps = 30.0

        duration_sec = total_frames / fps if total_frames > 0 else 0.0
        effective_fps = fps / step
        effective_min_scene_len = max(1, min_scene_len_frames // step)

        sys.stderr.write("[SceneIntel] PROGRESS 0%\n")
        sys.stderr.flush()

        diffs_cpu_list = []
        current_chunk = []
        frames_read = 0
        last_pct = 0

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            frames_read += 1
            if step > 1 and (frames_read - 1) % step != 0:
                continue

            small = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
            current_chunk.append(small)

            if total_frames > 0:
                pct = int(min(99, max(0, (frames_read / total_frames) * 100)))
                if pct > last_pct:
                    sys.stderr.write(f"[SceneIntel] PROGRESS {pct}%\n")
                    sys.stderr.flush()
                    last_pct = pct

            if len(current_chunk) >= batch_size + 1:
                batch_diffs = _compute_luma_diffs_pytorch(current_chunk, device)
                if batch_diffs is not None and len(batch_diffs) > 0:
                    diffs_cpu_list.append(batch_diffs)
                # Keep last frame for continuity with the next batch
                current_chunk = [current_chunk[-1]]

        if len(current_chunk) >= 2:
            batch_diffs = _compute_luma_diffs_pytorch(current_chunk, device)
            if batch_diffs is not None and len(batch_diffs) > 0:
                diffs_cpu_list.append(batch_diffs)
            current_chunk = []

        sys.stderr.write("[SceneIntel] PROGRESS 100%\n")
        sys.stderr.flush()

    except Exception as e:
        sys.stderr.write(f"[SceneIntel] OpenCV detection failed ({e})\n")
        return None
    finally:
        cap.release()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not diffs_cpu_list:
        sys.stderr.write("[SceneIntel] OpenCV decoded no diffs\n")
        return None

    diffs_cpu = np.concatenate(diffs_cpu_list)
    cuts = _detect_cuts_from_diffs(
        diffs_cpu,
        fps=effective_fps,
        adaptive_threshold=adaptive_threshold,
        min_scene_len_frames=effective_min_scene_len,
    )

    if duration_sec <= 0.0:
        duration_sec = _probe_duration(source_path)
    if duration_sec <= 0.0 and len(diffs_cpu) > 0:
        duration_sec = round((len(diffs_cpu) + 1) / effective_fps, 3)

    return _build_scenes_from_cuts(cuts, duration_sec)


def detect_scenes_gpu(
    source_path: str,
    frame_skip: int = FRAME_SKIP,
    adaptive_threshold: float = ADAPTIVE_THRESHOLD,
    min_scene_len_frames: int = MIN_SCENE_LEN_FRAMES,
) -> Optional[List[Dict]]:
    """Run PyTorch CUDA adaptive detector over video. None = failure (single scene fallback)."""
    if not source_path or not os.path.exists(source_path):
        sys.stderr.write(f"[SceneIntel] source file not found: {source_path}\n")
        return None

    # Tier 1: Fast stream ingestion via FFmpeg pipe (with hwaccel if available)
    try:
        scenes = _detect_scenes_ffmpeg_pipe(
            source_path=source_path,
            frame_skip=frame_skip,
            adaptive_threshold=adaptive_threshold,
            min_scene_len_frames=min_scene_len_frames,
            allow_hwaccel=True,
        )
        if scenes is not None:
            return scenes
    except Exception as e:
        sys.stderr.write(f"[SceneIntel] FFmpeg accelerated pipe error ({e})\n")

    # Tier 2: FFmpeg pipe retry in pure software mode (if hwaccel failed)
    try:
        sys.stderr.write("[SceneIntel] Retrying FFmpeg pipe in software decode mode...\n")
        scenes = _detect_scenes_ffmpeg_pipe(
            source_path=source_path,
            frame_skip=frame_skip,
            adaptive_threshold=adaptive_threshold,
            min_scene_len_frames=min_scene_len_frames,
            allow_hwaccel=False,
        )
        if scenes is not None:
            return scenes
    except Exception as e:
        sys.stderr.write(f"[SceneIntel] FFmpeg software retry failed ({e})\n")

    # Tier 3: OpenCV VideoCapture fallback
    sys.stderr.write("[SceneIntel] Attempting OpenCV fallback decoder...\n")
    return _detect_scenes_opencv(
        source_path=source_path,
        frame_skip=frame_skip,
        adaptive_threshold=adaptive_threshold,
        min_scene_len_frames=min_scene_len_frames,
    )


detect_scenes = detect_scenes_gpu


def run_cached(source_path: str, cache_dir: Optional[str]) -> Dict:
    """Cached scene detection. Never raises; failure = whole-video scene."""
    source_hash = compute_source_hash(source_path)
    cache_file = None
    if cache_dir and source_hash:
        cache_file = os.path.join(cache_dir, f"pyscenes_{source_hash}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    doc = json.load(f)
                if (doc.get("sourceHash") == source_hash
                        and doc.get("detectorVersion") == SCENE_DETECTOR_VERSION
                        and doc.get("configVersion") == SCENE_CONFIG_VERSION
                        and doc.get("scenes")):
                    sys.stderr.write(f"[SceneIntel] cache hit: {len(doc['scenes'])} scenes\n")
                    return doc
                sys.stderr.write("[SceneIntel] stale/corrupt cache removed\n")
                os.remove(cache_file)
            except Exception:
                sys.stderr.write("[SceneIntel] unreadable cache; recomputing\n")

    t0 = time.time()
    scenes = detect_scenes(source_path, frame_skip=FRAME_SKIP)
    elapsed = time.time() - t0
    used_fallback = False
    duration = scenes[-1]["end"] if scenes else 0.0
    if not scenes:
        used_fallback = True
        sys.stderr.write(
            "[SceneIntel] detection produced no scenes; single-scene fallback\n"
        )
        duration = _probe_duration(source_path)
        scenes = [{"sceneId": 0, "start": 0.0, "end": round(duration, 3)}]

    doc = {
        "sourceHash": source_hash,
        "detector": SCENE_DETECTOR_NAME,
        "detectorVersion": SCENE_DETECTOR_VERSION,
        "configVersion": SCENE_CONFIG_VERSION,
        # Part of the Rust serde contract. MUST be present on every path.
        "fallback": used_fallback,
        "config": {"adaptiveThreshold": ADAPTIVE_THRESHOLD, "minSceneLenFrames": MIN_SCENE_LEN_FRAMES, "frameSkip": FRAME_SKIP},
        "scenes": scenes,
        "elapsedSec": round(elapsed, 2),
        "createdAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if cache_file:
        try:
            os.makedirs(cache_dir, exist_ok=True)
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(doc, f, indent=1)
        except Exception as e:
            sys.stderr.write(f"[SceneIntel] cache write failed ({e})\n")
    # Human diagnostics MUST go to stderr: stdout is the machine-readable JSON channel
    sys.stderr.write(f"[SceneIntel] {len(scenes)} scenes in {elapsed:.1f}s "
                     f"(fallback={used_fallback})\n")
    return doc


def _probe_duration(path: str) -> float:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=60).stdout.strip()
        return float(out)
    except Exception:
        return 0.0


def main():
    parser = argparse.ArgumentParser(description="Source-level scene detection (PyTorch CUDA)")
    parser.add_argument("source_path")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--out", default=None, help="Write scene JSON here")
    args = parser.parse_args()
    doc = run_cached(args.source_path, args.cache_dir)
    text = json.dumps(doc, indent=1)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
