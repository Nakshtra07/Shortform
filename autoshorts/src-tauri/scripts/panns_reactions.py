#!/usr/bin/env python3
"""
AutoShorts 11.0 Phase 4 — PANNs Reaction Detection Sidecar

Source-level audio event detection using PANNs (Pre-trained Audio Neural Networks)
(CNN14 + DecisionLevelMax) over the source audio track.

CLI contract:
    python panns_reactions.py SOURCE [--model MODEL] [--threshold THRESHOLD]

    - SOURCE        absolute path to the source video/audio file

Output: a SINGLE JSON object on the LAST (only) stdout line:

    {"ok": true,  "events": [...], "diagnostics": {...}}
    {"ok": false, "events": [], "error": {"code": ..., "message": ...}, "diagnostics": {...}}

Telemetry goes to stderr. The `ok` flag plus the structured `error` object is the
ONLY way this sidecar reports a failure — there is no path that silently returns
"0 events" while the model never ran. Exit code is 0 on success, 2 on a
structured engine failure (missing checkpoint, unusable audio, inference error).

WHY THIS FILE USED TO RETURN 0 EVENTS IN 0.5s
-----------------------------------------------
`main()` defaulted to `detect_reactions_stub()` unless AUTOSHORTS_PANNS_REAL was
set, and the Rust caller never set that variable nor passed --use-real-model. The
stub did no inference at all: it just logged "PANNs model not available" and
returned []. Real inference now runs by default; the stub is reachable only via
an explicit --stub flag (and is marked as such in diagnostics).
"""

import sys
import os
import json
import argparse
import subprocess
import time
from typing import List, Dict, Any, Optional, Tuple

# Add script directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# CNN14 as constructed by panns_inference.SoundEventDetection is fixed at 32 kHz
# (models.py Cnn14_DecisionLevelMax / inference.py: sample_rate=32000). 16 kHz is
# the *AudioSet tagging* rate; feeding 16 kHz to this model is wrong.
MODEL_SAMPLE_RATE = 32000
# hop_size=320 at 32 kHz => 10 ms per frame (models.py: hop_size=320).
FRAME_DURATION_SEC = 320.0 / MODEL_SAMPLE_RATE
# Number of AudioSet classes; CNN14 is fixed at 527 (config.py: classes_num).
PANNS_CLASSES_NUM = 527
# A PANNs DecisionLevelMax checkpoint is ~320 MB. Anything smaller is a truncated
# download (this is the same sanity check panns_inference/inference.py uses).
MIN_CHECKPOINT_BYTES = 3 * 10**8
# Event shorter than this is noise, not a reaction.
MIN_EVENT_DURATION_SEC = 0.2


class PannsError(Exception):
    """Structured failure. `code` is stable and machine-checkable."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def log(msg):
    sys.stderr.write(f"[PANNs] {msg}\n")
    sys.stderr.flush()


def emit(payload: Dict[str, Any]):
    """Print the single JSON result line on stdout."""
    print(json.dumps(payload, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Checkpoint resolution — explicit, deterministic, and reported.
# ---------------------------------------------------------------------------

def checkpoint_candidates(model: str = "cnn14") -> List[str]:
    """
    Every path we will look at for a CNN14 checkpoint, in priority order.

    Sources, in order:
      1. AUTOSHORTS_PANNS_CHECKPOINT   (explicit override, always wins)
      2. <repo>/src-tauri/models/      and ./models/   (app-local models dir)
      3. <home>/panns_data/            (panns_inference's own default location)

    Returned as a list even when the env var is set, so callers can report the
    full search order on failure.
    """
    paths: List[str] = []

    env_ckpt = os.environ.get("AUTOSHORTS_PANNS_CHECKPOINT", "").strip()
    if env_ckpt:
        paths.append(env_ckpt)

    # App-local model directories, relative to this script and to the CWD.
    repo_root = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
    model_dirs = [
        os.path.join(repo_root, "src-tauri", "models"),
        os.path.join(repo_root, "models"),
        os.path.join(os.getcwd(), "models"),
        os.path.join(os.getcwd(), "src-tauri", "models"),
    ]
    filenames = [
        "Cnn14_DecisionLevelMax.pth",
        "Cnn14_mAP=0.431.pth",
        "Cnn14_DecisionLevelMax_mAP=0.385.pth",
    ]
    for d in model_dirs:
        for fn in filenames:
            paths.append(os.path.join(d, fn))

    # panns_inference's own default (inference.py: <home>/panns_data/...).
    paths.append(os.path.join(os.path.expanduser("~"), "panns_data", "Cnn14_DecisionLevelMax.pth"))
    paths.append(os.path.join(os.path.expanduser("~"), "panns_data", "Cnn14_mAP=0.431.pth"))

    # De-duplicate, keep order.
    seen = set()
    unique = []
    for p in paths:
        key = os.path.normcase(os.path.abspath(p))
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def resolve_checkpoint() -> str:
    """
    Find a usable PANNs CNN14 checkpoint or raise PannsError with a concrete,
    actionable reason. Never downloads anything.

    An explicit AUTOSHORTS_PANNS_CHECKPOINT override is AUTHORITATIVE: if it
    is set but unusable, that is reported as an error rather than silently
    falling back to some other checkpoint on disk. A silent substitution would
    mean the operator believes they are testing one model while a different
    one produces the numbers.
    """
    override = os.environ.get("AUTOSHORTS_PANNS_CHECKPOINT", "").strip()
    if override:
        if not os.path.isfile(override):
            raise PannsError(
                "checkpoint_missing",
                f"AUTOSHORTS_PANNS_CHECKPOINT points at a path that does not "
                f"exist: {override}",
            )
        size = os.path.getsize(override)
        if size < MIN_CHECKPOINT_BYTES:
            raise PannsError(
                "checkpoint_truncated",
                f"AUTOSHORTS_PANNS_CHECKPOINT points at a file too small to be "
                f"a CNN14 DecisionLevelMax checkpoint "
                f"({size} bytes < {MIN_CHECKPOINT_BYTES}): {override}",
            )
        log(f"checkpoint FROM OVERRIDE {override} ({size} bytes)")
        return override

    searched = checkpoint_candidates()
    existing = []
    too_small = []
    for path in searched:
        if not os.path.isfile(path):
            continue
        size = os.path.getsize(path)
        if size < MIN_CHECKPOINT_BYTES:
            too_small.append(f"{path} ({size} bytes < {MIN_CHECKPOINT_BYTES})")
            continue
        log(f"checkpoint FOUND {path} ({size} bytes)")
        return path

    if too_small:
        raise PannsError(
            "checkpoint_truncated",
            "PANNs checkpoint(s) found but too small to be a CNN14 DecisionLevelMax "
            f"checkpoint (need >= {MIN_CHECKPOINT_BYTES} bytes): {'; '.join(too_small)}. "
            f"Searched: {searched}",
        )
    raise PannsError(
        "checkpoint_missing",
        "PANNs CNN14 checkpoint not found. Set AUTOSHORTS_PANNS_CHECKPOINT or place "
        f"Cnn14_DecisionLevelMax.pth in one of: {searched}",
    )


# ---------------------------------------------------------------------------
# Audio extraction / probing
# ---------------------------------------------------------------------------

def get_audio_duration(source_path) -> float:
    """Get audio duration using ffprobe. Raises PannsError on probe failure."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "csv=p=0", source_path,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except FileNotFoundError as exc:
        raise PannsError("ffprobe_missing", f"ffprobe not found on PATH: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise PannsError("ffprobe_timeout", "ffprobe timed out after 60s") from exc
    if proc.returncode != 0 or not proc.stdout.strip():
        raise PannsError(
            "duration_probe_failed",
            f"ffprobe failed (rc={proc.returncode}): {proc.stderr.strip()[:400]}",
        )
    try:
        return float(proc.stdout.strip())
    except ValueError as exc:
        raise PannsError(
            "duration_parse_failed",
            f"ffprobe returned non-numeric duration {proc.stdout.strip()!r}",
        ) from exc


def load_audio(source_path, duration: float):
    """
    Decode the full source audio to mono float32 @ 32 kHz (the rate CNN14 is
    built for). Returns a C-contiguous, writable 1-D float32 array.
    """
    cmd = [
        "ffmpeg", "-v", "error",
        "-i", source_path,
        "-vn", "-ac", "1", "-ar", str(MODEL_SAMPLE_RATE),
        "-f", "f32le", "-",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=900)
    except FileNotFoundError as exc:
        raise PannsError("ffmpeg_missing", f"ffmpeg not found on PATH: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise PannsError("ffmpeg_timeout", "ffmpeg audio extraction timed out after 900s") from exc
    if proc.returncode != 0 or not proc.stdout:
        raise PannsError(
            "audio_extract_failed",
            f"ffmpeg failed (rc={proc.returncode}): {proc.stderr.decode('utf-8', 'replace').strip()[:400]}",
        )

    import numpy as np
    audio = np.frombuffer(proc.stdout, dtype=np.float32)
    if audio.size == 0:
        raise PannsError("audio_extract_empty", "ffmpeg produced zero audio samples")
    # .copy() makes it writable — torch warns (and would be UB) on read-only arrays.
    audio = np.ascontiguousarray(audio, dtype=np.float32).copy()
    if audio.ndim != 1:
        raise PannsError("audio_shape_invalid", f"expected 1-D mono audio, got shape {audio.shape}")
    return audio


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_detector(checkpoint_path: str, model: str, device: str = "cpu"):
    """
    Build a panns_inference.SoundEventDetection bound to an EXPLICIT checkpoint.

    We never pass checkpoint_path=None: panns_inference's own fallback would try
    to `wget` a checkpoint, which we must never do. Import/load failures are
    re-raised as structured PannsError with the real cause.
    """
    try:
        from panns_inference import SoundEventDetection
    except ImportError as exc:
        raise PannsError(
            "panns_inference_missing",
            f"panns_inference package is not importable: {exc}",
        ) from exc

    try:
        import torch
    except ImportError as exc:
        raise PannsError("torch_missing", f"torch is not importable: {exc}") from exc

    t0 = time.time()
    try:
        # checkpoint_path is always explicit: this disables panns_inference's
        # implicit download branch.
        sed = SoundEventDetection(checkpoint_path=checkpoint_path, device=device)
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        raise PannsError(
            "model_load_failed",
            f"failed to load PANNs CNN14 from {checkpoint_path}: "
            f"{type(exc).__name__}: {exc}",
        ) from exc
    log(f"model loaded in {time.time() - t0:.1f}s from {checkpoint_path}")

    # Prove weights actually landed on the network (a checkpoint that loaded
    # zero tensors would still produce logits, but meaningless ones).
    try:
        state = sed.model.state_dict()
    except AttributeError as exc:
        raise PannsError(
            "model_load_failed",
            f"SoundEventDetection has no .model to inspect: {exc}",
        ) from exc
    if not state:
        raise PannsError(
            "model_load_failed",
            f"PANNs model at {checkpoint_path} has an empty state_dict",
        )
    log(f"model state_dict tensors={len(state)}")
    return sed


_PANNS_CHUNK_SEC = 120.0   # audio seconds per forward pass (memory-bound)
_PANNS_OVERLAP_SEC = 2.0   # cross-chunk overlap so no event is split unseen


def _forward_framewise(sed, batch, model: str):
    """One forward pass over a (1, n_samples) batch with the STRICT output
    contract (3-D -> (n_frames, classes), finite). Raises PannsError on any
    mismatch so a model/ABI problem surfaces loudly."""
    import numpy as np

    try:
        framewise = sed.inference(batch)
    except (RuntimeError, ValueError, TypeError) as exc:
        raise PannsError(
            "inference_failed",
            f"PANNs inference raised {type(exc).__name__}: {exc}",
        ) from exc
    framewise = np.asarray(framewise)
    if framewise.ndim != 3:
        raise PannsError(
            "inference_shape_invalid",
            f"expected 3-D (batch, n_frames, classes) framewise output, "
            f"got {framewise.ndim}-D shape {framewise.shape}",
        )
    if framewise.shape[0] != 1:
        raise PannsError(
            "inference_shape_invalid",
            f"expected batch size 1, got framewise output shape {framewise.shape}",
        )
    framewise = framewise[0]
    if framewise.ndim != 2:
        raise PannsError(
            "inference_shape_invalid",
            f"expected (n_frames, classes) after dropping batch, got {framewise.shape}",
        )
    if framewise.shape[1] != PANNS_CLASSES_NUM:
        raise PannsError(
            "inference_shape_invalid",
            f"expected {PANNS_CLASSES_NUM} AudioSet classes, got {framewise.shape[1]} "
            f"(output shape {framewise.shape})",
        )
    if not np.isfinite(framewise).all():
        raise PannsError(
            "inference_nonfinite",
            f"framewise output contains NaN/Inf (shape {framewise.shape})",
        )
    return framewise


def run_inference(sed, audio, model: str):
    """
    Run CNN14 over the waveform and return the framewise output
    (n_frames, classes_num), frame i covering global source time i * 10 ms —
    the exact contract of a single full-audio pass.

    CNN14 expects (batch_size, data_length) float32 at 32 kHz. panns_inference's
    SoundEventDetection.inference runs the WHOLE input through the network in
    ONE forward pass, so a long source (a 25-min video allocates ~2.4 GB of
    activations) can die on DefaultCPUAllocator: not enough memory. The audio
    is therefore processed in bounded chunks with a 2 s overlap; per-frame
    global times (center=True, hop 320 @ 32 kHz) place every frame into its
    10 ms output slot, so the stitched output is frame-aligned with — and
    semantically identical to — the unchunked result. Short inputs keep the
    single-pass path unchanged.
    """
    import numpy as np

    if audio.ndim != 1:
        raise PannsError("audio_shape_invalid", f"inference expects 1-D audio, got {audio.shape}")
    total = int(audio.size)
    batch = np.ascontiguousarray(audio[np.newaxis, :], dtype=np.float32)
    log(
        f"inference input shape={batch.shape} dtype={batch.dtype} "
        f"sr={MODEL_SAMPLE_RATE} duration={audio.size / MODEL_SAMPLE_RATE:.1f}s"
    )

    t0 = time.time()
    chunk = max(1, int(_PANNS_CHUNK_SEC * MODEL_SAMPLE_RATE))
    overlap = int(_PANNS_OVERLAP_SEC * MODEL_SAMPLE_RATE)
    hop = 320  # models.py: hop_size=320 at 32 kHz => 10 ms per frame

    if total <= chunk:
        framewise = _forward_framewise(sed, batch, model)
    else:
        n_out = total // hop + 1
        stitched = None
        starts = list(range(0, total, chunk - overlap))
        for idx, s0 in enumerate(starts):
            s1 = min(total, s0 + chunk)
            keep_end = s1 if idx == len(starts) - 1 else min(s1, s0 + chunk - overlap)
            piece = np.ascontiguousarray(audio[s0:s1][np.newaxis, :], dtype=np.float32)
            fw = _forward_framewise(sed, piece, model)
            f_times = (s0 + np.arange(fw.shape[0], dtype=np.float64) * hop) / float(MODEL_SAMPLE_RATE)
            mask = (f_times >= s0 / float(MODEL_SAMPLE_RATE)) & (f_times < keep_end / float(MODEL_SAMPLE_RATE))
            if stitched is None:
                stitched = np.zeros((n_out, PANNS_CLASSES_NUM), dtype=fw.dtype)
            slots = np.rint(f_times[mask] / FRAME_DURATION_SEC).astype(np.int64)
            stitched[slots] = fw[mask]
        framewise = stitched
    elapsed = time.time() - t0

    log(
        f"inference DONE in {elapsed:.1f}s framewise_output={framewise.shape} "
        f"min={framewise.min():.3e} max={framewise.max():.4f}"
    )
    return framewise, elapsed


# ---------------------------------------------------------------------------
# Labels + event extraction
# ---------------------------------------------------------------------------

REACTION_KEYWORDS = [
    "laughter", "laugh", "chuckle", "chortle", "giggle", "snicker",
    "applause", "clapping",
    "cheering", "cheer", "crowd", "whoop", "yell", "shout", "scream",
    "gasp", "pant",
]


def load_labels():
    """Return (labels, classes_num) from panns_inference.config."""
    try:
        from panns_inference.config import labels, classes_num
    except ImportError as exc:
        raise PannsError(
            "labels_unavailable",
            f"panns_inference.config (AudioSet labels) not importable: {exc}",
        ) from exc
    if classes_num != PANNS_CLASSES_NUM or len(labels) != PANNS_CLASSES_NUM:
        raise PannsError(
            "labels_mismatch",
            f"expected {PANNS_CLASSES_NUM} AudioSet labels, got "
            f"classes_num={classes_num} len(labels)={len(labels)}",
        )
    return labels, classes_num


def reaction_label_indices(labels) -> List[Tuple[int, str]]:
    """
    (index, label) for every reaction-relevant AudioSet class.

    The index is the position in the PANNs `labels` list, which is exactly the
    column index of the framewise output — so there is no off-by-one risk here
    as long as we use enumerate() over the same list the model was built with.
    """
    hits = []
    for idx, label in enumerate(labels):
        low = label.lower()
        if any(kw in low for kw in REACTION_KEYWORDS):
            hits.append((idx, label))
    if not hits:
        raise PannsError(
            "no_reaction_labels",
            f"none of the {len(labels)} AudioSet labels matched the reaction keyword set",
        )
    return hits


def events_from_framewise(
    framewise,
    reaction_indices: List[Tuple[int, str]],
    threshold: float,
    min_duration: float = MIN_EVENT_DURATION_SEC,
    model: str = "cnn14",
    model_version: str = "1.0",
) -> List[Dict[str, Any]]:
    """
    Group contiguous above-threshold frames per class into events.

    framewise : (n_frames, classes_num) array of per-frame probabilities.
    Events are emitted in GLOBAL SOURCE TIME (frame_index * 10 ms), which is what
    the Rust side associates against candidate start/end ranges.
    """
    import numpy as np

    n_frames = framewise.shape[0]
    events: List[Dict[str, Any]] = []
    # Diagnostics: how close each class came, so a 0-event run is explicable.
    peaks: Dict[str, float] = {}

    for idx, label in reaction_indices:
        col = framewise[:, idx]
        peaks[label] = float(col.max()) if n_frames else 0.0
        open_event: Optional[Dict[str, float]] = None
        for f in range(n_frames):
            prob = float(col[f])
            if prob >= threshold:
                if open_event is None:
                    open_event = {"start": f * FRAME_DURATION_SEC, "max": prob}
                elif prob > open_event["max"]:
                    open_event["max"] = prob
            elif open_event is not None:
                end = f * FRAME_DURATION_SEC
                if end - open_event["start"] >= min_duration:
                    events.append(_make_event(label, open_event["start"], end,
                                              open_event["max"], model, model_version))
                open_event = None
        if open_event is not None:
            end = n_frames * FRAME_DURATION_SEC
            if end - open_event["start"] >= min_duration:
                events.append(_make_event(label, open_event["start"], end,
                                          open_event["max"], model, model_version))

    events.sort(key=lambda e: (e["start"], e["end"]))
    return events, peaks


def _make_event(label, start, end, conf, model, model_version) -> Dict[str, Any]:
    return {
        # AudioSet label verbatim — the Rust side matches on the exact label.
        "eventType": label,
        "start": float(round(start, 3)),
        "end": float(round(end, 3)),
        "confidence": float(round(conf, 6)),
        "model": model,
        "modelVersion": model_version,
    }


def detect_reactions(source_path, model, frame_threshold, min_event_duration,
                     model_version="1.0", device="cpu"):
    """
    Full PANNs pass over one source. Returns (events, diagnostics).

    `frame_threshold` is the per-frame DETECTION threshold. There is
    deliberately NO event-level confidence gate here: the caller (Rust) applies
    `confidence_threshold` so a single knob owns that decision and a stub/low
    threshold can never be applied twice.

    Raises PannsError with a concrete reason on any failure — this function has
    no silent empty-list path.
    """
    diagnostics: Dict[str, Any] = {
        "sampleRate": MODEL_SAMPLE_RATE,
        "frameDurationSec": FRAME_DURATION_SEC,
        "frameThreshold": frame_threshold,
        "minEventDurationSec": min_event_duration,
        "mode": "real",
    }

    duration = get_audio_duration(source_path)
    if duration <= 0:
        raise PannsError("duration_invalid", f"source duration is {duration}")
    diagnostics["sourceDurationSec"] = round(duration, 3)

    checkpoint = resolve_checkpoint()
    diagnostics["checkpointPath"] = checkpoint
    diagnostics["checkpointBytes"] = os.path.getsize(checkpoint)

    audio = load_audio(source_path, duration)
    diagnostics["audioSamples"] = int(audio.size)
    diagnostics["audioSeconds"] = round(audio.size / MODEL_SAMPLE_RATE, 3)
    diagnostics["audioPeak"] = round(float(abs(audio).max()), 6)

    labels, classes_num = load_labels()
    reaction_indices = reaction_label_indices(labels)
    diagnostics["classesNum"] = classes_num
    diagnostics["reactionLabels"] = [lbl for _, lbl in reaction_indices]

    sed = build_detector(checkpoint, model, device=device)
    framewise, infer_s = run_inference(sed, audio, model)
    diagnostics["inferenceSec"] = round(infer_s, 2)
    diagnostics["framewiseShape"] = list(framewise.shape)

    events, peaks = events_from_framewise(
        framewise, reaction_indices, frame_threshold, min_event_duration,
        model, model_version
    )
    diagnostics["reactionLabelPeakProb"] = {k: round(v, 6) for k, v in sorted(peaks.items())}
    diagnostics["reactionLabelPeakMax"] = round(max(peaks.values()) if peaks else 0.0, 6)

    log(
        f"detected {len(events)} events (frame_threshold={frame_threshold}, "
        f"min_dur={min_event_duration}s, reaction_peak_max={diagnostics['reactionLabelPeakMax']})"
    )
    return events, diagnostics


def detect_reactions_stub(source_path, model, threshold, model_version="1.0"):
    """
    Test-only stub. Produces NO real detections — it exists so the Rust parse
    path can be exercised without a checkpoint. Emitted events are explicitly
    marked "model": "stub" so they can never be mistaken for CNN14 output.
    """
    log("WARNING: running STUB detector — no CNN14 inference is performed")
    if os.environ.get("AUTOSHORTS_PANNS_STUB_EVENTS") == "1":
        events = [
            {"eventType": "Laughter", "start": 10.5, "end": 12.0, "confidence": 0.87,
             "model": "stub", "modelVersion": "0"},
            {"eventType": "Applause", "start": 45.0, "end": 47.5, "confidence": 0.92,
             "model": "stub", "modelVersion": "0"},
            {"eventType": "Cheering", "start": 89.0, "end": 92.0, "confidence": 0.78,
             "model": "stub", "modelVersion": "0"},
        ]
    else:
        events = []
    diagnostics = {"mode": "stub", "modelLoaded": False,
                   "reason": "stub detector requested via --stub"}
    return events, diagnostics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="AutoShorts Phase 4 — PANNs Reaction Detection")
    parser.add_argument("source", type=str, help="Path to source video/audio")
    parser.add_argument("--model", type=str, default=None, help="Model name (from env AUTOSHORTS_PANNS_MODEL)")
    parser.add_argument("--model-version", type=str, default=None, help="Model version (from env AUTOSHORTS_PANNS_MODEL_VERSION)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="EVENT-level confidence gate (from env AUTOSHORTS_PANNS_CONFIDENCE_THRESHOLD). "
                             "Applied by the Rust caller, not here.")
    parser.add_argument("--frame-threshold", type=float, default=None,
                        help="Per-frame DETECTION threshold (from env AUTOSHORTS_PANNS_FRAME_THRESHOLD)")
    parser.add_argument("--min-event-duration", type=float, default=None,
                        help="Minimum event duration seconds (from env AUTOSHORTS_PANNS_MIN_EVENT_DURATION)")
    parser.add_argument("--output", type=str, default=None, help="Optional output JSON file path")
    parser.add_argument("--device", type=str, default=os.environ.get("AUTOSHORTS_PANNS_DEVICE", "cpu"),
                        help="Inference device ('cpu' or 'cuda')")
    parser.add_argument("--stub", action="store_true", default=False,
                        help="Use the test stub detector (no CNN14 inference). Off by default.")
    parser.add_argument("--probe", action="store_true", default=False,
                        help="Resolve + load the checkpoint and report, without running inference.")
    args = parser.parse_args()

    source_path = args.source

    model = args.model or os.environ.get("AUTOSHORTS_PANNS_MODEL", "cnn14")
    model_version = args.model_version or os.environ.get("AUTOSHORTS_PANNS_MODEL_VERSION", "1.0")
    frame_threshold = (args.frame_threshold if args.frame_threshold is not None
                       else float(os.environ.get("AUTOSHORTS_PANNS_FRAME_THRESHOLD", "0.1")))
    min_event_duration = (args.min_event_duration if args.min_event_duration is not None
                          else float(os.environ.get("AUTOSHORTS_PANNS_MIN_EVENT_DURATION",
                                                     str(MIN_EVENT_DURATION_SEC))))
    use_stub = args.stub or os.environ.get("AUTOSHORTS_PANNS_STUB", "0") not in ("0", "false", "off")

    if not os.path.isfile(source_path):
        payload = {
            "ok": False, "events": [],
            "error": {"code": "source_not_found", "message": f"source not found: {source_path}"},
            "diagnostics": {"mode": "stub" if use_stub else "real"},
        }
        log(f"ERROR source_not_found: {source_path}")
        emit(payload)
        return 2

    try:
        if use_stub:
            events, diagnostics = detect_reactions_stub(source_path, model, frame_threshold, model_version)
        elif args.probe:
            ckpt = resolve_checkpoint()
            build_detector(ckpt, model, device=args.device)
            events, diagnostics = [], {
                "mode": "probe", "modelLoaded": True, "checkpointPath": ckpt,
                "checkpointBytes": os.path.getsize(ckpt),
            }
        else:
            events, diagnostics = detect_reactions(
                source_path, model, frame_threshold, min_event_duration,
                model_version, device=args.device
            )
    except PannsError as exc:
        log(f"ERROR {exc.code}: {exc.message}")
        emit({"ok": False, "events": [],
              "error": {"code": exc.code, "message": exc.message},
              "diagnostics": {"mode": "stub" if use_stub else "real"}})
        return 2
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        # Deliberately narrow: these are the failures a real run can produce, and
        # they are reported structurally. No bare `except Exception`.
        log(f"ERROR unexpected_{type(exc).__name__}: {exc}")
        emit({"ok": False, "events": [],
              "error": {"code": f"unexpected_{type(exc).__name__}", "message": str(exc)},
              "diagnostics": {"mode": "stub" if use_stub else "real"}})
        return 2

    payload = {"ok": True, "events": events, "diagnostics": diagnostics}
    emit(payload)

    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except OSError as exc:
            log(f"WARNING could not write --output {args.output}: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
