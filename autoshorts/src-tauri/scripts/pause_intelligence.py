#!/usr/bin/env python3
"""
AutoShorts 11.0 Phase 3 — Pause Intelligence (shared feature infrastructure)

Shared, cacheable audio analysis used by learned Smart Pacing (Phase 3) and
available to T7 prosody (future). Implements the Atria-recommended stack:

    AUDIO (16 kHz mono, decoded once)
      → Silero VAD v6.x  (speech probability + speech regions)
      → per-gap acoustic features (extends breath_detect.py's evidence)
      → LightGBM P(removable) scoring (model file optional)

ML/PROBABILITIES ARE SIGNALS ONLY. Nothing here edits a timeline; the
deterministic Smart Pacing safety rules in smart_pacing.py retain full
authority. Every function fails soft: missing/corrupt model, missing audio,
or import failure return None/empty and the caller falls back to the
existing deterministic behavior.

Model artifact contract (pause_classifier_v1):
    { "modelFile": "<path>.txt" (LightGBM Booster), "featureVersion": "1",
      "threshold": <float>, "trainedAt": <iso>, "datasetVersion": <str> }
The model is reloaded deterministically; any mismatch or load failure is a
documented fallback to existing thresholds.
"""

import os
import sys
import json
import math
import hashlib
import subprocess
from typing import Dict, List, Optional, Tuple

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# ─── Versions (cache keys + model metadata) ───────────────────────────────────
FEATURE_VERSION = "1"
PAUSE_MODEL_VERSION = "pause_classifier_v1"
SILERO_VAD_PKG = "silero-vad 6.2.3 (MIT)"

SAMPLE_RATE = 16000

# Flag semantics match the Rust env_flag: unset enables, 0/false/off disables.
def _flag_enabled(name: str) -> bool:
    v = os.environ.get(name)
    if v is None:
        return True
    return v.strip().lower() not in ("0", "false", "off")


def pause_learning_enabled() -> bool:
    return _flag_enabled("AUTOSHORTS_SMART_PACING_LEARNED")


# ─── Audio decoding (single pass, reused by features and VAD) ─────────────────

def decode_audio(source_path: str, start_sec: float, end_sec: float) -> Optional[np.ndarray]:
    """Decode [start_sec, end_sec) to f32 mono 16 kHz. None on failure."""
    try:
        cmd = [
            "ffmpeg", "-v", "error", "-ss", f"{max(0.0, start_sec):.3f}",
            "-t", f"{max(0.0, end_sec - start_sec):.3f}", "-i", source_path,
            "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
            "-f", "f32le", "-",
        ]
        raw = subprocess.run(cmd, capture_output=True, timeout=120).stdout
        if not raw:
            return None
        return np.frombuffer(raw, dtype=np.float32)
    except Exception:
        return None


def source_hash(path: str) -> str:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""


# ─── Silero VAD (lazy, cached per process) ────────────────────────────────────

_VAD_MODEL = None
_VAD_TRIED = False


def get_silero_vad():
    """Load Silero VAD once. Returns None when unavailable (fallback: none)."""
    global _VAD_MODEL, _VAD_TRIED
    if _VAD_TRIED:
        return _VAD_MODEL
    _VAD_TRIED = True
    try:
        import torch  # noqa: F401
        from silero_vad import load_silero_vad
        _VAD_MODEL = load_silero_vad()
    except Exception as e:
        sys.stderr.write(f"[PauseIntel] Silero VAD unavailable ({e}); VAD features disabled\n")
        _VAD_MODEL = None
    return _VAD_MODEL


def silero_speech_regions(x: np.ndarray) -> List[Tuple[float, float]]:
    """Speech regions (seconds, relative to buffer start). [] on failure."""
    model = get_silero_vad()
    if model is None or x is None or x.size == 0:
        return []
    try:
        import torch
        from silero_vad import get_speech_timestamps
        ts = get_speech_timestamps(torch.from_numpy(x), model, sampling_rate=SAMPLE_RATE)
        return [(t["start"] / SAMPLE_RATE, t["end"] / SAMPLE_RATE) for t in ts]
    except Exception as e:
        sys.stderr.write(f"[PauseIntel] VAD inference failed ({e}); no speech regions\n")
        return []


def silero_speech_probability(x: np.ndarray, t0: float, t1: float) -> Optional[float]:
    """Mean speech probability over [t0, t1) seconds within buffer x."""
    model = get_silero_vad()
    if model is None or x is None or x.size == 0:
        return None
    try:
        import torch
        i0 = max(0, int(t0 * SAMPLE_RATE))
        i1 = min(x.size, int(t1 * SAMPLE_RATE))
        if i1 - i0 < int(0.03 * SAMPLE_RATE):
            return None
        chunk = x[i0:i1]
        # Silero consumes 512-sample (32 ms) windows at 16 kHz.
        win = 512
        probs = []
        with torch.no_grad():
            for s in range(0, chunk.size - win + 1, win):
                probs.append(float(model(torch.from_numpy(chunk[s:s + win]), SAMPLE_RATE).item()))
        return float(np.mean(probs)) if probs else None
    except Exception:
        return None


# ─── Acoustic features (numpy-only core + light librosa-style extras) ─────────

def _rms_db(x: np.ndarray) -> float:
    if x is None or x.size == 0:
        return -120.0
    r = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    return 20.0 * math.log10(max(r, 1e-10))


def gap_acoustic_features(x: np.ndarray, gap_t0: float, gap_t1: float,
                          pre_t0: float, pre_t1: float,
                          post_t0: float, post_t1: float) -> Dict[str, float]:
    """Acoustic feature dict for one gap. Pure numpy; mirrors/extends the
    breath_detect.py evidence so both pipelines agree on definitions."""
    def seg(t0, t1):
        i0 = max(0, int(t0 * SAMPLE_RATE)); i1 = min(x.size, int(t1 * SAMPLE_RATE))
        return x[i0:i1] if i1 > i0 else None

    gap = seg(gap_t0, gap_t1)
    pre = seg(pre_t0, pre_t1)
    post = seg(post_t0, post_t1)

    feats: Dict[str, float] = {}
    feats["gapRmsDb"] = _rms_db(gap)
    feats["preSpeechRmsDb"] = _rms_db(pre)
    feats["postSpeechRmsDb"] = _rms_db(post)

    if gap is not None and gap.size > 32:
        # HF ratio (>2 kHz energy fraction) — same 2 kHz split as breath_detect.
        spec = np.abs(np.fft.rfft(gap.astype(np.float64)))
        freqs = np.fft.rfftfreq(gap.size, 1.0 / SAMPLE_RATE)
        total = float(np.sum(spec ** 2)) or 1.0
        feats["hfRatio"] = float(np.sum(spec[freqs > 2000.0] ** 2) / total)
        # Wiener spectral flatness (spectral entropy proxy).
        p = spec ** 2
        p = p / (p.sum() or 1.0)
        p = np.clip(p, 1e-12, None)
        feats["flatness"] = float(np.exp(np.mean(np.log(p))) / np.mean(p))
        # Zero-crossing rate (breath/noise discrimination).
        feats["zcr"] = float(np.mean(np.abs(np.diff(np.sign(gap)))) / 2.0)
        # Envelope dip depth: how far the gap floor sits below its own peaks.
        env = np.abs(gap)
        feats["dipDb"] = _rms_db(gap) - (20.0 * math.log10(max(float(np.percentile(env, 95)), 1e-10)))
    else:
        feats["hfRatio"] = 0.0
        feats["flatness"] = 1.0
        feats["zcr"] = 0.0
        feats["dipDb"] = -120.0

    feats["levelDropDb"] = feats["gapRmsDb"] - max(feats["preSpeechRmsDb"], feats["postSpeechRmsDb"])
    return feats


def structural_gap_features(gap: Dict, words: List[Dict], clip_start: float, clip_end: float) -> Dict[str, float]:
    """Structural/transcript features already implicit in the engine."""
    dur = float(gap.get("durationSec", gap.get("srcEndSec", 0.0)) - gap.get("srcStartSec", 0.0)) \
        if "durationSec" not in gap else float(gap.get("durationSec", 0.0))
    gs = float(gap.get("srcStartSec", 0.0))
    ge = float(gap.get("srcEndSec", 0.0))

    prev_w = gap.get("prevWord") or {}
    next_w = gap.get("nextWord") or {}
    prev_text = str(prev_w.get("text", "") or "")
    next_text = str(next_w.get("text", "") or "")

    prev_speaker = prev_w.get("speaker")
    next_speaker = next_w.get("speaker")
    speaker_change = 1.0 if (prev_speaker and next_speaker and prev_speaker != next_speaker) else 0.0

    ends_sentence = 1.0 if prev_text.rstrip().endswith((".", "!", "?", ";", ":", "।", "॥")) else 0.0
    ends_clause = 1.0 if prev_text.rstrip().endswith(",") else 0.0
    starts_filler = 1.0 if next_text.strip().lower().strip(".,!?") in (
        "um", "uh", "uhm", "umm", "erm", "hmm", "mm", "mmm", "ah", "er", "eh") else 0.0
    ends_filler = 1.0 if prev_text.strip().lower().strip(".,!?") in (
        "um", "uh", "uhm", "umm", "erm", "hmm", "mm", "mmm", "ah", "er", "eh") else 0.0

    # Speech rates around the gap (words/sec over ±2 s windows).
    def rate(t0, t1):
        n = sum(1 for w in words if float(w.get("end", 0.0)) > t0 and float(w.get("start", 0.0)) < t1)
        return n / max(0.1, min(t1, clip_end) - max(t0, clip_start))
    pre_rate = rate(gs - 2.0, gs)
    post_rate = rate(ge, ge + 2.0)

    # Position within the clip (leading/trailing dead-air signal).
    clip_dur = max(0.1, clip_end - clip_start)
    rel_pos = (gs - clip_start) / clip_dur

    return {
        "durationSec": max(0.0, dur),
        "speakerChange": speaker_change,
        "endsSentence": ends_sentence,
        "endsClause": ends_clause,
        "startsFiller": starts_filler,
        "endsFiller": ends_filler,
        "preSpeechRate": pre_rate,
        "postSpeechRate": post_rate,
        "relPosInClip": rel_pos,
    }


# ─── Feature assembly + caching ───────────────────────────────────────────────

def build_gap_features(source_path: str, gaps: List[Dict], words: List[Dict],
                       clip_start: float, clip_end: float,
                       vad_regions: Optional[List[Tuple[float, float]]] = None) -> Optional[List[Dict]]:
    """Full feature vectors for the engine's gaps, on the SOURCE timeline.

    `gaps` come from breath_detect.detect_gaps (absolute source seconds) and
    may carry prevWord/nextWord context. Returns []-compatible list aligned
    with `gaps`, or None when audio cannot be decoded (fallback: no features).
    """
    if not gaps:
        return []
    # Decode the span that covers all gaps ±2 s context, in one pass.
    lo = min(float(g.get("srcStartSec", 0.0)) for g in gaps) - 2.5
    hi = max(float(g.get("srcEndSec", 0.0)) for g in gaps) + 2.5
    lo = max(0.0, lo)
    hi = max(lo + 0.1, hi)
    x = decode_audio(source_path, lo, hi)
    if x is None:
        return None
    if vad_regions is None:
        vad_regions = silero_speech_regions(x)

    out = []
    for g in gaps:
        gs = float(g.get("srcStartSec", 0.0)) - lo
        ge = float(g.get("srcEndSec", 0.0)) - lo
        feats = gap_acoustic_features(x, gs, ge, max(0.0, gs - 1.0), gs, ge, ge + 1.0)
        feats.update(structural_gap_features(g, words, clip_start, clip_end))
        # VAD speech probability inside the gap (low = genuinely quiet).
        p = silero_speech_probability(x, gs, ge)
        feats["vadSpeechProb"] = p if p is not None else -1.0
        # VAD gap coverage: fraction of the gap not covered by speech regions.
        a, b = gs + lo, ge + lo
        speech = sum(max(0.0, min(b, e) - max(a, s)) for s, e in vad_regions)
        feats["vadSilenceFrac"] = 1.0 - min(1.0, speech / max(1e-6, b - a))
        # Echo breath_detect evidence when the caller attached it.
        ev = g.get("evidence") or {}
        for k in ("hfRatio", "aboveFloorDb", "contrastDb", "flatness"):
            if k in ev:
                feats[f"ev{k[0].upper()}{k[1:]}"] = float(ev[k])
        out.append(feats)
    return out


FEATURE_ORDER = [
    "gapRmsDb", "preSpeechRmsDb", "postSpeechRmsDb", "hfRatio", "flatness",
    "zcr", "dipDb", "levelDropDb", "durationSec", "speakerChange",
    "endsSentence", "endsClause", "startsFiller", "endsFiller",
    "preSpeechRate", "postSpeechRate", "relPosInClip", "vadSpeechProb",
    "vadSilenceFrac", "evHfRatio", "evAboveFloorDb", "evContrastDb", "evFlatness",
]


def features_to_row(f: Dict) -> List[float]:
    """Deterministic row vector; missing entries become 0.0."""
    return [float(f.get(k, 0.0)) for k in FEATURE_ORDER]


# ─── Model loading + scoring ──────────────────────────────────────────────────

_MODEL_CACHE: Dict[str, Tuple] = {}


def load_pause_classifier(model_path: str):
    """Load a trained LightGBM pause model. Returns (booster, meta) or None."""
    if not model_path or not os.path.exists(model_path):
        return None
    cached = _MODEL_CACHE.get(model_path)
    if cached is not None:
        return cached
    try:
        import lightgbm as lgb
        booster = lgb.Booster(model_file=model_path)
        meta_path = os.path.splitext(model_path)[0] + ".meta.json"
        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
        if meta.get("featureVersion") != FEATURE_VERSION:
            sys.stderr.write(
                f"[PauseIntel] model featureVersion {meta.get('featureVersion')} != {FEATURE_VERSION}; refusing to load\n")
            return None
        result = (booster, meta)
        _MODEL_CACHE[model_path] = result
        return result
    except Exception as e:
        sys.stderr.write(f"[PauseIntel] pause model load failed ({e}); deterministic thresholds decide\n")
        return None


def score_gaps_with_model(gaps: List[Dict], feature_rows: List[List[float]],
                          model_path: str) -> bool:
    """Attach pRemovable to each gap dict in place. Returns True when the
    model actually scored (False = caller must keep deterministic behavior).

    pRemovable = sum of the removable class probabilities (breath_pause +
    waiting_pause — exactly the categories the deterministic v2 stage may
    remove), per the model metadata."""
    loaded = load_pause_classifier(model_path) if pause_learning_enabled() else None
    if loaded is None:
        return False
    booster, meta = loaded
    try:
        proba = booster.predict(np.array(feature_rows, dtype=np.float64))
        classes = meta.get("classes") or []
        removable = meta.get("removableClasses") or [c for c in classes if c in ("breath_pause", "waiting_pause")]
        rem_idx = [i for i, c in enumerate(classes) if c in removable]
        thr = float(meta.get("threshold", 0.5))
        if rem_idx and proba.ndim == 2:
            p_rem = proba[:, rem_idx].sum(axis=1)
        else:
            p_rem = np.asarray(proba).ravel()
        for g, p in zip(gaps, p_rem):
            p = float(min(1.0, max(0.0, p)))
            g["pRemovable"] = p
            g["pRemovableAboveThreshold"] = bool(p >= thr)
        return True
    except Exception as e:
        sys.stderr.write(f"[PauseIntel] scoring failed ({e}); deterministic thresholds decide\n")
        return False


def find_pause_model() -> Optional[str]:
    """Locate the trained pause model (env override, then default location)."""
    env = os.environ.get("AUTOSHORTS_PAUSE_MODEL")
    if env and os.path.exists(env):
        return env
    candidates = [
        os.path.join(SCRIPT_DIR, "models", "pause_classifier_v1.txt"),
        os.path.join(SCRIPT_DIR, "..", "..", "..", "scratch", "phase3", "models", "pause_classifier_v1.txt"),
        os.path.join(SCRIPT_DIR, "..", "..", "..", "models", "pause_classifier_v1.txt"),
    ]
    for c in candidates:
        norm = os.path.normpath(c)
        if os.path.exists(norm):
            return norm
    return None


# ─── CLI (dataset extraction / scoring debug) ─────────────────────────────────

def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Pause Intelligence feature/scoring debug CLI")
    parser.add_argument("source_path")
    parser.add_argument("start_ms", type=float)
    parser.add_argument("end_ms", type=float)
    parser.add_argument("words_json")
    parser.add_argument("--out", default=None, help="Write features+scores JSON here")
    args = parser.parse_args(argv)

    with open(args.words_json, "r", encoding="utf-8") as f:
        words = json.load(f)
    if isinstance(words, dict):
        words = words.get("words", [])

    import breath_detect
    clip_start = args.start_ms / 1000.0
    clip_end = args.end_ms / 1000.0
    result = breath_detect.detect_gaps(args.source_path, clip_start, clip_end, words)
    gaps = result.get("gaps", []) if result.get("status") == "ok" else []
    feats = build_gap_features(args.source_path, gaps, words, clip_start, clip_end)
    rows = [features_to_row(f) for f in (feats or [])]
    model = find_pause_model()
    scored = score_gaps_with_model(gaps, rows, model) if model else False
    payload = {
        "sourceHash": source_hash(args.source_path),
        "featureVersion": FEATURE_VERSION,
        "gaps": [
            {**g, "features": (feats[i] if feats and i < len(feats) else None)}
            for i, g in enumerate(gaps)
        ],
        "modelScored": scored,
        "model": model,
    }
    text = json.dumps(payload, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
