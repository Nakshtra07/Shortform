#!/usr/bin/env python3
"""
AutoShorts 11.0 Phase 3 — T7 Prosodic Phrase-Boundary feature pipeline.

Provides:
    1. per-word-gap prosodic features (pause length, sentence ending, clause ending,
       speaker change, F0 slope, energy drop, VAD silence fraction, deterministic rule prior)
       computed from real audio;
    2. a deterministic rule-based prior P(boundary) (documented weights) usable for
       offline annotation triage and weak labeling;
    3. --label mode for producing annotation-ready or weak-labeled JSONL datasets
       (written to scripts/data/t7_labels.jsonl or custom path);
    4. --predict mode for sidecar inference using trained LightGBM model with graceful
       fallback to empty boundaries when model is missing or corrupt.
"""

import os
import sys
import json
import math
import argparse
from typing import List, Dict, Any, Optional

# Ensure UTF-8 output across Windows consoles and piped streams
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import numpy as np

FEATURE_VERSION = "t7_prosody_v1"
SAMPLE_RATE = 16000

FEATURE_ORDER = [
    "pauseSec",
    "endsSentence",
    "endsClause",
    "speakerChange",
    "f0SlopePre",
    "energyDropDb",
    "vadSilenceFrac",
    "pBoundaryRule",
]

DEFAULT_LABEL_THRESHOLD = 0.65
SENTENCE_TERMINATORS = (".", "!", "?", ";", ":", "।", "॥", "…")


def decode_audio(source_path: Optional[str], start_sec: float, end_sec: float) -> Optional[np.ndarray]:
    """Decode audio snippet to mono float32 at 16 kHz using ffmpeg."""
    if not source_path or not os.path.exists(source_path):
        return None
    if end_sec <= start_sec or round(end_sec - start_sec, 3) <= 0.0 or end_sec <= 0.0:
        return None
    try:
        import subprocess
        cmd = ["ffmpeg", "-v", "error", "-ss", f"{max(0.0, start_sec):.3f}",
               "-t", f"{max(0.0, end_sec - start_sec):.3f}", "-i", source_path,
               "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-"]
        raw = subprocess.run(cmd, capture_output=True, timeout=120).stdout
        return np.frombuffer(raw, dtype=np.float32) if raw else None
    except Exception:
        return None


def f0_slope_pre(x: np.ndarray, t0: float, t1: float) -> float:
    """Crude voicing-pooled F0 slope (Hz/s) over [t0,t1): autocorrelation on
    40 ms frames, voiced frames only. 0.0 when unvoiced/short."""
    i0, i1 = int(t0 * SAMPLE_RATE), int(t1 * SAMPLE_RATE)
    seg = x[i0:i1]
    frame = int(0.04 * SAMPLE_RATE)
    if seg.size < frame * 2:
        return 0.0
    f0s, times = [], []
    lag_min, lag_max = int(SAMPLE_RATE / 400), int(SAMPLE_RATE / 70)
    for s in range(0, seg.size - frame, frame // 2):
        fr = seg[s:s + frame].astype(np.float64)
        if float(np.sqrt(np.mean(fr ** 2))) < 1e-4:
            continue
        ac = np.correlate(fr, fr, mode="full")[frame - 1:]
        if ac[lag_min:lag_max].size == 0:
            continue
        peak = lag_min + int(np.argmax(ac[lag_min:lag_max]))
        if ac[peak] <= 0:
            continue
        f0s.append(SAMPLE_RATE / peak)
        times.append((s + frame / 2) / SAMPLE_RATE)
    if len(f0s) < 2:
        return 0.0
    z = np.polyfit(times, f0s, 1)
    return float(z[0])


def energy_contour_db(x: np.ndarray, t0: float, t1: float, hop: float = 0.02) -> List[float]:
    """RMS energy contour in dBFS over [t0, t1) using hop-second windows."""
    i0, i1 = int(t0 * SAMPLE_RATE), int(t1 * SAMPLE_RATE)
    seg = x[i0:i1]
    h = int(hop * SAMPLE_RATE)
    if seg.size < h:
        return []
    out = []
    for s in range(0, seg.size - h, h):
        r = float(np.sqrt(np.mean(seg[s:s + h].astype(np.float64) ** 2)))
        out.append(20.0 * math.log10(max(r, 1e-10)))
    return out


def vad_silence_frac(x: np.ndarray, t0: float, t1: float, pre_db: Optional[List[float]] = None,
                     threshold_db: float = -40.0, hop: float = 0.02) -> float:
    """
    Fraction of 20 ms frames in [t0, t1) that are silent.
    Silence threshold is adaptive: min(threshold_db, speech_mean - 15 dB) if pre_db has speech energy.
    Returns float in [0.0, 1.0].
    """
    if x is None or t1 <= t0:
        return 0.0
    thresh = threshold_db
    if pre_db and len(pre_db) > 0:
        m = float(np.mean(pre_db))
        if m > -60.0:  # Only adapt threshold if preceding audio had detectable energy
            thresh = min(threshold_db, m - 15.0)
    db_frames = energy_contour_db(x, t0, t1, hop=hop)
    if not db_frames:
        i0, i1 = int(t0 * SAMPLE_RATE), int(t1 * SAMPLE_RATE)
        seg = x[i0:i1]
        if seg.size == 0:
            return 0.0
        r = float(np.sqrt(np.mean(seg.astype(np.float64) ** 2)))
        db = 20.0 * math.log10(max(r, 1e-10))
        return 1.0 if db < thresh else 0.0
    silent = sum(1 for d in db_frames if d < thresh)
    return round(float(silent) / len(db_frames), 3)


def features_to_row(rec: Dict[str, Any]) -> List[float]:
    """Convert a gap record dict into an ordered feature vector for LightGBM."""
    row = []
    feats = rec.get("features", rec)
    for k in FEATURE_ORDER:
        v = feats.get(k)
        row.append(float(v) if v is not None else 0.0)
    return row


def build_gap_records(source_path: Optional[str], start_sec: float, end_sec: float,
                      words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Annotation-ready per-gap prosody records (word indices preserved)."""
    x = decode_audio(source_path, start_sec, end_sec) if source_path else None
    records = []
    for i in range(1, len(words)):
        prev_w, next_w = words[i - 1], words[i]
        try:
            gap_start = float(prev_w["end"])
            gap_end = float(next_w["start"])
        except (KeyError, TypeError, ValueError):
            continue
        pause = max(0.0, gap_end - gap_start)
        spk_prev = prev_w.get("speaker")
        spk_next = next_w.get("speaker")
        speaker_change = 1 if (
            spk_prev is not None
            and spk_next is not None
            and str(spk_prev).strip() != ""
            and str(spk_next).strip() != ""
            and str(spk_prev).strip() != str(spk_next).strip()
        ) else 0
        rec = {
            "gapIdx": i - 1,
            "wordIdxBefore": i - 1,
            "wordIdxAfter": i,
            "prevText": prev_w.get("text", ""),
            "nextText": next_w.get("text", ""),
            "pauseSec": round(pause, 3),
            "endsSentence": 1 if str(prev_w.get("text", "")).rstrip().endswith(SENTENCE_TERMINATORS) else 0,
            "endsClause": 1 if str(prev_w.get("text", "")).rstrip().endswith(",") else 0,
            "speakerChange": speaker_change,
        }
        if pause == 0:
            rec["f0SlopePre"] = 0.0
            rec["energyDropDb"] = 0.0
            rec["vadSilenceFrac"] = 0.0
        elif x is not None:
            rec["f0SlopePre"] = round(f0_slope_pre(x, max(0.0, gap_start - 0.6 - start_sec), gap_start - start_sec), 2)
            pre_db = energy_contour_db(x, max(0.0, gap_start - 0.4 - start_sec), gap_start - start_sec)
            gap_db = energy_contour_db(x, gap_start - start_sec, gap_end - start_sec)
            if pre_db and gap_db:
                rec["energyDropDb"] = round(float(np.mean(pre_db) - np.mean(gap_db)), 2)
            else:
                rec["energyDropDb"] = None
            rec["vadSilenceFrac"] = vad_silence_frac(x, gap_start - start_sec, gap_end - start_sec, pre_db=pre_db)
        else:
            rec["f0SlopePre"] = None
            rec["energyDropDb"] = None
            rec["vadSilenceFrac"] = None

        # Deterministic rule-based prior (annotation triage / weak labeling):
        # long pauses + punctuation dominate.
        p = 0.0
        p += min(1.0, pause / 0.6) * 0.55
        p += rec["endsSentence"] * 1.0
        p += rec["endsClause"] * 0.45
        p += 0.15 * (1 if (rec.get("energyDropDb") or 0) > 6.0 else 0)
        p -= rec["speakerChange"] * 1.0  # speaker turns are their own unit
        rec["pBoundaryRule"] = round(max(0.0, min(1.0, p)), 3)
        records.append(rec)
    return records


def run_label_mode(args, words: List[Dict[str, Any]]):
    records = build_gap_records(args.source_path, args.start_ms / 1000.0,
                                args.end_ms / 1000.0, words)
    human_annotations = {}
    if getattr(args, "annotations", None) and os.path.exists(args.annotations):
        with open(args.annotations, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if "gapIdx" in entry:
                        idx = int(entry["gapIdx"])
                        is_b = 1 if entry.get("isBoundary") in (1, True, "1", "true") else 0
                        human_annotations[idx] = is_b
                except Exception:
                    continue

    dataset_entries = []
    video_name = os.path.basename(args.source_path)
    for rec in records:
        gap_idx = rec["gapIdx"]
        if gap_idx in human_annotations:
            is_boundary = human_annotations[gap_idx]
            label_type = "human"
            confidence = 1.0
        else:
            p_rule = float(rec.get("pBoundaryRule", 0.0))
            is_boundary = 1 if p_rule >= DEFAULT_LABEL_THRESHOLD else 0
            label_type = "weak"
            confidence = round(p_rule if is_boundary == 1 else (1.0 - p_rule), 3)

        entry = {
            "video": video_name,
            "source": video_name,
            "sourcePath": args.source_path,
            "gapIdx": rec["gapIdx"],
            "wordIdxBefore": rec["wordIdxBefore"],
            "wordIdxAfter": rec["wordIdxAfter"],
            "prevText": rec["prevText"],
            "nextText": rec["nextText"],
            "pauseSec": rec["pauseSec"],
            "endsSentence": rec["endsSentence"],
            "endsClause": rec["endsClause"],
            "speakerChange": rec["speakerChange"],
            "f0SlopePre": rec["f0SlopePre"],
            "energyDropDb": rec["energyDropDb"],
            "vadSilenceFrac": rec["vadSilenceFrac"],
            "pBoundaryRule": rec["pBoundaryRule"],
            "features": {
                "pauseSec": rec["pauseSec"],
                "endsSentence": rec["endsSentence"],
                "endsClause": rec["endsClause"],
                "speakerChange": rec["speakerChange"],
                "f0SlopePre": rec["f0SlopePre"],
                "energyDropDb": rec["energyDropDb"],
                "vadSilenceFrac": rec["vadSilenceFrac"],
                "pBoundaryRule": rec["pBoundaryRule"],
            },
            "isBoundary": is_boundary,
            "labelType": label_type,
            "confidence": confidence,
        }
        dataset_entries.append(entry)

    out_path = args.out if args.out else os.path.join(SCRIPT_DIR, "data", "t7_labels.jsonl")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    mode = "a" if (getattr(args, "append", False) or args.out is None) else "w"
    with open(out_path, mode, encoding="utf-8") as f:
        for item in dataset_entries:
            f.write(json.dumps(item) + "\n")

    sys.stderr.write(f"[T7Prosody] Labeled {len(dataset_entries)} gaps -> {out_path}\n")
    print(json.dumps({"status": "ok", "labeledCount": len(dataset_entries), "out": out_path}))


def resolve_model_path(model_arg: Optional[str]) -> Optional[str]:
    if model_arg:
        return model_arg
    env_path = os.environ.get("AUTOSHORTS_T7_PROSODY_MODEL")
    if env_path:
        return env_path
    default_path = os.path.join(SCRIPT_DIR, "models", "t7_prosody_v1.txt")
    return default_path


def load_model(model_path: str):
    if not model_path or not os.path.exists(model_path):
        return None, None
    meta_path = os.path.splitext(model_path)[0] + ".meta.json"
    meta = {}
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("featureVersion") and meta.get("featureVersion") != FEATURE_VERSION:
                sys.stderr.write(
                    f"[T7Prosody] Model featureVersion mismatch: {meta.get('featureVersion')} != {FEATURE_VERSION}; fallback\n"
                )
                return None, None
        except Exception as e:
            sys.stderr.write(f"[T7Prosody] Metadata read failed ({e}); fallback\n")
            return None, None
    try:
        import lightgbm as lgb
        booster = lgb.Booster(model_file=model_path)
        return booster, meta
    except Exception as e:
        sys.stderr.write(f"[T7Prosody] Model load failed ({e}); fallback\n")
        return None, None


def emit_predict_fallback(out_path: Optional[str] = None) -> None:
    """Emit typed fallback payload for sidecar prediction mode."""
    payload = {
        "modelScored": False,
        "model": None,
        "boundaries": [],
    }
    text = json.dumps(payload, indent=1)
    if out_path:
        try:
            out_dir = os.path.dirname(os.path.abspath(out_path))
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(text)
            return
        except Exception:
            pass
    print(text)


def run_predict_mode(args, words: List[Dict[str, Any]]):
    try:
        model_path = resolve_model_path(getattr(args, "model", None))
        booster, meta = load_model(model_path)
        if booster is None:
            emit_predict_fallback(getattr(args, "out", None))
            return

        if words is None or not isinstance(words, list):
            emit_predict_fallback(getattr(args, "out", None))
            return

        records = build_gap_records(args.source_path, args.start_ms / 1000.0,
                                    args.end_ms / 1000.0, words)
        threshold = float((meta or {}).get("threshold", 0.65))
        boundaries = []
        if records:
            X = np.array([features_to_row(r) for r in records], dtype=np.float32)
            preds = booster.predict(X)
            if len(preds.shape) > 1 and preds.shape[1] > 1:
                preds = preds[:, 1]
            for rec, p in zip(records, preds):
                p_val = round(float(p), 4)
                boundaries.append({
                    "gapIdx": rec["gapIdx"],
                    "pBoundary": p_val,
                    "isBoundary": 1 if p_val >= threshold else 0,
                    "features": {
                        "pauseSec": rec["pauseSec"],
                        "endsSentence": rec["endsSentence"],
                        "endsClause": rec["endsClause"],
                        "speakerChange": rec["speakerChange"],
                        "f0SlopePre": rec["f0SlopePre"],
                        "energyDropDb": rec["energyDropDb"],
                        "vadSilenceFrac": rec["vadSilenceFrac"],
                        "pBoundaryRule": rec["pBoundaryRule"],
                    },
                })

        payload = {
            "modelScored": True,
            "featureVersion": FEATURE_VERSION,
            "model": os.path.basename(model_path) if model_path else "unknown",
            "boundaries": boundaries,
        }
        text = json.dumps(payload, indent=1)
        if args.out:
            out_dir = os.path.dirname(os.path.abspath(args.out))
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(args.out, "w", encoding="utf-8") as f:
                f.write(text)
        else:
            print(text)
    except Exception as e:
        sys.stderr.write(f"[T7Prosody] Predict mode fallback due to unexpected error: {e}\n")
        emit_predict_fallback(getattr(args, "out", None))


def main():
    parser = argparse.ArgumentParser(description="T7 prosodic boundary feature pipeline")
    parser.add_argument("source_path", help="Path to media file")
    parser.add_argument("start_ms", type=float, help="Start time in milliseconds")
    parser.add_argument("end_ms", type=float, help="End time in milliseconds")
    parser.add_argument("words_json", help="Path to transcript JSON containing words array")
    parser.add_argument("--out", default=None, help="Optional output path")

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--label", action="store_true", help="Run dataset labeling pipeline and output JSONL")
    mode_group.add_argument("--predict", action="store_true", help="Run model inference and output predictions JSON")

    parser.add_argument("--annotations", default=None, help="Path to human boundary annotations JSONL for --label mode")
    parser.add_argument("--append", action="store_true", default=False, help="Append to destination file in --label mode")
    parser.add_argument("--model", default=None, help="Path to trained LightGBM model for --predict mode")

    args = parser.parse_args()

    try:
        with open(args.words_json, "r", encoding="utf-8") as f:
            words = json.load(f)
        if isinstance(words, dict):
            words = words.get("words", [])
        if words is None or not isinstance(words, list):
            raise ValueError("Invalid words JSON: expected list or dict containing 'words' list")
    except Exception:
        if args.predict:
            emit_predict_fallback(getattr(args, "out", None))
            sys.exit(0)
        raise

    if args.label:
        run_label_mode(args, words)
    elif args.predict:
        run_predict_mode(args, words)
    else:
        # Default mode (Phase 3 baseline backward-compatible)
        records = build_gap_records(args.source_path, args.start_ms / 1000.0,
                                    args.end_ms / 1000.0, words)
        payload = {
            "source": os.path.basename(args.source_path),
            "featureVersion": FEATURE_VERSION,
            "learnedModel": "DATA-BLOCKED (no labeled T7 chunking dataset; see Phase 3 report)",
            "gaps": records,
        }
        text = json.dumps(payload, indent=1)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                f.write(text)
        else:
            print(text)


if __name__ == "__main__":
    main()
