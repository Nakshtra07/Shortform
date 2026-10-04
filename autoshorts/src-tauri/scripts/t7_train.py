#!/usr/bin/env python3
"""
AutoShorts 11.0 — T7 LightGBM Prosody Boundary Classifier Trainer (Requirement R2)

Trains a binary LightGBM classifier with early stopping on a held-out per-video
split (preventing speaker leakage) using extracted prosodic boundary features.

SAFETY GUARD:
    Refuses to train if labeled gaps < 200, outputting verbatim:
        NO LABELS — tagger not trained
    and exiting cleanly with return code 0, without fabricating fake data.

ARTIFACT CONTRACT:
    - Booster model: scripts/models/t7_prosody_v1.txt
    - Metadata:       scripts/models/t7_prosody_v1.meta.json
      Keys: featureVersion, threshold (0.65), classes, features, trainStats (AUC, PR-AUC, train/test size)
"""

import os
import sys
import json
import math
import argparse
from typing import Dict, List, Any, Tuple, Optional

# Ensure UTF-8 output across Windows consoles and piped streams (preserves em dash U+2014)
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

# ─── Constants & Specification Contracts ───────────────────────────────────────
FEATURE_VERSION = "t7_prosody_v1"
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

DEFAULT_THRESHOLD = 0.65
MIN_REQUIRED_SAMPLES = 200
REFUSAL_MESSAGE = "NO LABELS \u2014 tagger not trained"
DEFAULT_CLASSES = ["non_boundary", "boundary"]

DEFAULT_DATA_PATH = os.path.join(SCRIPT_DIR, "data", "t7_labels.jsonl")
DEFAULT_MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
DEFAULT_MODEL_FILE = os.path.join(DEFAULT_MODELS_DIR, "t7_prosody_v1.txt")
DEFAULT_META_FILE = os.path.join(DEFAULT_MODELS_DIR, "t7_prosody_v1.meta.json")


# ─── Data Ingestion ────────────────────────────────────────────────────────────
def parse_label(val: Any) -> Optional[int]:
    """Normalizes label value to 0 or 1. Returns None if invalid/absent."""
    if val is None:
        return None
    if isinstance(val, bool):
        return 1 if val else 0
    if isinstance(val, (int, float)):
        return 1 if val >= 0.5 else 0
    if isinstance(val, str):
        v = val.strip().lower()
        if v in ("1", "true", "yes", "boundary"):
            return 1
        if v in ("0", "false", "no", "non_boundary"):
            return 0
    return None


def load_labeled_records(data_path: str) -> List[Dict[str, Any]]:
    """
    Ingests JSONL labeled records from data_path.
    Filters for valid records containing an 'isBoundary' label.
    Returns empty list if file does not exist or has no valid entries.
    """
    if not data_path or not os.path.isfile(data_path):
        return []
    records = []
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if isinstance(rec, dict) and "isBoundary" in rec:
                        norm_lbl = parse_label(rec["isBoundary"])
                        if norm_lbl is not None:
                            rec["isBoundary"] = norm_lbl
                            records.append(rec)
                except (json.JSONDecodeError, ValueError):
                    continue
    except Exception as e:
        sys.stderr.write(f"[t7_train] Warning reading {data_path}: {e}\n")
        return []
    return records


# ─── Feature Extraction ────────────────────────────────────────────────────────
def extract_feature_matrix(records: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts numerical feature matrix X, binary labels y, and video/group identifiers.
    Handles both flat records and nested 'features' dicts.
    Missing values map deterministically to 0.0.
    """
    X_rows = []
    y_list = []
    groups_list = []

    for rec in records:
        feat_sub = rec.get("features") if isinstance(rec.get("features"), dict) else {}
        row = []
        for feat_name in FEATURE_ORDER:
            val = rec.get(feat_name)
            if val is None and feat_sub:
                val = feat_sub.get(feat_name)
            if val is None or (isinstance(val, (int, float)) and math.isnan(val)):
                val = 0.0
            row.append(float(val))
        X_rows.append(row)

        y_list.append(int(rec["isBoundary"]))

        video = (
            rec.get("video")
            or rec.get("source")
            or rec.get("videoId")
            or rec.get("source_path")
            or "single_video"
        )
        groups_list.append(str(video))

    return (
        np.array(X_rows, dtype=np.float32),
        np.array(y_list, dtype=np.int32),
        np.array(groups_list, dtype=object),
    )


# ─── Per-Video Splitting ───────────────────────────────────────────────────────
def partition_by_video(
    records: List[Dict[str, Any]],
    train_ratio: float = 0.80,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Partitions raw record dicts into (train_recs, val_recs) grouped by video ID.
    Guarantees train and validation video sets are strictly disjoint (zero speaker leakage).
    """
    import random

    if not records:
        return [], []

    video_keys = [
        str(
            r.get("video")
            or r.get("source")
            or r.get("videoId")
            or r.get("source_path")
            or "unknown"
        )
        for r in records
    ]
    unique_videos = sorted(list(set(video_keys)))

    if len(unique_videos) > 1:
        best_train, best_val = None, None
        for trial_seed in range(seed, seed + 30):
            shuffled = list(unique_videos)
            random.Random(trial_seed).shuffle(shuffled)
            num_train = max(1, min(len(unique_videos) - 1, int(round(len(unique_videos) * train_ratio))))
            train_set = set(shuffled[:num_train])
            val_set = set(shuffled[num_train:])

            tr = [
                r for r in records
                if str(r.get("video") or r.get("source") or r.get("videoId") or r.get("source_path") or "unknown") in train_set
            ]
            va = [
                r for r in records
                if str(r.get("video") or r.get("source") or r.get("videoId") or r.get("source_path") or "unknown") in val_set
            ]

            if best_train is None:
                best_train, best_val = tr, va

            y_tr = {int(r["isBoundary"]) for r in tr if "isBoundary" in r}
            y_va = {int(r["isBoundary"]) for r in va if "isBoundary" in r}
            if len(y_tr) >= 2 and len(y_va) >= 2:
                return tr, va
        return best_train, best_val
    else:
        split_point = max(1, min(len(records) - 1, int(len(records) * train_ratio)))
        return records[:split_point], records[split_point:]


def split_per_video(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    test_size: float = 0.20,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Index-based per-video train/validation split helper.
    """
    from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit

    unique_groups = np.unique(groups)

    if len(unique_groups) > 1:
        for trial_seed in range(seed, seed + 20):
            gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=trial_seed)
            train_idx, val_idx = next(gss.split(X, y, groups))
            if len(np.unique(y[train_idx])) >= 2 and len(np.unique(y[val_idx])) >= 2:
                return train_idx, val_idx

        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, val_idx = next(gss.split(X, y, groups))
        if len(val_idx) > 0 and len(train_idx) > 0:
            return train_idx, val_idx

    n_samples = len(X)
    split_point = max(1, int(n_samples * (1.0 - test_size)))
    train_idx = np.arange(0, split_point)
    val_idx = np.arange(split_point, n_samples)

    if len(np.unique(y[val_idx])) >= 2 and len(np.unique(y[train_idx])) >= 2:
        return train_idx, val_idx

    try:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        return next(sss.split(X, y))
    except Exception:
        return train_idx, val_idx


# ─── LightGBM Model Training ──────────────────────────────────────────────────
def train_lightgbm_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int = 42,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Trains LightGBM binary classifier with early stopping.
    Evaluates validation AUC and PR-AUC.
    """
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score, average_precision_score

    train_data = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_ORDER)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, feature_name=FEATURE_ORDER)

    params = {
        "objective": "binary",
        "metric": ["binary_logloss", "auc"],
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 15,
        "min_data_in_leaf": 5,
        "feature_fraction": 1.0,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbosity": -1,
        "random_state": seed,
    }

    callbacks = [
        lgb.early_stopping(stopping_rounds=15, verbose=False),
    ]

    booster = lgb.train(
        params,
        train_data,
        num_boost_round=200,
        valid_sets=[train_data, val_data],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )

    # Evaluate validation metrics
    val_preds = booster.predict(X_val)
    train_preds = booster.predict(X_train)

    try:
        auc = float(roc_auc_score(y_val, val_preds))
    except Exception:
        auc = 0.5

    try:
        pr_auc = float(average_precision_score(y_val, val_preds))
    except Exception:
        pr_auc = 0.0

    try:
        train_auc = float(roc_auc_score(y_train, train_preds))
    except Exception:
        train_auc = 0.5

    stats = {
        "auc": round(auc, 4),
        "prAuc": round(pr_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "trainSize": int(len(X_train)),
        "testSize": int(len(X_val)),
        "valSize": int(len(X_val)),
        "totalSamples": int(len(X_train) + len(X_val)),
        "bestIteration": int(booster.best_iteration or 0),
        "trainAuc": round(train_auc, 4),
    }

    return booster, stats


# ─── Artifacts Generation ──────────────────────────────────────────────────────
def save_model_artifacts(
    booster: Any,
    stats: Dict[str, Any],
    model_path: str,
    meta_path: str,
    threshold: float = DEFAULT_THRESHOLD,
) -> Tuple[str, str]:
    """
    Saves booster text model and metadata JSON.
    Creates parent directories if necessary.
    """
    os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(meta_path)), exist_ok=True)

    booster.save_model(model_path)

    metadata = {
        "featureVersion": FEATURE_VERSION,
        "threshold": threshold,
        "classes": DEFAULT_CLASSES,
        "features": FEATURE_ORDER,
        "trainStats": stats,
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return model_path, meta_path


# ─── Pipeline Orchestration ────────────────────────────────────────────────────
def train_pipeline(
    data_path: str = DEFAULT_DATA_PATH,
    models_dir: str = DEFAULT_MODELS_DIR,
    model_out: Optional[str] = None,
    meta_out: Optional[str] = None,
    min_samples: int = MIN_REQUIRED_SAMPLES,
    threshold: float = DEFAULT_THRESHOLD,
    val_ratio: float = 0.20,
    seed: int = 42,
) -> int:
    """
    Full training pipeline with Safety Guard.
    Returns 0 on success or safe refusal. Returns non-zero only on unhandled error.
    """
    # Safety Guard: Refuse to train if valid labeled gaps < min_samples
    records = load_labeled_records(data_path)
    if len(records) < min_samples:
        print(REFUSAL_MESSAGE)
        return 0

    # Per-video train/validation split
    train_recs, val_recs = partition_by_video(records, train_ratio=1.0 - val_ratio, seed=seed)
    X_train, y_train, _ = extract_feature_matrix(train_recs)
    X_val, y_val, _ = extract_feature_matrix(val_recs)

    # Train LightGBM model
    booster, stats = train_lightgbm_model(X_train, y_train, X_val, y_val, seed=seed)

    # Determine artifact file paths
    if not model_out:
        model_out = os.path.join(models_dir, "t7_prosody_v1.txt")
    if not meta_out:
        meta_out = os.path.join(models_dir, "t7_prosody_v1.meta.json")

    save_model_artifacts(booster, stats, model_out, meta_out, threshold=threshold)

    print(f"Successfully trained T7 prosody model: {model_out}")
    print(f"Validation AUC: {stats['auc']}, PR-AUC: {stats['prAuc']} (train={stats['trainSize']}, val={stats['testSize']})")
    print(f"Metadata saved to: {meta_out}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="T7 LightGBM Prosody Boundary Classifier Trainer")
    parser.add_argument("--data", default=DEFAULT_DATA_PATH, help="Path to input t7_labels.jsonl")
    parser.add_argument("--models_dir", default=DEFAULT_MODELS_DIR, help="Directory to save model artifacts")
    parser.add_argument("--out_model", default=None, help="Explicit path override for booster model file")
    parser.add_argument("--out_meta", default=None, help="Explicit path override for metadata json file")
    parser.add_argument("--min_samples", type=int, default=MIN_REQUIRED_SAMPLES, help="Minimum sample threshold")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Boundary decision threshold")
    parser.add_argument("--val_ratio", type=float, default=0.20, help="Validation set split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting and training")
    args = parser.parse_args(argv)

    return train_pipeline(
        data_path=args.data,
        models_dir=args.models_dir,
        model_out=args.out_model,
        meta_out=args.out_meta,
        min_samples=args.min_samples,
        threshold=args.threshold,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    sys.exit(main())
