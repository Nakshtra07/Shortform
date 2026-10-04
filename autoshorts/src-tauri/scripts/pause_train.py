#!/usr/bin/env python3
"""
AutoShorts 11.0 Phase 3 — Multi-Class LightGBM Pause Classifier Trainer

Trains a 6-class LightGBM classifier with early stopping on a held-out per-video
split (preventing speaker/source acoustic leakage) using 23-dimensional pause features.

SAFETY GUARD:
    Refuses to train if labeled records < 200, outputting verbatim:
        NO LABELS — classifier not trained
    and exiting cleanly with return code 0, without fabricating fake data.

ARTIFACT CONTRACT:
    - Booster model: scripts/models/pause_classifier_v1.txt
    - Metadata:       scripts/models/pause_classifier_v1.meta.json
      Keys: modelFile, featureVersion ("1"), threshold (0.65), classes,
            removableClasses, features, trainedAt, datasetVersion, trainStats
"""

import argparse
from datetime import datetime, timezone
import json
import math
import os
import random
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure UTF-8 output across Windows consoles and piped streams (preserves em dash U+2014)
if sys.platform == "win32":
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
FEATURE_VERSION = "1"
MODEL_FILENAME = "pause_classifier_v1.txt"
META_FILENAME = "pause_classifier_v1.meta.json"

CLASSES = [
    "breath_pause",
    "waiting_pause",
    "sentence_pause",
    "normal_word_gap",
    "nonspeech_unknown",
    "speaker_transition",
]

CLASS_TO_INDEX: Dict[str, int] = {name: idx for idx, name in enumerate(CLASSES)}
INDEX_TO_CLASS: Dict[int, str] = {idx: name for idx, name in enumerate(CLASSES)}

REMOVABLE_CLASSES = [
    "breath_pause",
    "waiting_pause",
]
REMOVABLE_INDICES = [CLASS_TO_INDEX[c] for c in REMOVABLE_CLASSES]  # [0, 1]

FEATURE_ORDER = [
    "gapRmsDb", "preSpeechRmsDb", "postSpeechRmsDb", "hfRatio", "flatness",
    "zcr", "dipDb", "levelDropDb", "durationSec", "speakerChange",
    "endsSentence", "endsClause", "startsFiller", "endsFiller",
    "preSpeechRate", "postSpeechRate", "relPosInClip", "vadSpeechProb",
    "vadSilenceFrac", "evHfRatio", "evAboveFloorDb", "evContrastDb", "evFlatness",
]

DEFAULT_THRESHOLD = 0.65
MIN_REQUIRED_SAMPLES = 200
REFUSAL_MESSAGE = "NO LABELS \u2014 classifier not trained"

DEFAULT_DATA_PATH = os.path.join(SCRIPT_DIR, "data", "pause_features_v1.jsonl")
DEFAULT_MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
DEFAULT_MODEL_FILE = os.path.join(DEFAULT_MODELS_DIR, MODEL_FILENAME)
DEFAULT_META_FILE = os.path.join(DEFAULT_MODELS_DIR, META_FILENAME)


# ─── Directory Setup ───────────────────────────────────────────────────────────
def ensure_models_dir(models_dir: str = DEFAULT_MODELS_DIR) -> str:
    """Ensure models directory exists and contains a .gitkeep file."""
    os.makedirs(models_dir, exist_ok=True)
    gitkeep = os.path.join(models_dir, ".gitkeep")
    if not os.path.exists(gitkeep):
        try:
            with open(gitkeep, "w", encoding="utf-8") as f:
                f.write("")
        except OSError:
            pass
    return models_dir


# ─── Data Ingestion & Parsing ──────────────────────────────────────────────────
def parse_label(val: Any) -> Optional[int]:
    """Normalizes label value to class index (0..5). Returns None if invalid."""
    if val is None:
        return None
    if isinstance(val, int):
        if 0 <= val < len(CLASSES):
            return val
        return None
    if isinstance(val, float):
        if val.is_integer() and 0 <= int(val) < len(CLASSES):
            return int(val)
        return None
    if isinstance(val, str):
        v = val.strip().lower()
        if v in CLASS_TO_INDEX:
            return CLASS_TO_INDEX[v]
        try:
            iv = int(v)
            if 0 <= iv < len(CLASSES):
                return iv
        except ValueError:
            pass
    return None


def extract_features(rec: Dict[str, Any]) -> Optional[List[float]]:
    """Extracts 23-dimensional feature vector. Returns None if invalid."""
    feats = rec.get("features")
    if isinstance(feats, list) and len(feats) == len(FEATURE_ORDER):
        try:
            row = []
            for v in feats:
                if v is None or (isinstance(v, (int, float)) and math.isnan(v)):
                    row.append(0.0)
                else:
                    row.append(float(v))
            return row
        except (ValueError, TypeError):
            return None
    elif isinstance(feats, dict):
        row = []
        for name in FEATURE_ORDER:
            val = feats.get(name, 0.0)
            if val is None or (isinstance(val, (int, float)) and math.isnan(val)):
                val = 0.0
            row.append(float(val))
        return row
    else:
        # Check top-level keys
        has_any = False
        row = []
        for name in FEATURE_ORDER:
            if name in rec:
                has_any = True
            val = rec.get(name, 0.0)
            if val is None or (isinstance(val, (int, float)) and math.isnan(val)):
                val = 0.0
            row.append(float(val))
        if has_any:
            return row
    return None


def load_labeled_records(data_path: str, quiet: bool = False) -> List[Dict[str, Any]]:
    """
    Ingests JSONL labeled records from data_path.
    Filters for valid records containing a recognized 6-class label and 23 features.
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
                    if not isinstance(rec, dict):
                        continue
                    norm_lbl = parse_label(rec.get("label"))
                    if norm_lbl is None:
                        continue
                    feat_vec = extract_features(rec)
                    if feat_vec is None:
                        continue

                    rec["_label_idx"] = norm_lbl
                    rec["_feat_vec"] = feat_vec

                    src_hash = (
                        rec.get("sourceHash")
                        or rec.get("source_hash")
                        or rec.get("video")
                        or rec.get("source")
                        or rec.get("videoId")
                        or rec.get("source_path")
                        or "single_source"
                    )
                    rec["_source_hash"] = str(src_hash)
                    records.append(rec)
                except (json.JSONDecodeError, ValueError):
                    continue
    except Exception as e:
        if not quiet:
            sys.stderr.write(f"[pause_train] Warning reading {data_path}: {e}\n")
        return []
    return records


# ─── Disjoint Per-Video Splitting ──────────────────────────────────────────────
def partition_by_video(
    records: List[Dict[str, Any]],
    test_size: float = 0.20,
    seed: int = 42,
    quiet: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Partitions records into (train_recs, test_recs) grouped strictly by sourceHash.
    Guarantees train and test sourceHash sets are strictly disjoint (zero speaker/video leakage).
    If there is only 1 unique video, falls back to intra-video split.
    """
    if not records:
        return [], []

    # Group by sourceHash
    records_by_video: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        vid = r["_source_hash"]
        records_by_video.setdefault(vid, []).append(r)

    unique_videos = sorted(list(records_by_video.keys()))

    if len(unique_videos) > 1:
        total_samples = len(records)
        target_test_samples = int(round(total_samples * test_size))
        all_labels = {r["_label_idx"] for r in records}

        best_train: Optional[List[Dict[str, Any]]] = None
        best_test: Optional[List[Dict[str, Any]]] = None
        best_score = float("-inf")

        for trial_seed in range(seed, seed + 50):
            rng = random.Random(trial_seed)
            shuffled_vids = list(unique_videos)
            rng.shuffle(shuffled_vids)

            test_vids: Set[str] = set()
            train_vids: Set[str] = set()
            cur_test_count = 0

            # Target ~20% of samples in test, with at least 1 video for train
            for v in shuffled_vids:
                v_count = len(records_by_video[v])
                if len(test_vids) < len(shuffled_vids) - 1 and (
                    cur_test_count + v_count <= target_test_samples or not test_vids
                ):
                    test_vids.add(v)
                    cur_test_count += v_count
                else:
                    train_vids.add(v)

            if not test_vids or not train_vids:
                test_vids = {shuffled_vids[0]}
                train_vids = set(shuffled_vids[1:])

            tr = [r for v in train_vids for r in records_by_video[v]]
            te = [r for v in test_vids for r in records_by_video[v]]

            tr_labels = {r["_label_idx"] for r in tr}
            class_coverage = len(tr_labels) / max(1, len(all_labels))
            actual_test_ratio = len(te) / max(1, total_samples)
            ratio_penalty = -abs(actual_test_ratio - test_size)

            score = (class_coverage * 10.0) + ratio_penalty

            if score > best_score:
                best_score = score
                best_train, best_test = tr, te
                if class_coverage == 1.0 and abs(actual_test_ratio - test_size) < 0.05:
                    break

        return (best_train or records), (best_test or [])
    else:
        # Single video fallback: intra-video split
        if not quiet:
            sys.stderr.write(
                "[pause_train] Warning: Single video/sourceHash in dataset; "
                "falling back to intra-video split to prevent empty sets.\n"
            )
        split_idx = max(1, min(len(records) - 1, int(round(len(records) * (1.0 - test_size)))))
        return records[:split_idx], records[split_idx:]


# ─── Evaluation Metrics ────────────────────────────────────────────────────────
def compute_evaluation_metrics(
    y_test: np.ndarray,
    proba_test: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
) -> Dict[str, float]:
    """
    Computes test logloss, multi_error, and removable precision/recall.
    """
    from sklearn.metrics import accuracy_score, log_loss, precision_score, recall_score

    if len(y_test) == 0:
        return {
            "removablePrecision": 0.0,
            "removableRecall": 0.0,
            "logloss": 0.0,
            "multiError": 0.0,
        }

    # 1. Multi-class log loss
    try:
        loss = float(log_loss(y_test, proba_test, labels=list(range(len(CLASSES)))))
    except Exception:
        loss = 0.0

    # 2. Multi-class error (1 - accuracy)
    pred_classes = np.argmax(proba_test, axis=1)
    multi_error = float(1.0 - accuracy_score(y_test, pred_classes))

    # 3. Removable precision and recall
    # P(removable) = P(breath_pause) + P(waiting_pause)
    p_rem = proba_test[:, REMOVABLE_INDICES].sum(axis=1)
    y_rem_pred = (p_rem >= threshold).astype(int)
    y_rem_true = np.isin(y_test, REMOVABLE_INDICES).astype(int)

    rem_precision = float(precision_score(y_rem_true, y_rem_pred, zero_division=0))
    rem_recall = float(recall_score(y_rem_true, y_rem_pred, zero_division=0))

    return {
        "removablePrecision": round(rem_precision, 4),
        "removableRecall": round(rem_recall, 4),
        "logloss": round(loss, 4),
        "multiError": round(multi_error, 4),
    }


# ─── LightGBM Model Training ──────────────────────────────────────────────────
def train_lightgbm_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
    seed: int = 42,
    num_boost_round: int = 300,
    early_stopping_rounds: int = 20,
    quiet: bool = False,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Trains 6-class LightGBM classifier with early stopping.
    Returns booster and evaluation trainStats.
    """
    import lightgbm as lgb

    train_data = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_ORDER)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data, feature_name=FEATURE_ORDER)

    params = {
        "objective": "multiclass",
        "num_class": len(CLASSES),
        "metric": ["multi_logloss", "multi_error"],
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 5,
        "feature_fraction": 1.0,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbosity": -1,
        "random_state": seed,
    }

    callbacks = []
    if early_stopping_rounds > 0 and len(X_test) > 0:
        callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False))

    booster = lgb.train(
        params,
        train_data,
        num_boost_round=num_boost_round,
        valid_sets=[train_data, test_data],
        valid_names=["train", "test"],
        callbacks=callbacks,
    )

    if len(X_test) > 0:
        test_proba = booster.predict(X_test)
        if test_proba.ndim == 1:
            test_proba = test_proba.reshape(-1, len(CLASSES))
        eval_metrics = compute_evaluation_metrics(y_test, test_proba, threshold=threshold)
    else:
        eval_metrics = {
            "removablePrecision": 0.0,
            "removableRecall": 0.0,
            "logloss": 0.0,
            "multiError": 0.0,
        }

    stats = {
        "totalSamples": int(len(X_train) + len(X_test)),
        "trainSamples": int(len(X_train)),
        "testSamples": int(len(X_test)),
        "removablePrecision": eval_metrics["removablePrecision"],
        "removableRecall": eval_metrics["removableRecall"],
        "logloss": eval_metrics["logloss"],
        "multiError": eval_metrics["multiError"],
        "bestIteration": int(booster.best_iteration or 0),
    }

    return booster, stats


# ─── Artifacts Generation ──────────────────────────────────────────────────────
def save_model_artifacts(
    booster: Any,
    stats: Dict[str, Any],
    model_path: str,
    meta_path: str,
    threshold: float = DEFAULT_THRESHOLD,
    dataset_version: str = "pause_features_v1",
) -> Tuple[str, str]:
    """
    Saves booster text model and metadata JSON.
    Creates parent directories if necessary.
    """
    os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(meta_path)), exist_ok=True)

    booster.save_model(model_path)

    metadata = {
        "modelFile": os.path.basename(model_path),
        "featureVersion": FEATURE_VERSION,
        "threshold": threshold,
        "classes": CLASSES,
        "removableClasses": REMOVABLE_CLASSES,
        "features": FEATURE_ORDER,
        "trainedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "datasetVersion": dataset_version,
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
    test_size: float = 0.20,
    seed: int = 42,
    num_boost_round: int = 300,
    early_stopping_rounds: int = 20,
    quiet: bool = False,
) -> int:
    """
    Full training pipeline with Safety Guard.
    Returns 0 on success or safe refusal. Returns non-zero only on unhandled error.
    """
    # 1. Ensure models directory exists and contains .gitkeep
    ensure_models_dir(models_dir)

    # 2. Safety Guard: Refuse to train if valid labeled records < min_samples
    records = load_labeled_records(data_path, quiet=quiet)
    if len(records) < min_samples:
        print(REFUSAL_MESSAGE)
        return 0

    # 3. Disjoint per-video train/test split
    train_recs, test_recs = partition_by_video(records, test_size=test_size, seed=seed, quiet=quiet)

    X_train = np.array([r["_feat_vec"] for r in train_recs], dtype=np.float32)
    y_train = np.array([r["_label_idx"] for r in train_recs], dtype=np.int32)

    X_test = np.array([r["_feat_vec"] for r in test_recs], dtype=np.float32)
    y_test = np.array([r["_label_idx"] for r in test_recs], dtype=np.int32)

    # 4. Train LightGBM 6-class classifier
    booster, stats = train_lightgbm_classifier(
        X_train,
        y_train,
        X_test,
        y_test,
        threshold=threshold,
        seed=seed,
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds,
        quiet=quiet,
    )

    # 5. Output paths
    if not model_out:
        model_out = os.path.join(models_dir, MODEL_FILENAME)
    if not meta_out:
        meta_out = os.path.join(models_dir, META_FILENAME)

    dataset_ver = os.path.splitext(os.path.basename(data_path))[0] if data_path else "pause_features_v1"

    save_model_artifacts(
        booster=booster,
        stats=stats,
        model_path=model_out,
        meta_path=meta_out,
        threshold=threshold,
        dataset_version=dataset_ver,
    )

    if not quiet:
        print(f"Successfully trained pause classifier: {model_out}")
        print(
            f"Test Metrics: logloss={stats['logloss']}, multi_error={stats['multiError']}, "
            f"removable_precision={stats['removablePrecision']}, removable_recall={stats['removableRecall']} "
            f"(train={stats['trainSamples']}, test={stats['testSamples']})"
        )
        print(f"Metadata exported to: {meta_out}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Multi-Class LightGBM Pause Classifier Trainer (`scripts/pause_train.py`)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data",
        default=DEFAULT_DATA_PATH,
        help=f"Path to input dataset JSONL (default: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument(
        "--models-dir",
        "--models_dir",
        dest="models_dir",
        default=DEFAULT_MODELS_DIR,
        help=f"Directory where model and metadata are exported (default: {DEFAULT_MODELS_DIR})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Removable decision threshold (default: {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--out-model",
        "--out_model",
        dest="out_model",
        default=None,
        help="Explicit path override for booster model file",
    )
    parser.add_argument(
        "--out-meta",
        "--out_meta",
        dest="out_meta",
        default=None,
        help="Explicit path override for metadata JSON file",
    )
    parser.add_argument(
        "--min-samples",
        "--min_samples",
        dest="min_samples",
        type=int,
        default=MIN_REQUIRED_SAMPLES,
        help=f"Minimum required labeled samples (default: {MIN_REQUIRED_SAMPLES})",
    )
    parser.add_argument(
        "--test-size",
        "--test_size",
        "--val-ratio",
        "--val_ratio",
        dest="test_size",
        type=float,
        default=0.20,
        help="Fraction of held-out test split (default: 0.20)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting and training (default: 42)",
    )
    parser.add_argument(
        "--num-boost-round",
        "--num_boost_round",
        dest="num_boost_round",
        type=int,
        default=300,
        help="Maximum boosting rounds (default: 300)",
    )
    parser.add_argument(
        "--early-stopping-rounds",
        "--early_stopping_rounds",
        dest="early_stopping_rounds",
        type=int,
        default=20,
        help="Early stopping rounds (default: 20)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose informational logs",
    )
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    return train_pipeline(
        data_path=args.data,
        models_dir=args.models_dir,
        model_out=args.out_model,
        meta_out=args.out_meta,
        min_samples=args.min_samples,
        threshold=args.threshold,
        test_size=args.test_size,
        seed=args.seed,
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping_rounds,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    sys.exit(main())
