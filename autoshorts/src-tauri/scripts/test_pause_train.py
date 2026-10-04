#!/usr/bin/env python3
"""
Unit and integration tests for Multi-Class LightGBM Training Pipeline (`scripts/pause_train.py`).

Verifies:
1. Safety gate: Refusal when samples < 200 with code 0 and verbatim em dash message.
2. Disjoint group splitting: Strict sourceHash separation between train and test (zero leakage).
3. 6-class LightGBM training and metadata JSON schema export with pinned featureVersion: "1".
4. Removable precision and recall computation.
5. End-to-end integration with `pause_intelligence.py` loading and scoring.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from typing import Any, Dict, List

import numpy as np

# Ensure scripts dir is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import pause_train
import pause_intelligence as pi


def generate_synthetic_record(
    source_hash: str,
    gap_idx: int,
    label: str,
    feature_override: Dict[str, float] = None,
) -> Dict[str, Any]:
    """Generates a synthetic labeled record matching pause_label.py schema."""
    features = []
    # 23 features in pause_train.FEATURE_ORDER
    for name in pause_train.FEATURE_ORDER:
        if feature_override and name in feature_override:
            features.append(feature_override[name])
        elif "Rms" in name:
            features.append(-35.0 + (gap_idx % 10))
        elif "Ratio" in name or "Frac" in name or "Prob" in name:
            features.append(round(0.1 + 0.03 * (gap_idx % 25), 4))
        elif "duration" in name:
            features.append(round(0.2 + 0.05 * (gap_idx % 20), 3))
        elif "Rate" in name:
            features.append(3.5)
        elif "pos" in name.lower():
            features.append(0.5)
        else:
            features.append(0.0)

    return {
        "sourceHash": source_hash,
        "gapIdx": gap_idx,
        "srcStartSec": round(gap_idx * 1.5, 3),
        "srcEndSec": round(gap_idx * 1.5 + 0.4, 3),
        "features": features,
        "label": label,
        "labelSource": "synthetic_test",
        "weakSupervision": False,
        "timestamp": "2026-10-03T10:00:00Z",
    }


class TestSafetyGate(unittest.TestCase):
    """Verifies sample count safety gate (< 200 samples)."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="pause_test_safety_")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_missing_file_refusal(self):
        non_existent = os.path.join(self.tmp_dir, "non_existent.jsonl")
        cmd = [sys.executable, os.path.join(SCRIPT_DIR, "pause_train.py"), "--data", non_existent]
        res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(res.returncode, 0)
        self.assertIn("NO LABELS \u2014 classifier not trained", res.stdout)

    def test_empty_file_refusal(self):
        empty_file = os.path.join(self.tmp_dir, "empty.jsonl")
        with open(empty_file, "w", encoding="utf-8") as f:
            pass
        cmd = [sys.executable, os.path.join(SCRIPT_DIR, "pause_train.py"), "--data", empty_file]
        res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(res.returncode, 0)
        self.assertIn("NO LABELS \u2014 classifier not trained", res.stdout)

    def test_under_200_samples_refusal(self):
        under_file = os.path.join(self.tmp_dir, "under_200.jsonl")
        with open(under_file, "w", encoding="utf-8") as f:
            for i in range(199):
                rec = generate_synthetic_record(
                    source_hash="video_abc",
                    gap_idx=i,
                    label=pause_train.CLASSES[i % len(pause_train.CLASSES)],
                )
                f.write(json.dumps(rec) + "\n")

        cmd = [sys.executable, os.path.join(SCRIPT_DIR, "pause_train.py"), "--data", under_file]
        res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(res.returncode, 0)
        self.assertIn("NO LABELS \u2014 classifier not trained", res.stdout)

        # Ensure no model artifacts were created
        models_out = os.path.join(self.tmp_dir, "models")
        self.assertFalse(os.path.exists(os.path.join(models_out, pause_train.MODEL_FILENAME)))


class TestDisjointGroupSplitting(unittest.TestCase):
    """Verifies per-video group splitting prevents speaker/source acoustic leakage."""

    def test_multi_video_disjoint_split(self):
        # 8 distinct videos, 30 records each = 240 records
        records = []
        for vid_idx in range(8):
            vid_hash = f"hash_{vid_idx:03d}"
            for g_idx in range(30):
                lbl = pause_train.CLASSES[(vid_idx + g_idx) % len(pause_train.CLASSES)]
                rec = generate_synthetic_record(vid_hash, g_idx, lbl)
                rec["_label_idx"] = pause_train.CLASS_TO_INDEX[lbl]
                rec["_feat_vec"] = rec["features"]
                rec["_source_hash"] = vid_hash
                records.append(rec)

        train_recs, test_recs = pause_train.partition_by_video(records, test_size=0.20, seed=42)

        self.assertGreater(len(train_recs), 0)
        self.assertGreater(len(test_recs), 0)
        self.assertEqual(len(train_recs) + len(test_recs), len(records))

        train_hashes = {r["_source_hash"] for r in train_recs}
        test_hashes = {r["_source_hash"] for r in test_recs}

        # Zero leakage requirement: train and test video hashes must be strictly disjoint!
        intersection = train_hashes.intersection(test_hashes)
        self.assertEqual(
            intersection,
            set(),
            f"Data leakage detected! Videos in both train and test: {intersection}",
        )
        self.assertTrue(train_hashes.isdisjoint(test_hashes))

    def test_single_video_fallback(self):
        # 210 records with the exact same sourceHash
        records = []
        vid_hash = "only_one_video_hash"
        for g_idx in range(210):
            lbl = pause_train.CLASSES[g_idx % len(pause_train.CLASSES)]
            rec = generate_synthetic_record(vid_hash, g_idx, lbl)
            rec["_label_idx"] = pause_train.CLASS_TO_INDEX[lbl]
            rec["_feat_vec"] = rec["features"]
            rec["_source_hash"] = vid_hash
            records.append(rec)

        train_recs, test_recs = pause_train.partition_by_video(records, test_size=0.20, seed=42, quiet=True)
        self.assertGreater(len(train_recs), 0)
        self.assertGreater(len(test_recs), 0)
        self.assertEqual(len(train_recs) + len(test_recs), 210)


class TestRemovableMetrics(unittest.TestCase):
    """Verifies precision and recall computation for removable classes."""

    def test_removable_gate_calculation(self):
        # 4 test samples:
        # Sample 0: class 0 (breath_pause) -> Removable ground truth = 1
        # Sample 1: class 1 (waiting_pause) -> Removable ground truth = 1
        # Sample 2: class 2 (sentence_pause) -> Removable ground truth = 0
        # Sample 3: class 3 (normal_word_gap) -> Removable ground truth = 0
        y_test = np.array([0, 1, 2, 3])

        # Predicted probabilities for the 6 classes
        # Sample 0: P(breath)=0.7, P(waiting)=0.1 => P(removable) = 0.8 >= 0.65 (TP)
        # Sample 1: P(breath)=0.2, P(waiting)=0.3 => P(removable) = 0.5 < 0.65 (FN)
        # Sample 2: P(breath)=0.4, P(waiting)=0.3 => P(removable) = 0.7 >= 0.65 (FP)
        # Sample 3: P(breath)=0.0, P(waiting)=0.1 => P(removable) = 0.1 < 0.65 (TN)
        proba = np.array([
            [0.70, 0.10, 0.10, 0.05, 0.03, 0.02],
            [0.20, 0.30, 0.30, 0.10, 0.05, 0.05],
            [0.40, 0.30, 0.20, 0.05, 0.03, 0.02],
            [0.00, 0.10, 0.10, 0.70, 0.05, 0.05],
        ])

        metrics = pause_train.compute_evaluation_metrics(y_test, proba, threshold=0.65)

        # TP = 1 (sample 0), FP = 1 (sample 2), FN = 1 (sample 1), TN = 1 (sample 3)
        # Precision = TP / (TP + FP) = 1 / 2 = 0.5
        # Recall = TP / (TP + FN) = 1 / 2 = 0.5
        self.assertAlmostEqual(metrics["removablePrecision"], 0.50, places=2)
        self.assertAlmostEqual(metrics["removableRecall"], 0.50, places=2)
        self.assertGreater(metrics["logloss"], 0.0)
        self.assertGreaterEqual(metrics["multiError"], 0.0)

    def test_empty_test_set_handling(self):
        metrics = pause_train.compute_evaluation_metrics(np.array([]), np.array([]))
        self.assertEqual(metrics["removablePrecision"], 0.0)
        self.assertEqual(metrics["removableRecall"], 0.0)
        self.assertEqual(metrics["logloss"], 0.0)
        self.assertEqual(metrics["multiError"], 0.0)


class TestModelTrainingAndArtifactExport(unittest.TestCase):
    """Verifies LightGBM 6-class training, artifact generation, and schema conformity."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="pause_test_train_")
        self.data_file = os.path.join(self.tmp_dir, "dataset.jsonl")
        self.models_dir = os.path.join(self.tmp_dir, "models")

        # Synthesize 240 samples across 6 distinct videos and 6 classes
        with open(self.data_file, "w", encoding="utf-8") as f:
            for vid_idx in range(6):
                vid_hash = f"video_hash_{vid_idx:03d}"
                for g_idx in range(40):
                    lbl = pause_train.CLASSES[(vid_idx + g_idx) % len(pause_train.CLASSES)]
                    rec = generate_synthetic_record(vid_hash, g_idx, lbl)
                    f.write(json.dumps(rec) + "\n")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_training_pipeline_and_metadata_schema(self):
        ret = pause_train.train_pipeline(
            data_path=self.data_file,
            models_dir=self.models_dir,
            min_samples=200,
            threshold=0.65,
            test_size=0.20,
            seed=42,
            num_boost_round=50,
            early_stopping_rounds=10,
            quiet=True,
        )
        self.assertEqual(ret, 0)

        # 1. Verify model text file exists
        model_file = os.path.join(self.models_dir, pause_train.MODEL_FILENAME)
        self.assertTrue(os.path.isfile(model_file))
        self.assertGreater(os.path.getsize(model_file), 100)

        # 2. Verify metadata JSON file exists and validate schema
        meta_file = os.path.join(self.models_dir, pause_train.META_FILENAME)
        self.assertTrue(os.path.isfile(meta_file))

        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

        # Validate pinned keys per contract
        self.assertEqual(meta["modelFile"], pause_train.MODEL_FILENAME)
        self.assertEqual(meta["featureVersion"], "1")
        self.assertEqual(meta["threshold"], 0.65)
        self.assertEqual(meta["classes"], pause_train.CLASSES)
        self.assertEqual(meta["removableClasses"], ["breath_pause", "waiting_pause"])
        self.assertEqual(meta["features"], pause_train.FEATURE_ORDER)
        self.assertEqual(len(meta["features"]), 23)
        self.assertIn("trainedAt", meta)

        train_stats = meta["trainStats"]
        self.assertEqual(train_stats["totalSamples"], 240)
        self.assertGreater(train_stats["trainSamples"], 0)
        self.assertGreater(train_stats["testSamples"], 0)
        self.assertEqual(train_stats["trainSamples"] + train_stats["testSamples"], 240)
        self.assertIn("removablePrecision", train_stats)
        self.assertIn("removableRecall", train_stats)
        self.assertIn("logloss", train_stats)
        self.assertIn("multiError", train_stats)

        # 3. Verify .gitkeep exists in models_dir
        gitkeep = os.path.join(self.models_dir, ".gitkeep")
        self.assertTrue(os.path.isfile(gitkeep))

        # 4. Integration with pause_intelligence.py
        # Test that pause_intelligence.load_pause_classifier loads this artifact seamlessly
        loaded = pi.load_pause_classifier(model_file)
        self.assertIsNotNone(loaded)
        booster, loaded_meta = loaded
        self.assertEqual(loaded_meta.get("featureVersion"), "1")

        # Test scoring gaps with score_gaps_with_model
        test_gaps = [
            {"srcStartSec": 1.0, "srcEndSec": 1.4},
            {"srcStartSec": 3.0, "srcEndSec": 3.3},
        ]
        test_rows = [
            [0.1] * 23,
            [-20.0] * 23,
        ]
        os.environ["AUTOSHORTS_SMART_PACING_LEARNED"] = "1"
        scored = pi.score_gaps_with_model(test_gaps, test_rows, model_file)
        self.assertTrue(scored)
        for g in test_gaps:
            self.assertIn("pRemovable", g)
            self.assertIn("pRemovableAboveThreshold", g)
            self.assertIsInstance(g["pRemovable"], float)
            self.assertIsInstance(g["pRemovableAboveThreshold"], bool)
            self.assertGreaterEqual(g["pRemovable"], 0.0)
            self.assertLessEqual(g["pRemovable"], 1.0)


if __name__ == "__main__":
    unittest.main()
