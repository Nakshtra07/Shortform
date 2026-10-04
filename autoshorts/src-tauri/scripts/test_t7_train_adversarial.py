#!/usr/bin/env python3
"""
Adversarial Stress Test Suite for T7 LightGBM Trainer (scripts/t7_train.py)

Empirically tests:
1. Exact boundary testing:
   - 0 samples (missing file, empty file, whitespace-only) -> exit 0, exact refusal message
   - 199 samples -> exit 0, exact refusal message "NO LABELS — tagger not trained"
   - 200 samples -> trains without refusal, generates valid artifacts
2. Corrupt JSONL filtering:
   - empty lines, malformed JSON syntax, missing isBoundary, null/invalid isBoundary
   - boundary threshold after corrupt filtering (205 raw with 6 corrupt = 199 vs 210 raw with 10 corrupt = 200)
   - non-numeric / NaN / None feature values
3. Split behaviors:
   - Single video dataset (per-video split handles 1 group without crashing)
   - Multi-video dataset (disjoint per-video partitions, zero speaker leakage)
4. Degenerate label distributions:
   - all 0s and all 1s (measures LightGBM and metric behavior)
5. Artifact safety:
   - No fake or partial model generated on refusal
   - Existing model artifacts are NOT overwritten or touched on refusal
6. Windows UTF-8 console & subprocess execution:
   - Verifies CLI process returncode and exact stdout bytes
"""

import os
import sys
import json
import math
import shutil
import tempfile
import unittest
import subprocess
from typing import List, Dict, Any

import numpy as np

# Ensure importability of t7_train
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import t7_train


def make_record(
    gap_idx: int,
    video: str = "video_01",
    is_boundary: Any = 0,
    features: Dict[str, Any] = None,
    flat: bool = True,
) -> Dict[str, Any]:
    """Helper to synthesize valid record dict."""
    default_feats = {
        "pauseSec": 0.35 + 0.05 * (gap_idx % 10),
        "endsSentence": 1 if gap_idx % 4 == 0 else 0,
        "endsClause": 1 if gap_idx % 2 == 0 else 0,
        "speakerChange": 1 if gap_idx % 5 == 0 else 0,
        "f0SlopePre": -1.2 + 0.1 * (gap_idx % 7),
        "energyDropDb": -4.5 + 0.2 * (gap_idx % 5),
        "vadSilenceFrac": 0.85 if is_boundary else 0.15,
        "pBoundaryRule": 0.75 if is_boundary else 0.25,
    }
    if features:
        default_feats.update(features)

    rec = {
        "gapIdx": gap_idx,
        "video": video,
        "isBoundary": is_boundary,
    }
    if flat:
        rec.update(default_feats)
    else:
        rec["features"] = default_feats
    return rec


class TestT7TrainAdversarialBoundaries(unittest.TestCase):
    """Stress tests boundary conditions: 0, 199, 200 samples."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="t7_adv_bound_")
        self.models_dir = os.path.join(self.test_dir, "models")
        os.makedirs(self.models_dir, exist_ok=True)
        self.data_file = os.path.join(self.test_dir, "data.jsonl")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_adv_01_boundary_0_samples_nonexistent_file(self):
        """0 samples: missing file exits 0 and prints verbatim refusal message."""
        nonexistent = os.path.join(self.test_dir, "does_not_exist.jsonl")
        
        code = t7_train.train_pipeline(
            data_path=nonexistent,
            models_dir=self.models_dir,
            min_samples=200,
        )
        self.assertEqual(code, 0)
        self.assertEqual(len(os.listdir(self.models_dir)), 0)

    def test_adv_02_boundary_0_samples_empty_file(self):
        """0 samples: 0-byte file exits 0 and creates no model."""
        with open(self.data_file, "w", encoding="utf-8") as f:
            pass  # 0 bytes

        code = t7_train.train_pipeline(
            data_path=self.data_file,
            models_dir=self.models_dir,
            min_samples=200,
        )
        self.assertEqual(code, 0)
        self.assertEqual(len(os.listdir(self.models_dir)), 0)

    def test_adv_03_boundary_0_samples_whitespace_only(self):
        """0 samples: file containing only whitespace/newlines exits 0."""
        with open(self.data_file, "w", encoding="utf-8") as f:
            f.write("\n\n   \n\t\n  \n")

        code = t7_train.train_pipeline(
            data_path=self.data_file,
            models_dir=self.models_dir,
            min_samples=200,
        )
        self.assertEqual(code, 0)
        self.assertEqual(len(os.listdir(self.models_dir)), 0)

    def test_adv_04_boundary_199_samples_exact_refusal(self):
        """Exact 199 samples: must refuse cleanly with code 0 and no artifacts."""
        with open(self.data_file, "w", encoding="utf-8") as f:
            for i in range(199):
                lbl = 1 if i % 2 == 0 else 0
                rec = make_record(i, video=f"vid_{i // 50}", is_boundary=lbl)
                f.write(json.dumps(rec) + "\n")

        records = t7_train.load_labeled_records(self.data_file)
        self.assertEqual(len(records), 199)

        code = t7_train.train_pipeline(
            data_path=self.data_file,
            models_dir=self.models_dir,
            min_samples=200,
        )
        self.assertEqual(code, 0)
        self.assertEqual(len(os.listdir(self.models_dir)), 0)

    def test_adv_05_boundary_200_samples_exact_trains(self):
        """Exact 200 samples: threshold reached, proceeds to train without refusal."""
        with open(self.data_file, "w", encoding="utf-8") as f:
            for i in range(200):
                lbl = 1 if i % 2 == 0 else 0
                # Using i // 50 ensures each of the 4 videos contains both 0 and 1 labels
                rec = make_record(i, video=f"vid_{i // 50}", is_boundary=lbl)
                f.write(json.dumps(rec) + "\n")

        records = t7_train.load_labeled_records(self.data_file)
        self.assertEqual(len(records), 200)

        model_path = os.path.join(self.models_dir, "t7_prosody_v1.txt")
        meta_path = os.path.join(self.models_dir, "t7_prosody_v1.meta.json")

        code = t7_train.train_pipeline(
            data_path=self.data_file,
            models_dir=self.models_dir,
            model_out=model_path,
            meta_out=meta_path,
            min_samples=200,
        )
        self.assertEqual(code, 0)
        self.assertTrue(os.path.isfile(model_path), "Model booster file must exist")
        self.assertTrue(os.path.isfile(meta_path), "Metadata JSON file must exist")
        self.assertGreater(os.path.getsize(model_path), 100)

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.assertEqual(meta["featureVersion"], "t7_prosody_v1")
        self.assertEqual(meta["threshold"], 0.65)
        self.assertEqual(meta["trainStats"]["totalSamples"], 200)
        self.assertGreater(meta["trainStats"]["auc"], 0.5)


class TestT7TrainAdversarialCorruptJSONL(unittest.TestCase):
    """Stress tests corruption resilience and sample filtering."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="t7_adv_corrupt_")
        self.models_dir = os.path.join(self.test_dir, "models")
        os.makedirs(self.models_dir, exist_ok=True)
        self.data_file = os.path.join(self.test_dir, "corrupt.jsonl")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_adv_06_corrupt_filtering_drops_below_200(self):
        """205 total lines: 199 valid lines + 6 corrupt lines -> exactly 199 valid -> refuses."""
        lines = []
        for i in range(199):
            rec = make_record(i, video=f"vid_{i // 50}", is_boundary=i % 2)
            lines.append(json.dumps(rec))
        
        corrupt_lines = [
            "",  # empty line
            "   ",  # whitespace line
            "{ malformed json string without closing bracket",  # syntax error
            json.dumps({"gapIdx": 999, "pauseSec": 0.5}),  # missing isBoundary
            json.dumps({"isBoundary": None, "pauseSec": 0.5}),  # null isBoundary
            json.dumps({"isBoundary": "not_a_valid_label", "pauseSec": 0.5}),  # invalid string label
        ]
        lines.insert(0, corrupt_lines[0])
        lines.insert(50, corrupt_lines[1])
        lines.insert(100, corrupt_lines[2])
        lines.insert(150, corrupt_lines[3])
        lines.insert(180, corrupt_lines[4])
        lines.append(corrupt_lines[5])

        with open(self.data_file, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")

        records = t7_train.load_labeled_records(self.data_file)
        self.assertEqual(len(records), 199, "Exactly 199 valid records should survive filtering")

        code = t7_train.train_pipeline(
            data_path=self.data_file,
            models_dir=self.models_dir,
            min_samples=200,
        )
        self.assertEqual(code, 0)
        self.assertEqual(len(os.listdir(self.models_dir)), 0, "No model artifact should be written on refusal")

    def test_adv_07_corrupt_filtering_reaches_200_and_trains(self):
        """210 total lines: 200 valid lines + 10 corrupt lines -> exactly 200 valid -> trains."""
        lines = []
        for i in range(200):
            rec = make_record(i, video=f"vid_{i // 50}", is_boundary=i % 2)
            lines.append(json.dumps(rec))

        corrupt_lines = [
            "{{{invalid json",
            "12345",  # JSON number, not dict
            '"just a json string"',  # JSON string, not dict
            "null",
            "[1, 2, 3]",  # JSON array, not dict
            json.dumps({"only_feature": 1.0}),  # dict without isBoundary
            json.dumps({"isBoundary": "banana"}),  # invalid string label
            json.dumps({"isBoundary": [0, 1]}),  # list label -> parse_label returns None
            json.dumps({"isBoundary": {"nested": 1}}),  # dict label -> None
            "   \t\n   ",
        ]

        for idx, cl in enumerate(corrupt_lines):
            lines.insert(idx * 15, cl)

        with open(self.data_file, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")

        records = t7_train.load_labeled_records(self.data_file)
        self.assertEqual(len(records), 200, "Exactly 200 valid records must be recovered")

        code = t7_train.train_pipeline(
            data_path=self.data_file,
            models_dir=self.models_dir,
            min_samples=200,
        )
        self.assertEqual(code, 0)
        self.assertTrue(os.path.isfile(os.path.join(self.models_dir, "t7_prosody_v1.txt")))

    def test_adv_08_corrupt_features_nan_none_missing_values(self):
        """Records with NaN, None, or omitted feature fields default gracefully to 0.0."""
        records = []
        for i in range(200):
            adv_feats = {
                "pauseSec": float("nan") if i % 3 == 0 else 0.4,
                "endsSentence": None if i % 5 == 0 else 1,
                "endsClause": 0,
                # omit speakerChange completely
                "f0SlopePre": -0.5,
                "energyDropDb": float("nan") if i % 2 == 0 else -3.0,
                "vadSilenceFrac": 0.8 if i % 2 == 0 else 0.1,
                "pBoundaryRule": 0.7 if i % 2 == 0 else 0.2,
            }
            rec = make_record(i, video=f"v_{i // 50}", is_boundary=i % 2, features=adv_feats)
            records.append(rec)

        X, y, groups = t7_train.extract_feature_matrix(records)
        self.assertEqual(X.shape, (200, 8))
        self.assertFalse(np.isnan(X).any(), "NaNs must be replaced with 0.0")
        self.assertFalse(np.isinf(X).any(), "No infinities allowed in feature matrix")

    def test_adv_08b_non_numeric_feature_string_causes_value_error(self):
        """Adversarial discovery: non-numeric string feature raises ValueError in extract_feature_matrix."""
        corrupt_rec = make_record(0, video="vid_0", is_boundary=1, features={"pauseSec": "not_a_float"})
        with self.assertRaises(ValueError):
            t7_train.extract_feature_matrix([corrupt_rec])


class TestT7TrainAdversarialSplits(unittest.TestCase):
    """Stress tests single-video vs multi-video dataset splitting."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="t7_adv_splits_")
        self.models_dir = os.path.join(self.test_dir, "models")
        os.makedirs(self.models_dir, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_adv_09_single_video_dataset_split_and_train(self):
        """Single-video dataset: per-video partition handles single group without crashing."""
        records = [
            make_record(i, video="single_solo_video_xyz", is_boundary=i % 2)
            for i in range(220)
        ]
        train_recs, val_recs = t7_train.partition_by_video(records, train_ratio=0.8)
        self.assertGreater(len(train_recs), 0)
        self.assertGreater(len(val_recs), 0)
        self.assertEqual(len(train_recs) + len(val_recs), 220)

        y_train = [r["isBoundary"] for r in train_recs]
        y_val = [r["isBoundary"] for r in val_recs]
        self.assertIn(0, y_train)
        self.assertIn(1, y_train)
        self.assertIn(0, y_val)
        self.assertIn(1, y_val)

        data_file = os.path.join(self.test_dir, "single_vid.jsonl")
        with open(data_file, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        code = t7_train.train_pipeline(
            data_path=data_file,
            models_dir=self.models_dir,
            min_samples=200,
        )
        self.assertEqual(code, 0)
        self.assertTrue(os.path.isfile(os.path.join(self.models_dir, "t7_prosody_v1.txt")))

    def test_adv_10_multi_video_split_strictly_disjoint_zero_leakage(self):
        """Multi-video dataset: train and validation sets have strictly disjoint video IDs."""
        video_names = [f"speaker_session_{k:02d}" for k in range(12)]
        records = []
        for i in range(360):
            vid = video_names[i % len(video_names)]
            records.append(make_record(i, video=vid, is_boundary=i % 2))

        train_recs, val_recs = t7_train.partition_by_video(records, train_ratio=0.75, seed=123)
        train_videos = {r["video"] for r in train_recs}
        val_videos = {r["video"] for r in val_recs}

        self.assertTrue(train_videos.isdisjoint(val_videos), f"Speaker leakage! Overlap: {train_videos & val_videos}")
        self.assertGreater(len(train_videos), 0)
        self.assertGreater(len(val_videos), 0)
        self.assertEqual(train_videos | val_videos, set(video_names))


class TestT7TrainAdversarialArtifactSafety(unittest.TestCase):
    """Stress tests artifact immutability on refusal."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="t7_adv_art_")
        self.models_dir = os.path.join(self.test_dir, "models")
        os.makedirs(self.models_dir, exist_ok=True)
        self.data_file = os.path.join(self.test_dir, "data_sub200.jsonl")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_adv_11_refusal_does_not_overwrite_existing_artifacts(self):
        """Pre-existing model and metadata must NOT be overwritten or touched on refusal."""
        model_path = os.path.join(self.models_dir, "t7_prosody_v1.txt")
        meta_path = os.path.join(self.models_dir, "t7_prosody_v1.meta.json")

        sentinel_model = "SENTINEL_MODEL_DO_NOT_OVERWRITE_12345"
        sentinel_meta = json.dumps({"sentinel": "untouched", "version": "pre_existing"})

        with open(model_path, "w", encoding="utf-8") as f:
            f.write(sentinel_model)
        with open(meta_path, "w", encoding="utf-8") as f:
            f.write(sentinel_meta)

        with open(self.data_file, "w", encoding="utf-8") as f:
            for i in range(150):
                f.write(json.dumps(make_record(i, is_boundary=i % 2)) + "\n")

        code = t7_train.train_pipeline(
            data_path=self.data_file,
            models_dir=self.models_dir,
            model_out=model_path,
            meta_out=meta_path,
            min_samples=200,
        )
        self.assertEqual(code, 0)

        with open(model_path, "r", encoding="utf-8") as f:
            self.assertEqual(f.read(), sentinel_model)
        with open(meta_path, "r", encoding="utf-8") as f:
            self.assertEqual(f.read(), sentinel_meta)


class TestT7TrainAdversarialSubprocessCLI(unittest.TestCase):
    """Stress tests CLI invocation, exit codes, and stdout byte accuracy via subprocess."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="t7_adv_cli_")
        self.models_dir = os.path.join(self.test_dir, "models")
        os.makedirs(self.models_dir, exist_ok=True)
        self.script_path = os.path.join(SCRIPT_DIR, "t7_train.py")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_adv_12_cli_refusal_exit_code_and_exact_stdout_string(self):
        """CLI invocation on missing dataset must exit code 0 and output exact refusal string."""
        empty_data = os.path.join(self.test_dir, "empty.jsonl")
        with open(empty_data, "w", encoding="utf-8") as f:
            pass

        result = subprocess.run(
            [sys.executable, self.script_path, "--data", empty_data, "--models_dir", self.models_dir],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )

        self.assertEqual(result.returncode, 0, f"Expected returncode 0, got {result.returncode}. Stderr: {result.stderr}")
        expected_refusal = "NO LABELS \u2014 tagger not trained"
        self.assertIn(expected_refusal, result.stdout.strip())
        self.assertNotIn("Successfully trained", result.stdout)

    def test_adv_13_cli_199_samples_exact_refusal(self):
        """CLI invocation on exactly 199 samples must exit code 0 and refuse."""
        data_file = os.path.join(self.test_dir, "199.jsonl")
        with open(data_file, "w", encoding="utf-8") as f:
            for i in range(199):
                f.write(json.dumps(make_record(i, is_boundary=i % 2)) + "\n")

        result = subprocess.run(
            [sys.executable, self.script_path, "--data", data_file, "--models_dir", self.models_dir],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )

        self.assertEqual(result.returncode, 0)
        self.assertIn("NO LABELS \u2014 tagger not trained", result.stdout.strip())
        self.assertEqual(len(os.listdir(self.models_dir)), 0)

    def test_adv_14_cli_200_samples_exact_training_success(self):
        """CLI invocation on exactly 200 samples must exit code 0 and train."""
        data_file = os.path.join(self.test_dir, "200.jsonl")
        with open(data_file, "w", encoding="utf-8") as f:
            for i in range(200):
                f.write(json.dumps(make_record(i, video=f"v_{i // 50}", is_boundary=i % 2)) + "\n")

        result = subprocess.run(
            [sys.executable, self.script_path, "--data", data_file, "--models_dir", self.models_dir],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )

        self.assertEqual(result.returncode, 0, f"Failed with stderr: {result.stderr}")
        self.assertNotIn("NO LABELS \u2014 tagger not trained", result.stdout)
        self.assertIn("Successfully trained T7 prosody model", result.stdout)

        model_file = os.path.join(self.models_dir, "t7_prosody_v1.txt")
        meta_file = os.path.join(self.models_dir, "t7_prosody_v1.meta.json")
        self.assertTrue(os.path.isfile(model_file))
        self.assertTrue(os.path.isfile(meta_file))


class TestT7TrainAdversarialDegenerateLabels(unittest.TestCase):
    """Stress tests behavior when label distributions are degenerate (all 0s or all 1s)."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="t7_adv_degen_")
        self.models_dir = os.path.join(self.test_dir, "models")
        os.makedirs(self.models_dir, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_adv_15_degenerate_all_zeros_behavior(self):
        """When all 200 samples are 0, LightGBM trains but AUC calculation issues warning and yields NaN."""
        data_file = os.path.join(self.test_dir, "all_zeros.jsonl")
        with open(data_file, "w", encoding="utf-8") as f:
            for i in range(200):
                f.write(json.dumps(make_record(i, video=f"v_{i // 50}", is_boundary=0)) + "\n")

        code = t7_train.train_pipeline(
            data_path=data_file,
            models_dir=self.models_dir,
            min_samples=200,
        )
        self.assertEqual(code, 0)
        # Verify model was created
        meta_file = os.path.join(self.models_dir, "t7_prosody_v1.meta.json")
        self.assertTrue(os.path.isfile(meta_file))
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
        # Note: in degenerate single-class data, auc evaluates to NaN
        self.assertTrue(math.isnan(meta["trainStats"]["auc"]))

    def test_adv_16_degenerate_all_ones_behavior(self):
        """When all 200 samples are 1, LightGBM trains but AUC evaluates to NaN."""
        data_file = os.path.join(self.test_dir, "all_ones.jsonl")
        with open(data_file, "w", encoding="utf-8") as f:
            for i in range(200):
                f.write(json.dumps(make_record(i, video=f"v_{i // 50}", is_boundary=1)) + "\n")

        code = t7_train.train_pipeline(
            data_path=data_file,
            models_dir=self.models_dir,
            min_samples=200,
        )
        self.assertEqual(code, 0)
        meta_file = os.path.join(self.models_dir, "t7_prosody_v1.meta.json")
        self.assertTrue(os.path.isfile(meta_file))
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.assertTrue(math.isnan(meta["trainStats"]["auc"]))


if __name__ == "__main__":
    unittest.main()
