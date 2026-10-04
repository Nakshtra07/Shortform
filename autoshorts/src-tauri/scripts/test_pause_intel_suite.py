#!/usr/bin/env python3
"""
AutoShorts 11.0 Phase 3 — Pause Intelligence Consolidated Test Suite

Comprehensive Python unit and integration test suite validating:
1. test_feature_vector_dimension_and_order
2. test_acoustic_feature_calculations
3. test_labeling_duration_filters
4. test_weak_supervision_mapping
5. test_trainer_refuses_under_200_samples
6. test_trainer_disjoint_video_partition
7. test_model_export_and_meta_schema
8. test_model_scoring_removable_classes
9. test_feature_version_mismatch_refusal
10. test_predict_cli_json_payload
"""

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import wave
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np

# Ensure script dir is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import pause_intelligence as pi
import pause_label as pl
import pause_train as pt


PINNED_23_FEATURES = [
    "gapRmsDb", "preSpeechRmsDb", "postSpeechRmsDb", "hfRatio", "flatness",
    "zcr", "dipDb", "levelDropDb", "durationSec", "speakerChange",
    "endsSentence", "endsClause", "startsFiller", "endsFiller",
    "preSpeechRate", "postSpeechRate", "relPosInClip", "vadSpeechProb",
    "vadSilenceFrac", "evHfRatio", "evAboveFloorDb", "evContrastDb", "evFlatness",
]


def create_test_wav(
    path: str,
    duration: float = 3.0,
    tone_spans: tuple = (),
    sample_rate: int = 16000,
) -> str:
    """Create a 16kHz mono WAV file with specified tone spans."""
    n = int(duration * sample_rate)
    x = np.zeros(n, dtype=np.float32)
    for t0, t1, freq, amp in tone_spans:
        i0 = int(t0 * sample_rate)
        i1 = min(n, int(t1 * sample_rate))
        if i1 > i0:
            t = np.arange(i1 - i0, dtype=np.float32) / sample_rate
            x[i0:i1] = amp * np.sin(2 * np.pi * freq * t)

    x_int16 = np.clip(x * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(x_int16.tobytes())
    return path


def generate_synthetic_dataset(
    num_samples: int = 240,
    num_videos: int = 6,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Generates synthetic labeled records conforming to pause_label/pause_train schema."""
    rng = np.random.RandomState(seed)
    records = []
    classes = pt.CLASSES

    for i in range(num_samples):
        vid_idx = i % num_videos
        vid_hash = f"video_source_{vid_idx:03d}"
        lbl = classes[i % len(classes)]

        feats = []
        for name in pt.FEATURE_ORDER:
            if "Rms" in name:
                feats.append(float(-45.0 + rng.uniform(0.0, 20.0)))
            elif "Ratio" in name or "Frac" in name or "Prob" in name:
                feats.append(float(rng.uniform(0.05, 0.95)))
            elif "duration" in name:
                feats.append(float(rng.uniform(0.08, 1.20)))
            elif "Rate" in name:
                feats.append(float(rng.uniform(2.0, 5.0)))
            elif "pos" in name.lower():
                feats.append(float(rng.uniform(0.0, 1.0)))
            else:
                feats.append(float(rng.choice([0.0, 1.0])))

        rec = {
            "sourceHash": vid_hash,
            "gapIdx": i,
            "srcStartSec": round(float(i * 1.5), 3),
            "srcEndSec": round(float(i * 1.5 + 0.35), 3),
            "features": feats,
            "label": lbl,
            "labelSource": "synthetic_generator",
            "weakSupervision": False,
            "timestamp": "2026-10-03T10:00:00Z",
            "_source_hash": vid_hash,
            "_label_idx": pt.CLASS_TO_INDEX[lbl],
            "_feat_vec": feats,
        }
        records.append(rec)
    return records


class TestPauseIntelligenceSuite(unittest.TestCase):
    """Consolidated test suite covering all 10 specifications of Task 3."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="pause_intel_suite_")
        # Ensure fresh model cache and enabled learning flag for every test
        pi._MODEL_CACHE.clear()
        self._orig_env = os.environ.get("AUTOSHORTS_SMART_PACING_LEARNED")
        os.environ["AUTOSHORTS_SMART_PACING_LEARNED"] = "1"

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        pi._MODEL_CACHE.clear()
        if self._orig_env is not None:
            os.environ["AUTOSHORTS_SMART_PACING_LEARNED"] = self._orig_env
        else:
            os.environ.pop("AUTOSHORTS_SMART_PACING_LEARNED", None)

    # ──────────────────────────────────────────────────────────────────────────
    # Test 1: Feature Vector Dimension and Order
    # ──────────────────────────────────────────────────────────────────────────
    def test_feature_vector_dimension_and_order(self):
        """Validates that FEATURE_ORDER has exactly 23 features in pinned order,
        and features_to_row converts feature dict to 23-float vector."""
        # 1. Pinned order check
        self.assertEqual(len(pi.FEATURE_ORDER), 23)
        self.assertEqual(pi.FEATURE_ORDER, PINNED_23_FEATURES)
        self.assertEqual(pl.FEATURE_ORDER, PINNED_23_FEATURES)
        self.assertEqual(pt.FEATURE_ORDER, PINNED_23_FEATURES)

        # 2. features_to_row with complete dict
        sample_dict = {name: float(idx * 1.5) for idx, name in enumerate(PINNED_23_FEATURES)}
        row = pi.features_to_row(sample_dict)
        self.assertIsInstance(row, list)
        self.assertEqual(len(row), 23)
        for idx, val in enumerate(row):
            self.assertIsInstance(val, float)
            self.assertEqual(val, float(idx * 1.5))

        # 3. features_to_row with partial dict (defaults to 0.0)
        partial_dict = {"gapRmsDb": -32.5, "hfRatio": 0.42, "speakerChange": 1.0}
        partial_row = pi.features_to_row(partial_dict)
        self.assertEqual(len(partial_row), 23)
        self.assertEqual(partial_row[0], -32.5)  # gapRmsDb is index 0
        self.assertEqual(partial_row[3], 0.42)   # hfRatio is index 3
        self.assertEqual(partial_row[9], 1.0)    # speakerChange is index 9
        self.assertEqual(partial_row[1], 0.0)    # preSpeechRmsDb missing -> 0.0

        # 4. features_to_row with empty dict
        empty_row = pi.features_to_row({})
        self.assertEqual(len(empty_row), 23)
        self.assertEqual(empty_row, [0.0] * 23)

    # ──────────────────────────────────────────────────────────────────────────
    # Test 2: Acoustic Feature Calculations
    # ──────────────────────────────────────────────────────────────────────────
    def test_acoustic_feature_calculations(self):
        """Validates gap_acoustic_features on synthetic audio signals:
        silence, pure sine tone (<2kHz), pure sine tone (>2kHz), and noise."""
        sr = 16000
        dur = 2.0
        n_samples = int(dur * sr)

        # 1. Pure silence signal
        silence_x = np.zeros(n_samples, dtype=np.float32)
        feats_silence = pi.gap_acoustic_features(
            silence_x,
            gap_t0=0.5, gap_t1=1.0,
            pre_t0=0.0, pre_t1=0.5,
            post_t0=1.0, post_t1=1.5,
        )
        self.assertLessEqual(feats_silence["gapRmsDb"], -90.0)
        self.assertEqual(feats_silence["zcr"], 0.0)
        self.assertEqual(feats_silence["hfRatio"], 0.0)

        # 2. Low-frequency sine tone (400 Hz < 2000 Hz)
        t = np.arange(n_samples, dtype=np.float32) / sr
        tone_400 = (0.707 * np.sin(2 * np.pi * 400.0 * t)).astype(np.float32)
        feats_400 = pi.gap_acoustic_features(
            tone_400,
            gap_t0=0.5, gap_t1=1.0,
            pre_t0=0.0, pre_t1=0.5,
            post_t0=1.0, post_t1=1.5,
        )
        # RMS should be near -3 dB (0.707 peak -> ~0.5 RMS -> -6 dB or ~ -3 dB for 0.707 amplitude)
        self.assertGreater(feats_400["gapRmsDb"], -10.0)
        self.assertLess(feats_400["gapRmsDb"], 0.0)
        # hfRatio should be negligible (< 0.05) since 400 Hz << 2000 Hz
        self.assertLess(feats_400["hfRatio"], 0.05)
        # Wiener flatness in [0, 1]; pure sine tone has low spectral flatness
        self.assertGreaterEqual(feats_400["flatness"], 0.0)
        self.assertLess(feats_400["flatness"], 0.10)
        # Zero crossing rate in [0, 1]
        self.assertGreaterEqual(feats_400["zcr"], 0.0)
        self.assertLessEqual(feats_400["zcr"], 1.0)
        # 400 Hz at 16kHz -> 2 zero-crossings per cycle -> 2 * 400 / 16000 = 0.05 crossings per sample
        self.assertAlmostEqual(feats_400["zcr"], (2.0 * 400.0) / sr, delta=0.01)

        # 3. High-frequency sine tone (4000 Hz > 2000 Hz)
        tone_4000 = (0.707 * np.sin(2 * np.pi * 4000.0 * t)).astype(np.float32)
        feats_4000 = pi.gap_acoustic_features(
            tone_4000,
            gap_t0=0.5, gap_t1=1.0,
            pre_t0=0.0, pre_t1=0.5,
            post_t0=1.0, post_t1=1.5,
        )
        # hfRatio should be high (> 0.95) since 4000 Hz >> 2000 Hz
        self.assertGreater(feats_4000["hfRatio"], 0.95)
        # ZCR for 4000 Hz should be ~10x that of 400 Hz
        self.assertGreater(feats_4000["zcr"], feats_400["zcr"] * 5.0)

        # 4. White noise (broadband)
        rng = np.random.RandomState(42)
        noise = rng.normal(0.0, 0.2, n_samples).astype(np.float32)
        feats_noise = pi.gap_acoustic_features(
            noise,
            gap_t0=0.5, gap_t1=1.0,
            pre_t0=0.0, pre_t1=0.5,
            post_t0=1.0, post_t1=1.5,
        )
        # Broadband noise has higher spectral flatness than pure tone
        self.assertGreater(feats_noise["flatness"], feats_400["flatness"])
        self.assertGreater(feats_noise["flatness"], 0.10)
        # White noise has energy up to 8 kHz, so >2 kHz portion should be significant
        self.assertGreater(feats_noise["hfRatio"], 0.40)

        # 5. Dip and Level drop assertions
        self.assertIn("dipDb", feats_400)
        self.assertIn("levelDropDb", feats_400)
        self.assertLessEqual(feats_400["dipDb"], 0.0)

    # ──────────────────────────────────────────────────────────────────────────
    # Test 3: Labeling Duration Filters
    # ──────────────────────────────────────────────────────────────────────────
    def test_labeling_duration_filters(self):
        """Uses pause_label filtering logic on gaps with varying durations:
        - 0.03s -> rejected (< 0.05s)
        - 0.30s -> accepted
        - 5.50s -> rejected (> 5.0s)"""
        # Test unit filter logic
        durations = [0.03, 0.049, 0.05, 0.30, 4.99, 5.0, 5.01, 5.50]
        results = []
        for d in durations:
            gs = 1.0
            ge = gs + d
            dur = ge - gs
            rejected = (dur < 0.05 or dur > 5.0)
            results.append((d, not rejected))

        accepted_durations = [d for d, acc in results if acc]
        rejected_durations = [d for d, acc in results if not acc]

        self.assertIn(0.03, rejected_durations)
        self.assertIn(5.50, rejected_durations)
        self.assertIn(0.30, accepted_durations)

        # Integration test via process_media_transcript
        wav_path = os.path.join(self.tmp_dir, "duration_filter_mock.wav")
        with open(wav_path, "wb") as f:
            f.write(b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00")

        words = [
            {"word": "w1", "start": 0.0, "end": 1.0, "speaker": "SPK0"},
            {"word": "w2", "start": 1.03, "end": 2.0, "speaker": "SPK0"},
            {"word": "w3", "start": 2.30, "end": 3.0, "speaker": "SPK0"},
            {"word": "w4", "start": 8.50, "end": 9.0, "speaker": "SPK0"},
        ]
        words_path = os.path.join(self.tmp_dir, "words.json")
        with open(words_path, "w", encoding="utf-8") as f:
            json.dump(words, f)

        mock_raw_gaps = [
            {"srcStartSec": 1.0, "srcEndSec": 1.03, "durationSec": 0.03},
            {"srcStartSec": 2.0, "srcEndSec": 2.30, "durationSec": 0.30},
            {"srcStartSec": 3.0, "srcEndSec": 8.50, "durationSec": 5.50},
        ]
        mock_features = [{"gapRmsDb": -30.0, "speakerChange": 0.0, "durationSec": 0.30}]

        with patch.object(pl.breath_detect, "detect_gaps", return_value={"status": "ok", "gaps": mock_raw_gaps}), \
             patch.object(pl.pi, "build_gap_features", return_value=mock_features):
            recs = pl.process_media_transcript(
                wav_path,
                words_path,
                clip_start=0.0,
                clip_end=10.0,
            )

        self.assertEqual(len(recs), 1)
        self.assertEqual(round(recs[0]["srcEndSec"] - recs[0]["srcStartSec"], 2), 0.30)
        self.assertEqual(recs[0]["srcStartSec"], 2.0)
        self.assertEqual(recs[0]["srcEndSec"], 2.30)

    # ──────────────────────────────────────────────────────────────────────────
    # Test 4: Weak Supervision Mapping
    # ──────────────────────────────────────────────────────────────────────────
    def test_weak_supervision_mapping(self):
        """Validates deterministic label assignment across all 6 classes:
        - speakerChange == 1.0 -> speaker_transition
        - Removable/quiet gap with hfRatio >= 0.15 -> breath_pause
        - Removable/quiet gap with hfRatio < 0.15 -> waiting_pause
        - Non-removable gap ending sentence -> sentence_pause
        - Non-removable gap inside sentence -> normal_word_gap
        - Explicit nonspeech -> nonspeech_unknown
        And validates human annotations override weak supervision with weakSupervision: false."""
        # 1. speaker_transition
        gap1 = {"category": "normal_word_gap"}
        feat1 = {"speakerChange": 1.0}
        lbl1, src1, weak1 = pl.resolve_gap_label(gap1, feat1, None)
        self.assertEqual(lbl1, "speaker_transition")
        self.assertTrue(weak1)
        self.assertEqual(src1, "weak_supervision_engine_decision")

        # 2. breath_pause (removable/quiet gap with hfRatio >= 0.15)
        gap2 = {"category": "breath_pause", "confidence": 0.85}
        feat2 = {"speakerChange": 0.0, "hfRatio": 0.22, "evHfRatio": 0.20}
        lbl2, src2, weak2 = pl.resolve_gap_label(gap2, feat2, None)
        self.assertEqual(lbl2, "breath_pause")
        self.assertTrue(weak2)

        # Also via VAD quiet condition
        gap2_vad = {}
        feat2_vad = {"speakerChange": 0.0, "vadSilenceFrac": 0.80, "levelDropDb": -10.0, "hfRatio": 0.30}
        lbl2_vad, _, weak2_vad = pl.resolve_gap_label(gap2_vad, feat2_vad, None)
        self.assertEqual(lbl2_vad, "breath_pause")
        self.assertTrue(weak2_vad)

        # 3. waiting_pause (removable/quiet gap with hfRatio < 0.15)
        gap3 = {"category": "waiting_pause", "confidence": 0.80}
        feat3 = {"speakerChange": 0.0, "hfRatio": 0.04, "evHfRatio": 0.02}
        lbl3, src3, weak3 = pl.resolve_gap_label(gap3, feat3, None)
        self.assertEqual(lbl3, "waiting_pause")
        self.assertTrue(weak3)

        # 4. sentence_pause (non-removable gap ending sentence)
        gap4 = {"category": "sentence_pause"}
        feat4 = {"speakerChange": 0.0, "endsSentence": 1.0, "vadSilenceFrac": 0.1, "levelDropDb": 0.0}
        lbl4, src4, weak4 = pl.resolve_gap_label(gap4, feat4, None)
        self.assertEqual(lbl4, "sentence_pause")
        self.assertTrue(weak4)

        # 5. normal_word_gap (non-removable gap inside sentence)
        gap5 = {"category": "normal_word_gap"}
        feat5 = {"speakerChange": 0.0, "endsSentence": 0.0, "vadSilenceFrac": 0.1, "levelDropDb": 0.0}
        lbl5, src5, weak5 = pl.resolve_gap_label(gap5, feat5, None)
        self.assertEqual(lbl5, "normal_word_gap")
        self.assertTrue(weak5)

        # 6. nonspeech_unknown (explicit nonspeech)
        gap6 = {"category": "nonspeech_unknown"}
        feat6 = {"speakerChange": 0.0, "endsSentence": 0.0, "vadSilenceFrac": 0.1, "levelDropDb": 0.0}
        lbl6, src6, weak6 = pl.resolve_gap_label(gap6, feat6, None)
        self.assertEqual(lbl6, "nonspeech_unknown")
        self.assertTrue(weak6)

        # 7. Human annotation overrides weak supervision
        for target_cls in pl.VALID_CLASSES:
            human_rec = {"class": target_cls, "sourceHash": "test_hash"}
            # Given conflicting heuristics (e.g. speakerChange=1.0)
            lbl_h, src_h, weak_h = pl.resolve_gap_label(
                {"category": "normal_word_gap"},
                {"speakerChange": 1.0},
                human_rec,
            )
            self.assertEqual(lbl_h, target_cls)
            self.assertEqual(src_h, "human_annotation")
            self.assertFalse(weak_h)

    # ──────────────────────────────────────────────────────────────────────────
    # Test 5: Trainer Refuses Under 200 Samples
    # ──────────────────────────────────────────────────────────────────────────
    def test_trainer_refuses_under_200_samples(self):
        """Validates that pause_train.py on a dataset with < 200 samples
        (e.g. 50 samples or non-existent file) prints verbatim
        'NO LABELS — classifier not trained' (with Unicode em dash \\u2014)
        and exits with return code 0."""
        # 1. File with 50 samples (< 200) via CLI verifies returncode 0 and verbatim em dash message
        under_200_file = os.path.join(self.tmp_dir, "under_200.jsonl")
        records_50 = generate_synthetic_dataset(num_samples=50, num_videos=3)
        with open(under_200_file, "w", encoding="utf-8") as f:
            for r in records_50:
                f.write(json.dumps(r) + "\n")

        models_dir = os.path.join(self.tmp_dir, "models")
        cmd = [
            sys.executable, os.path.join(SCRIPT_DIR, "pause_train.py"),
            "--data", under_200_file,
            "--models-dir", models_dir,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(res.returncode, 0)
        self.assertIn("NO LABELS \u2014 classifier not trained", res.stdout)

        # Verify no model files were written
        self.assertFalse(os.path.exists(os.path.join(models_dir, pt.MODEL_FILENAME)))
        self.assertFalse(os.path.exists(os.path.join(models_dir, pt.META_FILENAME)))

        # 2. Non-existent file also safely handled
        non_existent = os.path.join(self.tmp_dir, "non_existent.jsonl")
        ret_non_existent = pt.train_pipeline(data_path=non_existent, models_dir=models_dir, quiet=True)
        self.assertEqual(ret_non_existent, 0)
        self.assertFalse(os.path.exists(os.path.join(models_dir, pt.MODEL_FILENAME)))

    # ──────────────────────────────────────────────────────────────────────────
    # Test 6: Trainer Disjoint Video Partition
    # ──────────────────────────────────────────────────────────────────────────
    def test_trainer_disjoint_video_partition(self):
        """Validates partition_by_video on multi-video dataset, asserting that
        the intersection of sourceHash sets between train and test partitions
        is strictly empty (V_train ∩ V_test = ∅)."""
        # 240 samples across 8 distinct video source hashes
        records = generate_synthetic_dataset(num_samples=240, num_videos=8, seed=123)

        train_recs, test_recs = pt.partition_by_video(records, test_size=0.20, seed=42)

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

    # ──────────────────────────────────────────────────────────────────────────
    # Test 7: Model Export and Metadata Schema
    # ──────────────────────────────────────────────────────────────────────────
    def test_model_export_and_meta_schema(self):
        """Trains a model on 200+ synthetic gap records, verifying:
        - exported pause_classifier_v1.txt is valid Booster text
        - pause_classifier_v1.meta.json schema conforms to contract."""
        dataset_path = os.path.join(self.tmp_dir, "dataset_240.jsonl")
        records = generate_synthetic_dataset(num_samples=240, num_videos=6, seed=42)
        with open(dataset_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        models_dir = os.path.join(self.tmp_dir, "models")
        ret = pt.train_pipeline(
            data_path=dataset_path,
            models_dir=models_dir,
            min_samples=200,
            threshold=0.65,
            test_size=0.20,
            seed=42,
            num_boost_round=15,
            early_stopping_rounds=5,
            quiet=True,
        )
        self.assertEqual(ret, 0)

        # 1. Booster model file validation
        model_file = os.path.join(models_dir, pt.MODEL_FILENAME)
        self.assertTrue(os.path.isfile(model_file))
        self.assertGreater(os.path.getsize(model_file), 100)

        # Verify it loads cleanly as LightGBM Booster
        import lightgbm as lgb
        booster = lgb.Booster(model_file=model_file)
        self.assertEqual(booster.num_feature(), 23)

        # 2. Metadata JSON file schema validation
        meta_file = os.path.join(models_dir, pt.META_FILENAME)
        self.assertTrue(os.path.isfile(meta_file))

        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.assertEqual(meta["modelFile"], pt.MODEL_FILENAME)
        self.assertEqual(meta["featureVersion"], "1")
        self.assertEqual(meta["threshold"], 0.65)
        self.assertEqual(meta["classes"], pt.CLASSES)
        self.assertEqual(meta["removableClasses"], ["breath_pause", "waiting_pause"])
        self.assertEqual(meta["features"], pt.FEATURE_ORDER)
        self.assertEqual(len(meta["features"]), 23)

        # trainedAt must be valid ISO 8601 string
        self.assertIn("trainedAt", meta)
        self.assertIn("T", meta["trainedAt"])
        self.assertTrue(meta["trainedAt"].endswith("Z"))

        # trainStats validation
        self.assertIn("trainStats", meta)
        stats = meta["trainStats"]
        self.assertEqual(stats["totalSamples"], 240)
        self.assertGreater(stats["trainSamples"], 0)
        self.assertGreater(stats["testSamples"], 0)
        self.assertEqual(stats["trainSamples"] + stats["testSamples"], 240)
        self.assertIn("removablePrecision", stats)
        self.assertIn("removableRecall", stats)
        self.assertIn("logloss", stats)
        self.assertIn("multiError", stats)

    # ──────────────────────────────────────────────────────────────────────────
    # Test 8: Model Scoring Removable Classes
    # ──────────────────────────────────────────────────────────────────────────
    def test_model_scoring_removable_classes(self):
        """Uses pause_intelligence.score_gaps_with_model with trained model.
        Asserts:
        - pRemovable = proba[:, 0] + proba[:, 1] (breath_pause + waiting_pause)
        - high probability for non-removable classes (e.g. sentence_pause)
          does NOT contribute to pRemovable
        - pRemovableAboveThreshold is True only when pRemovable >= threshold."""
        # 1. Train a model on synthetic data
        dataset_path = os.path.join(self.tmp_dir, "score_dataset.jsonl")
        records = generate_synthetic_dataset(num_samples=240, num_videos=6, seed=42)
        with open(dataset_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        models_dir = os.path.join(self.tmp_dir, "models")
        pt.train_pipeline(
            data_path=dataset_path,
            models_dir=models_dir,
            min_samples=200,
            threshold=0.65,
            num_boost_round=10,
            quiet=True,
        )
        model_path = os.path.join(models_dir, pt.MODEL_FILENAME)

        # 2. Test scoring with trained model
        test_gaps = [
            {"srcStartSec": 0.5, "srcEndSec": 0.8},
            {"srcStartSec": 1.2, "srcEndSec": 1.5},
        ]
        test_rows = [
            [0.1] * 23,
            [-20.0] * 23,
        ]

        scored = pi.score_gaps_with_model(test_gaps, test_rows, model_path)
        self.assertTrue(scored)

        # Verify against booster predictions directly
        loaded = pi.load_pause_classifier(model_path)
        self.assertIsNotNone(loaded)
        booster, meta = loaded
        proba = booster.predict(np.array(test_rows, dtype=np.float64))

        for idx, gap in enumerate(test_gaps):
            expected_p_rem = float(proba[idx, 0] + proba[idx, 1])
            self.assertAlmostEqual(gap["pRemovable"], expected_p_rem, places=5)
            self.assertEqual(gap["pRemovableAboveThreshold"], (gap["pRemovable"] >= 0.65))

        # 3. Mathematical proof of non-removable class isolation
        # Construct mock booster returning specific class probabilities:
        # Case A: High removable probability: breath (0.50) + waiting (0.20) = 0.70 >= 0.65
        # Case B: High NON-REMOVABLE probability: sentence_pause = 0.85, removable = 0.10 < 0.65
        mock_booster = MagicMock()
        mock_booster.predict.return_value = np.array([
            [0.50, 0.20, 0.15, 0.05, 0.05, 0.05],  # Sum of [0, 1] = 0.70
            [0.05, 0.05, 0.85, 0.02, 0.02, 0.01],  # Class 2 is 0.85, sum of [0, 1] = 0.10
        ])
        mock_meta = {
            "modelFile": "mock.txt",
            "featureVersion": "1",
            "threshold": 0.65,
            "classes": pt.CLASSES,
            "removableClasses": ["breath_pause", "waiting_pause"],
        }

        mock_gaps = [
            {"name": "gap_removable"},
            {"name": "gap_sentence_pause"},
        ]

        with patch.object(pi, "load_pause_classifier", return_value=(mock_booster, mock_meta)):
            pi.score_gaps_with_model(mock_gaps, [[0.0] * 23, [0.0] * 23], "dummy_path.txt")

        # Gap A: removable sum = 0.70 >= 0.65 -> above threshold
        self.assertAlmostEqual(mock_gaps[0]["pRemovable"], 0.70, places=5)
        self.assertTrue(mock_gaps[0]["pRemovableAboveThreshold"])

        # Gap B: sentence_pause is 0.85, but removable sum is only 0.10 < 0.65
        self.assertAlmostEqual(mock_gaps[1]["pRemovable"], 0.10, places=5)
        self.assertFalse(mock_gaps[1]["pRemovableAboveThreshold"])

    # ──────────────────────────────────────────────────────────────────────────
    # Test 9: Feature Version Mismatch Refusal
    # ──────────────────────────────────────────────────────────────────────────
    def test_feature_version_mismatch_refusal(self):
        """Modifies/mocks metadata file with featureVersion: '999'.
        Asserts pause_intelligence.load_pause_classifier refuses to load
        (returns None) and score_gaps_with_model returns False."""
        pi._MODEL_CACHE.clear()

        # 1. Create a minimal valid booster file and an incompatible meta file
        mismatch_dir = os.path.join(self.tmp_dir, "mismatch_model")
        os.makedirs(mismatch_dir, exist_ok=True)
        model_file = os.path.join(mismatch_dir, "pause_classifier_v1.txt")
        meta_file = os.path.join(mismatch_dir, "pause_classifier_v1.meta.json")

        # Fast direct booster generation (5ms)
        import lightgbm as lgb
        train_data = lgb.Dataset(np.zeros((6, 23)), label=[0, 1, 2, 3, 4, 5], feature_name=pt.FEATURE_ORDER)
        booster = lgb.train({"objective": "multiclass", "num_class": 6, "verbosity": -1}, train_data, num_boost_round=1)
        booster.save_model(model_file)

        # Incompatible metadata file with featureVersion "999"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump({
                "modelFile": os.path.basename(model_file),
                "featureVersion": "999",
                "threshold": 0.65,
                "classes": pt.CLASSES,
            }, f)

        # Clear cache and verify refusal
        pi._MODEL_CACHE.clear()
        loaded = pi.load_pause_classifier(model_file)
        self.assertIsNone(loaded, "load_pause_classifier must return None on featureVersion mismatch")

        # Verify score_gaps_with_model fails soft
        test_gaps = [{"srcStartSec": 0.5, "srcEndSec": 0.8}]
        test_rows = [[0.0] * 23]
        scored = pi.score_gaps_with_model(test_gaps, test_rows, model_file)
        self.assertFalse(scored)
        self.assertNotIn("pRemovable", test_gaps[0])

    # ──────────────────────────────────────────────────────────────────────────
    # Test 10: Predict CLI JSON Payload
    # ──────────────────────────────────────────────────────────────────────────
    def test_predict_cli_json_payload(self):
        """Validates CLI execution of pause_intelligence.py with synthetic files,
        confirming JSON output schema containing sourceHash, featureVersion: '1',
        gaps, and modelScored."""
        # 1. Create test audio and words file
        wav_path = os.path.join(self.tmp_dir, "cli_test.wav")
        tone_spans = (
            (0.1, 0.4, 440.0, 0.8),
            (0.6, 0.8, 440.0, 0.8),
            (1.1, 1.4, 440.0, 0.8),
        )
        create_test_wav(wav_path, duration=1.5, tone_spans=tone_spans)

        words = [
            {"word": "Hello", "start": 0.1, "end": 0.4, "speaker": "SPK0"},
            {"word": "world", "start": 0.6, "end": 0.8, "speaker": "SPK0"},
            {"word": "done", "start": 1.1, "end": 1.4, "speaker": "SPK0"},
        ]
        words_path = os.path.join(self.tmp_dir, "cli_words.json")
        with open(words_path, "w", encoding="utf-8") as f:
            json.dump(words, f)

        # 2. Execute pause_intelligence.py CLI writing to --out
        out_json_path = os.path.join(self.tmp_dir, "cli_output.json")
        cmd = [
            sys.executable,
            os.path.join(SCRIPT_DIR, "pause_intelligence.py"),
            wav_path,
            "0", "1500",  # start_ms, end_ms
            words_path,
            "--out", out_json_path,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(res.returncode, 0, f"CLI stderr: {res.stderr}")
        self.assertTrue(os.path.isfile(out_json_path))

        with open(out_json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        # 3. Validate schema keys
        self.assertIn("sourceHash", payload)
        self.assertEqual(payload["sourceHash"], pi.source_hash(wav_path))
        self.assertIn("featureVersion", payload)
        self.assertEqual(payload["featureVersion"], "1")
        self.assertIn("gaps", payload)
        self.assertIsInstance(payload["gaps"], list)
        self.assertIn("modelScored", payload)
        self.assertIsInstance(payload["modelScored"], bool)

        # Also verify programmatic main execution produces valid payload
        out_json_path2 = os.path.join(self.tmp_dir, "cli_output2.json")
        pi.main([wav_path, "0", "1500", words_path, "--out", out_json_path2])
        self.assertTrue(os.path.isfile(out_json_path2))
        with open(out_json_path2, "r", encoding="utf-8") as f:
            payload2 = json.load(f)
        self.assertEqual(payload2["featureVersion"], "1")
        self.assertEqual(payload2["sourceHash"], pi.source_hash(wav_path))


if __name__ == "__main__":
    unittest.main()
