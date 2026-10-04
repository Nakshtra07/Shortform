#!/usr/bin/env python3
"""
AutoShorts 11.0 — T7 Prosodic Boundary Suite (Unit & Integration Tests)

Validates the complete T7 prosodic boundary tagging subsystem:
    1. Feature extraction schema (all 8 features present, correct ranges).
    2. Acoustic features (F0 slope, energy drop, VAD silence fraction).
    3. Sentence and clause punctuation rules (including Hindi danda/double danda).
    4. Deterministic rule prior (pBoundaryRule mathematical invariants).
    5. Label mode (--label with human annotations and weak-label fallback).
    6. Training safety guard (< 200 samples safe refusal exit 0).
    7. Training pipeline (>= 200 samples LightGBM training with per-video split).
    8. Sidecar inference (--predict mode and graceful degradation fallbacks).

Runner compatibility:
    python -m unittest autoshorts/src-tauri/scripts/test_t7_prosody_suite.py
    python -m unittest scripts/test_t7_prosody_suite.py
"""

import os
import sys
import json
import wave
import shutil
import tempfile
import unittest
import subprocess
from typing import Tuple

# Ensure script directory is on sys.path for direct imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import numpy as np

T7_PROSODY_PY = os.path.join(SCRIPT_DIR, "t7_prosody.py")
T7_TRAIN_PY = os.path.join(SCRIPT_DIR, "t7_train.py")

EXPECTED_FEATURE_KEYS = [
    "pauseSec",
    "endsSentence",
    "endsClause",
    "speakerChange",
    "f0SlopePre",
    "energyDropDb",
    "vadSilenceFrac",
    "pBoundaryRule",
]


def make_synth_wav(path: str, duration: float, tone_spans: Tuple[Tuple[float, float, float, float], ...] = ()) -> str:
    """Generate a 16 kHz mono 16-bit PCM WAV with tone spans (start, end, freq_hz, amp)."""
    sr = 16000
    n = int(duration * sr)
    x = np.zeros(n, dtype=np.float32)
    for t0, t1, freq, amp in tone_spans:
        i0 = max(0, int(t0 * sr))
        i1 = min(n, int(t1 * sr))
        if i1 > i0:
            t = np.arange(i1 - i0, dtype=np.float32) / sr
            x[i0:i1] = amp * np.sin(2.0 * np.pi * freq * t)

    int_samples = (np.clip(x, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(int_samples.tobytes())
    return path


def generate_synthetic_labels_jsonl(path: str, sample_count: int, num_videos: int = 4) -> None:
    """Generate balanced synthetic JSONL dataset with >= sample_count across distinct videos."""
    records = []
    rng = np.random.RandomState(42)
    video_ids = [f"vid_{i:02d}.mp4" for i in range(num_videos)]

    for i in range(sample_count):
        vid = video_ids[i % num_videos]
        pause = float(rng.uniform(0.1, 1.8))
        ends_sent = 1 if rng.rand() > 0.6 else 0
        ends_cl = 1 if (not ends_sent and rng.rand() > 0.5) else 0
        spk_chg = 1 if rng.rand() > 0.85 else 0
        f0_slope = float(rng.normal(-5.0, 20.0))
        energy_drop = float(rng.uniform(0.0, 18.0))
        vad_sil = float(min(1.0, pause / 0.8 * rng.uniform(0.8, 1.0)))

        # Rule prior computation
        p = min(1.0, pause / 0.6) * 0.55 + ends_sent * 1.0 + ends_cl * 0.45
        p += 0.15 if energy_drop > 6.0 else 0.0
        p -= spk_chg * 1.0
        p_boundary_rule = round(max(0.0, min(1.0, p)), 3)

        # Ground truth boundary (balanced)
        is_boundary = 1 if p_boundary_rule >= 0.65 or rng.rand() > 0.7 else 0

        rec = {
            "video": vid,
            "gapIdx": i,
            "wordIdxBefore": i,
            "wordIdxAfter": i + 1,
            "prevText": f"word_{i}.",
            "nextText": f"word_{i+1}",
            "pauseSec": round(pause, 3),
            "endsSentence": ends_sent,
            "endsClause": ends_cl,
            "speakerChange": spk_chg,
            "f0SlopePre": round(f0_slope, 2),
            "energyDropDb": round(energy_drop, 2),
            "vadSilenceFrac": round(vad_sil, 3),
            "pBoundaryRule": p_boundary_rule,
            "isBoundary": is_boundary,
            "labelType": "synthetic",
            "confidence": 1.0,
        }
        records.append(rec)

    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


class TestT7ProsodyFeaturePipeline(unittest.TestCase):
    """Tests feature extraction, acoustic measurements, and rule prior invariants."""

    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.mkdtemp(prefix="test_t7_pipeline_")
        cls.wav_multitonal = os.path.join(cls.tmp_dir, "multitonal.wav")
        # 3.0s clip: tone 0.0-0.5s (220Hz), silence 0.5-1.2s, tone 1.2-1.8s (180Hz), silence 1.8-3.0s
        make_synth_wav(cls.wav_multitonal, 3.0, (
            (0.0, 0.5, 220.0, 0.6),
            (1.2, 1.8, 180.0, 0.6),
        ))

        cls.wav_silence = os.path.join(cls.tmp_dir, "pure_silence.wav")
        make_synth_wav(cls.wav_silence, 2.0, ())

        cls.wav_continuous = os.path.join(cls.tmp_dir, "continuous_tone.wav")
        make_synth_wav(cls.wav_continuous, 2.0, ((0.0, 2.0, 250.0, 0.6),))

        cls.words = [
            {"text": "Hello", "start": 0.1, "end": 0.5, "speaker": "S1"},
            {"text": "world.", "start": 1.2, "end": 1.8, "speaker": "S1"},
            {"text": "Next?", "start": 2.2, "end": 2.7, "speaker": "S2"},
        ]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    def test_01_feature_schema_all_eight_keys_present(self):
        """Verifies all 8 required prosodic feature keys and metadata are returned."""
        import t7_prosody as tp
        records = tp.build_gap_records(self.wav_multitonal, 0.0, 3.0, self.words)
        self.assertEqual(len(records), len(self.words) - 1, "Exactly 1 record per word gap")

        for r in records:
            # Metadata keys
            for meta_k in ("gapIdx", "wordIdxBefore", "wordIdxAfter", "prevText", "nextText"):
                self.assertIn(meta_k, r, f"Missing metadata key: {meta_k}")
            # All 8 feature keys
            for feat_k in EXPECTED_FEATURE_KEYS:
                self.assertIn(feat_k, r, f"Missing feature key: {feat_k}")

    def test_02_vad_silence_fraction_calculation(self):
        """Verifies vadSilenceFrac approaches 1.0 on silence, <0.25 on continuous tone, 0.0 on zero pause."""
        import t7_prosody as tp
        words_gap = [
            {"text": "first", "start": 0.1, "end": 0.4, "speaker": "S1"},
            {"text": "second", "start": 1.4, "end": 1.8, "speaker": "S1"},
        ]
        # Pure silence WAV
        recs_sil = tp.build_gap_records(self.wav_silence, 0.0, 2.0, words_gap)
        self.assertGreaterEqual(recs_sil[0]["vadSilenceFrac"], 0.90, "Pure silence must have vadSilenceFrac >= 0.90")

        # Continuous tone WAV
        recs_tone = tp.build_gap_records(self.wav_continuous, 0.0, 2.0, words_gap)
        self.assertLess(recs_tone[0]["vadSilenceFrac"], 0.25, "Continuous tone must have vadSilenceFrac < 0.25")

        # Zero pause gap
        words_zero = [
            {"text": "fast", "start": 0.2, "end": 0.5, "speaker": "S1"},
            {"text": "pace", "start": 0.5, "end": 0.8, "speaker": "S1"},
        ]
        recs_zero = tp.build_gap_records(self.wav_silence, 0.0, 2.0, words_zero)
        self.assertEqual(recs_zero[0]["vadSilenceFrac"], 0.0, "Zero pause duration must have vadSilenceFrac == 0.0")

    def test_03_acoustic_features_measured_from_synthetic_audio(self):
        """Acoustic features f0SlopePre and energyDropDb are non-null floats on real audio; fail soft when audio is None."""
        import t7_prosody as tp
        recs = tp.build_gap_records(self.wav_multitonal, 0.0, 3.0, self.words)
        rec0 = recs[0]
        self.assertIsNotNone(rec0["f0SlopePre"])
        self.assertIsInstance(rec0["f0SlopePre"], (int, float))
        self.assertIsNotNone(rec0["energyDropDb"])
        self.assertIsInstance(rec0["energyDropDb"], (int, float))
        self.assertGreater(rec0["energyDropDb"], 5.0, "Energy must drop significantly into silence gap")

        # Missing audio source fails soft without exception
        recs_no_audio = tp.build_gap_records(None, 0.0, 3.0, self.words)
        self.assertIsNone(recs_no_audio[0]["f0SlopePre"])
        self.assertIsNone(recs_no_audio[0]["energyDropDb"])
        self.assertIsNotNone(recs_no_audio[0]["pBoundaryRule"])

    def test_04_ends_sentence_and_clause_punctuation_rules(self):
        """Verifies sentence terminators (., !, ?, ;, :, ।, ॥, …) set endsSentence=1; comma sets endsClause=1."""
        import t7_prosody as tp
        terminators = [".", "!", "?", ";", ":", "।", "॥", "…"]
        for term in terminators:
            words = [
                {"text": f"word{term}", "start": 0.0, "end": 0.5},
                {"text": "next", "start": 1.0, "end": 1.5},
            ]
            r = tp.build_gap_records(None, 0.0, 2.0, words)[0]
            self.assertEqual(r["endsSentence"], 1, f"Terminator '{term}' must set endsSentence=1")
            self.assertEqual(r["endsClause"], 0, f"Terminator '{term}' must not set endsClause=1")

        # Ellipsis with trailing whitespace
        words_ellipsis = [
            {"text": "waiting…  ", "start": 0.0, "end": 0.5},
            {"text": "next", "start": 1.0, "end": 1.5},
        ]
        r_el = tp.build_gap_records(None, 0.0, 2.0, words_ellipsis)[0]
        self.assertEqual(r_el["endsSentence"], 1, "Ellipsis with trailing spaces must set endsSentence=1")
        self.assertEqual(r_el["endsClause"], 0)

        # Comma clause boundary
        words_comma = [
            {"text": "however, ", "start": 0.0, "end": 0.5},
            {"text": "we", "start": 1.0, "end": 1.5},
        ]
        r_comma = tp.build_gap_records(None, 0.0, 2.0, words_comma)[0]
        self.assertEqual(r_comma["endsSentence"], 0)
        self.assertEqual(r_comma["endsClause"], 1)

        # Plain word
        words_plain = [
            {"text": "plain", "start": 0.0, "end": 0.5},
            {"text": "text", "start": 1.0, "end": 1.5},
        ]
        r_plain = tp.build_gap_records(None, 0.0, 2.0, words_plain)[0]
        self.assertEqual(r_plain["endsSentence"], 0)
        self.assertEqual(r_plain["endsClause"], 0)

    def test_05_speaker_change_detection(self):
        """Speaker ID transitions set speakerChange=1; identical or missing speakers set speakerChange=0."""
        import t7_prosody as tp
        # Different speakers
        words_diff = [
            {"text": "Alice", "start": 0.0, "end": 0.5, "speaker": "spk_1"},
            {"text": "Bob", "start": 1.0, "end": 1.5, "speaker": "spk_2"},
        ]
        self.assertEqual(tp.build_gap_records(None, 0.0, 2.0, words_diff)[0]["speakerChange"], 1)

        # Same speaker
        words_same = [
            {"text": "Alice", "start": 0.0, "end": 0.5, "speaker": "spk_1"},
            {"text": "Alice2", "start": 1.0, "end": 1.5, "speaker": "spk_1"},
        ]
        self.assertEqual(tp.build_gap_records(None, 0.0, 2.0, words_same)[0]["speakerChange"], 0)

        # Missing speaker
        words_none = [
            {"text": "None1", "start": 0.0, "end": 0.5},
            {"text": "None2", "start": 1.0, "end": 1.5, "speaker": "spk_1"},
        ]
        self.assertEqual(tp.build_gap_records(None, 0.0, 2.0, words_none)[0]["speakerChange"], 0)

    def test_05b_integer_speaker_zero_transitions(self):
        """Integer speaker 0 transitions (0 -> 1 and 1 -> 0) must trigger speakerChange=1 without falsy suppression."""
        import t7_prosody as tp
        # 0 -> 1 transition
        words_0_to_1 = [
            {"text": "Zero", "start": 0.0, "end": 0.5, "speaker": 0},
            {"text": "One", "start": 1.0, "end": 1.5, "speaker": 1},
        ]
        recs_01 = tp.build_gap_records(None, 0.0, 2.0, words_0_to_1)
        self.assertEqual(recs_01[0]["speakerChange"], 1, "Integer transition 0 -> 1 must trigger speakerChange=1")
        # Ensure rule prior reflects speaker change deduction
        self.assertEqual(recs_01[0]["pBoundaryRule"], 0.0, "Speaker turn must deduct 1.0 from rule prior")

        # 1 -> 0 transition
        words_1_to_0 = [
            {"text": "One", "start": 0.0, "end": 0.5, "speaker": 1},
            {"text": "Zero", "start": 1.0, "end": 1.5, "speaker": 0},
        ]
        recs_10 = tp.build_gap_records(None, 0.0, 2.0, words_1_to_0)
        self.assertEqual(recs_10[0]["speakerChange"], 1, "Integer transition 1 -> 0 must trigger speakerChange=1")

        # 0 -> 0 same speaker transition
        words_0_to_0 = [
            {"text": "ZeroA", "start": 0.0, "end": 0.5, "speaker": 0},
            {"text": "ZeroB", "start": 1.0, "end": 1.5, "speaker": 0},
        ]
        recs_00 = tp.build_gap_records(None, 0.0, 2.0, words_0_to_0)
        self.assertEqual(recs_00[0]["speakerChange"], 0, "Integer 0 -> 0 same speaker must not trigger speakerChange")

        # Mixed int and str representation of same speaker: 0 -> "0"
        words_mixed_same = [
            {"text": "ZeroInt", "start": 0.0, "end": 0.5, "speaker": 0},
            {"text": "ZeroStr", "start": 1.0, "end": 1.5, "speaker": "0"},
        ]
        recs_mixed = tp.build_gap_records(None, 0.0, 2.0, words_mixed_same)
        self.assertEqual(recs_mixed[0]["speakerChange"], 0, "Mixed int/str same speaker 0 -> '0' must not trigger speakerChange")

        # 0 -> None or empty string must not trigger speakerChange
        words_0_to_none = [
            {"text": "Zero", "start": 0.0, "end": 0.5, "speaker": 0},
            {"text": "None", "start": 1.0, "end": 1.5},
        ]
        self.assertEqual(tp.build_gap_records(None, 0.0, 2.0, words_0_to_none)[0]["speakerChange"], 0)

        # Empty string -> 0 must not trigger speakerChange
        words_empty_to_0 = [
            {"text": "Empty", "start": 0.0, "end": 0.5, "speaker": ""},
            {"text": "Zero", "start": 1.0, "end": 1.5, "speaker": 0},
        ]
        self.assertEqual(tp.build_gap_records(None, 0.0, 2.0, words_empty_to_0)[0]["speakerChange"], 0)

    def test_06_deterministic_rule_prior_bounded_and_weighted(self):
        """Mathematical verification that pBoundaryRule is bounded in [0.0, 1.0] and sentence ends dominate."""
        import t7_prosody as tp
        words_sent = [
            {"text": "Finished.", "start": 0.0, "end": 1.0, "speaker": "S1"},
            {"text": "Starting", "start": 2.0, "end": 3.0, "speaker": "S1"},
        ]
        r_sent = tp.build_gap_records(None, 0.0, 4.0, words_sent)[0]
        self.assertGreaterEqual(r_sent["pBoundaryRule"], 0.80, "Sentence end must produce high rule prior")
        self.assertLessEqual(r_sent["pBoundaryRule"], 1.00)

        words_spk = [
            {"text": "Turn1", "start": 0.0, "end": 0.5, "speaker": "S1"},
            {"text": "Turn2", "start": 0.6, "end": 1.0, "speaker": "S2"},
        ]
        r_spk = tp.build_gap_records(None, 0.0, 2.0, words_spk)[0]
        self.assertGreaterEqual(r_spk["pBoundaryRule"], 0.0, "Rule prior must not go below 0.0")

    def test_06b_decode_audio_guards_zero_and_non_positive_duration(self):
        """Verifies decode_audio returns None when end_sec <= start_sec, preventing FFmpeg EOF leaks."""
        import t7_prosody as tp
        # 1. Equal timestamps (start_sec == end_sec) on valid audio file must return None (NOT 1.0s leak)
        dec_equal = tp.decode_audio(self.wav_multitonal, 1.0, 1.0)
        self.assertIsNone(dec_equal, "decode_audio must return None when start_sec == end_sec")

        # 2. Equal zero timestamps (0.0, 0.0)
        dec_zero = tp.decode_audio(self.wav_multitonal, 0.0, 0.0)
        self.assertIsNone(dec_zero, "decode_audio must return None when start_sec == end_sec == 0.0")

        # 3. Inverted bounds (end_sec < start_sec)
        dec_inverted = tp.decode_audio(self.wav_multitonal, 2.0, 1.0)
        self.assertIsNone(dec_inverted, "decode_audio must return None when end_sec < start_sec")

        # 4. Sub-millisecond positive duration that rounds down to 0.000s in ffmpeg formatting
        dec_sub_ms = tp.decode_audio(self.wav_multitonal, 1.0, 1.0004)
        self.assertIsNone(dec_sub_ms, "decode_audio must return None when duration rounds to 0.000s")

        # 5. Non-positive interval / negative end_sec
        dec_neg = tp.decode_audio(self.wav_multitonal, -2.0, -1.0)
        self.assertIsNone(dec_neg, "decode_audio must return None for negative timestamp intervals")
        dec_neg_zero = tp.decode_audio(self.wav_multitonal, -1.0, 0.0)
        self.assertIsNone(dec_neg_zero, "decode_audio must return None when end_sec <= 0.0")

        # 6. Valid positive slice must still succeed and return float32 samples
        dec_valid = tp.decode_audio(self.wav_multitonal, 0.0, 1.0)
        self.assertIsNotNone(dec_valid, "decode_audio must succeed for valid positive duration")
        self.assertEqual(len(dec_valid), 16000, "1.0s decoded at 16kHz must yield exactly 16000 samples")
        self.assertEqual(dec_valid.dtype, np.float32)

        # 7. Nonexistent and None audio paths fail soft
        self.assertIsNone(tp.decode_audio(None, 0.0, 1.0))
        self.assertIsNone(tp.decode_audio("nonexistent_path_12345.wav", 0.0, 1.0))

    def test_06c_build_gap_records_with_zero_duration_audio_window(self):
        """Verifies build_gap_records succeeds with None acoustic features when audio duration is 0.0s."""
        import t7_prosody as tp
        recs = tp.build_gap_records(self.wav_multitonal, 1.0, 1.0, self.words)
        self.assertEqual(len(recs), len(self.words) - 1)
        for r in recs:
            self.assertIsNone(r["f0SlopePre"])
            self.assertIsNone(r["energyDropDb"])
            self.assertIsNone(r["vadSilenceFrac"])
            self.assertIsNotNone(r["pBoundaryRule"])


class TestT7ProsodyLabeling(unittest.TestCase):
    """Tests CLI --label mode: human annotation ingestion and weak-label fallback."""

    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.mkdtemp(prefix="test_t7_labeling_")
        cls.wav_path = os.path.join(cls.tmp_dir, "test.wav")
        make_synth_wav(cls.wav_path, 2.5, ((0.1, 0.6, 200.0, 0.5), (1.4, 2.0, 200.0, 0.5)))

        cls.words_path = os.path.join(cls.tmp_dir, "words.json")
        words_data = {
            "words": [
                {"text": "hello.", "start": 0.1, "end": 0.6, "speaker": "S1"},
                {"text": "world", "start": 1.4, "end": 1.9, "speaker": "S1"},
                {"text": "again", "start": 2.0, "end": 2.4, "speaker": "S1"},
            ]
        }
        with open(cls.words_path, "w", encoding="utf-8") as f:
            json.dump(words_data, f)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    def test_07_label_mode_with_human_annotations(self):
        """--label mode pairs human annotations and marks labelType='human' with confidence=1.0."""
        anno_path = os.path.join(self.tmp_dir, "human_anno.jsonl")
        with open(anno_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"gapIdx": 0, "isBoundary": 1}) + "\n")
            f.write(json.dumps({"gapIdx": 1, "isBoundary": 0}) + "\n")

        out_path = os.path.join(self.tmp_dir, "labeled_out.jsonl")
        res = subprocess.run([
            sys.executable, T7_PROSODY_PY,
            self.wav_path, "0", "2500", self.words_path,
            "--label", "--annotations", anno_path, "--out", out_path
        ], capture_output=True, text=True, timeout=30)

        self.assertEqual(res.returncode, 0, f"CLI label mode failed: {res.stderr}")
        self.assertTrue(os.path.exists(out_path), "Output JSONL must be created")

        with open(out_path, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]

        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0]["gapIdx"], 0)
        self.assertEqual(lines[0]["isBoundary"], 1)
        self.assertEqual(lines[0]["labelType"], "human")
        self.assertEqual(lines[0]["confidence"], 1.0)
        self.assertEqual(lines[1]["gapIdx"], 1)
        self.assertEqual(lines[1]["isBoundary"], 0)
        self.assertEqual(lines[1]["labelType"], "human")
        self.assertEqual(lines[1]["confidence"], 1.0)
        for feat in EXPECTED_FEATURE_KEYS:
            self.assertIn(feat, lines[0])

    def test_08_label_mode_weak_label_fallback(self):
        """--label mode without annotations falls back to pBoundaryRule >= 0.65 with labelType='weak'."""
        out_path = os.path.join(self.tmp_dir, "weak_labeled_out.jsonl")
        res = subprocess.run([
            sys.executable, T7_PROSODY_PY,
            self.wav_path, "0", "2500", self.words_path,
            "--label", "--out", out_path
        ], capture_output=True, text=True, timeout=30)

        self.assertEqual(res.returncode, 0, f"Weak labeling CLI failed: {res.stderr}")
        with open(out_path, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]

        self.assertEqual(len(lines), 2)
        # Gap 0 follows "hello." so pBoundaryRule is high -> isBoundary=1
        self.assertEqual(lines[0]["isBoundary"], 1)
        self.assertEqual(lines[0]["labelType"], "weak")
        self.assertAlmostEqual(lines[0]["confidence"], lines[0]["pBoundaryRule"], places=2)


class TestT7ProsodyTraining(unittest.TestCase):
    """Tests t7_train.py: safety refusal on < 200 samples, per-video split, artifact generation."""

    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.mkdtemp(prefix="test_t7_train_")
        cls.models_dir = os.path.join(cls.tmp_dir, "models")
        os.makedirs(cls.models_dir, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    def test_09_safety_guard_refuses_when_under_200_samples(self):
        """Refuses to train when labeled gaps < 200: outputs exact message, exits 0, creates no model."""
        under_200_path = os.path.join(self.tmp_dir, "under_200.jsonl")
        generate_synthetic_labels_jsonl(under_200_path, sample_count=50, num_videos=2)

        res = subprocess.run([
            sys.executable, T7_TRAIN_PY,
            "--data", under_200_path,
            "--models_dir", self.models_dir
        ], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)

        self.assertEqual(res.returncode, 0, "Safety guard refusal must exit code 0")
        combined_output = res.stdout + res.stderr
        self.assertTrue("NO LABELS — tagger not trained" in combined_output or "NO LABELS \u2014 tagger not trained" in combined_output,
                        f"Must print exact verbatim refusal message, got: {combined_output}")

        # Must not fabricate fake models
        model_file = os.path.join(self.models_dir, "t7_prosody_v1.txt")
        meta_file = os.path.join(self.models_dir, "t7_prosody_v1.meta.json")
        self.assertFalse(os.path.exists(model_file), "Model file must NOT be created under 200 samples")
        self.assertFalse(os.path.exists(meta_file), "Meta file must NOT be created under 200 samples")

    def test_10_per_video_split_prevents_speaker_leakage(self):
        """Verifies dataset partitioning groups exclusively by video ID with disjoint train/val video sets."""
        import t7_train
        labels_path = os.path.join(self.tmp_dir, "split_test.jsonl")
        generate_synthetic_labels_jsonl(labels_path, sample_count=240, num_videos=4)

        with open(labels_path, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]

        train_recs, val_recs = t7_train.partition_by_video(records, train_ratio=0.75)
        train_videos = {r["video"] for r in train_recs}
        val_videos = {r["video"] for r in val_recs}

        self.assertTrue(len(train_videos) > 0, "Train video set must not be empty")
        self.assertTrue(len(val_videos) > 0, "Validation video set must not be empty")
        self.assertTrue(train_videos.isdisjoint(val_videos),
                        "Train and validation videos must be completely disjoint (zero speaker leakage)")

    def test_11_training_pipeline_produces_valid_model_artifacts(self):
        """When labeled gaps >= 200, produces valid LightGBM model and metadata with AUC/PR-AUC."""
        import lightgbm as lgb
        valid_data_path = os.path.join(self.tmp_dir, "valid_240.jsonl")
        generate_synthetic_labels_jsonl(valid_data_path, sample_count=240, num_videos=4)

        res = subprocess.run([
            sys.executable, T7_TRAIN_PY,
            "--data", valid_data_path,
            "--models_dir", self.models_dir
        ], capture_output=True, text=True, timeout=60)

        self.assertEqual(res.returncode, 0, f"Training pipeline failed: {res.stderr}")

        model_file = os.path.join(self.models_dir, "t7_prosody_v1.txt")
        meta_file = os.path.join(self.models_dir, "t7_prosody_v1.meta.json")

        self.assertTrue(os.path.exists(model_file), "Trained model file must exist")
        self.assertTrue(os.path.exists(meta_file), "Metadata file must exist")

        # Load LightGBM booster
        booster = lgb.Booster(model_file=model_file)
        self.assertIsNotNone(booster)

        # Inspect metadata contract
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.assertEqual(meta.get("featureVersion"), "t7_prosody_v1")
        self.assertEqual(meta.get("threshold"), 0.65)
        self.assertEqual(meta.get("features"), EXPECTED_FEATURE_KEYS)

        train_stats = meta.get("trainStats", {})
        self.assertIn("auc", train_stats)
        self.assertIn("prAuc", train_stats)
        self.assertIn("trainSize", train_stats)
        self.assertIn("testSize", train_stats)
        self.assertGreater(train_stats["trainSize"], 0)
        self.assertGreater(train_stats["testSize"], 0)


class TestT7ProsodyInference(unittest.TestCase):
    """Tests CLI --predict sidecar mode: inference output schema and graceful fallbacks."""

    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.mkdtemp(prefix="test_t7_predict_")
        cls.wav_path = os.path.join(cls.tmp_dir, "test.wav")
        make_synth_wav(cls.wav_path, 2.5, ((0.1, 0.6, 200.0, 0.5), (1.4, 2.0, 200.0, 0.5)))

        cls.words_path = os.path.join(cls.tmp_dir, "words.json")
        words_data = {
            "words": [
                {"text": "hello.", "start": 0.1, "end": 0.6, "speaker": "S1"},
                {"text": "world", "start": 1.4, "end": 1.9, "speaker": "S1"},
            ]
        }
        with open(cls.words_path, "w", encoding="utf-8") as f:
            json.dump(words_data, f)

        # Train a fast synthetic model for testing predict mode
        cls.models_dir = os.path.join(cls.tmp_dir, "models")
        os.makedirs(cls.models_dir, exist_ok=True)
        synth_data = os.path.join(cls.tmp_dir, "data.jsonl")
        generate_synthetic_labels_jsonl(synth_data, sample_count=220, num_videos=4)
        subprocess.run([
            sys.executable, T7_TRAIN_PY,
            "--data", synth_data,
            "--models_dir", cls.models_dir
        ], capture_output=True, text=True, check=True)

        cls.model_path = os.path.join(cls.models_dir, "t7_prosody_v1.txt")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    def test_12_predict_mode_with_trained_model(self):
        """--predict mode returns modelScored=true and typed boundaries with pBoundary and features."""
        res = subprocess.run([
            sys.executable, T7_PROSODY_PY,
            self.wav_path, "0", "2500", self.words_path,
            "--predict", "--model", self.model_path
        ], capture_output=True, text=True, timeout=30)

        self.assertEqual(res.returncode, 0, f"Predict mode failed: {res.stderr}")
        data = json.loads(res.stdout)

        self.assertTrue(data.get("modelScored"))
        self.assertEqual(data.get("featureVersion"), "t7_prosody_v1")
        boundaries = data.get("boundaries", [])
        self.assertEqual(len(boundaries), 1)

        b = boundaries[0]
        self.assertEqual(b.get("gapIdx"), 0)
        self.assertIn("pBoundary", b)
        self.assertTrue(0.0 <= b["pBoundary"] <= 1.0)
        self.assertIn("isBoundary", b)
        self.assertIn("features", b)
        for feat in EXPECTED_FEATURE_KEYS:
            self.assertIn(feat, b["features"])

    def test_13_predict_mode_missing_model_fallback(self):
        """--predict mode with missing model exits 0 and returns modelScored=false, boundaries=[]."""
        missing_model = os.path.join(self.tmp_dir, "nonexistent_model.txt")
        res = subprocess.run([
            sys.executable, T7_PROSODY_PY,
            self.wav_path, "0", "2500", self.words_path,
            "--predict", "--model", missing_model
        ], capture_output=True, text=True, timeout=30)

        self.assertEqual(res.returncode, 0, "Missing model must degrade gracefully with exit 0")
        data = json.loads(res.stdout)
        self.assertFalse(data.get("modelScored"))
        self.assertIsNone(data.get("model"))
        self.assertEqual(data.get("boundaries"), [])

    def test_14_predict_mode_corrupt_or_mismatched_model_fallback(self):
        """--predict mode with corrupt model or version mismatch returns modelScored=false, boundaries=[]."""
        corrupt_model = os.path.join(self.tmp_dir, "corrupt_model.txt")
        with open(corrupt_model, "w", encoding="utf-8") as f:
            f.write("GARBAGE DATA - NOT LIGHTGBM")

        res = subprocess.run([
            sys.executable, T7_PROSODY_PY,
            self.wav_path, "0", "2500", self.words_path,
            "--predict", "--model", corrupt_model
        ], capture_output=True, text=True, timeout=30)

        self.assertEqual(res.returncode, 0, "Corrupt model must degrade gracefully with exit 0")
        data = json.loads(res.stdout)
        self.assertFalse(data.get("modelScored"))
        self.assertEqual(data.get("boundaries"), [])

    def test_15_predict_mode_missing_words_json_fallback(self):
        """--predict mode with missing words_json exits 0 and returns modelScored=false, boundaries=[]."""
        missing_words = os.path.join(self.tmp_dir, "nonexistent_words.json")
        res = subprocess.run([
            sys.executable, T7_PROSODY_PY,
            self.wav_path, "0", "2500", missing_words,
            "--predict", "--model", self.model_path
        ], capture_output=True, text=True, timeout=30)

        self.assertEqual(res.returncode, 0, f"Missing words_json must exit 0: {res.stderr}")
        data = json.loads(res.stdout)
        self.assertFalse(data.get("modelScored"))
        self.assertIsNone(data.get("model"))
        self.assertEqual(data.get("boundaries"), [])

    def test_16_predict_mode_corrupt_words_json_fallback(self):
        """--predict mode with corrupt words_json syntax exits 0 and returns modelScored=false, boundaries=[]."""
        corrupt_words = os.path.join(self.tmp_dir, "corrupt_words.json")
        with open(corrupt_words, "w", encoding="utf-8") as f:
            f.write("{broken_json: unclosed")

        res = subprocess.run([
            sys.executable, T7_PROSODY_PY,
            self.wav_path, "0", "2500", corrupt_words,
            "--predict", "--model", self.model_path
        ], capture_output=True, text=True, timeout=30)

        self.assertEqual(res.returncode, 0, f"Corrupt words_json must exit 0: {res.stderr}")
        data = json.loads(res.stdout)
        self.assertFalse(data.get("modelScored"))
        self.assertIsNone(data.get("model"))
        self.assertEqual(data.get("boundaries"), [])

    def test_17_predict_mode_null_words_json_fallback(self):
        """--predict mode with null or non-list words_json exits 0 and returns modelScored=false, boundaries=[]."""
        null_words = os.path.join(self.tmp_dir, "null_words.json")
        with open(null_words, "w", encoding="utf-8") as f:
            f.write("null")

        res = subprocess.run([
            sys.executable, T7_PROSODY_PY,
            self.wav_path, "0", "2500", null_words,
            "--predict", "--model", self.model_path
        ], capture_output=True, text=True, timeout=30)

        self.assertEqual(res.returncode, 0, f"Null words_json must exit 0: {res.stderr}")
        data = json.loads(res.stdout)
        self.assertFalse(data.get("modelScored"))
        self.assertIsNone(data.get("model"))
        self.assertEqual(data.get("boundaries"), [])

    def test_18_predict_mode_dict_null_words_fallback(self):
        """--predict mode with {'words': null} exits 0 and returns modelScored=false, boundaries=[]."""
        dict_null_words = os.path.join(self.tmp_dir, "dict_null_words.json")
        with open(dict_null_words, "w", encoding="utf-8") as f:
            f.write(json.dumps({"words": None}))

        res = subprocess.run([
            sys.executable, T7_PROSODY_PY,
            self.wav_path, "0", "2500", dict_null_words,
            "--predict", "--model", self.model_path
        ], capture_output=True, text=True, timeout=30)

        self.assertEqual(res.returncode, 0, f"Dict with null words must exit 0: {res.stderr}")
        data = json.loads(res.stdout)
        self.assertFalse(data.get("modelScored"))
        self.assertIsNone(data.get("model"))
        self.assertEqual(data.get("boundaries"), [])

    def test_19_predict_mode_fallback_with_out_file(self):
        """--predict mode with corrupt words_json and --out writes fallback JSON to file and exits 0."""
        corrupt_words = os.path.join(self.tmp_dir, "corrupt_for_out.json")
        with open(corrupt_words, "w", encoding="utf-8") as f:
            f.write("invalid json content")
        out_target = os.path.join(self.tmp_dir, "sub_out", "fallback_result.json")

        res = subprocess.run([
            sys.executable, T7_PROSODY_PY,
            self.wav_path, "0", "2500", corrupt_words,
            "--predict", "--model", self.model_path,
            "--out", out_target
        ], capture_output=True, text=True, timeout=30)

        self.assertEqual(res.returncode, 0, f"--out fallback must exit 0: {res.stderr}")
        self.assertTrue(os.path.isfile(out_target))
        with open(out_target, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertFalse(data.get("modelScored"))
        self.assertIsNone(data.get("model"))
        self.assertEqual(data.get("boundaries"), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
