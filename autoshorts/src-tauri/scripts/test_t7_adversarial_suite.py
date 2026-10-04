#!/usr/bin/env python3
"""
Adversarial Stress Test Suite for T7 Prosody Subsystem (scripts/t7_prosody.py & t7_train.py)

Empirically challenges:
1. Audio edge cases: 0.0s duration, corrupt/unreadable audio, missing audio, pure silence vs loud noise, inverted bounds.
2. Punctuation edge cases: trailing spaces/newlines, Hindi danda/double danda, ellipses, multiple marks, quotes/brackets.
3. Speaker change edge cases: integer speaker IDs (0 vs 1), string IDs, empty strings, None, missing keys.
4. Predict mode edge cases: missing model, corrupt model, version mismatch in meta, corrupt meta, empty/single words, missing/corrupt audio during predict.
5. Predict mode robustness: verify that --predict NEVER exits non-zero or produces invalid JSON on any malformed/corrupt input.
6. Train safety & corruption: verify <200 refusal and resilient JSONL parsing.
"""

import os
import sys
import json
import wave
import shutil
import tempfile
import unittest
import subprocess
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import t7_prosody as tp
import t7_train as tt

T7_PROSODY_PY = os.path.join(SCRIPT_DIR, "t7_prosody.py")
T7_TRAIN_PY = os.path.join(SCRIPT_DIR, "t7_train.py")


def make_test_wav(path: str, duration: float, freq: float = 440.0, amp: float = 0.5) -> str:
    """Create a standard PCM WAV file."""
    sr = 16000
    n = int(duration * sr)
    if amp == 0.0 or duration == 0.0:
        samples = np.zeros(n, dtype=np.int16)
    else:
        t = np.arange(n, dtype=np.float32) / sr
        x = amp * np.sin(2.0 * np.pi * freq * t)
        samples = (np.clip(x, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(samples.tobytes())
    return path


def make_corrupt_file(path: str, corrupt_type: str = "random") -> str:
    """Create a corrupted or non-audio file."""
    if corrupt_type == "empty":
        with open(path, "wb") as f:
            pass
    elif corrupt_type == "text":
        with open(path, "w", encoding="utf-8") as f:
            f.write("This is a plain text file pretending to be audio.\n")
    elif corrupt_type == "truncated_header":
        # 12 bytes of RIFF header without chunks
        with open(path, "wb") as f:
            f.write(b"RIFF\x24\x00\x00\x00WAVE")
    else:  # random bytes
        with open(path, "wb") as f:
            f.write(os.urandom(2048))
    return path


class TestAdversarialAudio(unittest.TestCase):
    """Stress tests on audio decoder and acoustic feature extraction."""

    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.mkdtemp(prefix="adv_audio_")
        cls.valid_wav = make_test_wav(os.path.join(cls.tmp_dir, "valid.wav"), 2.0, freq=300.0)
        cls.silence_wav = make_test_wav(os.path.join(cls.tmp_dir, "silence.wav"), 2.0, amp=0.0)
        cls.zero_wav = make_test_wav(os.path.join(cls.tmp_dir, "zero.wav"), 0.0)

        cls.corrupt_random = make_corrupt_file(os.path.join(cls.tmp_dir, "random.wav"), "random")
        cls.corrupt_empty = make_corrupt_file(os.path.join(cls.tmp_dir, "empty.wav"), "empty")
        cls.corrupt_text = make_corrupt_file(os.path.join(cls.tmp_dir, "text.wav"), "text")
        cls.corrupt_trunc = make_corrupt_file(os.path.join(cls.tmp_dir, "trunc.wav"), "truncated_header")

        cls.sample_words = [
            {"text": "Alpha", "start": 0.1, "end": 0.5, "speaker": "S1"},
            {"text": "Beta", "start": 1.2, "end": 1.8, "speaker": "S1"},
        ]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    def test_audio_01_zero_duration_audio(self):
        """0.0s audio duration or start_sec == end_sec: test behavior and reveal ffmpeg duration bug."""
        # 0s audio file
        dec = tp.decode_audio(self.zero_wav, 0.0, 0.0)
        self.assertIsNone(dec, "0s audio file should decode to None")

        # start_sec == end_sec on valid audio:
        dec2 = tp.decode_audio(self.valid_wav, 1.0, 1.0)
        self.assertIsNone(dec2, "decode_audio must return None when start_sec == end_sec")

        # build_gap_records with 0s duration
        recs = tp.build_gap_records(self.zero_wav, 0.0, 0.0, self.sample_words)
        self.assertEqual(len(recs), 1)
        self.assertIsNone(recs[0]["f0SlopePre"])
        self.assertIsNone(recs[0]["energyDropDb"])
        self.assertIsNone(recs[0]["vadSilenceFrac"])
        self.assertGreaterEqual(recs[0]["pBoundaryRule"], 0.0)

    def test_audio_02_corrupt_audio_files(self):
        """Corrupt audio files (random bytes, empty, text, truncated header) decode to None without crashing."""
        for name, path in [
            ("random", self.corrupt_random),
            ("empty", self.corrupt_empty),
            ("text", self.corrupt_text),
            ("truncated", self.corrupt_trunc),
        ]:
            dec = tp.decode_audio(path, 0.0, 2.0)
            self.assertIsNone(dec, f"Corrupt file ({name}) must decode to None")

            recs = tp.build_gap_records(path, 0.0, 2.0, self.sample_words)
            self.assertEqual(len(recs), 1, f"build_gap_records must succeed for corrupt {name}")
            self.assertIsNone(recs[0]["f0SlopePre"])
            self.assertIsNone(recs[0]["energyDropDb"])
            self.assertIsNone(recs[0]["vadSilenceFrac"])

    def test_audio_03_missing_audio_file(self):
        """Non-existent audio path or None does not crash."""
        dec = tp.decode_audio("nonexistent_path_98765.wav", 0.0, 2.0)
        self.assertIsNone(dec)

        recs = tp.build_gap_records("nonexistent_path_98765.wav", 0.0, 2.0, self.sample_words)
        self.assertEqual(len(recs), 1)
        self.assertIsNone(recs[0]["f0SlopePre"])

        recs_none = tp.build_gap_records(None, 0.0, 2.0, self.sample_words)
        self.assertEqual(len(recs_none), 1)
        self.assertIsNone(recs_none[0]["f0SlopePre"])

    def test_audio_04_pure_silence_vs_extreme_loud_noise(self):
        """Pure silence has high vadSilenceFrac; full-scale noise has low vadSilenceFrac."""
        words = [
            {"text": "A", "start": 0.2, "end": 0.5},
            {"text": "B", "start": 1.2, "end": 1.8},
        ]
        recs_sil = tp.build_gap_records(self.silence_wav, 0.0, 2.0, words)
        self.assertGreaterEqual(recs_sil[0]["vadSilenceFrac"], 0.90)

        # Extreme full-scale square wave
        square_wav = os.path.join(self.tmp_dir, "loud_square.wav")
        sr = 16000
        t = np.arange(2.0 * sr) / sr
        sq = np.sign(np.sin(2 * np.pi * 500 * t)).astype(np.float32)
        int_samples = (sq * 32767).astype(np.int16)
        with wave.open(square_wav, "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(sr)
            f.writeframes(int_samples.tobytes())

        recs_sq = tp.build_gap_records(square_wav, 0.0, 2.0, words)
        self.assertLessEqual(recs_sq[0]["vadSilenceFrac"], 0.10)

    def test_audio_05_inverted_or_out_of_bounds_timestamps(self):
        """Inverted start/end or words beyond clip range does not crash."""
        # start_sec > end_sec
        recs_inv = tp.build_gap_records(self.valid_wav, 2.0, 0.5, self.sample_words)
        self.assertEqual(len(recs_inv), 1)

        # Words out of range
        words_far = [
            {"text": "Far1", "start": 100.0, "end": 100.5},
            {"text": "Far2", "start": 101.0, "end": 101.5},
        ]
        recs_far = tp.build_gap_records(self.valid_wav, 0.0, 2.0, words_far)
        self.assertEqual(len(recs_far), 1)


class TestAdversarialPunctuation(unittest.TestCase):
    """Stress tests on sentence and clause ending detection."""

    def test_punct_01_trailing_whitespace_variations(self):
        """Sentence terminators with trailing spaces, tabs, and newlines must set endsSentence=1."""
        test_strings = [
            "hello. ",
            "hello.\t",
            "hello.\n",
            "hello.  \r\n",
            "world!   ",
            "really?  \t",
            "clause;   ",
            "colon: ",
            "hindi।   ",
            "double॥   ",
        ]
        for s in test_strings:
            words = [{"text": s, "start": 0.0, "end": 0.5}, {"text": "next", "start": 1.0, "end": 1.5}]
            recs = tp.build_gap_records(None, 0.0, 2.0, words)
            self.assertEqual(recs[0]["endsSentence"], 1, f"Failed for trailing whitespace on '{s}'")

    def test_punct_02_hindi_danda_variations(self):
        """Single and double Hindi danda with and without spaces set endsSentence=1."""
        dandas = ["वाक्य।", "वाक्य। ", "श्लोक॥", "श्लोक॥  ", "।", "॥"]
        for d in dandas:
            words = [{"text": d, "start": 0.0, "end": 0.5}, {"text": "दूसरा", "start": 1.0, "end": 1.5}]
            recs = tp.build_gap_records(None, 0.0, 2.0, words)
            self.assertEqual(recs[0]["endsSentence"], 1, f"Hindi danda '{d}' must set endsSentence=1")

    def test_punct_03_multiple_punctuation_marks(self):
        """Multiple punctuation marks like '!!!', '?!?', '...' must set endsSentence=1."""
        marks = ["wow!!!", "what!?!", "really???", "wait....", "done:;"]
        for m in marks:
            words = [{"text": m, "start": 0.0, "end": 0.5}, {"text": "next", "start": 1.0, "end": 1.5}]
            recs = tp.build_gap_records(None, 0.0, 2.0, words)
            self.assertEqual(recs[0]["endsSentence"], 1, f"Multiple marks '{m}' must set endsSentence=1")

    def test_punct_04_comma_clause_variations(self):
        """Commas with trailing spaces set endsClause=1, endsSentence=0."""
        commas = ["however,", "however, ", "wait,  \t", ","]
        for c in commas:
            words = [{"text": c, "start": 0.0, "end": 0.5}, {"text": "next", "start": 1.0, "end": 1.5}]
            recs = tp.build_gap_records(None, 0.0, 2.0, words)
            self.assertEqual(recs[0]["endsClause"], 1, f"Comma '{c}' must set endsClause=1")
            self.assertEqual(recs[0]["endsSentence"], 0, f"Comma '{c}' must not set endsSentence=1")

    def test_punct_05_ellipses_and_unicode(self):
        """Examine 3 dots '...' vs unicode ellipsis '…'."""
        # Standard ASCII three dots
        words_dots = [{"text": "waiting...", "start": 0.0, "end": 0.5}, {"text": "done", "start": 1.0, "end": 1.5}]
        recs_dots = tp.build_gap_records(None, 0.0, 2.0, words_dots)
        self.assertEqual(recs_dots[0]["endsSentence"], 1, "ASCII '...' must set endsSentence=1")

        # Unicode ellipsis
        words_uni = [{"text": "waiting…", "start": 0.0, "end": 0.5}, {"text": "done", "start": 1.0, "end": 1.5}]
        recs_uni = tp.build_gap_records(None, 0.0, 2.0, words_uni)
        # We record whether unicode ellipsis is currently recognized
        has_uni = (recs_uni[0]["endsSentence"] == 1)
        sys.stderr.write(f"\n[Adversarial Note] Unicode ellipsis recognized: {has_uni}\n")


class TestAdversarialSpeakerChange(unittest.TestCase):
    """Stress tests on speaker change detection."""

    def test_speaker_01_integer_speaker_ids(self):
        """CRITICAL: Integer speaker IDs (0 vs 1). Deepgram/Whisper often use int speaker indices."""
        words_int = [
            {"text": "Speaker Zero", "start": 0.0, "end": 0.5, "speaker": 0},
            {"text": "Speaker One", "start": 1.0, "end": 1.5, "speaker": 1},
        ]
        recs = tp.build_gap_records(None, 0.0, 2.0, words_int)
        sc = recs[0]["speakerChange"]
        sys.stderr.write(f"\n[Adversarial Note] Integer speaker change (0 -> 1) detected as: {sc}\n")
        # In python: 0 and 1 evaluates to 0 if not explicitly checking is not None!

    def test_speaker_02_none_and_empty_strings(self):
        """None or empty string speakers must not trigger false speakerChange."""
        pairs = [
            (None, None),
            (None, "spk1"),
            ("spk1", None),
            ("", ""),
            ("", "spk1"),
            ("spk1", ""),
            ("spk1", "spk1"),
        ]
        for spk1, spk2 in pairs:
            w = [
                {"text": "A", "start": 0.0, "end": 0.5, "speaker": spk1},
                {"text": "B", "start": 1.0, "end": 1.5, "speaker": spk2},
            ]
            recs = tp.build_gap_records(None, 0.0, 2.0, w)
            self.assertEqual(recs[0]["speakerChange"], 0, f"False speakerChange on ({spk1}, {spk2})")

    def test_speaker_03_distinct_string_ids(self):
        """Distinct non-empty string IDs trigger speakerChange=1."""
        w = [
            {"text": "A", "start": 0.0, "end": 0.5, "speaker": "spk_A"},
            {"text": "B", "start": 1.0, "end": 1.5, "speaker": "spk_B"},
        ]
        recs = tp.build_gap_records(None, 0.0, 2.0, w)
        self.assertEqual(recs[0]["speakerChange"], 1)


class TestAdversarialPredictFallback(unittest.TestCase):
    """Empirically verifies that --predict NEVER crashes on corrupt/missing inputs and outputs valid JSON."""

    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.mkdtemp(prefix="adv_predict_")
        cls.wav_path = make_test_wav(os.path.join(cls.tmp_dir, "test.wav"), 2.0)
        cls.corrupt_wav = make_corrupt_file(os.path.join(cls.tmp_dir, "corrupt.wav"), "random")

        cls.valid_words_path = os.path.join(cls.tmp_dir, "words_valid.json")
        with open(cls.valid_words_path, "w", encoding="utf-8") as f:
            json.dump([
                {"text": "hello.", "start": 0.1, "end": 0.6, "speaker": "S1"},
                {"text": "world", "start": 1.0, "end": 1.5, "speaker": "S1"},
            ], f)

        # Train a minimal valid model for testing
        cls.models_dir = os.path.join(cls.tmp_dir, "models")
        os.makedirs(cls.models_dir, exist_ok=True)
        data_path = os.path.join(cls.tmp_dir, "synth_data.jsonl")
        from test_t7_prosody_suite import generate_synthetic_labels_jsonl
        generate_synthetic_labels_jsonl(data_path, sample_count=220, num_videos=4)
        subprocess.run([
            sys.executable, T7_TRAIN_PY,
            "--data", data_path,
            "--models_dir", cls.models_dir
        ], check=True, capture_output=True)

        cls.valid_model = os.path.join(cls.models_dir, "t7_prosody_v1.txt")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    def _run_predict(self, audio_path, start_ms, end_ms, words_path, model_path=None):
        cmd = [
            sys.executable, T7_PROSODY_PY,
            audio_path, str(start_ms), str(end_ms), words_path,
            "--predict"
        ]
        if model_path:
            cmd.extend(["--model", model_path])
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60)

    def test_predict_01_missing_model_file(self):
        """Missing model file returns exit 0 and valid fallback JSON."""
        res = self._run_predict(self.wav_path, 0, 2000, self.valid_words_path,
                                model_path=os.path.join(self.tmp_dir, "no_model.txt"))
        self.assertEqual(res.returncode, 0)
        data = json.loads(res.stdout)
        self.assertFalse(data["modelScored"])
        self.assertIsNone(data["model"])
        self.assertEqual(data["boundaries"], [])

    def test_predict_02_corrupt_model_file(self):
        """Corrupt model file returns exit 0 and valid fallback JSON."""
        corrupt_model = os.path.join(self.tmp_dir, "bad_model.txt")
        with open(corrupt_model, "wb") as f:
            f.write(os.urandom(512))

        res = self._run_predict(self.wav_path, 0, 2000, self.valid_words_path,
                                model_path=corrupt_model)
        self.assertEqual(res.returncode, 0)
        data = json.loads(res.stdout)
        self.assertFalse(data["modelScored"])
        self.assertEqual(data["boundaries"], [])

    def test_predict_03_mismatched_feature_version_in_meta(self):
        """Metadata featureVersion mismatch triggers graceful fallback."""
        mismatch_model = os.path.join(self.tmp_dir, "mismatch_model.txt")
        mismatch_meta = os.path.join(self.tmp_dir, "mismatch_model.meta.json")
        shutil.copy(self.valid_model, mismatch_model)
        with open(mismatch_meta, "w", encoding="utf-8") as f:
            json.dump({"featureVersion": "t7_prosody_v999", "threshold": 0.65}, f)

        res = self._run_predict(self.wav_path, 0, 2000, self.valid_words_path,
                                model_path=mismatch_model)
        self.assertEqual(res.returncode, 0)
        data = json.loads(res.stdout)
        self.assertFalse(data["modelScored"])
        self.assertEqual(data["boundaries"], [])

    def test_predict_04_corrupt_meta_json(self):
        """Corrupted/invalid JSON in metadata file triggers graceful fallback."""
        bad_meta_model = os.path.join(self.tmp_dir, "bad_meta_model.txt")
        bad_meta = os.path.join(self.tmp_dir, "bad_meta_model.meta.json")
        shutil.copy(self.valid_model, bad_meta_model)
        with open(bad_meta, "w", encoding="utf-8") as f:
            f.write("{broken_json_not_valid:")

        res = self._run_predict(self.wav_path, 0, 2000, self.valid_words_path,
                                model_path=bad_meta_model)
        self.assertEqual(res.returncode, 0)
        data = json.loads(res.stdout)
        self.assertFalse(data["modelScored"])
        self.assertEqual(data["boundaries"], [])

    def test_predict_05_empty_words_list(self):
        """Empty words list returns exit 0, modelScored=true, boundaries=[]."""
        empty_words_path = os.path.join(self.tmp_dir, "words_empty.json")
        with open(empty_words_path, "w", encoding="utf-8") as f:
            json.dump([], f)

        res = self._run_predict(self.wav_path, 0, 2000, empty_words_path,
                                model_path=self.valid_model)
        self.assertEqual(res.returncode, 0)
        data = json.loads(res.stdout)
        self.assertTrue(data["modelScored"])
        self.assertEqual(data["boundaries"], [])

    def test_predict_06_single_word_list(self):
        """Single word list (0 gaps) returns exit 0, modelScored=true, boundaries=[]."""
        single_word_path = os.path.join(self.tmp_dir, "words_single.json")
        with open(single_word_path, "w", encoding="utf-8") as f:
            json.dump([{"text": "OnlyWord", "start": 0.0, "end": 0.5}], f)

        res = self._run_predict(self.wav_path, 0, 2000, single_word_path,
                                model_path=self.valid_model)
        self.assertEqual(res.returncode, 0)
        data = json.loads(res.stdout)
        self.assertTrue(data["modelScored"])
        self.assertEqual(data["boundaries"], [])

    def test_predict_07_corrupt_audio_during_predict(self):
        """Corrupt audio during predict mode runs successfully, falling back acoustic features."""
        res = self._run_predict(self.corrupt_wav, 0, 2000, self.valid_words_path,
                                model_path=self.valid_model)
        self.assertEqual(res.returncode, 0)
        data = json.loads(res.stdout)
        self.assertTrue(data["modelScored"])
        self.assertEqual(len(data["boundaries"]), 1)
        b0 = data["boundaries"][0]
        self.assertIn("pBoundary", b0)
        self.assertTrue(0.0 <= b0["pBoundary"] <= 1.0)
        # acoustic features should be None in record
        self.assertIsNone(b0["features"]["f0SlopePre"])

    def test_predict_08_missing_audio_during_predict(self):
        """Non-existent audio path during predict mode runs successfully."""
        res = self._run_predict("nonexistent_path_4321.wav", 0, 2000, self.valid_words_path,
                                model_path=self.valid_model)
        self.assertEqual(res.returncode, 0)
        data = json.loads(res.stdout)
        self.assertTrue(data["modelScored"])
        self.assertEqual(len(data["boundaries"]), 1)

    def test_predict_09_zero_duration_audio_during_predict(self):
        """0.0s audio duration (start_ms == end_ms) during predict mode runs successfully."""
        res = self._run_predict(self.wav_path, 1000, 1000, self.valid_words_path,
                                model_path=self.valid_model)
        self.assertEqual(res.returncode, 0)
        data = json.loads(res.stdout)
        self.assertTrue(data["modelScored"])
        self.assertEqual(len(data["boundaries"]), 1)


class TestAdversarialTrainSafety(unittest.TestCase):
    """Stress tests on trainer safety guards and corrupt JSONL tolerance."""

    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.mkdtemp(prefix="adv_train_")
        cls.models_dir = os.path.join(cls.tmp_dir, "models")
        os.makedirs(cls.models_dir, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    def test_train_01_empty_file_refusal(self):
        """Empty data file triggers exact refusal message and exits 0."""
        empty_data = os.path.join(self.tmp_dir, "empty.jsonl")
        with open(empty_data, "w", encoding="utf-8") as f:
            pass

        res = subprocess.run([
            sys.executable, T7_TRAIN_PY,
            "--data", empty_data,
            "--models_dir", self.models_dir
        ], capture_output=True, text=True, encoding="utf-8", errors="replace")

        self.assertEqual(res.returncode, 0)
        self.assertIn("NO LABELS", res.stdout)
        self.assertIn("tagger not trained", res.stdout)

    def test_train_02_corrupt_lines_in_jsonl(self):
        """JSONL with malformed lines, missing keys, and invalid labels parses resiliently."""
        corrupt_data = os.path.join(self.tmp_dir, "corrupt_data.jsonl")
        with open(corrupt_data, "w", encoding="utf-8") as f:
            f.write("\n")
            f.write("not json\n")
            f.write("{}\n")
            f.write('{"gapIdx": 1}\n')  # missing isBoundary
            f.write('{"gapIdx": 2, "isBoundary": "invalid_label"}\n')
            f.write('{"gapIdx": 3, "isBoundary": 1, "video": "v1", "features": {"pauseSec": 0.5}}\n')

        records = tt.load_labeled_records(corrupt_data)
        # Only gapIdx 3 is valid
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["gapIdx"], 3)
        self.assertEqual(records[0]["isBoundary"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
