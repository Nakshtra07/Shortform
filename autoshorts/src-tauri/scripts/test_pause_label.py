#!/usr/bin/env python3
"""
Unit and integration test suite for pause_label.py (Task 1).
"""

import json
import os
import shutil
import struct
import sys
import tempfile
import unittest
import wave

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import pause_label as pl
import pause_intelligence as pi


def create_test_wav(path: str, duration: float = 3.0, tone_spans: tuple = ()) -> str:
    """Create a 16kHz mono WAV file with specified tone spans."""
    sr = 16000
    n = int(duration * sr)
    x = np.zeros(n, dtype=np.float32)
    for t0, t1, freq, amp in tone_spans:
        i0 = int(t0 * sr)
        i1 = min(n, int(t1 * sr))
        if i1 > i0:
            t = np.arange(i1 - i0, dtype=np.float32) / sr
            x[i0:i1] = amp * np.sin(2 * np.pi * freq * t)

    # Convert to 16-bit PCM
    x_int16 = np.clip(x * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(x_int16.tobytes())
    return path


class TestPauseLabel(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="pause_label_test_")
        self.wav_path = os.path.join(self.temp_dir, "test_audio.wav")
        # 3 seconds total:
        # Word 1: 0.1 to 0.4 (tone 400Hz, amp 0.8)
        # Gap 1: 0.4 to 0.6 (0.2s duration -> kept)
        # Word 2: 0.6 to 0.9 (tone 400Hz, amp 0.8)
        # Gap 2: 0.9 to 0.93 (0.03s duration -> discarded <0.05s)
        # Word 3: 0.93 to 1.2 (tone 400Hz, amp 0.8)
        # Gap 3: 1.2 to 1.6 (0.4s duration -> kept)
        # Word 4: 1.6 to 1.9 (tone 400Hz, amp 0.8)
        tone_spans = (
            (0.1, 0.4, 400.0, 0.8),
            (0.6, 0.9, 400.0, 0.8),
            (0.93, 1.2, 400.0, 0.8),
            (1.6, 1.9, 400.0, 0.8),
        )
        create_test_wav(self.wav_path, duration=3.0, tone_spans=tone_spans)

        self.words = [
            {"word": "Hello.", "start": 0.1, "end": 0.4, "speaker": "SPEAKER_0"},
            {"word": "world,", "start": 0.6, "end": 0.9, "speaker": "SPEAKER_0"},
            {"word": "quick", "start": 0.93, "end": 1.2, "speaker": "SPEAKER_0"},
            {"word": "test!", "start": 1.6, "end": 1.9, "speaker": "SPEAKER_1"},
        ]
        self.words_path = os.path.join(self.temp_dir, "test_audio.words.json")
        with open(self.words_path, "w", encoding="utf-8") as f:
            json.dump(self.words, f)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_words(self):
        # List of words format
        w1 = pl.load_words(self.words_path)
        self.assertIsNotNone(w1)
        self.assertEqual(len(w1), 4)
        self.assertEqual(w1[0]["text"], "Hello.")

        # Dict with 'words' format
        dict_words_path = os.path.join(self.temp_dir, "dict_words.json")
        with open(dict_words_path, "w", encoding="utf-8") as f:
            json.dump({"words": self.words}, f)
        w2 = pl.load_words(dict_words_path)
        self.assertIsNotNone(w2)
        self.assertEqual(len(w2), 4)

    def test_degenerate_gap_filtering(self):
        # Gap between word 2 and word 3 is 0.9 to 0.93 (duration 0.03s < 0.05s).
        # It must be discarded.
        recs = pl.process_media_transcript(
            self.wav_path,
            self.words_path,
            clip_start=0.0,
            clip_end=3.0,
        )
        self.assertGreater(len(recs), 0)
        for r in recs:
            dur = r["srcEndSec"] - r["srcStartSec"]
            self.assertGreaterEqual(dur, 0.05)
            self.assertLessEqual(dur, 5.0)
            self.assertEqual(len(r["features"]), 23)
            self.assertIn(r["label"], pl.VALID_CLASSES)

    def test_speaker_transition_label(self):
        # Word 3 (speaker 0) -> Word 4 (speaker 1): gap from 1.2 to 1.6
        recs = pl.process_media_transcript(
            self.wav_path,
            self.words_path,
            clip_start=0.0,
            clip_end=3.0,
        )
        speaker_trans_recs = [r for r in recs if r["srcStartSec"] == 1.2]
        if speaker_trans_recs:
            self.assertEqual(speaker_trans_recs[0]["label"], "speaker_transition")

    def test_sentence_pause_label(self):
        # Word 1 ends with '.' ("Hello.") -> Gap 1: 0.4 to 0.6
        recs = pl.process_media_transcript(
            self.wav_path,
            self.words_path,
            clip_start=0.0,
            clip_end=3.0,
        )
        gap1_recs = [r for r in recs if r["srcStartSec"] == 0.4]
        if gap1_recs:
            # Word 1 ends sentence
            self.assertIn(gap1_recs[0]["label"], ("sentence_pause", "waiting_pause", "breath_pause"))

    def test_human_annotation_override(self):
        labels_path = os.path.join(self.temp_dir, "human_labels.jsonl")
        sh = pi.source_hash(self.wav_path)
        # Override gap at 0.4 with 'breath_pause'
        with open(labels_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "sourceHash": sh,
                "srcStartSec": 0.4,
                "srcEndSec": 0.6,
                "class": "breath_pause"
            }) + "\n")

        labels_data = pl.load_human_labels(labels_path)
        recs = pl.process_media_transcript(
            self.wav_path,
            self.words_path,
            labels_data=labels_data,
            clip_start=0.0,
            clip_end=3.0,
        )
        gap1_recs = [r for r in recs if r["srcStartSec"] == 0.4]
        self.assertEqual(len(gap1_recs), 1)
        self.assertEqual(gap1_recs[0]["label"], "breath_pause")
        self.assertEqual(gap1_recs[0]["labelSource"], "human_annotation")
        self.assertFalse(gap1_recs[0]["weakSupervision"])

    def test_jsonl_output_format(self):
        out_jsonl = os.path.join(self.temp_dir, "out.jsonl")
        recs = pl.process_media_transcript(self.wav_path, self.words_path)
        count = pl.write_records(recs, out_jsonl, overwrite=True)
        self.assertGreater(count, 0)
        self.assertTrue(os.path.isfile(out_jsonl))

        with open(out_jsonl, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        self.assertEqual(len(lines), count)
        for line in lines:
            self.assertIn("sourceHash", line)
            self.assertIn("gapIdx", line)
            self.assertIn("srcStartSec", line)
            self.assertIn("srcEndSec", line)
            self.assertIn("features", line)
            self.assertEqual(len(line["features"]), 23)
            self.assertIn("label", line)
            self.assertIn(line["label"], pl.VALID_CLASSES)
            self.assertIn("labelSource", line)
            self.assertIn("weakSupervision", line)
            self.assertIn("timestamp", line)

    def test_corpus_crawl(self):
        # Create a corpus subdir with pair
        c_dir = os.path.join(self.temp_dir, "subcorpus")
        os.makedirs(c_dir, exist_ok=True)
        c_wav = os.path.join(c_dir, "sample.wav")
        shutil.copy(self.wav_path, c_wav)
        c_words = os.path.join(c_dir, "sample.words.json")
        shutil.copy(self.words_path, c_words)

        pairs = pl.crawl_corpus(self.temp_dir)
        self.assertGreaterEqual(len(pairs), 2)  # main wav + subcorpus wav

    def test_cli_execution(self):
        out_jsonl = os.path.join(self.temp_dir, "cli_out.jsonl")
        ret = pl.main([
            self.wav_path,
            self.words_path,
            "--out", out_jsonl,
            "--overwrite"
        ])
        self.assertEqual(ret, 0)
        self.assertTrue(os.path.isfile(out_jsonl))
        with open(out_jsonl, "r", encoding="utf-8") as f:
            lines = [json.loads(l) for l in f if l.strip()]
        self.assertGreater(len(lines), 0)


if __name__ == "__main__":
    unittest.main()
