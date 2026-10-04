#!/usr/bin/env python3
"""
Test Suite for AutoShorts Multimodal Acoustic Feature Analyzer (Fixing F1)
Verifies:
- Real DSP extraction (RMS envelope, pre-hook silence, YIN pitch contour, spectral burstiness)
- Soundfile loading & FFmpeg pipe audio loading
- Truthful semantics: burstiness is measured onset spectral flux; laughter == burstiness
- No fabricated emotional claims or fake punctuation heuristics in evidence strings
- Graceful handling of missing files, silence, unvoiced frames, and forced failures
"""

import os
import sys
import tempfile
import unittest
import numpy as np
import soundfile as sf

# Add scripts directory to sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import subprocess

from multimodal_hook_analyzer import (
    load_audio_segment,
    extract_audio_segment_dsp,
    analyze_acoustic_signals,
    process_multimodal_pipeline,
)


class TestMultimodalAcousticSuite(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sr = 16000
        # Build synthetic audio (10.0 seconds total):
        # 0.0 - 2.0s: silence
        # 2.0 - 4.5s: low-energy conversation body (200Hz sine @ 0.10 amplitude + minor harmonics)
        # 4.5 - 5.5s: 1.0s deliberate pre-hook silence (< -40dB)
        # 5.5 - 8.0s: high-energy hook delivery (variable pitch 250->450Hz @ 0.85 amplitude with onset bursts)
        # 8.0 - 10.0s: low-energy tail (200Hz sine @ 0.10 amplitude)
        total_dur = 10.0
        n_samples = int(cls.sr * total_dur)
        cls.synthetic_audio = np.zeros(n_samples, dtype=np.float32)

        # 2.0 - 4.5s body
        idx_b_start = int(2.0 * cls.sr)
        idx_b_end = int(4.5 * cls.sr)
        t_body = np.linspace(0, 2.5, idx_b_end - idx_b_start, endpoint=False, dtype=np.float32)
        cls.synthetic_audio[idx_b_start:idx_b_end] = 0.10 * np.sin(2 * np.pi * 200 * t_body)

        # 4.5 - 5.5s is silence (remains 0.0)

        # 5.5 - 8.0s hook with energetic pitch modulation and transient bursts
        idx_h_start = int(5.5 * cls.sr)
        idx_h_end = int(8.0 * cls.sr)
        t_hook = np.linspace(0, 2.5, idx_h_end - idx_h_start, endpoint=False, dtype=np.float32)
        # Chirp/pitch inflection from 250Hz to 450Hz
        freq_mod = 250.0 + 80.0 * np.sin(2 * np.pi * 1.5 * t_hook)
        phase = 2 * np.pi * np.cumsum(freq_mod) / cls.sr
        hook_sig = 0.85 * np.sin(phase)
        # Add high-frequency transient pulses to trigger onset spectral burstiness
        pulses = np.zeros_like(hook_sig)
        for pulse_sec in [0.2, 0.6, 1.1, 1.7]:
            p_idx = int(pulse_sec * cls.sr)
            if p_idx + 400 < len(pulses):
                pulses[p_idx:p_idx + 400] = 0.7 * np.sin(2 * np.pi * 2500 * np.linspace(0, 0.025, 400))
        cls.synthetic_audio[idx_h_start:idx_h_end] = np.clip(hook_sig + pulses, -1.0, 1.0)

        # 8.0 - 10.0s tail
        idx_t_start = int(8.0 * cls.sr)
        idx_t_end = int(10.0 * cls.sr)
        t_tail = np.linspace(0, 2.0, idx_t_end - idx_t_start, endpoint=False, dtype=np.float32)
        cls.synthetic_audio[idx_t_start:idx_t_end] = 0.10 * np.sin(2 * np.pi * 200 * t_tail)

        # Write to temporary WAV file
        cls.temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        cls.temp_wav_path = cls.temp_wav.name
        cls.temp_wav.close()
        sf.write(cls.temp_wav_path, cls.synthetic_audio, cls.sr)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.temp_wav_path):
            try:
                os.unlink(cls.temp_wav_path)
            except OSError:
                pass

    def test_load_audio_segment_direct(self):
        """Test loading audio slice with soundfile."""
        segment = load_audio_segment(self.temp_wav_path, start_sec=2.0, dur_sec=3.0, sr=self.sr)
        self.assertIsNotNone(segment)
        self.assertIsInstance(segment, np.ndarray)
        self.assertEqual(len(segment), int(3.0 * self.sr))
        self.assertEqual(segment.dtype, np.float32)

    def test_load_audio_segment_missing_file(self):
        """Test that missing file returns None gracefully without exception."""
        segment = load_audio_segment("non_existent_file_xyz123.wav", start_sec=0.0, dur_sec=2.0)
        self.assertIsNone(segment)

    def test_extract_audio_segment_dsp_synthetic_signal(self):
        """
        Verify real DSP metrics on known synthetic audio:
        - Energy surge > 0.60
        - Peak strength > 0.60
        - Pause emphasis >= 0.60 (due to 1.0s silence before hook)
        - Pitch change variation captured via YIN
        - Burstiness measured from onset spectral flux
        - laughter == burstiness (truthful backward-compatible alias)
        """
        dsp = extract_audio_segment_dsp(
            audio=self.synthetic_audio,
            sr=self.sr,
            hook_rel_start=5.5,
            hook_rel_end=8.0,
            pre_hook_dur=2.0
        )

        self.assertIn("energy_change", dsp)
        self.assertIn("peak_strength", dsp)
        self.assertIn("pitch_change", dsp)
        self.assertIn("pause_emphasis", dsp)
        self.assertIn("burstiness", dsp)
        self.assertIn("laughter", dsp)
        self.assertIn("score", dsp)
        self.assertIn("evidence", dsp)

        # High energy hook vs low energy body
        self.assertGreater(dsp["energy_change"], 0.60, f"Expected energy_change > 0.60, got {dsp['energy_change']}")
        self.assertGreater(dsp["peak_strength"], 0.60, f"Expected peak_strength > 0.60, got {dsp['peak_strength']}")

        # 1.0s pre-hook silence should yield measurable pause emphasis
        self.assertGreaterEqual(dsp["pause_emphasis"], 0.60, f"Expected pause_emphasis >= 0.60, got {dsp['pause_emphasis']}")

        # Burstiness & laughter alias
        self.assertGreater(dsp["burstiness"], 0.0)
        self.assertEqual(dsp["burstiness"], dsp["laughter"], "Laughter must be identical to burstiness alias")

        # Score in [0, 1]
        self.assertTrue(0.0 <= dsp["score"] <= 1.0)

        # Truthful evidence strings: no fake heuristics or fabricated emotion
        evidence = dsp["evidence"]
        self.assertIsInstance(evidence, str)
        self.assertNotIn("laughter detected", evidence.lower())
        self.assertNotIn("rising vocal pitch contour with interrogative emphasis", evidence)
        self.assertNotIn("vocal energy surge with assertive peak volume", evidence)
        self.assertNotIn("?", evidence)
        self.assertNotIn("!", evidence)

    def test_extract_audio_segment_dsp_silence_unvoiced_stability(self):
        """Verify robust behavior with flat zero signal (no crash, valid defaults)."""
        silent_audio = np.zeros(int(self.sr * 4.0), dtype=np.float32)
        dsp = extract_audio_segment_dsp(
            audio=silent_audio,
            sr=self.sr,
            hook_rel_start=1.0,
            hook_rel_end=3.0,
            pre_hook_dur=1.0
        )

        self.assertIsInstance(dsp["score"], float)
        self.assertEqual(dsp["burstiness"], 0.0)
        self.assertEqual(dsp["laughter"], 0.0)
        # Neutral pitch fallback for unvoiced
        self.assertEqual(dsp["pitch_change"], 0.5)
        self.assertFalse(np.isnan(dsp["score"]))
        self.assertFalse(np.isinf(dsp["score"]))

    def test_analyze_acoustic_signals_with_audio_file(self):
        """Verify analyze_acoustic_signals end-to-end with candidates and real audio."""
        candidates = [
            {
                "start": 0.0,
                "end": 10.0,
                "hookStart": 5.5,
                "hookEnd": 8.0,
                "hook": "Energetic hook text without fake punctuation markers",
                "score": 0.85
            }
        ]

        status, err, results = analyze_acoustic_signals(self.temp_wav_path, candidates)
        self.assertEqual(status, "SUCCESS")
        self.assertIsNone(err)
        self.assertIn(0, results)

        cand_res = results[0]
        self.assertIn("energy_change", cand_res)
        self.assertIn("peak_strength", cand_res)
        self.assertIn("pitch_change", cand_res)
        self.assertIn("pause_emphasis", cand_res)
        self.assertIn("burstiness", cand_res)
        self.assertIn("laughter", cand_res)
        self.assertEqual(cand_res["burstiness"], cand_res["laughter"])

        evidence = cand_res.get("evidence", "")
        self.assertNotIn("laughter detected", evidence.lower())
        self.assertNotIn("interrogative emphasis", evidence)

    def test_analyze_acoustic_signals_missing_file(self):
        """Verify missing video/audio source gracefully returns SKIPPED."""
        status, err, results = analyze_acoustic_signals("non_existent_audio_file.wav", [{"start": 0.0, "end": 5.0}])
        self.assertEqual(status, "SKIPPED")
        self.assertIsNotNone(err)
        self.assertEqual(results, {})

    def test_analyze_acoustic_signals_force_fail(self):
        """Verify force_fail triggers FAILED status."""
        status, err, results = analyze_acoustic_signals(self.temp_wav_path, [{"start": 0.0, "end": 5.0}], force_fail=True)
        self.assertEqual(status, "FAILED")
        self.assertIn("Forced audio analyzer failure", err)
        self.assertEqual(results, {})

    def test_load_audio_segment_video_container_ffmpeg_pipe(self):
        """Verify audio segment loading from an MP4 video container via FFmpeg pipe."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            mp4_path = f.name

        try:
            # Generate a 3-second mp4 with audio and video test stream
            cmd = [
                "ffmpeg", "-y",
                "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
                "-f", "lavfi", "-i", "testsrc=duration=3:size=320x240:rate=25",
                "-c:v", "libx264", "-c:a", "aac",
                "-pix_fmt", "yuv420p",
                mp4_path
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

            # Load 1.5s slice @ 16kHz
            samples = load_audio_segment(mp4_path, start_sec=1.0, dur_sec=1.5, sr=self.sr)
            self.assertIsNotNone(samples)
            self.assertEqual(len(samples), int(1.5 * self.sr))
            self.assertEqual(samples.dtype, np.float32)
        finally:
            if os.path.exists(mp4_path):
                try:
                    os.unlink(mp4_path)
                except OSError:
                    pass

    def test_video_without_audio_stream_graceful_skip(self):
        """Verify that a video file without an audio stream gracefully returns SKIPPED."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            mp4_no_audio_path = f.name

        try:
            cmd = [
                "ffmpeg", "-y",
                "-f", "lavfi", "-i", "testsrc=duration=2:size=320x240:rate=25",
                "-c:v", "libx264", "-an",
                "-pix_fmt", "yuv420p",
                mp4_no_audio_path
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

            status, err, results = analyze_acoustic_signals(mp4_no_audio_path, [{"start": 0.0, "end": 2.0}])
            self.assertEqual(status, "SKIPPED")
            self.assertIn("No audio stream found", err)
            self.assertEqual(results, {})
        finally:
            if os.path.exists(mp4_no_audio_path):
                try:
                    os.unlink(mp4_no_audio_path)
                except OSError:
                    pass

    def test_pipeline_candidate_enrichment_burstiness(self):
        """Verify pipeline attaches both audioBurstiness and audioLaughter (backward compatibility)."""
        candidates = [
            {
                "start": 0.0,
                "end": 10.0,
                "hookStart": 5.5,
                "hookEnd": 8.0,
                "hook": "Testing pipeline candidate enrichment",
                "score": 0.85
            }
        ]

        result = process_multimodal_pipeline(self.temp_wav_path, candidates, force_fail_visual=True)
        self.assertIn(result["status"], ["SUCCESS", "PARTIAL"])
        cand = result["candidates"][0]

        self.assertIn("audioBurstiness", cand)
        self.assertIn("audioLaughter", cand)
        self.assertEqual(cand["audioBurstiness"], cand["audioLaughter"])
        self.assertGreater(cand["audioBurstiness"], 0.0)


if __name__ == "__main__":
    unittest.main()
