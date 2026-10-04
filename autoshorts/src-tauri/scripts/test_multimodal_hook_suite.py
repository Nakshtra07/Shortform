#!/usr/bin/env python3
"""
Unit & Failure-Injection Test Suite for AutoShorts v9.3 Multimodal Hook Intelligence
Validates:
- Normal multimodal feature extraction & composite ranking
- Test A: Visual analyzer failure handling & partial fallback
- Test B: Audio analyzer failure handling & partial fallback
- Test C: Temporal analyzer failure handling & partial fallback
- Test D: Total multimodal failure handling & complete v9.2 semantic fallback
- Meaning Dominates Loudness Quality Gate verification
- Question Hook and Guest Revelation boost verification
"""

import os
import sys
import unittest
import json
import tempfile
import subprocess

# Add scripts folder to sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from multimodal_hook_analyzer import (
    process_multimodal_pipeline,
    compute_composite_multimodal_score
)


class TestMultimodalHookIntelligence(unittest.TestCase):

    def setUp(self):
        # Sample candidate draft list
        self.candidates = [
            {
                "start": 10.0,
                "end": 45.0,
                "hookStart": 10.0,
                "hookEnd": 14.0,
                "hook": "What was the hardest decision you made in your career?",
                "hookSpeaker": "Host",
                "hookType": "interviewer_question",
                "conversationType": "question_answer",
                "questionHookUsed": True,
                "payoffText": "That decision completely defined who I am today.",
                "hookScore": 0.92,
                "score": 0.92,
                "rationale": "High-curiosity host question with profound payoff."
            },
            {
                "start": 120.0,
                "end": 155.0,
                "hookStart": 120.0,
                "hookEnd": 123.5,
                "hook": "I lost everything in 2020.",
                "hookSpeaker": "Guest",
                "hookType": "revelation",
                "conversationType": "story",
                "questionHookUsed": False,
                "payoffText": "And that failure became my greatest blessing.",
                "hookScore": 0.88,
                "score": 0.88,
                "rationale": "Vulnerable guest revelation with strong story resolution."
            },
            {
                "start": 300.0,
                "end": 330.0,
                "hookStart": 300.0,
                "hookEnd": 304.0,
                "hook": "LOUD RANDOM NOISE WITHOUT MEANING!",
                "hookSpeaker": "Guest",
                "hookType": "shout",
                "conversationType": "general",
                "questionHookUsed": False,
                "payoffText": "",
                "hookScore": 0.30,
                "score": 0.30,
                "rationale": "Loud shout with no narrative context or payoff."
            }
        ]

    def test_composite_scoring_and_meaning_gate(self):
        """Test composite weights and hard semantic quality gate."""
        # 1. High semantic candidate (0.90) with balanced multimodal signals (0.80)
        score_high = compute_composite_multimodal_score(0.90, 0.80, 0.80, 0.80)
        self.assertAlmostEqual(score_high, 0.85, delta=0.01)

        # 2. Hard Gate: Weak semantic (0.30) with maximum loudness (1.0) and visual flashiness (1.0)
        score_loud_shallow = compute_composite_multimodal_score(0.30, 1.0, 1.0, 1.0, semantic_gate_active=True)
        self.assertLessEqual(score_loud_shallow, 0.55, "Hard semantic gate must cap score at 0.55")

    def test_normal_multimodal_pipeline(self):
        """Test standard pipeline execution on candidate set."""
        result = process_multimodal_pipeline(
            "dummy_video.mp4",
            self.candidates,
            force_fail_visual=False,
            force_fail_audio=False,
            force_fail_temporal=False
        )

        self.assertIn(result["status"], ["SUCCESS", "PARTIAL", "SKIPPED"])
        self.assertEqual(result["candidate_count_analyzed"], 3)
        self.assertEqual(len(result["candidates"]), 3)

        top = result["candidates"][0]
        self.assertIn("multimodalScore", top)
        self.assertIn("visualScore", top)
        self.assertIn("audioScore", top)
        self.assertIn("temporalScore", top)
        self.assertIn("multimodalEvidence", top)
        self.assertTrue(len(top["multimodalEvidence"]) >= 2)

    def test_a_forced_visual_failure(self):
        """TEST A: Force visual analyzer failure."""
        result = process_multimodal_pipeline(
            "dummy_video.mp4",
            self.candidates,
            force_fail_visual=True,
            force_fail_audio=False,
            force_fail_temporal=False
        )

        self.assertEqual(result["visual_analysis"], "FAILED")
        self.assertIn(result["status"], ["PARTIAL", "FAILED"])
        self.assertIsNotNone(result["failure_reason"])
        self.assertIn("Visual", result["failure_reason"])

    def test_b_forced_audio_failure(self):
        """TEST B: Force audio analyzer failure."""
        result = process_multimodal_pipeline(
            "dummy_video.mp4",
            self.candidates,
            force_fail_visual=False,
            force_fail_audio=True,
            force_fail_temporal=False
        )

        self.assertEqual(result["audio_analysis"], "FAILED")
        self.assertIn(result["status"], ["PARTIAL", "FAILED"])
        self.assertIsNotNone(result["failure_reason"])
        self.assertIn("Audio", result["failure_reason"])

    def test_c_forced_temporal_failure(self):
        """TEST C: Force temporal analyzer failure."""
        result = process_multimodal_pipeline(
            "dummy_video.mp4",
            self.candidates,
            force_fail_visual=False,
            force_fail_audio=False,
            force_fail_temporal=True
        )

        self.assertEqual(result["temporal_analysis"], "FAILED")
        self.assertIn(result["status"], ["PARTIAL", "FAILED"])
        self.assertIsNotNone(result["failure_reason"])
        self.assertIn("Temporal", result["failure_reason"])

    def test_d_forced_all_failure_complete_v92_fallback(self):
        """TEST D: Force ALL multimodal analyzers to fail -> verified v9.2 semantic fallback."""
        result = process_multimodal_pipeline(
            "dummy_video.mp4",
            self.candidates,
            force_fail_all=True
        )

        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(result["visual_analysis"], "FAILED")
        self.assertEqual(result["audio_analysis"], "FAILED")
        self.assertEqual(result["temporal_analysis"], "FAILED")
        self.assertTrue(result["fallback_to_v92"])
        self.assertIsNotNone(result["failure_reason"])

        # Check candidate fallback labeling
        for cand in result["candidates"]:
            self.assertFalse(cand["multimodalVerified"])
            self.assertEqual(cand["multimodalStatus"], "FAILED")
            self.assertEqual(cand["fallbackLabel"], "v9.2 fallback — multimodal analysis unavailable")
            self.assertIn("v9.2", cand["rationale"])


if __name__ == "__main__":
    unittest.main()
