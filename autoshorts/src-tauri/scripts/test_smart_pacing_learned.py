#!/usr/bin/env python3
"""
Unit test suite for learned Pause Intelligence integration in smart_pacing.py.
Covers Task 5 requirements:
1. Model disabled / missing -> byte-identical edit plan to baseline.
2. Model enabled + removable class (breath_pause, waiting_pause) with P >= 0.65 -> gap marked removable.
3. Model enabled + non-removable class (sentence_pause, normal_word_gap, speaker_transition, nonspeech_unknown)
   with P >= 0.65 -> NEVER marked removable.
4. Keep-Wins-On-Conflict invariant: deterministic safety rules (word clearance, silence verification,
   speaker transitions, min duration) retain full veto power over model recommendations.
5. Soft degradation on missing dependencies, missing models, or scoring exceptions.
6. Verification of [PauseIntel] gaps_scored=<n> removable=<n> model=<model_name> logging.
"""

import io
import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

import smart_pacing as sp


class TestSmartPacingLearned(unittest.TestCase):
    def setUp(self):
        self.orig_env = {
            "AUTOSHORTS_SMART_PACING_LEARNED": os.environ.get("AUTOSHORTS_SMART_PACING_LEARNED"),
            "AUTOSHORTS_PAUSE_MODEL": os.environ.get("AUTOSHORTS_PAUSE_MODEL"),
            "AUTOSHORTS_PACING_TELEMETRY_DIR": os.environ.get("AUTOSHORTS_PACING_TELEMETRY_DIR"),
        }
        for k in ("AUTOSHORTS_SMART_PACING_LEARNED", "AUTOSHORTS_PAUSE_MODEL", "AUTOSHORTS_PACING_TELEMETRY_DIR"):
            os.environ.pop(k, None)

    def tearDown(self):
        for k, v in self.orig_env.items():
            if v is not None:
                os.environ[k] = v
            else:
                os.environ.pop(k, None)

    def test_01_model_disabled_byte_identical_fallback(self):
        """Model disabled via AUTOSHORTS_SMART_PACING_LEARNED=0 yields byte-identical plan to baseline."""
        words = [
            {"text": "hello", "start": 0.5, "end": 1.0, "speaker": "S1"},
            {"text": "world", "start": 2.5, "end": 3.0, "speaker": "S1"},
        ]
        silences = [(1.0, 2.5)]

        # Baseline plan with no model
        os.environ.pop("AUTOSHORTS_PAUSE_MODEL", None)
        os.environ.pop("AUTOSHORTS_SMART_PACING_LEARNED", None)
        plan_baseline = sp.plan_pacing(words, 0.0, 4.0, silences, v2=True)

        # Plan with model disabled explicitly
        os.environ["AUTOSHORTS_SMART_PACING_LEARNED"] = "0"
        os.environ["AUTOSHORTS_PAUSE_MODEL"] = "fake_model.txt"
        plan_disabled = sp.plan_pacing(words, 0.0, 4.0, silences, v2=True)

        self.assertEqual(
            json.dumps(plan_baseline, sort_keys=True),
            json.dumps(plan_disabled, sort_keys=True),
            "Disabled learned stage must produce byte-identical plan to baseline"
        )

        # Also verify gaps untouched in _attach_learned_pause_evidence
        gaps = [{"srcStartSec": 1.0, "srcEndSec": 2.0, "category": "breath_pause"}]
        sp._attach_learned_pause_evidence("source.mp4", 0.0, 4.0, words, gaps)
        self.assertNotIn("removable", gaps[0])
        self.assertNotIn("pRemovable", gaps[0])

    def test_02_model_missing_leaves_gaps_untouched(self):
        """Missing model file (find_pause_model returns None) leaves gaps untouched."""
        os.environ["AUTOSHORTS_SMART_PACING_LEARNED"] = "1"
        os.environ.pop("AUTOSHORTS_PAUSE_MODEL", None)

        gaps = [{"srcStartSec": 1.0, "srcEndSec": 2.0, "category": "breath_pause"}]
        with patch("pause_intelligence.find_pause_model", return_value=None):
            sp._attach_learned_pause_evidence("source.mp4", 0.0, 4.0, [], gaps)

        self.assertNotIn("removable", gaps[0])
        self.assertNotIn("pRemovable", gaps[0])

    def test_03_model_enabled_breath_pause_marked_removable(self):
        """breath_pause with P(removable) >= 0.65 is marked removable with evidence attached."""
        os.environ["AUTOSHORTS_SMART_PACING_LEARNED"] = "1"

        gaps = [
            {"srcStartSec": 1.0, "srcEndSec": 1.5, "category": "breath_pause"}
        ]
        words = [
            {"text": "hello", "start": 0.5, "end": 1.0, "speaker": "S1"},
            {"text": "world", "start": 1.5, "end": 2.0, "speaker": "S1"},
        ]

        def fake_score(gaps_list, rows, model_path):
            for g in gaps_list:
                g["pRemovable"] = 0.85
                g["pRemovableAboveThreshold"] = True
            return True

        with patch("pause_intelligence.find_pause_model", return_value="/tmp/pause_classifier_v1.txt"), \
             patch("pause_intelligence.build_gap_features", return_value=[{"durationSec": 0.5}]), \
             patch("pause_intelligence.features_to_row", return_value=[0.0] * 23), \
             patch("pause_intelligence.score_gaps_with_model", side_effect=fake_score):
            sp._attach_learned_pause_evidence("source.mp4", 0.0, 3.0, words, gaps)

        self.assertTrue(gaps[0].get("removable"))
        self.assertAlmostEqual(gaps[0]["pRemovable"], 0.85)
        self.assertTrue(gaps[0].get("evidence", {}).get("pause_intel"))
        self.assertAlmostEqual(gaps[0]["evidence"]["pRemovable"], 0.85)

    def test_04_model_enabled_waiting_pause_marked_removable(self):
        """waiting_pause with P(removable) >= 0.65 is marked removable."""
        os.environ["AUTOSHORTS_SMART_PACING_LEARNED"] = "1"

        gaps = [
            {"srcStartSec": 1.0, "srcEndSec": 1.8, "category": "waiting_pause"}
        ]
        words = [
            {"text": "start", "start": 0.5, "end": 1.0, "speaker": "S1"},
            {"text": "next", "start": 1.8, "end": 2.5, "speaker": "S1"},
        ]

        def fake_score(gaps_list, rows, model_path):
            for g in gaps_list:
                g["pRemovable"] = 0.72
                g["pRemovableAboveThreshold"] = True
            return True

        with patch("pause_intelligence.find_pause_model", return_value="/tmp/pause_classifier_v1.txt"), \
             patch("pause_intelligence.build_gap_features", return_value=[{"durationSec": 0.8}]), \
             patch("pause_intelligence.features_to_row", return_value=[0.0] * 23), \
             patch("pause_intelligence.score_gaps_with_model", side_effect=fake_score):
            sp._attach_learned_pause_evidence("source.mp4", 0.0, 3.0, words, gaps)

        self.assertTrue(gaps[0].get("removable"))
        self.assertAlmostEqual(gaps[0]["pRemovable"], 0.72)

    def test_05_below_threshold_never_marked_removable(self):
        """breath_pause with P(removable) < 0.65 is NEVER marked removable."""
        os.environ["AUTOSHORTS_SMART_PACING_LEARNED"] = "1"

        gaps = [
            {"srcStartSec": 1.0, "srcEndSec": 1.4, "category": "breath_pause"}
        ]

        def fake_score(gaps_list, rows, model_path):
            for g in gaps_list:
                g["pRemovable"] = 0.40
                g["pRemovableAboveThreshold"] = False
            return True

        with patch("pause_intelligence.find_pause_model", return_value="/tmp/pause_classifier_v1.txt"), \
             patch("pause_intelligence.build_gap_features", return_value=[{"durationSec": 0.4}]), \
             patch("pause_intelligence.features_to_row", return_value=[0.0] * 23), \
             patch("pause_intelligence.score_gaps_with_model", side_effect=fake_score):
            sp._attach_learned_pause_evidence("source.mp4", 0.0, 3.0, [], gaps)

        self.assertNotIn("removable", gaps[0])
        self.assertAlmostEqual(gaps[0]["pRemovable"], 0.40)

    def test_06_protected_class_sentence_pause_never_removable(self):
        """sentence_pause with P(removable) >= 0.65 must NEVER be marked removable."""
        os.environ["AUTOSHORTS_SMART_PACING_LEARNED"] = "1"

        gaps = [
            {"srcStartSec": 1.0, "srcEndSec": 2.0, "category": "sentence_pause"}
        ]

        def fake_score(gaps_list, rows, model_path):
            for g in gaps_list:
                g["pRemovable"] = 0.95
                g["pRemovableAboveThreshold"] = True
            return True

        with patch("pause_intelligence.find_pause_model", return_value="/tmp/pause_classifier_v1.txt"), \
             patch("pause_intelligence.build_gap_features", return_value=[{"durationSec": 1.0}]), \
             patch("pause_intelligence.features_to_row", return_value=[0.0] * 23), \
             patch("pause_intelligence.score_gaps_with_model", side_effect=fake_score):
            sp._attach_learned_pause_evidence("source.mp4", 0.0, 3.0, [], gaps)

        self.assertNotIn("removable", gaps[0], "sentence_pause must NEVER be marked removable")

    def test_07_protected_class_normal_word_gap_never_removable(self):
        """normal_word_gap with P(removable) >= 0.65 must NEVER be marked removable."""
        os.environ["AUTOSHORTS_SMART_PACING_LEARNED"] = "1"

        gaps = [
            {"srcStartSec": 1.0, "srcEndSec": 1.2, "category": "normal_word_gap"}
        ]

        def fake_score(gaps_list, rows, model_path):
            for g in gaps_list:
                g["pRemovable"] = 0.90
                g["pRemovableAboveThreshold"] = True
            return True

        with patch("pause_intelligence.find_pause_model", return_value="/tmp/pause_classifier_v1.txt"), \
             patch("pause_intelligence.build_gap_features", return_value=[{"durationSec": 0.2}]), \
             patch("pause_intelligence.features_to_row", return_value=[0.0] * 23), \
             patch("pause_intelligence.score_gaps_with_model", side_effect=fake_score):
            sp._attach_learned_pause_evidence("source.mp4", 0.0, 3.0, [], gaps)

        self.assertNotIn("removable", gaps[0], "normal_word_gap must NEVER be marked removable")

    def test_08_protected_class_speaker_transition_and_nonspeech_never_removable(self):
        """speaker_transition and nonspeech_unknown must NEVER be marked removable even with high P."""
        os.environ["AUTOSHORTS_SMART_PACING_LEARNED"] = "1"

        gaps = [
            {"srcStartSec": 1.0, "srcEndSec": 1.8, "category": "speaker_transition"},
            {"srcStartSec": 2.0, "srcEndSec": 2.5, "category": "nonspeech_unknown"},
        ]

        def fake_score(gaps_list, rows, model_path):
            for g in gaps_list:
                g["pRemovable"] = 0.88
                g["pRemovableAboveThreshold"] = True
            return True

        with patch("pause_intelligence.find_pause_model", return_value="/tmp/pause_classifier_v1.txt"), \
             patch("pause_intelligence.build_gap_features", return_value=[{"durationSec": 0.8}, {"durationSec": 0.5}]), \
             patch("pause_intelligence.features_to_row", return_value=[0.0] * 23), \
             patch("pause_intelligence.score_gaps_with_model", side_effect=fake_score):
            sp._attach_learned_pause_evidence("source.mp4", 0.0, 3.0, [], gaps)

        self.assertNotIn("removable", gaps[0], "speaker_transition must NEVER be marked removable")
        self.assertNotIn("removable", gaps[1], "nonspeech_unknown must NEVER be marked removable")

    def test_09_keep_wins_on_conflict_word_clearance_violation(self):
        """Deterministic safety rule (word clearance) vetoes model removable recommendation."""
        words = [
            {"text": "one", "start": 1.0, "end": 1.45, "speaker": "S1"},
            {"text": "two", "start": 1.50, "end": 2.0, "speaker": "S1"},
        ]
        silences = [(1.45, 1.50)]  # only 50ms gap -> violates BREATH_MIN_DUR (60ms) and clearance

        # Breath gap marked removable by model
        fake_gaps = [
            {
                "srcStartSec": 1.45,
                "srcEndSec": 1.50,
                "category": "breath_pause",
                "removable": True,
                "confidence": 0.95,
                "pRemovable": 0.95,
            }
        ]

        notes = []
        with patch.object(sp, "run_breath_detection", return_value=fake_gaps):
            plan = sp.plan_pacing(words, 0.0, 6.0, silences, v2=True, source_path="dummy.mp4")

        # The edit must be vetoed because 0.05s < BREATH_MIN_DUR (0.06s)
        breath_edits = [e for e in plan.get("edits", []) if e["editType"] == "breath_pause"]
        self.assertEqual(len(breath_edits), 0, "Deterministic safety rule must veto gap below minimum duration")

    def test_10_keep_wins_on_conflict_unverified_silence(self):
        """Acoustic verification failure vetoes model recommendation (gap kept)."""
        words = [
            {"text": "speech1", "start": 0.5, "end": 1.0, "speaker": "S1"},
            {"text": "speech2", "start": 1.6, "end": 2.2, "speaker": "S1"},
        ]
        # Silence does NOT cover the gap (e.g. background noise or unverified audio)
        silences = []

        fake_gaps = [
            {
                "srcStartSec": 1.0,
                "srcEndSec": 1.6,
                "category": "breath_pause",
                "removable": True,
                "confidence": 0.90,
                "pRemovable": 0.90,
            }
        ]

        with patch.object(sp, "run_breath_detection", return_value=fake_gaps):
            plan = sp.plan_pacing(words, 0.0, 6.0, silences, v2=True, source_path="dummy.mp4")

        # Zero edits because silence wasn't acoustically verified
        self.assertEqual(len(plan.get("edits", [])), 0, "Unverified silence must veto candidate cut")

    def test_11_keep_wins_on_conflict_speaker_change(self):
        """Speaker transition between words vetoes model cut recommendation."""
        words = [
            {"text": "speaker_one", "start": 0.5, "end": 1.0, "speaker": "S1"},
            {"text": "speaker_two", "start": 1.6, "end": 2.2, "speaker": "S2"},
        ]
        silences = [(1.0, 1.6)]

        fake_gaps = [
            {
                "srcStartSec": 1.0,
                "srcEndSec": 1.6,
                "category": "breath_pause",
                "removable": True,
                "confidence": 0.90,
                "pRemovable": 0.90,
            }
        ]

        with patch.object(sp, "run_breath_detection", return_value=fake_gaps):
            plan = sp.plan_pacing(words, 0.0, 6.0, silences, v2=True, source_path="dummy.mp4")

        # Crossed speaker boundary -> vetoed
        breath_edits = [e for e in plan.get("edits", []) if e["editType"] == "breath_pause"]
        self.assertEqual(len(breath_edits), 0, "Speaker change must veto cut")

    def test_12_log_emission_format(self):
        """Verify [PauseIntel] gaps_scored=<n> removable=<n> model=<model_name> format on stderr."""
        os.environ["AUTOSHORTS_SMART_PACING_LEARNED"] = "1"

        gaps = [
            {"srcStartSec": 1.0, "srcEndSec": 1.5, "category": "breath_pause"},
            {"srcStartSec": 2.0, "srcEndSec": 2.5, "category": "sentence_pause"},
        ]

        def fake_score(gaps_list, rows, model_path):
            gaps_list[0]["pRemovable"] = 0.80
            gaps_list[0]["pRemovableAboveThreshold"] = True
            gaps_list[1]["pRemovable"] = 0.90
            gaps_list[1]["pRemovableAboveThreshold"] = True
            return True

        captured_stderr = io.StringIO()
        with patch("sys.stderr", captured_stderr), \
             patch("pause_intelligence.find_pause_model", return_value="/models/test_classifier.txt"), \
             patch("pause_intelligence.build_gap_features", return_value=[{}, {}]), \
             patch("pause_intelligence.features_to_row", return_value=[0.0] * 23), \
             patch("pause_intelligence.score_gaps_with_model", side_effect=fake_score):
            sp._attach_learned_pause_evidence("source.mp4", 0.0, 4.0, [], gaps)

        output = captured_stderr.getvalue()
        self.assertIn("[PauseIntel] gaps_scored=2 removable=1 model=test_classifier.txt", output)

    def test_13_fail_soft_on_exceptions(self):
        """Any exception in scoring fails soft, logs error, and leaves gaps unchanged."""
        os.environ["AUTOSHORTS_SMART_PACING_LEARNED"] = "1"

        gaps = [{"srcStartSec": 1.0, "srcEndSec": 1.5, "category": "breath_pause"}]

        with patch("pause_intelligence.find_pause_model", return_value="model.txt"), \
             patch("pause_intelligence.build_gap_features", side_effect=RuntimeError("Simulated lightgbm failure")):
            # Must not raise
            sp._attach_learned_pause_evidence("source.mp4", 0.0, 3.0, [], gaps)

        self.assertNotIn("removable", gaps[0])
        self.assertNotIn("pRemovable", gaps[0])

    def test_14_detect_gaps_v2_expansion(self):
        """detect_gaps_v2 (run_breath_detection) expands reducible gaps with model-removable gaps."""
        # Mock breath_detect.detect_gaps
        mock_result = {
            "status": "ok",
            "gaps": [
                {"srcStartSec": 1.0, "srcEndSec": 1.5, "category": "breath_pause"},
                {"srcStartSec": 2.0, "srcEndSec": 2.5, "category": "dramatic_pause"},
                {"srcStartSec": 3.0, "srcEndSec": 3.5, "category": "sentence_pause"},
            ]
        }

        with patch("breath_detect.detect_gaps", return_value=mock_result), \
             patch.object(sp, "_attach_learned_pause_evidence") as mock_attach:
            # Side effect to simulate model marking dramatic_pause as removable
            def do_attach(source, start, end, words, gaps):
                gaps[1]["removable"] = True
                gaps[1]["evidence"] = {"pause_intel": True, "pRemovable": 0.88}
                # Even if sentence_pause had removable=True mistakenly set, reducible excludes it
                gaps[2]["removable"] = True
            mock_attach.side_effect = do_attach

            reducible = sp.detect_gaps_v2("dummy.mp4", 0.0, 5.0, [])

        # breath_pause is in reducible by category
        # dramatic_pause is in reducible by removable=True
        # sentence_pause is excluded by category guard
        self.assertEqual(len(reducible), 2)
        self.assertEqual(reducible[0]["category"], "breath_pause")
        self.assertEqual(reducible[1]["category"], "dramatic_pause")

    def test_15_model_enabled_valid_gap_creates_reduction_edit(self):
        """When model marks gap removable and all deterministic safety checks pass, edit is created."""
        words = [
            {"text": "hello", "start": 0.5, "end": 1.0, "speaker": "S1"},
            {"text": "world", "start": 1.8, "end": 2.5, "speaker": "S1"},
            {"text": "done", "start": 4.5, "end": 5.0, "speaker": "S1"},
        ]
        # Silence fully covers the 1.0 -> 1.8 interval
        silences = [(1.0, 1.8)]

        fake_gaps = [
            {
                "srcStartSec": 1.0,
                "srcEndSec": 1.8,
                "category": "breath_pause",
                "removable": True,
                "confidence": 0.92,
                "pRemovable": 0.92,
            }
        ]

        with patch.object(sp, "run_breath_detection", return_value=fake_gaps):
            plan = sp.plan_pacing(words, 0.0, 6.0, silences, v2=True, source_path="dummy.mp4")

        self.assertEqual(plan["status"], "ok")
        breath_edits = [e for e in plan["edits"] if e["editType"] == "breath_pause"]
        self.assertEqual(len(breath_edits), 1)
        self.assertAlmostEqual(breath_edits[0]["confidence"], 0.92)


if __name__ == "__main__":
    unittest.main(verbosity=2)
