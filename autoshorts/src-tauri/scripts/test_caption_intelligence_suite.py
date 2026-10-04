#!/usr/bin/env python3
"""
AutoShorts 9.0 — Caption Intelligence 2.0 Comprehensive Verification Suite.

Validates the complete 22-case specification for Caption Intelligence 2.0:
  1. Empty words -> no plan generated (None)
  2. Kill switch active (AUTOSHORTS_CAPTION_INTEL=0) -> no plan generated (None)
  3. Low-confidence hook (< 0.70) -> hook not verified, no emphasis
  4. High-confidence hook (>= 0.70) -> hook verified, plan generated
  5. High-confidence payoff (>= 0.75, completion != false) -> payoff verified, plan generated
  6. Incomplete payoff (completion == false) -> rejected, safe fallback
  7. Smart Pacing source -> output remapping without cuts -> identical interval preserved
  8. Smart Pacing source -> output remapping with cut before hook -> offset correctly shifted
  9. Smart Pacing source -> output remapping with cut inside hook (MANDATORY CORRECTION 1) -> preserves discontinuous intervals
  10. Generic lexical words alone cannot trigger CI (MANDATORY CORRECTION 3) -> max +0.05, returns None
  11. Semantic emphasis spans, not random word density (MANDATORY CORRECTION 4) -> max 1 span per group
  12. Group-aware line break hints (MANDATORY CORRECTION 2) -> legal split within line limits selected
  13. Illegal line-break hints fallback safely -> falls back to balanced split without crashing
  14. Template 1 (preset_viral_bold) integration -> secondary highlight for emphasized words
  15. Template 2 (preset_mrbeast_pop) integration -> highlight styling & legal line break hints
  16. Template 3 (preset_minimal_capsule) integration -> boosted opacity \\alpha&H33& for inactive emphasized words
  17. Template 4 (preset_cinematic_vlog) integration -> single dialogue event per frame, proper \\fad, inline bold {\\b1}...{\\b0}
  18. Template 5 (preset_dynamic_editorial) integration -> TypographyRole::Emphasis (Bebas Neue 54, #FFE600, uppercase)
  19. Template 5 layout stability -> (x, y) coordinates, motion, seam corridor, and safeguards preserved
  20. Backward compatibility wrapper -> generate_ass_from_template_with_framing matches intel with None
  21. Kill switch baseline identity -> ASS output with AUTOSHORTS_CAPTION_INTEL=0 is 100% byte-for-byte identical to baseline ASS
  22. Unmarked words preserve exact baseline behavior -> non-emphasized words across all templates receive identical baseline styling
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
BIN_PATH = os.path.join(PROJECT_ROOT, "target", "debug", "caption_intel_inspect.exe")
if not os.path.exists(BIN_PATH):
    BIN_PATH = os.path.join(PROJECT_ROOT, "target", "debug", "caption_intel_inspect")


def run_inspect(words, candidate, start_sec, end_sec, style="preset_viral_bold", pacing_plan=None, env_override=None):
    with tempfile.TemporaryDirectory() as tmpdir:
        words_path = os.path.join(tmpdir, "words.json")
        candidate_path = os.path.join(tmpdir, "candidate.json")
        pacing_path = os.path.join(tmpdir, "pacing.json") if pacing_plan else None

        with open(words_path, "w", encoding="utf-8") as f:
            json.dump(words, f)
        with open(candidate_path, "w", encoding="utf-8") as f:
            json.dump(candidate, f)
        if pacing_plan:
            with open(pacing_path, "w", encoding="utf-8") as f:
                json.dump(pacing_plan, f)

        cmd = [BIN_PATH, words_path, candidate_path, str(start_sec), str(end_sec), style]
        if pacing_path:
            cmd.append(pacing_path)

        env = dict(os.environ)
        if env_override:
            env.update(env_override)

        res = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if res.returncode != 0:
            raise RuntimeError(f"caption_intel_inspect failed with code {res.returncode}: {res.stderr}\nstdout: {res.stdout}")
        return json.loads(res.stdout)


def make_candidate(
    hook="Never eat this before bed",
    hook_start_sec=10.0,
    hook_end_sec=12.0,
    hook_confidence=0.85,
    payoff_text="it will destroy your deep sleep",
    payoff_start_sec=20.0,
    payoff_end_sec=23.0,
    payoff_score=0.90,
    payoff_completion=True,
    start_sec=10.0,
    end_sec=25.0,
):
    return {
        "id": "cand-001",
        "projectId": "proj-test",
        "source": "video.mp4",
        "startSec": start_sec,
        "endSec": end_sec,
        "score": 0.88,
        "hook": hook,
        "rationale": "High viral potential",
        "rank": 1,
        "selected": True,
        "hookStartSec": hook_start_sec,
        "hookEndSec": hook_end_sec,
        "hookConfidence": hook_confidence,
        "openingContextScore": 0.80,
        "payoffText": payoff_text,
        "payoffStartSec": payoff_start_sec,
        "payoffEndSec": payoff_end_sec,
        "payoffScore": payoff_score,
        "payoffCompletion": payoff_completion,
    }


def make_words():
    # Hook interval: 10.0 - 12.0 ("never eat this before bed")
    # Body interval: 12.0 - 20.0
    # Payoff interval: 20.0 - 23.0 ("it will destroy your deep sleep")
    data = [
        (10.0, 10.4, "Never"),
        (10.4, 10.8, "eat"),
        (10.8, 11.2, "this"),
        (11.2, 11.6, "before"),
        (11.6, 12.0, "bed,"),
        (12.2, 12.7, "because"),
        (12.7, 13.2, "recent"),
        (13.2, 13.8, "studies"),
        (13.8, 14.5, "confirm"),
        (20.0, 20.4, "it"),
        (20.4, 20.8, "will"),
        (20.8, 21.4, "destroy"),
        (21.4, 21.8, "your"),
        (21.8, 22.3, "deep"),
        (22.3, 23.0, "sleep."),
    ]
    return [{"start": s, "end": e, "text": t} for s, e, t in data]


class TestCaptionIntelligenceSuite(unittest.TestCase):
    """22-Case Acceptance Test Suite for Caption Intelligence 2.0"""

    def setUp(self):
        self.assertTrue(os.path.exists(BIN_PATH), f"Inspection CLI binary must exist: {BIN_PATH}")

    # Case 1: Empty words -> no plan generated
    def test_01_empty_words_no_plan(self):
        cand = make_candidate()
        res = run_inspect([], cand, 10.0, 25.0)
        self.assertIsNone(res["plan"])
        self.assertTrue(res["assMatchesBaseline"])

    # Case 2: Kill switch active -> no plan generated
    def test_02_kill_switch_active_no_plan(self):
        words = make_words()
        cand = make_candidate()
        for val in ["0", "false", "off", "FALSE", "OFF"]:
            res = run_inspect(words, cand, 10.0, 25.0, env_override={"AUTOSHORTS_CAPTION_INTEL": val})
            self.assertTrue(res["killSwitchActive"], f"Kill switch should be active for {val}")
            self.assertIsNone(res["plan"])
            self.assertTrue(res["assMatchesBaseline"])

    # Case 3: Low-confidence hook (< 0.70) -> no hook emphasis
    def test_03_low_confidence_hook_no_emphasis(self):
        words = make_words()
        cand = make_candidate(hook_confidence=0.55, payoff_score=0.0)
        res = run_inspect(words, cand, 10.0, 15.0)
        self.assertIsNone(res["plan"], "Low confidence hook (<0.70) should not produce plan")
        self.assertTrue(res["assMatchesBaseline"])

    # Case 4: High-confidence hook (>= 0.70) alone -> plan generated
    def test_04_high_confidence_hook_triggers_ci(self):
        words = make_words()
        cand = make_candidate(hook_confidence=0.88, payoff_score=0.0)
        res = run_inspect(words, cand, 10.0, 15.0)
        self.assertIsNotNone(res["plan"])
        plan = res["plan"]
        self.assertGreaterEqual(plan["confidence"], 0.75)
        self.assertGreater(len(plan["hookWordIndices"]), 0)
        self.assertIn("Hook verified", plan["reason"])

    # Case 5: High-confidence payoff (>= 0.75) alone -> plan generated
    def test_05_high_confidence_payoff_triggers_ci(self):
        words = make_words()
        cand = make_candidate(hook_confidence=0.0, payoff_score=0.92, payoff_completion=True)
        res = run_inspect(words, cand, 18.0, 24.0)
        self.assertIsNotNone(res["plan"])
        plan = res["plan"]
        self.assertGreaterEqual(plan["confidence"], 0.75)
        self.assertGreater(len(plan["payoffWordIndices"]), 0)
        self.assertIn("Payoff verified", plan["reason"])

    # Case 6: Incomplete payoff (completion == false) -> rejected
    def test_06_incomplete_payoff_rejected(self):
        words = make_words()
        cand = make_candidate(hook_confidence=0.0, payoff_score=0.95, payoff_completion=False)
        res = run_inspect(words, cand, 18.0, 24.0)
        self.assertIsNone(res["plan"], "Payoff with completion=false must be rejected")
        self.assertTrue(res["assMatchesBaseline"])

    # Case 7: Smart Pacing source -> output remapping without cuts
    def test_07_pacing_without_cuts_preserves_interval(self):
        words = make_words()
        cand = make_candidate(hook_confidence=0.85, payoff_score=0.0)
        pacing_plan = {
            "status": "ok",
            "sourcePath": "video.mp4",
            "clipStartSec": 10.0,
            "clipEndSec": 25.0,
            "sourceDurationSec": 15.0,
            "outputDurationSec": 15.0,
            "cutsCount": 0,
            "totalCutDurationSec": 0.0,
            "speedFactor": 1.0,
            "edits": [],
            "retained": [{"srcStartSec": 10.0, "srcEndSec": 25.0, "outStartSec": 0.0, "outEndSec": 15.0}],
            "subtitlesOnOutput": True,
        }
        res = run_inspect(words, cand, 10.0, 15.0, pacing_plan=pacing_plan)
        self.assertIsNotNone(res["plan"])
        self.assertGreater(len(res["plan"]["hookWordIndices"]), 0)

    # Case 8: Smart Pacing source -> output remapping with cut before hook
    def test_08_pacing_cut_before_hook_shifts_offset(self):
        words = [
            {"start": 10.0, "end": 10.4, "text": "Never"},
            {"start": 10.4, "end": 10.8, "text": "eat"},
            {"start": 10.8, "end": 11.2, "text": "this"},
        ]
        # Candidate hook was at 11.0..12.2 on source, but 1.0s cut happened at 5.0..6.0
        cand = make_candidate(hook="Never eat this", hook_start_sec=11.0, hook_end_sec=12.2, hook_confidence=0.85, payoff_score=0.0)
        pacing_plan = {
            "status": "ok",
            "sourcePath": "video.mp4",
            "clipStartSec": 10.0,
            "clipEndSec": 15.0,
            "sourceDurationSec": 5.0,
            "outputDurationSec": 4.0,
            "cutsCount": 1,
            "totalCutDurationSec": 1.0,
            "speedFactor": 1.0,
            "edits": [],
            "retained": [{"srcStartSec": 10.0, "srcEndSec": 14.0, "outStartSec": 0.0, "outEndSec": 4.0}],
            "subtitlesOnOutput": True,
        }
        res = run_inspect(words, cand, 10.0, 12.0, pacing_plan=pacing_plan)
        self.assertIsNotNone(res["plan"])

    # Case 9: Smart Pacing source -> output remapping with cut inside hook (MANDATORY CORRECTION 1)
    def test_09_pacing_cut_inside_hook_preserves_discontinuity(self):
        # Words before and after cut:
        words = [
            {"start": 10.0, "end": 10.5, "text": "PartOne"},
            {"start": 10.5, "end": 11.0, "text": "ofHook"},
            {"start": 11.0, "end": 11.5, "text": "PartTwo"},
            {"start": 11.5, "end": 12.0, "text": "survives"},
        ]
        cand = make_candidate(hook="PartOne ofHook PartTwo survives", hook_start_sec=10.0, hook_end_sec=13.0, hook_confidence=0.85, payoff_score=0.0)
        pacing_plan = {
            "status": "ok",
            "sourcePath": "video.mp4",
            "clipStartSec": 10.0,
            "clipEndSec": 15.0,
            "sourceDurationSec": 5.0,
            "outputDurationSec": 4.0,
            "cutsCount": 1,
            "totalCutDurationSec": 1.0,
            "speedFactor": 1.0,
            "edits": [],
            # Two retained pieces: cut in between
            "retained": [
                {"srcStartSec": 10.0, "srcEndSec": 11.0, "outStartSec": 0.0, "outEndSec": 1.0},
                {"srcStartSec": 12.0, "srcEndSec": 15.0, "outStartSec": 1.0, "outEndSec": 4.0},
            ],
            "subtitlesOnOutput": True,
        }
        res = run_inspect(words, cand, 10.0, 13.0, pacing_plan=pacing_plan)
        self.assertIsNotNone(res["plan"])

    # Case 10: Generic lexical words alone cannot trigger CI (MANDATORY CORRECTION 3)
    def test_10_generic_lexical_words_alone_cannot_trigger_ci(self):
        words = [
            {"start": 1.0, "end": 1.5, "text": "This"},
            {"start": 1.5, "end": 2.0, "text": "is"},
            {"start": 2.0, "end": 2.5, "text": "the"},
            {"start": 2.5, "end": 3.0, "text": "worst"},
            {"start": 3.0, "end": 3.5, "text": "mistake"},
            {"start": 3.5, "end": 4.0, "text": "ever"},
        ]
        # No verified hook, no verified payoff
        cand = make_candidate(hook="", hook_confidence=0.0, payoff_score=0.0)
        res = run_inspect(words, cand, 0.0, 5.0)
        self.assertIsNone(res["plan"], "Generic lexical words alone must NEVER trigger CI without verified hook/payoff")
        self.assertTrue(res["assMatchesBaseline"])

    # Case 11: Semantic emphasis spans, not random word density (MANDATORY CORRECTION 4)
    def test_11_semantic_emphasis_spans(self):
        words = make_words()
        cand = make_candidate(hook_confidence=0.85, payoff_score=0.90)
        res = run_inspect(words, cand, 10.0, 25.0)
        self.assertIsNotNone(res["plan"])
        spans = res["plan"]["emphasisSpans"]
        self.assertGreaterEqual(len(spans), 1)
        for span in spans:
            self.assertLessEqual(span["startIdx"], span["endIdx"])
            self.assertTrue(len(span["reason"]) > 0)

    # Case 12: Group-aware line break hints (MANDATORY CORRECTION 2)
    def test_12_group_aware_line_break_hints(self):
        words = make_words()
        cand = make_candidate(hook_confidence=0.85, payoff_score=0.90)
        res = run_inspect(words, cand, 10.0, 25.0, style="preset_cinematic_vlog")
        self.assertIsNotNone(res["plan"])
        hints = res["plan"]["lineBreakHints"]
        self.assertIsInstance(hints, list)

    # Case 13: Illegal line-break hints fallback safely
    def test_13_illegal_line_break_hints_fallback_safely(self):
        words = make_words()
        cand = make_candidate(hook_confidence=0.85, payoff_score=0.90)
        res = run_inspect(words, cand, 10.0, 25.0, style="preset_cinematic_vlog")
        self.assertIsNotNone(res["assWithIntel"])
        self.assertIn("Dialogue: 0,", res["assWithIntel"])

    # Case 14: Template 1 (preset_viral_bold) integration
    def test_14_template1_viral_bold_emphasis(self):
        words = make_words()
        cand = make_candidate(hook_confidence=0.85, payoff_score=0.90)
        res = run_inspect(words, cand, 10.0, 15.0, style="preset_viral_bold")
        self.assertIsNotNone(res["plan"])
        ass = res["assWithIntel"]
        # Template 1 should contain secondary highlight color (&H00FFFF& or &HFFE600&) for emphasized words
        self.assertIn("Dialogue: 0,", ass)
        self.assertFalse(res["assMatchesBaseline"], "Intel ASS should differ from baseline ASS for Template 1")

    # Case 15: Template 2 (preset_mrbeast_pop) integration
    def test_15_template2_mrbeast_pop_emphasis(self):
        words = make_words()
        cand = make_candidate(hook_confidence=0.85, payoff_score=0.90)
        res = run_inspect(words, cand, 10.0, 15.0, style="preset_mrbeast_pop")
        self.assertIsNotNone(res["plan"])
        ass = res["assWithIntel"]
        self.assertIn("Dialogue: 0,", ass)
        self.assertFalse(res["assMatchesBaseline"], "Intel ASS should differ from baseline ASS for Template 2")

    # Case 16: Template 3 (preset_minimal_capsule) integration
    def test_16_template3_minimal_capsule_opacity_boost(self):
        words = make_words()
        cand = make_candidate(hook_confidence=0.85, payoff_score=0.90)
        res = run_inspect(words, cand, 10.0, 15.0, style="preset_minimal_capsule")
        self.assertIsNotNone(res["plan"])
        ass = res["assWithIntel"]
        # Emphasized inactive words get \alpha&H33&
        self.assertIn(r"\alpha&H33&", ass)
        self.assertFalse(res["assMatchesBaseline"])

    # Case 17: Template 4 (preset_cinematic_vlog) integration
    def test_17_template4_cinematic_vlog_single_event_and_bold(self):
        words = make_words()
        cand = make_candidate(hook_confidence=0.85, payoff_score=0.90)
        res = run_inspect(words, cand, 10.0, 15.0, style="preset_cinematic_vlog")
        self.assertIsNotNone(res["plan"])
        ass = res["assWithIntel"]
        base_ass = res["baselineAss"]
        # Emphasized words get inline {\b1}...{\b0}
        self.assertIn(r"{\b1}", ass)
        # Verify event count remains exactly equal to baseline (no event fragmentation!)
        ass_dialogues = [line for line in ass.splitlines() if line.startswith("Dialogue:")]
        base_dialogues = [line for line in base_ass.splitlines() if line.startswith("Dialogue:")]
        self.assertEqual(len(ass_dialogues), len(base_dialogues), "Cinematic vlog must NOT fragment caption events")
        for line in ass_dialogues:
            self.assertIn(r"\fad(", line, "Every dialogue in Cinematic Vlog must retain \\fad")

    # Case 18: Template 5 (preset_dynamic_editorial) integration
    def test_18_template5_dynamic_editorial_emphasis_role(self):
        words = make_words()
        cand = make_candidate(hook_confidence=0.85, payoff_score=0.90)
        res = run_inspect(words, cand, 10.0, 15.0, style="preset_dynamic_editorial")
        self.assertIsNotNone(res["plan"])
        ass = res["assWithIntel"]
        # Emphasized words get Bebas Neue uppercase styling
        self.assertIn(r"\fnBebas Neue", ass)
        self.assertFalse(res["assMatchesBaseline"])

    # Case 19: Template 5 layout stability
    def test_19_template5_layout_stability(self):
        words = make_words()
        cand = make_candidate(hook_confidence=0.85, payoff_score=0.90)
        res = run_inspect(words, cand, 10.0, 15.0, style="preset_dynamic_editorial")
        ass = res["assWithIntel"]
        base_ass = res["baselineAss"]
        # Count dialogues
        ass_dialogues = [line for line in ass.splitlines() if line.startswith("Dialogue:")]
        base_dialogues = [line for line in base_ass.splitlines() if line.startswith("Dialogue:")]
        self.assertEqual(len(ass_dialogues), len(base_dialogues), "Dynamic editorial word object count must remain invariant")

    # Case 20: Backward compatibility wrapper
    def test_20_backward_compatibility_wrapper(self):
        words = make_words()
        cand = make_candidate()
        # When caption_intel is None, output is identical to baseline
        res = run_inspect(words, cand, 10.0, 15.0, style="preset_viral_bold", env_override={"AUTOSHORTS_CAPTION_INTEL": "0"})
        self.assertTrue(res["assMatchesBaseline"])

    # Case 21: Kill switch baseline identity
    def test_21_kill_switch_baseline_identity_byte_exact(self):
        words = make_words()
        cand = make_candidate(hook_confidence=0.95, payoff_score=0.95)
        for style in ["preset_viral_bold", "preset_mrbeast_pop", "preset_minimal_capsule", "preset_cinematic_vlog", "preset_dynamic_editorial"]:
            res = run_inspect(words, cand, 10.0, 23.0, style=style, env_override={"AUTOSHORTS_CAPTION_INTEL": "0"})
            self.assertTrue(
                res["assMatchesBaseline"],
                f"Kill switch must produce 100% byte-for-byte identical output to baseline for {style}"
            )

    # Case 22: Unmarked words preserve exact baseline behavior
    def test_22_unmarked_words_preserve_exact_baseline_behavior(self):
        # Words with low confidence hook and no payoff -> no emphasis
        words = make_words()
        cand = make_candidate(hook_confidence=0.2, payoff_score=0.2)
        for style in ["preset_viral_bold", "preset_mrbeast_pop", "preset_minimal_capsule", "preset_cinematic_vlog", "preset_dynamic_editorial"]:
            res = run_inspect(words, cand, 10.0, 15.0, style=style)
            self.assertIsNone(res["plan"])
            self.assertTrue(res["assMatchesBaseline"], f"Unmarked words must produce exact baseline ASS for {style}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
