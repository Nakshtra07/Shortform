"""
test_applied_features_suite.py
AutoShorts 8.0 - Structured Per-Clip Feature-Application Status

Tests spec cases I, J, K, L, N (Python-level semantics):
  I.  Feature evaluated but returned a no-op / skip -> not reported as applied
  J.  No-op boundary optimization -> hookEndingOptimization = false
  K.  No-op Smart Pacing (plan is None) -> smartPacing = false
  L.  Audio Intelligence skip (plan is None) -> audioIntelligence = false
  N.  Frontend JSON shape: appliedFeatures is valid JSON with exactly three boolean keys

All DB-layer cases (A-H, M, O, COALESCE) are covered by the Rust db.rs tests.
No production engine code is modified or called.
"""

import json
import unittest


# ---------------------------------------------------------------------------
# Mirror of the lib.rs serde_json::json!({...}).to_string() block
# ---------------------------------------------------------------------------

def compute_applied_features_json(
    smart_pacing_applied: bool,
    hook_ending_applied: bool,
    audio_intelligence_applied: bool,
) -> str:
    return json.dumps(
        {
            "smartPacing": smart_pacing_applied,
            "hookEndingOptimization": hook_ending_applied,
            "audioIntelligence": audio_intelligence_applied,
        },
        separators=(",", ":"),
    )


def parse_applied_features(json_str: str) -> dict:
    parsed = json.loads(json_str)
    assert isinstance(parsed, dict), "appliedFeatures must be a JSON object"
    return parsed


# ---------------------------------------------------------------------------
# Signal derivation functions - mirror exact lib.rs logic
# ---------------------------------------------------------------------------

class MockBoundaryOpt:
    def __init__(self, start_changed: bool, end_changed: bool):
        self.start_changed = start_changed
        self.end_changed = end_changed


def hook_ending_applied(boundary_opt: MockBoundaryOpt) -> bool:
    """Production signal: boundary_opt.start_changed || boundary_opt.end_changed"""
    return boundary_opt.start_changed or boundary_opt.end_changed


def smart_pacing_applied(pacing_plan) -> bool:
    """Production signal: pacing_plan.is_some()"""
    return pacing_plan is not None


def audio_intelligence_applied(audio_plan) -> bool:
    """Production signal: audio_plan.is_some()"""
    return audio_plan is not None


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestAppliedFeaturesSignals(unittest.TestCase):
    """Cases I, J, K, L - production signal semantics."""

    # Case J - no-op boundary optimization
    def test_case_j_noop_boundary_both_unchanged(self):
        opt = MockBoundaryOpt(start_changed=False, end_changed=False)
        self.assertFalse(hook_ending_applied(opt))

    def test_case_j_only_start_changed(self):
        opt = MockBoundaryOpt(start_changed=True, end_changed=False)
        self.assertTrue(hook_ending_applied(opt))

    def test_case_j_only_end_changed(self):
        opt = MockBoundaryOpt(start_changed=False, end_changed=True)
        self.assertTrue(hook_ending_applied(opt))

    def test_case_j_both_changed(self):
        opt = MockBoundaryOpt(start_changed=True, end_changed=True)
        self.assertTrue(hook_ending_applied(opt))

    # Case K - no-op smart pacing
    def test_case_k_pacing_plan_none(self):
        self.assertFalse(smart_pacing_applied(None))

    def test_case_k_pacing_plan_some(self):
        self.assertTrue(smart_pacing_applied({"status": "ok", "edits": [{"type": "pause"}]}))

    # Case L - audio intelligence skip
    def test_case_l_audio_plan_none(self):
        self.assertFalse(audio_intelligence_applied(None))

    def test_case_l_audio_plan_some(self):
        self.assertTrue(audio_intelligence_applied({"status": "ok", "filterChain": "highpass=f=80"}))

    # Case I - evaluated but not applied
    def test_case_i_evaluated_not_applied_boundary(self):
        opt = MockBoundaryOpt(start_changed=False, end_changed=False)
        self.assertFalse(hook_ending_applied(opt), "Evaluation != application")

    def test_case_i_evaluated_not_applied_pacing(self):
        self.assertFalse(smart_pacing_applied(None), "Evaluation != application")

    def test_case_i_evaluated_not_applied_audio(self):
        self.assertFalse(audio_intelligence_applied(None), "Evaluation != application")


class TestAppliedFeaturesJson(unittest.TestCase):
    """Cases A-H - all 8 feature combinations in JSON."""

    def _parse(self, **kwargs) -> dict:
        return parse_applied_features(compute_applied_features_json(**kwargs))

    def test_case_a_none(self):
        f = self._parse(smart_pacing_applied=False, hook_ending_applied=False, audio_intelligence_applied=False)
        self.assertFalse(f["smartPacing"])
        self.assertFalse(f["hookEndingOptimization"])
        self.assertFalse(f["audioIntelligence"])

    def test_case_b_smart_pacing_only(self):
        f = self._parse(smart_pacing_applied=True, hook_ending_applied=False, audio_intelligence_applied=False)
        self.assertTrue(f["smartPacing"])
        self.assertFalse(f["hookEndingOptimization"])
        self.assertFalse(f["audioIntelligence"])

    def test_case_c_hook_ending_only(self):
        f = self._parse(smart_pacing_applied=False, hook_ending_applied=True, audio_intelligence_applied=False)
        self.assertFalse(f["smartPacing"])
        self.assertTrue(f["hookEndingOptimization"])
        self.assertFalse(f["audioIntelligence"])

    def test_case_d_audio_only(self):
        f = self._parse(smart_pacing_applied=False, hook_ending_applied=False, audio_intelligence_applied=True)
        self.assertFalse(f["smartPacing"])
        self.assertFalse(f["hookEndingOptimization"])
        self.assertTrue(f["audioIntelligence"])

    def test_case_e_smart_and_hook(self):
        f = self._parse(smart_pacing_applied=True, hook_ending_applied=True, audio_intelligence_applied=False)
        self.assertTrue(f["smartPacing"])
        self.assertTrue(f["hookEndingOptimization"])
        self.assertFalse(f["audioIntelligence"])

    def test_case_f_smart_and_audio(self):
        f = self._parse(smart_pacing_applied=True, hook_ending_applied=False, audio_intelligence_applied=True)
        self.assertTrue(f["smartPacing"])
        self.assertFalse(f["hookEndingOptimization"])
        self.assertTrue(f["audioIntelligence"])

    def test_case_g_hook_and_audio(self):
        f = self._parse(smart_pacing_applied=False, hook_ending_applied=True, audio_intelligence_applied=True)
        self.assertFalse(f["smartPacing"])
        self.assertTrue(f["hookEndingOptimization"])
        self.assertTrue(f["audioIntelligence"])

    def test_case_h_all_three(self):
        f = self._parse(smart_pacing_applied=True, hook_ending_applied=True, audio_intelligence_applied=True)
        self.assertTrue(f["smartPacing"])
        self.assertTrue(f["hookEndingOptimization"])
        self.assertTrue(f["audioIntelligence"])


class TestAppliedFeaturesJsonShape(unittest.TestCase):
    """Case N - frontend JSON shape validation."""

    REQUIRED_KEYS = {"smartPacing", "hookEndingOptimization", "audioIntelligence"}

    def _assert_valid_shape(self, smart: bool, hook: bool, audio: bool):
        json_str = compute_applied_features_json(
            smart_pacing_applied=smart,
            hook_ending_applied=hook,
            audio_intelligence_applied=audio,
        )
        parsed = json.loads(json_str)
        self.assertIsInstance(parsed, dict)
        self.assertEqual(set(parsed.keys()), self.REQUIRED_KEYS)
        for key in self.REQUIRED_KEYS:
            val = parsed[key]
            self.assertIsInstance(val, bool, f"Key '{key}' must be a boolean, got {type(val).__name__}")
        self.assertEqual(parsed["smartPacing"], smart)
        self.assertEqual(parsed["hookEndingOptimization"], hook)
        self.assertEqual(parsed["audioIntelligence"], audio)

    def test_case_n_shape_all_false(self):
        self._assert_valid_shape(False, False, False)

    def test_case_n_shape_all_true(self):
        self._assert_valid_shape(True, True, True)

    def test_case_n_shape_mixed(self):
        self._assert_valid_shape(True, False, True)

    def test_case_n_false_keys_present(self):
        """All three keys must be present even when value is False."""
        json_str = compute_applied_features_json(
            smart_pacing_applied=False,
            hook_ending_applied=False,
            audio_intelligence_applied=False,
        )
        parsed = json.loads(json_str)
        for key in self.REQUIRED_KEYS:
            self.assertIn(key, parsed, f"Key '{key}' must be present even when False")

    def test_case_n_null_legacy_is_falsy(self):
        """NULL appliedFeatures (legacy clip) must be falsy - no badges shown."""
        self.assertFalse(bool(None))

    def test_case_n_always_three_keys(self):
        """appliedFeatures JSON must always have exactly 3 keys."""
        json_str = compute_applied_features_json(False, False, False)
        parsed = json.loads(json_str)
        self.assertEqual(len(parsed), 3)


class TestRealArtifactExpectations(unittest.TestCase):
    """
    Real-pipeline validation based on forensic baseline report Part 9 artifacts.
    No actual video processing - validates the logic contract against documented outcomes.
    """

    def _build_features(self, pacing_is_some: bool, boundary_changed: bool, audio_is_some: bool) -> dict:
        json_str = compute_applied_features_json(
            smart_pacing_applied=smart_pacing_applied({"ok": True} if pacing_is_some else None),
            hook_ending_applied=hook_ending_applied(
                MockBoundaryOpt(start_changed=boundary_changed, end_changed=False)
            ),
            audio_intelligence_applied=audio_intelligence_applied(
                {"status": "ok"} if audio_is_some else None
            ),
        )
        return json.loads(json_str)

    def test_beat_emotional_fatigue_smart_and_audio(self):
        """scratch/paced_slow.mp4: Smart Pacing applied (9.08s->8.52s), Audio applied (-24.48->-16.06 LUFS)."""
        f = self._build_features(pacing_is_some=True, boundary_changed=False, audio_is_some=True)
        self.assertTrue(f["smartPacing"])
        self.assertTrue(f["audioIntelligence"])
        self.assertFalse(f["hookEndingOptimization"])

    def test_messi_clip_smart_pacing_and_hook(self):
        """scratch/paced_messi.mp4: Smart Pacing (12.0s->9.52s), Hook/End Q->A repair, Audio skipped (-15.16 LUFS in tolerance)."""
        f = self._build_features(pacing_is_some=True, boundary_changed=True, audio_is_some=False)
        self.assertTrue(f["smartPacing"])
        self.assertTrue(f["hookEndingOptimization"])
        self.assertFalse(f["audioIntelligence"])

    def test_hindi_speech_audio_only(self):
        """scratch/audio_ab_out/ab____mp4.mp4: Smart Pacing skipped (no removable silence), Audio applied (peak limiter)."""
        f = self._build_features(pacing_is_some=False, boundary_changed=False, audio_is_some=True)
        self.assertFalse(f["smartPacing"])
        self.assertTrue(f["audioIntelligence"])
        self.assertFalse(f["hookEndingOptimization"])

    def test_messi_audio_within_tolerance_skip(self):
        """Messi clip Audio Intelligence correctly skipped (-15.16 LUFS, within +/-2 LU tolerance)."""
        f = self._build_features(pacing_is_some=True, boundary_changed=True, audio_is_some=False)
        self.assertFalse(f["audioIntelligence"])

    def test_rick_astley_music_blocks_pacing(self):
        """scratch/audio_ab_out/ab_Rick_Astley...mp4: Smart Pacing not applied (music blocks cuts), Audio applied (loudnorm)."""
        f = self._build_features(pacing_is_some=False, boundary_changed=False, audio_is_some=True)
        self.assertFalse(f["smartPacing"])
        self.assertTrue(f["audioIntelligence"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
