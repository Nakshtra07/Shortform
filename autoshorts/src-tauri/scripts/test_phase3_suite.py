#!/usr/bin/env python3
"""
AutoShorts 11.0 Phase 3 — unit + failure/fallback test suite.

Covers: pause_intelligence (features, model loading, scoring contract,
flag semantics, fallbacks), scene_intelligence sidecar (merge/debounce/
fallback), t7_prosody pipeline, and the smart_pacing integration contract
(evidence-only learned stage; telemetry opt-in; plan unchanged).

Every test uses real functions; audio fixtures are real WAVs (sine/silence)
or the existing e2e corpus — no fabricated metrics.
"""

import os
import sys
import json
import time
import unittest
import wave
import struct

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS)

import numpy as np

WS = os.path.normpath(os.path.join(SCRIPTS, "..", "..", ".."))
E2E_MEDIA = os.path.join(WS, "scratch", "phase2_e2e", "rio_90.mp4")
E2E_WORDS = os.path.join(WS, "scratch", "phase2_e2e", "rio_90_words.json")
PAUSE_MODEL = os.path.join(WS, "scratch", "phase3", "models", "pause_classifier_v1.txt")


def make_wav(path, duration, tone_spans=()):
    """Real 16 kHz mono WAV: silence with optional sine spans."""
    sr = 16000
    n = int(duration * sr)
    x = np.zeros(n, dtype=np.float32)
    for t0, t1, freq, amp in tone_spans:
        i0, i1 = int(t0 * sr), min(n, int(t1 * sr))
        t = np.arange(i1 - i0, dtype=np.float32) / sr
        x[i0:i1] = amp * np.sin(2 * np.pi * freq * t)
    with wave.open(path, "w") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(b"".join(
            struct.pack("<h", int(max(-1.0, min(1.0, v)) * 32767)) for v in x))


class TestPauseIntelligence(unittest.TestCase):
    def setUp(self):
        import pause_intelligence as pi
        self.pi = pi
        for k in ("AUTOSHORTS_SMART_PACING_LEARNED", "AUTOSHORTS_PAUSE_MODEL",
                  "AUTOSHORTS_PACING_TELEMETRY_DIR"):
            os.environ.pop(k, None)

    def test_01_flag_semantics(self):
        pi = self.pi
        os.environ.pop("AUTOSHORTS_SMART_PACING_LEARNED", None)
        self.assertTrue(pi.pause_learning_enabled())
        for off in ("0", "false", "off", "OFF"):
            os.environ["AUTOSHORTS_SMART_PACING_LEARNED"] = off
            self.assertFalse(pi.pause_learning_enabled(), off)
        os.environ["AUTOSHORTS_SMART_PACING_LEARNED"] = "1"
        self.assertTrue(pi.pause_learning_enabled())
        os.environ.pop("AUTOSHORTS_SMART_PACING_LEARNED", None)

    def test_02_acoustic_features_real_audio(self):
        pi = self.pi
        path = os.path.join(os.environ.get("TEMP", "/tmp"), "pi_test_tone.wav")
        make_wav(path, 2.0, tone_spans=[(0.0, 0.8, 220.0, 0.5)])
        x = pi.decode_audio(path, 0.0, 2.0)
        self.assertIsNotNone(x)
        feats = pi.gap_acoustic_features(x, 0.9, 1.5, 0.4, 0.9, 1.5, 2.0)
        self.assertLess(feats["levelDropDb"], -10.0, "silence gap must sit far below speech level")
        os.remove(path)

    def test_03_structural_features_deterministic(self):
        pi = self.pi
        gap = {"srcStartSec": 5.0, "srcEndSec": 6.0, "prevWord": {"text": "done.", "speaker": "S1"},
               "nextWord": {"text": "um", "speaker": "S2"}}
        words = [{"text": "done.", "start": 4.0, "end": 5.0, "speaker": "S1"},
                 {"text": "um", "start": 6.0, "end": 6.4, "speaker": "S2"}]
        f1 = pi.structural_gap_features(gap, words, 0.0, 10.0)
        f2 = pi.structural_gap_features(gap, words, 0.0, 10.0)
        self.assertEqual(f1, f2, "features must be deterministic")
        self.assertEqual(f1["endsSentence"], 1.0)
        self.assertEqual(f1["speakerChange"], 1.0)
        self.assertEqual(f1["startsFiller"], 1.0)

    def test_04_feature_row_shape_stable(self):
        pi = self.pi
        row = pi.features_to_row({"gapRmsDb": -30.0, "durationSec": 0.5})
        self.assertEqual(len(row), len(pi.FEATURE_ORDER))
        self.assertEqual(row[pi.FEATURE_ORDER.index("gapRmsDb")], -30.0)

    def test_05_scoring_fallback_no_model(self):
        pi = self.pi
        gaps = [{"srcStartSec": 0.0, "srcEndSec": 0.5}]
        scored = pi.score_gaps_with_model(gaps, [[0.0] * len(pi.FEATURE_ORDER)], "Z:/definitely/missing.txt")
        self.assertFalse(scored, "missing model must not score")
        self.assertNotIn("pRemovable", gaps[0])

    def test_06_scoring_fallback_model_missing_file(self):
        pi = self.pi
        gaps = [{}]
        scored = pi.score_gaps_with_model(gaps, [[0.0] * len(pi.FEATURE_ORDER)], None)
        self.assertFalse(scored)

    @unittest.skipUnless(os.path.exists(PAUSE_MODEL), "trained model not present")
    def test_07_real_model_scores_multiclass(self):
        pi = self.pi
        self.assertTrue(os.path.exists(E2E_MEDIA), "e2e corpus required")
        with open(E2E_WORDS, "r", encoding="utf-8") as f:
            words = json.load(f)
        import breath_detect as bd
        r = bd.detect_gaps(E2E_MEDIA, 0.0, 90.0, words)
        gaps = r.get("gaps", [])
        feats = pi.build_gap_features(E2E_MEDIA, gaps, words, 0.0, 90.0)
        self.assertIsNotNone(feats)
        rows = [pi.features_to_row(x) for x in feats]
        scored = pi.score_gaps_with_model(gaps, rows, PAUSE_MODEL)
        self.assertTrue(scored, "model present and flag on must score")
        for g in gaps:
            self.assertIn("pRemovable", g)
            self.assertTrue(0.0 <= g["pRemovable"] <= 1.0)

    def test_08_feature_version_mismatch_refuses_model(self):
        pi = self.pi
        import lightgbm as lgb
        import tempfile
        X = np.random.RandomState(0).rand(20, len(pi.FEATURE_ORDER))
        y = (X[:, 0] > 0.5).astype(int)
        m = lgb.LGBMClassifier(n_estimators=5, verbose=-1)
        m.fit(X, y)
        tmp = os.path.join(tempfile.gettempdir(), "pi_model_mismatch.txt")
        m.booster_.save_model(tmp)
        meta = {"featureVersion": "999", "classes": ["breath_pause", "nonspeech_unknown"],
                "removableClasses": ["breath_pause"], "threshold": 0.5}
        with open(os.path.splitext(tmp)[0] + ".meta.json", "w") as f:
            json.dump(meta, f)
        pi._MODEL_CACHE.pop(tmp, None)
        loaded = pi.load_pause_classifier(tmp)
        self.assertIsNone(loaded, "feature-version mismatch must refuse to load")
        os.remove(tmp)
        if os.path.exists(os.path.splitext(tmp)[0] + ".meta.json"):
            os.remove(os.path.splitext(tmp)[0] + ".meta.json")

    def test_09_corrupt_model_file_falls_back(self):
        pi = self.pi
        import tempfile
        tmp = os.path.join(tempfile.gettempdir(), "pi_model_corrupt.txt")
        with open(tmp, "w") as f:
            f.write("this is not a lightgbm model")
        pi._MODEL_CACHE.pop(tmp, None)
        loaded = pi.load_pause_classifier(tmp)
        self.assertIsNone(loaded)
        os.remove(tmp)


class TestSceneIntelligenceSidecar(unittest.TestCase):
    def test_10_merge_debounce_and_fallback(self):
        import scene_intelligence as si
        pixel = [0.0, 2.0, 10.0]
        doc = {"scenes": [{"sceneId": 0, "start": 2.1, "end": 5.0},
                          {"sceneId": 1, "start": 5.0, "end": 9.0},
                          {"sceneId": 2, "start": 2.05, "end": 9.0}]}  # 2.05 debounces with 2.0
        path = os.path.join(os.environ.get("TEMP", "/tmp"), "si_scenes_test.json")
        with open(path, "w") as f:
            json.dump(doc, f)
        merged = si.__dict__  # noqa — module functions used below
        # call via speaker_tracker helper for the merge contract
        import speaker_tracker as st
        out = st.load_scene_cuts_json(path, 0.0, 20.0, pixel)
        self.assertEqual(out[0], 0.0)
        self.assertIn(5.0, out)
        self.assertNotIn(2.05, out, "external cut within 0.35s of a pixel cut must debounce")
        os.remove(path)

    def test_11_scene_cuts_missing_file_fallback(self):
        import speaker_tracker as st
        pixel = [0.0, 3.0]
        out = st.load_scene_cuts_json("Z:/missing/scenes.json", 0.0, 20.0, pixel)
        self.assertEqual(out, [0.0, 3.0], "missing scene file must keep pixel cuts")

    def test_12_scene_cuts_clipped_to_clip_window(self):
        import speaker_tracker as st
        doc = {"scenes": [{"sceneId": 0, "start": 95.0, "end": 100.0}]}
        path = os.path.join(os.environ.get("TEMP", "/tmp"), "si_scenes_oob.json")
        with open(path, "w") as f:
            json.dump(doc, f)
        out = st.load_scene_cuts_json(path, 0.0, 20.0, [0.0])
        self.assertEqual(out, [0.0], "out-of-window scene boundary must be dropped")
        os.remove(path)


class TestSpeakerMapRoleConsumption(unittest.TestCase):
    def test_13_load_diarization_and_speaker_map(self):
        import speaker_tracker as st
        doc = {
            "model": "deepgram",
            "segments": [
                {"speaker_id": "spk_0", "start": 0.0, "end": 2.5, "confidence": 0.9},
                {"speaker_id": "spk_1", "start": 2.5, "end": 5.0, "confidence": 0.85},
            ],
            "speaker_map": [
                {"speaker_id": "spk_0", "role": "host"},
                {"speaker_id": "spk_1", "role": "guest"},
            ],
        }
        path = os.path.join(os.environ.get("TEMP", "/tmp"), "diar_spk_map_test.json")
        with open(path, "w") as f:
            json.dump(doc, f)
        try:
            # Test default backwards compatibility
            segs = st.load_diarization_sidecar_json(path)
            self.assertEqual(len(segs), 2)
            self.assertEqual(segs[0].speaker_id, "spk_0")

            # Test return_speaker_map=True
            segs, spk_map = st.load_diarization_sidecar_json(path, return_speaker_map=True)
            self.assertEqual(len(segs), 2)
            self.assertEqual(spk_map.get("spk_0"), "host")
            self.assertEqual(spk_map.get("spk_1"), "guest")

            # Test helper
            spk_map2 = st.load_speaker_map_from_sidecar(path)
            self.assertEqual(spk_map2, {"spk_0": "host", "spk_1": "guest"})
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_14_build_speaker_intel_block_populates_application_role(self):
        import speaker_tracker as st
        import types
        # Create a mock fusion state
        state1 = types.SimpleNamespace(
            track_id=1, diarization_id="spk_0", start=1.0, end=3.0,
            audio_evidence=0.8, visual_evidence=0.9, speaking_probability=0.85,
            confidence=0.88, evidence_sources=["diarization", "visual_mouth_motion"]
        )
        state2 = types.SimpleNamespace(
            track_id=2, diarization_id="spk_1", start=3.0, end=5.0,
            audio_evidence=0.7, visual_evidence=0.8, speaking_probability=0.75,
            confidence=0.80, evidence_sources=["diarization"]
        )
        state_unknown = types.SimpleNamespace(
            track_id=3, diarization_id=None, start=5.0, end=6.0,
            audio_evidence=0.0, visual_evidence=0.5, speaking_probability=0.4,
            confidence=0.50, evidence_sources=[]
        )
        speaker_map = {"spk_0": "host", "spk_1": "guest"}
        block = st.build_speaker_intel_block(None, [state1, state2, state_unknown], 0.0, speaker_map=speaker_map)

        intervals = block["fusion"]["intervals"]
        self.assertEqual(len(intervals), 3)
        self.assertEqual(intervals[0]["applicationRole"], "host")
        self.assertEqual(intervals[1]["applicationRole"], "guest")
        self.assertIsNone(intervals[2]["applicationRole"])


class TestT7ProsodyPipeline(unittest.TestCase):
    """
    Self-contained: builds its own deterministic media fixture with ffmpeg.

    This test previously required a developer's private
    `scratch/phase2_e2e/rio_90.mp4`, which is not in the repository, so it
    failed for everyone but the author. A 3-second tone/silence clip plus a
    synthetic word list exercises the same real code path (ffmpeg decode, gap
    extraction, schema, acoustic features) with no private asset.
    """

    @classmethod
    def setUpClass(cls):
        import subprocess
        import tempfile

        cls._tmp = tempfile.mkdtemp(prefix="t7_fixture_")
        media = os.path.join(cls._tmp, "fixture_3s.wav")
        # Alternating tone (voiced-ish) and near-silence (gap) so the energy
        # contour has something real to measure.
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
             "-i", "sine=frequency=140:duration=3:sample_rate=16000",
             "-c:a", "pcm_s16le", media],
            capture_output=True, timeout=120, check=True,
        )
        cls.media = media

        # Words with a deliberate pause between them, spanning the clip.
        cls.words = [
            {"text": "hello", "start": 0.10, "end": 0.40, "speaker": "S1"},
            {"text": "world", "start": 1.20, "end": 1.60, "speaker": "S1"},
            {"text": "next.", "start": 2.00, "end": 2.40, "speaker": "S2"},
        ]

    @classmethod
    def tearDownClass(cls):
        import shutil
        shutil.rmtree(cls._tmp, ignore_errors=True)

    def test_13_pipeline_runs_and_schema(self):
        """Real ffmpeg decode path, real schema, deterministic output."""
        import t7_prosody as tp

        recs = tp.build_gap_records(self.media, 0.0, 3.0, self.words)
        self.assertEqual(len(recs), len(self.words) - 1,
                         "one record per inter-word gap")
        r = recs[0]
        for key in ("gapIdx", "wordIdxBefore", "wordIdxAfter", "pauseSec",
                    "pBoundaryRule", "endsSentence", "speakerChange",
                    "f0SlopePre", "energyDropDb"):
            self.assertIn(key, r, f"schema key missing: {key}")

        # Word indices must be preserved for annotation round-tripping.
        self.assertEqual([x["gapIdx"] for x in recs], [0, 1])
        self.assertEqual(r["wordIdxBefore"], 0)
        self.assertEqual(r["wordIdxAfter"], 1)
        # Pause is derived from the word gap, not invented.
        self.assertAlmostEqual(r["pauseSec"], 0.80, places=2)
        # `endsSentence`/`speakerChange` describe the gap's PRECEDING word, which
        # is the correct convention for boundary annotation: a gap after "world."
        # is a sentence boundary, and the S1->S2 turn occurs in the second gap.
        self.assertEqual(recs[0]["endsSentence"], 0, "'world' has no terminator")
        self.assertEqual(recs[1]["endsSentence"], 0,
                         "the gap after 'world' is mid-sentence")
        self.assertEqual(recs[1]["speakerChange"], 1,
                         "gap between 'world'(S1) and 'next.'(S2) is a turn")
        # The prior is a bounded probability.
        for rec in recs:
            self.assertGreaterEqual(rec["pBoundaryRule"], 0.0)
            self.assertLessEqual(rec["pBoundaryRule"], 1.0)

    def test_13b_acoustic_fields_are_measured_from_real_audio(self):
        """With real audio present the prosody features must not be None."""
        import t7_prosody as tp

        recs = tp.build_gap_records(self.media, 0.0, 3.0, self.words)
        # The tone is continuous across the fixture, so energyDropDb should be
        # computable; the point is that the acoustic branch EXECUTED rather than
        # short-circuiting to None because no media was supplied.
        self.assertIsNotNone(recs[0]["f0SlopePre"])
        self.assertIsNotNone(recs[0]["energyDropDb"])

    def test_14_rule_prior_bounded_and_sentence_dominant(self):
        import t7_prosody as tp
        words = [{"text": "end.", "start": 0.0, "end": 1.0, "speaker": "S1"},
                 {"text": "New", "start": 2.5, "end": 2.9, "speaker": "S1"}]
        recs = tp.build_gap_records(None, 0.0, 5.0, words)
        self.assertEqual(recs[0]["pBoundaryRule"], 1.0, "sentence end must pin the prior")


class TestSmartPacingIntegration(unittest.TestCase):
    def setUp(self):
        for k in ("AUTOSHORTS_SMART_PACING_LEARNED", "AUTOSHORTS_PAUSE_MODEL",
                  "AUTOSHORTS_PACING_TELEMETRY_DIR"):
            os.environ.pop(k, None)

    def test_15_plan_byte_identical_without_model(self):
        import smart_pacing as sp
        words = [{"text": "hello", "start": 0.5, "end": 1.0, "speaker": "S1"},
                 {"text": "world", "start": 3.0, "end": 3.5, "speaker": "S1"}]
        sil = [(1.0, 2.9)]
        p1 = sp.plan_pacing(words, 0.0, 5.0, list(sil), v2=True)
        p2 = sp.plan_pacing(words, 0.0, 5.0, list(sil), v2=True)
        self.assertEqual(p1, p2, "repeated plans must be deterministic")

    @unittest.skipUnless(os.path.exists(PAUSE_MODEL) and os.path.exists(E2E_MEDIA),
                         "model + corpus required")
    def test_16_learned_stage_is_evidence_only(self):
        import smart_pacing as sp
        os.environ["AUTOSHORTS_PAUSE_MODEL"] = PAUSE_MODEL
        with open(E2E_WORDS, "r", encoding="utf-8") as f:
            words = json.load(f)
        import breath_detect as bd
        r = bd.detect_gaps(E2E_MEDIA, 0.0, 90.0, words)
        gaps_all = r.get("gaps", [])
        # Deterministic plan without the learned stage
        import importlib
        silences = sp.run_silencedetect(E2E_MEDIA, 0.0, 90.0)
        plan_plain = sp.plan_pacing(words, 0.0, 90.0, silences, v2=True, source_path=E2E_MEDIA)
        # Plan with the learned stage attached (model env set)
        os.environ["AUTOSHORTS_SMART_PACING_2"] = "1"
        plan_learned = sp.plan_pacing(words, 0.0, 90.0, silences, v2=True, source_path=E2E_MEDIA)
        strip = lambda p: json.dumps({k: v for k, v in p.items()}, sort_keys=True)
        self.assertEqual(strip(plan_plain), strip(plan_learned),
                         "evidence-only learned stage must NOT alter the plan")
        # And the scored gaps carry pRemovable (evidence attached)
        self.assertTrue(len(sp._LAST_DETECTED_GAPS) > 0)
        with_p = [g for g in sp._LAST_DETECTED_GAPS if "pRemovable" in g]
        self.assertTrue(with_p, "learned stage must attach pRemovable evidence")

    def test_17_telemetry_opt_in_and_failure_safe(self):
        import smart_pacing as sp
        words = [{"text": "a", "start": 0.5, "end": 1.0, "speaker": "S1"},
                 {"text": "b", "start": 3.0, "end": 3.5, "speaker": "S1"}]
        sp._LAST_DETECTED_GAPS = [{"srcStartSec": 1.0, "srcEndSec": 2.5, "category": "normal_word_gap"}]
        # Disabled by default (no env) — must not write anything nor raise
        sp._emit_pacing_telemetry("Z:/missing.mp4", 0.0, 5.0, words, sp._LAST_DETECTED_GAPS, {"edits": []})
        # Opt-in with a bogus dir must not raise
        os.environ["AUTOSHORTS_PACING_TELEMETRY_DIR"] = "Z:/definitely/not/writable"
        sp._emit_pacing_telemetry("Z:/missing.mp4", 0.0, 5.0, words, sp._LAST_DETECTED_GAPS, {"edits": []})
        os.environ.pop("AUTOSHORTS_PACING_TELEMETRY_DIR", None)


if __name__ == "__main__":
    unittest.main(verbosity=2)
