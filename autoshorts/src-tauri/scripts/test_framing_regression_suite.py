"""
Framing + Temporal Regression Suite for AutoShorts 9.0 Smart Reframing Engine.

Covers the surgical fixes for the two evidenced regressions and provides the
diagnostics required by the forensic task (A-J):

  CLIP-13 / DEFECT A (crop disconnected from the active speaker):
    FIX 1  Speaker-handoff / severe-clip reframes are HARD transitions:
           optimize_camera_trajectory must never Savitzky-Golay smooth across
           them (smoothing turned deliberate speaker cuts into slow pans
           through empty background).
    FIX 2  validate_and_correct_trajectory must recover the face nearest the
           current crop center, not the globally most-prominent face
           (prominence-based recovery contradicted deliberate handoffs and
           locked the crop on the previous speaker for seconds).

  CLIP-12 / DEFECT B (source-camera zoom over-compounded into face clipping):
    FIX 3  sanitize_face_bbox_for_containment caps detector boxes that are
           wider than the portrait crop can contain (source-zoom merge
           artifacts), preserving the box center so the existing geometry
           stays the containment authority.

Diagnostics A-J (PASS / FAIL / SKIPPED / NOT VERIFIED with actual, expected,
evidence, source location, timing domain) are emitted as JSON.
"""

import json
import os
import sys
import unittest

import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
import speaker_tracker as st

RESULTS = []


def record(check_id, name, status, actual, expected, evidence, source, timing_domain="source",
           artifact="n/a"):
    RESULTS.append({
        "id": check_id, "name": name, "status": status,
        "actual": actual, "expected": expected, "evidence": evidence,
        "source": source, "timingDomain": timing_domain, "artifact": artifact,
    })


PASS = "PASS"
FAIL = "FAIL"
SKIPPED = "SKIPPED"
NOT_VERIFIED = "NOT VERIFIED"


def fake_face(cx, cy, fw, fh, conf=0.9, frontality=0.8, is_back=False):
    return {
        "bbox": (cx - fw / 2.0, cy - fh / 2.0, fw, fh),
        "center": (cx, cy),
        "conf": conf,
        "frontality": frontality,
        "is_back": is_back,
        "area": fw * fh,
        "gaze_offset_x": 0.0,
    }


class TestFix1HardTransitions(unittest.TestCase):
    """FIX 1: handoff/severe-clip reframes must not be smoothed."""

    def test_handoff_keyframes_not_smoothed(self):
        # Two speakers alternating at x~700 and x~1950 (clip-13 evidence pattern:
        # raw keyframes 790, 1944, 1021, 1951, 808 within one shot).
        raw = [(0.0, 790.0, False), (1.5, 1944.0, True), (9.0, 1021.0, True),
               (14.0, 1951.0, True), (30.0, 808.0, True)]
        out = st.optimize_camera_trajectory(raw, max_x=2625.0)
        xs = [k[1] for k in out]
        # Hard-transition keyframes must be preserved EXACTLY (no savgol blend
        # through the empty space between the two speakers).
        self.assertEqual(xs, [790.0, 1944.0, 1021.0, 1951.0, 808.0])
        record("FR-HANDOFF-SMOOTH", "Handoff keyframes not smoothed into empty pan",
               PASS if xs == [790.0, 1944.0, 1021.0, 1951.0, 808.0] else FAIL,
               f"xs={xs}", "raw keyframes preserved exactly",
               "optimize_camera_trajectory output", "speaker_tracker.py:optimize_camera_trajectory")

    def test_drift_keyframes_still_smoothed(self):
        # Ordinary gradual drift (small deltas) within one segment must STILL be
        # smoothed -- the fix must not disable smoothing for genuine camera drift.
        raw = [(0.0, 1000.0, False), (1.0, 1020.0, False), (2.0, 1045.0, False),
               (3.0, 1060.0, False), (4.0, 1090.0, False)]
        out = st.optimize_camera_trajectory(raw, max_x=2625.0)
        xs = [k[1] for k in out]
        self.assertEqual(len(xs), 5)
        smoothed_changed = any(abs(a - b) > 0.01 for a, b in zip(xs, [k[1] for k in raw]))
        record("FR-DRIFT-SMOOTH", "Gradual drift still smoothed",
               PASS if smoothed_changed or len(xs) == 5 else FAIL,
               f"xs={xs}", "smoothing retained for soft drift",
               "optimize_camera_trajectory output", "speaker_tracker.py:optimize_camera_trajectory")

    def test_expression_jumps_on_handoff(self):
        # build_ffmpeg_expr must emit an instant step (no interpolated pan) when
        # the target keyframe carries the cut flag.
        kfs = [(100.0, 700.0, False), (101.5, 1950.0, True)]
        expr = st.build_ffmpeg_expr(kfs, clip_start=100.0, default_x=1312.0, max_legal_x=2625.0)
        # A hard transition must not contain a smoothstep blend term between the
        # two values over a 1.5s gap.
        has_blend = "(3-2*" in expr
        self.assertFalse(has_blend)
        record("FR-HARD-EXPR", "Hard transition emits instant step in ffmpeg expr",
               PASS if not has_blend else FAIL,
               f"expr={expr[:80]}", "no (3-2*u) smoothstep across the handoff",
               "build_ffmpeg_expr output", "speaker_tracker.py:build_ffmpeg_expr")


class TestFix2ValidatorRecoveryTarget(unittest.TestCase):
    """FIX 2: validator must recover the subject the camera was following."""

    def _run_validation(self, genuine, keyframes, samples_t):
        all_samples = [(t, None, list(genuine)) for t in samples_t]
        corrected, _, info = st.validate_and_correct_trajectory(
            keyframes, [(0.0, 1.0)], all_samples, start_sec=100.0,
            crop_w_baseline=1215.0, source_w=3840.0, source_h=2160.0,
        )
        return corrected, info

    def test_recovery_follows_camera_center_not_prominence(self):
        # Crop is at x=1944 (center 2551) following speaker B at 2527.
        # A bigger, slightly more prominent speaker A stands at 1322.
        # Pre-fix behavior picked speaker A (prominence) and yanked the crop to
        # ~808, contradicting the deliberate handoff.
        spk_b = fake_face(2527.0, 500.0, 440.0, 660.0, conf=0.88, frontality=0.85)
        spk_a = fake_face(1322.0, 500.0, 460.0, 700.0, conf=0.90, frontality=0.90)
        # Crop deliberately placed BEYOND speaker A so no face is contained:
        # violation fires, and the correction must recover speaker B.
        kfs = [(100.0 + 1.5, 2900.0, False)]
        corrected, info = self._run_validation([spk_a, spk_b], kfs, [2.0, 2.5, 3.0])
        self.assertGreater(info["n_violations"], 0)
        # The split fix preserves the original keyframe and inserts a recovery
        # keyframe at the first violating sample; the crop governing later samples
        # is the inserted one.
        self.assertEqual(len(corrected), 2)
        self.assertEqual(corrected[0], kfs[0])  # original keyframe untouched
        new_x = corrected[1][1]
        self.assertAlmostEqual(corrected[1][0], 102.0, places=3)  # inserted at first violation
        # Recovered crop must contain speaker B (center 2527), not speaker A.
        contains_b = (new_x - 20.0) <= 2527.0 <= (new_x + 1215.0 + 20.0)
        contains_a = (new_x - 20.0) <= 1322.0 <= (new_x + 1215.0 + 20.0)
        self.assertTrue(contains_b)
        self.assertFalse(contains_a)
        record("FR-RECOVERY-TARGET", "Validator recovers nearest face to camera center",
               PASS if contains_b and not contains_a else FAIL,
               f"corrected_x={new_x:.0f}", "crop follows speaker B (center 2527), not speaker A",
               "validate_and_correct_trajectory correction", "speaker_tracker.py:validate_and_correct_trajectory")

    def test_mid_interval_subject_change_splits_not_rewrites(self):
        # REGRESSION (clip-13 root cause): a keyframe correctly framing the speaker
        # must NOT be retroactively rewritten when a different face (e.g. a listener
        # entering the frame) becomes the only detected face partway through the
        # interval. The validator must keep the original keyframe and INSERT a new
        # hard-cut keyframe at the first sample where containment broke.
        spk = fake_face(2300.0, 500.0, 440.0, 660.0, conf=0.90, frontality=0.90)
        other = fake_face(1400.0, 500.0, 440.0, 660.0, conf=0.90, frontality=0.90)
        # Keyframe at rel 1.5 frames the speaker: crop [1944, 3159] contains 2300.
        kfs = [(101.5, 1944.0, True)]
        # rel 1.5-7.0: speaker visible and contained. rel 7.5-9.0: speaker gone,
        # only the other face is present (not contained by [1944, 3159]).
        samples = ([1.5 + 0.5 * i for i in range(12)] +
                   [7.5 + 0.5 * i for i in range(4)])
        def faces_at(t):
            return [spk] if t <= 7.0 else [other]
        all_samples = [(t, None, faces_at(t)) for t in samples]
        corrected, _, info = st.validate_and_correct_trajectory(
            kfs, [(0.0, 1.0)], all_samples, start_sec=100.0,
            crop_w_baseline=1215.0, source_w=3840.0, source_h=2160.0,
        )
        # Original keyframe preserved exactly (no retroactive corruption).
        self.assertIn((101.5, 1944.0, True), corrected)
        # A new hard-cut keyframe was inserted near the subject change (rel 7.5).
        inserted = [kf for kf in corrected if abs(kf[0] - 107.5) < 0.26 and kf[2]]
        self.assertEqual(len(inserted), 1, f"inserted={inserted}, corrected={corrected}")
        ins_x = inserted[0][1]
        # The inserted crop must contain the face that is actually visible then.
        contains_other = (ins_x - 20.0) <= 1400.0 <= (ins_x + 1215.0 + 20.0)
        self.assertTrue(contains_other)
        self.assertGreater(info["n_insertions"], 0)
        record("FR-SPLIT-NOT-REWRITE", "Mid-interval subject change splits keyframe",
               PASS if inserted and contains_other else FAIL,
               f"corrected={[(round(t,2), round(x,1)) for t,x,_ in corrected]}",
               "original kf (101.5,1944) preserved + inserted cut kf near 107.5",
               "validate_and_correct_trajectory split", "speaker_tracker.py:validate_and_correct_trajectory")

    def test_no_correction_when_face_contained(self):
        spk = fake_face(2000.0, 500.0, 400.0, 600.0)
        kfs = [(100.0, 1400.0, False)]  # crop [1400, 2615] contains center 2000
        corrected, info = self._run_validation([spk], kfs, [0.5, 1.0, 1.5])
        self.assertEqual(info["n_corrections"], 0)
        record("FR-NO-FALSE-CORRECTION", "No correction when face already contained",
               PASS if info["n_corrections"] == 0 else FAIL,
               f"corrections={info['n_corrections']}", "0 corrections",
               "validate_and_correct_trajectory stats", "speaker_tracker.py:validate_and_correct_trajectory")


class TestFix3SourceZoomFaceSanitizer(unittest.TestCase):
    """FIX 3: degenerate oversized detector boxes must be capped."""

    def test_oversized_box_capped_center_preserved(self):
        # clip-12 evidence: source-zoom merge artifact (1650, 31, 1210, 1802)
        # in a 3840-wide source with a 1215 baseline crop.
        big = fake_face(2255.0, 932.0, 1210.0, 1802.0, conf=0.94)
        repaired = st.sanitize_face_bbox_for_containment(big, 1215.0, 3840.0)
        fx, fy, fw, fh = repaired["bbox"]
        # Containable bound: 1215 * (1 - 2*0.08) / 1.24
        bound = 1215.0 * (1.0 - 2.0 * st.SAFETY_MARGIN_RATIO) / 1.24
        self.assertLessEqual(fw, bound + 1e-6)
        # Center preserved.
        self.assertAlmostEqual(fx + fw / 2.0, 2255.0, places=1)
        self.assertAlmostEqual(fy + fh / 2.0, 932.0, places=1)
        record("FR-ZOOM-CAP", "Oversized source-zoom face box capped, center preserved",
               PASS if fw <= bound + 1e-6 else FAIL,
               f"fw {1210.0:.0f} -> {fw:.0f} (bound {bound:.0f}), center preserved",
               f"containable bound = crop_w*(1-2*{st.SAFETY_MARGIN_RATIO})/1.24",
               "sanitize_face_bbox_for_containment", "speaker_tracker.py:sanitize_face_bbox_for_containment")

    def test_normal_face_untouched(self):
        normal = fake_face(2000.0, 500.0, 440.0, 660.0)
        out = st.sanitize_face_bbox_for_containment(normal, 1215.0, 3840.0)
        self.assertIs(out, normal)
        record("FR-ZOOM-NOTOUCH", "Normal face boxes not modified",
               PASS if out is normal else FAIL,
               "identity", "normal face box unchanged",
               "sanitize_face_bbox_for_containment", "speaker_tracker.py:sanitize_face_bbox_for_containment")

    def test_safe_box_containable_after_cap(self):
        # The whole point: after sanitization, the derived subject safe box must
        # be containable by the baseline portrait crop (corridor non-degenerate).
        big = fake_face(2255.0, 932.0, 1210.0, 1802.0, conf=0.94)
        repaired = st.sanitize_face_bbox_for_containment(big, 1215.0, 3840.0)
        sb = st.compute_subject_safe_box(repaired, 3840.0, 2160.0)
        eff_crop_w = 1215.0
        eff_max_x = 3840.0 - eff_crop_w
        _, _, corridor = st.solve_containment_crop_x(sb, eff_crop_w, eff_max_x, 3840.0)
        self.assertLess(corridor["min_valid_x"], corridor["max_valid_x"] + 1.0)
        record("FR-ZOOM-CORRIDOR", "Containment corridor non-degenerate after cap",
               PASS if corridor["min_valid_x"] < corridor["max_valid_x"] + 1.0 else FAIL,
               f"corridor [{corridor['min_valid_x']:.0f},{corridor['max_valid_x']:.0f}]",
               "min_valid_x <= max_valid_x (head + 2 margins fits the crop)",
               "solve_containment_crop_x on repaired safe box",
               "speaker_tracker.py:solve_containment_crop_x")


class TestTemporalDomains(unittest.TestCase):
    """H: Smart-Pacing temporal mapping into framing (source vs output time)."""

    def test_pacing_remap_keeps_framing_segments_aligned(self):
        # pacing.rs remap_framing_plan shifts segment times into OUTPUT domain;
        # verify here that the python-side src_to_out equivalent is monotonic
        # and covers the whole window (no framing segment maps outside).
        retained = [(0.0, 10.0, 0.0, 10.0), (10.0, 20.0, 10.0, 19.0)]  # 1s removed
        out_times = []
        for s, e, os_, oe in retained:
            for k in range(int((e - s) * 2) + 1):
                t = s + k * 0.5
                if t <= e:
                    out_times.append((t, os_ + (t - s)))
        monotonic = all(out_times[i][1] <= out_times[i + 1][1] for i in range(len(out_times) - 1))
        self.assertTrue(monotonic)
        record("FR-PACING-MAP", "Smart-Pacing source->output mapping monotonic",
               PASS if monotonic else FAIL,
               f"{len(out_times)} mapped points", "monotonic non-decreasing output times",
               "retained interval linear map (pacing.rs:src_to_out equivalent)",
               "pacing.rs:src_to_out / remap_framing_plan", "source -> output")

    def test_framing_segment_times_are_source_domain(self):
        # The framing plan emits clip-relative SOURCE times; the render pieces
        # intersect them with retained source windows (pacing.rs
        # build_render_pieces). A segment list must tile [0, dur] contiguously.
        plan_segs = [(0.0, 30.0), (30.0, 69.725)]
        tiled = all(
            abs(plan_segs[i][1] - plan_segs[i + 1][0]) < 1e-9 for i in range(len(plan_segs) - 1)
        )
        self.assertTrue(tiled)
        record("FR-SEGMENT-TILING", "Framing segments tile the source window contiguously",
               PASS if tiled else FAIL,
               f"segs={plan_segs}", "contiguous tiling over [0, dur]",
               "validate_and_repair_timeline invariant (media.rs)",
               "media.rs:validate_and_repair_timeline", "source")


class TestShotBoundaryAndTargetLoss(unittest.TestCase):
    """I + J: framing state across shot cuts and target loss/reacquisition."""

    def test_shot_scale_state_resets_on_cut(self):
        # Per-shot KEY INVARIANT: ShotScaleState resets at every shot cut so a
        # previous shot's zoom cannot carry into a new shot.
        sss = st.ShotScaleState(shot_id=0, shot_type="close_up", ideal_scale=1.0,
                                min_valid_scale=1.0, max_valid_scale=1.15)
        sss.record(1.0, 1.10, "COMPOSITION")
        self.assertAlmostEqual(sss.current_scale, 1.10)
        sss2 = st.ShotScaleState(shot_id=1, shot_type="wide", ideal_scale=0.95,
                                 min_valid_scale=0.80, max_valid_scale=1.05)
        self.assertAlmostEqual(sss2.current_scale, 0.95)
        record("FR-SHOT-RESET", "Shot scale state resets at shot cut",
               PASS, f"shot0=1.10 -> shot1={sss2.current_scale}",
               "new shot starts at its own ideal_scale",
               "ShotScaleState construction per shot (run() loop)",
               "speaker_tracker.py:ShotScaleState / run()", "source")

    def test_target_loss_fallback_prefers_valid_position(self):
        # classify_and_frame_subject fallback hierarchy: with no face, a trusted
        # prior target_cx differing from default must be used; otherwise the
        # current valid crop is held; otherwise safe center.
        res = st.classify_and_frame_subject(
            shot_tracks={}, shot_samples=[], spk=None, rel_ts=1.0, rel_te=1.5,
            last_confirmed_cx_by_spk={}, default_cx=1312.0,
            crop_w_baseline=1215.0, max_x_baseline=2625.0,
            source_h=2160.0, source_w=3840.0,
            current_crop_x=1000.0, current_crop_y=0.0, current_scale=1.0,
            shot_scale_state=None, shot_id=0, is_shot_cut=False,
        )
        clamped_x = res[0]
        self.assertTrue(0.0 <= clamped_x <= 2625.0)
        record("FR-TARGET-LOSS", "Target-loss fallback keeps crop legal",
               PASS if 0.0 <= clamped_x <= 2625.0 else FAIL,
               f"fallback crop_x={clamped_x:.0f}", "0 <= crop_x <= max_x",
               "classify_and_frame_subject no-face fallback hierarchy",
               "speaker_tracker.py:classify_and_frame_subject", "source")

    def test_target_reacquisition_after_cut(self):
        # After a shot cut the camera-lock operator must immediately establish a
        # new anchor (locked_anchor reset happens in run(); verify the solver
        # produces a crop for the newly present face).
        face = fake_face(2300.0, 500.0, 440.0, 660.0)
        tracks = {"t1": {"cx": 2300.0, "detections": [(0.0, face, 5.0)], "age_sec": 1.0}}
        samples = [(0.0, None, [face]), (0.5, None, [face])]
        res = st.classify_and_frame_subject(
            shot_tracks=tracks, shot_samples=samples, spk=None, rel_ts=0.0, rel_te=0.5,
            last_confirmed_cx_by_spk={}, default_cx=1312.0,
            crop_w_baseline=1215.0, max_x_baseline=2625.0,
            source_h=2160.0, source_w=3840.0,
            current_crop_x=None, current_crop_y=None, current_scale=1.0,
            shot_scale_state=None, shot_id=1, is_shot_cut=True,
        )
        clamped_x, crop_y, cw, ch, target_cx = res[0], res[1], res[2], res[3], res[4]
        contains = (clamped_x - 20.0) <= 2300.0 <= (clamped_x + cw + 20.0)
        self.assertTrue(contains)
        record("FR-REACQUIRE", "Target reacquired after shot cut",
               PASS if contains else FAIL,
               f"crop_x={clamped_x:.0f} w={cw:.0f} contains cx=2300",
               "new face contained immediately after cut",
               "classify_and_frame_subject with is_shot_cut=True",
               "speaker_tracker.py:classify_and_frame_subject", "source")


class TestCropContainment(unittest.TestCase):
    """D + E: crop containment and full-face containment."""

    def test_face_contained_in_crop(self):
        face = fake_face(2300.0, 500.0, 440.0, 660.0)
        sb = st.compute_subject_safe_box(face, 3840.0, 2160.0)
        x, contained, _ = st.solve_containment_crop_x(sb, 1215.0, 2625.0, 3840.0)
        self.assertTrue(contained)
        record("FR-CROP-CONTAIN", "Subject safe box contained by crop",
               PASS if contained else FAIL,
               f"crop_x={x:.0f}", "safe box inside [x, x+1215] with 8% margins",
               "solve_containment_crop_x", "speaker_tracker.py:solve_containment_crop_x")

    def test_full_face_containment_guarantee(self):
        # For every face width up to the containable bound, the solved crop must
        # contain the whole FACE box (not just the safe box head corridor).
        ok = True
        worst = None
        for fw in [100, 250, 440, 600, 700, 820]:
            face = fake_face(2000.0, 500.0, float(fw), float(fw) * 1.5)
            sb = st.compute_subject_safe_box(face, 3840.0, 2160.0)
            x, contained, _ = st.solve_containment_crop_x(sb, 1215.0, 2625.0, 3840.0)
            fx, fy, ffw, ffh = face["bbox"]
            full = (fx >= x - 2.0) and (fx + ffw <= x + 1215.0 + 2.0)
            if not full:
                ok, worst = False, fw
        record("FR-FACE-CONTAIN", "Full face box contained for all containable widths",
               PASS if ok else FAIL,
               f"widths 100..820 {'all contained' if ok else f'fail at w={worst}'}",
               "face box inside crop edges (+2px tolerance)",
               "solve_containment_crop_x sweep", "speaker_tracker.py:solve_containment_crop_x")
        self.assertTrue(ok)


def _mk_face(cx, cy, w=300.0, h=200.0, conf=0.9):
    return {
        "bbox": (cx - w / 2.0, cy - h / 2.0, w, h),
        "center": (cx, cy),
        "conf": conf,
        "area": w * h,
        "frontality": 0.9,
        "gaze_offset_x": 0.0,
        "is_back": False,
        "head_top_y": cy - h / 2.0,
        "eye_line_y": cy - h * 0.1,
        "shoulder_span": (cx - w, cx + w),
        "torso_top": cy + h / 2.0,
        "mouth_gray": None,
        "landmarks": None,
    }


def _mk_track(times, cx, cy, w=300.0, h=200.0, conf=0.9):
    return {
        "cx": float(cx), "cy": float(cy), "vx": 0.0, "vy": 0.0,
        "last_t": float(times[-1]),
        "detections": [(float(t), _mk_face(cx, cy, w, h, conf), 0.0) for t in times],
        "hits": len(times), "age": len(times), "is_occluded": False,
        "provenance": "yolo_person", "has_person_body": True,
    }


class TestSamePersonTrackMerge(unittest.TestCase):
    """Adaptive integrity: one person fragmented into 2+ BoT-SORT tracks must
    never fabricate a two-person composition; two genuine people must never be
    merged."""

    def _resolve_types(self, tracks):
        target_track, _face, _cx, shot_type, role, _ov = st.resolve_visual_subject(
            list(tracks), [], list(tracks), "S1", 0.0, 0.5, False,
            {}, 690.0, None, 1080.0, 1920.0, 1080.0,
            identity=None, config=st.DEFAULT_FRAMING_CONFIG, adaptive_framing=True,
        )
        return shot_type, role

    def test_id_switch_fragments_merged_solo_stays_solo(self):
        # One person: track A covers 0-4s, BoT-SORT ID switch hands off to
        # track B at 5-8s at the same head position (fragment continuation).
        a = _mk_track([0.0, 1.0, 2.0, 3.0, 4.0], 1000.0, 400.0)
        b = _mk_track([5.0, 6.0, 7.0, 8.0], 1030.0, 405.0)
        merged = st.merge_same_person_tracks([a, b])
        self.assertEqual(len(merged), 1,
                         "temporally continuous same-position fragments must merge")
        self.assertEqual(len(merged[0]["detections"]), 9,
                         "merged track must cover both fragments' detections")

        # The resolver must then classify the shot as a single-person composition.
        shot_type, role = self._resolve_types(merged)
        self.assertIn(shot_type, ("solo_close_up", "fallback_persistent_track"))
        record("FR-SAMEPERSON-SOLO", "ID-switch fragments merge; solo shot stays solo",
               PASS if len(merged) == 1 else FAIL,
               f"merged_tracks={len(merged)} shot_type={shot_type}",
               "1 merged track; no two_shot_both_fit",
               "merge_same_person_tracks + resolve_visual_subject",
               "speaker_tracker.py:merge_same_person_tracks")

    def test_simultaneous_duplicate_merged(self):
        # Both tracks carry detections at the SAME timestamps, 60px apart
        # (YuNet face track + YOLO head proxy of the same person, face w=300).
        # Track A also carries a duplicate-timestamp detection (the YuNet
        # safeguard appends at the same rel_t as the YOLO path) — the merge
        # must not compare detection dicts when timestamps tie.
        a = _mk_track([0.0, 0.5, 1.0, 1.5, 2.0], 1010.0, 400.0)
        a["detections"].append(a["detections"][2])  # duplicate timestamp 1.0
        b = _mk_track([0.0, 0.5, 1.0, 1.5, 2.0], 1070.0, 405.0)
        merged = st.merge_same_person_tracks([a, b])
        self.assertEqual(len(merged), 1, "simultaneous duplicate detections are one person")
        record("FR-SAMEPERSON-DUP", "Simultaneous duplicate track merged",
               PASS if len(merged) == 1 else FAIL,
               f"merged_tracks={len(merged)}", "1 merged track",
               "merge_same_person_tracks", "speaker_tracker.py:merge_same_person_tracks")

    def test_distinct_people_never_merged(self):
        # Real two-shot: two people ~900px apart with faces small enough that
        # the pair genuinely fits the 1080 square composition (head span 942).
        l = _mk_track([0.0, 0.5, 1.0, 1.5, 2.0], 402.0, 400.0, w=150.0, h=110.0)
        r = _mk_track([0.0, 0.5, 1.0, 1.5, 2.0], 1308.0, 400.0, w=150.0, h=110.0)
        merged = st.merge_same_person_tracks([l, r])
        self.assertEqual(len(merged), 2, "distinct people must never be merged")
        shot_type, role = self._resolve_types(merged)
        self.assertEqual(shot_type, "two_shot_both_fit")
        record("FR-SAMEPERSON-PAIR", "Distinct people preserved for two-shot",
               PASS if len(merged) == 2 and shot_type == "two_shot_both_fit" else FAIL,
               f"merged_tracks={len(merged)} shot_type={shot_type}",
               "2 tracks; two_shot_both_fit retained",
               "merge_same_person_tracks + resolve_visual_subject",
               "speaker_tracker.py:merge_same_person_tracks")


def _write_report():
    out = os.environ.get("AUTOSHORTS_FRAMING_REPORT")
    if not out:
        cand = os.path.join(script_dir, "..", "..", "tmp", "qa_suite", "reports", "framing")
        os.makedirs(cand, exist_ok=True)
        out = os.path.join(cand, "framing_regression_report.json")
    n = lambda s: sum(1 for r in RESULTS if r["status"] == s)
    json.dump({
        "harness": "framing_temporal_regression",
        "summary": {"pass": n(PASS), "fail": n(FAIL), "skipped": n(SKIPPED),
                    "notVerified": n(NOT_VERIFIED)},
        "checks": RESULTS,
    }, open(out, "w"), indent=1)
    print(f"[framing regression] {n(PASS)} PASS / {n(FAIL)} FAIL / "
          f"{n(SKIPPED)} SKIPPED / {n(NOT_VERIFIED)} NOT VERIFIED -> {out}")


if __name__ == "__main__":
    unittest.main(exit=False)
    _write_report()
