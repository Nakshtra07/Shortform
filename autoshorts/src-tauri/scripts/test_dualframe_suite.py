"""
Test Suite for AutoShorts 6.0 DualFrame Configuration & Layout Resolver State Machine.
Validates:
1. 1 subject -> single
2. 2 subjects for 1 frame (< 0.60s) -> single (persistence entry)
3. 2 persistent subjects (>= 0.60s) -> dual_stack
4. Left-to-right slot assignment: left subject -> TOP panel, right subject -> BOTTOM panel
5. Speaker turn switch while both visible -> slots DO NOT swap (slot stability)
6. 1 subject briefly occluded (< 0.75s) -> remains dual_stack
7. 1 subject absent for > 0.75s -> exits to single
8. Camera cut -> complete state reset
9. 3 subjects -> single
10. AUTOSHORTS_DUALFRAME_ENABLED=false -> single
11. Subject eligibility criteria
"""

import os
import sys
import unittest

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

import speaker_tracker
from speaker_tracker import (
    DualFrameConfig,
    DualFrameLayoutResolver,
    compute_panel_crop_dimensions,
    solve_dual_frame_trajectories,
    CropRectExpr,
)


def make_mock_track(
    track_id: int,
    cx: float,
    cy: float = 300.0,
    conf: float = 0.95,
    is_occluded: bool = False,
    is_back: bool = False,
    visual_prominence: float = 0.80,
    age_sec: float = 1.00,
    face_w: float = 160.0,
    face_h: float = 180.0,
    provenance: str = "yolo_person",
    has_person_body: bool = True,
    frontality: float = 0.90,
) -> dict:
    face = {
        "bbox": (cx - face_w / 2.0, cy - face_h / 2.0, face_w, face_h),
        "center": (cx, cy),
        "conf": conf,
        "area": face_w * face_h,
        "frontality": frontality,
        "gaze_offset_x": 0.0,
        "skin_ratio": 0.60,
        "is_back": is_back,
        "head_top_y": cy - 100.0,
        "eye_line_y": cy - 30.0,
    }
    return {
        "track_id": track_id,
        "cx": cx,
        "cy": cy,
        "vx": 0.0,
        "vy": 0.0,
        "last_t": 1.0,
        "last_mouth": None,
        "detections": [(0.0, face, 0.0), (age_sec, face, 0.0)],
        "hits": 10,
        "age": 10,
        "age_sec": age_sec,
        "is_occluded": is_occluded,
        "is_back": is_back,
        "visual_prominence": visual_prominence,
        "mot": 1.0,
        "provenance": provenance,
        "has_person_body": has_person_body,
    }


class TestDualFrameLayoutResolver(unittest.TestCase):

    def setUp(self):
        # Ensure fresh environment variable state
        if "AUTOSHORTS_DUALFRAME_ENABLED" in os.environ:
            del os.environ["AUTOSHORTS_DUALFRAME_ENABLED"]
        self.config = DualFrameConfig(
            enabled=True,
            dual_entry_persistence_sec=0.60,
            dual_exit_persistence_sec=0.75,
            min_segment_dur_sec=1.00,
            min_track_age_sec=0.50,
            max_lost_sec=0.60,
            panel_aspect_ratio=9.0 / 8.0,
        )
        self.resolver = DualFrameLayoutResolver(self.config)

    # 1. 1 subject -> single
    def test_01_single_subject_remains_single(self):
        tr1 = make_mock_track(1, cx=600.0)
        dec = self.resolver.update(rel_t=0.0, active_tracks={1: tr1})
        self.assertEqual(dec.mode, "single")
        self.assertEqual(self.resolver.mode, "single")
        self.assertIsNone(dec.top_track_id)
        self.assertIsNone(dec.bottom_track_id)

        # Advance time with only 1 subject
        dec2 = self.resolver.update(rel_t=1.5, active_tracks={1: tr1})
        self.assertEqual(dec2.mode, "single")
        self.assertIsNone(dec2.top_track_id)
        self.assertIsNone(dec2.bottom_track_id)

    # 2. 2 subjects for 1 frame (< 0.60s) -> single (persistence entry)
    def test_02_two_subjects_under_entry_persistence_remains_single(self):
        tr_left = make_mock_track(1, cx=400.0)
        tr_right = make_mock_track(2, cx=1400.0)

        # Frame 1 at t=0.0s (elapsed 0.0s < 0.60s)
        dec1 = self.resolver.update(rel_t=0.0, active_tracks={1: tr_left, 2: tr_right})
        self.assertEqual(dec1.mode, "single", "First frame of 2 subjects must remain single")
        self.assertEqual(self.resolver.mode, "single")

        # Frame 2 at t=0.50s (elapsed 0.50s < 0.60s)
        dec2 = self.resolver.update(rel_t=0.50, active_tracks={1: tr_left, 2: tr_right})
        self.assertEqual(dec2.mode, "single", "Under 0.60s persistence must remain single")
        self.assertEqual(self.resolver.mode, "single")

    # 3. 2 persistent subjects (>= 0.60s) -> dual_stack
    def test_03_two_subjects_persistent_enters_dual_stack(self):
        tr_left = make_mock_track(1, cx=400.0)
        tr_right = make_mock_track(2, cx=1400.0)

        # Start candidate timer at t=0.0s
        self.resolver.update(rel_t=0.0, active_tracks={1: tr_left, 2: tr_right})
        self.resolver.update(rel_t=0.30, active_tracks={1: tr_left, 2: tr_right})

        # At t=0.60s (elapsed >= 0.60s) -> enters dual_stack
        dec = self.resolver.update(rel_t=0.60, active_tracks={1: tr_left, 2: tr_right})
        self.assertEqual(dec.mode, "dual_stack", "At >= 0.60s persistence must transition to dual_stack")
        self.assertEqual(self.resolver.mode, "dual_stack")
        self.assertIsNotNone(dec.top_track_id)
        self.assertIsNotNone(dec.bottom_track_id)

    # 4. Left-to-right slot assignment: left subject -> TOP panel, right subject -> BOTTOM panel
    def test_04_left_to_right_slot_assignment(self):
        tr_left = make_mock_track(10, cx=350.0)
        tr_right = make_mock_track(20, cx=1450.0)

        self.resolver.update(rel_t=0.0, active_tracks={10: tr_left, 20: tr_right})
        dec = self.resolver.update(rel_t=0.65, active_tracks={10: tr_left, 20: tr_right})

        self.assertEqual(dec.mode, "dual_stack")
        self.assertEqual(dec.top_track_id, 10, "Left subject (cx=350) must be locked to TOP panel")
        self.assertEqual(dec.bottom_track_id, 20, "Right subject (cx=1450) must be locked to BOTTOM panel")
        self.assertEqual(self.resolver.locked_top_track_id, 10)
        self.assertEqual(self.resolver.locked_bottom_track_id, 20)

    # 5. Speaker turn switch while both visible -> slots DO NOT swap (slot stability)
    def test_05_speaker_turn_switch_slot_stability(self):
        tr_left = make_mock_track(1, cx=400.0)
        tr_right = make_mock_track(2, cx=1400.0)

        # Enter dual_stack
        self.resolver.update(rel_t=0.0, active_tracks={1: tr_left, 2: tr_right})
        self.resolver.update(rel_t=0.60, active_tracks={1: tr_left, 2: tr_right})

        # Speaker turn switch: Person 2 is now speaking, Person 1 is silent
        # Even if their physical X positions cross (Person 1 moves right, Person 2 moves left)
        tr_left_moved = make_mock_track(1, cx=1500.0)
        tr_right_moved = make_mock_track(2, cx=300.0)

        dec = self.resolver.update(
            rel_t=1.20,
            active_tracks={1: tr_left_moved, 2: tr_right_moved},
            active_spk="S2",
        )

        self.assertEqual(dec.mode, "dual_stack")
        self.assertEqual(dec.top_track_id, 1, "Top slot must NOT swap despite turn or motion")
        self.assertEqual(dec.bottom_track_id, 2, "Bottom slot must NOT swap despite turn or motion")
        self.assertEqual(self.resolver.locked_top_track_id, 1)
        self.assertEqual(self.resolver.locked_bottom_track_id, 2)

    # 6. 1 subject briefly occluded (< 0.75s) -> remains dual_stack
    def test_06_brief_occlusion_remains_dual_stack(self):
        tr_left = make_mock_track(1, cx=400.0)
        tr_right = make_mock_track(2, cx=1400.0)

        # Establish dual_stack at t=1.0s
        self.resolver.update(rel_t=0.40, active_tracks={1: tr_left, 2: tr_right})
        self.resolver.update(rel_t=1.00, active_tracks={1: tr_left, 2: tr_right})
        self.assertEqual(self.resolver.mode, "dual_stack")

        # Subject 2 is briefly occluded at t=1.30s (0.30s absence < 0.75s)
        dec_occ = self.resolver.update(rel_t=1.30, active_tracks={1: tr_left})
        self.assertEqual(dec_occ.mode, "dual_stack", "Brief occlusion < 0.75s must stay dual_stack")
        self.assertEqual(dec_occ.top_track_id, 1)
        self.assertEqual(dec_occ.bottom_track_id, 2)

        # Subject 2 still occluded at t=1.60s (0.60s absence < 0.75s)
        dec_occ2 = self.resolver.update(rel_t=1.60, active_tracks={1: tr_left})
        self.assertEqual(dec_occ2.mode, "dual_stack")

        # Subject 2 reappears at t=1.70s -> still dual_stack without interruption
        dec_rec = self.resolver.update(rel_t=1.70, active_tracks={1: tr_left, 2: tr_right})
        self.assertEqual(dec_rec.mode, "dual_stack")

    # 7. 1 subject absent for > 0.75s -> exits to single
    def test_07_subject_absent_exits_to_single(self):
        tr_left = make_mock_track(1, cx=400.0)
        tr_right = make_mock_track(2, cx=1400.0)

        # Establish dual_stack at t=1.0s
        self.resolver.update(rel_t=0.40, active_tracks={1: tr_left, 2: tr_right})
        self.resolver.update(rel_t=1.00, active_tracks={1: tr_left, 2: tr_right})
        self.assertEqual(self.resolver.mode, "dual_stack")

        # Subject 2 becomes absent starting after t=1.00s
        # At t=1.50s (0.50s absent <= 0.75s) -> still dual
        self.resolver.update(rel_t=1.50, active_tracks={1: tr_left})
        self.assertEqual(self.resolver.mode, "dual_stack")

        # At t=1.80s (0.80s absent > 0.75s) -> clean transition to single
        dec = self.resolver.update(rel_t=1.80, active_tracks={1: tr_left})
        self.assertEqual(dec.mode, "single", "Absence > 0.75s must exit to single")
        self.assertEqual(self.resolver.mode, "single")
        self.assertIsNone(dec.top_track_id)
        self.assertIsNone(dec.bottom_track_id)
        self.assertIsNone(self.resolver.locked_top_track_id)
        self.assertIsNone(self.resolver.locked_bottom_track_id)

    # 8. Camera cut -> complete state reset
    def test_08_camera_cut_complete_state_reset(self):
        tr_left = make_mock_track(1, cx=400.0)
        tr_right = make_mock_track(2, cx=1400.0)

        # Establish dual_stack
        self.resolver.update(rel_t=0.0, active_tracks={1: tr_left, 2: tr_right})
        self.resolver.update(rel_t=0.70, active_tracks={1: tr_left, 2: tr_right})
        self.assertEqual(self.resolver.mode, "dual_stack")

        # Camera cut at t=2.0s
        dec_cut = self.resolver.update(
            rel_t=2.0,
            active_tracks={1: tr_left, 2: tr_right},
            is_shot_cut=True,
        )

        self.assertEqual(dec_cut.mode, "single", "Camera cut must immediately reset state to single")
        self.assertEqual(self.resolver.mode, "single")
        self.assertIsNone(dec_cut.top_track_id)
        self.assertIsNone(dec_cut.bottom_track_id)
        self.assertIsNone(self.resolver.locked_top_track_id)
        self.assertIsNone(self.resolver.locked_bottom_track_id)

        # Persistence timer restarted: at t=2.2s (< 0.60s post-cut), still single
        dec_after = self.resolver.update(rel_t=2.2, active_tracks={1: tr_left, 2: tr_right})
        self.assertEqual(dec_after.mode, "single")

        # Only after new shot persists for >= 0.60s does it re-enter dual_stack
        dec_new_dual = self.resolver.update(rel_t=2.65, active_tracks={1: tr_left, 2: tr_right})
        self.assertEqual(dec_new_dual.mode, "dual_stack")

    # 9. 3 subjects -> single
    def test_09_three_subjects_remains_single(self):
        tr1 = make_mock_track(1, cx=300.0)
        tr2 = make_mock_track(2, cx=900.0)
        tr3 = make_mock_track(3, cx=1500.0)

        tracks = {1: tr1, 2: tr2, 3: tr3}
        for t in [0.0, 0.5, 1.0, 1.5, 2.0]:
            dec = self.resolver.update(rel_t=t, active_tracks=tracks)
            self.assertEqual(dec.mode, "single", f"3 subjects must remain single at t={t}")
            self.assertEqual(self.resolver.mode, "single")

    # 10. AUTOSHORTS_DUALFRAME_ENABLED=false -> single
    def test_10_disabled_via_env_var_remains_single(self):
        old_val = os.environ.get("AUTOSHORTS_DUALFRAME_ENABLED")
        try:
            os.environ["AUTOSHORTS_DUALFRAME_ENABLED"] = "false"
            disabled_config = DualFrameConfig()
            self.assertFalse(disabled_config.enabled)

            disabled_resolver = DualFrameLayoutResolver(disabled_config)
            tr_left = make_mock_track(1, cx=400.0)
            tr_right = make_mock_track(2, cx=1400.0)

            for t in [0.0, 0.5, 1.0, 1.5]:
                dec = disabled_resolver.update(rel_t=t, active_tracks={1: tr_left, 2: tr_right})
                self.assertEqual(dec.mode, "single", f"Disabled resolver must remain single at t={t}")
                self.assertEqual(disabled_resolver.mode, "single")
        finally:
            if old_val is not None:
                os.environ["AUTOSHORTS_DUALFRAME_ENABLED"] = old_val
            else:
                os.environ.pop("AUTOSHORTS_DUALFRAME_ENABLED", None)

    # 11. Subject eligibility criteria
    def test_11_subject_eligibility_criteria(self):
        # Valid subject
        tr_valid = make_mock_track(1, cx=500.0, conf=0.90, visual_prominence=0.70)
        self.assertTrue(self.resolver.is_subject_eligible(tr_valid))

        # Occluded subject
        tr_occ = make_mock_track(2, cx=500.0, is_occluded=True)
        self.assertFalse(self.resolver.is_subject_eligible(tr_occ))

        # Back of head
        tr_back = make_mock_track(3, cx=500.0, is_back=True)
        self.assertFalse(self.resolver.is_subject_eligible(tr_back))

        # Low visual prominence (background passerby)
        tr_passerby = make_mock_track(4, cx=500.0, visual_prominence=0.08)
        self.assertFalse(self.resolver.is_subject_eligible(tr_passerby))

        # Out of bounds geometry
        tr_oob = make_mock_track(5, cx=-100.0)
        self.assertFalse(self.resolver.is_subject_eligible(tr_oob))

    # 12. YuNet-only fallback background artifact is rejected
    def test_12_yunet_only_background_artifact_rejected(self):
        tr_real = make_mock_track(1, cx=818.0, cy=480.0, face_w=300.0, face_h=310.0, conf=0.91, provenance="yolo_person", has_person_body=True)
        # Mock painting/poster face detection: conf=0.54, 83x109 px, no person body
        tr_artifact = make_mock_track(
            1000,
            cx=210.0,
            cy=478.0,
            face_w=83.0,
            face_h=109.0,
            conf=0.54,
            provenance="yunet_face_only",
            has_person_body=False,
            visual_prominence=None,
        )
        # Must be rejected by eligibility
        self.assertFalse(self.resolver.is_subject_eligible(tr_artifact, source_w=1920.0, source_h=1080.0))
        self.assertTrue(self.resolver.is_subject_eligible(tr_real, source_w=1920.0, source_h=1080.0))

        # Combined in resolver: only 1 eligible subject -> remains single
        tracks = {1: tr_real, 1000: tr_artifact}
        for t in [0.0, 0.30, 0.60, 1.00, 1.50]:
            dec = self.resolver.update(rel_t=t, active_tracks=tracks, source_w=1920.0, source_h=1080.0)
            self.assertEqual(dec.mode, "single", f"Artifact + real person must remain single at t={t}")
            self.assertEqual(self.resolver.mode, "single")
            self.assertIsNone(dec.top_track_id)
            self.assertIsNone(dec.bottom_track_id)

    # 13. YuNet-only genuine ECU face-only shot is accepted as an exception
    def test_13_yunet_only_genuine_ecu_face_exception_accepted(self):
        tr_real = make_mock_track(1, cx=400.0, cy=300.0, conf=0.92, provenance="yolo_person", has_person_body=True)
        # Genuine extreme close-up: conf >= 0.70, width >= 125, height >= 108, area >= 13500, frontality >= 0.40
        tr_ecu = make_mock_track(
            2,
            cx=1400.0,
            cy=350.0,
            face_w=180.0,
            face_h=200.0,
            conf=0.88,
            provenance="yunet_face_only",
            has_person_body=False,
            frontality=0.85,
            visual_prominence=0.85,
        )
        self.assertTrue(self.resolver.is_subject_eligible(tr_ecu, source_w=1920.0, source_h=1080.0))

        # Enters dual_stack after persistence
        tracks = {1: tr_real, 2: tr_ecu}
        self.resolver.update(rel_t=0.0, active_tracks=tracks, source_w=1920.0, source_h=1080.0)
        dec = self.resolver.update(rel_t=0.65, active_tracks=tracks, source_w=1920.0, source_h=1080.0)
        self.assertEqual(dec.mode, "dual_stack")
        self.assertEqual(dec.top_track_id, 1)
        self.assertEqual(dec.bottom_track_id, 2)

    # 14. Rejection of identical track IDs (prevents self-pairing / cloning)
    def test_14_rejection_of_identical_track_ids(self):
        tr1_left = make_mock_track(1, cx=400.0)
        tr1_right = make_mock_track(1, cx=1400.0)
        # Even if somehow passed with identical track IDs:
        # Using a mock active_tracks list where both resolved candidates share ID 1
        active_tracks = {1: tr1_left}
        for t in [0.0, 0.30, 0.65, 1.00]:
            dec = self.resolver.update(rel_t=t, active_tracks=active_tracks, source_w=1920.0, source_h=1080.0)
            self.assertEqual(dec.mode, "single")
            self.assertIsNone(dec.top_track_id)
            self.assertIsNone(dec.bottom_track_id)

    # 15. Rejection of disproportionate scale ratio (< 15% area ratio)
    def test_15_rejection_of_disproportionate_scale_ratio(self):
        # tr_large area = 240 * 250 = 60,000
        tr_large = make_mock_track(1, cx=400.0, face_w=240.0, face_h=250.0)
        # tr_tiny area = 60 * 60 = 3,600 (ratio = 3600 / 60000 = 0.06 < 0.15)
        tr_tiny = make_mock_track(2, cx=1400.0, face_w=60.0, face_h=60.0)

        tracks = {1: tr_large, 2: tr_tiny}
        for t in [0.0, 0.30, 0.65, 1.00]:
            dec = self.resolver.update(rel_t=t, active_tracks=tracks, source_w=1920.0, source_h=1080.0)
            self.assertEqual(dec.mode, "single", f"Scale disparity must remain single at t={t}")
            self.assertEqual(self.resolver.mode, "single")

    # 16. Rejection of overlapping horizontal coordinates (< 12% width separation)
    def test_16_rejection_of_overlapping_horizontal_coordinates(self):
        # 1920 * 0.12 = 230.4 px minimum separation
        tr_a = make_mock_track(1, cx=600.0)
        tr_b = make_mock_track(2, cx=700.0)  # separation = 100 px < 230.4 px

        tracks = {1: tr_a, 2: tr_b}
        for t in [0.0, 0.30, 0.65, 1.00]:
            dec = self.resolver.update(rel_t=t, active_tracks=tracks, source_w=1920.0, source_h=1080.0)
            self.assertEqual(dec.mode, "single", f"Horizontal overlap must remain single at t={t}")
            self.assertEqual(self.resolver.mode, "single")

    # 17. Micro-detection filter (< 38x38 px or area < 1800)
    def test_17_micro_detection_filter(self):
        tr_micro = make_mock_track(1, cx=500.0, face_w=30.0, face_h=35.0)  # area = 1050 < 1800
        self.assertFalse(self.resolver.is_subject_eligible(tr_micro, source_w=1920.0, source_h=1080.0))


class TestDualFramePanelGeometryAndTrajectory(unittest.TestCase):
    """
    Test suite for Task 2: Independent 9:8 Panel Geometry & Dual Trajectory Generation.
    Validates:
    1. compute_panel_crop_dimensions produces exact 9:8 ratio (|crop_w - round(crop_h * 1.125)| <= 1) and even dimensions
    2. Independent crops for top subject (e.g. at left cx=200) and bottom subject (e.g. at right cx=1400) produce independent x expressions
    3. Vertical crop position preserves eye-line and clamps within 0 <= y <= H - crop_h
    4. Scale changes dynamically adjust crop box size without aspect distortion
    5. Dynamic subject scale from subject box
    """

    def test_01_panel_crop_dimensions_9_8_ratio_and_even(self):
        resolutions = [
            (1920.0, 1080.0),
            (1280.0, 720.0),
            (3840.0, 2160.0),
            (1080.0, 1080.0),
        ]
        scales = [0.80, 0.95, 1.00, 1.15, 1.30]

        for src_w, src_h in resolutions:
            for scale in scales:
                crop_w, crop_h = compute_panel_crop_dimensions(
                    source_w=src_w,
                    source_h=src_h,
                    current_scale=scale,
                )
                self.assertEqual(
                    crop_w % 2, 0,
                    f"crop_w must be even, got {crop_w} for {src_w}x{src_h} scale={scale}"
                )
                self.assertEqual(
                    crop_h % 2, 0,
                    f"crop_h must be even, got {crop_h} for {src_w}x{src_h} scale={scale}"
                )
                self.assertLessEqual(
                    abs(crop_w - round(crop_h * 1.125)), 1,
                    f"9:8 aspect ratio check failed: crop_w={crop_w}, crop_h={crop_h} for {src_w}x{src_h} scale={scale}"
                )
                self.assertGreaterEqual(crop_w, 2)
                self.assertLessEqual(crop_w, int(src_w))
                self.assertGreaterEqual(crop_h, 2)
                self.assertLessEqual(crop_h, int(src_h))

    def test_02_independent_crops_left_top_and_right_bottom(self):
        tr_top = make_mock_track(1, cx=200.0, cy=350.0)
        tr_bottom = make_mock_track(2, cx=1400.0, cy=350.0)

        res = solve_dual_frame_trajectories(
            top_track=tr_top,
            bottom_track=tr_bottom,
            source_w=1920.0,
            source_h=1080.0,
            start_sec=0.0,
        )

        top_crop = res["top_crop"]
        bottom_crop = res["bottom_crop"]

        # Verify independent x coordinates
        self.assertNotEqual(
            top_crop["x"], bottom_crop["x"],
            "Top and bottom crops must have independent x coordinates"
        )

        top_x = float(top_crop["x"])
        bottom_x = float(bottom_crop["x"])
        top_w = int(top_crop["w"])
        bottom_w = int(bottom_crop["w"])

        # Top subject at cx=200 should be near left bound (0 <= top_x <= 200)
        self.assertGreaterEqual(top_x, 0.0)
        self.assertLessEqual(top_x, 200.0)

        # Bottom subject at cx=1400 should be significantly to the right
        self.assertGreater(bottom_x, top_x + 200.0)
        self.assertLessEqual(bottom_x, 1920.0 - bottom_w)

        # Also test object-attribute access
        if hasattr(res, "top_crop"):
            self.assertEqual(res.top_crop.x, top_crop["x"])
            self.assertEqual(res.bottom_crop.x, bottom_crop["x"])

    def test_03_vertical_crop_position_eye_line_and_bounds_clamping(self):
        # 1. Normal eye-line
        tr_normal = make_mock_track(1, cx=600.0, cy=300.0)
        res_normal = solve_dual_frame_trajectories(
            top_track=tr_normal,
            bottom_track=tr_normal,
            source_w=1920.0,
            source_h=1080.0,
        )
        crop_h = int(res_normal["top_crop"]["h"])
        crop_y = float(res_normal["top_crop"]["y"])
        self.assertGreaterEqual(crop_y, 0.0)
        self.assertLessEqual(crop_y, 1080.0 - crop_h)
        # Preserves eye line: eye-line should be near 30% from crop top
        eye_ratio = (270.0 - crop_y) / float(crop_h)
        self.assertAlmostEqual(eye_ratio, 0.30, delta=0.15)

        # 2. Extreme high subject -> clamps to y >= 0
        tr_high = make_mock_track(2, cx=600.0, cy=50.0)
        res_high = solve_dual_frame_trajectories(
            top_track=tr_high,
            bottom_track=tr_high,
            source_w=1920.0,
            source_h=1080.0,
        )
        crop_y_high = float(res_high["top_crop"]["y"])
        self.assertEqual(crop_y_high, 0.0)

        # 3. Extreme low subject -> clamps to y <= source_h - crop_h
        tr_low = make_mock_track(3, cx=600.0, cy=1050.0)
        res_low = solve_dual_frame_trajectories(
            top_track=tr_low,
            bottom_track=tr_low,
            source_w=1920.0,
            source_h=1080.0,
        )
        crop_y_low = float(res_low["top_crop"]["y"])
        crop_h_low = int(res_low["top_crop"]["h"])
        self.assertEqual(crop_y_low, float(1080 - crop_h_low))

    def test_04_scale_changes_adjust_crop_box_size_without_distortion(self):
        crop_w_wide, crop_h_wide = compute_panel_crop_dimensions(
            source_w=1920.0,
            source_h=1080.0,
            current_scale=0.90,
        )
        crop_w_tight, crop_h_tight = compute_panel_crop_dimensions(
            source_w=1920.0,
            source_h=1080.0,
            current_scale=1.25,
        )

        # Tighter zoom must produce strictly smaller dimensions
        self.assertLess(crop_w_tight, crop_w_wide)
        self.assertLess(crop_h_tight, crop_h_wide)

        # Both maintain 9:8 ratio
        self.assertLessEqual(abs(crop_w_wide - round(crop_h_wide * 1.125)), 1)
        self.assertLessEqual(abs(crop_w_tight - round(crop_h_tight * 1.125)), 1)

        # Zero non-uniform stretching: aspect ratios must be almost identical
        aspect_wide = crop_w_wide / crop_h_wide
        aspect_tight = crop_w_tight / crop_h_tight
        self.assertAlmostEqual(aspect_wide, aspect_tight, delta=0.01)

    def test_05_dynamic_subject_scale_from_subject_box(self):
        # Passing subject box with smaller vs larger face size
        small_face_box = {"bbox": (500.0, 300.0, 80.0, 90.0), "center": (540.0, 345.0)}
        large_face_box = {"bbox": (400.0, 200.0, 400.0, 450.0), "center": (600.0, 425.0)}

        crop_w_small, crop_h_small = compute_panel_crop_dimensions(
            subject_box=small_face_box,
            source_w=1920.0,
            source_h=1080.0,
        )
        crop_w_large, crop_h_large = compute_panel_crop_dimensions(
            subject_box=large_face_box,
            source_w=1920.0,
            source_h=1080.0,
        )

        # Smaller subject gets zoomed in more (smaller crop box)
        self.assertLessEqual(crop_w_small, crop_w_large)
        self.assertLessEqual(crop_h_small, crop_h_large)
        self.assertLessEqual(abs(crop_w_small - round(crop_h_small * 1.125)), 1)
        self.assertLessEqual(abs(crop_w_large - round(crop_h_large * 1.125)), 1)

    def test_06_independent_motion_trajectories(self):
        # Top subject moves left-to-right from cx=200 to cx=1400 over 2.0s
        face_t0 = {"bbox": (120.0, 210.0, 160.0, 180.0), "center": (200.0, 300.0), "eye_line_y": 270.0}
        face_t1 = {"bbox": (1320.0, 210.0, 160.0, 180.0), "center": (1400.0, 300.0), "eye_line_y": 270.0}
        tr_top = {
            "track_id": 1,
            "cx": 1400.0,
            "cy": 300.0,
            "detections": [
                (0.0, face_t0, 0.95),
                (1.0, face_t0, 0.95),
                (2.0, face_t1, 0.95),
            ],
            "age_sec": 2.0,
        }
        # Bottom subject is stationary at cx=1400
        face_bot = {"bbox": (1320.0, 210.0, 160.0, 180.0), "center": (1400.0, 300.0), "eye_line_y": 270.0}
        tr_bottom = {
            "track_id": 2,
            "cx": 1400.0,
            "cy": 300.0,
            "detections": [
                (0.0, face_bot, 0.95),
                (2.0, face_bot, 0.95),
            ],
            "age_sec": 2.0,
        }

        res = solve_dual_frame_trajectories(
            top_track=tr_top,
            bottom_track=tr_bottom,
            source_w=1920.0,
            source_h=1080.0,
            start_sec=0.0,
        )

        # Top has motion: x expression should contain time condition or interpolation
        self.assertIn("t", res["top_crop"]["x"])
        # Bottom is stationary: x expression is constant number string
        self.assertTrue(res["bottom_crop"]["x"].isdigit())


def make_mock_face(cx: float, cy: float = 300.0, conf: float = 0.95):
    return {
        "bbox": (cx - 80.0, cy - 90.0, 160.0, 180.0),
        "center": (cx, cy),
        "conf": conf,
        "area": 28800.0,
        "frontality": 0.90,
        "gaze_offset_x": 0.0,
        "skin_ratio": 0.60,
        "is_back": False,
        "head_top_y": cy - 100.0,
        "eye_line_y": cy - 30.0,
        "shoulder_span": (cx - 150.0, cx + 150.0),
        "torso_top": cy + 90.0,
        "mouth_gray": None,
        "landmarks": None,
    }


class TestDualFrameLayoutPlanAndTelemetry(unittest.TestCase):
    """
    Test suite for Task 3: Python Full Layout Plan JSON Output & Telemetry.
    Validates:
    1. Full layout plan JSON output structure: mode, x, y, w, h, segments.
    2. Continuous coverage: segments cover 0.0 to clip duration without gaps.
    3. DualStack segments contain top_crop and bottom_crop with 9:8 aspect ratio.
    4. Fallback single mode when disabled via AUTOSHORTS_DUALFRAME_ENABLED=false or when only 1 person present.
    5. Structured telemetry verification in sys.stderr.
    """

    def setUp(self):
        if "AUTOSHORTS_DUALFRAME_ENABLED" in os.environ:
            del os.environ["AUTOSHORTS_DUALFRAME_ENABLED"]
        # Save/restore the backend env: this class forces the native backend
        # for determinism, but leaking the override into OTHER suites in a
        # combined pytest run silently flipped them to native too (an
        # unrelated suite's UltralyticsTracker assertions then failed).
        self._saved_backend = os.environ.get("AUTOSHORTS_CV_BACKEND")
        os.environ["AUTOSHORTS_CV_BACKEND"] = "native"

    def tearDown(self):
        if "AUTOSHORTS_DUALFRAME_ENABLED" in os.environ:
            del os.environ["AUTOSHORTS_DUALFRAME_ENABLED"]
        if self._saved_backend is not None:
            os.environ["AUTOSHORTS_CV_BACKEND"] = self._saved_backend
        else:
            os.environ.pop("AUTOSHORTS_CV_BACKEND", None)

    def _run_pipeline(self, all_samples, scene_cuts=None, duration_sec=10.0):
        import io
        from unittest.mock import patch
        import numpy as np

        if scene_cuts is None:
            scene_cuts = [0.0]
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        with patch("speaker_tracker.sample_video_and_detect_shots", return_value=(scene_cuts, all_samples)), \
             patch("sys.stdout", stdout_buf), \
             patch("sys.stderr", stderr_buf):
            speaker_tracker.run(
                source_path="mock_clip.mp4",
                start_ms=0.0,
                end_ms=duration_sec * 1000.0,
                crop_w=608.0,
                max_x=1312.0,
                default_x=656.0,
            )
        return stdout_buf.getvalue(), stderr_buf.getvalue()

    def test_01_full_layout_plan_json_output_structure(self):
        import numpy as np
        times = np.arange(0.0, 10.1, 0.5)
        dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        samples = [
            (t, dummy_frame, [make_mock_face(400.0), make_mock_face(1400.0)])
            for t in times
        ]

        stdout, stderr = self._run_pipeline(samples, duration_sec=10.0)
        lines = [line.strip() for line in stdout.splitlines() if line.strip() and not line.strip().startswith("[")]
        self.assertTrue(lines, f"Expected JSON output on stdout, got: {stdout}")
        import json
        plan = json.loads(lines[-1])

        # Required fields in root contract
        self.assertIn("mode", plan)
        self.assertIn("x", plan)
        self.assertIn("y", plan)
        self.assertIn("w", plan)
        self.assertIn("h", plan)
        self.assertIn("segments", plan)

        self.assertEqual(plan["mode"], "dual_stack")
        # Top-level fallback parameters must be valid non-empty strings
        self.assertIsInstance(plan["x"], str)
        self.assertIsInstance(plan["y"], str)
        self.assertIsInstance(plan["w"], str)
        self.assertIsInstance(plan["h"], str)
        self.assertGreater(len(plan["x"]), 0)
        self.assertGreater(len(plan["w"]), 0)
        self.assertIsInstance(plan["segments"], list)
        self.assertGreaterEqual(len(plan["segments"]), 1)

    def test_02_continuous_coverage_zero_gaps(self):
        import numpy as np
        clip_dur = 15.0
        times = np.arange(0.0, clip_dur + 0.1, 0.5)
        dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        # First 2 seconds: only 1 person at 600 -> single
        # 2.0s to 12.0s: 2 people -> dual_stack
        # 12.0s to 15.0s: right person leaves -> single
        samples = []
        for t in times:
            if t < 2.0:
                samples.append((t, dummy_frame, [make_mock_face(600.0)]))
            elif t <= 12.0:
                samples.append((t, dummy_frame, [make_mock_face(400.0), make_mock_face(1400.0)]))
            else:
                samples.append((t, dummy_frame, [make_mock_face(400.0)]))

        stdout, _ = self._run_pipeline(samples, duration_sec=clip_dur)
        lines = [line.strip() for line in stdout.splitlines() if line.strip() and not line.strip().startswith("[")]
        import json
        plan = json.loads(lines[-1])

        segments = plan["segments"]
        self.assertGreaterEqual(len(segments), 2, "Expected multiple segments across transition")

        # 1. Start at 0.0
        self.assertAlmostEqual(segments[0]["start"], 0.0, delta=0.01)
        # 2. End at clip duration
        self.assertAlmostEqual(segments[-1]["end"], clip_dur, delta=0.01)

        # 3. Contiguous without gaps or overlaps
        for i in range(len(segments) - 1):
            curr_end = segments[i]["end"]
            next_start = segments[i + 1]["start"]
            self.assertAlmostEqual(
                curr_end, next_start, delta=0.01,
                msg=f"Gap or overlap between segment {i} (end={curr_end}) and {i+1} (start={next_start})"
            )

        for s in segments:
            self.assertGreater(s["end"], s["start"], f"Invalid segment duration: {s}")

    def test_03_dual_stack_segment_geometry_9_8_ratio(self):
        import numpy as np
        times = np.arange(0.0, 8.1, 0.5)
        dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        samples = [
            (t, dummy_frame, [make_mock_face(350.0), make_mock_face(1500.0)])
            for t in times
        ]

        stdout, _ = self._run_pipeline(samples, duration_sec=8.0)
        lines = [line.strip() for line in stdout.splitlines() if line.strip() and not line.strip().startswith("[")]
        import json
        plan = json.loads(lines[-1])

        dual_segs = [s for s in plan["segments"] if s["mode"] == "dual_stack"]
        self.assertTrue(dual_segs, "Expected at least one dual_stack segment")

        for ds in dual_segs:
            self.assertIn("top_track_id", ds)
            self.assertIn("bottom_track_id", ds)
            self.assertIn("top_crop", ds)
            self.assertIn("bottom_crop", ds)

            top_crop = ds["top_crop"]
            bottom_crop = ds["bottom_crop"]

            for crop_dict, role in [(top_crop, "top"), (bottom_crop, "bottom")]:
                self.assertIn("x", crop_dict)
                self.assertIn("y", crop_dict)
                self.assertIn("w", crop_dict)
                self.assertIn("h", crop_dict)

                w = int(crop_dict["w"])
                h = int(crop_dict["h"])

                # Even pixel dimensions
                self.assertEqual(w % 2, 0, f"{role} crop width {w} must be even")
                self.assertEqual(h % 2, 0, f"{role} crop height {h} must be even")

                # Exact 9:8 ratio
                self.assertLessEqual(
                    abs(w - round(h * 1.125)), 1,
                    f"{role} crop aspect ratio must be 9:8, got {w}x{h}"
                )

            # Independent horizontal coordinates
            self.assertNotEqual(top_crop["x"], bottom_crop["x"])

    def test_04_fallback_single_mode_when_disabled_via_env(self):
        import numpy as np
        os.environ["AUTOSHORTS_DUALFRAME_ENABLED"] = "false"
        times = np.arange(0.0, 8.1, 0.5)
        dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        samples = [
            (t, dummy_frame, [make_mock_face(400.0), make_mock_face(1400.0)])
            for t in times
        ]

        stdout, _ = self._run_pipeline(samples, duration_sec=8.0)
        lines = [line.strip() for line in stdout.splitlines() if line.strip() and not line.strip().startswith("[")]
        import json
        plan = json.loads(lines[-1])

        self.assertEqual(plan["mode"], "single")
        for s in plan["segments"]:
            self.assertEqual(s["mode"], "single")
            self.assertIn("crop", s)
            self.assertNotIn("top_crop", s)
            self.assertNotIn("bottom_crop", s)

    def test_05_fallback_single_mode_when_only_one_person(self):
        import numpy as np
        times = np.arange(0.0, 8.1, 0.5)
        dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        samples = [
            (t, dummy_frame, [make_mock_face(960.0)])
            for t in times
        ]

        stdout, _ = self._run_pipeline(samples, duration_sec=8.0)
        lines = [line.strip() for line in stdout.splitlines() if line.strip() and not line.strip().startswith("[")]
        import json
        plan = json.loads(lines[-1])

        self.assertEqual(plan["mode"], "single")
        for s in plan["segments"]:
            self.assertEqual(s["mode"], "single")
            self.assertIn("crop", s)

    def test_06_structured_telemetry_in_stderr(self):
        import numpy as np
        times = np.arange(0.0, 8.1, 0.5)
        dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        samples = [
            (t, dummy_frame, [make_mock_face(400.0), make_mock_face(1400.0)])
            for t in times
        ]

        _, stderr = self._run_pipeline(samples, duration_sec=8.0)
        stderr_lines = [l.strip() for l in stderr.splitlines() if l.strip()]
        df_lines = [l for l in stderr_lines if l.startswith("[DualFrame]")]

        self.assertTrue(df_lines, f"Expected [DualFrame] telemetry in stderr, got:\n{stderr}")

        plan_summary = [l for l in df_lines if "layout_plan total_segments=" in l]
        self.assertTrue(plan_summary, "Missing [DualFrame] layout_plan total_segments=... line")
        self.assertIn("dual_segments=", plan_summary[0])

        enter_logs = [l for l in df_lines if "enter t=" in l]
        self.assertTrue(enter_logs, "Missing [DualFrame] enter t=... line")
        self.assertIn("top_track=", enter_logs[0])
        self.assertIn("bottom_track=", enter_logs[0])

    def test_07_moving_subject_in_nonzero_start_dual_segment_relative_keyframes(self):
        import numpy as np
        import json
        # First 2 seconds: only 1 person (single mode)
        # From 2.0 to 10.0: 2 persons with movement on person A (400 -> 700)
        times = np.arange(0.0, 10.1, 0.5)
        dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        samples = []
        for t in times:
            if t < 2.0:
                samples.append((t, dummy_frame, [make_mock_face(400.0)]))
            else:
                moving_cx = 400.0 + (t - 2.0) * 35.0  # Moves from 400 to ~680
                samples.append((t, dummy_frame, [make_mock_face(moving_cx), make_mock_face(1400.0)]))

        stdout, _ = self._run_pipeline(samples, duration_sec=10.0)
        lines = [line.strip() for line in stdout.splitlines() if line.strip() and not line.strip().startswith("[")]
        plan = json.loads(lines[-1])

        dual_segs = [s for s in plan["segments"] if s["mode"] == "dual_stack"]
        self.assertTrue(dual_segs, "Expected at least one dual_stack segment")
        ds = dual_segs[0]
        self.assertGreater(ds["start"], 1.0, f"Segment start should be >= 2.0s, got {ds['start']}")

        # Verify top_crop x expression is relative to segment start (starts at t ~ 0)
        top_x = ds["top_crop"]["x"]
        if "between(" in top_x:
            # First between(t, a, b) should have 'a' near 0.0, NOT offset by ds['start']
            import re
            m = re.search(r"between\(t,\s*([0-9\.]+),", top_x)
            if m:
                first_t = float(m.group(1))
                self.assertLess(first_t, 1.0, f"First keyframe time in segment must be relative to segment start (~0.0), got {first_t}")

    def test_08_plan_demotes_invalid_dual_segment_to_single(self):
        from speaker_tracker import DualFrameDecision, build_layout_plan, DualFrameConfig
        tr1 = make_mock_track(1, cx=600.0)
        # Duplicate track ID: top=1, bottom=1
        dec_dup = DualFrameDecision("dual_stack", 1, 1, 1.5)
        timeline_decisions = [
            {"start": 0.0, "end": 2.0, "decision": dec_dup, "tracks": {1: tr1}}
        ]

        plan = build_layout_plan(
            timeline_decisions=timeline_decisions,
            clip_dur=2.0,
            start_sec=0.0,
            source_w=1920.0,
            source_h=1080.0,
            crop_w_baseline=608.0,
            val_kf=[(0.0, 656.0), (2.0, 656.0)],
            y_keyframes=[(0.0, 0.0), (2.0, 0.0)],
            default_x=656.0,
            max_legal_x=1312,
            max_legal_y=0,
            final_crop_w=608,
            final_crop_h=1080,
            x_expr="656",
            y_expr="0",
            w_expr="608",
            h_expr="1080",
            config=DualFrameConfig(enabled=True, min_segment_dur_sec=0.5),
        )

        self.assertEqual(plan.mode, "single")
        for s in plan.segments:
            self.assertEqual(s.mode, "single", "Segment with duplicate top/bottom ID must be demoted to single")
            self.assertIsNotNone(s.crop)
            self.assertIsNone(s.top_crop)
            self.assertIsNone(s.bottom_crop)

    def test_09_segment_aware_face_bounds_single_and_dual(self):
        import json
        from speaker_tracker import DualFrameDecision, build_layout_plan, DualFrameConfig

        tr_top = {
            "id": 1,
            "cx": 400.0,
            "cy": 400.0,
            "age_sec": 5.0,
            "detections": [
                (2.5, {"bbox": [350.0, 200.0, 100.0, 100.0]}),
                (3.5, {"bbox": [360.0, 210.0, 100.0, 100.0]}),
            ],
        }
        tr_bot = {
            "id": 2,
            "cx": 1400.0,
            "cy": 400.0,
            "age_sec": 5.0,
            "detections": [
                (2.5, {"bbox": [1350.0, 300.0, 100.0, 100.0]}),
                (3.5, {"bbox": [1360.0, 310.0, 100.0, 100.0]}),
            ],
        }

        timeline_decisions = [
            {
                "start": 0.0,
                "end": 2.0,
                "decision": DualFrameDecision("single", None, None, 1.0),
                "face_bbox": [600.0, 200.0, 100.0, 120.0],
                "single_crop": (550.0, 0.0, 608.0, 1080.0),
                "tracks": {},
            },
            {
                "start": 2.0,
                "end": 5.0,
                "decision": DualFrameDecision("dual_stack", 1, 2, 2.0),
                "face_bbox": None,
                "single_crop": None,
                "tracks": {1: tr_top, 2: tr_bot},
            },
        ]

        plan = build_layout_plan(
            timeline_decisions=timeline_decisions,
            clip_dur=5.0,
            start_sec=0.0,
            source_w=1920.0,
            source_h=1080.0,
            crop_w_baseline=608.0,
            val_kf=[(0.0, 550.0), (5.0, 550.0)],
            y_keyframes=[(0.0, 0.0), (5.0, 0.0)],
            default_x=550.0,
            max_legal_x=1312,
            max_legal_y=0,
            final_crop_w=608,
            final_crop_h=1080,
            x_expr="550",
            y_expr="0",
            w_expr="608",
            h_expr="1080",
            config=DualFrameConfig(enabled=True, min_segment_dur_sec=0.5),
        )

        self.assertEqual(len(plan.segments), 2)

        # 1. Single segment face bounds
        seg_single = plan.segments[0]
        self.assertEqual(seg_single.mode, "single")
        self.assertIsNotNone(seg_single.face_bounds, "Single segment must have face_bounds")
        self.assertGreater(seg_single.face_bounds["top"], 0.0)
        self.assertLess(seg_single.face_bounds["bottom"], 1.0)

        # 2. Dual segment face bounds
        seg_dual = plan.segments[1]
        self.assertEqual(seg_dual.mode, "dual_stack")
        self.assertIsNotNone(seg_dual.top_face_bounds, "Dual segment must have top_face_bounds")
        self.assertIsNotNone(seg_dual.bottom_face_bounds, "Dual segment must have bottom_face_bounds")

        # Top panel bounds mapped to [0.0, 0.50]
        self.assertGreaterEqual(seg_dual.top_face_bounds["top"], 0.0)
        self.assertLessEqual(seg_dual.top_face_bounds["bottom"], 0.50)

        # Bottom panel bounds mapped to [0.50, 1.00]
        self.assertGreaterEqual(seg_dual.bottom_face_bounds["top"], 0.50)
        self.assertLessEqual(seg_dual.bottom_face_bounds["bottom"], 1.00)

        # 3. Root legacy fallback face bounds matches single segment
        self.assertEqual(plan.face_bounds, seg_single.face_bounds)

        # 4. JSON serialization roundtrip
        plan_json = plan.to_json()
        plan_dict = json.loads(plan_json)
        self.assertIn("face_bounds", plan_dict["segments"][0])
        self.assertIn("top_face_bounds", plan_dict["segments"][1])
        self.assertIn("bottom_face_bounds", plan_dict["segments"][1])


class TestCloseSpeakerPanelIsolation(unittest.TestCase):
    """
    Task 4 Part 1: Close-Speaker DualFrame Panel Isolation.
    Validates:
    1. Clearly separated speakers -> baseline crop, NO isolation zoom.
    2. Moderately close speakers -> boundary evaluation (isolation only if projected crop contaminated).
    3. Very close speakers -> isolation pinch-zoom activates on BOTH panels, other speaker excluded.
    4. Extreme overlap -> bounded max zoom (MAX_ISOLATION_SCALE), deterministic fallback, no pathological overzoom.
    5. Single speaker (other_track=None) -> Single framing unchanged.
    6. Invalid/face-only other track -> provenance protection unchanged (no crash, no zoom).
    All crops must preserve 9:8 aspect ratio and even dimensions.
    """

    def _make_iso_track(self, track_id, cx, cy=300.0, face_w=160.0, face_h=180.0, conf=0.95):
        face = {
            "bbox": (cx - face_w / 2.0, cy - face_h / 2.0, face_w, face_h),
            "center": (cx, cy),
            "conf": conf,
            "area": face_w * face_h,
            "frontality": 0.90,
            "gaze_offset_x": 0.0,
            "eye_line_y": cy - 30.0,
            "head_top_y": cy - face_h / 2.0 - 0.22 * face_h,
        }
        return {
            "track_id": track_id,
            "cx": cx,
            "cy": cy,
            "detections": [(0.0, face, conf), (1.0, face, conf), (2.0, face, conf)],
            "age_sec": 2.0,
            "provenance": "yolo_person",
            "has_person_body": True,
            "is_occluded": False,
            "is_back": False,
            "visual_prominence": 0.80,
        }

    def _assert_9_8_even(self, crop, role):
        w = int(crop["w"])
        h = int(crop["h"])
        self.assertEqual(w % 2, 0, f"{role} crop_w must be even, got {w}")
        self.assertEqual(h % 2, 0, f"{role} crop_h must be even, got {h}")
        self.assertLessEqual(abs(w - round(h * 1.125)), 1, f"{role} crop must be 9:8, got {w}x{h}")

    def _other_excluded(self, crop, other_face, source_w=1920.0):
        """True if the other speaker's safe box is fully outside the crop with clearance."""
        fx, fy, fw, fh = other_face["bbox"]
        head_left = max(0.0, fx - 0.12 * fw)
        head_right = min(source_w, fx + 1.12 * fw)
        shoulder_w = fw * 2.2
        sl = max(0.0, fx + fw / 2.0 - shoulder_w / 2.0)
        sr = min(source_w, fx + fw / 2.0 + shoulder_w / 2.0)
        other_left = min(head_left, sl)
        other_right = max(head_right, sr)
        x = float(crop["x"])
        w = float(crop["w"])
        clearance = 0.02 * source_w
        return (other_right <= x - clearance) or (other_left >= x + w + clearance)

    def test_01_separated_speakers_no_isolation_zoom(self):
        # Speakers 848px apart: baseline crop must isolate without any zoom
        tr_top = self._make_iso_track(1, cx=300.0)
        tr_bot = self._make_iso_track(2, cx=1500.0)
        res = solve_dual_frame_trajectories(tr_top, tr_bot, source_w=1920.0, source_h=1080.0)

        # Baseline scale for 160px face (ratio 0.083 -> 1.12): crop 1084x964
        self.assertEqual(int(res.top_crop["w"]), 1084)
        self.assertEqual(int(res.bottom_crop["w"]), 1084)
        self._assert_9_8_even(res.top_crop, "top")
        self._assert_9_8_even(res.bottom_crop, "bottom")
        # Both panels exclude the other speaker
        self.assertTrue(self._other_excluded(res.top_crop, tr_bot["detections"][0][1]))
        self.assertTrue(self._other_excluded(res.bottom_crop, tr_top["detections"][0][1]))

    def test_02_moderately_close_boundary_evaluation(self):
        # Speakers ~560px apart: baseline 1084px crop cannot exclude the other
        # speaker on both sides -> isolation must evaluate and zoom minimally
        tr_top = self._make_iso_track(1, cx=680.0)
        tr_bot = self._make_iso_track(2, cx=1240.0)
        res = solve_dual_frame_trajectories(tr_top, tr_bot, source_w=1920.0, source_h=1080.0)

        self._assert_9_8_even(res.top_crop, "top")
        self._assert_9_8_even(res.bottom_crop, "bottom")
        # Isolation zoom must be active (crop narrower than baseline 1084)
        self.assertLess(int(res.top_crop["w"]), 1084, "Top panel must pinch-zoom to isolate close speaker")
        self.assertLess(int(res.bottom_crop["w"]), 1084, "Bottom panel must pinch-zoom to isolate close speaker")
        # Minimal zoom: crop must not collapse to pathological tightness
        self.assertGreater(int(res.top_crop["w"]), 400)
        self.assertGreater(int(res.bottom_crop["w"]), 400)
        # Both panels exclude the other speaker
        self.assertTrue(self._other_excluded(res.top_crop, tr_bot["detections"][0][1]))
        self.assertTrue(self._other_excluded(res.bottom_crop, tr_top["detections"][0][1]))

    def test_03_very_close_speakers_isolation_zoom_both_panels(self):
        # Real failing geometry (clip-02 style): BIG right speaker + SMALL left speaker
        big_face = {
            "bbox": (978.5, 212.2, 190.6, 271.9), "center": (1073.8, 348.1),
            "eye_line_y": 307.0, "gaze_offset_x": 0.0, "frontality": 0.90,
        }
        small_face = {
            "bbox": (248.8, 171.9, 37.2, 51.5), "center": (267.4, 197.7),
            "eye_line_y": 180.0, "gaze_offset_x": 0.0, "frontality": 0.90,
        }
        tr_small = {
            "track_id": 1, "cx": 267.4, "cy": 197.7,
            "detections": [(0.0, small_face, 0.87), (1.0, small_face, 0.87), (2.0, small_face, 0.87)],
            "age_sec": 2.0, "provenance": "yolo_person", "has_person_body": True,
        }
        tr_big = {
            "track_id": 2, "cx": 1073.8, "cy": 348.1,
            "detections": [(0.0, big_face, 0.94), (1.0, big_face, 0.94), (2.0, big_face, 0.94)],
            "age_sec": 2.0, "provenance": "yolo_person", "has_person_body": True,
        }
        res = solve_dual_frame_trajectories(tr_small, tr_big, source_w=1920.0, source_h=1080.0)

        self._assert_9_8_even(res.top_crop, "top")
        self._assert_9_8_even(res.bottom_crop, "bottom")
        # SMALL speaker (top): base scale 1.30 -> must zoom tighter (crop < 934 baseline)
        self.assertLess(int(res.top_crop["w"]), 934, "Top panel (small face, base 1.30) must pinch-zoom")
        # BIG speaker (bottom): base 1.12 crop 1084 already isolates -> no unnecessary zoom
        self.assertEqual(int(res.bottom_crop["w"]), 1084, "Bottom panel must keep baseline (already isolated)")
        # Both panels exclude the other speaker
        self.assertTrue(self._other_excluded(res.top_crop, big_face))
        self.assertTrue(self._other_excluded(res.bottom_crop, small_face))

    def test_04_extreme_overlap_bounded_max_zoom(self):
        # Speakers nearly overlapping (centers 120px apart): full isolation is
        # geometrically impossible -> bounded fallback at MAX_ISOLATION_SCALE
        tr_top = self._make_iso_track(1, cx=900.0)
        tr_bot = self._make_iso_track(2, cx=1020.0)
        res = solve_dual_frame_trajectories(tr_top, tr_bot, source_w=1920.0, source_h=1080.0)

        self._assert_9_8_even(res.top_crop, "top")
        self._assert_9_8_even(res.bottom_crop, "bottom")
        # Bounded: crop must never be tighter than MAX_ISOLATION_SCALE allows
        # scale 1.80 -> crop_h = 1080/1.80 = 600 -> crop_w = 675
        min_allowed_w = 660  # tolerance for even-dim rounding
        self.assertGreaterEqual(int(res.top_crop["w"]), min_allowed_w,
                               "Top panel must not overzoom past MAX_ISOLATION_SCALE")
        self.assertGreaterEqual(int(res.bottom_crop["w"]), min_allowed_w,
                                "Bottom panel must not overzoom past MAX_ISOLATION_SCALE")
        # DualFrame must NOT be disabled: both crops exist and are legal
        for crop in (res.top_crop, res.bottom_crop):
            x = float(crop["x"]); w = float(crop["w"])
            self.assertGreaterEqual(x, 0.0)
            self.assertLessEqual(x + w, 1920.0)

    def test_05_single_speaker_unchanged(self):
        # other_track=None (single framing): must behave exactly as before
        tr = self._make_iso_track(1, cx=600.0)
        res = solve_dual_frame_trajectories(tr, None, source_w=1920.0, source_h=1080.0)
        # Bottom track None -> fallback crop (baseline scale 1.0: 1216x1080)
        self.assertEqual(int(res.bottom_crop["w"]), 1216)
        self.assertEqual(int(res.bottom_crop["h"]), 1080)
        # Top track solved normally with baseline scale for 160px face (1.12)
        self.assertEqual(int(res.top_crop["w"]), 1084)
        self._assert_9_8_even(res.top_crop, "top")
        self._assert_9_8_even(res.bottom_crop, "bottom")

    def test_06_invalid_other_track_no_crash_no_zoom(self):
        # Face-only / invalid other track (no detections): isolation must be a
        # no-op, not a crash, and must not zoom the panel
        tr_top = self._make_iso_track(1, cx=600.0)
        tr_invalid = {
            "track_id": 99, "cx": 1300.0, "cy": 300.0,
            "detections": [],  # no detections -> no pairing possible
            "age_sec": 2.0, "provenance": "yunet_only", "has_person_body": False,
        }
        res = solve_dual_frame_trajectories(tr_top, tr_invalid, source_w=1920.0, source_h=1080.0)
        # Top panel keeps baseline scale (no zoom from an unpairable other track)
        self.assertEqual(int(res.top_crop["w"]), 1084)
        self._assert_9_8_even(res.top_crop, "top")


class TestIsolationStabilityRefinement(unittest.TestCase):
    """
    Task: DualFrame close-speaker isolation stability refinement.
    Validates the two-part architecture:
    - PART 1 (clean pass-through): when the normal containment placement
      already excludes the other speaker, the normal crop trajectory passes
      through UNCHANGED (gaze preference, deadband, trajectory continuity).
    - PART 2 (constrained tracking): under contamination, isolation acts as a
      BOUNDARY/CONSTRAINT on the normal trajectory — the camera keeps
      tracking the intended speaker inside the safe region, with temporal
      hysteresis suppressing jitter-induced movement. No fresh isolation
      solve per keyframe, no Zone A/B flip-flop.
    Symmetry: identical logic for top and bottom panels (no panel-specific
    stabilization code).
    """

    SRC_W, SRC_H = 1920.0, 1080.0

    def _face(self, cx, cy=300.0, face_w=160.0, face_h=180.0, conf=0.95, gaze=0.0):
        return {
            "bbox": (cx - face_w / 2.0, cy - face_h / 2.0, face_w, face_h),
            "center": (cx, cy),
            "conf": conf,
            "area": face_w * face_h,
            "frontality": 0.90,
            "gaze_offset_x": gaze,
            "eye_line_y": cy - 30.0,
            "head_top_y": cy - face_h / 2.0 - 0.22 * face_h,
        }

    def _track(self, track_id, faces_with_times):
        dets = [(t, f, f["conf"]) for (t, f) in faces_with_times]
        last_face = faces_with_times[-1][1]
        return {
            "track_id": track_id,
            "cx": last_face["center"][0],
            "cy": last_face["center"][1],
            "detections": dets,
            "age_sec": 2.0,
            "provenance": "yolo_person",
            "has_person_body": True,
            "is_occluded": False,
            "is_back": False,
            "visual_prominence": 0.80,
        }

    def _other_excluded(self, crop, other_face, source_w=1920.0):
        """True if the other speaker's safe box is fully outside the crop with clearance."""
        fx, fy, fw, fh = other_face["bbox"]
        head_left = max(0.0, fx - 0.12 * fw)
        head_right = min(source_w, fx + 1.12 * fw)
        shoulder_w = fw * 2.2
        sl = max(0.0, fx + fw / 2.0 - shoulder_w / 2.0)
        sr = min(source_w, fx + fw / 2.0 + shoulder_w / 2.0)
        other_left = min(head_left, sl)
        other_right = max(head_right, sr)
        x = float(crop["x"])
        w = float(crop["w"])
        clearance = 0.02 * source_w
        return (other_right <= x - clearance) or (other_left >= x + w + clearance)

    def _eval_x(self, crop, t):
        """Evaluates a constant or if()-tree crop x expression at time t."""
        expr = crop["x"].strip()
        try:
            return float(expr)
        except ValueError:
            pass
        import re

        def ev(s, pos):
            m = re.match(r"if\(lt\(t,(-?\d+\.?\d*)\),", s[pos:])
            if m:
                thresh = float(m.group(1))
                pos += m.end()
                a, pos = ev(s, pos)
                assert s[pos] == ","
                b, pos = ev(s, pos + 1)
                assert s[pos] == ")"
                return (a if t < thresh else b), pos + 1
            if s[pos] == "(":
                v, pos = ev(s, pos + 1)
                while pos < len(s) and s[pos] in "+-*/":
                    op = s[pos]
                    rhs, pos = ev(s, pos + 1)
                    v = v + rhs if op == "+" else v - rhs if op == "-" else v * rhs if op == "*" else v / rhs
                assert s[pos] == ")"
                return v, pos + 1
            if s[pos] == "t":
                return t, pos + 1
            m = re.match(r"-?\d+\.?\d*", s[pos:])
            return float(m.group(0)), pos + m.end()

        v, _ = ev(expr, 0)
        return v

    def _assert_9_8_even(self, crop, role):
        w, h = int(crop["w"]), int(crop["h"])
        self.assertEqual(w % 2, 0, f"{role} crop_w must be even, got {w}")
        self.assertEqual(h % 2, 0, f"{role} crop_h must be even, got {h}")
        self.assertLessEqual(abs(w - round(h * 1.125)), 1, f"{role} crop must be 9:8, got {w}x{h}")

    # ── TEST 1: clean separated speakers → normal behavior, no isolation ──

    def test_1_clean_separated_normal_behavior(self):
        # Speakers far apart: normal containment placement is already clean.
        # The emitted crop must be IDENTICAL to a solo (other_track=None) solve.
        tr_top = self._track(1, [(0.0, self._face(300.0)), (1.0, self._face(300.0)), (2.0, self._face(300.0))])
        tr_bot = self._track(2, [(0.0, self._face(1500.0)), (1.0, self._face(1500.0)), (2.0, self._face(1500.0))])
        res_dual = solve_dual_frame_trajectories(tr_top, tr_bot, source_w=self.SRC_W, source_h=self.SRC_H)
        res_solo = solve_dual_frame_trajectories(tr_top, None, source_w=self.SRC_W, source_h=self.SRC_H)
        self.assertEqual(res_dual.top_crop["x"], res_solo.top_crop["x"],
                         "Clean top panel must pass through normal behavior unchanged")
        self.assertEqual(res_dual.top_crop["w"], res_solo.top_crop["w"])
        self.assertEqual(res_dual.top_crop["y"], res_solo.top_crop["y"])
        self._assert_9_8_even(res_dual.top_crop, "top")
        self._assert_9_8_even(res_dual.bottom_crop, "bottom")
        # And the other speaker is genuinely excluded
        self.assertTrue(self._other_excluded(res_dual.top_crop, tr_bot["detections"][0][1]))
        self.assertTrue(self._other_excluded(res_dual.bottom_crop, tr_top["detections"][0][1]))

    # ── TEST 2: clean + gaze preference preserved ──────────────────────────

    def test_2_clean_gaze_preference_preserved(self):
        # Gazing left (gaze_offset_x < -0.10) → normal solver prefers 0.54
        # (subject left of center). Clean panel must keep this preference.
        tr_top = self._track(1, [(0.0, self._face(300.0, gaze=-0.5)),
                                 (1.0, self._face(300.0, gaze=-0.5)),
                                 (2.0, self._face(300.0, gaze=-0.5))])
        tr_bot = self._track(2, [(0.0, self._face(1500.0)), (1.0, self._face(1500.0)), (2.0, self._face(1500.0))])
        res_dual = solve_dual_frame_trajectories(tr_top, tr_bot, source_w=self.SRC_W, source_h=self.SRC_H)
        res_solo = solve_dual_frame_trajectories(tr_top, None, source_w=self.SRC_W, source_h=self.SRC_H)
        self.assertEqual(res_dual.top_crop["x"], res_solo.top_crop["x"],
                         "Gaze preference must survive the clean pass-through")
        # Sanity: solo placement actually uses the gaze preference (0.54)
        # subject cx=300+safe-box offset; ideal = cx - crop_w*0.54
        # (verifying equality with solo is the contract; the solo value itself
        # is the pre-existing normal behavior)

    # ── TEST 3: clean + deadband preserved ────────────────────────────────

    def test_3_clean_deadband_preserved(self):
        # Subject drifts 1px/s (sub-deadband): normal solver holds the camera
        # still via the compositional deadband. A clean panel must keep that
        # deadband (the old unconditional isolation solve bypassed it).
        faces = [(float(t), self._face(600.0 + 1.0 * t)) for t in range(8)]
        tr_top = self._track(1, faces)
        tr_bot = self._track(2, [(0.0, self._face(1600.0)), (1.0, self._face(1600.0)), (2.0, self._face(1600.0))])
        res_dual = solve_dual_frame_trajectories(tr_top, tr_bot, source_w=self.SRC_W, source_h=self.SRC_H)
        res_solo = solve_dual_frame_trajectories(tr_top, None, source_w=self.SRC_W, source_h=self.SRC_H)
        self.assertEqual(res_dual.top_crop["x"], res_solo.top_crop["x"],
                         "Deadband behavior must survive the clean pass-through")
        # The drift is sub-deadband: solo emits a CONSTANT x
        self.assertEqual(res_solo.top_crop["x"], res_dual.top_crop["x"])
        try:
            float(res_solo.top_crop["x"])
        except ValueError:
            self.fail("Sub-deadband drift must produce a constant crop x, got an expression")

    # ── TEST 4: contamination requires isolation ───────────────────────────

    def test_4_contamination_requires_isolation(self):
        # Speakers close enough that the normal placement would include the
        # other speaker: the constrained solve MUST move the crop off the
        # normal placement and exclude the other speaker.
        tr_top = self._track(1, [(0.0, self._face(700.0)), (1.0, self._face(700.0)), (2.0, self._face(700.0))])
        tr_bot = self._track(2, [(0.0, self._face(1150.0)), (1.0, self._face(1150.0)), (2.0, self._face(1150.0))])
        res = solve_dual_frame_trajectories(tr_top, tr_bot, source_w=self.SRC_W, source_h=self.SRC_H)
        self._assert_9_8_even(res.top_crop, "top")
        self._assert_9_8_even(res.bottom_crop, "bottom")
        # Both panels exclude the other speaker (isolation active)
        self.assertTrue(self._other_excluded(res.top_crop, tr_bot["detections"][0][1]),
                        "Contaminated top panel must exclude the other speaker")
        self.assertTrue(self._other_excluded(res.bottom_crop, tr_top["detections"][0][1]),
                        "Contaminated bottom panel must exclude the other speaker")

    # ── TEST 5: other-speaker jitter → NO crop movement ────────────────────

    def test_5_other_speaker_jitter_no_crop_movement(self):
        # Target speaker static; other speaker jitters ±2px around a close
        # position. The camera must NOT move (hysteresis suppresses jitter).
        n = 12
        jitter = [2.0, -2.0, 1.5, -1.5, 2.5, -2.5, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5]
        own_faces = [(float(t), self._face(700.0)) for t in range(n)]
        other_faces = [(float(t), self._face(1150.0 + jitter[t])) for t in range(n)]
        tr_top = self._track(1, own_faces)
        tr_bot = self._track(2, other_faces)
        res = solve_dual_frame_trajectories(tr_top, tr_bot, source_w=self.SRC_W, source_h=self.SRC_H)
        xs = [self._eval_x(res.top_crop, float(t)) for t in range(n)]
        movement = max(xs) - min(xs)
        self.assertLessEqual(movement, 1.0,
                             f"Other-speaker jitter (±2px) must not move the camera; got range {movement:.2f}px")
        # Isolation still holds at every moment
        for t in range(n):
            other_face = other_faces[t][1]
            self.assertTrue(self._other_excluded(
                {"x": xs[t], "w": res.top_crop["w"]}, other_face),
                f"Other speaker must stay excluded at t={t}")

    # ── TEST 6: target movement → camera follows inside safe region ───────

    def test_6_target_movement_camera_follows_in_safe_region(self):
        # Target speaker genuinely moves (30px/s) toward a static, close
        # other speaker. The camera must follow the target (not freeze) while
        # keeping the other speaker excluded at every moment.
        #
        # Fixture geometry (corrected): the walk must remain ISOLATABLE at
        # every moment — the full-safe-box gap to the other speaker must
        # stay >= safety margin + clearance at some zoom <=
        # MAX_ISOLATION_SCALE. The original fixture (start 700 -> end 970
        # vs other at 1250) closed the gap from 198px to -72px (overlap):
        # no legal zoom can isolate that, so from t=4 onward the bounded
        # fallback necessarily admitted the other speaker — a physically
        # impossible scenario, not a production defect. The corrected start
        # (525 -> end 795) keeps the same genuine 30px/s movement toward
        # the other speaker but ENDS at the closest geometrically feasible
        # approach: the zoom solver isolates all 10 moments (scale 1.54,
        # no fallback) and the X constraint engages as the target nears the
        # isolation boundary, so the camera follows the movement naturally
        # inside the safe region.
        n = 10
        own_faces = [(float(t), self._face(525.0 + 30.0 * t)) for t in range(n)]
        other_faces = [(float(t), self._face(1250.0)) for t in range(n)]
        tr_top = self._track(1, own_faces)
        tr_bot = self._track(2, other_faces)
        res = solve_dual_frame_trajectories(tr_top, tr_bot, source_w=self.SRC_W, source_h=self.SRC_H)
        xs = [self._eval_x(res.top_crop, float(t)) for t in range(n)]
        # Camera follows: total movement must be substantial (>= 60px)
        total = max(xs) - min(xs)
        self.assertGreaterEqual(total, 60.0,
                                f"Camera must follow genuine target movement; got range {total:.2f}px")
        # Monotonic-ish following: final x > initial x
        self.assertGreater(xs[-1], xs[0], "Camera must track in the direction of target movement")
        # Other speaker excluded at every moment
        for t in range(n):
            self.assertTrue(self._other_excluded(
                {"x": xs[t], "w": res.top_crop["w"]}, other_faces[t][1]),
                f"Other speaker must stay excluded at t={t}")
        # Subject contained at every moment (head inside crop)
        for t in range(n):
            face = own_faces[t][1]
            fx, fy, fw, fh = face["bbox"]
            self.assertGreaterEqual(fx + fw, xs[t] - 2.0, f"Subject head must stay in crop at t={t}")
            self.assertLessEqual(fx, xs[t] + float(res.top_crop["w"]) + 2.0, f"Subject head must stay in crop at t={t}")

    # ── TEST 7: boundary change → smooth adjustment, no jump ──────────────

    def test_7_boundary_change_smooth_adjustment(self):
        # Other speaker genuinely shifts 60px closer mid-trajectory: the safe
        # region boundary genuinely changes → the camera adjusts to the new
        # boundary. The adjustment must be smooth (no jump larger than the
        # boundary change itself + hysteresis margin).
        n = 12
        own_faces = [(float(t), self._face(700.0)) for t in range(n)]
        # Other speaker steps 60px closer at t=6 (genuine boundary change)
        other_cx = [1150.0 if t < 6 else 1090.0 for t in range(n)]
        other_faces = [(float(t), self._face(other_cx[t])) for t in range(n)]
        tr_top = self._track(1, own_faces)
        tr_bot = self._track(2, other_faces)
        res = solve_dual_frame_trajectories(tr_top, tr_bot, source_w=self.SRC_W, source_h=self.SRC_H)
        xs = [self._eval_x(res.top_crop, float(t)) for t in range(n)]
        # Camera must adjust (boundary genuinely changed by 60px > hysteresis 5px)
        self.assertGreater(max(xs) - min(xs), 0.0,
                           "Genuine boundary change must adjust the camera")
        # No pathological jump: single-step movement bounded by boundary shift + margin
        max_step = max(abs(xs[i + 1] - xs[i]) for i in range(n - 1))
        self.assertLessEqual(max_step, 60.0 + 2 * speaker_tracker.ISOLATION_BOUNDARY_HYSTERESIS_PX + 1.0,
                             f"Boundary adjustment must be smooth; max step {max_step:.2f}px")
        # Other speaker excluded at every moment
        for t in range(n):
            self.assertTrue(self._other_excluded(
                {"x": xs[t], "w": res.top_crop["w"]}, other_faces[t][1]),
                f"Other speaker must stay excluded at t={t}")

    # ── TEST 8: one panel clean / one contaminated → independent ──────────

    def test_8_one_panel_clean_one_contaminated(self):
        # Top speaker's panel is clean (pass-through), bottom speaker's panel
        # is contaminated (constrained). Each panel solved independently.
        #
        # Fixture geometry (corrected): the original fixture (top at 300,
        # bottom at 700) left only a 48px full-safe-box gap between the
        # speakers while exclusion requires clearance (38.4) + round buffer
        # (1) + safety margin (>= 54 at MAX_ISOLATION_SCALE) ~= 93px — no
        # legal zoom could isolate the bottom panel, so the bounded fallback
        # necessarily admitted the top speaker. Physically impossible, not a
        # production defect.
        #
        # With equal crop widths, "top panel clean" and "bottom panel
        # contaminated" are exactly complementary for centered placements, so
        # the corrected fixture breaks the symmetry using the normal
        # solver's own gaze preference (core DualFrame framing, not isolation
        # machinery): both speakers gaze left (pref 0.54 -> each crop shifts
        # ~43px left). The TOP panel's crop hugs the left frame and stays
        # clear of the bottom speaker (genuinely clean, ~37px slack); the
        # BOTTOM panel's crop reaches left toward the top speaker (genuinely
        # contaminated: its normal placement would include the top speaker's
        # safe box by ~50px past clearance), so PART 2 constrained tracking
        # engages WITHOUT zoom (the safe region exists at baseline scale).
        tr_top = self._track(1, [(0.0, self._face(600.0, gaze=-0.5)),
                                 (1.0, self._face(600.0, gaze=-0.5)),
                                 (2.0, self._face(600.0, gaze=-0.5))])
        tr_bot = self._track(2, [(0.0, self._face(1350.0, gaze=-0.5)),
                                 (1.0, self._face(1350.0, gaze=-0.5)),
                                 (2.0, self._face(1350.0, gaze=-0.5))])
        res = solve_dual_frame_trajectories(tr_top, tr_bot, source_w=self.SRC_W, source_h=self.SRC_H)
        # Top panel: clean → FULL pass-through. x, w and y must all equal the
        # solo solve, so any leak across panels (zoom, constraint,
        # hysteresis) is detected.
        res_solo = solve_dual_frame_trajectories(tr_top, None, source_w=self.SRC_W, source_h=self.SRC_H)
        self.assertEqual(res.top_crop["x"], res_solo.top_crop["x"],
                         "Clean top panel must be identical to solo (pass-through)")
        self.assertEqual(res.top_crop["w"], res_solo.top_crop["w"],
                         "Clean top panel must not inherit zoom from the contaminated panel")
        self.assertEqual(res.top_crop["y"], res_solo.top_crop["y"],
                         "Clean top panel must not inherit crop movement from the contaminated panel")
        # Bottom panel: contaminated → must exclude the top speaker
        self.assertTrue(self._other_excluded(res.bottom_crop, tr_top["detections"][0][1]),
                        "Contaminated bottom panel must exclude the other speaker")
        # Bottom panel: isolation must have ACTIVELY engaged — the dual
        # placement differs from the bottom panel's own solo (normal) solve.
        # This guards the fixture against silently degenerating into a fully
        # separated clean case (which would no longer exercise isolation).
        res_bot_solo = solve_dual_frame_trajectories(None, tr_bot, source_w=self.SRC_W, source_h=self.SRC_H)
        self.assertNotEqual(res.bottom_crop["x"], res_bot_solo.bottom_crop["x"],
                             "Contaminated bottom panel must be constrained away from its normal placement")
        self._assert_9_8_even(res.top_crop, "top")
        self._assert_9_8_even(res.bottom_crop, "bottom")

    # ── TEST 9: impossible isolation → bounded fallback intact ────────────

    def test_9_impossible_isolation_bounded_fallback(self):
        # Speakers nearly overlapping: no isolated placement exists at any
        # zoom within MAX_ISOLATION_SCALE. The bounded fallback must hold:
        # crop stays legal, subject contained, no crash, no overzoom.
        tr_top = self._track(1, [(0.0, self._face(900.0)), (1.0, self._face(900.0)), (2.0, self._face(900.0))])
        tr_bot = self._track(2, [(0.0, self._face(1020.0)), (1.0, self._face(1020.0)), (2.0, self._face(1020.0))])
        res = solve_dual_frame_trajectories(tr_top, tr_bot, source_w=self.SRC_W, source_h=self.SRC_H)
        for role, crop in (("top", res.top_crop), ("bottom", res.bottom_crop)):
            x, w = float(crop["x"]), float(crop["w"])
            self.assertGreaterEqual(x, 0.0, f"{role} crop x must be legal")
            self.assertLessEqual(x + w, self.SRC_W, f"{role} crop must stay within source")
            self.assertGreaterEqual(int(crop["w"]), 660, f"{role} must not overzoom past MAX_ISOLATION_SCALE")
            self._assert_9_8_even(crop, role)
        # Subject head must remain contained in its own panel (fallback keeps subject)
        for crop, tr in ((res.top_crop, tr_top), (res.bottom_crop, tr_bot)):
            face = tr["detections"][0][1]
            fx, fy, fw, fh = face["bbox"]
            x, w = float(crop["x"]), float(crop["w"])
            self.assertGreaterEqual(fx + fw, x - 2.0, "Fallback must keep subject head in crop")
            self.assertLessEqual(fx, x + w + 2.0, "Fallback must keep subject head in crop")


class TestConditionalSubjectTightFraming(unittest.TestCase):
    """
    Task §27: CV-based conditional subject-tight framing for NORMAL DualFrame
    panels. Validates:
    A. Already well-composed panel (body occupancy >= enter threshold) →
       crop IDENTICAL to the baseline solve (zero tightening).
    B. Background-heavy panel (small person body vs baseline crop) →
       moderate tightening: crop smaller than baseline, occupancy increased,
       bounded by MAX_TIGHTEN_SCALE, 9:8 + even dims preserved.
    C. Mixed panels: top well-framed (unchanged) + bottom background-heavy
       (tightened) → per-panel independence.
    D. Person-box jitter (±3px) → constant crop w, stable x (no zoom/crop
       oscillation from CV noise).
    E. Genuine subject movement with body evidence → camera tracks in x,
       crop w stays stable (tightening is not a second camera tracker).
    F. Contamination WITH body evidence → sealed isolation activates
       exactly as without body evidence (isolation decision independent of
       body data; tightening never stacks on isolation).
    G. Isolation regression guard: sealed fixture exact values hold.
    H. No body evidence (face-only, no person_bbox) → pass-through: crop
       identical to the pre-existing behavior (guards all sealed fixtures).
    """

    SRC_W, SRC_H = 1920.0, 1080.0

    def _face(self, cx, cy=300.0, face_w=160.0, face_h=180.0, conf=0.95, gaze=0.0, person_w=None):
        face = {
            "bbox": (cx - face_w / 2.0, cy - face_h / 2.0, face_w, face_h),
            "center": (cx, cy),
            "conf": conf,
            "area": face_w * face_h,
            "frontality": 0.90,
            "gaze_offset_x": gaze,
            "eye_line_y": cy - 30.0,
            "head_top_y": cy - face_h / 2.0 - 0.22 * face_h,
        }
        if person_w is not None:
            # YOLO person body: corner format (px1, py1, px2, py2), body under the head
            ph = person_w * 2.0
            py1 = cy - face_h / 2.0
            face["person_bbox"] = (cx - person_w / 2.0, py1, cx + person_w / 2.0, py1 + ph)
        return face

    def _track(self, track_id, faces_with_times):
        dets = [(t, f, f["conf"]) for (t, f) in faces_with_times]
        last_face = faces_with_times[-1][1]
        return {
            "track_id": track_id,
            "cx": last_face["center"][0],
            "cy": last_face["center"][1],
            "detections": dets,
            "age_sec": 2.0,
            "provenance": "yolo_person",
            "has_person_body": True,
            "is_occluded": False,
            "is_back": False,
            "visual_prominence": 0.80,
        }

    def _assert_9_8_even(self, crop, role):
        w, h = int(crop["w"]), int(crop["h"])
        self.assertEqual(w % 2, 0, f"{role} crop_w must be even, got {w}")
        self.assertEqual(h % 2, 0, f"{role} crop_h must be even, got {h}")
        self.assertLessEqual(abs(w - round(h * 1.125)), 1, f"{role} crop must be 9:8, got {w}x{h}")

    def _eval_x(self, crop, t):
        """Evaluates a constant or if()-tree crop x expression at time t."""
        expr = crop["x"].strip()
        try:
            return float(expr)
        except ValueError:
            pass
        import re

        def ev(s, pos):
            m = re.match(r"if\(lt\(t,(-?\d+\.?\d*)\),", s[pos:])
            if m:
                thresh = float(m.group(1))
                pos += m.end()
                a, pos = ev(s, pos)
                assert s[pos] == ","
                b, pos = ev(s, pos + 1)
                assert s[pos] == ")"
                return (a if t < thresh else b), pos + 1
            if s[pos] == "(":
                v, pos = ev(s, pos + 1)
                while pos < len(s) and s[pos] in "+-*/":
                    op = s[pos]
                    rhs, pos = ev(s, pos + 1)
                    v = v + rhs if op == "+" else v - rhs if op == "-" else v * rhs if op == "*" else v / rhs
                assert s[pos] == ")"
                return v, pos + 1
            if s[pos] == "t":
                return t, pos + 1
            m = re.match(r"-?\d+\.?\d*", s[pos:])
            return float(m.group(0)), pos + m.end()

        v, _ = ev(expr, 0)
        return v

    # ── A: well-composed → zero tightening (true pass-through) ────────────

    def test_A_well_composed_panel_kept_unchanged(self):
        # 160px face → baseline scale 1.12 → crop 1084x964. Person body 500px
        # wide → occupancy 500/1084 = 0.461 >= 0.32 → ALREADY WELL COMPOSED.
        tr = self._track(1, [(0.0, self._face(400.0, person_w=500.0)),
                              (1.0, self._face(400.0, person_w=500.0)),
                              (2.0, self._face(400.0, person_w=500.0))])
        res_body = solve_dual_frame_trajectories(tr, None, source_w=self.SRC_W, source_h=self.SRC_H)
        # Baseline: identical fixture WITHOUT person_bbox (pre-existing behavior)
        tr_nobody = self._track(1, [(0.0, self._face(400.0)),
                                    (1.0, self._face(400.0)),
                                    (2.0, self._face(400.0))])
        res_base = solve_dual_frame_trajectories(tr_nobody, None, source_w=self.SRC_W, source_h=self.SRC_H)
        self.assertEqual(res_body.top_crop["w"], res_base.top_crop["w"],
                         "Well-composed panel must keep baseline crop width")
        self.assertEqual(res_body.top_crop["h"], res_base.top_crop["h"])
        self.assertEqual(res_body.top_crop["x"], res_base.top_crop["x"])
        self.assertEqual(res_body.top_crop["y"], res_base.top_crop["y"])
        self.assertEqual(int(res_body.top_crop["w"]), 1084, "Baseline 160px-face crop must be 1084")
        self._assert_9_8_even(res_body.top_crop, "top")

    # ── B: background-heavy → moderate tightening ─────────────────────────

    def test_B_background_heavy_panel_tightened_moderately(self):
        # 160px face → baseline 1084x964. Person body 300px wide →
        # occupancy 300/1084 = 0.277 < 0.32 → background-heavy. Tightening
        # searches to target occupancy 0.44 (crop_w ≈ 682) but the vertical
        # feasibility (meaningful torso preserved) caps it earlier; the
        # bounded best-feasible scale is applied (moderate, ≤ 1.50).
        tr = self._track(1, [(0.0, self._face(400.0, person_w=300.0)),
                              (1.0, self._face(400.0, person_w=300.0)),
                              (2.0, self._face(400.0, person_w=300.0))])
        res = solve_dual_frame_trajectories(tr, None, source_w=self.SRC_W, source_h=self.SRC_H)
        w = int(res.top_crop["w"])
        self.assertLess(w, 1084, "Background-heavy panel must tighten (crop narrower than baseline)")
        self.assertGreaterEqual(w, 660, "Tightening must stay bounded (never past MAX_TIGHTEN_SCALE)")
        self._assert_9_8_even(res.top_crop, "top")
        # Occupancy improved
        self.assertGreaterEqual(300.0 / w, 300.0 / 1084.0,
                                "Tightened crop must increase body occupancy")
        # Moderate: not face-only, not max zoom — crop must still be substantial
        self.assertGreater(w, 500, "Tightening must be moderate (preserve context, not face-only)")
        # Subject head stays contained
        face = tr["detections"][0][1]
        fx, fy, fw, fh = face["bbox"]
        x = self._eval_x(res.top_crop, 0.0)
        self.assertGreaterEqual(fx + fw, x - 2.0, "Subject head must stay in crop")
        self.assertLessEqual(fx, x + float(res.top_crop["w"]) + 2.0, "Subject head must stay in crop")

    # ── C: mixed panels → per-panel independence ─────────────────────────

    def test_C_mixed_panels_independent_decisions(self):
        # Top: well-framed (person 500 → occupancy 0.461) → unchanged.
        # Bottom: background-heavy (person 300 → occupancy 0.277) → tightened.
        # Speakers far apart (400 vs 1400) → clean, no isolation.
        tr_top = self._track(1, [(0.0, self._face(400.0, person_w=500.0)),
                                 (1.0, self._face(400.0, person_w=500.0)),
                                 (2.0, self._face(400.0, person_w=500.0))])
        tr_bot = self._track(2, [(0.0, self._face(1400.0, person_w=300.0)),
                                 (1.0, self._face(1400.0, person_w=300.0)),
                                 (2.0, self._face(1400.0, person_w=300.0))])
        res = solve_dual_frame_trajectories(tr_top, tr_bot, source_w=self.SRC_W, source_h=self.SRC_H)
        self.assertEqual(int(res.top_crop["w"]), 1084,
                         "Well-framed top panel must keep baseline (independent of bottom)")
        self.assertLess(int(res.bottom_crop["w"]), 1084,
                        "Background-heavy bottom panel must tighten (independent of top)")
        self._assert_9_8_even(res.top_crop, "top")
        self._assert_9_8_even(res.bottom_crop, "bottom")

    # ── D: person-box jitter → no zoom/crop oscillation ───────────────────

    def test_D_person_box_jitter_no_crop_oscillation(self):
        # Body width jitters ±3px around 300px (sub-threshold CV noise).
        # The median-based decision + quantized scale grid must produce the
        # SAME crop as the un-jittered fixture: constant w, stable x.
        widths_a = [297.0, 303.0, 300.0, 298.0, 302.0, 299.0, 301.0, 300.0, 298.0, 302.0]
        widths_b = [300.0, 300.0, 300.0, 300.0, 300.0, 300.0, 300.0, 300.0, 300.0, 300.0]
        faces_a = [(float(t), self._face(400.0, person_w=widths_a[t])) for t in range(10)]
        faces_b = [(float(t), self._face(400.0, person_w=widths_b[t])) for t in range(10)]
        res_a = solve_dual_frame_trajectories(self._track(1, faces_a), None, source_w=self.SRC_W, source_h=self.SRC_H)
        res_b = solve_dual_frame_trajectories(self._track(1, faces_b), None, source_w=self.SRC_W, source_h=self.SRC_H)
        self.assertEqual(res_a.top_crop["w"], res_b.top_crop["w"],
                         "±3px body jitter must not change the crop width")
        self.assertEqual(res_a.top_crop["x"], res_b.top_crop["x"],
                         "±3px body jitter must not move the camera")
        self.assertEqual(res_a.top_crop["h"], res_b.top_crop["h"])
        # Static subject → constant x (no expression)
        try:
            float(res_a.top_crop["x"])
        except ValueError:
            self.fail("Static subject with jittery body must keep a constant crop x")

    # ── E: genuine movement → camera tracks, w stable ─────────────────────

    def test_E_movement_camera_tracks_w_stable(self):
        # Subject genuinely moves 30px/s with body evidence. The camera must
        # follow in x (expression contains t) while the crop width stays
        # constant (tightening decided once per panel, not per keyframe).
        faces = [(float(t), self._face(400.0 + 30.0 * t, person_w=300.0)) for t in range(10)]
        tr = self._track(1, faces)
        res = solve_dual_frame_trajectories(tr, None, source_w=self.SRC_W, source_h=self.SRC_H)
        xs = [self._eval_x(res.top_crop, float(t)) for t in range(10)]
        total = max(xs) - min(xs)
        self.assertGreaterEqual(total, 60.0,
                                f"Camera must follow genuine movement; got range {total:.2f}px")
        self.assertGreater(xs[-1], xs[0], "Camera must track in the movement direction")
        # Crop width constant across the whole segment
        self.assertIn("t", res.top_crop["x"] if "t" in res.top_crop["x"] else "",
                      "Moving subject must produce a time-based x expression")
        # w is a plain constant string
        self.assertTrue(res.top_crop["w"].strip().isdigit(),
                        "Crop width must be a constant (no per-keyframe zoom)")

    # ── F: contamination + body → isolation wins, tightening skipped ──────

    def test_F_contamination_isolation_wins_over_tightening(self):
        # Sealed test_02 geometry (680/1240, contamination) WITH person
        # bodies. Isolation must activate exactly as without bodies: the
        # isolation decision is independent of body data, and tightening
        # never stacks on isolation.
        tr_top_body = self._track(1, [(0.0, self._face(680.0, person_w=300.0)),
                                       (1.0, self._face(680.0, person_w=300.0)),
                                       (2.0, self._face(680.0, person_w=300.0))])
        tr_bot_body = self._track(2, [(0.0, self._face(1240.0, person_w=300.0)),
                                      (1.0, self._face(1240.0, person_w=300.0)),
                                      (2.0, self._face(1240.0, person_w=300.0))])
        tr_top_nb = self._track(1, [(0.0, self._face(680.0)),
                                    (1.0, self._face(680.0)),
                                    (2.0, self._face(680.0))])
        tr_bot_nb = self._track(2, [(0.0, self._face(1240.0)),
                                    (1.0, self._face(1240.0)),
                                    (2.0, self._face(1240.0))])
        res_body = solve_dual_frame_trajectories(tr_top_body, tr_bot_body, source_w=self.SRC_W, source_h=self.SRC_H)
        res_nb = solve_dual_frame_trajectories(tr_top_nb, tr_bot_nb, source_w=self.SRC_W, source_h=self.SRC_H)
        for role, crop_b, crop_n in (("top", res_body.top_crop, res_nb.top_crop),
                                     ("bottom", res_body.bottom_crop, res_nb.bottom_crop)):
            self.assertEqual(crop_b["w"], crop_n["w"],
                             f"{role}: isolation crop must be independent of body evidence")
            self.assertEqual(crop_b["x"], crop_n["x"],
                             f"{role}: isolation placement must be independent of body evidence")
            self.assertEqual(crop_b["h"], crop_n["h"])
        # Isolation genuinely active: crops tighter than baseline 1084
        self.assertLess(int(res_body.top_crop["w"]), 1084, "Isolation must be active on the contaminated pair")
        self.assertLess(int(res_body.bottom_crop["w"]), 1084)

    # ── G: isolation regression guard (sealed exact values) ───────────────

    def test_G_isolation_regression_guard(self):
        # Re-assert the sealed fixture exact values from
        # TestCloseSpeakerPanelIsolation (the §26 sealed system must remain
        # behaviorally identical after §27).
        # Separated speakers → baseline 1084, no zoom
        tr_top = self._track(1, [(0.0, self._face(300.0)), (1.0, self._face(300.0)), (2.0, self._face(300.0))])
        tr_bot = self._track(2, [(0.0, self._face(1500.0)), (1.0, self._face(1500.0)), (2.0, self._face(1500.0))])
        res = solve_dual_frame_trajectories(tr_top, tr_bot, source_w=self.SRC_W, source_h=self.SRC_H)
        self.assertEqual(int(res.top_crop["w"]), 1084)
        self.assertEqual(int(res.bottom_crop["w"]), 1084)
        # Single speaker → bottom fallback 1216x1080
        res_single = solve_dual_frame_trajectories(tr_top, None, source_w=self.SRC_W, source_h=self.SRC_H)
        self.assertEqual(int(res_single.bottom_crop["w"]), 1216)
        self.assertEqual(int(res_single.bottom_crop["h"]), 1080)
        self.assertEqual(int(res_single.top_crop["w"]), 1084)
        # Very close speakers → bounded zoom, never past MAX_ISOLATION_SCALE
        tr_close_top = self._track(1, [(0.0, self._face(900.0)), (1.0, self._face(900.0)), (2.0, self._face(900.0))])
        tr_close_bot = self._track(2, [(0.0, self._face(1020.0)), (1.0, self._face(1020.0)), (2.0, self._face(1020.0))])
        res_close = solve_dual_frame_trajectories(tr_close_top, tr_close_bot, source_w=self.SRC_W, source_h=self.SRC_H)
        self.assertGreaterEqual(int(res_close.top_crop["w"]), 660)
        self.assertGreaterEqual(int(res_close.bottom_crop["w"]), 660)

    # ── H: no body evidence → pass-through (guards sealed fixtures) ───────

    def test_H_no_body_evidence_pass_through(self):
        # Face-only track (no person_bbox anywhere — the exact shape of all
        # 47 sealed fixtures): the crop must be IDENTICAL to the pre-existing
        # behavior. This is the structural guard that §27 cannot regress any
        # sealed test.
        tr = self._track(1, [(0.0, self._face(400.0)),
                             (1.0, self._face(400.0)),
                             (2.0, self._face(400.0))])
        res = solve_dual_frame_trajectories(tr, None, source_w=self.SRC_W, source_h=self.SRC_H)
        self.assertEqual(int(res.top_crop["w"]), 1084, "No-body fixture must keep the sealed 1084 baseline")
        self.assertEqual(int(res.top_crop["h"]), 964)
        self._assert_9_8_even(res.top_crop, "top")


if __name__ == "__main__":
    unittest.main()
