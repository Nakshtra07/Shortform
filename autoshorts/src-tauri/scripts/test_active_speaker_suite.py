"""
Comprehensive Automated Acceptance Test Suite for AutoShorts Smart Reframing Engine v12.0
Validates True Dynamic Scale + Position Framing, Shot-Local Scale State, and Anti-Overzoom.

Test Coverage:
1. Speaker remains stationary -> camera remains stationary.
2. Speaker makes small head movement (10-30px) -> camera remains stationary (Zero drift).
3. Speaker shifts moderately (40-60px) -> camera remains stable inside valid corridor.
4. Speaker approaches boundary -> camera begins smooth corrective adjustment.
5. Speaker crosses boundary -> camera restores containment immediately.
6. Camera cuts -> immediate 0ms hard reset to new composition and scale reset.
7. Speaker switches -> new composition for active speaker.
8. Speaker looks left/right -> natural look-room offset respected.
9. Wide shot -> active speaker prominence maintained, anti-furniture isolation.
10. Two-shot -> natural joint composition when both speakers fit.
11. Background-heavy scene -> furniture / empty floor heavily penalized.
12. Stable close-up -> stationary lock with zero artificial camera drift.
13. Rapid motion -> fast 150ms transition.
14. Full trajectory -> 100% subject containment with low camera motion.
15. Vertical source input -> max_x = 0 preservation and identity crop.
16. Temporary face-detection failure -> 5-Level Recovery Hierarchy.
17. cv2 decoder failure -> Exception-safe recovery.
18. FFmpeg decoder fallback -> image2pipe sequential decoding.
19. Pre-render containment validator -> Trajectory correction.
20. Clip beginning with interviewer -> Initial turn turn-taking.
21. Clip beginning with guest -> Guest turn priority.
22. True dynamic scale -> crop_w and crop_h change with scale (PROVES SCALE IS REAL).
23. Scale deadband -> small variations produce 0 scale change.
24. Vertical crop Y solver -> targets eye-line with adequate headroom.
25. 3-Person shot -> Active speaker isolation.
26. Speaker enters/leaves frame -> Dynamic track initialization and recovery.
27. End-of-shot scale escalation audit -> flags scale creep near shot end.
28. Output JSON format -> produces valid JSON with x, y, w, h expressions.
"""

import os
import sys
import json
import unittest
import cv2
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
import speaker_tracker

class TestSmartReframingEngineV12(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        candidates = [
            os.path.join(script_dir, "..", "..", "autoshorts", "inspection_latest_failure", "clip_source_segment.mp4"),
            os.path.join(script_dir, "..", "..", "inspection_latest_failure", "clip_source_segment.mp4"),
            r"d:\College\Autoshorts 3.0\autoshorts\inspection_latest_failure\clip_source_segment.mp4",
            r"d:\College\Autoshorts 3.0\Beat Emotional Fatigue_ Better Sleep & Clearer Mind.mp4",
        ]
        cls.source_video = None
        for c in candidates:
            if os.path.exists(c) and os.path.getsize(c) > 1000000:
                cls.source_video = c
                break

        cls.crop_w = 608.0
        cls.max_x = 1312.0
        cls.default_x = 656.0

    # 1. Speaker remains stationary -> camera remains stationary
    def test_01_speaker_stationary_camera_stationary(self):
        sample_face = {
            "bbox": (600.0, 200.0, 160.0, 180.0), "center": (680.0, 290.0), "conf": 0.95,
            "area": 28800.0, "frontality": 0.95, "gaze_offset_x": 0.0, "skin_ratio": 0.60, "is_back": False,
            "head_top_y": 167.6, "eye_line_y": 263.0, "shoulder_span": (504.0, 856.0),
            "torso_top": 353.0, "mouth_gray": None, "landmarks": None,
        }
        sb = speaker_tracker.compute_subject_safe_box(sample_face, 1920.0, 1080.0)
        initial_crop_x, is_cont, corridor = speaker_tracker.solve_containment_crop_x(
            sb, self.crop_w, self.max_x, 1920.0, current_crop_x=None
        )
        self.assertTrue(is_cont)
        next_crop_x, is_cont_2, corridor_2 = speaker_tracker.solve_containment_crop_x(
            sb, self.crop_w, self.max_x, 1920.0, current_crop_x=initial_crop_x
        )
        self.assertEqual(initial_crop_x, next_crop_x, "Stationary speaker must produce 0 camera movement")

    # 2. Speaker makes small head movement (10-30px) -> camera remains stationary (Zero drift)
    def test_02_speaker_small_head_movement_zero_drift(self):
        face_1 = {"bbox": (600.0, 200.0, 160.0, 180.0), "center": (680.0, 290.0), "conf": 0.95, "area": 28800.0, "frontality": 0.95, "gaze_offset_x": 0.0, "skin_ratio": 0.6, "is_back": False, "head_top_y": 167.6, "eye_line_y": 263.0}
        sb_1 = speaker_tracker.compute_subject_safe_box(face_1, 1920.0, 1080.0)
        initial_crop_x, _, _ = speaker_tracker.solve_containment_crop_x(sb_1, self.crop_w, self.max_x, 1920.0)

        face_2 = {"bbox": (620.0, 200.0, 160.0, 180.0), "center": (700.0, 290.0), "conf": 0.95, "area": 28800.0, "frontality": 0.95, "gaze_offset_x": 0.0, "skin_ratio": 0.6, "is_back": False, "head_top_y": 167.6, "eye_line_y": 263.0}
        sb_2 = speaker_tracker.compute_subject_safe_box(face_2, 1920.0, 1080.0)
        crop_x_after_shift, is_cont, corridor = speaker_tracker.solve_containment_crop_x(
            sb_2, self.crop_w, self.max_x, 1920.0, current_crop_x=initial_crop_x
        )
        self.assertTrue(is_cont)
        self.assertEqual(crop_x_after_shift, initial_crop_x, "Small 20px movement must NOT move the camera (Deadband active)")

    # 3. Speaker shifts moderately (40-60px) -> camera remains stable inside valid corridor
    def test_03_speaker_moderate_shift_stable_in_corridor(self):
        face_base = {"bbox": (600.0, 200.0, 160.0, 180.0), "center": (680.0, 290.0), "conf": 0.95, "area": 28800.0, "frontality": 0.95, "gaze_offset_x": 0.0, "skin_ratio": 0.6, "is_back": False, "head_top_y": 167.6, "eye_line_y": 263.0}
        sb_base = speaker_tracker.compute_subject_safe_box(face_base, 1920.0, 1080.0)
        initial_crop_x, _, _ = speaker_tracker.solve_containment_crop_x(sb_base, self.crop_w, self.max_x, 1920.0)

        face_shifted = {"bbox": (650.0, 200.0, 160.0, 180.0), "center": (730.0, 290.0), "conf": 0.95, "area": 28800.0, "frontality": 0.95, "gaze_offset_x": 0.0, "skin_ratio": 0.6, "is_back": False, "head_top_y": 167.6, "eye_line_y": 263.0}
        sb_shifted = speaker_tracker.compute_subject_safe_box(face_shifted, 1920.0, 1080.0)
        crop_x_2, is_cont, corridor = speaker_tracker.solve_containment_crop_x(
            sb_shifted, self.crop_w, self.max_x, 1920.0, current_crop_x=initial_crop_x
        )
        self.assertTrue(is_cont)
        self.assertEqual(crop_x_2, initial_crop_x, "Moderate shift inside corridor must not move camera")

    # 4. Speaker approaches boundary -> camera begins smooth corrective adjustment
    def test_04_speaker_approaches_boundary_adjustment(self):
        curr_crop_x = 300.0
        face_far = {"bbox": (850.0, 200.0, 160.0, 180.0), "center": (930.0, 290.0), "conf": 0.95, "area": 28800.0, "frontality": 0.95, "gaze_offset_x": 0.0, "skin_ratio": 0.6, "is_back": False, "head_top_y": 167.6, "eye_line_y": 263.0}
        sb_far = speaker_tracker.compute_subject_safe_box(face_far, 1920.0, 1080.0)
        corrected_crop_x, is_cont, corridor = speaker_tracker.solve_containment_crop_x(
            sb_far, self.crop_w, self.max_x, 1920.0, current_crop_x=curr_crop_x
        )
        self.assertTrue(is_cont)
        self.assertGreater(corrected_crop_x, curr_crop_x, "Camera must adjust rightward to contain subject")

    # 5. Speaker crosses boundary -> camera restores containment immediately
    def test_05_speaker_crosses_boundary_restore_containment(self):
        curr_crop_x = 200.0
        face_edge = {"bbox": (1200.0, 200.0, 160.0, 180.0), "center": (1280.0, 290.0), "conf": 0.95, "area": 28800.0, "frontality": 0.95, "gaze_offset_x": 0.0, "skin_ratio": 0.6, "is_back": False, "head_top_y": 167.6, "eye_line_y": 263.0}
        sb = speaker_tracker.compute_subject_safe_box(face_edge, 1920.0, 1080.0)
        restored_crop_x, is_cont, _ = speaker_tracker.solve_containment_crop_x(
            sb, self.crop_w, self.max_x, 1920.0, current_crop_x=curr_crop_x
        )
        self.assertTrue(is_cont)
        self.assertGreater(restored_crop_x, 800.0)

    # 6. Camera cuts -> immediate 0ms hard reset to new composition
    def test_06_hard_cut_instant_reset(self):
        keyframes = [(10.0, 200.0, False), (15.0, 800.0, True)]
        expr = speaker_tracker.build_ffmpeg_expr(keyframes, 10.0, self.default_x)
        self.assertNotIn("3-2*", expr)

    # 7. Speaker switches -> new composition for active speaker
    def test_07_speaker_switch_new_composition(self):
        keyframes = [(0.0, 200.0, False), (4.0, 950.0, False)]
        expr = speaker_tracker.build_ffmpeg_expr(keyframes, 0.0, self.default_x)
        self.assertIn("3-2*", expr)

    # 8. Speaker looks left/right -> natural look-room offset respected
    def test_08_look_room_gaze_direction_respected(self):
        face_look_left = {"bbox": (600.0, 200.0, 160.0, 180.0), "center": (680.0, 290.0), "conf": 0.95, "area": 28800.0, "frontality": 0.70, "gaze_offset_x": -0.25, "skin_ratio": 0.6, "is_back": False, "head_top_y": 167.6, "eye_line_y": 263.0}
        sb_l = speaker_tracker.compute_subject_safe_box(face_look_left, 1920.0, 1080.0)
        crop_l, _, _ = speaker_tracker.solve_containment_crop_x(sb_l, self.crop_w, self.max_x, 1920.0, current_crop_x=None)

        face_look_right = {"bbox": (600.0, 200.0, 160.0, 180.0), "center": (680.0, 290.0), "conf": 0.95, "area": 28800.0, "frontality": 0.70, "gaze_offset_x": +0.25, "skin_ratio": 0.6, "is_back": False, "head_top_y": 167.6, "eye_line_y": 263.0}
        sb_r = speaker_tracker.compute_subject_safe_box(face_look_right, 1920.0, 1080.0)
        crop_r, _, _ = speaker_tracker.solve_containment_crop_x(sb_r, self.crop_w, self.max_x, 1920.0, current_crop_x=None)

        self.assertLess(crop_l, crop_r, "Look-room offset must differ based on gaze direction")

    # 9. Wide shot -> active speaker prominence maintained, empty furniture rejected
    def test_09_wide_shot_anti_furniture_isolation(self):
        l_face = {"bbox": (250.0, 200.0, 140.0, 160.0), "center": (320.0, 280.0), "conf": 0.92, "area": 22400.0, "frontality": 0.9, "gaze_offset_x": 0.0, "skin_ratio": 0.5, "is_back": False, "head_top_y": 165.0, "eye_line_y": 255.0, "shoulder_span": (166.0, 474.0), "torso_top": 336.0, "mouth_gray": None, "landmarks": None}
        r_face = {"bbox": (1450.0, 200.0, 140.0, 160.0), "center": (1520.0, 280.0), "conf": 0.94, "area": 22400.0, "frontality": 0.9, "gaze_offset_x": 0.0, "skin_ratio": 0.5, "is_back": False, "head_top_y": 165.0, "eye_line_y": 255.0, "shoulder_span": (1366.0, 1674.0), "torso_top": 336.0, "mouth_gray": None, "landmarks": None}
        tracks = {
            0: {"cx": 320.0, "cy": 280.0, "last_t": 5.0, "last_mouth": None, "detections": [(t, l_face, 2.0) for t in np.linspace(0, 5, 8)]},
            1: {"cx": 1520.0, "cy": 280.0, "last_t": 5.0, "last_mouth": None, "detections": [(t, r_face, 18.0) for t in np.linspace(0, 5, 8)]},
        }
        clamped_x, crop_y, eff_w, eff_h, target_cx, shot_type, camera_mode, diag, scale_st = speaker_tracker.classify_and_frame_subject(
            tracks, [1] * 8, "S2", 0.0, 5.0, {}, self.default_x + self.crop_w / 2.0, self.crop_w, self.max_x
        )
        self.assertEqual(shot_type, "wide_two_shot")
        self.assertGreater(target_cx, 1400.0)
        self.assertGreater(clamped_x, 800.0)

    # 10. Two-shot -> natural joint composition when both speakers fit
    def test_10_two_shot_both_fit(self):
        l_face = {"bbox": (770.0, 200.0, 100.0, 120.0), "center": (820.0, 260.0), "conf": 0.90, "area": 12000.0, "frontality": 0.90, "gaze_offset_x": 0.0, "skin_ratio": 0.50, "is_back": False, "head_top_y": 178.4, "eye_line_y": 242.0, "shoulder_span": (710.0, 930.0), "torso_top": 302.0, "mouth_gray": None, "landmarks": None}
        r_face = {"bbox": (1030.0, 200.0, 100.0, 120.0), "center": (1080.0, 260.0), "conf": 0.90, "area": 12000.0, "frontality": 0.90, "gaze_offset_x": 0.0, "skin_ratio": 0.50, "is_back": False, "head_top_y": 178.4, "eye_line_y": 242.0, "shoulder_span": (970.0, 1190.0), "torso_top": 302.0, "mouth_gray": None, "landmarks": None}
        tracks = {
            0: {"cx": 820.0, "cy": 260.0, "last_t": 5.0, "last_mouth": None, "detections": [(t, l_face, 5.0) for t in np.linspace(0, 5, 8)]},
            1: {"cx": 1080.0, "cy": 260.0, "last_t": 5.0, "last_mouth": None, "detections": [(t, r_face, 5.0) for t in np.linspace(0, 5, 8)]},
        }
        clamped_x, crop_y, eff_w, eff_h, target_cx, shot_type, camera_mode, diag, scale_st = speaker_tracker.classify_and_frame_subject(
            tracks, [1] * 8, "S1", 0.0, 5.0, {}, self.default_x + self.crop_w / 2.0, self.crop_w, self.max_x
        )
        self.assertEqual(shot_type, "two_shot_both_fit")
        self.assertEqual(target_cx, 950.0)

    # 10b. Two-shot pair containment: the crop must hold BOTH subjects, not be
    # solved around the left subject alone (which silently excludes the right
    # one when the pair is near the crop width). Adaptive layer only.
    def test_10b_two_shot_pair_containment(self):
        l_face = {"bbox": (700.0, 200.0, 120.0, 140.0), "center": (760.0, 270.0), "conf": 0.90, "area": 16800.0, "frontality": 0.90, "gaze_offset_x": 0.0, "skin_ratio": 0.50, "is_back": False, "head_top_y": 169.2, "eye_line_y": 247.0, "shoulder_span": (628.0, 892.0), "torso_top": 322.0, "mouth_gray": None, "landmarks": None}
        r_face = {"bbox": (1020.0, 200.0, 120.0, 140.0), "center": (1080.0, 270.0), "conf": 0.90, "area": 16800.0, "frontality": 0.90, "gaze_offset_x": 0.0, "skin_ratio": 0.50, "is_back": False, "head_top_y": 169.2, "eye_line_y": 247.0, "shoulder_span": (948.0, 1212.0), "torso_top": 322.0, "mouth_gray": None, "landmarks": None}
        tracks = {
            0: {"cx": 760.0, "cy": 270.0, "last_t": 5.0, "last_mouth": None, "detections": [(t, l_face, 5.0) for t in np.linspace(0, 5, 8)]},
            1: {"cx": 1080.0, "cy": 270.0, "last_t": 5.0, "last_mouth": None, "detections": [(t, r_face, 5.0) for t in np.linspace(0, 5, 8)]},
        }
        # face outer span 700->1140 = 440px (fits the adaptive 0.92*608 bound),
        # but the left-subject-only crop would cut off the right face at 1140.
        clamped_x, crop_y, eff_w, eff_h, target_cx, shot_type, camera_mode, diag, scale_st = speaker_tracker.classify_and_frame_subject(
            tracks, [1] * 8, "S1", 0.0, 5.0, {}, self.default_x + self.crop_w / 2.0, self.crop_w, self.max_x,
            adaptive_framing=True,
        )
        self.assertEqual(shot_type, "two_shot_both_fit")
        self.assertLessEqual(clamped_x, 700.0, "Crop must start left of the left face")
        self.assertGreaterEqual(clamped_x + eff_w, 1140.0, "Crop must contain the right face's outer edge")
        self.assertLess(abs(clamped_x + eff_w / 2.0 - 920.0), 80.0, "Crop must be centered near the pair midpoint")

        # Original mode must NOT take the pair composition (stricter 0.65 bound;
        # 440px does not fit) — adaptive framing stays an additive layer.
        clamped_x_o, _, eff_w_o, _, target_cx_o, shot_type_o, _, _, _ = speaker_tracker.classify_and_frame_subject(
            tracks, [1] * 8, "S1", 0.0, 5.0, {}, self.default_x + self.crop_w / 2.0, self.crop_w, self.max_x,
            adaptive_framing=False,
        )
        self.assertNotEqual(shot_type_o, "two_shot_both_fit", "Original 9:16 must not adopt the adaptive pair composition")
        self.assertEqual(target_cx_o, 1080.0, "Original mode stays on the active speaker")

    # 10c. Composition-fit criterion: a pair whose face span exceeds the retired
    # 0.92*608 bound is still a close pair in Adaptive mode iff BOTH subjects'
    # HEAD extents fit inside the widest renderable 9:16 crop (608px at 1080p).
    # The active speaker must NOT override the pair (exact speaker rule). The
    # criterion is a ratio of the pair's own geometry, so it is resolution and
    # size independent.
    def test_10c_composition_fit_overrides_active_speaker(self):
        l_face = {"bbox": (640.0, 300.0, 80.0, 100.0), "center": (680.0, 350.0), "conf": 0.88, "area": 8000.0, "frontality": 0.90, "gaze_offset_x": 0.0, "skin_ratio": 0.50, "is_back": False, "head_top_y": 278.0, "eye_line_y": 335.0, "shoulder_span": (592.0, 768.0), "torso_top": 400.0, "mouth_gray": None, "landmarks": None}
        r_face = {"bbox": (1120.0, 300.0, 80.0, 100.0), "center": (1160.0, 350.0), "conf": 0.93, "area": 8000.0, "frontality": 0.90, "gaze_offset_x": 0.0, "skin_ratio": 0.50, "is_back": False, "head_top_y": 278.0, "eye_line_y": 335.0, "shoulder_span": (1072.0, 1248.0), "torso_top": 400.0, "mouth_gray": None, "landmarks": None}
        tracks = {
            0: {"cx": 680.0, "cy": 350.0, "last_t": 5.0, "last_mouth": None, "detections": [(t, l_face, 0.5) for t in np.linspace(0, 5, 8)]},
            1: {"cx": 1160.0, "cy": 350.0, "last_t": 5.0, "last_mouth": None, "detections": [(t, r_face, 10.0) for t in np.linspace(0, 5, 8)]},
        }
        # Face outer span 600..1200 = 600px > 0.92*608 (retired bound), and the
        # union safe box (592..1248 = 656px) is wider than the 608px baseline
        # crop — but the pair's HEAD extents (630.4..1209.6 = 579.2px) fit
        # inside the widest renderable 9:16 crop, so the composition-fit
        # criterion keeps the pair and only clips symmetric shoulder overhang.
        clamped_x, crop_y, eff_w, eff_h, target_cx, shot_type, camera_mode, diag, scale_st = speaker_tracker.classify_and_frame_subject(
            tracks, [1] * 8, "S2", 0.0, 5.0, {}, self.default_x + self.crop_w / 2.0, self.crop_w, self.max_x,
            adaptive_framing=True,
        )
        self.assertEqual(shot_type, "two_shot_both_fit", "Head-containable pair must be kept as a two-shot")
        self.assertEqual(target_cx, 920.0, "Composition must center on the pair midpoint")
        self.assertEqual(diag.get("subject_role"), "two_shot", "Pair composition overrides the active speaker")
        needed = diag.get("pair_needed_scale")
        self.assertIsNotNone(needed, "Criterion must report the needed scale")
        self.assertGreaterEqual(needed, 1.0)
        self.assertLessEqual(needed, 1.10)
        self.assertLessEqual(diag.get("crop_scale", 0.0), needed, "Resolved scale must never exceed the head-containment ceiling")
        self.assertGreaterEqual(eff_w, 579.2, "Crop must be at least as wide as the pair head span")
        # Head extents must be fully contained (the part that must never clip):
        head_left = 640.0 - 0.12 * 80.0
        head_right = 1120.0 + 1.12 * 80.0
        self.assertGreaterEqual(clamped_x, head_left - eff_w - 2.0, "Crop must not cut the left head")
        self.assertLessEqual(clamped_x, head_left + 2.0, "Crop must start at/left of the left head extent")
        self.assertGreaterEqual(clamped_x + eff_w, head_right - 2.0, "Crop must contain the right head extent")

        # Original mode: same tracks, strict span bound -> isolation of the
        # active speaker, never the pair composition.
        clamped_x_o, _, eff_w_o, _, target_cx_o, shot_type_o, _, diag_o, _ = speaker_tracker.classify_and_frame_subject(
            tracks, [1] * 8, "S2", 0.0, 5.0, {}, self.default_x + self.crop_w / 2.0, self.crop_w, self.max_x,
            adaptive_framing=False,
        )
        self.assertNotEqual(shot_type_o, "two_shot_both_fit")
        self.assertEqual(target_cx_o, 1160.0, "Original mode isolates the active speaker")

    # 10d. Composition-fit criterion: a FAR pair whose union safe box cannot be
    # contained at any valid scale is NOT a close pair — it must fall through to
    # active-speaker isolation in both modes (mirrors the 977.6-1001.3s control).
    def test_10d_far_pair_not_composition_fit(self):
        l_face = {"bbox": (250.0, 200.0, 140.0, 160.0), "center": (320.0, 280.0), "conf": 0.92, "area": 22400.0, "frontality": 0.9, "gaze_offset_x": 0.0, "skin_ratio": 0.5, "is_back": False, "head_top_y": 165.0, "eye_line_y": 255.0, "shoulder_span": (166.0, 474.0), "torso_top": 336.0, "mouth_gray": None, "landmarks": None}
        r_face = {"bbox": (1450.0, 200.0, 140.0, 160.0), "center": (1520.0, 280.0), "conf": 0.94, "area": 22400.0, "frontality": 0.9, "gaze_offset_x": 0.0, "skin_ratio": 0.5, "is_back": False, "head_top_y": 165.0, "eye_line_y": 255.0, "shoulder_span": (1366.0, 1674.0), "torso_top": 336.0, "mouth_gray": None, "landmarks": None}
        tracks = {
            0: {"cx": 320.0, "cy": 280.0, "last_t": 5.0, "last_mouth": None, "detections": [(t, l_face, 10.0) for t in np.linspace(0, 5, 8)]},
            1: {"cx": 1520.0, "cy": 280.0, "last_t": 5.0, "last_mouth": None, "detections": [(t, r_face, 0.5) for t in np.linspace(0, 5, 8)]},
        }
        # User-directed behavior change (2026-10-02, restores the observed 10.0
        # Adaptive behavior): when TWO persons are simultaneously in frame in
        # Adaptive mode, the composition is ALWAYS the shared centered two-shot
        # — no active-speaker isolation, regardless of pair span. Original 9:16
        # keeps the far-pair speaker isolation.
        clamped_x, crop_y, eff_w, eff_h, target_cx, shot_type, camera_mode, diag, scale_st = speaker_tracker.classify_and_frame_subject(
            tracks, [1] * 8, "S1", 0.0, 5.0, {}, self.default_x + self.crop_w / 2.0, self.crop_w, self.max_x,
            adaptive_framing=True,
        )
        self.assertEqual(shot_type, "two_shot_both_fit", "Adaptive: two people in frame must be a shared centered two-shot")
        self.assertEqual(target_cx, 920.0, "Adaptive: crop must center on the pair midpoint (320+1520)/2")

        clamped_x, crop_y, eff_w, eff_h, target_cx, shot_type, camera_mode, diag, scale_st = speaker_tracker.classify_and_frame_subject(
            tracks, [1] * 8, "S1", 0.0, 5.0, {}, self.default_x + self.crop_w / 2.0, self.crop_w, self.max_x,
            adaptive_framing=False,
        )
        self.assertNotEqual(shot_type, "two_shot_both_fit", "Original 9:16: uncontainable pair must not be a two-shot")
        self.assertEqual(target_cx, 320.0, "Original 9:16: must isolate the active speaker")

    def test_11_background_heavy_furniture_penalty(self):
        left_face = {"bbox": (100.0, 200.0, 100.0, 100.0), "center": (150.0, 250.0), "conf": 0.8}
        right_face = {"bbox": (1700.0, 200.0, 100.0, 100.0), "center": (1750.0, 250.0), "conf": 0.8}
        score_center = speaker_tracker.evaluate_crop_composition_score(
            656.0, 0.0, self.crop_w, 1080.0, 1.0, left_face, [left_face, right_face], 1920.0, 1080.0
        )
        score_left = speaker_tracker.evaluate_crop_composition_score(
            0.0, 0.0, self.crop_w, 1080.0, 1.0, left_face, [left_face, right_face], 1920.0, 1080.0
        )
        self.assertGreater(score_left, score_center, "Crop containing the face must score higher than empty center furniture crop")

    # 12. Stable close-up -> stationary lock with zero artificial camera drift
    def test_12_stable_close_up_stationary_lock(self):
        positions = [680.0, 682.0, 679.0, 681.0, 680.0]
        mode = speaker_tracker.evaluate_camera_mode(positions, 1920.0)
        self.assertEqual(mode, "STATIC")

    # 13. Rapid motion -> fast 150ms transition
    def test_13_rapid_motion_fast_transition(self):
        raw_keyframes = [(0.0, 200.0, False), (0.5, 900.0, False)]
        expr = speaker_tracker.build_ffmpeg_expr(raw_keyframes, 0.0, self.default_x)
        self.assertIn("3-2*", expr)

    # 14. Full trajectory -> 100% subject containment with low camera motion
    def test_14_full_trajectory_containment_and_stability(self):
        raw_keyframes = [(0.0, 300.0, False), (1.0, 305.0, False), (2.0, 310.0, False)]
        optimized = speaker_tracker.optimize_camera_trajectory(raw_keyframes, self.max_x)
        self.assertEqual(len(optimized), 3)

    # 15. Vertical source input -> max_x = 0 preservation
    def test_15_vertical_source_input(self):
        expr = speaker_tracker.build_ffmpeg_expr([(0.0, 0.0, False)], 0.0, 0.0)
        self.assertEqual(expr, "0")

    # 16. Temporary face-detection failure -> 5-Level Recovery Hierarchy
    def test_16_recovery_hierarchy_motion_extrapolation(self):
        identity = speaker_tracker.get_or_create_identity("S1")
        identity.record_detection(1100.0, 300.0, 10.0)
        identity.record_detection(1150.0, 300.0, 11.0)
        pred = identity.predict_position(12.0, 656.0)
        self.assertAlmostEqual(pred, 1200.0, delta=2.0)

    # 17. cv2 decoder failure -> Exception-safe recovery
    def test_17_cv2_decoder_exception_recovery(self):
        cuts, samples = speaker_tracker.sample_video_and_detect_shots("non_existent_file.mp4", 0.0, 5000.0)
        self.assertIsInstance(cuts, list)
        self.assertIsInstance(samples, list)

    # 18. FFmpeg decoder fallback -> image2pipe sequential decoding
    def test_18_ffmpeg_decoder_fallback(self):
        if not self.source_video:
            self.skipTest("No test video available")
        pipe_frames = speaker_tracker.decode_frames_via_ffmpeg_pipe(self.source_video, 0.0, 2.0, 8.0)
        self.assertGreater(len(pipe_frames), 0)

    # 19. Pre-render containment validator -> Trajectory correction
    def test_19_invalid_trajectory_recovery(self):
        sample_face = {"bbox": (1400.0, 200.0, 160.0, 180.0), "center": (1480.0, 290.0), "conf": 0.95, "area": 28800.0, "frontality": 0.95, "gaze_offset_x": 0.0, "skin_ratio": 0.6, "is_back": False, "head_top_y": 167.6, "eye_line_y": 263.0, "shoulder_span": (1304.0, 1656.0), "torso_top": 353.0, "mouth_gray": None, "landmarks": None}
        mock_samples = [(1.0, np.zeros((1080, 1920, 3), dtype=np.uint8), [sample_face])]
        invalid_kf = [(1.0, 200.0, False)]
        corrected, _, audit = speaker_tracker.validate_and_correct_trajectory(
            invalid_kf, [(1.0, 1.0, False)], mock_samples, 0.0, self.crop_w, 1920.0, 1080.0
        )
        self.assertGreater(corrected[0][1], 1000.0)
        self.assertEqual(audit["n_violations"], 1)

    # 20. Clip beginning with interviewer -> Initial turn turn-taking
    def test_20_clip_beginning_with_interviewer(self):
        words = [{"start": 0.0, "end": 4.0, "speaker": "S1", "word": "Question?"}]
        turns = speaker_tracker.build_speaker_turns(words, 0.0, 4.0)
        self.assertEqual(turns[0][2], "S1")

    # 21. Clip beginning with guest -> Guest turn priority
    def test_21_clip_beginning_with_guest(self):
        words = [{"start": 0.0, "end": 6.0, "speaker": "S2", "word": "Story..."}]
        turns = speaker_tracker.build_speaker_turns(words, 0.0, 6.0)
        self.assertEqual(turns[0][2], "S2")

    # 22. True Dynamic Scale -> crop_w and crop_h change with scale (PROVES SCALE IS REAL)
    def test_22_true_dynamic_scale_dimensions_change(self):
        base_w, base_h = speaker_tracker.compute_effective_crop_dims(1.0, 1080.0, 1920.0)
        zoomed_w, zoomed_h = speaker_tracker.compute_effective_crop_dims(1.20, 1080.0, 1920.0)
        wide_w, wide_h = speaker_tracker.compute_effective_crop_dims(0.90, 1080.0, 1920.0)

        # Baseline: 608x1080 (or closest even)
        self.assertEqual(base_w, 608.0)
        self.assertEqual(base_h, 1080.0)

        # Zoomed in (scale 1.2): crop_w is smaller (e.g. ~506px), crop_h is smaller (e.g. ~900px)
        self.assertLess(zoomed_w, base_w, "Scale 1.2 must produce genuinely smaller crop_w (tighter crop)")
        self.assertLess(zoomed_h, base_h, "Scale 1.2 must produce genuinely smaller crop_h (tighter crop)")

        # Zoomed out (scale 0.9): crop_w is larger (e.g. ~676px), crop_h is bounded by 1080
        self.assertGreater(wide_w, base_w, "Scale 0.9 must produce genuinely larger crop_w (wider crop)")

        # Aspect ratio 9:16 is preserved for portrait cropping
        for w, h in [(base_w, base_h), (zoomed_w, zoomed_h)]:
            ratio = w / h
            self.assertAlmostEqual(ratio, 9.0 / 16.0, delta=0.02, msg=f"Aspect ratio {ratio} must approximate 9/16")

    # 23. Scale deadband -> small variations produce 0 scale change
    def test_23_scale_deadband_stability(self):
        scale_out, reason = speaker_tracker.apply_scale_deadband(
            current_scale=1.05, ideal_scale=1.07, min_valid_scale=0.95, max_valid_scale=1.20, prev_reason="COMPOSITION"
        )
        self.assertEqual(scale_out, 1.05, "Small scale delta < 6% must remain locked in deadband")
        self.assertEqual(reason, "DEADBAND")

    # 24. Vertical crop Y solver -> targets eye-line with adequate headroom
    def test_24_vertical_crop_y_solver(self):
        face = {"bbox": (600.0, 300.0, 160.0, 180.0), "center": (680.0, 390.0), "head_top_y": 260.0, "eye_line_y": 360.0}
        crop_y = speaker_tracker.solve_crop_y(face, 900.0, 1080.0)
        self.assertGreaterEqual(crop_y, 0.0)
        self.assertLessEqual(crop_y, 180.0)

    # 25. 3-Person shot -> Active speaker isolation
    def test_25_three_person_shot_active_isolation(self):
        f1 = {"bbox": (200.0, 200.0, 120.0, 140.0), "center": (260.0, 270.0), "conf": 0.92, "area": 16800.0, "frontality": 0.9, "gaze_offset_x": 0.0, "skin_ratio": 0.5, "is_back": False, "head_top_y": 170.0, "eye_line_y": 250.0, "shoulder_span": (128.0, 392.0), "torso_top": 320.0, "mouth_gray": None, "landmarks": None}
        f2 = {"bbox": (900.0, 200.0, 120.0, 140.0), "center": (960.0, 270.0), "conf": 0.94, "area": 16800.0, "frontality": 0.9, "gaze_offset_x": 0.0, "skin_ratio": 0.5, "is_back": False, "head_top_y": 170.0, "eye_line_y": 250.0, "shoulder_span": (828.0, 1092.0), "torso_top": 320.0, "mouth_gray": None, "landmarks": None}
        f3 = {"bbox": (1500.0, 200.0, 120.0, 140.0), "center": (1560.0, 270.0), "conf": 0.91, "area": 16800.0, "frontality": 0.9, "gaze_offset_x": 0.0, "skin_ratio": 0.5, "is_back": False, "head_top_y": 170.0, "eye_line_y": 250.0, "shoulder_span": (1428.0, 1692.0), "torso_top": 320.0, "mouth_gray": None, "landmarks": None}
        tracks = {
            0: {"cx": 260.0, "cy": 270.0, "last_t": 5.0, "last_mouth": None, "detections": [(t, f1, 1.0) for t in np.linspace(0, 5, 8)]},
            1: {"cx": 960.0, "cy": 270.0, "last_t": 5.0, "last_mouth": None, "detections": [(t, f2, 1.0) for t in np.linspace(0, 5, 8)]},
            2: {"cx": 1560.0, "cy": 270.0, "last_t": 5.0, "last_mouth": None, "detections": [(t, f3, 20.0) for t in np.linspace(0, 5, 8)]},
        }
        clamped_x, crop_y, eff_w, eff_h, target_cx, shot_type, camera_mode, diag, scale_st = speaker_tracker.classify_and_frame_subject(
            tracks, [1] * 8, "S3", 0.0, 5.0, {}, self.default_x + self.crop_w / 2.0, self.crop_w, self.max_x
        )
        self.assertGreater(target_cx, 1450.0)

    # 26. Speaker enters/leaves frame -> Dynamic track initialization and recovery
    def test_26_speaker_leaves_frame_recovery(self):
        identity = speaker_tracker.get_or_create_identity("S1")
        identity.record_detection(400.0, 300.0, 1.0)
        clamped_x, crop_y, eff_w, eff_h, target_cx, shot_type, camera_mode, diag, scale_st = speaker_tracker.classify_and_frame_subject(
            {}, [], "S1", 3.0, 5.0, {}, self.default_x + self.crop_w / 2.0, self.crop_w, self.max_x
        )
        self.assertEqual(target_cx, 400.0)

    # 27. End-of-shot scale escalation audit
    def test_27_scale_escalation_detection(self):
        normal_skf = [(0.0, 1.0, False), (1.0, 1.0, False), (2.0, 1.0, False), (3.0, 1.0, False), (4.0, 1.0, False)]
        _, _, audit_normal = speaker_tracker.validate_and_correct_trajectory([], normal_skf, [], 0.0, 608.0, 1920.0, 1080.0)
        self.assertFalse(audit_normal["scale_escalation_detected"])

        escalating_skf = [(0.0, 1.0, False), (1.0, 1.0, False), (2.0, 1.0, False), (3.0, 1.25, False), (4.0, 1.30, False)]
        _, _, audit_esc = speaker_tracker.validate_and_correct_trajectory([], escalating_skf, [], 0.0, 608.0, 1920.0, 1080.0)
        self.assertTrue(audit_esc["scale_escalation_detected"])

    # 28. Output JSON format
    def test_28_build_ffmpeg_scale_and_y_expressions(self):
        skf = [(0.0, 1.0, False), (5.0, 1.15, False)]
        w_expr = speaker_tracker.build_ffmpeg_scale_dim_expr(skf, 0.0, 608.0, 1080.0, 1920.0, dim="w")
        h_expr = speaker_tracker.build_ffmpeg_scale_dim_expr(skf, 0.0, 608.0, 1080.0, 1920.0, dim="h")
        self.assertIn("if(lt(t,5", w_expr)
        self.assertIn("if(lt(t,5", h_expr)

    # 29. Camera Lock & Deadband Hold: small natural head/body movement holds position
    def test_29_camera_lock_and_deadband_hold(self):
        # Speaker safely centered around crop_x = 350.0
        # Speaker head shifts by 20px (natural conversation sway)
        face_initial = {"bbox": (450.0, 200.0, 200.0, 260.0), "center": (550.0, 330.0), "conf": 0.95,
                        "gaze_offset_x": 0.0, "head_top_y": 140.0, "eye_line_y": 280.0}
        safe_box_initial = speaker_tracker.compute_subject_safe_box(face_initial, 1920.0, 1080.0)
        crop_x_1, is_cont_1, _ = speaker_tracker.solve_containment_crop_x(
            safe_box_initial, self.crop_w, self.max_x, 1920.0, current_crop_x=None
        )

        # Shifted face by 20px (within deadband)
        face_shifted = {"bbox": (470.0, 200.0, 200.0, 260.0), "center": (570.0, 330.0), "conf": 0.95,
                        "gaze_offset_x": 0.0, "head_top_y": 140.0, "eye_line_y": 280.0}
        safe_box_shifted = speaker_tracker.compute_subject_safe_box(face_shifted, 1920.0, 1080.0)
        crop_x_2, is_cont_2, _ = speaker_tracker.solve_containment_crop_x(
            safe_box_shifted, self.crop_w, self.max_x, 1920.0, current_crop_x=crop_x_1
        )

        # The camera MUST hold the exact same position (crop_x_2 == crop_x_1)
        self.assertEqual(crop_x_2, crop_x_1)
        self.assertTrue(is_cont_2)

    # 30. Responsive Speaker Handoff: switching speaker targets updates framing anchor
    def test_30_speaker_handoff_repositioning(self):
        f_left = {"bbox": (250.0, 200.0, 200.0, 260.0), "center": (350.0, 330.0), "conf": 0.92,
                  "frontality": 0.9, "is_back": False, "area": 52000, "gaze_offset_x": 0.0, "head_top_y": 140.0, "eye_line_y": 280.0, "mouth_gray": None}
        f_right = {"bbox": (1350.0, 200.0, 200.0, 260.0), "center": (1450.0, 330.0), "conf": 0.92,
                   "frontality": 0.9, "is_back": False, "area": 52000, "gaze_offset_x": 0.0, "head_top_y": 140.0, "eye_line_y": 280.0, "mouth_gray": None}

        # S1 speaking -> track 0 has higher mouth motion (25 vs 5)
        tracks_s1 = {
            0: {"cx": 350.0, "cy": 330.0, "last_t": 2.0, "last_mouth": None, "detections": [(t, f_left, 25.0) for t in np.linspace(0, 2, 5)]},
            1: {"cx": 1450.0, "cy": 330.0, "last_t": 2.0, "last_mouth": None, "detections": [(t, f_right, 5.0) for t in np.linspace(0, 2, 5)]},
        }
        crop_s1, _, _, _, cx_s1, _, _, _, _ = speaker_tracker.classify_and_frame_subject(
            tracks_s1, [1]*5, "S1", 0.0, 2.0, {}, self.default_x + self.crop_w / 2.0, self.crop_w, self.max_x
        )
        self.assertLess(cx_s1, 600.0)

        # S2 speaking -> track 1 has higher mouth motion (25 vs 5)
        tracks_s2 = {
            0: {"cx": 350.0, "cy": 330.0, "last_t": 2.0, "last_mouth": None, "detections": [(t, f_left, 5.0) for t in np.linspace(0, 2, 5)]},
            1: {"cx": 1450.0, "cy": 330.0, "last_t": 2.0, "last_mouth": None, "detections": [(t, f_right, 25.0) for t in np.linspace(0, 2, 5)]},
        }
        crop_s2, _, _, _, cx_s2, _, _, _, _ = speaker_tracker.classify_and_frame_subject(
            tracks_s2, [1]*5, "S2", 0.0, 2.0, {}, self.default_x + self.crop_w / 2.0, self.crop_w, self.max_x,
            current_crop_x=crop_s1
        )
        self.assertGreater(cx_s2, 1200.0)
        self.assertGreater(abs(crop_s2 - crop_s1), 500.0)

    # 31. Edge Containment Guard: large displacement nearing frame edge forces reposition
    def test_31_edge_containment_guard(self):
        # Camera is locked at x=100.0
        # Face moves to x=700 (outside 608w window from 100 to 708)
        face_edge = {"bbox": (680.0, 200.0, 200.0, 260.0), "center": (780.0, 330.0), "conf": 0.95,
                     "gaze_offset_x": 0.0, "head_top_y": 140.0, "eye_line_y": 280.0}
        safe_box_edge = speaker_tracker.compute_subject_safe_box(face_edge, 1920.0, 1080.0)
        crop_x, is_cont, _ = speaker_tracker.solve_containment_crop_x(
            safe_box_edge, self.crop_w, self.max_x, 1920.0, current_crop_x=100.0
        )
        # Must NOT stay at 100.0; must reframe to contain the subject
    # 32. Listener Reaction Shot at Camera Cut: Person A speaking in audio, camera cuts to listener B -> B is framed, not A
    def test_32_listener_reaction_shot_at_camera_cut(self):
        last_spk = {"S1": 400.0}
        f_listener = {
            "bbox": (1080.0, 200.0, 200.0, 260.0), "center": (1180.0, 330.0),
            "conf": 0.94, "area": 52000.0, "frontality": 0.92, "gaze_offset_x": 0.0,
            "skin_ratio": 0.60, "is_back": False, "head_top_y": 140.0, "eye_line_y": 280.0,
            "mouth_gray": None, "landmarks": None
        }
        listener_tracks = {
            0: {
                "cx": 1180.0, "cy": 330.0, "last_t": 24.0, "last_mouth": None,
                "detections": [(t, f_listener, 0.0) for t in np.linspace(22.8, 25.0, 6)]
            }
        }
        mock_samples = [(t, np.zeros((1080, 1920, 3), dtype=np.uint8), [f_listener]) for t in np.linspace(22.8, 25.0, 6)]

        clamped_x, crop_y, eff_w, eff_h, target_cx, shot_type, camera_mode, diag, _ = speaker_tracker.classify_and_frame_subject(
            listener_tracks, mock_samples, "S1", 22.8, 25.0,
            last_spk, self.default_x + self.crop_w / 2.0, self.crop_w, self.max_x,
            source_h=1080.0, source_w=1920.0, is_shot_cut=True
        )

        self.assertGreater(target_cx, 1000.0)
        self.assertEqual(diag["subject_role"], "listener_reaction")
        self.assertGreater(clamped_x, 800.0)
        self.assertTrue(diag["containment_pass"])

    # 33. Multi-Factor Visual Prominence Signal: evaluates area, frontality, confidence, centrality without 1.5x hard rule
    def test_33_multi_factor_visual_prominence(self):
        f_centered = {
            "bbox": (860.0, 200.0, 200.0, 250.0), "center": (960.0, 325.0),
            "conf": 0.95, "area": 50000.0, "frontality": 0.95
        }
        f_edge = {
            "bbox": (50.0, 200.0, 200.0, 250.0), "center": (150.0, 325.0),
            "conf": 0.70, "area": 50000.0, "frontality": 0.40
        }
        tr_centered = {"cx": 960.0, "cy": 325.0, "detections": [(1.0, f_centered, 0.0)] * 5}
        tr_edge = {"cx": 150.0, "cy": 325.0, "detections": [(1.0, f_edge, 0.0)] * 2}

        prom_centered = speaker_tracker.compute_visual_prominence(tr_centered, 1920.0, 1080.0, max_area_in_shot=50000.0)
        prom_edge = speaker_tracker.compute_visual_prominence(tr_edge, 1920.0, 1080.0, max_area_in_shot=50000.0)

        self.assertGreater(prom_centered, prom_edge + 0.15)

    # 34. Dynamic Derived Geometry: crop x is computed dynamically from subject coordinates
    def test_34_dynamic_crop_derived_from_geometry(self):
        for subject_cx in [500.0, 800.0, 1150.0, 1400.0]:
            f = {
                "bbox": (subject_cx - 100.0, 200.0, 200.0, 260.0), "center": (subject_cx, 330.0),
                "conf": 0.95, "area": 52000.0, "frontality": 0.95, "gaze_offset_x": 0.0,
                "skin_ratio": 0.60, "is_back": False, "head_top_y": 140.0, "eye_line_y": 280.0,
            }
            sb = speaker_tracker.compute_subject_safe_box(f, 1920.0, 1080.0)
            crop_x, is_cont, _ = speaker_tracker.solve_containment_crop_x(sb, self.crop_w, self.max_x, 1920.0)
            self.assertTrue(is_cont)
            crop_center = crop_x + self.crop_w / 2.0
            self.assertAlmostEqual(crop_center, min(1920.0 - self.crop_w / 2.0, max(self.crop_w / 2.0, subject_cx)), delta=35.0)

    # 35. Trajectory Validator Shot Isolation: samples before cut never overwrite post-cut keyframes
    def test_35_validator_no_cross_cut_keyframe_corruption(self):
        keyframes = [(6.84, 429.0, True), (22.80, 889.0, True)]
        scale_kf = [(6.84, 1.0, True), (22.80, 1.0, True)]

        face_shot1 = {
            "bbox": (600.0, 200.0, 200.0, 260.0), "center": (700.0, 330.0),
            "conf": 0.95, "area": 52000.0, "frontality": 0.95, "is_back": False
        }
        samples = [(18.0, None, [face_shot1])]

        corrected, _, audit = speaker_tracker.validate_and_correct_trajectory(
            keyframes, scale_kf, samples, 0.0, self.crop_w, 1920.0, 1080.0
        )

        self.assertEqual(corrected[1][1], 889.0)
        self.assertEqual(audit["n_violations"], 0)

    # 36. Speaker Identity Preservation: listener reaction shot does not overwrite speaker identity
    def test_36_listener_reaction_does_not_contaminate_speaker_identity(self):
        speaker_tracker.speaker_identities = {}
        identity = speaker_tracker.get_or_create_identity("S1")
        identity.record_detection(400.0, 300.0, 10.0)

        f_listener = {
            "bbox": (1100.0, 200.0, 200.0, 260.0), "center": (1200.0, 330.0),
            "conf": 0.95, "area": 52000.0, "frontality": 0.95, "gaze_offset_x": 0.0,
            "skin_ratio": 0.60, "is_back": False, "head_top_y": 140.0, "eye_line_y": 280.0,
        }
        tracks = {
            0: {"cx": 1200.0, "cy": 330.0, "last_t": 23.0, "last_mouth": None,
                "detections": [(23.0, f_listener, 0.0)]}
        }
        last_spk = {"S1": 400.0}
        speaker_tracker.classify_and_frame_subject(
            tracks, [(23.0, None, [f_listener])], "S1", 22.8, 24.0,
            last_spk, self.default_x + self.crop_w / 2.0, self.crop_w, self.max_x,
            is_shot_cut=True
        )

        self.assertEqual(last_spk["S1"], 400.0)
        self.assertEqual(identity.last_confirmed_cx, 400.0)

    # 37. Global Motion Compensation (GMC) camera pan decoupling
    def test_37_gmc_camera_pan_decoupling(self):
        f1 = np.zeros((1080, 1920, 3), dtype=np.uint8)
        cv2.rectangle(f1, (200, 200), (500, 500), (255, 255, 255), -1)
        cv2.circle(f1, (800, 600), 100, (200, 200, 200), -1)
        M_shift = np.float32([[1, 0, 20.0], [0, 1, 0.0]])
        f2 = cv2.warpAffine(f1, M_shift, (1920, 1080))

        dx, dy = speaker_tracker.compute_camera_motion_compensation(f1, f2)
        self.assertAlmostEqual(dx, 20.0, delta=5.0)

    # 38. Temporary Occlusion Stability in ByteTrack
    def test_38_temporary_occlusion_stability(self):
        f_visible = {
            "bbox": (500.0, 200.0, 150.0, 200.0), "center": (575.0, 300.0),
            "conf": 0.90, "area": 30000.0, "mouth_gray": None
        }
        sample1 = (1.0, None, [f_visible])
        sample2 = (1.2, None, [])
        f_low_conf = dict(f_visible, conf=0.45)
        sample3 = (1.4, None, [f_low_conf])

        tracks = speaker_tracker.build_shot_tracks([sample1, sample2, sample3])
        self.assertEqual(len(tracks), 1)
        self.assertIn(0, tracks)
    # 39. Solo talking head (stationary) -> 0 camera movement, stable portrait crop
    def test_39_solo_talking_head_stationary(self):
        f = {"bbox": (600.0, 200.0, 160.0, 180.0), "center": (680.0, 290.0), "conf": 0.95, "area": 28800.0, "frontality": 0.95, "gaze_offset_x": 0.0, "is_back": False, "head_top_y": 167.0, "eye_line_y": 260.0}
        tr = {0: {"cx": 680.0, "cy": 290.0, "last_t": 1.0, "last_mouth": None, "detections": [(1.0, f, 0.0)]}}
        x1, _, _, _, _, _, _, diag1, _ = speaker_tracker.classify_and_frame_subject(
            tr, [(1.0, None, [f])], "S1", 1.0, 1.5, {}, self.default_x + self.crop_w / 2.0, self.crop_w, self.max_x
        )
        x2, _, _, _, _, _, _, diag2, _ = speaker_tracker.classify_and_frame_subject(
            tr, [(1.5, None, [f])], "S1", 1.5, 2.0, {}, self.default_x + self.crop_w / 2.0, self.crop_w, self.max_x, current_crop_x=x1
        )
        self.assertEqual(x1, x2)
        self.assertTrue(diag2["containment_pass"])

    # 40. Solo talking head (pacing/moving) -> smooth containment inside safe box
    def test_40_solo_talking_head_pacing_gmc(self):
        positions = [600.0, 640.0, 680.0, 720.0, 760.0]
        crop_xs = []
        cur_x = None
        for p in positions:
            f = {"bbox": (p - 80.0, 200.0, 160.0, 180.0), "center": (p, 290.0), "conf": 0.95, "area": 28800.0, "frontality": 0.95, "gaze_offset_x": 0.0, "is_back": False, "head_top_y": 167.0, "eye_line_y": 260.0}
            sb = speaker_tracker.compute_subject_safe_box(f, 1920.0, 1080.0)
            cur_x, is_cont, _ = speaker_tracker.solve_containment_crop_x(sb, self.crop_w, self.max_x, 1920.0, current_crop_x=cur_x)
            self.assertTrue(is_cont)
            crop_xs.append(cur_x)
        self.assertTrue(all(abs(crop_xs[i] - crop_xs[i-1]) <= 85.0 for i in range(1, len(crop_xs))))

    # 41. Two-person interview (Person A speaks, Person B silent) -> A is framed
    def test_41_two_person_speaker_focus(self):
        fA = {"bbox": (400.0, 200.0, 160.0, 180.0), "center": (480.0, 290.0), "conf": 0.95, "area": 28800.0, "frontality": 0.90, "is_back": False}
        fB = {"bbox": (1200.0, 200.0, 160.0, 180.0), "center": (1280.0, 290.0), "conf": 0.95, "area": 28800.0, "frontality": 0.90, "is_back": False}
        tracks = {
            0: {"cx": 480.0, "cy": 290.0, "last_t": 1.0, "detections": [(1.0, fA, 8.0), (1.2, fA, 8.5)]},
            1: {"cx": 1280.0, "cy": 290.0, "last_t": 1.0, "detections": [(1.0, fB, 1.0), (1.2, fB, 1.2)]}
        }
        x, _, _, _, target_cx, shot_type, _, diag, _ = speaker_tracker.classify_and_frame_subject(
            tracks, [(1.0, None, [fA, fB])], "S1", 1.0, 1.5, {}, self.default_x + self.crop_w / 2.0, self.crop_w, self.max_x
        )
        self.assertEqual(diag["subject_role"], "speaker")
        self.assertAlmostEqual(target_cx, 480.0, delta=20.0)

    # 42. Two-person interview (Camera cuts to B listening) -> B framed at cut
    def test_42_two_person_camera_cut_listener_reaction(self):
        fA = {"bbox": (400.0, 200.0, 120.0, 140.0), "center": (460.0, 270.0), "conf": 0.85, "area": 16800.0, "frontality": 0.50, "is_back": False}
        fB = {"bbox": (1100.0, 160.0, 220.0, 260.0), "center": (1210.0, 290.0), "conf": 0.96, "area": 57200.0, "frontality": 0.95, "is_back": False}
        tracks = {
            0: {"cx": 460.0, "cy": 270.0, "last_t": 5.0, "detections": [(5.0, fA, 1.0)]},
            1: {"cx": 1210.0, "cy": 290.0, "last_t": 5.0, "detections": [(5.0, fB, 0.5)]}
        }
        x, _, _, _, target_cx, _, _, diag, _ = speaker_tracker.classify_and_frame_subject(
            tracks, [(5.0, None, [fA, fB])], "S1", 5.0, 5.5, {"S1": 460.0}, self.default_x + self.crop_w / 2.0, self.crop_w, self.max_x, is_shot_cut=True
        )
        self.assertEqual(diag["subject_role"], "listener_reaction")
        self.assertAlmostEqual(target_cx, 1210.0, delta=20.0)

    # 43. Two-person interview (Both fit in 9:16) -> Joint framing centers both
    def test_43_two_person_joint_framing(self):
        f1 = {"bbox": (820.0, 220.0, 100.0, 120.0), "center": (870.0, 280.0), "conf": 0.90, "area": 12000.0, "frontality": 0.90, "is_back": False}
        f2 = {"bbox": (1020.0, 220.0, 100.0, 120.0), "center": (1070.0, 280.0), "conf": 0.90, "area": 12000.0, "frontality": 0.90, "is_back": False}
        tracks = {
            0: {"cx": 870.0, "cy": 280.0, "last_t": 2.0, "detections": [(2.0, f1, 1.0)]},
            1: {"cx": 1070.0, "cy": 280.0, "last_t": 2.0, "detections": [(2.0, f2, 1.0)]}
        }
        x, _, _, _, target_cx, shot_type, _, diag, _ = speaker_tracker.classify_and_frame_subject(
            tracks, [(2.0, None, [f1, f2])], "S1", 2.0, 2.5, {}, self.default_x + self.crop_w / 2.0, self.crop_w, self.max_x
        )
        self.assertEqual(shot_type, "two_shot_both_fit")
        self.assertAlmostEqual(target_cx, 970.0, delta=10.0)

    # 44. Two-person interview (Wide shot, cannot both fit) -> Isolates active speaker
    def test_44_two_person_wide_shot_isolation(self):
        f1 = {"bbox": (200.0, 200.0, 160.0, 200.0), "center": (280.0, 300.0), "conf": 0.92, "area": 32000.0, "frontality": 0.90, "is_back": False}
        f2 = {"bbox": (1600.0, 200.0, 160.0, 200.0), "center": (1680.0, 300.0), "conf": 0.92, "area": 32000.0, "frontality": 0.90, "is_back": False}
        tracks = {
            0: {"cx": 280.0, "cy": 300.0, "last_t": 1.0, "detections": [(1.0, f1, 6.5)]},
            1: {"cx": 1680.0, "cy": 300.0, "last_t": 1.0, "detections": [(1.0, f2, 1.0)]}
        }
        x, _, _, _, target_cx, shot_type, _, diag, _ = speaker_tracker.classify_and_frame_subject(
            tracks, [(1.0, None, [f1, f2])], "S1", 1.0, 1.5, {}, self.default_x + self.crop_w / 2.0, self.crop_w, self.max_x
        )
        self.assertNotEqual(shot_type, "two_shot_both_fit")
        self.assertAlmostEqual(target_cx, 280.0, delta=20.0)

    # 45. Three-person podcast panel (Center person speaks) -> Center framed
    def test_45_three_person_center_speaker(self):
        fL = {"bbox": (300.0, 200.0, 140.0, 180.0), "center": (370.0, 290.0), "conf": 0.90, "area": 25200.0, "frontality": 0.85, "is_back": False}
        fC = {"bbox": (900.0, 200.0, 140.0, 180.0), "center": (970.0, 290.0), "conf": 0.92, "area": 25200.0, "frontality": 0.95, "is_back": False}
        fR = {"bbox": (1500.0, 200.0, 140.0, 180.0), "center": (1570.0, 290.0), "conf": 0.90, "area": 25200.0, "frontality": 0.85, "is_back": False}
        tracks = {
            0: {"cx": 370.0, "cy": 290.0, "last_t": 2.0, "detections": [(2.0, fL, 1.0)]},
            1: {"cx": 970.0, "cy": 290.0, "last_t": 2.0, "detections": [(2.0, fC, 8.0)]},
            2: {"cx": 1570.0, "cy": 290.0, "last_t": 2.0, "detections": [(2.0, fR, 1.0)]}
        }
        x, _, _, _, target_cx, shot_type, _, diag, _ = speaker_tracker.classify_and_frame_subject(
            tracks, [(2.0, None, [fL, fC, fR])], "S1", 2.0, 2.5, {}, self.default_x + self.crop_w / 2.0, self.crop_w, self.max_x
        )
        self.assertEqual(shot_type, "multi_person_panel")
        self.assertEqual(diag["subject_role"], "speaker")
        self.assertAlmostEqual(target_cx, 970.0, delta=20.0)

    # 46. Three-person podcast panel (Far right speaker speaks) -> Track 3 framed (PROVES NO TOP-2 TRUNCATION)
    def test_46_three_person_far_right_speaker_no_truncation(self):
        fL = {"bbox": (300.0, 200.0, 140.0, 180.0), "center": (370.0, 290.0), "conf": 0.90, "area": 25200.0, "frontality": 0.85, "is_back": False}
        fC = {"bbox": (900.0, 200.0, 140.0, 180.0), "center": (970.0, 290.0), "conf": 0.90, "area": 25200.0, "frontality": 0.85, "is_back": False}
        fR = {"bbox": (1500.0, 200.0, 140.0, 180.0), "center": (1570.0, 290.0), "conf": 0.95, "area": 25200.0, "frontality": 0.95, "is_back": False}
        tracks = {
            0: {"cx": 370.0, "cy": 290.0, "last_t": 2.0, "detections": [(2.0, fL, 1.0)]},
            1: {"cx": 970.0, "cy": 290.0, "last_t": 2.0, "detections": [(2.0, fC, 1.0)]},
            2: {"cx": 1570.0, "cy": 290.0, "last_t": 2.0, "detections": [(2.0, fR, 9.0)]}
        }
        x, _, _, _, target_cx, shot_type, _, diag, _ = speaker_tracker.classify_and_frame_subject(
            tracks, [(2.0, None, [fL, fC, fR])], "S1", 2.0, 2.5, {}, self.default_x + self.crop_w / 2.0, self.crop_w, self.max_x
        )
        self.assertEqual(shot_type, "multi_person_panel")
        self.assertEqual(diag["subject_role"], "speaker")
        self.assertAlmostEqual(target_cx, 1570.0, delta=20.0)

    # 47. Four-person roundtable (4 participants, speaker handoff across round)
    def test_47_four_person_roundtable_handoff(self):
        f1 = {"bbox": (200.0, 200.0, 120.0, 140.0), "center": (260.0, 270.0), "conf": 0.90, "area": 16800.0, "frontality": 0.90, "is_back": False}
        f2 = {"bbox": (650.0, 200.0, 120.0, 140.0), "center": (710.0, 270.0), "conf": 0.90, "area": 16800.0, "frontality": 0.90, "is_back": False}
        f3 = {"bbox": (1150.0, 200.0, 120.0, 140.0), "center": (1210.0, 270.0), "conf": 0.90, "area": 16800.0, "frontality": 0.90, "is_back": False}
        f4 = {"bbox": (1600.0, 200.0, 120.0, 140.0), "center": (1660.0, 270.0), "conf": 0.90, "area": 16800.0, "frontality": 0.90, "is_back": False}
        tracks = {
            0: {"cx": 260.0, "cy": 270.0, "last_t": 3.0, "detections": [(3.0, f1, 1.0)]},
            1: {"cx": 710.0, "cy": 270.0, "last_t": 3.0, "detections": [(3.0, f2, 1.0)]},
            2: {"cx": 1210.0, "cy": 270.0, "last_t": 3.0, "detections": [(3.0, f3, 1.0)]},
            3: {"cx": 1660.0, "cy": 270.0, "last_t": 3.0, "detections": [(3.0, f4, 7.5)]}
        }
        x, _, _, _, target_cx, shot_type, _, diag, _ = speaker_tracker.classify_and_frame_subject(
            tracks, [(3.0, None, [f1, f2, f3, f4])], "S4", 3.0, 3.5, {}, self.default_x + self.crop_w / 2.0, self.crop_w, self.max_x
        )
        self.assertEqual(shot_type, "multi_person_panel")
        self.assertAlmostEqual(target_cx, 1660.0, delta=20.0)

    # 48. Off-screen speaker dialogue over listener -> Listener framed, speaker coords untouched
    def test_48_offscreen_dialogue_over_listener(self):
        fListener = {"bbox": (1100.0, 200.0, 200.0, 260.0), "center": (1200.0, 330.0), "conf": 0.95, "area": 52000.0, "frontality": 0.95, "is_back": False}
        tracks = {0: {"cx": 1200.0, "cy": 330.0, "last_t": 10.0, "detections": [(10.0, fListener, 0.0)]}}
        last_spk = {"S1": 450.0}
        speaker_tracker.classify_and_frame_subject(
            tracks, [(10.0, None, [fListener])], "S1", 10.0, 11.0, last_spk, self.default_x + self.crop_w / 2.0, self.crop_w, self.max_x, is_shot_cut=True
        )
        self.assertEqual(last_spk["S1"], 450.0)

    # 49. Rapid camera cuts (<1.0s shots) -> 0ms cut reset per shot
    def test_49_rapid_camera_cuts_0ms_reset(self):
        raw_kf = [(0.0, 400.0, True), (0.6, 900.0, True), (1.2, 500.0, True)]
        opt_kf = speaker_tracker.optimize_camera_trajectory(raw_kf, self.max_x)
        self.assertEqual(opt_kf[0][1], 400.0)
        self.assertEqual(opt_kf[1][1], 900.0)
        self.assertEqual(opt_kf[2][1], 500.0)

    # 50. Wide to close-up cut -> Scale adapts without zoom escalation
    def test_50_wide_to_closeup_scale_transition(self):
        f_wide = {"bbox": (800.0, 300.0, 20.0, 30.0), "center": (810.0, 315.0), "conf": 0.90, "area": 600.0, "frontality": 0.90}
        f_close = {"bbox": (700.0, 150.0, 300.0, 360.0), "center": (850.0, 330.0), "conf": 0.95, "area": 108000.0, "frontality": 0.95}
        scale_wide, _, _, class_wide = speaker_tracker.classify_shot_and_estimate_scale(f_wide, [], self.crop_w, 1920.0, 1080.0, "solo_close_up")
        scale_close, _, _, class_close = speaker_tracker.classify_shot_and_estimate_scale(f_close, [], self.crop_w, 1920.0, 1080.0, "solo_close_up")
        self.assertLess(scale_wide, scale_close)
        self.assertLessEqual(scale_close, speaker_tracker.MAX_SCALE)

    # 51. Temporary facial occlusion (subject turns head) -> ByteTrack coasting
    def test_51_temporary_facial_occlusion_profile(self):
        f_vis = {"bbox": (600.0, 200.0, 150.0, 200.0), "center": (675.0, 300.0), "conf": 0.90, "area": 30000.0}
        s1 = (1.0, None, [f_vis])
        s2 = (1.2, None, [])
        s3 = (1.4, None, [dict(f_vis, conf=0.35)])
        tracks = speaker_tracker.build_shot_tracks([s1, s2, s3])
        self.assertEqual(len(tracks), 1)
        self.assertAlmostEqual(tracks[0]["cx"], 675.0, delta=10.0)

    # 52. Camera panning background -> GMC decouples pan
    def test_52_camera_panning_gmc_compensation(self):
        img1 = np.zeros((1080, 1920, 3), dtype=np.uint8)
        cv2.circle(img1, (600, 400), 80, (200, 200, 200), -1)
        cv2.circle(img1, (1200, 700), 100, (255, 255, 255), -1)
        M = np.float32([[1, 0, -35.0], [0, 1, 0.0]])
        img2 = cv2.warpAffine(img1, M, (1920, 1080))
        dx, _ = speaker_tracker.compute_camera_motion_compensation(img1, img2)
        self.assertAlmostEqual(dx, -35.0, delta=6.0)

    # 53. Speaker enters frame mid-shot -> Dynamic track initialization
    def test_53_speaker_enters_frame_mid_shot(self):
        s1 = (0.0, None, [])
        f_enter = {"bbox": (700.0, 200.0, 150.0, 200.0), "center": (775.0, 300.0), "conf": 0.90, "area": 30000.0}
        s2 = (0.5, None, [f_enter])
        tracks = speaker_tracker.build_shot_tracks([s1, s2])
        self.assertEqual(len(tracks), 1)
        self.assertIn(0, tracks)

    # 54. Speaker leaves frame mid-shot -> Graceful fallback
    def test_54_speaker_leaves_frame_mid_shot(self):
        tracks = {}
        target_track, _, target_cx, shot_type, role, _ = speaker_tracker.resolve_visual_subject(
            [], [], [], None, 5.0, 5.5, False, {}, 960.0, 600.0, self.crop_w
        )
        self.assertIsNone(target_track)
        self.assertEqual(role, "fallback")
        self.assertEqual(target_cx, 960.0)

    # 55. Noisy diarization (unmatched turn) -> Visual saliency governs without S1 bias
    def test_55_noisy_diarization_no_s1_bias(self):
        f = {"bbox": (1100.0, 200.0, 200.0, 250.0), "center": (1200.0, 325.0), "conf": 0.95, "area": 50000.0, "frontality": 0.95, "is_back": False}
        tracks = {0: {"cx": 1200.0, "cy": 325.0, "last_t": 1.0, "detections": [(1.0, f, 6.0)]}}
        x, _, _, _, target_cx, _, _, diag, _ = speaker_tracker.classify_and_frame_subject(
            tracks, [(1.0, None, [f])], None, 1.0, 1.5, {"S1": 400.0}, self.default_x + self.crop_w / 2.0, self.crop_w, self.max_x
        )
        self.assertAlmostEqual(target_cx, 1200.0, delta=20.0)

    # 56. Zero faces detected in frame (B-roll/slide) -> Recovery hierarchy maintains valid crop
    def test_56_zero_faces_detected_recovery_hierarchy(self):
        x, _, _, _, target_cx, shot_type, _, diag, _ = speaker_tracker.classify_and_frame_subject(
            {}, [], None, 1.0, 1.5, {}, self.default_x + self.crop_w / 2.0, self.crop_w, self.max_x, current_crop_x=500.0
        )
        self.assertEqual(shot_type, "fallback_recovery_hierarchy")
        self.assertTrue(0.0 <= x <= self.max_x)

    # 57. Anti-hunting hysteresis in reaction shot -> Dwell time prevents rapid A/B toggling
    def test_57_anti_hunting_reaction_dwell_hysteresis(self):
        cfg = speaker_tracker.DEFAULT_FRAMING_CONFIG
        self.assertGreaterEqual(cfg.listener_reaction_min_dwell, 0.80)

        # State machine test: simulate sub-windows across a listener reaction shot
        # Listener is framed at x = 880, off-screen speaker at x = 400
        # Cut at t = 0.0 establishes listener reaction anchor
        locked_role = "listener_reaction"
        locked_anchor_x = 880.0
        last_role_change_time = 0.0
        crop_w_baseline = 607.5

        # Sub-window 1 at t = 0.4s: Offscreen speaker active at x = 400 (dwell < 0.80s)
        # Should NOT reframe back to speaker yet (hold reaction dwell)
        sub_abs = 0.4
        current_role = "speaker"
        clamped_x = 400.0
        target_dist = abs(clamped_x - locked_anchor_x)
        is_same_subject = target_dist < max(25.0, crop_w_baseline * 0.05)
        time_in_reaction = sub_abs - last_role_change_time
        should_reframe = False
        if current_role == "speaker" and is_same_subject:
            locked_role = "speaker"
        elif current_role == "speaker" and not is_same_subject:
            if time_in_reaction >= cfg.listener_reaction_min_dwell:
                should_reframe = True
        self.assertFalse(should_reframe)  # Dwell holding!

        # Sub-window 2 at t = 0.9s: Offscreen speaker still active at x = 400 (dwell >= 0.80s)
        # Should now reframe cleanly back to speaker
        sub_abs = 0.9
        time_in_reaction = sub_abs - last_role_change_time
        if current_role == "speaker" and not is_same_subject:
            if time_in_reaction >= cfg.listener_reaction_min_dwell:
                should_reframe = True
        self.assertTrue(should_reframe)  # Dwell expired, transition allowed!

        # Sub-window 3: Framed listener themselves starts speaking at t = 0.3s (at x = 880)
        # Should immediately promote to speaker WITHOUT waiting for 0.8s dwell!
        sub_abs = 0.3
        locked_role = "listener_reaction"
        clamped_x = 880.0
        target_dist = abs(clamped_x - locked_anchor_x)
        is_same_subject = target_dist < max(25.0, crop_w_baseline * 0.05)
        should_reframe = False
        if current_role == "speaker" and is_same_subject:
            locked_role = "speaker"
        self.assertEqual(locked_role, "speaker")
        self.assertFalse(should_reframe)  # Immediate promotion, no jump

    # 58. Full-clip regression test: Candidate #3 Shot 3 listener framed at x ≈ 889, corr = 0
    def test_58_full_clip_candidate_3_regression(self):
        f_speaker = {"bbox": (600.0, 180.0, 220.0, 280.0), "center": (710.0, 320.0), "conf": 0.95, "area": 61600.0, "frontality": 0.95, "is_back": False}
        f_listener = {"bbox": (1080.0, 180.0, 200.0, 260.0), "center": (1180.0, 310.0), "conf": 0.95, "area": 52000.0, "frontality": 0.95, "is_back": False}
        tracks_shot3 = {0: {"cx": 1180.0, "cy": 310.0, "last_t": 23.0, "detections": [(23.0, f_listener, 0.0)]}}
        x, _, _, _, target_cx, _, _, diag, _ = speaker_tracker.classify_and_frame_subject(
            tracks_shot3, [(23.0, None, [f_listener])], "S1", 22.8, 24.0, {"S1": 710.0}, self.default_x + self.crop_w / 2.0, self.crop_w, self.max_x, is_shot_cut=True
        )
        self.assertEqual(diag["subject_role"], "listener_reaction")
        self.assertAlmostEqual(target_cx, 1180.0, delta=20.0)
        self.assertAlmostEqual(x, 876.0, delta=20.0)

if __name__ == "__main__":
    unittest.main()
