import io
import sys
import unittest
from unittest.mock import patch
import numpy as np

from speaker_tracker import (
    FramingConfig,
    DEFAULT_FRAMING_CONFIG,
    run,
    apply_scale_deadband,
    MIN_SCALE,
    MAX_SCALE,
)


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


class TestFramingConfigProductionBinding(unittest.TestCase):
    def test_apply_scale_deadband_honors_explicit_deadband(self):
        # 1. Scale up beyond deadband
        scale_def, reason_def = apply_scale_deadband(1.0, 1.10, deadband=0.08)
        self.assertEqual(scale_def, 1.10)
        self.assertEqual(reason_def, "SCALE_UP")

        # 2. Scale up within wider deadband -> locked
        scale_cust, reason_cust = apply_scale_deadband(1.0, 1.10, deadband=0.15)
        self.assertEqual(scale_cust, 1.0)
        self.assertEqual(reason_cust, "DEADBAND")

        # 3. Scale down beyond deadband
        scale_down, reason_down = apply_scale_deadband(1.20, 1.10, deadband=0.08)
        self.assertEqual(scale_down, 1.10)
        self.assertEqual(reason_down, "SCALE_DOWN")

        # 4. Scale down within wider deadband -> locked
        scale_down_locked, reason_down_locked = apply_scale_deadband(1.20, 1.10, deadband=0.15)
        self.assertEqual(scale_down_locked, 1.20)
        self.assertEqual(reason_down_locked, "DEADBAND")

        # 5. Default deadband (DEFAULT_FRAMING_CONFIG.scale_deadband = 0.08)
        scale_def_none, reason_def_none = apply_scale_deadband(1.0, 1.05)
        self.assertEqual(scale_def_none, 1.0)
        self.assertEqual(reason_def_none, "DEADBAND")

    @patch("speaker_tracker.sample_video_and_detect_shots")
    def test_production_run_loop_respects_framing_config(self, mock_sample):
        # Synthetic video clip: 2.5 seconds, sampled every 0.25s
        # Face starts at cx=800, shifts to cx=920 at t=1.0s (120px shift).
        # With crop_w_baseline=608, face width=160:
        # avail_margin = 448, max_center_dev = 145.92px.
        # Since shift 120px < 145.92px, off_center is False and severe_clip is False.
        # Framing decisions are governed by DISP_THRESH and SUSTAINED_DUR.
        times = np.arange(0.0, 2.6, 0.25)
        dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        samples = []
        for t in times:
            cx = 800.0 if t < 1.0 else 920.0
            samples.append((t, dummy_frame, [make_mock_face(cx)]))

        mock_sample.return_value = ([0.0], samples)

        # Config A: High displacement threshold ratio (0.25 -> DISP_THRESH = 152px)
        # Shift of 120px is below threshold, so the camera stays locked at initial anchor.
        cfg_high_thresh = FramingConfig(
            displacement_thresh_ratio=0.25,
            reframe_cooldown=0.2,
            sustained_displacement_dur=0.2,
        )

        # Config B: Low displacement threshold ratio (0.10 -> DISP_THRESH = 70px)
        # Shift of 120px exceeds threshold, so camera reframes to follow subject.
        cfg_sensitive = FramingConfig(
            displacement_thresh_ratio=0.10,
            reframe_cooldown=0.2,
            sustained_displacement_dur=0.2,
        )

        stdout_buf = io.StringIO()
        with patch("sys.stdout", stdout_buf):
            plan_high = run("mock.mp4", 0.0, 2500.0, 608, 1312, 656, config=cfg_high_thresh)
            plan_sens = run("mock.mp4", 0.0, 2500.0, 608, 1312, 656, config=cfg_sensitive)

        self.assertIsNotNone(plan_high, "run() must return the generated SmartFramingPlan")
        self.assertIsNotNone(plan_sens, "run() must return the generated SmartFramingPlan")
        self.assertNotEqual(
            plan_high.x,
            plan_sens.x,
            "Production run() loop must produce different trajectories when FramingConfig changes",
        )


if __name__ == "__main__":
    unittest.main()
