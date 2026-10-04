"""
AutoShorts 5.0 — Ultralytics CV Adapter Test Suite
Tests UltralyticsTracker, BoT-SORT integration, YuNet fusion, shot resets,
telemetry, and A/B backend toggling.
"""

import unittest
import os
import sys
import io
import time
import subprocess
import numpy as np

# ─── Project .venv resolution (Warning 2 fix) ─────────────────────────────────
# The Ultralytics backend requires torch/ultralytics, which live in the project
# virtual environment (d:/College/Autoshorts 9.0/.venv). If this suite was
# launched with a system Python that lacks those dependencies, re-exec the suite
# with the project .venv interpreter so the REAL Ultralytics backend is
# exercised (never the native OpenCV fallback as a substitute).
def _has_ultralytics_deps() -> bool:
    try:
        import torch  # noqa: F401
        import ultralytics  # noqa: F401
        return True
    except ImportError:
        return False

def _find_project_venv_python() -> str | None:
    """Locate the AutoShorts project .venv interpreter (mirrors media.rs find_python_cmd())."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "..", "..", "..", ".venv", "Scripts", "python.exe"),  # <repo>/autoshorts/.venv
        os.path.join(here, "..", "..", ".venv", "Scripts", "python.exe"),        # <repo>/src-tauri/.venv
        os.path.join(here, "..", ".venv", "Scripts", "python.exe"),              # <repo>/scripts/.venv
        os.path.join(here, "..", "..", "..", ".venv", "bin", "python"),          # POSIX variants
        os.path.join(here, "..", "..", ".venv", "bin", "python"),
        os.path.join(here, "..", ".venv", "bin", "python"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None

def _reexec_with_project_venv_if_needed():
    if _has_ultralytics_deps():
        return
    venv_python = _find_project_venv_python()
    if venv_python is None:
        # No project venv found — report clearly instead of silently substituting
        # the native backend and claiming the Ultralytics test passed.
        sys.stderr.write(
            "[Ultralytics Suite] WARNING: torch/ultralytics not importable and no project .venv found.\n"
            f"[Ultralytics Suite] Interpreter: {sys.executable}\n"
            "[Ultralytics Suite] Ultralytics-specific tests will be skipped; native fallback is NOT proof of the Ultralytics backend.\n"
        )
        return
    if os.path.abspath(venv_python) == os.path.abspath(sys.executable):
        return
    sys.stderr.write(
        f"[Ultralytics Suite] Re-executing with project .venv interpreter: {venv_python}\n"
    )
    sys.stderr.flush()
    result = subprocess.run([venv_python, os.path.abspath(__file__)], cwd=os.path.dirname(os.path.abspath(__file__)))
    sys.exit(result.returncode)

_reexec_with_project_venv_if_needed()

# Ensure speaker_tracker is importable
sys.path.insert(0, os.path.dirname(__file__))
import speaker_tracker


class TestUltralyticsAdapterSuite(unittest.TestCase):

    def setUp(self):
        self.original_backend = os.environ.get("AUTOSHORTS_CV_BACKEND")

    def tearDown(self):
        if self.original_backend is not None:
            os.environ["AUTOSHORTS_CV_BACKEND"] = self.original_backend
        else:
            os.environ.pop("AUTOSHORTS_CV_BACKEND", None)

    def test_yolo_model_discovery_and_loading(self):
        """Verify that yolo11n.pt model file is discovered and loads correctly."""
        model_path = speaker_tracker.find_yolo_model()
        self.assertIsNotNone(model_path, "yolo11n.pt model file should be found")
        self.assertTrue(os.path.exists(model_path), f"Path must exist: {model_path}")
        self.assertGreater(os.path.getsize(model_path), 5_000_000, "Model file size should be ~5.6 MB")

        model = speaker_tracker.get_ultralytics_model()
        self.assertIsNotNone(model, "YOLO model singleton should load successfully")
        self.assertEqual(model.task, "detect", "Model task must be 'detect'")

    def test_backend_toggle_switch(self):
        """Verify that AUTOSHORTS_CV_BACKEND environment variable toggles backends."""
        fake_face = {
            "bbox": (100.0, 100.0, 80.0, 80.0),
            "center": (140.0, 140.0),
            "conf": 0.90,
            "area": 6400.0,
            "frontality": 0.85,
            "gaze_offset_x": 0.0,
            "skin_ratio": 0.5,
            "is_back": False,
            "head_top_y": 80.0,
            "eye_line_y": 120.0,
            "shoulder_span": (50.0, 230.0),
            "torso_top": 170.0,
            "mouth_gray": None,
            "landmarks": None,
        }
        sample = (0.0, None, [fake_face])

        # Test Native mode
        os.environ["AUTOSHORTS_CV_BACKEND"] = "native"
        tracks_native = speaker_tracker.build_shot_tracks([sample])
        self.assertIn(0, tracks_native)
        self.assertEqual(tracks_native[0]["cx"], 140.0)

        # Test Ultralytics mode (with fallback on None frame)
        os.environ["AUTOSHORTS_CV_BACKEND"] = "ultralytics"
        tracks_ultra = speaker_tracker.build_shot_tracks([sample])
        self.assertIn(0, tracks_ultra)
        self.assertEqual(tracks_ultra[0]["cx"], 140.0)

    def test_ultralytics_tracking_on_video_frames(self):
        """Verify Ultralytics BoT-SORT tracking and YuNet fusion on real/synthetic images."""
        model = speaker_tracker.get_ultralytics_model()
        if model is None:
            self.skipTest("Ultralytics model not available")

        h, w = 720, 1280
        frame1 = np.ones((h, w, 3), dtype=np.uint8) * 128
        frame2 = np.ones((h, w, 3), dtype=np.uint8) * 128

        face1 = {
            "bbox": (300.0, 200.0, 100.0, 100.0),
            "center": (350.0, 250.0),
            "conf": 0.85,
            "area": 10000.0,
            "frontality": 0.90,
            "gaze_offset_x": 0.0,
            "skin_ratio": 0.5,
            "is_back": False,
            "head_top_y": 180.0,
            "eye_line_y": 235.0,
            "shoulder_span": (250.0, 450.0),
            "torso_top": 285.0,
            "mouth_gray": np.zeros((20, 32), dtype=np.uint8),
            "landmarks": None,
        }
        face2 = dict(face1)
        face2["center"] = (352.0, 250.0)

        shot_samples = [
            (0.0, frame1, [face1]),
            (0.125, frame2, [face2]),
        ]

        tracker = speaker_tracker.UltralyticsTracker(model)
        shot_tracks = tracker.track_shot(shot_samples)
        self.assertGreater(len(shot_tracks), 0, "Should have tracked at least 1 subject")

        for tid, tr in shot_tracks.items():
            self.assertIn("cx", tr)
            self.assertIn("cy", tr)
            self.assertIn("detections", tr)
            self.assertIn("hits", tr)
            self.assertIn("is_occluded", tr)

    def test_shot_reset_clears_tracker_state(self):
        """Verify that _reset_tracker clears BoT-SORT state between shots."""
        model = speaker_tracker.get_ultralytics_model()
        if model is None:
            self.skipTest("Ultralytics model not available")

        tracker = speaker_tracker.UltralyticsTracker(model)
        tracker._reset_tracker()
        if hasattr(model, 'predictor') and hasattr(model.predictor, 'trackers'):
            for t in model.predictor.trackers:
                if hasattr(t, 'frame_id'):
                    self.assertEqual(t.frame_id, 0)

    def test_structured_telemetry_format(self):
        """Verify that structured telemetry is output to stderr without crashing stdout."""
        sample = (0.0, None, [])
        stderr_capture = io.StringIO()
        old_stderr = sys.stderr
        try:
            sys.stderr = stderr_capture
            os.environ["AUTOSHORTS_CV_BACKEND"] = "native"
            speaker_tracker.build_shot_tracks([sample])
            output = stderr_capture.getvalue()
            self.assertIn("[CV Backend]", output)
            self.assertIn("backend=native", output)
        finally:
            sys.stderr = old_stderr


if __name__ == "__main__":
    unittest.main()
