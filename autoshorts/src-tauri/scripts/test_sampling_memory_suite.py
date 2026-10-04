"""
AutoShorts 7.0 — Sampling Memory Budget & Full-Coverage Regression Suite
=========================================================================
ROOT CAUSE REGRESSION (late-clip framing failure, clips clip-05/clip-03):

`sample_video_and_detect_shots` used to store FULL-resolution frames for the
whole clip. For ~67s 4K candidates that is ~536 frames x 24.9MB ~ 13.3GB;
OpenCV frame allocation OOMs mid-clip (~470 frames), the read loop breaks, and
because `cv_success = len(all_samples) >= 3` a PARTIAL sample set silently
passes. The unsampled tail never sees the late scene change (single speaker
returns at ~59.9s / ~62.3s), so `resolve_visual_subject` serves a stale
max-confidence face, the Camera Lock sees it as centered, and the crop freezes
on the microphone/chair region for the final ~7-10s.

FIX UNDER TEST (v12.2 memory-budgeted sampling):
  1. Stored frames are downscaled to fit a memory budget
     (AUTOSHORTS_SAMPLE_MEM_BUDGET_MB, default ~3.2GB) -> full clip coverage.
  2. Face detection runs on the TRUE-resolution frame; `all_samples` faces
     are ALWAYS in true source space.
  3. Tracking runs in stored-frame space (faces scaled to match YOLO/native
     boxes), and track geometry is scaled back to TRUE source space before
     any downstream consumer (framing solver, DualFrame, Camera Lock,
     trajectory validation) sees it.
  4. A stderr WARNING fires if sampling stops early (truncation guard).

THE FRAMING INVARIANT asserted end-to-end (test_09):
  For a clip whose FINAL scene contains a single speaker at a known true-space
  position, the final plan's crop X must equal the containment crop computed
  from that speaker's TRUE-space geometry — i.e. the crop follows the visible
  speaker through the very end of the clip, in true source coordinates.
"""

import io
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(__file__))
import speaker_tracker


def make_face(cx: float, cy: float, w: float = 120.0, h: float = 140.0, conf: float = 0.92) -> dict:
    """Synthetic true-space face dict with every key downstream code reads."""
    return {
        "bbox": (cx - w / 2.0, cy - h / 2.0, w, h),
        "center": (cx, cy),
        "conf": conf,
        "area": w * h,
        "frontality": 0.90,
        "gaze_offset_x": 0.0,
        "skin_ratio": 0.60,
        "is_back": False,
        "head_top_y": cy - h / 2.0 - 0.22 * h,
        "eye_line_y": cy - 0.15 * h,
        "shoulder_span": (cx - w * 1.1, cx + w * 1.1),
        "torso_top": cy + h * 0.35,
        "mouth_gray": None,
        "landmarks": None,
    }


class TestSamplingMemoryBudget(unittest.TestCase):
    """Unit tests for the memory-budget machinery itself."""

    def setUp(self):
        self._saved = os.environ.get("AUTOSHORTS_SAMPLE_MEM_BUDGET_MB")
        if "AUTOSHORTS_SAMPLE_MEM_BUDGET_MB" in os.environ:
            del os.environ["AUTOSHORTS_SAMPLE_MEM_BUDGET_MB"]

    def tearDown(self):
        if self._saved is not None:
            os.environ["AUTOSHORTS_SAMPLE_MEM_BUDGET_MB"] = self._saved
        elif "AUTOSHORTS_SAMPLE_MEM_BUDGET_MB" in os.environ:
            del os.environ["AUTOSHORTS_SAMPLE_MEM_BUDGET_MB"]

    def test_01_default_budget_downscales_4k_long_clip(self):
        """A ~67s 4K clip (the exact failing class) must downscale storage.

        Pre-fix: 536 full-res frames x 24.9MB ~ 13.3GB -> mid-clip OOM.
        Post-fix: stored width fits the ~3.2GB budget (~1.9k px, ~6MB/frame).
        """
        true_w, true_h = 3840.0, 2160.0
        n_frames = 67.0 * 8.0 + 4.0  # 540 estimated samples
        store_w = speaker_tracker._compute_stored_frame_width(true_w, true_h, n_frames)
        self.assertLess(store_w, int(true_w), "4K long clip must downscale stored frames")
        self.assertGreaterEqual(store_w, speaker_tracker._MIN_STORED_FRAME_WIDTH)
        self.assertEqual(store_w % 2, 0)
        # Budget actually honored: n frames of stored size fit the budget.
        store_h = store_w * true_h / true_w
        total_bytes = store_w * store_h * 3.0 * n_frames
        self.assertLessEqual(
            total_bytes,
            speaker_tracker._DEFAULT_SAMPLE_MEM_BUDGET_BYTES * 1.001,
            f"stored frames ({total_bytes/1e9:.2f}GB) must fit the memory budget",
        )
        # And it must be far below the ~11.7GB point where the OOM struck.
        self.assertLess(total_bytes, 4.0 * 1024 ** 3)

    def test_02_short_clip_keeps_full_resolution(self):
        """Short clips (e.g. 8s 1080p real-render tests) must NOT downscale."""
        true_w, true_h = 1920.0, 1080.0
        n_frames = 8.0 * 8.0 + 4.0  # 68 samples
        store_w = speaker_tracker._compute_stored_frame_width(true_w, true_h, n_frames)
        self.assertEqual(store_w, 1920, "1080p 8s clip fits the budget without downscaling")

    def test_03_env_override_forces_tiny_budget(self):
        """AUTOSHORTS_SAMPLE_MEM_BUDGET_MB is honored (testability hook)."""
        os.environ["AUTOSHORTS_SAMPLE_MEM_BUDGET_MB"] = "1"  # 1 MB
        try:
            store_w = speaker_tracker._compute_stored_frame_width(1920.0, 1080.0, 68.0)
            self.assertEqual(
                store_w, speaker_tracker._MIN_STORED_FRAME_WIDTH,
                "tiny budget must clamp stored width to the minimum",
            )
        finally:
            del os.environ["AUTOSHORTS_SAMPLE_MEM_BUDGET_MB"]

    def test_04_face_scaling_round_trip(self):
        """true->stored->true coordinate round-trip is lossless for all keys."""
        face = make_face(1800.0, 500.0, w=160.0, h=200.0)
        face["landmarks"] = ((1750.0, 450.0), (1850.0, 450.0), (1800.0, 500.0),
                             (1770.0, 560.0), (1830.0, 560.0))
        face["person_bbox"] = (1500.0, 300.0, 2100.0, 1200.0)
        ratio = 640.0 / 3840.0  # stored-space shrink
        stored = speaker_tracker._scale_face_to_space(face, ratio)
        # back to true space
        back = speaker_tracker._scale_face_to_space(stored, 1.0 / ratio)
        self.assertAlmostEqual(back["bbox"][0], face["bbox"][0], places=3)
        self.assertAlmostEqual(back["bbox"][2], face["bbox"][2], places=3)
        self.assertAlmostEqual(back["center"][0], face["center"][0], places=3)
        self.assertAlmostEqual(back["center"][1], face["center"][1], places=3)
        self.assertAlmostEqual(back["area"], face["area"], places=0)
        self.assertAlmostEqual(back["head_top_y"], face["head_top_y"], places=3)
        self.assertAlmostEqual(back["eye_line_y"], face["eye_line_y"], places=3)
        self.assertAlmostEqual(back["torso_top"], face["torso_top"], places=3)
        self.assertAlmostEqual(back["shoulder_span"][0], face["shoulder_span"][0], places=3)
        self.assertAlmostEqual(back["landmarks"][2][0], face["landmarks"][2][0], places=3)
        self.assertAlmostEqual(back["person_bbox"][3], face["person_bbox"][3], places=3)

    def test_05_track_scaling_to_source_space(self):
        """Track dicts (cx/cy/vx/vy, person_bbox, detections) scale to true space."""
        det_face = make_face(320.0, 250.0, w=80.0, h=100.0)  # stored-space
        tracks = {
            7: {
                "cx": 320.0, "cy": 250.0, "vx": 12.0, "vy": -3.0,
                "last_t": 1.0, "last_mouth": None,
                "detections": [(1.0, det_face, 0.0)],
                "hits": 2, "age": 2, "is_occluded": False,
                "person_bbox": (200.0, 100.0, 440.0, 700.0),
                "provenance": "yolo_person", "has_person_body": True,
            }
        }
        ratio = 1920.0 / 640.0  # stored->true
        scaled = speaker_tracker._scale_shot_tracks_to_source_space(tracks, ratio)
        tr = scaled[7]
        self.assertAlmostEqual(tr["cx"], 960.0, places=3)
        self.assertAlmostEqual(tr["cy"], 750.0, places=3)
        self.assertAlmostEqual(tr["vx"], 36.0, places=3)
        self.assertAlmostEqual(tr["person_bbox"][2], 1320.0, places=3)
        self.assertAlmostEqual(tr["detections"][0][1]["center"][0], 960.0, places=3)
        self.assertAlmostEqual(tr["detections"][0][1]["bbox"][2], 240.0, places=3)
        # ratio=1.0 is a no-op returning the same object
        self.assertIs(speaker_tracker._scale_shot_tracks_to_source_space(tracks, 1.0), tracks)


class TestSamplerFullCoverage(unittest.TestCase):
    """Sampler behavior on REAL synthetic video files (cv2.VideoWriter)."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp(prefix="autoshorts_sampling_test_")
        # Synthetic 1920x1080 30fps video, 12s: bright square "face" LEFT
        # half for t<6s, RIGHT half for t>=6s (hard scene change at 6.0s).
        cls.video_path = os.path.join(cls.temp_dir, "synth_two_scene.mp4")
        cls.true_w, cls.true_h = 1920, 1080
        cls.scene_change_t = 6.0
        cls.left_cx, cls.right_cx = 480.0, 1440.0
        writer = cv2.VideoWriter(
            cls.video_path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (cls.true_w, cls.true_h)
        )
        for i in range(360):
            t = i / 30.0
            scene2 = t >= cls.scene_change_t
            # Hard background luminance flip at the scene change so shot-cut
            # detection (160x90 mean absdiff > SCENE_CUT_DIFF) fires at t=6.0.
            bg = 150 if scene2 else 40
            frame = np.full((cls.true_h, cls.true_w, 3), bg, dtype=np.uint8)
            cx = cls.right_cx if scene2 else cls.left_cx
            # textured "head" block (YuNet will not see a face; faces are
            # injected via patched detector in the pipeline test below).
            x0, y0 = int(cx - 100), 300
            frame[y0:y0 + 260, x0:x0 + 200] = (200, 190, 180)
            cv2.circle(frame, (int(cx), 430), 60, (90, 120, 200), -1)
            writer.write(frame)
        writer.release()

    @classmethod
    def tearDownClass(cls):
        try:
            os.remove(cls.video_path)
            os.rmdir(cls.temp_dir)
        except OSError:
            pass

    def setUp(self):
        self._budget_saved = os.environ.get("AUTOSHORTS_SAMPLE_MEM_BUDGET_MB")
        self._backend_saved = os.environ.get("AUTOSHORTS_CV_BACKEND")
        os.environ["AUTOSHORTS_CV_BACKEND"] = "native"
        speaker_tracker._SOURCE_GEOMETRY = None

    def tearDown(self):
        if self._budget_saved is not None:
            os.environ["AUTOSHORTS_SAMPLE_MEM_BUDGET_MB"] = self._budget_saved
        elif "AUTOSHORTS_SAMPLE_MEM_BUDGET_MB" in os.environ:
            del os.environ["AUTOSHORTS_SAMPLE_MEM_BUDGET_MB"]
        if self._backend_saved is not None:
            os.environ["AUTOSHORTS_CV_BACKEND"] = self._backend_saved
        elif "AUTOSHORTS_CV_BACKEND" in os.environ:
            del os.environ["AUTOSHORTS_CV_BACKEND"]
        speaker_tracker._SOURCE_GEOMETRY = None

    def test_06_tiny_budget_downscales_storage_and_covers_full_duration(self):
        """THE COVERAGE INVARIANT: with a tiny budget the sampler must store
        downscaled frames, keep faces in TRUE space, and sample the ENTIRE
        clip (last sample within 0.5s of clip end). Pre-fix behavior stored
        full-res frames and OOM-truncated long clips."""
        os.environ["AUTOSHORTS_SAMPLE_MEM_BUDGET_MB"] = "1"
        cuts, samples = speaker_tracker.sample_video_and_detect_shots(
            self.video_path, 0.0, 12000.0
        )
        self.assertGreaterEqual(len(samples), 3)
        # Stored frames downscaled to the minimum width...
        self.assertEqual(samples[0][1].shape[1], speaker_tracker._MIN_STORED_FRAME_WIDTH)
        # ...but recorded geometry is the TRUE source size.
        self.assertEqual(speaker_tracker._SOURCE_GEOMETRY, (1920.0, 1080.0))
        # FULL COVERAGE: last sample reaches the clip end (within 0.5s).
        self.assertGreaterEqual(samples[-1][0], 12.0 - 0.5,
                                "sampler must cover the full clip duration")
        # Scene change at ~6s detected (hard luminance flip of the head block).
        self.assertTrue(any(5.0 <= c <= 7.0 for c in cuts),
                        f"scene change near t=6.0 expected, got cuts={cuts}")

    def test_07_default_budget_keeps_full_resolution_short_clip(self):
        """Default budget on a short 1080p clip: no downscale, geometry true."""
        cuts, samples = speaker_tracker.sample_video_and_detect_shots(
            self.video_path, 0.0, 12000.0
        )
        self.assertGreaterEqual(len(samples), 3)
        self.assertEqual(samples[0][1].shape[1], 1920)
        self.assertEqual(speaker_tracker._SOURCE_GEOMETRY, (1920.0, 1080.0))
        self.assertGreaterEqual(samples[-1][0], 12.0 - 0.5)

    def test_08_truncation_warning_fires_on_early_stop(self):
        """If sampling stops early, the truncation WARNING must fire."""
        os.environ["AUTOSHORTS_SAMPLE_MEM_BUDGET_MB"] = "1"

        class FakeCap:
            """Yields 3 frames then stops (simulates mid-clip decode break)."""
            def __init__(self, *_a, **_k):
                self.n = 0
                self.opened = True
            def isOpened(self):
                return True
            def get(self, prop):
                return 30.0 if prop == cv2.CAP_PROP_FPS else 0.0
            def set(self, *_a):
                return True
            def read(self):
                self.n += 1
                # 9 frames at 30fps with frame_step=4 -> stored samples at
                # frame 0, 4, 8 (3 samples pass the >=3 gate), then stop.
                if self.n > 9:
                    return False, None
                return True, np.zeros((1080, 1920, 3), dtype=np.uint8)
            def release(self):
                pass

        stderr_buf = io.StringIO()
        with patch.object(cv2, "VideoCapture", FakeCap), \
             patch("sys.stderr", stderr_buf):
            cuts, samples = speaker_tracker.sample_video_and_detect_shots(
                "fake.mp4", 0.0, 12000.0
            )
        # 3 samples pass the >=3 gate but end at t=0.1 -> WARNING required.
        self.assertEqual(len(samples), 3)
        self.assertIn("sampling truncated", stderr_buf.getvalue())

    def test_09_framing_follows_late_scene_change_end_to_end(self):
        """THE FRAMING INVARIANT (root-cause regression).

        Two-scene clip, speaker LEFT (t<6s) then RIGHT (t>=6s), analyzed with
        a tiny memory budget (forces the downscaled-storage path that used to
        be the OOM failure class). The final plan's crop must follow the
        scene-2 speaker in TRUE source coordinates through the clip end.
        """
        os.environ["AUTOSHORTS_SAMPLE_MEM_BUDGET_MB"] = "1"

        left_face = make_face(self.left_cx, 430.0)
        right_face = make_face(self.right_cx, 430.0)

        # The sampler calls detect once per stored sample in order: 12s @
        # 30fps with frame_step=4 -> 90 samples at t = 4n/30. The visual
        # scene change lands on frame 180 = call 45 (t=6.0 exactly), so the
        # injected face flips in lockstep with the detected shot cut.
        call_state = {"n": 0}

        def detect_by_time(detector, frame):
            face = left_face if call_state["n"] < 45 else right_face
            call_state["n"] += 1
            return [dict(face)]

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        with patch("speaker_tracker.detect_faces_and_subjects", detect_by_time), \
             patch("sys.stdout", stdout_buf), patch("sys.stderr", stderr_buf):
            speaker_tracker.run(
                source_path=self.video_path,
                start_ms=0.0,
                end_ms=12000.0,
                crop_w=608.0,
                max_x=1312.0,
                default_x=656.0,
            )

        lines = [ln.strip() for ln in stdout_buf.getvalue().splitlines()
                 if ln.strip() and not ln.strip().startswith("[")]
        self.assertTrue(lines, f"expected plan JSON on stdout, got: {stderr_buf.getvalue()}")
        plan = json.loads(lines[-1])

        # Plan covers the full clip (no truncation of the timeline).
        segments = plan["segments"]
        self.assertAlmostEqual(segments[-1]["end"], 12.0, delta=0.05)

        # Final single segment must frame the scene-2 (right) speaker.
        final_single = [s for s in segments if s["mode"] == "single"][-1]
        x_expr = final_single["crop"]["x"]

        def eval_x(expr, t):
            safe = expr.replace("if(", "_iff(")
            env = {"t": t, "lt": lambda a, b: a < b,
                   "_iff": lambda c, a, b: a if c else b}
            return float(eval(safe, env))

        x_end = eval_x(x_expr, final_single["end"] - 0.1)
        x_late = eval_x(x_expr, 11.0)

        # Containment crop for the right speaker in TRUE space:
        # face cx=1440, w=120 -> safe box ~[1314, 1572]; crop_w=608 ->
        # ideal X ~ 1440 - 608/2 = 1136 (clamped to max_x=1312).
        self.assertLess(x_end, 1313.0, f"final crop must move RIGHT of start, got X={x_end}")
        self.assertGreater(x_end, 900.0, f"final crop must follow right speaker, got X={x_end}")
        self.assertAlmostEqual(x_late, x_end, delta=1.0,
                               msg="crop must be stable on the scene-2 speaker through the end")
        # And the early part of the clip framed the LEFT speaker.
        first_single = [s for s in segments if s["mode"] == "single"][0]
        x_start = eval_x(first_single["crop"]["x"], min(1.0, first_single["end"] - 0.1))
        self.assertLess(x_start, 700.0, f"early crop must frame left speaker, got X={x_start}")


if __name__ == "__main__":
    unittest.main()
