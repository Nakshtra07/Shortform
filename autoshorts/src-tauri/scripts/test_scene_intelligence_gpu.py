import os
import sys
import unittest
import torch
import numpy as np

# Add scripts directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scene_intelligence

class TestSceneIntelligenceGPU(unittest.TestCase):
    def test_cuda_available(self):
        self.assertTrue(torch.cuda.is_available(), "CUDA must be available for GPU scene intelligence")

    def test_pytorch_adaptive_detector_detects_cuts(self):
        # Create a synthetic sequence of 150 frames (30fps = 5.0s)
        # Shot 1: Frames 0..49 (black, 0)
        # Shot 2: Frames 50..99 (white, 255) -> Cut at frame 50 (1.667s)
        # Shot 3: Frames 100..149 (gray, 128) -> Cut at frame 100 (3.333s)
        frames = []
        for i in range(150):
            if i < 50:
                val = 10
            elif i < 100:
                val = 240
            else:
                val = 120
            frame = np.full((180, 320, 3), val, dtype=np.uint8)
            frames.append(frame)

        cuts = scene_intelligence.detect_cuts_from_frames_pytorch(frames, fps=30.0)
        # Should detect cuts near 1.667s and 3.333s
        self.assertGreaterEqual(len(cuts), 2)
        cut1, cut2 = cuts[0], cuts[1]
        self.assertAlmostEqual(cut1, 50 / 30.0, delta=0.2)
        self.assertAlmostEqual(cut2, 100 / 30.0, delta=0.2)

    def test_failsoft_fallback_on_invalid_file(self):
        doc = scene_intelligence.run_cached("invalid_file_nonexistent_12345.mp4", None)
        self.assertTrue(doc.get("fallback"), "Missing file must produce fallback=True")
        self.assertGreaterEqual(len(doc.get("scenes", [])), 1)

    def test_memory_cleanup(self):
        if not torch.cuda.is_available():
            return
        torch.cuda.empty_cache()
        baseline = torch.cuda.memory_allocated()
        # Run a small batch
        frames = [np.full((180, 320, 3), 128, dtype=np.uint8) for _ in range(64)]
        _ = scene_intelligence.detect_cuts_from_frames_pytorch(frames, fps=30.0)
        torch.cuda.empty_cache()
        after = torch.cuda.memory_allocated()
        self.assertLessEqual(after - baseline, 5 * 1024 * 1024, "VRAM must be cleaned up to <5MB delta")

    def test_luma_diffs_supports_grayscale_and_bgr(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bgr_frames = [np.full((180, 320, 3), 10 if i < 2 else 240, dtype=np.uint8) for i in range(4)]
        diffs_bgr = scene_intelligence._compute_luma_diffs_pytorch(bgr_frames, device)
        gray_frames = [np.full((180, 320), 10 if i < 2 else 240, dtype=np.uint8) for i in range(4)]
        diffs_gray = scene_intelligence._compute_luma_diffs_pytorch(gray_frames, device)
        self.assertIsNotNone(diffs_bgr)
        self.assertIsNotNone(diffs_gray)
        self.assertTrue(np.allclose(diffs_bgr, diffs_gray))

    def test_build_scenes_from_cuts(self):
        # Empty cuts -> single scene
        scenes = scene_intelligence._build_scenes_from_cuts([], 10.0)
        self.assertEqual(len(scenes), 1)
        self.assertEqual(scenes[0]["start"], 0.0)
        self.assertEqual(scenes[0]["end"], 10.0)

        # Cuts at 3.0, 7.0
        scenes = scene_intelligence._build_scenes_from_cuts([3.0, 7.0], 10.0)
        self.assertEqual(len(scenes), 3)
        self.assertEqual(scenes[0], {"sceneId": 0, "start": 0.0, "end": 3.0})
        self.assertEqual(scenes[1], {"sceneId": 1, "start": 3.0, "end": 7.0})
        self.assertEqual(scenes[2], {"sceneId": 2, "start": 7.0, "end": 10.0})

    def test_probe_video_stream_missing_file_safe(self):
        info = scene_intelligence._probe_video_stream("nonexistent_video_path.mp4")
        self.assertEqual(info["codec"], "")
        self.assertEqual(info["duration"], 0.0)
        self.assertEqual(info["fps"], 30.0)

    def test_build_scenes_from_cuts_boundary_rounding_safety(self):
        # Cut very close to duration: 9.9996s vs 10.0s
        scenes = scene_intelligence._build_scenes_from_cuts([9.9996], 10.0)
        self.assertIsNotNone(scenes)
        self.assertGreaterEqual(len(scenes), 1)
        for s in scenes:
            self.assertGreater(s["end"], s["start"], f"Scene {s} must have strictly positive duration")
            self.assertLessEqual(s["end"], 10.0, f"Scene {s} must not exceed video duration")

        # Empty cuts, tiny duration
        scenes = scene_intelligence._build_scenes_from_cuts([], 0.05)
        self.assertIsNotNone(scenes)
        self.assertEqual(len(scenes), 1)
        self.assertGreater(scenes[0]["end"], scenes[0]["start"])

    def test_ffmpeg_pipe_fallback_to_opencv(self):
        original_pipe = scene_intelligence._detect_scenes_ffmpeg_pipe
        original_cv = scene_intelligence._detect_scenes_opencv
        try:
            # Force FFmpeg pipe to fail
            scene_intelligence._detect_scenes_ffmpeg_pipe = lambda *a, **kw: None
            # Mock opencv to return simulated scenes
            mock_scenes = [{"sceneId": 0, "start": 0.0, "end": 5.0}]
            scene_intelligence._detect_scenes_opencv = lambda *a, **kw: mock_scenes

            # Call detect_scenes_gpu with a dummy path that exists (this script itself)
            result = scene_intelligence.detect_scenes_gpu(__file__)
            self.assertEqual(result, mock_scenes)
        finally:
            scene_intelligence._detect_scenes_ffmpeg_pipe = original_pipe
            scene_intelligence._detect_scenes_opencv = original_cv

    def test_multi_tier_fallback_hierarchy(self):
        original_pipe = scene_intelligence._detect_scenes_ffmpeg_pipe
        original_cv = scene_intelligence._detect_scenes_opencv
        pipe_calls = []

        def mock_pipe(*args, **kwargs):
            allow_hwaccel = kwargs.get("allow_hwaccel", True)
            pipe_calls.append(allow_hwaccel)
            if allow_hwaccel:
                # Simulate hardware acceleration failure (e.g. NVDEC error)
                return None
            # Software fallback succeeds
            return [{"sceneId": 0, "start": 0.0, "end": 10.0}]

        try:
            scene_intelligence._detect_scenes_ffmpeg_pipe = mock_pipe
            scene_intelligence._detect_scenes_opencv = lambda *a, **kw: None

            result = scene_intelligence.detect_scenes_gpu(__file__)
            self.assertIsNotNone(result)
            self.assertEqual(len(result), 1)
            # Verify tier 1 was attempted with hwaccel=True, then tier 2 with hwaccel=False
            self.assertEqual(pipe_calls, [True, False])
        finally:
            scene_intelligence._detect_scenes_ffmpeg_pipe = original_pipe
            scene_intelligence._detect_scenes_opencv = original_cv

if __name__ == "__main__":
    unittest.main()
