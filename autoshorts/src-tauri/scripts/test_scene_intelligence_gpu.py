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

if __name__ == "__main__":
    unittest.main()
