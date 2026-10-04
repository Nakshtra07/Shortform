"""
Unit tests for Resolution-Independent Head Geometry Threshold Normalization (F7).
Validates:
  1. 1080p exact equivalence and boundary (134x134 -> False, 135x135 -> True)
  2. 720p proportional scaling and boundary (89x89 -> False, 90x90 -> True)
  3. 4K proportional scaling and boundary (268x268 -> False, 269x269 -> True)
  4. 1080x1920 (9:16 portrait) scaling and axis orientation (fy=700 -> False, fy=1000 -> True)
  5. Low-front and extreme-edge proportional scaling across resolutions.
  6. Robustness to zero/degenerate frame dimensions.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))

import speaker_tracker
from speaker_tracker import (
    REF_FRAME_AREA_1080P,
    FRAC_BOTTOM_FOREGROUND,
    FRAC_LOW_FRONT_LARGE,
    FRAC_EXTREME_EDGE_HUGE,
    is_back_of_head_candidate,
)


class TestGeometryScalingSuite(unittest.TestCase):
    def test_constants_math(self):
        """Assert exact reference area and fraction definitions."""
        self.assertAlmostEqual(REF_FRAME_AREA_1080P, 1920.0 * 1080.0, places=4)
        self.assertAlmostEqual(FRAC_BOTTOM_FOREGROUND, 18000.0 / (1920.0 * 1080.0), places=8)
        self.assertAlmostEqual(FRAC_LOW_FRONT_LARGE, 14000.0 / (1920.0 * 1080.0), places=8)
        self.assertAlmostEqual(FRAC_EXTREME_EDGE_HUGE, 35000.0 / (1920.0 * 1080.0), places=8)

    def test_1080p_exact_equivalence_and_boundary(self):
        """
        1080p landscape (1920x1080):
        Bottom foreground threshold is exactly 18000.0 px^2 at fy > 1080 * 0.48 = 518.4.
        134x134 = 17956 -> False
        135x135 = 18225 -> True
        """
        w, h = 1920, 1080
        fy = 600.0  # fy > h * 0.48 (518.4)
        fx = 800.0  # centered
        conf = 0.90  # clean detection (suppresses low_front and extreme_edge)
        skin_ratio = 0.50
        sym_err = 0.0

        # Sub-threshold
        self.assertFalse(
            is_back_of_head_candidate(fx, fy, 134.0, 134.0, w, h, conf, skin_ratio, sym_err),
            "1080p 134x134 (17956 px^2) should be False"
        )
        # Super-threshold
        self.assertTrue(
            is_back_of_head_candidate(fx, fy, 135.0, 135.0, w, h, conf, skin_ratio, sym_err),
            "1080p 135x135 (18225 px^2) should be True"
        )

    def test_720p_proportional_scaling_and_boundary(self):
        """
        720p landscape (1280x720):
        Frame area = 921,600 (4/9 of 1080p).
        Bottom foreground threshold = 18000 * (4/9) = 8000.0 px^2.
        89x89 = 7921 -> False
        90x90 = 8100 -> True
        """
        w, h = 1280, 720
        fy = 400.0  # fy > h * 0.48 (345.6)
        fx = 500.0  # centered
        conf = 0.90
        skin_ratio = 0.50
        sym_err = 0.0

        # Sub-threshold
        self.assertFalse(
            is_back_of_head_candidate(fx, fy, 89.0, 89.0, w, h, conf, skin_ratio, sym_err),
            "720p 89x89 (7921 px^2 vs 8000 threshold) should be False"
        )
        # Super-threshold
        self.assertTrue(
            is_back_of_head_candidate(fx, fy, 90.0, 90.0, w, h, conf, skin_ratio, sym_err),
            "720p 90x90 (8100 px^2 vs 8000 threshold) should be True"
        )

    def test_4k_proportional_scaling_and_boundary(self):
        """
        4K landscape (3840x2160):
        Frame area = 8,294,400 (4x of 1080p).
        Bottom foreground threshold = 18000 * 4 = 72000.0 px^2.
        268x268 = 71824 -> False
        269x269 = 72361 -> True
        """
        w, h = 3840, 2160
        fy = 1200.0  # fy > h * 0.48 (1036.8)
        fx = 1500.0  # centered
        conf = 0.90
        skin_ratio = 0.50
        sym_err = 0.0

        # Sub-threshold
        self.assertFalse(
            is_back_of_head_candidate(fx, fy, 268.0, 268.0, w, h, conf, skin_ratio, sym_err),
            "4K 268x268 (71824 px^2 vs 72000 threshold) should be False"
        )
        # Super-threshold
        self.assertTrue(
            is_back_of_head_candidate(fx, fy, 269.0, 269.0, w, h, conf, skin_ratio, sym_err),
            "4K 269x269 (72361 px^2 vs 72000 threshold) should be True"
        )

    def test_portrait_9_16_scaling_and_axis_orientation(self):
        """
        Portrait 9:16 (1080x1920):
        Frame area = 1080 * 1920 = 2,073,600 (identical to 1080p landscape).
        Bottom threshold = 18000.0 px^2.
        Vertical threshold line = h * 0.48 = 1920 * 0.48 = 921.6.
        Head of size 150x150 (22500 px^2 > 18000):
          fy = 700.0 (upper-middle: fy < 921.6) -> False
          fy = 1000.0 (bottom: fy > 921.6) -> True
        """
        w, h = 1080, 1920
        fx = 450.0  # centered
        fw, fh = 150.0, 150.0  # area = 22500 > 18000
        conf = 0.90
        skin_ratio = 0.50
        sym_err = 0.0

        # Upper-middle (fy=700 < 921.6)
        self.assertFalse(
            is_back_of_head_candidate(fx, 700.0, fw, fh, w, h, conf, skin_ratio, sym_err),
            "Portrait fy=700 is upper-middle (< 921.6) -> should be False"
        )
        # Bottom (fy=1000 > 921.6)
        self.assertTrue(
            is_back_of_head_candidate(fx, 1000.0, fw, fh, w, h, conf, skin_ratio, sym_err),
            "Portrait fy=1000 is bottom (> 921.6) -> should be True"
        )

    def test_low_front_large_scaling(self):
        """
        is_low_front_large:
        conf < 0.78 and (skin_ratio < 0.35 or sym_err > 0.32) and face_area > FRAC_LOW_FRONT_LARGE * frame_area
        1080p: threshold = 14000.0. 118x118 (13924) -> False, 119x119 (14161) -> True
        720p: threshold = 14000 * 4/9 = 6222.22. 78x78 (6084) -> False, 79x79 (6241) -> True
        """
        # 1080p
        w, h = 1920, 1080
        fy = 300.0  # top half, so bottom_foreground is False
        fx = 800.0
        conf = 0.70  # < 0.78
        skin_ratio = 0.20  # < 0.35
        sym_err = 0.10

        self.assertFalse(is_back_of_head_candidate(fx, fy, 118.0, 118.0, w, h, conf, skin_ratio, sym_err))
        self.assertTrue(is_back_of_head_candidate(fx, fy, 119.0, 119.0, w, h, conf, skin_ratio, sym_err))

        # 720p
        w720, h720 = 1280, 720
        fy720 = 200.0
        fx720 = 500.0
        self.assertFalse(is_back_of_head_candidate(fx720, fy720, 78.0, 78.0, w720, h720, conf, skin_ratio, sym_err))
        self.assertTrue(is_back_of_head_candidate(fx720, fy720, 79.0, 79.0, w720, h720, conf, skin_ratio, sym_err))

    def test_extreme_edge_huge_scaling(self):
        """
        is_extreme_edge_huge:
        (fx < w * 0.15 or fx + fw > w * 0.85) and face_area > FRAC_EXTREME_EDGE_HUGE * frame_area and conf < 0.80
        1080p: threshold = 35000.0. 187x187 (34969) -> False, 188x188 (35344) -> True
        """
        w, h = 1920, 1080
        fy = 300.0
        conf = 0.75  # < 0.80
        skin_ratio = 0.50
        sym_err = 0.0

        # Left edge (fx = 100 < 1920 * 0.15 = 288)
        self.assertFalse(is_back_of_head_candidate(100.0, fy, 187.0, 187.0, w, h, conf, skin_ratio, sym_err))
        self.assertTrue(is_back_of_head_candidate(100.0, fy, 188.0, 188.0, w, h, conf, skin_ratio, sym_err))

        # Right edge (fx = 1700 => fx + fw = 1888 > 1920 * 0.85 = 1632)
        self.assertFalse(is_back_of_head_candidate(1700.0, fy, 187.0, 187.0, w, h, conf, skin_ratio, sym_err))
        self.assertTrue(is_back_of_head_candidate(1700.0, fy, 188.0, 188.0, w, h, conf, skin_ratio, sym_err))

    def test_degenerate_dimensions_guard(self):
        """Guard against zero or negative dimensions."""
        self.assertFalse(is_back_of_head_candidate(0.0, 0.0, 0.0, 0.0, 0, 0, 0.5, 0.5, 0.0))
        self.assertFalse(is_back_of_head_candidate(10.0, 10.0, 50.0, 50.0, -100, -100, 0.9, 0.5, 0.0))


if __name__ == "__main__":
    unittest.main()
