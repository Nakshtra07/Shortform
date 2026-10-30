"""Resolution-Independent Framing Scaler Unit & Integration Test Suite.

Verifies:
1. Mathematical invariants of ResolutionScaler across 1080p, 4K UHD, 720p HD, and portrait 9:16.
2. Exact 1080p equivalence (scale_diag == 1.0, scale_area == 1.0, pixel metrics match baseline exactly).
3. 4K UHD scaling (scale_diag == 2.0, scale_area == 4.0, distance doubles, area quadruples).
4. 720p HD scaling (scale_diag == 2/3, scale_area == 4/9).
5. Robustness against degenerate / zero / negative dimensions.
6. Integration with speaker_tracker components:
   - DualFramePanelManager.is_subject_eligible distant face rejection scaling
   - resolve_visual_subject horizontal center jump scaling
   - NativeTracker tracking association threshold scaling
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
import pytest

# Ensure scripts dir is on sys.path
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import speaker_tracker
from speaker_tracker import (
    REF_FRAME_W,
    REF_FRAME_H,
    REF_DIAGONAL_1080P,
    REF_AREA_1080P,
    ResolutionScaler,
    DualFrameLayoutResolver,
    resolve_visual_subject,
    NativeTracker,
    FramingConfig,
)


# ==============================================================================
# 1. Mathematical Invariant & Scaling Factor Tests
# ==============================================================================

class TestResolutionScalerMath:
    def test_reference_constants(self):
        """Verify baseline 1080p reference constants."""
        assert REF_FRAME_W == 1920.0
        assert REF_FRAME_H == 1080.0
        assert math.isclose(REF_DIAGONAL_1080P, math.hypot(1920.0, 1080.0), rel_tol=1e-9)
        assert math.isclose(REF_AREA_1080P, 1920.0 * 1080.0, rel_tol=1e-9)
        assert REF_AREA_1080P == 2073600.0

    def test_1080p_exact_equivalence(self):
        """Verify that 1080p yields scale factors of 1.0 and identical metrics."""
        scaler = ResolutionScaler(1920.0, 1080.0)

        assert math.isclose(scaler.scale_diag, 1.0, rel_tol=1e-9)
        assert math.isclose(scaler.scale_area, 1.0, rel_tol=1e-9)

        # Distance thresholds at 1080p
        assert math.isclose(scaler.dist(180.0), 180.0, rel_tol=1e-9)
        assert math.isclose(scaler.dist(140.0), 140.0, rel_tol=1e-9)
        assert math.isclose(scaler.dist(150.0), 150.0, rel_tol=1e-9)
        assert math.isclose(scaler.dist(250.0), 250.0, rel_tol=1e-9)
        assert math.isclose(scaler.dist(17.5), 17.5, rel_tol=1e-9)

        # Area threshold at 1080p
        assert math.isclose(scaler.area_val(1800.0), 1800.0, rel_tol=1e-9)

    def test_4k_uhd_scaling(self):
        """Verify that 4K UHD (3840x2160) doubles Euclidean distance and quadruples 2D area."""
        scaler = ResolutionScaler(3840.0, 2160.0)

        assert scaler.scale_diag == pytest.approx(2.0, rel=1e-6)
        assert scaler.scale_area == pytest.approx(4.0, rel=1e-6)

        # Distance thresholds double
        assert scaler.dist(180.0) == pytest.approx(360.0, rel=1e-6)
        assert scaler.dist(140.0) == pytest.approx(280.0, rel=1e-6)
        assert scaler.dist(150.0) == pytest.approx(300.0, rel=1e-6)
        assert scaler.dist(250.0) == pytest.approx(500.0, rel=1e-6)
        assert scaler.dist(17.5) == pytest.approx(35.0, rel=1e-6)

        # Area threshold quadruples (1800 -> 7200)
        assert scaler.area_val(1800.0) == pytest.approx(7200.0, rel=1e-6)

    def test_720p_hd_scaling(self):
        """Verify that 720p HD (1280x720) scales by 2/3 for distance and 4/9 for area."""
        scaler = ResolutionScaler(1280.0, 720.0)

        assert scaler.scale_diag == pytest.approx(2.0 / 3.0, rel=1e-6)
        assert scaler.scale_area == pytest.approx(4.0 / 9.0, rel=1e-6)

        # Distance thresholds scale by 2/3
        assert scaler.dist(180.0) == pytest.approx(120.0, rel=1e-6)
        assert scaler.dist(140.0) == pytest.approx(140.0 * (2.0 / 3.0), rel=1e-6)
        assert scaler.dist(150.0) == pytest.approx(100.0, rel=1e-6)
        assert scaler.dist(250.0) == pytest.approx(250.0 * (2.0 / 3.0), rel=1e-6)
        assert scaler.dist(17.5) == pytest.approx(17.5 * (2.0 / 3.0), rel=1e-6)

        # Area threshold scales by 4/9 (1800 * 4/9 = 800)
        assert scaler.area_val(1800.0) == pytest.approx(800.0, rel=1e-6)

    def test_portrait_9_16_scaling(self):
        """Verify that 1080x1920 (9:16 portrait) preserves diagonal and area scaling."""
        scaler = ResolutionScaler(1080.0, 1920.0)

        assert scaler.scale_diag == pytest.approx(1.0, rel=1e-6)
        assert scaler.scale_area == pytest.approx(1.0, rel=1e-6)
        assert scaler.dist(180.0) == pytest.approx(180.0, rel=1e-6)
        assert scaler.area_val(1800.0) == pytest.approx(1800.0, rel=1e-6)

    def test_zero_and_negative_dimension_clamping(self):
        """Verify that non-positive dimensions are safely clamped to >= 1.0 without ZeroDivisionError."""
        scaler_zero = ResolutionScaler(0.0, 0.0)
        assert scaler_zero.w >= 1.0
        assert scaler_zero.h >= 1.0
        assert not math.isnan(scaler_zero.scale_diag)
        assert not math.isinf(scaler_zero.scale_diag)
        assert scaler_zero.dist(100.0) > 0.0

        scaler_neg = ResolutionScaler(-1920.0, -1080.0)
        assert scaler_neg.w >= 1.0
        assert scaler_neg.h >= 1.0
        assert not math.isnan(scaler_neg.scale_area)


# ==============================================================================
# 2. Integration Tests with speaker_tracker Components
# ==============================================================================

class TestResolutionScalingIntegration:
    def test_dualframe_distant_face_rejection_scaling(self):
        """Verify DualFrame distant face rejection adapts to 1080p, 4K, and 720p."""
        resolver = DualFrameLayoutResolver()

        # Baseline face in 1080p:
        # Bounding box width=50, height=50 -> area=2500 (> 1800 in 1080p)
        track_1080p = {
            "provenance": "yolo_person",
            "has_person_body": True,
            "is_occluded": False,
            "is_back": False,
            "cx": 960.0,
            "cy": 540.0,
            "visual_prominence": 0.50,
            "age_sec": 2.0,
            "detections": [
                (0.0, {"conf": 0.90, "bbox": (935, 515, 50, 50), "area": 2500.0, "is_back": False}),
            ],
        }

        # In 1080p: area 2500 >= 1800.0 -> accepted
        assert resolver.is_subject_eligible(track_1080p, source_w=1920.0, source_h=1080.0) is True

        # In 4K: area 2500 is < 7200.0 (4K threshold) -> rejected as distant micro-spec
        # (also min_w=3840*0.02=76.8 > 50, min_h=2160*0.035=75.6 > 50)
        assert resolver.is_subject_eligible(track_1080p, source_w=3840.0, source_h=2160.0) is False

        # In 4K: scaled face width=100, height=100 -> area=10000 (>= 7200.0) -> accepted
        track_4k = {
            "provenance": "yolo_person",
            "has_person_body": True,
            "is_occluded": False,
            "is_back": False,
            "cx": 1920.0,
            "cy": 1080.0,
            "visual_prominence": 0.50,
            "age_sec": 2.0,
            "detections": [
                (0.0, {"conf": 0.90, "bbox": (1870, 1030, 100, 100), "area": 10000.0, "is_back": False}),
            ],
        }
        assert resolver.is_subject_eligible(track_4k, source_w=3840.0, source_h=2160.0) is True

        # In 720p: area 1200 is < 1800 (would fail in 1080p) but >= 800 (720p threshold)
        # min_w = 1280 * 0.02 = 25.6 <= 35, min_h = 720 * 0.035 = 25.2 <= 35
        track_720p = {
            "provenance": "yolo_person",
            "has_person_body": True,
            "is_occluded": False,
            "is_back": False,
            "cx": 640.0,
            "cy": 360.0,
            "visual_prominence": 0.50,
            "age_sec": 2.0,
            "detections": [
                (0.0, {"conf": 0.90, "bbox": (622, 342, 35, 35), "area": 1225.0, "is_back": False}),
            ],
        }
        assert resolver.is_subject_eligible(track_720p, source_w=1280.0, source_h=720.0) is True

    def test_resolve_visual_subject_horizontal_jump_scaling(self):
        """Verify horizontal center jump (250px baseline) scales proportionally in 4K and 720p."""
        cfg = FramingConfig()

        # Construct single genuine track with low mouth motion (mot = 1.0)
        track = {
            "track_id": 1,
            "cx": 1300.0,
            "mot": 1.0,  # below active threshold 4.0
            "detections": [
                (0.0, {"center": (1300.0, 540.0), "conf": 0.9, "frontality": 0.8, "is_back": False, "area": 5000.0, "bbox": (1250, 490, 100, 100)}, 1.0),
            ],
        }

        last_confirmed_cx = {"spk_1": 1000.0}  # jump = |1300 - 1000| = 300px

        # In 1080p: jump of 300px > 250px -> triggers listener_reaction
        res_1080p = resolve_visual_subject(
            genuine_tracks=[track],
            ots_listener_tracks=[],
            pers_tracks=[track],
            spk="spk_1",
            rel_ts=0.0,
            rel_te=0.5,
            is_shot_cut=False,
            last_confirmed_cx_by_spk=last_confirmed_cx,
            default_cx=960.0,
            current_crop_x=600.0,
            crop_w_baseline=608.0,
            source_w=1920.0,
            source_h=1080.0,
            config=cfg,
        )
        # subject_role is index 4 in resolve_visual_subject return tuple
        assert res_1080p[4] == "listener_reaction"

        # In 4K: jump of 300px is <= scaler.dist(250.0) (500px in 4K) -> stays speaker
        res_4k = resolve_visual_subject(
            genuine_tracks=[track],
            ots_listener_tracks=[],
            pers_tracks=[track],
            spk="spk_1",
            rel_ts=0.0,
            rel_te=0.5,
            is_shot_cut=False,
            last_confirmed_cx_by_spk=last_confirmed_cx,
            default_cx=1920.0,
            current_crop_x=1200.0,
            crop_w_baseline=1216.0,
            source_w=3840.0,
            source_h=2160.0,
            config=cfg,
        )
        assert res_4k[4] == "speaker"

        # In 4K with jump of 600px > 500px -> triggers listener_reaction
        track_far = dict(track)
        track_far["cx"] = 1650.0
        track_far["detections"] = [
            (0.0, {"center": (1650.0, 540.0), "conf": 0.9, "frontality": 0.8, "is_back": False, "area": 5000.0, "bbox": (1600, 490, 100, 100)}, 1.0),
        ]
        res_4k_far = resolve_visual_subject(
            genuine_tracks=[track_far],
            ots_listener_tracks=[],
            pers_tracks=[track_far],
            spk="spk_1",
            rel_ts=0.0,
            rel_te=0.5,
            is_shot_cut=False,
            last_confirmed_cx_by_spk=last_confirmed_cx,
            default_cx=1920.0,
            current_crop_x=1200.0,
            crop_w_baseline=1216.0,
            source_w=3840.0,
            source_h=2160.0,
            config=cfg,
        )
        assert res_4k_far[4] == "listener_reaction"

    def test_native_tracker_distance_scaling(self):
        """Verify NativeTracker detection matching uses scaled distance thresholds."""
        # Frame 0: detection at cx=100.0, cy=100.0
        # Frame 1: detection displaced by 200px (cx=300.0, cy=100.0)
        # Stage 1 dist_thresh = scaler.dist(180.0)
        # In 1080p: dist = 200 > 180 -> stage 1 does NOT match (creates new track or stage 2 if < 140, which it isn't)
        # In 4K: dist_thresh = 360.0 -> dist = 200 <= 360 -> stage 1 matches and associates track!

        det_f0 = {"conf": 0.90, "center": (100.0, 100.0), "mouth_gray": None}
        det_f1 = {"conf": 0.90, "center": (300.0, 100.0), "mouth_gray": None}

        # Run 1080p tracking
        tracks_1080p = NativeTracker.track_shot(
            shot_samples=[(0.0, None, [det_f0]), (0.1, None, [det_f1])],
            source_w=1920.0,
            source_h=1080.0,
        )
        # Since dist 200 > 180, it cannot match in 1080p -> 2 distinct tracks created
        assert len(tracks_1080p) == 2

        # Run 4K tracking
        tracks_4k = NativeTracker.track_shot(
            shot_samples=[(0.0, None, [det_f0]), (0.1, None, [det_f1])],
            source_w=3840.0,
            source_h=2160.0,
        )
        # In 4K, dist_thresh is 360.0 >= 200.0 -> single track associates both detections
        assert len(tracks_4k) == 1
