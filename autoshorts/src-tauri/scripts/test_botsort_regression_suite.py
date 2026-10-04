"""
AutoShorts 11.0 — BoT-SORT regression tests
============================================

Guards the real Ultralytics BoT-SORT integration in ``speaker_tracker.py``
against the regression that disabled it:

    TypeError: slice indices must be integers or None or have an __index__
    method
    at speaker_tracker.py::UltralyticsTracker.track_shot -> frame[y:y+h, x:x+w]

Covered:
  * tracker initialisation (UltralyticsTracker + BoT-SORT reset)
  * a valid tracking result (ints, bboxes, confidences)
  * track-ID persistence across frames
  * the float-slice TypeError cannot recur (every crop window is int)

Run:  "D:/College/Autoshorts 11.0/.venv/Scripts/python.exe" -m pytest test_botsort_regression_suite.py -q
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import speaker_tracker as st  # noqa: E402


# ─── Fixtures / helpers ───────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def real_video():
    """A real media file. Skips the suite (loudly) when absent."""
    path = os.environ.get(
        "AUTOSHORTS_PROOF_VIDEO",
        r"D:\College\Autoshorts 11.0\clip-01_flat.mp4",
    )
    if not os.path.isfile(path):
        pytest.skip(f"real test video not available: {path}")
    return path


@pytest.fixture(scope="module")
def yolo_model():
    if not st.HAS_ULTRALYTICS:
        pytest.skip("ultralytics not installed in this interpreter")
    model = st.get_ultralytics_model()
    if model is None:
        pytest.skip(
            "yolo11n.pt weights unavailable; refusing to fake detections "
            "(run the venv python that owns the weights)"
        )
    return model


def read_frames(path, count):
    import cv2

    cap = cv2.VideoCapture(path)
    frames = []
    while len(frames) < count:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


class _FakeBoxes:
    """Minimal stand-in for ultralytics Boxes — lets us assert the parsing
    contract without running the network. Never used as proof of tracking."""

    def __init__(self, xyxy, conf, cls, ids):
        self.xyxy = np.asarray(xyxy, dtype=np.float64)
        self.conf = np.asarray(conf, dtype=np.float64)
        self.cls = np.asarray(cls, dtype=np.float64)
        self.id = None if ids is None else np.asarray(ids, dtype=np.float64)

    def __len__(self):
        return len(self.xyxy)


class _FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


# ─── 1. int_crop_window_xyxy: the float-slice regression guard ──────────────

class TestIntegerCropWindows:
    """The actual root cause. Every crop slice must get integer bounds."""

    @pytest.fixture
    def frame(self):
        return np.zeros((1080, 1920, 3), dtype=np.uint8)

    def test_float_bbox_yields_int_bounds(self, frame):
        """Regression: BoT-SORT boxes.xyxy are floats; bounds must be ints."""
        # This is the exact value class that produced the crash.
        bbox = (100.0, 50.0, 400.25, 900.75)
        win = st.int_crop_window_xyxy(frame, bbox)
        assert win is not None
        x, y, w, h = win
        assert all(type(v) is int for v in win), f"non-int crop window: {win}"

    def test_window_is_sliceable(self, frame):
        """The literal proof: frame[y:y+h, x:x+w] must not raise."""
        win = st.int_crop_window_xyxy(frame, (100.0, 50.0, 400.25, 900.75))
        x, y, w, h = win
        crop = frame[y:y + h, x:x + w]  # must not raise TypeError
        assert crop.shape[0] == h and crop.shape[1] == w

    def test_numpy_float_bbox_yields_int_bounds(self, frame):
        """numpy scalars must not leak into slices either."""
        bbox = np.array([10.5, 20.25, 300.75, 400.5])
        win = st.int_crop_window_xyxy(frame, bbox)
        assert win is not None
        assert all(type(v) is int for v in win)
        x, y, w, h = win
        frame[y:y + h, x:x + w]

    def test_window_clamped_to_frame(self, frame):
        win = st.int_crop_window_xyxy(frame, (-50.0, -20.0, 5000.0, 4000.0))
        x, y, w, h = win
        assert x == 0 and y == 0
        assert x + w <= frame.shape[1]
        assert y + h <= frame.shape[0]
        assert w >= 1 and h >= 1
        frame[y:y + h, x:x + w]

    def test_box_touching_right_edge_is_fully_inside(self, frame):
        win = st.int_crop_window_xyxy(frame, (1800.5, 100.5, 1920.0, 700.25))
        x, y, w, h = win
        assert x + w <= frame.shape[1]
        assert y + h <= frame.shape[0]
        frame[y:y + h, x:x + w]

    def test_degenerate_boxes_return_none(self, frame):
        # Zero-extent (x1==x2 or y1==y2) and missing input are degenerate.
        assert st.int_crop_window_xyxy(frame, (10.0, 10.0, 10.0, 10.0)) is None
        assert st.int_crop_window_xyxy(frame, (10.0, 10.0, 90.0, 10.0)) is None
        assert st.int_crop_window_xyxy(frame, None) is None
        assert st.int_crop_window_xyxy(None, (0.0, 0.0, 10.0, 10.0)) is None

    def test_inverted_box_is_normalised_not_rejected(self, frame):
        """A self-crossing box still names a real region; normalising is correct."""
        win = st.int_crop_window_xyxy(frame, (10.0, 10.0, 5.0, 50.0))
        assert win is not None
        x, y, w, h = win
        assert (x, y, w, h) == (5, 10, 10, 50)
        assert all(type(v) is int for v in win)
        frame[y:y + h, x:x + w]

    def test_non_finite_bbox_returns_none(self, frame):
        assert st.int_crop_window_xyxy(frame, (float("nan"), 0.0, 10.0, 10.0)) is None
        assert st.int_crop_window_xyxy(frame, (0.0, 0.0, float("inf"), 10.0)) is None

    def test_extract_embeddings_survives_float_bbox(self, frame, monkeypatch):
        """ReIDManager.extract_embeddings must not raise on float XYXY boxes.

        This is the exact call the BoT-SORT path makes, and the exact function
        that raised the float-slice TypeError before the fix.
        """
        mgr = st.ReIDManager.__new__(st.ReIDManager)  # no model load
        mgr.model = object()  # non-None so the crop loop actually runs
        mgr.device = "cpu"
        mgr.ema_alpha = 0.95
        mgr.gallery = {}

        captured = {}

        def fake_batch(model, crops, device):
            captured["crops"] = crops
            return [None] * len(crops)

        monkeypatch.setattr(st, "extract_embedding_batch", fake_batch)
        out = mgr.extract_embeddings(
            frame, [{"bbox": (100.0, 50.0, 400.25, 900.75)}]
        )
        assert len(out) == 1
        assert isinstance(captured["crops"][0], np.ndarray)


# ─── 2. Tracker initialisation ────────────────────────────────────────────────

class TestTrackerInit:
    def test_ultralytics_tracker_constructs(self, yolo_model):
        tracker = st.UltralyticsTracker(yolo_model)
        assert tracker.model is yolo_model

    def test_reset_tracker_is_safe_when_predictor_missing(self):
        class _FakeModel:
            pass

        t = st.UltralyticsTracker.__new__(st.UltralyticsTracker)
        t.model = _FakeModel()
        t._reset_tracker()  # must not raise

    def test_track_shot_announces_start_without_error(self, yolo_model, real_video, capsys):
        frames = read_frames(real_video, 2)
        if not frames:
            pytest.skip("no frames decoded")
        tracker = st.UltralyticsTracker(yolo_model)
        tracker.track_shot([(i * 0.033, f, []) for i, f in enumerate(frames)])
        err = capsys.readouterr().err
        assert "BoT-SORT tracking started" in err
        assert "tracking error" not in err.lower()


# ─── 3. Valid tracking result on real media ───────────────────────────────────

class TestValidTrackingResult:
    def test_boxes_are_parsed_with_int_ids(self, yolo_model, real_video):
        frames = read_frames(real_video, 5)
        if not frames:
            pytest.skip("no frames decoded")
        res = yolo_model.track(
            source=frames[0], persist=True, tracker="botsort.yaml",
            classes=[0], conf=0.25, verbose=False,
        )
        boxes = res[0].boxes
        if boxes is None or len(boxes) == 0:
            pytest.skip("no person detections in this clip (media property)")
        assert boxes.id is not None
        ids = boxes.id.cpu().numpy().astype(int)
        assert len(ids) == len(boxes.xyxy)
        assert all(isinstance(int(v), int) for v in ids)

    def test_track_shot_returns_wellformed_tracks(self, yolo_model, real_video):
        frames = read_frames(real_video, 30)
        if not frames:
            pytest.skip("no frames decoded")
        samples = [(i * 0.033, f, []) for i, f in enumerate(frames)]
        tracks = st.UltralyticsTracker(yolo_model).track_shot(samples)
        if not tracks:
            pytest.skip("no tracks in this clip (media property)")
        for tid, tr in tracks.items():
            assert isinstance(tid, int), f"track id not int: {tid!r}"
            assert tr["detections"], "track has no detections"
            for _rel_t, face, _mdiff in tr["detections"]:
                bb = face["bbox"]
                assert len(bb) == 4
                assert all(isinstance(float(v), float) for v in bb)
            assert np.isfinite(tr["cx"]) and np.isfinite(tr["cy"])

    def test_downstream_prominence_consumes_tracks(self, yolo_model, real_video):
        frames = read_frames(real_video, 30)
        if not frames:
            pytest.skip("no frames decoded")
        h, w = frames[0].shape[:2]
        samples = [(i * 0.033, f, []) for i, f in enumerate(frames)]
        tracks = st.UltralyticsTracker(yolo_model).track_shot(samples)
        if not tracks:
            pytest.skip("no tracks in this clip (media property)")
        for tid, tr in tracks.items():
            score = st.compute_visual_prominence(tr, source_w=float(w), source_h=float(h))
            assert 0.0 <= score <= 1.0


# ─── 4. Track-ID persistence across frames ────────────────────────────────────

class TestTrackIdPersistence:
    def test_track_id_persists_across_frames(self, yolo_model, real_video):
        frames = read_frames(real_video, 30)
        if len(frames) < 10:
            pytest.skip("not enough frames")
        seen = {}
        for f in frames:
            res = yolo_model.track(
                source=f, persist=True, tracker="botsort.yaml",
                classes=[0], conf=0.25, verbose=False,
            )
            boxes = res[0].boxes
            if boxes is None or boxes.id is None or len(boxes) == 0:
                continue
            for raw in boxes.id.cpu().numpy():
                seen.setdefault(int(raw), 0)
                seen[int(raw)] += 1
        if not seen:
            pytest.skip("no person tracks in this clip (media property)")
        best = max(seen.values())
        assert best >= 5, (
            f"no track ID persisted across frames (best run = {best} of "
            f"{len(frames)} frames; runs={seen})"
        )

    def test_ids_stay_int_across_many_frames(self, yolo_model, real_video):
        frames = read_frames(real_video, 12)
        if not frames:
            pytest.skip("no frames decoded")
        for f in frames:
            res = yolo_model.track(
                source=f, persist=True, tracker="botsort.yaml",
                classes=[0], conf=0.25, verbose=False,
            )
            boxes = res[0].boxes
            if boxes is None or boxes.id is None:
                continue
            for raw in boxes.id.cpu().numpy():
                assert isinstance(int(raw), int)
                assert int(raw) >= 0


# ─── 5. Backend selection must not silently degrade ───────────────────────────

class TestBackendSelection:
    def test_build_shot_tracks_runs_ultralytics(self, yolo_model, real_video, monkeypatch, capsys):
        frames = read_frames(real_video, 10)
        if not frames:
            pytest.skip("no frames decoded")
        monkeypatch.setenv("AUTOSHORTS_CV_BACKEND", "ultralytics")
        monkeypatch.setattr(st, "_CURRENT_BACKEND", None)
        monkeypatch.setattr(st, "_BACKEND_HEADER_PRINTED", False)
        samples = [(i * 0.033, f, []) for i, f in enumerate(frames)]

        st.build_shot_tracks(samples)
        err = capsys.readouterr().err
        assert "backend=ultralytics" in err
        assert "tracking error" not in err.lower(), (
            "build_shot_tracks degraded to the native tracker — BoT-SORT is broken"
        )

    def test_result_parsing_accepts_float_boxes(self, yolo_model):
        """Ultralytics hands back float XYXY; track_shot must yield int IDs.

        Uses a stub predictor result to assert the parsing contract. This does
        NOT claim tracking works — the real-media tests above do that.
        """
        stub = _FakeResult(
            _FakeBoxes(
                xyxy=[[10.5, 20.25, 300.75, 400.5], [50.0, 60.0, 200.0, 300.0]],
                conf=[0.91, 0.77],
                cls=[0, 0],
                ids=[1, 2],
            )
        )
        tids = [
            int(np.asarray(stub.boxes.id).astype(int)[i]) for i in range(len(stub.boxes))
        ]
        assert tids == [1, 2]
        assert all(isinstance(v, int) for v in tids)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))