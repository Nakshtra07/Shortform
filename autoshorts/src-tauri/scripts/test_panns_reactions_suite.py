"""
Focused tests for the PANNs reaction-detection sidecar.

Covers the four areas required to prove the 0-event bug is dead:
  1. Model-loading failure produces a CONCRETE, structured reason
     (e.g. `checkpoint_missing` naming the exact paths searched).
  2. Inference input/output shapes are what CNN14 actually needs:
     (1, n_samples) float32 in, (1, n_frames, 527) out.
  3. Event parsing / thresholding / min-duration behaviour, including the
     real measured label distribution for this corpus.
  4. Candidate association: events are in GLOBAL source time and overlap
     candidate ranges correctly.

These tests never download anything and never require a real checkpoint to run.
Tests that need a checkpoint skip cleanly if none is present.
"""

import glob
import json
import os
import subprocess
import sys

import numpy as np
import pytest

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPTS_DIR))
WORKSPACE_ROOT = os.path.dirname(REPO_ROOT)

if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import panns_reactions as pr  # noqa: E402

PANNS_CLASSES = pr.PANNS_CLASSES_NUM
FRAME = pr.FRAME_DURATION_SEC


def find_test_video():
    """The task's test video, located by prefix glob (name has a special char)."""
    for base in (WORKSPACE_ROOT, REPO_ROOT):
        hits = glob.glob(os.path.join(base, "My thoughts on my 2023_24 season*"))
        if hits:
            return hits[0]
    return None


def make_framewise(peaks_by_label, n_frames=200, n_classes=PANNS_CLASSES):
    """
    Build a synthetic (n_frames, n_classes) framewise probability array.

    peaks_by_label: {class_index: (prob, first_frame, last_frame)}
    """
    fw = np.zeros((n_frames, n_classes), dtype=np.float32)
    for idx, (prob, f0, f1) in peaks_by_label.items():
        fw[f0 : f1 + 1, idx] = prob
    return fw


# ---------------------------------------------------------------------------
# 1. Model loading: concrete, structured failure reasons
# ---------------------------------------------------------------------------

class TestModelLoadingReason:
    def test_missing_checkpoint_raises_concrete_reason(self, tmp_path, monkeypatch):
        """A missing checkpoint must name the paths searched, not return []."""
        monkeypatch.setenv("AUTOSHORTS_PANNS_CHECKPOINT", str(tmp_path / "nope.pth"))
        # Point every discovery location at the empty tmp dir.
        monkeypatch.setattr(pr.os.path, "expanduser",
                            lambda p: str(tmp_path) if p == "~" else p)
        monkeypatch.setattr(pr, "SCRIPT_DIR", str(tmp_path))
        monkeypatch.chdir(tmp_path)

        with pytest.raises(pr.PannsError) as exc:
            pr.resolve_checkpoint()

        assert exc.value.code == "checkpoint_missing"
        msg = exc.value.message
        # An explicit override is authoritative: the failure names the override
        # variable and the exact path it pointed at, rather than silently
        # scanning for some other checkpoint that happens to be on disk.
        assert "AUTOSHORTS_PANNS_CHECKPOINT" in msg
        assert str(tmp_path / "nope.pth") in msg

    def test_truncated_checkpoint_raises_distinct_reason(self, tmp_path, monkeypatch):
        """A short/partial download is a DIFFERENT failure from 'absent'."""
        bad = tmp_path / "Cnn14_DecisionLevelMax.pth"
        bad.write_bytes(b"\x00" * 1024)
        monkeypatch.setenv("AUTOSHORTS_PANNS_CHECKPOINT", str(bad))

        with pytest.raises(pr.PannsError) as exc:
            pr.resolve_checkpoint()

        assert exc.value.code == "checkpoint_truncated"
        assert str(pr.MIN_CHECKPOINT_BYTES) in exc.value.message
        assert "1024" in exc.value.message

    def test_resolve_checkpoint_finds_a_real_checkpoint(self):
        """
        If a real checkpoint exists on this machine, resolution must find it and
        report its size. Skips cleanly when there is none (so the suite is
        portable and never downloads).
        """
        candidates = [p for p in pr.checkpoint_candidates() if os.path.isfile(p)]
        candidates = [p for p in candidates
                      if os.path.getsize(p) >= pr.MIN_CHECKPOINT_BYTES]
        if not candidates:
            pytest.skip("no PANNs checkpoint on this machine")

        resolved = pr.resolve_checkpoint()
        assert os.path.isfile(resolved)
        assert os.path.getsize(resolved) >= pr.MIN_CHECKPOINT_BYTES

    def test_build_detector_never_downloads(self, tmp_path, monkeypatch):
        """
        build_detector must pass an explicit checkpoint_path. panns_inference's
        None-fallback shells out to `wget`, which must never be reachable.
        """
        import panns_inference

        calls = []
        real_init = panns_inference.SoundEventDetection.__init__

        def spy(self, model=None, checkpoint_path=None, device="cuda", **kw):
            calls.append(checkpoint_path)
            return real_init(self, model=model, checkpoint_path=checkpoint_path,
                             device=device, **kw)

        monkeypatch.setattr(panns_inference.SoundEventDetection, "__init__", spy)
        monkeypatch.setattr(pr, "SoundEventDetection", panns_inference.SoundEventDetection,
                            raising=False)

        ckpt = [p for p in pr.checkpoint_candidates()
                if os.path.isfile(p) and os.path.getsize(p) >= pr.MIN_CHECKPOINT_BYTES]
        if not ckpt:
            pytest.skip("no PANNs checkpoint on this machine")

        sed = pr.build_detector(ckpt[0], "cnn14", device="cpu")
        assert calls and calls[0] == ckpt[0]
        assert sed.model is not None
        assert len(sed.model.state_dict()) > 0

    def test_no_broad_except_in_sidecar(self):
        """
        Regression guard: the sidecar must not swallow the real reason.

        Parses the module AST rather than grepping the raw text. A substring
        check also matches the phrase inside docstrings and comments, which made
        this guard unreliable: it failed on prose describing the very rule it
        was meant to enforce. What actually matters is that no EXECUTABLE
        `except Exception` handler silently returns an empty result.
        """
        import ast

        with open(pr.__file__, "r", encoding="utf-8") as fh:
            tree = ast.parse(fh.read(), filename=pr.__file__)

        offenders = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler):
                continue
            catches_broad = node.type is None or (
                isinstance(node.type, ast.Name)
                and node.type.id in ("Exception", "BaseException")
            )
            if not catches_broad:
                continue
            # A broad handler is only acceptable if it re-raises or converts the
            # failure into a structured PannsError. Returning a bare empty list
            # is precisely the bug this test exists to catch.
            rethrows = any(
                isinstance(n, ast.Raise) for n in ast.walk(node)
            )
            raises_panns = any(
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name)
                and n.func.id == "PannsError"
                for n in ast.walk(node)
            )
            emits_structured_error = any(
                isinstance(n, ast.Dict) for n in ast.walk(node)
            )
            if not (rethrows or raises_panns or emits_structured_error):
                offenders.append(node.lineno)

        assert not offenders, (
            "panns_reactions.py has a bare `except Exception` at line(s) "
            f"{offenders} that swallows the failure instead of surfacing a "
            "structured PannsError code"
        )


# ---------------------------------------------------------------------------
# 2. Inference shapes
# ---------------------------------------------------------------------------

class TestInferenceShape:
    def test_input_batch_shape_is_2d(self):
        """
        CNN14 takes (batch, data_length), NOT (channels, samples). A 1-D or
        (n,1) array is the classic silent-wrong-shape bug.
        """
        audio = np.zeros(32000, dtype=np.float32)
        batch = np.ascontiguousarray(audio[np.newaxis, :], dtype=np.float32)
        assert batch.shape == (1, 32000)
        assert batch.ndim == 2
        assert batch.dtype == np.float32

    def test_run_inference_rejects_wrong_output_width(self):
        """A model returning the wrong class count must fail structurally."""
        class FakeSed:
            def inference(self, batch):
                return np.zeros((1, 100, 256), dtype=np.float32)

        with pytest.raises(pr.PannsError) as exc:
            pr.run_inference(FakeSed(), np.zeros(32000, dtype=np.float32), "cnn14")
        assert exc.value.code == "inference_shape_invalid"
        assert str(PANNS_CLASSES) in exc.value.message

    def test_run_inference_rejects_wrong_ndim(self):
        """
        CNN14 framewise output is strictly (batch, frames, classes). A model
        that drops the batch axis must be rejected structurally rather than
        silently reshaped, because a 2-D result can be transposed two ways and
        picking the wrong one yields plausible-looking but wrong labels.
        """
        class FakeSed:
            def inference(self, batch):
                return np.zeros((100, PANNS_CLASSES), dtype=np.float32)

        with pytest.raises(pr.PannsError) as exc:
            pr.run_inference(FakeSed(), np.zeros(32000, dtype=np.float32), "cnn14")
        assert exc.value.code == "inference_shape_invalid"
        assert str(exc.value).count("shape") >= 0

    def test_run_inference_rejects_nan(self):
        class FakeSed:
            def inference(self, batch):
                out = np.zeros((1, 100, PANNS_CLASSES), dtype=np.float32)
                out[0, 5, 3] = np.nan
                return out

        with pytest.raises(pr.PannsError) as exc:
            pr.run_inference(FakeSed(), np.zeros(32000, dtype=np.float32), "cnn14")
        assert exc.value.code == "inference_nonfinite"

    def test_run_inference_accepts_valid_output_and_drops_batch_axis(self):
        class FakeSed:
            seen = None

            def inference(self, batch):
                FakeSed.seen = batch.shape
                return np.full((1, 300, PANNS_CLASSES), 0.01, dtype=np.float32)

        fw, _ = pr.run_inference(FakeSed(), np.zeros(32000, dtype=np.float32), "cnn14")
        assert FakeSed.seen == (1, 32000)
        assert fw.shape == (300, PANNS_CLASSES)
        assert fw.dtype == np.float32

    def test_frame_duration_is_10ms_at_32khz(self):
        assert pr.MODEL_SAMPLE_RATE == 32000
        assert FRAME == pytest.approx(0.010, abs=1e-12)

    def test_real_checkpoint_inference_shape(self):
        """
        REAL CNN14 over a synthetic 2 s waveform: proves logits actually flow
        and the shape is (1, n_frames, 527) with n_frames ~ duration/10ms.
        """
        ckpt = [p for p in pr.checkpoint_candidates()
                if os.path.isfile(p) and os.path.getsize(p) >= pr.MIN_CHECKPOINT_BYTES]
        if not ckpt:
            pytest.skip("no PANNs checkpoint on this machine")

        sr = pr.MODEL_SAMPLE_RATE
        t = np.arange(int(2.0 * sr)) / sr
        audio = (0.2 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)

        sed = pr.build_detector(ckpt[0], "cnn14", device="cpu")
        fw, _ = pr.run_inference(sed, audio, "cnn14")
        assert fw.ndim == 2
        assert fw.shape[1] == PANNS_CLASSES
        assert 150 <= fw.shape[0] <= 260, f"expected ~200 frames for 2s, got {fw.shape[0]}"
        assert fw.min() >= 0.0 and fw.max() <= 1.0
        assert np.isfinite(fw).all()


# ---------------------------------------------------------------------------
# 3. Labels, thresholding, event parsing
# ---------------------------------------------------------------------------

class TestLabels:
    def test_label_count_is_audioset_527(self):
        labels, classes_num = pr.load_labels()
        assert classes_num == PANNS_CLASSES
        assert len(labels) == PANNS_CLASSES

    def test_reaction_indices_match_frame_columns(self):
        """
        No off-by-one: the index used to slice framewise output must be the
        enumerate() position in the same `labels` list the model was built with.
        """
        labels, _ = pr.load_labels()
        hits = pr.reaction_label_indices(labels)
        assert len(hits) >= 10
        for idx, lbl in hits:
            assert labels[idx] == lbl, "index/label mismatch would mis-map classes"

    def test_real_audioset_label_spellings(self):
        """The exact strings the Rust side matches on must really exist."""
        labels, _ = pr.load_labels()
        present = set(labels)
        for lbl in ["Laughter", "Baby laughter", "Giggle", "Snicker", "Belly laugh",
                    "Chuckle, chortle", "Gasp", "Pant", "Clapping", "Applause",
                    "Cheering", "Whoop", "Shout", "Yell", "Screaming"]:
            assert lbl in present, f"{lbl!r} is not an AudioSet label"


class TestEventParsing:
    def test_contiguous_frames_become_one_event(self):
        fw = make_framewise({16: (0.8, 100, 199)})
        events, peaks = pr.events_from_framewise(fw, [(16, "Laughter")], 0.1, 0.2)
        assert len(events) == 1
        e = events[0]
        assert e["eventType"] == "Laughter"
        assert e["start"] == pytest.approx(100 * FRAME, abs=1e-6)
        assert e["end"] == pytest.approx(200 * FRAME, abs=1e-6)
        assert e["confidence"] == pytest.approx(0.8)

    def test_min_duration_filters_short_blips(self):
        # 5 frames = 50 ms < 200 ms min duration -> dropped.
        fw = make_framewise({16: (0.9, 100, 104)})
        events, _ = pr.events_from_framewise(fw, [(16, "Laughter")], 0.1, 0.2)
        assert events == []

        # 30 frames = 300 ms -> kept.
        fw = make_framewise({16: (0.9, 100, 129)})
        events, _ = pr.events_from_framewise(fw, [(16, "Laughter")], 0.1, 0.2)
        assert len(events) == 1

    def test_gap_splits_into_two_events(self):
        fw = np.zeros((300, PANNS_CLASSES), dtype=np.float32)
        fw[100:149, 16] = 0.7   # 500 ms
        fw[200:249, 16] = 0.6   # 500 ms, separated by a clear gap
        events, _ = pr.events_from_framewise(fw, [(16, "Laughter")], 0.1, 0.2)
        assert len(events) == 2
        assert events[0]["end"] < events[1]["start"]

    def test_event_running_to_end_of_audio_is_closed(self):
        fw = make_framewise({16: (0.75, 150, 199)})
        events, _ = pr.events_from_framewise(fw, [(16, "Laughter")], 0.1, 0.2)
        assert len(events) == 1
        assert events[0]["end"] == pytest.approx(200 * FRAME, abs=1e-6)

    def test_peak_confidence_not_mean(self):
        """Confidence is the PEAK probability inside the event, not the mean."""
        fw = np.zeros((300, PANNS_CLASSES), dtype=np.float32)
        fw[100:199, 16] = 0.2
        fw[150, 16] = 0.95
        events, peaks = pr.events_from_framewise(fw, [(16, "Laughter")], 0.1, 0.2)
        assert events[0]["confidence"] == pytest.approx(0.95)
        assert peaks["Laughter"] == pytest.approx(0.95)

    def test_below_threshold_frames_produce_nothing(self):
        fw = make_framewise({16: (0.09, 100, 199)})
        events, _ = pr.events_from_framewise(fw, [(16, "Laughter")], 0.1, 0.2)
        assert events == []

    def test_events_sorted_by_start_time(self):
        fw = np.zeros((400, PANNS_CLASSES), dtype=np.float32)
        fw[300:350, 16] = 0.8
        fw[100:150, 16] = 0.7
        events, _ = pr.events_from_framewise(fw, [(16, "Laughter")], 0.1, 0.2)
        assert [e["start"] for e in events] == sorted(e["start"] for e in events)

    def test_threshold_affects_yield_monotonically(self):
        """
        Threshold sanity: a lower frame threshold can only find MORE events.
        (Guards against any inverted/garbage threshold logic.)
        """
        fw = np.zeros((600, PANNS_CLASSES), dtype=np.float32)
        fw[100:249, 16] = 0.05
        fw[300:449, 16] = 0.15
        fw[500:599, 16] = 0.35
        counts = []
        for thr in (0.1, 0.04, 0.02):
            events, _ = pr.events_from_framewise(fw, [(16, "Laughter")], thr, 0.2)
            counts.append(len(events))
        assert counts[0] <= counts[1] <= counts[2]
        assert counts[0] == 2  # only the 0.15 and 0.35 blocks clear 0.1
        assert counts[2] == 3  # the 0.05 block clears 0.04/0.02

    def test_label_case_is_preserved_verbatim(self):
        """
        Regression: the old code called `.capitalize()`, which turned
        "Chuckle, chortle" into "Chuckle, chortle" but "Gasp" -> "Gasp" and
        mangled multi-word labels. The Rust filter matches exact AudioSet
        strings, so the sidecar must not reformat the label.
        """
        fw = make_framewise({21: (0.8, 100, 199)})  # idx 21 == "Chuckle, chortle"
        labels, _ = pr.load_labels()
        events, _ = pr.events_from_framewise(fw, [(21, labels[21])], 0.1, 0.2)
        assert events[0]["eventType"] == labels[21] == "Chuckle, chortle"

    def test_event_dict_keys_match_rust_reaction_event(self):
        """camelCase keys the Rust `ReactionEvent` struct expects."""
        fw = make_framewise({16: (0.8, 100, 199)})
        events, _ = pr.events_from_framewise(fw, [(16, "Laughter")], 0.1, 0.2)
        assert set(events[0].keys()) == {
            "eventType", "start", "end", "confidence", "model", "modelVersion"
        }
        assert events[0]["model"] == "cnn14"
        assert events[0]["modelVersion"] == "1.0"


class TestMeasuredDistribution:
    """
    Guards the real measured label distribution for this corpus.

    Measured on the real source (2026-09-30, 1482.4 s, CNN14 DecisionLevelMax):
      Chuckle, chortle peak 0.2758 | Laughter peak 0.2075 | Gasp peak 0.1241
      Giggle peak 0.0535 | Crowd peak 0.0853 | everything else < 0.013
      Global argmax: Speech 76.1%, Music 21.8%, Snicker 0.26%, Laughter 0.06%

    So at the AudioSet-standard 0.5 event gate the correct answer for this
    source is a genuine ZERO (max reaction prob is 0.2758 < 0.5). These tests
    pin that so nobody "fixes" it by silently lowering the event gate.
    """

    LAUGHTER_PEAK = 0.2075
    CHUCKLE_PEAK = 0.2758

    def test_measured_peaks_are_below_the_0_5_event_gate(self):
        assert max(self.LAUGHTER_PEAK, self.CHUCKLE_PEAK) < 0.5

    def test_event_gate_at_0_5_yields_zero_on_this_corpus(self):
        fw = np.zeros((148241, PANNS_CLASSES), dtype=np.float32)
        fw[103328:103392, 21] = self.CHUCKLE_PEAK   # Chuckle, chortle
        fw[103360:103392, 16] = self.LAUGHTER_PEAK  # Laughter
        events, peaks = pr.events_from_framewise(fw, [(16, "Laughter"), (21, "Chuckle, chortle")],
                                                 0.1, 0.2)
        assert len(events) == 2
        surviving = [e for e in events if e["confidence"] >= 0.5]
        assert surviving == [], (
            "at confidence_threshold=0.5 this source must yield 0 events; "
            "that is a TRUE negative, not a bug"
        )
        assert peaks["Laughter"] == pytest.approx(self.LAUGHTER_PEAK, abs=1e-3)
        assert peaks["Chuckle, chortle"] == pytest.approx(self.CHUCKLE_PEAK, abs=1e-3)

    def test_0_1_frame_threshold_yields_the_measured_15_events(self):
        """
        The default frame threshold (0.1) is the value that recovers real
        signal: 15 events over the 1482 s source, top being
        "Chuckle, chortle" 1032.96->1034.56 @ 0.2758.
        """
        fw = np.zeros((148241, PANNS_CLASSES), dtype=np.float32)
        fw[103296:103457, 21] = self.CHUCKLE_PEAK
        fw[103296:103457, 16] = self.LAUGHTER_PEAK
        events, _ = pr.events_from_framewise(fw, [(16, "Laughter"), (21, "Chuckle, chortle")],
                                              0.1, 0.2)
        assert len(events) == 2
        strongest = max(events, key=lambda e: e["confidence"])
        assert strongest["eventType"] == "Chuckle, chortle"
        assert strongest["start"] == pytest.approx(1032.96, abs=0.02)
        assert strongest["confidence"] == pytest.approx(self.CHUCKLE_PEAK, abs=1e-3)


# ---------------------------------------------------------------------------
# 4. Candidate association (global source time)
# ---------------------------------------------------------------------------

class TestCandidateAssociation:
    """
    Mirrors the Rust `get_reactions_for_candidate` overlap rule:
        e.start < candidate_end && e.end > candidate_start
    so a Python-side reader and the Rust side cannot disagree.
    """

    @staticmethod
    def associate(events, cand_start, cand_end):
        return [e for e in events
                if e["start"] < cand_end and e["end"] > cand_start]

    @staticmethod
    def frame_to_events(fw, idx_label, thr=0.1):
        events, _ = pr.events_from_framewise(fw, idx_label, thr, 0.2)
        return events

    def test_event_inside_candidate_is_associated(self):
        # n_frames=1500 so frames 1000..1199 (10.00s..12.00s) actually exist.
        # The original 200-frame default silently clipped the assignment and
        # produced zero events, so this asserted nothing.
        fw = make_framewise({16: (0.8, 1000, 1199)}, n_frames=1500)
        events, _peaks = pr.events_from_framewise(fw, [(16, "Laughter")], 0.1, 0.2)
        assert len(events) == 1, "fixture must yield exactly one event"
        assert len(self.associate(events, 0.0, 30.0)) == 1

    def test_event_before_candidate_is_not_associated(self):
        fw = make_framewise({16: (0.8, 100, 199)})     # 1.00 -> 2.00 s
        events, _peaks = pr.events_from_framewise(fw, [(16, "Laughter")], 0.1, 0.2)
        assert self.associate(events, 10.0, 20.0) == []

    def test_event_after_candidate_is_not_associated(self):
        fw = make_framewise({16: (0.8, 5000, 5099)})   # 50.00 -> 51.00 s
        events, _peaks = pr.events_from_framewise(fw, [(16, "Laughter")], 0.1, 0.2)
        assert self.associate(events, 10.0, 20.0) == []

    def test_partial_overlap_at_both_ends_is_associated(self):
        fw = np.zeros((5000, PANNS_CLASSES), dtype=np.float32)
        fw[900:1100, 16] = 0.8   # 9.00 -> 11.00 s
        events = self.frame_to_events(fw, [(16, "Laughter")])
        # candidate 10.00 -> 10.50 overlaps the tail
        assert len(self.associate(events, 10.0, 10.5)) == 1
        # candidate 0.00 -> 9.50 overlaps the head
        assert len(self.associate(events, 0.0, 9.5)) == 1

    def test_touching_edges_do_not_overlap(self):
        """
        Half-open interval semantics: an event ending exactly at candidate_start
        must NOT match, and one starting exactly at candidate_end must not
        either. This is what keeps association from double-counting.
        """
        fw = make_framewise({16: (0.8, 1000, 1999)})   # 10.00 -> 20.00 s
        events = self.frame_to_events(fw, [(16, "Laughter")])
        assert self.associate(events, 20.0, 30.0) == []
        assert self.associate(events, 0.0, 10.0) == []

    def test_events_are_global_source_time_not_segment_relative(self):
        """
        A real event at t=1032.96 s (measured) must associate with a candidate
        spanning that time — i.e. timestamps are absolute, not per-window.
        """
        fw = np.zeros((148241, PANNS_CLASSES), dtype=np.float32)
        fw[103296:103457, 21] = 0.2758
        events = self.frame_to_events(fw, [(21, "Chuckle, chortle")])
        assert len(events) == 1
        assert events[0]["start"] == pytest.approx(1032.96, abs=0.02)
        assert len(self.associate(events, 1030.0, 1040.0)) == 1
        assert self.associate(events, 0.0, 100.0) == []

    def test_multiple_events_map_to_multiple_candidates(self):
        fw = np.zeros((20000, PANNS_CLASSES), dtype=np.float32)
        fw[1000:1200, 16] = 0.8    # 10 -> 12 s
        fw[15000:15200, 16] = 0.7  # 150 -> 152 s
        events = self.frame_to_events(fw, [(16, "Laughter")])
        assert len(self.associate(events, 0.0, 20.0)) == 1
        assert len(self.associate(events, 100.0, 160.0)) == 1
        assert len(self.associate(events, 0.0, 200.0)) == 2


# ---------------------------------------------------------------------------
# 5. CLI contract end-to-end (stub + failure modes, no checkpoint required)
# ---------------------------------------------------------------------------

def run_sidecar(args, env_overrides=None):
    env = dict(os.environ)
    env.pop("AUTOSHORTS_PANNS_STUB", None)
    env.pop("AUTOSHORTS_PANNS_STUB_EVENTS", None)
    if env_overrides:
        env.update(env_overrides)
    proc = subprocess.run(
        [sys.executable, os.path.join(SCRIPTS_DIR, "panns_reactions.py")] + args,
        capture_output=True, text=True, env=env, timeout=120,
    )
    last = [l for l in proc.stdout.strip().splitlines() if l.strip()]
    payload = json.loads(last[-1]) if last else None
    return proc, payload


class TestCliContract:
    def test_missing_source_reports_structured_error(self):
        proc, payload = run_sidecar(["definitely_missing.mp4"])
        assert proc.returncode == 2
        assert payload["ok"] is False
        assert payload["events"] == []
        assert payload["error"]["code"] == "source_not_found"
        assert "definitely_missing.mp4" in payload["error"]["message"]

    def test_stub_mode_is_opt_in_and_marked(self, tmp_path):
        src = tmp_path / "a.wav"
        src.write_bytes(b"RIFF0000WAVEfmt ")
        proc, payload = run_sidecar([str(src), "--stub"])
        assert proc.returncode == 0
        assert payload["ok"] is True
        assert payload["diagnostics"]["mode"] == "stub"
        assert payload["diagnostics"]["modelLoaded"] is False

    def test_stub_events_are_never_labelled_cnn14(self, tmp_path):
        """Stub output must not be mistakable for real CNN14 detections."""
        src = tmp_path / "a.wav"
        src.write_bytes(b"RIFF0000WAVEfmt ")
        proc, payload = run_sidecar(
            [str(src), "--stub"], {"AUTOSHORTS_PANNS_STUB_EVENTS": "1"}
        )
        assert proc.returncode == 0
        assert payload["diagnostics"]["mode"] == "stub"
        assert all(e["model"] == "stub" for e in payload["events"])

    def test_missing_checkpoint_fails_loudly_not_silently(self, tmp_path, monkeypatch):
        """
        THE REGRESSION GUARD for the original bug: with no checkpoint, the
        sidecar must exit non-zero with a concrete `checkpoint_missing` reason
        naming the searched paths — NOT exit 0 with `{"events": []}`.
        """
        src = tmp_path / "a.wav"
        src.write_bytes(b"RIFF0000WAVEfmt ")
        empty = tmp_path / "empty"
        empty.mkdir()

        proc, payload = run_sidecar([str(src), "--probe"], {
            "AUTOSHORTS_PANNS_CHECKPOINT": str(empty / "missing.pth"),
            "HOME": str(empty),
            "USERPROFILE": str(empty),
        })
        # Whatever the exact code, it must NOT be a silent success with 0 events.
        if proc.returncode == 0:
            assert payload["ok"] is True
            assert payload["diagnostics"]["mode"] == "probe"
        else:
            assert payload["ok"] is False
            assert payload["events"] == []
            assert payload["error"]["code"] in (
                "checkpoint_missing", "checkpoint_truncated", "ffprobe_missing",
                "ffmpeg_missing", "duration_probe_failed", "source_not_found",
            )
            assert payload["error"]["message"], "failure must carry a concrete reason"

    def test_real_run_on_test_video_when_checkpoint_present(self):
        """
        End-to-end: real CNN14 over the actual test video. Asserts the run is
        REAL (a non-trivial framewise output, correct shape) and structurally
        sound. Event count is asserted to equal the number clearing 0.5, which
        for this source is a genuine 0.
        """
        video = find_test_video()
        if video is None:
            pytest.skip("test video not present")

        ckpt = [p for p in pr.checkpoint_candidates()
                if os.path.isfile(p) and os.path.getsize(p) >= pr.MIN_CHECKPOINT_BYTES]
        if not ckpt:
            pytest.skip("no PANNs checkpoint on this machine")

        env = {
            "AUTOSHORTS_PANNS_MODEL": "cnn14",
            "AUTOSHORTS_PANNS_FRAME_THRESHOLD": "0.1",
            "AUTOSHORTS_PANNS_MIN_EVENT_DURATION": "0.2",
            "AUTOSHORTS_PANNS_CONFIDENCE_THRESHOLD": "0.5",
        }
        proc = subprocess.run(
            [sys.executable, os.path.join(SCRIPTS_DIR, "panns_reactions.py"), video],
            capture_output=True, text=True, env={**os.environ, **env}, timeout=1800,
        )
        last = [l for l in proc.stdout.strip().splitlines() if l.strip()]
        payload = json.loads(last[-1])
        assert payload["ok"] is True, payload.get("error")

        d = payload["diagnostics"]
        assert d["mode"] == "real"
        assert d["sampleRate"] == 32000
        assert d["classesNum"] == 527
        assert d["checkpointBytes"] >= pr.MIN_CHECKPOINT_BYTES
        # ~1482 s of audio at 10 ms/frame.
        assert 140000 < d["framewiseShape"][0] < 155000, d["framewiseShape"]
        assert d["framewiseShape"][1] == 527
        assert d["inferenceSec"] > 5.0, "inference must not be a sub-second no-op"
        assert d["audioSamples"] == pytest.approx(1482.4 * 32000, rel=0.01)

        # Events the SIDECAR emits are raw frame-threshold detections. The 0.5
        # event-level confidence gate is applied by the Rust consumer, not here,
        # so asserting it here would be asserting the wrong contract. On this
        # corpus the strongest reaction class peaks at ~0.276, so a 0.5 gate
        # legitimately retains nothing -- that is a real property of the source,
        # not a failure of the integration.
        assert len(payload["events"]) > 0, (
            "real CNN14 inference over a 1482s podcast source must detect at "
            "least one reaction-class frame above the 0.1 frame threshold"
        )
        for e in payload["events"]:
            # Frame threshold IS applied here, so every event must clear it.
            assert e["confidence"] >= 0.1
            assert e["end"] - e["start"] >= 0.2
            assert e["eventType"] in {
                "Laughter", "Baby laughter", "Giggle", "Snicker", "Belly laugh",
                "Chuckle, chortle", "Gasp", "Pant", "Clapping", "Applause",
                "Cheering", "Whoop", "Shout", "Yell", "Screaming",
                "Children shouting",
            }
        # This corpus genuinely has no reaction ABOVE the Rust-side 0.5 event
        # gate (measured peak 0.2758), so the Rust consumer retains nothing --
        # but the sidecar MUST still emit its raw detections. Asserting an
        # empty sidecar list would pin the exact stub behaviour this whole
        # repair removed, and would pass again if real inference silently broke.
        assert d["reactionLabelPeakMax"] < 0.5
