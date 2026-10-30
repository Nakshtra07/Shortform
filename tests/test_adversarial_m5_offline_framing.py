"""Milestone 5 Empirical Challenger Test Suite: Offline Invariant & Framing Math.

Empirically stress-tests:
1. Strict Air-Gap Verification:
   - Blocks all network and socket operations (socket.socket, create_connection, urllib, http.client).
   - Verifies autoclip_adapter.py and all autoclip_studio services execute offline with ZERO network/LLM calls.
   - Verifies draft.language == 'source' invariant and fail-closed text_json placeholder behavior.
2. Framing Algorithm Stress-Testing:
   - Hold-segment tracking (min_hold=2) on speaker-switching sequences, glances, oscillation, and multi-speaker transitions.
   - Non-face frames and scenes: verify mode: "fit" is assigned with blurred backdrop filter generation.
   - Boundary aspect ratios: all 16 combinations of 16:9, 9:16, 1:1, 21:9 and extreme 32:9, 9:32, 100:1.
"""
from __future__ import annotations

import io
import json
import math
import os
import re
import socket
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# Ensure scripts and autoclip_studio are importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "autoshorts" / "src-tauri" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import autoclip_adapter
from autoclip_adapter import (
    build_draft_and_scene,
    generate_srt_string,
    group_words_into_cues,
    run_adapter,
    validate_payload,
)
from autoclip_studio import (
    TITLE_STYLES,
    CropPoint,
    Draft,
    Packaging,
    Scene,
    caption_layout,
    framing,
    platform_strategy,
    publish_export,
    render,
    title_art,
    video_encoder,
)
from autoclip_studio.ffmpeg_utils import get_ffmpeg_path


class AirgapBreachError(RuntimeError):
    """Raised when any socket or network operation is attempted."""
    pass


# ==============================================================================
# 1. STRICT AIR-GAP & ZERO-EXTERNAL-CALL VERIFICATION
# ==============================================================================

class TestAirgapOfflineInvariants:
    """Verifies complete offline isolation: ZERO socket or network operations permitted."""

    @pytest.fixture
    def strict_airgap(self, monkeypatch):
        """Block all socket creation, connection, and DNS lookups."""
        def deny_call(*args, **kwargs):
            raise AirgapBreachError("CRITICAL AIRGAP BREACH: Network or socket operation attempted!")

        class BlockedSocket:
            def __init__(self, *args, **kwargs):
                raise AirgapBreachError("CRITICAL AIRGAP BREACH: Socket creation attempted!")

        # Monkeypatch standard socket interfaces
        monkeypatch.setattr(socket, "socket", BlockedSocket)
        monkeypatch.setattr(socket, "create_connection", deny_call)
        monkeypatch.setattr(socket, "getaddrinfo", deny_call)
        monkeypatch.setattr(socket, "gethostbyname", deny_call)
        monkeypatch.setattr(socket, "gethostbyname_ex", deny_call)
        monkeypatch.setattr(socket, "getfqdn", deny_call)

        # Monkeypatch urllib.request.urlopen
        import urllib.request
        monkeypatch.setattr(urllib.request, "urlopen", deny_call)

        # Monkeypatch http.client connections
        import http.client
        monkeypatch.setattr(http.client.HTTPConnection, "connect", deny_call)
        monkeypatch.setattr(http.client.HTTPSConnection, "connect", deny_call)

        yield

    @pytest.fixture
    def synthetic_media(self, tmp_path):
        """Create a short 1-second synthetic media file for offline rendering."""
        vid_path = tmp_path / "airgap_source_1s.mp4"
        cmd = [
            get_ffmpeg_path(),
            "-y",
            "-f", "lavfi", "-i", "testsrc=duration=1:size=320x240:rate=30",
            "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
            "-t", "1",
            "-c:v", "libx264",
            "-c:a", "aac",
            str(vid_path),
        ]
        res = subprocess.run(cmd, capture_output=True, timeout=30)
        assert res.returncode == 0, f"FFmpeg failed: {res.stderr.decode()}"
        assert vid_path.is_file()
        return vid_path

    def test_airgap_run_adapter_end_to_end(self, strict_airgap, synthetic_media, tmp_path, monkeypatch):
        """Run complete autoclip_adapter pipeline with real synthetic video under active socket blocking."""
        test_data_dir = tmp_path / "airgap_data"
        monkeypatch.setenv("AUTOCLIP_DATA_DIR", str(test_data_dir))

        target_out = tmp_path / "airgap_rendered_output.mp4"
        payload = {
            "project_id": "proj_airgap_verify",
            "video_path": str(synthetic_media),
            "output_path": str(target_out),
            "candidate": {
                "id": "cand_airgap_01",
                "start_sec": 0.1,
                "end_sec": 0.8,
                "hook": "Offline Invariant Guarantee",
                "score": 0.99,
            },
            "transcript_words": [
                {"text": "Zero", "start": 0.15, "end": 0.35, "speaker": "A"},
                {"text": "network", "start": 0.35, "end": 0.55, "speaker": "A"},
                {"text": "calls!", "start": 0.55, "end": 0.75, "speaker": "A"},
            ],
            "presentation_config": {
                "platform_strategy": "bilibili",
                "subtitle_style": "clean",
                "title_style": "plain",
                "template": "none",
            },
        }

        # Must execute cleanly with zero socket calls
        result = run_adapter(payload)
        assert result["status"] == "success"
        assert target_out.is_file()
        assert target_out.stat().st_size > 0
        assert result["strategy_id"] == "bilibili"

    def test_airgap_autoclip_studio_services_sweep(self, strict_airgap):
        """Invoke all autoclip_studio submodules and verify zero network calls."""
        # 1. Platform strategy lookup
        for strat_name in ["douyin", "tiktok", "bilibili", "youtube_shorts", "xiaohongshu", "original"]:
            strat = platform_strategy(strat_name)
            assert strat.id == strat_name

        # 2. Caption layout typography & wrapping
        screens = caption_layout.split_screens("离线模式文本排版测试，保证无外部网络调用。", 15.0)
        assert len(screens) >= 1
        timed = caption_layout.timed_screens("测试字幕", 0.0, 5.0, 15.0)
        assert len(timed) >= 1

        # 3. Title artwork rendering for all styles (using version 6 as configured in adapter)
        for style in TITLE_STYLES:
            png = title_art.png_bytes("Offline Title", style, 1080, 1920, version=6)
            assert isinstance(png, bytes)
            assert len(png) > 100

        # 4. Video encoder detection
        enc = video_encoder.encoder()
        assert isinstance(enc, str)

        # 5. Framing math operations
        wf = framing.window_fraction(1920, 1080, 1080, 1920)
        assert wf == pytest.approx(0.31640625, abs=1e-5)
        cx = framing.crop_x_for_center(0.5, wf)
        assert cx == pytest.approx(0.5)

        # 6. Publish export filter construction
        req = publish_export.ExportRequest("proj_dummy", "draft_dummy", layout="blur")
        spec = {"layout": "blur", "w": 1080, "h": 1920}
        graph, last = publish_export._build_filter(req, spec, None, None, None, "clean")
        assert "blurred_backdrop" in graph or "split=2[bg][fg]" in graph

    def test_airgap_llm_placeholder_fails_closed(self):
        """Verify render.text_json explicitly refuses any LLM calls."""
        with pytest.raises(RuntimeError, match="External text model calls disabled in offline mode"):
            render.text_json("prompt", {})

    def test_airgap_draft_language_is_strictly_source(self):
        """Verify build_draft_and_scene strictly sets draft.language = 'source'."""
        dummy_payload = {
            "project_id": "proj_p",
            "video_path": "dummy.mp4",
            "output_path": "out.mp4",
            "candidate": {"id": "c1", "start_sec": 1.0, "end_sec": 4.0, "hook": "Test"},
            "presentation_config": {"platform_strategy": "douyin"},
        }
        draft, scene, strat = build_draft_and_scene(dummy_payload, [])
        assert draft.language == "source"

    def test_airgap_adversarial_non_source_language_fails_closed(self, tmp_path, synthetic_media, monkeypatch):
        """Adversarial stress: if draft.language is maliciously changed to 'zh' with English text,
        render_draft must fail closed via text_json rather than making a network request.
        """
        test_data_dir = tmp_path / "airgap_data_adv"
        monkeypatch.setenv("AUTOCLIP_DATA_DIR", str(test_data_dir))

        proj_id = "proj_adv_lang"
        proj_dir = test_data_dir / "projects" / proj_id
        meta_dir = proj_dir / "metadata"
        meta_dir.mkdir(parents=True, exist_ok=True)
        # Write English SRT entries so _needs_translation returns True
        srt_file = meta_dir / "input.srt"
        srt_file.write_text("1\n00:00:00,100 --> 00:00:00,800\nThis is English text that needs translation.\n", encoding="utf-8")

        scene = Scene(id="s1", start=0.1, end=0.8, crop_x=0.5)
        draft = Draft(
            id="cand_adv",
            title="Adversarial Translation Attack",
            hook="English Hook",
            scenes=[scene],
            language="zh",  # Non-source language to trigger translation path
            aspect="portrait",
            layout="fit",
            subtitles=True,
            packaging=None,
        )

        with pytest.raises(RuntimeError, match="External text model calls disabled in offline mode"):
            render.render_draft(
                project_id=proj_id,
                video=synthetic_media,
                draft=draft,
                job_id="adv_job",
                progress=lambda p: None,
            )

    def test_airgap_subprocess_cli_with_injected_socket_blocker(self, synthetic_media, tmp_path):
        """Spawn autoclip_adapter.py in a separate Python subprocess where sitecustomize blocks socket."""
        adapter_script = SCRIPTS_DIR / "autoclip_adapter.py"
        test_data_dir = tmp_path / "subp_airgap_data"
        target_out = tmp_path / "subp_airgap_out.mp4"
        payload_file = tmp_path / "subp_payload.json"

        payload = {
            "project_id": "proj_subp_airgap",
            "video_path": str(synthetic_media),
            "output_path": str(target_out),
            "candidate": {
                "id": "cand_subp_airgap",
                "start_sec": 0.1,
                "end_sec": 0.7,
                "hook": "Subprocess Airgap Test",
            },
            "transcript_words": [
                {"text": "Isolated", "start": 0.15, "end": 0.35},
                {"text": "Execution", "start": 0.4, "end": 0.65},
            ],
            "presentation_config": {
                "platform_strategy": "bilibili",
            },
        }
        payload_file.write_text(json.dumps(payload), encoding="utf-8")

        # Create sitecustomize directory with socket blocking
        site_dir = tmp_path / "site_blocker"
        site_dir.mkdir(parents=True, exist_ok=True)
        blocker_code = (
            "import socket\n"
            "class BlockedSocket:\n"
            "    def __init__(self, *args, **kwargs):\n"
            "        raise RuntimeError('Subprocess Airgap Breach!')\n"
            "def deny(*a, **k):\n"
            "    raise RuntimeError('Subprocess Airgap Breach!')\n"
            "socket.socket = BlockedSocket\n"
            "socket.create_connection = deny\n"
        )
        (site_dir / "sitecustomize.py").write_text(blocker_code, encoding="utf-8")

        env = os.environ.copy()
        env["AUTOCLIP_DATA_DIR"] = str(test_data_dir)
        env["PYTHONPATH"] = str(site_dir) + os.pathsep + env.get("PYTHONPATH", "")

        proc = subprocess.run(
            [sys.executable, str(adapter_script), "--payload", str(payload_file)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env,
            timeout=60,
        )
        assert proc.returncode == 0, f"Subprocess failed with stderr: {proc.stderr}\nstdout: {proc.stdout}"
        stdout_json = json.loads(proc.stdout)
        assert stdout_json["status"] == "success"
        assert target_out.is_file()


# ==============================================================================
# 2. FRAMING ALGORITHM: HOLD-SEGMENT TRACKING STRESS TESTS
# ==============================================================================

class TestFramingHoldSegmentTracking:
    """Stress tests framing.auto_frame hold-segment tracking (min_hold=2) on speaker-switching sequences."""

    def test_single_glance_rejected_in_auto_frame(self):
        """A single-sample glance to a secondary speaker must NOT trigger a crop window switch."""
        scene = Scene(id="scene_glance", start=0.0, end=15.0, crop_x=0.5)
        draft = Draft(id="d_glance", title="Glance Test", scenes=[scene], aspect="portrait", layout="crop")

        # 15s long shot with samples at intervals of 2.5s:
        # Speaker A at x=0.2 (t=0.3, 2.8, 5.3), glance to B at x=0.85 (t=7.8), return to A at x=0.2 (t=10.3, 12.8)
        shot_samples = [
            (0.3, 0.20),
            (2.8, 0.20),
            (5.3, 0.21),
            (7.8, 0.85),  # Single glance spike
            (10.3, 0.20), # Return
            (12.8, 0.20),
        ]
        scans = [{"shots": [(0.0, 15.0, shot_samples)], "faces": 6, "samples": 6}]

        res = framing.auto_frame(Path("dummy.mp4"), draft, 1920, 1080, scans=scans)
        scene_info = res["scenes"][0]

        # Invariants:
        assert scene_info["switches"] == 0, "Single glance must not cause a switch"
        assert len(scene_info["crop_track"]) == 1
        assert scene_info["crop_track"][0]["start"] == 0.0
        # Window remains on Speaker A
        expected_crop_x = framing.crop_x_for_center(0.20, res["window_fraction"])
        assert scene_info["crop_track"][0]["crop_x"] == pytest.approx(expected_crop_x, abs=0.01)

    def test_persistent_turn_switches_at_start_of_turn(self):
        """When a new speaker persists for >= min_hold (2) samples, window switches at start of turn."""
        scene = Scene(id="scene_persist", start=0.0, end=15.0, crop_x=0.5)
        draft = Draft(id="d_persist", title="Persist Test", scenes=[scene], aspect="portrait", layout="crop")

        # Speaker A at x=0.2 (t=0.3, 2.8), Speaker B at x=0.8 (t=5.3, 7.8, 10.3, 12.8)
        shot_samples = [
            (0.3, 0.20),
            (2.8, 0.20),
            (5.3, 0.80),  # First sample of Speaker B's turn
            (7.8, 0.82),  # Second sample (persisted!)
            (10.3, 0.81),
            (12.8, 0.80),
        ]
        scans = [{"shots": [(0.0, 15.0, shot_samples)], "faces": 6, "samples": 6}]

        res = framing.auto_frame(Path("dummy.mp4"), draft, 1920, 1080, scans=scans)
        scene_info = res["scenes"][0]

        assert scene_info["switches"] == 1
        assert len(scene_info["crop_track"]) == 2

        track0, track1 = scene_info["crop_track"]
        assert track0["start"] == 0.0
        assert track0["mode"] == "crop"

        # The switch must take place at t=5.3 (start timestamp of Speaker B's first sample!)
        assert track1["start"] == pytest.approx(5.3, abs=0.01)
        assert track1["mode"] == "crop"
        expected_b = framing.crop_x_for_center(0.81, res["window_fraction"])
        assert track1["crop_x"] == pytest.approx(expected_b, abs=0.05)

    def test_rapid_alternating_glances_suppressed(self):
        """Oscillating between two speakers with 1 sample each: camera must hold steady on first speaker."""
        scene = Scene(id="scene_osc", start=0.0, end=18.0, crop_x=0.5)
        draft = Draft(id="d_osc", title="Oscillation Test", scenes=[scene], aspect="portrait", layout="crop")

        shot_samples = [
            (0.3, 0.2),
            (2.8, 0.8),  # glance 1
            (5.3, 0.2),  # return
            (7.8, 0.8),  # glance 2
            (10.3, 0.2), # return
            (12.8, 0.8), # glance 3
            (15.3, 0.2), # return
        ]
        scans = [{"shots": [(0.0, 18.0, shot_samples)], "faces": 7, "samples": 7}]

        res = framing.auto_frame(Path("dummy.mp4"), draft, 1920, 1080, scans=scans)
        scene_info = res["scenes"][0]

        assert scene_info["switches"] == 0
        assert len(scene_info["crop_track"]) == 1
        expected_a = framing.crop_x_for_center(0.2, res["window_fraction"])
        assert scene_info["crop_track"][0]["crop_x"] == pytest.approx(expected_a, abs=0.01)

    def test_three_speaker_conversation_transitions(self):
        """Speaker A (2 samples) -> Speaker B (1 sample glance) -> Speaker C (2 samples persist) -> Speaker A (2 samples persist)."""
        scene = Scene(id="scene_3spk", start=0.0, end=20.0, crop_x=0.5)
        draft = Draft(id="d_3spk", title="Three Speaker Test", scenes=[scene], aspect="portrait", layout="crop")

        shot_samples = [
            (0.3, 0.15),  # Speaker A
            (2.8, 0.15),  # Speaker A
            (5.3, 0.50),  # Speaker B (glance)
            (7.8, 0.85),  # Speaker C (sample 1)
            (10.3, 0.86), # Speaker C (sample 2 - persisted!)
            (12.8, 0.85), # Speaker C
            (15.3, 0.15), # Speaker A (sample 1)
            (17.8, 0.16), # Speaker A (sample 2 - persisted!)
        ]
        scans = [{"shots": [(0.0, 20.0, shot_samples)], "faces": 8, "samples": 8}]

        res = framing.auto_frame(Path("dummy.mp4"), draft, 1920, 1080, scans=scans)
        scene_info = res["scenes"][0]

        # Must ignore B, switch to C at t=7.8, then switch back to A at t=15.3
        assert scene_info["switches"] == 2
        assert len(scene_info["crop_track"]) == 3

        t0, t1, t2 = scene_info["crop_track"]
        assert t0["start"] == 0.0
        assert t1["start"] == pytest.approx(7.8, abs=0.01)
        assert t2["start"] == pytest.approx(15.3, abs=0.01)

    def test_subthreshold_jitter_rejected(self):
        """Jitter below min_jump (0.12) must not trigger any new crop points."""
        scene = Scene(id="scene_jitter", start=0.0, end=15.0, crop_x=0.5)
        draft = Draft(id="d_jitter", title="Jitter Test", scenes=[scene], aspect="portrait", layout="crop")

        # Center drifts slightly between 0.48 and 0.53 (diff <= 0.05 < 0.12)
        shot_samples = [
            (0.3, 0.50),
            (2.8, 0.53),
            (5.3, 0.48),
            (7.8, 0.52),
            (10.3, 0.49),
            (12.8, 0.51),
        ]
        scans = [{"shots": [(0.0, 15.0, shot_samples)], "faces": 6, "samples": 6}]

        res = framing.auto_frame(Path("dummy.mp4"), draft, 1920, 1080, scans=scans)
        scene_info = res["scenes"][0]
        assert scene_info["switches"] == 0
        assert len(scene_info["crop_track"]) == 1

    def test_short_shots_under_6s_do_not_switch_internally(self):
        """In shots <= LONG_SHOT (6.0s), frame_shot locks to median center without internal switches."""
        scene = Scene(id="scene_short", start=0.0, end=5.0, crop_x=0.5)
        draft = Draft(id="d_short", title="Short Shot Test", scenes=[scene], aspect="portrait", layout="crop")

        # Even if samples switch between 0.2 and 0.8:
        shot_samples = [
            (0.3, 0.2),
            (2.5, 0.8),
            (4.0, 0.8),
        ]
        scans = [{"shots": [(0.0, 5.0, shot_samples)], "faces": 3, "samples": 3}]

        res = framing.auto_frame(Path("dummy.mp4"), draft, 1920, 1080, scans=scans)
        scene_info = res["scenes"][0]
        assert scene_info["switches"] == 0
        assert len(scene_info["crop_track"]) == 1


# ==============================================================================
# 3. FRAMING ALGORITHM: NON-FACE FRAMES, FIT MODE & BLURRED BACKDROP
# ==============================================================================

class TestFramingNonFaceFitAndBlurredBackdrop:
    """Stress tests non-face frames and verifies mode: 'fit' with blurred backdrop filter generation."""

    def test_non_face_shot_in_multishot_scene_produces_fit_mode(self):
        """A scene with Shot 1 (talking head) and Shot 2 (slides/no face)."""
        scene = Scene(id="scene_slides", start=0.0, end=14.0, crop_x=0.5)
        draft = Draft(id="d_slides", title="Slide Test", scenes=[scene], aspect="portrait", layout="crop")

        shot1_samples = [(0.3, 0.35), (2.8, 0.35), (5.3, 0.35)]  # Shot 1: [0.0, 7.0) with face
        shot2_samples = [(0.3, None), (2.8, None), (5.3, None)]  # Shot 2: [7.0, 14.0) no face

        scans = [{
            "shots": [
                (0.0, 7.0, shot1_samples),
                (7.0, 14.0, shot2_samples),
            ],
            "faces": 3,
            "samples": 6,
        }]

        res = framing.auto_frame(Path("dummy.mp4"), draft, 1920, 1080, scans=scans)
        scene_info = res["scenes"][0]

        # Invariants:
        assert scene_info["fit_shots"] == 1
        assert len(scene_info["crop_track"]) == 2

        pt1, pt2 = scene_info["crop_track"]
        assert pt1["start"] == 0.0
        assert pt1["mode"] == "crop"

        assert pt2["start"] == 7.0
        assert pt2["mode"] == "fit"
        assert pt2["crop_x"] == 0.5

    def test_layout_filter_generates_blurred_backdrop_when_fit_shots_present(self):
        """Verify layout_filter produces the 3-way split with blurred_backdrop when fit mode is used."""
        scene = Scene(
            id="scene_filter",
            start=0.0,
            end=14.0,
            crop_x=0.35,
            crop_track=[
                CropPoint(start=0.0, crop_x=0.35, mode="crop"),
                CropPoint(start=7.0, crop_x=0.50, mode="fit"),
            ],
        )

        filter_graph = framing.layout_filter(scene, fallback=0.5, w=1080, h=1920)

        # 1. Must use 3-way split: [c] for cropped, [bg] for blurred background, [fg] for fitted foreground
        assert "[0:v]split=3[c][bg][fg]" in filter_graph

        # 2. Must contain blurred_backdrop logic with 1/8 downscaled gblur
        # sw = 1080 // 8 // 2 * 2 = 134, sh = 1920 // 8 // 2 * 2 = 240
        assert "scale=134:240:force_original_aspect_ratio=increase" in filter_graph
        assert "gblur=sigma=6" in filter_graph
        assert "colorlevels=romax=0.35:gomax=0.35:bomax=0.35" in filter_graph
        assert "scale=1080:1920" in filter_graph

        # 3. Must overlay fitted foreground over blurred background
        assert "[bg2][fg2]overlay=(W-w)/2:(H-h)/2[fitted]" in filter_graph

        # 4. Must enable fitted stream only during the fit span between(t,7.0,14.0)
        assert "[cropped][fitted]overlay=0:0:enable='between(t,7.0,14.0)'[base]" in filter_graph

    def test_purely_cropped_scene_omits_blurred_backdrop(self):
        """When all shots have faces (no fit mode), layout_filter produces clean single-stream crop."""
        scene = Scene(
            id="scene_pure_crop",
            start=0.0,
            end=10.0,
            crop_x=0.35,
            crop_track=[
                CropPoint(start=0.0, crop_x=0.35, mode="crop"),
            ],
        )

        filter_graph = framing.layout_filter(scene, fallback=0.5, w=1080, h=1920)

        # Must NOT split or instantiate blurred_backdrop
        assert "split=3" not in filter_graph
        assert "blurred_backdrop" not in filter_graph
        assert "gblur" not in filter_graph
        assert "[0:v]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920" in filter_graph
        assert "[base]" in filter_graph

    def test_majority_non_face_samples_assigns_fit_mode(self):
        """If more than half of the samples in a shot have no face, shot is assigned fit mode."""
        shot_samples = [
            (0.3, None),
            (1.5, 0.4),   # single stray face (e.g. background poster or false positive)
            (3.0, None),
            (4.5, None),
        ]
        # len(seen) = 1, len(samples) = 4. len(seen)*2 = 2 < 4 -> fit mode!
        points = framing.frame_shot(shot_samples, shot_start=2.0, shot_length=5.0, fraction=0.316)
        assert len(points) == 1
        assert points[0]["mode"] == "fit"
        assert points[0]["crop_x"] == 0.5
        assert points[0]["start"] == 2.0

    def test_entirely_non_face_draft_resets_crop_track(self):
        """When NO scenes in the draft have any faces, crop_track is reset to None (preserving user layout)."""
        scene1 = Scene(id="s1", start=0.0, end=5.0)
        scene2 = Scene(id="s2", start=5.0, end=10.0)
        draft = Draft(id="d_empty", title="No Faces", scenes=[scene1, scene2], aspect="portrait")

        scans = [
            {"shots": [(0.0, 5.0, [(1.0, None), (3.0, None)])], "faces": 0, "samples": 2},
            {"shots": [(0.0, 5.0, [(1.0, None), (3.0, None)])], "faces": 0, "samples": 2},
        ]

        res = framing.auto_frame(Path("dummy.mp4"), draft, 1920, 1080, scans=scans)
        for s in res["scenes"]:
            assert s["crop_x"] is None
            assert s["crop_track"] is None
            assert s["fit_shots"] == 0
            assert s["switches"] == 0

    def test_multiscene_draft_one_scene_has_faces_one_has_none(self):
        """When Scene 1 has faces and Scene 2 has none, Scene 2 keeps fit mode and blurred backdrop."""
        scene1 = Scene(id="s1", start=0.0, end=5.0)
        scene2 = Scene(id="s2", start=5.0, end=10.0)
        draft = Draft(id="d_mixed", title="Mixed Scenes", scenes=[scene1, scene2], aspect="portrait")

        scans = [
            {"shots": [(0.0, 5.0, [(1.0, 0.4), (3.0, 0.4)])], "faces": 2, "samples": 2},
            {"shots": [(0.0, 5.0, [(1.0, None), (3.0, None)])], "faces": 0, "samples": 2},
        ]

        res = framing.auto_frame(Path("dummy.mp4"), draft, 1920, 1080, scans=scans)
        s1_info = res["scenes"][0]
        s2_info = res["scenes"][1]

        # Scene 1: crop
        assert s1_info["crop_track"][0]["mode"] == "crop"

        # Scene 2: fit is preserved because any(faces) across draft is True!
        assert s2_info["crop_track"][0]["mode"] == "fit"
        assert s2_info["crop_track"][0]["crop_x"] == 0.5
        assert s2_info["fit_shots"] == 1


# ==============================================================================
# 4. FRAMING ALGORITHM: BOUNDARY ASPECT RATIOS (16:9, 9:16, 1:1, 21:9)
# ==============================================================================

class TestFramingBoundaryAspectRatios:
    """Stress tests boundary aspect ratios (16:9, 9:16, 1:1, 21:9) and extreme limits."""

    STANDARD_RATIOS = {
        "16:9": (1920, 1080),
        "9:16": (1080, 1920),
        "1:1":  (1080, 1080),
        "21:9": (2560, 1080),
    }

    @pytest.mark.parametrize("src_name, src_dims", list(STANDARD_RATIOS.items()))
    @pytest.mark.parametrize("dst_name, dst_dims", list(STANDARD_RATIOS.items()))
    def test_full_aspect_ratio_matrix_16_combinations(self, src_name, src_dims, dst_name, dst_dims):
        """Exhaustively verify all 16 combinations of 16:9, 9:16, 1:1, 21:9 input and output."""
        src_w, src_h = src_dims
        dst_w, dst_h = dst_dims

        # 1. Window fraction
        wf = framing.window_fraction(src_w, src_h, dst_w, dst_h)
        assert 0.0 < wf <= 1.0, f"Invalid window_fraction {wf} for {src_name} -> {dst_name}"

        # Mathematical formula verification
        scale = max(dst_w / src_w, dst_h / src_h)
        expected_wf = min(1.0, dst_w / (src_w * scale))
        assert wf == pytest.approx(expected_wf, rel=1e-6)

        # 2. Crop x for center across span [0.0, 1.0]
        for cx in [0.0, 0.25, 0.5, 0.75, 1.0]:
            crop_x = framing.crop_x_for_center(cx, wf)
            assert 0.0 <= crop_x <= 1.0

        # 3. auto_frame execution
        clean_src = src_name.replace(":", "_")
        clean_dst = dst_name.replace(":", "_")
        scene = Scene(id=f"scene_{clean_src}_{clean_dst}", start=0.0, end=10.0, crop_x=0.5)
        draft = Draft(id=f"d_{clean_src}_{clean_dst}", title="Aspect Test", scenes=[scene], aspect="portrait")
        scans = [{"shots": [(0.0, 10.0, [(1.0, 0.3), (5.0, 0.7), (8.0, 0.7)])], "faces": 3, "samples": 3}]

        res = framing.auto_frame(Path("dummy.mp4"), draft, src_w, src_h, window=(dst_w, dst_h), scans=scans)
        assert res["window_fraction"] == round(wf, 3)

        track = res["scenes"][0]["crop_track"]
        assert len(track) >= 1
        for pt in track:
            assert 0.0 <= pt["crop_x"] <= 1.0
            assert pt["mode"] in ("crop", "fit")

        # 4. Crop expression syntax
        scene_with_track = Scene(id="s", start=0.0, end=10.0, crop_x=track[0]["crop_x"],
                                 crop_track=[CropPoint(**p) for p in track])
        expr = framing.crop_expression(scene_with_track, fallback=0.5)
        assert isinstance(expr, str)
        assert len(expr) > 0
        # Valid FFmpeg characters only
        assert re.match(r"^[0-9.,()a-z_iflt<>=+-]+$", expr)

        # 5. Layout filter generation
        filt = framing.layout_filter(scene_with_track, fallback=0.5, w=dst_w, h=dst_h)
        assert f"scale={dst_w}:{dst_h}" in filt or f"crop={dst_w}:{dst_h}" in filt
        assert "[base]" in filt

    @pytest.mark.parametrize("src_w, src_h, dst_w, dst_h", [
        # Ultra-wide 32:9 -> 9:16 portrait
        (5120, 1440, 1080, 1920),
        # Ultra-tall 9:32 portrait -> 9:16 portrait
        (1080, 3840, 1080, 1920),
        # Extreme panoramic 100:1 -> 9:16
        (10000, 100, 1080, 1920),
        # Extreme vertical sliver 1:100 -> 9:16
        (100, 10000, 1080, 1920),
        # 16:9 -> Ultra-wide 32:9
        (1920, 1080, 5120, 1440),
        # 16:9 -> Ultra-tall 9:32
        (1920, 1080, 1080, 3840),
        # Tiny resolution 2x2
        (2, 2, 1080, 1920),
        (1080, 1920, 2, 2),
    ])
    def test_extreme_aspect_ratios_numerical_stability(self, src_w, src_h, dst_w, dst_h):
        """Ensure no divide-by-zero, NaN, Inf, or out-of-bounds float issues at extreme limits."""
        wf = framing.window_fraction(src_w, src_h, dst_w, dst_h)
        assert isinstance(wf, float)
        assert not math.isnan(wf)
        assert not math.isinf(wf)
        assert 0.0 < wf <= 1.0

        for cx in [-10.0, 0.0, 0.5, 1.0, 10.0]:
            cx_res = framing.crop_x_for_center(cx, wf)
            assert not math.isnan(cx_res)
            assert not math.isinf(cx_res)
            assert 0.0 <= cx_res <= 1.0

    def test_platform_strategy_aspect_resolution_consistency(self):
        """Verify each platform strategy correctly maps to expected aspect ratios and dimensions."""
        strategies = {
            "douyin": ("portrait", 1080, 1920, "blur"),
            "tiktok": ("portrait", 1080, 1920, "crop"),
            "instagram_reels": ("portrait", 1080, 1920, "crop"),
            "youtube_shorts": ("portrait", 1080, 1920, "crop"),
            "bilibili": ("landscape", 1920, 1080, "fit"),
            "xiaohongshu": ("portrait", 1080, 1920, "blur"),
            "original": ("original", None, None, "none"),
        }
        for strat_id, (expected_aspect, exp_w, exp_h, exp_layout) in strategies.items():
            strat = platform_strategy(strat_id)
            assert strat.aspect == expected_aspect
            assert strat.layout == exp_layout
            if exp_w and exp_h:
                # auto_frame default window for this aspect
                out_w, out_h = {"portrait": (1080, 1920), "landscape": (1920, 1080)}[strat.aspect]
                assert out_w == exp_w
                assert out_h == exp_h
                # Test window fraction with standard 16:9 source
                wf = framing.window_fraction(1920, 1080, out_w, out_h)
                assert 0.0 < wf <= 1.0
