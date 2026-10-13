"""Comprehensive unit and integration test suite for autoclip_adapter.py.

Verifies:
1. CLI argument parsing (--payload, --payload-json, --stdin, --output).
2. Word-to-cue grouping logic (Latin, CJK, punctuation, pauses, speaker transitions, 8-word cap).
3. AutoClip Scene and Draft model construction and constraints.
4. Offline invariant: draft.language == 'source' (guaranteeing 0 billable API calls).
5. Error handling: fail-closed with structured JSON and non-zero exit on corrupt input/missing files.
6. Execution pipeline: auto_frame -> crop assignment -> render_draft -> target file move.
7. Subprocess execution for CLI compatibility with Tauri Rust sidecar.
"""
from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure scripts dir is on sys.path
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import autoclip_adapter
from autoclip_adapter import (
    group_words_into_cues,
    generate_srt_string,
    join_words_text,
    validate_payload,
    build_draft_and_scene,
    run_adapter,
    main,
)
from autoclip_studio import TITLE_STYLES, CropPoint, Draft, Scene


# ==============================================================================
# 1. Word Grouping & Subtitle Clause Cues Tests
# ==============================================================================

class TestWordGrouping:
    def test_empty_words_returns_empty_cues(self):
        assert group_words_into_cues([]) == []

    def test_latin_words_joined_with_spaces(self):
        words = [
            {"text": "This", "start": 1.0, "end": 1.2, "speaker": "A"},
            {"text": "is", "start": 1.2, "end": 1.4, "speaker": "A"},
            {"text": "great", "start": 1.4, "end": 1.8, "speaker": "A"},
        ]
        cues = group_words_into_cues(words)
        assert len(cues) == 1
        assert cues[0]["text"] == "This is great"
        assert cues[0]["start"] == 1.0
        assert cues[0]["end"] == 1.8
        assert cues[0]["speaker"] == "A"

    def test_cjk_words_joined_without_spaces(self):
        words = [
            {"text": "今天", "start": 0.5, "end": 0.9, "speaker": "S1"},
            {"text": "天气", "start": 0.9, "end": 1.3, "speaker": "S1"},
            {"text": "真好", "start": 1.3, "end": 1.7, "speaker": "S1"},
        ]
        cues = group_words_into_cues(words)
        assert len(cues) == 1
        assert cues[0]["text"] == "今天天气真好"
        assert cues[0]["start"] == 0.5
        assert cues[0]["end"] == 1.7

    def test_latin_punctuation_splits(self):
        words = [
            {"text": "Hello,", "start": 0.0, "end": 0.3},
            {"text": "world!", "start": 0.35, "end": 0.6},
            {"text": "Are", "start": 0.65, "end": 0.8},
            {"text": "you", "start": 0.8, "end": 1.0},
            {"text": "ready?", "start": 1.0, "end": 1.3},
            {"text": "Yes.", "start": 1.35, "end": 1.6},
        ]
        cues = group_words_into_cues(words)
        assert len(cues) == 4
        assert cues[0]["text"] == "Hello,"
        assert cues[1]["text"] == "world!"
        assert cues[2]["text"] == "Are you ready?"
        assert cues[3]["text"] == "Yes."

    def test_cjk_punctuation_splits(self):
        words = [
            {"text": "首先，", "start": 0.0, "end": 0.4},
            {"text": "第一步完成。", "start": 0.45, "end": 1.0},
            {"text": "真的很棒！", "start": 1.05, "end": 1.5},
            {"text": "你觉得呢？", "start": 1.55, "end": 2.0},
        ]
        cues = group_words_into_cues(words)
        assert len(cues) == 4
        assert cues[0]["text"] == "首先，"
        assert cues[1]["text"] == "第一步完成。"
        assert cues[2]["text"] == "真的很棒！"
        assert cues[3]["text"] == "你觉得呢？"

    def test_speaker_transition_splits(self):
        words = [
            {"text": "Hello", "start": 1.0, "end": 1.2, "speaker": "Alice"},
            {"text": "there", "start": 1.2, "end": 1.4, "speaker": "Alice"},
            {"text": "Hi", "start": 1.45, "end": 1.6, "speaker": "Bob"},
            {"text": "Alice", "start": 1.6, "end": 1.8, "speaker": "Bob"},
        ]
        cues = group_words_into_cues(words)
        assert len(cues) == 2
        assert cues[0]["text"] == "Hello there"
        assert cues[0]["speaker"] == "Alice"
        assert cues[1]["text"] == "Hi Alice"
        assert cues[1]["speaker"] == "Bob"

    def test_silence_gap_splits(self):
        # Gap >= 0.4s triggers a split; gap < 0.4s does not
        words = [
            {"text": "First", "start": 1.0, "end": 1.3, "speaker": "A"},
            {"text": "part", "start": 1.4, "end": 1.7, "speaker": "A"},     # gap = 0.1s (< 0.4s) -> continue
            {"text": "second", "start": 2.15, "end": 2.4, "speaker": "A"},  # gap = 0.45s (>= 0.4s) -> split
            {"text": "part", "start": 2.45, "end": 2.7, "speaker": "A"},
        ]
        cues = group_words_into_cues(words)
        assert len(cues) == 2
        assert cues[0]["text"] == "First part"
        assert cues[1]["text"] == "second part"

    def test_maximum_eight_words_split(self):
        # 10 words without punctuation or pause (gap=0.05s < 0.4s) -> split at 8 words
        words = [
            {"text": f"word{i}", "start": float(i) * 0.2, "end": float(i) * 0.2 + 0.15, "speaker": "A"}
            for i in range(10)
        ]
        cues = group_words_into_cues(words)
        assert len(cues) == 2
        assert len(cues[0]["text"].split()) == 8
        assert len(cues[1]["text"].split()) == 2

    def test_generate_srt_string(self):
        cues = [
            {"start": 1.5, "end": 3.75, "text": "First subtitle cue"},
            {"start": 4.0, "end": 6.2, "text": "Second subtitle cue"},
        ]
        srt = generate_srt_string(cues)
        assert "1\n00:00:01,500 --> 00:00:03,750\nFirst subtitle cue\n" in srt
        assert "2\n00:00:04,000 --> 00:00:06,200\nSecond subtitle cue\n" in srt


# ==============================================================================
# 2. Model Construction & Offline Invariant Tests
# ==============================================================================

class TestModelConstruction:
    @pytest.fixture
    def base_payload(self):
        return {
            "project_id": "proj_test_123",
            "video_path": "dummy.mp4",
            "output_path": "output.mp4",
            "candidate": {
                "id": "cand_alpha_01",
                "start_sec": 10.0,
                "end_sec": 25.5,
                "hook": "Unlocking the Future of AI",
                "score": 0.95,
            },
            "transcript_words": [],
            "presentation_config": {
                "platform_strategy": "douyin",
                "subtitle_style": "accent",
                "title_style": "comic",
                "title_motion": True,
                "title_scale": 1.1,
                "title_y": 0.15,
                "template": "interview_zh",
            },
        }

    def test_offline_invariant_language_is_source(self, base_payload):
        draft, scene, strat = build_draft_and_scene(base_payload, [])
        assert draft.language == "source", "CRITICAL: draft.language must be 'source' to prevent external LLM calls"

    def test_scene_model_attributes(self, base_payload):
        draft, scene, strat = build_draft_and_scene(base_payload, [])
        assert scene.id == "scene_cand_alpha_01"
        assert scene.start == 10.0
        assert scene.end == 25.5
        assert scene.label == "片段"
        assert scene.crop_x == 0.5
        assert len(draft.scenes) == 1
        assert draft.scenes[0] is scene

    def test_draft_attributes_and_strategy_resolution(self, base_payload):
        draft, scene, strat = build_draft_and_scene(base_payload, [])
        assert draft.id == "cand_alpha_01"
        assert draft.title == "Unlocking the Future of AI"
        assert draft.hook == "Unlocking the Future of AI"
        assert draft.aspect == "portrait"
        assert draft.layout == "blur"
        assert draft.title_style == "comic"
        assert draft.title_template_version == 6
        assert draft.title_motion is True
        assert draft.title_scale == 1.1
        assert draft.title_y == 0.15
        assert draft.subtitle_style == "accent"
        assert draft.subtitles is True
        assert draft.original_audio is True

    def test_title_template_version_selection(self, base_payload):
        # Styles in TITLE_STYLES (comic, neon, arena, editorial, pixel, frosted) -> 6
        for style in TITLE_STYLES:
            base_payload["presentation_config"]["title_style"] = style
            draft, _, _ = build_draft_and_scene(base_payload, [])
            assert draft.title_template_version == 6

        # Plain / impact / card -> 1
        for style in ["plain", "impact", "card"]:
            base_payload["presentation_config"]["title_style"] = style
            draft, _, _ = build_draft_and_scene(base_payload, [])
            assert draft.title_template_version == 1

    def test_packaging_for_classic_templates(self, base_payload):
        cues = [
            {"start": 12.0, "end": 14.0, "text": "Inside scene cue"},
            {"start": 30.0, "end": 32.0, "text": "Outside scene cue"},
        ]
        base_payload["presentation_config"]["template"] = "interview_zh"
        draft, _, _ = build_draft_and_scene(base_payload, cues)
        assert draft.packaging is not None
        assert draft.packaging.template == "interview_zh"
        assert draft.packaging.fallback is True
        # Only cues within scene [10.0, 25.5] should be included
        assert len(draft.packaging.cues) == 1
        assert draft.packaging.cues[0].text == "Inside scene cue"

    def test_packaging_omitted_for_non_classic_template(self, base_payload):
        base_payload["presentation_config"]["template"] = "none"
        draft, _, _ = build_draft_and_scene(base_payload, [])
        assert draft.packaging is None

    def test_title_scale_and_y_clamping(self, base_payload):
        # Exceeds max
        base_payload["presentation_config"]["title_scale"] = 2.5
        base_payload["presentation_config"]["title_y"] = 0.95
        draft, _, _ = build_draft_and_scene(base_payload, [])
        assert draft.title_scale == 1.2
        assert draft.title_y == 0.70

        # Below min
        base_payload["presentation_config"]["title_scale"] = 0.1
        base_payload["presentation_config"]["title_y"] = 0.01
        draft, _, _ = build_draft_and_scene(base_payload, [])
        assert draft.title_scale == 0.75
        assert draft.title_y == 0.06

    def test_candidate_interval_too_short_fails(self, base_payload):
        base_payload["candidate"]["start_sec"] = 5.0
        base_payload["candidate"]["end_sec"] = 5.05
        with pytest.raises(ValueError, match="interval too short"):
            build_draft_and_scene(base_payload, [])


# ==============================================================================
# 3. Payload Validation & Error Handling Tests
# ==============================================================================

class TestPayloadValidation:
    def test_non_dict_payload_raises(self):
        with pytest.raises(ValueError, match="Payload must be a JSON object"):
            validate_payload(["not", "a", "dict"])

    def test_missing_project_id_raises(self):
        with pytest.raises(ValueError, match="Missing or invalid 'project_id'"):
            validate_payload({"video_path": "v.mp4", "output_path": "o.mp4", "candidate": {}})

    def test_missing_video_path_raises(self):
        with pytest.raises(ValueError, match="Missing or invalid 'video_path'"):
            validate_payload({"project_id": "p1", "output_path": "o.mp4", "candidate": {}})

    def test_video_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError, match="Source video file not found"):
            validate_payload({
                "project_id": "p1",
                "video_path": "non_existent_video_12345.mp4",
                "output_path": "o.mp4",
                "candidate": {"id": "c1", "start_sec": 1.0, "end_sec": 2.0},
            })

    def test_missing_candidate_or_fields_raises(self, tmp_path):
        dummy_video = tmp_path / "vid.mp4"
        dummy_video.write_bytes(b"dummy")

        with pytest.raises(ValueError, match="Missing or invalid 'candidate'"):
            validate_payload({"project_id": "p1", "video_path": str(dummy_video), "output_path": "o.mp4"})

        with pytest.raises(ValueError, match="missing required field 'id'"):
            validate_payload({
                "project_id": "p1",
                "video_path": str(dummy_video),
                "output_path": "o.mp4",
                "candidate": {"start_sec": 1.0, "end_sec": 2.0},
            })

    def test_invalid_time_boundaries_raise(self, tmp_path):
        dummy_video = tmp_path / "vid.mp4"
        dummy_video.write_bytes(b"dummy")

        # negative start
        with pytest.raises(ValueError, match="cannot be negative"):
            validate_payload({
                "project_id": "p1",
                "video_path": str(dummy_video),
                "output_path": "o.mp4",
                "candidate": {"id": "c1", "start_sec": -1.0, "end_sec": 2.0},
            })

        # end <= start
        with pytest.raises(ValueError, match="must be greater than start_sec"):
            validate_payload({
                "project_id": "p1",
                "video_path": str(dummy_video),
                "output_path": "o.mp4",
                "candidate": {"id": "c1", "start_sec": 5.0, "end_sec": 5.0},
            })


# ==============================================================================
# 4. CLI Argument Parsing & Fail-Closed Tests
# ==============================================================================

class TestCLIInterface:
    def test_cli_no_args_fails_closed(self, capsys):
        code = main([])
        assert code == 1
        out = json.loads(capsys.readouterr().out)
        assert out["status"] == "error"
        assert out["output_path"] is None
        assert "No input payload provided" in out["error"]

    def test_cli_unrecognized_flag_fails_closed(self, capsys):
        code = main(["--bogus-option"])
        assert code == 1
        out = json.loads(capsys.readouterr().out)
        assert out["status"] == "error"
        assert "CLI argument error" in out["error"]

    def test_cli_missing_payload_file_fails_closed(self, capsys):
        code = main(["--payload", "non_existent_payload_file.json"])
        assert code == 1
        out = json.loads(capsys.readouterr().out)
        assert out["status"] == "error"
        assert "Payload file not found" in out["error"]

    def test_cli_corrupt_json_fails_closed(self, tmp_path, capsys):
        bad_json_file = tmp_path / "bad.json"
        bad_json_file.write_text("{ not valid json : 123 }", encoding="utf-8")
        code = main(["--payload", str(bad_json_file)])
        assert code == 1
        out = json.loads(capsys.readouterr().out)
        assert out["status"] == "error"
        assert "JSONDecodeError" in out["traceback"]

    def test_cli_output_file_written_on_error(self, tmp_path):
        err_out_file = tmp_path / "result_error.json"
        code = main(["--payload-json", "{ invalid }", "--output", str(err_out_file)])
        assert code == 1
        assert err_out_file.is_file()
        data = json.loads(err_out_file.read_text(encoding="utf-8"))
        assert data["status"] == "error"


# ==============================================================================
# 5. Execution Pipeline (auto_frame -> render_draft -> file move) Tests
# ==============================================================================

class TestExecutionPipeline:
    @patch("autoclip_adapter.render_draft")
    @patch("autoclip_adapter.auto_frame")
    @patch("autoclip_adapter._probe")
    def test_run_adapter_end_to_end_mocked(self, mock_probe, mock_auto_frame, mock_render_draft, tmp_path, monkeypatch):
        # Setup temporary directories
        test_data_dir = tmp_path / "data"
        monkeypatch.setenv("AUTOCLIP_DATA_DIR", str(test_data_dir))

        dummy_video = tmp_path / "source.mp4"
        dummy_video.write_bytes(b"dummy_video_bytes")
        target_output = tmp_path / "final_output" / "clip.mp4"

        project_id = "test_pipe_proj"
        cand_id = "cand_001"

        payload = {
            "project_id": project_id,
            "video_path": str(dummy_video),
            "output_path": str(target_output),
            "candidate": {
                "id": cand_id,
                "start_sec": 2.0,
                "end_sec": 6.5,
                "hook": "Pipeline Test Hook",
                "score": 0.99,
            },
            "transcript_words": [
                {"text": "Hello", "start": 2.1, "end": 2.5, "speaker": "S1"},
                {"text": "World!", "start": 2.6, "end": 3.0, "speaker": "S1"},
            ],
            "presentation_config": {
                "platform_strategy": "douyin",
                "subtitle_style": "box",
                "title_style": "neon",
                "template": "interview_zh",
            },
        }

        # Mock probe return
        mock_probe.return_value = {"width": 1920, "height": 1080, "duration": 10.0}

        # Mock auto_frame return with crop track
        mock_auto_frame.return_value = {
            "window_fraction": 0.316,
            "scenes": [
                {
                    "id": f"scene_{cand_id}",
                    "crop_x": 0.52,
                    "crop_track": [
                        {"start": 0.0, "crop_x": 0.52, "mode": "crop"},
                        {"start": 2.0, "crop_x": 0.50, "mode": "fit"},
                    ],
                }
            ],
        }

        # Mock render_draft to simulate creating the rendered video
        def fake_render(project_id, video, draft, job_id, progress, **kwargs):
            studio_out_dir = test_data_dir / "projects" / project_id / "output" / "studio"
            studio_out_dir.mkdir(parents=True, exist_ok=True)
            out_file = studio_out_dir / f"{job_id}.mp4"
            out_file.write_bytes(b"fake_rendered_mp4_bytes")
            progress(100)
            return {
                "title": draft.title,
                "duration": 4.5,
                "width": 1080,
                "height": 1920,
                "warnings": [],
            }

        mock_render_draft.side_effect = fake_render

        # Execute adapter
        result = run_adapter(payload)

        # Verifications
        assert result["status"] == "success"
        assert result["output_path"] == str(target_output.resolve())
        assert result["duration"] == 4.5
        assert result["width"] == 1080
        assert result["height"] == 1920
        assert result["strategy_id"] == "douyin"
        assert result["framing_info"]["window_fraction"] == 0.316
        assert target_output.is_file()
        assert target_output.read_bytes() == b"fake_rendered_mp4_bytes"

        # Verify SRT was written to metadata/input.srt
        srt_file = test_data_dir / "projects" / project_id / "metadata" / "input.srt"
        assert srt_file.is_file()
        srt_content = srt_file.read_text(encoding="utf-8")
        assert "Hello World!" in srt_content

        # Verify auto_frame called with probed dimensions
        mock_auto_frame.assert_called_once()
        call_args = mock_auto_frame.call_args[0]
        assert call_args[1].scenes[0].crop_x == 0.52
        assert len(call_args[1].scenes[0].crop_track) == 2
        assert call_args[1].scenes[0].crop_track[1].mode == "fit"


# ==============================================================================
# 6. Real Media / CLI Integration via Subprocess
# ==============================================================================

class TestCLIIntegration:
    @patch("autoclip_adapter.render_draft")
    @patch("autoclip_adapter.auto_frame")
    @patch("autoclip_adapter._probe")
    def test_cli_main_success_via_payload_argument(self, mock_probe, mock_auto_frame, mock_render_draft, tmp_path, monkeypatch, capsys):
        test_data_dir = tmp_path / "data"
        monkeypatch.setenv("AUTOCLIP_DATA_DIR", str(test_data_dir))

        dummy_video = tmp_path / "vid.mp4"
        dummy_video.write_bytes(b"dummy")
        target_output = tmp_path / "out.mp4"
        payload_file = tmp_path / "payload.json"
        result_file = tmp_path / "res.json"

        mock_probe.return_value = {"width": 1920, "height": 1080, "duration": 5.0}
        mock_auto_frame.return_value = {"window_fraction": 0.316, "scenes": []}

        def fake_render(project_id, video, draft, job_id, progress, **kwargs):
            out_file = test_data_dir / "projects" / project_id / "output" / "studio" / f"{job_id}.mp4"
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_bytes(b"mp4_content")
            return {"title": draft.title, "duration": 3.0, "width": 1080, "height": 1920, "warnings": []}

        mock_render_draft.side_effect = fake_render

        payload = {
            "project_id": "cli_proj",
            "video_path": str(dummy_video),
            "output_path": str(target_output),
            "candidate": {"id": "c1", "start_sec": 1.0, "end_sec": 4.0, "hook": "CLI Test"},
            "transcript_words": [],
            "presentation_config": {"platform_strategy": "tiktok"},
        }
        payload_file.write_text(json.dumps(payload), encoding="utf-8")

        code = main(["--payload", str(payload_file), "--output", str(result_file)])
        assert code == 0

        stdout_data = json.loads(capsys.readouterr().out)
        assert stdout_data["status"] == "success"
        assert stdout_data["strategy_id"] == "tiktok"

        file_data = json.loads(result_file.read_text(encoding="utf-8"))
        assert file_data["status"] == "success"
        assert target_output.is_file()

    @patch("autoclip_adapter.render_draft")
    @patch("autoclip_adapter.auto_frame")
    @patch("autoclip_adapter._probe")
    def test_cli_main_success_via_stdin(self, mock_probe, mock_auto_frame, mock_render_draft, tmp_path, monkeypatch, capsys):
        test_data_dir = tmp_path / "data"
        monkeypatch.setenv("AUTOCLIP_DATA_DIR", str(test_data_dir))

        dummy_video = tmp_path / "vid.mp4"
        dummy_video.write_bytes(b"dummy")
        target_output = tmp_path / "out_stdin.mp4"

        mock_probe.return_value = {"width": 1920, "height": 1080, "duration": 5.0}
        mock_auto_frame.return_value = {"window_fraction": 0.316, "scenes": []}

        def fake_render(project_id, video, draft, job_id, progress, **kwargs):
            out_file = test_data_dir / "projects" / project_id / "output" / "studio" / f"{job_id}.mp4"
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_bytes(b"mp4_content")
            return {"title": draft.title, "duration": 2.0, "width": 1080, "height": 1920, "warnings": []}

        mock_render_draft.side_effect = fake_render

        payload = {
            "project_id": "stdin_proj",
            "video_path": str(dummy_video),
            "output_path": str(target_output),
            "candidate": {"id": "c_stdin", "start_sec": 0.5, "end_sec": 2.5, "hook": "Stdin Test"},
            "transcript_words": [],
            "presentation_config": {"platform_strategy": "youtube_shorts"},
        }

        monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))
        code = main(["--stdin"])
        assert code == 0

        stdout_data = json.loads(capsys.readouterr().out)
        assert stdout_data["status"] == "success"
        assert stdout_data["strategy_id"] == "youtube_shorts"
        assert target_output.is_file()


# ==============================================================================
# 7. Real Synthetic Media & Subprocess Invocation Tests
# ==============================================================================

class TestSyntheticMediaAndSubprocess:
    @pytest.fixture
    def synthetic_video(self, tmp_path):
        from autoclip_studio.ffmpeg_utils import get_ffmpeg_path
        vid_path = tmp_path / "synthetic_1s.mp4"
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
        assert res.returncode == 0, f"Failed to generate synthetic video: {res.stderr.decode()}"
        assert vid_path.is_file()
        return vid_path

    def test_end_to_end_synthetic_media(self, synthetic_video, tmp_path, monkeypatch):
        test_data_dir = tmp_path / "data"
        monkeypatch.setenv("AUTOCLIP_DATA_DIR", str(test_data_dir))

        target_out = tmp_path / "out_synth.mp4"
        payload = {
            "project_id": "proj_synth_e2e",
            "video_path": str(synthetic_video),
            "output_path": str(target_out),
            "candidate": {
                "id": "cand_e2e",
                "start_sec": 0.1,
                "end_sec": 0.8,
                "hook": "Synthetic Test Video",
                "score": 0.88,
            },
            "transcript_words": [
                {"text": "Hello", "start": 0.15, "end": 0.4, "speaker": "S1"},
                {"text": "World!", "start": 0.45, "end": 0.75, "speaker": "S1"},
            ],
            "presentation_config": {
                "platform_strategy": "bilibili",
                "subtitle_style": "clean",
                "title_style": "plain",
                "template": "none",
            },
        }

        result = run_adapter(payload)
        assert result["status"] == "success"
        assert target_out.is_file()
        assert target_out.stat().st_size > 0
        assert result["duration"] is not None
        assert result["width"] == 1920
        assert result["height"] == 1080
        assert result["strategy_id"] == "bilibili"
        assert "scenes" in result["framing_info"]

    def test_subprocess_cli_invocation_payload_file(self, synthetic_video, tmp_path):
        adapter_script = SCRIPTS_DIR / "autoclip_adapter.py"
        test_data_dir = tmp_path / "data_subp"
        target_out = tmp_path / "out_subp.mp4"
        payload_file = tmp_path / "payload_subp.json"

        payload = {
            "project_id": "proj_subp",
            "video_path": str(synthetic_video),
            "output_path": str(target_out),
            "candidate": {
                "id": "cand_subp",
                "start_sec": 0.1,
                "end_sec": 0.7,
                "hook": "Subprocess CLI Test",
            },
            "transcript_words": [
                {"text": "Subprocess", "start": 0.15, "end": 0.35},
                {"text": "Test", "start": 0.4, "end": 0.65},
            ],
            "presentation_config": {
                "platform_strategy": "bilibili",
            },
        }
        payload_file.write_text(json.dumps(payload), encoding="utf-8")

        env = os.environ.copy()
        env["AUTOCLIP_DATA_DIR"] = str(test_data_dir)

        proc = subprocess.run(
            [sys.executable, str(adapter_script), "--payload", str(payload_file)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env,
            timeout=60,
        )
        assert proc.returncode == 0, f"Subprocess failed with stderr: {proc.stderr}"
        stdout_json = json.loads(proc.stdout)
        assert stdout_json["status"] == "success"
        assert target_out.is_file()

    def test_subprocess_cli_invocation_stdin(self, synthetic_video, tmp_path):
        adapter_script = SCRIPTS_DIR / "autoclip_adapter.py"
        test_data_dir = tmp_path / "data_stdin"
        target_out = tmp_path / "out_stdin.mp4"

        payload = {
            "project_id": "proj_subp_stdin",
            "video_path": str(synthetic_video),
            "output_path": str(target_out),
            "candidate": {
                "id": "cand_subp_stdin",
                "start_sec": 0.1,
                "end_sec": 0.7,
                "hook": "Stdin Subprocess Test",
            },
            "transcript_words": [],
            "presentation_config": {
                "platform_strategy": "bilibili",
            },
        }
        payload_str = json.dumps(payload)

        env = os.environ.copy()
        env["AUTOCLIP_DATA_DIR"] = str(test_data_dir)

        proc = subprocess.run(
            [sys.executable, str(adapter_script), "--stdin"],
            input=payload_str,
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env,
            timeout=60,
        )
        assert proc.returncode == 0, f"Subprocess failed with stderr: {proc.stderr}"
        stdout_json = json.loads(proc.stdout)
        assert stdout_json["status"] == "success"
        assert target_out.is_file()

    def test_subprocess_cli_error_fails_closed(self, tmp_path):
        adapter_script = SCRIPTS_DIR / "autoclip_adapter.py"
        proc = subprocess.run(
            [sys.executable, str(adapter_script), "--payload-json", "{ invalid json }"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
        )
        assert proc.returncode == 1
        stdout_json = json.loads(proc.stdout)
        assert stdout_json["status"] == "error"
        assert stdout_json["output_path"] is None
        assert "error" in stdout_json
        assert "traceback" in stdout_json

    def test_default_data_directory_fallback_outside_src_tauri(self, monkeypatch, tmp_path):
        """Verify that when AUTOCLIP_DATA_DIR is unset, default path is outside src-tauri."""
        monkeypatch.delenv("AUTOCLIP_DATA_DIR", raising=False)
        monkeypatch.delenv("AUTOSHORTS_DATA_DIR", raising=False)
        import autoclip_adapter
        proj_dir = autoclip_adapter.setup_data_directory("probe_project_id")
        resolved_proj = proj_dir.resolve()
        src_tauri_dir = (SCRIPTS_DIR.parent).resolve()
        assert not str(resolved_proj).startswith(str(src_tauri_dir)), (
            f"AutoClip data dir {resolved_proj} must not be inside src-tauri ({src_tauri_dir})"
        )


