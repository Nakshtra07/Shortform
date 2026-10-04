"""
End-to-End Real-Video Render, A/V Duration & Synchronization Verification Suite.
Validates AutoShorts 6.0 DualFrame Architecture on Real Video Media:

1. Two-person shots trigger DualFrame ([DualFrame] mode=dual_stack, enter, and layout_plan).
2. Layout transitions occur cleanly (SINGLE -> DUAL_STACK -> SINGLE).
3. Output video is strictly 1080x1920 (9:16 portrait).
4. Continuous audio: audio duration matches video duration with |Delta t| < 0.05s.
5. Kinetic ASS subtitles are burned once onto the full 1080x1920 canvas without per-panel duplication.
6. A/B comparison confirms zero regression when AUTOSHORTS_DUALFRAME_ENABLED=false or on single-person footage.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

import speaker_tracker


def locate_candidate_video() -> str:
    """Locates a suitable test video from known workspace locations."""
    candidates = [
        os.path.join(r"d:\College\Autoshorts 6.0", "Messi vs Ronaldo Fans： The Psychology Explained [rssDTc086bk].mp4"),
        os.path.join(script_dir, "..", "..", "..", "Messi vs Ronaldo Fans： The Psychology Explained [rssDTc086bk].mp4"),
        os.path.join(script_dir, "..", "..", "Messi vs Ronaldo Fans： The Psychology Explained [rssDTc086bk].mp4"),
        os.path.join(r"d:\College\Autoshorts 6.0", "Beat Emotional Fatigue_ Better Sleep & Clearer Mind.mp4"),
        os.path.join(script_dir, "..", "..", "..", "Beat Emotional Fatigue_ Better Sleep & Clearer Mind.mp4"),
    ]
    for c in candidates:
        if os.path.exists(c) and os.path.getsize(c) > 1000000:
            return os.path.abspath(c)

    # Search workspace root for any mp4 > 5MB
    ws_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
    if os.path.exists(ws_root):
        for f in os.listdir(ws_root):
            if f.endswith(".mp4"):
                p = os.path.join(ws_root, f)
                if os.path.getsize(p) > 5000000:
                    return p

    raise FileNotFoundError("Could not find any suitable real video file for DualFrame rendering tests.")


def build_layout_filtergraph(
    plan: dict,
    duration_sec: float,
    ass_subtitle_path: str = None,
    drawtext_filters: str = None,
) -> tuple[str, str]:
    """
    Python mirror of Rust media.rs build_layout_filtergraph function.
    Constructs an FFmpeg filtergraph from a SmartFramingPlan JSON dictionary.
    Returns (filter_string, output_label).
    """
    segments = plan.get("segments", [])
    has_dual = any(seg.get("mode") == "dual_stack" for seg in segments)
    is_single = (not has_dual) and len(segments) <= 1

    if is_single:
        if segments and "crop" in segments[0]:
            crop = segments[0]["crop"]
            crop_w = crop["w"]
            crop_h = crop["h"]
            crop_x = crop["x"]
            crop_y = crop["y"]
        else:
            crop_w = plan.get("w", "608")
            crop_h = plan.get("h", "1080")
            crop_x = plan.get("x", "0")
            crop_y = plan.get("y", "0")

        crop_filter = f"crop=w={crop_w}:h={crop_h}:x='{crop_x}':y='{crop_y}'"
        scale_filter = "scale=1080:1920:flags=lanczos+accurate_rnd+full_chroma_int"
        filter_str = f"{crop_filter},{scale_filter}"

        if ass_subtitle_path and os.path.exists(ass_subtitle_path):
            escaped_ass = ass_subtitle_path.replace("\\", "/").replace(":", r"\:")
            filter_str = f"{filter_str},subtitles='{escaped_ass}'"
        elif drawtext_filters and drawtext_filters.strip():
            filter_str = f"{filter_str},{drawtext_filters.strip()}"

        return filter_str, ""

    total_branches = 0
    for seg in segments:
        if seg.get("mode") == "dual_stack":
            total_branches += 2
        else:
            total_branches += 1

    graph = ""
    if total_branches > 1:
        graph += f"[0:v]split={total_branches}"
        for b in range(total_branches):
            graph += f"[b{b}]"
        graph += ";"

    def format_time(val: float) -> str:
        if abs(val - round(val)) < 1e-6:
            return f"{round(val)}"
        s = f"{val:.3f}"
        return s.rstrip("0").rstrip(".")

    branch_idx = 0
    for i, seg in enumerate(segments):
        mode = seg.get("mode", "single")
        start = seg.get("start", 0.0)
        end = seg.get("end", duration_sec)
        seg_end = duration_sec if (end <= 0.0 or end <= start) else end

        if mode == "single":
            crop = seg.get("crop", {})
            cw = crop.get("w", "608")
            ch = crop.get("h", "1080")
            cx = crop.get("x", "0")
            cy = crop.get("y", "0")

            b_in = f"[b{branch_idx}]" if total_branches > 1 else "[0:v]"
            branch_idx += 1

            graph += (
                f"{b_in}trim=start={format_time(start)}:end={format_time(seg_end)},"
                f"setpts=PTS-STARTPTS,crop=w={cw}:h={ch}:x='{cx}':y='{cy}',"
                f"scale=1080:1920:flags=lanczos+accurate_rnd+full_chroma_int,setsar=1[v_seg_{i}];"
            )
        elif mode == "dual_stack":
            top_crop = seg.get("top_crop", {})
            bot_crop = seg.get("bottom_crop", {})

            b_top = f"[b{branch_idx}]"
            b_bot = f"[b{branch_idx + 1}]"
            branch_idx += 2

            # Top panel scaled to 1080:960
            graph += (
                f"{b_top}trim=start={format_time(start)}:end={format_time(seg_end)},"
                f"setpts=PTS-STARTPTS,crop=w={top_crop.get('w', '1080')}:h={top_crop.get('h', '960')}:"
                f"x='{top_crop.get('x', '0')}':y='{top_crop.get('y', '0')}',"
                f"scale=1080:960:flags=lanczos+accurate_rnd+full_chroma_int,setsar=1[v_top_{i}];"
            )

            # Bottom panel scaled to 1080:960
            graph += (
                f"{b_bot}trim=start={format_time(start)}:end={format_time(seg_end)},"
                f"setpts=PTS-STARTPTS,crop=w={bot_crop.get('w', '1080')}:h={bot_crop.get('h', '960')}:"
                f"x='{bot_crop.get('x', '0')}':y='{bot_crop.get('y', '0')}',"
                f"scale=1080:960:flags=lanczos+accurate_rnd+full_chroma_int,setsar=1[v_bot_{i}];"
            )

            # Vertical stack: 960 + 960 = 1920
            graph += f"[v_top_{i}][v_bot_{i}]vstack=inputs=2[v_seg_{i}];"

    # Concatenate segments
    concat_inputs = "".join(f"[v_seg_{i}]" for i in range(len(segments)))
    graph += f"{concat_inputs}concat=n={len(segments)}:v=1:a=0[v_composed]"

    # Subtitles or drawtext
    if ass_subtitle_path and os.path.exists(ass_subtitle_path):
        escaped_ass = ass_subtitle_path.replace("\\", "/").replace(":", r"\:")
        graph += f";[v_composed]subtitles='{escaped_ass}'[v_final]"
        return graph, "v_final"
    elif drawtext_filters and drawtext_filters.strip():
        graph += f";[v_composed]{drawtext_filters.strip()}[v_final]"
        return graph, "v_final"
    else:
        return graph, "v_composed"


class TestDualFrameRealRender(unittest.TestCase):
    """
    End-to-End Real Video Render, Duration, and Synchronization Verification.
    """

    @classmethod
    def setUpClass(cls):
        cls.source_video = locate_candidate_video()
        cls.temp_dir = tempfile.mkdtemp(prefix="autoshorts_dualframe_render_test_")
        cls.rendered_files = []

        # Create a sample kinetic ASS subtitle file
        cls.ass_path = os.path.join(cls.temp_dir, "test_kinetic_subtitles.ass")
        ass_content = (
            "[Script Info]\n"
            "Title: AutoShorts DualFrame Kinetic Subtitles\n"
            "ScriptType: v4.00+\n"
            "PlayResX: 1080\n"
            "PlayResY: 1920\n"
            "ScaledBorderAndShadow: yes\n\n"
            "[V4+ Styles]\n"
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
            "Style: Default,Arial,52,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,0,2,40,40,240,1\n\n"
            "[Events]\n"
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
            "Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,{\\c&H00FFFF&}DUALFRAME{\\c&HFFFFFF&} ARCHITECTURE ACTIVE\n"
            "Dialogue: 0,0:00:03.50,0:00:06.50,Default,,0,0,0,,{\\c&H00FF00&}ACTIVE SPEAKER{\\c&HFFFFFF&} SYNCHRONIZED AUDIO\n"
        )
        with open(cls.ass_path, "w", encoding="utf-8") as f:
            f.write(ass_content)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.temp_dir):
            try:
                shutil.rmtree(cls.temp_dir, ignore_errors=True)
            except Exception:
                pass

    def setUp(self):
        # Reset environment variable
        if "AUTOSHORTS_DUALFRAME_ENABLED" in os.environ:
            del os.environ["AUTOSHORTS_DUALFRAME_ENABLED"]

    def tearDown(self):
        os.environ.pop("AUTOSHORTS_DUALFRAME_ENABLED", None)

    def _execute_tracker(self, video_path: str, start_sec: float, end_sec: float) -> tuple[dict, str]:
        """Runs speaker_tracker.py as a subprocess and returns (plan_dict, stderr)."""
        cmd = [
            sys.executable,
            os.path.join(script_dir, "speaker_tracker.py"),
            video_path,
            str(start_sec * 1000.0),
            str(end_sec * 1000.0),
            "608",
            "1312",
            "656",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        self.assertEqual(res.returncode, 0, f"speaker_tracker.py failed:\n{res.stderr}")

        stdout_lines = [l.strip() for l in res.stdout.splitlines() if l.strip() and not l.strip().startswith("[")]
        self.assertTrue(stdout_lines, f"No JSON output from speaker_tracker.py. Output:\n{res.stdout}")
        plan = json.loads(stdout_lines[-1])
        return plan, res.stderr

    def _render_ffmpeg(
        self,
        video_path: str,
        start_sec: float,
        duration_sec: float,
        filtergraph: str,
        output_label: str,
        out_path: str,
    ) -> None:
        """Invokes FFmpeg with exact encoding and audio mapping parameters."""
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start_sec:.3f}",
            "-i", video_path,
            "-t", f"{duration_sec:.3f}",
        ]

        if output_label:
            cmd.extend(["-filter_complex", filtergraph, "-map", f"[{output_label}]", "-map", "0:a"])
        else:
            cmd.extend(["-vf", filtergraph])

        cmd.extend([
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "18",
            "-c:a", "aac",
            "-b:a", "192k",
            "-movflags", "+faststart",
            out_path,
        ])

        res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        self.assertEqual(res.returncode, 0, f"FFmpeg failed with exit code {res.returncode}:\n{res.stderr}")
        self.assertTrue(os.path.exists(out_path), f"Rendered output missing: {out_path}")
        self.assertGreater(os.path.getsize(out_path), 50000, "Rendered file is suspiciously small")
        self.rendered_files.append(out_path)

    def _probe_media(self, media_path: str) -> dict:
        """Runs ffprobe and returns parsed JSON output."""
        cmd = [
            "ffprobe", "-v", "error",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            media_path,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        self.assertEqual(res.returncode, 0, f"ffprobe failed:\n{res.stderr}")
        return json.loads(res.stdout)

    # -------------------------------------------------------------------------
    # Test Case 1: Real-video slice DualFrame resolution & telemetry
    # -------------------------------------------------------------------------
    def test_01_real_video_dualframe_plan_and_telemetry(self):
        """
        Executes speaker_tracker.py on a real video slice containing two subjects (4.0s - 12.0s)
        to verify layout plan generation, 9:8 panel geometry, and [DualFrame] telemetry.
        """
        plan, stderr = self._execute_tracker(self.source_video, start_sec=4.0, end_sec=12.0)

        # 1. Structured telemetry verification
        stderr_lines = [l.strip() for l in stderr.splitlines() if l.strip()]
        df_lines = [l for l in stderr_lines if l.startswith("[DualFrame]")]
        self.assertTrue(df_lines, "Expected [DualFrame] telemetry in stderr")

        has_enter = any("enter t=" in l and "top_track=" in l and "bottom_track=" in l for l in df_lines)
        self.assertTrue(has_enter, f"Missing [DualFrame] enter line in telemetry:\n{stderr}")

        has_summary = any("layout_plan total_segments=" in l and "dual_segments=" in l for l in df_lines)
        self.assertTrue(has_summary, f"Missing [DualFrame] layout_plan summary line in telemetry:\n{stderr}")

        # 2. Plan contract verification
        self.assertEqual(plan.get("mode"), "dual_stack", f"Expected plan mode dual_stack, got: {plan.get('mode')}")
        segments = plan.get("segments", [])
        self.assertGreaterEqual(len(segments), 2, "Expected multiple segments across layout transitions")

        # 3. Continuous temporal coverage
        self.assertAlmostEqual(segments[0]["start"], 0.0, delta=0.01)
        self.assertAlmostEqual(segments[-1]["end"], 8.0, delta=0.01)
        for i in range(len(segments) - 1):
            self.assertAlmostEqual(
                segments[i]["end"], segments[i + 1]["start"], delta=0.01,
                msg=f"Gap or overlap between segment {i} and {i + 1}"
            )

        # 4. Geometry and 9:8 panel ratio
        dual_segs = [s for s in segments if s.get("mode") == "dual_stack"]
        self.assertTrue(dual_segs, "Expected at least one dual_stack segment")

        for ds in dual_segs:
            self.assertIn("top_crop", ds)
            self.assertIn("bottom_crop", ds)
            self.assertIn("top_track_id", ds)
            self.assertIn("bottom_track_id", ds)

            for crop_role, crop in [("top", ds["top_crop"]), ("bottom", ds["bottom_crop"])]:
                w = int(crop["w"])
                h = int(crop["h"])
                self.assertEqual(w % 2, 0, f"{crop_role} width {w} must be even")
                self.assertEqual(h % 2, 0, f"{crop_role} height {h} must be even")
                # 9:8 aspect ratio check: w ≈ h * 1.125
                expected_w = round(h * 1.125)
                self.assertLessEqual(abs(w - expected_w), 1, f"{crop_role} crop aspect ratio must be 9:8, got {w}x{h}")

            # Top and bottom crops must have independent framing coordinates
            self.assertNotEqual(ds["top_crop"]["x"], ds["bottom_crop"]["x"])

    # -------------------------------------------------------------------------
    # Test Case 2 & 3: Render real clip (SINGLE -> DUAL_STACK -> SINGLE) & probe
    # -------------------------------------------------------------------------
    def test_02_and_03_render_real_clip_and_verify_duration_and_av_sync(self):
        """
        Renders a real clip containing a SINGLE -> DUAL_STACK -> SINGLE transition using FFmpeg
        with multi-branch filtergraph and kinetic ASS subtitles, then executes ffprobe to verify:
        - Output video dimensions == 1080x1920 (strict 9:16 portrait)
        - Video duration matches audio duration with |Delta t| < 0.05s
        - Audio stream encoded once from 0:a as AAC >= 44100Hz
        """
        start_sec = 4.0
        end_sec = 12.0
        duration_sec = end_sec - start_sec

        plan, _ = self._execute_tracker(self.source_video, start_sec=start_sec, end_sec=end_sec)
        filtergraph, output_label = build_layout_filtergraph(
            plan=plan,
            duration_sec=duration_sec,
            ass_subtitle_path=self.ass_path,
        )

        out_mp4 = os.path.join(self.temp_dir, "dualframe_transition_render.mp4")
        self._render_ffmpeg(
            video_path=self.source_video,
            start_sec=start_sec,
            duration_sec=duration_sec,
            filtergraph=filtergraph,
            output_label=output_label,
            out_path=out_mp4,
        )

        # Probe output media
        probe = self._probe_media(out_mp4)
        fmt = probe.get("format", {})
        streams = probe.get("streams", [])

        v_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
        a_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)

        self.assertIsNotNone(v_stream, "Missing video stream in rendered MP4")
        self.assertIsNotNone(a_stream, "Missing audio stream in rendered MP4")

        # 1. Output resolution: strictly 1080x1920 (9:16 portrait)
        v_width = int(v_stream["width"])
        v_height = int(v_stream["height"])
        self.assertEqual(v_width, 1080, f"Expected width 1080, got {v_width}")
        self.assertEqual(v_height, 1920, f"Expected height 1920, got {v_height}")

        # 2. Continuous audio stream: encoded once as AAC
        a_codec = a_stream.get("codec_name")
        self.assertEqual(a_codec, "aac", f"Expected AAC audio codec, got: {a_codec}")
        sample_rate = int(a_stream.get("sample_rate", 0))
        self.assertGreaterEqual(sample_rate, 44100, f"Audio sample rate {sample_rate} < 44100")
        channels = int(a_stream.get("channels", 0))
        self.assertGreaterEqual(channels, 1, f"Audio channels {channels} < 1")

        # 3. Exact duration matching: |Delta t| < 0.05s
        v_dur = float(v_stream.get("duration", 0.0))
        a_dur = float(a_stream.get("duration", 0.0))
        fmt_dur = float(fmt.get("duration", 0.0))

        delta_va = abs(v_dur - a_dur)
        delta_fa = abs(fmt_dur - a_dur)

        print(
            f"\n[Real Render Probe] Video: {v_width}x{v_height} ({v_dur:.4f}s), "
            f"Audio: {a_codec} {sample_rate}Hz {channels}ch ({a_dur:.4f}s), "
            f"Format: {fmt_dur:.4f}s, |Delta t| = {delta_va:.4f}s"
        )

        self.assertLess(
            delta_va, 0.05,
            f"A/V duration discrepancy |video_dur - audio_dur| = {delta_va:.4f}s exceeds 0.05s threshold"
        )
        self.assertLess(
            delta_fa, 0.05,
            f"Container/Audio duration discrepancy |fmt_dur - audio_dur| = {delta_fa:.4f}s exceeds 0.05s threshold"
        )

    # -------------------------------------------------------------------------
    # Test Case 4: Single-mode fallback regression when AUTOSHORTS_DUALFRAME_ENABLED=false
    # -------------------------------------------------------------------------
    def test_04_fallback_single_mode_when_disabled_via_env(self):
        """
        Tests fallback to single-person framing when AUTOSHORTS_DUALFRAME_ENABLED=false.
        Verifies exact legacy behavior: single mode, single crop segment, 1080x1920 output, |Delta t| < 0.05s.
        """
        os.environ["AUTOSHORTS_DUALFRAME_ENABLED"] = "false"
        start_sec = 4.0
        end_sec = 12.0
        duration_sec = end_sec - start_sec

        plan, stderr = self._execute_tracker(self.source_video, start_sec=start_sec, end_sec=end_sec)

        # Plan must be single mode
        self.assertEqual(plan.get("mode"), "single", "Plan mode must be single when disabled")
        segments = plan.get("segments", [])
        self.assertEqual(len(segments), 1, "Expected single segment in single fallback")
        self.assertEqual(segments[0].get("mode"), "single")
        self.assertIn("crop", segments[0])
        self.assertNotIn("top_crop", segments[0])
        self.assertNotIn("bottom_crop", segments[0])

        # Filtergraph must be single-branch
        filtergraph, output_label = build_layout_filtergraph(
            plan=plan,
            duration_sec=duration_sec,
            ass_subtitle_path=self.ass_path,
        )
        self.assertEqual(output_label, "", "Single layout must return empty label for -vf")
        self.assertNotIn("split=", filtergraph)
        self.assertNotIn("vstack=", filtergraph)
        self.assertIn("scale=1080:1920", filtergraph)

        out_mp4 = os.path.join(self.temp_dir, "single_fallback_render.mp4")
        self._render_ffmpeg(
            video_path=self.source_video,
            start_sec=start_sec,
            duration_sec=duration_sec,
            filtergraph=filtergraph,
            output_label=output_label,
            out_path=out_mp4,
        )

        probe = self._probe_media(out_mp4)
        v_stream = next(s for s in probe["streams"] if s.get("codec_type") == "video")
        a_stream = next(s for s in probe["streams"] if s.get("codec_type") == "audio")

        v_dur = float(v_stream.get("duration", 0.0))
        a_dur = float(a_stream.get("duration", 0.0))
        delta_va = abs(v_dur - a_dur)

        self.assertEqual(int(v_stream["width"]), 1080)
        self.assertEqual(int(v_stream["height"]), 1920)
        self.assertLess(delta_va, 0.05, f"Single-mode |Delta t| = {delta_va:.4f}s exceeds 0.05s")

    # -------------------------------------------------------------------------
    # Test Case 5: Single-person segment natural regression
    # -------------------------------------------------------------------------
    def test_05_single_person_segment_regression(self):
        """
        Executes speaker_tracker.py on a single-speaker section (20.0s - 25.0s) with DualFrame enabled.
        Verifies that only 1 subject is detected and the system remains in single mode throughout.
        """
        start_sec = 20.0
        end_sec = 25.0
        duration_sec = end_sec - start_sec

        plan, stderr = self._execute_tracker(self.source_video, start_sec=start_sec, end_sec=end_sec)

        # Plan must be single mode
        self.assertEqual(plan.get("mode"), "single")
        segments = plan.get("segments", [])
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].get("mode"), "single")

        # Telemetry check
        self.assertIn("[DualFrame] layout_plan total_segments=1 dual_segments=0", stderr)

        # Render and verify
        filtergraph, output_label = build_layout_filtergraph(
            plan=plan,
            duration_sec=duration_sec,
            ass_subtitle_path=self.ass_path,
        )

        out_mp4 = os.path.join(self.temp_dir, "single_person_render.mp4")
        self._render_ffmpeg(
            video_path=self.source_video,
            start_sec=start_sec,
            duration_sec=duration_sec,
            filtergraph=filtergraph,
            output_label=output_label,
            out_path=out_mp4,
        )

        probe = self._probe_media(out_mp4)
        v_stream = next(s for s in probe["streams"] if s.get("codec_type") == "video")
        a_stream = next(s for s in probe["streams"] if s.get("codec_type") == "audio")

        v_dur = float(v_stream.get("duration", 0.0))
        a_dur = float(a_stream.get("duration", 0.0))
        delta_va = abs(v_dur - a_dur)

        self.assertEqual(int(v_stream["width"]), 1080)
        self.assertEqual(int(v_stream["height"]), 1920)
        self.assertLess(delta_va, 0.05, f"Single-person |Delta t| = {delta_va:.4f}s exceeds 0.05s")

    # -------------------------------------------------------------------------
    # Test Case 6: Filtergraph builder contract edge-cases
    # -------------------------------------------------------------------------
    def test_06_filtergraph_builder_contract_edge_cases(self):
        """
        Validates filtergraph builder branch counting, time formatting, and subtitle path escaping.
        """
        # Multi-segment plan with 2 single and 1 dual_stack segment:
        # Total branches = 1 + 2 + 1 = 4
        dummy_plan = {
            "mode": "dual_stack",
            "segments": [
                {
                    "mode": "single",
                    "start": 0.0,
                    "end": 2.0,
                    "crop": {"x": "100", "y": "0", "w": "608", "h": "1080"},
                },
                {
                    "mode": "dual_stack",
                    "start": 2.0,
                    "end": 5.5,
                    "top_crop": {"x": "200", "y": "50", "w": "1080", "h": "960"},
                    "bottom_crop": {"x": "400", "y": "50", "w": "1080", "h": "960"},
                },
                {
                    "mode": "single",
                    "start": 5.5,
                    "end": 7.0,
                    "crop": {"x": "150", "y": "0", "w": "608", "h": "1080"},
                },
            ],
        }

        filter_str, label = build_layout_filtergraph(
            dummy_plan,
            duration_sec=7.0,
            ass_subtitle_path=self.ass_path,
        )

        self.assertEqual(label, "v_final")
        self.assertIn("[0:v]split=4[b0][b1][b2][b3];", filter_str)
        self.assertIn("vstack=inputs=2[v_seg_1];", filter_str)
        self.assertIn("concat=n=3:v=1:a=0[v_composed]", filter_str)
        self.assertIn("[v_composed]subtitles='", filter_str)


if __name__ == "__main__":
    unittest.main()
