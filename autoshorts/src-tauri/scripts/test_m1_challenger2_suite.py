#!/usr/bin/env python3
"""
Adversarial Challenge Test Suite - Milestone 1 Challenger 2
Validates:
1. Offline Guarantees & Zero Network Socket Calls on importing and using autoclip_studio
2. TitleArt Rendering Across ALL Styles ('comic', 'neon', 'arena', 'editorial', 'pixel', 'frosted')
3. Pydantic Model Serialization & Deserialization Stress Testing on Draft, Scene, CropPoint
"""
import io
import math
import os
import re
import socket
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image

# Ensure scripts directory is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


# ============================================================================
# 1. SOCKET / NETWORK FIREWALL HARNESS
# ============================================================================
class NetworkAccessAttempted(RuntimeError):
    pass


class SocketFirewall:
    """Strict socket firewall that intercepts all network connection and resolution attempts."""
    def __init__(self):
        self.call_log = []
        self._orig_socket_connect = socket.socket.connect
        self._orig_socket_connect_ex = socket.socket.connect_ex
        self._orig_socket_bind = socket.socket.bind
        self._orig_socket_send = socket.socket.send
        self._orig_socket_sendto = socket.socket.sendto
        self._orig_socket_sendall = socket.socket.sendall
        self._orig_create_connection = socket.create_connection
        self._orig_getaddrinfo = socket.getaddrinfo
        self._orig_gethostbyname = socket.gethostbyname

    def _block(self, func_name, *args, **kwargs):
        msg = f"Network call attempted: {func_name}(args={args}, kwargs={kwargs})"
        self.call_log.append(msg)
        raise NetworkAccessAttempted(msg)

    def __enter__(self):
        self.call_log.clear()
        socket.socket.connect = lambda *a, **k: self._block("socket.connect", *a, **k)
        socket.socket.connect_ex = lambda *a, **k: self._block("socket.connect_ex", *a, **k)
        socket.socket.bind = lambda *a, **k: self._block("socket.bind", *a, **k)
        socket.socket.send = lambda *a, **k: self._block("socket.send", *a, **k)
        socket.socket.sendto = lambda *a, **k: self._block("socket.sendto", *a, **k)
        socket.socket.sendall = lambda *a, **k: self._block("socket.sendall", *a, **k)
        socket.create_connection = lambda *a, **k: self._block("socket.create_connection", *a, **k)
        socket.getaddrinfo = lambda *a, **k: self._block("socket.getaddrinfo", *a, **k)
        socket.gethostbyname = lambda *a, **k: self._block("socket.gethostbyname", *a, **k)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        socket.socket.connect = self._orig_socket_connect
        socket.socket.connect_ex = self._orig_socket_connect_ex
        socket.socket.bind = self._orig_socket_bind
        socket.socket.send = self._orig_socket_send
        socket.socket.sendto = self._orig_socket_sendto
        socket.socket.sendall = self._orig_socket_sendall
        socket.create_connection = self._orig_create_connection
        socket.getaddrinfo = self._orig_getaddrinfo
        socket.gethostbyname = self._orig_gethostbyname


# ============================================================================
# TEST SUITE 1: OFFLINE ISOLATION & ZERO NETWORK CALLS
# ============================================================================
class TestOfflineGuarantees(unittest.TestCase):
    def test_firewall_positive_control(self):
        """Sanity check: ensure our firewall actually triggers if a network call is attempted."""
        with SocketFirewall() as fw:
            with self.assertRaises(NetworkAccessAttempted):
                s = socket.socket()
                s.connect(("1.1.1.1", 80))
            self.assertTrue(len(fw.call_log) > 0)

    def test_autoclip_studio_modules_import_and_execution_offline(self):
        """Verify that importing and calling autoclip_studio modules makes ZERO network calls."""
        with SocketFirewall() as fw:
            # 1. Import main package and all submodules
            import autoclip_studio
            from autoclip_studio import (
                platform_strategy,
                framing,
                title_art,
                models,
                caption_layout,
                publish_export,
                video_encoder,
                time_utils,
            )

            # 2. Execute platform_strategy functions
            strategies = [
                "douyin", "tiktok", "instagram_reels", "youtube_shorts",
                "youtube_long", "bilibili", "xiaohongshu", "original"
            ]
            for sid in strategies:
                strat = platform_strategy(sid)
                self.assertIsNotNone(strat)
                self.assertEqual(strat.id, sid)
                spec = strat.export_spec()
                self.assertIn("strategy_id", spec)
                summary = strat.public_summary()
                self.assertIn("aspect", summary)

            all_strats = autoclip_studio.list_platform_strategies()
            self.assertGreaterEqual(len(all_strats), 8)

            # 3. Execute framing functions
            installed = framing.is_installed()
            self.assertTrue(installed, "OpenCV and YuNet model must be reported installed")
            cv2_mod = framing.load_cv2()
            self.assertIsNotNone(cv2_mod)
            self.assertTrue(framing.MODEL.exists(), f"Model file must exist at {framing.MODEL}")

            # Face detector instantiation
            import cv2
            detector = cv2.FaceDetectorYN.create(str(framing.MODEL), "", (320, 320), score_threshold=0.6)
            self.assertIsNotNone(detector)

            # Framing math
            wf = framing.window_fraction(1920, 1080, 1080, 1920)
            self.assertAlmostEqual(wf, 0.31640625, places=5)
            cx = framing.crop_x_for_center(0.5, wf)
            self.assertAlmostEqual(cx, 0.5, places=5)
            track = framing.segment_track([(0.0, 0.5), (1.0, 0.5), (2.5, 0.8), (3.0, 0.8)])
            self.assertIsInstance(track, list)

            # 4. Execute title_art functions
            motion_comic = title_art.overlay_motion("comic", True, 1920)
            self.assertEqual(motion_comic[0], "0")
            motion_arena = title_art.overlay_motion("arena", True, 1920)
            self.assertIn("-36", motion_arena[0])

            # Generate real art
            art_img = title_art.artwork("OFFLINE TEST", "comic", 1080, 1920, version=6)
            self.assertEqual(art_img.size, (1080, 1920))
            png_bytes = title_art.png_bytes("OFFLINE TEST", "comic", 1080, 1920, version=6)
            self.assertTrue(png_bytes.startswith(b"\x89PNG\r\n\x1a\n"))

            # 5. Execute models creation and validation
            cp = models.CropPoint(start=0.0, crop_x=0.5, mode="crop")
            sc = models.Scene(id="test_scene", start=0.0, end=10.0, crop_track=[cp])
            df = models.Draft(
                id="test_draft",
                title="Offline Test Draft",
                scenes=[sc],
                title_style="comic",
                title_template_version=6,
            )
            dumped_json = df.model_dump_json()
            reloaded_df = models.Draft.model_validate_json(dumped_json)
            self.assertEqual(reloaded_df.id, "test_draft")

            # 6. Execute caption_layout functions
            w_cjk = caption_layout.width("测试")
            self.assertEqual(w_cjk, 2.0)
            w_mixed = caption_layout.width("测试 AutoShorts")
            self.assertGreater(w_mixed, 2.0)
            screens = caption_layout.split_screens("第一句字幕，用于测试分行。第二句字幕用于测试断句。", 14)
            self.assertTrue(len(screens) >= 1)

            # Assert ZERO network calls were logged by firewall
            self.assertEqual(
                len(fw.call_log), 0,
                f"Network calls were attempted during offline execution: {fw.call_log}"
            )


# ============================================================================
# TEST SUITE 2: TITLEART RENDERING ACROSS ALL 6 STYLES
# ============================================================================
class TestTitleArtRendering(unittest.TestCase):
    STYLES = ["comic", "neon", "arena", "editorial", "pixel", "frosted"]
    CANVASES = [
        (1080, 1920),  # 9:16 Portrait
        (1080, 1080),  # 1:1 Square
        (1920, 1080),  # 16:9 Landscape
        (720, 1280),   # 9:16 720p
    ]

    def setUp(self):
        from autoclip_studio import title_art
        self.title_art = title_art

    def _verify_png_image(self, raw_bytes: bytes, target_w: int, target_h: int, check_visible_pixels=True):
        self.assertIsInstance(raw_bytes, bytes)
        self.assertGreater(len(raw_bytes), 100, "PNG output must not be trivial empty file")
        self.assertTrue(raw_bytes.startswith(b"\x89PNG\r\n\x1a\n"), "Must have PNG magic header")
        
        # Load through PIL to verify image decodability
        img = Image.open(io.BytesIO(raw_bytes))
        self.assertEqual(img.format, "PNG")
        self.assertEqual(img.mode, "RGBA")
        self.assertEqual(img.size, (target_w, target_h), f"Image dimensions {img.size} must match canvas ({target_w}, {target_h})")

        if check_visible_pixels:
            # Verify the image is not purely transparent (it actually rendered text)
            alpha_channel = img.getchannel("A")
            bbox = alpha_channel.getbbox()
            self.assertIsNotNone(bbox, "Rendered TitleArt must contain visible non-transparent pixels")
            rendered_w = bbox[2] - bbox[0]
            rendered_h = bbox[3] - bbox[1]
            self.assertGreater(rendered_w, 10, "Rendered artwork width must be significant")
            self.assertGreater(rendered_h, 10, "Rendered artwork height must be significant")
        return img

    def test_all_styles_render_valid_png_version_6_portrait(self):
        """Generate TitleArt across ALL styles in version 6 at 1080x1920."""
        text = "STAGE CLEAR\nBONUS LEVEL"
        for style in self.STYLES:
            with self.subTest(style=style, version=6):
                raw = self.title_art.png_bytes(text, style, 1080, 1920, version=6)
                self._verify_png_image(raw, 1080, 1920, check_visible_pixels=True)

    def test_all_styles_render_valid_png_version_3_portrait(self):
        """Generate TitleArt across ALL styles in version 3 at 1080x1920."""
        text = "ARCADE EDITION\nSTAGE CLEAR"
        for style in self.STYLES:
            with self.subTest(style=style, version=3):
                raw = self.title_art.png_bytes(text, style, 1080, 1920, version=3)
                self._verify_png_image(raw, 1080, 1920, check_visible_pixels=True)

    def test_all_styles_across_diverse_canvas_dimensions(self):
        """Verify each style produces an image exactly matching canvas across square, landscape, etc."""
        text = "MULTICANVAS\nTEST"
        for w, h in self.CANVASES:
            for style in self.STYLES:
                with self.subTest(style=style, canvas=f"{w}x{h}"):
                    raw = self.title_art.png_bytes(text, style, w, h, version=6)
                    self._verify_png_image(raw, w, h, check_visible_pixels=True)

    def test_title_art_with_cjk_characters(self):
        """Verify TitleArt renders CJK (Chinese) characters across all styles."""
        cjk_text = "独家揭秘\n颠覆认知"
        for style in self.STYLES:
            with self.subTest(style=style, text="cjk"):
                raw = self.title_art.png_bytes(cjk_text, style, 1080, 1920, version=6)
                self._verify_png_image(raw, 1080, 1920, check_visible_pixels=True)

    def test_title_art_with_mixed_latin_and_cjk(self):
        """Verify TitleArt renders mixed Latin and CJK characters."""
        mixed_text = "TOP 10 精彩瞬间\nEPISODE 01"
        for style in self.STYLES:
            with self.subTest(style=style, text="mixed"):
                raw = self.title_art.png_bytes(mixed_text, style, 1080, 1920, version=6)
                self._verify_png_image(raw, 1080, 1920, check_visible_pixels=True)

    def test_title_art_single_line_and_three_lines(self):
        """Verify TitleArt handles 1-line and 3-line text correctly."""
        for lines_count, text in [(1, "SINGLE LINE"), (3, "FIRST LINE\nSECOND LINE\nTHIRD LINE")]:
            for style in self.STYLES:
                with self.subTest(style=style, lines=lines_count):
                    raw = self.title_art.png_bytes(text, style, 1080, 1920, version=6)
                    self._verify_png_image(raw, 1080, 1920, check_visible_pixels=True)

    def test_title_art_empty_and_whitespace_text(self):
        """Empty or whitespace text should produce a transparent canvas matching dimensions."""
        for empty_text in ["", "   ", "\n\t  "]:
            for style in self.STYLES:
                with self.subTest(style=style, empty_case=repr(empty_text)):
                    raw = self.title_art.png_bytes(empty_text, style, 1080, 1920, version=6)
                    img = self._verify_png_image(raw, 1080, 1920, check_visible_pixels=False)
                    alpha = img.getchannel("A")
                    self.assertIsNone(alpha.getbbox(), "Empty text must produce fully transparent canvas")

    def test_canvas_bounds_validation(self):
        """Canvas < 16 or > 16M pixels must raise ValueError."""
        with self.assertRaises(ValueError):
            self.title_art.png_bytes("HELLO", "comic", 15, 1080, version=6)
        with self.assertRaises(ValueError):
            self.title_art.png_bytes("HELLO", "comic", 1080, 15, version=6)
        with self.assertRaises(ValueError):
            # 5000 x 5000 = 25,000,000 > 16,777,216
            self.title_art.png_bytes("HELLO", "comic", 5000, 5000, version=6)

    def test_too_long_text_rejection(self):
        """Text exceeding line limit or shrinking below min_size must raise ValueError."""
        excessive_text = "THIS IS AN EXTREMELY LONG PARAGRAPH THAT CANNOT POSSIBLY FIT INTO THREE LINES EVEN AT THE MINIMUM FONT SIZE OF THE CANVAS" * 4
        with self.assertRaises(ValueError):
            self.title_art.png_bytes(excessive_text, "comic", 1080, 1920, version=6)

    def test_version_style_matrix_behavior(self):
        """Verify version fallbacks and constraints."""
        # Version 1 only supports comic, neon, arena
        for style in ["comic", "neon", "arena"]:
            raw = self.title_art.png_bytes("TITLE", style, 1080, 1920, version=1)
            self._verify_png_image(raw, 1080, 1920)
        for unsupported_style in ["editorial", "pixel", "frosted"]:
            with self.assertRaises(ValueError):
                self.title_art.png_bytes("TITLE", unsupported_style, 1080, 1920, version=1)

        # Version 2 supports editorial
        raw_ed = self.title_art.png_bytes("EDITORIAL", "editorial", 1080, 1920, version=2)
        self._verify_png_image(raw_ed, 1080, 1920)

        # Version 4 and 5 punch comic
        raw_c4 = self.title_art.png_bytes("COMIC", "comic", 1080, 1920, version=4)
        self._verify_png_image(raw_c4, 1080, 1920)
        raw_c5 = self.title_art.png_bytes("COMIC", "comic", 1080, 1920, version=5)
        self._verify_png_image(raw_c5, 1080, 1920)


# ============================================================================
# TEST SUITE 3: PYDANTIC MODEL SERIALIZATION & DESERIALIZATION STRESS TESTS
# ============================================================================
class TestPydanticModelSerialization(unittest.TestCase):
    def setUp(self):
        from autoclip_studio import models
        self.models = models

    # --- CropPoint Tests ---
    def test_crop_point_valid_defaults_and_custom(self):
        cp = self.models.CropPoint(start=0.0, crop_x=0.5)
        self.assertEqual(cp.start, 0.0)
        self.assertEqual(cp.crop_x, 0.5)
        self.assertEqual(cp.mode, "crop")

        cp_fit = self.models.CropPoint(start=5.25, crop_x=0.8, mode="fit")
        self.assertEqual(cp_fit.mode, "fit")

    def test_crop_point_missing_required_fields(self):
        with self.assertRaises(self.models.BaseModel.__pydantic_core_schema__['schema']['type'] if False else Exception):
            self.models.CropPoint.model_validate({})
        with self.assertRaises(Exception):
            self.models.CropPoint.model_validate({"start": 1.0})  # missing crop_x
        with self.assertRaises(Exception):
            self.models.CropPoint.model_validate({"crop_x": 0.5})  # missing start

    def test_crop_point_boundary_and_invalid_values(self):
        # start >= 0
        self.models.CropPoint(start=0.0, crop_x=0.5)
        with self.assertRaises(Exception):
            self.models.CropPoint(start=-0.001, crop_x=0.5)

        # crop_x in [0, 1]
        self.models.CropPoint(start=1.0, crop_x=0.0)
        self.models.CropPoint(start=1.0, crop_x=1.0)
        with self.assertRaises(Exception):
            self.models.CropPoint(start=1.0, crop_x=-0.01)
        with self.assertRaises(Exception):
            self.models.CropPoint(start=1.0, crop_x=1.01)

        # inf and nan rejection
        with self.assertRaises(Exception):
            self.models.CropPoint(start=1.0, crop_x=float("nan"))
        with self.assertRaises(Exception):
            self.models.CropPoint(start=1.0, crop_x=float("inf"))
        with self.assertRaises(Exception):
            self.models.CropPoint(start=float("inf"), crop_x=0.5)

        # invalid mode
        with self.assertRaises(Exception):
            self.models.CropPoint(start=1.0, crop_x=0.5, mode="invalid_mode")

        # extra fields forbidden
        with self.assertRaises(Exception):
            self.models.CropPoint.model_validate({"start": 1.0, "crop_x": 0.5, "extra": "forbidden"})

    def test_crop_point_serialization_roundtrip(self):
        cp = self.models.CropPoint(start=3.1415, crop_x=0.75, mode="fit")
        dumped = cp.model_dump()
        self.assertEqual(dumped, {"start": 3.1415, "crop_x": 0.75, "mode": "fit"})
        json_str = cp.model_dump_json()
        restored = self.models.CropPoint.model_validate_json(json_str)
        self.assertEqual(cp, restored)

    # --- Scene Tests ---
    def test_scene_valid_defaults(self):
        sc = self.models.Scene(id="sc-01", start=0.0, end=5.0)
        self.assertEqual(sc.id, "sc-01")
        self.assertEqual(sc.label, "片段")
        self.assertEqual(sc.evidence, "")
        self.assertIsNone(sc.crop_x)
        self.assertIsNone(sc.crop_track)
        self.assertIsNone(sc.framing_source)
        self.assertFalse(sc.framing_adjusted)

    def test_scene_missing_fields(self):
        with self.assertRaises(Exception):
            self.models.Scene.model_validate({"start": 0.0, "end": 5.0})  # missing id
        with self.assertRaises(Exception):
            self.models.Scene.model_validate({"id": "sc1", "end": 5.0})    # missing start
        with self.assertRaises(Exception):
            self.models.Scene.model_validate({"id": "sc1", "start": 0.0})  # missing end

    def test_scene_id_constraints(self):
        # valid characters: ^[a-zA-Z0-9_-]+$
        self.models.Scene(id="a-b_C1", start=0.0, end=1.0)
        # invalid characters
        with self.assertRaises(Exception):
            self.models.Scene(id="sc 1", start=0.0, end=1.0)  # space
        with self.assertRaises(Exception):
            self.models.Scene(id="sc#1", start=0.0, end=1.0)  # hash
        with self.assertRaises(Exception):
            self.models.Scene(id="", start=0.0, end=1.0)      # empty
        with self.assertRaises(Exception):
            self.models.Scene(id="a" * 101, start=0.0, end=1.0)  # > 100 chars

    def test_scene_time_interval_constraints(self):
        # start >= 0
        with self.assertRaises(Exception):
            self.models.Scene(id="sc1", start=-0.5, end=1.0)

        # end > 0
        with self.assertRaises(Exception):
            self.models.Scene(id="sc1", start=0.0, end=0.0)

        # end - start >= 0.1
        self.models.Scene(id="sc1", start=1.0, end=1.1001)
        with self.assertRaises(Exception):
            self.models.Scene(id="sc1", start=1.0, end=1.05)
        with self.assertRaises(Exception):
            self.models.Scene(id="sc1", start=5.0, end=4.0)

    def test_scene_crop_track_constraints(self):
        points = [self.models.CropPoint(start=float(i), crop_x=0.5) for i in range(10)]
        sc = self.models.Scene(id="sc1", start=0.0, end=20.0, crop_track=points)
        self.assertEqual(len(sc.crop_track), 10)

        # > 400 points should fail
        too_many = [self.models.CropPoint(start=float(i)*0.1, crop_x=0.5) for i in range(401)]
        with self.assertRaises(Exception):
            self.models.Scene(id="sc1", start=0.0, end=50.0, crop_track=too_many)

    def test_scene_extra_fields_forbidden(self):
        with self.assertRaises(Exception):
            self.models.Scene.model_validate({"id": "sc1", "start": 0.0, "end": 1.0, "unknown_key": "bad"})

    # --- Draft Tests ---
    def test_draft_minimal_and_defaults(self):
        sc = self.models.Scene(id="sc_01", start=0.0, end=10.0)
        df = self.models.Draft(id="df_01", title="Test Title", scenes=[sc])
        self.assertEqual(df.id, "df_01")
        self.assertEqual(df.title, "Test Title")
        self.assertEqual(df.language, "source")
        self.assertEqual(df.aspect, "original")
        self.assertEqual(df.layout, "fit")
        self.assertEqual(df.crop_x, 0.5)
        self.assertEqual(df.title_style, "plain")
        self.assertEqual(df.title_template_version, 1)
        self.assertTrue(df.title_motion)
        self.assertEqual(df.title_scale, 1.0)
        self.assertEqual(df.title_y, 0.12)
        self.assertIsNone(df.title_accent)
        self.assertTrue(df.subtitles)
        self.assertEqual(df.subtitle_style, "clean")
        self.assertTrue(df.original_audio)
        self.assertEqual(df.revision, 1)
        self.assertEqual(df.origin, "manual")
        self.assertIsNone(df.packaging)

    def test_draft_missing_required_fields(self):
        sc = self.models.Scene(id="s1", start=0.0, end=1.0)
        with self.assertRaises(Exception):
            self.models.Draft.model_validate({"title": "T", "scenes": [sc.model_dump()]})  # missing id
        with self.assertRaises(Exception):
            self.models.Draft.model_validate({"id": "d1", "scenes": [sc.model_dump()]})    # missing title
        with self.assertRaises(Exception):
            self.models.Draft.model_validate({"id": "d1", "title": "T"})                  # missing scenes

    def test_draft_scenes_bounds(self):
        # Empty scenes list forbidden
        with self.assertRaises(Exception):
            self.models.Draft(id="d1", title="T", scenes=[])
        # > 30 scenes forbidden
        too_many_scenes = [self.models.Scene(id=f"s_{i}", start=0.0, end=1.0) for i in range(31)]
        with self.assertRaises(Exception):
            self.models.Draft(id="d1", title="T", scenes=too_many_scenes)

    def test_draft_title_style_and_version_matrix(self):
        sc = self.models.Scene(id="s1", start=0.0, end=1.0)

        # Version 6: only 6 design presets allowed
        for valid_v6 in ["comic", "neon", "arena", "editorial", "pixel", "frosted"]:
            df = self.models.Draft(id="d1", title="T", scenes=[sc], title_style=valid_v6, title_template_version=6)
            self.assertEqual(df.title_style, valid_v6)
        for invalid_v6 in ["plain", "impact", "card"]:
            with self.assertRaises(Exception):
                self.models.Draft(id="d1", title="T", scenes=[sc], title_style=invalid_v6, title_template_version=6)

        # Version 4 and 5: only comic allowed
        self.models.Draft(id="d1", title="T", scenes=[sc], title_style="comic", title_template_version=4)
        self.models.Draft(id="d1", title="T", scenes=[sc], title_style="comic", title_template_version=5)
        for non_comic in ["neon", "arena", "editorial", "pixel", "frosted"]:
            with self.assertRaises(Exception):
                self.models.Draft(id="d1", title="T", scenes=[sc], title_style=non_comic, title_template_version=4)
            with self.assertRaises(Exception):
                self.models.Draft(id="d1", title="T", scenes=[sc], title_style=non_comic, title_template_version=5)

        # Editorial: requires version >= 2
        with self.assertRaises(Exception):
            self.models.Draft(id="d1", title="T", scenes=[sc], title_style="editorial", title_template_version=1)
        self.models.Draft(id="d1", title="T", scenes=[sc], title_style="editorial", title_template_version=2)
        self.models.Draft(id="d1", title="T", scenes=[sc], title_style="editorial", title_template_version=3)
        self.models.Draft(id="d1", title="T", scenes=[sc], title_style="editorial", title_template_version=6)

        # Pixel and Frosted: require version 3 or 6
        for pf in ["pixel", "frosted"]:
            for bad_ver in [1, 2]:
                with self.assertRaises(Exception):
                    self.models.Draft(id="d1", title="T", scenes=[sc], title_style=pf, title_template_version=bad_ver)
            self.models.Draft(id="d1", title="T", scenes=[sc], title_style=pf, title_template_version=3)
            self.models.Draft(id="d1", title="T", scenes=[sc], title_style=pf, title_template_version=6)

    def test_draft_numeric_boundaries(self):
        sc = self.models.Scene(id="s1", start=0.0, end=1.0)
        # title_scale: ge=.75, le=1.2
        self.models.Draft(id="d1", title="T", scenes=[sc], title_scale=0.75)
        self.models.Draft(id="d1", title="T", scenes=[sc], title_scale=1.2)
        with self.assertRaises(Exception):
            self.models.Draft(id="d1", title="T", scenes=[sc], title_scale=0.749)
        with self.assertRaises(Exception):
            self.models.Draft(id="d1", title="T", scenes=[sc], title_scale=1.201)

        # title_y: ge=.06, le=.70
        self.models.Draft(id="d1", title="T", scenes=[sc], title_y=0.06)
        self.models.Draft(id="d1", title="T", scenes=[sc], title_y=0.70)
        with self.assertRaises(Exception):
            self.models.Draft(id="d1", title="T", scenes=[sc], title_y=0.059)
        with self.assertRaises(Exception):
            self.models.Draft(id="d1", title="T", scenes=[sc], title_y=0.701)

        # title_accent hex regex
        self.models.Draft(id="d1", title="T", scenes=[sc], title_accent="#ff00aa")
        self.models.Draft(id="d1", title="T", scenes=[sc], title_accent="#123456")
        with self.assertRaises(Exception):
            self.models.Draft(id="d1", title="T", scenes=[sc], title_accent="invalid_color")
        with self.assertRaises(Exception):
            self.models.Draft(id="d1", title="T", scenes=[sc], title_accent="#fff")  # must be 6 hex digits

    def test_full_complex_draft_serialization_roundtrip(self):
        """Serialize a complex Draft with multiple scenes, crop points, and packaging to JSON and reload."""
        cues = [
            self.models.PackagingCue(start=0.0, end=2.0, text="Cue 1", original="Original 1"),
            self.models.PackagingCue(start=2.0, end=4.5, text="Cue 2", original="Original 2"),
        ]
        speakers = [self.models.PackagingSpeaker(at=0.5, name="Alice", role="Host")]
        tags = [self.models.PackagingMark(at=1.0, text="Highlight")]
        pkg = self.models.Packaging(
            version=1,
            template="interview_zh",
            audience_language="zh",
            title_lines=["Title Line 1", "Title Line 2"],
            title_accent_line=0,
            cues=cues,
            speakers=speakers,
            tags=tags,
            style="classic",
        )

        scenes = [
            self.models.Scene(
                id="scene_001",
                label="Intro",
                start=0.0,
                end=3.5,
                crop_track=[
                    self.models.CropPoint(start=0.0, crop_x=0.45, mode="crop"),
                    self.models.CropPoint(start=1.8, crop_x=0.65, mode="crop"),
                ]
            ),
            self.models.Scene(
                id="scene_002",
                label="Slide Presentation",
                start=3.5,
                end=8.0,
                crop_track=[
                    self.models.CropPoint(start=0.0, crop_x=0.5, mode="fit")
                ]
            ),
        ]

        original_draft = self.models.Draft(
            id="draft_full_01",
            title="Full Complex Draft",
            hook="Incredible Hook",
            scenes=scenes,
            language="zh",
            aspect="portrait",
            layout="blur",
            crop_x=0.55,
            title_style="comic",
            title_template_version=6,
            title_motion=True,
            title_scale=1.1,
            title_y=0.15,
            title_accent="#ffe52d",
            subtitles=True,
            subtitle_style="box",
            original_audio=True,
            revision=2,
            updated_at="2026-10-12T09:00:00Z",
            origin="autoclip_adapter",
            packaging=pkg,
            caption_mask=(0.7, 0.95),
        )

        # JSON Roundtrip
        json_output = original_draft.model_dump_json()
        reconstituted = self.models.Draft.model_validate_json(json_output)

        self.assertEqual(original_draft.id, reconstituted.id)
        self.assertEqual(len(reconstituted.scenes), 2)
        self.assertEqual(len(reconstituted.scenes[0].crop_track), 2)
        self.assertEqual(reconstituted.scenes[1].crop_track[0].mode, "fit")
        self.assertEqual(reconstituted.packaging.template, "interview_zh")
        self.assertEqual(len(reconstituted.packaging.cues), 2)
        self.assertEqual(reconstituted.caption_mask, (0.7, 0.95))
        self.assertEqual(original_draft, reconstituted)


if __name__ == "__main__":
    unittest.main(verbosity=2)
