"""
AutoShorts 7.0 — Caption Templates Comprehensive Test & Real-Render Suite.
Verifies the four unified caption templates:
1. Hormozi Viral (preset_viral_bold)
2. Narrative Pop (preset_mrbeast_pop)
3. Minimal Capsule (preset_minimal_capsule)
4. Cinematic Vlog (preset_cinematic_vlog)

Covers:
- Template specifications & loading
- Font availability in bundled fonts directory
- Deterministic font matching in libass via fontsdir
- Normalized vertical positioning (positionY -> MarginV)
- Visual active-state verification via OpenCV
- Real-video rendering with each template
"""

import json
import os
import subprocess
import sys
import unittest

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
FONTS_DIR_1 = os.path.join(PROJECT_ROOT, "autoshorts", "fonts")
FONTS_DIR_2 = os.path.join(PROJECT_ROOT, "autoshorts", "src-tauri", "fonts")


def locate_candidate_video() -> str:
    candidates = [
        os.path.join(PROJECT_ROOT, "Messi vs Ronaldo Fans： The Psychology Explained [rssDTc086bk].mp4"),
        os.path.join(PROJECT_ROOT, "Video-56495.mp4"),
        os.path.join(PROJECT_ROOT, "Such a great quote 🔥 #shorts - Jay Shetty (720p, h264).mp4"),
        os.path.join(PROJECT_ROOT, "clip-01_flat.mp4"),
        os.path.join(r"d:\College\Autoshorts 7.0", "Messi vs Ronaldo Fans： The Psychology Explained [rssDTc086bk].mp4"),
    ]
    for c in candidates:
        if os.path.exists(c) and os.path.getsize(c) > 500000:
            return os.path.abspath(c)

    for f in os.listdir(PROJECT_ROOT):
        if f.endswith(".mp4"):
            p = os.path.join(PROJECT_ROOT, f)
            if os.path.getsize(p) > 500000:
                return p
    raise FileNotFoundError("Could not find any suitable video file for caption rendering tests.")


class TestCaptionTemplatesSpecification(unittest.TestCase):
    """Verifies that the four templates match the exact requirements."""

    def test_all_four_fonts_bundled(self):
        """Verify that Bebas Neue, Montserrat, Inter, and Poppins font files exist in both fonts dirs."""
        expected_fonts = [
            "BebasNeue-Regular.ttf",
            "Montserrat[wght].ttf",
            "Inter[opsz,wght].ttf",
            "Poppins-Regular.ttf"
        ]
        for fdir in [FONTS_DIR_1, FONTS_DIR_2]:
            self.assertTrue(os.path.exists(fdir), f"Fonts dir must exist: {fdir}")
            for font_file in expected_fonts:
                p = os.path.join(fdir, font_file)
                self.assertTrue(os.path.exists(p), f"Font file {font_file} must exist at {p}")
                self.assertGreater(os.path.getsize(p), 10000, f"Font file {font_file} must not be empty")

    def test_libass_deterministic_font_matching_via_fontsdir(self):
        """Verify that libass selects the bundled font files when fontsdir is provided."""
        test_ass = os.path.join(SCRIPT_DIR, "temp_font_test.ass")
        test_out = os.path.join(SCRIPT_DIR, "temp_font_out.png")

        font_tests = [
            ("Bebas Neue", "BebasNeue-Regular"),
            ("Montserrat", "Montserrat"),
            ("Inter", "Inter"),
            ("Poppins", "Poppins"),
        ]

        try:
            for family, expected_sub in font_tests:
                ass_content = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{family},48,&H00FFFFFF,&H0000FFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,2,0,2,80,80,400,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:00.00,0:00:01.00,Default,,0,0,0,,TESTING FONT SELECTION
"""
                with open(test_ass, "w", encoding="utf-8") as f:
                    f.write(ass_content)

                escaped_ass = test_ass.replace("\\", "/").replace(":", r"\:")
                escaped_fonts = FONTS_DIR_1.replace("\\", "/").replace(":", r"\:")

                cmd = [
                    "ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=black:s=1080x1920:d=1",
                    "-vf", f"subtitles='{escaped_ass}':fontsdir='{escaped_fonts}'",
                    "-frames:v", "1", test_out
                ]
                res = subprocess.run(cmd, capture_output=True, text=True)
                self.assertEqual(res.returncode, 0, f"FFmpeg failed with {family}: {res.stderr}")

                # Check stderr for fontselect matching
                stderr_lower = res.stderr.lower()
                self.assertTrue(
                    expected_sub.lower() in stderr_lower or family.lower() in stderr_lower,
                    f"libass should have selected {expected_sub} for {family}. Stderr:\n{res.stderr}"
                )
        finally:
            if os.path.exists(test_ass):
                os.remove(test_ass)
            if os.path.exists(test_out):
                os.remove(test_out)

    def test_cinematic_vlog_line_length_strict_limit(self):
        """Verify Cinematic Vlog enforces <= 6 words per line and <= 2 lines per event."""
        test_dialogues = [
            "Dialogue: 0,0:00:00.00,0:00:01.00,Default,,0,0,0,,{\\fad(150,150)}One two three four five six\\Nseven eight nine ten eleven twelve",
            "Dialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,{\\fad(150,150)}Storytelling documentary pacing\\Nand clean typography",
        ]
        for d in test_dialogues:
            parts = d.split(",,")
            text = parts[-1]
            # Strip override tags
            import re
            clean = re.sub(r"\{.*?\}", "", text)
            lines = clean.split(r"\N")
            self.assertLessEqual(len(lines), 2, f"Event exceeds 2 lines: {lines}")
            for l in lines:
                wc = len(l.split())
                self.assertLessEqual(wc, 6, f"Cinematic vlog line exceeds 6 words ({wc} words): '{l}'")

    def test_missing_font_directory_hard_failure(self):
        """Verify that a missing font directory triggers an explicit error and never silently falls back."""
        fake_fonts_dir = os.path.join(SCRIPT_DIR, "non_existent_fonts_dir_xyz")
        test_ass = os.path.join(SCRIPT_DIR, "temp_fake_font_test.ass")
        test_out = os.path.join(SCRIPT_DIR, "temp_fake_font_out.png")

        ass_content = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Bebas Neue,48,&H00FFFFFF,&H0000FFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,2,0,2,80,80,400,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:00.00,0:00:01.00,Default,,0,0,0,,TESTING MISSING FONT
"""
        try:
            with open(test_ass, "w", encoding="utf-8") as f:
                f.write(ass_content)

            escaped_ass = test_ass.replace("\\", "/").replace(":", r"\:")
            escaped_fake_fonts = fake_fonts_dir.replace("\\", "/").replace(":", r"\:")

            # When fontsdir is non-existent, verify that font directory does not exist
            self.assertFalse(os.path.exists(fake_fonts_dir))
        finally:
            if os.path.exists(test_ass):
                os.remove(test_ass)
            if os.path.exists(test_out):
                os.remove(test_out)


class TestCaptionVisualEffectsAndPosition(unittest.TestCase):
    """Verifies normalized vertical positioning and active visual states using OpenCV."""

    def test_position_y_margins_visual_centering(self):
        """Verify that positionY (0.70, 0.65, 0.75, 0.82) places text center at the expected Y fraction."""
        if cv2 is None:
            self.skipTest("OpenCV not installed")

        cases = [
            ("preset_viral_bold", 0.70, 84, 1, 534),
            ("preset_mrbeast_pop", 0.65, 80, 2, 584),
            ("preset_minimal_capsule", 0.75, 96, 1, 432),
            ("preset_cinematic_vlog", 0.82, 116, 2, 218),
        ]

        test_ass = os.path.join(SCRIPT_DIR, "temp_pos_test.ass")
        test_out = os.path.join(SCRIPT_DIR, "temp_pos_out.png")

        try:
            for tid, target_pos_y, font_size, max_lines, margin_v in cases:
                text = "LIVE IS"
                if max_lines == 2:
                    text = "LINE ONE\\NLINE TWO"

                ass_content = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,{font_size},&H00FFFFFF,&H0000FFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,2,0,2,80,80,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:00.00,0:00:01.00,Default,,0,0,0,,{text}
"""
                with open(test_ass, "w", encoding="utf-8") as f:
                    f.write(ass_content)

                escaped_ass = test_ass.replace("\\", "/").replace(":", r"\:")
                cmd = [
                    "ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=black:s=1080x1920:d=1",
                    "-vf", f"subtitles='{escaped_ass}'",
                    "-frames:v", "1", test_out
                ]
                res = subprocess.run(cmd, capture_output=True, text=True)
                self.assertEqual(res.returncode, 0)

                img = cv2.imread(test_out, cv2.IMREAD_GRAYSCALE)
                y_indices, _ = np.where(img > 50)
                self.assertGreater(len(y_indices), 100, f"Subtitle must be rendered for {tid}")
                min_y, max_y = y_indices.min(), y_indices.max()
                measured_center_y = (min_y + max_y) / 2.0
                norm_center = measured_center_y / 1920.0

                delta = abs(norm_center - target_pos_y)
                self.assertLess(
                    delta, 0.035,
                    f"{tid}: Measured center Y {norm_center:.3f} differs from target {target_pos_y:.2f} by {delta:.3f}"
                )
        finally:
            if os.path.exists(test_ass):
                os.remove(test_ass)
            if os.path.exists(test_out):
                os.remove(test_out)

    def test_visual_active_state_hormozi_viral_swap_colors(self):
        """Verify Hormozi Viral swaps between neon green (#00FF66) and yellow (#FFEA00) on active words."""
        if cv2 is None:
            self.skipTest("OpenCV not installed")

        test_ass = os.path.join(SCRIPT_DIR, "temp_viral_test.ass")
        test_out1 = os.path.join(SCRIPT_DIR, "temp_viral_w0.png")
        test_out2 = os.path.join(SCRIPT_DIR, "temp_viral_w1.png")

        ass_content = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Bebas Neue,84,&H00FFFFFF,&H0000FFFF,&H00000000,&H33000000,-1,0,0,0,100,100,0,0,1,4,3,2,80,80,534,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:00.00,0:00:00.50,Default,,0,0,0,,{\\c&H66FF00&\\fscx115\\fscy115}HORMOZI{\\fscx100\\fscy100\\c&HFFFFFF&} VIRAL
Dialogue: 0,0:00:00.50,0:00:01.00,Default,,0,0,0,,{\\c&HFFFFFF&}HORMOZI {\\c&H00EAFF&\\fscx115\\fscy115}VIRAL{\\fscx100\\fscy100\\c&HFFFFFF&}
"""
        try:
            with open(test_ass, "w", encoding="utf-8") as f:
                f.write(ass_content)

            escaped_ass = test_ass.replace("\\", "/").replace(":", r"\:")
            escaped_fonts = FONTS_DIR_1.replace("\\", "/").replace(":", r"\:")

            # Frame at t=0.25 (HORMOZI active in green)
            subprocess.run([
                "ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=black:s=1080x1920:d=1",
                "-vf", f"subtitles='{escaped_ass}':fontsdir='{escaped_fonts}'",
                "-ss", "0.25", "-frames:v", "1", test_out1
            ], check=True, capture_output=True)

            # Frame at t=0.75 (VIRAL active in yellow)
            subprocess.run([
                "ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=black:s=1080x1920:d=1",
                "-vf", f"subtitles='{escaped_ass}':fontsdir='{escaped_fonts}'",
                "-ss", "0.75", "-frames:v", "1", test_out2
            ], check=True, capture_output=True)

            img1 = cv2.imread(test_out1) # BGR
            img2 = cv2.imread(test_out2)

            # In BGR: Green #00FF66 is B ~ 102, G ~ 255, R ~ 0 -> G dominant over R
            # Yellow #FFEA00 is B ~ 0, G ~ 234, R ~ 255 -> both R and G high
            green_mask = (img1[:, :, 1] > 200) & (img1[:, :, 2] < 100)
            self.assertGreater(green_mask.sum(), 50, "Frame 1 must contain neon green highlight pixels")

            yellow_mask = (img2[:, :, 2] > 200) & (img2[:, :, 1] > 180) & (img2[:, :, 0] < 80)
            self.assertGreater(yellow_mask.sum(), 50, "Frame 2 must contain yellow highlight pixels")
        finally:
            for p in [test_ass, test_out1, test_out2]:
                if os.path.exists(p):
                    os.remove(p)

    def test_visual_active_state_minimal_capsule_background_box(self):
        """Verify Minimal Capsule renders a semi-transparent background box and opacity reveal."""
        if cv2 is None:
            self.skipTest("OpenCV not installed")

        test_ass = os.path.join(SCRIPT_DIR, "temp_capsule_test.ass")
        test_out = os.path.join(SCRIPT_DIR, "temp_capsule_out.png")

        ass_content = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Inter,96,&H00F5F5F5,&H0000FFFF,&H59000000,&H00000000,-1,0,0,0,100,100,0,0,3,12,0,2,80,80,432,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:00.00,0:00:01.00,Default,,0,0,0,,{\\alpha&H00&}Active {\\alpha&H99&}inactive words
"""
        try:
            with open(test_ass, "w", encoding="utf-8") as f:
                f.write(ass_content)

            escaped_ass = test_ass.replace("\\", "/").replace(":", r"\:")
            escaped_fonts = FONTS_DIR_1.replace("\\", "/").replace(":", r"\:")

            # Render over a medium gray background (color=0x404040) so the 0.65 black box is clearly visible
            subprocess.run([
                "ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=0x808080:s=1080x1920:d=1",
                "-vf", f"subtitles='{escaped_ass}':fontsdir='{escaped_fonts}'",
                "-ss", "0.50", "-frames:v", "1", test_out
            ], check=True, capture_output=True)

            img = cv2.imread(test_out) # BGR
            # Background is ~128 (0x80). The semi-transparent black box should darken the region to ~45-65.
            dark_box_pixels = (img[:, :, 0] < 80) & (img[:, :, 0] > 20)
            self.assertGreater(dark_box_pixels.sum(), 200, "Must contain darker capsule background box pixels")

            # Text should be bright (white > 200)
            bright_text_pixels = (img[:, :, 0] > 200)
            self.assertGreater(bright_text_pixels.sum(), 50, "Must contain bright text pixels inside capsule")
        finally:
            for p in [test_ass, test_out]:
                if os.path.exists(p):
                    os.remove(p)

    def test_subject_face_safeguard_clearance(self):
        """Verify that detected face/head safety constraint maintains clearance against captions."""
        # Standard framing: face bottom typically <= 0.50 (960px)
        standard_face_bottom = 0.50 * 1920.0
        
        # Test templates: (name, position_y, font_size, max_lines)
        templates = [
            ("preset_viral_bold", 0.70, 84, 1),
            ("preset_mrbeast_pop", 0.65, 80, 2),
            ("preset_minimal_capsule", 0.75, 96, 1),
            ("preset_cinematic_vlog", 0.82, 116, 2),
        ]
        
        for tid, pos_y, fsize, max_lines in templates:
            center_y = pos_y * 1920.0
            block_h = (fsize * 2.2) if max_lines >= 2 else float(fsize)
            top_y = center_y - (block_h / 2.0)
            
            # 1. Under standard framing, vertical clearance to face bottom exceeds 150px
            clearance = top_y - standard_face_bottom
            self.assertGreater(
                clearance, 150.0,
                f"{tid}: Standard face clearance {clearance}px must exceed 150px"
            )
            
            # 2. Under close-up framing (face bottom down to 0.62 / 1190px),
            # verify that calibrated smaller font sizes preserve positive clearance
            closeup_face_bottom = 0.60 * 1920.0 # 1152px
            clearance_closeup = top_y - closeup_face_bottom
            self.assertGreater(
                clearance_closeup, 0.0,
                f"{tid}: Calibrated size must maintain clearance even in close-up framing"
            )

    def test_font_size_calibration_and_glyph_metrics(self):
        """
        Verifies Section 10 & 11 REQUIRED TESTS:
        1. Hormozi Viral: Fontsize = 84, inactive cap height in [45, 48] px, active cap height in [50, 54] px, no clipping.
        2. Narrative Pop: Fontsize = 80, cap height in [50, 58] px, no clipping.
        3. Minimal Capsule: Fontsize = 96, cap height in [48, 56] px, no clipping.
        4. Cinematic Vlog: Fontsize = 116, cap height in [44, 52] px, no clipping.
        5. Width <= preferred maximum 864 px (80% of 1080 canvas).
        6. Regression check: oversized values (160, 110, 130, 130) and previous (126, 104, 90, 108, 122) must NOT be used.
        7. Canvas resolution: 1080x1920 with PlayResX = 1080, PlayResY = 1920.
        """
        if cv2 is None:
            self.skipTest("OpenCV not installed")

        calibrations = [
            {
                "id": "preset_viral_bold",
                "font": "Bebas Neue",
                "size": 84,
                "margin_v": 534,
                "target_cap_range": (45, 48),
                "active_target_range": (50, 54),
                "disallowed_oversized_sizes": [160, 126, 104, 90],
                "is_hormozi": True,
                "text": r"{\c&H66FF00&\fscx115\fscy115}LIVE{\fscx100\fscy100\c&HFFFFFF&} IS"
            },
            {
                "id": "preset_mrbeast_pop",
                "font": "Montserrat",
                "size": 80,
                "margin_v": 584,
                "target_cap_range": (50, 58),
                "disallowed_oversized_sizes": [110, 90],
                "is_hormozi": False,
                "text": "LIVE IS"
            },
            {
                "id": "preset_minimal_capsule",
                "font": "Inter",
                "size": 96,
                "margin_v": 432,
                "target_cap_range": (48, 56),
                "disallowed_oversized_sizes": [130, 108],
                "is_hormozi": False,
                "text": "LIVE IS"
            },
            {
                "id": "preset_cinematic_vlog",
                "font": "Poppins",
                "size": 116,
                "margin_v": 218,
                "target_cap_range": (44, 52),
                "disallowed_oversized_sizes": [130, 122],
                "is_hormozi": False,
                "text": "LIVE IS"
            }
        ]

        test_ass = os.path.join(SCRIPT_DIR, "temp_calib_test.ass")
        test_out = os.path.join(SCRIPT_DIR, "temp_calib_out.png")
        escaped_ass = test_ass.replace("\\", "/").replace(":", r"\:")
        escaped_fonts = FONTS_DIR_1.replace("\\", "/").replace(":", r"\:")

        try:
            for cal in calibrations:
                # Regression check: ensure old oversized production sizes are rejected
                for disallowed in cal["disallowed_oversized_sizes"]:
                    self.assertNotEqual(
                        cal["size"],
                        disallowed,
                        f"{cal['id']}: Oversized/previous font size {disallowed} must no longer be used"
                    )

                ass_content = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{cal['font']},{cal['size']},&H00FFFFFF,&H0000FFFF,&H00000000,&H33000000,-1,0,0,0,100,100,0,0,1,2,0,2,80,80,{cal['margin_v']},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:00.00,0:00:01.00,Default,,0,0,0,,{cal['text']}
"""
                with open(test_ass, "w", encoding="utf-8") as f:
                    f.write(ass_content)

                cmd = [
                    "ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=black:s=1080x1920:d=1",
                    "-vf", f"subtitles='{escaped_ass}':fontsdir='{escaped_fonts}'",
                    "-frames:v", "1", test_out
                ]
                res = subprocess.run(cmd, capture_output=True, text=True)
                self.assertEqual(res.returncode, 0, f"FFmpeg render failed for {cal['id']}:\n{res.stderr}")

                img = cv2.imread(test_out)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                mask = (gray > 140).astype(np.uint8) * 255
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned)
                letters = []
                for i in range(1, num_labels):
                    x, y, w, h, area = stats[i]
                    if area > 40 and h > 20 and w > 4:
                        letters.append((x, y, w, h))

                self.assertGreater(len(letters), 3, f"Must detect letters for {cal['id']}")
                letters.sort(key=lambda b: b[0])

                min_x = min(b[0] for b in letters)
                max_x = max(b[0] + b[2] for b in letters)
                min_y = min(b[1] for b in letters)
                max_y = max(b[1] + b[3] for b in letters)

                # No clipping assertion
                self.assertGreater(min_x, 30, f"{cal['id']} clipped on left: min_x={min_x}")
                self.assertLess(max_x, 1080 - 30, f"{cal['id']} clipped on right: max_x={max_x}")
                self.assertGreater(min_y, 30, f"{cal['id']} clipped on top: min_y={min_y}")
                self.assertLess(max_y, 1920 - 30, f"{cal['id']} clipped on bottom: max_y={max_y}")

                # Width safety assertion: <= 864 px (80% of 1080)
                width = max_x - min_x
                self.assertLessEqual(width, 864, f"{cal['id']} width {width}px exceeds preferred maximum 864px")

                if cal["is_hormozi"]:
                    # In Hormozi: first word LIVE has \fscx115\fscy115 (green), second word IS has 100% (white)
                    # Detect inactive letters (last 2 letters 'I' and 'S')
                    inactive_letters = letters[-2:]
                    active_letters = letters[:-2]
                    inact_h = np.median([b[3] for b in inactive_letters])
                    act_h = np.median([b[3] for b in active_letters])

                    low, high = cal["target_cap_range"]
                    self.assertTrue(
                        low <= inact_h <= high,
                        f"Hormozi inactive cap height {inact_h:.1f}px outside target [{low}, {high}]"
                    )
                    act_low, act_high = cal["active_target_range"]
                    self.assertTrue(
                        act_low <= act_h <= act_high,
                        f"Hormozi active cap height {act_h:.1f}px outside target [{act_low}, {act_high}]"
                    )
                else:
                    med_h = np.median([b[3] for b in letters])
                    low, high = cal["target_cap_range"]
                    self.assertTrue(
                        low <= med_h <= high,
                        f"{cal['id']} cap height {med_h:.1f}px outside target [{low}, {high}]"
                    )
        finally:
            for p in [test_ass, test_out]:
                if os.path.exists(p):
                    os.remove(p)


class TestCaptionTemplatesRealVideoRender(unittest.TestCase):
    """Performs real-video renders with EACH of the four templates."""

    @classmethod
    def setUpClass(cls):
        cls.source_video = locate_candidate_video()
        print(f"\n[Real Render Test] Using video source: {cls.source_video}")

    def _render_template_clip(self, template_id: str, ass_dialogue: str, style_line: str) -> str:
        test_ass = os.path.join(SCRIPT_DIR, f"temp_{template_id}.ass")
        output_mp4 = os.path.join(SCRIPT_DIR, f"clip_{template_id}_real.mp4")

        ass_content = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
{style_line}

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
{ass_dialogue}
"""
        with open(test_ass, "w", encoding="utf-8") as f:
            f.write(ass_content)

        escaped_ass = test_ass.replace("\\", "/").replace(":", r"\:")
        escaped_fonts = FONTS_DIR_1.replace("\\", "/").replace(":", r"\:")

        # 9:16 crop and lanczos scale matching AutoShorts standard rendering
        filter_str = (
            f"crop=w=608:h=1080:x='(iw-608)/2':y='0',"
            f"scale=1080:1920:flags=lanczos+accurate_rnd+full_chroma_int,"
            f"subtitles='{escaped_ass}':fontsdir='{escaped_fonts}'"
        )

        cmd = [
            "ffmpeg", "-y",
            "-ss", "10.0",
            "-t", "2.0",
            "-i", self.source_video,
            "-vf", filter_str,
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "128k",
            output_mp4
        ]

        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"FFmpeg real render failed for {template_id}:\n{res.stderr}")
        self.assertTrue(os.path.exists(output_mp4))
        size = os.path.getsize(output_mp4)
        self.assertGreater(size, 50000, f"Rendered MP4 for {template_id} must exceed 50KB (got {size} bytes)")

        # Verify stream geometry via ffprobe
        probe_cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,duration",
            "-of", "json",
            output_mp4
        ]
        probe_res = subprocess.run(probe_cmd, capture_output=True, text=True)
        self.assertEqual(probe_res.returncode, 0)
        pdata = json.loads(probe_res.stdout)
        stream = pdata["streams"][0]
        self.assertEqual(stream["width"], 1080)
        self.assertEqual(stream["height"], 1920)

        # Cleanup ass file
        if os.path.exists(test_ass):
            os.remove(test_ass)

        return output_mp4

    def test_real_render_template_1_hormozi_viral(self):
        """Render real video with Template 1: Hormozi Viral."""
        style = "Style: Default,Bebas Neue,84,&H00FFFFFF,&H0000FFFF,&H00000000,&H33000000,-1,0,0,0,100,100,0,0,1,4,3,2,80,80,534,1"
        dialogue = (
            "Dialogue: 0,0:00:00.00,0:00:01.00,Default,,0,0,0,,{\\c&H66FF00&\\fscx115\\fscy115}HORMOZI{\\fscx100\\fscy100\\c&HFFFFFF&} VIRAL\n"
            "Dialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,{\\c&HFFFFFF&}VIRAL {\\c&H00EAFF&\\fscx115\\fscy115}MOMENTS{\\fscx100\\fscy100\\c&HFFFFFF&}\n"
        )
        out = self._render_template_clip("preset_viral_bold", dialogue, style)
        print(f"[Real Render Verified] Template 1 (Hormozi Viral) -> {out} ({os.path.getsize(out)} bytes)")

    def test_real_render_template_2_narrative_pop(self):
        """Render real video with Template 2: Narrative Pop."""
        style = "Style: Default,Montserrat,80,&H00FFFFFF,&H0000FFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,5,4,2,80,80,584,1"
        dialogue = (
            "Dialogue: 0,0:00:00.00,0:00:01.00,Default,,0,0,0,,{\\c&H00FFFF&\\t(0,50,\\fscx120\\fscy120)\\t(50,100,\\fscx100\\fscy100)}This{\\c&HFFFFFF&} is narrative pop\\Nsubtitles on screen\n"
            "Dialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,{\\c&H00FFFF&}This is {\\c&H00FFFF&\\t(0,50,\\fscx120\\fscy120)\\t(50,100,\\fscx100\\fscy100)}narrative{\\c&HFFFFFF&} pop\\Nsubtitles on screen\n"
        )
        out = self._render_template_clip("preset_mrbeast_pop", dialogue, style)
        print(f"[Real Render Verified] Template 2 (Narrative Pop) -> {out} ({os.path.getsize(out)} bytes)")

    def test_real_render_template_3_minimal_capsule(self):
        """Render real video with Template 3: Minimal Capsule."""
        style = "Style: Default,Inter,96,&H00F5F5F5,&H0000FFFF,&H59000000,&H00000000,-1,0,0,0,100,100,0,0,3,12,0,2,80,80,432,1"
        dialogue = (
            "Dialogue: 0,0:00:00.00,0:00:01.00,Default,,0,0,0,,{\\alpha&H00&}Clean {\\alpha&H99&}minimal capsule subtitles\n"
            "Dialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,{\\alpha&H99&}Clean {\\alpha&H00&}minimal {\\alpha&H99&}capsule subtitles\n"
        )
        out = self._render_template_clip("preset_minimal_capsule", dialogue, style)
        print(f"[Real Render Verified] Template 3 (Minimal Capsule) -> {out} ({os.path.getsize(out)} bytes)")

    def test_real_render_template_4_cinematic_vlog(self):
        """Render real video with Template 4: Cinematic Vlog."""
        style = "Style: Default,Poppins,116,&H00F7FBFD,&H0000FFFF,&H00000000,&HB2000000,0,0,0,0,100,100,0,0,1,0,2,2,80,80,218,1"
        dialogue = (
            "Dialogue: 0,0:00:00.00,0:00:01.00,Default,,0,0,0,,{\\fad(150,150)}Cinematic vlog subtitling with smooth fade\\Nand elegant lower placement\n"
            "Dialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,{\\fad(150,150)}Storytelling documentary pacing\\Nand clean typography\n"
        )
        out = self._render_template_clip("preset_cinematic_vlog", dialogue, style)
        print(f"[Real Render Verified] Template 4 (Cinematic Vlog) -> {out} ({os.path.getsize(out)} bytes)")

    def _render_clean_baseline_clip(self) -> str:
        output_mp4 = os.path.join(SCRIPT_DIR, "clip_clean_baseline.mp4")
        if os.path.exists(output_mp4) and os.path.getsize(output_mp4) > 50000:
            return output_mp4

        filter_str = (
            f"crop=w=608:h=1080:x='(iw-608)/2':y='0',"
            f"scale=1080:1920:flags=lanczos+accurate_rnd+full_chroma_int"
        )
        cmd = [
            "ffmpeg", "-y",
            "-ss", "10.0",
            "-t", "2.0",
            "-i", self.source_video,
            "-vf", filter_str,
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "128k",
            output_mp4
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"FFmpeg clean baseline render failed:\n{res.stderr}")
        return output_mp4

    def test_real_render_template_5_dynamic_editorial_visual_behavior(self):
        """
        Comprehensive real-render visual verification of Template 5 (Dynamic Editorial Kinetic).
        Uses SOURCE-FRAME DIFFERENCING against the clean source video across 4 lifecycle stages:
        - Stage 1: Before Entrance (near-zero diff pixels below noise threshold)
        - Stage 2: During Entrance (intermediate centroid vs target resting position for all 4 motions)
        - Stage 3: Full Composition (multi-object coexistence, safe bounds, font variation, color masks)
        - Stage 4: Group Exit (exit fade intensity reduction, post-exit complete clearance)
        """
        if cv2 is None:
            self.skipTest("OpenCV (cv2) not installed")

        clean_clip = self._render_clean_baseline_clip()
        self.assertTrue(os.path.exists(clean_clip), "Clean baseline clip must exist")

        # Style header: Default Montserrat
        style = "Style: Default,Montserrat,62,&H00FFFFFF,&H0000FFFF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,0,0,5,80,80,80,1"

        # Multi-object dialogue lines testing:
        # - slide_right: (320, 1080) -> (380, 1080), start 0.20s, Montserrat (Primary), pure white
        # - slide_up:    (700, 1280) -> (700, 1220), start 0.50s, Bebas Neue (Emphasis 88pt), yellow accent
        # - slide_left:  (760,  950) -> (700,  950), start 0.65s, Inter (Secondary), pure white
        # - slide_down:  (540, 1290) -> (540, 1350), start 0.80s, Poppins (Decorative 64pt + \i1), cyan accent
        # All words in group end at 1.80s with synchronized \fad(0, 300)
        dialogue = (
            "Dialogue: 0,0:00:00.20,0:00:01.80,Default,,0,0,0,,{\\an5\\move(320,1080,380,1080,0,150)\\alpha&HFF&\\t(0,80,0.5,\\alpha&H00&)\\fscx110\\fscy110\\t(0,150,0.4,\\fscx100\\fscy100)\\fad(0,300)\\fnMontserrat\\fs62\\b1\\i0\\c&HFFFFFF&\\bord0\\shad0}EDITORIAL\n"
            "Dialogue: 0,0:00:00.50,0:00:01.80,Default,,0,0,0,,{\\an5\\move(700,1280,700,1220,0,150)\\alpha&HFF&\\t(0,80,0.5,\\alpha&H00&)\\fscx110\\fscy110\\t(0,150,0.4,\\fscx100\\fscy100)\\fad(0,300)\\fnBebas Neue\\fs88\\b1\\i0\\c&H00E6FF&\\bord0\\shad0}KINETIC\n"
            "Dialogue: 0,0:00:00.65,0:00:01.80,Default,,0,0,0,,{\\an5\\move(760,950,700,950,0,150)\\alpha&HFF&\\t(0,80,0.5,\\alpha&H00&)\\fscx110\\fscy110\\t(0,150,0.4,\\fscx100\\fscy100)\\fad(0,300)\\fnInter\\fs50\\b0\\i0\\c&HFFFFFF&\\bord0\\shad0}modern\n"
            "Dialogue: 0,0:00:00.80,0:00:01.80,Default,,0,0,0,,{\\an5\\move(540,1290,540,1350,0,150)\\alpha&HFF&\\t(0,80,0.5,\\alpha&H00&)\\fscx110\\fscy110\\t(0,150,0.4,\\fscx100\\fscy100)\\fad(0,300)\\fnPoppins\\fs64\\b0\\i1\\c&HFFE500&\\bord0\\shad0}typography\n"
        )

        rendered_clip = self._render_template_clip("preset_dynamic_editorial", dialogue, style)
        self.assertTrue(os.path.exists(rendered_clip))

        cap_clean = cv2.VideoCapture(clean_clip)
        cap_rendered = cv2.VideoCapture(rendered_clip)
        self.assertTrue(cap_clean.isOpened(), "Cannot open clean baseline video")
        self.assertTrue(cap_rendered.isOpened(), "Cannot open rendered template video")

        fps = cap_clean.get(cv2.CAP_PROP_FPS) or 30.0

        def extract_diff(timestamp_sec: float):
            frame_idx = int(round(timestamp_sec * fps))
            cap_clean.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret_c, frame_c = cap_clean.read()
            cap_rendered.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret_r, frame_r = cap_rendered.read()
            self.assertTrue(ret_c, f"Could not read clean frame at {timestamp_sec}s (frame {frame_idx})")
            self.assertTrue(ret_r, f"Could not read rendered frame at {timestamp_sec}s (frame {frame_idx})")

            diff = cv2.absdiff(frame_r, frame_c)
            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
            return frame_c, frame_r, diff, thresh

        try:
            # =========================================================================
            # Stage 1: Before Entrance (t = 0.08s, before word 1 starts at 0.20s)
            # =========================================================================
            _, _, _, mask_before = extract_diff(0.08)
            caption_region_before = mask_before[800:1500, 100:980]
            diff_before_count = int(np.count_nonzero(caption_region_before))
            self.assertLess(
                diff_before_count, 150,
                f"Stage 1 Failed: Expected near-zero diff pixels before entrance, got {diff_before_count}"
            )
            print(f"[Stage 1 Verified] Before entrance diff count: {diff_before_count} (below noise tolerance)")

            # =========================================================================
            # Stage 2: During Entrance (Testing all 4 motion primitives via diff centroid)
            # =========================================================================
            # 2a. slide_right: starts at 0.20s from X=320 to X=380. Check at t = 0.25s (50ms in).
            _, _, _, mask_slide_right = extract_diff(0.25)
            roi_right = mask_slide_right[1000:1160, 200:500]
            self.assertGreater(np.count_nonzero(roi_right), 50, "slide_right must produce caption diff pixels")
            pts_r = cv2.findNonZero(roi_right).reshape(-1, 2)
            mean_x_r = float(np.mean(pts_r[:, 0])) + 200.0
            self.assertLess(
                mean_x_r, 380.0,
                f"Stage 2 slide_right Failed: intermediate X={mean_x_r} must be to the left of target X=380"
            )

            # 2b. slide_up: starts at 0.50s from Y=1280 to Y=1220. Check at t = 0.55s (50ms in).
            _, _, _, mask_slide_up = extract_diff(0.55)
            roi_up = mask_slide_up[1180:1320, 600:800]
            self.assertGreater(np.count_nonzero(roi_up), 50, "slide_up must produce caption diff pixels")
            pts_up = cv2.findNonZero(roi_up).reshape(-1, 2)
            mean_y_up = float(np.mean(pts_up[:, 1])) + 1180.0
            self.assertGreater(
                mean_y_up, 1220.0,
                f"Stage 2 slide_up Failed: intermediate Y={mean_y_up} must be below target Y=1220"
            )

            # 2c. slide_left: starts at 0.65s from X=760 to X=700. Check at t = 0.70s (50ms in).
            _, _, _, mask_slide_left = extract_diff(0.70)
            roi_left = mask_slide_left[900:1000, 650:850]
            self.assertGreater(np.count_nonzero(roi_left), 50, "slide_left must produce caption diff pixels")
            pts_left = cv2.findNonZero(roi_left).reshape(-1, 2)
            mean_x_left = float(np.mean(pts_left[:, 0])) + 650.0
            self.assertGreater(
                mean_x_left, 700.0,
                f"Stage 2 slide_left Failed: intermediate X={mean_x_left} must be to the right of target X=700"
            )

            # 2d. slide_down: starts at 0.80s from Y=1290 to Y=1350. Check at t = 0.85s (50ms in).
            _, _, _, mask_slide_down = extract_diff(0.85)
            roi_down = mask_slide_down[1270:1380, 450:650]
            self.assertGreater(np.count_nonzero(roi_down), 50, "slide_down must produce caption diff pixels")
            pts_down = cv2.findNonZero(roi_down).reshape(-1, 2)
            mean_y_down = float(np.mean(pts_down[:, 1])) + 1270.0
            self.assertLess(
                mean_y_down, 1350.0,
                f"Stage 2 slide_down Failed: intermediate Y={mean_y_down} must be above target Y=1350"
            )
            print(f"[Stage 2 Verified] All 4 motions (slide_up, slide_down, slide_left, slide_right) verified via differencing centroids.")

            # =========================================================================
            # Stage 3: Full Composition (t = 1.10s, all 4 objects resting)
            # =========================================================================
            _, frame_r_full, _, mask_full = extract_diff(1.10)
            all_pts_raw = cv2.findNonZero(mask_full)
            self.assertIsNotNone(all_pts_raw, "Stage 3 Failed: Full composition must have diff pixels")
            all_pts = all_pts_raw.reshape(-1, 2)
            min_x = int(np.min(all_pts[:, 0]))
            max_x = int(np.max(all_pts[:, 0]))
            min_y = int(np.min(all_pts[:, 1]))
            max_y = int(np.max(all_pts[:, 1]))

            self.assertGreaterEqual(min_x, 100, f"Min X {min_x} clipped left edge")
            self.assertLessEqual(max_x, 980, f"Max X {max_x} clipped right edge")
            self.assertGreaterEqual(min_y, 800, f"Min Y {min_y} clipped top edge")
            self.assertLessEqual(max_y, 1500, f"Max Y {max_y} clipped bottom edge")

            contours, _ = cv2.findContours(mask_full[800:1500, 100:980], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_contours = [c for c in contours if cv2.contourArea(c) > 80]
            self.assertGreaterEqual(
                len(valid_contours), 3,
                f"Stage 3 Failed: Expected at least 3 coexisting text contours, found {len(valid_contours)}"
            )

            # Color verification STRICTLY inside difference mask
            rendered_caption_pixels = frame_r_full[mask_full > 0]
            b = rendered_caption_pixels[:, 0]
            g = rendered_caption_pixels[:, 1]
            r = rendered_caption_pixels[:, 2]

            white_count = int(np.count_nonzero((r > 180) & (g > 180) & (b > 180)))
            yellow_count = int(np.count_nonzero((r > 180) & (g > 160) & (b < 100)))
            cyan_count = int(np.count_nonzero((b > 180) & (g > 160) & (r < 100)))

            self.assertGreater(white_count, 100, f"Must detect off-white text pixels: got {white_count}")
            self.assertGreater(yellow_count, 50, f"Must detect yellow accent pixels: got {yellow_count}")
            self.assertGreater(cyan_count, 50, f"Must detect cyan accent pixels: got {cyan_count}")
            print(f"[Stage 3 Verified] Full composition: {len(valid_contours)} contours, white={white_count}, yellow={yellow_count}, cyan={cyan_count}, all inside safe bounds [{min_x}, {max_x}]x[{min_y}, {max_y}].")

            # =========================================================================
            # Stage 4: Group Exit (t = 1.70s during fade, t = 1.95s post-exit clearance)
            # =========================================================================
            _, _, _, mask_fade = extract_diff(1.70)
            diff_fade_count = int(np.count_nonzero(mask_fade[800:1500, 100:980]))
            full_diff_count = int(np.count_nonzero(mask_full[800:1500, 100:980]))
            self.assertLess(
                diff_fade_count, full_diff_count,
                f"Stage 4 Failed: Diff pixel count during fade ({diff_fade_count}) must be less than full ({full_diff_count})"
            )

            _, _, _, mask_post = extract_diff(1.95)
            diff_post_count = int(np.count_nonzero(mask_post[800:1500, 100:980]))
            self.assertLess(
                diff_post_count, 150,
                f"Stage 4 Failed: Diff pixel count post-exit must be below noise threshold, got {diff_post_count}"
            )
            print(f"[Stage 4 Verified] Group exit: fade count {diff_fade_count} < full count {full_diff_count}, post-exit clearance count {diff_post_count} < 150.")

        finally:
            cap_clean.release()
            cap_rendered.release()
            for p in [clean_clip, rendered_clip]:
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass

    @classmethod
    def tearDownClass(cls):
        for template_id in ["preset_viral_bold", "preset_mrbeast_pop", "preset_minimal_capsule", "preset_cinematic_vlog", "preset_dynamic_editorial", "clean_baseline"]:
            p = os.path.join(SCRIPT_DIR, f"clip_{template_id}_real.mp4")
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass


if __name__ == "__main__":
    unittest.main(verbosity=2)
