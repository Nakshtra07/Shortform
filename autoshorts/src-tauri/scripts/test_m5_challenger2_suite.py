"""Milestone 5 Challenger 2 Test Suite: Subtitle & TitleArt Stress Challenger.

Adversarially stress-tests:
1. caption_layout.py:
   - CJK (1.0) and Latin (0.55) character width calculation.
   - split_screens invariant: NEVER exceeds 2 lines per screen.
   - Continuous strings without whitespace, mixed CJK/Latin, long punctuation sequences.
   - Boundary limits, edge cases, and property-based random stress.
2. title_art.py:
   - TitleArt generation for all styles (comic, neon, arena, editorial, pixel, frosted).
   - Valid PNG headers, dimensions, mode, and multi-line CJK/Latin rendering.
   - Error handling, dimension boundaries, and motion overlays.
3. autoclip_adapter.py:
   - Word grouping clause splitting on punctuation (Latin & CJK).
   - Silence gaps >= 0.4s (including exact boundary 0.399 vs 0.400).
   - Speaker transitions.
   - Max 8 words cap and scene boundary filtering.
"""
from __future__ import annotations

import io
import os
import random
import re
import sys
import unittest
from pathlib import Path

from PIL import Image

# Ensure scripts dir is on sys.path
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from autoclip_studio.caption_layout import (
    line_count,
    lines_for,
    split_screens,
    timed_screens,
    width,
    _cut,
    wrap_two,
    _fit_two,
)
from autoclip_studio.title_art import (
    STYLES as TITLE_STYLES,
    artwork,
    overlay_motion,
    png_bytes,
)
import autoclip_adapter
from autoclip_adapter import (
    build_draft_and_scene,
    generate_srt_string,
    group_words_into_cues,
    has_clause_punctuation,
    is_cjk,
    join_words_text,
)


class TestCaptionLayoutWidth(unittest.TestCase):
    """Adversarially verify CJK and Latin width calculations."""

    def test_latin_character_widths(self):
        """Latin letters, numbers, spaces, and ASCII symbols must have width 0.55."""
        latin_samples = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 `~!@#$%^&*()-_=+[{]}\\|;:'\",<.>/?"
        for ch in latin_samples:
            w = width(ch)
            self.assertAlmostEqual(
                w,
                0.55,
                places=5,
                msg=f"Character '{ch}' (U+{ord(ch):04X}) expected width 0.55, got {w}",
            )

    def test_cjk_character_widths(self):
        """Standard CJK ideographs and fullwidth punctuation within CJK regex must have width 1.0."""
        cjk_samples = (
            "你好世界人工智能自然语言处理深度学习视频剪辑"
            "，。！？；：、（）《》【】"
            "\u3000"  # Ideographic space
        )
        for ch in cjk_samples:
            w = width(ch)
            self.assertAlmostEqual(
                w,
                1.0,
                places=5,
                msg=f"Character '{ch}' (U+{ord(ch):04X}) expected width 1.0, got {w}",
            )

    def test_cjk_general_punctuation_width_empirical_behavior(self):
        """Empirically verify that characters outside [　-〿㐀-鿿＀-￯] receive width 0.55.

        In particular, Chinese curly quotation marks (U+201C, U+201D, U+2018, U+2019),
        horizontal ellipsis (U+2026), and Japanese kana (U+3042, U+30A2) evaluate to 0.55
        due to the exact regex range in caption_layout.py.
        """
        quirk_chars = {
            "“": 0.55,  # U+201C
            "”": 0.55,  # U+201D
            "‘": 0.55,  # U+2018
            "’": 0.55,  # U+2019
            "…": 0.55,  # U+2026
            "—": 0.55,  # U+2014
            "あ": 0.55,  # U+3042 Hiragana
            "ア": 0.55,  # U+30A2 Katakana
        }
        for ch, expected_w in quirk_chars.items():
            self.assertAlmostEqual(
                width(ch),
                expected_w,
                places=5,
                msg=f"Character '{ch}' (U+{ord(ch):04X}) expected width {expected_w}",
            )

    def test_mixed_string_width_exact_calculation(self):
        """Mixed CJK and Latin strings must compute exact weighted sum."""
        # 10 Latin chars ("AutoShorts ") + 4 CJK chars ("自动剪辑")
        # 10 * 0.55 + 4 * 1.0 = 5.5 + 4.0 = 9.5
        text = "AutoShorts 自动剪辑"
        expected = 11 * 0.55 + 4 * 1.0  # 10 letters + 1 space = 11 Latin chars
        self.assertAlmostEqual(width(text), expected, places=5)

        # Latin: 6 chars ("v1.0.0") = 6 * 0.55 = 3.3; CJK: 4 chars ("正式发布") = 4 * 1.0 = 4.0
        # Total = 7.3
        text2 = "v1.0.0正式发布"
        expected2 = 6 * 0.55 + 4 * 1.0
        self.assertAlmostEqual(width(text2), expected2, places=5)

    def test_empty_string_width(self):
        self.assertEqual(width(""), 0.0)


class TestCaptionLayoutSplitScreens(unittest.TestCase):
    """Adversarially stress-test split_screens to guarantee lines per screen <= 2."""

    def _assert_never_exceeds_two_lines(self, screens: list[str], context: str = ""):
        """Oracle assertion: every screen MUST contain <= 2 lines."""
        for idx, screen in enumerate(screens):
            n_lines = line_count(screen)
            self.assertLessEqual(
                n_lines,
                2,
                msg=f"VIOLATION: Screen {idx} has {n_lines} lines (exceeds 2) in context '{context}'. Screen content: {repr(screen)}",
            )
            # Verbatim check on hard line-break count
            hard_breaks = screen.count("\\N")
            self.assertLessEqual(
                hard_breaks,
                1,
                msg=f"VIOLATION: Screen {idx} contains {hard_breaks} hard breaks (max 1 permitted). Screen content: {repr(screen)}",
            )

    def test_split_screens_standard_sentences(self):
        text = "这是一段用于测试普通文本分割的字幕，它应该被优雅地拆分成多个屏幕，并且每个屏幕至多包含两行字幕。"
        for limit in [8.0, 10.0, 12.0, 15.0, 20.0]:
            screens = split_screens(text, limit)
            self.assertGreater(len(screens), 0)
            self._assert_never_exceeds_two_lines(screens, f"standard cjk limit={limit}")

    def test_split_screens_continuous_strings_without_whitespace_latin(self):
        """Continuous long Latin string without any whitespace."""
        text = "SupercalifragilisticexpialidociousEvenThoughTheSoundOfItIsSomethingQuiteAtrocious" * 3
        for limit in [5.0, 10.0, 15.0, 25.0]:
            screens = split_screens(text, limit)
            self.assertGreater(len(screens), 0)
            self._assert_never_exceeds_two_lines(screens, f"continuous latin limit={limit}")

    def test_split_screens_continuous_strings_without_whitespace_cjk(self):
        """Continuous long CJK string without any whitespace or punctuation."""
        text = "这是一段完全没有标点符号也没有空格的超级长中文测试文本用来验证断句切割算法是否会在达到行宽限制时安全且确定地执行截断而绝不溢出两行" * 3
        for limit in [6.0, 8.0, 12.0, 16.0]:
            screens = split_screens(text, limit)
            self.assertGreater(len(screens), 0)
            self._assert_never_exceeds_two_lines(screens, f"continuous cjk limit={limit}")

    def test_split_screens_mixed_cjk_latin_no_whitespace(self):
        """Mixed CJK and Latin without whitespace."""
        text = "AutoShorts11PresentationMode2DouyinTikTokBilibiliXiaohongshu全平台多端架构适配验证" * 3
        for limit in [5.0, 10.0, 15.0]:
            screens = split_screens(text, limit)
            self.assertGreater(len(screens), 0)
            self._assert_never_exceeds_two_lines(screens, f"mixed continuous limit={limit}")

    def test_split_screens_long_punctuation_sequences(self):
        """Extremely long repetitive punctuation sequences."""
        punctuation_cases = [
            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!",
            "..................................................",
            "？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？",
            "，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，",
            "!?.!?.!?.!?.!?.!?.!?.!?.!?.!?.!?.!?.!?.!?.!?.!?.",
            "……，，，。。。？？？！！！；；；：：：、、、、",
            "Hello!!!!!! World?????? Yes...... No,,,,,, Ok;;;;;;",
            "你好！！！！世界？？？？真的……太棒了，，，，完美。。。。",
        ]
        for text in punctuation_cases:
            for limit in [5.0, 8.0, 12.0, 20.0]:
                screens = split_screens(text, limit)
                self.assertGreater(len(screens), 0)
                self._assert_never_exceeds_two_lines(screens, f"punctuation '{text[:10]}' limit={limit}")

    def test_split_screens_extreme_limits(self):
        """Sub-character and huge line limits."""
        text = "AutoShorts 极速切片测试"
        # Sub-character limit: smaller than single CJK (1.0) and Latin (0.55)
        for tiny_limit in [0.1, 0.5, 0.9]:
            screens = split_screens(text, tiny_limit)
            self.assertGreater(len(screens), 0)
            self._assert_never_exceeds_two_lines(screens, f"tiny limit={tiny_limit}")

        # Huge limit
        huge_screens = split_screens(text, 1000.0)
        self.assertEqual(len(huge_screens), 1)
        self._assert_never_exceeds_two_lines(huge_screens, "huge limit=1000.0")

    def test_split_screens_empty_and_whitespace_inputs(self):
        """Empty and whitespace inputs should return empty screen lists."""
        self.assertEqual(split_screens("", 10.0), [])
        self.assertEqual(split_screens("   ", 10.0), [])
        self.assertEqual(split_screens("\n\t\r", 10.0), [])

    def test_split_screens_fuzz_property_based(self):
        """Randomized property-based testing across 100 generated adversarial inputs."""
        cjk_vocab = ["我", "你", "他", "剪辑", "模式", "短视频", "测试", "爆款", "发布", "的", "了", "和", "在"]
        latin_vocab = ["AutoShorts", "Mode", "AI", "Video", "Super", "Fast", "Render", "Clip", "11", "Tauri"]
        punct_vocab = [",", ".", "!", "?", "，", "。", "！", "？", "；", " ", ""]

        rng = random.Random(42)
        for i in range(100):
            # Synthesize random sentence
            tokens = []
            for _ in range(rng.randint(3, 25)):
                choice = rng.random()
                if choice < 0.4:
                    tokens.append(rng.choice(cjk_vocab))
                elif choice < 0.8:
                    tokens.append(rng.choice(latin_vocab))
                else:
                    tokens.append(rng.choice(punct_vocab))
            synthetic_text = "".join(tokens)
            limit = rng.uniform(2.0, 25.0)

            screens = split_screens(synthetic_text, limit)
            if synthetic_text.strip():
                self.assertGreater(len(screens), 0)
            self._assert_never_exceeds_two_lines(screens, f"fuzz iter={i} limit={limit:.2f}")

    def test_timed_screens_invariants(self):
        """Verify timed_screens produces valid spans and <= 2 lines per screen."""
        text = "AutoShorts 11.0 提供了全新的 AutoClip Studio 演示模式，支持抖音和TikTok双重视觉风格。"
        start_t = 10.0
        end_t = 18.0
        res = timed_screens(text, start=start_t, end=end_t, line_limit=10.0)
        self.assertGreater(len(res), 0)
        self.assertAlmostEqual(res[0][0], start_t, places=4)
        self.assertAlmostEqual(res[-1][1], end_t, places=4)

        for idx, (st, en, scr, orig) in enumerate(res):
            self.assertLessEqual(st, en)
            self.assertLessEqual(line_count(scr), 2)
            self.assertLessEqual(scr.count("\\N"), 1)


class TestTitleArtGeneration(unittest.TestCase):
    """Adversarially stress-test TitleArt generation across all styles and versions."""

    def test_titleart_all_six_styles_generate_valid_png(self):
        """Generate TitleArt for all 6 styles: comic, neon, arena, editorial, pixel, frosted."""
        styles = sorted(TITLE_STYLES)
        self.assertEqual(
            set(styles),
            {"comic", "neon", "arena", "editorial", "pixel", "frosted"},
            "Must support exactly the 6 required styles",
        )

        sample_text = "VIRAL HOOK"
        w, h = 1080, 1920

        for style in styles:
            raw = png_bytes(sample_text, style, w, h, version=6)
            self.assertIsInstance(raw, bytes)
            self.assertGreater(len(raw), 500, f"PNG bytes too small for style {style}")
            self.assertTrue(
                raw.startswith(b"\x89PNG\r\n\x1a\n"),
                f"Style {style} output missing PNG magic header",
            )

            # Load into PIL and verify dimensions and mode
            img = Image.open(io.BytesIO(raw))
            self.assertEqual(img.size, (w, h), f"Style {style} produced incorrect size")
            self.assertEqual(img.mode, "RGBA", f"Style {style} produced mode {img.mode}, expected RGBA")

    def test_titleart_multiline_and_cjk_all_styles(self):
        """Verify all 6 styles handle multi-line and Chinese characters."""
        cjk_text = "百万播放\n爆款短视频"
        w, h = 1080, 1920

        for style in sorted(TITLE_STYLES):
            raw = png_bytes(cjk_text, style, w, h, version=6)
            self.assertTrue(raw.startswith(b"\x89PNG\r\n\x1a\n"))
            img = Image.open(io.BytesIO(raw))
            self.assertEqual(img.size, (w, h))

    def test_titleart_multiple_resolutions(self):
        """Verify rendering at 720x1280, 1080x1080, and 1920x1080."""
        resolutions = [(720, 1280), (1080, 1080), (1920, 1080)]
        for w, h in resolutions:
            for style in ["comic", "neon", "editorial", "pixel", "frosted"]:
                raw = png_bytes("TEST DIMS", style, w, h, version=6)
                img = Image.open(io.BytesIO(raw))
                self.assertEqual(img.size, (w, h), f"Failed for {w}x{h} with style {style}")

    def test_titleart_empty_text_returns_transparent_canvas(self):
        """Empty or whitespace text returns transparent image without error."""
        for empty_text in ["", "   ", "\n"]:
            for style in sorted(TITLE_STYLES):
                raw = png_bytes(empty_text, style, 1080, 1920, version=6)
                img = Image.open(io.BytesIO(raw))
                self.assertEqual(img.size, (1080, 1920))
                # Image should be fully transparent (empty canvas)
                bbox = img.getbbox()
                self.assertIsNone(bbox, f"Empty text for style {style} should have None bbox, got {bbox}")

    def test_titleart_error_conditions(self):
        """Invalid styles and extreme canvas dimensions must raise ValueError."""
        # Invalid style
        with self.assertRaises(ValueError):
            png_bytes("TEST", "unknown_style_xyz", 1080, 1920, version=6)

        # Dimensions too small (<16)
        with self.assertRaises(ValueError):
            png_bytes("TEST", "comic", 15, 1080, version=6)
        with self.assertRaises(ValueError):
            png_bytes("TEST", "comic", 1080, 15, version=6)

        # Dimensions exceed 16MP
        with self.assertRaises(ValueError):
            png_bytes("TEST", "comic", 5000, 5000, version=6)

        # Excessive text that cannot fit within 3 lines
        excessive = "一二三四五六七八九十 " * 40
        with self.assertRaises(ValueError):
            png_bytes(excessive, "comic", 1080, 1920, version=6)

    def test_titleart_overlay_motion(self):
        """Verify motion overlay expressions for all styles."""
        # Disabled returns ('0', '0')
        self.assertEqual(overlay_motion("comic", False, 1920), ("0", "0"))
        # Pixel and frosted do not move
        self.assertEqual(overlay_motion("pixel", True, 1920), ("0", "0"))
        self.assertEqual(overlay_motion("frosted", True, 1920), ("0", "0"))
        # Arena has horizontal slide
        arena_x, arena_y = overlay_motion("arena", True, 1920)
        self.assertIn("-36", arena_x)
        self.assertEqual(arena_y, "0")
        # Comic has vertical entrance
        comic_x, comic_y = overlay_motion("comic", True, 1920)
        self.assertEqual(comic_x, "0")
        self.assertIn("max(0,1-t/0.18)", comic_y)


class TestWordGroupingAdapter(unittest.TestCase):
    """Adversarially stress-test word grouping and clause segmentation in autoclip_adapter.py."""

    def test_clause_splitting_on_latin_punctuation(self):
        """Splits on Latin punctuation marks: . , ! ? ;"""
        words = [
            {"text": "First,", "start": 1.0, "end": 1.2, "speaker": "A"},
            {"text": "second.", "start": 1.25, "end": 1.45, "speaker": "A"},
            {"text": "Third!", "start": 1.5, "end": 1.7, "speaker": "A"},
            {"text": "Fourth?", "start": 1.75, "end": 1.95, "speaker": "A"},
            {"text": "Fifth;", "start": 2.0, "end": 2.2, "speaker": "A"},
            {"text": "Sixth", "start": 2.25, "end": 2.45, "speaker": "A"},
        ]
        cues = group_words_into_cues(words)
        self.assertEqual(len(cues), 6)
        self.assertEqual(cues[0]["text"], "First,")
        self.assertEqual(cues[1]["text"], "second.")
        self.assertEqual(cues[2]["text"], "Third!")
        self.assertEqual(cues[3]["text"], "Fourth?")
        self.assertEqual(cues[4]["text"], "Fifth;")
        self.assertEqual(cues[5]["text"], "Sixth")

    def test_clause_splitting_on_cjk_punctuation(self):
        """Splits on CJK punctuation marks: ， 。 ！ ？ ；"""
        words = [
            {"text": "首先，", "start": 1.0, "end": 1.3, "speaker": "A"},
            {"text": "第一步完成。", "start": 1.35, "end": 1.8, "speaker": "A"},
            {"text": "太棒了！", "start": 1.85, "end": 2.2, "speaker": "A"},
            {"text": "你觉得呢？", "start": 2.25, "end": 2.6, "speaker": "A"},
            {"text": "结论如下；", "start": 2.65, "end": 3.0, "speaker": "A"},
            {"text": "继续前进", "start": 3.05, "end": 3.4, "speaker": "A"},
        ]
        cues = group_words_into_cues(words)
        self.assertEqual(len(cues), 6)
        self.assertEqual(cues[0]["text"], "首先，")
        self.assertEqual(cues[1]["text"], "第一步完成。")
        self.assertEqual(cues[2]["text"], "太棒了！")
        self.assertEqual(cues[3]["text"], "你觉得呢？")
        self.assertEqual(cues[4]["text"], "结论如下；")
        self.assertEqual(cues[5]["text"], "继续前进")

    def test_clause_splitting_silence_gap(self):
        """Verify silence gap splitting behavior.

        - Gap of 0.1s (< 0.4) MUST NOT split.
        - Gap of 0.45s (>= 0.4) MUST split.
        - Exact float subtraction: 1.0 - 0.6 evaluates to 0.4 in float, which splits.
        - Precision quirk: 1.9 - 1.5 evaluates to 0.3999999999999999 (< 0.4) in IEEE 754,
          empirically demonstrating why floating point rounding is relevant.
        """
        # Case 1: Gap = 0.1s (no split)
        words_no_split = [
            {"text": "WordA", "start": 1.000, "end": 1.500, "speaker": "A"},
            {"text": "WordB", "start": 1.600, "end": 2.000, "speaker": "A"},
        ]
        cues1 = group_words_into_cues(words_no_split)
        self.assertEqual(len(cues1), 1, "Gap of 0.1s should NOT split clause")
        self.assertEqual(cues1[0]["text"], "WordA WordB")

        # Case 2: Gap = 0.45s (split)
        words_split = [
            {"text": "WordA", "start": 1.000, "end": 1.500, "speaker": "A"},
            {"text": "WordB", "start": 1.950, "end": 2.200, "speaker": "A"},
        ]
        cues2 = group_words_into_cues(words_split)
        self.assertEqual(len(cues2), 2, "Gap of 0.45s MUST split clause")
        self.assertEqual(cues2[0]["text"], "WordA")
        self.assertEqual(cues2[1]["text"], "WordB")

        # Case 3: Exact float representation where (1.0 - 0.6) == 0.4
        words_exact_float = [
            {"text": "WordA", "start": 0.200, "end": 0.600, "speaker": "A"},
            {"text": "WordB", "start": 1.000, "end": 1.400, "speaker": "A"},
        ]
        cues_exact = group_words_into_cues(words_exact_float)
        self.assertEqual(len(cues_exact), 2, "Exact float difference of 0.4 MUST split clause")

        # Case 4: Empirical IEEE 754 float quirk: 1.9 - 1.5 = 0.3999999999999999
        words_float_quirk = [
            {"text": "WordA", "start": 1.000, "end": 1.500, "speaker": "A"},
            {"text": "WordB", "start": 1.900, "end": 2.200, "speaker": "A"},
        ]
        cues_quirk = group_words_into_cues(words_float_quirk)
        # Because (1.9 - 1.5) is strictly < 0.4 in raw float arithmetic, it does not split without rounding
        self.assertEqual(len(cues_quirk), 1, "Raw float (1.9 - 1.5) is 0.3999999999999999 (<0.4), so does not split")

        # Case 5: Gap = 1.5s (long pause, split)
        words_long_pause = [
            {"text": "BeforePause", "start": 1.0, "end": 1.5, "speaker": "A"},
            {"text": "AfterPause", "start": 3.0, "end": 3.5, "speaker": "A"},
        ]
        cues3 = group_words_into_cues(words_long_pause)
        self.assertEqual(len(cues3), 2)

    def test_clause_splitting_on_speaker_change(self):
        """Speaker changes must split clauses immediately, even with 0.0s gap."""
        words = [
            {"text": "SpeakerOne", "start": 1.0, "end": 1.5, "speaker": "Host"},
            {"text": "SpeakerTwo", "start": 1.5, "end": 2.0, "speaker": "Guest"},
        ]
        cues = group_words_into_cues(words)
        self.assertEqual(len(cues), 2, "Speaker change MUST split into separate cues")
        self.assertEqual(cues[0]["speaker"], "Host")
        self.assertEqual(cues[0]["text"], "SpeakerOne")
        self.assertEqual(cues[1]["speaker"], "Guest")
        self.assertEqual(cues[1]["text"], "SpeakerTwo")

    def test_max_eight_words_clause_cap(self):
        """Maximum 8 words per clause when pauses are small (< 0.4s)."""
        # Gap = 0.02s between words (< 0.4s), same speaker, no punctuation
        words_10 = [
            {"text": f"word{i}", "start": float(i) * 0.1, "end": float(i) * 0.1 + 0.08, "speaker": "A"}
            for i in range(1, 11)
        ]
        cues = group_words_into_cues(words_10)
        self.assertEqual(len(cues), 2)
        # First cue has 8 words
        self.assertEqual(len(cues[0]["text"].split()), 8)
        self.assertEqual(cues[0]["text"], "word1 word2 word3 word4 word5 word6 word7 word8")
        # Second cue has remaining 2 words
        self.assertEqual(len(cues[1]["text"].split()), 2)
        self.assertEqual(cues[1]["text"], "word9 word10")

    def test_scene_boundary_filtering(self):
        """Words strictly outside [scene_start, scene_end] are filtered out."""
        words = [
            {"text": "BeforeScene", "start": 0.0, "end": 4.9, "speaker": "A"},
            {"text": "InsideScene1", "start": 5.0, "end": 6.0, "speaker": "A"},
            {"text": "InsideScene2", "start": 6.1, "end": 7.0, "speaker": "A"},
            {"text": "AfterScene", "start": 10.1, "end": 11.0, "speaker": "A"},
        ]
        cues = group_words_into_cues(words, scene_start=5.0, scene_end=10.0)
        self.assertEqual(len(cues), 1)
        self.assertEqual(cues[0]["text"], "InsideScene1 InsideScene2")

    def test_adversarial_corrupt_word_data(self):
        """Gracefully handle inverted timestamps, negative durations, and missing keys."""
        words = [
            None,
            "not_a_dict",
            {"text": "", "start": 1.0, "end": 1.5},
            {"text": "   ", "start": 1.5, "end": 2.0},
            {"text": "Inverted", "start": 5.0, "end": 4.0},  # end < start
            {"text": "Normal", "start": 5.2, "end": 5.5},
            {"text": "UnsortedEarlier", "start": 0.5, "end": 0.9},
        ]
        cues = group_words_into_cues(words)
        self.assertGreater(len(cues), 0)
        # Check that UnsortedEarlier came first
        self.assertTrue(cues[0]["text"].startswith("UnsortedEarlier"))

    def test_generate_srt_string_format(self):
        """Verify standard SRT format."""
        cues = [
            {"start": 1.250, "end": 3.750, "text": "Hello world"},
            {"start": 4.000, "end": 5.500, "text": "Goodbye world"},
        ]
        srt = generate_srt_string(cues)
        self.assertIn("1\n00:00:01,250 --> 00:00:03,750\nHello world", srt)
        self.assertIn("2\n00:00:04,000 --> 00:00:05,500\nGoodbye world", srt)


class TestAdapterDraftSceneConstruction(unittest.TestCase):
    """Verify build_draft_and_scene enforces TitleArt version 6 and offline invariants."""

    def test_draft_construction_with_title_styles(self):
        for style in ["comic", "neon", "arena", "editorial", "pixel", "frosted"]:
            payload = {
                "project_id": "test_proj_m5",
                "video_path": "dummy.mp4",
                "output_path": "out.mp4",
                "candidate": {
                    "id": "cand_01",
                    "start_sec": 10.0,
                    "end_sec": 25.0,
                    "hook": "Top Secret Headline",
                },
                "presentation_config": {
                    "platform_strategy": "douyin",
                    "title_style": style,
                    "title_scale": 1.1,
                    "title_y": 0.15,
                },
            }
            cues = [{"start": 10.0, "end": 12.0, "text": "Top Secret Headline"}]
            draft, scene, strat = build_draft_and_scene(payload, cues)

            self.assertEqual(draft.title_style, style)
            self.assertEqual(
                draft.title_template_version,
                6,
                f"Style {style} must use title_template_version=6",
            )
            self.assertEqual(draft.language, "source", "Must enforce offline language='source'")
            self.assertAlmostEqual(draft.title_scale, 1.1)
            self.assertAlmostEqual(draft.title_y, 0.15)


if __name__ == "__main__":
    unittest.main()
