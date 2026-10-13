"""Adversarial stress-testing suite for Milestone 1: Presentation Engine Isolation & Asset Staging.

Empirical verification of:
1. framing.window_fraction & framing.crop_x_for_center across extreme & edge aspect ratios.
2. caption_layout.py typography width, clause splitting, CJK/Latin wrapping, and invariant:
   split_screens NEVER outputs more than 2 lines per screen.
3. framing.segment_track hold segment dampening (min_hold=2) and resistance to single-glance spikes.
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path
import pytest

# Ensure scripts and autoclip_studio are importable
SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "autoshorts" / "src-tauri" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from autoclip_studio import caption_layout as cl
from autoclip_studio import framing


# ============================================================================
# 1. FRAMING MATH ADVERSARIAL STRESS TESTS
# ============================================================================

class TestFramingMathAdversarial:
    """Stress tests for framing.window_fraction and framing.crop_x_for_center."""

    @pytest.mark.parametrize("src_w, src_h, out_w, out_h, expected_fraction", [
        # Standard 16:9 -> 9:16 portrait
        (1920, 1080, 1080, 1920, (1080 / 1920) / (1920 / 1080)),  # 81/256 = 0.31640625
        (1280, 720, 1080, 1920, (1080 / 1920) / (1280 / 720)),   # 0.31640625
        # 9:16 -> 9:16 (identical aspect)
        (1080, 1920, 1080, 1920, 1.0),
        (720, 1280, 1080, 1920, 1.0),
        # 1:1 square -> 9:16 portrait
        (1080, 1080, 1080, 1920, (1080 / 1920) / 1.0),            # 9/16 = 0.5625
        (720, 720, 1080, 1920, 0.5625),
        # Ultra-wide 21:9 -> 9:16 portrait
        (2560, 1080, 1080, 1920, (1080 / 1920) / (2560 / 1080)),  # 243/1024 = 0.2373046875
        (3440, 1440, 1080, 1920, (1080 / 1920) / (3440 / 1440)),  # ~0.235465
        # Super ultra-wide 32:9 -> 9:16 portrait
        (5120, 1440, 1080, 1920, (1080 / 1920) / (5120 / 1440)),  # 81/512 = 0.158203125
        # Vertical tall 9:21 -> 9:16 portrait (source is narrower than target)
        (1080, 2520, 1080, 1920, 1.0),                             # AR_src = 9/21 < 9/16 -> covers height, fills width -> 1.0
        # Ultra-tall vertical 9:32 -> 9:16 portrait
        (1080, 3840, 1080, 1920, 1.0),
        # Landscape 16:9 output with various sources
        (1920, 1080, 1920, 1080, 1.0),                             # 16:9 -> 16:9
        (1080, 1920, 1920, 1080, 1.0),                             # 9:16 -> 16:9 (narrower -> 1.0)
        (1080, 1080, 1920, 1080, 1.0),                             # 1:1 -> 16:9 (narrower -> 1.0)
        (2560, 1080, 1920, 1080, (1920 / 1080) / (2560 / 1080)),  # 21:9 -> 16:9 = 1920/2560 = 0.75
        (5120, 1440, 1920, 1080, (1920 / 1080) / (5120 / 1440)),  # 32:9 -> 16:9 = 0.5
        # Extreme aspect ratios
        (10000, 100, 1080, 1920, (1080 / 1920) / (10000 / 100)),  # 100:1 source -> 0.005625
        (100, 10000, 1080, 1920, 1.0),                             # 1:100 source -> 1.0
        # Tiny resolution edge cases
        (16, 9, 1080, 1920, 0.31640625),
        (9, 16, 1080, 1920, 1.0),
        (1, 1, 1080, 1920, 0.5625),
    ])
    def test_window_fraction_exact_values(self, src_w, src_h, out_w, out_h, expected_fraction):
        frac = framing.window_fraction(src_w, src_h, out_w, out_h)
        assert frac == pytest.approx(expected_fraction, rel=1e-5)
        assert 0.0 < frac <= 1.0

    def test_window_fraction_exhaustive_sweep_invariants(self):
        """Invariant: For any positive dimensions, window_fraction is always in (0, 1.0]."""
        dimensions = [1, 2, 9, 16, 100, 480, 720, 1080, 1440, 1920, 2560, 3840, 7680]
        for src_w in [16, 720, 1080, 1920, 3840]:
            for src_h in [9, 720, 1080, 1920, 2160]:
                for out_w in [720, 1080, 1920]:
                    for out_h in [720, 1080, 1920]:
                        f = framing.window_fraction(src_w, src_h, out_w, out_h)
                        assert isinstance(f, float)
                        assert not math.isnan(f)
                        assert not math.isinf(f)
                        assert 0.0 < f <= 1.0
                        # Mathematical relation:
                        # scale = max(out_w / src_w, out_h / src_h)
                        # expected = min(1.0, out_w / (src_w * scale))
                        expected = min(1.0, out_w / (src_w * max(out_w / src_w, out_h / src_h)))
                        assert f == pytest.approx(expected, rel=1e-7)

    @pytest.mark.parametrize("fraction", [
        0.0001, 0.1, 0.237, 0.31640625, 0.5, 0.75, 0.99, 1.0, 1.5
    ])
    def test_crop_x_for_center_invariants(self, fraction):
        """Invariants for crop_x_for_center:
        1. Always clamped within [0.0, 1.0].
        2. Centered at 0.5 yields 0.5.
        3. Monotonically non-decreasing with center_x.
        4. Clamps at borders [0.0, fraction/2] -> 0.0, [1 - fraction/2, 1.0] -> 1.0.
        5. When fraction >= 1.0, always returns 0.5.
        """
        # Test 1: center=0.5 -> 0.5
        assert framing.crop_x_for_center(0.5, fraction) == pytest.approx(0.5, abs=1e-6)

        # Test 2: when fraction >= 1.0, always 0.5
        if fraction >= 1.0:
            for cx in [-0.5, 0.0, 0.2, 0.5, 0.8, 1.0, 1.5]:
                assert framing.crop_x_for_center(cx, fraction) == 0.5
            return

        # Test 3: Left and right clamps
        left_clamp_boundary = fraction / 2.0
        right_clamp_boundary = 1.0 - fraction / 2.0

        assert framing.crop_x_for_center(0.0, fraction) == 0.0
        assert framing.crop_x_for_center(left_clamp_boundary * 0.5, fraction) == 0.0
        assert framing.crop_x_for_center(left_clamp_boundary, fraction) == pytest.approx(0.0, abs=1e-6)

        assert framing.crop_x_for_center(1.0, fraction) == 1.0
        assert framing.crop_x_for_center(right_clamp_boundary + (1.0 - right_clamp_boundary) * 0.5, fraction) == 1.0
        assert framing.crop_x_for_center(right_clamp_boundary, fraction) == pytest.approx(1.0, abs=1e-6)

        # Test 4: Extreme out-of-range center_x inputs
        assert framing.crop_x_for_center(-10.0, fraction) == 0.0
        assert framing.crop_x_for_center(10.0, fraction) == 1.0

        # Test 5: Monotonicity sweep
        prev = -1.0
        for i in range(101):
            cx = i / 100.0
            val = framing.crop_x_for_center(cx, fraction)
            assert 0.0 <= val <= 1.0
            assert val >= prev - 1e-9, f"Monotonicity violated at cx={cx}: {val} < {prev}"
            prev = val

    def test_crop_x_for_center_geometric_reconstruction(self):
        """Verify the FFmpeg geometric correspondence:
        In FFmpeg, crop filter places window at x = (iw - ow) * crop_x.
        With ow = iw * fraction, the window left edge is x = iw * (1 - fraction) * crop_x.
        The window center in pixels is left_edge + ow / 2.
        As a fraction of iw, center_normalized = (1 - fraction) * crop_x + fraction / 2.
        For any center_x within reachable bounds [fraction/2, 1 - fraction/2],
        center_normalized MUST equal center_x exactly!
        """
        for fraction in [0.1582, 0.237, 0.31640625, 0.5625, 0.75]:
            left_bound = fraction / 2.0
            right_bound = 1.0 - fraction / 2.0
            steps = 50
            for i in range(steps + 1):
                target_center = left_bound + (right_bound - left_bound) * (i / steps)
                crop_x = framing.crop_x_for_center(target_center, fraction)
                reconstructed_center = (1.0 - fraction) * crop_x + fraction / 2.0
                assert reconstructed_center == pytest.approx(target_center, abs=1e-6), (
                    f"Geometric mismatch: target={target_center}, reconstructed={reconstructed_center}"
                )


# ============================================================================
# 2. CAPTION LAYOUT ADVERSARIAL STRESS TESTS
# ============================================================================

class TestCaptionLayoutAdversarial:
    """Stress tests for caption_layout.py wrapping and max 2 lines invariant."""

    @pytest.mark.parametrize("line_limit", [3.0, 5.0, 8.0, 12.0, 15.0, 20.0, 30.0])
    @pytest.mark.parametrize("text", [
        # Mixed CJK and Latin
        "AutoShorts 11.0 架构升级：Mode 1 与 Mode 2 双轨独立并行，零跨模态污染！",
        "Hello 世界！Today is a great day. 今天天气真好，我们一起去公园散步吧。",
        "Paul Graham 曾经在 Essays 里提到过：Do Things that Don't Scale，影响了无数创业者。",
        "NVIDIA NIM H100 GPU 算力集群加速 4K60p 视频渲染与实时人脸检测！",
        "iPhone 16 Pro Max 搭载 A18 Pro 芯片，性能提升明显，续航表现突出。",
        # Extremely long CJK clause without any punctuation
        "这是一个没有任何标点符号的非常漫长且连续的中文字符串用来测试文本自动断行算法在没有标点辅助时能否根据语法助词以及连续汉字合理折行同时绝对遵守每屏最多两行的铁律",
        # Extremely long Latin text with spaces
        "This is an extraordinarily long English clause without any punctuation marks designed to stress test whether greedy word wrapping correctly cuts text into screens while strictly enforcing the golden invariant of at most two lines per screen under all circumstances.",
        # Extremely long Latin text without ANY spaces (continuous word)
        "supercalifragilisticexpialidociousantidisestablishmentarianismpneumonoultramicroscopicsilicovolcanoconiosisunbelievablylongcontinuouslatinwordthatneverendstestinghowthecutterhandlesunbreakablelatin",
        # Consecutive punctuation sequences
        "等等……什么？？！！真的假的？！？！不可能吧。。。。。。",
        "——……——……，，，、、、：：；；！！！！？？？？",
        "Hello... World!?!! Are you sure?! No way...... Seriously???",
        # Special characters & symbols
        "Special symbols: @#$%^&*()_+-=[]{}|;':\",./<>?~`! Emojis: 🔥🚀✨💯🎉🐶🐱🍕🍔🍟",
        "【重磅推荐】《黑神话：悟空》Steam 在线人数突破 200 万！国产 3A 大作崛起！",
        # Mixed numbers, currency, percentages
        "营收增长了 150.75%，净利润达到 $3,450,000 美元，同比增长 32.5%。",
        # Edge cases: particles only
        "的的的的了了了着着着过过过得得得地地地",
        # Short strings
        "短",
        "Hi",
        "OK!",
        "好",
        "1",
    ])
    def test_split_screens_never_exceeds_two_lines(self, text, line_limit):
        """CRITICAL GOLDEN INVARIANT: split_screens NEVER outputs more than 2 lines per screen."""
        screens = cl.split_screens(text, line_limit)
        assert isinstance(screens, list)
        if not text.strip():
            assert screens == []
            return

        assert len(screens) > 0
        for idx, screen in enumerate(screens):
            # Invariant 1: Line count must be <= 2
            cnt = cl.line_count(screen)
            assert cnt <= 2, f"Screen {idx} exceeded 2 lines (got {cnt}): {screen!r}"

            # Invariant 2: Explicit \N break count must be <= 1
            n_count = screen.count("\\N")
            assert n_count <= 1, f"Screen {idx} has {n_count} '\\N' breaks: {screen!r}"

            # Invariant 3: Slices split by \N must have len <= 2
            lines = screen.split("\\N")
            assert len(lines) <= 2, f"Screen {idx} split into {len(lines)} lines: {screen!r}"

            # Invariant 4: No empty lines within the screen
            for l in lines:
                assert len(l.strip()) > 0, f"Screen {idx} contains empty line: {screen!r}"

    @pytest.mark.parametrize("line_limit", [5.0, 10.0, 15.0, 20.0])
    def test_text_preservation(self, line_limit):
        """Verify that split_screens preserves all non-whitespace characters."""
        texts = [
            "这些年把同样的能量，带给了无数创始人，而且这种手把手的方式一直延续到今天，影响了整整一代人",
            "AutoShorts 11.0 presentation architecture strictly isolates Mode 1 and Mode 2.",
            "快速测试：1234567890，ABCDEFG，你好世界！",
        ]
        for text in texts:
            screens = cl.split_screens(text, line_limit)
            reconstructed = "".join(s.replace("\\N", "").replace(" ", "") for s in screens)
            original_compact = text.replace(" ", "")
            assert reconstructed == original_compact, (
                f"Text altered during wrapping!\nOriginal: {original_compact}\nReconstructed: {reconstructed}"
            )

    def test_empty_and_whitespace_inputs(self):
        """Edge cases with empty, whitespace, and standalone punctuation."""
        assert cl.split_screens("", 15.0) == []
        assert cl.split_screens("   ", 15.0) == []
        assert cl.split_screens("\n\t  \r", 15.0) == []
        # Standalone punctuation
        punct_screens = cl.split_screens("，，，", 15.0)
        assert len(punct_screens) <= 1
        for s in punct_screens:
            assert cl.line_count(s) <= 2

    def test_width_computation_cjk_vs_latin(self):
        """Width accounting: CJK=1.0, Latin=0.55."""
        assert cl.width("测") == 1.0
        assert cl.width("试") == 1.0
        assert cl.width("测试") == 2.0
        assert cl.width("a") == pytest.approx(0.55)
        assert cl.width("ab") == pytest.approx(1.10)
        assert cl.width(" ") == pytest.approx(0.55)
        assert cl.width("，") == 1.0  # fullwidth CJK comma
        assert cl.width(",") == pytest.approx(0.55)  # ASCII comma
        assert cl.width("你好 World") == pytest.approx(2.0 + 0.55 + 5 * 0.55)  # 2 + 0.55 + 2.75 = 5.30

    def test_timed_screens_adversarial_invariants(self):
        """Stress-test timed_screens with extreme mismatched subtitle and original lengths."""
        # Case A: Very long original under short Chinese
        short_zh = "这是一个短标题。"
        huge_en = (
            "This is an exceptionally and extraordinarily long English sentence that continues across "
            "multiple clauses and phrases without pausing or stopping, aiming to force timed_screens "
            "to scale screens or truncate gracefully without ever exceeding the two lines per screen limit."
        )
        screens = cl.timed_screens(short_zh, 0.0, 10.0, line_limit=15.0, original=huge_en, original_limit=25.0)
        assert len(screens) >= 1
        assert screens[0][0] == pytest.approx(0.0)
        assert screens[-1][1] == pytest.approx(10.0)

        for start, end, text, orig in screens:
            assert start < end
            assert cl.line_count(text) <= 2
            assert cl.line_count(orig) <= 2
            assert text.count("\\N") <= 1
            assert orig.count("\\N") <= 1

        # Case B: Very long Chinese under short English
        huge_zh = (
            "在人工智能与大模型快速发展的今天，视频生成、数字人播报、智能剪辑和自动化脚本编写等技术 "
            "正在深刻改变内容创作与分发的格局，每一个创作者都拥有了前所未有的超级杠杆与表达工具。"
        )
        short_en = "AI is changing video creation."
        screens2 = cl.timed_screens(huge_zh, 5.0, 25.0, line_limit=12.0, original=short_en, original_limit=20.0)
        assert len(screens2) >= 2
        assert screens2[0][0] == pytest.approx(5.0)
        assert screens2[-1][1] == pytest.approx(25.0)

        for start, end, text, orig in screens2:
            assert start < end
            assert cl.line_count(text) <= 2
            assert cl.line_count(orig) <= 2


# ============================================================================
# 3. SEGMENT TRACK HOLD DAMPENING STRESS TESTS
# ============================================================================

class TestSegmentTrackHoldAdversarial:
    """Stress tests for framing.segment_track hold logic (min_hold=2, min_jump=0.12)."""

    def test_empty_and_single_sample(self):
        assert framing.segment_track([]) == []
        assert framing.segment_track([(0.0, 0.5)]) == [{"start": 0.0, "crop_x": 0.5}]
        assert framing.segment_track([(10.0, 0.35)]) == [{"start": 0.0, "crop_x": 0.35}]

    def test_identical_samples(self):
        samples = [(i * 2.5, 0.5) for i in range(10)]
        track = framing.segment_track(samples)
        assert track == [{"start": 0.0, "crop_x": 0.5}]

    def test_single_glance_spike_never_jumps(self):
        """Crucial requirement: a single glance to another speaker must NOT move the camera."""
        # Baseline at 0.3, spike to 0.8 at t=5.0, return to 0.3 at t=7.5
        samples = [
            (0.0, 0.3),
            (2.5, 0.31),
            (5.0, 0.85),  # Single glance spike (> min_jump=0.12)
            (7.5, 0.3),   # Returned to baseline
            (10.0, 0.32),
        ]
        track = framing.segment_track(samples, min_hold=2, min_jump=0.12)
        assert len(track) == 1
        assert track[0]["start"] == 0.0
        assert track[0]["crop_x"] == 0.3

    def test_rapid_alternating_glances_never_jump(self):
        """Oscillating between two speakers with 1 sample each: camera must hold steady."""
        samples = [
            (0.0, 0.2),
            (2.5, 0.8),  # glance 1
            (5.0, 0.2),  # return
            (7.5, 0.8),  # glance 2
            (10.0, 0.2), # return
            (12.5, 0.8), # glance 3
            (15.0, 0.2), # return
        ]
        track = framing.segment_track(samples, min_hold=2, min_jump=0.12)
        assert len(track) == 1
        assert track == [{"start": 0.0, "crop_x": 0.2}]

    def test_persistent_switch_moves_camera_at_first_transition_sample(self):
        """When a new speaker persists for min_hold (2) samples, track updates with start at pending[0]."""
        samples = [
            (0.0, 0.3),
            (2.5, 0.3),
            (5.0, 0.8),   # Sample 1 of new speaker
            (7.5, 0.82),  # Sample 2 of new speaker (persisted!)
            (10.0, 0.81), # Continuing
        ]
        track = framing.segment_track(samples, min_hold=2, min_jump=0.12)
        assert len(track) == 2
        assert track[0] == {"start": 0.0, "crop_x": 0.3}
        assert track[1]["start"] == 5.0  # Start timestamp of FIRST sample of the turn!
        assert track[1]["crop_x"] == pytest.approx(0.81, abs=0.01)

    def test_three_way_glance_vs_persistence(self):
        """Speaker A (0.2) -> glance Speaker B (0.5, 1 sample) -> glance Speaker C (0.9, 1 sample) -> return to A."""
        samples = [
            (0.0, 0.2),
            (2.5, 0.5),  # B (glance)
            (5.0, 0.9),  # C (glance - note jump from B is 0.4 >= min_jump, resetting pending!)
            (7.5, 0.2),  # A (return)
            (10.0, 0.2),
        ]
        track = framing.segment_track(samples, min_hold=2, min_jump=0.12)
        assert len(track) == 1
        assert track == [{"start": 0.0, "crop_x": 0.2}]

        # Now test where Speaker C persists:
        samples_c_persists = [
            (0.0, 0.2),
            (2.5, 0.5),   # B (glance)
            (5.0, 0.9),   # C (sample 1)
            (7.5, 0.91),  # C (sample 2 - persisted!)
            (10.0, 0.9),
        ]
        track_c = framing.segment_track(samples_c_persists, min_hold=2, min_jump=0.12)
        assert len(track_c) == 2
        assert track_c[0] == {"start": 0.0, "crop_x": 0.2}
        assert track_c[1]["start"] == 5.0
        assert track_c[1]["crop_x"] == pytest.approx(0.905, abs=0.01)

    def test_subthreshold_jitter_rejected(self):
        """Jitter below min_jump (0.12) should never trigger a new segment."""
        # Fluctuations within [0.25, 0.35] relative to 0.30 (max diff 0.05 < 0.12)
        samples = [
            (0.0, 0.30),
            (2.5, 0.35),
            (5.0, 0.26),
            (7.5, 0.34),
            (10.0, 0.28),
            (12.5, 0.32),
        ]
        track = framing.segment_track(samples, min_hold=2, min_jump=0.12)
        assert len(track) == 1
        assert track == [{"start": 0.0, "crop_x": 0.3}]

    def test_exact_jump_threshold_boundary(self):
        """Test boundary right below and right above min_jump (0.12)."""
        # Exactly below: diff = 0.119
        samples_below = [(0.0, 0.50), (2.5, 0.619), (5.0, 0.619)]
        track_below = framing.segment_track(samples_below, min_hold=2, min_jump=0.12)
        assert len(track_below) == 1

        # Exactly above: diff = 0.121
        samples_above = [(0.0, 0.50), (2.5, 0.621), (5.0, 0.621)]
        track_above = framing.segment_track(samples_above, min_hold=2, min_jump=0.12)
        assert len(track_above) == 2
        assert track_above[1]["start"] == 2.5
        assert track_above[1]["crop_x"] == 0.621

    @pytest.mark.parametrize("min_hold", [1, 2, 3, 4])
    def test_configurable_min_hold(self, min_hold):
        """Verify min_hold parameter scaling."""
        # 3 consecutive samples of new speaker at t=5.0, 7.5, 10.0
        samples = [
            (0.0, 0.2),
            (2.5, 0.2),
            (5.0, 0.8),
            (7.5, 0.8),
            (10.0, 0.8),
            (12.5, 0.2),
        ]
        track = framing.segment_track(samples, min_hold=min_hold, min_jump=0.12)
        if min_hold <= 3:
            # Should have switched to 0.8
            assert any(p["crop_x"] == 0.8 for p in track)
        else:
            # min_hold=4 requires 4 samples, 0.8 only had 3 -> should NOT switch
            assert not any(p["crop_x"] == 0.8 for p in track)


# ============================================================================
# 4. RANDOMIZED PROPERTY FUZZING HARNESS
# ============================================================================

class TestRandomizedFuzzingHarness:
    """Intensive property-based fuzzer generating thousands of adversarial inputs."""

    def test_fuzz_framing_math_10000_iterations(self):
        """Fuzz window_fraction and crop_x_for_center across 10,000 random dimension pairs."""
        import random
        rng = random.Random(42)

        for _ in range(10000):
            # Extreme random dimensions from 1 to 10000
            src_w = rng.randint(1, 10000)
            src_h = rng.randint(1, 10000)
            out_w = rng.randint(1, 10000)
            out_h = rng.randint(1, 10000)

            frac = framing.window_fraction(src_w, src_h, out_w, out_h)
            assert 0.0 < frac <= 1.0, f"Invalid frac {frac} for {(src_w, src_h, out_w, out_h)}"

            # Random center_x including out-of-bounds
            center_x = rng.uniform(-2.0, 3.0)
            crop_x = framing.crop_x_for_center(center_x, frac)
            assert 0.0 <= crop_x <= 1.0, f"Invalid crop_x {crop_x} for center={center_x}, frac={frac}"

    def test_fuzz_caption_layout_5000_iterations(self):
        """Fuzz caption wrapping with 5,000 random text strings and verify max 2 lines invariant."""
        import random
        rng = random.Random(1337)

        cjk_chars = "的一是在不了有和人这中大为上个国我以要他时来用生到作地于出就分对成会可主发年动同工也能下过子说产种面而方后多定行学法所民得经十三之进着等部度家电力里如水化高自二理起小物现实加量都两体制机当使点从业本去把性好应开它"
        latin_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "
        punct_chars = "，。！？!?；;、,：:…—-_=+/\\@#$%^&*()[]{}'\"`~"

        for _ in range(5000):
            # Generate random mixed text
            length = rng.randint(1, 150)
            text_type = rng.choice(["mixed", "cjk_heavy", "latin_heavy", "punct_heavy"])
            chars = []
            for _ in range(length):
                roll = rng.random()
                if text_type == "mixed":
                    if roll < 0.4:
                        chars.append(rng.choice(cjk_chars))
                    elif roll < 0.8:
                        chars.append(rng.choice(latin_chars))
                    else:
                        chars.append(rng.choice(punct_chars))
                elif text_type == "cjk_heavy":
                    chars.append(rng.choice(cjk_chars) if roll < 0.8 else rng.choice(punct_chars))
                elif text_type == "latin_heavy":
                    chars.append(rng.choice(latin_chars) if roll < 0.8 else rng.choice(punct_chars))
                else:
                    chars.append(rng.choice(punct_chars))

            # Pre-existing ASS line-break control sequences (\N) in raw text are sanitized by callers
            # (e.g. publish_export.py / packaging_render.py _esc) so random text should not contain literal '\N'.
            text = "".join(chars).replace("\\N", " ").replace("\\n", " ")
            limit = rng.uniform(1.0, 35.0)

            screens = cl.split_screens(text, limit)
            for s in screens:
                # Golden Invariant: Never > 2 lines per screen
                assert cl.line_count(s) <= 2, f"Fuzz violation: {s!r} in screens of {text!r} with limit={limit}"
                assert s.count("\\N") <= 1

    def test_literal_ass_linebreak_adversarial_boundary(self):
        """Adversarial Discovery: If raw text contains pre-existing literal '\\N' ASS control codes,
        caption_layout.py does not strip or split on '\\N' (as it assumes raw transcript text).
        Therefore, joining two lines with '\\N' produces a screen with multiple '\\N' tokens.
        This proves why callers (packaging_render._esc / publish_export) MUST sanitize/escape raw text.
        """
        raw_text_with_control_code = "Line A\\NLine B with another clause，还有第二句中文。"
        screens = cl.split_screens(raw_text_with_control_code, 15.0)
        # Because the first clause already contains '\N', when split_screens pairs it with another line,
        # the screen contains more than 1 '\N' token:
        first_screen = screens[0]
        assert "\\N" in first_screen
        # Demonstrating the boundary:
        assert cl.line_count("Pre\\NExisting") == 2
        # Once escaped as per packaging_render._esc (replacing '\\' with '＼'), invariant holds:
        escaped_text = raw_text_with_control_code.replace("\\", "＼")
        escaped_screens = cl.split_screens(escaped_text, 15.0)
        for s in escaped_screens:
            assert cl.line_count(s) <= 2


    def test_fuzz_segment_track_2000_iterations(self):
        """Fuzz segment_track with 2,000 random tracks containing glance spikes and noise."""
        import random
        rng = random.Random(2026)

        for _ in range(2000):
            n_samples = rng.randint(1, 50)
            samples = []
            t = 0.0
            x = rng.uniform(0.1, 0.9)
            samples.append((t, x))

            for _ in range(n_samples - 1):
                t += rng.uniform(0.5, 3.0)
                event = rng.choice(["same", "jitter", "glance", "switch"])
                if event == "same":
                    samples.append((t, x))
                elif event == "jitter":
                    # Subthreshold jitter (< 0.12)
                    samples.append((t, x + rng.uniform(-0.08, 0.08)))
                elif event == "glance":
                    # Single spike far away
                    glance_x = rng.uniform(0.0, 1.0)
                    samples.append((t, glance_x))
                    t += rng.uniform(0.5, 3.0)
                    samples.append((t, x))  # return immediately
                else:  # switch
                    x = rng.uniform(0.1, 0.9)
                    samples.append((t, x))

            track = framing.segment_track(samples, min_hold=2, min_jump=0.12)
            assert len(track) >= 1
            assert track[0]["start"] == 0.0
            # Monotonic timestamps
            for i in range(len(track) - 1):
                assert track[i]["start"] < track[i + 1]["start"], f"Non-monotonic track: {track}"

