#!/usr/bin/env python3
"""
test_hook_ending_optimization_suite.py — AutoShorts 8.0
Hook & Ending Optimization 2.0 — Comprehensive Verification Suite (Cases A–O)

Exercises the REAL production pipeline via the `boundary_inspect` CLI
(the exact code path the Tauri app uses at render time):
    optimize_boundaries → pacing on the OPTIMIZED range → framing on the
    OPTIMIZED range → word remap → ASS/SRT captions.

Timeline mapping (spec):
    SOURCE    = original candidate start/end (never mutated)
    OPTIMIZED = new start/end chosen by the optimizer
    OUTPUT    = post-pacing timeline (Smart Pacing's responsibility)

Division of labor (spec):
    Hook/Ending Optimization chooses better CONTENT boundaries.
    Smart Pacing removes safe unnecessary TIME inside those boundaries.

Cases:
    A  Strong hook + strong conclusion → NO change (false negatives OK)
    B  Weak setup opening → forward skip to strong statement
    C  Context-required opening → preserved (no unprovable moves)
    D  Strong conclusion ending → NO change
    E  Trailing wind-down → end trimmed earlier
    F  Cut-off conclusion → end extended to completion
    G  Unsafe contextless opening → preserved
    H  Incomplete conclusion far away → preserved
    I  Multi-speaker Q→A: (a) backward repair includes the question,
       (b) forward skip blocked by a question
    J  Small gain → NO change + REAL kill-switch via subprocess env
    K  Smart Pacing integration: pacing receives the OPTIMIZED range
    L  Caption sync: remapped words exclude skipped setup; ASS reflects
       the final timeline
    M  Framing/DualFrame: framing runs on the OPTIMIZED range; kill-switch
       control run keeps the SOURCE range
    N  Isolation (two-speaker): Q→A preserved, framing valid
    O  Determinism: identical inputs → identical outputs

Usage:
    python test_hook_ending_optimization_suite.py
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.abspath(__file__))
EXE = os.path.join(ROOT, "autoshorts", "src-tauri", "target", "debug",
                   "boundary_inspect.exe")
BEAT_VIDEO = os.path.join(ROOT, "Beat Emotional Fatigue_ Better Sleep & Clearer Mind.mp4")
MESSI_VIDEO = os.path.join(
    ROOT, "Messi vs Ronaldo Fans： The Psychology Explained [rssDTc086bk].mp4")

if not os.path.exists(EXE):
    print("FATAL: boundary_inspect.exe not found at", EXE)
    print("Build it first:  cd autoshorts/src-tauri && cargo build --bin boundary_inspect")
    sys.exit(1)

NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


# ── Helpers ─────────────────────────────────────────────────────────────────

def w(text, start, end, speaker=None):
    return {"text": text, "start": round(start, 3), "end": round(end, 3),
            "speaker": speaker}


def transcript(words, duration=None):
    """Full NormalizedTranscript JSON shape (camelCase)."""
    if duration is None:
        duration = max(x["end"] for x in words) + 5.0 if words else 60.0
    return {"language": "en", "duration": duration, "speakers": [],
            "words": words, "segments": []}


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)


def run_inspect(start, end, words, candidate=None, source=None, style=None,
                env_extra=None):
    """Run boundary_inspect with the given inputs. Empty-string args are
    preserved (Python subprocess does not drop them, unlike PowerShell)."""
    tmpdir = tempfile.mkdtemp(prefix="boundary_case_")
    words_path = os.path.join(tmpdir, "words.json")
    write_json(words_path, words if isinstance(words, list) else transcript(words))
    cand_path = os.path.join(tmpdir, "candidate.json")
    write_json(cand_path, candidate or {})

    args = [EXE, str(start), str(end), words_path, cand_path]
    args.append(source or "")
    if style:
        args.append(style)

    env = dict(os.environ)
    env.pop("AUTOSHORTS_BOUNDARY_OPTIMIZATION", None)
    if env_extra:
        env.update(env_extra)

    r = subprocess.run(args, capture_output=True, text=True, encoding="utf-8",
                       timeout=600, creationflags=NO_WINDOW, env=env)
    if r.returncode != 0:
        raise AssertionError(
            "boundary_inspect failed (exit {}): {}".format(r.returncode,
                                                          r.stderr[:2000]))
    return json.loads(r.stdout)


def opt_of(d):
    return d["optimization"]


def body_words(start, count, step=0.4, dur=0.2, prefix="body"):
    """Filler body words (content-bearing tokens) to satisfy duration floors."""
    return [w("{}{}".format(prefix, i), start + i * step, start + i * step + dur)
            for i in range(count)]


# ── A. Strong hook + strong conclusion → NO change ─────────────────────────

class TestCaseA(unittest.TestCase):
    """CASE A: strong standalone hook + strong complete conclusion → the
    optimizer must keep both boundaries (false negatives are acceptable,
    incorrect changes are not)."""

    def test_no_change(self):
        words = [
            w("The", 10.0, 10.1), w("biggest", 10.1, 10.3), w("mistake", 10.3, 10.6),
            w("people", 10.6, 10.8), w("make", 10.8, 11.0), w("is", 11.0, 11.1),
            w("overthinking.", 11.1, 11.6),
        ] + body_words(12.0, 30) + [
            w("And", 25.0, 25.2), w("that's", 25.2, 25.4), w("why", 25.4, 25.5),
            w("I", 25.5, 25.6), w("stopped", 25.6, 26.0), w("doing", 26.0, 26.2),
            w("it.", 26.2, 26.6),
        ]
        d = run_inspect(10.0, 27.0, words)
        o = opt_of(d)
        self.assertFalse(o["startChanged"], "strong hook must not move")
        self.assertFalse(o["endChanged"], "strong conclusion must not move")
        self.assertEqual(o["optimizedStartSec"], 10.0)
        self.assertEqual(o["optimizedEndSec"], 27.0)


# ── B. Weak setup opening → forward skip ────────────────────────────────────

class TestCaseB(unittest.TestCase):
    """CASE B: provably low-value setup ("So yeah we were talking about
    that." — all closed-vocab) before a strong standalone statement → the
    start skips forward to the statement."""

    def test_forward_skip(self):
        words = [
            w("So", 20.0, 20.2), w("yeah", 20.2, 20.4), w("we", 20.4, 20.5),
            w("were", 20.5, 20.6), w("talking", 20.6, 20.9), w("about", 20.9, 21.1),
            w("that.", 21.1, 21.4),
            w("The", 22.0, 22.1), w("biggest", 22.1, 22.3), w("mistake", 22.3, 22.6),
            w("people", 22.6, 22.8), w("make", 22.8, 23.0), w("is", 23.0, 23.1),
            w("overthinking.", 23.1, 23.6),
        ] + body_words(24.0, 35) + [w("end.", 39.6, 40.0)]
        d = run_inspect(20.0, 40.0, words)
        o = opt_of(d)
        self.assertTrue(o["startChanged"], "setup should be skipped")
        self.assertAlmostEqual(o["optimizedStartSec"], 21.85, delta=0.3)
        self.assertFalse(o["endChanged"])
        self.assertIn("setup", (o["startReason"] or "").lower())

    def test_blocked_by_content_word(self):
        # "psychology" is a content word → the span is NOT provably
        # low-value → keep the original boundary.
        words = [
            w("So", 30.0, 30.2), w("yeah", 30.2, 30.4), w("we", 30.4, 30.5),
            w("were", 30.5, 30.6), w("talking", 30.6, 30.9), w("about", 30.9, 31.1),
            w("psychology.", 31.1, 31.6),
            w("The", 32.0, 32.1), w("biggest", 32.1, 32.3), w("mistake", 32.3, 32.6),
            w("people", 32.6, 32.8), w("make", 32.8, 33.0), w("is", 33.0, 33.1),
            w("overthinking.", 33.1, 33.6),
        ] + body_words(34.0, 30) + [w("end.", 46.0, 46.4)]
        d = run_inspect(30.0, 47.0, words)
        o = opt_of(d)
        self.assertFalse(o["startChanged"], "content word must block the skip")


# ── C. Context-required opening → preserved ─────────────────────────────────

class TestCaseC(unittest.TestCase):
    """CASE C: the setup span contains material whose removal could damage
    context (question, pronoun antecedent) → preserve the original start."""

    def test_question_blocks_skip(self):
        # The skipped span ends with "?" → Q→A protection blocks the move.
        words = [
            w("So", 40.0, 40.2), w("why", 40.2, 40.4), w("did", 40.4, 40.5),
            w("you", 40.5, 40.6), w("leave?", 40.6, 41.0),
            w("I", 42.0, 42.1), w("left", 42.1, 42.4), w("because", 42.4, 42.6),
            w("I", 42.6, 42.7), w("was", 42.7, 42.8), w("tired.", 42.8, 43.2),
        ] + body_words(44.0, 35) + [w("end.", 58.0, 58.4)]
        d = run_inspect(40.0, 59.0, words)
        o = opt_of(d)
        self.assertFalse(o["startChanged"], "question must block the skip")

    def test_hook_anchor_protection(self):
        # Hook anchor inside the skipped span → move rejected even when
        # the span is otherwise all-vocab.
        words = [
            w("So", 50.0, 50.2), w("yeah", 50.2, 50.4), w("we", 50.4, 50.5),
            w("were", 50.5, 50.6), w("talking", 50.6, 50.9), w("about", 50.9, 51.1),
            w("that.", 51.1, 51.4),
            w("The", 52.0, 52.1), w("biggest", 52.1, 52.3), w("mistake", 52.3, 52.6),
            w("people", 52.6, 52.8), w("make", 52.8, 53.0), w("is", 53.0, 53.1),
            w("overthinking.", 53.1, 53.6),
        ] + body_words(54.0, 35) + [w("end.", 68.0, 68.4)]
        cand = {"hook": "So yeah we were talking about that.",
                "hookStartSec": 50.0, "hookEndSec": 51.4,
                "hookConfidence": 0.90}
        d = run_inspect(50.0, 69.0, words, candidate=cand)
        o = opt_of(d)
        self.assertFalse(o["startChanged"], "hook anchor must block the skip")


# ── D. Strong conclusion ending → NO change ─────────────────────────────────

class TestCaseD(unittest.TestCase):
    """CASE D: the clip already ends on a strong complete conclusion →
    no end move (no extension to NEW sentences after a complete end)."""

    def test_no_change(self):
        words = [
            w("The", 80.0, 80.1), w("result", 80.1, 80.4), w("changed", 80.4, 80.7),
            w("my", 80.7, 80.8), w("life.", 80.8, 81.2),
        ] + body_words(82.0, 35) + [
            w("And", 96.0, 96.2), w("that's", 96.2, 96.4), w("why", 96.4, 96.5),
            w("I", 96.5, 96.6), w("stopped", 96.6, 97.0), w("doing", 97.0, 97.2),
            w("it.", 97.2, 97.6),
        ]
        d = run_inspect(80.0, 98.0, words)
        o = opt_of(d)
        self.assertFalse(o["endChanged"], "strong conclusion must not move")
        self.assertFalse(o["startChanged"])


# ── E. Trailing wind-down → end trimmed earlier ─────────────────────────────

class TestCaseE(unittest.TestCase):
    """CASE E: trailing wind-down ("So yeah that's basically the whole
    story." — all closed-vocab) after a strong conclusion → trim the end."""

    def test_wind_down_trim(self):
        words = [
            w("The", 80.0, 80.1), w("result", 80.1, 80.4), w("changed", 80.4, 80.7),
            w("my", 80.7, 80.8), w("life.", 80.8, 81.2),
        ] + body_words(82.0, 35) + [
            w("conclusion.", 96.0, 96.5),
            w("So", 97.0, 97.2), w("yeah,", 97.2, 97.4), w("that's", 97.4, 97.6),
            w("basically", 97.6, 97.9), w("the", 97.9, 98.0), w("whole", 98.0, 98.2),
            w("story.", 98.2, 98.6),
        ]
        d = run_inspect(80.0, 99.0, words)
        o = opt_of(d)
        self.assertTrue(o["endChanged"], "wind-down should be trimmed")
        self.assertAlmostEqual(o["optimizedEndSec"], 96.6, delta=0.25)
        self.assertFalse(o["startChanged"])
        self.assertIn("wind-down", (o["endReason"] or "").lower())

    def test_trim_blocked_by_content_word(self):
        # "So I decided to quit." — "decided"/"quit" are content words →
        # the span is not provably wind-down → keep the original end.
        words = [
            w("The", 90.0, 90.1), w("result", 90.1, 90.4), w("changed", 90.4, 90.7),
            w("my", 90.7, 90.8), w("life.", 90.8, 91.2),
        ] + body_words(92.0, 35) + [
            w("So", 106.0, 106.2), w("I", 106.2, 106.3), w("decided", 106.3, 106.7),
            w("to", 106.7, 106.8), w("quit.", 106.8, 107.2),
        ]
        d = run_inspect(90.0, 108.0, words)
        o = opt_of(d)
        self.assertFalse(o["endChanged"], "content word must block the trim")

    def test_trim_blocked_by_payoff_endpoint(self):
        # A candidate with a verified payoff_end_sec must NEVER have its endpoint trimmed
        words = [
            w("The", 80.0, 80.1), w("result", 80.1, 80.4), w("changed", 80.4, 80.7),
            w("my", 80.7, 80.8), w("life.", 80.8, 81.2),
        ] + body_words(82.0, 35) + [
            w("conclusion.", 96.0, 96.5),
            w("So", 97.0, 97.2), w("yeah,", 97.2, 97.4), w("that's", 97.4, 97.6),
            w("basically", 97.6, 97.9), w("the", 97.9, 98.0), w("whole", 98.0, 98.2),
            w("story.", 98.2, 98.6),
        ]
        cand = {"payoffEndSec": 98.6}
        d = run_inspect(80.0, 98.6, words, candidate=cand)
        o = opt_of(d)
        self.assertFalse(o["endChanged"], "payoff endpoint must block wind-down trim")
        self.assertEqual(o["optimizedEndSec"], 98.6)

        d2 = run_inspect(80.0, 99.0, words, candidate=cand)
        o2 = opt_of(d2)
        self.assertTrue(o2["endChanged"], "differing raw end snaps to payoff endpoint")
        self.assertEqual(o2["optimizedEndSec"], 98.6)
        self.assertEqual(o2["endReason"], "authoritative payoff endpoint")


# ── F. Cut-off conclusion → end extended ────────────────────────────────────

class TestCaseF(unittest.TestCase):
    """CASE F: the clip cuts mid-sentence and the sentence completes within
    the bounded window (≤6s, ≤20 words) → extend the end to completion."""

    def test_end_extend(self):
        words = [
            w("And", 100.0, 100.1), w("that's", 100.1, 100.3), w("why", 100.3, 100.4),
            w("I", 100.4, 100.5), w("finally", 100.5, 100.8), w("stopped", 100.8, 101.2),
            w("doing", 101.2, 101.4), w("it.", 101.4, 101.8),
            w("The", 102.0, 102.1), w("lesson", 102.1, 102.5), w("was", 102.5, 102.7),
            w("worth", 102.7, 103.0), w("every", 103.0, 103.2), w("second", 103.2, 103.6),
            w("of", 103.6, 103.7), w("pain.", 103.7, 104.2),
        ]
        d = run_inspect(100.0, 103.0, words)
        o = opt_of(d)
        self.assertTrue(o["endChanged"], "mid-sentence cut should be completed")
        self.assertGreaterEqual(o["optimizedEndSec"], 104.2,
                                "new end must include the full sentence")


# ── G. Unsafe contextless opening → preserved ──────────────────────────────

class TestCaseG(unittest.TestCase):
    """CASE G: pronoun opening ("She stole everything.") with no provable
    antecedent → the optimizer must NOT move the start (pronoun grounding
    is unprovable; only Q→A repair is allowed backward)."""

    def test_pronoun_opening_preserved(self):
        words = [
            w("Maria", 70.0, 70.3), w("was", 70.3, 70.4), w("my", 70.4, 70.5),
            w("partner.", 70.5, 70.9),
            w("She", 71.2, 71.4), w("stole", 71.4, 71.7), w("everything.", 71.7, 72.2),
        ] + body_words(73.0, 35) + [w("end.", 87.0, 87.4)]
        d = run_inspect(71.2, 88.0, words)
        o = opt_of(d)
        self.assertFalse(o["startChanged"],
                         "pronoun opening must be preserved (no unprovable moves)")


# ── H. Incomplete conclusion far away → preserved ──────────────────────────

class TestCaseH(unittest.TestCase):
    """CASE H: the clip cuts mid-sentence but completion is outside the
    bounded window (>6s away) → keep the original end."""

    def test_far_completion_preserved(self):
        words = [
            w("The", 110.0, 110.1), w("lesson", 110.1, 110.5), w("was", 110.5, 110.7),
            w("painful.", 118.0, 118.4),
        ]
        d = run_inspect(110.0, 111.0, words)
        o = opt_of(d)
        self.assertFalse(o["endChanged"], "far completion must block extension")


# ── I. Multi-speaker Q→A ────────────────────────────────────────────────────

class TestCaseI(unittest.TestCase):
    """CASE I: multi-speaker question→answer. (a) An answer opening
    ("Because ...") preceded by a short question → backward repair
    includes the question (cross-speaker allowed). (b) A forward skip
    whose skipped span contains a question is blocked."""

    def test_qa_backward_repair(self):
        words = [
            w("Why", 60.0, 60.2, "Host"), w("did", 60.2, 60.3, "Host"),
            w("she", 60.3, 60.4, "Host"), w("quit?", 60.4, 60.8, "Host"),
            w("Because", 61.2, 61.5, "Guest"), w("the", 61.5, 61.6, "Guest"),
            w("job", 61.6, 61.8, "Guest"), w("was", 61.8, 61.9, "Guest"),
            w("destroying", 61.9, 62.3, "Guest"), w("her.", 62.3, 62.7, "Guest"),
        ] + body_words(63.0, 40) + [w("end.", 79.0, 79.4)]
        d = run_inspect(61.2, 80.0, words)
        o = opt_of(d)
        self.assertTrue(o["startChanged"], "Q→A repair should fire")
        self.assertLessEqual(o["optimizedStartSec"], 60.0 + 1e-9,
                             "new start at/before the question")
        self.assertGreaterEqual(o["optimizedStartSec"], 59.0,
                                "new start not before the question's lead")
        reason = (o["startReason"] or "").lower()
        self.assertIn("repair", reason, "reason must describe the Q->A repair")
        self.assertIn("question", reason)

    def test_forward_blocked_by_question(self):
        words = [
            w("So", 40.0, 40.2), w("why", 40.2, 40.4), w("did", 40.4, 40.5),
            w("you", 40.5, 40.6), w("leave?", 40.6, 41.0),
            w("I", 42.0, 42.1), w("left", 42.1, 42.4), w("because", 42.4, 42.6),
            w("I", 42.6, 42.7), w("was", 42.7, 42.8), w("tired.", 42.8, 43.2),
        ] + body_words(44.0, 35) + [w("end.", 58.0, 58.4)]
        d = run_inspect(40.0, 59.0, words)
        o = opt_of(d)
        self.assertFalse(o["startChanged"], "question must block the forward skip")


# ── J. Small gain → NO change + REAL kill-switch ────────────────────────────

class TestCaseJ(unittest.TestCase):
    """CASE J: (a) a move whose gain is below the minimum threshold keeps
    the original boundary; (b) the REAL kill-switch
    (AUTOSHORTS_BOUNDARY_OPTIMIZATION=0) disables the optimizer end-to-end
    via subprocess env (no code stubs)."""

    def test_small_gain_no_change(self):
        # Old opening is deficient (setup opener "So yeah") but the only
        # candidate landing sentence is itself deficient (continuation
        # opener "And so ...") → gain below threshold → no move.
        words = [
            w("So", 150.0, 150.2), w("yeah", 150.2, 150.4), w("we", 150.4, 150.5),
            w("were", 150.5, 150.6), w("talking", 150.6, 150.9), w("about", 150.9, 151.1),
            w("that.", 151.1, 151.4),
            w("And", 152.0, 152.1), w("so", 152.1, 152.3), w("the", 152.3, 152.6),
            w("results", 152.6, 153.0), w("followed.", 153.0, 153.4),
        ] + body_words(154.0, 35) + [w("end.", 168.0, 168.4)]
        d = run_inspect(150.0, 169.0, words)
        o = opt_of(d)
        self.assertFalse(o["startChanged"],
                         "gain below threshold must keep the original boundary")

    def test_kill_switch_off(self):
        # Same fixture as case B (a move that WOULD fire) — with the
        # kill-switch env set, nothing moves.
        words = [
            w("So", 20.0, 20.2), w("yeah", 20.2, 20.4), w("we", 20.4, 20.5),
            w("were", 20.5, 20.6), w("talking", 20.6, 20.9), w("about", 20.9, 21.1),
            w("that.", 21.1, 21.4),
            w("The", 22.0, 22.1), w("biggest", 22.1, 22.3), w("mistake", 22.3, 22.6),
            w("people", 22.6, 22.8), w("make", 22.8, 23.0), w("is", 23.0, 23.1),
            w("overthinking.", 23.1, 23.6),
        ] + body_words(24.0, 35) + [w("end.", 39.6, 40.0)]
        d = run_inspect(20.0, 40.0, words,
                        env_extra={"AUTOSHORTS_BOUNDARY_OPTIMIZATION": "0"})
        o = opt_of(d)
        self.assertFalse(o["startChanged"], "kill-switch must disable the move")
        self.assertEqual(o["optimizedStartSec"], 20.0)
        self.assertEqual(o["optimizedEndSec"], 40.0)


# ── K. Smart Pacing integration ─────────────────────────────────────────────

class TestCaseK(unittest.TestCase):
    """CASE K: pacing receives the OPTIMIZED range (not the SOURCE range).
    Uses the real Beat video: synthetic setup words sit in real silence at
    39.0–39.6, the strong statement starts at 39.72 (real speech)."""

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(BEAT_VIDEO):
            raise unittest.SkipTest("Beat video not found: " + BEAT_VIDEO)
        # Synthetic transcript: a complete setup sentence ("So yeah.")
        # sits in the real silence before the strong statement at 39.72.
        words = [
            w("So", 39.0, 39.2), w("yeah.", 39.2, 39.45),
            w("The", 39.72, 39.8), w("biggest", 39.8, 39.95), w("mistake", 39.95, 40.2),
            w("people", 40.2, 40.4), w("make", 40.4, 40.6), w("is", 40.6, 40.7),
            w("overthinking.", 40.7, 41.2),
        ] + body_words(41.5, 16, step=0.4, prefix="real")
        cls.words = words
        cls.d = run_inspect(38.92, 48.0, words, source=BEAT_VIDEO)

    def test_optimization_fires(self):
        o = opt_of(self.d)
        self.assertTrue(o["startChanged"])
        self.assertAlmostEqual(o["optimizedStartSec"], 39.57, delta=0.3)

    def test_pacing_range_is_optimized_range(self):
        o = opt_of(self.d)
        pr = self.d["pacingRange"]
        self.assertAlmostEqual(pr["startSec"], o["optimizedStartSec"], delta=1e-6)
        self.assertAlmostEqual(pr["endSec"], o["optimizedEndSec"], delta=1e-6)

    def test_framing_range_is_optimized_range(self):
        o = opt_of(self.d)
        fr = self.d["framingRange"]
        self.assertAlmostEqual(fr["startSec"], o["optimizedStartSec"], delta=1e-6)
        self.assertAlmostEqual(fr["endSec"], o["optimizedEndSec"], delta=1e-6)

    def test_pacing_plan_on_optimized_range(self):
        plan = self.d.get("pacing")
        if plan:  # engine may legitimately return no plan
            self.assertAlmostEqual(plan["clipStartSec"],
                                  opt_of(self.d)["optimizedStartSec"], delta=1e-6)
            self.assertAlmostEqual(plan["clipEndSec"],
                                  opt_of(self.d)["optimizedEndSec"], delta=1e-6)


# ── L. Caption sync ─────────────────────────────────────────────────────────

class TestCaseL(unittest.TestCase):
    """CASE L: after a forward skip, the caption words exclude the skipped
    setup and the ASS reflects the final timeline (first caption word is
    the strong statement, not the setup). Runs the full downstream path on
    the real Beat video (same fixture as case K)."""

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(BEAT_VIDEO):
            raise unittest.SkipTest("Beat video not found: " + BEAT_VIDEO)
        words = [
            w("So", 39.0, 39.2), w("yeah.", 39.2, 39.45),
            w("The", 39.72, 39.8), w("biggest", 39.8, 39.95), w("mistake", 39.95, 40.2),
            w("people", 40.2, 40.4), w("make", 40.4, 40.6), w("is", 40.6, 40.7),
            w("overthinking.", 40.7, 41.2),
        ] + body_words(41.5, 16, step=0.4, prefix="real")
        cls.d = run_inspect(38.92, 48.0, words, source=BEAT_VIDEO)

    def test_caption_words_exclude_setup(self):
        o = opt_of(self.d)
        self.assertTrue(o["startChanged"], "fixture must trigger the skip")
        rw = self.d["remappedWords"]
        self.assertIsNotNone(rw, "caption words must exist")
        # Cross-path invariant (legacy AND paced): the first word that
        # overlaps the OPTIMIZED range is the strong statement, never the
        # skipped setup.
        in_range = [x for x in rw
                    if x["end"] > o["optimizedStartSec"] + 1e-9
                    and x["start"] < o["optimizedEndSec"] - 1e-9]
        self.assertTrue(in_range, "words must overlap the optimized range")
        self.assertEqual(in_range[0]["text"], "The",
                         "first caption word must be the strong statement")
        self.assertNotIn("So", [x["text"] for x in in_range[:3]],
                         "skipped setup must not lead captions")

    def test_ass_reflects_final_timeline(self):
        ass = self.d["ass"]
        self.assertTrue(ass, "ASS must be generated")
        self.assertNotIn("So yeah", ass, "setup phrase must not appear in captions")
        # Caption text is uppercased by the ASS generator.
        self.assertIn("BIGGEST", ass, "strong statement must appear in captions")
        # ASS times are output-relative: the first dialogue must start at
        # (first retained word start − optimized start), i.e. the skipped
        # setup is absent from the caption timeline.
        o = opt_of(self.d)
        first_dialogue = next(
            (ln for ln in ass.splitlines() if ln.startswith("Dialogue:")), None)
        self.assertIsNotNone(first_dialogue, "ASS must contain dialogue events")
        start_field = first_dialogue.split(",")[1]  # "H:MM:SS.CC"
        h, m, rest = start_field.split(":")
        secs = int(h) * 3600 + int(m) * 60 + float(rest)
        expected = 39.72 - o["optimizedStartSec"]
        self.assertAlmostEqual(secs, expected, delta=0.05,
                                msg="first caption must sit on the optimized timeline")


# ── M. Framing / DualFrame eligibility ──────────────────────────────────────

class TestCaseM(unittest.TestCase):
    """CASE M: framing (the DualFrame/isolation input) runs on the
    OPTIMIZED range. Kill-switch control run keeps the SOURCE range."""

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(BEAT_VIDEO):
            raise unittest.SkipTest("Beat video not found: " + BEAT_VIDEO)
        # Same fixture as case K: complete setup sentence in real silence.
        words = [
            w("So", 39.0, 39.2), w("yeah.", 39.2, 39.45),
            w("The", 39.72, 39.8), w("biggest", 39.8, 39.95), w("mistake", 39.95, 40.2),
            w("people", 40.2, 40.4), w("make", 40.4, 40.6), w("is", 40.6, 40.7),
            w("overthinking.", 40.7, 41.2),
        ] + body_words(41.5, 16, step=0.4, prefix="real")
        cls.words = words
        cls.d_on = run_inspect(38.92, 48.0, words, source=BEAT_VIDEO)
        cls.d_off = run_inspect(38.92, 48.0, words, source=BEAT_VIDEO,
                                env_extra={"AUTOSHORTS_BOUNDARY_OPTIMIZATION": "0"})

    def test_optimization_fired_non_vacuous(self):
        # Guard: the ON run must actually move the start, otherwise the
        # range-comparison tests below are vacuous.
        o = opt_of(self.d_on)
        self.assertTrue(o["startChanged"], "fixture must trigger optimization")
        self.assertAlmostEqual(o["optimizedStartSec"], 39.57, delta=0.3)

    def test_framing_present_on_optimized_range(self):
        self.assertIsNotNone(self.d_on["framing"],
                             "framing must run on the optimized range")
        self.assertEqual(self.d_on["framing"]["mode"], "single")

    def test_kill_switch_keeps_source_range(self):
        o = opt_of(self.d_off)
        self.assertFalse(o["startChanged"])
        self.assertAlmostEqual(self.d_off["pacingRange"]["startSec"], 38.92, delta=1e-6)
        self.assertAlmostEqual(self.d_off["framingRange"]["startSec"], 38.92, delta=1e-6)

    def test_framing_range_matches_optimized(self):
        o = opt_of(self.d_on)
        self.assertAlmostEqual(self.d_on["framingRange"]["startSec"],
                               o["optimizedStartSec"], delta=1e-6)


# ── N. Isolation (two-speaker) ──────────────────────────────────────────────

class TestCaseN(unittest.TestCase):
    """CASE N: two-speaker isolation — Q→A repair preserved across
    speakers, framing valid on the optimized range."""

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(MESSI_VIDEO):
            raise unittest.SkipTest("Messi video not found: " + MESSI_VIDEO)
        words = [
            w("Why", 60.0, 60.2, "Host"), w("did", 60.2, 60.3, "Host"),
            w("she", 60.3, 60.4, "Host"), w("quit?", 60.4, 60.8, "Host"),
            w("Because", 61.2, 61.5, "Guest"), w("the", 61.5, 61.6, "Guest"),
            w("job", 61.6, 61.8, "Guest"), w("was", 61.8, 61.9, "Guest"),
            w("destroying", 61.9, 62.3, "Guest"), w("her.", 62.3, 62.7, "Guest"),
        ] + body_words(63.0, 40) + [w("end.", 79.0, 79.4)]
        cls.words = words
        cls.d = run_inspect(61.2, 80.0, words, source=MESSI_VIDEO)

    def test_qa_repair_fires(self):
        o = opt_of(self.d)
        self.assertTrue(o["startChanged"], "Q→A repair should fire (cross-speaker)")
        self.assertLessEqual(o["optimizedStartSec"], 60.0 + 1e-9)

    def test_framing_valid(self):
        self.assertIsNotNone(self.d["framing"], "framing must be produced")
        self.assertIn(self.d["framing"]["mode"], ("single", "dual", "dual_stack", "isolation"))


# ── O. Determinism ──────────────────────────────────────────────────────────

class TestCaseO(unittest.TestCase):
    """CASE O: identical inputs must produce identical outputs (two full
    CLI runs, byte-identical optimization JSON)."""

    def test_determinism(self):
        words = [
            w("So", 160.0, 160.2), w("yeah", 160.2, 160.4), w("we", 160.4, 160.5),
            w("were", 160.5, 160.6), w("talking", 160.6, 160.9), w("about", 160.9, 161.1),
            w("that.", 161.1, 161.4),
            w("The", 162.0, 162.1), w("biggest", 162.1, 162.3), w("mistake", 162.3, 162.6),
            w("people", 162.6, 162.8), w("make", 162.8, 163.0), w("is", 163.0, 163.1),
            w("overthinking.", 163.1, 163.6),
            w("So", 164.0, 164.2), w("yeah,", 164.2, 164.4), w("that's", 164.4, 164.6),
            w("basically", 164.6, 164.9), w("it.", 164.9, 165.2),
        ]
        a = run_inspect(160.0, 166.0, words)
        b = run_inspect(160.0, 166.0, words)
        self.assertEqual(json.dumps(a["optimization"], sort_keys=True),
                         json.dumps(b["optimization"], sort_keys=True),
                         "identical inputs must produce identical optimization")
        self.assertEqual(a["summary"], b["summary"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
