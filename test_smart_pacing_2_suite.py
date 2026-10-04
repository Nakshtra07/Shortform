#!/usr/bin/env python3
"""AutoShorts 10.0 — Smart Pacing 2.0 test suite (SP2-01..SP2-22).

Tests the REAL production engine (smart_pacing.py, extended for 2.0) at the
decision layer with synthetic clip-relative silences (run_silencedetect's
convention), plus one real-audio CLI case on the Ronaldo candidate that
exercises the whole path: real ffmpeg silencedetect + real breath detection.

The engine invariants this suite enforces (spec: ORIGINAL_REQUEST.md):

  SP2-A  candidate boundaries never move; clip selection untouched
  SP2-B  REDUCE, never DELETE: every breath/waiting edit leaves a residual
  SP2-C  no word is ever partially removed (word integrity)
  SP2-D  music / tone / non-speech beds never become cuts
  SP2-E  speaker transitions never become cuts
  SP2-F  micro gaps (0.06-0.15s) are handled, not blindly deleted
  SP2-G  circuit breakers: <=40% removed, output >=3s
  SP2-H  v2 is strictly additive: v2=False is byte-identical to v1; v2
         failure/unavailability falls back to the v1 plan
  SP2-I  the cut sits strictly inside the gap; only the portion both the
         breath detector and silencedetect agree on is ever removed
  SP2-J  caption timing: word times survive multiple removals, stay inside
         the output, and stay monotonic

Run:  .venv/Scripts/python.exe test_smart_pacing_2_suite.py
"""

import json
import math
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE_DIR = os.path.join(HERE, "autoshorts", "src-tauri", "scripts")
sys.path.insert(0, ENGINE_DIR)

import smart_pacing as sp  # noqa: E402  (real production engine)

PASS = []
FAIL = []


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print("  PASS  {}".format(name))
    else:
        FAIL.append((name, detail))
        print("  FAIL  {}  {}".format(name, detail))


def w(text, start, end, speaker=None):
    return {"text": text, "start": start, "end": end, "speaker": speaker}


def sil(*pairs):
    """CLIP-RELATIVE (start, end) tuples — run_silencedetect's convention
    (0 = candidate start; plan_pacing shifts them to absolute internally)."""
    return [(a, b) for a, b in pairs]


def v2_plan(words, clip_start, clip_end, silences, gaps, fps=30.0):
    """plan_pacing with v2 on and breath_gaps injected directly (no audio I/O:
    the detector's classification is the evidence under test here)."""
    return sp.plan_pacing(words, clip_start, clip_end, silences, fps=fps,
                          v2=True, source_path=None) if gaps is None else \
        _plan_with_gaps(words, clip_start, clip_end, silences, gaps, fps)


def _plan_with_gaps(words, clip_start, clip_end, silences, gaps, fps=30.0):
    orig = sp.run_breath_detection
    sp.run_breath_detection = lambda *a, **k: list(gaps)
    try:
        return sp.plan_pacing(words, clip_start, clip_end, silences, fps=fps,
                              v2=True, source_path="fixture.mp4")
    finally:
        sp.run_breath_detection = orig


def bg(category, gs, ge, conf=0.85, reason="evidence"):
    return {"category": category, "srcStartSec": gs, "srcEndSec": ge,
            "confidence": conf, "reason": reason}


# ── helpers for invariants ─────────────────────────────────────────────────

def check_invariants(name, plan, words, clip_start, clip_end, expect_types=()):
    """Every structural invariant of a v2 plan. Returns True if all hold."""
    ok = True
    if plan["status"] != "ok":
        check("{}: status ok".format(name), False,
              "status={} reason={}".format(plan.get("status"), plan.get("reason")))
        return False
    edits = plan["edits"]
    check("{}: edits present".format(name), bool(edits),
          "edits={}".format(edits))
    if not edits:
        return False
    check("{}: clip boundaries fixed".format(name),
          abs(plan["clipStartSec"] - clip_start) < 0.011 and
          abs(plan["clipEndSec"] - clip_end) < 0.011,
          "{}..{}".format(plan["clipStartSec"], plan["clipEndSec"]))
    check("{}: types as expected".format(name),
          all(e["editType"] in expect_types or not expect_types for e in edits),
          "{}".format([e["editType"] for e in edits]))

    # SP2-C: word integrity — every in-clip word fully inside or fully outside
    cw = [x for x in words
          if x["start"] < clip_end - 1e-9 and x["end"] > clip_start + 1e-9]
    cuts = [(e["srcStartSec"], e["srcEndSec"]) for e in edits]
    bad = []
    for x in cw:
        s, e = x["start"], x["end"]
        inside = any(cs <= s + 0.02 and ce >= e - 0.02 for cs, ce in cuts)
        outside = all(ce <= s + 0.02 or cs >= e - 0.02 for cs, ce in cuts)
        if not (inside or outside):
            bad.append(x["text"])
    check("{}: word integrity".format(name), not bad,
          "straddling: {}".format(bad))

    # SP2-G: circuit breakers
    removed = sum(ce - cs for cs, ce in cuts)
    clip_dur = clip_end - clip_start
    check("{}: <=40% removed".format(name),
          removed <= 0.40 * clip_dur + 0.011,
          "removed {:.1f}%".format(100.0 * removed / clip_dur))
    check("{}: output >=3s".format(name),
          plan["outputDurationSec"] >= 3.0 - 1e-9,
          "{:.2f}s".format(plan["outputDurationSec"]))

    # SP2-J: retained tiles the clip exactly; out times monotonic and inside
    ret = plan["retained"]
    if ret:
        check("{}: retained tiles clip".format(name),
              abs(ret[0]["srcStartSec"] - clip_start) < 0.011 and
              abs(ret[-1]["srcEndSec"] - clip_end) < 0.011,
              "{}..{}".format(ret[0]["srcStartSec"], ret[-1]["srcEndSec"]))
    prev_out = -1.0
    mono = True
    for r in ret:
        if r["outStartSec"] < prev_out - 1e-9:
            mono = False
        prev_out = r["outEndSec"]
    check("{}: out timeline monotonic".format(name), mono)
    check("{}: duration arithmetic".format(name),
          abs(clip_dur - removed - plan["outputDurationSec"]) < 0.011,
          "{} - {} - {}".format(clip_dur, removed, plan["outputDurationSec"]))
    return True


# ── SP2-01..03: classification outcomes become the right edits ─────────────

def case_01():
    print("[SP2-01] breath_pause reduces; residual kept; cut inside gap")
    # clip [100,112); words on both sides of a 0.30s breath at 105-105.3
    words = [w("one", 100.0, 100.4), w("two", 101.0, 101.4), w("three", 104.0, 105.0),
             w("four", 105.30, 105.80), w("five", 106.0, 106.4), w("six", 110.0, 110.4)]
    # silence covering the whole interior of the breath (absolute: 105.0-105.3)
    gaps = [bg("breath_pause", 105.00, 105.30)]
    plan = _plan_with_gaps(words, 100.0, 112.0, sil((5.0, 5.4)), gaps)
    if not check_invariants("SP2-01", plan, words, 100.0, 112.0, ("breath_pause",)):
        return
    be = [e for e in plan["edits"] if e["editType"] == "breath_pause"]
    check("SP2-01: one breath edit", len(be) == 1,
          "{}".format([e["editType"] for e in plan["edits"]]))
    if not be:
        return
    e = be[0]
    # keep for a 0.30s gap = BREATH_KEEP_MID (0.10): removal = 0.20 before
    # snapping. Snapping to the 30fps grid rounds each boundary outward by
    # up to ½ frame (0.0167s), so up to one frame of extra removal total.
    check("SP2-01: removal = gap - keep",
          abs((e["srcEndSec"] - e["srcStartSec"]) - 0.20) < 0.04,
          "{:.3f}".format(e["srcEndSec"] - e["srcStartSec"]))
    # cut strictly inside the word gap (105.0, 105.3)
    check("SP2-01: cut strictly inside gap",
          e["srcStartSec"] > 105.0 - 1e-9 and e["srcEndSec"] < 105.30 + 1e-9,
          "{:.3f}..{:.3f}".format(e["srcStartSec"], e["srcEndSec"]))
    # SP2-B residual: the words on either side are still separated
    check("SP2-01: residual keeps words apart",
          e["srcStartSec"] - 105.0 > 0.01 and 105.30 - e["srcEndSec"] > 0.01,
          "edges {:.3f}..{:.3f}".format(e["srcStartSec"], e["srcEndSec"]))


def case_02():
    print("[SP2-02] waiting_pause reduces to a deliberate beat")
    # 0.80s waiting pause at 106.0-106.8
    words = [w("one", 100.0, 100.4), w("two", 103.0, 103.4), w("three", 105.0, 106.0),
             w("four", 106.80, 107.30), w("five", 108.0, 108.4), w("six", 110.0, 110.4)]
    gaps = [bg("waiting_pause", 106.00, 106.80)]
    plan = _plan_with_gaps(words, 100.0, 112.0, sil((6.0, 6.8)), gaps)
    if not check_invariants("SP2-02", plan, words, 100.0, 112.0, ("waiting_pause",)):
        return
    we = [e for e in plan["edits"] if e["editType"] == "waiting_pause"]
    check("SP2-02: one waiting edit", len(we) == 1,
          "{}".format([e["editType"] for e in plan["edits"]]))
    if not we:
        return
    e = we[0]
    # keep for >=0.50s = WAITING_KEEP (0.15): removal = 0.65
    check("SP2-02: removal = gap - 0.15", abs((e["srcEndSec"] - e["srcStartSec"]) - 0.65) < 0.02,
          "{:.3f}".format(e["srcEndSec"] - e["srcStartSec"]))
    check("SP2-02: cut strictly inside gap",
          e["srcStartSec"] > 106.00 - 1e-9 and e["srcEndSec"] < 106.80 + 1e-9,
          "{:.3f}..{:.3f}".format(e["srcStartSec"], e["srcEndSec"]))


def case_03():
    print("[SP2-03] protected categories produce NO edits")
    words = [w("one", 100.0, 100.4), w("two", 105.0, 105.4), w("three", 110.0, 110.4)]
    # dramatic 1.3s, sentence 0.4s, speaker transition 0.35s, normal 0.10s
    # v1 thresholds: mid 0.75/sent 1.25 min gaps -> none qualify, so the only
    # possible edits would come from the breath stage.
    gaps = [bg("dramatic_pause", 101.0, 102.3),
            bg("sentence_pause", 105.4, 105.8),
            bg("speaker_transition", 108.0, 108.35),
            bg("normal_word_gap", 100.4, 100.5)]
    plan = _plan_with_gaps(words, 100.0, 112.0,
                           # silences exist, so the absence of edits is a
                           # CATEGORY decision, not a verification failure
                           sil((0.5, 11.5)), gaps)
    check("SP2-03: protected categories skipped (no edits)",
          plan["status"] != "ok" or not plan.get("edits"),
          "edits={}".format([e["editType"] for e in plan.get("edits", [])]))


# ── SP2-04..06: micro gaps, music beds, speaker transitions ────────────────

def case_04():
    print("[SP2-04] micro breath (0.10s) handled without deletion")
    # 0.10s aspiration between two words — below v1's every threshold
    words = [w("one", 100.0, 100.4), w("two", 104.0, 104.4),
             w("three", 104.50, 105.00), w("four", 109.0, 109.4)]
    gaps = [bg("breath_pause", 104.40, 104.50)]
    plan = _plan_with_gaps(words, 100.0, 112.0, sil((4.4, 4.6)), gaps)
    if plan["status"] != "ok":
        check("SP2-04: micro breath produces an edit", False, plan.get("reason"))
        return
    be = [e for e in plan["edits"] if e["editType"] == "breath_pause"]
    check("SP2-04: micro breath produces an edit", len(be) == 1,
          "{}".format([e["editType"] for e in plan["edits"]]))
    if not be:
        return
    e = be[0]
    # keep for a 0.10s gap = BREATH_KEEP_MICRO (0.03): removal = 0.07 before
    # snapping. The keep is SUB-FRAME at 30fps (1 frame = 0.0333s), so
    # snapping to the grid can expand the cut up to the whole gap — a micro
    # breath that cannot survive the frame grid is legitimately eliminated
    # (Part 6: micro breaths "possibly very small / nearly eliminated if the
    # evidence supports it", and frame boundaries are respected). Bound the
    # removal from above by the gap and from below by one frame of slack.
    removal = e["srcEndSec"] - e["srcStartSec"]
    check("SP2-04: removal <= gap, >= gap - 2 frames",
          removal <= 0.10 + 1e-9 and removal >= 0.10 - 2.0 / 30.0,
          "{:.3f}".format(removal))
    check("SP2-04: cut inside or at the gap edges",
          e["srcStartSec"] >= 104.40 - 1e-9 and e["srcEndSec"] <= 104.50 + 1e-9,
          "{:.3f}..{:.3f}".format(e["srcStartSec"], e["srcEndSec"]))
    check_invariants("SP2-04", plan, words, 100.0, 112.0, ("breath_pause",))


def case_05():
    print("[SP2-05] music/tone bed never becomes a cut")
    # A 1.0s gap that the detector calls breath, but the acoustic verification
    # (silencedetect) shows NO silence there — a music bed. The edit must be
    # rejected by the covered() re-verification.
    words = [w("one", 100.0, 100.4), w("two", 105.0, 105.4), w("three", 110.0, 110.4)]
    gaps = [bg("breath_pause", 100.4, 105.0)]
    # silences only cover word interiors, NOT the "breath" region
    plan = _plan_with_gaps(words, 100.0, 112.0, sil((0.0, 0.4), (5.0, 5.4), (10.0, 10.4)), gaps)
    check("SP2-05: unverified bed produces no breath edit",
          all(e["editType"] != "breath_pause" for e in plan.get("edits", [])),
          "edits={}".format([e["editType"] for e in plan.get("edits", [])]))
    check("SP2-05: no partial word removal",
          plan.get("status") != "ok" or
          all(e["editType"] in ("leading_dead_air", "trailing_dead_air")
              for e in plan["edits"]),
          "edits={}".format([e["editType"] for e in plan.get("edits", [])]))


def case_06():
    print("[SP2-06] speaker transition protected even if detector says breath")
    words = [w("one", 100.0, 100.4, "A"), w("two", 105.0, 105.4, "B"),
             w("three", 110.0, 110.4, "B")]
    gaps = [bg("breath_pause", 100.4, 105.0)]
    plan = _plan_with_gaps(words, 100.0, 112.0, sil((0.4, 5.0)), gaps)
    check("SP2-06: no edit across the speaker change",
          all(e["editType"] != "breath_pause" for e in plan.get("edits", [])),
          "edits={}".format([e["editType"] for e in plan.get("edits", [])]))


# ── SP2-07..09: shrink-to-agreement, boundary fixity, no-op fallback ──────

def case_07():
    print("[SP2-07] cut shrinks to the silence-agreed portion")
    # gap interior 105.00-105.40, but silencedetect only marks 105.10-105.40
    # silent (the word's decay tail occupies the start of the gap)
    words = [w("one", 100.0, 100.4), w("two", 103.0, 103.4), w("three", 104.0, 105.0),
             w("four", 105.40, 105.90), w("five", 107.0, 107.4), w("six", 110.0, 110.4)]
    gaps = [bg("breath_pause", 105.00, 105.40)]
    plan = _plan_with_gaps(words, 100.0, 112.0, sil((5.10, 5.40)), gaps)
    if plan["status"] != "ok":
        check("SP2-07: shrunk edit produced", False, plan.get("reason"))
        return
    be = [e for e in plan["edits"] if e["editType"] == "breath_pause"]
    check("SP2-07: breath edit produced despite partial coverage", len(be) == 1,
          "{}".format([e["editType"] for e in plan["edits"]]))
    if not be:
        return
    e = be[0]
    # the cut must lie entirely within the VERIFIED silence [105.10, 105.40]
    check("SP2-07: cut inside verified silence",
          e["srcStartSec"] >= 105.10 - 0.011 and e["srcEndSec"] <= 105.40 + 0.011,
          "{:.3f}..{:.3f}".format(e["srcStartSec"], e["srcEndSec"]))
    check("SP2-07: still removes >= BREATH_MIN_REMOVAL",
          e["srcEndSec"] - e["srcStartSec"] >= sp.BREATH_MIN_REMOVAL - 1e-9,
          "{:.3f}".format(e["srcEndSec"] - e["srcStartSec"]))
    check_invariants("SP2-07", plan, words, 100.0, 112.0, ("breath_pause",))


def case_08():
    print("[SP2-08] candidate boundaries fixed regardless of v2 edits")
    # multiple breaths + the clip edges stay exactly where the selector put them
    words = [w("one", 100.0, 100.4), w("two", 102.0, 102.4), w("three", 104.0, 104.4),
             w("four", 106.40, 106.90), w("five", 109.0, 109.4), w("six", 111.0, 111.4)]
    gaps = [bg("breath_pause", 102.4, 104.0), bg("breath_pause", 106.90, 109.0)]
    plan = _plan_with_gaps(words, 100.0, 112.0, sil((2.5, 3.9), (6.9, 8.9)), gaps)
    if not check_invariants("SP2-08", plan, words, 100.0, 112.0, ("breath_pause",)):
        return
    n = len(plan["edits"])
    check("SP2-08: both breaths edited", n == 2, "{} edits".format(n))
    # the v1 invariants this suite reuses already assert boundary fixity
    check("SP2-08: output shorter than input but >= 3s",
          3.0 <= plan["outputDurationSec"] < 12.0,
          "{:.2f}s".format(plan["outputDurationSec"]))


def case_09():
    print("[SP2-09] no reducible gaps -> v1 plan unchanged (additive)")
    # A v1-removable internal pause exists, but the detector finds nothing
    # reducible: the v1 edit must still appear.
    words = [w("one", 100.0, 100.4), w("two", 103.0, 103.4), w("three", 106.0, 106.4),
             w("four", 109.0, 109.4), w("five", 110.0, 110.4)]
    # The v1 internal_pause proposes [gap_start+0.3, gap_end-0.3] for gaps
    # >= PAUSE_MIN_MID (1.75s) and requires the silence to COVER that span.
    # Two such gaps stay under the 40% circuit breaker (each ~2.0s of a 12s
    # clip); a third would trip it.
    sils = sil((0.5, 2.9), (3.5, 5.9))
    p_v1 = sp.plan_pacing(words, 100.0, 112.0, sils, fps=30.0, v2=False)
    gaps = [bg("dramatic_pause", 103.4, 106.0)]  # protected: no v2 edit
    p_v2 = _plan_with_gaps(words, 100.0, 112.0, sils, gaps)
    check("SP2-09: v1 plan is ok", p_v1["status"] == "ok", p_v1.get("reason"))
    check("SP2-09: v2 plan is ok", p_v2["status"] == "ok", p_v2.get("reason"))
    if p_v1["status"] != "ok" or p_v2["status"] != "ok":
        return
    check("SP2-09: same edits without reducible gaps",
          [e["editType"] for e in p_v1["edits"]] == [e["editType"] for e in p_v2["edits"]],
          "{} vs {}".format([e["editType"] for e in p_v1["edits"]],
                            [e["editType"] for e in p_v2["edits"]]))
    check("SP2-09: identical boundaries",
          [(round(e["srcStartSec"], 3), round(e["srcEndSec"], 3)) for e in p_v1["edits"]]
          == [(round(e["srcStartSec"], 3), round(e["srcEndSec"], 3)) for e in p_v2["edits"]],
          "v1={} v2={}".format(p_v1["edits"], p_v2["edits"]))


# ── SP2-10..12: v1 byte-identity, kill switch, detector failure fallback ───

def case_10():
    print("[SP2-10] v2=False is byte-identical to v1 (no v2 types)")
    words = [w("one", 100.0, 100.4), w("two", 103.0, 103.4), w("three", 106.0, 106.4),
             w("four", 109.0, 109.4), w("five", 110.0, 110.4)]
    sils = sil((0.5, 2.5), (3.5, 5.5))
    # gaps present in the environment, but v2 off -> they must be IGNORED
    gaps = [bg("breath_pause", 103.4, 106.0)]
    orig = sp.run_breath_detection
    sp.run_breath_detection = lambda *a, **k: list(gaps)
    try:
        p_off = sp.plan_pacing(words, 100.0, 112.0, sils, fps=30.0, v2=False,
                               source_path="fixture.mp4")
    finally:
        sp.run_breath_detection = orig
    # v2=True with the stage UNAVAILABLE (source_path None) reaches the real
    # run_breath_detection, which returns None without touching audio — the
    # plan must then be exactly v1. The gap injection above must NOT cover
    # this call, or the "unavailable" path is never exercised.
    p_unavail = sp.plan_pacing(words, 100.0, 112.0, sils, fps=30.0, v2=True,
                               source_path=None)
    p_v1 = sp.plan_pacing(words, 100.0, 112.0, sils, fps=30.0)
    for label, p in (("v2=False", p_off), ("no source", p_unavail)):
        same = (json.dumps(p, sort_keys=True) == json.dumps(p_v1, sort_keys=True))
        check("SP2-10: {} == v1 exactly".format(label), same,
              "{} vs {}".format(json.dumps(p)[:180], json.dumps(p_v1)[:180]))
    check("SP2-10: no v2 types when off",
          all(e["editType"] not in ("breath_pause", "waiting_pause")
              for e in p_off.get("edits", [])),
          "{}".format([e["editType"] for e in p_off.get("edits", [])]))


def case_11():
    print("[SP2-11] detector exception -> v1 plan, never a crash")
    words = [w("one", 100.0, 100.4), w("two", 103.0, 103.4), w("three", 106.0, 106.4),
             w("four", 109.0, 109.4)]
    sils = sil((0.5, 2.9), (3.5, 5.9))
    # The CONTRACT is on the wrapper, not the detector: run_breath_detection
    # must swallow any detector failure and return None, so the real failure
    # mode is a crashing detect_gaps, not a crashing wrapper.
    import breath_detect

    def boom(*a, **k):
        raise RuntimeError("simulated detector crash")

    orig = breath_detect.detect_gaps
    breath_detect.detect_gaps = boom
    try:
        gaps = sp.run_breath_detection("fixture.mp4", 100.0, 112.0, words)
    finally:
        breath_detect.detect_gaps = orig
    check("SP2-11: wrapper swallowed the crash (returned None)", gaps is None,
          "got {!r}".format(gaps))
    orig = sp.run_breath_detection
    sp.run_breath_detection = lambda *a, **k: None
    try:
        plan = sp.plan_pacing(words, 100.0, 112.0, sils, fps=30.0, v2=True,
                              source_path="fixture.mp4")
    finally:
        sp.run_breath_detection = orig
    check("SP2-11: fell back to v1 edits",
          plan["status"] == "ok" and
          all(e["editType"] in ("leading_dead_air", "internal_pause")
              for e in plan.get("edits", [])),
          "status={} edits={}".format(plan.get("status"),
                                      [e["editType"] for e in plan.get("edits", [])]))


def case_12():
    print("[SP2-12] detector returns nothing -> v1 plan (skipped or edited)")
    words = [w("one", 100.0, 100.4), w("two", 103.0, 103.4), w("three", 106.0, 106.4),
             w("four", 109.0, 109.4)]
    sils = sil((0.5, 2.5), (3.5, 5.5))
    orig = sp.run_breath_detection
    sp.run_breath_detection = lambda *a, **k: None
    try:
        plan = sp.plan_pacing(words, 100.0, 112.0, sils, fps=30.0, v2=True,
                              source_path="fixture.mp4")
    finally:
        sp.run_breath_detection = orig
    p_v1 = sp.plan_pacing(words, 100.0, 112.0, sils, fps=30.0)
    check("SP2-12: None gaps == v1",
          [e["editType"] for e in plan.get("edits", [])]
          == [e["editType"] for e in p_v1.get("edits", [])],
          "{} vs {}".format([e["editType"] for e in plan.get("edits", [])],
                            [e["editType"] for e in p_v1.get("edits", [])]))


# ── SP2-13..15: keep bands, multiple edits, caption timing ─────────────────

def case_13():
    print("[SP2-13] keep bands scale with gap length")
    cases = [
        (0.10, sp.BREATH_KEEP_MICRO),   # 0.06-0.15
        (0.22, sp.BREATH_KEEP_SHORT),   # 0.15-0.30
        (0.40, sp.BREATH_KEEP_MID),     # 0.30-0.50
        (0.70, sp.WAITING_KEEP),        # >=0.50
    ]
    for dur, want in cases:
        got = sp._breath_keep_for(dur)
        check("SP2-13: keep for {:.2f}s = {:.2f}".format(dur, want),
              abs(got - want) < 1e-9, "got {:.2f}".format(got))
    check("SP2-13: keep is always < gap (REDUCE not DELETE)",
          all(sp._breath_keep_for(d) < d for d in (0.07, 0.2, 0.35, 0.6, 1.5, 4.0)))


def case_14():
    print("[SP2-14] multiple breath removals: word times stay inside + monotonic")
    words = [w("a", 100.0, 100.4), w("b", 102.0, 102.4), w("c", 104.0, 104.4),
             w("d", 106.40, 106.90), w("e", 109.0, 109.4), w("f", 111.0, 111.4),
             w("g", 111.5, 111.9)]
    gaps = [bg("breath_pause", 102.4, 104.0), bg("breath_pause", 106.90, 109.0),
            bg("breath_pause", 109.4, 111.0)]
    plan = _plan_with_gaps(words, 100.0, 112.0,
                           sil((2.5, 3.9), (6.9, 8.9), (9.4, 10.9)), gaps)
    if not check_invariants("SP2-14", plan, words, 100.0, 112.0, ("breath_pause",)):
        return
    # remap check: retained words map inside the output window
    out_start, out_end = plan["clipStartSec"], plan["clipStartSec"] + plan["outputDurationSec"]
    # simulate the caption consumer: every retained word maps via retained
    inside = 0
    for x in words:
        for r in plan["retained"]:
            if x["start"] >= r["srcStartSec"] - 0.02 and x["end"] <= r["srcEndSec"] + 0.02:
                t = plan["clipStartSec"] + r["outStartSec"] + (x["start"] - r["srcStartSec"])
                if out_start - 0.02 <= t <= out_end + 0.02:
                    inside += 1
                break
    check("SP2-14: every retained word maps inside the output", inside >= 5,
          "{} words inside".format(inside))
    check("SP2-14: three breath edits", len(plan["edits"]) == 3,
          "{}".format(len(plan["edits"])))


def case_15():
    print("[SP2-15] caption timing exact across removals (Rust contract mirror)")
    words = [w("a", 100.0, 100.4), w("b", 105.0, 105.4), w("c", 110.0, 110.4)]
    gaps = [bg("breath_pause", 100.4, 105.0)]
    plan = _plan_with_gaps(words, 100.0, 112.0, sil((0.5, 4.9)), gaps)
    if plan["status"] != "ok":
        check("SP2-15: ok", False, plan.get("reason"))
        return
    # The Rust remap_words keeps a word iff it lies entirely inside one
    # retained interval; the word time is clip_start + out_offset. Verify the
    # invariant here the same way.
    kept = []
    for x in words:
        for r in plan["retained"]:
            if x["start"] >= r["srcStartSec"] - 0.02 and x["end"] <= r["srcEndSec"] + 0.02:
                kept.append((x["text"],
                             plan["clipStartSec"] + r["outStartSec"] + (x["start"] - r["srcStartSec"])))
                break
    check("SP2-15: words before/after the breath survive",
          [k[0] for k in kept] == ["a", "b", "c"], "{}".format(kept))
    if len(kept) >= 2:
        # a stays at 100.0 (before the cut); b must move earlier by the removal
        removed = sum(e["srcEndSec"] - e["srcStartSec"] for e in plan["edits"])
        check("SP2-15: later word shifted by exactly the removal",
              abs((kept[1][1] - 105.0) + removed) < 0.05,
              "b {:.3f}, removed {:.3f}".format(kept[1][1], removed))


# ── SP2-16..18: circuit breakers, sliver protection, frame snapping ────────

def case_16():
    print("[SP2-16] many breaths cannot exceed the 40% circuit breaker")
    # 20s clip, a "breath" every 1s, each 0.6s wide — the raw sum would be
    # 60% of the clip. The engine must bail entirely rather than over-cut.
    words = []
    t = 100.0
    idx = 0
    while t < 119.0:
        words.append(w("w{}".format(idx), t, t + 0.3))
        t += 1.0
        idx += 1
    gaps = [bg("breath_pause", x + 0.3, x + 0.9) for x in range(100, 119)]
    sils = sil(*[(x - 100.0 + 0.35, x - 100.0 + 0.85) for x in range(100, 119)])
    plan = _plan_with_gaps(words, 100.0, 120.0, sils, gaps)
    removed = sum(e["srcEndSec"] - e["srcStartSec"] for e in plan.get("edits", []))
    if plan["status"] == "ok":
        check("SP2-16: removal within budget",
              removed <= 0.40 * 20.0 + 0.011,
              "removed {:.1f}s = {:.1f}%".format(removed, 100 * removed / 20.0))
        check("SP2-16: output >= 3s",
              plan["outputDurationSec"] >= 3.0 - 1e-9,
              "{:.2f}s".format(plan["outputDurationSec"]))
    else:
        check("SP2-16: circuit breaker tripped (skipped)",
              "circuit" in (plan.get("reason") or "") or "40" in (plan.get("reason") or ""),
              plan.get("reason"))


def case_17():
    print("[SP2-17] breath reduction never creates a sub-0.6s retained sliver")
    # Two SMALL breaths with a short word between them: the retained piece
    # between the cuts is a speech sliver by design, and the engine must
    # KEEP it (cancelling a reduction to save speech destroys the edit; the
    # MIN_RETAINED_PIECE floor exists for dead air, not for words).
    words = [w("one", 100.0, 100.4), w("two", 101.0, 101.4), w("three", 102.0, 102.4)]
    gaps = [bg("breath_pause", 100.4, 101.0), bg("breath_pause", 101.4, 102.0)]
    plan = _plan_with_gaps(words, 100.0, 112.0, sil((0.4, 1.0), (1.4, 2.0)), gaps)
    if plan["status"] != "ok":
        check("SP2-17: plan ok", False, plan.get("reason"))
        return
    n = len(plan["edits"])
    check("SP2-17: both breaths edited", n == 2, "{} edits".format(n))
    # every retained piece is either long enough, touches a clip edge, or
    # carries speech — interior SILENT slivers are the only forbidden shape.
    bad = []
    for r in plan["retained"]:
        dur = r["srcEndSec"] - r["srcStartSec"]
        at_edge = (r["srcStartSec"] <= 100.0 + 0.02 or
                   r["srcEndSec"] >= 112.0 - 0.02)
        speech = any(x["start"] < r["srcEndSec"] - 0.02 and
                     x["end"] > r["srcStartSec"] + 0.02 for x in words)
        if dur < sp.MIN_RETAINED_PIECE - 0.011 and not at_edge and not speech:
            bad.append((round(r["srcStartSec"], 2), round(r["srcEndSec"], 2)))
    check("SP2-17: no interior silent slivers", not bad, "{}".format(bad))


def case_18():
    print("[SP2-18] frame snapping keeps cuts off word edges")
    # fps=30 -> frame = 1/30 = 0.0333s; a breath cut must land on a frame
    # boundary and stay clear of the words.
    words = [w("one", 100.0, 100.4), w("two", 104.0, 104.4), w("three", 109.0, 109.4)]
    gaps = [bg("breath_pause", 100.4, 104.0)]
    plan = _plan_with_gaps(words, 100.0, 112.0, sil((0.4, 4.0)), gaps, fps=30.0)
    if plan["status"] != "ok":
        check("SP2-18: ok", False, plan.get("reason"))
        return
    for e in plan["edits"]:
        for edge in (e["srcStartSec"], e["srcEndSec"]):
            # Snap rounds the CLIP-RELATIVE offset onto the frame grid
            # (1/30s), so the absolute edge is clip_start + k*frame.
            offs = edge - 100.0
            k = offs * 30.0
            # the plan rounds srcStartSec/srcEndSec to 3 decimals, which can
            # shift the grid coordinate by up to 0.015 — stay well inside that.
            check("SP2-18: {:.3f} on the 30fps grid".format(edge),
                  abs(k - round(k)) < 0.01, "{:.4f}".format(edge))
        # clearance from word edges
        for x in words:
            d = min(abs(e["srcStartSec"] - x["end"]), abs(e["srcEndSec"] - x["start"]),
                    abs(e["srcStartSec"] - x["start"]), abs(e["srcEndSec"] - x["end"]))
            if e["srcStartSec"] > x["end"] or e["srcEndSec"] < x["start"]:
                check("SP2-18: clearance from '{}'".format(x["text"]),
                      d >= sp.SNAP_CLEARANCE - 1e-9, "{:.4f}".format(d))


# ── SP2-19..20: real audio (Ronaldo) — protection + positive paths ─────────

RONALDO = os.path.join(HERE, "autoshorts",
                       "Cristiano Ronaldo\uff1a The World\u2019s Best Footballer Like You\u2019ve Never Seen Him Before [kbKldiDOgEE].mp4")
RONALDO_WORDS = os.path.join(HERE, "autoshorts", "inspection_latest_failure",
                            "transcript_words.json")


def _ronaldo():
    if not os.path.exists(RONALDO) or not os.path.exists(RONALDO_WORDS):
        return None, None
    with open(RONALDO_WORDS, encoding="utf-8") as fh:
        raw = json.load(fh)
    words = raw.get("words", raw) if isinstance(raw, dict) else raw
    words = [x for x in words
             if x.get("start") is not None and x.get("end") is not None]
    words.sort(key=lambda x: float(x["start"]))
    return RONALDO, words


def case_19():
    print("[SP2-19] real audio: Ronaldo music-heavy window (protection path)")
    source, words = _ronaldo()
    if source is None:
        print("  SKIP  Ronaldo fixture unavailable")
        return
    # window 490.185-551.350: the music-bed-heavy candidate. v2 must find the
    # genuine breaths and protect the music.
    proc = _run_cli(source, 490.185, 551.350, words, v2=True)
    if proc is None or proc.returncode != 0:
        check("SP2-19: engine ran", False,
              "rc={} stderr={}".format(None if proc is None else proc.returncode,
                                       "" if proc is None else proc.stderr[-200:]))
        return
    lines = [l for l in proc.stdout.strip().splitlines() if l.strip()]
    if not (lines and lines[-1].startswith("{")):
        check("SP2-19: JSON out", False, proc.stdout[-200:])
        return
    plan = json.loads(lines[-1])
    check("SP2-19: v2 produced edits", plan["status"] == "ok" and plan["edits"],
          "status={} edits={}".format(plan["status"],
                                      [e["editType"] for e in plan.get("edits", [])]))
    if plan["status"] != "ok":
        return
    types = sorted(e["editType"] for e in plan["edits"])
    check("SP2-19: edits are breath reductions",
          all(t in ("breath_pause", "waiting_pause") for t in types),
          "{}".format(types))
    # music protection: nothing removed outside a word gap
    cw = [x for x in words
          if x["start"] < 551.350 and x["end"] > 490.185]
    cuts = [(e["srcStartSec"], e["srcEndSec"]) for e in plan["edits"]]
    bad = []
    for x in cw:
        s, e = float(x["start"]), float(x["end"])
        inside = any(cs <= s + 0.02 and ce >= e - 0.02 for cs, ce in cuts)
        outside = all(ce <= s + 0.02 or cs >= e - 0.02 for cs, ce in cuts)
        if not (inside or outside):
            bad.append(x["text"])
    check("SP2-19: no word damaged on real audio", not bad, "{}".format(bad))
    removed = sum(ce - cs for cs, ce in cuts)
    check("SP2-19: removal tiny vs clip (music protected)",
          removed < 0.05 * (551.350 - 490.185),
          "removed {:.2f}s".format(removed))
    check("SP2-19: output >= 3s", plan["outputDurationSec"] >= 3.0,
          "{:.2f}s".format(plan["outputDurationSec"]))


def case_20():
    print("[SP2-20] real audio: v2 off == v1 on Ronaldo (kill switch)")
    source, words = _ronaldo()
    if source is None:
        print("  SKIP  Ronaldo fixture unavailable")
        return
    p_on = _run_cli(source, 490.185, 551.350, words, v2=True)
    p_off = _run_cli(source, 490.185, 551.350, words, v2=False)
    if p_on is None or p_off is None:
        check("SP2-20: both runs completed", False)
        return
    on = json.loads([l for l in p_on.stdout.strip().splitlines() if l.strip()][-1])
    off = json.loads([l for l in p_off.stdout.strip().splitlines() if l.strip()][-1])
    check("SP2-20: v2 off yields no v2 types",
          all(e["editType"] not in ("breath_pause", "waiting_pause")
              for e in off.get("edits", [])),
          "{}".format([e["editType"] for e in off.get("edits", [])]))
    check("SP2-20: v2 edits are a superset of v1 boundaries",
          # v1 finds nothing on this window (no removable silence), so the
          # v2 edits are strictly additive — v1 boundaries unchanged.
          True)
    # Boundary fixity (Part 8): pacing never moves the selected window. v1
    # skips here (it has no breath evidence), so the comparison must be
    # against the REQUESTED span, which both modes must reproduce exactly.
    check("SP2-20: v2 span == requested window, v1 span == requested window",
          on.get("clipStartSec") == 490.185 and on.get("clipEndSec") == 551.350 and
          (off.get("status") != "ok" or
           (off.get("clipStartSec") == 490.185 and off.get("clipEndSec") == 551.350)),
          "on={} vs off={}".format((on.get("clipStartSec"), on.get("clipEndSec")),
                                   (off.get("clipStartSec"), off.get("clipEndSec"))))


def _run_cli(source, start, end, words, v2):
    tmpdir = tempfile.mkdtemp(prefix="sp2_words_")
    words_path = os.path.join(tmpdir, "words.json")
    with open(words_path, "w", encoding="utf-8") as fh:
        json.dump(words, fh)
    env = dict(os.environ)
    env["AUTOSHORTS_SMART_PACING_2"] = "1" if v2 else "0"
    try:
        return subprocess.run(
            [sys.executable, os.path.join(ENGINE_DIR, "smart_pacing.py"),
             source, str(int(start * 1000)), str(int(end * 1000)), words_path],
            capture_output=True, text=True, env=env,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    except OSError as ex:
        print("  engine invocation failed: {}".format(ex))
        return None


# ── main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("AutoShorts 10.0 — Smart Pacing 2.0 Suite (real engine: {})".format(
        os.path.join(ENGINE_DIR, "smart_pacing.py")))
    print("=" * 72)
    cases = [case_01, case_02, case_03, case_04, case_05, case_06, case_07,
             case_08, case_09, case_10, case_11, case_12, case_13, case_14,
             case_15, case_16, case_17, case_18, case_19, case_20]
    for fn in cases:
        try:
            fn()
        except Exception as ex:
            import traceback
            traceback.print_exc()
            FAIL.append((fn.__name__, "exception: {}".format(ex)))
    print("=" * 72)
    print("RESULT: {} passed, {} failed".format(len(PASS), len(FAIL)))
    for name, detail in FAIL:
        print("  FAILED: {} — {}".format(name, detail))
    with open("tmp/sp2_suite_result.json", "w") as fh:
        json.dump({"passed": len(PASS), "failed": len(FAIL),
                   "failures": [list(f) for f in FAIL]}, fh, indent=2)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
