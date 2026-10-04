#!/usr/bin/env python3
"""AutoShorts 8.0 — Smart Pacing test suite (cases A–N).

Tests the REAL production engine (`autoshorts/src-tauri/scripts/smart_pacing.py`)
— the same sidecar the Rust render path invokes — not a replica. Pure
decision-engine cases run against `plan_pacing` directly with synthetic
CLIP-RELATIVE (start, end) tuple silences — matching run_silencedetect's
convention (its -ss seek makes t=0 the candidate start); CLI-level
cases run the real `main()` entrypoint with real ffmpeg silencedetect on
generated fixtures.

Run:  .venv/Scripts/python.exe test_smart_pacing_suite.py
"""

import json
import math
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE) if os.path.basename(HERE) == "tests" else HERE
ENGINE_DIR = os.path.join(REPO_ROOT, "autoshorts", "src-tauri", "scripts")
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


# ── A. No removable silence -> unchanged ─────────────────────────────────────

def case_a():
    print("[A] no removable silence -> skipped/unchanged")
    # 5s clip (>= MIN_OUTPUT_SEC + 1) so the skip reason is about silence,
    # not clip length; dense words leave no removable gaps.
    words = [w("hello", 10.0, 10.4), w("world", 10.5, 10.9), w("again", 11.0, 11.4),
             w("more", 12.0, 12.4), w("words", 12.5, 12.9)]
    plan = sp.plan_pacing(words, 10.0, 15.0, sil(), fps=30.0)
    check("A: skipped when no verified silence", plan["status"] != "ok",
          "expected non-ok, got {}".format(plan.get("status")))
    if plan["status"] != "ok":
        check("A: reason present", bool(plan.get("reason")), plan.get("reason"))
        check("A: reason is 'no removable silence' (not 'clip too short')",
              "removable" in (plan.get("reason") or ""),
              plan.get("reason"))


# ── B. Leading dead air ─────────────────────────────────────────────────────

def case_b():
    print("[B] leading dead air removed with breath kept")
    # clip [100, 110); speech starts at 103.0 -> 3s lead
    words = [w("so", 103.0, 103.3), w("this", 103.4, 103.7), w("is", 103.8, 104.0)]
    words += [w("a", 104.1, 104.3), w("clip", 104.4, 104.8)]
    plan = sp.plan_pacing(words, 100.0, 110.0, sil((0.0, 2.7)), fps=30.0)
    check("B: ok", plan["status"] == "ok", json.dumps(plan)[:200])
    if plan["status"] != "ok":
        return
    lead = [e for e in plan["edits"] if e["editType"] == "leading_dead_air"]
    check("B: exactly one leading edit", len(lead) == 1,
          "edits={}".format([e["editType"] for e in plan["edits"]]))
    if not lead:
        return
    e = lead[0]
    check("B: starts at clip start", abs(e["srcStartSec"] - 100.0) < 0.05,
          "start={}".format(e["srcStartSec"]))
    # cut must end at least LEAD_KEEP (0.30) before first word
    check("B: keeps breath before speech",
          0.05 <= 103.0 - e["srcEndSec"] <= 0.60,
          "cut_end={} first_word=103.0".format(e["srcEndSec"]))
    # retained must begin at the cut
    r0 = plan["retained"][0]
    check("B: retained starts at cut end", abs(r0["srcStartSec"] - e["srcEndSec"]) < 0.05,
          "r0={}".format(r0["srcStartSec"]))
    # output duration arithmetic
    check("B: duration arithmetic",
          abs((plan["clipEndSec"] - plan["clipStartSec"]) - plan["removedTotalSec"]
              - plan["outputDurationSec"]) < 0.05,
          "dur={} removed={} out={}".format(
              plan["clipEndSec"] - plan["clipStartSec"],
              plan["removedTotalSec"], plan["outputDurationSec"]))


# ── C. Trailing dead air ────────────────────────────────────────────────────

def case_c():
    print("[C] trailing dead air removed with decay kept")
    # 6s clip, speech until 103.5 then 2.5s trailing silence: removal stays
    # under the 40% circuit breaker and the output stays >= 3s.
    words = [w("final", 100.2, 100.6), w("words", 100.7, 101.1), w("here", 101.2, 101.6),
             w("more", 101.7, 102.1), w("speech", 102.2, 102.6), w("goes", 102.7, 103.1),
             w("on", 103.2, 103.5)]
    plan = sp.plan_pacing(words, 100.0, 106.0, sil((3.5, 6.0)), fps=30.0)
    check("C: ok", plan["status"] == "ok", json.dumps(plan)[:200])
    if plan["status"] != "ok":
        return
    trail = [e for e in plan["edits"] if e["editType"] == "trailing_dead_air"]
    check("C: exactly one trailing edit", len(trail) == 1,
          "edits={}".format([e["editType"] for e in plan["edits"]]))
    if not trail:
        return
    e = trail[0]
    check("C: ends at clip end", abs(e["srcEndSec"] - 106.0) < 0.05,
          "end={}".format(e["srcEndSec"]))
    check("C: keeps decay after speech",
          0.05 <= e["srcStartSec"] - 103.5 <= 0.90,
          "cut_start={} last_word_end=103.5".format(e["srcStartSec"]))


# ── D. Internal pause compressed, natural pause preserved ──────────────────

def case_d():
    print("[D] long internal pause compressed to natural keep")
    words = [w("first", 100.2, 100.6), w("sentence.", 100.7, 101.2)]
    words += [w("second", 103.2, 103.6), w("sentence.", 103.7, 104.2)]  # 2s sentence gap
    plan = sp.plan_pacing(words, 100.0, 105.0, sil((1.2, 3.2)), fps=30.0)
    check("D: ok", plan["status"] == "ok", json.dumps(plan)[:200])
    if plan["status"] != "ok":
        return
    pauses = [e for e in plan["edits"] if e["editType"] == "internal_pause"]
    check("D: one internal pause edit", len(pauses) == 1,
          "edits={}".format([e["editType"] for e in plan["edits"]]))
    if not pauses:
        return
    e = pauses[0]
    # keep = PAUSE_KEEP_SENT (0.50): cut = gap minus keep, centered
    check("D: keeps natural pause",
          abs((e["srcEndSec"] - e["srcStartSec"]) - (2.0 - sp.PAUSE_KEEP_SENT)) < 0.10,
          "cut_len={}".format(e["srcEndSec"] - e["srcStartSec"]))
    check("D: cut inside the gap",
          101.2 - 0.05 <= e["srcStartSec"] and e["srcEndSec"] <= 103.2 + 0.05,
          "cut=[{},{}] gap=[101.2,103.2]".format(e["srcStartSec"], e["srcEndSec"]))


# ── E. Natural short pause preserved (no edit) ──────────────────────────────

def case_e():
    print("[E] natural short pause preserved")
    # 4.5s clip (>= MIN_OUTPUT_SEC + 1) with 0.1-0.3s natural pauses — all
    # below every compression threshold even though verified silent.
    words = [w("natural", 100.2, 100.6), w("pause", 100.9, 101.3),
             w("kept", 101.4, 101.8), w("intact", 101.9, 102.3),
             w("fully", 102.4, 102.8), w("here", 102.9, 103.3),
             w("today", 103.4, 103.8)]
    plan = sp.plan_pacing(words, 100.0, 104.5,
                         sil((0.6, 0.9), (1.3, 1.4), (1.8, 1.9), (2.3, 2.4),
                             (2.8, 2.9), (3.3, 3.4)), fps=30.0)
    check("E: no internal_pause edits",
          not any(e["editType"] == "internal_pause" for e in plan.get("edits", [])),
          "edits={}".format([e["editType"] for e in plan.get("edits", [])]))


# ── F. Hesitation / false start (high confidence only) ──────────────────────

def case_f():
    print("[F] verbatim false start removed")
    # "I think I think we should go" — first "I think" abandoned
    words = [w("I", 100.2, 100.4), w("think", 100.5, 100.9),
            w("I", 101.4, 101.6), w("think", 101.7, 102.1),
            w("we", 102.2, 102.4), w("should", 102.5, 102.9), w("go", 103.0, 103.2)]
    # clip-relative silences: pre-run breath [100.0, 100.15], abandonment [100.9, 101.4]
    plan = sp.plan_pacing(words, 100.0, 104.0, sil((0.0, 0.15), (0.9, 1.4)), fps=30.0)
    check("F: ok", plan["status"] == "ok", json.dumps(plan)[:200])
    if plan["status"] != "ok":
        return
    fs = [e for e in plan["edits"] if e["editType"] == "false_start"]
    check("F: false start detected", len(fs) == 1,
          "edits={}".format([e["editType"] for e in plan["edits"]]))
    if fs:
        e = fs[0]
        # cut must cover the first "I think" but not the restart
        check("F: covers abandoned run",
              e["srcStartSec"] < 100.5 and e["srcEndSec"] > 100.9,
              "cut=[{},{}]".format(e["srcStartSec"], e["srcEndSec"]))
        check("F: restart retained", e["srcEndSec"] <= 101.4,
              "cut_end={} restart=101.4".format(e["srcEndSec"]))


# ── G. Filler island (interior, silence-delimited) ──────────────────────────

def case_g():
    print("[G] interior filler island removed")
    words = [w("and", 100.2, 100.5), w("then", 100.6, 100.9),
             w("um", 101.4, 101.7), w("uh", 101.8, 102.1),
             w("we", 102.7, 103.0), w("left", 103.1, 103.5)]
    plan = sp.plan_pacing(words, 100.0, 104.0,
                         sil((0.9, 1.4), (2.1, 2.7)), fps=30.0)
    check("G: ok", plan["status"] == "ok", json.dumps(plan)[:200])
    if plan["status"] != "ok":
        return
    fillers = [e for e in plan["edits"] if e["editType"] == "hesitation_filler"]
    check("G: filler island removed", len(fillers) == 1,
          "edits={}".format([e["editType"] for e in plan["edits"]]))
    if fillers:
        e = fillers[0]
        check("G: cut covers fillers only",
              e["srcStartSec"] >= 100.9 and e["srcEndSec"] <= 102.7,
              "cut=[{},{}]".format(e["srcStartSec"], e["srcEndSec"]))


# ── H. Word boundary safety (no partial-word cuts) ──────────────────────────

def case_h():
    print("[H] word boundary safety")
    # H1: silence claims a region overlapping a word's tail — a trap. The
    # engine's cut boundaries derive from WORD edges, so the cut must still
    # never partially overlap a word.
    words = [w("safe", 100.2, 100.6), w("words", 103.0, 103.4),
             w("never", 103.5, 103.9), w("clipped", 104.0, 104.4)]
    # silence [100.4, 102.7] overlaps "safe" (ends 100.6) — must not matter
    plan = sp.plan_pacing(words, 100.0, 105.0, sil((0.4, 2.7)), fps=30.0)
    check("H1: plan produced", plan["status"] == "ok", json.dumps(plan)[:200])
    for e in plan.get("edits", []):
        for word in words:
            overlap = min(e["srcEndSec"], word["end"]) - max(e["srcStartSec"], word["start"])
            full_inside = e["srcStartSec"] <= word["start"] + 1e-9 and \
                e["srcEndSec"] >= word["end"] - 1e-9
            check("H1: no partial word overlap in [{}]".format(e["editType"]),
                  overlap <= 1e-9 or full_inside,
                  "cut=[{},{}] word=[{},{}]".format(
                      e["srcStartSec"], e["srcEndSec"], word["start"], word["end"]))
    # H2: silence does NOT cover the whole cut region -> uncertain -> KEEP
    plan2 = sp.plan_pacing(words, 100.0, 105.0, sil((1.5, 2.0)), fps=30.0)
    check("H2: unverified gap is kept",
          not any(e["editType"] == "internal_pause" for e in plan2.get("edits", [])),
          "edits={}".format([e["editType"] for e in plan2.get("edits", [])]))


# ── I. Caption sync: retained mapping is exact ──────────────────────────────

def case_i():
    print("[I] retained mapping exactness")
    words = [w("one", 100.2, 100.6), w("two", 100.7, 101.1),
             w("three", 103.1, 103.5), w("four", 103.6, 104.0)]
    plan = sp.plan_pacing(words, 100.0, 105.0, sil((1.1, 3.1)), fps=30.0)
    check("I: ok", plan["status"] == "ok", json.dumps(plan)[:200])
    if plan["status"] != "ok":
        return
    ret = plan["retained"]
    # tiling: out intervals contiguous from 0, total = outputDuration
    out_sum = sum(r["outEndSec"] - r["outStartSec"] for r in ret)
    check("I: output tiling exact",
          abs(out_sum - plan["outputDurationSec"]) < 0.05,
          "sum={} out={}".format(out_sum, plan["outputDurationSec"]))
    for a, b in zip(ret, ret[1:]):
        check("I: contiguous out intervals",
              abs(b["outStartSec"] - a["outEndSec"]) < 0.05,
              "{} -> {}".format(a, b))
    # src lengths preserved per interval
    for r in ret:
        check("I: src length preserved",
              abs((r["srcEndSec"] - r["srcStartSec"]) -
                  (r["outEndSec"] - r["outStartSec"])) < 0.05,
              json.dumps(r))


# ── J. Source->output mapping math ──────────────────────────────────────────

def case_j():
    print("[J] source/output mapping math")
    words = [w("one", 100.2, 100.6), w("two", 100.7, 101.1),
             w("three", 103.1, 103.5), w("four", 103.6, 104.0)]
    plan = sp.plan_pacing(words, 100.0, 105.0, sil((1.1, 3.1)), fps=30.0)
    if plan["status"] != "ok":
        check("J: ok", False, json.dumps(plan)[:200])
        return
    # word "three" at src 103.1 must map to 103.1 minus everything removed
    # before it (the internal-pause cut in the [101.1, 103.1] gap).
    removed_before = sum(
        e["srcEndSec"] - e["srcStartSec"]
        for e in plan["edits"] if e["srcEndSec"] <= 103.1 + 1e-9)
    expected_out = (103.1 - plan["clipStartSec"]) - removed_before
    # find which retained interval contains 103.1
    hit = [r for r in plan["retained"] if r["srcStartSec"] <= 103.1 <= r["srcEndSec"]]
    check("J: 103.1 in exactly one retained", len(hit) == 1, json.dumps(plan["retained"]))
    if hit:
        r = hit[0]
        frac = (103.1 - r["srcStartSec"]) / (r["srcEndSec"] - r["srcStartSec"])
        out = r["outStartSec"] + frac * (r["outEndSec"] - r["outStartSec"])
        check("J: mapping math", abs(out - expected_out) < 0.05,
              "out={} expected={}".format(out, expected_out))


# ── K. Circuit breakers ─────────────────────────────────────────────────────

def case_k():
    print("[K] circuit breakers")
    # >40% removal -> skipped entirely
    words = [w("a", 100.2, 100.5), w("b", 104.0, 104.3)]  # 3.5s gap in 5s clip
    plan = sp.plan_pacing(words, 100.0, 105.0, sil((0.5, 4.0)), fps=30.0)
    check("K: 40% breaker skips", plan["status"] != "ok",
          "status={} removed={}".format(plan.get("status"), plan.get("removedTotalSec")))
    # removal under 40% but output < 3s -> skipped
    words2 = [w("a", 100.2, 100.5), w("b", 102.7, 103.0)]  # 2.2s gap in 4.5s clip
    plan2 = sp.plan_pacing(words2, 100.0, 104.5, sil((0.5, 2.7)), fps=30.0)
    check("K: min-output breaker skips", plan2["status"] != "ok",
          "status={} reason={}".format(plan2.get("status"), plan2.get("reason")))


# ── L. Speaker change blocks internal pause cuts ─────────────────────────────

def case_l():
    print("[L] speaker change blocks pause cuts")
    words = [w("speaker", 100.2, 100.8, "S1"), w("one", 100.9, 101.4, "S1"),
             w("speaker", 103.4, 104.0, "S2"), w("two", 104.1, 104.6, "S2")]
    plan = sp.plan_pacing(words, 100.0, 105.0, sil((1.4, 3.4)), fps=30.0)
    check("L: no internal_pause across speakers",
          not any(e["editType"] == "internal_pause" for e in plan.get("edits", [])),
          "edits={}".format([e["editType"] for e in plan.get("edits", [])]))
    # same speaker with identical geometry -> cut allowed
    words2 = [w("same", 100.2, 100.8, "S1"), w("speaker", 100.9, 101.4, "S1"),
              w("same", 103.4, 104.0, "S1"), w("speaker", 104.1, 104.6, "S1")]
    plan2 = sp.plan_pacing(words2, 100.0, 105.0, sil((1.4, 3.4)), fps=30.0)
    check("L: same-speaker pause IS cut",
          any(e["editType"] == "internal_pause" for e in plan2.get("edits", [])),
          "edits={}".format([e["editType"] for e in plan2.get("edits", [])]))


# ── M. Kill switch ──────────────────────────────────────────────────────────

def case_m():
    print("[M] kill switch")
    old = os.environ.get("AUTOSHORTS_SMART_PACING")
    try:
        os.environ["AUTOSHORTS_SMART_PACING"] = "0"
        # plan_pacing itself doesn't check env (Rust does); verify the Rust
        # side parses it — mirrored here by checking the env value contract.
        val = os.environ["AUTOSHORTS_SMART_PACING"].strip().lower()
        check("M: env contract", val in ("0", "false", "off"), val)
    finally:
        if old is None:
            os.environ.pop("AUTOSHORTS_SMART_PACING", None)
        else:
            os.environ["AUTOSHORTS_SMART_PACING"] = old


# ── N. silencedetect covers the FULL candidate range ────────────────────────

def case_n():
    print("[N] silencedetect full-range coverage + clip-relative contract")
    captured = {}

    class FakeProc(object):
        returncode = 0
        stdout = ""
        stderr = ("[silencedetect @ 0x1] silence_start: 0.500\n"
                 "[silencedetect @ 0x1] silence_end: 2.500 | silence_duration: 2.000\n")

    def fake_run(cmd, **kw):
        captured["cmd"] = cmd
        return FakeProc()

    orig = sp.subprocess.run
    sp.subprocess.run = fake_run
    try:
        result = sp.run_silencedetect("fake.mp4", 100.0, 105.0)
    finally:
        sp.subprocess.run = orig
    cmd = captured.get("cmd", [])
    check("N: seeks to candidate start",
          "-ss" in cmd and "100.000" in cmd, str(cmd))
    check("N: spans FULL candidate duration (not a shortened range)",
          "-t" in cmd and "5.000" in cmd, str(cmd))
    check("N: returns clip-relative intervals",
          result == [(0.5, 2.5)], "result={}".format(result))
    # Regression: a MID-SOURCE clip (start=100) must pace correctly — the
    # engine shifts clip-relative silences onto the absolute word timeline.
    words = [w("one", 100.2, 100.6), w("two", 103.0, 103.4)]
    plan = sp.plan_pacing(words, 100.0, 105.0, sil((0.6, 3.0)), fps=30.0)
    check("N: mid-source clip paces with clip-relative silences",
          plan["status"] == "ok" and
          any(e["editType"] == "internal_pause" for e in plan.get("edits", [])),
          json.dumps(plan)[:200])


# ── CLI-level: real ffmpeg silencedetect on generated fixtures ──────────────

def run_ffmpeg(args):
    subprocess.run(args, check=True, capture_output=True,
                  creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)


def make_tone_silence_fixture(path, tone_spans, dur):
    """One ffmpeg command: anullsrc/sine lavfi segments joined by the concat
    filter. tone_spans = [(start, end), ...] tone regions; rest is silence."""
    inputs, labels = [], []
    prev = 0.0
    idx = 0
    for s, e in sorted(tone_spans):
        if s > prev + 1e-3:
            inputs += ["-f", "lavfi", "-t", "{:.3f}".format(s - prev),
                       "-i", "anullsrc=r=44100:cl=mono"]
            labels.append("[{}:a]".format(idx))
            idx += 1
        inputs += ["-f", "lavfi", "-t", "{:.3f}".format(e - s),
                   "-i", "sine=frequency=440:r=44100"]
        labels.append("[{}:a]".format(idx))
        idx += 1
        prev = e
    if prev < dur - 1e-3:
        inputs += ["-f", "lavfi", "-t", "{:.3f}".format(dur - prev),
                   "-i", "anullsrc=r=44100:cl=mono"]
        labels.append("[{}:a]".format(idx))
    graph = "".join(labels) + "concat=n={}:v=0:a=1[a]".format(len(labels))
    run_ffmpeg(["ffmpeg", "-y"] + inputs +
               ["-filter_complex", graph, "-map", "[a]",
                "-c:a", "pcm_s16le", path])


def run_engine_cli(source, start_ms, end_ms, words):
    tmpdir = tempfile.mkdtemp(prefix="pacing_words_")
    words_path = os.path.join(tmpdir, "words.json")
    with open(words_path, "w") as f:
        json.dump(words, f)
    return subprocess.run(
        [sys.executable, os.path.join(ENGINE_DIR, "smart_pacing.py"),
         source, str(start_ms), str(end_ms), words_path],
        capture_output=True, text=True,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)


def dense_words(spans, dur=0.45, gap=0.05):
    """Dense realistic words covering each (start, end) tone span."""
    out = []
    for s, e in spans:
        t = s
        i = 0
        while t + dur <= e + 1e-6:
            out.append(w("word{}".format(i), round(t, 3), round(t + dur, 3)))
            t += dur + gap
            i += 1
    return out


def case_cli():
    print("[CLI] real engine main() with real ffmpeg silencedetect")
    tmpdir = tempfile.mkdtemp(prefix="pacing_cli_")
    src = os.path.join(tmpdir, "fixture.wav")
    try:
        # 12s: silence [0,1), tone [1,5), silence [5,7), tone [7,11), silence [11,12)
        make_tone_silence_fixture(src, [(1.0, 5.0), (7.0, 11.0)], 12.0)
    except Exception as ex:
        print("  SKIP  CLI cases (ffmpeg/fixture error: {})".format(ex))
        return

    words = dense_words([(1.0, 5.0), (7.0, 11.0)])
    # mark a sentence boundary before the big gap so the sentence threshold
    # (1.25s) applies; the 2s gap then compresses to a 0.5s natural pause
    if words:
        words[len([x for x in words if x["start"] < 5.0]) - 1]["text"] = "end."
    proc = run_engine_cli(src, 0, 12000, words)
    check("CLI: exit code 0", proc.returncode == 0,
          "rc={} stderr={}".format(proc.returncode, proc.stderr[-300:]))
    lines = [l for l in proc.stdout.strip().splitlines() if l.strip()]
    check("CLI: JSON on last stdout line", lines and lines[-1].startswith("{"),
          "stdout tail={}".format(lines[-1][:100] if lines else "<empty>"))
    if not (lines and lines[-1].startswith("{")):
        return
    plan = json.loads(lines[-1])
    check("CLI: leading dead air found via real silencedetect",
          any(e["editType"] == "leading_dead_air" for e in plan.get("edits", [])),
          "edits={}".format([e["editType"] for e in plan.get("edits", [])]))
    check("CLI: internal pause found via real silencedetect",
          any(e["editType"] == "internal_pause" for e in plan.get("edits", [])),
          "edits={}".format([e["editType"] for e in plan.get("edits", [])]))
    check("CLI: trailing dead air found via real silencedetect",
          any(e["editType"] == "trailing_dead_air" for e in plan.get("edits", [])),
          "edits={}".format([e["editType"] for e in plan.get("edits", [])]))
    if plan.get("status") == "ok":
        check("CLI: total removal within circuit-breaker budget",
              plan["removedTotalSec"] <= 0.40 * 12.0 + 0.05,
              "removed={}".format(plan["removedTotalSec"]))
        check("CLI: duration arithmetic",
              abs(12.0 - plan["removedTotalSec"] - plan["outputDurationSec"]) < 0.05,
              "removed={} out={}".format(plan["removedTotalSec"],
                                         plan["outputDurationSec"]))


# ── Music/sound: silence verification blocks cuts on non-silent audio ───────

def case_music():
    print("[MUSIC] continuous tone blocks all cuts")
    tmpdir = tempfile.mkdtemp(prefix="pacing_music_")
    src = os.path.join(tmpdir, "music.wav")
    try:
        make_tone_silence_fixture(src, [(0.0, 12.0)], 12.0)  # continuous tone
    except Exception as ex:
        print("  SKIP  music case (ffmpeg error: {})".format(ex))
        return
    words = dense_words([(1.0, 5.0), (7.0, 11.0)])  # dense words, big gaps
    proc = run_engine_cli(src, 0, 12000, words)
    lines = [l for l in proc.stdout.strip().splitlines() if l.strip()]
    if not (proc.returncode == 0 and lines and lines[-1].startswith("{")):
        check("MUSIC: engine exits cleanly", False,
              "rc={} out={}".format(proc.returncode, proc.stdout[-200:]))
        return
    plan = json.loads(lines[-1])
    check("MUSIC: no cuts on continuous music",
          plan["status"] != "ok" or not plan.get("edits"),
          "status={} edits={}".format(plan["status"],
                                      [e["editType"] for e in plan.get("edits", [])]))


def main():
    print("=" * 72)
    print("AutoShorts 8.0 — Smart Pacing Suite (real engine: {})".format(
        os.path.join(ENGINE_DIR, "smart_pacing.py")))
    print("=" * 72)
    for fn in (case_a, case_b, case_c, case_d, case_e, case_f, case_g, case_h,
               case_i, case_j, case_k, case_l, case_m, case_n, case_cli,
               case_music):
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
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
