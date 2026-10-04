#!/usr/bin/env python3
"""AutoShorts 8.0 — Audio Intelligence test suite (cases A–P).

Tests the REAL production engine
(`autoshorts/src-tauri/scripts/audio_intelligence.py`) — the same sidecar
the Rust render path invokes — not a replica. Decision-engine cases run
against `decide()` directly with synthetic analysis dicts; CLI-level cases
run the real `main()` entrypoint with real ffmpeg analysis on generated
fixtures (the exact lavfi recipes verified against FFmpeg 9 during
development); the Rust-integration case runs the real `audio_inspect`
binary end-to-end (sidecar plan -> Rust validation -> render -> measured
output loudness).

Spec mandate: "Good audio -> preserve it. Problematic audio -> make the
smallest safe correction. Uncertain audio -> do not touch it."

Run:  .venv/Scripts/python.exe test_audio_intelligence_suite.py
"""

import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE) if os.path.basename(HERE) == "tests" else HERE
ENGINE_DIR = os.path.join(REPO_ROOT, "autoshorts", "src-tauri", "scripts")
sys.path.insert(0, ENGINE_DIR)

import audio_intelligence as ai  # noqa: E402  (real production engine)

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


def stage(plan_or_stages, name):
    stages = (plan_or_stages.get("stages")
              if isinstance(plan_or_stages, dict)
              else plan_or_stages)
    return next((s for s in stages if s.get("name") == name), None)


def applied(plan_or_stages, name):
    s = stage(plan_or_stages, name)
    return bool(s and s.get("applied"))


def mk_analysis(**over):
    """Analysis dict in the engine's internal shape (snake_case, astats
    sub-dict) — the same shape `decide()` reads."""
    a = {
        "input_i": -16.0, "input_tp": -6.0, "input_lra": 6.0,
        "input_thresh": -26.0, "target_offset": 0.0,
        "astats": {"peak_db": -6.0, "rms_db": -20.0,
                   "clipped": False, "min_level": -0.5, "max_level": 0.5},
        "highpass_mean_db": -20.5,
        "noise_floor_db": -55.0,
    }
    a.update(over)
    return a


# ── A. Good audio -> preserved (no processing) ──────────────────────────────

def case_a():
    print("[A] good audio -> preserved")
    # -16.1 LUFS (within 2 LU of target), clean peak, no rumble, no noise,
    # single speaker.
    analysis = mk_analysis(input_i=-16.1, input_tp=-3.0,
                           astats={"peak_db": -3.0, "rms_db": -20.0,
                                   "clipped": False,
                                   "min_level": -0.7, "max_level": 0.7},
                           highpass_mean_db=-20.5)
    stages, chain, reason = ai.decide(analysis, [], 10.0, {})
    check("A: no filters applied", chain == "", "chain={!r}".format(chain))
    check("A: reason documents preservation",
          "no processing" in reason, reason)
    check("A: all stages recorded as skipped",
          all(not s["applied"] for s in stages),
          json.dumps([s["applied"] for s in stages]))


# ── B. Quiet speech -> loudness normalized to target ────────────────────────

def case_b():
    print("[B] quiet speech -> loudnorm to -16 LUFS")
    analysis = mk_analysis(input_i=-24.5, input_tp=-8.8, input_lra=1.8,
                           input_thresh=-35.4, target_offset=-0.2,
                           astats={"peak_db": -8.8, "rms_db": -22.0,
                                   "clipped": False,
                                   "min_level": -0.36, "max_level": 0.36},
                           highpass_mean_db=-22.9)  # delta 0.9 dB < 2 dB gate
    stages, chain, reason = ai.decide(analysis, [], 10.0, {})
    check("B: loudnorm applied", applied(stages, "loudness"), chain)
    check("B: chain is loudnorm + aresample",
          chain.startswith("loudnorm=") and "aresample=48000" in chain,
          chain)
    check("B: linear mode (dynamics preserved)", "linear=true" in chain, chain)
    check("B: no limiter after loudnorm (TP guaranteed)",
          "alimiter" not in chain, chain)
    check("B: measured values embedded (two-pass)",
          "measured_I=-24.5" in chain, chain)


# ── C. Already-loud audio within tolerance -> untouched ─────────────────────

def case_c():
    print("[C] within-tolerance loudness -> untouched")
    analysis = mk_analysis(input_i=-15.0, input_tp=-2.0,
                           astats={"peak_db": -2.0, "rms_db": -18.0,
                                   "clipped": False,
                                   "min_level": -0.79, "max_level": 0.79},
                           highpass_mean_db=-18.5)
    stages, chain, reason = ai.decide(analysis, [], 10.0, {})
    check("C: no loudnorm within tolerance", "loudnorm" not in chain, chain)
    s = stage(stages, "loudness") or {}
    check("C: loudness stage documents why",
          "within" in s.get("reason", ""), s.get("reason", ""))


# ── D. Clipping protection (alimiter) ──────────────────────────────────────

def case_d():
    print("[D] clipped audio -> limiter, not reconstruction")
    # Clipped samples (|level| >= 0.999) but loudness already at target so
    # the limiter is the ONLY correction (smallest safe correction).
    analysis = mk_analysis(input_i=-16.0, input_tp=-0.2,
                           astats={"peak_db": -0.1, "rms_db": -14.0,
                                   "clipped": True,
                                   "min_level": -1.018, "max_level": 1.013},
                           highpass_mean_db=-14.5)
    stages, chain, reason = ai.decide(analysis, [], 10.0, {})
    check("D: alimiter engaged for clipped audio",
          "alimiter=limit=0.841:level=false" in chain, chain)
    check("D: level=false (no auto-attenuation)", "level=false" in chain, chain)
    check("D: no loudnorm (already at target)", "loudnorm" not in chain, chain)


# ── E. Predicted peak overshoot -> limiter ──────────────────────────────────

def case_e():
    print("[E] predicted peak overshoot after boost -> limiter")
    # Speaker boost of 6 dB on a -5 dBTP input predicts +1.0 dBTP > -1.0 gate.
    # (Mismatch 14 dB -> gain = min(6, 14/2) = 6 dB.)
    analysis = mk_analysis(input_i=-16.0, input_tp=-5.0,
                           astats={"peak_db": -5.0, "rms_db": -18.0,
                                   "clipped": False,
                                   "min_level": -0.56, "max_level": 0.56},
                           highpass_mean_db=-18.5)
    speakers = {"A": {"mean_db": -20.0, "intervals": [(0.0, 4.0)]},
                "B": {"mean_db": -34.0, "intervals": [(5.0, 9.0)]}}
    stages, chain, reason = ai.decide(analysis, [], 10.0, speakers)
    check("E: speaker boost applied", "volume=" in chain, chain)
    check("E: limiter engaged for predicted overshoot",
          "alimiter" in chain, chain)


# ── F. Speaker-to-speaker consistency ──────────────────────────────────────

def case_f():
    print("[F] speaker mismatch -> capped boost on quiet speaker only")
    analysis = mk_analysis()
    # 13.9 dB mismatch -> gain = min(6, 13.9/2) = 6 dB (capped)
    speakers = {"A": {"mean_db": -18.0, "intervals": [(0.0, 4.0)]},
                "B": {"mean_db": -31.9, "intervals": [(5.0, 9.0)]}}
    stages, chain, reason = ai.decide(analysis, [], 10.0, speakers)
    check("F: boost capped at 6 dB", "volume=6.00dB" in chain, chain)
    check("F: enable expression targets only B's speech",
          "enable='between(t,5,9)'" in chain, chain)
    s = stage(stages, "speaker_balance") or {}
    check("F: mismatch documented",
          abs(s.get("mismatchDb", 0) - 13.9) < 0.2,
          json.dumps(s))


# ── G. Silent speaker is NEVER amplified ─────────────────────────────────────

def case_g():
    print("[G] effectively-silent speaker never amplified")
    analysis = mk_analysis()
    # B is pure silence (-82 dB) — boosting it would amplify noise.
    speakers = {"A": {"mean_db": -18.0, "intervals": [(0.0, 4.0)]},
                "B": {"mean_db": -82.5, "intervals": [(5.0, 9.0)]}}
    stages, chain, reason = ai.decide(analysis, [], 10.0, speakers)
    check("G: no volume filter for silent speaker",
          "volume=" not in chain, chain)
    s = stage(stages, "speaker_balance") or {}
    check("G: reason documents the silence guard",
          "silence" in s.get("reason", ""), s.get("reason", ""))


# ── H. Extreme silence -> untouched (never amplified) ───────────────────────

def case_h():
    print("[H] genuine silence -> untouched")
    # input_i below MIN_SPEECH_I (-50): the whole clip is effectively
    # silence — amplifying it would only amplify noise.
    analysis = mk_analysis(input_i=-60.0, input_tp=-50.0, input_lra=0.0,
                           input_thresh=-70.0,
                           astats={"peak_db": -50.0, "rms_db": -60.0,
                                   "clipped": False,
                                   "min_level": -0.001, "max_level": 0.001},
                           highpass_mean_db=-60.0, noise_floor_db=None)
    stages, chain, reason = ai.decide(analysis, [], 10.0, {})
    check("H: silence left untouched", chain == "", "chain={!r}".format(chain))
    check("H: reason documents the guard",
          "silence" in reason.lower(), reason)


# ── I. Rumble -> highpass ───────────────────────────────────────────────────

def case_i():
    print("[I] sub-100Hz rumble -> highpass at 80 Hz")
    # Rumble gate: fullband RMS minus highpassed RMS >= 2 dB.
    analysis = mk_analysis(astats={"peak_db": -6.0, "rms_db": -20.0,
                                   "clipped": False,
                                   "min_level": -0.5, "max_level": 0.5},
                            highpass_mean_db=-31.0)
    stages, chain, reason = ai.decide(analysis, [], 10.0, {})
    check("I: highpass applied", "highpass=f=80" in chain, chain)
    check("I: highpass is FIRST in the chain", chain.startswith("highpass="),
          chain)


# ── J. Noise floor above gate -> conservative denoise ───────────────────────

def case_j():
    print("[J] loud noise floor -> afftdn with measured floor")
    analysis = mk_analysis(noise_floor_db=-31.4)
    stages, chain, reason = ai.decide(analysis, [], 10.0, {})
    check("J: afftdn applied", "afftdn=nf=" in chain, chain)
    # nf = floor + 6 dB margin, clamped to [-80, -20]
    check("J: nf includes the 6 dB safety margin",
          "afftdn=nf=-25.4" in chain, chain)
    check("J: conservative nr=10", "nr=10" in chain, chain)


# ── K. Uncertain audio (missing measurements) -> untouched ──────────────────

def case_k():
    print("[K] missing measurements -> untouched")
    # No loudnorm JSON at all (analysis unavailable) — nothing may fire.
    analysis = mk_analysis(input_i=None, input_tp=None, input_lra=None,
                           input_thresh=None, target_offset=None,
                           astats={"peak_db": None, "rms_db": None,
                                   "clipped": None,
                                   "min_level": None, "max_level": None},
                           highpass_mean_db=None, noise_floor_db=None)
    stages, chain, reason = ai.decide(analysis, [], 10.0, {})
    check("K: unmeasured audio untouched", chain == "",
          "chain={!r}".format(chain))
    # Loudness justified but LRA/thresh missing -> still untouched.
    analysis2 = mk_analysis(input_i=-30.0, input_lra=None,
                            input_thresh=None)
    stages2, chain2, reason2 = ai.decide(analysis2, [], 10.0, {})
    check("K: incomplete loudness measurements -> no loudnorm",
          "loudnorm" not in chain2, chain2)


# ── L. Silence ratio guard (mostly-silent clips) ───────────────────────────

def case_l():
    print("[L] mostly-silent clip -> no amplification of silence")
    # 85% silence: normalizing the integrated loudness would mostly
    # amplify silence. (Silence ratio comes from the silence_intervals
    # argument, not the analysis dict.)
    analysis = mk_analysis(input_i=-40.0, input_tp=-20.0, input_lra=2.0,
                           input_thresh=-50.0,
                           astats={"peak_db": -20.0, "rms_db": -35.0,
                                   "clipped": False,
                                   "min_level": -0.01, "max_level": 0.01},
                           highpass_mean_db=-35.5)
    stages, chain, reason = ai.decide(analysis, [(0.0, 8.5)], 10.0, {})
    check("L: mostly-silent clip untouched", chain == "",
          "chain={!r}".format(chain))


# ── M. Stage ordering is fixed ─────────────────────────────────────────────

def case_m():
    print("[M] chain order: highpass -> afftdn -> volume -> loudnorm -> aresample")
    analysis = mk_analysis(input_i=-24.0, input_tp=-8.0, input_lra=4.0,
                           input_thresh=-34.0,
                           astats={"peak_db": -8.0, "rms_db": -20.0,
                                   "clipped": False,
                                   "min_level": -0.4, "max_level": 0.4},
                           highpass_mean_db=-30.0, noise_floor_db=-31.0)
    speakers = {"A": {"mean_db": -20.0, "intervals": [(0.0, 4.0)]},
                "B": {"mean_db": -28.0, "intervals": [(5.0, 9.0)]}}
    stages, chain, reason = ai.decide(analysis, [], 10.0, speakers)
    order = ["highpass=f=80", "afftdn=nf=", "volume=", "loudnorm=",
             "aresample=48000"]
    pos = [chain.find(tok) for tok in order]
    check("M: all stages present in order",
          all(p >= 0 for p in pos) and pos == sorted(pos),
          "chain={} pos={}".format(chain, pos))


# ── N. CLI: real engine main() on generated fixtures ────────────────────────

def run_ffmpeg(args):
    subprocess.run(args, check=True, capture_output=True,
                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)


def run_engine_cli(source, start_ms, end_ms, words=None, pacing=None):
    cmd = [sys.executable, os.path.join(ENGINE_DIR, "audio_intelligence.py"),
           source, str(start_ms), str(end_ms)]
    tmpdir = tempfile.mkdtemp(prefix="audio_words_")
    if words is not None:
        words_path = os.path.join(tmpdir, "words.json")
        with open(words_path, "w") as f:
            json.dump(words, f)
        cmd.append(words_path)
    if pacing is not None:
        pacing_path = os.path.join(tmpdir, "pacing.json")
        with open(pacing_path, "w") as f:
            json.dump(pacing, f)
        cmd.append(pacing_path)
    return subprocess.run(cmd, capture_output=True, text=True,
                          creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)


def last_json(stdout):
    lines = [l for l in stdout.strip().splitlines() if l.strip()]
    if not lines or not lines[-1].startswith("{"):
        return None
    return json.loads(lines[-1])


def case_n():
    print("[N] CLI: real fixtures through real main()")
    tmpdir = tempfile.mkdtemp(prefix="audio_cli_")

    # N1: quiet-but-clean speech (600 Hz sine at -24 dBFS) -> loudness only.
    # (Sine default amplitude is 1/8 = -18 dBFS; -6 dB more = -24 dBFS RMS,
    # comfortably above the -35 dB silence line.)
    src1 = os.path.join(tmpdir, "quiet.wav")
    run_ffmpeg(["ffmpeg", "-y", "-f", "lavfi", "-t", "10",
                "-i", "sine=frequency=600:r=48000",
                "-af", "volume=-6dB", "-c:a", "pcm_s16le", src1])
    plan = last_json(run_engine_cli(src1, 0, 10000).stdout)
    check("N1: plan emitted", plan is not None, "no JSON")
    if plan:
        check("N1: loudness applied on quiet fixture",
              applied(plan, "loudness"), json.dumps(plan.get("stages"))[:200])
        check("N1: chain shape", plan["filterChain"].startswith("loudnorm="),
              plan["filterChain"])

    # N2: genuine silence -> untouched (status ok, empty chain).
    src2 = os.path.join(tmpdir, "silent.wav")
    run_ffmpeg(["ffmpeg", "-y", "-f", "lavfi", "-t", "10",
                "-i", "anullsrc=r=48000:cl=mono", "-c:a", "pcm_s16le", src2])
    plan2 = last_json(run_engine_cli(src2, 0, 10000).stdout)
    check("N2: silence plan emitted", plan2 is not None, "no JSON")
    if plan2:
        check("N2: silence left untouched (empty chain)",
              plan2["filterChain"] == "", plan2["filterChain"])
        check("N2: reason documents silence",
              "silence" in (plan2.get("reason") or "").lower(),
              plan2.get("reason"))

    # N3: hard-clipped audio (sine +20 dB into s16) -> clipped detected.
    src3 = os.path.join(tmpdir, "clipped.wav")
    run_ffmpeg(["ffmpeg", "-y", "-f", "lavfi", "-t", "10",
                "-i", "sine=frequency=600:r=48000",
                "-af", "volume=20dB", "-c:a", "pcm_s16le", src3])
    plan3 = last_json(run_engine_cli(src3, 0, 10000).stdout)
    check("N3: clipped detected", plan3 is not None and
          plan3["analysis"].get("clipped") is True,
          json.dumps(plan3.get("analysis", {}))[:200] if plan3 else "no JSON")
    if plan3:
        check("N3: limiter or loudnorm engaged",
              "alimiter" in plan3["filterChain"] or
              "loudnorm" in plan3["filterChain"], plan3["filterChain"])

    # N4: rumble (50 Hz louder than speech band) -> highpass.
    # Mixed spectrum: 50 Hz at -6 dB + 600 Hz at -24 dB. The highpassed
    # mean drops well below the fullband mean -> rumble gate fires. A pure
    # 50 Hz tone would leave nothing after highpass (remeasure guard).
    src4 = os.path.join(tmpdir, "rumble.wav")
    run_ffmpeg(["ffmpeg", "-y",
                "-f", "lavfi", "-t", "10", "-i", "sine=frequency=50:r=48000",
                "-f", "lavfi", "-t", "10", "-i", "sine=frequency=600:r=48000",
                "-filter_complex",
                "[0:a]volume=-6dB[r];[1:a]volume=-24dB[s];"
                "[r][s]amix=inputs=2:normalize=0[a]",
                "-map", "[a]", "-c:a", "pcm_s16le", src4])
    plan4 = last_json(run_engine_cli(src4, 0, 10000).stdout)
    check("N4: rumble highpassed",
          plan4 is not None and "highpass=f=80" in plan4["filterChain"],
          plan4["filterChain"] if plan4 else "no JSON")

    # N5: no audio stream at all -> skipped cleanly.
    src5 = os.path.join(tmpdir, "noaudio.mp4")
    run_ffmpeg(["ffmpeg", "-y", "-f", "lavfi", "-t", "5",
                "-i", "testsrc=size=320x240:rate=10",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", src5])
    plan5 = last_json(run_engine_cli(src5, 0, 5000).stdout)
    check("N5: no-audio source skipped",
          plan5 is not None and plan5["status"] == "skipped",
          json.dumps(plan5)[:200] if plan5 else "no JSON")


# ── O. Pacing-aware analysis (output timeline) ──────────────────────────────

def case_o():
    print("[O] pacing plan -> analysis on the OUTPUT timeline")
    tmpdir = tempfile.mkdtemp(prefix="audio_pacing_")
    # 12s fixture: tone [1,5) + tone [7,11), silence elsewhere.
    src = os.path.join(tmpdir, "paced.wav")
    inputs = ["-f", "lavfi", "-t", "1", "-i", "anullsrc=r=48000:cl=mono",
              "-f", "lavfi", "-t", "4", "-i", "sine=frequency=600:r=48000",
              "-f", "lavfi", "-t", "2", "-i", "anullsrc=r=48000:cl=mono",
              "-f", "lavfi", "-t", "4", "-i", "sine=frequency=600:r=48000",
              "-f", "lavfi", "-t", "1", "-i", "anullsrc=r=48000:cl=mono"]
    labels = "[0:a][1:a][2:a][3:a][4:a]concat=n=5:v=0:a=1[a]"
    run_ffmpeg(["ffmpeg", "-y"] + inputs +
               ["-filter_complex", labels, "-map", "[a]",
                "-c:a", "pcm_s16le", src])

    # Pacing plan removing [5,7) (the 2s silence): output = 10s.
    pacing = {"status": "ok", "clipStartSec": 0.0, "clipEndSec": 12.0,
              "outputDurationSec": 10.0, "removedTotalSec": 2.0,
              "edits": [{"editType": "internal_pause", "srcStartSec": 5.0,
                         "srcEndSec": 7.0, "outStartSec": 0.0,
                         "confidence": 0.9, "reason": "test"}],
              "retained": [
                  {"srcStartSec": 0.0, "srcEndSec": 5.0,
                   "outStartSec": 0.0, "outEndSec": 5.0},
                  {"srcStartSec": 7.0, "srcEndSec": 12.0,
                   "outStartSec": 5.0, "outEndSec": 10.0}]}
    plan_plain = last_json(run_engine_cli(src, 0, 12000).stdout)
    plan_paced = last_json(run_engine_cli(src, 0, 12000, [], pacing).stdout)
    check("O: both plans emitted", plan_plain and plan_paced, "missing JSON")
    if not (plan_plain and plan_paced):
        return
    # The edit map removes 2s of silence. Integrated loudness is gated
    # (silence contributes ~nothing), so the honest output-timeline proof is
    # the astats RMS (mean includes silence) plus the silence ratio.
    ri = plan_plain["analysis"].get("astatsRmsDb")
    ro = plan_paced["analysis"].get("astatsRmsDb")
    check("O: output-timeline RMS differs",
          ri is not None and ro is not None and abs(ri - ro) > 0.3,
          "plain={} paced={}".format(ri, ro))
    check("O: paced silence ratio lower",
          (plan_paced["analysis"].get("silenceRatio") or 0) <
          (plan_plain["analysis"].get("silenceRatio") or 1),
          "plain={} paced={}".format(plan_plain["analysis"].get("silenceRatio"),
                                     plan_paced["analysis"].get("silenceRatio")))


# ── P. Rust integration: audio_inspect end-to-end ───────────────────────────

AUDIO_INSPECT = os.path.join(REPO_ROOT, "autoshorts", "src-tauri", "target",
                             "debug", "audio_inspect.exe")


def case_p():
    print("[P] Rust integration: audio_inspect end-to-end")
    if not os.path.exists(AUDIO_INSPECT):
        print("  SKIP  P (audio_inspect.exe not built)")
        return
    # Real footage: Beat 30-40s is quiet speech (-24.5 LUFS) -> loudness.
    src = os.path.join(HERE,
                       "Beat Emotional Fatigue_ Better Sleep & Clearer Mind.mp4")
    if not os.path.exists(src):
        print("  SKIP  P (no local footage)")
        return
    tmpdir = tempfile.mkdtemp(prefix="audio_rust_")
    words_path = os.path.join(tmpdir, "words.json")
    with open(words_path, "w") as f:
        json.dump([], f)
    out_path = os.path.join(tmpdir, "ab_render.mp4")
    proc = subprocess.run(
        [AUDIO_INSPECT, src, "30", "40", words_path, "-", out_path],
        capture_output=True, text=True,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    check("P: exit 0", proc.returncode == 0,
          "rc={} stderr={}".format(proc.returncode, proc.stderr[-300:]))
    # JSON is the LAST stdout line (render diagnostics print first).
    lines = [l for l in proc.stdout.strip().splitlines() if l.strip()]
    if not (lines and lines[-1].startswith("{")):
        check("P: JSON output", False,
              "stdout tail={}".format(lines[-1][:100] if lines else "<empty>"))
        return
    try:
        out = json.loads(lines[-1])
    except Exception as ex:
        check("P: JSON output", False, "parse error: {}".format(ex))
        return
    check("P: plan validated by Rust", out.get("filterChain") is not None,
          json.dumps(out)[:200])
    check("P: render produced", out.get("renderPath") is not None,
          json.dumps(out.get("renderError"))[:200])
    # Measure the rendered output: must hit the -16 LUFS target.
    probe = subprocess.run(
        ["ffmpeg", "-i", out_path, "-af", "loudnorm=print_format=json",
         "-f", "null", os.devnull],
        capture_output=True, text=True,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    m = None
    for line in probe.stderr.splitlines():
        if '"input_i"' in line:
            m = line.split('"')[3]
            break
    try:
        measured = float(m)
    except (TypeError, ValueError):
        check("P: output loudness measurable", False,
              "no input_i in stderr")
        return
    check("P: rendered output hits -16 LUFS target",
          abs(measured - (-16.0)) < 1.0, "measured={}".format(measured))


def main():
    print("=" * 72)
    print("AutoShorts 8.0 — Audio Intelligence Suite (real engine: {})".format(
        os.path.join(ENGINE_DIR, "audio_intelligence.py")))
    print("=" * 72)
    for fn in (case_a, case_b, case_c, case_d, case_e, case_f, case_g,
               case_h, case_i, case_j, case_k, case_l, case_m, case_n,
               case_o, case_p):
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
