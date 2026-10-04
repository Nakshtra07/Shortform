#!/usr/bin/env python3
"""
AutoShorts 8.0 — Audio Intelligence / Speech Quality engine (sidecar).

Sits conceptually AFTER Smart Pacing (the edit map is authoritative) and
BEFORE the final render. The system never asks "can I process this?" —
it asks "is processing PROVEN necessary and PROVEN safe?" When uncertain,
the audio is left untouched.

Design contract (mirrors smart_pacing.py conventions):
  * ANALYSIS  — streaming FFmpeg passes over the bounded candidate range
                (never the whole file, never RAM-loaded).
  * DECISION  — conservative, evidence-gated stage rules.
  * APPLY     — the chosen filters are appended to the render's audio chain
                by Rust (single encode, processing happens exactly once).

Analysis timeline = OUTPUT timeline. When a pacing plan is supplied, every
measurement runs through the same atrim/concat edit map the render uses, so
analysis and processing never see different audio (no second independent
audio timeline).

CLI contract:
    python audio_intelligence.py SOURCE START_MS END_MS [WORDS_JSON] [PACING_JSON]

    - SOURCE       absolute path to the source video
    - START_MS     optimized range start in milliseconds (absolute source time)
    - END_MS       optimized range end in milliseconds (absolute source time)
    - WORDS_JSON   optional path to [{"text","start","end","speaker"}, ...]
                   on the SOURCE timeline (same file the framing tracker gets)
    - PACING_JSON  optional path to the SmartPacingPlan JSON (camelCase). When
                   present and valid, analysis follows the edit map.

Output: the LAST stdout line is the JSON plan (camelCase keys, matching the
Rust `AudioIntelligencePlan` serde struct). Telemetry goes to stderr. Exit
code 0 whenever the engine ran (status=skipped is a SUCCESS); non-zero means
the engine itself failed.

Plan JSON:
{
  "status": "ok" | "skipped" | "error",
  "reason": "...",
  "stages": [                     // every stage reports its decision
    {"name": "denoise", "applied": false, "reason": "...", ...}
  ],
  "filterChain": "highpass=f=80,afftdn=nf=-34:nr=10,...",   // "" when untouched
  "analysis": { ... raw measurements ... }
}

Loudness policy (documented, evidence-based):
  Target -16.0 LUFS integrated / -1.5 dBTP true peak.
  * -16 LUFS is the de-facto speech/podcast standard (Apple/Spotify podcasts,
    EBU R128 short-form guidance) and sits safely below platform loudness
    normalization ceilings (YouTube -14) so nothing re-normalizes on us.
  * The current pipeline applies NO loudness processing (source passthrough
    at -23..-26 LUFS measured on real footage — too quiet for mobile speech).
  * Tolerance band: |input_i - target| <= 2.0 LU -> "already good", no stage.
"""

import json
import math
import os
import re
import subprocess
import sys

# ── Policy constants (all conservative, all documented) ─────────────────────

TARGET_I = -16.0          # LUFS integrated target (documented above)
TARGET_TP = -1.5          # dBTP true-peak ceiling
TARGET_LRA = 11.0         # loudness-range ceiling for linear loudnorm
TOLERANCE_LU = 2.0        # |input_i - target| within this -> no loudness stage

NOISE_FLOOR_GATE = -38.0  # dB — floor must be at least this loud to denoise
NOISE_FLOOR_MARGIN = 6.0  # afftdn nf = floor + margin (never above -20)
AFFTDN_NR = 10.0          # conservative noise-reduction amount (0-97 scale)
NOISE_PROBE_DB = -26.0    # silencedetect threshold for floor hunting: hiss
                          # louder than the -35 dB silence line is invisible
                          # to it; floors above this probe stay unmeasured
FLOOR_MARGIN_DB = 6.0     # floor must be this far below content RMS to count
                          # (guards against mistaking quiet speech for noise)

RUMBLE_GATE_DB = 2.0      # fullband-vs-highpassed mean delta -> rumble present
HIGHPASS_HZ = 80          # speech-band-preserving high-pass

SPEAKER_MISMATCH_LU = 6.0  # dB between speaker means -> correction justified
SPEAKER_GAIN_CAP_DB = 6.0   # max boost applied to the quieter speaker

CLIP_TP_GATE = -1.0       # dBTP above this -> peak-safety stage engages
LIMITER_CEIL = -1.5       # dBFS limiter ceiling (linear: 10^(-1.5/20))

SILENCE_RATIO_MAX = 0.80  # >80% silence -> treat as genuine silence, skip
MIN_SPEECH_I = -50.0      # integrated loudness below this -> no speech energy

MAX_ANALYSIS_SEC = 600    # hard bound on any single analysis pass
EPS = 1e-6


def log(msg):
    sys.stderr.write("[Audio Intelligence] {}\n".format(msg))
    sys.stderr.flush()


# ── FFmpeg plumbing ─────────────────────────────────────────────────────────

def run_ffmpeg(args, timeout_sec):
    """Run ffmpeg, return (returncode, stderr_text). Never raises."""
    try:
        proc = subprocess.run(
            ["ffmpeg", "-nostats"] + args,
            capture_output=True, text=True, timeout=timeout_sec,
            encoding="utf-8", errors="replace",
        )
        return proc.returncode, proc.stderr or ""
    except (OSError, subprocess.TimeoutExpired) as exc:
        log("ffmpeg failed to run: {}".format(exc))
        return -1, ""


def fmt(v):
    s = "{:.3f}".format(v)
    return s.rstrip("0").rstrip(".") if "." in s else s


def edit_map_chain(pacing):
    """Build the atrim/concat chain that mirrors the render's audio edit map.

    Returns (prelude, out_label) where prelude maps [0:a] -> [a_edited] on the
    OUTPUT timeline (t=0 at output start), or ("", None) when there is no
    pacing plan (analysis then runs on the plain seeked range). A single
    retained interval is still an edit map (edge trims) and gets a matching
    plain atrim — exactly what the render's single-piece audio path does.
    """
    if not pacing:
        return "", None
    retained = pacing.get("retained") or []
    if not retained:
        return "", None
    clip_start = float(pacing.get("clipStartSec", 0.0))
    if len(retained) == 1:
        s = float(retained[0]["srcStartSec"]) - clip_start
        e = float(retained[0]["srcEndSec"]) - clip_start
        return ("[0:a]atrim=start={}:end={},asetpts=PTS-STARTPTS[a_edited]"
                .format(fmt(s), fmt(e)), "a_edited")
    parts = ["[0:a]asplit={}".format(len(retained))]
    for i in range(len(retained)):
        parts.append("[ae{}]".format(i))
    parts.append(";")
    clip_start = float(pacing.get("clipStartSec", 0.0))
    for i, r in enumerate(retained):
        s = float(r["srcStartSec"]) - clip_start
        e = float(r["srcEndSec"]) - clip_start
        parts.append("[ae{}]atrim=start={}:end={},asetpts=PTS-STARTPTS[aet{}];"
                     .format(i, fmt(s), fmt(e), i))
    joined = "".join("[aet{}]".format(i) for i in range(len(retained)))
    parts.append("{}concat=n={}:v=0:a=1[a_edited]".format(joined, len(retained)))
    return "".join(parts), "a_edited"


def analysis_cmd(source, start_sec, dur_sec, filter_complex, maps):
    """Assemble the bounded analysis command (input seek + -t bound)."""
    args = [
        "-ss", "{:.3f}".format(start_sec),
        "-t", "{:.3f}".format(dur_sec),
        "-i", source,
        "-filter_complex", filter_complex,
    ]
    for m in maps:
        args += ["-map", m]
    args += ["-f", "null", "-"]
    return args


# ── Parsers ──────────────────────────────────────────────────────────────────

def parse_loudnorm_json(stderr):
    """Extract the LAST loudnorm JSON block from stderr."""
    blocks = re.findall(r"\{[^{}]*\"input_i\"[^{}]*\}", stderr, re.S)
    if not blocks:
        return None
    try:
        return json.loads(blocks[-1])
    except ValueError:
        return None


def parse_db(v):
    """Parse a loudnorm/astats dB string; -inf/inf/nan -> None."""
    if v is None:
        return None
    s = str(v).strip()
    if s in ("-inf", "inf", "nan", "-NaN", "NaN", ""):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def parse_astats_overall(stderr, instance_hint=None):
    """Parse the LAST 'Overall' block from astats output.

    FFmpeg 9 removed 'Number of Clipped samples' from astats entirely, so
    clipping is derived from Min/Max level (linear scale, full scale = 1.0):
    decoded audio that was hard-clipped (or codec-overshot) shows levels
    beyond +/-1.0; clean audio stays below. 0.999 catches integer-clipped
    samples (32767/32768) and near-full-scale codec overshoot."""
    blocks = stderr.split("Overall")
    if len(blocks) < 2:
        return None
    tail = blocks[-1]
    out = {}
    for key, pat in [
        ("peak_db", r"Peak level dB:\s*(-?[\d.]+)"),
        ("rms_db", r"RMS level dB:\s*(-?[\d.]+)"),
        ("noise_floor_db", r"Noise floor dB:\s*(-?[\d.]+)"),
        ("min_level", r"Min level:\s*(-?[\d.]+)"),
        ("max_level", r"Max level:\s*(-?[\d.]+)"),
    ]:
        m = re.search(pat, tail)
        out[key] = float(m.group(1)) if m else None
    if out["min_level"] is not None or out["max_level"] is not None:
        extreme = max(abs(v) for v in (out["min_level"], out["max_level"])
                      if v is not None)
        out["clipped"] = extreme >= 0.999
    else:
        out["clipped"] = None
    return out


def parse_volumedetect(stderr, instance_idx=0):
    """Parse the Nth volumedetect RESULTS block.

    FFmpeg 9 emits an init block per instance (n_samples: 0, no metrics)
    BEFORE the results block, at a different filter address. Splitting on
    the instance marker therefore yields init blocks too — only segments
    containing 'mean_volume:' are real results."""
    blocks = re.split(r"\[Parsed_volumedetect_\d+ @", stderr)
    results = [seg for seg in blocks[1:] if "mean_volume:" in seg]
    if instance_idx >= len(results):
        return None
    seg = results[instance_idx]
    mean = re.search(r"mean_volume:\s*(-?[\d.]+)\s*dB", seg)
    mx = re.search(r"max_volume:\s*(-?[\d.]+)\s*dB", seg)
    if not mean:
        return None
    return {
        "mean_db": float(mean.group(1)),
        "max_db": float(mx.group(1)) if mx else None,
    }


def parse_silencedetect(stderr, dur_sec):
    """Parse silencedetect intervals (clip-relative). Returns list of (s,e)."""
    starts, ends = [], []
    for line in stderr.splitlines():
        m = re.search(r"silence_start:\s*(-?\d+(?:\.\d+)?)", line)
        if m:
            starts.append(max(0.0, float(m.group(1))))
        m = re.search(r"silence_end:\s*(-?\d+(?:\.\d+)?)", line)
        if m:
            ends.append(max(0.0, float(m.group(1))))
    intervals = []
    open_start = starts.pop(0) if starts else None
    for e in ends:
        s = open_start if open_start is not None else 0.0
        intervals.append((s, e))
        open_start = starts.pop(0) if starts else None
    if open_start is not None:
        intervals.append((open_start, dur_sec))
    return intervals


# ── Analysis passes ─────────────────────────────────────────────────────────

def pass_overall(source, start_sec, dur_sec, pacing):
    """Pass 1: overall metrics on the OUTPUT timeline.

    One ffmpeg invocation, three parallel branches (each parser targets a
    unique output format, so stderr interleaving cannot confuse them):
      [a1] loudnorm JSON               — I / TP / LRA / threshold
      [a2] astats                       — peak / RMS / clipped samples
      [a3] highpass@100 + volumedetect  — sub-100Hz rumble estimate
    """
    prelude, edited = edit_map_chain(pacing)
    a_in = "[{}]".format(edited) if edited else "[0:a]"
    branches = (
        ";[a1]loudnorm=I={}:TP={}:LRA={}:print_format=json[o1]"
        .format(TARGET_I, TARGET_TP, TARGET_LRA)
        + ";[a2]astats[o2]"
        + ";[a3]highpass=f=100,volumedetect[o3]"
    )
    if edited:
        fc = prelude + ";{}asplit=3[a1][a2][a3]".format(a_in) + branches
    else:
        fc = "[0:a]asplit=3[a1][a2][a3]" + branches
    rc, err = run_ffmpeg(
        analysis_cmd(source, start_sec, dur_sec, fc, ["[o1]", "[o2]", "[o3]"]),
        timeout_sec=max(60, int(dur_sec * 4)),
    )
    if rc != 0:
        if "matches no streams" in err:
            log("source has no audio stream")
        else:
            log("overall analysis pass failed rc={}".format(rc))
        return None
    ln = parse_loudnorm_json(err)
    if ln is None:
        log("no loudnorm JSON in analysis output")
        return None
    stats = parse_astats_overall(err)
    hp = parse_volumedetect(err, instance_idx=0)
    return {
        "input_i": parse_db(ln.get("input_i")),
        "input_tp": parse_db(ln.get("input_tp")),
        "input_lra": parse_db(ln.get("input_lra")),
        "input_thresh": parse_db(ln.get("input_thresh")),
        "target_offset": parse_db(ln.get("target_offset")),
        "normalization_type": ln.get("normalization_type"),
        "astats": stats,
        "highpass_mean_db": hp["mean_db"] if hp else None,
    }


def pass_silence(source, start_sec, dur_sec, total_sec, pacing):
    """Pass 2: silencedetect (-35dB, 0.30s) on the OUTPUT timeline.

    `total_sec` is the analysis-stream duration (the pacing output duration
    when an edit map is active) so trailing silence is clamped correctly.
    ffmpeg may exit non-zero when trailing silence runs into EOF; the stderr
    metrics are still valid, so only a hard failure returns None.
    """
    prelude, edited = edit_map_chain(pacing)
    a_in = "[{}]".format(edited) if edited else "[0:a]"
    fc = (prelude + ";" if edited else "") + \
         "{}silencedetect=noise=-35dB:d=0.30[o]".format(a_in)
    rc, err = run_ffmpeg(
        analysis_cmd(source, start_sec, dur_sec, fc, ["[o]"]),
        timeout_sec=max(60, int(dur_sec * 4)),
    )
    if rc != 0 and "silence_" not in err:
        return None
    return parse_silencedetect(err, total_sec)


def pass_noise_floor(source, start_sec, dur_sec, silence_intervals, pacing,
                     content_rms_db=None):
    """Pass 3: measure the noise floor inside detected quiet regions.

    Hiss in the treatable range (-38..-26 dB) is LOUDER than the -35 dB
    silence-detection line, so it never registers as silence there. When the
    primary pass found no measurable silence, a second probe at -26 dB
    hunts for quiet regions that still exist above the hiss. A floor only
    counts when it sits FLOOR_MARGIN_DB below the content RMS — quiet
    speech is never mistaken for noise."""
    intervals = list(silence_intervals or [])
    if not intervals:
        # Floor hunt: quiet regions at the higher probe threshold.
        prelude, edited = edit_map_chain(pacing)
        a_in = "[{}]".format(edited) if edited else "[0:a]"
        fc = (prelude + ";" if edited else "") + \
             "{}silencedetect=noise={}dB:d=0.30[o]"\
             .format(a_in, fmt(NOISE_PROBE_DB))
        rc, err = run_ffmpeg(
            analysis_cmd(source, start_sec, dur_sec, fc, ["[o]"]),
            timeout_sec=max(60, int(dur_sec * 4)),
        )
        if rc == 0 or "silence_" in err:
            intervals = parse_silencedetect(err, dur_sec)
    if not intervals:
        return None
    ranked = sorted(intervals, key=lambda iv: iv[1] - iv[0], reverse=True)
    chosen = [iv for iv in ranked if iv[1] - iv[0] >= 0.25][:8]
    if not chosen:
        return None
    prelude, edited = edit_map_chain(pacing)
    a_in = "[{}]".format(edited) if edited else "[0:a]"
    parts = ["{}asplit={}".format(a_in, len(chosen))]
    for i in range(len(chosen)):
        parts.append("[an{}]".format(i))
    parts.append(";")
    for i, (s, e) in enumerate(chosen):
        parts.append("[an{}]atrim=start={}:end={},asetpts=PTS-STARTPTS[ant{}];"
                     .format(i, fmt(s), fmt(e), i))
    joined = "".join("[ant{}]".format(i) for i in range(len(chosen)))
    parts.append("{}concat=n={}:v=0:a=1,astats[o]".format(joined, len(chosen)))
    fc = (prelude + ";" if edited else "") + "".join(parts)
    rc, err = run_ffmpeg(
        analysis_cmd(source, start_sec, dur_sec, fc, ["[o]"]),
        timeout_sec=max(60, int(dur_sec * 4)),
    )
    if rc != 0:
        return None
    stats = parse_astats_overall(err)
    if not stats or stats.get("rms_db") is None:
        return None
    floor = stats["rms_db"]
    # Margin guard: a "floor" within FLOOR_MARGIN_DB of the content RMS is
    # quiet content, not noise — leave it alone.
    if content_rms_db is not None and floor > content_rms_db - FLOOR_MARGIN_DB:
        log("quiet-region RMS {:.1f} dB within {} dB of content RMS {:.1f} dB "
            "— not a noise floor".format(floor, FLOOR_MARGIN_DB, content_rms_db))
        return None
    return floor


def pass_speaker_means(source, start_sec, dur_sec, words, pacing):
    """Pass 4: per-speaker mean level over that speaker's merged word
    intervals.

    Words arrive on the SOURCE timeline; when a pacing plan exists they are
    remapped to the OUTPUT timeline first (mirroring Rust remap_words).
    Returns {speaker: {"mean_db": float, "intervals": [[s, e], ...]}} with
    intervals on the analysis timeline (output-relative when paced,
    clip-relative otherwise), or None when unusable.
    """
    if not words:
        return None
    speakers = {}
    for w in words:
        spk = (w.get("speaker") or "").strip()
        if spk:
            speakers.setdefault(spk, []).append((float(w["start"]), float(w["end"])))
    if len(speakers) < 2:
        return None
    # Remap word intervals to the output timeline when pacing is active.
    if pacing:
        retained = pacing.get("retained") or []
        out_map = [(float(r["srcStartSec"]), float(r["srcEndSec"]),
                    float(r["outStartSec"])) for r in retained]
        remapped = {}
        for spk, ivs in speakers.items():
            out_ivs = []
            for (s, e) in ivs:
                for (rs, re_, os_) in out_map:
                    if s >= rs - 0.02 and e <= re_ + 0.02:
                        out_ivs.append((os_ + (s - rs), os_ + (e - rs)))
                        break
            if out_ivs:
                remapped[spk] = out_ivs
        speakers = remapped
    else:
        # No pacing: analysis t is clip-relative (t=0 at start_sec).
        speakers = {
            spk: [(s - start_sec, e - start_sec) for (s, e) in ivs]
            for spk, ivs in speakers.items()
        }
    if len(speakers) < 2 or len(speakers) > 6:
        return None

    result = {}
    for spk, ivs in speakers.items():
        merged = merge_intervals(ivs)
        if not merged:
            continue
        total = sum(e - s for (s, e) in merged)
        if total < 0.5:
            continue  # too little material to measure reliably
        prelude, edited = edit_map_chain(pacing)
        a_in = "[{}]".format(edited) if edited else "[0:a]"
        parts = ["{}asplit={}".format(a_in, len(merged))]
        for i in range(len(merged)):
            parts.append("[as{}]".format(i))
        parts.append(";")
        for i, (s, e) in enumerate(merged):
            parts.append("[as{}]atrim=start={}:end={},asetpts=PTS-STARTPTS[ast{}];"
                         .format(i, fmt(s), fmt(e), i))
        joined = "".join("[ast{}]".format(i) for i in range(len(merged)))
        parts.append("{}concat=n={}:v=0:a=1,volumedetect[o]"
                     .format(joined, len(merged)))
        fc = (prelude + ";" if edited else "") + "".join(parts)
        rc, err = run_ffmpeg(
            analysis_cmd(source, start_sec, dur_sec, fc, ["[o]"]),
            timeout_sec=max(60, int(dur_sec * 4)),
        )
        if rc != 0:
            continue
        vd = parse_volumedetect(err, instance_idx=0)
        if vd and vd.get("mean_db") is not None:
            result[spk] = {"mean_db": vd["mean_db"], "intervals": merged}
    return result if len(result) >= 2 else None


def merge_intervals(ivs, tol=0.25):
    """Merge overlapping/nearby intervals (gap <= tol); drop sub-50ms slivers.

    The 250ms merge tolerance keeps the volume-enable expression compact
    while never splitting a speaker's continuous speech."""
    if not ivs:
        return []
    ivs = sorted(ivs)
    out = [list(ivs[0])]
    for s, e in ivs[1:]:
        if s <= out[-1][1] + tol:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for (s, e) in out if e - s >= 0.05]


# ── Decision policy ─────────────────────────────────────────────────────────

MAX_ENABLE_TERMS = 48  # volume-enable expression interval cap


def decide(analysis, silence_intervals, dur_sec, speaker_means, remeasure=None):
    """Produce the staged decision. Returns (stages, filter_chain, reason).

    `remeasure` (optional callable) re-measures loudness AFTER a pre-filter
    chain (two-pass loudnorm); it returns a dict of fresh input_* values or
    None on failure. `analysis` is updated in place when it succeeds.
    """
    stages = []
    filters = []

    input_i = analysis.get("input_i")
    input_tp = analysis.get("input_tp")
    input_lra = analysis.get("input_lra")
    input_thresh = analysis.get("input_thresh")
    target_offset = analysis.get("target_offset")
    astats = analysis.get("astats") or {}
    clipped = astats.get("clipped")
    fullband_mean = astats.get("rms_db")  # fullband reference (astats RMS)
    hp_mean = analysis.get("highpass_mean_db")

    # ── Gate 1: genuine silence / no speech energy ──────────────────────────
    silence_total = sum(e - s for (s, e) in (silence_intervals or []))
    silence_ratio = (silence_total / dur_sec) if dur_sec > 0 else 0.0
    if input_i is None or input_i < MIN_SPEECH_I:
        stages.append({"name": "gate", "applied": False,
                       "reason": "no measurable speech energy (integrated "
                                 "loudness unmeasurable or below {:.0f} LUFS) — "
                                 "genuine silence is never amplified"
                                 .format(MIN_SPEECH_I)})
        return stages, "", "genuine silence — left untouched"
    if silence_ratio > SILENCE_RATIO_MAX:
        stages.append({"name": "gate", "applied": False,
                       "reason": "silence ratio {:.0%} exceeds {:.0%} — "
                                 "genuine silence is never amplified"
                                 .format(silence_ratio, SILENCE_RATIO_MAX)})
        return stages, "", "genuine silence — left untouched"

    # ── Stage 1: HIGHPASS (rumble) — measured sub-100Hz energy ───────────────
    # Decided FIRST so the chain order is highpass -> afftdn (rumble removed
    # before the FFT denoiser sees the signal).
    if (fullband_mean is not None and hp_mean is not None
            and (fullband_mean - hp_mean) >= RUMBLE_GATE_DB):
        filters.append("highpass=f={}".format(HIGHPASS_HZ))
        stages.append({"name": "highpass", "applied": True,
                       "rumbleDb": round(fullband_mean - hp_mean, 2),
                       "reason": "sub-100Hz energy {:.1f} dB below fullband mean "
                                 "— high-pass at {} Hz (speech band preserved)"
                                 .format(fullband_mean - hp_mean, HIGHPASS_HZ)})
    else:
        delta = (round(fullband_mean - hp_mean, 2)
                 if (fullband_mean is not None and hp_mean is not None) else None)
        stages.append({"name": "highpass", "applied": False,
                       "rumbleDb": delta,
                       "reason": "no significant sub-100Hz rumble "
                                 "({} dB delta < {:.0f} dB gate)"
                                 .format(delta if delta is not None else "unmeasured",
                                         RUMBLE_GATE_DB)})

    # ── Stage 2: DENOISE (afftdn) — only with a measured, loud floor ─────────
    noise_floor = analysis.get("noise_floor_db")
    if noise_floor is not None and noise_floor > NOISE_FLOOR_GATE:
        nf = max(-80.0, min(-20.0, noise_floor + NOISE_FLOOR_MARGIN))
        filters.append("afftdn=nf={}:nr={}".format(fmt(nf), fmt(AFFTDN_NR)))
        stages.append({"name": "denoise", "applied": True,
                       "noiseFloorDb": noise_floor,
                       "afftdnNf": nf,
                       "reason": "measured noise floor {:.1f} dB exceeds {:.0f} dB "
                                 "gate — conservative FFT denoise (nr={:.0f})"
                                 .format(noise_floor, NOISE_FLOOR_GATE, AFFTDN_NR)})
    else:
        why = ("noise floor {:.1f} dB below {:.0f} dB gate — no denoising"
               .format(noise_floor, NOISE_FLOOR_GATE)
               if noise_floor is not None
               else "no measurable silence to estimate a noise floor — "
                    "denoising unproven, audio untouched")
        stages.append({"name": "denoise", "applied": False, "reason": why})

    # ── Stage 3: SPEAKER BALANCE — measured per-speaker mismatch ─────────────
    applied_boost = 0.0
    if speaker_means and len(speaker_means) >= 2:
        loudest = max(v["mean_db"] for v in speaker_means.values())
        quietest = min(v["mean_db"] for v in speaker_means.values())
        mismatch = loudest - quietest
        if mismatch > SPEAKER_MISMATCH_LU:
            quiet_speaker = min(speaker_means,
                                key=lambda s: speaker_means[s]["mean_db"])
            quiet_mean = speaker_means[quiet_speaker]["mean_db"]
            if quiet_mean < MIN_SPEECH_I:
                stages.append({"name": "speaker_balance", "applied": False,
                               "mismatchDb": round(mismatch, 2),
                               "reason": "quieter speaker's speech ({:.1f} dB) "
                                         "is effectively silence — never "
                                         "amplified".format(quiet_mean)})
            else:
                ivs = speaker_means[quiet_speaker]["intervals"]
                if len(ivs) > MAX_ENABLE_TERMS:
                    stages.append({"name": "speaker_balance", "applied": False,
                                   "mismatchDb": round(mismatch, 2),
                                   "reason": "mismatch {:.1f} dB justified a boost "
                                             "but speech is too fragmented "
                                             "({} intervals) — correction "
                                             "skipped as unsafe"
                                             .format(mismatch, len(ivs))})
                else:
                    gain = min(SPEAKER_GAIN_CAP_DB, mismatch / 2.0)
                    enable = "+".join("between(t,{},{})".format(fmt(s), fmt(e))
                                      for (s, e) in ivs)
                    filters.append("volume={:.2f}dB:enable='{}'".format(gain, enable))
                    applied_boost = gain
                    stages.append({"name": "speaker_balance", "applied": True,
                                   "mismatchDb": round(mismatch, 2),
                                   "boostedSpeaker": quiet_speaker,
                                   "gainDb": round(gain, 2),
                                   "intervals": len(ivs),
                                   "reason": "speaker mismatch {:.1f} dB exceeds "
                                             "{:.0f} dB gate — quieter speaker "
                                             "boosted {:.1f} dB (capped) only "
                                             "during their speech"
                                             .format(mismatch, SPEAKER_MISMATCH_LU,
                                                     gain)})
        else:
            stages.append({"name": "speaker_balance", "applied": False,
                           "mismatchDb": round(mismatch, 2),
                           "reason": "speaker mismatch {:.1f} dB within natural "
                                     "variation (< {:.0f} dB gate)"
                                     .format(mismatch, SPEAKER_MISMATCH_LU)})
    else:
        stages.append({"name": "speaker_balance", "applied": False,
                       "reason": "fewer than two measurable speakers"})

    # ── Two-pass loudness: re-measure after pre-filters when they matter ────
    loudness_needed = (input_i is not None
                       and abs(input_i - TARGET_I) > TOLERANCE_LU)
    remeasured = False
    if filters and remeasure and (loudness_needed or applied_boost > 0):
        fresh = remeasure(",".join(filters))
        if fresh is None or fresh.get("input_i") is None:
            return (stages, "",
                    "re-measurement after pre-filters failed — uncertain "
                    "audio is not touched")
        for k in ("input_i", "input_tp", "input_lra", "input_thresh",
                  "target_offset"):
            if fresh.get(k) is not None:
                analysis[k] = fresh[k]
        input_i = analysis["input_i"]
        input_tp = analysis["input_tp"]
        input_lra = analysis["input_lra"]
        input_thresh = analysis["input_thresh"]
        target_offset = analysis["target_offset"]
        loudness_needed = abs(input_i - TARGET_I) > TOLERANCE_LU
        remeasured = True

    # ── Stage 4: LOUDNESS (loudnorm linear, two-pass measured) ───────────────
    if loudness_needed and input_lra is not None and input_thresh is not None \
            and input_tp is not None:
        lra_target = max(TARGET_LRA, min(50.0, input_lra))
        offset = target_offset if target_offset is not None else 0.0
        ln = ("loudnorm=I={}:TP={}:LRA={}:measured_I={}:measured_LRA={}:"
              "measured_TP={}:measured_thresh={}:offset={}:linear=true"
              .format(TARGET_I, TARGET_TP, lra_target,
                      input_i, input_lra, input_tp, input_thresh, offset))
        filters.append(ln)
        filters.append("aresample=48000")  # loudnorm outputs 192 kHz
        stage = {"name": "loudness", "applied": True,
                 "inputI": input_i, "targetI": TARGET_I,
                 "lraTarget": lra_target, "remeasured": remeasured,
                 "reason": "integrated loudness {:.1f} LUFS deviates > {:.0f} LU "
                           "from the {:.0f} LUFS target — linear two-pass "
                           "loudnorm (dynamics preserved)"
                           .format(input_i, TOLERANCE_LU, TARGET_I)}
        stages.append(stage)
    elif loudness_needed:
        stages.append({"name": "loudness", "applied": False,
                       "reason": "loudness correction justified but measurements "
                                 "incomplete — uncertain audio is not touched"})
    else:
        stages.append({"name": "loudness", "applied": False,
                       "reason": "integrated loudness {:.1f} LUFS already within "
                                 "{:.0f} LU of target — no normalization"
                                 .format(input_i, TOLERANCE_LU)})

    # ── Stage 5: PEAK SAFETY (alimiter) — only when loudnorm is absent ──────
    # loudnorm's TP=-1.5 dBTP already guarantees the true-peak ceiling (it
    # falls back to dynamic normalization when linear gain would overshoot),
    # so a limiter after it would be redundant processing.
    loudnorm_applied = any(s.get("name") == "loudness" and s.get("applied")
                           for s in stages)
    if loudnorm_applied:
        stages.append({"name": "peak_safety", "applied": False,
                       "reason": "true-peak ceiling guaranteed by loudnorm "
                                 "TP={:.1f} dBTP — no extra limiter"
                                 .format(TARGET_TP)})
    else:
        # Predict the post-boost true peak (speaker boost is the only gain).
        predicted_tp = (input_tp + applied_boost) if input_tp is not None else None
        if clipped:
            stages.append({"name": "peak_safety", "applied": True,
                           "clipped": True,
                           "reason": "samples at/over full scale (Min/Max level "
                                     ">= 0.999) — limiter engaged (mitigation, "
                                     "not reconstruction)"})
            filters.append("alimiter=limit={}:level=false:attack=5:release=50"
                           .format(fmt(math.pow(10, LIMITER_CEIL / 20.0))))
        elif predicted_tp is not None and predicted_tp > CLIP_TP_GATE:
            stages.append({"name": "peak_safety", "applied": True,
                           "inputTp": input_tp,
                           "predictedTp": round(predicted_tp, 2),
                           "reason": "true peak {:.1f} dBTP (predicted {:.1f} "
                                     "after boost) exceeds {:.0f} dBTP gate — "
                                     "limiter engaged"
                                     .format(input_tp, predicted_tp, CLIP_TP_GATE)})
            filters.append("alimiter=limit={}:level=false:attack=5:release=50"
                           .format(fmt(math.pow(10, LIMITER_CEIL / 20.0))))
        else:
            tp_txt = ("{:.1f} dBTP within safe range".format(input_tp)
                      if input_tp is not None else "true peak unmeasured")
            stages.append({"name": "peak_safety", "applied": False,
                           "reason": tp_txt + " — no limiter"})

    chain = ",".join(filters)
    if chain:
        reason = "processed: " + ", ".join(
            s["name"] for s in stages if s.get("applied"))
    else:
        reason = "audio already good — no processing needed"
    return stages, chain, reason


# ── Main ────────────────────────────────────────────────────────────────────

def load_json_file(path):
    try:
        with open(path, "r", encoding="utf-8-sig") as fh:
            return json.load(fh)
    except (OSError, ValueError) as exc:
        log("cannot read {}: {}".format(path, exc))
        return None


def main():
    args = sys.argv[1:]
    if len(args) < 3:
        sys.stderr.write(
            "usage: audio_intelligence.py SOURCE START_MS END_MS "
            "[WORDS_JSON] [PACING_JSON]\n")
        return 2
    source = args[0]
    start_sec = int(args[1]) / 1000.0
    end_sec = int(args[2]) / 1000.0
    words = load_json_file(args[3]) if len(args) > 3 and args[3] else None
    pacing = load_json_file(args[4]) if len(args) > 4 and args[4] else None
    # Shape-validate: words must be a list of dicts; anything else is treated
    # as no transcript (speaker analysis simply won't run).
    if words is not None and not (
            isinstance(words, list)
            and all(isinstance(w, dict) for w in words)):
        log("words JSON is not a list of objects — ignoring transcript")
        words = None
    # Shape-validate: pacing must be a dict with a usable retained map.
    if pacing is not None and not (
            isinstance(pacing, dict)
            and isinstance(pacing.get("retained"), list)):
        log("pacing JSON is not a plan object — treating as no pacing")
        pacing = None

    if not os.path.isfile(source):
        emit({"status": "error", "reason": "source not found: {}".format(source),
              "stages": [], "filterChain": "", "analysis": {}})
        return 0
    if end_sec <= start_sec:
        emit({"status": "skipped", "reason": "non-positive range",
              "stages": [], "filterChain": "", "analysis": {}})
        return 0
    if pacing and pacing.get("status") != "ok":
        log("pacing plan status={} — treating as no pacing".format(
            pacing.get("status")))
        pacing = None

    dur_sec = min(end_sec - start_sec, MAX_ANALYSIS_SEC)
    # Analysis-stream duration: the pacing OUTPUT duration when an edit map
    # is active (trailing-silence clamping needs the true stream length).
    total_sec = dur_sec
    if pacing:
        try:
            total_sec = min(float(pacing.get("outputDurationSec") or dur_sec),
                            dur_sec)
        except (TypeError, ValueError):
            total_sec = dur_sec

    # Pass 1: overall metrics (edit-map aware).
    analysis = pass_overall(source, start_sec, dur_sec, pacing)
    if analysis is None:
        emit({"status": "skipped",
              "reason": "analysis unavailable — audio untouched (uncertain)",
              "stages": [], "filterChain": "", "analysis": {}})
        return 0

    # Pass 2: silence intervals (edit-map aware).
    silence = pass_silence(source, start_sec, dur_sec, total_sec, pacing) or []

    # Pass 3: noise floor inside quiet regions (floor hunt when needed).
    analysis["noise_floor_db"] = pass_noise_floor(
        source, start_sec, dur_sec, silence, pacing,
        content_rms_db=(analysis.get("astats") or {}).get("rms_db"))

    # Pass 4: per-speaker means (edit-map aware, word-timestamp based).
    speaker_means = pass_speaker_means(
        source, start_sec, dur_sec, words, pacing)

    # Two-phase re-measure callback: loudness AFTER pre-filters, so the
    # loudnorm measured_* values describe exactly the stream it will process.
    def remeasure(pre_chain):
        prelude, edited = edit_map_chain(pacing)
        a_in = "[{}]".format(edited) if edited else "[0:a]"
        fc = (prelude + ";" if edited else "") + \
             "{}{},loudnorm=I={}:TP={}:LRA={}:print_format=json[o]"\
             .format(a_in, pre_chain, TARGET_I, TARGET_TP, TARGET_LRA)
        rc, err = run_ffmpeg(
            analysis_cmd(source, start_sec, dur_sec, fc, ["[o]"]),
            timeout_sec=max(60, int(dur_sec * 4)),
        )
        if rc != 0:
            return None
        ln = parse_loudnorm_json(err)
        if ln is None:
            return None
        return {
            "input_i": parse_db(ln.get("input_i")),
            "input_tp": parse_db(ln.get("input_tp")),
            "input_lra": parse_db(ln.get("input_lra")),
            "input_thresh": parse_db(ln.get("input_thresh")),
            "target_offset": parse_db(ln.get("target_offset")),
        }

    stages, chain, reason = decide(
        analysis, silence, total_sec, speaker_means, remeasure)

    emit({
        "status": "ok",
        "reason": reason,
        "stages": stages,
        "filterChain": chain,
        "analysis": {
            "inputI": analysis.get("input_i"),
            "inputTp": analysis.get("input_tp"),
            "inputLra": analysis.get("input_lra"),
            "inputThresh": analysis.get("input_thresh"),
            "targetOffset": analysis.get("target_offset"),
            "astatsPeakDb": (analysis.get("astats") or {}).get("peak_db"),
            "astatsRmsDb": (analysis.get("astats") or {}).get("rms_db"),
            "clipped": (analysis.get("astats") or {}).get("clipped"),
            "minLevel": (analysis.get("astats") or {}).get("min_level"),
            "maxLevel": (analysis.get("astats") or {}).get("max_level"),
            "highpassMeanDb": analysis.get("highpass_mean_db"),
            "noiseFloorDb": analysis.get("noise_floor_db"),
            "silenceRatio": round(
                sum(e - s for (s, e) in silence) / total_sec, 3)
            if total_sec > 0 else None,
            "speakerMeansDb": ({spk: v["mean_db"]
                                for spk, v in speaker_means.items()}
                               if speaker_means else None),
        },
    })
    return 0


def emit(plan):
    print(json.dumps(plan, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
