#!/usr/bin/env python3
"""AutoShorts 10.0 — Breath / speech-pause DETECTOR (Smart Pacing 2.0).

A dedicated, evidence-driven detector for INTERNAL speech pauses inside a
fixed candidate timeline. It is deliberately NOT a general audio engine
(Part 11): it reads word timestamps and measures acoustic features INSIDE
each word gap, then classifies the gap into conceptual pause categories
(Part 4):

    A. BREATH / NATURAL SPEECH PAUSE
    B. HESITATION / WAITING PAUSE
    C. NORMAL WORD GAP
    D. SENTENCE / MEANING PAUSE
    E. DRAMATIC / INTENTIONAL PAUSE
    F. SPEAKER / SCENE TRANSITION
    G. NON-SPEECH / UNKNOWN

The detector never assumes LOW VOLUME == BREATH (Part 3): silence, sentence
pauses, music, room tone, and detector dropouts are also low-energy. It
combines three measured acoustic features with the transcript structure:

  * HF_RATIO   fraction of gap-interior energy above 2 kHz. Aspiration
               (breath noise, "h", fricative tails) is broadband and lands
               in [0.15, 1.0]; voiced phonetic boundaries, plosive gaps and
               detector artifacts with no high-frequency content land near
               [0.002, 0.05]. This is the single strongest discriminator
               found on real AutoShorts 10.0 audio (see scratch/
               breath_calibration.py).
  * ABOVE_FLOOR gap-interior energy in dB above the clip's own 5th-percentile
               silence floor. Dead air sits at [+35, +60] dB; a gap that is
               merely "quiet speech / room tone / music bed" stays near the
               floor and is NOT treated as removable air.
  * CONTRAST   speech reference level (p75 of word-interior RMS) minus gap
               energy. A true dead/waiting pause is far below the speaker;
               music or sound effects filling the gap have low contrast and
               must not be cut (Part 13).

Derived from measurements on the real Ronaldo candidate (490-551s):
  0.06-0.10s gaps: HF ~0.005  -> phonetic boundaries (protected)
  0.15-0.25s gaps: HF ~0.55-1.0 -> aspiration / micro breath (reducible)
  0.50-0.90s gaps: above_floor ~+23 dB, contrast ~+2..6 dB (compressible)
  sentence gaps: above_floor ~+50 dB (dead air; v1 already compresses)

This module is READ-ONLY with respect to Audio Intelligence (Part 11): it
runs its own narrow analysis pass over the candidate window only, exactly
as the v1 engine already runs its own silencedetect pass. It does not
modify any other signal-processing pipeline.

CLI contract (mirrors smart_pacing.py conventions):
    python breath_detect.py SOURCE START_MS END_MS WORDS_JSON

Output: the LAST stdout line is a JSON object:
    {
      "gaps": [
        {"idx": 0,
         "srcStartSec": ..., "srcEndSec": ...,
         "durationSec": ...,
         "category": "breath_pause" | "waiting_pause" | "normal_word_gap" |
                     "sentence_pause" | "dramatic_pause" |
                     "speaker_transition" | "nonspeech_unknown",
         "confidence": 0.0-1.0,
         "evidence": {"hfRatio": ..., "aboveFloorDb": ..., "contrastDb": ...,
                      "flatness": ..., "speechRefDb": ..., "floorDb": ...},
         "reason": "..."}
      ],
      "speechRefDb": ..., "floorDb": ..., "sampleRate": 16000
    }

All times are ABSOLUTE source times. A failure to analyze (no audio, no
numpy, decode error) yields status "error" and an empty gap list so the
caller can fall back to the v1 engine without inventing evidence.
"""

import json
import math
import subprocess
import sys

try:
    import numpy as np
except ImportError:  # numpy is not always present; the caller degrades
    np = None

SAMPLE_RATE = 16000
WIN_MS = 20
HOP_MS = 10
DECODE_RAMP = 64  # ffmpeg f32le decoder warm-up samples to discard

# ── Evidence thresholds (calibrated on real AutoShorts 10.0 audio) ──────────

# HF energy fraction separator between aspiration and voiced/phonetic content.
# The 2 kHz line sits above the low formants of normal speech (F1 ~ 300-900
# Hz, F2 ~ 900-2500 Hz) so that voiced phonetic leakage into a short gap is
# NOT read as aspiration. Real glottal spectra also roll off (~-12 dB/oct),
# so high harmonics contribute little; a whole-segment FFT (not per-block
# windowed ratios) keeps harmonic lines narrow and measures the true noise
# floor between them.
HF_BREATH_MIN = 0.15  # at/above this -> breath-like broadband content
HF_PHONETIC_MAX = 0.08  # below this -> no aspiration; treat as phonetic gap

# Gap-interior energy above the clip silence floor (dB). Below this the
# gap is quiet but not necessarily dead air: real breaths on real
# AutoShorts audio measure +23..26 dB above the floor, so a flat 30 dB bar
# would protect every genuine breath. Low-level gaps therefore fall through
# to the tonality/aspiration guards below, which decide on evidence.
DEAD_AIR_MIN = 30.0  # clearly dead air: real silence, not room tone

# Gap energy relative to the speaker reference (median word-interior level).
WAITING_CONTRAST_MIN = 12.0  # a waiting pause is clearly below the speaker
# Content filling a gap that is as loud as the speaker is NOT a pause - it
# is music, sound effects, or speech-level room content. Calibrated on the
# real candidate: music/SFX beds sit within ~6 dB of the speaker median,
# genuine breaths sit 7-25 dB below it.
CONTENT_CONTRAST_MAX = 6.0

# Spectral flatness guards. Breath/aspiration is noise-like (high flatness);
# sustained tones and some music beds are spectrally structured (low).
TONAL_FLATNESS_MAX = 0.005  # at/below this AND quiet -> tonal content

# A gap whose interior is essentially AT the clip's own silence floor (< this
# many dB above it) is indistinguishable from the noise floor itself: room
# tone, hum, or dither. Collapsing it would create an on/off discontinuity in
# the background and gain nothing. Real content gaps (music beds, breaths,
# sentence air) all measure +14 dB or more above the floor on real
# AutoShorts audio, so this bar cannot catch them.
FLOOR_MARGIN_MIN = 10.0

# Minimum usable interior length for reliable measurement (seconds). Below
# this the gap is too short to measure meaningfully and is classified by
# duration/context evidence only.
MIN_MEASURABLE = 0.045

# Gap duration bands (seconds).
DUR_MICRO_MIN = 0.06  # micro pauses: 0.06-0.15s
DUR_SHORT_MIN = 0.15  # short pauses: 0.15-0.30s
DUR_BREATH_MIN = 0.30  # breath band: 0.30-0.90s
DUR_WAITING_MIN = 0.50  # waiting band: >= 0.50s
DUR_LONG_MIN = 0.90  # long pauses: >= 0.90s (v1 handles these)

# A dramatic/intentional pause is a long, very quiet gap that is NOT
# between two sentence terminators and NOT adjacent to a speaker change.
# We protect it unless the evidence says it is merely dead waiting time.
DRAMATIC_MIN = 1.10  # long enough to be a deliberate beat

SENTENCE_TERMINATORS = (".", "!", "?", ";", ":", "\u0964", "\u0965")

EPS = 1e-9


def log(msg):
    sys.stderr.write("[BreathDetect] {}\n".format(msg))
    sys.stderr.flush()


# ── Audio decode helpers ─────────────────────────────────────────────────────

def decode_range(source_path, t0, t1):
    """Decode [t0, t1] seconds of the source audio to mono float32."""
    if t1 - t0 < 1e-3:
        t1 = t0 + 0.01
    p = subprocess.run(
        ["ffmpeg", "-nostats", "-ss", "{:.3f}".format(t0),
         "-t", "{:.3f}".format(t1 - t0),
         "-i", source_path, "-vn",
         "-f", "f32le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-"],
        capture_output=True)
    if np is None:
        return None
    x = np.frombuffer(p.stdout, dtype="<f4")
    return x[DECODE_RAMP:] if x.size > DECODE_RAMP * 2 else x


def rms_db(x):
    if x is None or x.size == 0:
        return None
    m = float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))
    if m <= 1e-12:
        return None
    return 20 * math.log10(m)


def _fft_hf_ratio(seg, sample_rate, hf_hz=2000.0):
    """Fraction of gap-interior energy above `hf_hz` (band energy ratio).

    Computed on a single whole-segment FFT rather than per-block windowed
    ratios. The per-block approach is unusable for this discriminator: a
    windowed voiced segment leaks the harmonic tails of its lowest bins
    across the whole spectrum, and a 100 ms phonetic boundary then measures
    HF ~0.1-0.3 and is misread as aspiration. On the whole segment the
    harmonic lines stay narrow and the energy between them is genuinely the
    aspiration noise floor, which is what we actually want to measure.
    """
    n = seg.size
    if n < 256:
        return None
    spec = np.abs(np.fft.rfft(seg * np.hanning(n))) ** 2
    freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)
    total = float(np.sum(spec)) + 1e-15
    return float(np.sum(spec[freqs > hf_hz])) / total


def _spectral_flatness(seg):
    """Wiener-entropy flatness in [0,1]. Tonal music/sine beds are ~0;
    breath/noise is high."""
    n = seg.size
    if n < 512:
        return None
    win = min(2048, n)
    vals = []
    for off in range(0, n - win + 1, win):
        block = seg[off:off + win]
        spec = np.abs(np.fft.rfft(block * np.hanning(block.size))) ** 2
        spec = np.maximum(spec, 1e-15)
        vals.append(float(np.exp(np.mean(np.log(spec))) / np.mean(spec)))
    return float(np.mean(vals)) if vals else None


# ── Gap analysis ─────────────────────────────────────────────────────────────

def _word_tail_end(word):
    """The time at which the previous word's acoustic tail has decayed
    enough for a clean gap-interior measurement."""
    dur = float(word["end"]) - float(word["start"])
    # Longer words carry more reverberant/decay tail; allow up to 60ms.
    return float(word["end"]) + min(0.060, max(0.020, 0.15 * dur))


def _word_head_start(word):
    """The time at which the next word's onset (breath build-up, attack)
    begins. Speech onsets are preceded by rising aspiration energy, so the
    interior must stop before this."""
    dur = float(word["end"]) - float(word["start"])
    return float(word["start"]) - min(0.060, max(0.020, 0.15 * dur))


def _interior_bounds(gap_start, gap_end, prev_word=None, next_word=None):
    """Interior window of a gap that excludes word tails and onsets.

    Word tails decay with reverb and onsets are preceded by rising
    aspiration, both of which pollute a naive centered measurement. We
    therefore back off from each edge by the word's own decay allowance,
    falling back to a proportional window for very short gaps."""
    dur = gap_end - gap_start
    if dur < 1e-4:
        return None
    back_off = 0.040
    if prev_word is not None:
        back_off = max(back_off, _word_tail_end(prev_word) - gap_start)
    if next_word is not None:
        back_off = max(back_off, gap_end - _word_head_start(next_word))
    back_off = min(back_off, dur * 0.40)
    a, b = gap_start + back_off, gap_end - back_off
    if b - a < MIN_MEASURABLE:
        a = gap_start + dur * 0.35
        b = gap_end - dur * 0.35
    if b - a < 0.010:
        return None
    return a, b


def _word_interior_db(x, rel, word):
    """RMS of a 40ms window centered on the word's interior."""
    mid = (word["start"] + word["end"]) / 2.0
    half = 0.020
    i = rel(mid)
    s, e = max(i - int(half * SAMPLE_RATE), 0), min(i + int(half * SAMPLE_RATE), x.size)
    if e - s < 2:
        return None
    return rms_db(x[s:e])


def analyze_gap(ctx, gap_start, gap_end, prev_word, next_word):
    """Measure one word gap and classify it. `ctx` holds decoded audio and
    reference levels for the candidate window (see build_context)."""
    bounds = _interior_bounds(gap_start, gap_end, prev_word, next_word)
    dur = gap_end - gap_start
    ev = {"hfRatio": None, "aboveFloorDb": None, "contrastDb": None,
          "flatness": None, "speechRefDb": round(ctx["speech_ref_db"], 1) if ctx.get("speech_ref_db") else None,
          "floorDb": round(ctx["floor_db"], 1) if ctx.get("floor_db") else None}

    if bounds is None or ctx.get("x") is None:
        # Too short to measure: classify on structure alone.
        return _classify(ctx, gap_start, gap_end, prev_word, next_word, ev)

    a, b = bounds
    rel = ctx["rel"]
    ia, ib = rel(a), rel(b)
    ia, ib = max(ia, 0), min(ib, ctx["x"].size)
    if ib - ia < int(0.008 * SAMPLE_RATE):
        return _classify(ctx, gap_start, gap_end, prev_word, next_word, ev)

    seg = ctx["x"][ia:ib]
    gap_db = rms_db(seg)
    if gap_db is not None:
        ev["aboveFloorDb"] = round(gap_db - ctx["floor_db"], 1) if ctx.get("floor_db") is not None else None
        ev["contrastDb"] = round(ctx["speech_ref_db"] - gap_db, 1) if ctx.get("speech_ref_db") is not None else None
    ev["hfRatio"] = _fft_hf_ratio(seg, SAMPLE_RATE)
    ev["flatness"] = _spectral_flatness(seg)

    return _classify(ctx, gap_start, gap_end, prev_word, next_word, ev)


def _ends_sentence(word):
    t = (word.get("text") or "").strip()
    return t.endswith(SENTENCE_TERMINATORS)


def _is_filler_like(word):
    t = (word.get("text") or "").strip().lower().strip(".,!?;:")
    return t in ("um", "uh", "uhm", "umm", "erm", "hmm", "mm", "mmm", "ah", "er", "eh")


def _classify(ctx, gap_start, gap_end, prev_word, next_word, ev):
    """Evidence combination into a pause category (Part 4).

    The categories are ordered so the most protective decision wins when
    evidence is weak: speaker transition > dramatic/meaning > sentence >
    non-speech/unknown > waiting > breath > normal.
    """
    dur = gap_end - gap_start
    hf = ev.get("hfRatio")
    above = ev.get("aboveFloorDb")
    contrast = ev.get("contrastDb")
    flat = ev.get("flatness")

    same_speaker = (prev_word.get("speaker") or None) == (next_word.get("speaker") or None)
    prev_ends_sent = _ends_sentence(prev_word)
    prev_is_filler = _is_filler_like(prev_word)
    next_is_filler = _is_filler_like(next_word)

    # F. SPEAKER / SCENE TRANSITION — must never be collapsed (Part 4).
    if not same_speaker:
        return _result("speaker_transition", 0.95, dur, ev,
                       "speaker change across the gap - protected")

    # G. NON-SPEECH / UNKNOWN — protected unless proven to be removable air.
    # True digital silence inside the interior means the decoder produced no
    # energy at all: an honest detector dropout or a truncated decode. Never
    # treat absence of signal as a removable pause (Part 3).
    if above is None and contrast is None:
        if hf is None:
            return _result("nonspeech_unknown", 0.60, dur, ev,
                           "gap too short to measure - no evidence of removable air")
        return _result("nonspeech_unknown", 0.75, dur, ev,
                       "true digital silence in the gap interior - detector dropout - protected")
    if above is not None and above < FLOOR_MARGIN_MIN:
        return _result("nonspeech_unknown", 0.70, dur, ev,
                       "gap at the clip noise floor ({:+.1f} dB) - room tone / dither - protected".format(above))

    # Content-filled gaps (music / SFX / speech-level room tone) are as loud
    # as the speaker themselves. This is the single strongest protection
    # against cutting music or sound effects (Part 13) and against assuming
    # LOW VOLUME == BREATH (Part 3): a music bed is low relative to peak
    # speech but NOT low relative to the speaker's typical level.
    if contrast is not None and contrast < CONTENT_CONTRAST_MAX:
        if flat is not None and flat < TONAL_FLATNESS_MAX:
            return _result("nonspeech_unknown", 0.80, dur, ev,
                           "speech-level spectrally-structured content (music/tone) - protected")
        return _result("nonspeech_unknown", 0.78, dur, ev,
                       "gap as loud as the speaker - content-filled, not a pause")

    # C. NORMAL WORD GAP — a very short gap with no high-frequency
    # aspiration is a phonetic boundary or plosive transition at any level
    # band (Part 5: never blindly cut 60-150ms gaps).
    if dur < DUR_SHORT_MIN and hf is not None and hf < HF_PHONETIC_MAX:
        return _result("normal_word_gap", 0.90, dur, ev,
                       "micro gap ({:.0f}ms) without aspiration - phonetic boundary preserved".format(dur * 1000))

    # Between ROOM_TONE_MAX and DEAD_AIR_MIN the gap is quiet but not dead.
    # It is only reducible air if it is broadband AND clearly below the
    # speaker (contrast). A gap in this band that is speech-level is music
    # or sound effects; one without broadband aspiration is room tone, hum,
    # voiced phonetic leakage, or a music tail. Calibrated on the real
    # candidate: real breaths measure +23..26 dB above the floor, so a flat
    # 30 dB bar would have protected every genuine breath.
    if above is not None and above < DEAD_AIR_MIN:
        if flat is not None and flat < TONAL_FLATNESS_MAX:
            return _result("nonspeech_unknown", 0.72, dur, ev,
                           "low-level tonal content below the dead-air bar - protected")
        if hf is not None and hf < HF_BREATH_MIN:
            return _result("nonspeech_unknown", 0.72, dur, ev,
                           "low-level gap without broadband aspiration - protected")

    # E. DRAMATIC / INTENTIONAL PAUSE — protected unless strong evidence
    # says it is merely dead waiting time (Part 13).
    if dur >= DRAMATIC_MIN:
        dead_air = above is not None and above >= 45.0 and contrast is not None and contrast >= 18.0
        if dead_air and not prev_ends_sent:
            return _result("waiting_pause", 0.72, dur, ev,
                           "long gap measured as dead air mid-sentence - reduced")
        if prev_ends_sent:
            return _result("sentence_pause", 0.88, dur, ev,
                           "long sentence-final pause - v1 compression applies")
        return _result("dramatic_pause", 0.85, dur, ev,
                       "long mid-sentence pause preserved as intentional beat")

    # D. SENTENCE / MEANING PAUSE — handled conservatively: v1 compresses
    # these only above PAUSE_MIN_SENT, and only when verified silent.
    if prev_ends_sent or next_is_filler:
        return _result("sentence_pause", 0.85, dur, ev,
                       "sentence/meaning boundary pause - conservative")

    # A/B. The reducible speech-adjacent pauses. Breath/aspiration requires
    # BOTH broadband HF content AND being clearly below the speaker level
    # (never LOW VOLUME alone - Part 3).
    if hf is not None and hf >= HF_BREATH_MIN and contrast is not None and contrast >= CONTENT_CONTRAST_MAX:
        if dur >= DUR_WAITING_MIN:
            return _result("waiting_pause", 0.80, dur, ev,
                           "mid-thought gap {:.0f}ms below speech by {:.0f}dB with broadband "
                           "content - waiting".format(dur * 1000, contrast))
        # A. BREATH / NATURAL SPEECH PAUSE (micro band included; the same
        # evidence bar applies at every duration - Part 5).
        return _result("breath_pause", 0.82, dur, ev,
                       "broadband speech-adjacent gap {:.0f}ms (HF {:.2f}) below speech by "
                       "{:.0f}dB - breath/aspiration".format(dur * 1000, hf, contrast))

    # C. NORMAL WORD GAP — no aspiration evidence -> phonetic boundary,
    # plosive transition, or detector artifact (Part 5). Protected.
    if hf is not None and hf < HF_PHONETIC_MAX:
        return _result("normal_word_gap", 0.90, dur, ev,
                       "gap lacks high-frequency aspiration - phonetic boundary preserved")
    if hf is not None:
        return _result("normal_word_gap", 0.75, dur, ev,
                       "broadband content but not clearly below speech - preserved")
    return _result("nonspeech_unknown", 0.60, dur, ev,
                   "insufficient evidence to classify as removable")


def _result(category, confidence, dur, ev, reason):
    return {
        "category": category,
        "confidence": round(confidence, 2),
        "durationSec": round(dur, 4),
        "evidence": {k: v for k, v in ev.items() if v is not None},
        "reason": reason,
    }


# ── Candidate-window context ─────────────────────────────────────────────────

def build_context(source_path, clip_start, clip_end, words):
    """Decode the candidate window ONCE and compute the reference levels
    every gap measurement needs: the speaker level (p75 of word-interior
    RMS) and the clip silence floor (p5 of 30ms-window energies)."""
    if np is None:
        log("numpy unavailable - cannot measure breath evidence")
        return None
    x = decode_range(source_path, clip_start, clip_end)
    if x is None or x.size < SAMPLE_RATE * 0.5:
        log("no decodable audio in the candidate window")
        return None

    rel = lambda t: int(round((t - clip_start) * SAMPLE_RATE))  # noqa: E731

    # Speaker reference: the MEDIAN word-interior level. The median is
    # robust to the loudest words (shouted emphasis, onomatopoeia) which
    # would otherwise inflate the reference and make real pauses look like
    # speech-level content. The candidate is then judged gap-by-gap
    # relative to the speaker's *typical* level.
    word_db = []
    for wd in words:
        if wd["end"] - wd["start"] < 0.05:
            continue
        v = _word_interior_db(x, rel, wd)
        if v is not None and v > -90.0:
            word_db.append(v)
    speech_ref_db = float(np.median(word_db)) if word_db else None

    # Silence floor: p5 of 30ms windows across the whole candidate window.
    win = int(0.030 * SAMPLE_RATE)
    hop = max(1, win // 2)
    env = np.array([
        v for v in (rms_db(x[i:i + win])
                    for i in range(0, max(1, x.size - win + 1), hop))
        if v is not None
    ])
    floor_db = float(np.percentile(env, 5)) if env.size else None

    return {"x": x, "rel": rel, "speech_ref_db": speech_ref_db,
            "floor_db": floor_db, "sample_rate": SAMPLE_RATE}


# ── Entry point ──────────────────────────────────────────────────────────────

def detect_gaps(source_path, clip_start, clip_end, words, min_gap=0.05):
    """Detect and classify internal speech pauses in [clip_start, clip_end].

    Returns a dict with `gaps` (one per word gap) and reference levels. On
    any analysis failure, `status` is "error" and gaps is empty so callers
    must NOT treat absence of evidence as evidence of absence."""
    if np is None:
        return {"status": "error", "reason": "numpy unavailable", "gaps": []}
    if clip_end <= clip_start:
        return {"status": "error", "reason": "empty candidate range", "gaps": []}

    cw = [w for w in (words or [])
          if w.get("end") is not None and w.get("start") is not None
          and float(w["end"]) > float(w["start"])]
    cw.sort(key=lambda w: float(w["start"]))
    if len(cw) < 2:
        return {"status": "error", "reason": "fewer than two words", "gaps": []}

    ctx = build_context(source_path, clip_start, clip_end, cw)
    if ctx is None:
        return {"status": "error", "reason": "audio analysis unavailable", "gaps": []}

    gaps = []
    for i in range(len(cw) - 1):
        prev_word = cw[i]
        next_word = cw[i + 1]
        gap_start = float(prev_word["end"])
        gap_end = float(next_word["start"])
        if gap_end - gap_start < min_gap:
            continue
        res = analyze_gap(ctx, gap_start, gap_end, prev_word, next_word)
        res["idx"] = i
        res["srcStartSec"] = round(gap_start, 3)
        res["srcEndSec"] = round(gap_end, 3)
        gaps.append(res)

    return {
        "status": "ok",
        "gaps": gaps,
        "speechRefDb": round(ctx["speech_ref_db"], 1) if ctx.get("speech_ref_db") else None,
        "floorDb": round(ctx["floor_db"], 1) if ctx.get("floor_db") else None,
        "sampleRate": SAMPLE_RATE,
    }


def main(argv):
    if len(argv) < 5:
        sys.stderr.write(__doc__)
        return 2
    source_path, start_ms, end_ms, words_json = argv[1:5]
    clip_start = float(start_ms) / 1000.0
    clip_end = float(end_ms) / 1000.0
    try:
        with open(words_json, "r", encoding="utf-8") as fh:
            words = json.load(fh)
        if isinstance(words, dict):
            words = words.get("words", [])
    except (OSError, ValueError) as exc:
        log("words JSON unreadable: {}".format(exc))
        print(json.dumps({"status": "error", "reason": "words JSON unreadable", "gaps": []}))
        return 0

    try:
        result = detect_gaps(source_path, clip_start, clip_end, words)
    except Exception as exc:  # never crash a render for detection
        log("detector exception: {}".format(exc))
        print(json.dumps({"status": "error", "reason": "detector exception", "gaps": []}))
        return 0

    counts = {}
    for g in result.get("gaps", []):
        counts[g["category"]] = counts.get(g["category"], 0) + 1
    if result.get("status") == "ok":
        log("detected {} gap(s): {}".format(
            len(result["gaps"]),
            ", ".join("{}={}".format(k, v) for k, v in sorted(counts.items())) or "none"))
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
