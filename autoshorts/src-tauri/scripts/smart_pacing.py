#!/usr/bin/env python3
"""
AutoShorts 8.0 — Retention-Aware Smart Pacing engine (sidecar).

Sits conceptually AFTER candidate selection and BEFORE final rendering.
The system never asks "Can I cut this?" — it asks
"Can I PROVE this time interval is safe to remove?" When uncertain,
the original content is KEPT.

Edit types (editorial priority order, all conservative v1):
  1. leading_dead_air    — silence before the first word (keep 0.30s)
  2. trailing_dead_air   — silence after the last word (keep 0.45s)
  3. internal_pause      — unusually long pauses compressed to a natural
                           keep (never fully removed, never across a
                           speaker change)
  4. hesitation_filler   — isolated um/uh/uhm/... islands delimited by
                           >=0.45s of verified silence on BOTH sides
  5. false_start         — verbatim-restart abandoned starts (<=5 words,
                           sentence-initial, >=0.4s abandonment pause)

Every removal is acoustically verified with FFmpeg silencedetect
(-35 dB). If silence cannot be verified, the cut is NOT made. If the
silence analysis itself fails, NO cuts are made at all (status=error ->
caller renders the legacy unmodified clip).

CLI contract (mirrors speaker_tracker.py conventions):
    python smart_pacing.py SOURCE START_MS END_MS WORDS_JSON
    - SOURCE      absolute path to the source video
    - START_MS    candidate start in milliseconds (absolute source time)
    - END_MS      candidate end in milliseconds (absolute source time)
    - WORDS_JSON  path to a JSON file: [{"text","start","end","speaker"}, ...]
                  (absolute source times, same file the framing tracker gets)

Output: the LAST stdout line is the JSON pacing plan (camelCase keys,
matching the Rust `SmartPacingPlan` serde struct). All telemetry goes to
stderr. Exit code is 0 whenever the engine ran (even for status=skipped);
a non-zero exit means the engine itself failed.

Plan JSON:
{
  "status": "ok" | "skipped" | "error",
  "reason": "..."                       (optional human-readable summary),
  "clipStartSec": 120.0,                (absolute source secs)
  "clipEndSec": 168.5,
  "outputDurationSec": 66.9,
  "removedTotalSec": 1.6,
  "edits": [
    {"editType": "leading_dead_air", "srcStartSec": ..., "srcEndSec": ...,
     "outStartSec": ..., "confidence": 0.92, "reason": "..."}
  ],
  "retained": [
    {"srcStartSec": ..., "srcEndSec": ..., "outStartSec": ..., "outEndSec": ...}
  ]
}

The `retained` list is the AUTHORITATIVE source->output mapping: it tiles
[clipStartSec, clipEndSec] exactly and [0, outputDurationSec] exactly.
"""

import json
import math
import os
import re
import subprocess
import sys

# ── Conservative thresholds (v1) ────────────────────────────────────────────
# Tuned for "noticeably tighter, not maximally compressed". Every value is a
# deliberate compromise between retention gain and audible-safety margin.

LEAD_MIN_GAP = 0.65          # leading silence must be at least this long
LEAD_KEEP = 0.30              # natural breath kept before the first word
TRAIL_MIN_GAP = 0.85          # trailing silence must be at least this long
TRAIL_KEEP = 0.45             # natural decay kept after the last word

PAUSE_MIN_SENT = 1.25         # sentence-boundary pause: compress only if >= this
PAUSE_KEEP_SENT = 0.50        # natural pause kept at sentence boundaries
PAUSE_MIN_MID = 1.75          # mid-sentence pause: compress only if >= this
PAUSE_KEEP_MID = 0.60         # natural pause kept mid-sentence
PAUSE_MIN_HES = 0.90          # gap merged with a removed hesitation: >= this
PAUSE_KEEP_HES = 0.45         # keep when the gap already contains a removal
PAUSE_MIN_REMOVAL = 0.40      # never compress a pause by less than this

FALSE_START_MAX_WORDS = 5     # abandoned run length cap (words)
FALSE_START_PAUSE = 0.40      # abandonment pause between run and restart
FALSE_START_GUARD = 0.05      # guard bitten out of the surrounding pauses
FALSE_START_PRE_MIN = 0.15    # minimum pause before the run (breath)

FILLER_WORDS = {              # pure hesitation tokens ONLY — never "like",
    "um", "uh", "uhm", "umm", "erm", "hmm", "mm", "mmm",  # "you know",
    "ah", "er", "eh",         # "i mean", "so", "well": those carry meaning.
}
FILLER_MAX_RUN = 2            # at most this many consecutive fillers
FILLER_EDGE_PAUSE = 0.45      # verified silence required on BOTH sides
FILLER_GUARD = 0.05           # guard bitten out of each surrounding pause

WORD_CLEARANCE = 0.06         # min distance cut boundary -> word edge
SNAP_CLEARANCE = 0.03         # min clearance AFTER frame snapping
SILENCE_TOL = 0.04            # tolerance when matching silence coverage
EDGE_TOL = 0.02               # tolerance for filler/false-start edge windows
EDGE_WINDOW = 0.10            # edge window length verified around word cuts

MIN_RETAINED_PIECE = 0.60     # retained pieces shorter than this cancel the
                               # smaller adjacent removal
MIN_EDIT_REMOVAL = 0.30       # every individual edit must remove >= this
MAX_REMOVED_FRACTION = 0.40   # circuit breaker: >40% removed -> bail entirely
MIN_OUTPUT_SEC = 3.0          # never pace a clip below this output length
MIN_INTERSECTION = 0.10       # sub-frame intersections are dropped

NOISE_DB = -35                # silencedetect threshold
SILENCE_MIN_D = 0.10          # silencedetect minimum duration

SENTENCE_TERMINATORS = (".", "!", "?", ";", ":", "\u0964", "\u0965")  # . ! ? ; : । ॥

EPS = 1e-6

# ── Smart Pacing 2.0: speech-adjacent breath / waiting pauses ────────────────
# v2 adds ONE detection stage on the fixed candidate timeline: it measures the
# interior of every word gap and classifies it (breath, waiting, normal,
# sentence, dramatic, speaker transition, non-speech). Only the two
# REDUCIBLE categories reach the plan, and every one of them is then merged
# into the SAME edit list and run through the SAME merge / snap / safety /
# validation pipeline as every v1 edit. There is no second pacing engine.
#
# REDUCE, never DELETE (Part 6): each reducible gap is compressed toward a
# natural residual length instead of being removed outright. The residual is
# what makes the output sound human; the removed middle is the waiting time.

BREATH_KEEP_MICRO = 0.03      # 0.06-0.15s aspiration: nearly eliminated, but
                              # a sliver of the breath survives so the
                              # surrounding phonemes are not glued together
BREATH_KEEP_SHORT = 0.06      # 0.15-0.30s: keep a short natural breath
BREATH_KEEP_MID = 0.10        # 0.30-0.90s: keep a comfortable breath
WAITING_KEEP = 0.15           # >=0.50s waiting pause: keep a deliberate beat
BREATH_MIN_REMOVAL = 0.02     # never shave a gap by less than this (an edit
                              # that only moves 20ms is inaudible and just
                              # risks word-edge collisions)
WAITING_MIN_REMOVAL = 0.10    # a waiting pause must actually shrink to count
BREATH_MIN_DUR = 0.06         # below this a gap is a phonetic boundary
WAITING_MIN_DUR = 0.50        # below this a reducible gap is a breath, not a
                              # noticeable wait

# Duration bands mirrored from breath_detect.py, used to pick the natural
# residual length for a reducible gap (see _breath_keep_for).
DUR_SHORT_BAND = 0.15         # 0.15-0.30s: short breath
DUR_BREATH_BAND = 0.30        # 0.30-0.50s: comfortable breath


def log(msg):
    sys.stderr.write("[Smart Pacing] {}\n".format(msg))
    sys.stderr.flush()


def _log_notes(notes):
    """Dump the decision log. Auditable across BOTH outcomes (Part 9): a
    skipped plan needs its reasons exactly as much as an applied one."""
    if notes:
        log("notes: " + "; ".join(notes[:12]))


# ── Word helpers ────────────────────────────────────────────────────────────

def normalize_word_text(text):
    return re.sub(r"[^\w\u0900-\u097F']", "", (text or "").strip().lower())


def ends_sentence(text):
    t = (text or "").strip()
    return t.endswith(SENTENCE_TERMINATORS)


def is_filler(text):
    return normalize_word_text(text) in FILLER_WORDS


def same_speaker(a, b):
    return (a.get("speaker") or None) == (b.get("speaker") or None)


def clip_words(words, clip_start, clip_end):
    """Filter/sort the full-transcript words to the candidate range.

    Words straddling the clip boundary are clamped to the bounds so the
    leading/trailing gap math stays sound (a candidate that snaps mid-word
    must not produce a negative or bogus gap).
    """
    out = []
    for w in words or []:
        try:
            s = float(w["start"])
            e = float(w["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if e <= s:
            continue
        if e <= clip_start + EPS or s >= clip_end - EPS:
            continue
        out.append({
            "text": str(w.get("text", "")),
            "start": max(s, clip_start),
            "end": min(e, clip_end),
            "speaker": w.get("speaker"),
        })
    out.sort(key=lambda w: w["start"])
    # drop exact duplicates (diarization artifacts)
    deduped = []
    for w in out:
        if deduped and abs(w["start"] - deduped[-1]["start"]) < 1e-3 \
                and abs(w["end"] - deduped[-1]["end"]) < 1e-3:
            continue
        deduped.append(w)
    return deduped


# ── Silence helpers ──────────────────────────────────────────────────────────

def normalize_silences(silences, clip_start, clip_end):
    """Merge/clip raw (start, end) silence intervals to the candidate range."""
    dur = clip_end - clip_start
    clipped = []
    for s, e in silences or []:
        s = max(float(s), 0.0)
        e = min(float(e), dur)
        if e - s <= EPS:
            continue
        clipped.append((s, e))
    clipped.sort()
    merged = []
    for s, e in clipped:
        if merged and s <= merged[-1][1] + EPS:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def covered(silences, a, b, tol=SILENCE_TOL):
    """True if some verified silence interval covers [a, b] within tol."""
    if b - a <= EPS:
        return True
    for s, e in silences:
        if s <= a + tol and e >= b - tol:
            return True
    return False


def largest_covered_subinterval(silences, a, b, tol=SILENCE_TOL):
    """The widest sub-interval of [a, b] that a verified silence covers.

    A breath gap often starts inside the previous word's decay tail: the
    breath detector measures the gap INTERIOR (it already excludes the
    tail), but the v1 silencedetect pass only marks the part that has
    actually fallen below threshold as silent. Rather than demanding that
    the whole intended cut be silent, we reduce only the portion both
    systems agree on. Every removed second stays independently verified.
    """
    best = None
    for s, e in silences:
        lo = max(s, a)
        hi = min(e, b)
        if hi - lo <= EPS:
            continue
        # Require the silence to actually span the sub-interval, not merely
        # touch it, so the cut sits inside verified silence on both sides.
        if s <= lo + tol and e >= hi - tol and (best is None or hi - lo > best[1] - best[0]):
            best = (lo, hi)
    return best


def edge_windows_verified(silences, cut_start, cut_end):
    """For cuts that remove speech (fillers / false starts): the pauses on
    both sides of the cut must be acoustically silent in the 0.10s windows
    adjacent to the cut boundaries."""
    before = covered(silences, cut_start - EDGE_WINDOW, cut_start, EDGE_TOL)
    after = covered(silences, cut_end, cut_end + EDGE_WINDOW, EDGE_TOL)
    return before and after


# ── Frame snapping ──────────────────────────────────────────────────────────

def snap(t, fps):
    if not fps or fps <= 0:
        return t
    return round(t * fps) / fps


# ── Edit assembly ───────────────────────────────────────────────────────────

def make_edit(edit_type, start, end, confidence, reason):
    return {
        "editType": edit_type,
        "srcStartSec": start,
        "srcEndSec": end,
        "confidence": confidence,
        "reason": reason,
    }


def propose_edits(cw, clip_start, clip_end, silences, source_path=None, breath_gaps=None):
    """Propose the raw (unsnaped, unmerged) edit list. Returns (edits, notes).

    Smart Pacing 2.0: when `breath_gaps` is supplied (the result of the
    breath-detection stage, see propose_breath_edits), the reducible gaps are
    folded into this same edit list as `breath_pause` / `waiting_pause`
    edits, then handled by exactly the same downstream pipeline.
    """
    edits = []
    notes = []
    first, last = cw[0], cw[-1]

    # 1. Leading dead air -----------------------------------------------------
    lead = first["start"] - clip_start
    if lead >= LEAD_MIN_GAP:
        cut_end = first["start"] - LEAD_KEEP
        if cut_end - clip_start >= MIN_EDIT_REMOVAL and \
                covered(silences, clip_start, cut_end):
            edits.append(make_edit(
                "leading_dead_air", clip_start, cut_end, 0.92,
                "leading silence {:.2f}s verified silent (kept {:.2f}s breath)".format(lead, LEAD_KEEP)))
        else:
            notes.append("leading gap {:.2f}s not removable (threshold/verification)".format(lead))

    # 2. Trailing dead air ----------------------------------------------------
    trail = clip_end - last["end"]
    if trail >= TRAIL_MIN_GAP:
        cut_start = last["end"] + TRAIL_KEEP
        if clip_end - cut_start >= MIN_EDIT_REMOVAL and \
                covered(silences, cut_start, clip_end):
            edits.append(make_edit(
                "trailing_dead_air", cut_start, clip_end, 0.92,
                "trailing silence {:.2f}s verified silent (kept {:.2f}s decay)".format(trail, TRAIL_KEEP)))
        else:
            notes.append("trailing gap {:.2f}s not removable (threshold/verification)".format(trail))

    # 3. False starts (verbatim restarts) — word-level, decided BEFORE pauses
    removed_word_idx = set()

    i = 0
    n = len(cw)
    while i < n:
        run_len = min(FALSE_START_MAX_WORDS, (n - i) // 2)
        matched = 0
        while run_len >= 1:
            run = cw[i:i + run_len]
            restart = cw[i + run_len:i + 2 * run_len]
            if len(restart) == run_len and \
                    all(normalize_word_text(a["text"]) == normalize_word_text(b["text"])
                        for a, b in zip(run, restart)):
                pause = restart[0]["start"] - run[-1]["end"]
                pre_pause = (run[0]["start"] - cw[i - 1]["end"]) if i > 0 else run[0]["start"] - clip_start
                sentence_initial = (i == 0) or ends_sentence(cw[i - 1]["text"])
                speakers = [w.get("speaker") for w in run] + [w.get("speaker") for w in restart]
                known_speaker = len(set(speakers)) == 1 and (
                    len({w.get("speaker") for w in cw}) <= 1 or speakers[0] is not None)
                if pause >= FALSE_START_PAUSE and pre_pause >= FALSE_START_PRE_MIN \
                        and sentence_initial and known_speaker \
                        and not any(k in removed_word_idx for k in range(i, i + 2 * run_len)):
                    matched = run_len
                    break
            run_len -= 1
        if matched:
            run = cw[i:i + matched]
            restart = cw[i + matched:i + 2 * matched]
            pre_pause = (run[0]["start"] - cw[i - 1]["end"]) if i > 0 else run[0]["start"] - clip_start
            post_pause = restart[0]["start"] - run[-1]["end"]
            cut_start = run[0]["start"] - min(FALSE_START_GUARD, pre_pause / 2.0)
            cut_end = run[-1]["end"] + min(FALSE_START_GUARD, post_pause / 2.0)
            if edge_windows_verified(silences, cut_start, cut_end):
                edits.append(make_edit(
                    "false_start", cut_start, cut_end, 0.85,
                    "verbatim restart of {} word(s) abandoned after {:.2f}s pause".format(matched, post_pause)))
                for k in range(i, i + matched):
                    removed_word_idx.add(k)
                # skip past the removed run; the restart becomes the new anchor
                i += matched
                continue
        i += 1

    # 4. Filler islands — word-level ------------------------------------------
    i = 0
    while i < n:
        if is_filler(cw[i]["text"]) and i > 0 and i < n - 1 and i not in removed_word_idx:
            j = i
            while j + 1 < n and j - i + 1 < FILLER_MAX_RUN and \
                    is_filler(cw[j + 1]["text"]) and (j + 1) not in removed_word_idx:
                j += 1
            prev, nxt = cw[i - 1], cw[j + 1]
            pre = cw[i]["start"] - prev["end"]
            post = nxt["start"] - cw[j]["end"]
            interior = (i - 1 not in removed_word_idx) and (j + 1 not in removed_word_idx)
            if interior and pre >= FILLER_EDGE_PAUSE and post >= FILLER_EDGE_PAUSE \
                    and same_speaker(prev, cw[i]) and same_speaker(cw[j], nxt):
                cut_start = cw[i]["start"] - FILLER_GUARD
                cut_end = cw[j]["end"] + FILLER_GUARD
                if edge_windows_verified(silences, cut_start, cut_end):
                    edits.append(make_edit(
                        "hesitation_filler", cut_start, cut_end, 0.75,
                        "filler '{}' delimited by {:.2f}s/{:.2f}s verified silence".format(
                            " ".join(w["text"] for w in cw[i:j + 1]), pre, post)))
                    for k in range(i, j + 1):
                        removed_word_idx.add(k)
                    i = j + 1
                    continue
        i += 1

    # 5. Internal pause compression — over RETAINED words, with hesitation-
    #    merged effective gaps (a gap that already contains a removed filler /
    #    false start uses the tighter hesitation thresholds).
    for a_idx in range(n - 1):
        b_idx = a_idx + 1
        if a_idx in removed_word_idx or b_idx in removed_word_idx:
            continue
        a, b = cw[a_idx], cw[b_idx]
        gap = b["start"] - a["end"]
        if gap <= EPS:
            continue
        if not same_speaker(a, b):
            continue  # NEVER cut across a speaker change
        gap_lo, gap_hi = a["end"], b["start"]
        spans_removed = any(
            e["srcEndSec"] > gap_lo + EPS and e["srcStartSec"] < gap_hi - EPS
            for e in edits)
        if spans_removed:
            min_gap, keep = PAUSE_MIN_HES, PAUSE_KEEP_HES
        elif ends_sentence(a["text"]):
            min_gap, keep = PAUSE_MIN_SENT, PAUSE_KEEP_SENT
        else:
            min_gap, keep = PAUSE_MIN_MID, PAUSE_KEEP_MID
        if gap < min_gap:
            continue
        cut_start = a["end"] + keep / 2.0
        cut_end = b["start"] - keep / 2.0
        if cut_end - cut_start < PAUSE_MIN_REMOVAL:
            continue
        if not covered(silences, cut_start, cut_end):
            continue
        kind = "hesitation-merged" if spans_removed else (
            "sentence" if ends_sentence(a["text"]) else "mid-sentence")
        edits.append(make_edit(
            "internal_pause", cut_start, cut_end, 0.90,
            "{} pause {:.2f}s compressed to {:.2f}s (verified silent)".format(kind, gap, keep)))

    # 6. Smart Pacing 2.0 — speech-adjacent breath / waiting pauses. The
    #    detector already classified every gap; the two REDUCIBLE categories
    #    become edits here, on the fixed candidate timeline. Every other
    #    category (normal word gap, sentence, dramatic, speaker transition,
    #    non-speech) is deliberately absent from this list, which is how
    #    those categories end up protected.
    if breath_gaps:
        edits.extend(propose_breath_edits(cw, clip_start, clip_end, silences, breath_gaps, notes))

    return edits, notes


# ── Smart Pacing 2.0: breath / waiting pause reduction ──────────────────────

def _breath_keep_for(dur):
    """Natural residual length for a reducible gap of `dur` seconds.

    REDUCE, never DELETE (Part 6): the gap is compressed toward a natural
    breath length instead of being removed. Longer reducible gaps keep a
    longer residual because a listener expects a longer beat there.
    """
    if dur >= WAITING_MIN_DUR:
        return WAITING_KEEP
    if dur >= DUR_BREATH_BAND:
        return BREATH_KEEP_MID
    if dur >= DUR_SHORT_BAND:
        return BREATH_KEEP_SHORT
    return BREATH_KEEP_MICRO


def propose_breath_edits(cw, clip_start, clip_end, silences, gaps, notes):
    """Turn the detector's reducible gaps into reduction edits.

    The detector returns gaps in ABSOLUTE source time, already classified.
    The categories and their handling:

        breath_pause    -> REDUCE to a short natural breath
        waiting_pause   -> REDUCE to a deliberate beat
        everything else -> no edit here (protected by construction: the
                           edit list simply never contains them)

    Every proposal is re-anchored to the retained words of THIS clip, re-
    checked for word clearance, re-checked for a speaker change, and
    re-verified acoustically. The detector's judgment is evidence, not
    authority: the same guards that gate every v1 edit gate these.
    """
    edits = []
    if not gaps:
        return edits

    by_span = {}
    for w in cw:
        by_span[(round(w["start"], 3), round(w["end"], 3))] = w

    for g in gaps:
        category = g.get("category")
        if category not in ("breath_pause", "waiting_pause"):
            if g.get("removable") is True and category not in ("sentence_pause", "speaker_transition", "normal_word_gap", "nonspeech_unknown"):
                pred_c = g.get("predictedClass")
                category = pred_c if pred_c in ("breath_pause", "waiting_pause") else "breath_pause"
            else:
                continue
        try:
            gs = float(g["srcStartSec"])
            ge = float(g["srcEndSec"])
        except (KeyError, TypeError, ValueError):
            continue
        if ge - gs < BREATH_MIN_DUR - 1e-9:
            continue
        # Anchor to the retained words inside this clip: the detector may see
        # words outside the candidate range; the plan only ever cuts inside
        # it, and the boundary words were clamped by clip_words().
        prev_w = _word_ending_before(cw, gs)
        next_w = _word_starting_after(cw, ge)
        if prev_w is None or next_w is None:
            notes.append("breath gap {:.3f}-{:.3f}s outside retained words - kept".format(gs, ge))
            continue
        if not same_speaker(prev_w, next_w):
            notes.append("breath gap {:.3f}s crosses a speaker change - kept".format(ge - gs))
            continue
        gap_lo, gap_hi = prev_w["end"], next_w["start"]
        if gap_hi - gap_lo < BREATH_MIN_DUR - 1e-9:
            continue

        keep = _breath_keep_for(gap_hi - gap_lo)
        cut_start = gap_lo + keep / 2.0
        cut_end = gap_hi - keep / 2.0
        if category == "waiting_pause":
            if cut_end - cut_start < WAITING_MIN_REMOVAL:
                notes.append("waiting pause {:.3f}s shrinks by <{:.2f}s - kept".format(
                    gap_hi - gap_lo, WAITING_MIN_REMOVAL))
                continue
        else:
            if cut_end - cut_start < BREATH_MIN_REMOVAL:
                notes.append("breath gap {:.3f}s shrinks by <{:.2f}s - kept".format(
                    gap_hi - gap_lo, BREATH_MIN_REMOVAL))
                continue
        # No clearance margin: the cut is strictly interior to the word gap,
        # and word_clearance_ok() only rejects cuts that touch or contain a
        # word edge (Part 7). The residual breath is the listener-facing
        # guarantee that no phoneme is clipped.
        #
        # Acoustic re-verification: the v1 silencedetect pass only marks what
        # has actually fallen below threshold, so the agreed portion can be
        # narrower than the intended cut (a gap often starts inside the
        # previous word's decay tail). Shrink to the portion BOTH systems
        # agree on instead of dropping the gap: every removed second stays
        # independently verified, and shrinking only ever removes less.
        agreed = largest_covered_subinterval(silences, cut_start, cut_end)
        if agreed is None:
            notes.append("breath gap {:.3f}s not verified silent - kept".format(gap_hi - gap_lo))
            continue
        cut_start, cut_end = agreed
        min_removal = WAITING_MIN_REMOVAL if category == "waiting_pause" else BREATH_MIN_REMOVAL
        if cut_end - cut_start < min_removal - 1e-9:
            notes.append("{} {:.3f}s verified portion {:.2f}s < {:.2f}s - kept".format(
                "waiting pause" if category == "waiting_pause" else "breath gap",
                gap_hi - gap_lo, cut_end - cut_start, min_removal))
            continue
        edits.append(make_edit(
            category, cut_start, cut_end, float(g.get("confidence", 0.80)),
            "{} {:.3f}s reduced to {:.2f}s ({})".format(
                "waiting pause" if category == "waiting_pause" else "breath pause",
                gap_hi - gap_lo, keep, g.get("reason", "breath evidence"))))
    return edits


def _word_ending_before(cw, t):
    out = None
    for w in cw:
        if w["end"] <= t + EPS:
            out = w
        else:
            break
    return out


def _word_starting_after(cw, t):
    for w in cw:
        if w["start"] >= t - EPS:
            return w
    return None


# ── Merge / snap / safety pipeline ──────────────────────────────────────────

def merge_edits(edits):
    if not edits:
        return []
    edits = sorted(edits, key=lambda e: e["srcStartSec"])
    merged = [edits[0]]
    for e in edits[1:]:
        last = merged[-1]
        if e["srcStartSec"] <= last["srcEndSec"] + EPS:
            if e["srcEndSec"] > last["srcEndSec"]:
                last["srcEndSec"] = e["srcEndSec"]
            last["confidence"] = min(last["confidence"], e["confidence"])
            last["reason"] = "{} + {}".format(last["reason"], e["reason"])
        else:
            merged.append(e)
    return merged


def snap_edits(edits, fps, clip_start):
    """Snap cut boundaries to the OUTPUT frame grid. The output frame grid is
    CLIP-RELATIVE (ffmpeg `-ss` makes filtergraph t=0 at the candidate start),
    so snapping must round the offset from clip_start, not the absolute time."""
    for e in edits:
        e["srcStartSec"] = clip_start + snap(e["srcStartSec"] - clip_start, fps)
        e["srcEndSec"] = clip_start + snap(e["srcEndSec"] - clip_start, fps)
    return edits


def word_clearance_ok(cuts, cw, clearance):
    """Every cut boundary must keep `clearance` from every retained word edge."""
    for cs, ce in cuts:
        for w in cw:
            if cs - EPS <= w["start"] <= ce + EPS or cs - EPS <= w["end"] <= ce + EPS:
                # boundary touches this word's edge -> distance check
                if min(abs(cs - w["start"]), abs(cs - w["end"]),
                       abs(ce - w["start"]), abs(ce - w["end"])) < clearance - 1e-9:
                    if not (cs <= w["start"] + 1e-9 and ce >= w["end"] - 1e-9):
                        return False
            if cs < w["start"] and ce > w["end"]:
                continue  # word fully inside the cut (intended removal)
            if cs < w["end"] - 1e-9 and ce > w["start"] + 1e-9:
                # partial overlap with a word that is not fully inside
                return False
    return True


def build_retained(cuts, clip_start, clip_end):
    retained = []
    cursor = clip_start
    for cs, ce in cuts:
        if cs - cursor > EPS:
            retained.append((cursor, cs))
        cursor = max(cursor, ce)
    if clip_end - cursor > EPS:
        retained.append((cursor, clip_end))
    return retained


def plan_pacing(words, clip_start, clip_end, silences, fps=None, v2=False,
                source_path=None):
    """Public entry (signature stable for tests): runs the deterministic core,
    then emits opt-in per-gap telemetry (Phase 3 data pipeline). Telemetry
    never alters the plan and never raises."""
    plan = _plan_pacing_core(words, clip_start, clip_end, silences, fps=fps,
                             v2=v2, source_path=source_path)
    try:
        _emit_pacing_telemetry(source_path, clip_start, clip_end, words,
                               globals().get("_LAST_DETECTED_GAPS", []), plan)
    except Exception:
        pass
    return plan


def _plan_pacing_core(words, clip_start, clip_end, silences, fps=None, v2=False,
                      source_path=None):
    """Pure decision engine. `silences` are CLIP-RELATIVE (0 = clip start)
    verified-silent intervals. Returns the plan dict.

    Smart Pacing 2.0 (`v2=True`): adds ONE detection stage on the fixed
    candidate timeline — breath_detect measures every word gap and classifies
    it. The two reducible categories are folded into the same edit list and
    the same merge / snap / safety / validation pipeline as every v1 edit.
    Candidate boundaries are never moved; clip selection is untouched.
    """
    clip_dur = clip_end - clip_start
    notes = []
    if clip_dur < MIN_OUTPUT_SEC + 1.0:
        _log_notes(notes)
        return {"status": "skipped",
                "reason": "clip too short for smart pacing ({:.2f}s)".format(clip_dur)}

    cw = clip_words(words, clip_start, clip_end)
    if not cw:
        _log_notes(notes)
        return {"status": "skipped", "reason": "no transcript words in clip range"}

    silences = normalize_silences(silences, clip_start, clip_end)
    # Silences arrive CLIP-RELATIVE (run_silencedetect seeks with -ss, so its
    # timestamps start at 0 = candidate start). Everything downstream - words,
    # edit boundaries, covered() queries - is in ABSOLUTE source time, so shift
    # the verified silences onto the absolute timeline exactly once, here.
    silences = [(s + clip_start, e + clip_start) for s, e in silences]

    breath_gaps = None
    if v2:
        breath_gaps = run_breath_detection(source_path, clip_start, clip_end, words)

    edits, notes = propose_edits(cw, clip_start, clip_end, silences,
                                 breath_gaps=breath_gaps)
    edits = merge_edits(edits)
    edits = snap_edits(edits, fps, clip_start)

    # Safety filter 1: word clearance + no partial word overlap.
    kept = []
    for e in edits:
        removal = e["srcEndSec"] - e["srcStartSec"]
        # The minimum removal is editType-aware: breath reductions are
        # intentionally small (Part 6 REDUCE, not DELETE), so they get their
        # own floor. Every v1 type keeps the v1 floor, which keeps the v1
        # output byte-identical when v2 is off.
        if e["editType"] == "breath_pause":
            min_removal = BREATH_MIN_REMOVAL
        elif e["editType"] == "waiting_pause":
            min_removal = WAITING_MIN_REMOVAL
        else:
            min_removal = MIN_EDIT_REMOVAL
        if removal < min_removal - 1e-9:
            notes.append("dropped {} removal {:.2f}s below minimum".format(e["editType"], removal))
            continue
        trial = [(x["srcStartSec"], x["srcEndSec"]) for x in kept] + \
            [(e["srcStartSec"], e["srcEndSec"])]
        # Breath/waiting cuts are anchored to word EDGES by construction
        # (cut = word_end + keep/2, snapped to the frame grid), so a zero
        # distance from a word edge is the intended placement, not a hazard:
        # no phoneme is clipped as long as the cut stays strictly inside the
        # word gap, which word_clearance_ok's partial-overlap clause still
        # enforces. The SNAP_CLEARANCE margin guards the v1 edits, whose
        # dead-air boundaries are placed without any word anchoring.
        clearance = 0.0 if e["editType"] in ("breath_pause", "waiting_pause") \
            else SNAP_CLEARANCE
        if word_clearance_ok(trial, cw, clearance):
            kept.append(e)
        else:
            notes.append("dropped {} (word clearance)".format(e["editType"]))
    edits = kept

    # Safety filter 2: sliver pieces cancel the smaller adjacent removal.
    # A "sliver" is a retained piece whose retention value is nil — a
    # sub-MIN_RETAINED_PIECE span carrying NO speech. Such dead air is better
    # absorbed (at a clip edge, when the adjacent removal can cover it) or
    # defended by cancelling the smaller adjacent removal. A span that
    # contains words is NOT dead air: the floor exists to protect speech, and
    # cancelling a reduction to save a speech piece would destroy a
    # legitimate edit while the piece keeps its words either way.
    while True:
        cuts = [(e["srcStartSec"], e["srcEndSec"]) for e in edits]
        retained = build_retained(cuts, clip_start, clip_end)
        sliver = next(((s, e) for s, e in retained
                       if e - s < MIN_RETAINED_PIECE
                       and not any(x["start"] < e - EPS and x["end"] > s + EPS
                                    for x in cw)), None)
        if not sliver:
            break
        s, e = sliver
        at_edge = (s <= clip_start + EPS) or (e >= clip_end - EPS)
        if at_edge:
            absorbed = False
            for ed in edits:
                if abs(ed["srcStartSec"] - e) < 1e-6 and covered(silences, s, ed["srcStartSec"]):
                    ed["srcStartSec"] = s
                    absorbed = True
                    notes.append("absorbed {:.2f}s edge sliver into {}".format(
                        e - s, ed["editType"]))
                    break
                if abs(ed["srcEndSec"] - s) < 1e-6 and covered(silences, ed["srcEndSec"], e):
                    ed["srcEndSec"] = e
                    absorbed = True
                    notes.append("absorbed {:.2f}s edge sliver into {}".format(
                        e - s, ed["editType"]))
                    break
            if absorbed:
                continue
        adjacent = []
        for idx, ed in enumerate(edits):
            if abs(ed["srcEndSec"] - s) < 1e-6:
                adjacent.append((idx, ed["srcEndSec"] - ed["srcStartSec"]))
            if abs(ed["srcStartSec"] - e) < 1e-6:
                adjacent.append((idx, ed["srcStartSec"] - ed["srcEndSec"]))
        if not adjacent:
            break
        idx = min(adjacent, key=lambda t: abs(t[1]))[0]
        notes.append("cancelled {} (would create {:.2f}s sliver)".format(
            edits[idx]["editType"], e - s))
        edits.pop(idx)

    # Safety filter 3: circuit breakers — bail ENTIRELY rather than produce
    # something extreme.
    removed_total = sum(e["srcEndSec"] - e["srcStartSec"] for e in edits)
    if removed_total > MAX_REMOVED_FRACTION * clip_dur + 1e-9:
        _log_notes(notes)
        return {"status": "skipped",
                "reason": "circuit breaker: would remove {:.1f}% of the clip".format(
                    100.0 * removed_total / clip_dur)}
    out_dur = clip_dur - removed_total
    if edits and out_dur < MIN_OUTPUT_SEC:
        _log_notes(notes)
        return {"status": "skipped",
                "reason": "circuit breaker: output would be {:.2f}s < {:.2f}s".format(out_dur, MIN_OUTPUT_SEC)}

    # Safety filter 4: every in-clip word is either fully removed or fully
    # retained (defensive; clearance already guarantees this).
    cuts = [(e["srcStartSec"], e["srcEndSec"]) for e in edits]
    bad = []
    for w in cw:
        inside = any(cs <= w["start"] + 0.02 and ce >= w["end"] - 0.02 for cs, ce in cuts)
        outside = all(ce <= w["start"] + 0.02 or cs >= w["end"] - 0.02 for cs, ce in cuts)
        if not (inside or outside):
            bad.append(w)
    if bad:
        _log_notes(notes)
        return {"status": "skipped",
                "reason": "defensive reject: {} word(s) straddle cut boundaries".format(len(bad))}

    if not edits:
        _log_notes(notes)
        return {"status": "skipped",
                "reason": "no removable silence found - original kept"}

    # Assemble the authoritative mapping.
    retained = build_retained(cuts, clip_start, clip_end)
    out_cursor = 0.0
    retained_out = []
    for s, e in retained:
        retained_out.append({
            "srcStartSec": round(s, 3), "srcEndSec": round(e, 3),
            "outStartSec": round(out_cursor, 3),
            "outEndSec": round(out_cursor + (e - s), 3),
        })
        out_cursor += e - s

    out_cursor = 0.0
    cursor = clip_start
    for e in edits:
        span_before = sum(r["srcEndSec"] - r["srcStartSec"] for r in retained_out
                          if r["srcEndSec"] <= e["srcStartSec"] + EPS)
        e["outStartSec"] = round(span_before, 3)

    plan = {
        "status": "ok",
        "clipStartSec": round(clip_start, 3),
        "clipEndSec": round(clip_end, 3),
        "outputDurationSec": round(out_dur, 3),
        "removedTotalSec": round(removed_total, 3),
        "edits": [
            {"editType": e["editType"], "srcStartSec": round(e["srcStartSec"], 3),
             "srcEndSec": round(e["srcEndSec"], 3), "outStartSec": e["outStartSec"],
             "confidence": e["confidence"], "reason": e["reason"]}
            for e in edits
        ],
        "retained": retained_out,
    }
    _log_notes(notes)
    return plan


# ── Smart Pacing 2.0: breath detection stage ─────────────────────────────────

# Full detected-gap list from the most recent run_breath_detection call (this
# process handles exactly one pacing request, so a module stash is safe and
# keeps the function's public return contract untouched for tests).
_LAST_DETECTED_GAPS = []

def run_breath_detection(source_path, clip_start, clip_end, words):
    """Run the breath/pause detector over the fixed candidate window.

    Returns the gap list, or None whenever the stage is unavailable or fails
    — in which case the plan simply contains no v2 edits and falls back to
    exactly the v1 edit list. The caller never treats absence of evidence as
    evidence of absence (Part 3).
    """
    if not source_path:
        return None
    try:
        import breath_detect
    except ImportError:
        log("v2: breath_detect unavailable - falling back to v1 edits")
        return None
    try:
        result = breath_detect.detect_gaps(source_path, clip_start, clip_end, words)
    except Exception as exc:  # never break a render for detection
        log("v2: breath detection failed ({}): falling back to v1 edits".format(exc))
        return None
    if not result or result.get("status") != "ok":
        log("v2: breath detection {} - falling back to v1 edits".format(
            (result or {}).get("status", "unavailable")))
        return None
    gaps = result.get("gaps", [])
    global _LAST_DETECTED_GAPS
    _LAST_DETECTED_GAPS = gaps
    _attach_learned_pause_evidence(source_path, clip_start, clip_end, words, gaps)
    reducible = [
        g for g in gaps
        if g.get("category") in ("breath_pause", "waiting_pause")
        or (g.get("removable") is True and g.get("category") not in ("sentence_pause", "speaker_transition", "normal_word_gap", "nonspeech_unknown"))
    ]
    log("v2: {} reducible gap(s) of {} detected".format(
        len(reducible), len(gaps)))
    return reducible


detect_gaps_v2 = run_breath_detection
detect_pacing_cuts = plan_pacing


def _attach_learned_pause_evidence(source_path, clip_start, clip_end, words, gaps):
    """Phase 3: attach learned pause-type evidence to each gap.

    The Atria-contract LightGBM model (pause_intelligence.py) scores each gap
    with pRemovable = P(breath) + P(waiting). Deterministic safety rules
    retain full veto authority. Model missing/flagged off/failed => gaps
    untouched (documented fallback, plan unchanged).
    """
    if not gaps:
        return
    try:
        import pause_intelligence as pi
    except Exception:
        return
    if not pi.pause_learning_enabled():
        return
    model_path = pi.find_pause_model()
    if not model_path:
        return
    try:
        feats = pi.build_gap_features(source_path, gaps, words, clip_start, clip_end)
        if not feats:
            return
        rows = [pi.features_to_row(f) for f in feats]
        scored = pi.score_gaps_with_model(gaps, rows, model_path)
        if scored:
            removable_count = 0
            protected_classes = {
                "sentence_pause",
                "normal_word_gap",
                "nonspeech_unknown",
                "speaker_transition",
            }
            allowed_classes = {"breath_pause", "waiting_pause"}
            for g in gaps:
                p_rem = float(g.get("pRemovable") or 0.0)
                is_above_thr = (p_rem >= 0.65) or bool(g.get("pRemovableAboveThreshold"))
                pred_class = g.get("predictedClass") or g.get("category")
                orig_cat = g.get("category")

                # Removable Category Gate: Only breath_pause and waiting_pause are allowed to be marked removable.
                # Protected classes (sentence_pause, normal_word_gap, nonspeech_unknown, speaker_transition)
                # must NEVER be marked removable by the model.
                if (
                    is_above_thr
                    and pred_class in allowed_classes
                    and pred_class not in protected_classes
                    and orig_cat not in protected_classes
                ):
                    g["removable"] = True
                    if not isinstance(g.get("evidence"), dict):
                        g["evidence"] = {}
                    g["evidence"]["pause_intel"] = True
                    g["evidence"]["pRemovable"] = p_rem
                    removable_count += 1
            model_name = os.path.basename(model_path) if model_path else "unknown"
            sys.stderr.write("[PauseIntel] gaps_scored={} removable={} model={}\n".format(
                len(gaps), removable_count, model_name))
            sys.stderr.flush()
            log("v3-learned: pause model scored {} gap(s) (removable={})".format(len(gaps), removable_count))
    except Exception as exc:  # never break a render for learned evidence
        log("v3-learned: pause scoring failed ({}): gaps unchanged".format(exc))


def _emit_pacing_telemetry(source_path, clip_start, clip_end, words, gaps, plan):
    """Phase 3 data pipeline: append per-gap (features, decision) rows so real
    runs accumulate weak-label training data for future model promotions.
    Opt-in via AUTOSHORTS_PACING_TELEMETRY_DIR; failure never affects a render."""
    out_dir = os.environ.get("AUTOSHORTS_PACING_TELEMETRY_DIR")
    if not out_dir or not gaps:
        return
    try:
        import time
        os.makedirs(out_dir, exist_ok=True)
        import pause_intelligence as pi
        feats = pi.build_gap_features(source_path, gaps, words, clip_start, clip_end) or []
        removed_mids = [
            (float(e["srcStartSec"]) + float(e["srcEndSec"])) / 2.0
            for e in (plan or {}).get("edits", [])
        ]
        rows = []
        for g, f in zip(gaps, feats + [None] * max(0, len(gaps) - len(feats))):
            mid = (float(g.get("srcStartSec", 0.0)) + float(g.get("srcEndSec", 0.0))) / 2.0
            rows.append({
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "sourceHash": pi.source_hash(source_path),
                "category": g.get("category"),
                "detectorConfidence": g.get("confidence"),
                "pRemovable": g.get("pRemovable"),
                "removed": int(any(abs(mid - m) < 1e-6 for m in removed_mids)),
                "features": f,
            })
        name = "pacing_telemetry_{}.jsonl".format(time.strftime("%Y%m%d", time.gmtime()))
        with open(os.path.join(out_dir, name), "a", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        log("telemetry: {} gap record(s) appended".format(len(rows)))
    except Exception as exc:
        log("telemetry: failed ({}); render unaffected".format(exc))


# ── FFmpeg / ffprobe integration ────────────────────────────────────────────

def run_silencedetect(source_path, start_sec, end_sec):
    """Run FFmpeg silencedetect over the FULL candidate range.

    Returns clip-relative silence intervals, or None if the analysis could
    not be performed (in which case the caller must make NO cuts).
    """
    dur = end_sec - start_sec
    cmd = [
        "ffmpeg", "-nostats",
        "-ss", "{:.3f}".format(start_sec),
        "-t", "{:.3f}".format(dur),
        "-i", source_path,
        "-vn", "-af", "silencedetect=noise={}dB:d={:.2f}".format(NOISE_DB, SILENCE_MIN_D),
        "-f", "null", "-",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=max(60, int(dur * 4)), errors="replace")
    except (OSError, subprocess.TimeoutExpired) as exc:
        log("silencedetect failed to run: {}".format(exc))
        return None
    if proc.returncode != 0:
        log("silencedetect exited with {}: {}".format(
            proc.returncode, (proc.stderr or "").strip()[-300:]))
        return None

    starts, ends = [], []
    for line in (proc.stderr or "").splitlines():
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
        intervals.append((open_start, dur))  # silence runs to end of range
    if starts:  # trailing unmatched starts without an end pair
        for s in starts:
            intervals.append((s, dur))
    return intervals


def probe_fps(source_path):
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=avg_frame_rate", "-of", "csv=p=0", source_path],
            capture_output=True, text=True, timeout=30, errors="replace")
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    raw = (proc.stdout or "").strip()
    m = re.match(r"^(\d+)\s*/\s*(\d+)$", raw)
    if not m:
        return None
    num, den = int(m.group(1)), int(m.group(2))
    if den <= 0 or num <= 0:
        return None
    fps = num / den
    if not (1.0 <= fps <= 240.0):
        return None
    return fps


# ── CLI ─────────────────────────────────────────────────────────────────────

def main(argv):
    if len(argv) < 5:
        sys.stderr.write(__doc__)
        return 2
    source_path, start_ms, end_ms, words_json = argv[1:5]
    start_sec = float(start_ms) / 1000.0
    end_sec = float(end_ms) / 1000.0
    if end_sec <= start_sec:
        print(json.dumps({"status": "skipped", "reason": "empty candidate range"}))
        return 0

    try:
        with open(words_json, "r", encoding="utf-8") as fh:
            words = json.load(fh)
        if isinstance(words, dict):
            words = words.get("words", [])
    except (OSError, ValueError) as exc:
        log("words JSON unreadable: {}".format(exc))
        print(json.dumps({"status": "error", "reason": "words JSON unreadable"}))
        return 0

    # Smart Pacing 2.0 is ON by default; "0"/"false"/"off" disables it and
    # yields the exact v1 plan. The kill switch is independent of the v1
    # switch so v2 can be disabled without disabling pacing as a whole.
    v2_enabled = os.environ.get("AUTOSHORTS_SMART_PACING_2", "1").strip().lower() not in ("0", "false", "off")

    try:
        silences = run_silencedetect(source_path, start_sec, end_sec)
        if silences is None:
            # Acoustic verification unavailable -> NO cuts (conservative).
            print(json.dumps({"status": "error",
                              "reason": "silence verification unavailable — no cuts made"}))
            return 0
        fps = probe_fps(source_path)
        plan = plan_pacing(words, start_sec, end_sec, silences, fps,
                           v2=v2_enabled, source_path=source_path)
    except Exception as exc:  # engine bug -> conservative, never crash the render
        log("engine exception: {}".format(exc))
        print(json.dumps({"status": "error", "reason": "engine exception"}))
        return 0

    if plan.get("status") == "ok" and plan.get("edits"):
        log("plan: {} edit(s), removed {:.2f}s of {:.2f}s -> output {:.2f}s".format(
            len(plan["edits"]), plan["removedTotalSec"],
            plan["clipEndSec"] - plan["clipStartSec"], plan["outputDurationSec"]))
    else:
        log("no edits ({} {})".format(plan.get("status"), plan.get("reason", "")))
    print(json.dumps(plan))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
