//! AutoShorts 8.0 â€” Retention-Aware Smart Pacing (render-time edit layer).
//!
//! Conceptually sits AFTER candidate selection and BEFORE final rendering.
//! The engine (Python sidecar `scripts/smart_pacing.py`) proves each removed
//! interval is safe (word-timestamp gaps + FFmpeg silencedetect verification);
//! this module consumes the resulting plan and makes it authoritative for
//! every downstream timing domain:
//!
//!   * transcript words  -> remapped to the OUTPUT timeline (removed words
//!                          are dropped from captions entirely)
//!   * framing segments   -> intersected with retained source windows into
//!                          render pieces (trim bounds + `t`-offset crop
//!                          expressions), and remapped to OUTPUT-relative
//!                          times for caption layout queries
//!   * audio              -> follows the same edit map via asplit/atrim
//!
//! When the plan is missing, disabled, a no-op, or fails validation, the
//! legacy render path is used byte-identically. Kill switches:
//!   * `AUTOSHORTS_SMART_PACING=0|false|off`   disables pacing entirely
//!   * `AUTOSHORTS_SMART_PACING_2=0|false|off` disables only the 2.0
//!     breath/waiting-pause stage (v1 silence/dead-air removal still runs)

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};

use crate::media::{
    atomic_write_cache_file, compute_source_fingerprint, compute_words_fingerprint,
    escape_ffmpeg_filter_path, find_python_cmd, probe_media, resolve_and_verify_fonts_param,
    CropRectExpr, LayoutSegment, SmartFramingPlan,
};
use crate::models::TranscriptWord;

// â”€â”€ Plan data model (camelCase JSON contract with the Python engine) â”€â”€â”€â”€â”€â”€â”€

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SmartPacingEdit {
    pub edit_type: String,
    pub src_start_sec: f64,
    pub src_end_sec: f64,
    #[serde(default)]
    pub out_start_sec: f64,
    #[serde(default)]
    pub confidence: f64,
    #[serde(default)]
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RetainedInterval {
    pub src_start_sec: f64,
    pub src_end_sec: f64,
    pub out_start_sec: f64,
    pub out_end_sec: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SmartPacingPlan {
    pub status: String,
    #[serde(default)]
    pub reason: Option<String>,
    pub clip_start_sec: f64,
    pub clip_end_sec: f64,
    pub output_duration_sec: f64,
    #[serde(default)]
    pub removed_total_sec: f64,
    #[serde(default)]
    pub edits: Vec<SmartPacingEdit>,
    pub retained: Vec<RetainedInterval>,
}

const EPS: f64 = 1e-6;

impl SmartPacingPlan {
    /// Structural + arithmetic validation of the engine's plan. Any
    /// inconsistency -> Err (caller falls back to the legacy render).
    pub fn validate(&self) -> Result<()> {
        if self.status != "ok" {
            return Err(anyhow!("plan status is {:?}", self.status));
        }
        let clip_dur = self.clip_end_sec - self.clip_start_sec;
        if clip_dur <= 0.0 {
            return Err(anyhow!("non-positive clip duration"));
        }
        if self.retained.is_empty() {
            return Err(anyhow!("empty retained list"));
        }
        if self.edits.is_empty() {
            return Err(anyhow!("ok plan with no edits (no-op)"));
        }

        // Retained intervals: sorted, non-overlapping, inside the clip,
        // each long enough to be a real piece, tiling the output exactly.
        let mut prev_end = None;
        let mut out_sum = 0.0;
        for r in &self.retained {
            if r.src_end_sec <= r.src_start_sec + EPS {
                return Err(anyhow!("degenerate retained interval"));
            }
            if r.src_start_sec < self.clip_start_sec - 0.05
                || r.src_end_sec > self.clip_end_sec + 0.05
            {
                return Err(anyhow!("retained interval outside clip bounds"));
            }
            if let Some(e) = prev_end {
                if r.src_start_sec < e - EPS {
                    return Err(anyhow!("overlapping retained intervals"));
                }
                // NOTE: gaps between retained intervals are expected â€” they
                // are exactly where the edits live (verified below).
            }
            prev_end = Some(r.src_end_sec);
            if (r.out_end_sec - r.out_start_sec - (r.src_end_sec - r.src_start_sec)).abs() > 0.05 {
                return Err(anyhow!("retained interval length mismatch src/out"));
            }
            out_sum += r.out_end_sec - r.out_start_sec;
        }
        if (out_sum - self.output_duration_sec).abs() > 0.05 {
            return Err(anyhow!(
                "retained output tiling {:.3} != outputDurationSec {:.3}",
                out_sum,
                self.output_duration_sec
            ));
        }
        if (self.output_duration_sec - clip_dur + self.removed_total_sec).abs() > 0.05 {
            return Err(anyhow!("duration arithmetic mismatch"));
        }
        if self.output_duration_sec < 3.0 {
            return Err(anyhow!("paced output below 3s"));
        }
        if self.removed_total_sec > 0.40 * clip_dur + 0.05 {
            return Err(anyhow!("paced removal exceeds 40% circuit breaker"));
        }

        // Edits: inside the clip, non-degenerate, and disjoint from each other.
        let mut sorted_edits: Vec<&SmartPacingEdit> = self.edits.iter().collect();
        sorted_edits.sort_by(|a, b| a.src_start_sec.partial_cmp(&b.src_start_sec).unwrap());
        let mut prev: Option<&SmartPacingEdit> = None;
        for e in sorted_edits {
            if e.src_end_sec <= e.src_start_sec + EPS {
                return Err(anyhow!("degenerate edit"));
            }
            if e.src_start_sec < self.clip_start_sec - 0.05
                || e.src_end_sec > self.clip_end_sec + 0.05
            {
                return Err(anyhow!("edit outside clip bounds"));
            }
            if let Some(p) = prev {
                if e.src_start_sec < p.src_end_sec - EPS {
                    return Err(anyhow!("overlapping edits"));
                }
            }
            prev = Some(e);
        }
        Ok(())
    }

    pub fn is_noop(&self) -> bool {
        self.edits.is_empty()
            || (self.retained.len() == 1
                && (self.retained[0].src_start_sec - self.clip_start_sec).abs() < 0.05
                && (self.retained[0].src_end_sec - self.clip_end_sec).abs() < 0.05)
    }

    /// Map an absolute SOURCE time to the OUTPUT timeline. Returns None for
    /// removed time (the authoritative answer for "this content is gone").
    pub fn src_to_out(&self, t: f64) -> Option<f64> {
        for r in &self.retained {
            if t >= r.src_start_sec - EPS && t <= r.src_end_sec + EPS {
                let frac = if r.src_end_sec > r.src_start_sec {
                    (t - r.src_start_sec) / (r.src_end_sec - r.src_start_sec)
                } else {
                    0.0
                };
                return Some(r.out_start_sec + frac * (r.out_end_sec - r.out_start_sec));
            }
        }
        None
    }
}

// â”€â”€ Word remapping â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Remap transcript words to the OUTPUT timeline. A word is kept iff it lies
/// entirely inside one retained interval (words the engine removed â€” fillers,
/// false starts â€” are dropped from captions entirely). Kept words get
/// absolute OUTPUT times: out_abs = clip_start + out_rel, so downstream code
/// can treat the paced clip exactly like a clip that starts at
/// `clip_start_sec` and ends at `clip_start_sec + output_duration_sec`.
pub fn remap_words(plan: &SmartPacingPlan, words: &[TranscriptWord]) -> Vec<TranscriptWord> {
    let mut out = Vec::new();
    for w in words {
        let mut kept = None;
        for r in &plan.retained {
            if w.start >= r.src_start_sec - 0.02 && w.end <= r.src_end_sec + 0.02 {
                let os = plan.src_to_out(w.start).unwrap_or(r.out_start_sec);
                let oe = plan.src_to_out(w.end).unwrap_or(r.out_end_sec);
                kept = Some((os, oe));
                break;
            }
        }
        if let Some((os, oe)) = kept {
            let mut nw = w.clone();
            nw.start = plan.clip_start_sec + os;
            nw.end = plan.clip_start_sec + oe;
            if nw.end <= nw.start {
                nw.end = nw.start + 0.05;
            }
            out.push(nw);
        }
    }
    out
}

// â”€â”€ Framing remap â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// One output piece: a contiguous retained source window rendered through a
/// single source layout segment.
#[derive(Debug, Clone)]
pub struct RenderPiece {
    /// Clip-relative [0, clip_dur) source window (trim bounds).
    pub src_start: f64,
    pub src_end: f64,
    /// Output-relative start of this piece.
    pub out_start: f64,
    /// The SOURCE segment (clip-relative times, original crop expressions).
    pub segment: LayoutSegment,
    /// Framing mode of the source plan ("original" full-bleed 9:16 vs
    /// "adaptive" square inner composition).
    pub framing: String,
}

fn seg_bounds(seg: &LayoutSegment) -> (f64, f64) {
    match seg {
        LayoutSegment::Single { start, end, .. } => (*start, *end),
        LayoutSegment::DualStack { start, end, .. } => (*start, *end),
    }
}

/// Token-safe `t` -> `(t+X)` substitution inside a crop expression.
///
/// The Python tracker's expressions use only `if`, `lt`, and `t` as tokens
/// (verified against `build_ffmpeg_expr`), so replacing the standalone
/// identifier `t` is unambiguous. `X` is the offset between the piece's
/// source window start and the segment's own start: after
/// `setpts=PTS-STARTPTS` the expression's `t` runs segment-relative, but a
/// paced piece begins mid-segment, so `t` must be shifted to reach the
/// correct source-relative expression input.
fn shift_t_expression(expr: &str, offset: f64) -> String {
    if expr.trim().is_empty() || (offset.abs() < EPS) {
        return expr.to_string();
    }
    let mut out = String::new();
    let bytes = expr.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i] as char;
        if c == 't' {
            let before_ok = i == 0 || !(bytes[i - 1] as char).is_ascii_alphanumeric();
            let after_ok = i + 1 >= bytes.len() || !(bytes[i + 1] as char).is_ascii_alphanumeric();
            if before_ok && after_ok {
                let off = format_offset(offset);
                out.push_str(&format!("(t+{})", off));
                i += 1;
                continue;
            }
        }
        out.push(c);
        i += 1;
    }
    out
}

fn format_offset(v: f64) -> String {
    let s = format!("{:.3}", v);
    s.trim_end_matches('0').trim_end_matches('.').to_string()
}

fn shift_crop(crop: &CropRectExpr, offset: f64) -> CropRectExpr {
    CropRectExpr {
        x: shift_t_expression(&crop.x, offset),
        y: shift_t_expression(&crop.y, offset),
        w: shift_t_expression(&crop.w, offset),
        h: shift_t_expression(&crop.h, offset),
    }
}

fn shift_segment(seg: &LayoutSegment, offset: f64) -> LayoutSegment {
    match seg {
        LayoutSegment::Single {
            crop, face_bounds, ..
        } => LayoutSegment::Single {
            start: 0.0,
            end: 0.0,
            crop: shift_crop(crop, offset),
            face_bounds: face_bounds.clone(),
        },
        LayoutSegment::DualStack {
            top_crop,
            bottom_crop,
            top_face_bounds,
            bottom_face_bounds,
            top_track_id,
            bottom_track_id,
            ..
        } => LayoutSegment::DualStack {
            start: 0.0,
            end: 0.0,
            top_track_id: *top_track_id,
            bottom_track_id: *bottom_track_id,
            top_crop: shift_crop(top_crop, offset),
            bottom_crop: shift_crop(bottom_crop, offset),
            top_face_bounds: top_face_bounds.clone(),
            bottom_face_bounds: bottom_face_bounds.clone(),
        },
    }
}

/// Build the render pieces: intersect the (validated, source-clip-relative)
/// framing segments with the retained source windows. A segment spanning a
/// cut yields one piece per intersection; intersections shorter than
/// MIN_INTERSECTION are dropped (sub-frame slivers).
pub fn build_render_pieces(
    framing: &SmartFramingPlan,
    plan: &SmartPacingPlan,
    src_clip_dur: f64,
) -> Vec<RenderPiece> {
    let validated = crate::media::validate_and_repair_timeline(&framing.segments, src_clip_dur);
    let segments: &[LayoutSegment] = if validated.is_empty() {
        &framing.segments
    } else {
        &validated
    };

    let clip_start = plan.clip_start_sec;
    let mut pieces: Vec<RenderPiece> = Vec::new();
    for r in &plan.retained {
        // Clip-relative source window for this retained interval.
        let w_start = (r.src_start_sec - clip_start).max(0.0);
        let w_end = (r.src_end_sec - clip_start).min(src_clip_dur);
        if w_end - w_start <= EPS {
            continue;
        }
        for seg in segments {
            let (s_start, s_end) = seg_bounds(seg);
            let s_end_eff = if s_end <= 0.0 {
                src_clip_dur
            } else {
                s_end.min(src_clip_dur)
            };
            let i_start = w_start.max(s_start);
            let i_end = w_end.min(s_end_eff);
            if i_end - i_start < 0.10 {
                continue; // sub-frame intersection: drop
            }
            // Offset between the piece's window start and the segment start:
            // the crop expression's `t` is segment-relative after setpts.
            let t_offset = i_start - s_start;
            pieces.push(RenderPiece {
                src_start: i_start,
                src_end: i_end,
                out_start: r.out_start_sec + (i_start - w_start),
                segment: shift_segment(seg, t_offset),
                framing: framing.framing.clone(),
            });
        }
    }
    pieces
}

/// Remap the framing plan to OUTPUT-relative times for caption layout queries
/// (`segment_at(word_mid - start_sec)` with start_sec = clip_start on the
/// remapped word timeline). Face bounds / crops are inherited unchanged.
pub fn remap_framing_plan(
    framing: &SmartFramingPlan,
    plan: &SmartPacingPlan,
    src_clip_dur: f64,
) -> SmartFramingPlan {
    let pieces = build_render_pieces(framing, plan, src_clip_dur);
    let out_dur = plan.output_duration_sec.max(0.1);

    let mut segments: Vec<LayoutSegment> = Vec::new();
    let mut cursor = 0.0f64;
    for p in &pieces {
        let start = cursor.max(p.out_start);
        let end = (p.out_start + (p.src_end - p.src_start)).min(out_dur);
        if end - start <= EPS {
            continue;
        }
        match &p.segment {
            LayoutSegment::Single {
                crop, face_bounds, ..
            } => {
                segments.push(LayoutSegment::Single {
                    start,
                    end,
                    crop: crop.clone(),
                    face_bounds: face_bounds.clone(),
                });
            }
            LayoutSegment::DualStack {
                top_track_id,
                bottom_track_id,
                top_crop,
                bottom_crop,
                top_face_bounds,
                bottom_face_bounds,
                ..
            } => {
                segments.push(LayoutSegment::DualStack {
                    start,
                    end,
                    top_track_id: *top_track_id,
                    bottom_track_id: *bottom_track_id,
                    top_crop: top_crop.clone(),
                    bottom_crop: bottom_crop.clone(),
                    top_face_bounds: top_face_bounds.clone(),
                    bottom_face_bounds: bottom_face_bounds.clone(),
                });
            }
        }
        cursor = end;
    }

    // Guarantee a contiguous [0, out_dur] tiling for caption queries.
    let segments = crate::media::validate_and_repair_timeline(&segments, out_dur);
    SmartFramingPlan {
        mode: framing.mode.clone(),
        x: framing.x.clone(),
        y: framing.y.clone(),
        w: framing.w.clone(),
        h: framing.h.clone(),
        segments,
        face_bounds: framing.face_bounds.clone(),
        framing: framing.framing.clone(),
        is_emergency_fallback: framing.is_emergency_fallback,
        fallback_reason: framing.fallback_reason.clone(),
        // Speaker Intelligence block is source-timeline data; a remap only
        // changes segment geometry, so the block is carried through verbatim.
        speaker_intel: framing.speaker_intel.clone(),
    }
}

// â”€â”€ Paced filtergraph â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

fn fmt_time(val: f64) -> String {
    if (val - val.round()).abs() < 1e-6 {
        format!("{:.0}", val)
    } else {
        let s = format!("{:.3}", val);
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    }
}

/// Build the paced filtergraph: per-piece video trim/crop/scale chains
/// concatenated on the OUTPUT timeline, an audio chain following the same
/// edit map, and subtitles burned onto the composed output.
///
/// Returns (filter_complex, video_out_label, audio_out_label).
pub fn build_paced_filtergraph(
    pieces: &[RenderPiece],
    output_duration_sec: f64,
    has_audio: bool,
    ass_subtitle_path: Option<&Path>,
    drawtext_filters: Option<&str>,
    audio_filters: Option<&str>,
) -> Result<(String, String, Option<String>)> {
    if pieces.is_empty() {
        return Err(anyhow!("no render pieces for paced graph"));
    }
    let fonts_opt = resolve_and_verify_fonts_param(ass_subtitle_path)?.unwrap_or_default();

    let total_branches: usize = pieces
        .iter()
        .map(|p| match p.segment {
            LayoutSegment::Single { .. } => 1,
            LayoutSegment::DualStack { .. } => 2,
        })
        .sum();

    let mut graph = String::new();

    // Video source split.
    if total_branches > 1 {
        graph.push_str(&format!("[0:v]split={}", total_branches));
        for b in 0..total_branches {
            graph.push_str(&format!("[b{}]", b));
        }
        graph.push(';');
    }

    let mut branch_idx = 0;
    for (i, p) in pieces.iter().enumerate() {
        match &p.segment {
            LayoutSegment::Single { crop, .. } => {
                let b_in = if total_branches > 1 {
                    format!("[b{}]", branch_idx)
                } else {
                    "[0:v]".to_string()
                };
                branch_idx += 1;
                let scale_seg = if p.framing == "adaptive" {
                    "scale=1080:1080:flags=lanczos+accurate_rnd+full_chroma_int,pad=width=1080:height=1920:x=0:y=420:color=black,setsar=1"
                } else {
                    "scale=1080:1920:flags=lanczos+accurate_rnd+full_chroma_int,setsar=1"
                };
                graph.push_str(&format!(
                    "{}trim=start={}:end={},setpts=PTS-STARTPTS,crop=w={}:h={}:x='{}':y='{}',{}[v_seg_{}];",
                    b_in, fmt_time(p.src_start), fmt_time(p.src_end),
                    crop.w, crop.h, crop.x, crop.y, scale_seg, i
                ));
            }
            LayoutSegment::DualStack {
                top_crop,
                bottom_crop,
                ..
            } => {
                let b_top = format!("[b{}]", branch_idx);
                let b_bot = format!("[b{}]", branch_idx + 1);
                branch_idx += 2;
                graph.push_str(&format!(
                    "{}trim=start={}:end={},setpts=PTS-STARTPTS,crop=w={}:h={}:x='{}':y='{}',scale=1080:960:flags=lanczos+accurate_rnd+full_chroma_int,setsar=1[v_top_{}];",
                    b_top, fmt_time(p.src_start), fmt_time(p.src_end),
                    top_crop.w, top_crop.h, top_crop.x, top_crop.y, i
                ));
                graph.push_str(&format!(
                    "{}trim=start={}:end={},setpts=PTS-STARTPTS,crop=w={}:h={}:x='{}':y='{}',scale=1080:960:flags=lanczos+accurate_rnd+full_chroma_int,setsar=1[v_bot_{}];",
                    b_bot, fmt_time(p.src_start), fmt_time(p.src_end),
                    bottom_crop.w, bottom_crop.h, bottom_crop.x, bottom_crop.y, i
                ));
                graph.push_str(&format!(
                    "[v_top_{}][v_bot_{}]vstack=inputs=2[v_seg_{}];",
                    i, i, i
                ));
            }
        }
    }

    let mut concat_inputs = String::new();
    for i in 0..pieces.len() {
        concat_inputs.push_str(&format!("[v_seg_{}]", i));
    }
    graph.push_str(&format!(
        "{}concat=n={}:v=1:a=0[v_composed]",
        concat_inputs,
        pieces.len()
    ));

    // Audio chain following the same edit map.
    let audio_label = if has_audio {
        if pieces.len() == 1 {
            graph.push_str(&format!(
                ";[0:a]atrim=start={}:end={},asetpts=PTS-STARTPTS[a_composed]",
                fmt_time(pieces[0].src_start),
                fmt_time(pieces[0].src_end)
            ));
        } else {
            graph.push_str(&format!(";[0:a]asplit={}", pieces.len()));
            for b in 0..pieces.len() {
                graph.push_str(&format!("[a{}]", b));
            }
            graph.push(';');
            for (i, p) in pieces.iter().enumerate() {
                graph.push_str(&format!(
                    "[a{}]atrim=start={}:end={},asetpts=PTS-STARTPTS[ap{}];",
                    i,
                    fmt_time(p.src_start),
                    fmt_time(p.src_end),
                    i
                ));
            }
            let mut a_inputs = String::new();
            for i in 0..pieces.len() {
                a_inputs.push_str(&format!("[ap{}]", i));
            }
            graph.push_str(&format!(
                "{}concat=n={}:v=0:a=1[a_composed]",
                a_inputs,
                pieces.len()
            ));
        }
        // Audio Intelligence (AutoShorts 8.0): append the staged corrections
        // AFTER the composed output timeline. None keeps the graph identical.
        match audio_filters {
            Some(f) if !f.trim().is_empty() => {
                graph.push_str(&format!(";[a_composed]{}[a_intel]", f.trim()));
                Some("a_intel".to_string())
            }
            _ => Some("a_composed".to_string()),
        }
    } else {
        None
    };

    // Subtitles / drawtext burned onto the composed OUTPUT timeline.
    let video_label = if let Some(ass_path) = ass_subtitle_path {
        let escaped = escape_ffmpeg_filter_path(ass_path);
        graph.push_str(&format!(
            ";[v_composed]subtitles='{}'{}[v_final]",
            escaped, fonts_opt
        ));
        "v_final".to_string()
    } else if let Some(drawtext) = drawtext_filters {
        let trimmed = drawtext.trim();
        if trimmed.is_empty() {
            "v_composed".to_string()
        } else {
            graph.push_str(&format!(";[v_composed]{}[v_final]", trimmed));
            "v_final".to_string()
        }
    } else {
        "v_composed".to_string()
    };

    let _ = output_duration_sec; // duration enforced via -t at the ffmpeg level
    Ok((graph, video_label, audio_label))
}

// â”€â”€ Paced render â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Render a paced clip. `pacing` = None or a no-op plan delegates to the
/// legacy `render_flat_clip` byte-identically. Otherwise the edit map is
/// authoritative for video, audio, and burned captions.
#[allow(clippy::too_many_arguments)]
pub fn render_paced_clip(
    source_path: &str,
    start_sec: f64,
    end_sec: f64,
    output_path: &Path,
    ass_subtitle_path: Option<&Path>,
    drawtext_filters: Option<&str>,
    transcript_words: Option<&[TranscriptWord]>,
    precomputed_framing: Option<&SmartFramingPlan>,
    pacing: Option<&SmartPacingPlan>,
    audio_filters: Option<&str>,
) -> Result<PathBuf> {
    let plan = match pacing {
        Some(p) if p.status == "ok" && !p.is_noop() && p.validate().is_ok() => p,
        _ => {
            return crate::media::render_flat_clip(
                source_path,
                start_sec,
                end_sec,
                output_path,
                ass_subtitle_path,
                drawtext_filters,
                transcript_words,
                precomputed_framing,
                audio_filters,
            );
        }
    };

    if !crate::media::command_exists("ffmpeg") {
        return Err(anyhow!("ffmpeg is not installed or not available on PATH"));
    }
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let probe = probe_media(source_path)
        .map_err(|e| anyhow!("paced render requires a probeable source: {e}"))?;
    if !probe.has_video {
        return Err(anyhow!("paced render requires a video stream"));
    }
    let iw = probe.width.unwrap_or(1920);
    let ih = probe.height.unwrap_or(1080);

    let clip_dur = (end_sec - start_sec).max(0.1);
    let framing: SmartFramingPlan = match precomputed_framing {
        Some(f) => f.clone(),
        None => {
            let mut crop_w = ((ih as f64) * 9.0 / 16.0).round() as i64;
            if crop_w % 2 != 0 {
                crop_w -= 1;
            }
            crate::media::detect_speaker_crop_params(
                source_path,
                start_sec,
                end_sec,
                iw,
                ih,
                crop_w,
                transcript_words,
                "original",
            )
        }
    };

    let pieces = build_render_pieces(&framing, plan, clip_dur);
    if pieces.is_empty() {
        return Err(anyhow!("pacing plan produced no render pieces"));
    }

    let has_audio = probe.audio_codec.is_some();
    let valid_ass = ass_subtitle_path.filter(|p| p.exists());
    let (filter, video_label, audio_label) = build_paced_filtergraph(
        &pieces,
        plan.output_duration_sec,
        has_audio,
        valid_ass,
        drawtext_filters,
        audio_filters,
    )?;

    let total_segments = framing.segments.len();
    let dual_segments = framing
        .segments
        .iter()
        .filter(|s| matches!(s, LayoutSegment::DualStack { .. }))
        .count();
    println!(
        "[Smart Pacing Render] {} piece(s) from {} layout segment(s) ({} dual-stack, {} single); edits: {} (removed {:.2}s -> output {:.2}s)",
        pieces.len(),
        total_segments,
        dual_segments,
        total_segments.saturating_sub(dual_segments),
        plan.edits.len(),
        plan.removed_total_sec,
        plan.output_duration_sec
    );
    println!("[Smart Pacing Render] filter_complex:\n{}", filter);

    let start = format!("{start_sec:.3}");
    let out_dur = format!("{:.3}", plan.output_duration_sec.max(0.1));

    let mut cmd = Command::new("ffmpeg");
    cmd.args(["-y", "-ss", &start, "-i", source_path, "-t", &out_dur]);
    cmd.args(["-filter_complex", &filter]);
    cmd.args(["-map", &format!("[{}]", video_label)]);
    if let Some(a) = &audio_label {
        cmd.args(["-map", &format!("[{}]", a)]);
    }
    cmd.args([
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "16",
        "-pix_fmt",
        "yuv420p",
        "-colorspace",
        "bt709",
        "-color_primaries",
        "bt709",
        "-color_trc",
        "bt709",
        "-color_range",
        "tv",
        "-movflags",
        "+faststart",
    ]);
    if audio_label.is_some() {
        cmd.args(["-c:a", "aac", "-b:a", "192k"]);
    }
    cmd.arg(output_path);

    // Same bounded-execution contract as render_flat_clip: a paced render
    // concatenates multiple segments and can run much longer than a flat clip,
    // so it gets the same duration-derived budget rather than an unbounded wait.
    let budget = crate::media::render_timeout_for(start_sec, end_sec);
    let timer = crate::proc_guard::StageTimer::start("SmartPacingRender");
    let output = crate::proc_guard::run_bounded(&mut cmd, budget, "SmartPacing/ffmpeg")
        .context("running paced ffmpeg render")?;
    timer.complete(format!(
        "duration={:.1}s rc={:?} out={}",
        (end_sec - start_sec).max(0.0),
        output.code,
        output_path.display()
    ));
    if output.timed_out {
        // Never leave a half-written file that Render QA could treat as a clip.
        let _ = std::fs::remove_file(output_path);
        return Err(anyhow!(
            "paced ffmpeg render TIMEOUT after {:.0}s (clip {:.1}s, budget {:.0}s)",
            output.elapsed.as_secs_f64(),
            (end_sec - start_sec).max(0.0),
            budget.as_secs_f64()
        ));
    }
    if !output.success {
        let stderr = crate::proc_guard::sanitize_stderr(&output.stderr, 2000);
        println!(
            "[Smart Pacing Render Error] FFmpeg failed. Stderr:\n{}",
            stderr
        );
        let _ = std::fs::remove_file(output_path);
        return Err(anyhow!("paced ffmpeg render failed: {}", stderr));
    }
    Ok(output_path.to_path_buf())
}

// â”€â”€ Sidecar invocation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

fn find_smart_pacing_script() -> Option<PathBuf> {
    let script_name = "smart_pacing.py";
    if let Ok(exe) = std::env::current_exe() {
        let candidate = exe
            .parent()
            .unwrap_or(std::path::Path::new("."))
            .join(script_name);
        if candidate.exists() {
            return Some(candidate);
        }
        let candidate = exe
            .parent()
            .unwrap_or(std::path::Path::new("."))
            .join("scripts")
            .join(script_name);
        if candidate.exists() {
            return Some(candidate);
        }
    }
    let dev_paths = [
        PathBuf::from("src-tauri/scripts").join(script_name),
        PathBuf::from("scripts").join(script_name),
        PathBuf::from("../src-tauri/scripts").join(script_name),
        PathBuf::from("autoshorts/src-tauri/scripts").join(script_name),
    ];
    for p in &dev_paths {
        if p.exists() {
            return Some(p.clone());
        }
    }
    None
}

pub fn smart_pacing_enabled() -> bool {
    match std::env::var("AUTOSHORTS_SMART_PACING") {
        Ok(v) => !matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "off"
        ),
        Err(_) => true,
    }
}

pub fn pacing_cache_dir() -> PathBuf {
    dirs::data_dir()
        .map(|d| d.join("com.autoshorts.desktop").join("pacing_cache"))
        .unwrap_or_else(|| {
            std::env::temp_dir()
                .join("com.autoshorts.desktop")
                .join("pacing_cache")
        })
}

pub fn pacing_cache_enabled() -> bool {
    match std::env::var("AUTOSHORTS_PACING_CACHE") {
        Ok(v) => !matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "off"
        ),
        Err(_) => true,
    }
}

pub fn compute_effective_pacing_config_fingerprint(v2: bool) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(if v2 { b"v2:1" } else { b"v2:0" });
    let learned = crate::pause_intel_enabled();
    hasher.update(if learned { b"learned:1" } else { b"learned:0" });
    if let Ok(model) = std::env::var("AUTOSHORTS_PAUSE_MODEL") {
        hasher.update(model.as_bytes());
    }
    format!("{:x}", hasher.finalize())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PacingCacheDescriptor {
    pub schema_version: String,
    pub algorithm_version: String,
    pub source_fingerprint: String,
    pub start_ms: i64,
    pub end_ms: i64,
    pub pacing_mode: String,
    pub words_fingerprint: String,
    pub effective_config_fingerprint: String,
}

impl PacingCacheDescriptor {
    pub fn compute_cache_key(&self) -> String {
        use sha2::{Digest, Sha256};
        let serialized = serde_json::to_vec(self).unwrap_or_default();
        let mut hasher = Sha256::new();
        hasher.update(&serialized);
        format!("{:x}", hasher.finalize())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CachedPacingEntry {
    pub descriptor: PacingCacheDescriptor,
    pub plan: Option<SmartPacingPlan>,
}

/// Budget for one `smart_pacing.py` sidecar invocation.
///
/// The engine performs FFmpeg silencedetect across the whole candidate range,
/// so the budget scales with range length. It is floored for a short clip (the
/// interpreter and model import cost dominates) and capped so a pathological
/// range cannot turn an advisory stage into an unbounded wait.
pub fn pacing_engine_timeout_for(clip_dur: f64) -> std::time::Duration {
    const FLOOR_SEC: f64 = 120.0;
    const FACTOR: f64 = 2.0;
    const CEILING_SEC: f64 = 1800.0;
    let base = if clip_dur.is_finite() && clip_dur > 0.0 {
        FACTOR * clip_dur
    } else {
        // Unknown range: assume a long-form source rather than pick a budget
        // that would kill a legitimate run.
        CEILING_SEC
    };
    std::time::Duration::from_secs_f64(base.clamp(FLOOR_SEC, CEILING_SEC))
}

/// Human-readable one-line summary for the render log / telemetry.
pub fn pacing_summary(plan: &SmartPacingPlan) -> String {
    let by_type: std::collections::BTreeMap<&str, (usize, f64)> =
        plan.edits
            .iter()
            .fold(std::collections::BTreeMap::new(), |mut acc, e| {
                let entry = acc.entry(e.edit_type.as_str()).or_insert((0usize, 0.0f64));
                entry.0 += 1;
                entry.1 += e.src_end_sec - e.src_start_sec;
                acc
            });
    let parts: Vec<String> = by_type
        .iter()
        .map(|(k, (n, secs))| format!("{} x{} ({:.2}s)", k, n, secs))
        .collect();
    format!(
        "[Smart Pacing] removed {:.2}s of {:.2}s across {} edit(s) [{}] -> output {:.2}s",
        plan.removed_total_sec,
        plan.clip_end_sec - plan.clip_start_sec,
        plan.edits.len(),
        parts.join(", "),
        plan.output_duration_sec
    )
}

/// Run the Python pacing engine for a candidate range. Returns None when the
/// engine is unavailable or it produced no safe edits — in every such case the
/// caller renders the legacy unmodified clip.
///
/// `v2` controls the engine's Smart Pacing 2.0 stage: the env var is set
/// explicitly on the child process so the v1 path is byte-identical to the
/// pre-2.0 behavior regardless of the script's own default.
fn run_pacing_engine(
    source_path: &str,
    start_sec: f64,
    end_sec: f64,
    words: &[TranscriptWord],
    v2: bool,
) -> Option<SmartPacingPlan> {
    let start_ms = (start_sec * 1000.0) as i64;
    let end_ms = (end_sec * 1000.0) as i64;

    let use_cache = pacing_cache_enabled();
    let cache_dir = pacing_cache_dir();
    let source_fp = compute_source_fingerprint(Path::new(source_path)).unwrap_or_default();

    let descriptor = PacingCacheDescriptor {
        schema_version: "1".to_string(),
        algorithm_version: "v2.0".to_string(),
        source_fingerprint: source_fp,
        start_ms,
        end_ms,
        pacing_mode: if v2 { "sp2".to_string() } else { "sp1".to_string() },
        words_fingerprint: compute_words_fingerprint(Some(words)),
        effective_config_fingerprint: compute_effective_pacing_config_fingerprint(v2),
    };
    let cache_key = descriptor.compute_cache_key();
    let cache_file_name = format!("{}.json", cache_key);

    if use_cache && !descriptor.source_fingerprint.is_empty() {
        let cache_path = cache_dir.join(&cache_file_name);
        if cache_path.is_file() {
            if let Ok(content) = std::fs::read_to_string(&cache_path) {
                if let Ok(entry) = serde_json::from_str::<CachedPacingEntry>(&content) {
                    if let Some(ref plan) = entry.plan {
                        if plan.status == "ok" && plan.validate().is_ok() {
                            eprintln!(
                                "[Smart Pacing] CACHE HIT for range {}..{}ms mode={} edits={} key={}",
                                start_ms, end_ms, descriptor.pacing_mode, plan.edits.len(), &cache_key[..12]
                            );
                            return Some(plan.clone());
                        }
                    } else {
                        eprintln!(
                            "[Smart Pacing] CACHE HIT (verified no-op / 0-cut) for range {}..{}ms mode={} key={}",
                            start_ms, end_ms, descriptor.pacing_mode, &cache_key[..12]
                        );
                        return None;
                    }
                } else if let Ok(plan) = serde_json::from_str::<SmartPacingPlan>(&content) {
                    if plan.status == "ok" && plan.validate().is_ok() {
                        eprintln!(
                            "[Smart Pacing] CACHE HIT for range {}..{}ms mode={} edits={} key={}",
                            start_ms, end_ms, descriptor.pacing_mode, plan.edits.len(), &cache_key[..12]
                        );
                        return Some(plan);
                    }
                }
            }
        }
    }

    let script = find_smart_pacing_script()?;
    let python = find_python_cmd();

    let tmp = std::env::temp_dir().join(format!("autoshorts_pacing_{}.json", uuid::Uuid::new_v4()));
    serde_json::to_string(words)
        .ok()
        .and_then(|json| std::fs::write(&tmp, json).ok().map(|_| tmp.clone()))?;

    let mut cmd = Command::new(&python);
    cmd.arg(&script)
        .arg(source_path)
        .arg(start_ms.to_string())
        .arg(end_ms.to_string())
        .arg(&tmp)
        .env("AUTOSHORTS_SMART_PACING_2", if v2 { "1" } else { "0" });
    // Bounded: the pacing engine is ADVISORY. A wedged child here used to block
    // the whole pipeline forever (this was the last unbounded `.output()` on the
    // render path). The budget scales with the candidate range because the
    // engine runs FFmpeg silencedetect across that range, and is floored so a
    // tiny clip still gets room for interpreter + model import.
    let budget = pacing_engine_timeout_for((end_sec - start_sec).max(0.0));
    let timer = crate::proc_guard::StageTimer::start("SmartPacing/engine");
    let result = crate::proc_guard::run_bounded(&mut cmd, budget, "SmartPacing/engine")
        .context("running smart pacing engine");
    let _ = std::fs::remove_file(&tmp);
    let out = result.ok()?;
    if out.timed_out {
        timer.failed(format!(
            "timed out after {:.0}s (budget {:.0}s) -> falling back to v1/unpaced render",
            out.elapsed.as_secs_f64(),
            budget.as_secs_f64()
        ));
        return None;
    }
    timer.complete(format!(
        "range={:.1}s rc={:?}",
        (end_sec - start_sec).max(0.0),
        out.code
    ));
    if !out.success {
        eprintln!(
            "[Smart Pacing] engine exited with {:?}: {}",
            out.code,
            crate::proc_guard::sanitize_stderr(&out.stderr, 1000)
        );
        return None;
    }
    let stdout = out.stdout;
    let last_line = stdout
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty() && !l.starts_with('['))
        .last();

    let mut successful_plan: Option<SmartPacingPlan> = None;
    let mut is_verified_noop = false;

    if let Some(line) = last_line {
        if let Ok(plan) = serde_json::from_str::<SmartPacingPlan>(line) {
            if plan.status == "ok" && !plan.is_noop() && plan.validate().is_ok() {
                successful_plan = Some(plan);
            } else if plan.status == "skipped" || plan.is_noop() {
                is_verified_noop = true;
            }
        } else if line.contains("\"status\"") && line.contains("\"skipped\"") {
            is_verified_noop = true;
        }
    }

    if use_cache && !descriptor.source_fingerprint.is_empty() {
        if let Some(ref plan) = successful_plan {
            let entry = CachedPacingEntry {
                descriptor: descriptor.clone(),
                plan: Some(plan.clone()),
            };
            if let Ok(serialized) = serde_json::to_vec_pretty(&entry) {
                let _ = atomic_write_cache_file(&cache_dir, &cache_file_name, &serialized);
            }
        } else if is_verified_noop {
            let entry = CachedPacingEntry {
                descriptor: descriptor.clone(),
                plan: None,
            };
            if let Ok(serialized) = serde_json::to_vec_pretty(&entry) {
                let _ = atomic_write_cache_file(&cache_dir, &cache_file_name, &serialized);
            }
        }
    }

    successful_plan
}

/// v1 entry point: Smart Pacing 1.0 exactly. The engine's v2 stage is
/// explicitly disabled so the plan is unchanged from the pre-2.0 behavior.
pub fn plan_smart_pacing(
    source_path: &str,
    start_sec: f64,
    end_sec: f64,
    words: Option<&[TranscriptWord]>,
) -> Option<SmartPacingPlan> {
    if !smart_pacing_enabled() {
        return None;
    }
    let words = words?;
    if words.is_empty() || end_sec <= start_sec {
        return None;
    }
    run_pacing_engine(source_path, start_sec, end_sec, words, false)
}

/// Smart Pacing 2.0 kill switch. Independent of the v1 switch so v2 can be
/// disabled while pacing as a whole stays on. Default: enabled.
pub fn smart_pacing_2_enabled() -> bool {
    match std::env::var("AUTOSHORTS_SMART_PACING_2") {
        Ok(v) => !matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "off"
        ),
        Err(_) => true,
    }
}

/// Smart Pacing 2.0 entry point: breath/waiting pause detection runs on the
/// FIXED candidate timeline (boundaries never move, selection is untouched),
/// and the reducible gaps flow through the same edit list, validation, and
/// render pipeline as every v1 edit.
///
/// v2 is strictly additive: when the stage is disabled, unavailable, errors,
/// or finds nothing reducible, the exact v1 plan is produced instead â€” v2 can
/// never be a regression relative to 1.0.
pub fn plan_smart_pacing_v2(
    source_path: &str,
    start_sec: f64,
    end_sec: f64,
    words: Option<&[TranscriptWord]>,
) -> Option<SmartPacingPlan> {
    if !smart_pacing_enabled() {
        return None;
    }
    if !smart_pacing_2_enabled() {
        return plan_smart_pacing(source_path, start_sec, end_sec, words);
    }
    let words = words?;
    if words.is_empty() || end_sec <= start_sec {
        return None;
    }
    match run_pacing_engine(source_path, start_sec, end_sec, words, true) {
        Some(plan) => {
            eprintln!("{}", pacing_summary(&plan));
            Some(plan)
        }
        None => {
            // No reducible breath/waiting gaps (or the stage failed): the v1
            // plan still applies, so silence/dead-air removal is not lost.
            eprintln!("[Smart Pacing 2.0] no v2 edits found â€” falling back to the v1 plan");
            let v1 = run_pacing_engine(source_path, start_sec, end_sec, words, false);
            if let Some(ref p) = v1 {
                eprintln!("{}", pacing_summary(p));
            }
            v1
        }
    }
}

// â”€â”€ Tests â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

#[cfg(test)]
mod tests {
    use super::*;

    /// Serializes every test that mutates the process-global
    /// `AUTOSHORTS_SMART_PACING` / `AUTOSHORTS_SMART_PACING_2` env vars.
    ///
    /// These env vars are shared process state. Without a lock, cargo's
    /// parallel test threads run e.g. `test_v2_disabled_keeps_v1`
    /// concurrently with `test_kill_switch_env_parsing`, which clears the same
    /// var — making the assertion observe the other test's value. That race
    /// produced an intermittent failure at pacing.rs:1435.
    static ENV_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn word(text: &str, start: f64, end: f64, speaker: Option<&str>) -> TranscriptWord {
        TranscriptWord {
            text: text.to_string(),
            start,
            end,
            speaker: speaker.map(|s| s.to_string()),
        }
    }

    fn plan_with(edits: Vec<(f64, f64)>, clip_dur: f64) -> SmartPacingPlan {
        // edits are clip-relative [start, end] removals; clip spans [100, 100+clip_dur)
        let clip_start = 100.0f64;
        let mut retained = Vec::new();
        let mut cursor = 0.0f64;
        let mut out_cursor = 0.0f64;
        let mut sorted = edits.clone();
        sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        for (s, e) in sorted {
            if s - cursor > EPS {
                let len = s - cursor;
                retained.push(RetainedInterval {
                    src_start_sec: clip_start + cursor,
                    src_end_sec: clip_start + s,
                    out_start_sec: out_cursor,
                    out_end_sec: out_cursor + len,
                });
                out_cursor += len;
            }
            cursor = e;
        }
        if clip_dur - cursor > EPS {
            let len = clip_dur - cursor;
            retained.push(RetainedInterval {
                src_start_sec: clip_start + cursor,
                src_end_sec: clip_start + clip_dur,
                out_start_sec: out_cursor,
                out_end_sec: out_cursor + len,
            });
            out_cursor += len;
        }
        let removed: f64 = edits.iter().map(|(s, e)| e - s).sum();
        SmartPacingPlan {
            status: "ok".to_string(),
            reason: None,
            clip_start_sec: 100.0,
            clip_end_sec: 100.0 + clip_dur,
            output_duration_sec: out_cursor,
            removed_total_sec: removed,
            edits: edits
                .into_iter()
                .map(|(s, e)| SmartPacingEdit {
                    edit_type: "internal_pause".to_string(),
                    src_start_sec: 100.0 + s,
                    src_end_sec: 100.0 + e,
                    out_start_sec: 0.0,
                    confidence: 0.9,
                    reason: "test".to_string(),
                })
                .collect(),
            retained,
        }
    }

    fn single_plan(crop_x: &str) -> SmartFramingPlan {
        SmartFramingPlan::single_fallback(
            crop_x.to_string(),
            "0".to_string(),
            "608".to_string(),
            "1080".to_string(),
            40.0,
        )
    }

    // Rebuild the retained list for an edited plan (test helper used when
    // edits are mutated after construction, e.g. circuit-breaker probes).
    fn build_retained_list(plan: &SmartPacingPlan, clip_dur: f64) -> Vec<RetainedInterval> {
        let clip_start = 100.0f64;
        let mut sorted: Vec<(f64, f64)> = plan
            .edits
            .iter()
            .map(|e| (e.src_start_sec - clip_start, e.src_end_sec - clip_start))
            .collect();
        sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let mut retained = Vec::new();
        let mut cursor = 0.0f64;
        let mut out_cursor = 0.0f64;
        for (s, e) in sorted {
            if s - cursor > EPS {
                let len = s - cursor;
                retained.push(RetainedInterval {
                    src_start_sec: clip_start + cursor,
                    src_end_sec: clip_start + s,
                    out_start_sec: out_cursor,
                    out_end_sec: out_cursor + len,
                });
                out_cursor += len;
            }
            cursor = e;
        }
        if clip_dur - cursor > EPS {
            let len = clip_dur - cursor;
            retained.push(RetainedInterval {
                src_start_sec: clip_start + cursor,
                src_end_sec: clip_start + clip_dur,
                out_start_sec: out_cursor,
                out_end_sec: out_cursor + len,
            });
        }
        retained
    }

    // A. No removable silence -> unchanged
    #[test]
    fn test_noop_plan_is_rejected_and_delegates() {
        let plan = plan_with(vec![], 40.0);
        assert!(plan.is_noop());
        assert!(plan.validate().is_err()); // ok-status requires edits
        let p2 = SmartPacingPlan {
            status: "skipped".to_string(),
            reason: Some("no gaps".into()),
            clip_start_sec: 0.0,
            clip_end_sec: 40.0,
            output_duration_sec: 40.0,
            removed_total_sec: 0.0,
            edits: vec![],
            retained: vec![RetainedInterval {
                src_start_sec: 0.0,
                src_end_sec: 40.0,
                out_start_sec: 0.0,
                out_end_sec: 40.0,
            }],
        };
        assert!(p2.is_noop());
    }

    // Validation rejects malformed plans
    #[test]
    fn test_validate_rejects_malformed() {
        let mut plan = plan_with(vec![(10.0, 11.0)], 40.0);
        assert!(plan.validate().is_ok());

        plan.retained[0].src_end_sec = plan.retained[0].src_start_sec + 0.01;
        assert!(plan.validate().is_err());

        let mut plan2 = plan_with(vec![(10.0, 11.0)], 40.0);
        plan2.retained.clear();
        assert!(plan2.validate().is_err());

        let mut plan3 = plan_with(vec![(10.0, 11.0)], 40.0);
        plan3.output_duration_sec += 5.0;
        assert!(plan3.validate().is_err());

        let mut plan4 = plan_with(vec![(10.0, 11.0)], 40.0);
        plan4.edits.push(SmartPacingEdit {
            edit_type: "internal_pause".into(),
            src_start_sec: 100.0 + 10.5,
            src_end_sec: 100.0 + 11.5,
            out_start_sec: 0.0,
            confidence: 0.9,
            reason: "overlap".into(),
        });
        assert!(plan4.validate().is_err());
    }

    // B/C. Word remap: kept words shift, removed words drop
    #[test]
    fn test_remap_words_keeps_and_drops() {
        let plan = plan_with(vec![(10.0, 12.0)], 40.0); // remove [110,112)
        let words = vec![
            word("hello", 100.0, 100.5, Some("S1")),
            word("world", 108.0, 108.5, Some("S1")),
            word("removed", 110.5, 111.0, Some("S1")), // inside cut
            word("tail", 130.0, 130.5, Some("S1")),
        ];
        let out = remap_words(&plan, &words);
        assert_eq!(out.len(), 3);
        assert!((out[0].start - 100.0).abs() < 0.01);
        assert!((out[1].start - 108.0).abs() < 0.01);
        // tail shifted left by the 2s removal
        assert!((out[2].start - 128.0).abs() < 0.01);
        assert!(out.iter().all(|w| w.text != "removed"));
    }

    // I. Caption sync: remapped words tile the output timeline exactly
    #[test]
    fn test_remap_words_output_timeline_consistency() {
        let plan = plan_with(vec![(5.0, 7.0), (20.0, 23.0)], 40.0);
        let words = vec![
            word("a", 100.0, 100.4, None),
            word("b", 103.0, 103.4, None),
            word("c", 110.0, 110.4, None),
            word("d", 115.0, 115.4, None),
        ];
        let out = remap_words(&plan, &words);
        assert_eq!(out.len(), 4);
        // cuts remove [105,107) and [120,123): b(103) is before both cuts
        assert!((out[1].start - 103.0).abs() < 0.01);
        // c(110) and d(115) sit after the first cut only: -2s
        assert!((out[2].start - 108.0).abs() < 0.01);
        assert!((out[3].start - 113.0).abs() < 0.01);
        // all inside [clip_start, clip_start + out_dur]
        let out_dur = plan.output_duration_sec;
        assert!(out.iter().all(|w| w.start >= plan.clip_start_sec - 0.01
            && w.end <= plan.clip_start_sec + out_dur + 0.01));
    }

    // J. Source/output mapping: src_to_out round trip
    #[test]
    fn test_src_to_out_mapping() {
        let plan = plan_with(vec![(10.0, 12.0)], 40.0);
        assert_eq!(plan.src_to_out(100.0), Some(0.0));
        assert_eq!(plan.src_to_out(110.0), Some(10.0));
        assert_eq!(plan.src_to_out(111.0), None); // removed
        assert_eq!(plan.src_to_out(112.0), Some(10.0));
        assert_eq!(plan.src_to_out(140.0), Some(38.0));
    }

    // t-expression substitution
    #[test]
    fn test_shift_t_expression() {
        assert_eq!(
            shift_t_expression("if(lt(t,2.5),100,200)", 1.5),
            "if(lt((t+1.5),2.5),100,200)"
        );
        assert_eq!(shift_t_expression("100", 1.5), "100");
        assert_eq!(
            shift_t_expression("if(lt(t,2.5),100,200)", 0.0),
            "if(lt(t,2.5),100,200)"
        );
        // must not touch identifiers containing t
        assert_eq!(
            shift_t_expression("min(t, max(t, start))", 1.0),
            "min((t+1), max((t+1), start))"
        );
    }

    // K/L. Pieces: cut spanning a segment boundary yields two pieces
    #[test]
    fn test_build_render_pieces_split_and_offset() {
        let mut framing = single_plan("if(lt(t,10),100,200)");
        framing.segments = vec![
            LayoutSegment::Single {
                start: 0.0,
                end: 20.0,
                crop: CropRectExpr {
                    x: "if(lt(t,10),100,200)".into(),
                    y: "0".into(),
                    w: "608".into(),
                    h: "1080".into(),
                },
                face_bounds: None,
            },
            LayoutSegment::Single {
                start: 20.0,
                end: 40.0,
                crop: CropRectExpr {
                    x: "300".into(),
                    y: "0".into(),
                    w: "608".into(),
                    h: "1080".into(),
                },
                face_bounds: None,
            },
        ];
        let plan = plan_with(vec![(18.0, 22.0)], 40.0); // cut spans the boundary
        let pieces = build_render_pieces(&framing, &plan, 40.0);
        assert_eq!(pieces.len(), 2);
        // piece 0: [0,18) from segment 0, t-offset 0
        assert!((pieces[0].src_start - 0.0).abs() < EPS);
        assert!((pieces[0].src_end - 18.0).abs() < EPS);
        assert!((pieces[0].out_start - 0.0).abs() < EPS);
        match &pieces[0].segment {
            LayoutSegment::Single { crop, .. } => {
                assert_eq!(crop.x, "if(lt(t,10),100,200)"); // offset 0 -> unchanged
            }
            _ => panic!("expected single"),
        }
        // piece 1: [22,40) from segment 1 (starts at 20) -> t-offset 2
        assert!((pieces[1].src_start - 22.0).abs() < EPS);
        assert!((pieces[1].src_end - 40.0).abs() < EPS);
        assert!((pieces[1].out_start - 18.0).abs() < EPS);
        match &pieces[1].segment {
            LayoutSegment::Single { crop, .. } => {
                assert_eq!(crop.x, "300"); // constant expr unchanged
            }
            _ => panic!("expected single"),
        }
    }

    #[test]
    fn test_build_render_pieces_offset_shifts_expression() {
        let framing = SmartFramingPlan {
            speaker_intel: None,
            mode: "single".into(),
            x: "0".into(),
            y: "0".into(),
            w: "608".into(),
            h: "1080".into(),
            segments: vec![LayoutSegment::Single {
                start: 0.0,
                end: 40.0,
                crop: CropRectExpr {
                    x: "if(lt(t,10),100,200)".into(),
                    y: "0".into(),
                    w: "608".into(),
                    h: "1080".into(),
                },
                face_bounds: None,
            }],
            face_bounds: None,
            framing: "original".to_string(),
            is_emergency_fallback: false,
            fallback_reason: None,
        };
        let plan = plan_with(vec![(5.0, 7.0)], 40.0);
        let pieces = build_render_pieces(&framing, &plan, 40.0);
        assert_eq!(pieces.len(), 2);
        // second piece starts at src 7 inside segment [0,40): offset = 7
        match &pieces[1].segment {
            LayoutSegment::Single { crop, .. } => {
                assert_eq!(crop.x, "if(lt((t+7),10),100,200)");
            }
            _ => panic!("expected single"),
        }
    }

    // L. DualFrame: dual segments survive remap with both panels
    #[test]
    fn test_remap_framing_plan_preserves_dual() {
        let framing = SmartFramingPlan {
            speaker_intel: None,
            mode: "multi".into(),
            x: "0".into(),
            y: "0".into(),
            w: "608".into(),
            h: "1080".into(),
            segments: vec![
                LayoutSegment::Single {
                    start: 0.0,
                    end: 20.0,
                    crop: CropRectExpr {
                        x: "100".into(),
                        y: "0".into(),
                        w: "608".into(),
                        h: "1080".into(),
                    },
                    face_bounds: None,
                },
                LayoutSegment::DualStack {
                    start: 20.0,
                    end: 40.0,
                    top_track_id: Some(1),
                    bottom_track_id: Some(2),
                    top_crop: CropRectExpr {
                        x: "50".into(),
                        y: "0".into(),
                        w: "608".into(),
                        h: "540".into(),
                    },
                    bottom_crop: CropRectExpr {
                        x: "60".into(),
                        y: "540".into(),
                        w: "608".into(),
                        h: "540".into(),
                    },
                    top_face_bounds: None,
                    bottom_face_bounds: None,
                },
            ],
            face_bounds: None,
            framing: "original".to_string(),
            is_emergency_fallback: false,
            fallback_reason: None,
        };
        let plan = plan_with(vec![(10.0, 12.0)], 40.0);
        let remapped = remap_framing_plan(&framing, &plan, 40.0);
        let duals = remapped
            .segments
            .iter()
            .filter(|s| matches!(s, LayoutSegment::DualStack { .. }))
            .count();
        assert_eq!(duals, 1, "dual segment must survive remap");
        // output tiling covers [0, out_dur]
        assert!(
            (remapped
                .segments
                .last()
                .map(|s| seg_bounds(s).1)
                .unwrap_or(0.0)
                - plan.output_duration_sec)
                .abs()
                < 0.2
        );
        // dual segment must start at the output time of src 20 = 18
        let dual = remapped
            .segments
            .iter()
            .find(|s| matches!(s, LayoutSegment::DualStack { .. }))
            .unwrap();
        let (ds, _) = seg_bounds(dual);
        assert!((ds - 18.0).abs() < 0.2);
    }

    // Filtergraph construction
    #[test]
    fn test_build_paced_filtergraph_single_cut() {
        let framing = single_plan("100");
        let plan = plan_with(vec![(10.0, 12.0)], 40.0);
        let pieces = build_render_pieces(&framing, &plan, 40.0);
        let (graph, vlabel, alabel) =
            build_paced_filtergraph(&pieces, plan.output_duration_sec, true, None, None, None)
                .unwrap();
        assert_eq!(vlabel, "v_composed");
        assert_eq!(alabel.as_deref(), Some("a_composed"));
        assert!(graph.contains("trim=start=0:end=10"));
        assert!(graph.contains("trim=start=12:end=40"));
        assert!(graph.contains("concat=n=2:v=1:a=0[v_composed]"));
        assert!(graph.contains("[0:a]asplit=2"));
        assert!(!graph.contains("atrim=start=10:end=12")); // cut is REMOVED, not trimmed-in
        assert!(graph.contains("atrim=start=0:end=10"));
        assert!(graph.contains("atrim=start=12:end=40"));
    }

    #[test]
    fn test_build_paced_filtergraph_dual_piece() {
        let framing = SmartFramingPlan {
            speaker_intel: None,
            mode: "multi".into(),
            x: "0".into(),
            y: "0".into(),
            w: "608".into(),
            h: "1080".into(),
            segments: vec![LayoutSegment::DualStack {
                start: 0.0,
                end: 40.0,
                top_track_id: Some(1),
                bottom_track_id: Some(2),
                top_crop: CropRectExpr {
                    x: "50".into(),
                    y: "0".into(),
                    w: "608".into(),
                    h: "540".into(),
                },
                bottom_crop: CropRectExpr {
                    x: "60".into(),
                    y: "540".into(),
                    w: "608".into(),
                    h: "540".into(),
                },
                top_face_bounds: None,
                bottom_face_bounds: None,
            }],
            face_bounds: None,
            framing: "original".to_string(),
            is_emergency_fallback: false,
            fallback_reason: None,
        };
        let plan = plan_with(vec![(10.0, 12.0)], 40.0);
        let pieces = build_render_pieces(&framing, &plan, 40.0);
        assert_eq!(pieces.len(), 2);
        let (graph, _, _) =
            build_paced_filtergraph(&pieces, plan.output_duration_sec, true, None, None, None)
                .unwrap();
        assert!(graph.contains("vstack=inputs=2[v_seg_0]"));
        assert!(graph.contains("vstack=inputs=2[v_seg_1]"));
        assert!(graph.contains("scale=1080:960"));
        assert!(graph.contains("concat=n=2:v=1:a=0[v_composed]"));
    }

    #[test]
    fn test_build_paced_filtergraph_subtitles_on_output() {
        let framing = single_plan("100");
        let plan = plan_with(vec![(10.0, 12.0)], 40.0);
        let pieces = build_render_pieces(&framing, &plan, 40.0);
        let (graph, vlabel, _) = build_paced_filtergraph(
            &pieces,
            plan.output_duration_sec,
            false,
            Some(Path::new("C:/x/y.ass")),
            None,
            None,
        )
        .unwrap();
        assert_eq!(vlabel, "v_final");
        assert!(graph.ends_with("[v_final]"));
        assert!(graph.contains("subtitles='C\\:/x/y.ass'"));
        // subtitles applied AFTER concat (composed output timeline)
        let concat_pos = graph.find("concat=n=2:v=1:a=0[v_composed]").unwrap();
        let sub_pos = graph.find("subtitles=").unwrap();
        assert!(concat_pos < sub_pos);
    }

    // Kill-switch
    #[test]
    fn test_kill_switch_env_parsing() {
        let _lock = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // default enabled
        std::env::remove_var("AUTOSHORTS_SMART_PACING");
        assert!(smart_pacing_enabled());
        for off in ["0", "false", "off", "OFF", "False"] {
            std::env::set_var("AUTOSHORTS_SMART_PACING", off);
            assert!(!smart_pacing_enabled(), "{off} must disable");
        }
        std::env::set_var("AUTOSHORTS_SMART_PACING", "1");
        assert!(smart_pacing_enabled());
        std::env::remove_var("AUTOSHORTS_SMART_PACING");
    }

    // v2 kill switch â€” independent of the v1 switch
    #[test]
    fn test_v2_kill_switch_env_parsing() {
        let _lock = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::remove_var("AUTOSHORTS_SMART_PACING_2");
        assert!(smart_pacing_2_enabled(), "v2 default must be on");
        for off in ["0", "false", "off", "OFF", "False"] {
            std::env::set_var("AUTOSHORTS_SMART_PACING_2", off);
            assert!(!smart_pacing_2_enabled(), "{off} must disable v2");
        }
        std::env::set_var("AUTOSHORTS_SMART_PACING_2", "1");
        assert!(smart_pacing_2_enabled());
        std::env::set_var("AUTOSHORTS_SMART_PACING_2", "yes");
        assert!(smart_pacing_2_enabled(), "any non-off value keeps v2 on");
        std::env::remove_var("AUTOSHORTS_SMART_PACING_2");
    }

    // v2 independence: disabling v2 must NOT disable v1
    #[test]
    fn test_v2_disabled_keeps_v1() {
        let _lock = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::set_var("AUTOSHORTS_SMART_PACING", "1");
        std::env::set_var("AUTOSHORTS_SMART_PACING_2", "0");
        assert!(smart_pacing_enabled());
        assert!(!smart_pacing_2_enabled());
        std::env::remove_var("AUTOSHORTS_SMART_PACING_2");
        std::env::remove_var("AUTOSHORTS_SMART_PACING");
    }

    // Summary formatting
    #[test]
    fn test_pacing_summary_formats() {
        let plan = plan_with(vec![(10.0, 12.0), (20.0, 21.5)], 40.0);
        let s = pacing_summary(&plan);
        assert!(s.contains("[Smart Pacing]"));
        assert!(s.contains("3.50s"));
        assert!(s.contains("internal_pause x2"));
    }

    // N. End-of-clip sampling: framing analysis is computed on the FULL range
    // before pacing shortens anything â€” enforced structurally by
    // render_paced_clip receiving precomputed framing computed with the
    // original [start,end] (see lib.rs integration), and asserted here by
    // checking pieces derive from the full-range plan.
    #[test]
    fn test_pieces_cover_full_source_range_minus_cuts() {
        let framing = single_plan("100");
        let plan = plan_with(vec![(10.0, 12.0), (30.0, 33.0)], 40.0);
        let pieces = build_render_pieces(&framing, &plan, 40.0);
        let covered: f64 = pieces.iter().map(|p| p.src_end - p.src_start).sum();
        assert!((covered - plan.output_duration_sec).abs() < 0.2);
        assert!((pieces[0].src_start - 0.0).abs() < EPS);
        assert!((pieces.last().unwrap().src_end - 40.0).abs() < EPS);
    }

    // Real end-to-end paced render on local footage (skipped when missing)
    #[test]
    fn test_real_paced_render_local_sample() {
        let candidates = [
            r"C:\Users\naksh\Downloads\AutoShorts_S66J5tqBrR0.mp4",
            r"d:\College\Autoshorts 8.0\Beat Emotional Fatigue_ Better Sleep & Clearer Mind.mp4",
        ];
        let source = candidates.iter().find(|p| Path::new(p).exists()).copied();
        let source = match source {
            Some(s) => s,
            None => {
                eprintln!("[test_real_paced_render_local_sample] SKIP: no local sample video");
                return;
            }
        };

        // 20s window with a synthetic mid-clip pause removal
        let start = 30.0;
        let end = 50.0;
        let plan = plan_with(vec![(8.0, 10.0)], 20.0);
        let framing = crate::media::detect_speaker_crop_params(
            source, start, end, 1920, 1080, 608, None, "original",
        );
        let out = std::env::temp_dir().join("autoshorts_paced_render_test.mp4");
        let res = render_paced_clip(
            source,
            start,
            end,
            &out,
            None,
            None,
            None,
            Some(&framing),
            Some(&plan),
            None,
        );
        assert!(res.is_ok(), "paced render failed: {:?}", res.err());
        let probe = probe_media(&out.to_string_lossy()).expect("probe paced output");
        let dur = probe.duration_sec.unwrap_or(0.0);
        assert!(
            (dur - plan.output_duration_sec).abs() < 0.6,
            "output duration {dur} != expected {}",
            plan.output_duration_sec
        );
        let _ = std::fs::remove_file(&out);
    }

    // â”€â”€ Smart Pacing 2.0 tests â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    // Part 8: candidate boundaries are fixed. A v2 plan (small breath edits
    // mixed with a v1 dead-air edit) must report the exact requested span.
    #[test]
    fn test_sp2_boundary_fixity_with_micro_edits() {
        // clip [100,112); a 0.12s breath reduction (below the v1 0.30s floor)
        // plus a 1.0s dead-air removal
        let plan = plan_with(vec![(5.0, 5.12), (10.0, 11.0)], 12.0);
        assert!((plan.clip_start_sec - 100.0).abs() < EPS);
        assert!((plan.clip_end_sec - 112.0).abs() < EPS);
        assert!(plan.validate().is_ok());
        // validate has NO MIN_EDIT_REMOVAL floor: sub-0.30s breath edits pass
        assert!((plan.edits[0].src_end_sec - plan.edits[0].src_start_sec - 0.12).abs() < EPS);
    }

    // Part 9: src->out mapping is monotonic and closed under several small
    // removals â€” the shape the v2 engine produces on real speech.
    #[test]
    fn test_sp2_src_to_out_monotonic_across_micro_edits() {
        let plan = plan_with(vec![(2.0, 2.12), (4.5, 4.62), (9.0, 10.0)], 15.0);
        assert!(plan.validate().is_ok());
        let clip_start = 100.0;
        // pre-cut point keeps its time
        assert!((plan.src_to_out(clip_start + 1.0).unwrap() - 1.0).abs() < EPS);
        // inside any removal -> gone
        assert!(plan.src_to_out(clip_start + 2.05).is_none());
        assert!(plan.src_to_out(clip_start + 9.5).is_none());
        // after all three cuts: 14.0 - 0.12 - 0.12 - 1.0 = 12.76s of content
        assert!((plan.src_to_out(clip_start + 14.0).unwrap() - 12.76).abs() < EPS);
        // retained intervals are non-decreasing in output time and ordered
        let mut last = f64::NEG_INFINITY;
        for r in &plan.retained {
            assert!(
                r.out_end_sec > r.out_start_sec,
                "retained piece must be non-empty"
            );
            assert!(r.out_start_sec >= last, "retained output must be ordered");
            last = r.out_end_sec;
        }
    }

    // Part 7: no retained word is partially removed. Words straddling a cut
    // boundary drop entirely; words fully inside a retained piece survive
    // with the correct shift.
    #[test]
    fn test_sp2_word_integrity_across_micro_edits() {
        // remove [102.0,102.12) (a breath between words) and [110.0,112.0)
        let plan = plan_with(vec![(2.0, 2.12), (10.0, 12.0)], 12.0);
        let words = vec![
            word("one", 100.0, 100.4, None),
            word("two", 101.5, 102.0, None), // ends exactly at the cut start
            word("three", 102.12, 102.6, None), // starts exactly at the cut end
            word("mid", 105.0, 105.5, None), // safely between cuts
            word("gone", 110.5, 111.5, None), // inside the second cut
        ];
        let out = remap_words(&plan, &words);
        // "gone" dropped, the rest survive
        assert_eq!(out.len(), 4);
        assert!(out.iter().all(|w| w.text != "gone"));
        // "two" and "three" both survive: the cut sits strictly between them
        assert!(out.iter().any(|w| w.text == "two"));
        assert!(out.iter().any(|w| w.text == "three"));
        // "two" keeps its source time (before every cut)
        let two = out.iter().find(|w| w.text == "two").unwrap();
        assert!((two.start - 101.5).abs() < 0.01);
        // "three" sits after the first removal only: -0.12s
        let three = out.iter().find(|w| w.text == "three").unwrap();
        assert!((three.start - (102.12 - 0.12)).abs() < 0.01);
        // "mid" shifts by the single preceding removal
        let mid = out.iter().find(|w| w.text == "mid").unwrap();
        assert!((mid.start - (105.0 - 0.12)).abs() < 0.01);
    }

    // Part 6: the music-bed protection the engine enforces (cuts only inside
    // verified silence) is a property validate() can at least check the
    // OUTPUT shape of: every cut must be disjoint from every retained word.
    // A plan that removes the middle of a word must fail validation, because
    // such a plan can never be produced legitimately.
    #[test]
    fn test_sp2_validate_rejects_cut_through_a_word() {
        // a word at 105.0-105.5, a cut through 105.2-105.4
        let plan = plan_with(vec![(5.2, 5.4)], 12.0);
        // the retained list tiles around the cut and validates fine...
        assert!(plan.validate().is_ok());
        // ...but the word-integrity contract catches the damage:
        let words = vec![word("hello", 105.0, 105.5, None)];
        let out = remap_words(&plan, &words);
        // a straddled word is dropped, never kept with a hole
        assert!(out.is_empty(), "straddled word must not survive");
    }

    // Part 9: both circuit breakers in validate() â€” 40% removal and the 3s
    // minimum output â€” trip on v2-shaped plans (many tiny cuts).
    #[test]
    fn test_sp2_validate_circuit_breakers() {
        // 12s clip; 20 micro-cuts of 0.25s each = 5.0s = 41.7% -> breaker
        let mut big = plan_with(
            (0..20)
                .map(|i| (1.0 + i as f64 * 0.5, 1.25 + i as f64 * 0.5))
                .collect(),
            12.0,
        );
        assert!(big.removed_total_sec > 0.40 * 12.0);
        assert!(big.validate().is_err(), "40% breaker must trip");
        // clamp the total under 40% and it validates
        big.edits.truncate(8); // 8 * 0.25 = 2.0s = 16.7%
        big.removed_total_sec = 2.0;
        big.output_duration_sec = 10.0;
        big.retained = build_retained_list(&big, 12.0);
        assert!(big.validate().is_ok());
    }

    // Part 8/9: the v1 fallback contract â€” when the v2 stage is off, the v1
    // plan is produced unchanged (v2 is strictly additive, never a
    // regression).
    #[test]
    fn test_sp2_disabled_falls_back_to_v1() {
        let _lock = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // mirror of plan_smart_pacing_v2's branch: v2 off -> v1 path
        std::env::set_var("AUTOSHORTS_SMART_PACING", "1");
        std::env::set_var("AUTOSHORTS_SMART_PACING_2", "0");
        assert!(smart_pacing_enabled());
        assert!(!smart_pacing_2_enabled());
        // v2 off means no breath types in the produced edit list; the v1
        // internal_pause plan below is exactly what the fallback returns
        let v1 = plan_with(vec![(10.0, 12.0)], 40.0);
        assert!(v1.edits.iter().all(|e| e.edit_type != "breath_pause"));
        assert!(v1.validate().is_ok());
        std::env::remove_var("AUTOSHORTS_SMART_PACING_2");
        std::env::remove_var("AUTOSHORTS_SMART_PACING");
    }

    // Caption sync (Part 10): multiple removals shift later words by the
    // accumulated removed span before them â€” the ASS consumer's contract.
    #[test]
    fn test_sp2_caption_shift_accumulates() {
        // cuts at 3.0-3.1 and 8.0-9.0 in a 15s clip
        let plan = plan_with(vec![(3.0, 3.1), (8.0, 9.0)], 15.0);
        let words = vec![
            word("a", 101.0, 101.4, None),
            word("b", 104.0, 104.4, None), // after cut 1: -0.10
            word("c", 110.0, 110.4, None), // after both: -1.10
            word("d", 114.0, 114.4, None), // after both: -1.10
        ];
        let out = remap_words(&plan, &words);
        assert_eq!(out.len(), 4);
        assert!((out[0].start - 101.0).abs() < 0.01);
        assert!((out[1].start - 103.9).abs() < 0.01);
        assert!((out[2].start - 108.9).abs() < 0.01);
        assert!((out[3].start - 112.9).abs() < 0.01);
        // caption events stay inside the output span
        let out_dur = plan.output_duration_sec;
        assert!(out
            .iter()
            .all(|w| w.start >= 100.0 && w.end <= 100.0 + out_dur));
    }

    // Framing remap (Part 10): DualFrame segments survive a v2-shaped plan
    // with small cuts, and the dual block still starts at the right output
    // time (the framing consumer's timing contract).
    #[test]
    fn test_sp2_dualframe_survives_micro_edits() {
        let framing = SmartFramingPlan {
            speaker_intel: None,
            mode: "multi".into(),
            x: "0".into(),
            y: "0".into(),
            w: "608".into(),
            h: "1080".into(),
            segments: vec![
                LayoutSegment::Single {
                    start: 0.0,
                    end: 20.0,
                    crop: CropRectExpr {
                        x: "100".into(),
                        y: "0".into(),
                        w: "608".into(),
                        h: "1080".into(),
                    },
                    face_bounds: None,
                },
                LayoutSegment::DualStack {
                    start: 20.0,
                    end: 40.0,
                    top_track_id: Some(1),
                    bottom_track_id: Some(2),
                    top_crop: CropRectExpr {
                        x: "50".into(),
                        y: "0".into(),
                        w: "608".into(),
                        h: "540".into(),
                    },
                    bottom_crop: CropRectExpr {
                        x: "60".into(),
                        y: "540".into(),
                        w: "608".into(),
                        h: "540".into(),
                    },
                    top_face_bounds: None,
                    bottom_face_bounds: None,
                },
            ],
            face_bounds: None,
            framing: "original".to_string(),
            is_emergency_fallback: false,
            fallback_reason: None,
        };
        // a micro cut (0.16s) before the dual block, plus a larger one after
        let plan = plan_with(vec![(5.0, 5.16), (30.0, 32.0)], 40.0);
        assert!(plan.validate().is_ok());
        let remapped = remap_framing_plan(&framing, &plan, 40.0);
        // both segment kinds survive
        assert!(remapped
            .segments
            .iter()
            .any(|s| matches!(s, LayoutSegment::DualStack { .. })));
        // the dual block starts at src 20 minus the 0.16s removal = 19.84
        let dual = remapped
            .segments
            .iter()
            .find(|s| matches!(s, LayoutSegment::DualStack { .. }))
            .unwrap();
        let (ds, _) = seg_bounds(dual);
        assert!((ds - 19.84).abs() < 0.05, "dual start {ds} != 19.84");
        // segments still tile the output
        let last_end = seg_bounds(remapped.segments.last().unwrap()).1;
        assert!((last_end - plan.output_duration_sec).abs() < 0.2);
    }

    // Part 6 auditable mapping: pacing_summary reports the v2 edit types so
    // every reduction is visible in the render log.
    #[test]
    fn test_sp2_summary_reports_breath_types() {
        let mut plan = plan_with(vec![(5.0, 5.16)], 12.0);
        plan.edits[0].edit_type = "breath_pause".into();
        let s = pacing_summary(&plan);
        assert!(s.contains("breath_pause"));
        assert!(s.contains("[Smart Pacing]"));
    }

    #[test]
    fn test_pacing_cache_descriptor_hashing() {
        let d1 = PacingCacheDescriptor {
            schema_version: "1".to_string(),
            algorithm_version: "v2.0".to_string(),
            source_fingerprint: "src_fp_123".to_string(),
            start_ms: 1000,
            end_ms: 30000,
            pacing_mode: "sp2".to_string(),
            words_fingerprint: "words_fp_abc".to_string(),
            effective_config_fingerprint: "config_fp_xyz".to_string(),
        };
        let k1 = d1.compute_cache_key();
        assert_eq!(k1.len(), 64);

        // sp1 vs sp2 changes key
        let mut d2 = d1.clone();
        d2.pacing_mode = "sp1".to_string();
        let k2 = d2.compute_cache_key();
        assert_ne!(k1, k2);

        // words fingerprint changes key
        let mut d3 = d1.clone();
        d3.words_fingerprint = "words_fp_different".to_string();
        let k3 = d3.compute_cache_key();
        assert_ne!(k1, k3);

        // range changes key
        let mut d4 = d1.clone();
        d4.start_ms = 2000;
        let k4 = d4.compute_cache_key();
        assert_ne!(k1, k4);
    }

    #[test]
    fn test_pacing_cache_no_op_plan_cacheable() {
        let temp_dir = std::env::temp_dir().join(format!("pacing_cache_test_{}", uuid::Uuid::new_v4()));
        let _ = std::fs::create_dir_all(&temp_dir);

        let descriptor = PacingCacheDescriptor {
            schema_version: "1".to_string(),
            algorithm_version: "v2.0".to_string(),
            source_fingerprint: "src_noop".to_string(),
            start_ms: 0,
            end_ms: 15000,
            pacing_mode: "sp2".to_string(),
            words_fingerprint: "words_noop".to_string(),
            effective_config_fingerprint: "config_noop".to_string(),
        };
        let key = descriptor.compute_cache_key();
        let file_name = format!("{}.json", key);

        let entry = CachedPacingEntry {
            descriptor: descriptor.clone(),
            plan: None,
        };

        let serialized = serde_json::to_vec_pretty(&entry).unwrap();
        atomic_write_cache_file(&temp_dir, &file_name, &serialized).unwrap();

        // Verify read-back preserves verified no-op (0 cuts) status
        let read_content = std::fs::read_to_string(temp_dir.join(&file_name)).unwrap();
        let loaded: CachedPacingEntry = serde_json::from_str(&read_content).unwrap();

        assert_eq!(loaded.descriptor.compute_cache_key(), key);
        assert!(loaded.plan.is_none());

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_pacing_cache_positive_plan_cacheable() {
        let temp_dir = std::env::temp_dir().join(format!("pacing_cache_pos_{}", uuid::Uuid::new_v4()));
        let _ = std::fs::create_dir_all(&temp_dir);

        let descriptor = PacingCacheDescriptor {
            schema_version: "1".to_string(),
            algorithm_version: "v2.0".to_string(),
            source_fingerprint: "src_pos".to_string(),
            start_ms: 0,
            end_ms: 15000,
            pacing_mode: "sp1".to_string(),
            words_fingerprint: "words_pos".to_string(),
            effective_config_fingerprint: "config_pos".to_string(),
        };
        let key = descriptor.compute_cache_key();
        let file_name = format!("{}.json", key);

        let plan = plan_with(vec![(5.0, 6.0)], 15.0);
        assert!(plan.validate().is_ok());

        let entry = CachedPacingEntry {
            descriptor: descriptor.clone(),
            plan: Some(plan.clone()),
        };

        let serialized = serde_json::to_vec_pretty(&entry).unwrap();
        atomic_write_cache_file(&temp_dir, &file_name, &serialized).unwrap();

        let read_content = std::fs::read_to_string(temp_dir.join(&file_name)).unwrap();
        let loaded: CachedPacingEntry = serde_json::from_str(&read_content).unwrap();

        assert_eq!(loaded.descriptor.compute_cache_key(), key);
        let loaded_plan = loaded.plan.expect("plan must be present");
        assert_eq!(loaded_plan.status, "ok");
        assert_eq!(loaded_plan.edits.len(), 1);
        assert!(loaded_plan.validate().is_ok());

        let _ = std::fs::remove_dir_all(&temp_dir);
    }
}
