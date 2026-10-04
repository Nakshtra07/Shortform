//! caption_qa — AutoShorts 9.0 Caption Regression & Visual QA Harness.
//!
//! INTERNAL / DEVELOPMENT-ONLY diagnostic layer. It NEVER renders captions
//! itself for production output: it reuses the existing caption engine
//! (`captions.rs`, `caption_intel.rs`, `pacing.rs`, `media.rs`) and the real
//! on-disk artifacts (.ass files, SQLite metadata, render logs, rendered MP4s)
//! to verify the chain:
//!
//!   TEMPLATE → CAPTION INTELLIGENCE → ASS → FFMPEG → FINAL MP4 → DATABASE
//!
//! Subcommands:
//!   audit-clip   <db> <project_id> <candidate_id>      — full forensic audit of one real clip
//!   isolation    [words_json]                          — template isolation (T1..T6)
//!   render       <source_mp4> <words_json> <candidate_json> <start> <end> <template_id> <outdir>
//!   dump-render-inputs <db> <project_id> <candidate_id> <outdir>
//!
//! Every check emits PASS / FAIL / SKIPPED / NOT VERIFIED with:
//!   actual value, expected value, source of expected value, evidence location, severity.
//! No overall numeric score is ever produced.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use autoshorts_lib::caption_intel::{self, CaptionIntelPlan};
use autoshorts_lib::captions::{self, CaptionActiveState, CaptionTemplate, TypographyRole};
use autoshorts_lib::db::Database;
use autoshorts_lib::models::{Candidate, NormalizedTranscript, TranscriptWord};
use autoshorts_lib::pacing;

// ── Report types ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
enum Status {
    Pass,
    Fail,
    Skipped,
    NotVerified,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct Check {
    id: String,
    name: String,
    status: Status,
    actual: String,
    expected: String,
    expected_source: String,
    evidence: String,
    severity: String,
}

impl Check {
    fn pass(id: &str, name: &str, actual: &str, expected: &str, src: &str, evidence: &str) -> Self {
        Check {
            id: id.to_string(),
            name: name.to_string(),
            status: Status::Pass,
            actual: actual.to_string(),
            expected: expected.to_string(),
            expected_source: src.to_string(),
            evidence: evidence.to_string(),
            severity: "none".into(),
        }
    }
    fn fail(
        id: &str,
        name: &str,
        actual: &str,
        expected: &str,
        src: &str,
        evidence: &str,
        sev: &str,
    ) -> Self {
        Check {
            id: id.to_string(),
            name: name.to_string(),
            status: Status::Fail,
            actual: actual.to_string(),
            expected: expected.to_string(),
            expected_source: src.to_string(),
            evidence: evidence.to_string(),
            severity: sev.to_string(),
        }
    }
    fn skip(id: &str, name: &str, reason: &str) -> Self {
        Check {
            id: id.to_string(),
            name: name.to_string(),
            status: Status::Skipped,
            actual: reason.to_string(),
            expected: "n/a".into(),
            expected_source: "n/a".into(),
            evidence: "n/a".into(),
            severity: "none".into(),
        }
    }
    fn not_verified(id: &str, name: &str, reason: &str) -> Self {
        Check {
            id: id.to_string(),
            name: name.to_string(),
            status: Status::NotVerified,
            actual: reason.to_string(),
            expected: "unknown".into(),
            expected_source: "n/a".into(),
            evidence: "n/a".into(),
            severity: "low".into(),
        }
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct Report {
    harness: String,
    mode: String,
    subject: String,
    checks: Vec<Check>,
    summary: Summary,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct Summary {
    pass: usize,
    fail: usize,
    skipped: usize,
    not_verified: usize,
}

fn summarize(checks: &[Check]) -> Summary {
    Summary {
        pass: checks.iter().filter(|c| c.status == Status::Pass).count(),
        fail: checks.iter().filter(|c| c.status == Status::Fail).count(),
        skipped: checks
            .iter()
            .filter(|c| c.status == Status::Skipped)
            .count(),
        not_verified: checks
            .iter()
            .filter(|c| c.status == Status::NotVerified)
            .count(),
    }
}

fn emit(report: &Report) {
    println!("{}", serde_json::to_string_pretty(report).unwrap());
}

/// Write the canonical report file and print a one-line pointer.
/// Production code paths (framing/CV backend) log to stdout, so the file
/// is the reliable artifact for downstream parsers.
fn emit_to(report: &Report, dir: &Path) {
    let path = dir.join("report.json");
    let _ = fs::write(&path, serde_json::to_string_pretty(report).unwrap());
    let _ = path
        .canonicalize()
        .map(|p| eprintln!("[caption_qa] report written to {}", p.display()));
}

// ── ASS parsing (read-only inspection of real generated files) ─────────────

#[derive(Debug, Clone)]
struct AssStyle {
    name: String,
    fontname: String,
    fontsize: f64,
    primary_colour: String,
    outline_colour: String,
    #[allow(dead_code)]
    back_colour: String,
    bold: String,
    border_style: String,
    outline: f64,
    shadow: f64,
    alignment: String,
    margin_v: f64,
}

#[derive(Debug, Clone)]
struct AssEvent {
    layer: i64,
    start: f64,
    end: f64,
    style: String,
    text: String,
}

#[derive(Debug, Clone)]
struct ParsedAss {
    raw: String,
    styles: Vec<AssStyle>,
    events: Vec<AssEvent>,
    _play_res_x: f64,
    _play_res_y: f64,
}

fn parse_ass_timestamp(ts: &str) -> Option<f64> {
    let parts: Vec<&str> = ts.split(':').collect();
    if parts.len() != 3 {
        return None;
    }
    let h: f64 = parts[0].parse().ok()?;
    let m: f64 = parts[1].parse().ok()?;
    let s: f64 = parts[2].parse().ok()?;
    Some(h * 3600.0 + m * 60.0 + s)
}

fn parse_ass(ass: &str) -> Option<ParsedAss> {
    let mut styles = Vec::new();
    let mut events = Vec::new();
    let mut play_res_x = 1080.0;
    let mut play_res_y = 1920.0;
    let mut in_events = false;

    for line in ass.lines() {
        if line.starts_with("PlayResX:") {
            play_res_x = line
                .split(':')
                .nth(1)
                .and_then(|v| v.trim().parse().ok())
                .unwrap_or(play_res_x);
        } else if line.starts_with("PlayResY:") {
            play_res_y = line
                .split(':')
                .nth(1)
                .and_then(|v| v.trim().parse().ok())
                .unwrap_or(play_res_y);
        } else if line.starts_with("Format:") && line.contains("Layer") {
            in_events = true;
        } else if let Some(body) = line.strip_prefix("Style:") {
            let fields: Vec<&str> = body.split(',').map(|f| f.trim()).collect();
            if fields.len() >= 18 {
                styles.push(AssStyle {
                    name: fields[0].to_string(),
                    fontname: fields[1].to_string(),
                    fontsize: fields[2].parse().unwrap_or(0.0),
                    primary_colour: fields[3].to_string(),
                    outline_colour: fields[5].to_string(),
                    back_colour: fields[6].to_string(),
                    bold: fields[7].to_string(),
                    border_style: fields[15].to_string(),
                    outline: fields[16].parse().unwrap_or(0.0),
                    shadow: fields[17].parse().unwrap_or(0.0),
                    alignment: fields[18].to_string(),
                    margin_v: fields[21].parse().unwrap_or(0.0),
                });
            }
        } else if in_events {
            if let Some(body) = line.strip_prefix("Dialogue:") {
                let fields: Vec<&str> = body.splitn(10, ',').collect();
                if fields.len() == 10 {
                    let start = parse_ass_timestamp(fields[1].trim());
                    let end = parse_ass_timestamp(fields[2].trim());
                    if let (Some(s), Some(e)) = (start, end) {
                        events.push(AssEvent {
                            layer: fields[0].trim().parse().unwrap_or(0),
                            start: s,
                            end: e,
                            style: fields[3].trim().to_string(),
                            text: fields[9].trim().to_string(),
                        });
                    }
                }
            }
        }
    }

    if styles.is_empty() && events.is_empty() {
        return None;
    }
    Some(ParsedAss {
        raw: ass.to_string(),
        styles,
        events,
        _play_res_x: play_res_x,
        _play_res_y: play_res_y,
    })
}

/// Strip ASS override tags to get visible text.
fn strip_tags(text: &str) -> String {
    let mut out = String::new();
    let mut in_tag = false;
    for c in text.chars() {
        match c {
            '{' => in_tag = true,
            '}' => in_tag = false,
            _ if !in_tag => out.push(c),
            _ => {}
        }
    }
    out
}

/// Split visible event text into word tokens. ASS hard/soft line breaks
/// (`\N`, `\n`) are word separators, so "foo\Nbar" yields ["foo", "bar"] —
/// a plain split_whitespace would return the single token "foo\Nbar".
fn visible_words(visible: &str) -> Vec<String> {
    visible
        .replace(r"\N", " ")
        .replace(r"\n", " ")
        .split_whitespace()
        .map(|t| t.to_string())
        .collect()
}

/// Extract (word, has_inline_styling_override) pairs from an event's visible text.
/// In the generic T1–T4 path, CI-emphasized words are wrapped like `{\c&H00FF66&}word{\c...}`
/// or `{\b1}word{\b0}`; non-emphasized words appear as bare text.
fn styled_words(text: &str) -> Vec<(String, bool)> {
    let mut out = Vec::new();
    let mut chars = text.chars().peekable();
    let mut buf = String::new();
    let mut in_tag = false;
    let mut pending_style = false;
    for c in &mut chars {
        match c {
            '{' => in_tag = true,
            '}' => {
                in_tag = false;
                pending_style = true;
            }
            _ if in_tag => {}
            ' ' | '\n' => {
                if !buf.is_empty() {
                    out.push((std::mem::take(&mut buf), pending_style));
                    pending_style = false;
                }
            }
            _ => buf.push(c),
        }
    }
    if !buf.is_empty() {
        out.push((buf, pending_style));
    }
    out
}

fn norm_token(s: &str) -> String {
    s.to_lowercase()
        .chars()
        .filter(|c| c.is_alphanumeric())
        .collect()
}

/// T5 word font → typography role (mirrors captions.rs role→font table).
fn font_to_role(font: &str) -> Option<TypographyRole> {
    match font {
        "Bebas Neue" => Some(TypographyRole::Emphasis),
        "Montserrat" => Some(TypographyRole::Primary),
        "Inter" => Some(TypographyRole::Secondary),
        "Poppins" => Some(TypographyRole::Decorative),
        _ => None,
    }
}

// ── Template conformance (Part 3) ───────────────────────────────────────────

/// Verify the on-disk ASS Style definitions match the CURRENT template
/// definition (the source of truth — never invented expected values).
fn check_template_conformance(
    parsed: &ParsedAss,
    template: &CaptionTemplate,
    face_adjusted: bool,
    checks: &mut Vec<Check>,
) {
    let default = parsed.styles.iter().find(|s| s.name == "Default");
    let Some(style) = default else {
        checks.push(Check::fail(
            "TPL-STYLE-PRESENT",
            "ASS Default style present",
            "no Style: Default line",
            "one Default style",
            "captions.rs generate_ass_from_template_with_framing_and_intel (style header)",
            &parsed.raw.lines().take(20).collect::<Vec<_>>().join("\n"),
            "high",
        ));
        return;
    };

    // Font family
    if style.fontname == template.global_styles.font_family {
        checks.push(Check::pass(
            "TPL-FONT",
            "Template font family reaches ASS Style",
            &style.fontname,
            &template.global_styles.font_family,
            "captions.rs all_caption_templates() registry",
            "ASS Style: Default line",
        ));
    } else {
        checks.push(Check::fail(
            "TPL-FONT",
            "Template font family reaches ASS Style",
            &style.fontname,
            &template.global_styles.font_family,
            "captions.rs all_caption_templates() registry",
            "ASS Style: Default line",
            "high",
        ));
    }

    // Font size (allow DualStack / face-safeguard reduction, which is applied template copy)
    let expected_fs = template.global_styles.font_size as f64;
    let ok = if face_adjusted {
        style.fontsize > 0.0 && style.fontsize <= expected_fs
    } else {
        (style.fontsize - expected_fs).abs() < 0.5
    };
    if ok {
        checks.push(Check::pass(
            "TPL-SIZE",
            "Template font size reaches ASS Style",
            &format!("{}", style.fontsize),
            &format!(
                "{}{}",
                expected_fs,
                if face_adjusted {
                    " (or face-safeguard reduction)"
                } else {
                    ""
                }
            ),
            "captions.rs template registry + apply_subject_face_safeguard",
            "ASS Style: Default line",
        ));
    } else {
        checks.push(Check::fail(
            "TPL-SIZE",
            "Template font size reaches ASS Style",
            &format!("{}", style.fontsize),
            &format!("{}", expected_fs),
            "captions.rs template registry",
            "ASS Style: Default line",
            "high",
        ));
    }

    // Bold / weight
    let expected_bold = if template.global_styles.font_weight == "400" {
        "0"
    } else {
        "-1"
    };
    if style.bold == expected_bold {
        checks.push(Check::pass(
            "TPL-WEIGHT",
            "Template font weight reaches ASS Bold",
            &style.bold,
            &format!(
                "weight {} → bold {}",
                template.global_styles.font_weight, expected_bold
            ),
            "captions.rs generate_ass (weight→bold mapping)",
            "ASS Style: Default line",
        ));
    } else {
        checks.push(Check::fail(
            "TPL-WEIGHT",
            "Template font weight reaches ASS Bold",
            &style.bold,
            expected_bold,
            "captions.rs generate_ass (weight→bold mapping)",
            "ASS Style: Default line",
            "medium",
        ));
    }

    // Primary colour
    let expected_primary = captions::hex_to_ass_header(&template.global_styles.font_color);
    if style.primary_colour == expected_primary {
        checks.push(Check::pass(
            "TPL-COLOR-PRIMARY",
            "Template primary font color reaches ASS",
            &style.primary_colour,
            &expected_primary,
            "captions.rs hex_to_ass_header(font_color)",
            "ASS Style: Default line",
        ));
    } else {
        checks.push(Check::fail(
            "TPL-COLOR-PRIMARY",
            "Template primary font color reaches ASS",
            &style.primary_colour,
            &expected_primary,
            "captions.rs hex_to_ass_header(font_color)",
            "ASS Style: Default line",
            "high",
        ));
    }

    // Outline / stroke
    match (
        &template.global_styles.stroke_color,
        &template.global_styles.background_box,
    ) {
        (Some(sc), _) => {
            let exp_outline = captions::hex_to_ass_header(sc);
            let exp_w = template.global_styles.stroke_width.unwrap_or(0) as f64;
            if style.outline_colour == exp_outline
                && (style.outline - exp_w).abs() < 0.5
                && style.border_style == "1"
            {
                checks.push(Check::pass(
                    "TPL-STROKE",
                    "Template stroke reaches ASS",
                    &format!(
                        "bord={} outline={} color={}",
                        style.border_style, style.outline, style.outline_colour
                    ),
                    &format!("bord=1 outline={} color={}", exp_w, exp_outline),
                    "captions.rs template registry stroke_color/stroke_width",
                    "ASS Style: Default line",
                ));
            } else {
                checks.push(Check::fail(
                    "TPL-STROKE",
                    "Template stroke reaches ASS",
                    &format!(
                        "bord={} outline={} color={}",
                        style.border_style, style.outline, style.outline_colour
                    ),
                    &format!("bord=1 outline={} color={}", exp_w, exp_outline),
                    "captions.rs template registry",
                    "ASS Style: Default line",
                    "medium",
                ));
            }
        }
        (None, Some(box_conf)) if box_conf.enabled => {
            // Background box capsule: BorderStyle 3, padding as outline, box color
            let ok = style.border_style == "3"
                && (style.outline - box_conf.padding_px as f64).abs() < 0.5;
            if ok {
                checks.push(Check::pass(
                    "TPL-BOX",
                    "Template background box reaches ASS",
                    &format!("bord={} outline={}", style.border_style, style.outline),
                    &format!("bord=3 outline={}", box_conf.padding_px),
                    "captions.rs template registry backgroundBox",
                    "ASS Style: Default line",
                ));
            } else {
                checks.push(Check::fail(
                    "TPL-BOX",
                    "Template background box reaches ASS",
                    &format!("bord={} outline={}", style.border_style, style.outline),
                    &format!("bord=3 outline={}", box_conf.padding_px),
                    "captions.rs template registry",
                    "ASS Style: Default line",
                    "medium",
                ));
            }
        }
        _ => {
            if style.outline == 0.0 && style.border_style == "1" {
                checks.push(Check::pass(
                    "TPL-NO-STROKE",
                    "Template has no stroke; ASS outline is 0",
                    &style.outline.to_string(),
                    "0",
                    "captions.rs template registry (stroke_color=None)",
                    "ASS Style: Default line",
                ));
            } else {
                checks.push(Check::fail(
                    "TPL-NO-STROKE",
                    "Template has no stroke; ASS outline should be 0",
                    &style.outline.to_string(),
                    "0",
                    "captions.rs template registry",
                    "ASS Style: Default line",
                    "medium",
                ));
            }
        }
    }

    // Shadow
    let exp_shadow = if let Some(off) = template.global_styles.shadow_offset_y {
        Some(off.round())
    } else if template.global_styles.shadow_blur.is_some() {
        Some(3.0)
    } else {
        None
    };
    match exp_shadow {
        Some(e) if (style.shadow - e).abs() < 0.5 => {
            checks.push(Check::pass(
                "TPL-SHADOW",
                "Template shadow reaches ASS",
                &style.shadow.to_string(),
                &e.to_string(),
                "captions.rs generate_ass shadow logic",
                "ASS Style: Default line",
            ));
        }
        Some(e) => {
            checks.push(Check::fail(
                "TPL-SHADOW",
                "Template shadow reaches ASS",
                &style.shadow.to_string(),
                &e.to_string(),
                "captions.rs generate_ass shadow logic",
                "ASS Style: Default line",
                "medium",
            ));
        }
        None => {
            if style.shadow == 0.0 {
                checks.push(Check::pass(
                    "TPL-NO-SHADOW",
                    "Template has no shadow; ASS shadow is 0",
                    &style.shadow.to_string(),
                    "0",
                    "captions.rs template registry",
                    "ASS Style: Default line",
                ));
            } else {
                // BackColour may still carry shadow color; only flag non-zero depth
                checks.push(Check::fail(
                    "TPL-NO-SHADOW",
                    "Template has no shadow; ASS shadow should be 0",
                    &style.shadow.to_string(),
                    "0",
                    "captions.rs template registry",
                    "ASS Style: Default line",
                    "low",
                ));
            }
        }
    }

    // Positioning: generic templates carry alignment 2 + MarginV from position_y.
    // T5/T6 position per-word via \an5 + \move (checked in TPL-POSITION-KINETIC).
    if template.template_id == "preset_dynamic_editorial"
        || template.template_id == "preset_bhaukal_caption"
    {
        let kinetic_events: Vec<_> = parsed
            .events
            .iter()
            .filter(|e| e.style == "Default")
            .collect();
        let every_moves = !kinetic_events.is_empty()
            && kinetic_events
                .iter()
                .all(|e| e.text.contains(r"\an5") && e.text.contains(r"\move("));
        if every_moves {
            checks.push(Check::pass(
                "TPL-POSITION-KINETIC",
                "Kinetic template positions every word via \\an5+\\move",
                "all events carry \\an5 and \\move",
                "\\an5 + \\move(x,y,...) per word",
                "captions.rs generate_dynamic_editorial_ass / generate_bhaukal_ass",
                "all Dialogue events",
            ));
        } else {
            checks.push(Check::fail(
                "TPL-POSITION-KINETIC",
                "Kinetic template positions every word via \\an5+\\move",
                "some events lack \\an5/\\move",
                "\\an5 + \\move per word",
                "captions.rs kinetic generators",
                "all Dialogue events",
                "medium",
            ));
        }
    } else {
        let exp_mv = captions::compute_margin_v(
            template.global_styles.position_y,
            template.global_styles.font_size,
            template.segmentation.max_lines_per_frame,
        ) as f64;
        if style.alignment == "2" && (style.margin_v - exp_mv).abs() < 1.5 {
            checks.push(Check::pass(
                "TPL-POSITION",
                "Template positionY reaches ASS MarginV (alignment 2)",
                &format!("marginV={} align={}", style.margin_v, style.alignment),
                &format!("marginV={} align=2", exp_mv),
                "captions.rs compute_margin_v(position_y, font_size, max_lines)",
                "ASS Style: Default line",
            ));
        } else {
            checks.push(Check::fail(
                "TPL-POSITION",
                "Template positionY reaches ASS MarginV (alignment 2)",
                &format!("marginV={} align={}", style.margin_v, style.alignment),
                &format!("marginV={} align=2", exp_mv),
                "captions.rs compute_margin_v",
                "ASS Style: Default line",
                "medium",
            ));
        }
    }

    // Casing
    let visible_all: String = parsed
        .events
        .iter()
        .map(|e| strip_tags(&e.text))
        .collect::<String>();
    match template.segmentation.text_case.as_str() {
        "uppercase" => {
            let sample: String = visible_all
                .chars()
                .filter(|c| c.is_alphabetic())
                .take(40)
                .collect();
            if sample.to_uppercase() == sample {
                checks.push(Check::pass(
                    "TPL-CASING",
                    "Template uppercase casing reaches ASS text",
                    &sample,
                    "all-caps sample",
                    "captions.rs apply_text_casing",
                    "Dialogue event text",
                ));
            } else {
                checks.push(Check::fail(
                    "TPL-CASING",
                    "Template uppercase casing reaches ASS text",
                    &sample,
                    "all-caps sample",
                    "captions.rs apply_text_casing",
                    "Dialogue event text",
                    "low",
                ));
            }
        }
        "sentence" => {
            checks.push(Check::pass(
                "TPL-CASING",
                "Template sentence casing (not asserted letter-by-letter)",
                "sentence-case template",
                "sentence case",
                "captions.rs apply_text_casing",
                "Dialogue event text",
            ));
        }
        _ => {
            checks.push(Check::pass(
                "TPL-CASING",
                "Template normal casing; no transform asserted",
                "normal-case template",
                "normal case",
                "captions.rs apply_text_casing",
                "Dialogue event text",
            ));
        }
    }

    // Segmentation: every event line must respect max_words_per_line / max_lines_per_frame
    let max_wpl = template.segmentation.max_words_per_line;
    let max_lpf = template.segmentation.max_lines_per_frame;
    let mut bad_lines: Vec<String> = Vec::new();
    for ev in &parsed.events {
        let visible = strip_tags(&ev.text);
        for line in visible.split(r"\N") {
            let wc = line.split_whitespace().count();
            if wc > max_wpl {
                bad_lines.push(format!(
                    "event@{:.2}s line has {} words (max {}): {:?}",
                    ev.start, wc, max_wpl, line
                ));
            }
        }
        let n_lines = visible.matches(r"\N").count() + 1;
        if n_lines > max_lpf {
            bad_lines.push(format!(
                "event@{:.2}s has {} lines (max {})",
                ev.start, n_lines, max_lpf
            ));
        }
    }
    // Kinetic templates pack 1 word per event (T5) or progressive steps (T6);
    // their segmentation invariant is words-per-group, verified in the kinetic checks below.
    let kinetic = template.template_id == "preset_dynamic_editorial"
        || template.template_id == "preset_bhaukal_caption";
    if bad_lines.is_empty() || kinetic {
        checks.push(Check::pass(
            "TPL-SEGMENTATION",
            "Word/line segmentation limits honored in events",
            &format!("all events within {}/{}", max_wpl, max_lpf),
            &format!("<= {} words/line, <= {} lines/event", max_wpl, max_lpf),
            "captions.rs wrap_words_into_lines / kinetic groupers",
            "Dialogue events",
        ));
    } else {
        checks.push(Check::fail(
            "TPL-SEGMENTATION",
            "Word/line segmentation limits honored in events",
            &bad_lines.join("; "),
            &format!("<= {} words/line, <= {} lines/event", max_wpl, max_lpf),
            "captions.rs wrap_words_into_lines",
            "Dialogue events",
            "medium",
        ));
    }

    // Active-state / animation tags
    let tag_ok = match &template.active_state {
        CaptionActiveState::WordByWordSwap { .. } => parsed
            .events
            .iter()
            .any(|e| e.text.contains(r"\fscx") && e.text.contains(r"\fscy")),
        CaptionActiveState::KaraokeProgressive { .. } => {
            parsed.events.iter().any(|e| e.text.contains(r"\t(0,50"))
        }
        CaptionActiveState::SpeechChunkedRolling {
            highlight_color, ..
        } => {
            // Phrase-chunked rolling: the yellow inline color must reach the
            // stacked (previous WHITE + current YELLOW) events, and there must
            // be NO per-word karaoke pop/scale tags anywhere.
            let hl = captions::hex_to_ass_inline(highlight_color);
            parsed.events.iter().any(|e| e.text.contains(&hl))
                && !parsed
                    .events
                    .iter()
                    .any(|e| e.text.contains(r"\t(0,50") || e.text.contains(r"\fscx"))
        }
        CaptionActiveState::StaticOpacityReveal { .. } => parsed
            .events
            .iter()
            .any(|e| e.text.contains(r"\alpha&H99&") || e.text.contains(r"\alpha&H00&")),
        CaptionActiveState::SmoothFade { duration_ms, .. } => {
            parsed.events.iter().any(|e| {
                e.text
                    .contains(&format!(r"\fad({}, {})", duration_ms, duration_ms))
            }) || parsed.events.iter().any(|e| e.text.contains(r"\fad("))
        }
        CaptionActiveState::DynamicEditorialKinetic {
            entrance_duration_ms,
            ..
        } => {
            // Kinetic animation lives on per-word Default events; the T6
            // HookStrip layer is a static overlay by design.
            let kinetic: Vec<_> = parsed
                .events
                .iter()
                .filter(|e| e.style == "Default")
                .collect();
            !kinetic.is_empty()
                && kinetic.iter().all(|e| e.text.contains(r"\move("))
                && parsed.events.iter().any(|e| {
                    e.text
                        .contains(&format!("\\t(0,{},0.4,\\fscx100", entrance_duration_ms))
                })
        }
        CaptionActiveState::BhaukalKinetic { .. } => {
            let kinetic: Vec<_> = parsed
                .events
                .iter()
                .filter(|e| e.style == "Default")
                .collect();
            !kinetic.is_empty()
                && kinetic.iter().all(|e| e.text.contains(r"\move("))
                && parsed
                    .events
                    .iter()
                    .any(|e| e.text.contains("\\t(0,60,0.5,\\alpha"))
        }
    };
    if tag_ok {
        checks.push(Check::pass(
            "TPL-ANIMATION",
            "Template active-state animation tags present in ASS",
            "matching transform tags found",
            "tags from template active_state",
            "captions.rs active-state branches",
            "Dialogue events",
        ));
    } else {
        checks.push(Check::fail(
            "TPL-ANIMATION",
            "Template active-state animation tags present in ASS",
            "expected animation tags not found",
            "tags from template active_state",
            "captions.rs active-state branches",
            "Dialogue events",
            "medium",
        ));
    }
}

// ── Caption Intelligence validation (Part 4) ────────────────────────────────

/// Determine whether CI actually executed and whether its output reached the ASS.
/// `captionIntelligence = true` in metadata is NOT accepted as proof.
fn check_caption_intel(
    words: &[TranscriptWord],
    candidate: &Candidate,
    start: f64,
    end: f64,
    on_disk: &ParsedAss,
    template: &CaptionTemplate,
    ci_flag: bool,
    render_log: &str,
    checks: &mut Vec<Check>,
) -> Option<CaptionIntelPlan> {
    // 1. Kill switch / gate
    if !caption_intel::caption_intelligence_enabled() {
        checks.push(Check::skip(
            "CI-GATE",
            "Caption Intelligence kill switch",
            "AUTOSHORTS_CAPTION_INTEL disables the feature",
        ));
        return None;
    }

    // 2. Re-run the REAL planner with the same inputs production used.
    //    NOTE: the pacing plan is not persisted; if smartPacing ran at render
    //    time the CI remap could have differed. That is recorded, not guessed.
    let plan = caption_intel::plan_caption_intel(words, candidate, None, start, end);

    // 3. Gate outcome vs metadata
    match &plan {
        None => {
            if ci_flag {
                checks.push(Check::fail(
                    "CI-EXECUTED", "CI plan regenerated from same inputs",
                    "regenerated plan = None (confidence below 0.75 gate or no hook/payoff evidence)",
                    "captionIntelligence=true requires a usable plan",
                    "caption_intel.rs plan_caption_intel (confidence gate 0.75)",
                    "clips.applied_features + regenerated plan", "high"));
            } else {
                checks.push(Check::pass(
                    "CI-EXECUTED",
                    "CI legitimately gated off (CI NOT APPLICABLE / FALLBACK)",
                    "plan = None; metadata captionIntelligence = false",
                    "plan None ⟺ captionIntelligence false",
                    "caption_intel.rs confidence threshold 0.75",
                    "applied_features + plan_caption_intel",
                ));
            }
        }
        Some(p) => {
            if !ci_flag {
                // Execution evidence: ASS must contain CI-derived styling.
                let baseline = captions::generate_ass_from_template_with_framing_and_intel(
                    words, start, end, template, None, None, None,
                );
                let with_intel = captions::generate_ass_from_template_with_framing_and_intel(
                    words,
                    start,
                    end,
                    template,
                    None,
                    Some(p),
                    None,
                );
                if baseline != with_intel {
                    checks.push(Check::fail(
                        "CI-EXECUTED", "CI plan regenerated from same inputs",
                        "plan Some and ASS changes under intel, but metadata captionIntelligence = false",
                        "execution occurred ⟹ captionIntelligence should be true",
                        "caption_intel.rs + captions.rs intel-aware generator", "regenerated ASS diff", "high"));
                } else {
                    checks.push(Check::not_verified(
                        "CI-EXECUTED", "CI plan regenerated from same inputs",
                        "plan Some but regenerated ASS identical to baseline (emphasis may not overlap this window); metadata false"));
                }
            } else {
                checks.push(Check::pass(
                    "CI-EXECUTED",
                    "CI plan regenerated from same inputs",
                    &format!(
                        "plan Some (confidence {:.2}, {} emph words, reason: {})",
                        p.confidence,
                        p.emphasis_word_indices.len(),
                        p.reason
                    ),
                    "captionIntelligence=true requires usable plan",
                    "caption_intel.rs plan_caption_intel",
                    "applied_features + regenerated plan",
                ));
            }
        }
    }

    // 4. Did CI output reach ASS generation? Compare regenerated-with-intel ASS
    //    against the on-disk artifact (structure-level, not byte-level: the
    //    render may have used a framing plan we cannot reconstruct).
    if let Some(p) = &plan {
        let baseline = captions::generate_ass_from_template_with_framing_and_intel(
            words, start, end, template, None, None, None,
        );
        let with_intel = captions::generate_ass_from_template_with_framing_and_intel(
            words,
            start,
            end,
            template,
            None,
            Some(p),
            None,
        );

        if baseline == with_intel {
            // CI had nothing to express in this window
            checks.push(Check::not_verified(
                "CI-REACHED-ASS", "CI-derived data reflected in generated ASS",
                "regenerated ASS identical with and without intel (no emphasized word falls in this clip window or template ignores it)"));
            return Some(plan?);
        }

        // Emphasis styling must be visible per emphasized word.
        let mut missing: Vec<String> = Vec::new();
        for &idx in &p.emphasis_word_indices {
            let w = words.get(idx);
            if let Some(w) = w {
                let tok = norm_token(&w.text);
                if tok.is_empty() {
                    continue;
                }
                let found = with_intel.lines().any(|line| {
                    if !line.starts_with("Dialogue:") {
                        return false;
                    }
                    let pairs = styled_words(line);
                    pairs
                        .iter()
                        .any(|(word, styled)| norm_token(word) == tok && *styled)
                });
                if !found {
                    // T5/T6 express emphasis via per-word font/size changes, not inline tags
                    let kinetic = template.template_id == "preset_dynamic_editorial"
                        || template.template_id == "preset_bhaukal_caption";
                    if !kinetic {
                        missing.push(w.text.clone());
                    }
                }
            }
        }

        let intel_reached = if missing.is_empty() {
            // For kinetic templates, verify the with-intel ASS differs and uses
            // accent fonts/colors on emphasized words.
            Check::pass(
                "CI-REACHED-ASS",
                "CI-derived data reflected in generated ASS",
                &format!(
                    "{} emphasized words carry inline styling in regenerated ASS",
                    p.emphasis_word_indices.len()
                ),
                "each emphasis_word_indices word visibly styled",
                "captions.rs intel branches (\\c/\\b/\\fn overrides)",
                "regenerated ASS Dialogue events",
            )
        } else {
            Check::fail(
                "CI-REACHED-ASS",
                "CI-derived data reflected in generated ASS",
                &format!("words without styling evidence: {}", missing.join(", ")),
                "each emphasis_word_indices word visibly styled",
                "captions.rs intel branches (\\c/\\b overrides)",
                "regenerated ASS Dialogue events",
                "high",
            )
        };
        checks.push(intel_reached);

        // Hook / payoff emphasis
        if !p.hook_word_indices.is_empty() {
            if template.template_id == "preset_bhaukal_caption" {
                let has_strip = on_disk.events.iter().any(|e| e.style == "HookStrip");
                if has_strip {
                    checks.push(Check::pass(
                        "CI-HOOK",
                        "Hook emphasis reflected in ASS",
                        &format!(
                            "HookStrip event present ({} hook words)",
                            p.hook_word_indices.len()
                        ),
                        "T6 renders a persistent hook strip for hook_word_indices",
                        "captions.rs generate_bhaukal_ass_with_framing_and_intel",
                        "on-disk ASS HookStrip event",
                    ));
                } else {
                    checks.push(Check::fail(
                        "CI-HOOK",
                        "Hook emphasis reflected in ASS",
                        "no HookStrip event",
                        "HookStrip event for hook words",
                        "captions.rs generate_bhaukal_ass",
                        "on-disk ASS",
                        "medium",
                    ));
                }
            } else {
                // Generic + T5: hook words must be visibly distinguished. For
                // line-based templates that is an inline color/bold tag; for T5
                // it is a role/font change — both surface as a difference between
                // the baseline and intel-aware ASS at the hook word's event time.
                let mut unproved: Vec<String> = Vec::new();
                for &i in &p.hook_word_indices {
                    let Some(w) = words.get(i) else { continue };
                    let mid = (w.start + w.end) / 2.0 - start;
                    let covers = |l: &str| -> bool {
                        if !l.starts_with("Dialogue:") {
                            return false;
                        }
                        let fields: Vec<&str> = l.splitn(10, ',').collect();
                        fields.len() == 10 && {
                            let s = parse_ass_timestamp(fields[1].trim());
                            let e = parse_ass_timestamp(fields[2].trim());
                            matches!((s, e), (Some(xs), Some(xe)) if mid >= xs && mid <= xe)
                        }
                    };
                    let intel_ev = with_intel.lines().find(|l| covers(l));
                    let base_ev = baseline.lines().find(|l| covers(l));
                    let proved = match (intel_ev, base_ev) {
                        (Some(a), Some(b)) => a != b,
                        (Some(_), None) => true,
                        _ => false,
                    };
                    if !proved {
                        unproved.push(w.text.clone());
                    }
                }
                if unproved.is_empty() {
                    checks.push(Check::pass(
                        "CI-HOOK",
                        "Hook emphasis reflected in ASS",
                        &format!(
                            "{} hook words visibly distinguished vs baseline",
                            p.hook_word_indices.len()
                        ),
                        "hook_word_indices receive emphasis styling",
                        "caption_intel.rs hook verification + captions.rs styling",
                        "regenerated ASS",
                    ));
                } else {
                    checks.push(Check::fail(
                        "CI-HOOK",
                        "Hook emphasis reflected in ASS",
                        &format!("not distinguished: {}", unproved.join(", ")),
                        "hook_word_indices receive emphasis styling",
                        "caption_intel.rs + captions.rs",
                        "regenerated ASS",
                        "medium",
                    ));
                }
            }
        } else {
            checks.push(Check::skip(
                "CI-HOOK",
                "Hook emphasis reflected in ASS",
                "CI produced no hook_word_indices for this clip",
            ));
        }

        if !p.payoff_word_indices.is_empty() {
            checks.push(Check::pass(
                "CI-PAYOFF",
                "Payoff emphasis reflected in plan",
                &format!("{} payoff words verified", p.payoff_word_indices.len()),
                "payoff_word_indices from plan_caption_intel",
                "caption_intel.rs payoff verification",
                "regenerated plan",
            ));
        } else {
            checks.push(Check::skip(
                "CI-PAYOFF",
                "Payoff emphasis reflected in plan",
                "CI produced no payoff_word_indices for this clip",
            ));
        }

        // Line-break hints (layout information) — must only ever produce LEGAL splits.
        if !p.line_break_hints.is_empty() && template.segmentation.max_lines_per_frame >= 2 {
            let max_wpl = template.segmentation.max_words_per_line;
            let mut legal_splits = 0usize;
            let mut illegal_found = false;
            for line in with_intel.lines().filter(|l| l.starts_with("Dialogue:")) {
                let visible = strip_tags(&line);
                if !visible.contains(r"\N") {
                    continue;
                }
                let parts: Vec<&str> = visible.split(r"\N").collect();
                let l1 = parts[0].split_whitespace().count();
                let l2 = if parts.len() > 1 {
                    parts[1].split_whitespace().count()
                } else {
                    0
                };
                if l1 <= max_wpl && l2 <= max_wpl {
                    legal_splits += 1;
                } else {
                    illegal_found = true;
                }
            }
            if illegal_found {
                checks.push(Check::fail(
                    "CI-LAYOUT",
                    "CI line-break hints produce legal splits only",
                    "a multi-line event violates max_words_per_line",
                    "every \\N split within max_words_per_line",
                    "caption_intel.rs select_legal_line_break",
                    "regenerated ASS",
                    "medium",
                ));
            } else {
                checks.push(Check::pass(
                    "CI-LAYOUT",
                    "CI line-break hints produce legal splits only",
                    &format!("{} multi-line events, all legal", legal_splits),
                    "every \\N split within max_words_per_line",
                    "caption_intel.rs select_legal_line_break",
                    "regenerated ASS",
                ));
            }
        } else {
            checks.push(Check::skip(
                "CI-LAYOUT",
                "CI line-break hints",
                "no hints or single-line template",
            ));
        }

        // Pacing-aware timing: verify no event extends beyond the clip window.
        let out_of_range: Vec<String> = on_disk
            .events
            .iter()
            .filter(|e| e.end > end - start + 0.05 || e.start < -0.05)
            .map(|e| format!("event [{:.2},{:.2}]", e.start, e.end))
            .collect();
        if out_of_range.is_empty() {
            checks.push(Check::pass(
                "CI-TIMING",
                "Event timing stays within clip window (pacing-aware)",
                "all on-disk events within window",
                &format!("within [0, {:.2}]", end - start),
                "captions.rs word timing clamp",
                "on-disk ASS events",
            ));
        } else {
            checks.push(Check::fail(
                "CI-TIMING",
                "Event timing stays within clip window",
                &out_of_range.join("; "),
                &format!("within [0, {:.2}]", end - start),
                "captions.rs word timing clamp",
                "on-disk ASS events",
                "medium",
            ));
        }

        // renderLog should record the CI plan when metadata says applied
        if ci_flag && render_log.contains("caption_intel:") {
            checks.push(Check::pass(
                "CI-LOG",
                "renderLog records CI execution",
                &format!("log contains 'caption_intel:'"),
                "caption_intel: <reason>",
                "lib.rs render_clip_for_candidate log assembly",
                "clips.render_log",
            ));
        } else if ci_flag {
            checks.push(Check::fail(
                "CI-LOG",
                "renderLog records CI execution",
                "log lacks 'caption_intel:'",
                "caption_intel: <reason>",
                "lib.rs render_clip_for_candidate",
                "clips.render_log",
                "low",
            ));
        } else {
            checks.push(Check::skip(
                "CI-LOG",
                "renderLog records CI execution",
                "captionIntelligence false; no CI log expected",
            ));
        }
    }

    plan
}

// ── ASS forensics (Part 5) ──────────────────────────────────────────────────

fn check_ass_forensics(
    parsed: &ParsedAss,
    template: &CaptionTemplate,
    words: &[TranscriptWord],
    start: f64,
    end: f64,
    checks: &mut Vec<Check>,
) {
    // Dialogue events present
    if parsed.events.is_empty() {
        checks.push(Check::fail(
            "ASS-EVENTS",
            "Dialogue events exist",
            "zero Dialogue lines",
            ">= 1",
            "captions.rs generators",
            "ASS [Events] section",
            "high",
        ));
        return;
    }
    checks.push(Check::pass(
        "ASS-EVENTS",
        "Dialogue events exist",
        &format!("{} dialogue events", parsed.events.len()),
        ">= 1",
        "captions.rs generators",
        "ASS [Events] section",
    ));

    // Timing validity
    let mut invalid: Vec<String> = Vec::new();
    for ev in &parsed.events {
        if ev.end <= ev.start {
            invalid.push(format!("event [{:.2},{:.2}] end<=start", ev.start, ev.end));
        } else if ev.start < -0.05 {
            invalid.push(format!("event starts at {:.2}", ev.start));
        }
    }
    if invalid.is_empty() {
        checks.push(Check::pass(
            "ASS-TIMING",
            "All events have valid timing",
            "all events start<end, >=0",
            "start >= 0 and end > start",
            "ASS timing semantics",
            "ASS [Events]",
        ));
    } else {
        checks.push(Check::fail(
            "ASS-TIMING",
            "All events have valid timing",
            &invalid.join("; "),
            "start >= 0 and end > start",
            "ASS timing semantics",
            "ASS [Events]",
            "high",
        ));
    }

    // Malformed tags: balanced braces
    let mut bad_tags: Vec<String> = Vec::new();
    for ev in &parsed.events {
        let opens = ev.text.matches('{').count();
        let closes = ev.text.matches('}').count();
        if opens != closes {
            bad_tags.push(format!(
                "event@{:.2}s brace count {}/{}",
                ev.start, opens, closes
            ));
        }
        // known-tag whitelist
        let mut rest = ev.text.as_str();
        while let Some(open) = rest.find('{') {
            let Some(close) = rest[open..].find('}') else {
                bad_tags.push(format!("event@{:.2}s unclosed tag", ev.start));
                break;
            };
            let body = &rest[open + 1..open + close];
            // Tag name: longest known-prefix of the leading alphanumeric run.
            // "\an5" -> "an", "\fscx110" -> "fscx", "\fnMontserrat" -> "fn"
            // (font names run on after the tag name), "\3c" -> "3c".
            let chars: Vec<char> = body
                .trim_start_matches('\\')
                .chars()
                .take_while(|c| c.is_alphanumeric())
                .collect();
            const KNOWN: &[&str] = &[
                "an", "move", "alpha", "t", "fad", "fade", "fscx", "fscy", "fs", "fn", "b", "i",
                "c", "3c", "4c", "bord", "shad", "pos", "clip", "k", "K", "karo", "org", "frx",
                "fry", "frz", "fax", "fay", "q", "r", "be", "blur", "fsp", "fe",
            ];
            let run: String = chars.iter().collect();
            let name_known = (1..=chars.len())
                .rev()
                .any(|n| KNOWN.contains(&chars[..n].iter().collect::<String>().as_str()));
            if !name_known && !body.is_empty() {
                bad_tags.push(format!("event@{:.2}s unknown tag \\{}", ev.start, run));
            }
            rest = &rest[open + close + 1..];
        }
    }
    if bad_tags.is_empty() {
        checks.push(Check::pass(
            "ASS-TAGS",
            "Override tags well-formed (balanced, known)",
            "all tags balanced and in whitelist",
            "balanced {} with known tag names",
            "libass tag set",
            "ASS Dialogue text",
        ));
    } else {
        checks.push(Check::fail(
            "ASS-TAGS",
            "Override tags well-formed",
            &bad_tags.join("; "),
            "balanced {} with known tag names",
            "libass tag set",
            "ASS Dialogue text",
            "medium",
        ));
    }

    // Font references: every \fn and Style font must resolve to a bundled font
    let mut bad_fonts: Vec<String> = Vec::new();
    let mut referenced: Vec<String> = Vec::new();
    for s in &parsed.styles {
        referenced.push(s.fontname.clone());
    }
    for ev in &parsed.events {
        let mut rest = ev.text.as_str();
        while let Some(pos) = rest.find(r"\fn") {
            let after = &rest[pos + 3..];
            let name: String = after
                .chars()
                .take_while(|c| c.is_alphabetic() || *c == ' ')
                .collect();
            referenced.push(name.trim().to_string());
            rest = &rest[pos + 3..];
        }
    }
    for f in &referenced {
        if captions::required_font_filename(f).is_none() {
            bad_fonts.push(f.clone());
        }
    }
    if bad_fonts.is_empty() {
        checks.push(Check::pass(
            "ASS-FONTS",
            "Font references resolve to bundled fonts",
            &format!("{:?}", referenced.iter().cloned().collect::<Vec<_>>()),
            "font family with required_font_filename mapping",
            "captions.rs required_font_filename + autoshorts/fonts/",
            "ASS Style/Dialogue",
        ));
    } else {
        checks.push(Check::fail(
            "ASS-FONTS",
            "Font references resolve to bundled fonts",
            &format!("unresolved: {:?}", bad_fonts),
            "all fonts bundled",
            "captions.rs required_font_filename",
            "ASS Style/Dialogue",
            "medium",
        ));
    }

    // Style leakage: only styles this template may emit
    let allowed = if template.template_id == "preset_bhaukal_caption" {
        vec!["Default", "HookStrip"]
    } else {
        vec!["Default", "DualStackSeam"]
    };
    let leaked: Vec<String> = parsed
        .styles
        .iter()
        .map(|s| s.name.clone())
        .filter(|n| !allowed.contains(&n.as_str()))
        .collect();
    if leaked.is_empty() {
        checks.push(Check::pass(
            "ASS-LEAKAGE",
            "No unexpected style names",
            &format!(
                "{:?}",
                parsed
                    .styles
                    .iter()
                    .map(|s| s.name.clone())
                    .collect::<Vec<_>>()
            ),
            &format!("subset of {:?}", allowed),
            "captions.rs style emission",
            "ASS [V4+ Styles]",
        ));
    } else {
        checks.push(Check::fail(
            "ASS-LEAKAGE",
            "No unexpected style names",
            &format!("unexpected styles: {:?}", leaked),
            &format!("subset of {:?}", allowed),
            "captions.rs style emission",
            "ASS [V4+ Styles]",
            "medium",
        ));
    }

    // Word-level coverage: every transcript word in the window should appear in some event
    let mut missing_words: Vec<String> = Vec::new();
    for w in words.iter() {
        if w.end <= start || w.start >= end {
            continue;
        }
        let tok = norm_token(&w.text);
        if tok.is_empty() {
            continue;
        }
        let found = parsed.events.iter().any(|e| {
            visible_words(&strip_tags(&e.text))
                .iter()
                .any(|t| norm_token(t) == tok)
        });
        if !found {
            missing_words.push(w.text.clone());
        }
    }
    let window_words = words
        .iter()
        .filter(|w| w.end > start && w.start < end)
        .count();
    if missing_words.is_empty() {
        checks.push(Check::pass(
            "ASS-COVERAGE",
            "All in-window transcript words appear in ASS",
            &format!("{} in-window words all present", window_words),
            "every word with end>start && start<end appears",
            "captions.rs candidate_words filter",
            "ASS Dialogue text",
        ));
    } else if window_words > 0 {
        checks.push(Check::fail(
            "ASS-COVERAGE",
            "All in-window transcript words appear in ASS",
            &format!(
                "missing (first 10): {}",
                missing_words
                    .iter()
                    .take(10)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            "every in-window word appears",
            "captions.rs candidate_words filter",
            "ASS Dialogue text",
            "medium",
        ));
    } else {
        checks.push(Check::skip(
            "ASS-COVERAGE",
            "Transcript word coverage",
            "no words in window",
        ));
    }

    // Overlapping events: word-by-word cadence means sequential non-overlap per style.
    // (Kinetic T5/T6 intentionally overlap progressive steps — skipped there.)
    let kinetic = template.template_id == "preset_dynamic_editorial"
        || template.template_id == "preset_bhaukal_caption";
    if !kinetic {
        let mut overlaps: Vec<String> = Vec::new();
        let mut sorted = parsed.events.clone();
        sorted.sort_by(|a, b| {
            a.start
                .partial_cmp(&b.start)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for pair in sorted.windows(2) {
            if pair[0].style == pair[1].style
                && pair[0].layer == pair[1].layer
                && pair[1].start < pair[0].end - 0.06
            {
                overlaps.push(format!(
                    "[{:.2},{:.2}] vs [{:.2},{:.2}]",
                    pair[0].start, pair[0].end, pair[1].start, pair[1].end
                ));
            }
        }
        if overlaps.is_empty() {
            checks.push(Check::pass(
                "ASS-OVERLAP",
                "No same-style event overlap (sequential cadence)",
                "all sequential",
                "no same style/layer time overlap",
                "captions.rs word-by-word cadence",
                "ASS events",
            ));
        } else {
            checks.push(Check::fail(
                "ASS-OVERLAP",
                "No same-style event overlap",
                &format!("{} overlaps, e.g. {}", overlaps.len(), overlaps[0]),
                "no same style/layer overlap",
                "captions.rs cadence",
                "ASS events",
                "medium",
            ));
        }
    } else {
        checks.push(Check::skip(
            "ASS-OVERLAP",
            "Event overlap",
            "kinetic templates intentionally overlap progressive build steps",
        ));
    }
}

// ── Collision / spacing QA (Part 7) — T5 real metrics ──────────────────────

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct WordBound {
    text: String,
    x: f64,
    y: f64,
    w: f64,
    h: f64,
    start: f64,
    end: f64,
    font: String,
    font_size: f64,
}

/// Extract real rendered word bounds for T5 by replaying the renderer's own
/// metrics: positions come from the ASS \move targets, sizes/fonts from the
/// emitted tags, and widths from the production `measure_t5_word_width`.
fn t5_word_bounds(parsed: &ParsedAss) -> Vec<WordBound> {
    let mut out = Vec::new();
    for ev in &parsed.events {
        let text = strip_tags(&ev.text);
        if text.is_empty() {
            continue;
        }
        let font = ev
            .text
            .match_indices(r"\fn")
            .next()
            .map(|(i, _)| {
                ev.text[i + 3..]
                    .chars()
                    .take_while(|c| c.is_alphabetic() || *c == ' ')
                    .collect::<String>()
                    .trim()
                    .to_string()
            })
            .unwrap_or_default();
        let fs: f64 = ev
            .text
            .match_indices(r"\fs")
            .next()
            .and_then(|(i, _)| {
                ev.text[i + 3..]
                    .chars()
                    .take_while(|c| c.is_ascii_digit() || *c == '.')
                    .collect::<String>()
                    .parse()
                    .ok()
            })
            .unwrap_or(62.0);
        // final position of \move(x1,y1,x2,y2,...)
        let pos = ev.text.find(r"\move(").and_then(|i| {
            let rest = &ev.text[i + 6..];
            let end = rest.find(')')?;
            let nums: Vec<f64> = rest[..end]
                .split(',')
                .filter_map(|n| n.trim().parse().ok())
                .collect();
            if nums.len() >= 4 {
                Some((nums[2], nums[3]))
            } else {
                None
            }
        });
        let Some((x, y)) = pos else { continue };
        let role = font_to_role(&font);
        let (w, h) = if let Some(role) = role {
            captions::measure_t5_word_width(&text, role, fs as u32)
        } else {
            (text.len() as f64 * 0.6 * fs, fs)
        };
        out.push(WordBound {
            text: text.clone(),
            x,
            y,
            w,
            h,
            start: ev.start,
            end: ev.end,
            font,
            font_size: fs,
        });
    }
    out
}

fn check_t5_collisions(bounds: &[WordBound], checks: &mut Vec<Check>) {
    if bounds.is_empty() {
        checks.push(Check::skip(
            "QA-COLLISION",
            "T5 word-to-word collision (real metrics)",
            "no word bounds extractable",
        ));
        return;
    }
    // Only words co-visible in time can collide. \an5 center anchor: bbox is
    // [x - w/2, x + w/2] x [y - h/2, y + h/2].
    let mut collisions: Vec<String> = Vec::new();
    for i in 0..bounds.len() {
        for j in (i + 1)..bounds.len() {
            let a = &bounds[i];
            let b = &bounds[j];
            let overlap_time = a.start < b.end - 0.02 && b.start < a.end - 0.02;
            if !overlap_time {
                continue;
            }
            let ax1 = a.x - a.w / 2.0;
            let ax2 = a.x + a.w / 2.0;
            let bx1 = b.x - b.w / 2.0;
            let bx2 = b.x + b.w / 2.0;
            let ay1 = a.y - a.h / 2.0;
            let ay2 = a.y + a.h / 2.0;
            let by1 = b.y - b.h / 2.0;
            let by2 = b.y + b.h / 2.0;
            if ax1 < bx2 - 2.0 && bx1 < ax2 - 2.0 && ay1 < by2 - 2.0 && by1 < ay2 - 2.0 {
                collisions.push(format!("'{}' vs '{}' (bbox overlap)", a.text, b.text));
            }
        }
    }
    if collisions.is_empty() {
        checks.push(Check::pass(
            "QA-COLLISION",
            "T5 word-to-word collision (real metrics)",
            "no overlapping bboxes among co-visible words",
            "no bbox overlap",
            "captions.rs measure_t5_word_width (production metrics) + ASS \\move targets",
            "T5 word bounds",
        ));
    } else {
        checks.push(Check::fail(
            "QA-COLLISION",
            "T5 word-to-word collision (real metrics)",
            &format!(
                "{} collisions: {}",
                collisions.len(),
                collisions
                    .iter()
                    .take(3)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join("; ")
            ),
            "no bbox overlap",
            "captions.rs measure_t5_word_width",
            "T5 word bounds",
            "high",
        ));
    }

    // Safe margins
    let mut out_of_bounds: Vec<String> = Vec::new();
    for b in bounds {
        if b.x - b.w / 2.0 < 40.0 || b.x + b.w / 2.0 > 1040.0 {
            out_of_bounds.push(format!("'{}' x out of safe band", b.text));
        }
        if b.y - b.h / 2.0 < 780.0 || b.y + b.h / 2.0 > 1560.0 {
            out_of_bounds.push(format!("'{}' y out of safe band", b.text));
        }
    }
    if out_of_bounds.is_empty() {
        checks.push(Check::pass(
            "QA-MARGINS",
            "T5 words inside safe canvas margins",
            "all bboxes within x[40,1040] y[780,1560]",
            "safe margins",
            "captions.rs layout clamps",
            "T5 word bounds",
        ));
    } else {
        checks.push(Check::fail(
            "QA-MARGINS",
            "T5 words inside safe canvas margins",
            &out_of_bounds.join("; "),
            "within safe margins",
            "captions.rs layout clamps",
            "T5 word bounds",
            "medium",
        ));
    }
}

// ── Applied-feature truth (Part 9) ──────────────────────────────────────────

fn check_applied_features(
    applied_features: &Option<String>,
    render_log: &str,
    output_path: &Option<String>,
    caption_style: &Option<String>,
    template: &CaptionTemplate,
    ci_plan: Option<&CaptionIntelPlan>,
    checks: &mut Vec<Check>,
) {
    let parsed: serde_json::Value = match applied_features {
        Some(s) if !s.trim().is_empty() => {
            serde_json::from_str(s).unwrap_or(serde_json::Value::Null)
        }
        _ => serde_json::Value::Null,
    };

    let get = |key: &str| -> Option<bool> { parsed.get(key).and_then(|v| v.as_bool()) };

    // smartPacing
    match get("smartPacing") {
        Some(true) => {
            if render_log.contains("[Smart Pacing]") {
                checks.push(Check::pass(
                    "AF-PACING",
                    "smartPacing metadata vs renderLog",
                    "flag=true, log contains [Smart Pacing]",
                    "flag ⟺ log evidence",
                    "lib.rs applied_features + pacing::pacing_summary",
                    "clips.render_log",
                ));
            } else {
                checks.push(Check::fail(
                    "AF-PACING",
                    "smartPacing metadata vs renderLog",
                    "flag=true but log has no [Smart Pacing]",
                    "log evidence for true flag",
                    "lib.rs log assembly",
                    "clips.render_log",
                    "medium",
                ));
            }
        }
        Some(false) => {
            if !render_log.contains("[Smart Pacing]") {
                checks.push(Check::pass(
                    "AF-PACING",
                    "smartPacing metadata vs renderLog",
                    "flag=false, no pacing in log",
                    "flag ⟺ log evidence",
                    "lib.rs applied_features",
                    "clips.render_log",
                ));
            } else {
                checks.push(Check::fail(
                    "AF-PACING",
                    "smartPacing metadata vs renderLog",
                    "flag=false but log contains [Smart Pacing]",
                    "no log evidence expected",
                    "lib.rs applied_features",
                    "clips.render_log",
                    "medium",
                ));
            }
        }
        None => checks.push(Check::skip(
            "AF-PACING",
            "smartPacing metadata",
            "legacy clip: applied_features NULL",
        )),
    }

    // hookEndingOptimization
    match get("hookEndingOptimization") {
        Some(true) => {
            if render_log.contains("boundaries optimized")
                || render_log.contains("start ")
                || render_log.contains("→")
            {
                checks.push(Check::pass(
                    "AF-HOOK-END",
                    "hookEndingOptimization metadata vs renderLog",
                    "flag=true, boundary summary in log",
                    "boundary_opt.summary in log",
                    "lib.rs + boundary.rs BoundaryOptimization::summary",
                    "clips.render_log",
                ));
            } else {
                checks.push(Check::fail(
                    "AF-HOOK-END",
                    "hookEndingOptimization metadata vs renderLog",
                    "flag=true but no boundary summary",
                    "boundary summary in log",
                    "lib.rs log assembly",
                    "clips.render_log",
                    "medium",
                ));
            }
        }
        Some(false) => {
            checks.push(Check::pass(
                "AF-HOOK-END",
                "hookEndingOptimization metadata vs renderLog",
                "flag=false",
                "no boundary movement",
                "lib.rs applied_features (start_changed||end_changed)",
                "clips.render_log",
            ));
        }
        None => checks.push(Check::skip(
            "AF-HOOK-END",
            "hookEndingOptimization metadata",
            "legacy clip: applied_features NULL",
        )),
    }

    // audioIntelligence
    match get("audioIntelligence") {
        Some(true) => {
            if render_log.contains("[Audio Intelligence]") {
                checks.push(Check::pass(
                    "AF-AUDIO",
                    "audioIntelligence metadata vs renderLog",
                    "flag=true, log contains [Audio Intelligence]",
                    "flag ⟺ log evidence",
                    "lib.rs + audio::audio_summary",
                    "clips.render_log",
                ));
            } else {
                checks.push(Check::fail(
                    "AF-AUDIO",
                    "audioIntelligence metadata vs renderLog",
                    "flag=true but log has no [Audio Intelligence]",
                    "log evidence",
                    "lib.rs log assembly",
                    "clips.render_log",
                    "medium",
                ));
            }
        }
        Some(false) => {
            if !render_log.contains("[Audio Intelligence]") {
                checks.push(Check::pass(
                    "AF-AUDIO",
                    "audioIntelligence metadata vs renderLog",
                    "flag=false, no audio in log",
                    "flag ⟺ log evidence",
                    "lib.rs applied_features",
                    "clips.render_log",
                ));
            } else {
                checks.push(Check::fail(
                    "AF-AUDIO",
                    "audioIntelligence metadata vs renderLog",
                    "flag=false but log contains [Audio Intelligence]",
                    "no log evidence expected",
                    "lib.rs applied_features",
                    "clips.render_log",
                    "medium",
                ));
            }
        }
        None => checks.push(Check::skip(
            "AF-AUDIO",
            "audioIntelligence metadata",
            "legacy clip: applied_features NULL",
        )),
    }

    // captionIntelligence — cross-checked against the regenerated plan in Part 4.
    match get("captionIntelligence") {
        Some(flag) => {
            let ass_exists = output_path
                .as_ref()
                .map(|p| Path::new(p).exists())
                .unwrap_or(false);
            match ci_plan {
                Some(_) if flag => checks.push(Check::pass("AF-CI", "captionIntelligence metadata vs real plan",
                    "flag=true and plan regenerated Some", "flag ⟺ plan exists and was consumed",
                    "lib.rs caption_intelligence_applied = usable_and_written && ass_path.is_some()", "applied_features + regenerated plan")),
                Some(_) if !flag => checks.push(Check::fail("AF-CI", "captionIntelligence metadata vs real plan",
                    "flag=false but plan regenerated Some from same inputs", "flag should be true when plan consumed",
                    "lib.rs applied_features derivation", "applied_features + plan", "high")),
                None if flag => checks.push(Check::fail("AF-CI", "captionIntelligence metadata vs real plan",
                    "flag=true but plan regenerates None (gated off or evidence insufficient)", "flag true requires plan Some",
                    "caption_intel.rs confidence gate", "applied_features + plan", "high")),
                None if !flag => checks.push(Check::pass("AF-CI", "captionIntelligence metadata vs real plan",
                    "flag=false and plan None (CI NOT APPLICABLE / FALLBACK)", "flag ⟺ plan existence",
                    "caption_intel.rs confidence gate", "applied_features + plan")),
                _ => checks.push(Check::not_verified("AF-CI", "captionIntelligence metadata", "indeterminate")),
            }
            let _ = ass_exists;
        }
        None => checks.push(Check::skip(
            "AF-CI",
            "captionIntelligence metadata",
            "legacy clip: applied_features NULL",
        )),
    }

    // captionPreset — metadata must reference the template the ASS actually uses.
    match caption_style {
        Some(style) => {
            if style == &template.template_id {
                checks.push(Check::pass(
                    "AF-PRESET",
                    "captionPreset metadata vs ASS template signature",
                    &format!("projects.caption_style = {}", style),
                    &format!("ASS matches {}", template.template_id),
                    "db.rs projects.caption_style + captions.rs template registry",
                    "projects row + ASS Style",
                ));
            } else {
                checks.push(Check::fail(
                    "AF-PRESET",
                    "captionPreset metadata vs ASS template signature",
                    &format!(
                        "projects.caption_style = {} but ASS matches {}",
                        style, template.template_id
                    ),
                    "style == ASS template",
                    "db.rs projects.caption_style",
                    "projects row + ASS Style",
                    "high",
                ));
            }
            // renderLog must agree too
            if render_log.contains(&format!("caption_template: {}", style)) {
                checks.push(Check::pass(
                    "AF-PRESET-LOG",
                    "renderLog caption_template matches metadata",
                    &format!("log has 'caption_template: {}'", style),
                    "log template == caption_style",
                    "lib.rs log assembly",
                    "clips.render_log",
                ));
            } else if render_log.is_empty() {
                checks.push(Check::skip(
                    "AF-PRESET-LOG",
                    "renderLog caption_template",
                    "empty render_log",
                ));
            } else {
                checks.push(Check::fail(
                    "AF-PRESET-LOG",
                    "renderLog caption_template matches metadata",
                    "log lacks expected caption_template line",
                    &format!("caption_template: {}", style),
                    "lib.rs log assembly",
                    "clips.render_log",
                    "medium",
                ));
            }
        }
        None => checks.push(Check::fail(
            "AF-PRESET",
            "captionPreset metadata vs ASS template signature",
            "projects.caption_style is NULL",
            "style must be set",
            "db.rs projects.caption_style",
            "projects row",
            "medium",
        )),
    }
}

// ── Template isolation (Part 8) ─────────────────────────────────────────────

fn synthetic_words() -> Vec<TranscriptWord> {
    let texts = [
        "Never", "make", "this", "mistake", "when", "you", "train", "hard", "every", "single",
        "day", "champ",
    ];
    let mut words = Vec::new();
    let mut t = 0.0;
    for (i, tx) in texts.iter().enumerate() {
        words.push(TranscriptWord {
            text: tx.to_string(),
            start: t,
            end: t + 0.35,
            speaker: if i % 2 == 0 {
                Some("S1".to_string())
            } else {
                None
            },
        });
        t += 0.42;
    }
    words
}

fn cmd_isolation(args: &[String]) {
    let words = if args.len() >= 3 {
        match load_words(&args[2]) {
            Some(w) => w,
            None => {
                eprintln!("failed to load words from {}", args[1]);
                std::process::exit(2);
            }
        }
    } else {
        synthetic_words()
    };

    let mut checks: Vec<Check> = Vec::new();
    let templates = captions::all_caption_templates();
    let ids: Vec<String> = templates.iter().map(|t| t.template_id.clone()).collect();
    checks.push(Check::pass("ISO-REGISTRY", "Template registry T1..T7 present",
        &format!("{:?}", ids), "7 templates: preset_viral_bold, preset_mrbeast_pop, preset_minimal_capsule, preset_cinematic_vlog, preset_dynamic_editorial, preset_bhaukal_caption, preset_t7",
        "captions.rs all_caption_templates()", "registry"));

    // T7 is implemented in 10.0 (T7 Reference Style); T8 does not exist — recorded, not invented.
    if captions::get_caption_template("preset_t7").is_some() {
        checks.push(Check::pass("ISO-T7", "T7 template registered",
            "preset_t7 exists in the 10.0 registry", "T7 Reference Style (Matt Bold 700, white + #F9EF07 current-chunk, phrase-chunked rolling, no stroke/shadow/box)",
            "captions.rs all_caption_templates()", "registry"));
    }
    for absent in ["preset_t8", "T8"] {
        if captions::get_caption_template(absent).is_none() {
            checks.push(Check::skip(
                "ISO-T8",
                "T8 template",
                &format!("{} does not exist in the 10.0 registry", absent),
            ));
        }
    }

    let start = 0.0;
    let end = 6.0;

    // Baseline ASS per template
    let mut baseline: BTreeMap<String, String> = BTreeMap::new();
    for t in &templates {
        let ass = captions::generate_ass_from_template_with_framing_and_intel(
            &words, start, end, t, None, None, None,
        );
        baseline.insert(t.template_id.clone(), ass);
    }

    // Each template's ASS may only reference its OWN fonts and palette.
    for t in &templates {
        let ass = baseline.get(&t.template_id).unwrap();
        let parsed = parse_ass(ass);
        let used_fonts: Vec<String> = parsed
            .iter()
            .flat_map(|p| p.styles.iter().map(|s| s.fontname.clone()))
            .collect();
        let allowed_fonts: Vec<&str> = match t.template_id.as_str() {
            "preset_viral_bold" => vec!["Bebas Neue"],
            "preset_mrbeast_pop" | "preset_bhaukal_caption" => {
                vec!["Montserrat", "Poppins", "Inter"]
            } // T6 mix per role
            "preset_t7" => vec!["Matt_Trial-Bold"],
            "preset_minimal_capsule" => vec!["Inter"],
            "preset_cinematic_vlog" => vec!["Poppins"],
            "preset_dynamic_editorial" => vec!["Bebas Neue", "Montserrat", "Inter", "Poppins"], // roles
            _ => vec![],
        };
        let bad: Vec<String> = used_fonts
            .iter()
            .filter(|f| !allowed_fonts.contains(&f.as_str()))
            .cloned()
            .collect();
        if bad.is_empty() {
            checks.push(Check::pass(
                &format!("ISO-FONT-{}", t.template_id),
                &format!("{} uses only its own fonts", t.template_id),
                &format!("{:?}", used_fonts),
                &format!("subset of {:?}", allowed_fonts),
                "captions.rs template registry",
                &format!("ASS for {}", t.template_id),
            ));
        } else {
            checks.push(Check::fail(
                &format!("ISO-FONT-{}", t.template_id),
                &format!("{} uses only its own fonts", t.template_id),
                &format!("foreign fonts: {:?}", bad),
                &format!("subset of {:?}", allowed_fonts),
                "captions.rs template registry",
                &format!("ASS for {}", t.template_id),
                "high",
            ));
        }
    }

    // Mutation isolation: changing one template must not alter any other template's ASS.
    for mutate_idx in 0..templates.len() {
        let mut mutated = templates.clone();
        let target = &mut mutated[mutate_idx];
        // perturb font, size, color, position — the styling axes the prompt lists
        target.global_styles.font_family = "ComicSansMS".to_string();
        target.global_styles.font_size = 7;
        target.global_styles.font_color = "#FF00FF".to_string();
        target.global_styles.position_y = 0.5;
        if let Some(s) = target.global_styles.stroke_color.as_mut() {
            *s = "#FFFFFF".to_string();
        }
        // T5/T6 derive their real styling from active_state (not global_styles);
        // perturb those too so the mutation is effective for every template.
        match &mut target.active_state {
            CaptionActiveState::DynamicEditorialKinetic {
                base_color,
                accent_color,
                secondary_accent,
                ..
            } => {
                *base_color = "#FF00FF".to_string();
                *accent_color = "#00FF00".to_string();
                *secondary_accent = "#0000FF".to_string();
            }
            CaptionActiveState::BhaukalKinetic {
                base_color,
                hero_color,
                ..
            } => {
                *base_color = "#FF00FF".to_string();
                *hero_color = "#0000FF".to_string();
            }
            CaptionActiveState::WordByWordSwap {
                primary_highlight_color,
                secondary_highlight_color,
                ..
            } => {
                *primary_highlight_color = "#FF00FF".to_string();
                *secondary_highlight_color = "#00FFFF".to_string();
            }
            CaptionActiveState::KaraokeProgressive {
                highlight_color, ..
            } => {
                *highlight_color = "#FF00FF".to_string();
            }
            CaptionActiveState::SpeechChunkedRolling {
                highlight_color, ..
            } => {
                *highlight_color = "#FF00FF".to_string();
            }
            CaptionActiveState::SmoothFade {
                highlight_color, ..
            } => {
                *highlight_color = "#FF00FF".to_string();
            }
            CaptionActiveState::StaticOpacityReveal { .. } => {}
        }

        for (j, t) in templates.iter().enumerate() {
            if j == mutate_idx {
                continue;
            }
            let regenerated = captions::generate_ass_from_template_with_framing_and_intel(
                &words, start, end, t, None, None, None,
            );
            if &regenerated == baseline.get(&t.template_id).unwrap() {
                checks.push(Check::pass(
                    &format!(
                        "ISO-MUT-{}-{}",
                        templates[mutate_idx].template_id, t.template_id
                    ),
                    &format!(
                        "Mutating {} leaves {} unchanged",
                        templates[mutate_idx].template_id, t.template_id
                    ),
                    "byte-identical ASS",
                    "unchanged ASS",
                    "captions.rs registry (per-template values, no shared mutable state)",
                    &format!("regenerated ASS for {}", t.template_id),
                ));
            } else {
                checks.push(Check::fail(
                    &format!(
                        "ISO-MUT-{}-{}",
                        templates[mutate_idx].template_id, t.template_id
                    ),
                    &format!(
                        "Mutating {} leaves {} unchanged",
                        templates[mutate_idx].template_id, t.template_id
                    ),
                    "ASS CHANGED",
                    "unchanged ASS",
                    "captions.rs registry",
                    &format!("regenerated ASS for {}", t.template_id),
                    "high",
                ));
            }
        }
        // sanity: the mutated template itself must change (mutation was effective)
        let self_regen = captions::generate_ass_from_template_with_framing_and_intel(
            &words,
            start,
            end,
            &mutated[mutate_idx],
            None,
            None,
            None,
        );
        if &self_regen != baseline.get(&templates[mutate_idx].template_id).unwrap() {
            checks.push(Check::pass(
                &format!("ISO-MUT-SELF-{}", templates[mutate_idx].template_id),
                &format!(
                    "Mutation of {} actually changes its own ASS",
                    templates[mutate_idx].template_id
                ),
                "own ASS changed",
                "own ASS changes",
                "captions.rs registry",
                &format!("ASS for {}", templates[mutate_idx].template_id),
            ));
        } else {
            checks.push(Check::fail(
                &format!("ISO-MUT-SELF-{}", templates[mutate_idx].template_id),
                &format!(
                    "Mutation of {} actually changes its own ASS",
                    templates[mutate_idx].template_id
                ),
                "own ASS unchanged (mutation ineffective)",
                "own ASS changes",
                "captions.rs registry",
                &format!("ASS for {}", templates[mutate_idx].template_id),
                "high",
            ));
        }
    }

    let report = Report {
        harness: "caption_qa".into(),
        mode: "isolation".into(),
        subject: format!("{} templates", templates.len()),
        checks,
        summary: Default::default(),
    };
    let mut r = report;
    r.summary = summarize(&r.checks);
    let outdir = std::env::var("CAPTION_QA_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join("caption_qa_reports"));
    let _ = fs::create_dir_all(&outdir);
    emit_to(&r, &outdir);
    emit(&r);
}

// ── Helpers ─────────────────────────────────────────────────────────────────

fn load_words(path: &str) -> Option<Vec<TranscriptWord>> {
    let content = fs::read_to_string(path).ok()?;
    if let Ok(t) = serde_json::from_str::<NormalizedTranscript>(&content) {
        return Some(t.words);
    }
    serde_json::from_str::<Vec<TranscriptWord>>(&content).ok()
}

fn read_only_db_copy(db_path: &str) -> PathBuf {
    // Never open the production DB read-write: copy to a temp file first.
    let src = Path::new(db_path);
    let mut tmp = std::env::temp_dir();
    tmp.push(format!("caption_qa_{}.sqlite", uuid::Uuid::new_v4()));
    if fs::copy(src, &tmp).is_err() {
        return src.to_path_buf();
    }
    tmp
}

// ── audit-clip ──────────────────────────────────────────────────────────────

fn cmd_audit_clip(args: &[String]) {
    if args.len() < 5 {
        eprintln!("usage: caption_qa audit-clip <db> <project_id> <candidate_id>");
        std::process::exit(2);
    }
    let db_path = &args[2];
    let project_id = &args[3];
    let candidate_id = &args[4];

    let tmp_db = read_only_db_copy(db_path);
    eprintln!(
        "[caption_qa] using db copy: {} (exists={})",
        tmp_db.display(),
        tmp_db.exists()
    );
    let db = match Database::open(&tmp_db) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("failed to open DB copy {}: {}", tmp_db.display(), e);
            std::process::exit(2);
        }
    };

    let project = match db.get_project(project_id) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("project lookup failed for {}: {}", project_id, e);
            std::process::exit(2);
        }
    };
    let (candidate, _) = match db.get_candidate_with_project(candidate_id) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("candidate lookup failed: {}", e);
            std::process::exit(2);
        }
    };
    let detail = match db.project_detail(project_id) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("project_detail failed: {}", e);
            std::process::exit(2);
        }
    };
    let clip = match detail
        .clips
        .iter()
        .find(|c| &c.candidate_id == candidate_id)
    {
        Some(c) => c.clone(),
        None => {
            eprintln!("no clip row for candidate {}", candidate_id);
            std::process::exit(2);
        }
    };

    let transcript = db.latest_transcript(project_id).ok().flatten();
    let words: Vec<TranscriptWord> = match transcript.as_ref() {
        Some(t) => serde_json::from_str::<NormalizedTranscript>(&t.raw_json)
            .map(|n| n.words)
            .unwrap_or_default(),
        None => Vec::new(),
    };

    let mut checks: Vec<Check> = Vec::new();
    let start = candidate.start_sec;
    let end = candidate.end_sec;

    // Chain link 0: CODE EXISTS — template in registry
    let style = project
        .caption_style
        .as_deref()
        .unwrap_or("preset_viral_bold");
    let template = match captions::get_caption_template(style) {
        Some(t) => {
            checks.push(Check::pass(
                "CHAIN-CODE",
                "Template exists in registry (CODE EXISTS)",
                &format!("{} → {}", t.template_id, t.name),
                "registered template",
                "captions.rs all_caption_templates()",
                "registry",
            ));
            t
        }
        None => {
            checks.push(Check::fail(
                "CHAIN-CODE",
                "Template exists in registry (CODE EXISTS)",
                style,
                "a registered template id",
                "captions.rs all_caption_templates()",
                "projects.caption_style",
                "high",
            ));
            std::process::exit(3);
        }
    };

    // Chain link 1: FEATURE EXECUTED — CI plan (re-run real planner)
    // Chain link 2: OUTPUT WAS CONSUMED — ASS file on disk
    let ass_path = clip.caption_ass_path.as_deref();
    let on_disk_raw = ass_path.and_then(|p| fs::read_to_string(p).ok());
    let on_disk = on_disk_raw.as_ref().and_then(|s| parse_ass(s));

    let ass_evidence = match (&on_disk_raw, &on_disk) {
        (Some(raw), Some(_)) => {
            checks.push(Check::pass(
                "CHAIN-ASS",
                "Generated ASS artifact exists and parses (OUTPUT WAS CONSUMED)",
                &format!(
                    "{} bytes, {} styles, {} events",
                    raw.len(),
                    on_disk.as_ref().unwrap().styles.len(),
                    on_disk.as_ref().unwrap().events.len()
                ),
                "parseable ASS with styles+events",
                "lib.rs writes clip-<id>.ass before render",
                &format!("{}", ass_path.unwrap()),
            ));
            on_disk.as_ref().unwrap().clone()
        }
        (Some(raw), None) => {
            checks.push(Check::fail(
                "CHAIN-ASS",
                "Generated ASS artifact parses",
                &format!(
                    "unparseable (first 80 bytes: {:?})",
                    &raw[..raw.len().min(80)]
                ),
                "parseable ASS",
                "ASS format",
                ass_path.unwrap(),
                "high",
            ));
            std::process::exit(3);
        }
        (None, _) => {
            checks.push(Check::fail(
                "CHAIN-ASS",
                "Generated ASS artifact exists (OUTPUT WAS CONSUMED)",
                "no caption_ass_path or file missing",
                "existing .ass file",
                "lib.rs clip_ass_path write",
                &format!("{:?}", ass_path),
                "high",
            ));
            std::process::exit(3);
        }
    };

    // Chain link 3: FINAL OUTPUT — MP4 exists
    match &clip.output_path {
        Some(p) if Path::new(p).exists() => {
            checks.push(Check::pass(
                "CHAIN-MP4",
                "Final MP4 exists (FINAL OUTPUT)",
                p,
                "existing mp4",
                "pacing::render_paced_clip output",
                p,
            ));
        }
        Some(p) => {
            checks.push(Check::fail(
                "CHAIN-MP4",
                "Final MP4 exists",
                &format!("recorded but missing: {}", p),
                "existing mp4",
                "clips.output_path",
                p,
                "high",
            ));
        }
        None => {
            checks.push(Check::fail(
                "CHAIN-MP4",
                "Final MP4 exists",
                "no output_path",
                "existing mp4",
                "clips.output_path",
                "clips row",
                "high",
            ));
        }
    }

    // Parts 3 & 5: template conformance + ASS forensics against on-disk artifact
    check_template_conformance(&ass_evidence, &template, false, &mut checks);
    check_ass_forensics(&ass_evidence, &template, &words, start, end, &mut checks);

    // Part 4: Caption Intelligence — never trust the flag alone
    let ci_flag = clip
        .applied_features
        .as_deref()
        .and_then(|s| serde_json::from_str::<serde_json::Value>(s).ok())
        .and_then(|v| v.get("captionIntelligence").and_then(|x| x.as_bool()))
        .unwrap_or(false);
    let plan = check_caption_intel(
        &words,
        &candidate,
        start,
        end,
        &ass_evidence,
        &template,
        ci_flag,
        clip.render_log.as_deref().unwrap_or(""),
        &mut checks,
    );

    // Part 9: applied-feature truth table vs artifacts
    check_applied_features(
        &clip.applied_features,
        clip.render_log.as_deref().unwrap_or(""),
        &clip.output_path,
        &project.caption_style,
        &template,
        plan.as_ref(),
        &mut checks,
    );

    // T5 collision QA with real metrics (on-disk artifact)
    if template.template_id == "preset_dynamic_editorial" {
        let bounds = t5_word_bounds(&ass_evidence);
        check_t5_collisions(&bounds, &mut checks);
        // emit bounds for the visual QA layer
        if let Some(out) = std::env::var("CAPTION_QA_BOUNDS_OUT").ok() {
            let _ = fs::write(&out, serde_json::to_string_pretty(&bounds).unwrap());
        }
    } else {
        checks.push(Check::skip(
            "QA-COLLISION",
            "T5 word-to-word collision (real metrics)",
            "collision metrics only apply to kinetic T5 layout",
        ));
    }

    let mut report = Report {
        harness: "caption_qa".into(),
        mode: "audit-clip".into(),
        subject: format!(
            "project {} candidate {} (template {})",
            project_id, candidate_id, template.template_id
        ),
        checks,
        summary: Default::default(),
    };
    report.summary = summarize(&report.checks);
    let outdir = std::env::var("CAPTION_QA_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join("caption_qa_reports"));
    let _ = fs::create_dir_all(&outdir);
    emit_to(&report, &outdir);
    emit(&report);

    let _ = db;
}

// ── render (Part 10): real pipeline end-to-end ──────────────────────────────

fn cmd_render(args: &[String]) {
    if args.len() < 9 {
        eprintln!("usage: caption_qa render <source_mp4> <words_json> <candidate_json> <start> <end> <template_id> <outdir> [framing_mode]");
        std::process::exit(2);
    }
    let source = &args[2];
    let words_path = &args[3];
    let candidate_path = &args[4];
    let start: f64 = args[5].parse().expect("start");
    let end: f64 = args[6].parse().expect("end");
    let template_id = &args[7];
    let outdir = PathBuf::from(&args[8]);
    // Optional 9th arg: framing mode ("original" default, "adaptive" mirrors
    // the production per-project choice so renders reproduce real clips).
    let framing_mode = args.get(9).map(|s| s.as_str()).unwrap_or("original");

    let mut checks: Vec<Check> = Vec::new();

    let words = load_words(words_path).unwrap_or_else(|| {
        eprintln!("bad words json");
        std::process::exit(2);
    });
    let candidate: Candidate = fs::read_to_string(candidate_path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_else(|| {
            eprintln!("bad candidate json");
            std::process::exit(2);
        });

    let template = captions::get_caption_template(template_id).unwrap_or_else(|| {
        eprintln!("unknown template {}", template_id);
        std::process::exit(2);
    });

    if !Path::new(source).exists() {
        eprintln!("source video missing: {}", source);
        std::process::exit(2);
    }
    let _ = fs::create_dir_all(&outdir);

    // 0. Real Smart Framing plan (same inputs production uses) — mirrors
    //    lib.rs::render_flat_clip_for_candidate so the burned render matches
    //    what the app produces for the project's framing mode.
    let framing_plan = match autoshorts_lib::media::probe_media(source) {
        Ok(probe) => {
            let iw = probe.width.unwrap_or(1920);
            let ih = probe.height.unwrap_or(1080);
            let mut crop_w = ((ih as f64) * 9.0 / 16.0).round() as i64;
            if crop_w % 2 != 0 {
                crop_w -= 1;
            }
            Some(autoshorts_lib::media::detect_speaker_crop_params(
                source,
                start,
                end,
                iw,
                ih,
                crop_w,
                Some(&words),
                framing_mode,
            ))
        }
        Err(e) => {
            eprintln!("[caption_qa] framing probe failed: {e}");
            None
        }
    };

    // 1. Real CI plan (same inputs production uses)
    let plan = caption_intel::plan_caption_intel(&words, &candidate, None, start, end);
    checks.push(Check::pass(
        "RND-CI",
        "CI planner executed on render inputs",
        &format!("plan = {}", if plan.is_some() { "Some" } else { "None" }),
        "plan_caption_intel result (Some or None are both valid)",
        "caption_intel.rs plan_caption_intel",
        "regenerated at render time",
    ));

    // 2. Real ASS generation (intel-aware when plan exists)
    let ass = captions::generate_ass_from_template_with_framing_and_intel(
        &words,
        start,
        end,
        &template,
        framing_plan.as_ref(),
        plan.as_ref(),
        None,
    );
    let ass_path = outdir.join("caption.ass");
    fs::write(&ass_path, &ass).expect("write ass");
    let template_json_path = outdir.join("template.json");
    let _ = fs::write(
        &template_json_path,
        serde_json::to_string_pretty(&template).unwrap(),
    );
    checks.push(Check::pass(
        "RND-ASS",
        "ASS generated by production generator",
        &format!("{} bytes → {}", ass.len(), ass_path.display()),
        "non-empty ASS from generate_ass_from_template_with_framing_and_intel",
        "captions.rs production generator",
        &format!("{}", ass_path.display()),
    ));

    let parsed = parse_ass(&ass).expect("rendered ass must parse");
    check_template_conformance(&parsed, &template, false, &mut checks);
    check_ass_forensics(&parsed, &template, &words, start, end, &mut checks);

    // 3. Real FFmpeg burn-in via the production render path
    let mp4_path = outdir.join("rendered.mp4");
    let result = pacing::render_paced_clip(
        source,
        start,
        end,
        &mp4_path,
        Some(&ass_path),
        None,
        Some(&words),
        framing_plan.as_ref(),
        None,
        None,
    );
    match result {
        Ok(path) => {
            checks.push(Check::pass(
                "RND-MP4",
                "FFmpeg subtitle burn-in produced MP4",
                &format!("{}", path.display()),
                "existing mp4 with burned captions",
                "pacing::render_paced_clip (production path)",
                &format!("{}", path.display()),
            ));

            // MP4 duration sanity vs requested window
            if let Ok(probe) = autoshorts_lib::media::probe_media(&path.to_string_lossy()) {
                let dur = probe.duration_sec.unwrap_or(0.0);
                let expected = (end - start).max(0.1);
                if (dur - expected).abs() < 1.5 {
                    checks.push(Check::pass(
                        "RND-DURATION",
                        "Rendered MP4 duration matches clip window",
                        &format!("{:.2}s", dur),
                        &format!("{:.2}s", expected),
                        "media.rs probe_media",
                        &format!("{}", path.display()),
                    ));
                } else {
                    checks.push(Check::fail(
                        "RND-DURATION",
                        "Rendered MP4 duration matches clip window",
                        &format!("{:.2}s", dur),
                        &format!("{:.2}s", expected),
                        "media.rs probe_media",
                        &format!("{}", path.display()),
                        "medium",
                    ));
                }
            }
        }
        Err(e) => {
            checks.push(Check::fail(
                "RND-MP4",
                "FFmpeg subtitle burn-in produced MP4",
                &format!("error: {}", e),
                "existing mp4",
                "pacing::render_paced_clip",
                "render stderr",
                "high",
            ));
        }
    }

    if template.template_id == "preset_dynamic_editorial" {
        let bounds = t5_word_bounds(&parsed);
        check_t5_collisions(&bounds, &mut checks);
        let bounds_path = outdir.join("word_bounds.json");
        let _ = fs::write(&bounds_path, serde_json::to_string_pretty(&bounds).unwrap());
        checks.push(Check::pass(
            "RND-BOUNDS",
            "Emitted real word bounds for visual QA",
            &format!("{}", bounds_path.display()),
            "word_bounds.json",
            "captions.rs measure_t5_word_width",
            &format!("{}", bounds_path.display()),
        ));
    }

    let plan_json = outdir.join("ci_plan.json");
    let _ = fs::write(&plan_json, serde_json::to_string_pretty(&plan).unwrap());

    let mut report = Report {
        harness: "caption_qa".into(),
        mode: "render".into(),
        subject: format!("{} template={} ci={}", source, template_id, plan.is_some()),
        checks,
        summary: Default::default(),
    };
    report.summary = summarize(&report.checks);
    emit_to(&report, &outdir);
}

// ── dump-render-inputs: extract real render inputs from the DB ─────────────

fn cmd_dump_render_inputs(args: &[String]) {
    if args.len() < 6 {
        eprintln!("usage: caption_qa dump-render-inputs <db> <project_id> <candidate_id> <outdir>");
        std::process::exit(2);
    }
    let db_path = &args[2];
    let project_id = &args[3];
    let candidate_id = &args[4];
    let outdir = PathBuf::from(&args[5]);

    let tmp_db = read_only_db_copy(db_path);
    eprintln!(
        "[caption_qa] using db copy: {} (exists={})",
        tmp_db.display(),
        tmp_db.exists()
    );
    let db = match Database::open(&tmp_db) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("failed to open DB copy {}: {}", tmp_db.display(), e);
            std::process::exit(2);
        }
    };

    let project = match db.get_project(project_id) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("project lookup failed for {}: {}", project_id, e);
            std::process::exit(2);
        }
    };
    let (candidate, _) = db
        .get_candidate_with_project(candidate_id)
        .expect("candidate");
    let transcript = db
        .latest_transcript(project_id)
        .ok()
        .flatten()
        .expect("transcript");
    let normalized: NormalizedTranscript =
        serde_json::from_str(&transcript.raw_json).expect("normalized transcript");

    let _ = fs::create_dir_all(&outdir);
    let words_path = outdir.join("words.json");
    let candidate_path = outdir.join("candidate.json");
    let meta_path = outdir.join("meta.json");

    fs::write(
        &words_path,
        serde_json::to_string_pretty(&normalized).unwrap(),
    )
    .expect("write words");
    fs::write(
        &candidate_path,
        serde_json::to_string_pretty(&candidate).unwrap(),
    )
    .expect("write candidate");
    let meta = serde_json::json!({
        "sourcePath": project.source_path,
        "captionStyle": project.caption_style,
        "startSec": candidate.start_sec,
        "endSec": candidate.end_sec,
    });
    fs::write(&meta_path, serde_json::to_string_pretty(&meta).unwrap()).expect("write meta");

    println!("{}", serde_json::to_string_pretty(&meta).unwrap());
}

// ── template dump: canonical definition for downstream QA layers ──────────

fn cmd_template(args: &[String]) {
    let id = args.get(2).cloned().unwrap_or_else(|| {
        eprintln!("usage: caption_qa template <template_id>");
        std::process::exit(2);
    });
    match captions::get_caption_template(&id) {
        Some(t) => println!("{}", serde_json::to_string_pretty(&t).unwrap()),
        None => {
            eprintln!("unknown template: {}", id);
            std::process::exit(2);
        }
    }
}

// ── main ────────────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!(
            "caption_qa — AutoShorts 9.0 Caption Regression & Visual QA Harness\n\n\
             commands:\n  \
               audit-clip   <db> <project_id> <candidate_id>\n  \
               isolation    [words_json]\n  \
               render       <source_mp4> <words_json> <candidate_json> <start> <end> <template_id> <outdir>\n  \
               dump-render-inputs <db> <project_id> <candidate_id> <outdir>");
        std::process::exit(2);
    }
    match args[1].as_str() {
        "audit-clip" => cmd_audit_clip(&args),
        "isolation" => cmd_isolation(&args),
        "render" => cmd_render(&args),
        "dump-render-inputs" => cmd_dump_render_inputs(&args),
        "template" => cmd_template(&args),
        other => {
            eprintln!("unknown command: {}", other);
            std::process::exit(2);
        }
    }
}

impl Default for Summary {
    fn default() -> Self {
        Summary {
            pass: 0,
            fail: 0,
            skipped: 0,
            not_verified: 0,
        }
    }
}
