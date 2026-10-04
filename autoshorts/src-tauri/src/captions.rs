use crate::models::TranscriptWord;
pub use crate::t7_prosody::T7Boundary;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CaptionTemplate {
    #[serde(rename = "templateId")]
    pub template_id: String,
    pub name: String,
    pub segmentation: CaptionSegmentation,
    #[serde(rename = "globalStyles")]
    pub global_styles: CaptionGlobalStyles,
    #[serde(rename = "activeState")]
    pub active_state: CaptionActiveState,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CaptionSegmentation {
    #[serde(rename = "maxWordsPerLine")]
    pub max_words_per_line: usize,
    #[serde(rename = "maxLinesPerFrame")]
    pub max_lines_per_frame: usize,
    #[serde(rename = "case")]
    pub text_case: String, // "uppercase" | "sentence" | "normal"
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CaptionBackgroundBox {
    pub enabled: bool,
    pub color: String,
    pub opacity: f64,
    #[serde(rename = "paddingPx")]
    pub padding_px: u32,
    #[serde(rename = "borderRadiusPx")]
    pub border_radius_px: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CaptionGlobalStyles {
    #[serde(rename = "fontFamily")]
    pub font_family: String,
    #[serde(rename = "fontSize")]
    pub font_size: u32,
    #[serde(rename = "fontWeight")]
    pub font_weight: String,
    #[serde(rename = "fontColor")]
    pub font_color: String,
    pub alignment: String,
    #[serde(rename = "positionY")]
    pub position_y: f64,
    #[serde(rename = "strokeColor")]
    pub stroke_color: Option<String>,
    #[serde(rename = "strokeWidth")]
    pub stroke_width: Option<u32>,
    #[serde(rename = "shadowColor")]
    pub shadow_color: Option<String>,
    #[serde(rename = "shadowBlur")]
    pub shadow_blur: Option<f64>,
    #[serde(rename = "shadowOffsetY")]
    pub shadow_offset_y: Option<f64>,
    #[serde(rename = "backgroundBox")]
    pub background_box: Option<CaptionBackgroundBox>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AnimationConfig {
    #[serde(rename = "type")]
    pub animation_type: String,
    #[serde(rename = "scaleFactor")]
    pub scale_factor: Option<f64>,
    #[serde(rename = "durationMs")]
    pub duration_ms: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type")]
pub enum CaptionActiveState {
    #[serde(rename = "word_by_word_swap")]
    WordByWordSwap {
        #[serde(rename = "primaryHighlightColor")]
        primary_highlight_color: String,
        #[serde(rename = "secondaryHighlightColor")]
        secondary_highlight_color: String,
        animation: AnimationConfig,
    },
    #[serde(rename = "karaoke_progressive")]
    KaraokeProgressive {
        #[serde(rename = "highlightColor")]
        highlight_color: String,
        animation: AnimationConfig,
    },
    #[serde(rename = "speech_chunked_rolling")]
    SpeechChunkedRolling {
        #[serde(rename = "highlightColor")]
        highlight_color: String,
    },
    #[serde(rename = "static_opacity_reveal")]
    StaticOpacityReveal {
        #[serde(rename = "inactiveOpacity")]
        inactive_opacity: f64,
        #[serde(rename = "activeOpacity")]
        active_opacity: f64,
    },
    #[serde(rename = "smooth_fade")]
    SmoothFade {
        #[serde(rename = "highlightColor")]
        highlight_color: String,
        #[serde(rename = "durationMs")]
        duration_ms: u32,
    },
    #[serde(rename = "dynamic_editorial_kinetic")]
    DynamicEditorialKinetic {
        #[serde(rename = "baseColor")]
        base_color: String,
        #[serde(rename = "accentColor")]
        accent_color: String,
        #[serde(rename = "secondaryAccent")]
        secondary_accent: String,
        #[serde(rename = "entranceDurationMs")]
        entrance_duration_ms: u32,
        #[serde(rename = "exitFadeMs")]
        exit_fade_ms: u32,
    },
    #[serde(rename = "bhaukal_kinetic")]
    BhaukalKinetic {
        #[serde(rename = "baseColor")]
        base_color: String,
        #[serde(rename = "heroColor")]
        hero_color: String,
        #[serde(rename = "entranceDurationMs")]
        entrance_duration_ms: u32,
        #[serde(rename = "exitFadeMs")]
        exit_fade_ms: u32,
    },
}

pub fn all_caption_templates() -> Vec<CaptionTemplate> {
    vec![
        // 1. Hormozi Viral
        CaptionTemplate {
            template_id: "preset_viral_bold".to_string(),
            name: "Hormozi Viral".to_string(),
            segmentation: CaptionSegmentation {
                max_words_per_line: 2,
                max_lines_per_frame: 1,
                text_case: "uppercase".to_string(),
            },
            global_styles: CaptionGlobalStyles {
                font_family: "Bebas Neue".to_string(),
                font_size: 84,
                font_weight: "900".to_string(),
                font_color: "#FFFFFF".to_string(),
                alignment: "center".to_string(),
                position_y: 0.70,
                stroke_color: Some("#000000".to_string()),
                stroke_width: Some(4),
                shadow_color: Some("rgba(0, 0, 0, 0.8)".to_string()),
                shadow_blur: Some(10.0),
                shadow_offset_y: None,
                background_box: None,
            },
            active_state: CaptionActiveState::WordByWordSwap {
                primary_highlight_color: "#00FF66".to_string(),
                secondary_highlight_color: "#FFEA00".to_string(),
                animation: AnimationConfig {
                    animation_type: "scale_up".to_string(),
                    scale_factor: Some(1.15),
                    duration_ms: 80,
                },
            },
        },
        // 2. Narrative Pop
        CaptionTemplate {
            template_id: "preset_mrbeast_pop".to_string(),
            name: "Narrative Pop".to_string(),
            segmentation: CaptionSegmentation {
                max_words_per_line: 4,
                max_lines_per_frame: 2,
                text_case: "sentence".to_string(),
            },
            global_styles: CaptionGlobalStyles {
                font_family: "Montserrat".to_string(),
                font_size: 80,
                font_weight: "800".to_string(),
                font_color: "#FFFFFF".to_string(),
                alignment: "center".to_string(),
                position_y: 0.65,
                stroke_color: Some("#000000".to_string()),
                stroke_width: Some(5),
                shadow_color: Some("rgba(0,0,0,0.5)".to_string()),
                shadow_blur: None,
                shadow_offset_y: Some(4.0),
                background_box: None,
            },
            active_state: CaptionActiveState::KaraokeProgressive {
                highlight_color: "#FFFF00".to_string(),
                animation: AnimationConfig {
                    animation_type: "pop_bounce".to_string(),
                    scale_factor: None,
                    duration_ms: 100,
                },
            },
        },
        // 3. Minimal Capsule
        CaptionTemplate {
            template_id: "preset_minimal_capsule".to_string(),
            name: "Minimal Capsule".to_string(),
            segmentation: CaptionSegmentation {
                max_words_per_line: 5,
                max_lines_per_frame: 1,
                text_case: "normal".to_string(),
            },
            global_styles: CaptionGlobalStyles {
                font_family: "Inter".to_string(),
                font_size: 96,
                font_weight: "600".to_string(),
                font_color: "#F5F5F5".to_string(),
                alignment: "center".to_string(),
                position_y: 0.75,
                stroke_color: None,
                stroke_width: None,
                shadow_color: None,
                shadow_blur: None,
                shadow_offset_y: None,
                background_box: Some(CaptionBackgroundBox {
                    enabled: true,
                    color: "#000000".to_string(),
                    opacity: 0.65,
                    padding_px: 12,
                    border_radius_px: 8,
                }),
            },
            active_state: CaptionActiveState::StaticOpacityReveal {
                inactive_opacity: 0.40,
                active_opacity: 1.00,
            },
        },
        // 4. Cinematic Vlog
        CaptionTemplate {
            template_id: "preset_cinematic_vlog".to_string(),
            name: "Cinematic Vlog".to_string(),
            segmentation: CaptionSegmentation {
                max_words_per_line: 6,
                max_lines_per_frame: 2,
                text_case: "normal".to_string(),
            },
            global_styles: CaptionGlobalStyles {
                font_family: "Poppins".to_string(),
                font_size: 116,
                font_weight: "400".to_string(),
                font_color: "#FDFBF7".to_string(),
                alignment: "center".to_string(),
                position_y: 0.82,
                stroke_color: None,
                stroke_width: None,
                shadow_color: Some("rgba(0, 0, 0, 0.3)".to_string()),
                shadow_blur: Some(6.0),
                shadow_offset_y: Some(2.0),
                background_box: None,
            },
            active_state: CaptionActiveState::SmoothFade {
                highlight_color: "#FFFFFF".to_string(),
                duration_ms: 150,
            },
        },
        // 5. Dynamic Editorial Kinetic
        CaptionTemplate {
            template_id: "preset_dynamic_editorial".to_string(),
            name: "Dynamic Editorial Kinetic".to_string(),
            segmentation: CaptionSegmentation {
                max_words_per_line: 4,
                max_lines_per_frame: 1,
                text_case: "normal".to_string(),
            },
            global_styles: CaptionGlobalStyles {
                font_family: "Montserrat".to_string(),
                font_size: 62,
                font_weight: "800".to_string(),
                font_color: "#FFFFFF".to_string(),
                alignment: "center".to_string(),
                position_y: 0.65,
                stroke_color: None,
                stroke_width: None,
                shadow_color: None,
                shadow_blur: None,
                shadow_offset_y: None,
                background_box: None,
            },
            active_state: CaptionActiveState::DynamicEditorialKinetic {
                base_color: "#FFFFFF".to_string(),
                accent_color: "#FFE600".to_string(),
                secondary_accent: "#00E5FF".to_string(),
                entrance_duration_ms: 150,
                exit_fade_ms: 300,
            },
        },
        // 6. Bhaukal Caption (Template 6, independent peer of T1-T5)
        CaptionTemplate {
            template_id: "preset_bhaukal_caption".to_string(),
            name: "Bhaukal caption".to_string(),
            segmentation: CaptionSegmentation {
                max_words_per_line: 3,
                max_lines_per_frame: 2,
                text_case: "normal".to_string(),
            },
            global_styles: CaptionGlobalStyles {
                font_family: "Montserrat".to_string(),
                font_size: 52,
                font_weight: "800".to_string(),
                font_color: "#FFFFFF".to_string(),
                alignment: "center".to_string(),
                position_y: 0.55,
                stroke_color: Some("#000000".to_string()),
                stroke_width: Some(3),
                shadow_color: Some("rgba(0, 0, 0, 0.6)".to_string()),
                shadow_blur: Some(8.0),
                shadow_offset_y: Some(2.0),
                background_box: None,
            },
            active_state: CaptionActiveState::BhaukalKinetic {
                base_color: "#FFFFFF".to_string(),
                hero_color: "#FFFFFF".to_string(),
                entrance_duration_ms: 120,
                exit_fade_ms: 250,
            },
        },
        // 7. T7 Reference Style — phrase-chunked rolling captions.
        // Modeled frame-by-frame on the reference edit: short speech chunks
        // (1-3 words) appear as stable visual units; the previous chunk stays
        // WHITE above the newly arrived YELLOW chunk for the new chunk's
        // duration. Chunk boundaries follow speech rhythm (micro-pauses,
        // sentence/clause ends), never a fixed word count. Plain solid text
        // only — no stroke, shadow, or box. Colors, size and position are
        // measured off the reference frames (#FFFFFF / #F9EF07, cap-height
        // 77px on a 1080-wide canvas -> size 110; the block sits at 0.60 of
        // the 9:16 canvas, with a CapCut-tight 0.93em line pitch between the
        // two stacked lines). Font: Matt Bold (Matt_Trial Bold, weight 700),
        // cap-height ratio 0.70 — same as Poppins ExtraBold, so size 110
        // preserves the reference-matched cap height.
        CaptionTemplate {
            template_id: "preset_t7".to_string(),
            name: "T7 Reference Style".to_string(),
            segmentation: CaptionSegmentation {
                max_words_per_line: 3,
                max_lines_per_frame: 2,
                text_case: "normal".to_string(),
            },
            global_styles: CaptionGlobalStyles {
                font_family: "Matt_Trial-Bold".to_string(),
                font_size: 110,
                font_weight: "700".to_string(),
                font_color: "#FFFFFF".to_string(),
                alignment: "center".to_string(),
                // Reference caption band (averaged text-row profile of the
                // reference render) centers at 0.60 of the canvas.
                position_y: 0.60,
                stroke_color: None,
                stroke_width: None,
                shadow_color: None,
                shadow_blur: None,
                shadow_offset_y: None,
                background_box: None,
            },
            active_state: CaptionActiveState::SpeechChunkedRolling {
                highlight_color: "#F9EF07".to_string(),
            },
        },
    ]
}

pub fn get_caption_template(id: &str) -> Option<CaptionTemplate> {
    all_caption_templates()
        .into_iter()
        .find(|t| t.template_id == id)
}

/// Convert hex color string like "#FFFFFF" or "#00FF66" to ASS inline format `&HBBGGRR&`.
pub fn hex_to_ass_inline(hex: &str) -> String {
    let clean = hex.trim().trim_start_matches('#');
    if clean.len() == 6 {
        let r = &clean[0..2];
        let g = &clean[2..4];
        let b = &clean[4..6];
        format!("&H{}{}{}&", b, g, r)
    } else {
        "&HFFFFFF&".to_string()
    }
}

/// Convert hex color string like "#FFFFFF" to ASS header format `&H00BBGGRR`.
pub fn hex_to_ass_header(hex: &str) -> String {
    let clean = hex.trim().trim_start_matches('#');
    if clean.len() == 6 {
        let r = &clean[0..2];
        let g = &clean[2..4];
        let b = &clean[4..6];
        format!("&H00{}{}{}", b, g, r)
    } else {
        "&H00FFFFFF".to_string()
    }
}

/// Convert rgba string like "rgba(0, 0, 0, 0.8)" to ASS header format `&HAABBGGRR`.
pub fn rgba_to_ass_header(rgba_str: &str) -> String {
    let trimmed = rgba_str.trim();
    if trimmed.starts_with("rgba") {
        let inner = trimmed
            .trim_start_matches("rgba(")
            .trim_end_matches(')')
            .replace(' ', "");
        let parts: Vec<&str> = inner.split(',').collect();
        if parts.len() == 4 {
            let r: u8 = parts[0].parse().unwrap_or(0);
            let g: u8 = parts[1].parse().unwrap_or(0);
            let b: u8 = parts[2].parse().unwrap_or(0);
            let a_f: f64 = parts[3].parse().unwrap_or(1.0);
            // In ASS, alpha is 0x00 for completely opaque and 0xFF for completely transparent
            let ass_alpha = ((1.0 - a_f.clamp(0.0, 1.0)) * 255.0).round() as u8;
            return format!("&H{:02X}{:02X}{:02X}{:02X}", ass_alpha, b, g, r);
        }
    } else if trimmed.starts_with('#') {
        return hex_to_ass_header(trimmed);
    }
    "&H80000000".to_string()
}

/// Format seconds as ASS timestamp "H:MM:SS.cs".
pub fn format_ass_timestamp(secs: f64) -> String {
    let total_cs = (secs.max(0.0) * 100.0).round() as u64;
    let cs = total_cs % 100;
    let total_secs = total_cs / 100;
    let s = total_secs % 60;
    let total_mins = total_secs / 60;
    let m = total_mins % 60;
    let h = total_mins / 60;
    format!("{}:{:02}:{:02}.{:02}", h, m, s, cs)
}

/// Apply text casing (uppercase, sentence, normal).
fn apply_text_casing(text: &str, case_type: &str, is_first_in_sentence: bool) -> String {
    match case_type {
        "uppercase" => text.to_uppercase(),
        "sentence" => {
            if is_first_in_sentence {
                let mut chars = text.chars();
                match chars.next() {
                    None => String::new(),
                    Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
                }
            } else {
                text.to_string()
            }
        }
        _ => text.to_string(), // "normal"
    }
}

/// Computes ASS vertical margin (MarginV) from normalized positionY and font geometry.
/// For 1 line: block_height = FontSize
/// For 2 lines: block_height = FontSize * 2.2 (accounting for line-height and leading)
pub fn compute_margin_v(position_y: f64, font_size: u32, max_lines: usize) -> i64 {
    let block_height = if max_lines >= 2 {
        (font_size as f64) * 2.2
    } else {
        font_size as f64
    };
    ((1.0 - position_y) * 1920.0 - (block_height / 2.0))
        .round()
        .max(40.0) as i64
}

/// Subject face/head region bounding box (normalized 0.0..1.0 or pixel coordinates in 1080x1920 canvas).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SubjectFaceBounds {
    pub top: f64,
    pub bottom: f64,
    pub left: f64,
    pub right: f64,
}

/// Computes the visual bounding box (left, top, right, bottom in 1080x1920 pixels) of a caption template.
pub fn compute_caption_bounding_box(template: &CaptionTemplate) -> (f64, f64, f64, f64) {
    let center_y = template.global_styles.position_y * 1920.0;
    let block_height = if template.segmentation.max_lines_per_frame >= 2 {
        (template.global_styles.font_size as f64) * 2.2
    } else {
        template.global_styles.font_size as f64
    };
    let top = center_y - (block_height / 2.0);
    let bottom = center_y + (block_height / 2.0);
    let left = 80.0;
    let right = 1000.0;
    (left, top, right, bottom)
}

/// Applies a visual safety constraint preventing captions from visually covering a detected face/head region.
/// Conceptually: face region -> caption bounding box -> check overlap -> reduce caption size slightly
/// as primary solution, or adjust position within safety limits.
pub fn apply_subject_face_safeguard(
    template: &CaptionTemplate,
    face_bounds: Option<&SubjectFaceBounds>,
) -> CaptionTemplate {
    let mut adjusted = template.clone();
    let Some(face) = face_bounds else {
        return adjusted;
    };

    let face_bottom_px = if face.bottom <= 1.0 && face.bottom > 0.0 {
        face.bottom * 1920.0
    } else {
        face.bottom
    };

    let (_, caption_top_px, _, _) = compute_caption_bounding_box(&adjusted);
    let safety_buffer = 24.0;

    if face_bottom_px + safety_buffer > caption_top_px {
        let intrusion = (face_bottom_px + safety_buffer) - caption_top_px;
        // Primary solution: size reduction (up to 15%)
        let max_reduction = (adjusted.global_styles.font_size as f64 * 0.15).round() as u32;
        let size_reduction = ((intrusion / 1.5).round() as u32).min(max_reduction);
        adjusted.global_styles.font_size = adjusted
            .global_styles
            .font_size
            .saturating_sub(size_reduction);

        // If intrusion persists, use position adjustment (shift slightly down, max 0.88)
        let (_, new_top, _, _) = compute_caption_bounding_box(&adjusted);
        if face_bottom_px + safety_buffer > new_top {
            let remaining_gap = (face_bottom_px + safety_buffer) - new_top;
            let y_shift = (remaining_gap / 1920.0).min(0.06);
            adjusted.global_styles.position_y =
                (adjusted.global_styles.position_y + y_shift).min(0.88);
        }
    }

    adjusted
}

/// Resolves required bundled font file name for a template font family.
pub fn required_font_filename(font_family: &str) -> Option<&'static str> {
    let normalized = font_family.to_lowercase().replace([' ', '-', '_'], "");
    match normalized.as_str() {
        "bebasneue" => Some("BebasNeue-Regular.ttf"),
        "montserrat" => Some("Montserrat[wght].ttf"),
        "inter" => Some("Inter[opsz,wght].ttf"),
        "poppins" => Some("Poppins-Regular.ttf"),
        "matttrialbold" => Some("Matt_Trial-Bold.otf"),
        _ => None,
    }
}

/// Safely wraps formatted words into lines ensuring:
/// 1. Every generated line contains <= max_words_per_line (e.g. <= 6 for Cinematic Vlog)
/// 2. Every generated event contains <= max_lines_per_frame (e.g. <= 2)
pub fn wrap_words_into_lines(
    formatted_words: &[String],
    max_words_per_line: usize,
    max_lines_per_frame: usize,
) -> String {
    wrap_words_into_lines_with_hints(
        formatted_words,
        max_words_per_line,
        max_lines_per_frame,
        None,
    )
}

pub fn wrap_words_into_lines_with_hints(
    formatted_words: &[String],
    max_words_per_line: usize,
    max_lines_per_frame: usize,
    preferred_split_at: Option<usize>,
) -> String {
    let max_words_per_line = max_words_per_line.max(1);
    let max_lines_per_frame = max_lines_per_frame.max(1);
    let max_total_words = max_words_per_line * max_lines_per_frame;

    // Cap total words to the maximum allowed for this event
    let words = if formatted_words.len() > max_total_words {
        &formatted_words[..max_total_words]
    } else {
        formatted_words
    };

    let total = words.len();
    if max_lines_per_frame <= 1 || total <= max_words_per_line {
        return words.join(" ");
    }

    // For 2 lines: balanced split ensuring both lines <= max_words_per_line
    let min_line1 = total.saturating_sub(max_words_per_line).max(1);
    let max_line1 = max_words_per_line.min(total.saturating_sub(1));
    let target = (total + 1) / 2;

    let split_at = if let Some(pref) = preferred_split_at {
        if pref >= min_line1 && pref <= max_line1 && (total - pref) <= max_words_per_line {
            pref
        } else {
            target.max(min_line1).min(max_words_per_line)
        }
    } else {
        target.max(min_line1).min(max_words_per_line)
    };

    let line1 = words[..split_at].join(" ");
    let line2 = words[split_at..].join(" ");
    format!("{}\\N{}", line1, line2)
}

/// Speech-rhythm chunking for T7. Groups transcript words into short phrase
/// chunks (1-3 words) whose boundaries follow how the speaker actually
/// speaks: micro-pauses, sentence terminators and clause commas are primary
/// boundaries; word count is only a ceiling, never the driving rule.
///
/// This deliberately contrasts with fixed-size grouping (e.g. "every N words"
/// or character-count splits): a single word framed by pauses becomes its own
/// chunk, while three words delivered as one rhythm unit stay together.
pub fn segment_speech_chunks<'a>(
    words: &'a [(usize, &TranscriptWord)],
    max_words: usize,
    pause_threshold: f64,
    max_chunk_dur: f64,
    t7_boundaries: Option<&[T7Boundary]>,
) -> Vec<Vec<(usize, &'a TranscriptWord)>> {
    let max_words = max_words.max(1);
    let mut chunks: Vec<Vec<(usize, &TranscriptWord)>> = Vec::new();
    let mut current: Vec<(usize, &TranscriptWord)> = Vec::new();

    let has_alnum = |t: &str| t.chars().any(|c| c.is_alphanumeric());
    let mut gap_seq_idx: usize = 0;

    for &word_ref in words {
        let (_, word) = word_ref;
        if !has_alnum(&word.text) {
            // Standalone punctuation token (rare from Deepgram); punctuation
            // that matters is already attached to the preceding word.
            continue;
        }
        if let Some(&(prev_orig_idx, prev)) = current.last() {
            let pause = word.start - prev.end;
            let projected_dur = word.end - current[0].1.start;
            let ends_sentence = crate::transcription::ends_with_sentence_terminator(&prev.text);
            let ends_clause = prev.text.ends_with(',');
            let at_ceiling = current.len() >= max_words;
            let too_long = projected_dur > max_chunk_dur;

            // SOFT threshold modulation:
            // Baseline pause threshold is pause_threshold (e.g. 1.5s).
            // When t7_boundaries is present and p_boundary >= 0.65, reduce effective
            // pause threshold from 1.5s down to 0.9s for THAT gap only.
            // When t7_boundaries is None or confidence < 0.50, retain baseline threshold.
            let mut effective_pause_threshold = pause_threshold;
            if let Some(boundaries) = t7_boundaries {
                let matched = boundaries
                    .iter()
                    .find(|b| b.gap_idx == prev_orig_idx)
                    .or_else(|| boundaries.iter().find(|b| b.gap_idx == gap_seq_idx));

                if let Some(b) = matched {
                    if b.p_boundary >= 0.65 {
                        if (pause_threshold - 1.5).abs() < 1e-4 {
                            effective_pause_threshold = 0.9;
                        } else {
                            effective_pause_threshold = (pause_threshold * 0.6).min(0.9);
                        }
                    }
                }
            }

            // A boundary is a real prosodic break (pause/punctuation) or a
            // hard ceiling hit — never an arbitrary word position.
            if pause > effective_pause_threshold || ends_sentence || ends_clause || at_ceiling || too_long {
                chunks.push(std::mem::take(&mut current));
            }
            gap_seq_idx += 1;
        }
        current.push(word_ref);
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    chunks
}

/// T7 Phrase-Paired Unit representation.
/// T7 groups speech chunks into independent phrase units:
/// either a single emphatic chunk (Solo) or a two-chunk phrase pair (Pair).
/// When paired, Line 1 (first) appears in White, and Line 2 (second) arrives
/// in Yellow. When the unit ends, both lines clear together. Chunks are never
/// recycled across phrase units, eliminating conveyor-belt line jumps.
#[derive(Debug, Clone)]
pub enum T7PhraseUnit<'a> {
    Solo {
        chunk: Vec<(usize, &'a TranscriptWord)>,
        speech_start: f64,
        speech_end: f64,
    },
    Pair {
        first: Vec<(usize, &'a TranscriptWord)>,
        second: Vec<(usize, &'a TranscriptWord)>,
        first_start: f64,
        first_end: f64,
        second_start: f64,
        second_end: f64,
    },
}

pub fn group_t7_phrase_units<'a>(
    words: &'a [(usize, &TranscriptWord)],
    max_words_per_line: usize,
    max_chunk_dur: f64,
    max_pair_dur: f64,
    t7_boundaries: Option<&[T7Boundary]>,
) -> Vec<T7PhraseUnit<'a>> {
    let chunks = segment_speech_chunks(
        words,
        max_words_per_line,
        0.12, // micro-pause boundary (s)
        max_chunk_dur,
        t7_boundaries,
    );

    let mut units = Vec::new();
    let mut i = 0;
    while i < chunks.len() {
        let c1 = &chunks[i];
        let c1_start = c1[0].1.start;
        let c1_end = c1.last().unwrap().1.end;
        let c1_last_text = c1.last().unwrap().1.text.trim();
        let ends_sentence = crate::transcription::ends_with_sentence_terminator(c1_last_text);

        if i + 1 >= chunks.len() {
            units.push(T7PhraseUnit::Solo {
                chunk: c1.clone(),
                speech_start: c1_start,
                speech_end: c1_end,
            });
            i += 1;
            continue;
        }

        let c2 = &chunks[i + 1];
        let c2_start = c2[0].1.start;
        let c2_end = c2.last().unwrap().1.end;
        let pause = c2_start - c1_end;
        let pair_dur = c2_end - c1_start;

        let speaker_diff = match (
            c1.last().unwrap().1.speaker.as_deref(),
            c2[0].1.speaker.as_deref(),
        ) {
            (Some(s1), Some(s2)) => s1 != s2,
            _ => false,
        };

        // Form a solo chunk if:
        // 1. Current chunk ends a sentence (., !, ?)
        // 2. Significant pause to next chunk (> 0.35s)
        // 3. Speaker changes
        // 4. Combined pair duration would exceed safety ceiling (e.g. 2.6s)
        if ends_sentence || pause > 0.35 || speaker_diff || pair_dur > max_pair_dur {
            units.push(T7PhraseUnit::Solo {
                chunk: c1.clone(),
                speech_start: c1_start,
                speech_end: c1_end,
            });
            i += 1;
        } else {
            units.push(T7PhraseUnit::Pair {
                first: c1.clone(),
                second: c2.clone(),
                first_start: c1_start,
                first_end: c1_end,
                second_start: c2_start,
                second_end: c2_end,
            });
            i += 2; // Advance past both chunks: second chunk is never recycled!
        }
    }

    units
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EditorialMotion {
    SlideUp,
    SlideDown,
    SlideLeft,
    SlideRight,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TypographyRole {
    Emphasis,
    Primary,
    Secondary,
    Decorative,
}

#[derive(Debug, Clone)]
pub struct EditorialWordPlacement {
    pub x: i64,
    pub y: i64,
    pub motion: EditorialMotion,
    pub role: TypographyRole,
}

pub fn get_editorial_layout_variants(group_size: usize) -> Vec<Vec<EditorialWordPlacement>> {
    match group_size {
        1 => vec![
            // Variant 0: hero_center (preferred single-speaker center Y=1245)
            vec![EditorialWordPlacement {
                x: 540,
                y: 1245,
                motion: EditorialMotion::SlideUp,
                role: TypographyRole::Emphasis,
            }],
            // Variant 1: hero_upper
            vec![EditorialWordPlacement {
                x: 540,
                y: 1205,
                motion: EditorialMotion::SlideDown,
                role: TypographyRole::Primary,
            }],
            // Variant 2: hero_lower
            vec![EditorialWordPlacement {
                x: 540,
                y: 1285,
                motion: EditorialMotion::SlideUp,
                role: TypographyRole::Emphasis,
            }],
        ],
        2 => vec![
            // Variant 0: compact_stacked_pair
            vec![
                EditorialWordPlacement {
                    x: 540,
                    y: 1210,
                    motion: EditorialMotion::SlideRight,
                    role: TypographyRole::Primary,
                },
                EditorialWordPlacement {
                    x: 540,
                    y: 1280,
                    motion: EditorialMotion::SlideUp,
                    role: TypographyRole::Emphasis,
                },
            ],
            // Variant 1: compact_inline_pair
            vec![
                EditorialWordPlacement {
                    x: 450,
                    y: 1245,
                    motion: EditorialMotion::SlideDown,
                    role: TypographyRole::Emphasis,
                },
                EditorialWordPlacement {
                    x: 630,
                    y: 1245,
                    motion: EditorialMotion::SlideUp,
                    role: TypographyRole::Primary,
                },
            ],
            // Variant 2: compact_staggered_pair
            vec![
                EditorialWordPlacement {
                    x: 510,
                    y: 1215,
                    motion: EditorialMotion::SlideLeft,
                    role: TypographyRole::Secondary,
                },
                EditorialWordPlacement {
                    x: 570,
                    y: 1275,
                    motion: EditorialMotion::SlideRight,
                    role: TypographyRole::Emphasis,
                },
            ],
        ],
        3 => vec![
            // Variant 0: compact_hero_top_two_bottom
            vec![
                EditorialWordPlacement {
                    x: 540,
                    y: 1210,
                    motion: EditorialMotion::SlideRight,
                    role: TypographyRole::Emphasis,
                },
                EditorialWordPlacement {
                    x: 450,
                    y: 1280,
                    motion: EditorialMotion::SlideUp,
                    role: TypographyRole::Secondary,
                },
                EditorialWordPlacement {
                    x: 630,
                    y: 1280,
                    motion: EditorialMotion::SlideLeft,
                    role: TypographyRole::Primary,
                },
            ],
            // Variant 1: compact_two_top_hero_bottom
            vec![
                EditorialWordPlacement {
                    x: 450,
                    y: 1210,
                    motion: EditorialMotion::SlideLeft,
                    role: TypographyRole::Primary,
                },
                EditorialWordPlacement {
                    x: 630,
                    y: 1210,
                    motion: EditorialMotion::SlideDown,
                    role: TypographyRole::Secondary,
                },
                EditorialWordPlacement {
                    x: 540,
                    y: 1280,
                    motion: EditorialMotion::SlideUp,
                    role: TypographyRole::Emphasis,
                },
            ],
            // Variant 2: compact_stair
            vec![
                EditorialWordPlacement {
                    x: 480,
                    y: 1180,
                    motion: EditorialMotion::SlideRight,
                    role: TypographyRole::Secondary,
                },
                EditorialWordPlacement {
                    x: 540,
                    y: 1245,
                    motion: EditorialMotion::SlideLeft,
                    role: TypographyRole::Emphasis,
                },
                EditorialWordPlacement {
                    x: 600,
                    y: 1310,
                    motion: EditorialMotion::SlideUp,
                    role: TypographyRole::Primary,
                },
            ],
        ],
        _ => vec![
            // Variant 0: compact_2x2_grid
            vec![
                EditorialWordPlacement {
                    x: 450,
                    y: 1210,
                    motion: EditorialMotion::SlideRight,
                    role: TypographyRole::Primary,
                },
                EditorialWordPlacement {
                    x: 630,
                    y: 1210,
                    motion: EditorialMotion::SlideDown,
                    role: TypographyRole::Secondary,
                },
                EditorialWordPlacement {
                    x: 450,
                    y: 1280,
                    motion: EditorialMotion::SlideUp,
                    role: TypographyRole::Emphasis,
                },
                EditorialWordPlacement {
                    x: 630,
                    y: 1280,
                    motion: EditorialMotion::SlideLeft,
                    role: TypographyRole::Decorative,
                },
            ],
            // Variant 1: compact_staggered_2x2
            vec![
                EditorialWordPlacement {
                    x: 430,
                    y: 1210,
                    motion: EditorialMotion::SlideDown,
                    role: TypographyRole::Secondary,
                },
                EditorialWordPlacement {
                    x: 610,
                    y: 1210,
                    motion: EditorialMotion::SlideRight,
                    role: TypographyRole::Primary,
                },
                EditorialWordPlacement {
                    x: 470,
                    y: 1280,
                    motion: EditorialMotion::SlideLeft,
                    role: TypographyRole::Emphasis,
                },
                EditorialWordPlacement {
                    x: 650,
                    y: 1280,
                    motion: EditorialMotion::SlideUp,
                    role: TypographyRole::Decorative,
                },
            ],
            // Variant 2: compact_diamond
            vec![
                EditorialWordPlacement {
                    x: 540,
                    y: 1175,
                    motion: EditorialMotion::SlideRight,
                    role: TypographyRole::Primary,
                },
                EditorialWordPlacement {
                    x: 440,
                    y: 1245,
                    motion: EditorialMotion::SlideLeft,
                    role: TypographyRole::Emphasis,
                },
                EditorialWordPlacement {
                    x: 640,
                    y: 1245,
                    motion: EditorialMotion::SlideDown,
                    role: TypographyRole::Secondary,
                },
                EditorialWordPlacement {
                    x: 540,
                    y: 1315,
                    motion: EditorialMotion::SlideUp,
                    role: TypographyRole::Decorative,
                },
            ],
        ],
    }
}

// ── Template 6: Bhaukal Caption (independent peer of T1–T5) ───────────────
// Visual language (from reference video): predominantly white/near-white,
// hierarchy carried by font + scale + weight + phrase structure, NOT rainbow
// colors. Clean bold sans for supporting text; large elegant italic hero for
// important concepts / metrics / payoff. Restrained fade/slide only.
// Font note (approximation, not an exact-match claim): bundled Montserrat
// (sans) + Poppins rendered italic (hero/script approximation) + Inter
// (hook strip). All three resolve via required_font_filename(); T1–T5 fonts
// are untouched.

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BhaukalRole {
    Normal,
    Emphasis,
    Hero,
    Metric,
    Payoff,
}

#[derive(Debug, Clone)]
pub struct BhaukalPlacement {
    pub x: i64,
    pub y: i64,
    pub role: BhaukalRole,
}

/// Left / center-left biased lockups. Deliberately distinct coordinates from
/// T5's quadrant/diagonal placements so T6 has its own visual identity.
pub fn get_bhaukal_layout_variants(group_size: usize) -> Vec<Vec<BhaukalPlacement>> {
    match group_size {
        1 => vec![
            // Solo hero: open left space, mid-frame
            vec![BhaukalPlacement {
                x: 360,
                y: 1000,
                role: BhaukalRole::Hero,
            }],
            // Solo upper-left
            vec![BhaukalPlacement {
                x: 330,
                y: 900,
                role: BhaukalRole::Emphasis,
            }],
            // Solo central hero (used only when face-safe)
            vec![BhaukalPlacement {
                x: 540,
                y: 1050,
                role: BhaukalRole::Hero,
            }],
        ],
        2 => vec![
            // Context (small, upper) + keyword (hero, lower)
            vec![
                BhaukalPlacement {
                    x: 330,
                    y: 920,
                    role: BhaukalRole::Normal,
                },
                BhaukalPlacement {
                    x: 360,
                    y: 1080,
                    role: BhaukalRole::Hero,
                },
            ],
            // Stacked left pair
            vec![
                BhaukalPlacement {
                    x: 350,
                    y: 960,
                    role: BhaukalRole::Emphasis,
                },
                BhaukalPlacement {
                    x: 350,
                    y: 1120,
                    role: BhaukalRole::Normal,
                },
            ],
            // Mirrored pair (used when speaker occupies the left)
            vec![
                BhaukalPlacement {
                    x: 730,
                    y: 920,
                    role: BhaukalRole::Normal,
                },
                BhaukalPlacement {
                    x: 700,
                    y: 1080,
                    role: BhaukalRole::Hero,
                },
            ],
        ],
        _ => vec![
            // Progressive triple: context -> keyword -> hero phrase
            vec![
                BhaukalPlacement {
                    x: 320,
                    y: 880,
                    role: BhaukalRole::Normal,
                },
                BhaukalPlacement {
                    x: 340,
                    y: 1020,
                    role: BhaukalRole::Emphasis,
                },
                BhaukalPlacement {
                    x: 370,
                    y: 1180,
                    role: BhaukalRole::Hero,
                },
            ],
            // Compact left stack
            vec![
                BhaukalPlacement {
                    x: 350,
                    y: 920,
                    role: BhaukalRole::Normal,
                },
                BhaukalPlacement {
                    x: 350,
                    y: 1050,
                    role: BhaukalRole::Emphasis,
                },
                BhaukalPlacement {
                    x: 350,
                    y: 1190,
                    role: BhaukalRole::Hero,
                },
            ],
            // Mirrored triple for left-side speakers
            vec![
                BhaukalPlacement {
                    x: 740,
                    y: 880,
                    role: BhaukalRole::Normal,
                },
                BhaukalPlacement {
                    x: 720,
                    y: 1020,
                    role: BhaukalRole::Emphasis,
                },
                BhaukalPlacement {
                    x: 690,
                    y: 1180,
                    role: BhaukalRole::Hero,
                },
            ],
        ],
    }
}

/// Maps a Bhaukal role to (font_family, base_font_size, bold, italic,
/// casing, char_width_ratio). Hero/Metric/Payoff use Poppins italic as a
/// bundled approximation of the reference's elegant script/italic serif.
pub fn bhaukal_role_style(role: BhaukalRole) -> (&'static str, u32, i64, i64, &'static str, f64) {
    match role {
        BhaukalRole::Normal => ("Montserrat", 44u32, 1, 0, "normal", 0.62),
        BhaukalRole::Emphasis => ("Montserrat", 58u32, 1, 0, "normal", 0.60),
        BhaukalRole::Hero => ("Poppins", 76u32, 1, 1, "normal", 0.56),
        BhaukalRole::Metric => ("Poppins", 84u32, 1, 1, "normal", 0.54),
        BhaukalRole::Payoff => ("Poppins", 92u32, 1, 1, "normal", 0.54),
    }
}

fn bhaukal_face_center_x(face: &SubjectFaceBounds) -> f64 {
    let l = if face.left <= 1.0 && face.left >= 0.0 {
        face.left * 1080.0
    } else {
        face.left
    };
    let r = if face.right <= 1.0 && face.right > 0.0 {
        face.right * 1080.0
    } else {
        face.right
    };
    (l + r) / 2.0
}

fn bhaukal_face_bottom_px(face: &SubjectFaceBounds) -> f64 {
    if face.bottom <= 1.0 && face.bottom > 0.0 {
        face.bottom * 1920.0
    } else {
        face.bottom
    }
}

/// T6-specific positioning layer. Consumes (not redesigns) framing data:
/// active segment, face bounds, DualStack seam corridor. Never covers a face
/// to match the reference; falls back to a safe position instead.
pub fn bhaukal_safe_position(
    base: &BhaukalPlacement,
    segment: Option<&crate::media::LayoutSegment>,
    framing_plan: Option<&crate::media::SmartFramingPlan>,
) -> BhaukalPlacement {
    let mut out = base.clone();
    // Mirror to open side when the speaker occupies the placement side.
    let face_opt: Option<&SubjectFaceBounds> = match segment {
        Some(crate::media::LayoutSegment::Single { face_bounds, .. }) => face_bounds.as_ref(),
        Some(crate::media::LayoutSegment::DualStack { .. }) => None,
        None => framing_plan.and_then(|p| {
            if p.segments.is_empty() {
                p.face_bounds.as_ref()
            } else {
                None
            }
        }),
    };
    if let Some(face) = face_opt {
        let cx = bhaukal_face_center_x(face);
        let base_left_side = out.x < 540;
        let face_left_side = cx < 540.0;
        if base_left_side == face_left_side {
            // Speaker sits on our side: mirror horizontally into open space.
            out.x = (1080 - out.x).clamp(80, 1000);
        }
        let face_bottom = bhaukal_face_bottom_px(face);
        let word_top = (out.y - 40) as f64;
        if face_bottom + 24.0 > word_top {
            let push = ((face_bottom + 24.0) - word_top).round() as i64;
            out.y = (out.y + push).min(1500);
        }
    }
    if let Some(crate::media::LayoutSegment::DualStack {
        top_face_bounds,
        bottom_face_bounds,
        ..
    }) = segment
    {
        let s_top = if let Some(f) = top_face_bounds {
            (bhaukal_face_bottom_px(f) + 24.0).clamp(800.0, 960.0)
        } else {
            880.0
        };
        let s_bot = if let Some(f) = bottom_face_bounds {
            let ft = if f.top <= 1.0 && f.top > 0.0 {
                f.top * 1920.0
            } else {
                f.top
            };
            let fb = bhaukal_face_bottom_px(f);
            (ft - 0.20 * (fb - ft) - 10.0).clamp(960.0, 1120.0)
        } else {
            1040.0
        };
        if s_top < s_bot {
            out.y = out.y.clamp(s_top.round() as i64, s_bot.round() as i64);
        } else {
            out.y = out.y.clamp(880, 1040);
        }
    }
    out.x = out.x.clamp(80, 1000);
    out.y = out.y.clamp(400, 1500);
    out
}

fn bhaukal_is_metric_token(text: &str) -> bool {
    text.chars().any(|c| c.is_ascii_digit()) || text.contains('$') || text.contains('%')
}

/// Generates restrained dual-typography ASS subtitles for Template 6.
/// Phrase groups of 1–3 words; each group renders as progressive build steps
/// (context -> keyword -> completed phrase) using REAL word timings. An
/// optional persistent hook strip (Layer 1, top) renders only when Caption
/// Intelligence supplies hook word indices.
pub fn generate_bhaukal_ass(
    words: &[TranscriptWord],
    start_sec: f64,
    end_sec: f64,
    template: &CaptionTemplate,
) -> String {
    generate_bhaukal_ass_with_framing(words, start_sec, end_sec, template, None)
}

pub fn generate_bhaukal_ass_with_framing(
    words: &[TranscriptWord],
    start_sec: f64,
    end_sec: f64,
    template: &CaptionTemplate,
    framing_plan: Option<&crate::media::SmartFramingPlan>,
) -> String {
    generate_bhaukal_ass_with_framing_and_intel(
        words,
        start_sec,
        end_sec,
        template,
        framing_plan,
        None,
    )
}

pub fn generate_bhaukal_ass_with_framing_and_intel(
    words: &[TranscriptWord],
    start_sec: f64,
    end_sec: f64,
    template: &CaptionTemplate,
    framing_plan: Option<&crate::media::SmartFramingPlan>,
    caption_intel: Option<&crate::caption_intel::CaptionIntelPlan>,
) -> String {
    let candidate_words: Vec<(usize, &TranscriptWord)> = words
        .iter()
        .enumerate()
        .filter(|(_, w)| w.end > start_sec && w.start < end_sec)
        .collect();
    if candidate_words.is_empty() {
        return String::new();
    }
    let (base_color, hero_color, entrance_duration_ms, exit_fade_ms) = match &template.active_state
    {
        CaptionActiveState::BhaukalKinetic {
            base_color,
            hero_color,
            entrance_duration_ms,
            exit_fade_ms,
        } => (
            base_color.as_str(),
            hero_color.as_str(),
            *entrance_duration_ms,
            *exit_fade_ms,
        ),
        _ => ("#FFFFFF", "#FFFFFF", 120, 250),
    };
    let base_inline = hex_to_ass_inline(base_color);
    let hero_inline = hex_to_ass_inline(hero_color);
    let clip_dur = (end_sec - start_sec).max(0.1);

    let mut ass = format!(
        r#"[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Montserrat,52,&H00FFFFFF,&H0000FFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,2,5,80,80,80,1
Style: HookStrip,Inter,34,&H00FFFFFF,&H0000FFFF,&H64000000,&H96000000,-1,0,0,0,100,100,0,0,1,2,1,8,60,60,60,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"#
    );

    // Persistent hook/title strip (T6-only layer): small clean sans at top.
    if let Some(plan) = caption_intel {
        if !plan.hook_word_indices.is_empty() {
            let mut hook_text: String = plan
                .hook_word_indices
                .iter()
                .filter_map(|gi| words.get(*gi))
                .map(|w| {
                    w.text
                        .trim()
                        .trim_matches(|c| c == '{' || c == '}')
                        .to_string()
                })
                .filter(|t| !t.is_empty())
                .collect::<Vec<_>>()
                .join(" ");
            hook_text.truncate(120);
            if !hook_text.trim().is_empty() {
                let clean = hook_text.replace('{', "").replace('}', "");
                ass.push_str(&format!(
                    "Dialogue: 1,{},{},HookStrip,,0,0,0,,{}\n",
                    format_ass_timestamp(0.0),
                    format_ass_timestamp(clip_dur),
                    clean
                ));
            }
        }
    }

    // Speech-driven phrase groups: ~1–3 words per kinetic card.
    let conjunctions = [
        "and", "but", "or", "so", "because", "when", "if", "that", "with", "from", "into", "every",
        "on", "your", "my", "our", "their", "this", "which", "aur", "lekin", "par", "ki", "toh",
        "kyunki", "agar", "jab", "tab",
    ];
    let mut groups: Vec<Vec<(usize, &TranscriptWord)>> = Vec::new();
    let mut curr: Vec<(usize, &TranscriptWord)> = Vec::new();
    for &(idx, word) in &candidate_words {
        if let Some(&(_, prev)) = curr.last() {
            let pause = word.start - prev.end;
            let duration = word.end - curr[0].1.start;
            let count = curr.len();
            let clean_w = word
                .text
                .to_lowercase()
                .trim_matches(|c: char| !c.is_alphanumeric())
                .to_string();
            let ends_punct = crate::transcription::ends_with_sentence_terminator(&prev.text);
            let ends_clause = prev.text.ends_with(',');
            let is_pause = pause > 0.28;
            let reaches_max = count >= 3;
            let is_conjunction = count >= 1 && conjunctions.contains(&clean_w.as_str());
            let max_dur = duration > 1.8;
            if is_pause
                || ends_punct
                || reaches_max
                || max_dur
                || (count >= 2 && (ends_clause || is_conjunction))
            {
                groups.push(std::mem::take(&mut curr));
            }
        }
        curr.push((idx, word));
    }
    if !curr.is_empty() {
        groups.push(curr);
    }

    for (group_idx, group) in groups.iter().enumerate() {
        if group.is_empty() {
            continue;
        }
        let k = group.len().clamp(1, 3);
        let variants = get_bhaukal_layout_variants(k);
        let placements = &variants[group_idx % variants.len()];
        let group_end = group.last().unwrap().1.end;
        let group_rel_end = (group_end - start_sec).min(clip_dur);
        let group_mid = (group.first().unwrap().1.start + group.last().unwrap().1.end) / 2.0;
        let segment = framing_plan.and_then(|plan| plan.segment_at(group_mid - start_sec));

        // Resolve per-word roles: intel decides WHAT is important, T6 decides HOW it looks.
        let mut roles: Vec<BhaukalRole> = placements.iter().map(|p| p.role).collect();
        if caption_intel.is_none() {
            // Restrained fallback without semantic evidence: never hero without proof.
            for r in roles.iter_mut() {
                if matches!(
                    r,
                    BhaukalRole::Hero | BhaukalRole::Metric | BhaukalRole::Payoff
                ) {
                    *r = BhaukalRole::Emphasis;
                }
            }
        }
        if let Some(plan) = caption_intel {
            for (i, (global_idx, w)) in group.iter().enumerate() {
                let in_payoff = plan.payoff_word_indices.contains(global_idx);
                let in_emph = plan.emphasis_word_indices.contains(global_idx);
                let metric = bhaukal_is_metric_token(&w.text);
                if in_payoff {
                    roles[i] = if metric {
                        BhaukalRole::Metric
                    } else {
                        BhaukalRole::Payoff
                    };
                } else if metric && in_emph {
                    roles[i] = BhaukalRole::Metric;
                } else if in_emph && group.len() == 1 {
                    roles[i] = BhaukalRole::Hero;
                } else if in_emph {
                    // Last emphasized word of a build becomes the hero of that card.
                    let is_last_emph = !group[i + 1..]
                        .iter()
                        .any(|(gi, _)| plan.emphasis_word_indices.contains(gi));
                    roles[i] = if is_last_emph {
                        BhaukalRole::Hero
                    } else {
                        BhaukalRole::Emphasis
                    };
                }
            }
        }
        // Cap hero dominance: at most one Metric/Payoff + one Hero per card.
        {
            let mut hero_seen = 0;
            for r in roles.iter_mut() {
                if matches!(
                    r,
                    BhaukalRole::Hero | BhaukalRole::Metric | BhaukalRole::Payoff
                ) {
                    hero_seen += 1;
                    if hero_seen > 2 {
                        *r = BhaukalRole::Emphasis;
                    }
                }
            }
        }

        // Progressive construction: step j shows words[0..=j] with real timings.
        // Steps replace (never overlap): step j holds until the next word
        // starts, so no animation ever spans a removed pacing region.
        for (step, &(_, step_word)) in group.iter().enumerate() {
            let step_rel_start = (step_word.start - start_sec).max(0.0);
            let hold_until = group
                .get(step + 1)
                .map(|(_, next)| (next.start - start_sec).max(0.0))
                .unwrap_or(group_rel_end);
            let step_rel_end = hold_until.max(step_rel_start + 0.08).min(clip_dur);
            if step_rel_end <= step_rel_start {
                continue;
            }
            let mut parts: Vec<String> = Vec::new();
            let mut est_width: f64 = 0.0;
            for (i, (_, w)) in group.iter().enumerate().take(step + 1) {
                let role = roles[i.min(roles.len() - 1)];
                let (font_family, base_fs, bold_val, italic_val, _casing, char_ratio) =
                    bhaukal_role_style(role);
                let color_inline = match role {
                    BhaukalRole::Normal | BhaukalRole::Emphasis => &base_inline,
                    BhaukalRole::Hero | BhaukalRole::Metric | BhaukalRole::Payoff => &hero_inline,
                };
                let clean_raw = w.text.trim().replace('{', "").replace('}', "");
                est_width +=
                    (clean_raw.chars().count().max(1) as f64) * char_ratio * (base_fs as f64)
                        + 14.0;
                parts.push(format!(
                    r"{{\fn{}\fs{}\b{}\i{}\c{}\3c&H000000&\bord3\shad2\4c&H80000000&}}{}",
                    font_family, base_fs, bold_val, italic_val, color_inline, clean_raw
                ));
            }
            // Width safety: shrink hero sizes proportionally if the lockup overflows.
            let mut size_scale = 1.0;
            if est_width > 920.0 {
                size_scale = (920.0 / est_width).max(0.45);
            }
            let line_text = if (size_scale - 1.0).abs() < f64::EPSILON {
                parts.join(r"{\c&HFFFFFF&} ")
            } else {
                // Re-emit with scaled sizes (deterministic, no guessing).
                let mut scaled: Vec<String> = Vec::new();
                for (i, (_, w)) in group.iter().enumerate().take(step + 1) {
                    let role = roles[i.min(roles.len() - 1)];
                    let (font_family, base_fs, bold_val, italic_val, _casing, _ratio) =
                        bhaukal_role_style(role);
                    let fs = ((base_fs as f64 * size_scale).round() as u32).max(24);
                    let color_inline = match role {
                        BhaukalRole::Normal | BhaukalRole::Emphasis => &base_inline,
                        BhaukalRole::Hero | BhaukalRole::Metric | BhaukalRole::Payoff => {
                            &hero_inline
                        }
                    };
                    let clean_raw = w.text.trim().replace('{', "").replace('}', "");
                    scaled.push(format!(
                        r"{{\fn{}\fs{}\b{}\i{}\c{}\3c&H000000&\bord3\shad2\4c&H80000000&}}{}",
                        font_family, fs, bold_val, italic_val, color_inline, clean_raw
                    ));
                }
                scaled.join(r"{\c&HFFFFFF&} ")
            };

            let anchor = bhaukal_safe_position(
                &placements[step.min(placements.len() - 1)],
                segment,
                framing_plan,
            );
            // Restrained entrance: 30px slide-up + soft fade (no bounce/zoom/karaoke).
            let move_tag = format!(
                r"\move({},{},{},{},0,{})",
                anchor.x,
                anchor.y + 30,
                anchor.x,
                anchor.y,
                entrance_duration_ms
            );
            let word_dur_ms = ((step_rel_end - step_rel_start) * 1000.0) as u32;
            let actual_fade = exit_fade_ms
                .min(word_dur_ms.saturating_sub(entrance_duration_ms + 40))
                .max(60);
            let event_text = format!(
                r"{{\an5{move_tag}\alpha&HFF&\t(0,60,0.5,\alpha&H00&)\fad(0,{actual_fade})}}{line_text}"
            );
            ass.push_str(&format!(
                "Dialogue: 0,{},{},Default,,0,0,0,,{}\n",
                format_ass_timestamp(step_rel_start),
                format_ass_timestamp(step_rel_end),
                event_text
            ));
        }
    }

    ass
}

/// Generates multi-object kinetic editorial ASS subtitles for Template 5.
pub fn generate_dynamic_editorial_ass(
    words: &[TranscriptWord],
    start_sec: f64,
    end_sec: f64,
    template: &CaptionTemplate,
) -> String {
    generate_dynamic_editorial_ass_with_framing(words, start_sec, end_sec, template, None)
}

/// Measures the rendered bounding box width and height for a T5 word.
/// Uses character-specific, font-specific proportional widths plus a conservative safety margin
/// to ensure words never collide when packed compactly.
pub fn measure_t5_word_width(text: &str, role: TypographyRole, font_size: u32) -> (f64, f64) {
    let fs = font_size as f64;
    let mut total_em = 0.0f64;

    for ch in text.chars() {
        let em = match role {
            TypographyRole::Emphasis => {
                // Bebas Neue: condensed uppercase gothic
                match ch {
                    'I' | '1' | '.' | ',' | '\'' | '!' | '|' | ':' | ';' => 0.28,
                    'M' | 'W' | '@' | '%' => 0.72,
                    'J' | 'L' => 0.38,
                    ' ' => 0.30,
                    _ => 0.50,
                }
            }
            TypographyRole::Primary => {
                // Montserrat: wide geometric sans-serif (bold)
                match ch {
                    'i' | 'l' | '1' | '.' | ',' | '\'' | '!' | '|' | ':' | ';' => 0.35,
                    'I' | 'j' | 'r' | 't' | 'f' => 0.42,
                    'm' | 'w' | 'M' | 'W' | '@' | '%' => 0.95,
                    'A'..='Z' => 0.76,
                    'a'..='z' => 0.64,
                    '0'..='9' => 0.68,
                    ' ' => 0.35,
                    _ => 0.62,
                }
            }
            TypographyRole::Secondary => {
                // Inter: neo-grotesque sans-serif (regular)
                match ch {
                    'i' | 'l' | '1' | '.' | ',' | '\'' | '!' | '|' | ':' | ';' => 0.30,
                    'j' | 'r' | 't' | 'f' => 0.38,
                    'm' | 'w' | 'M' | 'W' | '@' | '%' => 0.88,
                    'A'..='Z' => 0.70,
                    'a'..='z' => 0.56,
                    '0'..='9' => 0.60,
                    ' ' => 0.30,
                    _ => 0.54,
                }
            }
            TypographyRole::Decorative => {
                // Poppins: geometric sans-serif (italic)
                match ch {
                    'i' | 'l' | '1' | '.' | ',' | '\'' | '!' | '|' | ':' | ';' => 0.32,
                    'j' | 'r' | 't' | 'f' => 0.40,
                    'm' | 'w' | 'M' | 'W' | '@' | '%' => 0.90,
                    'A'..='Z' => 0.74,
                    'a'..='z' => 0.60,
                    '0'..='9' => 0.64,
                    ' ' => 0.32,
                    _ => 0.56,
                }
            }
        };
        total_em += em;
    }

    let conservative_w = (total_em * 1.08 * fs).max(fs * 0.4);
    let height = fs * 1.05;
    (conservative_w, height)
}

pub fn generate_dynamic_editorial_ass_with_framing(
    words: &[TranscriptWord],
    start_sec: f64,
    end_sec: f64,
    template: &CaptionTemplate,
    framing_plan: Option<&crate::media::SmartFramingPlan>,
) -> String {
    generate_dynamic_editorial_ass_with_framing_and_intel(
        words,
        start_sec,
        end_sec,
        template,
        framing_plan,
        None,
    )
}

pub fn generate_dynamic_editorial_ass_with_framing_and_intel(
    words: &[TranscriptWord],
    start_sec: f64,
    end_sec: f64,
    template: &CaptionTemplate,
    framing_plan: Option<&crate::media::SmartFramingPlan>,
    caption_intel: Option<&crate::caption_intel::CaptionIntelPlan>,
) -> String {
    let candidate_words: Vec<(usize, &TranscriptWord)> = words
        .iter()
        .enumerate()
        .filter(|(_, w)| w.end > start_sec && w.start < end_sec)
        .collect();

    if candidate_words.is_empty() {
        return String::new();
    }

    let (base_color, accent_color, secondary_accent, entrance_duration_ms, exit_fade_ms) =
        match &template.active_state {
            CaptionActiveState::DynamicEditorialKinetic {
                base_color,
                accent_color,
                secondary_accent,
                entrance_duration_ms,
                exit_fade_ms,
            } => (
                base_color.as_str(),
                accent_color.as_str(),
                secondary_accent.as_str(),
                *entrance_duration_ms,
                *exit_fade_ms,
            ),
            _ => ("#FFFFFF", "#FFE600", "#00E5FF", 150, 300),
        };

    let base_inline = hex_to_ass_inline(base_color);
    let accent_inline = hex_to_ass_inline(accent_color);
    let secondary_inline = hex_to_ass_inline(secondary_accent);

    let mut ass = format!(
        r#"[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Montserrat,62,&H00FFFFFF,&H0000FFFF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,0,0,5,80,80,80,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"#
    );

    let conjunctions = [
        "and", "but", "or", "so", "because", "when", "if", "that", "with", "from", "into", "every",
        "on", "your", "my", "our", "their", "this", "which", "aur", "lekin", "par", "ki", "toh",
        "kyunki", "agar", "jab", "tab",
    ];

    let mut groups: Vec<Vec<(usize, &TranscriptWord)>> = Vec::new();
    let mut curr_group: Vec<(usize, &TranscriptWord)> = Vec::new();

    for &(idx, word) in &candidate_words {
        if let Some(&(_, prev)) = curr_group.last() {
            let pause = word.start - prev.end;
            let duration = word.end - curr_group[0].1.start;
            let count = curr_group.len();

            let clean_w = word
                .text
                .to_lowercase()
                .trim_matches(|c: char| !c.is_alphanumeric())
                .to_string();
            let ends_punct = crate::transcription::ends_with_sentence_terminator(&prev.text);
            let ends_clause = prev.text.ends_with(',');
            let is_pause = pause > 0.28;
            let reaches_max = count >= 4;
            let is_conjunction = count >= 2 && conjunctions.contains(&clean_w.as_str());
            let max_dur = duration > 2.0;

            if is_pause
                || ends_punct
                || reaches_max
                || max_dur
                || (count >= 2 && (ends_clause || is_conjunction))
            {
                groups.push(std::mem::take(&mut curr_group));
            }
        }
        curr_group.push((idx, word));
    }
    if !curr_group.is_empty() {
        groups.push(curr_group);
    }

    struct PreparedWord<'a> {
        _global_idx: usize,
        word: &'a TranscriptWord,
        role: TypographyRole,
        font_family: &'static str,
        font_size: u32,
        bold_val: u8,
        italic_val: u8,
        color_inline: &'a str,
        formatted_word: String,
        width: f64,
        height: f64,
    }

    for (group_idx, group) in groups.iter().enumerate() {
        if group.is_empty() {
            continue;
        }

        let group_end = group.last().unwrap().1.end;
        let group_rel_end = (group_end - start_sec).min(end_sec - start_sec);

        let group_mid = (group.first().unwrap().1.start + group.last().unwrap().1.end) / 2.0;
        let segment = framing_plan.and_then(|plan| plan.segment_at(group_mid - start_sec));
        let is_group_dual = matches!(segment, Some(crate::media::LayoutSegment::DualStack { .. }));

        let (safe_top, safe_bottom) = if let Some(crate::media::LayoutSegment::DualStack {
            top_face_bounds,
            bottom_face_bounds,
            ..
        }) = segment
        {
            let s_top = if let Some(top_f) = top_face_bounds {
                let top_face_bottom_px = if top_f.bottom <= 1.0 && top_f.bottom > 0.0 {
                    top_f.bottom * 1920.0
                } else {
                    top_f.bottom
                };
                (top_face_bottom_px + 24.0).clamp(800.0, 960.0)
            } else {
                880.0
            };

            let s_bot = if let Some(bot_f) = bottom_face_bounds {
                let bot_face_top_px = if bot_f.top <= 1.0 && bot_f.top > 0.0 {
                    bot_f.top * 1920.0
                } else {
                    bot_f.top
                };
                let bot_face_bottom_px = if bot_f.bottom <= 1.0 && bot_f.bottom > 0.0 {
                    bot_f.bottom * 1920.0
                } else {
                    bot_f.bottom
                };
                let fh = bot_face_bottom_px - bot_face_top_px;
                let hairline_px = bot_face_top_px - (0.20 * fh) - 10.0;
                hairline_px.clamp(960.0, 1120.0)
            } else {
                1040.0
            };

            if s_top >= s_bot {
                eprintln!(
                    "[DualFrame LAKCB] Warning: Safe Top ({:.1}) >= Safe Bottom ({:.1}), falling back to default seam corridor [880, 1040]",
                    s_top, s_bot
                );
                (880, 1040)
            } else {
                (s_top.round() as i64, s_bot.round() as i64)
            }
        } else {
            (880, 1040)
        };

        let n_words = group.len();
        let mut prepared: Vec<PreparedWord> = Vec::with_capacity(n_words);

        for (i, &(global_idx, w)) in group.iter().enumerate() {
            let mut role = match n_words {
                1 => TypographyRole::Emphasis,
                2 => {
                    let v = group_idx % 2;
                    if (i + v) % 2 == 1 {
                        TypographyRole::Emphasis
                    } else {
                        TypographyRole::Primary
                    }
                }
                3 => match i {
                    0 => TypographyRole::Secondary,
                    1 => TypographyRole::Emphasis,
                    _ => TypographyRole::Decorative,
                },
                4 => match i {
                    0 => TypographyRole::Secondary,
                    1 => TypographyRole::Primary,
                    2 => TypographyRole::Emphasis,
                    _ => TypographyRole::Decorative,
                },
                _ => match i % 3 {
                    1 => TypographyRole::Emphasis,
                    2 => TypographyRole::Primary,
                    _ => TypographyRole::Secondary,
                },
            };

            // Caption Intelligence 2.0: Elevate emphasized words to TypographyRole::Emphasis
            if let Some(plan) = caption_intel {
                if plan.emphasis_word_indices.contains(&global_idx) {
                    role = TypographyRole::Emphasis;
                }
            }

            let (font_family, base_font_size, bold_val, italic_val, color_inline, casing) =
                match role {
                    TypographyRole::Emphasis => (
                        "Bebas Neue",
                        88u32,
                        1,
                        0,
                        accent_inline.as_str(),
                        "uppercase",
                    ),
                    TypographyRole::Primary => {
                        ("Montserrat", 62u32, 1, 0, base_inline.as_str(), "uppercase")
                    }
                    TypographyRole::Secondary => {
                        ("Inter", 50u32, 0, 0, base_inline.as_str(), "normal")
                    }
                    TypographyRole::Decorative => (
                        "Poppins",
                        80u32,
                        1,
                        1,
                        secondary_inline.as_str(),
                        "uppercase",
                    ),
                };

            let clean_raw = w.text.trim().replace('{', "").replace('}', "");
            let formatted_word = if casing == "uppercase" {
                clean_raw.to_uppercase()
            } else {
                clean_raw
            };

            let (w_px, h_px) = measure_t5_word_width(&formatted_word, role, base_font_size);

            prepared.push(PreparedWord {
                _global_idx: global_idx,
                word: w,
                role,
                font_family,
                font_size: base_font_size,
                bold_val,
                italic_val,
                color_inline,
                formatted_word,
                width: w_px,
                height: h_px,
            });
        }

        let word_gap = 26.0f64;
        let line_gap = 14.0f64;

        // Partition words into compact lines based on word count & widths
        let lines: Vec<Vec<usize>> = match n_words {
            1 => vec![vec![0]],
            2 => {
                if prepared[0].width + word_gap + prepared[1].width <= 620.0 {
                    vec![vec![0, 1]]
                } else {
                    vec![vec![0], vec![1]]
                }
            }
            3 => {
                let total_3 =
                    prepared[0].width + prepared[1].width + prepared[2].width + 2.0 * word_gap;
                if total_3 <= 660.0 {
                    vec![vec![0, 1, 2]]
                } else if prepared[0].role == TypographyRole::Emphasis {
                    vec![vec![0], vec![1, 2]]
                } else if prepared[2].role == TypographyRole::Emphasis {
                    vec![vec![0, 1], vec![2]]
                } else {
                    let diff_1_2 = (prepared[0].width
                        - (prepared[1].width + word_gap + prepared[2].width))
                        .abs();
                    let diff_2_1 = ((prepared[0].width + word_gap + prepared[1].width)
                        - prepared[2].width)
                        .abs();
                    if diff_1_2 <= diff_2_1 {
                        vec![vec![0], vec![1, 2]]
                    } else {
                        vec![vec![0, 1], vec![2]]
                    }
                }
            }
            4 => {
                vec![vec![0, 1], vec![2, 3]]
            }
            _ => {
                let max_line_w = 720.0;
                let mut cur_line = Vec::new();
                let mut cur_w = 0.0;
                let mut result = Vec::new();

                for idx in 0..n_words {
                    let w_len = prepared[idx].width;
                    let needed = if cur_line.is_empty() {
                        w_len
                    } else {
                        word_gap + w_len
                    };
                    if !cur_line.is_empty() && (cur_w + needed > max_line_w || cur_line.len() >= 3)
                    {
                        result.push(cur_line);
                        cur_line = vec![idx];
                        cur_w = w_len;
                    } else {
                        cur_line.push(idx);
                        cur_w += needed;
                    }
                }
                if !cur_line.is_empty() {
                    result.push(cur_line);
                }
                result
            }
        };

        // Check if any line width exceeds 860px and scale font sizes down if needed
        let max_canvas_line_w = 860.0f64;
        for line in &lines {
            let line_w: f64 = line.iter().map(|&idx| prepared[idx].width).sum::<f64>()
                + (line.len().saturating_sub(1) as f64) * word_gap;
            if line_w > max_canvas_line_w {
                let scale = (max_canvas_line_w / line_w).max(0.60);
                for &idx in line {
                    let p = &mut prepared[idx];
                    p.font_size = ((p.font_size as f64) * scale).round() as u32;
                    let (nw, nh) = measure_t5_word_width(&p.formatted_word, p.role, p.font_size);
                    p.width = nw;
                    p.height = nh;
                }
            }
        }

        // Calculate line heights and total block height
        let line_heights: Vec<f64> = lines
            .iter()
            .map(|line| {
                line.iter()
                    .map(|&idx| prepared[idx].height)
                    .fold(0.0f64, f64::max)
            })
            .collect();
        let total_block_h: f64 =
            line_heights.iter().sum::<f64>() + (lines.len().saturating_sub(1) as f64) * line_gap;

        // Vertical centering
        let group_center_y = if is_group_dual {
            let seam_center = ((safe_top + safe_bottom) as f64) / 2.0;
            let mut cy = seam_center;
            if cy - total_block_h / 2.0 < safe_top as f64 {
                cy = safe_top as f64 + total_block_h / 2.0;
            }
            if cy + total_block_h / 2.0 > safe_bottom as f64 {
                cy = safe_bottom as f64 - total_block_h / 2.0;
            }
            cy
        } else {
            // Single speaker preferred center: Y ≈ 1245.0
            let mut cy = 1245.0f64;
            let group_top = cy - total_block_h / 2.0;

            let face_opt = match segment {
                Some(crate::media::LayoutSegment::Single { face_bounds, .. }) => {
                    face_bounds.as_ref()
                }
                _ => {
                    if let Some(plan) = framing_plan {
                        if plan.segments.is_empty() {
                            plan.face_bounds.as_ref()
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                }
            };

            if let Some(face) = face_opt {
                let face_bottom_px = if face.bottom <= 1.0 && face.bottom > 0.0 {
                    face.bottom * 1920.0
                } else {
                    face.bottom
                };
                let min_safe_top = face_bottom_px + 28.0;
                if group_top < min_safe_top {
                    let push_down = min_safe_top - group_top;
                    cy += push_down;
                }
            }

            // Lower safe-region bounds (protect bottom overlay)
            let max_bottom = 1520.0;
            if cy + total_block_h / 2.0 > max_bottom {
                cy = max_bottom - total_block_h / 2.0;
            }

            cy
        };

        // Compute line center Y coordinates
        let top_of_block = group_center_y - total_block_h / 2.0;
        let mut acc_y = top_of_block;
        let mut line_center_ys = Vec::with_capacity(lines.len());
        for lh in &line_heights {
            line_center_ys.push(acc_y + lh / 2.0);
            acc_y += lh + line_gap;
        }

        // Layout words and emit dialogue events
        for (m, line) in lines.iter().enumerate() {
            let line_w: f64 = line.iter().map(|&idx| prepared[idx].width).sum::<f64>()
                + (line.len().saturating_sub(1) as f64) * word_gap;
            let line_start_x = 540.0 - line_w / 2.0;
            let mut acc_x = line_start_x;
            let target_y = line_center_ys[m].round() as i64;

            for (j, &idx) in line.iter().enumerate() {
                let prep = &prepared[idx];
                let target_x = (acc_x + prep.width / 2.0).round() as i64;
                acc_x += prep.width + word_gap;

                let motion = match (m % 2, j % 2) {
                    (0, 0) => EditorialMotion::SlideRight,
                    (0, 1) => EditorialMotion::SlideLeft,
                    (1, 0) => EditorialMotion::SlideUp,
                    _ => EditorialMotion::SlideDown,
                };

                let offset_px = 32;
                let (start_x, start_y) = match motion {
                    EditorialMotion::SlideUp => (target_x, target_y + offset_px),
                    EditorialMotion::SlideDown => (target_x, target_y - offset_px),
                    EditorialMotion::SlideLeft => (target_x + offset_px, target_y),
                    EditorialMotion::SlideRight => (target_x - offset_px, target_y),
                };

                let move_tag = format!(
                    r"\move({},{},{},{},0,{})",
                    start_x, start_y, target_x, target_y, entrance_duration_ms
                );

                let word_rel_start = (prep.word.start - start_sec).max(0.0);
                let effective_end = group_rel_end.max(word_rel_start + 0.10);
                let word_dur_ms = ((effective_end - word_rel_start) * 1000.0) as u32;
                let actual_fade_ms = exit_fade_ms
                    .min(word_dur_ms.saturating_sub(entrance_duration_ms + 40))
                    .max(80);
                let fad_tag = format!(r"\fad(0,{})", actual_fade_ms);

                let event_text = format!(
                    r"{{\an5{move_tag}\alpha&HFF&\t(0,80,0.5,\alpha&H00&)\fscx110\fscy110\t(0,{},0.4,\fscx100\fscy100){fad_tag}\fn{}\fs{}\b{}\i{}\c{}\bord0\shad0}}{}",
                    entrance_duration_ms,
                    prep.font_family,
                    prep.font_size,
                    prep.bold_val,
                    prep.italic_val,
                    prep.color_inline,
                    prep.formatted_word
                );

                ass.push_str(&format!(
                    "Dialogue: 0,{},{},Default,,0,0,0,,{}\n",
                    format_ass_timestamp(word_rel_start),
                    format_ass_timestamp(effective_end),
                    event_text
                ));
            }
        }
    }

    ass
}

/// Generate ASS subtitle content from a CaptionTemplate.
pub fn generate_ass_from_template(
    words: &[TranscriptWord],
    start_sec: f64,
    end_sec: f64,
    template: &CaptionTemplate,
) -> String {
    generate_ass_from_template_with_framing(words, start_sec, end_sec, template, None)
}

pub fn generate_ass_from_template_with_framing(
    words: &[TranscriptWord],
    start_sec: f64,
    end_sec: f64,
    template: &CaptionTemplate,
    framing_plan: Option<&crate::media::SmartFramingPlan>,
) -> String {
    generate_ass_from_template_with_framing_and_intel(
        words,
        start_sec,
        end_sec,
        template,
        framing_plan,
        None,
        None,
    )
}

pub fn generate_ass_from_template_with_framing_and_intel(
    words: &[TranscriptWord],
    start_sec: f64,
    end_sec: f64,
    template: &CaptionTemplate,
    framing_plan: Option<&crate::media::SmartFramingPlan>,
    caption_intel: Option<&crate::caption_intel::CaptionIntelPlan>,
    t7_boundaries: Option<&[crate::t7_prosody::T7Boundary]>,
) -> String {
    if template.template_id == "preset_bhaukal_caption" {
        return generate_bhaukal_ass_with_framing_and_intel(
            words,
            start_sec,
            end_sec,
            template,
            framing_plan,
            caption_intel,
        );
    }
    if template.template_id == "preset_dynamic_editorial" {
        return generate_dynamic_editorial_ass_with_framing_and_intel(
            words,
            start_sec,
            end_sec,
            template,
            framing_plan,
            caption_intel,
        );
    }

    let candidate_words: Vec<(usize, &TranscriptWord)> = words
        .iter()
        .enumerate()
        .filter(|(_, w)| w.end > start_sec && w.start < end_sec)
        .collect();

    if candidate_words.is_empty() {
        return String::new();
    }

    // Apply subject face safeguard in single layout if face bounds are detected
    let single_face_bounds = if let Some(plan) = framing_plan {
        plan.segments
            .iter()
            .find_map(|s| match s {
                crate::media::LayoutSegment::Single { face_bounds, .. } => face_bounds.as_ref(),
                _ => None,
            })
            .or_else(|| {
                if plan.segments.is_empty() {
                    plan.face_bounds.as_ref()
                } else {
                    None
                }
            })
    } else {
        None
    };

    // Subject face safeguard for the GENERIC single-layout templates.
    //
    // This safeguard SHRINKS THE FONT when a subject face is detected. That is
    // correct for the generic templates, but it is wrong for T7, whose
    // reference identity is fixed (Matt Bold 110) and whose face-safety is
    // enforced POSITIONALLY instead — see the `preset_t7` branch later in this
    // same function, which runs a real solver (face bottom + 155px clearance,
    // rigid 120px band, asymmetric jitter suppression).
    //
    // So T7 must be excluded from the FONT-shrinking safeguard here; its
    // face-safety invariant is satisfied downstream by the positional solver.
    // This is not an exemption from face-safety — it is a choice of WHICH
    // mechanism enforces it (move the block, never resize it).
    let active_template = if let Some(face) = single_face_bounds {
        if template.template_id == "preset_t7" {
            // T7 keeps its exact reference typography; the T7 positional
            // solver below guarantees the captions still clear the face.
            template.clone()
        } else {
            apply_subject_face_safeguard(template, Some(face))
        }
    } else {
        template.clone()
    };

    let font_name = &active_template.global_styles.font_family;
    let font_size = active_template.global_styles.font_size;
    let primary_header_color = hex_to_ass_header(&active_template.global_styles.font_color);
    let primary_inline_color = hex_to_ass_inline(&active_template.global_styles.font_color);

    // Font weight: 900, 800, 600 -> -1 (bold); 400 -> 0 (regular)
    let bold_val = match active_template.global_styles.font_weight.as_str() {
        "400" => 0,
        _ => -1,
    };

    // BorderStyle: 3 for backgroundBox capsule, 1 for stroke/outline
    let is_capsule = active_template
        .global_styles
        .background_box
        .as_ref()
        .map(|b| b.enabled)
        .unwrap_or(false);

    let (border_style, outline_val, outline_color, shadow_val, back_color) = if is_capsule {
        let box_conf = active_template
            .global_styles
            .background_box
            .as_ref()
            .unwrap();
        let box_alpha = ((1.0 - box_conf.opacity.clamp(0.0, 1.0)) * 255.0).round() as u8;
        let clean_hex = box_conf.color.trim().trim_start_matches('#');
        let (r, g, b) = if clean_hex.len() == 6 {
            (&clean_hex[0..2], &clean_hex[2..4], &clean_hex[4..6])
        } else {
            ("00", "00", "00")
        };
        let box_color = format!("&H{:02X}{}{}{}", box_alpha, b, g, r);
        (
            3,
            box_conf.padding_px as i64,
            box_color,
            0,
            "&H00000000".to_string(),
        )
    } else {
        let outline = active_template.global_styles.stroke_width.unwrap_or(0) as i64;
        let stroke_col = active_template
            .global_styles
            .stroke_color
            .as_deref()
            .map(hex_to_ass_header)
            .unwrap_or_else(|| "&H00000000".to_string());

        let shadow = if let Some(offset) = active_template.global_styles.shadow_offset_y {
            offset.round() as i64
        } else if active_template.global_styles.shadow_blur.is_some() {
            3
        } else {
            0
        };

        let shadow_col = active_template
            .global_styles
            .shadow_color
            .as_deref()
            .map(rgba_to_ass_header)
            .unwrap_or_else(|| "&H80000000".to_string());

        (1, outline, stroke_col, shadow, shadow_col)
    };

    let margin_v = compute_margin_v(
        active_template.global_styles.position_y,
        font_size,
        active_template.segmentation.max_lines_per_frame,
    );

    let has_dual_stack = framing_plan.map_or(false, |plan| {
        plan.segments
            .iter()
            .any(|s| matches!(s, crate::media::LayoutSegment::DualStack { .. }))
    });

    let dual_style_line = if has_dual_stack {
        let block_height = if active_template.segmentation.max_lines_per_frame >= 2 {
            (font_size as f64) * 2.2
        } else {
            font_size as f64
        };

        let mut dual_font_size = font_size;
        let mut current_block_h = block_height;
        let mut dual_anchor_y = 960.0;

        // Protected visual boundaries for split-screen dynamically derived from dual segments:
        let (raw_safe_top, raw_safe_bottom) = if let Some(plan) = framing_plan {
            let mut top_bound: Option<f64> = None;
            let mut bot_bound: Option<f64> = None;
            for seg in &plan.segments {
                if let crate::media::LayoutSegment::DualStack {
                    top_face_bounds,
                    bottom_face_bounds,
                    ..
                } = seg
                {
                    if let Some(top_f) = top_face_bounds {
                        let top_face_bottom_px = if top_f.bottom <= 1.0 && top_f.bottom > 0.0 {
                            top_f.bottom * 1920.0
                        } else {
                            top_f.bottom
                        };
                        let val = (top_face_bottom_px + 24.0).clamp(800.0, 960.0);
                        top_bound = Some(top_bound.map_or(val, |prev: f64| prev.max(val)));
                    }
                    if let Some(bot_f) = bottom_face_bounds {
                        let bot_face_top_px = if bot_f.top <= 1.0 && bot_f.top > 0.0 {
                            bot_f.top * 1920.0
                        } else {
                            bot_f.top
                        };
                        let bot_face_bottom_px = if bot_f.bottom <= 1.0 && bot_f.bottom > 0.0 {
                            bot_f.bottom * 1920.0
                        } else {
                            bot_f.bottom
                        };
                        let fh = bot_face_bottom_px - bot_face_top_px;
                        let hairline_px = bot_face_top_px - (0.20 * fh) - 10.0;
                        let val = hairline_px.clamp(960.0, 1120.0);
                        bot_bound = Some(bot_bound.map_or(val, |prev: f64| prev.min(val)));
                    }
                }
            }
            (top_bound.unwrap_or(850.0), bot_bound.unwrap_or(1060.0))
        } else {
            (850.0, 1060.0)
        };

        let (safe_top, safe_bottom) = if raw_safe_top >= raw_safe_bottom {
            eprintln!(
                "[DualFrame LAKCB] Warning: Safe Top ({:.1}) >= Safe Bottom ({:.1}), falling back to default seam corridor [850.0, 1060.0]",
                raw_safe_top, raw_safe_bottom
            );
            (850.0, 1060.0)
        } else {
            (raw_safe_top, raw_safe_bottom)
        };

        let max_corridor_h = safe_bottom - safe_top;
        if current_block_h > max_corridor_h - 20.0 {
            let scale_factor = (max_corridor_h - 20.0).max(40.0) / current_block_h;
            dual_font_size = ((dual_font_size as f64 * scale_factor).round() as u32).max(40);
            current_block_h = if active_template.segmentation.max_lines_per_frame >= 2 {
                (dual_font_size as f64) * 2.2
            } else {
                dual_font_size as f64
            };
        }

        let cur_top = dual_anchor_y - (current_block_h / 2.0);
        let cur_bottom = dual_anchor_y + (current_block_h / 2.0);

        if cur_bottom > safe_bottom {
            let shift = cur_bottom - safe_bottom;
            dual_anchor_y = (dual_anchor_y - shift).max(safe_top + current_block_h / 2.0);
        } else if cur_top < safe_top {
            let shift = safe_top - cur_top;
            dual_anchor_y = (dual_anchor_y + shift).min(safe_bottom - current_block_h / 2.0);
        }

        let dual_margin_v = ((1.0 - (dual_anchor_y / 1920.0)) * 1920.0 - (current_block_h / 2.0))
            .round()
            .max(40.0) as i64;

        format!(
            "Style: DualStackSeam,{font_name},{dual_font_size},{primary_header_color},&H0000FFFF,{outline_color},{back_color},{bold_val},0,0,0,100,100,0,0,{border_style},{outline_val},{shadow_val},2,80,80,{dual_margin_v},1\n"
        )
    } else {
        String::new()
    };

    let mut ass = format!(
        r#"[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_name},{font_size},{primary_header_color},&H0000FFFF,{outline_color},{back_color},{bold_val},0,0,0,100,100,0,0,{border_style},{outline_val},{shadow_val},2,80,80,{margin_v},1
{dual_style_line}
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"#
    );

    // Group words into frames based on max_words_per_line and max_lines_per_frame
    let max_words_per_line = template.segmentation.max_words_per_line;
    let max_words_per_frame = max_words_per_line * template.segmentation.max_lines_per_frame;

    // ── T7: phrase-chunked rolling captions ────────────────────────────────
    // Speech-rhythm-driven, explicitly NON-karaoke: chunks are stable visual
    // units (no per-word highlight, no active-word sweep). The previous chunk
    // stays WHITE on the top line while the newly arrived chunk is YELLOW on
    // the bottom line; once the new chunk finishes speaking the previous one
    // leaves and the current chunk holds in WHITE until the next arrives.
    // ── T7: phrase-chunked rolling captions (phrase-paired display units) ──
    // Reconstructs the reference behavior: independent 2-line phrase units
    // (Line 1 White lead-in -> Line 2 Yellow arrival -> stable coexistence ->
    // simultaneous clean clear -> next phrase starts fresh) or 1-line units
    // for isolated emphatic chunks. Line 1 is pinned at y1 and Line 2 at y2.
    // Chunks are NEVER recycled or shifted upward from Line 2 into Line 1.
    // Zero conveyor belt motion, zero vertical snapping, zero per-word karaoke.
    if template.template_id == "preset_t7" {
        let hl_inline = match &active_template.active_state {
            CaptionActiveState::SpeechChunkedRolling { highlight_color } => {
                hex_to_ass_inline(highlight_color)
            }
            _ => "&H07EFF9&".to_string(), // #F9EF07
        };
        let text_case = &active_template.segmentation.text_case;
        let clip_dur = end_sec - start_sec;

        let format_chunk = |c: &Vec<(usize, &TranscriptWord)>| -> String {
            c.iter()
                .enumerate()
                .map(|(i, (_, w))| {
                    let clean = w.text.trim().replace('{', "").replace('}', "");
                    apply_text_casing(&clean, text_case, i == 0)
                })
                .collect::<Vec<_>>()
                .join(" ")
        };

        let units = group_t7_phrase_units(
            &candidate_words,
            max_words_per_line,
            1.5, // max chunk duration ceiling (s)
            2.6, // max phrase pair duration ceiling (s)
            t7_boundaries,
        );

        let _is_adaptive = framing_plan.map_or(false, |plan| plan.framing == "adaptive");
        let mut current_y1: Option<i64> = None;

        // Post-phrase hold duration aligned to reference (0.25s)
        let post_hold: f64 = 0.25;
        let max_lead: f64 = 0.040; // 40ms maximum anticipation target

        let mut prev_unit_end_rel = 0.0;

        for (u_idx, unit) in units.iter().enumerate() {
            let (mid_abs, is_empty_unit) = match unit {
                T7PhraseUnit::Solo {
                    chunk,
                    speech_start,
                    speech_end,
                } => ((*speech_start + *speech_end) / 2.0, chunk.is_empty()),
                T7PhraseUnit::Pair {
                    first,
                    second,
                    first_start,
                    second_end,
                    ..
                } => (
                    (*first_start + *second_end) / 2.0,
                    first.is_empty() || second.is_empty(),
                ),
            };
            if is_empty_unit {
                continue;
            }

            let mid_rel = (mid_abs - start_sec).max(0.0);
            let face_bounds = framing_plan.and_then(|plan| {
                plan.segment_at(mid_rel)
                    .and_then(|seg| match seg {
                        crate::media::LayoutSegment::Single { face_bounds, .. } => {
                            face_bounds.as_ref()
                        }
                        _ => None,
                    })
                    .or_else(|| plan.face_bounds.as_ref())
            });

            // T7 Face-Safe Placement solver:
            // 1. Baseline target: default reference position is y_1 = 1020, y_2 = 1140.
            // 2. Clearance: Matt Bold 110 ascender is ~85px, plus 70px dynamic speech/jaw motion buffer = 155px below face bottom.
            // face.bottom in SmartFramingPlan is already normalized to the final 1080x1920 canvas
            // (speaker_tracker.py maps crop coordinates to the canvas in both original 9:16 and adaptive modes).
            let y1_target = if let Some(face) = face_bounds {
                let face_bottom_px = if face.bottom <= 1.0 && face.bottom > 0.0 {
                    face.bottom * 1920.0
                } else {
                    face.bottom
                };
                (face_bottom_px + 155.0).round() as i64
            } else {
                1020
            };

            // 3. Active Composition Prioritization:
            // - Active square bottom is y = 1500; with 20px padding, max y2 inside square is 1480 (y1 <= 1360).
            // - If y1_target <= 1360: keep inside active composition, y1 = max(1020, y1_target).
            // - If y1_target > 1360: PRESERVE FACE VISIBILITY FIRST over letterbox avoidance.
            //   T7 then extends downward past 1480px into the lower region.
            //
            //   The hard floor is the CANVAS, not an arbitrary margin: the rigid
            //   band is y2 = y1 + 120 and the play surface is 1920px tall, so
            //   y1 <= 1800 keeps the second line exactly on-canvas. The previous
            //   clamp of 1740 could NOT satisfy the 155px clearance for a face
            //   below ~1585px, which meant the safety invariant was silently
            //   violated for low faces. Face clearance wins; the canvas bound
            //   is the only hard limit.
            let y1_candidate = if y1_target <= 1360 {
                1020.max(y1_target)
            } else {
                y1_target.min(1800)
            };

            // 4. Unified Rigid Band & Jitter Suppression (Safe Asymmetric Hysteresis):
            // - If moving DOWN (y1_candidate > prev): face is lower, MUST move down immediately to protect face clearance.
            // - If moving UP (y1_candidate < prev): face moved higher, prev is already safely below the face;
            //   hold prev if within 30px to prevent visual jitter.
            let y1 = match current_y1 {
                Some(prev) if y1_candidate < prev && (prev - y1_candidate) <= 30 => prev,
                _ => {
                    current_y1 = Some(y1_candidate);
                    y1_candidate
                }
            };
            let y2 = y1 + 120;
            let y_solo = y1 + 60;
            // Find when the next unit's speech begins (if any) to prevent overlap
            let next_unit_speech_start_rel = units
                .get(u_idx + 1)
                .map(|next_u| match next_u {
                    T7PhraseUnit::Solo { speech_start, .. } => (speech_start - start_sec).max(0.0),
                    T7PhraseUnit::Pair { first_start, .. } => (first_start - start_sec).max(0.0),
                })
                .unwrap_or(clip_dur);

            match unit {
                T7PhraseUnit::Solo {
                    chunk,
                    speech_start,
                    speech_end,
                } => {
                    if chunk.is_empty() {
                        continue;
                    }
                    let t_start_raw = (speech_start - start_sec).max(0.0);
                    let t_end_raw = (speech_end - start_sec)
                        .min(clip_dur)
                        .max(t_start_raw + 0.05);

                    // Compute anticipation:
                    // If near opening frame (< 0.35s), lead in from 0.00s.
                    // Otherwise, only lead if there was a preceding pause (> 0.15s), capped at 40ms.
                    let gap_before = t_start_raw - prev_unit_end_rel;
                    let lead = if t_start_raw < 0.35 && u_idx == 0 {
                        t_start_raw
                    } else if gap_before > 0.15 {
                        max_lead.min(gap_before * 0.25)
                    } else {
                        0.0
                    };
                    let disp_start = (t_start_raw - lead).max(prev_unit_end_rel).max(0.0);
                    let hold_end = (t_end_raw + post_hold)
                        .min(next_unit_speech_start_rel)
                        .min(clip_dur)
                        .max(disp_start + 0.10);

                    let mid_abs = (*speech_start + *speech_end) / 2.0;
                    let is_event_dual = framing_plan.map_or(false, |plan| {
                        plan.segment_at(mid_abs - start_sec).map_or(false, |s| {
                            matches!(s, crate::media::LayoutSegment::DualStack { .. })
                        })
                    });
                    let event_style = if is_event_dual {
                        "DualStackSeam"
                    } else {
                        "Default"
                    };

                    let solo_text = format_chunk(chunk);
                    if is_event_dual {
                        ass.push_str(&format!(
                            "Dialogue: 0,{},{},{},,0,0,0,,{}\n",
                            format_ass_timestamp(disp_start),
                            format_ass_timestamp(hold_end),
                            event_style,
                            solo_text,
                        ));
                    } else {
                        ass.push_str(&format!(
                            "Dialogue: 0,{},{},{},,0,0,0,,{{\\pos(540,{y_solo})}}{}\n",
                            format_ass_timestamp(disp_start),
                            format_ass_timestamp(hold_end),
                            event_style,
                            solo_text,
                        ));
                    }
                    prev_unit_end_rel = hold_end;
                }
                T7PhraseUnit::Pair {
                    first,
                    second,
                    first_start,
                    first_end,
                    second_start,
                    second_end,
                } => {
                    if first.is_empty() || second.is_empty() {
                        continue;
                    }
                    let t_first_start_raw = (first_start - start_sec).max(0.0);
                    let t_first_end_raw = (first_end - start_sec)
                        .min(clip_dur)
                        .max(t_first_start_raw + 0.05);
                    let t_second_start_raw = (second_start - start_sec)
                        .min(clip_dur)
                        .max(t_first_start_raw);
                    let t_second_end_raw = (second_end - start_sec)
                        .min(clip_dur)
                        .max(t_second_start_raw + 0.05);

                    // Stage 1 (Line 1 White) start:
                    let gap_before = t_first_start_raw - prev_unit_end_rel;
                    let lead1 = if t_first_start_raw < 0.35 && u_idx == 0 {
                        t_first_start_raw
                    } else if gap_before > 0.15 {
                        max_lead.min(gap_before * 0.25)
                    } else {
                        0.0
                    };
                    let disp_start1 = (t_first_start_raw - lead1).max(prev_unit_end_rel).max(0.0);

                    // Stage 2 (Line 2 Yellow arrival) start:
                    // Second chunk must NEVER arrive before first chunk ends its core speech.
                    let gap_between = t_second_start_raw - t_first_end_raw;
                    let lead2 = if gap_between > 0.10 {
                        max_lead.min(gap_between * 0.25)
                    } else {
                        0.0
                    };
                    let disp_start2 = (t_second_start_raw - lead2)
                        .max(t_first_end_raw)
                        .max(disp_start1 + 0.06);

                    // Unified phrase unit end (both lines clear simultaneously after hold):
                    let hold_end = (t_second_end_raw + post_hold)
                        .min(next_unit_speech_start_rel)
                        .min(clip_dur)
                        .max(disp_start2 + 0.10);

                    let mid_abs = (*first_start + *second_end) / 2.0;
                    let is_event_dual = framing_plan.map_or(false, |plan| {
                        plan.segment_at(mid_abs - start_sec).map_or(false, |s| {
                            matches!(s, crate::media::LayoutSegment::DualStack { .. })
                        })
                    });
                    let event_style = if is_event_dual {
                        "DualStackSeam"
                    } else {
                        "Default"
                    };

                    let first_text = format_chunk(first);
                    let second_text = format_chunk(second);

                    if is_event_dual {
                        // In DualStack mode, single event stacked with newline
                        if disp_start2 > disp_start1 + 0.06 {
                            ass.push_str(&format!(
                                "Dialogue: 0,{},{},{},,0,0,0,,{}\n",
                                format_ass_timestamp(disp_start1),
                                format_ass_timestamp(disp_start2),
                                event_style,
                                first_text,
                            ));
                        }
                        ass.push_str(&format!(
                            "Dialogue: 0,{},{},{},,0,0,0,,{}\\N{{\\c{hl}}}{}{{\\c{primary}}}\n",
                            format_ass_timestamp(disp_start2),
                            format_ass_timestamp(hold_end),
                            event_style,
                            first_text,
                            second_text,
                            hl = hl_inline,
                            primary = primary_inline_color,
                        ));
                    } else {
                        // Standard T7 pinned dual-layer layout:
                        // Stage 1: Line 1 alone at y1 in White
                        if disp_start2 > disp_start1 + 0.06 {
                            ass.push_str(&format!(
                                "Dialogue: 0,{},{},{},,0,0,0,,{{\\pos(540,{y1})}}{}\n",
                                format_ass_timestamp(disp_start1),
                                format_ass_timestamp(disp_start2),
                                event_style,
                                first_text,
                            ));
                        }
                        // Stage 2: Line 1 remains at y1 (White), Line 2 arrives at y2 (Yellow).
                        // Both end at the exact same millisecond hold_end!
                        ass.push_str(&format!(
                            "Dialogue: 1,{},{},{},,0,0,0,,{{\\pos(540,{y1})}}{}\n",
                            format_ass_timestamp(disp_start2),
                            format_ass_timestamp(hold_end),
                            event_style,
                            first_text,
                        ));
                        ass.push_str(&format!(
                            "Dialogue: 0,{},{},{},,0,0,0,,{{\\pos(540,{y2})}}{{\\c{hl}}}{}{{\\c{primary}}}\n",
                            format_ass_timestamp(disp_start2),
                            format_ass_timestamp(hold_end),
                            event_style,
                            second_text,
                            hl = hl_inline,
                            primary = primary_inline_color,
                        ));
                    }
                    prev_unit_end_rel = hold_end;
                }
            }
        }

        return ass;
    }

    let conjunctions = [
        "and", "but", "or", "so", "because", "when", "if", "that", "with", "from", "into", "every",
        "on", "your", "my", "our", "their", "this", "which", "aur", "lekin", "par", "ki", "toh",
        "kyunki", "agar", "jab", "tab",
    ];

    let mut frames: Vec<Vec<(usize, &TranscriptWord)>> = Vec::new();
    let mut curr_frame: Vec<(usize, &TranscriptWord)> = Vec::new();

    for &(global_idx, word) in &candidate_words {
        if let Some(&(_, prev)) = curr_frame.last() {
            let pause = word.start - prev.end;
            let duration = word.end - curr_frame[0].1.start;
            let count = curr_frame.len();

            let clean_w = word
                .text
                .to_lowercase()
                .trim_matches(|c: char| !c.is_alphanumeric())
                .to_string();
            let ends_punct = crate::transcription::ends_with_sentence_terminator(&prev.text);
            let ends_clause = prev.text.ends_with(',');
            let is_pause = pause > 0.28;
            let is_conjunction = count >= 2 && conjunctions.contains(&clean_w.as_str());
            let reaches_max = count >= max_words_per_frame;
            let max_dur = duration > 2.2;

            if is_pause
                || ends_punct
                || reaches_max
                || max_dur
                || (count >= max_words_per_line && (ends_clause || is_conjunction))
            {
                frames.push(std::mem::take(&mut curr_frame));
            }
        }
        curr_frame.push((global_idx, word));
    }
    if !curr_frame.is_empty() {
        frames.push(curr_frame);
    }

    // If SmoothFade (Cinematic Vlog), generate one dialogue event per timed caption event (frame)
    // with smooth fade-in and fade-out. The caption remains stable on screen for its intended
    // interval and transitions cleanly to the next event without unwanted per-word flickering/flashing.
    if let CaptionActiveState::SmoothFade { duration_ms, .. } = &active_template.active_state {
        let fade_ms = *duration_ms;
        for frame in &frames {
            if frame.is_empty() {
                continue;
            }

            let frame_start = (frame[0].1.start - start_sec).max(0.0);
            let frame_end = (frame.last().unwrap().1.end - start_sec)
                .min(end_sec - start_sec)
                .max(frame_start + 0.1);

            let mut formatted_words = Vec::new();
            for (idx, &(global_idx, w)) in frame.iter().enumerate() {
                let clean_raw = w.text.trim().replace('{', "").replace('}', "");
                let is_first = idx == 0;
                let text = apply_text_casing(
                    &clean_raw,
                    &active_template.segmentation.text_case,
                    is_first,
                );
                let is_emphasized = caption_intel
                    .map(|plan| plan.emphasis_word_indices.contains(&global_idx))
                    .unwrap_or(false);

                if is_emphasized {
                    formatted_words.push(format!(r"{{\b1}}{text}{{\b0}}"));
                } else {
                    formatted_words.push(text);
                }
            }

            let preferred_break = caption_intel.and_then(|plan| {
                let frame_global_indices: Vec<usize> =
                    frame.iter().map(|(g_idx, _)| *g_idx).collect();
                crate::caption_intel::select_legal_line_break(
                    &frame_global_indices,
                    &plan.line_break_hints,
                    max_words_per_line,
                    active_template.segmentation.max_lines_per_frame,
                )
            });

            let dialogue_text = wrap_words_into_lines_with_hints(
                &formatted_words,
                max_words_per_line,
                active_template.segmentation.max_lines_per_frame,
                preferred_break,
            );

            let final_text = format!(r"{{\fad({fade_ms},{fade_ms})}}{dialogue_text}");

            let is_event_dual = if let Some(plan) = framing_plan {
                let frame_abs_mid = (frame[0].1.start + frame.last().unwrap().1.end) / 2.0;
                // LayoutSegment start/end are clip-relative [0, clip_dur]; frame timestamps are
                // absolute source-video times. Convert to clip-relative before the segment lookup.
                plan.segment_at(frame_abs_mid - start_sec)
                    .map_or(false, |s| {
                        matches!(s, crate::media::LayoutSegment::DualStack { .. })
                    })
            } else {
                false
            };

            let event_style = if is_event_dual {
                "DualStackSeam"
            } else {
                "Default"
            };
            let start_str = format_ass_timestamp(frame_start);
            let end_str = format_ass_timestamp(frame_end);

            ass.push_str(&format!(
                "Dialogue: 0,{},{},{},,0,0,0,,{}\n",
                start_str, end_str, event_style, final_text
            ));
        }

        return ass;
    }

    // Generate dialogue events for each word in each frame (word-by-word active state templates)
    for frame in &frames {
        if frame.is_empty() {
            continue;
        }

        let preferred_break = caption_intel.and_then(|plan| {
            let frame_global_indices: Vec<usize> = frame.iter().map(|(g_idx, _)| *g_idx).collect();
            crate::caption_intel::select_legal_line_break(
                &frame_global_indices,
                &plan.line_break_hints,
                max_words_per_line,
                active_template.segmentation.max_lines_per_frame,
            )
        });

        for (active_idx, &(_, active_word)) in frame.iter().enumerate() {
            let word_start = (active_word.start - start_sec).max(0.0);
            let word_end = (active_word.end - start_sec)
                .min(end_sec - start_sec)
                .max(word_start + 0.05);

            let mut formatted_words = Vec::new();

            for (idx, &(global_idx, w)) in frame.iter().enumerate() {
                let clean_raw = w.text.trim().replace('{', "").replace('}', "");
                let is_first = idx == 0;
                let text = apply_text_casing(
                    &clean_raw,
                    &active_template.segmentation.text_case,
                    is_first,
                );
                let is_emphasized = caption_intel
                    .map(|plan| plan.emphasis_word_indices.contains(&global_idx))
                    .unwrap_or(false);

                match &active_template.active_state {
                    CaptionActiveState::WordByWordSwap {
                        primary_highlight_color,
                        secondary_highlight_color,
                        ..
                    } => {
                        if idx == active_idx {
                            let hl_color = if active_idx % 2 == 0 {
                                hex_to_ass_inline(primary_highlight_color)
                            } else {
                                hex_to_ass_inline(secondary_highlight_color)
                            };
                            formatted_words.push(format!(
                                r"{{\c{hl_color}\fscx115\fscy115}}{text}{{\fscx100\fscy100\c{primary_inline_color}}}"
                            ));
                        } else if is_emphasized {
                            let hl_color = hex_to_ass_inline(secondary_highlight_color);
                            formatted_words.push(format!(
                                r"{{\c{hl_color}}}{text}{{\c{primary_inline_color}}}"
                            ));
                        } else {
                            formatted_words.push(text);
                        }
                    }

                    CaptionActiveState::KaraokeProgressive {
                        highlight_color, ..
                    } => {
                        let hl = hex_to_ass_inline(highlight_color);
                        if idx < active_idx {
                            // Already spoken word in progressive karaoke
                            formatted_words
                                .push(format!(r"{{\c{hl}}}{text}{{\c{primary_inline_color}}}"));
                        } else if idx == active_idx {
                            // Active word: pop bounce animation
                            formatted_words.push(format!(
                                r"{{\c{hl}\t(0,50,\fscx120\fscy120)\t(50,100,\fscx100\fscy100)}}{text}{{\fscx100\fscy100\c{primary_inline_color}}}"
                            ));
                        } else if is_emphasized {
                            formatted_words
                                .push(format!(r"{{\c{hl}}}{text}{{\c{primary_inline_color}}}"));
                        } else {
                            // Upcoming word
                            formatted_words.push(text);
                        }
                    }

                    CaptionActiveState::StaticOpacityReveal { .. } => {
                        if idx == active_idx {
                            formatted_words.push(format!(r"{{\alpha&H00&}}{text}"));
                        } else if is_emphasized {
                            formatted_words.push(format!(r"{{\alpha&H33&}}{text}"));
                        } else {
                            formatted_words.push(format!(r"{{\alpha&H99&}}{text}"));
                        }
                    }

                    CaptionActiveState::SmoothFade { .. } => {
                        if is_emphasized {
                            formatted_words.push(format!(r"{{\b1}}{text}{{\b0}}"));
                        } else {
                            formatted_words.push(text);
                        }
                    }

                    CaptionActiveState::DynamicEditorialKinetic { .. } => {
                        formatted_words.push(text);
                    }

                    // Unreachable via production dispatch (T6 early-returns to its
                    // own generator; T7 early-returns to the phrase-chunked branch);
                    // plain passthrough keeps T1–T5 output unchanged.
                    CaptionActiveState::BhaukalKinetic { .. } => {
                        formatted_words.push(text);
                    }
                    CaptionActiveState::SpeechChunkedRolling { .. } => {
                        formatted_words.push(text);
                    }
                }
            }

            // Use safe line wrapping ensuring line and event constraints
            let dialogue_text = wrap_words_into_lines_with_hints(
                &formatted_words,
                max_words_per_line,
                active_template.segmentation.max_lines_per_frame,
                preferred_break,
            );

            let final_text = dialogue_text;

            let is_event_dual = if let Some(plan) = framing_plan {
                let word_abs_mid = (active_word.start + active_word.end) / 2.0;
                // LayoutSegment start/end are clip-relative [0, clip_dur]; word timestamps are
                // absolute source-video times. Convert to clip-relative before the segment lookup.
                plan.segment_at(word_abs_mid - start_sec)
                    .map_or(false, |s| {
                        matches!(s, crate::media::LayoutSegment::DualStack { .. })
                    })
            } else {
                false
            };

            let event_style = if is_event_dual {
                "DualStackSeam"
            } else {
                "Default"
            };

            let start_str = format_ass_timestamp(word_start);
            let end_str = format_ass_timestamp(word_end);

            ass.push_str(&format!(
                "Dialogue: 0,{},{},{},,0,0,0,,{}\n",
                start_str, end_str, event_style, final_text
            ));
        }
    }

    ass
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::TranscriptWord;

    #[test]
    fn test_wrap_words_into_lines_cinematic_vlog_constraints() {
        // Rule: Cinematic Vlog max 6 words per line, max 2 lines per event

        // 1 word -> 1 line
        let words1 = vec!["hello".to_string()];
        let out1 = wrap_words_into_lines(&words1, 6, 2);
        assert_eq!(out1, "hello");
        assert!(!out1.contains(r"\N"));

        // 6 words -> 1 line
        let words6: Vec<String> = (1..=6).map(|i| format!("word{i}")).collect();
        let out6 = wrap_words_into_lines(&words6, 6, 2);
        assert!(
            !out6.contains(r"\N"),
            "6 words must fit in a single line: {}",
            out6
        );
        assert_eq!(out6.split_whitespace().count(), 6);

        // 7 words -> 2 lines, each line <= 6 words
        let words7: Vec<String> = (1..=7).map(|i| format!("w{i}")).collect();
        let out7 = wrap_words_into_lines(&words7, 6, 2);
        let lines7: Vec<&str> = out7.split(r"\N").collect();
        assert_eq!(lines7.len(), 2, "7 words must split into 2 lines: {}", out7);
        for l in &lines7 {
            let wc = l.split_whitespace().count();
            assert!(wc <= 6 && wc > 0, "Line exceeds 6 words: '{}'", l);
        }

        // 12 words -> 2 lines, each line <= 6 words
        let words12: Vec<String> = (1..=12).map(|i| format!("w{i}")).collect();
        let out12 = wrap_words_into_lines(&words12, 6, 2);
        let lines12: Vec<&str> = out12.split(r"\N").collect();
        assert_eq!(lines12.len(), 2);
        assert_eq!(lines12[0].split_whitespace().count(), 6);
        assert_eq!(lines12[1].split_whitespace().count(), 6);

        // 13 words -> capped at 2 lines, each <= 6 words (max 12 words formatted)
        let words13: Vec<String> = (1..=13).map(|i| format!("w{i}")).collect();
        let out13 = wrap_words_into_lines(&words13, 6, 2);
        let lines13: Vec<&str> = out13.split(r"\N").collect();
        assert_eq!(lines13.len(), 2);
        for l in &lines13 {
            let wc = l.split_whitespace().count();
            assert!(
                wc <= 6,
                "Line cannot exceed 6 words in 13-word test: '{}'",
                l
            );
        }

        // 18 words -> capped at 2 lines, each <= 6 words
        let words18: Vec<String> = (1..=18).map(|i| format!("w{i}")).collect();
        let out18 = wrap_words_into_lines(&words18, 6, 2);
        let lines18: Vec<&str> = out18.split(r"\N").collect();
        assert_eq!(lines18.len(), 2);
        for l in &lines18 {
            let wc = l.split_whitespace().count();
            assert!(
                wc <= 6,
                "Line cannot exceed 6 words in 18-word test: '{}'",
                l
            );
        }
    }

    #[test]
    fn test_cinematic_vlog_ass_generation_line_limits() {
        let words: Vec<TranscriptWord> = (0..20)
            .map(|i| TranscriptWord {
                text: format!("word{}", i + 1),
                start: i as f64 * 0.3,
                end: (i as f64 * 0.3) + 0.25,
                speaker: Some("S1".to_string()),
            })
            .collect();

        let template = get_caption_template("preset_cinematic_vlog")
            .expect("Cinematic Vlog template must exist");
        let ass = generate_ass_from_template(&words, 0.0, 6.0, &template);

        for line in ass.lines() {
            if let Some(dialogue_part) = line.strip_prefix("Dialogue: ") {
                let parts: Vec<&str> = dialogue_part.splitn(10, ',').collect();
                assert_eq!(
                    parts.len(),
                    10,
                    "ASS Dialogue line must have 10 comma-separated parts"
                );
                let text = parts[9];

                // Split by \N
                let sublines: Vec<&str> = text.split(r"\N").collect();
                assert!(
                    sublines.len() <= 2,
                    "Cinematic Vlog event cannot have more than 2 lines: {}",
                    text
                );

                for sl in sublines {
                    // Strip ASS override tags like {\c&H...&} or {\fad(...)}
                    let mut clean_text = String::new();
                    let mut in_tag = false;
                    for c in sl.chars() {
                        if c == '{' {
                            in_tag = true;
                        } else if c == '}' {
                            in_tag = false;
                        } else if !in_tag {
                            clean_text.push(c);
                        }
                    }

                    let wc = clean_text.split_whitespace().count();
                    assert!(
                        wc <= 6,
                        "Cinematic Vlog line exceeded 6 words (found {} words in '{}')",
                        wc,
                        clean_text
                    );
                }
            }
        }
    }

    #[test]
    fn test_compute_margin_v_formula_agreement() {
        // L = 1, position_y = 0.80, font_size = 52
        // H_block = 52.0
        // Y_px = 1920 * 0.80 = 1536.0
        // MarginV = 1920 - 1536.0 - 26.0 = 358
        let mv1 = compute_margin_v(0.80, 52, 1);
        assert_eq!(mv1, 358, "L=1 formula mismatch");

        // L = 2, position_y = 0.80, font_size = 38
        // H_block = 2.2 * 38.0 = 83.6
        // Y_px = 1920 * 0.80 = 1536.0
        // MarginV = round(1920 - 1536.0 - 41.8) = round(342.2) = 342
        let mv2 = compute_margin_v(0.80, 38, 2);
        assert_eq!(mv2, 342, "L=2 formula mismatch");

        // L = 1, position_y = 0.75, font_size = 48
        // H_block = 48.0
        // Y_px = 1920 * 0.75 = 1440.0
        // MarginV = 1920 - 1440.0 - 24.0 = 456
        let mv3 = compute_margin_v(0.75, 48, 1);
        assert_eq!(mv3, 456, "L=1 at position_y=0.75 mismatch");
    }

    #[test]
    fn test_required_font_filename_mappings() {
        assert_eq!(
            required_font_filename("Bebas Neue"),
            Some("BebasNeue-Regular.ttf")
        );
        assert_eq!(
            required_font_filename("bebas neue"),
            Some("BebasNeue-Regular.ttf")
        );
        assert_eq!(
            required_font_filename("Montserrat"),
            Some("Montserrat[wght].ttf")
        );
        assert_eq!(
            required_font_filename("Inter"),
            Some("Inter[opsz,wght].ttf")
        );
        assert_eq!(
            required_font_filename("Poppins"),
            Some("Poppins-Regular.ttf")
        );
        assert_eq!(
            required_font_filename("Matt_Trial-Bold"),
            Some("Matt_Trial-Bold.otf")
        );
        assert_eq!(required_font_filename("Arial"), None);
    }

    #[test]
    fn test_dynamic_editorial_layout_variants_matrix() {
        for k in 1..=4 {
            let variants = get_editorial_layout_variants(k);
            assert_eq!(
                variants.len(),
                3,
                "Each group size k={} must have 3 layout variants",
                k
            );
            for (v_idx, var) in variants.iter().enumerate() {
                assert_eq!(
                    var.len(),
                    k,
                    "Variant {} for k={} must contain exactly {} word placements",
                    v_idx,
                    k,
                    k
                );
                for p in var {
                    // Safe viewport bounds: X in [140, 940], Y in [850, 1450]
                    assert!(
                        p.x >= 140 && p.x <= 940,
                        "X coordinate {} out of safe bounds [140, 940]",
                        p.x
                    );
                    assert!(
                        p.y >= 850 && p.y <= 1450,
                        "Y coordinate {} out of safe bounds [850, 1450]",
                        p.y
                    );
                }
            }
        }
    }

    #[test]
    fn test_dynamic_editorial_ass_generation_tags_and_coexistence() {
        let words = vec![
            TranscriptWord {
                text: "DYNAMIC".to_string(),
                start: 0.10,
                end: 0.40,
                speaker: Some("S1".to_string()),
            },
            TranscriptWord {
                text: "EDITORIAL".to_string(),
                start: 0.45,
                end: 0.80,
                speaker: Some("S1".to_string()),
            },
            TranscriptWord {
                text: "KINETIC".to_string(),
                start: 0.85,
                end: 1.30,
                speaker: Some("S1".to_string()),
            },
        ];

        let template = get_caption_template("preset_dynamic_editorial")
            .expect("preset_dynamic_editorial template must exist");
        let ass = generate_ass_from_template(&words, 0.0, 2.0, &template);

        assert!(ass.contains("[Script Info]"));
        assert!(ass.contains("[V4+ Styles]"));
        assert!(ass.contains("[Events]"));

        let dialogue_lines: Vec<&str> =
            ass.lines().filter(|l| l.starts_with("Dialogue:")).collect();
        assert_eq!(
            dialogue_lines.len(),
            3,
            "Must generate 3 dialogue lines for 3-word group"
        );

        // Check that all 3 dialogue lines share the exact same group end timestamp (coexistence)
        let mut end_timestamps = Vec::new();
        for line in &dialogue_lines {
            let parts: Vec<&str> = line.splitn(10, ',').collect();
            assert_eq!(parts.len(), 10);
            let end_ts = parts[2];
            end_timestamps.push(end_ts);

            let event_text = parts[9];
            assert!(
                event_text.contains(r"\an5"),
                "Must use center-middle alignment: {}",
                event_text
            );
            assert!(
                event_text.contains(r"\move("),
                "Must contain \\move tag: {}",
                event_text
            );
            assert!(
                event_text.contains(r"\alpha&HFF&\t(0,80,0.5,\alpha&H00&)"),
                "Must contain opacity entrance: {}",
                event_text
            );
            assert!(
                event_text.contains(r"\fad(0,"),
                "Must contain group exit fade: {}",
                event_text
            );
        }

        assert_eq!(
            end_timestamps[0], end_timestamps[1],
            "Words in group must share same group end timestamp"
        );
        assert_eq!(
            end_timestamps[1], end_timestamps[2],
            "Words in group must share same group end timestamp"
        );

        // Verify typography variation across the 3 words
        assert!(
            ass.contains(r"\fnInter"),
            "Must include Inter for secondary word"
        );
        assert!(
            ass.contains(r"\fnBebas Neue"),
            "Must include Bebas Neue for emphasis word"
        );
        assert!(
            ass.contains(r"\fnPoppins"),
            "Must include Poppins for decorative word"
        );
        assert!(
            ass.contains(r"\i1"),
            "Must use ASS oblique italic tag for Poppins decorative role"
        );
    }

    #[test]
    fn test_smart_framing_plan_face_bounds_deserialization() {
        let json_with_face = r#"{
            "mode": "single",
            "x": "420",
            "y": "0",
            "w": "1080",
            "h": "1920",
            "segments": [
                {
                    "mode": "single",
                    "start": 0.0,
                    "end": 10.0,
                    "crop": { "x": "420", "y": "0", "w": "1080", "h": "1920" }
                }
            ],
            "face_bounds": {
                "top": 0.18,
                "bottom": 0.45,
                "left": 0.30,
                "right": 0.70
            }
        }"#;

        let plan: crate::media::SmartFramingPlan =
            serde_json::from_str(json_with_face).expect("Must deserialize plan with face_bounds");
        assert_eq!(plan.mode, "single");
        assert!(plan.face_bounds.is_some());
        let face = plan.face_bounds.unwrap();
        assert!((face.top - 0.18).abs() < 1e-6);
        assert!((face.bottom - 0.45).abs() < 1e-6);

        let json_without_face = r#"{
            "mode": "single",
            "x": "0",
            "y": "0",
            "w": "1080",
            "h": "1920"
        }"#;
        let plan_no_face: crate::media::SmartFramingPlan = serde_json::from_str(json_without_face)
            .expect("Must deserialize backward-compatible plan");
        assert!(plan_no_face.face_bounds.is_none());
    }

    #[test]
    fn test_apply_subject_face_safeguard_activated_in_ass_generation() {
        let words = vec![
            TranscriptWord {
                text: "HELLO".to_string(),
                start: 0.10,
                end: 0.60,
                speaker: Some("S1".to_string()),
            },
            TranscriptWord {
                text: "WORLD".to_string(),
                start: 0.60,
                end: 1.20,
                speaker: Some("S1".to_string()),
            },
        ];

        let template = get_caption_template("preset_viral_bold").unwrap();

        // 1. Normal face: chin at 0.40 (768px), caption top at 1344 - 45 = 1299px -> no collision
        let normal_plan = crate::media::SmartFramingPlan {
            mode: "single".to_string(),
            x: "0".to_string(),
            y: "0".to_string(),
            w: "1080".to_string(),
            h: "1920".to_string(),
            segments: vec![],
            face_bounds: Some(SubjectFaceBounds {
                top: 0.15,
                bottom: 0.40,
                left: 0.30,
                right: 0.70,
            }),
            framing: "original".to_string(),
            is_emergency_fallback: false,
            fallback_reason: None,
            ..Default::default()
        };
        let ass_normal = generate_ass_from_template_with_framing(
            &words,
            0.0,
            2.0,
            &template,
            Some(&normal_plan),
        );
        // Baseline Hormozi size 84
        assert!(
            ass_normal.contains("Style: Default,Bebas Neue,84,"),
            "Normal face preserves configured font size 84: {}",
            ass_normal
        );

        // 2. Intrusion face: chin at 0.70 (1344px) -> collides with caption top at 1299px!
        let intrusion_plan = crate::media::SmartFramingPlan {
            mode: "single".to_string(),
            x: "0".to_string(),
            y: "0".to_string(),
            w: "1080".to_string(),
            h: "1920".to_string(),
            segments: vec![],
            face_bounds: Some(SubjectFaceBounds {
                top: 0.20,
                bottom: 0.70,
                left: 0.25,
                right: 0.75,
            }),
            framing: "original".to_string(),
            is_emergency_fallback: false,
            fallback_reason: None,
            ..Default::default()
        };
        let ass_safeguarded = generate_ass_from_template_with_framing(
            &words,
            0.0,
            2.0,
            &template,
            Some(&intrusion_plan),
        );
        // Font size must be reduced from 84 (up to 15% reduction -> ~72-74)
        assert!(
            !ass_safeguarded.contains("Style: Default,Bebas Neue,84,"),
            "Intrusion must reduce font size below 84"
        );
    }

    #[test]
    fn test_dual_stack_caption_seam_placement() {
        let words = vec![
            TranscriptWord {
                text: "DUAL".to_string(),
                start: 0.20,
                end: 0.80,
                speaker: Some("S1".to_string()),
            },
            TranscriptWord {
                text: "FRAME".to_string(),
                start: 0.85,
                end: 1.50,
                speaker: Some("S2".to_string()),
            },
        ];

        let template = get_caption_template("preset_viral_bold").unwrap();
        let dual_plan = crate::media::SmartFramingPlan {
            mode: "dual_stack".to_string(),
            x: "0".to_string(),
            y: "0".to_string(),
            w: "1080".to_string(),
            h: "1920".to_string(),
            segments: vec![crate::media::LayoutSegment::DualStack {
                start: 0.0,
                end: 2.0,
                top_track_id: Some(1),
                bottom_track_id: Some(2),
                top_crop: crate::media::CropRectExpr {
                    x: "0".to_string(),
                    y: "0".to_string(),
                    w: "1080".to_string(),
                    h: "960".to_string(),
                },
                bottom_crop: crate::media::CropRectExpr {
                    x: "0".to_string(),
                    y: "960".to_string(),
                    w: "1080".to_string(),
                    h: "960".to_string(),
                },
                top_face_bounds: None,
                bottom_face_bounds: None,
            }],
            face_bounds: None,
            framing: "original".to_string(),
            is_emergency_fallback: false,
            fallback_reason: None,
            ..Default::default()
        };

        let ass =
            generate_ass_from_template_with_framing(&words, 0.0, 2.0, &template, Some(&dual_plan));
        assert!(
            ass.contains("Style: DualStackSeam,"),
            "Must define DualStackSeam style for dual-stack segments: {}",
            ass
        );
        assert!(
            ass.contains("Dialogue: 0,0:00:00.20,0:00:00.80,DualStackSeam,"),
            "Dialogue during dual stack segment must use DualStackSeam style: {}",
            ass
        );

        // Verify DualStackSeam MarginV:
        // Initial anchor Y=960. For font_size=84, 1 line: block_height=84.
        // MarginV = (1.0 - 0.50)*1920 - 42 = 960 - 42 = 918.
        assert!(
            ass.contains(",918,1"),
            "MarginV for Y=960 and size 84 must be 918: {}",
            ass
        );
    }

    #[test]
    fn test_dynamic_editorial_dual_stack_clamping() {
        let words = vec![
            TranscriptWord {
                text: "EDITORIAL".to_string(),
                start: 0.10,
                end: 0.50,
                speaker: Some("S1".to_string()),
            },
            TranscriptWord {
                text: "KINETIC".to_string(),
                start: 0.50,
                end: 0.90,
                speaker: Some("S1".to_string()),
            },
            TranscriptWord {
                text: "POWER".to_string(),
                start: 0.90,
                end: 1.40,
                speaker: Some("S1".to_string()),
            },
        ];

        let template = get_caption_template("preset_dynamic_editorial").unwrap();
        let dual_plan = crate::media::SmartFramingPlan {
            mode: "dual_stack".to_string(),
            x: "0".to_string(),
            y: "0".to_string(),
            w: "1080".to_string(),
            h: "1920".to_string(),
            segments: vec![crate::media::LayoutSegment::DualStack {
                start: 0.0,
                end: 2.0,
                top_track_id: Some(1),
                bottom_track_id: Some(2),
                top_crop: crate::media::CropRectExpr {
                    x: "0".to_string(),
                    y: "0".to_string(),
                    w: "1080".to_string(),
                    h: "960".to_string(),
                },
                bottom_crop: crate::media::CropRectExpr {
                    x: "0".to_string(),
                    y: "960".to_string(),
                    w: "1080".to_string(),
                    h: "960".to_string(),
                },
                top_face_bounds: None,
                bottom_face_bounds: None,
            }],
            face_bounds: None,
            framing: "original".to_string(),
            is_emergency_fallback: false,
            fallback_reason: None,
            ..Default::default()
        };

        let ass = generate_dynamic_editorial_ass_with_framing(
            &words,
            0.0,
            2.0,
            &template,
            Some(&dual_plan),
        );

        // Every word's target Y in \move must be clamped within the seam corridor [880, 1040]
        for line in ass.lines().filter(|l| l.starts_with("Dialogue:")) {
            // Find \move(start_x, start_y, target_x, target_y, 0, duration)
            if let Some(move_idx) = line.find(r"\move(") {
                let move_str = &line[move_idx + 6..];
                let close_paren = move_str
                    .find(')')
                    .expect("Move tag must have closing paren");
                let coords: Vec<&str> = move_str[..close_paren].split(',').collect();
                assert_eq!(coords.len(), 6);
                let target_y: i64 = coords[3].parse().expect("target_y must be integer");
                assert!(
                    (880..=1040).contains(&target_y),
                    "Target Y must be clamped to safe seam corridor [880, 1040], found: {}",
                    target_y
                );
            }
        }
    }

    #[test]
    fn test_cinematic_vlog_no_repeated_flicker_within_single_caption_event() {
        // A multi-word sentence spoken contiguously in a single frame
        let words = vec![
            TranscriptWord {
                text: "You".to_string(),
                start: 0.00,
                end: 0.25,
                speaker: Some("S1".to_string()),
            },
            TranscriptWord {
                text: "know,".to_string(),
                start: 0.25,
                end: 0.50,
                speaker: Some("S1".to_string()),
            },
            TranscriptWord {
                text: "what".to_string(),
                start: 0.50,
                end: 0.75,
                speaker: Some("S1".to_string()),
            },
            TranscriptWord {
                text: "do".to_string(),
                start: 0.75,
                end: 1.00,
                speaker: Some("S1".to_string()),
            },
            TranscriptWord {
                text: "people".to_string(),
                start: 1.00,
                end: 1.30,
                speaker: Some("S1".to_string()),
            },
            TranscriptWord {
                text: "not".to_string(),
                start: 1.30,
                end: 1.55,
                speaker: Some("S1".to_string()),
            },
            TranscriptWord {
                text: "see".to_string(),
                start: 1.55,
                end: 1.90,
                speaker: Some("S1".to_string()),
            },
        ];

        let template = get_caption_template("preset_cinematic_vlog")
            .expect("Cinematic Vlog template must exist");
        let ass = generate_ass_from_template(&words, 0.0, 2.0, &template);

        let dialogue_lines: Vec<&str> = ass
            .lines()
            .filter(|l| l.starts_with("Dialogue: "))
            .collect();
        // Regression test: A single caption event must NOT be split into per-word dialogue lines (which caused repeated flicker).
        // It must emit exactly ONE stable dialogue line for the whole frame.
        assert_eq!(
            dialogue_lines.len(),
            1,
            "Cinematic Vlog must generate exactly 1 dialogue line for a single caption event, got {}",
            dialogue_lines.len()
        );

        let dialogue = dialogue_lines[0];
        // Must contain the full sentence
        assert!(dialogue.contains("You know, what do"));
        assert!(dialogue.contains("people not see"));
        // Must start at 0.00 and end at 1.90
        assert!(dialogue.contains("0:00:00.00,0:00:01.90"));
        // Must have smooth fade tag
        assert!(dialogue.contains(r"{\fad(150,150)}"));
        // Must NOT contain per-word inline color highlight tags that cause opacity/color flipping
        assert!(!dialogue.contains(r"{\c&HFFFFFF&}"));
    }

    #[test]
    fn test_segment_aware_plan_deserialization_and_segment_at() {
        let json_str = r#"{
            "mode": "dual_stack",
            "x": "0", "y": "0", "w": "1080", "h": "1920",
            "segments": [
                {
                    "mode": "single",
                    "start": 0.0,
                    "end": 3.0,
                    "crop": { "x": "400", "y": "0", "w": "608", "h": "1080" },
                    "face_bounds": { "top": 0.20, "bottom": 0.45, "left": 0.35, "right": 0.65 }
                },
                {
                    "mode": "dual_stack",
                    "start": 3.0,
                    "end": 8.0,
                    "top_track_id": 1,
                    "bottom_track_id": 2,
                    "top_crop": { "x": "200", "y": "0", "w": "1080", "h": "960" },
                    "bottom_crop": { "x": "400", "y": "960", "w": "1080", "h": "960" },
                    "top_face_bounds": { "top": 0.10, "bottom": 0.35, "left": 0.30, "right": 0.60 },
                    "bottom_face_bounds": { "top": 0.60, "bottom": 0.85, "left": 0.40, "right": 0.70 }
                }
            ],
            "face_bounds": { "top": 0.20, "bottom": 0.45, "left": 0.35, "right": 0.65 }
        }"#;

        let plan: crate::media::SmartFramingPlan = serde_json::from_str(json_str)
            .expect("Must deserialize plan with segment-level face bounds");

        assert_eq!(plan.segments.len(), 2);

        // Test segment 0 (single)
        let seg0 = plan.segment_at(1.5).expect("Must find segment at t=1.5");
        match seg0 {
            crate::media::LayoutSegment::Single { face_bounds, .. } => {
                let fb = face_bounds
                    .as_ref()
                    .expect("Single segment must have face_bounds");
                assert_eq!(fb.top, 0.20);
                assert_eq!(fb.bottom, 0.45);
            }
            _ => panic!("Expected single segment at t=1.5"),
        }

        // Test segment 1 (dual_stack)
        let seg1 = plan.segment_at(5.0).expect("Must find segment at t=5.0");
        match seg1 {
            crate::media::LayoutSegment::DualStack {
                top_face_bounds,
                bottom_face_bounds,
                ..
            } => {
                let top_fb = top_face_bounds
                    .as_ref()
                    .expect("Dual segment must have top_face_bounds");
                assert_eq!(top_fb.top, 0.10);
                assert_eq!(top_fb.bottom, 0.35);

                let bot_fb = bottom_face_bounds
                    .as_ref()
                    .expect("Dual segment must have bottom_face_bounds");
                assert_eq!(bot_fb.top, 0.60);
                assert_eq!(bot_fb.bottom, 0.85);
            }
            _ => panic!("Expected dual segment at t=5.0"),
        }

        // Test edge boundary clamping
        assert!(plan.segment_at(-1.0).is_some());
        assert!(plan.segment_at(10.0).is_some());
    }

    #[test]
    fn test_dualframe_safe_corridor_dynamic_validation_and_collapse_fallback() {
        let words = vec![
            TranscriptWord {
                text: "COLLAPSE".to_string(),
                start: 0.10,
                end: 0.50,
                speaker: Some("S1".to_string()),
            },
            TranscriptWord {
                text: "TEST".to_string(),
                start: 0.50,
                end: 1.00,
                speaker: Some("S1".to_string()),
            },
        ];
        let template = get_caption_template("preset_dynamic_editorial").unwrap();

        // Construct plan where safe_top (e.g. 0.50 * 1920 + 24 = 984px) >= safe_bottom (e.g. 0.51 * 1920 - 10 = 969px)
        let inverted_plan = crate::media::SmartFramingPlan {
            mode: "dual_stack".to_string(),
            x: "0".to_string(),
            y: "0".to_string(),
            w: "1080".to_string(),
            h: "1920".to_string(),
            segments: vec![crate::media::LayoutSegment::DualStack {
                start: 0.0,
                end: 2.0,
                top_track_id: Some(1),
                bottom_track_id: Some(2),
                top_crop: crate::media::CropRectExpr {
                    x: "0".to_string(),
                    y: "0".to_string(),
                    w: "1080".to_string(),
                    h: "960".to_string(),
                },
                bottom_crop: crate::media::CropRectExpr {
                    x: "0".to_string(),
                    y: "960".to_string(),
                    w: "1080".to_string(),
                    h: "960".to_string(),
                },
                top_face_bounds: Some(SubjectFaceBounds {
                    top: 0.20,
                    bottom: 0.50,
                    left: 0.30,
                    right: 0.60,
                }),
                bottom_face_bounds: Some(SubjectFaceBounds {
                    top: 0.51,
                    bottom: 0.80,
                    left: 0.30,
                    right: 0.60,
                }),
            }],
            face_bounds: None,
            framing: "original".to_string(),
            is_emergency_fallback: false,
            fallback_reason: None,
            ..Default::default()
        };

        // Must NOT panic, must fall back safely to [880, 1040] corridor
        let ass = generate_dynamic_editorial_ass_with_framing(
            &words,
            0.0,
            2.0,
            &template,
            Some(&inverted_plan),
        );
        for line in ass.lines().filter(|l| l.starts_with("Dialogue:")) {
            if let Some(move_idx) = line.find(r"\move(") {
                let move_str = &line[move_idx + 6..];
                let close_paren = move_str
                    .find(')')
                    .expect("Move tag must have closing paren");
                let coords: Vec<&str> = move_str[..close_paren].split(',').collect();
                let target_y: i64 = coords[3].parse().expect("target_y must be integer");
                assert!(
                    (880..=1040).contains(&target_y),
                    "Target Y must fall back to safe default corridor [880, 1040] when corridor collapses, got: {}",
                    target_y
                );
            }
        }
    }

    #[test]
    fn test_templates_1_to_4_isolated_from_dual_ghost_bounds() {
        let words = vec![
            TranscriptWord {
                text: "CLEAN".to_string(),
                start: 0.10,
                end: 0.50,
                speaker: Some("S1".to_string()),
            },
            TranscriptWord {
                text: "TYPOGRAPHY".to_string(),
                start: 0.50,
                end: 1.00,
                speaker: Some("S1".to_string()),
            },
        ];
        let template = get_caption_template("preset_viral_bold").unwrap();

        // Plan with dual segment containing face bounds, but NO single segments with face bounds
        let dual_only_plan = crate::media::SmartFramingPlan {
            mode: "dual_stack".to_string(),
            x: "0".to_string(),
            y: "0".to_string(),
            w: "1080".to_string(),
            h: "1920".to_string(),
            segments: vec![crate::media::LayoutSegment::DualStack {
                start: 0.0,
                end: 2.0,
                top_track_id: Some(1),
                bottom_track_id: Some(2),
                top_crop: crate::media::CropRectExpr {
                    x: "0".to_string(),
                    y: "0".to_string(),
                    w: "1080".to_string(),
                    h: "960".to_string(),
                },
                bottom_crop: crate::media::CropRectExpr {
                    x: "0".to_string(),
                    y: "960".to_string(),
                    w: "1080".to_string(),
                    h: "960".to_string(),
                },
                top_face_bounds: Some(SubjectFaceBounds {
                    top: 0.10,
                    bottom: 0.40,
                    left: 0.30,
                    right: 0.60,
                }),
                bottom_face_bounds: Some(SubjectFaceBounds {
                    top: 0.60,
                    bottom: 0.90,
                    left: 0.30,
                    right: 0.60,
                }),
            }],
            // Legacy root face_bounds that should NOT affect the Default style because segment metadata exists!
            face_bounds: Some(SubjectFaceBounds {
                top: 0.60,
                bottom: 0.90,
                left: 0.30,
                right: 0.60,
            }),
            framing: "original".to_string(),
            is_emergency_fallback: false,
            fallback_reason: None,
            ..Default::default()
        };

        let ass = generate_ass_from_template_with_framing(
            &words,
            0.0,
            2.0,
            &template,
            Some(&dual_only_plan),
        );
        // Default style font size must match template's original 84 (not shrunken by ghost intrusion)
        assert!(
            ass.contains("Style: Default,Bebas Neue,84,"),
            "Default style must preserve original font size 84: {}",
            ass
        );
    }

    #[test]
    fn test_nonzero_start_sec_segment_aware_dual_style_selection() {
        // Regression test for the absolute-vs-clip-relative segment lookup bug:
        // LayoutSegment start/end are clip-relative [0, clip_dur], but word timestamps are
        // absolute source-video times. With start_sec > 0, an unconverted lookup always
        // exceeded every segment bound and fell through to the LAST segment, so every
        // caption event inherited the last segment's mode instead of the active one.
        let words = vec![
            TranscriptWord {
                text: "SINGLE".to_string(),
                start: 100.20,
                end: 100.80,
                speaker: Some("S1".to_string()),
            },
            TranscriptWord {
                text: "OPEN".to_string(),
                start: 100.85,
                end: 101.50,
                speaker: Some("S1".to_string()),
            },
            TranscriptWord {
                text: "DUAL".to_string(),
                start: 103.20,
                end: 103.80,
                speaker: Some("S1".to_string()),
            },
            TranscriptWord {
                text: "FRAME".to_string(),
                start: 103.85,
                end: 104.50,
                speaker: Some("S2".to_string()),
            },
            TranscriptWord {
                text: "CLOSE".to_string(),
                start: 107.20,
                end: 107.80,
                speaker: Some("S1".to_string()),
            },
            TranscriptWord {
                text: "OUT".to_string(),
                start: 107.85,
                end: 108.50,
                speaker: Some("S1".to_string()),
            },
        ];

        let template = get_caption_template("preset_viral_bold").unwrap();

        // Clip-relative segments: Single [0,3], DualStack [3,6], Single [6,10].
        // The plan ENDS on a Single segment — the exact configuration that previously
        // forced every event to Default even during the dual window.
        let mixed_plan = crate::media::SmartFramingPlan {
            mode: "dual_stack".to_string(),
            x: "0".to_string(),
            y: "0".to_string(),
            w: "1080".to_string(),
            h: "1920".to_string(),
            segments: vec![
                crate::media::LayoutSegment::Single {
                    start: 0.0,
                    end: 3.0,
                    crop: crate::media::CropRectExpr {
                        x: "0".to_string(),
                        y: "0".to_string(),
                        w: "1080".to_string(),
                        h: "1920".to_string(),
                    },
                    face_bounds: Some(SubjectFaceBounds {
                        top: 0.10,
                        bottom: 0.40,
                        left: 0.30,
                        right: 0.70,
                    }),
                },
                crate::media::LayoutSegment::DualStack {
                    start: 3.0,
                    end: 6.0,
                    top_track_id: Some(1),
                    bottom_track_id: Some(2),
                    top_crop: crate::media::CropRectExpr {
                        x: "0".to_string(),
                        y: "0".to_string(),
                        w: "1080".to_string(),
                        h: "960".to_string(),
                    },
                    bottom_crop: crate::media::CropRectExpr {
                        x: "0".to_string(),
                        y: "960".to_string(),
                        w: "1080".to_string(),
                        h: "960".to_string(),
                    },
                    top_face_bounds: Some(SubjectFaceBounds {
                        top: 0.10,
                        bottom: 0.35,
                        left: 0.30,
                        right: 0.60,
                    }),
                    bottom_face_bounds: Some(SubjectFaceBounds {
                        top: 0.60,
                        bottom: 0.85,
                        left: 0.40,
                        right: 0.70,
                    }),
                },
                crate::media::LayoutSegment::Single {
                    start: 6.0,
                    end: 10.0,
                    crop: crate::media::CropRectExpr {
                        x: "0".to_string(),
                        y: "0".to_string(),
                        w: "1080".to_string(),
                        h: "1920".to_string(),
                    },
                    face_bounds: Some(SubjectFaceBounds {
                        top: 0.10,
                        bottom: 0.40,
                        left: 0.30,
                        right: 0.70,
                    }),
                },
            ],
            face_bounds: None,
            framing: "original".to_string(),
            is_emergency_fallback: false,
            fallback_reason: None,
            ..Default::default()
        };

        let ass = generate_ass_from_template_with_framing(
            &words,
            100.0,
            110.0,
            &template,
            Some(&mixed_plan),
        );

        // Words at absolute 103.20–104.50 are clip-relative 3.20–4.50 → inside the DualStack
        // segment → their events MUST use DualStackSeam (previously all got Default).
        assert!(
            ass.contains("Dialogue: 0,0:00:03.20,0:00:03.80,DualStackSeam,"),
            "Word inside dual window (clip-relative 3.20-3.80) must use DualStackSeam style: {}",
            ass
        );
        assert!(
            ass.contains("Dialogue: 0,0:00:03.85,0:00:04.50,DualStackSeam,"),
            "Second word inside dual window must use DualStackSeam style: {}",
            ass
        );

        // Words in single windows (clip-relative 0.20-1.50 and 7.20-8.50) must use Default.
        assert!(
            ass.contains("Dialogue: 0,0:00:00.20,0:00:00.80,Default,"),
            "Word in first single window must use Default style: {}",
            ass
        );
        assert!(
            ass.contains("Dialogue: 0,0:00:07.20,0:00:07.80,Default,"),
            "Word in closing single window must use Default style: {}",
            ass
        );

        // Segment-transition correctness: events on both sides of the 3.0s and 6.0s
        // boundaries resolve to the correct side (no last-segment bleed-through).
        let dual_events = ass
            .lines()
            .filter(|l| l.contains(",DualStackSeam,"))
            .count();
        let default_events = ass
            .lines()
            .filter(|l| l.starts_with("Dialogue:") && l.contains(",Default,"))
            .count();
        assert!(
            dual_events >= 2,
            "Must have DualStackSeam events during the dual window, found {}",
            dual_events
        );
        assert!(
            default_events >= 4,
            "Must have Default events during single windows, found {}",
            default_events
        );
    }

    #[test]
    fn test_nonzero_start_sec_dual_seam_respects_bottom_face_bounds() {
        // With start_sec > 0, the DualStackSeam corridor must still be derived from the
        // active dual segment's face geometry: the lower caption must never fall into the
        // lower speaker's face region. Bottom face top at 0.60 (1152px) → hairline
        // 1152 - 0.20*(1620-1152) - 10 = 1048.6 → clamped to [960,1120] → 1048.6.
        // Top face bottom 0.35 (672px) + 24 = 696 → clamped to [800,960] → 800.
        // Corridor [800, 1048.6]; anchor 960 with block 84 fits (960+42=1002 <= 1048.6).
        let words = vec![
            TranscriptWord {
                text: "DUAL".to_string(),
                start: 503.20,
                end: 503.80,
                speaker: Some("S1".to_string()),
            },
            TranscriptWord {
                text: "FRAME".to_string(),
                start: 503.85,
                end: 504.50,
                speaker: Some("S2".to_string()),
            },
        ];

        let template = get_caption_template("preset_viral_bold").unwrap();

        let dual_plan = crate::media::SmartFramingPlan {
            mode: "dual_stack".to_string(),
            x: "0".to_string(),
            y: "0".to_string(),
            w: "1080".to_string(),
            h: "1920".to_string(),
            segments: vec![crate::media::LayoutSegment::DualStack {
                start: 3.0,
                end: 6.0,
                top_track_id: Some(1),
                bottom_track_id: Some(2),
                top_crop: crate::media::CropRectExpr {
                    x: "0".to_string(),
                    y: "0".to_string(),
                    w: "1080".to_string(),
                    h: "960".to_string(),
                },
                bottom_crop: crate::media::CropRectExpr {
                    x: "0".to_string(),
                    y: "960".to_string(),
                    w: "1080".to_string(),
                    h: "960".to_string(),
                },
                top_face_bounds: Some(SubjectFaceBounds {
                    top: 0.10,
                    bottom: 0.35,
                    left: 0.30,
                    right: 0.60,
                }),
                bottom_face_bounds: Some(SubjectFaceBounds {
                    top: 0.60,
                    bottom: 0.85,
                    left: 0.40,
                    right: 0.70,
                }),
            }],
            face_bounds: None,
            framing: "original".to_string(),
            is_emergency_fallback: false,
            fallback_reason: None,
            ..Default::default()
        };

        let ass = generate_ass_from_template_with_framing(
            &words,
            500.0,
            510.0,
            &template,
            Some(&dual_plan),
        );

        // Events during the dual window must use DualStackSeam (clip-relative 3.20-4.50).
        assert!(
            ass.contains("Dialogue: 0,0:00:03.20,0:00:03.80,DualStackSeam,"),
            "Dual-window word must use DualStackSeam style: {}",
            ass
        );

        // Extract DualStackSeam MarginV and verify the caption block stays above the
        // lower speaker's hairline (bottom face top 1152px): block bottom = MarginV + block_h
        // must be <= 1046 + tolerance. Anchor 960, block 84 → MarginV 918, block bottom 1002.
        let style_line = ass
            .lines()
            .find(|l| l.starts_with("Style: DualStackSeam,"))
            .expect("DualStackSeam style must exist");
        let fields: Vec<&str> = style_line.split(',').collect();
        // Format: Name,Fontname,Fontsize,...,Alignment,MarginL,MarginR,MarginV,Encoding
        // MarginV is field index 21 (0-based).
        let margin_v: i64 = fields[21].trim().parse().expect("MarginV must be integer");
        let font_size: i64 = fields[2].trim().parse().expect("Fontsize must be integer");
        let block_h = font_size; // 1 line template
        let block_bottom = margin_v + block_h;
        assert!(
            block_bottom <= 1049,
            "DualStackSeam caption block bottom ({}px) must stay above lower speaker hairline (~1046px)",
            block_bottom
        );
        assert!(
            margin_v >= 800 - 42,
            "DualStackSeam caption must not be pushed above the corridor top (800px): MarginV {}",
            margin_v
        );
    }

    #[test]
    fn test_nonzero_start_sec_dynamic_editorial_dual_group_resolution() {
        // Template 5 (Dynamic Editorial) resolves the segment from the word-group midpoint.
        // With start_sec > 0 the midpoint must be converted to clip-relative time before the
        // lookup, otherwise the last segment always wins.
        // A single-word group deterministically uses preferred single-speaker center (y=1245):
        //  - single window: word stays at y=1245 (no seam shift)
        //  - dual window: word is shifted/clamped into the seam corridor [800, 1046]
        let single_word = vec![TranscriptWord {
            text: "EDITORIAL".to_string(),
            start: 200.10,
            end: 200.60,
            speaker: Some("S1".to_string()),
        }];
        let dual_word = vec![TranscriptWord {
            text: "EDITORIAL".to_string(),
            start: 203.10,
            end: 203.60,
            speaker: Some("S1".to_string()),
        }];

        let template = get_caption_template("preset_dynamic_editorial").unwrap();

        // Plan: Single [0,2] then DualStack [2,6]. Single word at absolute 200.10-200.60 →
        // clip-relative 0.10-0.60 → Single segment. Dual word at absolute 203.10-203.60 →
        // clip-relative 3.10-3.60 → DualStack segment.
        let mixed_plan = crate::media::SmartFramingPlan {
            mode: "dual_stack".to_string(),
            x: "0".to_string(),
            y: "0".to_string(),
            w: "1080".to_string(),
            h: "1920".to_string(),
            segments: vec![
                crate::media::LayoutSegment::Single {
                    start: 0.0,
                    end: 2.0,
                    crop: crate::media::CropRectExpr {
                        x: "0".to_string(),
                        y: "0".to_string(),
                        w: "1080".to_string(),
                        h: "1920".to_string(),
                    },
                    face_bounds: Some(SubjectFaceBounds {
                        top: 0.10,
                        bottom: 0.40,
                        left: 0.30,
                        right: 0.70,
                    }),
                },
                crate::media::LayoutSegment::DualStack {
                    start: 2.0,
                    end: 6.0,
                    top_track_id: Some(1),
                    bottom_track_id: Some(2),
                    top_crop: crate::media::CropRectExpr {
                        x: "0".to_string(),
                        y: "0".to_string(),
                        w: "1080".to_string(),
                        h: "960".to_string(),
                    },
                    bottom_crop: crate::media::CropRectExpr {
                        x: "0".to_string(),
                        y: "960".to_string(),
                        w: "1080".to_string(),
                        h: "960".to_string(),
                    },
                    top_face_bounds: Some(SubjectFaceBounds {
                        top: 0.10,
                        bottom: 0.35,
                        left: 0.30,
                        right: 0.60,
                    }),
                    bottom_face_bounds: Some(SubjectFaceBounds {
                        top: 0.60,
                        bottom: 0.85,
                        left: 0.40,
                        right: 0.70,
                    }),
                },
            ],
            face_bounds: None,
            framing: "original".to_string(),
            is_emergency_fallback: false,
            fallback_reason: None,
            ..Default::default()
        };

        // Single-window word: placement y=1245 must be preserved (no seam shift).
        let ass_single = generate_dynamic_editorial_ass_with_framing(
            &single_word,
            200.0,
            206.0,
            &template,
            Some(&mixed_plan),
        );
        let single_target_y = ass_single
            .lines()
            .find(|l| l.starts_with("Dialogue:"))
            .and_then(|l| l.find(r"\move(").map(|i| &l[i + 6..]))
            .and_then(|s| s.find(')').map(|c| &s[..c]))
            .and_then(|s| {
                s.split(',')
                    .nth(3)
                    .map(|v| v.parse::<i64>().expect("target_y must be integer"))
            });
        assert_eq!(
            single_target_y,
            Some(1245),
            "Single-window word must keep placement y=1245 (no seam corridor shift): {:?}",
            single_target_y
        );

        // Dual-window word: placement must be shifted into the seam corridor [800, 1046].
        // Corridor: top = 0.35*1920+24 = 696 → clamped to 800; bottom hairline =
        // 0.60*1920 - 0.20*(0.85-0.60)*1920 - 10 = 1152 - 96 - 10 = 1046.
        let ass_dual = generate_dynamic_editorial_ass_with_framing(
            &dual_word,
            200.0,
            206.0,
            &template,
            Some(&mixed_plan),
        );
        let dual_target_y = ass_dual
            .lines()
            .find(|l| l.starts_with("Dialogue:"))
            .and_then(|l| l.find(r"\move(").map(|i| &l[i + 6..]))
            .and_then(|s| s.find(')').map(|c| &s[..c]))
            .and_then(|s| {
                s.split(',')
                    .nth(3)
                    .map(|v| v.parse::<i64>().expect("target_y must be integer"))
            });
        assert!(
            matches!(dual_target_y, Some(y) if (800..=1046).contains(&y)),
            "Dual-window word must be shifted into the seam corridor [800, 1046], got: {:?}",
            dual_target_y
        );
        assert_ne!(
            dual_target_y,
            Some(1150),
            "Dual-window word must NOT keep the unshifted hero_center placement"
        );
    }

    // Real-data verification harness: reads plan/words JSON from env vars set by
    // scripts/verify_dualframe_fix_real_render.py and generates the ASS through the
    // actual production code path. Skipped when env vars are absent.
    #[test]
    fn verify_real_clip01_dualframe_fix() {
        let plan_path = match std::env::var("AUTOSHORTS_VERIFY_PLAN") {
            Ok(p) => p,
            Err(_) => return,
        };
        let words_path = std::env::var("AUTOSHORTS_VERIFY_WORDS").expect("AUTOSHORTS_VERIFY_WORDS");
        let out_path = std::env::var("AUTOSHORTS_VERIFY_OUT").expect("AUTOSHORTS_VERIFY_OUT");

        let plan_json = std::fs::read_to_string(&plan_path).expect("plan JSON readable");
        let words_json = std::fs::read_to_string(&words_path).expect("words JSON readable");

        let plan: crate::media::SmartFramingPlan =
            serde_json::from_str(&plan_json).expect("plan deserializes");
        let raw_words: Vec<serde_json::Value> =
            serde_json::from_str(&words_json).expect("words deserialize");

        let words: Vec<TranscriptWord> = raw_words
            .iter()
            .map(|w| TranscriptWord {
                text: w["text"].as_str().unwrap_or("").to_string(),
                start: w["start"].as_f64().unwrap_or(0.0),
                end: w["end"].as_f64().unwrap_or(0.0),
                speaker: w["speaker"].as_str().map(|s| s.to_string()),
            })
            .collect();

        let start_sec = words.first().map(|w| w.start).unwrap_or(0.0);
        let end_sec = words.last().map(|w| w.end).unwrap_or(start_sec + 1.0);

        let template = get_caption_template("preset_viral_bold").unwrap();
        let ass = generate_ass_from_template_with_framing(
            &words,
            start_sec,
            end_sec,
            &template,
            Some(&plan),
        );

        std::fs::write(&out_path, &ass).expect("ASS written");
        println!("[verify] wrote {} ({} bytes)", out_path, ass.len());
    }

    // ── Template 6: Bhaukal Caption — dedicated suite (21 aspects) ──────────
    fn bhaukal_word(text: &str, start: f64, end: f64) -> TranscriptWord {
        TranscriptWord {
            text: text.to_string(),
            start,
            end,
            speaker: Some("S1".to_string()),
        }
    }

    fn bhaukal_template() -> CaptionTemplate {
        get_caption_template("preset_bhaukal_caption").expect("T6 registered")
    }

    fn bhaukal_intel(
        emph: Vec<usize>,
        hook: Vec<usize>,
        payoff: Vec<usize>,
    ) -> crate::caption_intel::CaptionIntelPlan {
        crate::caption_intel::CaptionIntelPlan {
            confidence: 0.92,
            emphasis_spans: vec![],
            emphasis_word_indices: emph,
            line_break_hints: vec![],
            hook_word_indices: hook,
            payoff_word_indices: payoff,
            reason: "test".to_string(),
        }
    }

    fn bhaukal_dialogues(ass: &str) -> Vec<&str> {
        ass.lines().filter(|l| l.starts_with("Dialogue:")).collect()
    }

    fn bhaukal_ts_to_secs(ts: &str) -> f64 {
        let ts = ts.trim();
        let (h, rest) = ts.split_once(':').expect("h");
        let (m, s) = rest.split_once(':').expect("m");
        h.parse::<f64>().unwrap_or(0.0) * 3600.0
            + m.parse::<f64>().unwrap_or(0.0) * 60.0
            + s.parse::<f64>().unwrap_or(0.0)
    }

    fn bhaukal_event_span(line: &str) -> (f64, f64) {
        let parts: Vec<&str> = line.split(',').collect();
        (bhaukal_ts_to_secs(parts[1]), bhaukal_ts_to_secs(parts[2]))
    }

    fn bhaukal_move_xy(line: &str) -> (i64, i64) {
        let i = line.find(r"\move(").expect("move tag");
        let inner = &line[i + 6..];
        let end = inner.find(')').expect("move end");
        let nums: Vec<i64> = inner[..end]
            .split(',')
            .map(|v| v.trim().parse::<i64>().expect("int"))
            .collect();
        (nums[2], nums[3])
    }

    #[test]
    fn bhaukal_01_registration_and_t1_t5_selectable() {
        let all = all_caption_templates();
        assert_eq!(all.len(), 7, "T1–T5 + T6 + T7");
        for id in [
            "preset_viral_bold",
            "preset_mrbeast_pop",
            "preset_minimal_capsule",
            "preset_cinematic_vlog",
            "preset_dynamic_editorial",
            "preset_bhaukal_caption",
            "preset_t7",
        ] {
            assert!(get_caption_template(id).is_some(), "selectable: {}", id);
        }
        let t6 = bhaukal_template();
        assert_eq!(t6.name, "Bhaukal caption");
        assert_eq!(t6.segmentation.max_words_per_line, 3);
        assert_eq!(t6.segmentation.max_lines_per_frame, 2);
        assert!(matches!(
            t6.active_state,
            CaptionActiveState::BhaukalKinetic { .. }
        ));
    }

    #[test]
    fn bhaukal_02_selection_dispatch_routes_to_bhaukal() {
        let words = vec![
            bhaukal_word("Hello", 0.1, 0.5),
            bhaukal_word("world", 0.5, 0.9),
        ];
        let t6 = bhaukal_template();
        let ass =
            generate_ass_from_template_with_framing_and_intel(&words, 0.0, 2.0, &t6, None, None, None);
        assert!(
            ass.contains("Style: HookStrip,Inter,34,"),
            "T6 header carries hook-strip style"
        );
        let intel = bhaukal_intel(vec![1], vec![], vec![]);
        let ass_intel = generate_ass_from_template_with_framing_and_intel(
            &words,
            0.0,
            2.0,
            &t6,
            None,
            Some(&intel),
            None,
        );
        assert!(
            ass_intel.contains(r"\i1"),
            "T6 hero words use italic approximation"
        );
        // Peer check: T1 through the same dispatcher keeps its own identity.
        let t1 = get_caption_template("preset_viral_bold").unwrap();
        let ass1 =
            generate_ass_from_template_with_framing_and_intel(&words, 0.0, 2.0, &t1, None, None, None);
        assert!(ass1.contains("Bebas Neue"), "T1 still Bebas");
        assert!(!ass1.contains("HookStrip"), "T1 gains no T6 layer");
    }

    #[test]
    fn bhaukal_03_typography_roles_scale_hierarchy() {
        let (_, normal_fs, _, normal_i, _, _) = bhaukal_role_style(BhaukalRole::Normal);
        let (_, emph_fs, _, _, _, _) = bhaukal_role_style(BhaukalRole::Emphasis);
        let (hero_font, hero_fs, _, hero_i, _, _) = bhaukal_role_style(BhaukalRole::Hero);
        let (_, metric_fs, _, _, _, _) = bhaukal_role_style(BhaukalRole::Metric);
        let (_, payoff_fs, _, _, _, _) = bhaukal_role_style(BhaukalRole::Payoff);
        assert!(
            normal_fs < emph_fs
                && emph_fs < hero_fs
                && hero_fs <= metric_fs
                && metric_fs <= payoff_fs,
            "supporting -> important -> hero -> payoff grows by scale"
        );
        assert_eq!(normal_i, 0);
        assert_eq!(hero_i, 1, "hero uses italic approximation");
        assert_eq!(hero_font, "Poppins");
    }

    #[test]
    fn bhaukal_04_layout_variants_are_left_biased_and_bounded() {
        for k in [1usize, 2, 3, 4] {
            for variant in get_bhaukal_layout_variants(k) {
                for p in &variant {
                    assert!((80..=1000).contains(&p.x), "x in canvas: {:?}", p);
                    assert!((400..=1500).contains(&p.y), "y in canvas: {:?}", p);
                }
            }
        }
        let solo = &get_bhaukal_layout_variants(1)[0][0];
        assert!(
            solo.x < 540,
            "default solo card is left-biased, got x={}",
            solo.x
        );
        assert_ne!(
            (solo.x, solo.y),
            (540, 1150),
            "T6 identity is distinct from T5 hero_center"
        );
    }

    #[test]
    fn bhaukal_05_emphasis_integration_elevates_scale() {
        let words = vec![
            bhaukal_word("you", 0.0, 0.3),
            bhaukal_word("will", 0.3, 0.6),
            bhaukal_word("WIN", 0.6, 1.0),
        ];
        let t6 = bhaukal_template();
        let plain = generate_bhaukal_ass_with_framing_and_intel(&words, 0.0, 2.0, &t6, None, None);
        let intel = bhaukal_intel(vec![2], vec![], vec![]);
        let emph =
            generate_bhaukal_ass_with_framing_and_intel(&words, 0.0, 2.0, &t6, None, Some(&intel));
        assert!(
            plain.contains(r"\fs44"),
            "no-intel card stays supporting scale"
        );
        assert!(
            !plain.contains(r"\fs76"),
            "no accidental hero without evidence"
        );
        assert!(
            emph.contains(r"\fs76"),
            "emphasized build hero renders at hero scale"
        );
    }

    #[test]
    fn bhaukal_06_hook_strip_only_with_hook_evidence() {
        let words = vec![
            bhaukal_word("Your", 0.0, 0.3),
            bhaukal_word("income", 0.3, 0.7),
            bhaukal_word("grows", 0.7, 1.1),
        ];
        let t6 = bhaukal_template();
        let no_intel =
            generate_bhaukal_ass_with_framing_and_intel(&words, 0.0, 3.0, &t6, None, None);
        assert!(
            bhaukal_dialogues(&no_intel)
                .iter()
                .all(|l| !l.starts_with("Dialogue: 1,")),
            "no hook strip without intel"
        );
        let intel = bhaukal_intel(vec![], vec![0, 1, 2], vec![]);
        let hooked =
            generate_bhaukal_ass_with_framing_and_intel(&words, 0.0, 3.0, &t6, None, Some(&intel));
        let hooked_lines = bhaukal_dialogues(&hooked);
        let strip: Vec<&&str> = hooked_lines
            .iter()
            .filter(|l| l.starts_with("Dialogue: 1,"))
            .collect();
        assert_eq!(strip.len(), 1, "exactly one persistent strip");
        assert!(
            strip[0].contains("HookStrip") && strip[0].contains("Your income grows"),
            "strip text = hook words"
        );
        let (s, e) = bhaukal_event_span(strip[0]);
        assert!(
            s.abs() < 1e-6 && (e - 3.0).abs() < 0.02,
            "strip spans the full clip"
        );
    }

    #[test]
    fn bhaukal_07_payoff_gets_strongest_scale() {
        let words = vec![
            bhaukal_word("this", 0.0, 0.3),
            bhaukal_word("changes", 0.3, 0.7),
            bhaukal_word("everything", 0.7, 1.2),
        ];
        let t6 = bhaukal_template();
        let intel = bhaukal_intel(vec![], vec![], vec![2]);
        let ass =
            generate_bhaukal_ass_with_framing_and_intel(&words, 0.0, 2.0, &t6, None, Some(&intel));
        assert!(ass.contains(r"\fs92"), "payoff renders at strongest scale");
    }

    #[test]
    fn bhaukal_08_metric_hero_only_with_evidence() {
        let words = vec![
            bhaukal_word("$1", 0.0, 0.4),
            bhaukal_word("million", 0.4, 0.9),
        ];
        let t6 = bhaukal_template();
        let plain = generate_bhaukal_ass_with_framing_and_intel(&words, 0.0, 2.0, &t6, None, None);
        assert!(
            !plain.contains(r"\fs84"),
            "bare numbers are not auto-heroed"
        );
        let intel = bhaukal_intel(vec![0, 1], vec![], vec![]);
        let hero =
            generate_bhaukal_ass_with_framing_and_intel(&words, 0.0, 2.0, &t6, None, Some(&intel));
        assert!(
            hero.contains(r"\fs84"),
            "$1 million becomes Metric hero with evidence"
        );
    }

    #[test]
    fn bhaukal_09_progressive_construction_accumulates() {
        let words = vec![
            bhaukal_word("you", 0.0, 0.25),
            bhaukal_word("will", 0.30, 0.55),
            bhaukal_word("make", 0.60, 0.90),
        ];
        let t6 = bhaukal_template();
        let ass = generate_bhaukal_ass_with_framing_and_intel(&words, 0.0, 2.0, &t6, None, None);
        let all09 = bhaukal_dialogues(&ass);
        let body: Vec<&&str> = all09
            .iter()
            .filter(|l| l.starts_with("Dialogue: 0,"))
            .collect();
        assert_eq!(body.len(), 3, "3-word group builds in 3 replacement steps");
        assert!(body[0].ends_with("}you"), "step 1 = context");
        assert!(
            body[1].contains("you") && body[1].ends_with("}will"),
            "step 2 = context + keyword"
        );
        assert!(body[2].ends_with("}make"), "step 3 = completed phrase");
    }

    #[test]
    fn bhaukal_10_phrase_timing_uses_real_word_times() {
        let words = vec![
            bhaukal_word("you", 10.0, 10.25),
            bhaukal_word("will", 10.30, 10.55),
            bhaukal_word("make", 10.60, 10.90),
        ];
        let t6 = bhaukal_template();
        let ass = generate_bhaukal_ass_with_framing_and_intel(&words, 10.0, 12.0, &t6, None, None);
        let all10 = bhaukal_dialogues(&ass);
        let body: Vec<&&str> = all10
            .iter()
            .filter(|l| l.starts_with("Dialogue: 0,"))
            .collect();
        assert_eq!(body.len(), 3);
        let spans: Vec<(f64, f64)> = body.iter().map(|l| bhaukal_event_span(l)).collect();
        assert!(
            (spans[0].0 - 0.0).abs() < 0.02
                && (spans[1].0 - 0.30).abs() < 0.02
                && (spans[2].0 - 0.60).abs() < 0.02,
            "step starts equal word starts (clip-relative): {:?}",
            spans
        );
        assert!(
            (spans[0].1 - 0.30).abs() < 0.02
                && (spans[1].1 - 0.60).abs() < 0.02
                && (spans[2].1 - 0.90).abs() < 0.02,
            "steps hand off at real word boundaries: {:?}",
            spans
        );
    }

    #[test]
    fn bhaukal_11_no_animation_spans_removed_regions() {
        // Post-pacing-style input: retained words with a removed middle.
        let words = vec![
            bhaukal_word("alpha", 0.0, 0.3),
            bhaukal_word("beta", 5.0, 5.3),
        ];
        let t6 = bhaukal_template();
        let ass = generate_bhaukal_ass_with_framing_and_intel(&words, 0.0, 6.0, &t6, None, None);
        for line in bhaukal_dialogues(&ass)
            .iter()
            .filter(|l| l.starts_with("Dialogue: 0,"))
        {
            let (s, e) = bhaukal_event_span(line);
            assert!(
                !(s < 2.5 && e > 2.5),
                "no event spans the removed gap: {:?}",
                (s, e)
            );
        }
    }

    #[test]
    fn bhaukal_12_positions_within_canvas_and_left_default() {
        let words = vec![
            bhaukal_word("open", 0.0, 0.4),
            bhaukal_word("left", 0.45, 0.9),
        ];
        let t6 = bhaukal_template();
        let ass = generate_bhaukal_ass_with_framing_and_intel(&words, 0.0, 2.0, &t6, None, None);
        assert!(ass.contains(r"\an5"), "middle-center anchor");
        let mut saw_left = false;
        for line in bhaukal_dialogues(&ass)
            .iter()
            .filter(|l| l.starts_with("Dialogue: 0,"))
        {
            let (x, y) = bhaukal_move_xy(line);
            assert!(
                (80..=1000).contains(&x) && (400..=1500).contains(&y),
                "safe canvas: {:?}",
                (x, y)
            );
            if x < 540 {
                saw_left = true;
            }
        }
        assert!(saw_left, "default cluster occupies left negative space");
    }

    #[test]
    fn bhaukal_13_face_safe_fallback_mirrors_and_pushes_down() {
        let words = vec![bhaukal_word("hero", 0.0, 0.6)];
        let t6 = bhaukal_template();
        let plan_with = |l: f64, r: f64, b: f64| crate::media::SmartFramingPlan {
            mode: "single".to_string(),
            x: "0".to_string(),
            y: "0".to_string(),
            w: "1080".to_string(),
            h: "1920".to_string(),
            segments: vec![],
            face_bounds: Some(SubjectFaceBounds {
                top: 0.20,
                bottom: b,
                left: l,
                right: r,
            }),
            framing: "original".to_string(),
            is_emergency_fallback: false,
            fallback_reason: None,
            ..Default::default()
        };
        // Speaker right -> caption stays left.
        let right = generate_bhaukal_ass_with_framing(
            &words,
            0.0,
            2.0,
            &t6,
            Some(&plan_with(0.55, 0.95, 0.45)),
        );
        for line in bhaukal_dialogues(&right)
            .iter()
            .filter(|l| l.starts_with("Dialogue: 0,"))
        {
            assert!(bhaukal_move_xy(line).0 < 540, "keeps open left space");
        }
        // Speaker left -> caption mirrors right, never over the face.
        let left = generate_bhaukal_ass_with_framing(
            &words,
            0.0,
            2.0,
            &t6,
            Some(&plan_with(0.05, 0.45, 0.45)),
        );
        for line in bhaukal_dialogues(&left)
            .iter()
            .filter(|l| l.starts_with("Dialogue: 0,"))
        {
            assert!(bhaukal_move_xy(line).0 > 540, "mirrors to open right space");
        }
        // Deep face -> pushed below chin + buffer.
        let deep = generate_bhaukal_ass_with_framing(
            &words,
            0.0,
            2.0,
            &t6,
            Some(&plan_with(0.30, 0.70, 0.75)),
        );
        for line in bhaukal_dialogues(&deep)
            .iter()
            .filter(|l| l.starts_with("Dialogue: 0,"))
        {
            let (_, y) = bhaukal_move_xy(line);
            assert!(
                (y as f64) >= 0.75 * 1920.0 + 24.0 - 40.0,
                "below chin + buffer: {}",
                y
            );
        }
    }

    #[test]
    fn bhaukal_14_width_safety_shrinks_overflow() {
        let words = vec![bhaukal_word(
            "supercalifragilisticexpialidociousness",
            0.0,
            0.8,
        )];
        let t6 = bhaukal_template();
        let ass = generate_bhaukal_ass_with_framing(&words, 0.0, 2.0, &t6, None);
        let line = bhaukal_dialogues(&ass)
            .into_iter()
            .find(|l| l.starts_with("Dialogue: 0,"))
            .expect("event");
        assert!(line.contains(r"\fs"), "sized event emitted");
        assert!(!line.contains(r"\fs76"), "overflow hero shrinks to fit");
        assert!(
            !line.contains(r"\fs0") && !line.contains(r"\fs1,"),
            "never collapses to unreadable"
        );
    }

    #[test]
    fn bhaukal_15_white_monochrome_no_rainbow() {
        let words = vec![
            bhaukal_word("you", 0.0, 0.3),
            bhaukal_word("will", 0.3, 0.6),
            bhaukal_word("Believe", 0.6, 1.0),
        ];
        let t6 = bhaukal_template();
        let intel = bhaukal_intel(vec![0, 1, 2], vec![0], vec![2]);
        let ass =
            generate_bhaukal_ass_with_framing_and_intel(&words, 0.0, 2.0, &t6, None, Some(&intel));
        for bad in [
            "&H00FFFF&",
            "&HFFFF00&",
            "&HFF0000&",
            "&H00FF00&",
            "&HFF00FF&",
            "&H00E5FF&",
            "&HFFE600&",
        ] {
            assert!(
                !ass.to_uppercase().contains(&bad.to_uppercase()),
                "no rainbow accent {}",
                bad
            );
        }
        assert!(ass.contains("&HFFFFFF&"), "white/near-white hierarchy");
    }

    #[test]
    fn bhaukal_16_pacing_remap_flows_through() {
        use crate::pacing::{RetainedInterval, SmartPacingPlan};
        let src = vec![
            bhaukal_word("keep", 0.0, 0.4),
            bhaukal_word("drop", 2.0, 2.4),
            bhaukal_word("keep2", 4.0, 4.5),
        ];
        let plan = SmartPacingPlan {
            status: "ok".to_string(),
            reason: None,
            clip_start_sec: 0.0,
            clip_end_sec: 5.0,
            output_duration_sec: 3.0,
            removed_total_sec: 2.0,
            edits: vec![],
            retained: vec![
                RetainedInterval {
                    src_start_sec: 0.0,
                    src_end_sec: 1.0,
                    out_start_sec: 0.0,
                    out_end_sec: 1.0,
                },
                RetainedInterval {
                    src_start_sec: 3.5,
                    src_end_sec: 5.0,
                    out_start_sec: 1.0,
                    out_end_sec: 2.5,
                },
            ],
        };
        let remapped = crate::pacing::remap_words(&plan, &src);
        assert_eq!(remapped.len(), 2, "removed word drops out");
        let t6 = bhaukal_template();
        let ass = generate_bhaukal_ass_with_framing(&remapped, 0.0, 2.5, &t6, None);
        assert!(!ass.contains("drop"), "removed speech never captioned");
        assert!(
            ass.contains("keep") && ass.contains("keep2"),
            "retained speech captioned on output timeline"
        );
    }

    #[test]
    fn bhaukal_17_discontinuous_intervals_stay_discontinuous() {
        let words = vec![
            bhaukal_word("first", 10.0, 10.4),
            bhaukal_word("second", 14.0, 14.5),
        ];
        let t6 = bhaukal_template();
        let ass = generate_bhaukal_ass_with_framing(&words, 10.0, 15.0, &t6, None);
        let spans: Vec<(f64, f64)> = bhaukal_dialogues(&ass)
            .iter()
            .filter(|l| l.starts_with("Dialogue: 0,"))
            .map(|l| bhaukal_event_span(l))
            .collect();
        assert_eq!(spans.len(), 2);
        assert!(
            spans[0].1 <= 0.45 && spans[1].0 >= 3.95,
            "cut region stays empty: {:?}",
            spans
        );
    }

    #[test]
    fn bhaukal_18_dualframe_seam_clamp() {
        let words = vec![
            bhaukal_word("level", 203.10, 203.50),
            bhaukal_word("of", 203.55, 203.80),
            bhaukal_word("belief", 203.85, 204.30),
        ];
        let t6 = bhaukal_template();
        let plan = crate::media::SmartFramingPlan {
            mode: "dual_stack".to_string(),
            x: "0".to_string(),
            y: "0".to_string(),
            w: "1080".to_string(),
            h: "1920".to_string(),
            segments: vec![crate::media::LayoutSegment::DualStack {
                start: 2.0,
                end: 6.0,
                top_track_id: Some(1),
                bottom_track_id: Some(2),
                top_crop: crate::media::CropRectExpr {
                    x: "0".to_string(),
                    y: "0".to_string(),
                    w: "1080".to_string(),
                    h: "960".to_string(),
                },
                bottom_crop: crate::media::CropRectExpr {
                    x: "0".to_string(),
                    y: "960".to_string(),
                    w: "1080".to_string(),
                    h: "960".to_string(),
                },
                top_face_bounds: Some(SubjectFaceBounds {
                    top: 0.10,
                    bottom: 0.35,
                    left: 0.30,
                    right: 0.60,
                }),
                bottom_face_bounds: Some(SubjectFaceBounds {
                    top: 0.60,
                    bottom: 0.85,
                    left: 0.40,
                    right: 0.70,
                }),
            }],
            face_bounds: None,
            framing: "original".to_string(),
            is_emergency_fallback: false,
            fallback_reason: None,
            ..Default::default()
        };
        let ass = generate_bhaukal_ass_with_framing(&words, 200.0, 206.0, &t6, Some(&plan));
        for line in bhaukal_dialogues(&ass)
            .iter()
            .filter(|l| l.starts_with("Dialogue: 0,"))
        {
            let (_, y) = bhaukal_move_xy(line);
            assert!(
                (800..=1046).contains(&y),
                "dual-window lockup inside seam corridor: {}",
                y
            );
        }
    }

    #[test]
    fn bhaukal_19_deterministic_output() {
        let words = vec![
            bhaukal_word("you", 0.0, 0.3),
            bhaukal_word("make", 0.35, 0.7),
            bhaukal_word("money", 0.75, 1.1),
        ];
        let t6 = bhaukal_template();
        let intel = bhaukal_intel(vec![1, 2], vec![0, 1], vec![2]);
        let a =
            generate_bhaukal_ass_with_framing_and_intel(&words, 0.0, 2.0, &t6, None, Some(&intel));
        let b =
            generate_bhaukal_ass_with_framing_and_intel(&words, 0.0, 2.0, &t6, None, Some(&intel));
        assert_eq!(a, b, "byte-identical across runs");
    }

    #[test]
    fn bhaukal_20_t1_t5_outputs_unchanged() {
        let words = vec![
            bhaukal_word("HELLO", 0.1, 0.6),
            bhaukal_word("WORLD", 0.6, 1.2),
        ];
        let t1 = get_caption_template("preset_viral_bold").unwrap();
        let a1 = generate_ass_from_template(&words, 0.0, 2.0, &t1);
        assert!(
            a1.contains("Bebas Neue") && a1.contains("HELLO"),
            "T1 intact"
        );
        let t5 = get_caption_template("preset_dynamic_editorial").unwrap();
        let a5 = generate_ass_from_template(&words, 0.0, 2.0, &t5);
        assert!(
            a5.contains("Montserrat") && a5.contains(r"\move("),
            "T5 kinetic intact"
        );
        for (name, ass) in [("T1", &a1), ("T5", &a5)] {
            assert!(
                !ass.contains("HookStrip") && !ass.contains("preset_bhaukal"),
                "{} free of T6 coupling",
                name
            );
        }
    }

    #[test]
    fn bhaukal_21_empty_and_out_of_range_words_emit_nothing() {
        let t6 = bhaukal_template();
        assert!(
            generate_bhaukal_ass(&[], 0.0, 2.0, &t6).is_empty(),
            "empty words"
        );
        let words = vec![bhaukal_word("late", 50.0, 51.0)];
        assert!(
            generate_bhaukal_ass(&words, 0.0, 2.0, &t6).is_empty(),
            "out-of-window words"
        );
    }

    // Real-data harness: reference-inspired script through the REAL Caption
    // Intelligence planner into the T6 generator. Writes ASS to
    // $AUTOSHORTS_BHAUKAL_OUT; skipped when unset. Used for the visual
    // reference-render comparison (burned with ffmpeg, CI enabled).
    #[test]
    fn bhaukal_real_render_harness() {
        let out_path = match std::env::var("AUTOSHORTS_BHAUKAL_OUT") {
            Ok(p) => p,
            Err(_) => return,
        };
        let script = [
            ("Your", 0.20, 0.45),
            ("income", 0.45, 0.85),
            ("grows", 0.85, 1.15),
            ("with", 1.15, 1.35),
            ("you", 1.35, 1.60),
            ("If", 2.00, 2.20),
            ("you", 2.20, 2.40),
            ("believe", 2.40, 2.90),
            ("you", 3.30, 3.50),
            ("will", 3.50, 3.70),
            ("make", 3.70, 4.10),
            ("money", 4.10, 4.60),
            ("to", 5.00, 5.15),
            ("your", 5.15, 5.40),
            ("level", 5.40, 5.90),
            ("of", 5.90, 6.05),
            ("Belief", 6.05, 6.60),
            ("You", 7.00, 7.20),
            ("make", 7.20, 7.50),
            ("$1", 7.50, 7.80),
            ("million", 7.80, 8.30),
        ];
        let words: Vec<TranscriptWord> = script
            .iter()
            .map(|(t, s, e)| bhaukal_word(t, *s, *e))
            .collect();
        let candidate = crate::models::Candidate {
            id: "cand-bhaukal".to_string(),
            project_id: "proj-bhaukal".to_string(),
            start_sec: 0.0,
            end_sec: 9.0,
            score: 0.93,
            hook: "Your income grows with you".to_string(),
            rationale: "reference-inspired harness".to_string(),
            rank: 1,
            selected: true,
            hook_start_sec: Some(0.2),
            hook_end_sec: Some(1.6),
            hook_confidence: Some(0.90),
            opening_context_score: Some(0.88),
            payoff_text: Some("You make $1 million".to_string()),
            payoff_start_sec: Some(7.0),
            payoff_end_sec: Some(8.3),
            payoff_score: Some(0.91),
            payoff_completion: Some(true),
            metadata_json: None,
        };
        let intel = crate::caption_intel::plan_caption_intel(&words, &candidate, None, 0.0, 9.0)
            .expect("harness intel plan must clear confidence gate");
        let t6 = bhaukal_template();
        let ass =
            generate_bhaukal_ass_with_framing_and_intel(&words, 0.0, 9.0, &t6, None, Some(&intel));
        assert!(ass.contains("Dialogue: 1,"), "harness renders hook strip");
        assert!(
            ass.contains(r"\fs84") || ass.contains(r"\fs92"),
            "harness heroes the metric/payoff"
        );
        std::fs::write(&out_path, &ass).expect("ASS written");
        println!(
            "[bhaukal-harness] intel={:?} wrote {} ({} bytes)",
            intel.reason,
            out_path,
            ass.len()
        );
    }

    fn t7_test_word(text: &str, start: f64, end: f64) -> TranscriptWord {
        TranscriptWord {
            text: text.to_string(),
            start,
            end,
            speaker: Some("S1".to_string()),
        }
    }

    #[test]
    fn test_t7_phrase_units_no_line_recycling() {
        let raw_words = vec![
            t7_test_word("if", 1.0, 1.2),
            t7_test_word("somebody", 1.2, 1.4), // 0.20s pause to 1.6s -> triggers speech chunk boundary (>0.12s)
            t7_test_word("who's", 1.6, 1.8),
            t7_test_word("watching", 1.8, 2.0), // 0.20s pause to 2.2s -> triggers speech chunk boundary
            t7_test_word("this", 2.2, 2.3),
            t7_test_word("wants", 2.3, 2.5),
            t7_test_word("to", 2.5, 2.7),
            t7_test_word("become", 2.7, 3.0),
        ];
        let word_refs: Vec<(usize, &TranscriptWord)> = raw_words.iter().enumerate().collect();
        let units = group_t7_phrase_units(&word_refs, 3, 1.5, 2.6, None);

        // Must form at least 2 distinct pairs
        assert!(
            units.len() >= 2,
            "Expected multiple phrase units, got {}",
            units.len()
        );

        // First unit must be pair ("if somebody", "who's watching")
        match &units[0] {
            T7PhraseUnit::Pair { first, second, .. } => {
                let first_txt: Vec<_> = first.iter().map(|(_, w)| w.text.as_str()).collect();
                let second_txt: Vec<_> = second.iter().map(|(_, w)| w.text.as_str()).collect();
                assert_eq!(first_txt, vec!["if", "somebody"]);
                assert_eq!(second_txt, vec!["who's", "watching"]);
            }
            _ => panic!("Expected first unit to be Pair, got {:?}", units[0]),
        }

        // Second unit must NOT reuse "who's watching" as its first line!
        match &units[1] {
            T7PhraseUnit::Pair { first, .. } => {
                let first_txt: Vec<_> = first.iter().map(|(_, w)| w.text.as_str()).collect();
                assert_eq!(
                    first_txt,
                    vec!["this", "wants", "to"],
                    "Line 2 must NEVER be recycled into Line 1 of next unit!"
                );
            }
            T7PhraseUnit::Solo { chunk, .. } => {
                let first_txt: Vec<_> = chunk.iter().map(|(_, w)| w.text.as_str()).collect();
                assert_eq!(
                    first_txt,
                    vec!["this", "wants", "to"],
                    "Line 2 must NEVER be recycled into Line 1 of next unit!"
                );
            }
        }
    }

    #[test]
    fn test_t7_phrase_units_solo_on_long_pause_and_terminator() {
        let raw_words = vec![
            t7_test_word("extraordinary.", 1.0, 1.4),
            t7_test_word("First", 2.5, 2.8),
            t7_test_word("of", 2.8, 2.9),
            t7_test_word("all", 2.9, 3.2),
        ];
        let word_refs: Vec<(usize, &TranscriptWord)> = raw_words.iter().enumerate().collect();
        let units = group_t7_phrase_units(&word_refs, 3, 1.5, 2.6, None);

        assert_eq!(units.len(), 2);
        match &units[0] {
            T7PhraseUnit::Solo { chunk, .. } => {
                assert_eq!(chunk[0].1.text, "extraordinary.");
            }
            _ => panic!("Expected sentence-terminator chunk to be Solo"),
        }
    }

    #[test]
    fn test_t7_ass_phrase_paired_structure() {
        let raw_words = vec![
            t7_test_word("if", 1.0, 1.2),
            t7_test_word("somebody", 1.2, 1.5),
            t7_test_word("who's", 1.6, 1.8),
            t7_test_word("watching", 1.8, 2.1),
            t7_test_word("this", 2.6, 2.8),
            t7_test_word("podcast.", 2.9, 3.3),
        ];
        let t7 = get_caption_template("preset_t7").expect("preset_t7 registered");
        let ass = generate_ass_from_template_with_framing(&raw_words, 0.0, 5.0, &t7, None);

        let lines: Vec<&str> = ass.lines().filter(|l| l.starts_with("Dialogue:")).collect();
        assert!(!lines.is_empty(), "T7 ASS must emit dialogue lines");

        // Verify pinned pos tags
        assert!(
            ass.contains(r"{\pos(540,1020)}"),
            "T7 must pin Line 1 at y=1020"
        );
        assert!(
            ass.contains(r"{\pos(540,1140)}"),
            "T7 must pin Line 2 at y=1140"
        );
        assert!(
            ass.contains(r"{\c&H07EFF9&}"),
            "T7 must highlight Line 2 in yellow #F9EF07"
        );

        // Verify no conveyor belt: Line 2 of unit 1 ("who's watching") must never appear at \pos(540,1020)
        for line in &lines {
            if line.contains(r"\pos(540,1020)") {
                assert!(
                    !line.contains("who's watching"),
                    "Line 2 must NEVER move upward into Line 1"
                );
            }
        }
    }

    #[test]
    fn test_t7_face_safe_avoidance_and_containment() {
        let raw_words = vec![
            t7_test_word("if", 1.0, 1.2),
            t7_test_word("somebody", 1.2, 1.5),
            t7_test_word("who's", 1.6, 1.8),
            t7_test_word("watching", 1.8, 2.1),
        ];
        let t7 = get_caption_template("preset_t7").expect("preset_t7 registered");

        // 1. High face: face bottom = 906px on 1920 canvas (norm = 906/1920 = 0.471875)
        // y1_target = 906 + 155 = 1061 <= 1360 -> y1 = max(1020, 1061) = 1061
        let high_face_plan = crate::media::SmartFramingPlan {
            mode: "single".to_string(),
            x: "0".to_string(),
            y: "0".to_string(),
            w: "1080".to_string(),
            h: "1080".to_string(),
            segments: vec![crate::media::LayoutSegment::Single {
                start: 0.0,
                end: 5.0,
                crop: crate::media::CropRectExpr {
                    x: "0".into(),
                    y: "0".into(),
                    w: "1080".into(),
                    h: "1080".into(),
                },
                face_bounds: Some(SubjectFaceBounds {
                    top: 0.15,
                    bottom: 906.0 / 1920.0,
                    left: 0.3,
                    right: 0.7,
                }),
            }],
            face_bounds: None,
            framing: "adaptive".to_string(),
            is_emergency_fallback: false,
            fallback_reason: None,
            ..Default::default()
        };
        let ass_high = generate_ass_from_template_with_framing(
            &raw_words,
            0.0,
            5.0,
            &t7,
            Some(&high_face_plan),
        );
        assert!(
            ass_high.contains(r"{\pos(540,1061)}"),
            "High face must clear face at y1=1061"
        );
        assert!(
            ass_high.contains(r"{\pos(540,1181)}"),
            "High face must place y2=1181"
        );

        // 2. Mid face: face bottom = 1122px on 1920 canvas (norm = 1122/1920 = 0.584375)
        // y1_target = 1122 + 155 = 1277 <= 1360 -> y1 = max(1020, 1277) = 1277, y2 = 1397 <= 1480
        let mid_face_plan = crate::media::SmartFramingPlan {
            mode: "single".to_string(),
            x: "0".to_string(),
            y: "0".to_string(),
            w: "1080".to_string(),
            h: "1080".to_string(),
            segments: vec![crate::media::LayoutSegment::Single {
                start: 0.0,
                end: 5.0,
                crop: crate::media::CropRectExpr {
                    x: "0".into(),
                    y: "0".into(),
                    w: "1080".into(),
                    h: "1080".into(),
                },
                face_bounds: Some(SubjectFaceBounds {
                    top: 0.35,
                    bottom: 1122.0 / 1920.0,
                    left: 0.3,
                    right: 0.7,
                }),
            }],
            face_bounds: None,
            framing: "adaptive".to_string(),
            is_emergency_fallback: false,
            fallback_reason: None,
            ..Default::default()
        };
        let ass_mid = generate_ass_from_template_with_framing(
            &raw_words,
            0.0,
            5.0,
            &t7,
            Some(&mid_face_plan),
        );
        assert!(
            ass_mid.contains(r"{\pos(540,1277)}"),
            "Mid face must adjust to y1=1277 to clear face"
        );
        assert!(
            ass_mid.contains(r"{\pos(540,1397)}"),
            "Mid face must adjust to y2=1397 inside active square"
        );

        // 3. Low face: face bottom = 1338px on 1920 canvas (norm = 1338/1920 = 0.696875)
        // y1_target = 1338 + 155 = 1493 > 1360 -> preserve face visibility first! y1 = 1493, y2 = 1613
        let low_face_plan = crate::media::SmartFramingPlan {
            mode: "single".to_string(),
            x: "0".to_string(),
            y: "0".to_string(),
            w: "1080".to_string(),
            h: "1080".to_string(),
            segments: vec![crate::media::LayoutSegment::Single {
                start: 0.0,
                end: 5.0,
                crop: crate::media::CropRectExpr {
                    x: "0".into(),
                    y: "0".into(),
                    w: "1080".into(),
                    h: "1080".into(),
                },
                face_bounds: Some(SubjectFaceBounds {
                    top: 0.55,
                    bottom: 1338.0 / 1920.0,
                    left: 0.3,
                    right: 0.7,
                }),
            }],
            face_bounds: None,
            framing: "adaptive".to_string(),
            is_emergency_fallback: false,
            fallback_reason: None,
            ..Default::default()
        };
        let ass_low = generate_ass_from_template_with_framing(
            &raw_words,
            0.0,
            5.0,
            &t7,
            Some(&low_face_plan),
        );
        assert!(
            ass_low.contains(r"{\pos(540,1493)}"),
            "Low face must preserve face visibility first by placing y1=1493"
        );
        assert!(
            ass_low.contains(r"{\pos(540,1613)}"),
            "Low face must place y2=1613 in lower region"
        );
    }

    /// Regression guard for the T7 face-safety invariant (Phase 4 wiring audit,
    /// blocker B4). The audit reported that T7 "opts out" of the subject face
    /// safeguard. That exemption lived in the GENERIC single-layout path
    /// (`apply_subject_face_safeguard`), which SHRINKS the font — but it was
    /// unreachable for T7, because T7 is dispatched earlier to its own
    /// phrase-paired generator with a real POSITIONAL solver.
    ///
    /// This test pins the observable invariant so a future refactor cannot
    /// reintroduce either failure mode:
    ///   1. T7 font size is NEVER shrunk by the face safeguard (Matt Bold 110
    ///      is part of T7's visual identity and must stay exact).
    ///   2. T7 caption geometry is always placed below the face bottom with at
    ///      least the documented 155px clearance — for a high, mid, and low
    ///      face.
    #[test]
    fn test_t7_face_safety_invariant_holds_and_font_never_shrinks() {
        let raw_words = vec![
            t7_test_word("if", 1.0, 1.2),
            t7_test_word("somebody", 1.2, 1.5),
            t7_test_word("who's", 1.6, 1.8),
            t7_test_word("watching", 1.8, 2.1),
        ];
        let t7 = get_caption_template("preset_t7").expect("preset_t7 registered");
        const CANVAS_H: f64 = 1920.0;
        const CLEARANCE_PX: f64 = 155.0;

        // Sanity: the template itself is Matt Bold 110 (reference identity).
        assert_eq!(t7.global_styles.font_family, "Matt_Trial-Bold");
        assert_eq!(t7.global_styles.font_size, 110);

        for face_bottom_px in [600.0f64, 906.0, 1122.0, 1338.0, 1600.0] {
            let plan = crate::media::SmartFramingPlan {
                mode: "single".to_string(),
                x: "0".to_string(),
                y: "0".to_string(),
                w: "1080".to_string(),
                h: "1080".to_string(),
                segments: vec![crate::media::LayoutSegment::Single {
                    start: 0.0,
                    end: 5.0,
                    crop: crate::media::CropRectExpr {
                        x: "0".into(),
                        y: "0".into(),
                        w: "1080".into(),
                        h: "1080".into(),
                    },
                    face_bounds: Some(SubjectFaceBounds {
                        top: (face_bottom_px - 400.0) / CANVAS_H,
                        bottom: face_bottom_px / CANVAS_H,
                        left: 0.3,
                        right: 0.7,
                    }),
                }],
                face_bounds: None,
                framing: "adaptive".to_string(),
                is_emergency_fallback: false,
                fallback_reason: None,
                ..Default::default()
            };

            let ass =
                generate_ass_from_template_with_framing(&raw_words, 0.0, 5.0, &t7, Some(&plan));

            // (1) Font size must remain exactly 110 — the face safeguard must
            //     never shrink T7 (Style line carries Fontsize).
            assert!(
                ass.contains(",Matt_Trial-Bold,110,"),
                "T7 font size must stay 110 for face_bottom={face_bottom_px}, got:\n{ass}"
            );

            // (2) Every emitted \pos() must sit below the face with clearance.
            //
            //     The T7 solver emits line 1 at `y1`, line 2 at `y1 + 120`, and a
            //     centered solo line at `y1 + 60`. `y1` is what the solver clamps,
            //     so we recover it as the smallest emitted y and validate:
            //       - y1 clears the face by CLEARANCE_PX
            //       - y1 is canvas-bounded
            //       - the lowest emitted line stays on-canvas
            let mut ys: Vec<i64> = Vec::new();
            for line in ass.lines() {
                let Some(idx) = line.find("\\pos(540,") else {
                    continue;
                };
                let rest = &line[idx + "\\pos(540,".len()..];
                let end = rest.find(')').expect("malformed \\pos tag");
                let y: i64 = rest[..end]
                    .parse()
                    .unwrap_or_else(|_| panic!("unparseable \\pos y in: {line}"));
                ys.push(y);
            }
            assert!(
                !ys.is_empty(),
                "expected at least one positioned T7 caption for face_bottom={face_bottom_px}"
            );

            let y1 = *ys.iter().min().expect("non-empty ys");
            let lowest = *ys.iter().max().expect("non-empty ys");
            assert!(
                (y1 as f64) >= face_bottom_px + CLEARANCE_PX,
                "T7 line-1 y={y1} violates face clearance for face_bottom={face_bottom_px}"
            );
            assert!(
                y1 <= 1800,
                "T7 line-1 y={y1} must stay clamped to the canvas bound (<=1800)"
            );
            assert!(
                lowest <= 1920,
                "T7 lowest caption line y={lowest} must stay on the 1920px canvas"
            );
            // No line may sit above the face at all.
            for y in &ys {
                assert!(
                    (*y as f64) > face_bottom_px,
                    "T7 line at y={y} overlaps the face (bottom={face_bottom_px})"
                );
            }
        }
    }
}
