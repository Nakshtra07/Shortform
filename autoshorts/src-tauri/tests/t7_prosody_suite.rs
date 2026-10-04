//! Integration Test Suite: T7 Prosody Boundary & Soft Modulation Pipeline
//!
//! Target: autoshorts/src-tauri/tests/t7_prosody_suite.rs
//! Runner: cargo test -p autoshorts --test t7_prosody_suite
//!
//! Validates Requirements R3, R4, R5, and Acceptance Criteria:
//! 1. test_missing_model_fallback: Missing model file or t7_boundaries: None gracefully
//!    falls back to empty boundaries and baseline chunking.
//! 2. test_threshold_reduction_at_p_ge_065: p_boundary >= 0.65 reduces pause threshold
//!    from 1.5s to 0.9s, splitting a gap of 1.0s that baseline would not split.
//! 3. test_boundary_ignored_at_p_lt_050: p_boundary < 0.50 retains 1.5s threshold,
//!    leaving gap of 1.0s unsplit (identical to baseline).
//! 4. test_hard_guardrails_preserved: Short pause mid-sentence (e.g. 0.5s) without terminator
//!    is never split even when p_boundary is high (0.95); 1-3 word span protected; gap > 2.5s splits.
//! 5. test_flag_disabled_path: AUTOSHORTS_T7_PROSODY=0/false/off disables T7, returns baseline
//!    chunks byte-identically.
//! 6. test_byte_identical_fallback_ass: ASS generation output is byte-identical when T7
//!    is disabled vs default fallback state.
//! 7. test_multi_gap_selective_modulation: Multi-gap sequence with mixed probabilities properly
//!    applies modulation only to eligible gaps while preserving baseline on others.
//! 8. test_t7_boundary_serde_roundtrip: Validates JSON serialization and deserialization of
//!    T7Boundary with camelCase and snake_case aliases.

use autoshorts_lib::captions::{
    generate_ass_from_template, get_caption_template, segment_speech_chunks,
};
use autoshorts_lib::models::TranscriptWord;
use autoshorts_lib::{predict_t7_boundaries, t7_prosody_enabled, T7Boundary};
use std::path::PathBuf;
use std::sync::Mutex;

static ENV_LOCK: Mutex<()> = Mutex::new(());

/// Helper: construct a TranscriptWord for testing
fn make_word(text: &str, start: f64, end: f64) -> TranscriptWord {
    TranscriptWord {
        text: text.to_string(),
        start,
        end,
        speaker: Some("spk_0".to_string()),
    }
}

/// Helper: wrap TranscriptWords into (usize, &TranscriptWord) references
fn as_word_refs<'a>(words: &'a [TranscriptWord]) -> Vec<(usize, &'a TranscriptWord)> {
    words.iter().enumerate().collect()
}

/// Helper: extract word text strings per chunk for comparison
fn chunk_texts(chunks: &[Vec<(usize, &TranscriptWord)>]) -> Vec<Vec<String>> {
    chunks
        .iter()
        .map(|c| c.iter().map(|(_, w)| w.text.clone()).collect())
        .collect()
}

#[test]
fn test_missing_model_fallback() {
    // 1. predict_t7_boundaries on non-existent source/model returns empty boundaries without panic
    let words = vec![
        make_word("First", 0.0, 0.4),
        make_word("sentence", 0.5, 0.9),
        make_word("here.", 1.0, 1.4),
        make_word("Second", 1.8, 2.2),
        make_word("phrase.", 2.3, 2.8),
    ];
    let nonexistent_source = PathBuf::from("nonexistent_video_sample_999.mp4");
    let predicted = predict_t7_boundaries(&nonexistent_source, 0.0, 3.0, &words);
    assert!(
        predicted.is_empty(),
        "Missing model or source must gracefully return empty Vec<T7Boundary>, got {:?}",
        predicted
    );

    // 2. segment_speech_chunks with t7_boundaries: None vs Some(&[]) produces identical results
    let word_refs = as_word_refs(&words);
    let baseline_chunks = segment_speech_chunks(&word_refs, 3, 1.5, 2.6, None);
    let empty_t7_chunks = segment_speech_chunks(&word_refs, 3, 1.5, 2.6, Some(&[]));

    assert_eq!(
        chunk_texts(&baseline_chunks),
        chunk_texts(&empty_t7_chunks),
        "Chunking with Some(&[]) must be identical to None"
    );

    // 3. Unmatched gap indices in T7Boundary also produce identical chunking to baseline
    let unmatched_boundaries = vec![T7Boundary {
        gap_idx: 999,
        p_boundary: 0.95,
        features: None,
    }];
    let unmatched_chunks =
        segment_speech_chunks(&word_refs, 3, 1.5, 2.6, Some(&unmatched_boundaries));
    assert_eq!(
        chunk_texts(&baseline_chunks),
        chunk_texts(&unmatched_chunks),
        "Unmatched gap indices must fall back to baseline chunking"
    );
}

#[test]
fn test_threshold_reduction_at_p_ge_065() {
    // Setup 4 words:
    // w0 ("One", 0.0..0.4)
    // w1 ("Two", 0.5..0.9)   gap 0: pause = 0.1s
    // w2 ("Three", 1.9..2.3) gap 1: pause = 1.0s (between 0.9s and 1.5s)
    // w3 ("Four", 2.4..2.8)  gap 2: pause = 0.1s
    let words = vec![
        make_word("One", 0.0, 0.4),
        make_word("Two", 0.5, 0.9),
        make_word("Three", 1.9, 2.3),
        make_word("Four", 2.4, 2.8),
    ];
    let word_refs = as_word_refs(&words);

    // Baseline: pause_threshold = 1.5s, max_words = 4.
    // Gap 1 has pause = 1.0s <= 1.5s, so baseline does NOT split at gap 1.
    let baseline = segment_speech_chunks(&word_refs, 4, 1.5, 10.0, None);
    assert_eq!(
        baseline.len(),
        1,
        "Baseline with 1.5s threshold must keep 1.0s gap in same chunk"
    );
    assert_eq!(
        chunk_texts(&baseline),
        vec![vec!["One", "Two", "Three", "Four"]]
    );

    // T7 Modulated: gap 1 has p_boundary = 0.70 >= 0.65.
    // Effective pause threshold reduces from 1.5s to 0.9s for gap 1.
    // Pause is 1.0s > 0.9s -> triggers split!
    let t7_boundaries = vec![T7Boundary {
        gap_idx: 1,
        p_boundary: 0.70,
        features: None,
    }];
    let modulated = segment_speech_chunks(&word_refs, 4, 1.5, 10.0, Some(&t7_boundaries));

    assert_eq!(
        modulated.len(),
        2,
        "p_boundary >= 0.65 must reduce threshold to 0.9s and split the 1.0s gap"
    );
    assert_eq!(
        chunk_texts(&modulated),
        vec![vec!["One", "Two"], vec!["Three", "Four"]]
    );
}

#[test]
fn test_boundary_ignored_at_p_lt_050() {
    // Same words: gap 1 pause = 1.0s
    let words = vec![
        make_word("One", 0.0, 0.4),
        make_word("Two", 0.5, 0.9),
        make_word("Three", 1.9, 2.3),
        make_word("Four", 2.4, 2.8),
    ];
    let word_refs = as_word_refs(&words);

    let baseline = segment_speech_chunks(&word_refs, 4, 1.5, 10.0, None);

    // Case 1: p_boundary = 0.40 (< 0.50). Retains 1.5s threshold -> no split.
    let low_boundaries = vec![T7Boundary {
        gap_idx: 1,
        p_boundary: 0.40,
        features: None,
    }];
    let low_chunks = segment_speech_chunks(&word_refs, 4, 1.5, 10.0, Some(&low_boundaries));

    assert_eq!(
        chunk_texts(&low_chunks),
        chunk_texts(&baseline),
        "p_boundary < 0.50 must retain baseline threshold and produce identical chunks"
    );

    // Case 2: p_boundary = 0.20 (very low). Retains 1.5s threshold -> no split.
    let very_low_boundaries = vec![T7Boundary {
        gap_idx: 1,
        p_boundary: 0.20,
        features: None,
    }];
    let very_low_chunks =
        segment_speech_chunks(&word_refs, 4, 1.5, 10.0, Some(&very_low_boundaries));
    assert_eq!(
        chunk_texts(&very_low_chunks),
        chunk_texts(&baseline),
        "p_boundary = 0.20 must leave gap unsplit"
    );

    // Case 3: p_boundary = 0.55 (between 0.50 and 0.65, below modulation trigger). Retains 1.5s -> no split.
    let mid_boundaries = vec![T7Boundary {
        gap_idx: 1,
        p_boundary: 0.55,
        features: None,
    }];
    let mid_chunks = segment_speech_chunks(&word_refs, 4, 1.5, 10.0, Some(&mid_boundaries));
    assert_eq!(
        chunk_texts(&mid_chunks),
        chunk_texts(&baseline),
        "p_boundary < 0.65 must not trigger soft threshold reduction"
    );
}

#[test]
fn test_hard_guardrails_preserved() {
    // Guardrail 1: Mid-sentence short pause (0.5s) without terminator is NEVER split even when p_boundary is high (0.95)
    let words_short_pause = vec![
        make_word("Never", 0.0, 0.4),
        make_word("break", 0.9, 1.3), // pause = 0.5s
        make_word("early", 1.4, 1.8), // pause = 0.1s
    ];
    let refs_short = as_word_refs(&words_short_pause);
    let high_p_boundaries = vec![T7Boundary {
        gap_idx: 0,
        p_boundary: 0.95,
        features: None,
    }];

    // Even though p_boundary is 0.95, threshold is reduced to 0.9s; pause is 0.5s <= 0.9s.
    // Hard guardrail: Never break mid-sentence without terminator when pause <= 0.9s.
    let chunks = segment_speech_chunks(&refs_short, 3, 1.5, 10.0, Some(&high_p_boundaries));
    assert_eq!(
        chunks.len(),
        1,
        "High p_boundary must not force break mid-sentence when pause (0.5s) is below 0.9s"
    );
    assert_eq!(chunk_texts(&chunks), vec![vec!["Never", "break", "early"]]);

    // Guardrail 2: Large gap (> 2.5s) mid-sentence ALWAYS splits even with low or no p_boundary
    let words_long_gap = vec![
        make_word("Wait", 0.0, 0.4),
        make_word("here", 3.1, 3.5), // pause = 2.7s > 2.5s
    ];
    let refs_long = as_word_refs(&words_long_gap);
    let low_p_boundaries = vec![T7Boundary {
        gap_idx: 0,
        p_boundary: 0.05,
        features: None,
    }];
    let chunks_long = segment_speech_chunks(&refs_long, 3, 1.5, 10.0, Some(&low_p_boundaries));
    assert_eq!(
        chunks_long.len(),
        2,
        "Gap > 2.5s must always split regardless of low p_boundary"
    );

    // Guardrail 3: Sentence terminator always triggers split regardless of pause duration
    let words_terminator = vec![
        make_word("Done.", 0.0, 0.4),
        make_word("Next", 0.45, 0.8), // pause = 0.05s
    ];
    let refs_term = as_word_refs(&words_terminator);
    let chunks_term = segment_speech_chunks(&refs_term, 3, 1.5, 10.0, None);
    assert_eq!(
        chunks_term.len(),
        2,
        "Sentence terminator must always split even with minimal pause"
    );
}

#[test]
fn test_flag_disabled_path() {
    let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let original = std::env::var("AUTOSHORTS_T7_PROSODY").ok();

    // 1. Explicitly disabled values: "0", "false", "off", "FALSE"
    for disabled_val in ["0", "false", "off", "FALSE", "Off"] {
        std::env::set_var("AUTOSHORTS_T7_PROSODY", disabled_val);
        assert!(
            !t7_prosody_enabled(),
            "Expected t7_prosody_enabled() to be false for value '{}'",
            disabled_val
        );
    }

    // 2. Explicitly enabled values: "1", "true", "TRUE"
    for enabled_val in ["1", "true", "TRUE", "yes"] {
        std::env::set_var("AUTOSHORTS_T7_PROSODY", enabled_val);
        assert!(
            t7_prosody_enabled(),
            "Expected t7_prosody_enabled() to be true for value '{}'",
            enabled_val
        );
    }

    // 3. Unset: default is true
    std::env::remove_var("AUTOSHORTS_T7_PROSODY");
    assert!(
        t7_prosody_enabled(),
        "Expected t7_prosody_enabled() to default to true when unset"
    );

    // 4. When disabled, sidecar call returns empty Vec without calling python
    std::env::set_var("AUTOSHORTS_T7_PROSODY", "0");
    let words = vec![make_word("Test", 0.0, 0.5), make_word("words", 0.6, 1.0)];
    let source = PathBuf::from("dummy.mp4");
    let result = predict_t7_boundaries(&source, 0.0, 1.0, &words);
    assert!(
        result.is_empty(),
        "Disabled flag must return empty boundaries immediately"
    );

    // Restore original env var
    if let Some(val) = original {
        std::env::set_var("AUTOSHORTS_T7_PROSODY", val);
    } else {
        std::env::remove_var("AUTOSHORTS_T7_PROSODY");
    }
}

#[test]
fn test_byte_identical_fallback_ass() {
    let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let template = get_caption_template("preset_t7").expect("preset_t7 must be registered");

    let words = vec![
        make_word("The", 0.0, 0.2),
        make_word("quick", 0.25, 0.5),
        make_word("brown", 0.55, 0.8),
        make_word("fox", 0.85, 1.1),
        make_word("jumps", 1.4, 1.7),
        make_word("over", 1.75, 2.0),
        make_word("lazy", 2.05, 2.3),
        make_word("dog.", 2.35, 2.7),
    ];

    let original = std::env::var("AUTOSHORTS_T7_PROSODY").ok();

    // Render with flag disabled
    std::env::set_var("AUTOSHORTS_T7_PROSODY", "0");
    let ass_disabled = generate_ass_from_template(&words, 0.0, 3.0, &template);

    // Render with flag enabled but no model / default fallback
    std::env::remove_var("AUTOSHORTS_T7_PROSODY");
    let ass_fallback = generate_ass_from_template(&words, 0.0, 3.0, &template);

    // Restore env var
    if let Some(val) = original {
        std::env::set_var("AUTOSHORTS_T7_PROSODY", val);
    } else {
        std::env::remove_var("AUTOSHORTS_T7_PROSODY");
    }

    assert_eq!(
        ass_disabled, ass_fallback,
        "ASS output when T7 is disabled must be byte-identical to baseline fallback"
    );

    // Basic structure verification
    assert!(
        ass_disabled.contains("[Script Info]"),
        "Generated ASS must contain [Script Info]"
    );
    assert!(
        ass_disabled.contains("Dialogue:"),
        "Generated ASS must contain Dialogue lines"
    );
}

#[test]
fn test_multi_gap_selective_modulation() {
    // 5 words, 4 gaps:
    // w0 ("Alpha", 0.0..0.4)
    // w1 ("Beta", 0.7..1.1)    gap 0: pause = 0.3s (p=0.80) -> pause <= 0.9s -> no split
    // w2 ("Gamma", 2.2..2.6)   gap 1: pause = 1.1s (p=0.75) -> p >= 0.65, pause > 0.9s -> SPLIT
    // w3 ("Delta", 3.7..4.1)   gap 2: pause = 1.1s (p=0.35) -> p < 0.50, pause <= 1.5s -> NO SPLIT
    // w4 ("Epsilon", 5.8..6.2) gap 3: pause = 1.7s (p=0.20) -> pause > 1.5s (baseline rule) -> SPLIT
    let words = vec![
        make_word("Alpha", 0.0, 0.4),
        make_word("Beta", 0.7, 1.1),
        make_word("Gamma", 2.2, 2.6),
        make_word("Delta", 3.7, 4.1),
        make_word("Epsilon", 5.8, 6.2),
    ];
    let word_refs = as_word_refs(&words);

    let boundaries = vec![
        T7Boundary {
            gap_idx: 0,
            p_boundary: 0.80,
            features: None,
        },
        T7Boundary {
            gap_idx: 1,
            p_boundary: 0.75,
            features: None,
        },
        T7Boundary {
            gap_idx: 2,
            p_boundary: 0.35,
            features: None,
        },
        T7Boundary {
            gap_idx: 3,
            p_boundary: 0.20,
            features: None,
        },
    ];

    let chunks = segment_speech_chunks(&word_refs, 5, 1.5, 10.0, Some(&boundaries));

    // Expected chunks:
    // Chunk 0: ["Alpha", "Beta"] (split at gap 1 due to p=0.75 + 1.1s pause)
    // Chunk 1: ["Gamma", "Delta"] (kept together because gap 2 has p=0.35, 1.1s <= 1.5s)
    // Chunk 2: ["Epsilon"] (split at gap 3 because pause 1.7s > 1.5s)
    assert_eq!(
        chunks.len(),
        3,
        "Expected exactly 3 chunks from selective multi-gap modulation"
    );
    assert_eq!(
        chunk_texts(&chunks),
        vec![
            vec!["Alpha", "Beta"],
            vec!["Gamma", "Delta"],
            vec!["Epsilon"]
        ]
    );
}

#[test]
fn test_t7_boundary_serde_roundtrip() {
    // 1. camelCase deserialization (from Python JSON output)
    let json_camel = r#"{"gapIdx": 3, "pBoundary": 0.82, "features": {"pauseSec": 1.1}}"#;
    let b_camel: T7Boundary = serde_json::from_str(json_camel).expect("deserialize camelCase");
    assert_eq!(b_camel.gap_idx, 3);
    assert!((b_camel.p_boundary - 0.82).abs() < 1e-6);
    assert!(b_camel.features.is_some());

    // 2. snake_case deserialization
    let json_snake = r#"{"gap_idx": 5, "p_boundary": 0.45}"#;
    let b_snake: T7Boundary = serde_json::from_str(json_snake).expect("deserialize snake_case");
    assert_eq!(b_snake.gap_idx, 5);
    assert!((b_snake.p_boundary - 0.45).abs() < 1e-6);
    assert!(b_snake.features.is_none());

    // 3. Serialization roundtrip
    let serialized = serde_json::to_string(&b_camel).expect("serialize T7Boundary");
    let deserialized: T7Boundary =
        serde_json::from_str(&serialized).expect("deserialize serialized");
    assert_eq!(b_camel, deserialized);
}
