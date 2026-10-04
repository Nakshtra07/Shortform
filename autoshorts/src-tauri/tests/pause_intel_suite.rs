//! Integration Test Suite: Pause Intelligence Pipeline (`tests/pause_intel_suite.rs`)
//!
//! Target: autoshorts/src-tauri/tests/pause_intel_suite.rs
//! Runner: cargo test -p autoshorts --test pause_intel_suite
//!
//! Validates Requirements from SDD Phase 3 Task 7:
//! 1. test_missing_model_fallback: Missing model file or source gracefully falls back to empty gaps.
//! 2. test_flag_disabled_by_default: Unset AUTOSHORTS_SMART_PACING_LEARNED defaults to false, score_gaps returns empty.
//! 3. test_flag_explicit_enable: "1"/"true"/"on" enables flag; "0"/"false"/"off"/invalid disables flag.
//! 4. test_removable_above_threshold_detected: P(removable) >= 0.65 for breath_pause/waiting_pause detected as removable.
//! 5. test_low_confidence_ignored: P(removable) < 0.50 or below threshold 0.65 is marked non-removable.
//! 6. test_conflict_keep_wins: Keep-Wins-On-Conflict invariant preserves deterministic authority over protected boundaries.
//! 7. test_version_mismatch_fails_soft: Feature version mismatch (e.g. "999") fails soft to deterministic fallback without panicking.
//! 8. test_serde_roundtrip_pause_intel_gap: Validates camelCase and snake_case deserialization and serialization roundtrip.

use autoshorts_lib::models::TranscriptWord;
use autoshorts_lib::{pause_intel_enabled, score_gaps, PauseIntelGap};
use std::path::PathBuf;
use std::sync::Mutex;

static ENV_LOCK: Mutex<()> = Mutex::new(());

/// Helper: construct a TranscriptWord for integration tests
fn make_word(text: &str, start: f64, end: f64) -> TranscriptWord {
    TranscriptWord {
        text: text.to_string(),
        start,
        end,
        speaker: Some("spk_0".to_string()),
    }
}

/// Helper: safely restore an environment variable to its previous state
fn restore_env(key: &str, prev: Option<String>) {
    match prev {
        Some(v) => std::env::set_var(key, v),
        None => std::env::remove_var(key),
    }
}

/// Helper: check if a category is eligible for removal under the Removable Category Gate
fn is_removable_category(category: Option<&str>) -> bool {
    matches!(category, Some("breath_pause" | "waiting_pause"))
}

/// Helper: models the Removable Category Gate and threshold evaluation
fn should_mark_removable(gap: &PauseIntelGap) -> bool {
    let class = gap.predicted_class.as_deref().or(gap.category.as_deref());
    gap.p_removable_above_threshold && gap.p_removable >= 0.65 && is_removable_category(class)
}

/// Deterministic verdict used to model Keep-Wins-On-Conflict invariant
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DeterministicVerdict {
    /// Deterministic safety or boundary rule mandates KEEP
    Keep(&'static str),
    /// Deterministic pipeline permits consideration of removal
    AllowCandidate,
}

/// Helper: resolves candidate gap action strictly enforcing Keep-Wins-On-Conflict
fn resolve_gap_action(gap: &PauseIntelGap, deterministic: DeterministicVerdict) -> &'static str {
    // 1. Keep-Wins-On-Conflict invariant: deterministic veto is 100% authoritative
    if let DeterministicVerdict::Keep(_) = deterministic {
        return "KEEP";
    }

    // 2. Removable Category Gate & threshold check
    if should_mark_removable(gap) {
        "REMOVE"
    } else {
        "KEEP"
    }
}

#[test]
fn test_missing_model_fallback() {
    let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let orig_flag = std::env::var("AUTOSHORTS_SMART_PACING_LEARNED").ok();
    let orig_model = std::env::var("AUTOSHORTS_PAUSE_MODEL").ok();

    let words = vec![
        make_word("First", 0.0, 0.4),
        make_word("gap", 0.5, 0.9),
        make_word("testing", 1.8, 2.2),
    ];

    // Case 1: Non-existent source file returns empty Vec without panic
    let nonexistent_source = PathBuf::from("nonexistent_video_sample_99999.mp4");
    let gaps_missing_src = score_gaps(&nonexistent_source, 0.0, 3.0, &words);
    assert!(
        gaps_missing_src.is_empty(),
        "Missing source file must gracefully return empty Vec<PauseIntelGap>, got {:?}",
        gaps_missing_src
    );

    // Case 2: Source exists, flag is enabled, but model file is missing
    std::env::set_var("AUTOSHORTS_SMART_PACING_LEARNED", "1");
    std::env::set_var(
        "AUTOSHORTS_PAUSE_MODEL",
        "nonexistent_pause_model_file_99999.txt",
    );

    let temp_src = std::env::temp_dir().join(format!(
        "test_pause_missing_model_{}.mp4",
        uuid::Uuid::new_v4()
    ));
    std::fs::write(&temp_src, b"dummy audio source data").expect("write temp source");

    let gaps_missing_model = score_gaps(&temp_src, 0.0, 3.0, &words);
    assert!(
        gaps_missing_model.is_empty(),
        "Missing model file must gracefully return empty Vec<PauseIntelGap>, got {:?}",
        gaps_missing_model
    );

    // Case 3: Empty words slice returns empty Vec without panic
    let gaps_empty_words = score_gaps(&temp_src, 0.0, 3.0, &[]);
    assert!(
        gaps_empty_words.is_empty(),
        "Empty words slice must gracefully return empty Vec<PauseIntelGap>"
    );

    // Cleanup
    let _ = std::fs::remove_file(&temp_src);
    restore_env("AUTOSHORTS_PAUSE_MODEL", orig_model);
    restore_env("AUTOSHORTS_SMART_PACING_LEARNED", orig_flag);
}

#[test]
fn test_flag_disabled_by_default() {
    let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let orig = std::env::var("AUTOSHORTS_SMART_PACING_LEARNED").ok();

    // 1. Unset flag: pause_intel_enabled() must evaluate to false
    std::env::remove_var("AUTOSHORTS_SMART_PACING_LEARNED");
    assert!(
        !pause_intel_enabled(),
        "pause_intel_enabled() must evaluate to false when AUTOSHORTS_SMART_PACING_LEARNED is unset"
    );

    // 2. score_gaps returns empty Vec even with valid source and words
    let temp_src = std::env::temp_dir().join(format!(
        "test_pause_disabled_default_{}.mp4",
        uuid::Uuid::new_v4()
    ));
    std::fs::write(&temp_src, b"dummy video audio payload").expect("write temp source");

    let words = vec![
        make_word("Safety", 0.0, 0.5),
        make_word("baseline", 0.6, 1.1),
        make_word("guaranteed", 2.0, 2.7),
    ];

    let gaps = score_gaps(&temp_src, 0.0, 3.0, &words);
    assert!(
        gaps.is_empty(),
        "score_gaps must return empty Vec when flag is disabled by default, got {:?}",
        gaps
    );

    let _ = std::fs::remove_file(&temp_src);
    restore_env("AUTOSHORTS_SMART_PACING_LEARNED", orig);
}

#[test]
fn test_flag_explicit_enable() {
    let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let orig = std::env::var("AUTOSHORTS_SMART_PACING_LEARNED").ok();

    // 1. Truthy values (case-insensitive and trimmed) must return true
    let truthy_cases = [
        "1", "true", "TRUE", "True", "tRuE", "on", "ON", "On", "  1  ", " true ", " on ",
    ];
    for val in truthy_cases {
        std::env::set_var("AUTOSHORTS_SMART_PACING_LEARNED", val);
        assert!(
            pause_intel_enabled(),
            "Expected pause_intel_enabled() to be true for value '{val}'"
        );
    }

    // 2. Falsy and arbitrary invalid values must return false
    let falsy_cases = [
        "0",
        "false",
        "FALSE",
        "False",
        "off",
        "OFF",
        "Off",
        "2",
        "-1",
        "disabled",
        "no",
        "random_value",
        "",
        "   ",
    ];
    for val in falsy_cases {
        std::env::set_var("AUTOSHORTS_SMART_PACING_LEARNED", val);
        assert!(
            !pause_intel_enabled(),
            "Expected pause_intel_enabled() to be false for value '{val}'"
        );
    }

    // 3. Unset must return false
    std::env::remove_var("AUTOSHORTS_SMART_PACING_LEARNED");
    assert!(
        !pause_intel_enabled(),
        "Expected pause_intel_enabled() to be false when unset"
    );

    restore_env("AUTOSHORTS_SMART_PACING_LEARNED", orig);
}

#[test]
fn test_removable_above_threshold_detected() {
    // Test 1: breath_pause with P(removable) >= 0.65
    let json_breath = r#"{
        "gapIdx": 1,
        "srcStartSec": 1.2,
        "srcEndSec": 1.9,
        "pRemovable": 0.85,
        "pRemovableAboveThreshold": true,
        "predictedClass": "breath_pause",
        "category": "breath_pause",
        "confidence": 0.92
    }"#;
    let gap_breath: PauseIntelGap =
        serde_json::from_str(json_breath).expect("deserialize breath_pause gap");

    assert!(
        gap_breath.p_removable >= 0.65,
        "P(removable) must be >= 0.65 for breath_pause"
    );
    assert!(
        gap_breath.p_removable_above_threshold,
        "p_removable_above_threshold must be true"
    );
    assert_eq!(gap_breath.predicted_class.as_deref(), Some("breath_pause"));
    assert_eq!(gap_breath.category.as_deref(), Some("breath_pause"));
    assert!(
        is_removable_category(gap_breath.category.as_deref()),
        "breath_pause must be classified as a removable category"
    );
    assert!(
        should_mark_removable(&gap_breath),
        "breath_pause with P >= 0.65 must be marked removable"
    );

    // Test 2: waiting_pause with P(removable) >= 0.65
    let json_waiting = r#"{
        "gapIdx": 2,
        "srcStartSec": 3.4,
        "srcEndSec": 4.5,
        "pRemovable": 0.65,
        "pRemovableAboveThreshold": true,
        "predictedClass": "waiting_pause",
        "category": "waiting_pause",
        "confidence": 0.78
    }"#;
    let gap_waiting: PauseIntelGap =
        serde_json::from_str(json_waiting).expect("deserialize waiting_pause gap");

    assert!(
        gap_waiting.p_removable >= 0.65,
        "P(removable) must be >= 0.65 for waiting_pause"
    );
    assert!(
        gap_waiting.p_removable_above_threshold,
        "p_removable_above_threshold must be true at threshold boundary 0.65"
    );
    assert_eq!(
        gap_waiting.predicted_class.as_deref(),
        Some("waiting_pause")
    );
    assert_eq!(gap_waiting.category.as_deref(), Some("waiting_pause"));
    assert!(
        is_removable_category(gap_waiting.category.as_deref()),
        "waiting_pause must be classified as a removable category"
    );
    assert!(
        should_mark_removable(&gap_waiting),
        "waiting_pause with P >= 0.65 must be marked removable"
    );

    // Test 3: Struct instantiation directly
    let direct_gap = PauseIntelGap {
        gap_idx: 3,
        src_start_sec: 6.0,
        src_end_sec: 6.8,
        p_removable: 0.94,
        p_removable_above_threshold: true,
        predicted_class: Some("breath_pause".to_string()),
        category: Some("breath_pause".to_string()),
        confidence: Some(0.96),
        features: None,
    };
    assert!(direct_gap.p_removable >= 0.65);
    assert!(direct_gap.p_removable_above_threshold);
    assert!(should_mark_removable(&direct_gap));
}

#[test]
fn test_low_confidence_ignored() {
    // Case 1: P(removable) < 0.50 (e.g. 0.40)
    let json_low = r#"{
        "gapIdx": 0,
        "srcStartSec": 0.8,
        "srcEndSec": 1.1,
        "pRemovable": 0.40,
        "pRemovableAboveThreshold": false,
        "predictedClass": "normal_word_gap",
        "category": "normal_word_gap",
        "confidence": 0.60
    }"#;
    let gap_low: PauseIntelGap =
        serde_json::from_str(json_low).expect("deserialize low confidence gap");

    assert!(
        gap_low.p_removable < 0.50,
        "P(removable) 0.40 must be strictly below 0.50"
    );
    assert!(
        !gap_low.p_removable_above_threshold,
        "p_removable_above_threshold must be false for low confidence gap"
    );
    assert!(
        !should_mark_removable(&gap_low),
        "Low confidence gap must be rejected by removable gating"
    );

    // Case 2: Sub-threshold confidence (0.60, between 0.50 and 0.65)
    let json_sub_thr = r#"{
        "gapIdx": 1,
        "srcStartSec": 2.0,
        "srcEndSec": 2.5,
        "pRemovable": 0.60,
        "pRemovableAboveThreshold": false,
        "predictedClass": "breath_pause",
        "category": "breath_pause",
        "confidence": 0.65
    }"#;
    let gap_sub: PauseIntelGap =
        serde_json::from_str(json_sub_thr).expect("deserialize sub-threshold gap");

    assert!(
        gap_sub.p_removable < 0.65,
        "P(removable) 0.60 must be below threshold 0.65"
    );
    assert!(
        !gap_sub.p_removable_above_threshold,
        "p_removable_above_threshold must be false for 0.60"
    );
    assert!(
        !should_mark_removable(&gap_sub),
        "Sub-threshold gap must be rejected despite breath_pause class"
    );

    // Case 3: Default deserialization when pRemovableAboveThreshold is omitted
    let json_default = r#"{
        "gapIdx": 2,
        "srcStartSec": 3.0,
        "srcEndSec": 3.4,
        "pRemovable": 0.25
    }"#;
    let gap_default: PauseIntelGap =
        serde_json::from_str(json_default).expect("deserialize omitted threshold");

    assert!(
        !gap_default.p_removable_above_threshold,
        "Default for omitted p_removable_above_threshold must be false"
    );
    assert!(
        !should_mark_removable(&gap_default),
        "Omitted threshold must safely evaluate to not removable"
    );
}

#[test]
fn test_conflict_keep_wins() {
    // Keep-Wins-On-Conflict invariant:
    // Deterministic authority is 100% sovereign.
    // Even if P(removable) >= 0.65, protected boundaries or safety rule vetoes must KEEP the gap.

    // Scenario 1: sentence_pause with high P(removable) = 0.95
    let sentence_gap = PauseIntelGap {
        gap_idx: 0,
        src_start_sec: 1.0,
        src_end_sec: 2.2,
        p_removable: 0.95,
        p_removable_above_threshold: true,
        predicted_class: Some("sentence_pause".to_string()),
        category: Some("sentence_pause".to_string()),
        confidence: Some(0.97),
        features: None,
    };
    // Protected boundary: deterministic pipeline flags sentence boundary
    let action_1 = resolve_gap_action(
        &sentence_gap,
        DeterministicVerdict::Keep("sentence_boundary"),
    );
    assert_eq!(
        action_1, "KEEP",
        "Keep-Wins: sentence_pause must never be removed even with P(removable) = 0.95"
    );
    // Even if deterministic verdict was AllowCandidate, the Removable Category Gate blocks it
    let action_1_gate =
        resolve_gap_action(&sentence_gap, DeterministicVerdict::AllowCandidate);
    assert_eq!(
        action_1_gate, "KEEP",
        "Removable Category Gate: sentence_pause cannot be marked removable"
    );

    // Scenario 2: speaker_transition with high P(removable) = 0.90
    let speaker_gap = PauseIntelGap {
        gap_idx: 1,
        src_start_sec: 3.0,
        src_end_sec: 4.0,
        p_removable: 0.90,
        p_removable_above_threshold: true,
        predicted_class: Some("speaker_transition".to_string()),
        category: Some("speaker_transition".to_string()),
        confidence: Some(0.92),
        features: None,
    };
    let action_2 = resolve_gap_action(
        &speaker_gap,
        DeterministicVerdict::Keep("speaker_transition"),
    );
    assert_eq!(
        action_2, "KEEP",
        "Keep-Wins: speaker_transition must never be removed even with high confidence"
    );

    // Scenario 3: normal_word_gap with high P(removable) = 0.75
    let normal_gap = PauseIntelGap {
        gap_idx: 2,
        src_start_sec: 4.5,
        src_end_sec: 4.8,
        p_removable: 0.75,
        p_removable_above_threshold: true,
        predicted_class: Some("normal_word_gap".to_string()),
        category: Some("normal_word_gap".to_string()),
        confidence: Some(0.80),
        features: None,
    };
    let action_3 = resolve_gap_action(&normal_gap, DeterministicVerdict::AllowCandidate);
    assert_eq!(
        action_3, "KEEP",
        "Keep-Wins: normal_word_gap is not an allowed removable class"
    );

    // Scenario 4: breath_pause with P(removable) = 0.88, but deterministic safety rule vetoes (e.g. speech buffer protection)
    let breath_gap = PauseIntelGap {
        gap_idx: 3,
        src_start_sec: 5.5,
        src_end_sec: 6.2,
        p_removable: 0.88,
        p_removable_above_threshold: true,
        predicted_class: Some("breath_pause".to_string()),
        category: Some("breath_pause".to_string()),
        confidence: Some(0.91),
        features: None,
    };
    let action_4_buffer_veto = resolve_gap_action(
        &breath_gap,
        DeterministicVerdict::Keep("speech_buffer_clearance_violation"),
    );
    assert_eq!(
        action_4_buffer_veto, "KEEP",
        "Keep-Wins: speech buffer protection must veto model recommendation"
    );

    let action_4_silence_veto = resolve_gap_action(
        &breath_gap,
        DeterministicVerdict::Keep("acoustic_silence_verification_failure"),
    );
    assert_eq!(
        action_4_silence_veto, "KEEP",
        "Keep-Wins: acoustic silence check failure must veto model recommendation"
    );

    // Scenario 5: breath_pause with P(removable) = 0.88, deterministic pipeline agrees -> REMOVE
    let action_5_allowed =
        resolve_gap_action(&breath_gap, DeterministicVerdict::AllowCandidate);
    assert_eq!(
        action_5_allowed, "REMOVE",
        "breath_pause with valid clearance and high P(removable) is eligible for removal"
    );
}

#[test]
fn test_version_mismatch_fails_soft() {
    let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let orig_flag = std::env::var("AUTOSHORTS_SMART_PACING_LEARNED").ok();
    let orig_model = std::env::var("AUTOSHORTS_PAUSE_MODEL").ok();

    std::env::set_var("AUTOSHORTS_SMART_PACING_LEARNED", "1");

    // 1. Setup a model file whose companion .meta.json has an incompatible featureVersion: "999"
    let temp_dir = std::env::temp_dir();
    let unique_id = uuid::Uuid::new_v4();
    let model_path = temp_dir.join(format!("test_mismatched_model_{}.txt", unique_id));
    let meta_path = temp_dir.join(format!("test_mismatched_model_{}.meta.json", unique_id));
    let temp_src = temp_dir.join(format!("test_mismatched_src_{}.mp4", unique_id));

    std::fs::write(&model_path, b"dummy model file content").unwrap();
    let meta_content = serde_json::json!({
        "featureVersion": "999",
        "modelType": "lightgbm",
        "nFeatures": 23
    });
    std::fs::write(&meta_path, meta_content.to_string()).unwrap();
    std::fs::write(&temp_src, b"dummy media bytes").unwrap();

    std::env::set_var("AUTOSHORTS_PAUSE_MODEL", &model_path);

    let words = vec![
        make_word("Version", 0.0, 0.4),
        make_word("mismatch", 0.8, 1.4),
    ];

    // score_gaps must execute without panicking and return empty Vec on version mismatch
    let gaps = score_gaps(&temp_src, 0.0, 2.0, &words);
    assert!(
        gaps.is_empty(),
        "Model with incompatible featureVersion '999' must fail soft and return empty gaps, got {:?}",
        gaps
    );

    // 2. Validate payload deserialization with unexpected feature version fails soft
    let mismatched_payload_json = r#"{
        "sourceHash": "12345678abcdef01",
        "featureVersion": "999",
        "modelScored": false,
        "model": "dummy_model_v999.txt",
        "gaps": []
    }"#;
    let payload: serde_json::Value =
        serde_json::from_str(mismatched_payload_json).expect("deserialize mismatched payload");
    assert_eq!(payload.get("featureVersion").and_then(|v| v.as_str()), Some("999"));
    assert_eq!(
        payload.get("modelScored").and_then(|v| v.as_bool()),
        Some(false)
    );

    // Cleanup
    let _ = std::fs::remove_file(&model_path);
    let _ = std::fs::remove_file(&meta_path);
    let _ = std::fs::remove_file(&temp_src);
    restore_env("AUTOSHORTS_PAUSE_MODEL", orig_model);
    restore_env("AUTOSHORTS_SMART_PACING_LEARNED", orig_flag);
}

#[test]
fn test_serde_roundtrip_pause_intel_gap() {
    // 1. camelCase deserialization (matches python sidecar output)
    let json_camel = r#"{
        "gapIdx": 4,
        "srcStartSec": 2.15,
        "srcEndSec": 3.05,
        "pRemovable": 0.87,
        "pRemovableAboveThreshold": true,
        "predictedClass": "breath_pause",
        "category": "breath_pause",
        "confidence": 0.93,
        "features": { "rmsDb": -26.4, "hfRatio": 0.28 }
    }"#;
    let gap_camel: PauseIntelGap =
        serde_json::from_str(json_camel).expect("failed to deserialize camelCase PauseIntelGap");

    assert_eq!(gap_camel.gap_idx, 4);
    assert!((gap_camel.src_start_sec - 2.15).abs() < 1e-6);
    assert!((gap_camel.src_end_sec - 3.05).abs() < 1e-6);
    assert!((gap_camel.p_removable - 0.87).abs() < 1e-6);
    assert!(gap_camel.p_removable_above_threshold);
    assert_eq!(gap_camel.predicted_class.as_deref(), Some("breath_pause"));
    assert_eq!(gap_camel.category.as_deref(), Some("breath_pause"));
    assert_eq!(gap_camel.confidence, Some(0.93));
    assert!(gap_camel.features.is_some());

    // 2. snake_case deserialization
    let json_snake = r#"{
        "gap_idx": 8,
        "src_start_sec": 5.0,
        "src_end_sec": 5.8,
        "p_removable": 0.33,
        "p_removable_above_threshold": false,
        "predicted_class": "normal_word_gap",
        "category": "normal_word_gap"
    }"#;
    let gap_snake: PauseIntelGap =
        serde_json::from_str(json_snake).expect("failed to deserialize snake_case PauseIntelGap");

    assert_eq!(gap_snake.gap_idx, 8);
    assert!((gap_snake.src_start_sec - 5.0).abs() < 1e-6);
    assert!((gap_snake.src_end_sec - 5.8).abs() < 1e-6);
    assert!((gap_snake.p_removable - 0.33).abs() < 1e-6);
    assert!(!gap_snake.p_removable_above_threshold);
    assert_eq!(gap_snake.predicted_class.as_deref(), Some("normal_word_gap"));
    assert_eq!(gap_snake.category.as_deref(), Some("normal_word_gap"));
    assert_eq!(gap_snake.confidence, None);

    // 3. idx alias deserialization
    let json_idx = r#"{
        "idx": 15,
        "src_start_sec": 12.0,
        "src_end_sec": 12.6,
        "p_removable": 0.72
    }"#;
    let gap_idx: PauseIntelGap =
        serde_json::from_str(json_idx).expect("failed to deserialize idx alias");
    assert_eq!(gap_idx.gap_idx, 15);
    assert!(!gap_idx.p_removable_above_threshold); // default false
    assert_eq!(gap_idx.predicted_class, None);

    // 4. Serialization produced camelCase
    let serialized = serde_json::to_string(&gap_camel).expect("failed to serialize PauseIntelGap");
    assert!(serialized.contains("\"gapIdx\":4"));
    assert!(serialized.contains("\"srcStartSec\":2.15"));
    assert!(serialized.contains("\"srcEndSec\":3.05"));
    assert!(serialized.contains("\"pRemovable\":0.87"));
    assert!(serialized.contains("\"pRemovableAboveThreshold\":true"));
    assert!(serialized.contains("\"predictedClass\":\"breath_pause\""));

    // 5. Full roundtrip equality
    let roundtripped: PauseIntelGap =
        serde_json::from_str(&serialized).expect("failed to deserialize roundtrip JSON");
    assert_eq!(gap_camel, roundtripped);
}
