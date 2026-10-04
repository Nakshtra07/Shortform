//! AutoShorts 11.0 Phase 3 — Pause Intelligence sidecar wrapper
//!
//! Wraps `scripts/pause_intelligence.py` as a sidecar, bounded via `proc_guard::run_bounded` (60s timeout).
//! Caches results per (source_hash, feature_version, start_sec, end_sec) under
//! `%APPDATA%/com.autoshorts.desktop/pause_intel_cache/`. Missing or corrupt model gracefully
//! degrades to empty gaps.

use crate::models::TranscriptWord;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Duration;

pub const FEATURE_VERSION: &str = "1";
pub const SIDECAR_TIMEOUT_SECS: u64 = 60;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct PauseIntelGap {
    #[serde(alias = "gap_idx", alias = "idx")]
    pub gap_idx: usize,
    #[serde(alias = "src_start_sec")]
    pub src_start_sec: f64,
    #[serde(alias = "src_end_sec")]
    pub src_end_sec: f64,
    #[serde(alias = "p_removable", default)]
    pub p_removable: f64,
    #[serde(alias = "p_removable_above_threshold", default)]
    pub p_removable_above_threshold: bool,
    #[serde(alias = "predicted_class", default)]
    pub predicted_class: Option<String>,
    #[serde(default)]
    pub category: Option<String>,
    #[serde(default)]
    pub confidence: Option<f64>,
    #[serde(default)]
    pub features: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PausePredictPayload {
    #[serde(default)]
    pub source_hash: Option<String>,
    #[serde(default)]
    pub feature_version: Option<String>,
    #[serde(default)]
    pub model_scored: bool,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub gaps: Vec<PauseIntelGap>,
}

/// Returns the cache directory for Pause Intelligence sidecar results.
pub fn pause_intel_cache_dir() -> PathBuf {
    dirs::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("com.autoshorts.desktop")
        .join("pause_intel_cache")
}

/// Locates the `pause_intelligence.py` sidecar script.
pub fn find_sidecar_script() -> Option<PathBuf> {
    let script_name = "pause_intelligence.py";
    let mut candidates = Vec::new();

    if let Ok(cargo_manifest) = std::env::var("CARGO_MANIFEST_DIR") {
        candidates.push(PathBuf::from(cargo_manifest).join("scripts").join(script_name));
    }
    if let Ok(cwd) = std::env::current_dir() {
        candidates.push(cwd.join("scripts").join(script_name));
        candidates.push(
            cwd.join("autoshorts")
                .join("src-tauri")
                .join("scripts")
                .join(script_name),
        );
        if let Some(parent) = cwd.parent() {
            candidates.push(
                parent
                    .join("autoshorts")
                    .join("src-tauri")
                    .join("scripts")
                    .join(script_name),
            );
            candidates.push(parent.join("scripts").join(script_name));
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            candidates.push(parent.join("scripts").join(script_name));
            if let Some(grandparent) = parent.parent() {
                candidates.push(grandparent.join("scripts").join(script_name));
            }
        }
    }

    candidates.into_iter().find(|p| p.exists())
}

/// Locates the `pause_classifier_v1.txt` model file if trained.
pub fn find_model_file() -> Option<PathBuf> {
    let model_name = "pause_classifier_v1.txt";
    let mut candidates = Vec::new();

    if let Ok(env_path) = std::env::var("AUTOSHORTS_PAUSE_MODEL") {
        let p = PathBuf::from(env_path);
        if p.exists() {
            return Some(p);
        }
    }

    if let Ok(cargo_manifest) = std::env::var("CARGO_MANIFEST_DIR") {
        let manifest_dir = PathBuf::from(cargo_manifest);
        candidates.push(
            manifest_dir
                .join("scripts")
                .join("models")
                .join(model_name),
        );
        candidates.push(
            manifest_dir
                .join("models")
                .join(model_name),
        );
    }
    if let Ok(cwd) = std::env::current_dir() {
        candidates.push(cwd.join("scripts").join("models").join(model_name));
        candidates.push(cwd.join("models").join(model_name));
        candidates.push(
            cwd.join("autoshorts")
                .join("src-tauri")
                .join("scripts")
                .join("models")
                .join(model_name),
        );
        if let Some(parent) = cwd.parent() {
            candidates.push(
                parent
                    .join("autoshorts")
                    .join("src-tauri")
                    .join("scripts")
                    .join("models")
                    .join(model_name),
            );
            candidates.push(parent.join("scripts").join("models").join(model_name));
            candidates.push(parent.join("models").join(model_name));
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            candidates.push(parent.join("scripts").join("models").join(model_name));
            candidates.push(parent.join("models").join(model_name));
            if let Some(grandparent) = parent.parent() {
                candidates.push(grandparent.join("scripts").join("models").join(model_name));
                candidates.push(grandparent.join("models").join(model_name));
            }
        }
    }

    candidates.into_iter().find(|p| p.exists())
}

/// Computes a fast, stable Sha256 hash for the source media file.
pub fn compute_source_hash(source_path: &Path) -> Result<String> {
    let meta = std::fs::metadata(source_path)?;
    let mut hasher = Sha256::new();
    hasher.update(meta.len().to_le_bytes());
    if let Ok(modified) = meta.modified() {
        if let Ok(dur) = modified.duration_since(std::time::UNIX_EPOCH) {
            hasher.update(dur.as_secs().to_le_bytes());
        }
    }

    let mut file = std::fs::File::open(source_path)?;
    let mut buffer = [0u8; 65536];
    let mut total_read = 0;
    while total_read < 4 * 1024 * 1024 {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
        total_read += n;
    }

    Ok(format!("{:x}", hasher.finalize()))
}

/// Scores candidate gaps using the pause intelligence sidecar or disk cache.
///
/// On ANY failure (missing model, missing source, script error, timeout, corrupt cache),
/// this gracefully returns an empty Vec, guaranteeing safe fallback for callers.
pub fn score_gaps<P: AsRef<Path>>(
    source_path: P,
    start_sec: f64,
    end_sec: f64,
    words: &[TranscriptWord],
) -> Vec<PauseIntelGap> {
    let source_ref = source_path.as_ref();

    // 1. Feature flag check
    if !crate::pause_intel_enabled() {
        return Vec::new();
    }

    // 2. Empty inputs check
    if words.is_empty() {
        return Vec::new();
    }

    // 3. Source path existence
    if !source_ref.exists() {
        return Vec::new();
    }

    // 4. Check for model file
    let model_file = match find_model_file() {
        Some(m) => m,
        None => {
            eprintln!("[PauseIntel] No trained model found; returning empty gaps");
            return Vec::new();
        }
    };

    // 5. Script existence
    let script_path = match find_sidecar_script() {
        Some(s) => s,
        None => {
            eprintln!("[PauseIntel] pause_intelligence.py script not found; returning empty gaps");
            return Vec::new();
        }
    };

    // 6. Check cache
    let cache_dir = pause_intel_cache_dir();
    let source_hash = compute_source_hash(source_ref).unwrap_or_else(|_| "unknown_src".to_string());
    let cache_file_name = format!(
        "{}_{}_{:.2}_{:.2}.json",
        &source_hash[..16.min(source_hash.len())],
        FEATURE_VERSION,
        start_sec,
        end_sec
    );
    let cache_path = cache_dir.join(&cache_file_name);

    if cache_path.exists() {
        if let Ok(content) = std::fs::read_to_string(&cache_path) {
            if let Ok(cached) = serde_json::from_str::<Vec<PauseIntelGap>>(&content) {
                return cached;
            }
        }
    }

    // 7. Write words to temp file
    let temp_words_path = std::env::temp_dir().join(format!("pause_intel_words_{}.json", uuid::Uuid::new_v4()));
    let words_json = serde_json::json!({ "words": words });
    if let Err(e) = std::fs::write(&temp_words_path, words_json.to_string()) {
        eprintln!("[PauseIntel] Failed to write temp words json: {}", e);
        return Vec::new();
    }

    // 8. Invoke sidecar
    let python = crate::media::find_python_cmd();
    let mut cmd = std::process::Command::new(&python);
    cmd.arg(&script_path)
        .arg(source_ref)
        .arg(format!("{:.1}", start_sec * 1000.0))
        .arg(format!("{:.1}", end_sec * 1000.0))
        .arg(&temp_words_path)
        .env("AUTOSHORTS_PAUSE_MODEL", &model_file);

    let output = match crate::proc_guard::run_bounded(
        &mut cmd,
        Duration::from_secs(SIDECAR_TIMEOUT_SECS),
        "PauseIntel",
    ) {
        Ok(out) => out,
        Err(e) => {
            eprintln!("[PauseIntel] Sidecar spawn failed: {}", e);
            let _ = std::fs::remove_file(&temp_words_path);
            return Vec::new();
        }
    };

    let _ = std::fs::remove_file(&temp_words_path);

    if output.timed_out {
        eprintln!(
            "[PauseIntel] Sidecar TIMED OUT after {}s; returning empty gaps",
            output.elapsed.as_secs_f64()
        );
        return Vec::new();
    }

    if !output.success {
        eprintln!(
            "[PauseIntel] Sidecar exited with error (rc={:?}): {}",
            output.code,
            crate::proc_guard::sanitize_stderr(&output.stderr, 400)
        );
        return Vec::new();
    }

    // 9. Parse output (gracefully isolate JSON substring if any log lines precede)
    let stdout_str = output.stdout.trim();
    let json_str = match (stdout_str.find('{'), stdout_str.rfind('}')) {
        (Some(start), Some(end)) if start <= end => &stdout_str[start..=end],
        _ => stdout_str,
    };

    let payload: PausePredictPayload = match serde_json::from_str(json_str) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("[PauseIntel] Failed to parse sidecar JSON output: {}", e);
            return Vec::new();
        }
    };

    if !payload.gaps.is_empty() {
        if let Ok(_) = std::fs::create_dir_all(&cache_dir) {
            if let Ok(serialized) = serde_json::to_string(&payload.gaps) {
                let _ = std::fs::write(&cache_path, serialized);
            }
        }
    }

    payload.gaps
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn make_word(text: &str, start: f64, end: f64) -> TranscriptWord {
        TranscriptWord {
            text: text.to_string(),
            start,
            end,
            speaker: Some("spk_0".to_string()),
        }
    }

    #[test]
    fn test_pause_intel_gap_serde_camel_case() {
        let json = r#"{
            "gapIdx": 2,
            "srcStartSec": 1.25,
            "srcEndSec": 2.50,
            "pRemovable": 0.85,
            "pRemovableAboveThreshold": true,
            "predictedClass": "breath_pause",
            "category": "breath_pause",
            "confidence": 0.92,
            "features": { "rms_db": -24.5 }
        }"#;

        let gap: PauseIntelGap = serde_json::from_str(json).expect("failed to deserialize camelCase PauseIntelGap");
        assert_eq!(gap.gap_idx, 2);
        assert!((gap.src_start_sec - 1.25).abs() < 1e-6);
        assert!((gap.src_end_sec - 2.50).abs() < 1e-6);
        assert!((gap.p_removable - 0.85).abs() < 1e-6);
        assert!(gap.p_removable_above_threshold);
        assert_eq!(gap.predicted_class.as_deref(), Some("breath_pause"));
        assert_eq!(gap.category.as_deref(), Some("breath_pause"));
        assert_eq!(gap.confidence, Some(0.92));
        assert!(gap.features.is_some());

        // Roundtrip back to JSON
        let serialized = serde_json::to_string(&gap).expect("failed to serialize PauseIntelGap");
        assert!(serialized.contains("\"gapIdx\":2"));
        assert!(serialized.contains("\"srcStartSec\":1.25"));
        assert!(serialized.contains("\"pRemovable\":0.85"));
        assert!(serialized.contains("\"pRemovableAboveThreshold\":true"));
        assert!(serialized.contains("\"predictedClass\":\"breath_pause\""));
    }

    #[test]
    fn test_pause_intel_gap_serde_snake_case_aliases() {
        let json = r#"{
            "gap_idx": 5,
            "src_start_sec": 3.0,
            "src_end_sec": 4.1,
            "p_removable": 0.35,
            "p_removable_above_threshold": false,
            "predicted_class": "normal_word_gap",
            "category": "normal_word_gap"
        }"#;

        let gap: PauseIntelGap = serde_json::from_str(json).expect("failed to deserialize snake_case PauseIntelGap");
        assert_eq!(gap.gap_idx, 5);
        assert!((gap.src_start_sec - 3.0).abs() < 1e-6);
        assert!((gap.src_end_sec - 4.1).abs() < 1e-6);
        assert!((gap.p_removable - 0.35).abs() < 1e-6);
        assert!(!gap.p_removable_above_threshold);
        assert_eq!(gap.predicted_class.as_deref(), Some("normal_word_gap"));
        assert_eq!(gap.category.as_deref(), Some("normal_word_gap"));

        // Test with `idx` alias and omitted defaults
        let json_idx = r#"{
            "idx": 7,
            "src_start_sec": 10.0,
            "src_end_sec": 10.8,
            "p_removable": 0.77
        }"#;

        let gap_idx: PauseIntelGap = serde_json::from_str(json_idx).expect("failed to deserialize idx alias");
        assert_eq!(gap_idx.gap_idx, 7);
        assert!(!gap_idx.p_removable_above_threshold); // default false
        assert_eq!(gap_idx.predicted_class, None);
        assert_eq!(gap_idx.category, None);
        assert_eq!(gap_idx.confidence, None);
        assert_eq!(gap_idx.features, None);
    }

    #[test]
    fn test_pause_predict_payload_serde() {
        let json = r#"{
            "sourceHash": "deadbeef12345678",
            "featureVersion": "1",
            "modelScored": true,
            "model": "scripts/models/pause_classifier_v1.txt",
            "gaps": [
                {
                    "gapIdx": 0,
                    "srcStartSec": 0.5,
                    "srcEndSec": 1.2,
                    "pRemovable": 0.80,
                    "pRemovableAboveThreshold": true
                }
            ]
        }"#;

        let payload: PausePredictPayload = serde_json::from_str(json).expect("failed to deserialize PausePredictPayload");
        assert_eq!(payload.source_hash.as_deref(), Some("deadbeef12345678"));
        assert_eq!(payload.feature_version.as_deref(), Some("1"));
        assert!(payload.model_scored);
        assert_eq!(payload.model.as_deref(), Some("scripts/models/pause_classifier_v1.txt"));
        assert_eq!(payload.gaps.len(), 1);
        assert_eq!(payload.gaps[0].gap_idx, 0);
        assert!(payload.gaps[0].p_removable_above_threshold);
    }

    #[test]
    fn test_missing_model_returns_empty_vec() {
        let _guard = ENV_LOCK.lock().unwrap();

        let orig_flag = std::env::var("AUTOSHORTS_SMART_PACING_LEARNED").ok();
        std::env::set_var("AUTOSHORTS_SMART_PACING_LEARNED", "1");

        // Point AUTOSHORTS_PAUSE_MODEL to a non-existent file
        let orig = std::env::var("AUTOSHORTS_PAUSE_MODEL").ok();
        std::env::set_var("AUTOSHORTS_PAUSE_MODEL", "non_existent_model_file_12345.txt");

        // Create a dummy source file so source_ref.exists() is true
        let temp_src = std::env::temp_dir().join(format!("test_src_{}.mp4", uuid::Uuid::new_v4()));
        std::fs::write(&temp_src, b"dummy audio content").unwrap();

        let words = vec![make_word("hello", 0.0, 1.0), make_word("world", 1.5, 2.0)];
        let gaps = score_gaps(&temp_src, 0.0, 2.0, &words);

        // Cleanup
        let _ = std::fs::remove_file(&temp_src);
        match orig {
            Some(v) => std::env::set_var("AUTOSHORTS_PAUSE_MODEL", v),
            None => std::env::remove_var("AUTOSHORTS_PAUSE_MODEL"),
        }
        match orig_flag {
            Some(v) => std::env::set_var("AUTOSHORTS_SMART_PACING_LEARNED", v),
            None => std::env::remove_var("AUTOSHORTS_SMART_PACING_LEARNED"),
        }

        assert!(gaps.is_empty(), "Expected empty Vec on missing model");
    }

    #[test]
    fn test_cache_directory_path_formatting() {
        let dir = pause_intel_cache_dir();
        let path_str = dir.to_string_lossy();
        assert!(
            path_str.ends_with("pause_intel_cache"),
            "Cache directory path should end with pause_intel_cache, got: {}",
            path_str
        );
        assert!(
            path_str.contains("com.autoshorts.desktop"),
            "Cache directory path should contain com.autoshorts.desktop, got: {}",
            path_str
        );
    }

    #[test]
    fn test_hash_computation_on_test_file() {
        let temp_file = std::env::temp_dir().join(format!("test_hash_{}.dat", uuid::Uuid::new_v4()));
        let content = b"AutoShorts 11.0 Phase 3 Pause Intelligence SHA256 Test Content";
        let mut f = std::fs::File::create(&temp_file).expect("create temp file");
        f.write_all(content).expect("write content");
        drop(f);

        let hash_result = compute_source_hash(&temp_file);
        assert!(hash_result.is_ok(), "compute_source_hash failed: {:?}", hash_result.err());
        let hash = hash_result.unwrap();

        assert_eq!(hash.len(), 64, "SHA256 hex string should be 64 characters");

        // Determinism check: computing hash again on unchanged file yields identical result
        let hash_repeat = compute_source_hash(&temp_file).unwrap();
        assert_eq!(hash, hash_repeat, "Hash must be deterministic");

        let _ = std::fs::remove_file(&temp_file);
    }

    #[test]
    fn test_disabled_flag_returns_empty_vec() {
        let _guard = ENV_LOCK.lock().unwrap();

        let orig = std::env::var("AUTOSHORTS_SMART_PACING_LEARNED").ok();

        let temp_src = std::env::temp_dir().join(format!("test_src_flag_{}.mp4", uuid::Uuid::new_v4()));
        std::fs::write(&temp_src, b"dummy audio content").unwrap();

        let words = vec![make_word("hello", 0.0, 1.0), make_word("world", 1.5, 2.0)];

        // Test 1: Unset defaults to disabled
        std::env::remove_var("AUTOSHORTS_SMART_PACING_LEARNED");
        let gaps_unset = score_gaps(&temp_src, 0.0, 2.0, &words);
        assert!(gaps_unset.is_empty(), "Expected empty Vec when flag is unset (default disabled)");

        // Test 2: Explicitly "0"
        std::env::set_var("AUTOSHORTS_SMART_PACING_LEARNED", "0");
        let gaps_zero = score_gaps(&temp_src, 0.0, 2.0, &words);
        assert!(gaps_zero.is_empty(), "Expected empty Vec when flag is '0'");

        // Test 3: Explicitly "false"
        std::env::set_var("AUTOSHORTS_SMART_PACING_LEARNED", "false");
        let gaps_false = score_gaps(&temp_src, 0.0, 2.0, &words);
        assert!(gaps_false.is_empty(), "Expected empty Vec when flag is 'false'");

        let _ = std::fs::remove_file(&temp_src);
        match orig {
            Some(v) => std::env::set_var("AUTOSHORTS_SMART_PACING_LEARNED", v),
            None => std::env::remove_var("AUTOSHORTS_SMART_PACING_LEARNED"),
        }
    }

    #[test]
    fn test_empty_words_or_missing_source_returns_empty_vec() {
        let words = vec![make_word("hello", 0.0, 1.0)];
        let gaps_missing_file = score_gaps("non_existent_source_99999.mp4", 0.0, 5.0, &words);
        assert!(gaps_missing_file.is_empty(), "Missing source file must return empty gaps");

        let temp_src = std::env::temp_dir().join(format!("test_src_empty_{}.mp4", uuid::Uuid::new_v4()));
        std::fs::write(&temp_src, b"dummy audio content").unwrap();
        let gaps_empty_words = score_gaps(&temp_src, 0.0, 5.0, &[]);
        let _ = std::fs::remove_file(&temp_src);
        assert!(gaps_empty_words.is_empty(), "Empty words slice must return empty gaps");
    }

    #[test]
    fn test_cache_hit_serves_cached_gaps() {
        let _guard = ENV_LOCK.lock().unwrap();

        let orig_flag = std::env::var("AUTOSHORTS_SMART_PACING_LEARNED").ok();
        std::env::set_var("AUTOSHORTS_SMART_PACING_LEARNED", "1");

        let temp_model = std::env::temp_dir().join(format!("test_model_{}.txt", uuid::Uuid::new_v4()));
        std::fs::write(&temp_model, b"dummy model").unwrap();

        let orig_model = std::env::var("AUTOSHORTS_PAUSE_MODEL").ok();
        std::env::set_var("AUTOSHORTS_PAUSE_MODEL", &temp_model);

        let temp_src = std::env::temp_dir().join(format!("test_src_cache_{}.mp4", uuid::Uuid::new_v4()));
        std::fs::write(&temp_src, b"cached source media content").unwrap();

        let source_hash = compute_source_hash(&temp_src).expect("hash source");
        let cache_dir = pause_intel_cache_dir();
        let _ = std::fs::create_dir_all(&cache_dir);

        let start_sec = 1.0;
        let end_sec = 3.0;
        let cache_file_name = format!(
            "{}_{}_{:.2}_{:.2}.json",
            &source_hash[..16.min(source_hash.len())],
            FEATURE_VERSION,
            start_sec,
            end_sec
        );
        let cache_path = cache_dir.join(&cache_file_name);

        let expected_gaps = vec![PauseIntelGap {
            gap_idx: 42,
            src_start_sec: 1.2,
            src_end_sec: 1.8,
            p_removable: 0.95,
            p_removable_above_threshold: true,
            predicted_class: Some("breath_pause".to_string()),
            category: Some("breath_pause".to_string()),
            confidence: Some(0.99),
            features: None,
        }];
        std::fs::write(&cache_path, serde_json::to_string(&expected_gaps).unwrap()).unwrap();

        let words = vec![make_word("hello", 1.0, 1.2), make_word("world", 1.8, 2.5)];
        let gaps = score_gaps(&temp_src, start_sec, end_sec, &words);

        // Cleanup
        let _ = std::fs::remove_file(&cache_path);
        let _ = std::fs::remove_file(&temp_src);
        let _ = std::fs::remove_file(&temp_model);
        match orig_model {
            Some(v) => std::env::set_var("AUTOSHORTS_PAUSE_MODEL", v),
            None => std::env::remove_var("AUTOSHORTS_PAUSE_MODEL"),
        }
        match orig_flag {
            Some(v) => std::env::set_var("AUTOSHORTS_SMART_PACING_LEARNED", v),
            None => std::env::remove_var("AUTOSHORTS_SMART_PACING_LEARNED"),
        }

        assert_eq!(gaps, expected_gaps, "score_gaps must serve cached gaps on cache hit");
    }
}
