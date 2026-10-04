//! AutoShorts 11.0 Phase 3 — T7 Prosodic Boundary sidecar wrapper
//!
//! Wraps `t7_prosody.py` as a sidecar, bounded via `proc_guard::run_bounded` (30s timeout).
//! Caches results per (source_hash, feature_version, start_sec, end_sec) under
//! `%APPDATA%/com.autoshorts.desktop/t7_cache/`. Missing or corrupt model gracefully
//! degrades to empty boundaries.

use crate::models::TranscriptWord;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Duration;

pub const FEATURE_VERSION: &str = "t7_prosody_v1";
pub const SIDECAR_TIMEOUT_SECS: u64 = 30;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct T7Boundary {
    #[serde(alias = "gap_idx")]
    pub gap_idx: usize,
    #[serde(alias = "p_boundary")]
    pub p_boundary: f64,
    #[serde(default)]
    pub features: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct T7PredictPayload {
    #[serde(default)]
    pub model_scored: bool,
    #[serde(default)]
    pub feature_version: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub boundaries: Vec<T7Boundary>,
}

/// Returns the cache directory for T7 prosody sidecar results.
pub fn t7_cache_dir() -> PathBuf {
    dirs::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("com.autoshorts.desktop")
        .join("t7_cache")
}

/// Locates the `t7_prosody.py` sidecar script.
pub fn find_sidecar_script() -> Option<PathBuf> {
    let script_name = "t7_prosody.py";
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

/// Locates the `t7_prosody_v1.txt` model file if trained.
pub fn find_model_file() -> Option<PathBuf> {
    let model_name = "t7_prosody_v1.txt";
    let mut candidates = Vec::new();

    if let Ok(cargo_manifest) = std::env::var("CARGO_MANIFEST_DIR") {
        candidates.push(
            PathBuf::from(cargo_manifest)
                .join("scripts")
                .join("models")
                .join(model_name),
        );
    }
    if let Ok(cwd) = std::env::current_dir() {
        candidates.push(cwd.join("scripts").join("models").join(model_name));
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

/// Invokes the `t7_prosody.py` sidecar in `--predict` mode or serves cached predictions.
///
/// On ANY failure (missing model, missing source, script error, timeout, corrupt cache),
/// this gracefully returns an empty Vec, guaranteeing byte-identical fallback for callers.
pub fn predict_t7_boundaries<P: AsRef<Path>>(
    source_path: P,
    start_sec: f64,
    end_sec: f64,
    words: &[TranscriptWord],
) -> Vec<T7Boundary> {
    let source_ref = source_path.as_ref();

    // 1. Feature flag check
    if !crate::t7_prosody_enabled() {
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
            eprintln!("[T7Prosody] No trained model found (DATA-BLOCKED); returning empty boundaries");
            return Vec::new();
        }
    };

    // 5. Script existence
    let script_path = match find_sidecar_script() {
        Some(s) => s,
        None => {
            eprintln!("[T7Prosody] t7_prosody.py script not found; returning empty boundaries");
            return Vec::new();
        }
    };

    // 6. Check cache
    let cache_dir = t7_cache_dir();
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
            if let Ok(cached) = serde_json::from_str::<Vec<T7Boundary>>(&content) {
                return cached;
            }
        }
    }

    // 7. Write words to temp file
    let temp_words_path = std::env::temp_dir().join(format!("t7_words_{}.json", uuid::Uuid::new_v4()));
    let words_json = serde_json::json!({ "words": words });
    if let Err(e) = std::fs::write(&temp_words_path, words_json.to_string()) {
        eprintln!("[T7Prosody] Failed to write temp words json: {}", e);
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
        .arg("--predict")
        .arg("--model")
        .arg(&model_file);

    let output = match crate::proc_guard::run_bounded(
        &mut cmd,
        Duration::from_secs(SIDECAR_TIMEOUT_SECS),
        "T7Prosody",
    ) {
        Ok(out) => out,
        Err(e) => {
            eprintln!("[T7Prosody] Sidecar spawn failed: {}", e);
            let _ = std::fs::remove_file(&temp_words_path);
            return Vec::new();
        }
    };

    let _ = std::fs::remove_file(&temp_words_path);

    if output.timed_out {
        eprintln!(
            "[T7Prosody] Sidecar TIMED OUT after {}s; returning empty boundaries",
            output.elapsed.as_secs_f64()
        );
        return Vec::new();
    }

    if !output.success {
        eprintln!(
            "[T7Prosody] Sidecar exited with error (rc={:?}): {}",
            output.code,
            crate::proc_guard::sanitize_stderr(&output.stderr, 400)
        );
        return Vec::new();
    }

    // 9. Parse output
    let payload: T7PredictPayload = match serde_json::from_str(&output.stdout) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("[T7Prosody] Failed to parse sidecar JSON output: {}", e);
            return Vec::new();
        }
    };

    if !payload.boundaries.is_empty() {
        if let Ok(_) = std::fs::create_dir_all(&cache_dir) {
            if let Ok(serialized) = serde_json::to_string(&payload.boundaries) {
                let _ = std::fs::write(&cache_path, serialized);
            }
        }
    }

    payload.boundaries
}
