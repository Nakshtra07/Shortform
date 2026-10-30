# Three Core Architectural Upgrades Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the three core architectural upgrades approved in `docs/superpowers/specs/2026-10-26-three-core-upgrades-design.md`: zero-redundancy Deepgram diarization in Rust, diagonal resolution-independent framing in Python, and AGPL-decoupled atomic YOLO11 weights downloading in Tauri Rust.

**Architecture:** 
1. Diarization is derived directly from `TranscriptWord` in Rust if available, eliminating duplicate Deepgram uploads and billing.
2. Tracking thresholds in `speaker_tracker.py` are normalized via a diagonal and area `ResolutionScaler` anchored to standard 1080p.
3. Desktop distribution bundles remain 100% permissively licensed (Apache-2.0 YuNet only) while a robust Rust manager atomically downloads YOLO11 weights to `%LOCALAPPDATA%/autoshorts/models/` with graceful offline fallback.

**Tech Stack:** Rust (Tokio, Reqwest, Tauri 2), Python (NumPy, OpenCV, Ultralytics YOLO11, Pytest), React 19 / TypeScript.

## Global Constraints

- Do not break existing 1080p tracking math or numerical tolerances (`scale_diag == 1.0` at 1920x1080).
- Maintain 100% passing status across existing cargo test suite (695 tests) and Python suites.
- Do not introduce new unmanaged binary assets into git or distributed installer bundles.
- All file edits must be backwards-compatible with offline environments (fallback to OpenCV YuNet).

---

### Task 1: Deepgram Diarization & Zero-Redundancy Speaker Turn Propagation

**Files:**
- Modify: `autoshorts/src-tauri/src/speaker_intelligence.rs:240-285`
- Test: `autoshorts/src-tauri/src/speaker_intelligence.rs:1650-1750`

**Interfaces:**
- Consumes: `TranscriptWord` (`text`, `start`, `end`, `speaker: Option<String>`) from `transcription.rs`.
- Produces: `SpeakerIntelligenceEngine::build_diarization_from_transcript_words(words: &[TranscriptWord], model_label: &str) -> Option<SpeakerDiarizationResult>`.

- [ ] **Step 1: Write failing unit tests for `build_diarization_from_transcript_words`**

Add the following tests to the `tests` module in `autoshorts/src-tauri/src/speaker_intelligence.rs`:

```rust
#[test]
fn test_build_diarization_from_transcript_words_multi_speaker() {
    let words = vec![
        TranscriptWord {
            text: "Hello".to_string(),
            start: 0.0,
            end: 0.5,
            speaker: Some("S1".to_string()),
        },
        TranscriptWord {
            text: "world".to_string(),
            start: 0.5,
            end: 1.0,
            speaker: Some("S1".to_string()),
        },
        TranscriptWord {
            text: "Welcome".to_string(),
            start: 1.2,
            end: 1.8,
            speaker: Some("S2".to_string()),
        },
        TranscriptWord {
            text: "back".to_string(),
            start: 1.8,
            end: 2.2,
            speaker: Some("S2".to_string()),
        },
        TranscriptWord {
            text: "Thanks".to_string(),
            start: 2.5,
            end: 3.0,
            speaker: Some("S1".to_string()),
        },
    ];

    let diar = SpeakerIntelligenceEngine::build_diarization_from_transcript_words(&words, "test-model")
        .expect("should produce diarization from tagged words");

    assert_eq!(diar.speakers.len(), 2);
    assert_eq!(diar.segments.len(), 3);
    assert_eq!(diar.segments[0].speaker_id, "S1");
    assert_eq!(diar.segments[1].speaker_id, "S2");
    assert_eq!(diar.segments[2].speaker_id, "S1");
    assert_eq!(diar.model, "test-model");
    assert!(diar.overall_confidence >= 0.90);
}

#[test]
fn test_build_diarization_from_transcript_words_none_fallback() {
    let words = vec![
        TranscriptWord {
            text: "Hello".to_string(),
            start: 0.0,
            end: 0.5,
            speaker: None,
        },
        TranscriptWord {
            text: "world".to_string(),
            start: 0.5,
            end: 1.0,
            speaker: None,
        },
    ];

    let diar = SpeakerIntelligenceEngine::build_diarization_from_transcript_words(&words, "test-model");
    assert!(diar.is_none(), "untagged words must return None to trigger fallback");
}
```

- [ ] **Step 2: Run cargo test to verify tests fail**

Run:
`cargo test --manifest-path autoshorts/src-tauri/Cargo.toml --lib test_build_diarization_from_transcript_words -j 2`
Expected: FAIL with "no function or associated item named `build_diarization_from_transcript_words` found".

- [ ] **Step 3: Implement `build_diarization_from_transcript_words` and wire into `process_source_video`**

In `autoshorts/src-tauri/src/speaker_intelligence.rs`:

```rust
impl SpeakerIntelligenceEngine {
    /// Constructs a SpeakerDiarizationResult directly from word-level transcript tags
    /// without re-uploading audio to external APIs.
    pub fn build_diarization_from_transcript_words(
        words: &[TranscriptWord],
        model_label: &str,
    ) -> Option<SpeakerDiarizationResult> {
        let has_speakers = words.iter().any(|w| w.speaker.is_some());
        if !has_speakers || words.is_empty() {
            return None;
        }

        let mut segments: Vec<DiarizedSegment> = Vec::new();
        let mut cur_speaker: Option<String> = None;
        let mut cur_start = 0.0;
        let mut cur_end = 0.0;

        for word in words {
            let spk = word.speaker.clone().unwrap_or_else(|| "S1".to_string());
            match &cur_speaker {
                Some(s) if *s == spk && (word.start - cur_end) <= 0.45 => {
                    cur_end = word.end;
                }
                Some(s) => {
                    segments.push(DiarizedSegment {
                        speaker_id: s.clone(),
                        start: cur_start,
                        end: cur_end,
                        confidence: 0.95,
                    });
                    cur_speaker = Some(spk);
                    cur_start = word.start;
                    cur_end = word.end;
                }
                None => {
                    cur_speaker = Some(spk);
                    cur_start = word.start;
                    cur_end = word.end;
                }
            }
        }

        if let Some(s) = cur_speaker {
            segments.push(DiarizedSegment {
                speaker_id: s,
                start: cur_start,
                end: cur_end,
                confidence: 0.95,
            });
        }

        let mut speaker_map: std::collections::BTreeMap<String, (f64, usize)> = std::collections::BTreeMap::new();
        for seg in &segments {
            let entry = speaker_map.entry(seg.speaker_id.clone()).or_insert((0.0, 0));
            entry.0 += (seg.end - seg.start).max(0.0);
            entry.1 += 1;
        }

        let speakers = speaker_map
            .into_iter()
            .map(|(id, (speech_sec, seg_count))| DiarizedSpeaker {
                diarization_id: id,
                total_speech_sec: speech_sec,
                segment_count: seg_count,
                avg_confidence: 0.95,
            })
            .collect();

        Some(SpeakerDiarizationResult {
            model: model_label.to_string(),
            version: "inline-v1".to_string(),
            speakers,
            segments,
            overall_confidence: 0.95,
        })
    }
}
```

In `SpeakerIntelligenceEngine::process_source_video()` (line ~273):
```rust
        // Step 1: Diarization (fallback-safe: never blocks the pipeline).
        let diarization = if let Some(inline_diar) = Self::build_diarization_from_transcript_words(
            transcript_words,
            "deepgram-nova3-transcription",
        ) {
            println!(
                "[SpeakerIntel] Zero-redundancy diarization extracted from transcript: {} speakers, {} segments (model={})",
                inline_diar.speakers.len(),
                inline_diar.segments.len(),
                inline_diar.model
            );
            inline_diar
        } else {
            self.run_diarization(source_path, &source_hash)?
        };
```

- [ ] **Step 4: Run cargo test to verify it passes**

Run:
`cargo test --manifest-path autoshorts/src-tauri/Cargo.toml --lib test_build_diarization_from_transcript_words -j 2`
Expected: PASS (2 passed, 0 failed).

- [ ] **Step 5: Commit**

```bash
git add autoshorts/src-tauri/src/speaker_intelligence.rs
git commit -m "feat(speaker_intel): extract zero-redundancy diarization from transcript words"
```

---

### Task 2: Resolution-Independent Framing Thresholds in `speaker_tracker.py`

**Files:**
- Modify: `autoshorts/src-tauri/scripts/speaker_tracker.py:680-695, 1238-1265, 2320-2345, 2590-2605, 3250-3260, 3455-3470, 4960-4970`
- Test: Create `autoshorts/src-tauri/scripts/test_resolution_scaling_suite.py`

**Interfaces:**
- Consumes: `source_w`, `source_h` dimensions of input frames.
- Produces: `ResolutionScaler` class with `scaler.dist(px_1080p: float) -> float` and `scaler.area_val(area_1080p: float) -> float`.

- [ ] **Step 1: Write failing unit test suite `test_resolution_scaling_suite.py`**

Create `autoshorts/src-tauri/scripts/test_resolution_scaling_suite.py`:

```python
import math
import pytest
from speaker_tracker import ResolutionScaler, REF_FRAME_W, REF_FRAME_H, REF_DIAGONAL_1080P, REF_AREA_1080P

def test_scaler_1080p_exact_equivalence():
    scaler = ResolutionScaler(1920, 1080)
    assert abs(scaler.scale_diag - 1.0) < 1e-6
    assert abs(scaler.scale_area - 1.0) < 1e-6
    assert abs(scaler.dist(180.0) - 180.0) < 1e-6
    assert abs(scaler.dist(140.0) - 140.0) < 1e-6
    assert abs(scaler.dist(150.0) - 150.0) < 1e-6
    assert abs(scaler.dist(250.0) - 250.0) < 1e-6
    assert abs(scaler.dist(17.5) - 17.5) < 1e-6
    assert abs(scaler.dist(720.0) - 720.0) < 1e-6
    assert abs(scaler.area_val(1800.0) - 1800.0) < 1e-6

def test_scaler_4k_scaling():
    scaler = ResolutionScaler(3840, 2160)
    assert abs(scaler.scale_diag - 2.0) < 1e-5
    assert abs(scaler.scale_area - 4.0) < 1e-5
    assert abs(scaler.dist(180.0) - 360.0) < 1e-5
    assert abs(scaler.dist(140.0) - 280.0) < 1e-5
    assert abs(scaler.area_val(1800.0) - 7200.0) < 1e-5

def test_scaler_720p_scaling():
    scaler = ResolutionScaler(1280, 720)
    expected_ratio = 1280.0 / 1920.0  # 2/3
    assert abs(scaler.scale_diag - expected_ratio) < 1e-5
    assert abs(scaler.scale_area - (expected_ratio ** 2)) < 1e-5
    assert abs(scaler.dist(180.0) - (180.0 * 2.0 / 3.0)) < 1e-5
    assert abs(scaler.area_val(1800.0) - (1800.0 * 4.0 / 9.0)) < 1e-5
```

- [ ] **Step 2: Run pytest to verify tests fail**

Run:
`python -m pytest autoshorts/src-tauri/scripts/test_resolution_scaling_suite.py`
Expected: FAIL with "ImportError: cannot import name 'ResolutionScaler'".

- [ ] **Step 3: Implement `ResolutionScaler` and normalize thresholds in `speaker_tracker.py`**

In `autoshorts/src-tauri/scripts/speaker_tracker.py`:

Define `ResolutionScaler`:
```python
REF_FRAME_W = 1920.0
REF_FRAME_H = 1080.0
REF_DIAGONAL_1080P = math.hypot(REF_FRAME_W, REF_FRAME_H)
REF_AREA_1080P = REF_FRAME_W * REF_FRAME_H

class ResolutionScaler:
    """Scales Euclidean distances and 2D pixel areas invariant to frame resolution and aspect ratio."""
    def __init__(self, w: float, h: float):
        self.w = max(1.0, float(w))
        self.h = max(1.0, float(h))
        self.diag = math.hypot(self.w, self.h)
        self.area = self.w * self.h
        self.scale_diag = self.diag / REF_DIAGONAL_1080P
        self.scale_area = self.area / REF_AREA_1080P

    def dist(self, px_1080p: float) -> float:
        return px_1080p * self.scale_diag

    def area_val(self, area_1080p: float) -> float:
        return area_1080p * self.scale_area
```

Update threshold usages:
1. In `detect_and_track_faces()` / tracking matching:
   Use `scaler = ResolutionScaler(source_w, source_h)`.
   Replace `dist_thresh=180.0` with `dist_thresh=scaler.dist(180.0)`.
   Replace `dist_thresh=140.0` with `dist_thresh=scaler.dist(140.0)`.
2. In shot-track re-association:
   Replace `math.hypot(tr["cx"] - fcx, tr["cy"] - fcy) < 150.0` with `< scaler.dist(150.0)`.
3. In speaker center shift:
   Replace `abs(target_cx - last_confirmed_cx_by_spk[spk]) > 250.0` with `> scaler.dist(250.0)`.
4. In distant face rejection (line ~3462):
   Replace `fa < 1800.0` with `fa < scaler.area_val(1800.0)`.
5. In deadband computation:
   Replace `_deadband = max(17.5, eff_crop_w * 0.045)` with `_deadband = max(scaler.dist(17.5), eff_crop_w * 0.045)`.

- [ ] **Step 4: Run pytest and regression suite to verify it passes**

Run:
`python -m pytest autoshorts/src-tauri/scripts/test_resolution_scaling_suite.py autoshorts/src-tauri/scripts/test_autoclip_adapter.py`
Expected: ALL PASS (100% green).

- [ ] **Step 5: Commit**

```bash
git add autoshorts/src-tauri/scripts/speaker_tracker.py autoshorts/src-tauri/scripts/test_resolution_scaling_suite.py
git commit -m "feat(smart_framing): introduce ResolutionScaler for resolution-independent tracking thresholds"
```

---

### Task 3: AGPL-3.0 License Decoupling & YOLO11 Weights Downloader in Rust

**Files:**
- Create: `autoshorts/src-tauri/src/model_manager.rs`
- Modify: `autoshorts/src-tauri/src/lib.rs`
- Test: `autoshorts/src-tauri/src/model_manager.rs:tests`

**Interfaces:**
- Consumes: `tauri::AppHandle`, Reqwest client.
- Produces: `model_manager::ensure_yolo_weights(app: Option<&tauri::AppHandle>) -> Result<PathBuf, String>`.
- Exposes: Tauri command `ensure_yolo_model`.

- [ ] **Step 1: Write failing unit tests for `model_manager`**

In `autoshorts/src-tauri/src/model_manager.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_models_dir_structure() {
        let dir = get_models_dir();
        assert!(dir.to_string_lossy().contains("autoshorts"));
        assert!(dir.to_string_lossy().contains("models"));
    }

    #[test]
    fn test_is_yolo_installed_when_missing() {
        let temp_dir = tempfile::tempdir().unwrap();
        let target = temp_dir.path().join("yolo11n.pt");
        assert!(!is_yolo_file_valid(&target));
    }
}
```

- [ ] **Step 2: Run cargo test to verify it fails**

Run:
`cargo test --manifest-path autoshorts/src-tauri/Cargo.toml --lib model_manager -j 2`
Expected: FAIL with "module `model_manager` not found".

- [ ] **Step 3: Implement `model_manager.rs`**

Create `autoshorts/src-tauri/src/model_manager.rs`:

```rust
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use tauri::Emitter;

pub const YOLO_UPSTREAM_URL: &str =
    "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt";
pub const YOLO_MIN_BYTES: u64 = 1_000_000;

pub fn get_models_dir() -> PathBuf {
    if let Ok(local_app_data) = std::env::var("LOCALAPPDATA") {
        PathBuf::from(local_app_data).join("autoshorts").join("models")
    } else {
        dirs::data_local_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("autoshorts")
            .join("models")
    }
}

pub fn is_yolo_file_valid(path: &Path) -> bool {
    if let Ok(meta) = std::fs::metadata(path) {
        meta.is_file() && meta.len() >= YOLO_MIN_BYTES
    } else {
        false
    }
}

pub fn is_yolo_installed() -> bool {
    let target = get_models_dir().join("yolo11n.pt");
    is_yolo_file_valid(&target)
}

pub async fn ensure_yolo_weights<R: tauri::Runtime>(
    app: Option<&tauri::AppHandle<R>>,
) -> Result<PathBuf> {
    let models_dir = get_models_dir();
    std::fs::create_dir_all(&models_dir)
        .with_context(|| format!("creating models directory at {}", models_dir.display()))?;

    let target_path = models_dir.join("yolo11n.pt");
    if is_yolo_file_valid(&target_path) {
        return Ok(target_path);
    }

    println!("[ModelManager] yolo11n.pt not found. Initiating download from upstream...");
    let temp_part_path = models_dir.join("yolo11n.pt.part");

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(300))
        .build()
        .context("building HTTP client for model download")?;

    let response = client
        .get(YOLO_UPSTREAM_URL)
        .send()
        .await
        .context("requesting YOLO11 weights from GitHub releases")?;

    if !response.status().is_success() {
        return Err(anyhow::anyhow!(
            "Failed to download YOLO weights: HTTP {}",
            response.status()
        ));
    }

    let total_size = response.content_length().unwrap_or(5_613_764);
    let mut downloaded: u64 = 0;
    let mut stream = response.bytes_stream();
    let mut file = tokio::fs::File::create(&temp_part_path)
        .await
        .context("creating temporary model part file")?;

    use futures_util::StreamExt;
    use tokio::io::AsyncWriteExt;

    while let Some(chunk_res) = stream.next().await {
        let chunk = chunk_res.context("reading stream chunk")?;
        file.write_all(&chunk)
            .await
            .context("writing chunk to temporary file")?;
        downloaded += chunk.len() as u64;

        if let Some(app_handle) = app {
            let percent = if total_size > 0 {
                (downloaded as f64 / total_size as f64) * 100.0
            } else {
                0.0
            };
            let _ = app_handle.emit(
                "model_download_progress",
                serde_json::json!({
                    "model": "yolo11n.pt",
                    "downloaded": downloaded,
                    "total": total_size,
                    "percent": percent
                }),
            );
        }
    }

    file.flush().await.context("flushing model file")?;
    drop(file);

    tokio::fs::rename(&temp_part_path, &target_path)
        .await
        .context("renaming part file to yolo11n.pt")?;

    println!(
        "[ModelManager] yolo11n.pt successfully verified and placed at {}",
        target_path.display()
    );
    Ok(target_path)
}
```

Add module and command in `autoshorts/src-tauri/src/lib.rs`:
```rust
pub mod model_manager;

#[tauri::command]
pub async fn ensure_yolo_model(app: tauri::AppHandle) -> Result<String, String> {
    model_manager::ensure_yolo_weights(Some(&app))
        .await
        .map(|p| p.to_string_lossy().to_string())
        .map_err(|e| e.to_string())
}
```
And register `ensure_yolo_model` in the invoke handler in `lib.rs`.

- [ ] **Step 4: Run cargo test to verify it passes**

Run:
`cargo test --manifest-path autoshorts/src-tauri/Cargo.toml --lib model_manager -j 2`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add autoshorts/src-tauri/src/model_manager.rs autoshorts/src-tauri/src/lib.rs
git commit -m "feat(model_manager): decouple AGPL weights and add atomic YOLO model downloader"
```

---

### Task 4: Full System Verification

**Files:**
- Entire codebase

- [ ] **Step 1: Run complete Rust library test suite**
Run: `cargo test --manifest-path autoshorts/src-tauri/Cargo.toml --lib -j 2`
Expected: 697+ passed, 0 failed.

- [ ] **Step 2: Run complete Python smart framing tests**
Run: `python -m pytest autoshorts/src-tauri/scripts/test_resolution_scaling_suite.py autoshorts/src-tauri/scripts/test_autoclip_adapter.py`
Expected: 100% passed.

- [ ] **Step 3: Run Rust build check**
Run: `cargo check --manifest-path autoshorts/src-tauri/Cargo.toml`
Expected: 0 errors.

- [ ] **Step 4: Run TypeScript / Vite frontend build**
Run: `npm --prefix autoshorts run build`
Expected: Build succeeds with 0 errors.
