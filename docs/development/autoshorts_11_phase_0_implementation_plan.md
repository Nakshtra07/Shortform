# AutoShorts 11.0 — Phase 0 Implementation Plan (Revised)
# Correctness, Data Persistence, and Foundation Infrastructure

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct all five foundational defects (F1, F6, F8, F9, F7) identified in `ML_RESEARCH_REPORT_AUTOSHORTS_11.0.md`, replacing fake acoustic heuristics with real DSP extraction via librosa/FFmpeg (with honest acoustic burstiness naming), unlocking 100% complete `CandidateDraft` persistence with observable corruption error paths, eliminating hardcoded paths and silent framing fallbacks with explicit emergency telemetry, unifying the production tracking loop under an authoritative `FramingConfig`, and making back-of-head geometry thresholds resolution-independent across 720p, 1080p, 4K, and 9:16 portrait.

**Architecture:** AutoShorts remains a hybrid architecture where ML and DSP produce scores, probabilities, and advisory metadata while deterministic rules maintain hard safety guarantees. Phase 0 requires zero new machine learning models and zero new binary dependencies, strictly hardening existing runtime Python scripts and Rust database/media pipelines with full backward compatibility.

**Tech Stack:** Rust (Tauri, Rusqlite, Serde), Python 3.12 (librosa 1.0.0, soundfile, numpy, scipy, opencv-python), FFmpeg, Vitest / React / TypeScript.

---

## Global Constraints

- **Strict Invariant 1 (Hybrid Rule):** ML and DSP produce scores, probabilities, and metadata; existing deterministic engines and hard safety constraints govern all final rendering decisions.
- **Strict Invariant 2 (Locked Payoff Endpoint):** The payoff endpoint contract remains strictly locked (`LLM payoff -> transcript alignment -> payoff_end -> candidate_end`). No duration caps (75s/90s), no payoff truncation, and Hook Intelligence cannot alter payoff boundaries.
- **Strict Invariant 3 (Adaptive Framing Invariant):** Two persistent valid people in Adaptive mode must stay together in a shared 1080×1080 composition, with no active-speaker follow and no DualFrame.
- **Strict Invariant 4 (Caption Safety):** T7 phrase-paired speech-chunked timing, Matt Bold 110, non-karaoke, and face-safe placement invariants are strictly preserved.
- **Strict Invariant 5 (Audio Intelligence):** Audio Intelligence remains OFF by default on clean studio audio.
- **Strict Invariant 6 (Authoritative Workspace Isolation):** Work is performed strictly in the confirmed authoritative AutoShorts 11.0 project directory (`d:\College\Autoshorts 11.0`). No copying or overwriting between versioned project directories (`10.0`, `11.0`, etc.) shall occur.
- **Strict Invariant 7 (Zero Regression):** All 282 Rust unit tests, 16 existing Python test suites, and frontend `npm run build` must remain green at every step.

---

## File Structure & Responsibilities

| File Path | Responsibility | Changes in Phase 0 |
|---|---|---|
| `autoshorts/src-tauri/scripts/multimodal_hook_analyzer.py` | Python sidecar extracting visual, acoustic, and temporal signals for hook evaluation | **Task 1 (F1):** Replace fake punctuation heuristics with real audio DSP extraction via FFmpeg + `librosa` / `soundfile` (RMS energy envelope, pitch F0 via `yin`, silence detection, high-frequency onset burstiness). Document burstiness honestly as `burstiness` / `laughter_proxy` without claiming genuine laughter detection. |
| `autoshorts/src-tauri/scripts/test_multimodal_acoustic_suite.py` | New Python unit test suite | **Task 1 (F1):** Dedicated test suite verifying real audio loading, silence pause detection, pitch tracking, burstiness measurement, and graceful fallback on audio-less media. |
| `autoshorts/src-tauri/src/db.rs` | Rust SQLite persistence layer | **Task 2 (F6):** Migrate `candidates` table to store full `CandidateDraft` metadata (all ~30+ fields) via `metadata_json` column. Implement observable 3-state handling: absent (legacy), valid (full draft), and corrupted (explicit `MetadataError`). |
| `autoshorts/src-tauri/src/models.rs` | Rust domain data structures | **Task 2 (F6):** Add `metadata_json` to `Candidate`, implement `try_metadata_draft()` with explicit error handling, and implement `PartialEq` on `CandidateDraft` to enable exhaustive equality assertions. |
| `autoshorts/src-tauri/scripts/speaker_tracker.py` | 5,550-line Python visual reframing & tracking engine | **Task 3 (F8):** Remove hardcoded `Autoshorts 5.0` paths.<br>**Task 4 (F9):** Wire dead constants to authoritative `FramingConfig`; wire `config` into the production `run()` entry point and reframe loop.<br>**Task 5 (F7):** Normalize back-of-head pixel thresholds (18000, 14000, 35000) to frame-area fractions. |
| `autoshorts/src-tauri/src/media.rs` | Rust media processor and sidecar launcher | **Task 3 (F8):** Make `find_speaker_tracker_script` search dynamically and emit explicit emergency telemetry when falling back to center crop, distinguishing emergency recovery from successful tracking. |
| `autoshorts/src-tauri/scripts/test_framing_config_suite.py` | New Python test suite | **Task 4 (F9):** Verify that passing custom `FramingConfig` into the production `run()` tracking loop alters runtime thresholds and camera reframing decisions. |
| `autoshorts/src-tauri/scripts/test_geometry_scaling_suite.py` | New Python test suite | **Task 5 (F7):** Verify resolution-independent geometry scaling across 1280×720, 1920×1080, 3840×2160, and 1080×1920 (portrait), testing boundary conditions and confirming width/height orientation integrity. |
| `autoshorts/src-tauri/scripts/test_path_resolution.py` | New Python test suite | **Task 3 (F8):** Verify no hardcoded paths remain and test env-var path overrides. |

---

## Implementation Tasks

### Task 1: Real Acoustic Feature Analyzer with Truthful Semantics (Fixing F1)

**Files:**
- Modify: `autoshorts/src-tauri/scripts/multimodal_hook_analyzer.py:186-270`
- Create: `autoshorts/src-tauri/scripts/test_multimodal_acoustic_suite.py`

**Interfaces:**
- Consumes: `video_path: str`, `candidates: List[Dict[str, Any]]`, `force_fail: bool`
- Produces: `Tuple[str, Optional[str], Dict[int, Dict[str, Any]]]` where each candidate result contains:
  - `energy_change: float` (0.0 to 1.0, normalized RMS ratio)
  - `peak_strength: float` (0.0 to 1.0, normalized hook peak RMS)
  - `pitch_change: float` (0.0 to 1.0, YIN F0 variation/slope)
  - `pause_emphasis: float` (0.0 to 1.0, pre-hook silence duration)
  - `burstiness: float` (0.0 to 1.0, onset spectral burstiness)
  - `laughter: float` (alias to `burstiness` for backward compatibility, explicitly documented as `laughter_proxy`, NOT a trained laughter detector)
  - `score: float` (0.0 to 1.0, composite acoustic score)
  - `evidence: str` (truthful DSP summary of measured decibels, hertz, and seconds)

- [ ] **Step 1: Write the failing unit test for real acoustic DSP analysis**

Create `autoshorts/src-tauri/scripts/test_multimodal_acoustic_suite.py`:
```python
import os
import unittest
import numpy as np
import soundfile as sf
import tempfile
from multimodal_hook_analyzer import analyze_acoustic_signals, extract_audio_segment_dsp

class TestMultimodalAcousticSuite(unittest.TestCase):
    def setUp(self):
        # Create a synthetic 16kHz WAV file with:
        # 0.0 - 1.0s: silence (RMS < -50 dB)
        # 1.0 - 2.5s: 150 Hz tone (moderate energy)
        # 2.5 - 4.0s: 300 Hz tone (high energy burst)
        self.sr = 16000
        dur = 5.0
        t = np.linspace(0, dur, int(self.sr * dur), endpoint=False)
        audio = np.zeros_like(t)
        
        # 1.0 to 2.5s: 150 Hz at 0.2 amplitude
        idx_body = (t >= 1.0) & (t < 2.5)
        audio[idx_body] = 0.2 * np.sin(2 * np.pi * 150 * t[idx_body])
        
        # 2.5 to 4.0s: 300 Hz at 0.8 amplitude
        idx_hook = (t >= 2.5) & (t < 4.0)
        audio[idx_hook] = 0.8 * np.sin(2 * np.pi * 300 * t[idx_hook])
        
        self.tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(self.tmp_wav.name, audio, self.sr)
        self.tmp_wav.close()

    def tearDown(self):
        if os.path.exists(self.tmp_wav.name):
            os.remove(self.tmp_wav.name)

    def test_real_audio_dsp_extraction(self):
        candidates = [{
            "start": 0.0,
            "end": 5.0,
            "hookStart": 2.5,
            "hookEnd": 4.0,
            "hook": "This is a real acoustic test without punctuation"
        }]
        status, err, results = analyze_acoustic_signals(self.tmp_wav.name, candidates)
        self.assertEqual(status, "SUCCESS")
        self.assertIsNone(err)
        cand_res = results[0]
        
        # Verify metrics are grounded in actual audio DSP
        self.assertGreater(cand_res["peak_strength"], 0.6)
        self.assertGreater(cand_res["energy_change"], 0.6)
        self.assertIn("burstiness", cand_res)
        self.assertEqual(cand_res["laughter"], cand_res["burstiness"]) # Verify proxy alias
        
        # Verify evidence does not claim laughter or interrogative pitch without cause
        self.assertIn("Audio:", cand_res["evidence"])
        self.assertNotIn("rising vocal pitch contour with interrogative emphasis", cand_res["evidence"])
        self.assertNotIn("laughter detected", cand_res["evidence"].lower())

    def test_missing_audio_fails_gracefully(self):
        candidates = [{"start": 0.0, "end": 5.0, "hookStart": 0.0, "hookEnd": 2.0}]
        status, err, results = analyze_acoustic_signals("non_existent_file.mp4", candidates)
        self.assertEqual(status, "SKIPPED")
        self.assertIn("Audio source not found", err)

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```powershell
& "d:\College\Autoshorts 11.0\.venv\Scripts\python.exe" autoshorts/src-tauri/scripts/test_multimodal_acoustic_suite.py
```
Expected: FAIL because `extract_audio_segment_dsp` is not defined and current `analyze_acoustic_signals` checks for `?`/`!` instead of reading audio.

- [ ] **Step 3: Implement real acoustic analysis with truthful semantics**

In `autoshorts/src-tauri/scripts/multimodal_hook_analyzer.py`:
- Implement `load_audio_segment(source_path, start_sec, dur_sec, sr=16000)` reading via FFmpeg stdout pipe or `soundfile`.
- Implement `extract_audio_segment_dsp(audio, sr, hook_rel_start, hook_rel_end, pre_hook_dur=2.0)`:
  - Compute RMS energy envelope with 20ms hop size.
  - Calculate `energy_change` from hook vs candidate RMS ratio.
  - Calculate `peak_strength` from max hook RMS.
  - Calculate `pause_emphasis` from measured silence frames (< -38 dB relative peak) in pre-hook window.
  - Calculate `pitch_change` from YIN F0 standard deviation in voiced frames.
  - Calculate `burstiness` from onset spectral flux in high frequencies.
  - Set `laughter = burstiness` as an explicit proxy for schema compatibility:
    `"burstiness": round(burstiness, 3),`
    `"laughter": round(burstiness, 3), # Backward-compatible proxy field; represents acoustic onset burstiness, NOT genuine laughter detection`
  - Generate truthful evidence strings:
    - `"Audio: measured {silence_sec:.1f}s pre-hook silence (< -38dB relative peak)"`
    - `"Audio: measured +{energy_db:+.1f}dB RMS energy surge on hook delivery"`
    - `"Audio: measured F0 dynamic inflection (std {f0_std:.1f} Hz) via YIN contour"`
    - `"Audio: high acoustic onset burstiness ({burstiness:.2f})"`
    - Do NOT assert "laughter detected" or fabricated emotional classifications.
- Update `analyze_acoustic_signals` to iterate candidates and invoke real DSP extraction.

- [ ] **Step 4: Run test to verify it passes**

Run:
```powershell
& "d:\College\Autoshorts 11.0\.venv\Scripts\python.exe" autoshorts/src-tauri/scripts/test_multimodal_acoustic_suite.py
```
Expected: `Ran 2 tests in ...s. OK`

---

### Task 2: Exhaustive CandidateDraft Persistence & Observable Corruption Error Handling (Fixing F6)

**Files:**
- Modify: `autoshorts/src-tauri/src/models.rs`
- Modify: `autoshorts/src-tauri/src/db.rs`

**Interfaces:**
- Consumes: `drafts: &[CandidateDraft]` in `replace_candidates(project_id, drafts)`
- Produces: Persistent SQLite records retaining all 30+ fields with 100% fidelity round-trip recovery, plus an explicit 3-state error model for metadata deserialization.

- [ ] **Step 1: Write failing Rust unit tests for full draft equality and corrupted JSON detection**

In `autoshorts/src-tauri/src/db.rs` (under `mod tests`):
```rust
#[test]
fn test_candidate_draft_full_persistence_roundtrip_all_fields() {
    let db = Database::open(std::path::Path::new(":memory:")).expect("open in-memory db");
    let project = db
        .create_project("/tmp/test.mp4", "local", "preset_viral_bold", "adaptive", Some(60.0))
        .expect("create project");

    // Populate EVERY field on CandidateDraft with distinct non-default values
    let original = CandidateDraft {
        start: 10.5,
        end: 42.0,
        score: 0.955,
        hook: "Full persistence test hook".into(),
        rationale: "Full persistence rationale".into(),
        hook_start: Some(10.5),
        hook_end: Some(14.0),
        payoff_text: Some("Payoff completion line".into()),
        payoff_start: Some(38.0),
        payoff_end: Some(41.8),
        structure: Some("insight_explanation".into()),
        hook_score: Some(0.91),
        coherence_score: Some(0.89),
        payoff_score: Some(0.94),
        question_hook_used: Some(true),
        question_start: Some(10.5),
        question_end: Some(12.0),
        question_text: Some("Why does this happen?".into()),
        question_hook_score: Some(0.88),
        question_relevance_score: Some(0.92),
        answer_start: Some(12.0),
        answer_end: Some(14.0),
        answer_text: Some("Because of this specific reason.".into()),
        answer_strength_score: Some(0.90),
        language: Some("hi".into()),
        script: Some("Devanagari".into()),
        hook_type: Some("question".into()),
        speakers: Some(vec!["Host".into(), "Guest".into()]),
        summary: Some("Detailed summary of the segment.".into()),
        curiosity_score: Some(0.93),
        emotional_impact_score: Some(0.85),
        surprise_score: Some(0.79),
        value_score: Some(0.96),
        story_quality_score: Some(0.87),
        context_completeness_score: Some(0.91),
        shareability_score: Some(0.89),
        hook_speaker: Some("Host".into()),
        context_text: Some("Introductory context".into()),
        conversation_type: Some("interview".into()),
        start_reason: Some("natural_pause".into()),
        end_reason: Some("payoff_completion".into()),
        semantic_complete: Some(true),
        interviewer_hook_strength: Some(0.88),
        guest_hook_strength: Some(0.92),
        answer_completeness: Some(0.94),
        semantic_closure: Some(0.95),
        multimodal_verified: true,
        multimodal_status: Some("SUCCESS".into()),
        fallback_label: Some("none".into()),
        multimodal_score: Some(0.93),
        visual_score: Some(0.87),
        audio_score: Some(0.91),
        temporal_score: Some(0.85),
        semantic_score: Some(0.94),
        visual_scene_change: Some(0.45),
        visual_reaction_strength: Some(0.72),
        visual_expression_change: Some(0.68),
        visual_gesture_strength: Some(0.55),
        visual_framing_change: Some(0.30),
        visual_saliency: Some(0.82),
        audio_energy_change: Some(0.78),
        audio_peak_strength: Some(0.84),
        audio_pitch_change: Some(0.65),
        audio_pause_emphasis: Some(0.82),
        audio_laughter: Some(0.25),
        temporal_escalation: Some(0.70),
        temporal_turning_point: Some(0.60),
        temporal_narrative_progression: Some(0.85),
        temporal_payoff_alignment: Some(0.90),
        multimodal_evidence: Some(vec!["Visual: gesture".into(), "Audio: surge".into()]),
        hook_topic_clarity: Some(0.96),
        hook_curiosity: Some(0.94),
        hook_relevance: Some(0.92),
        hook_contrast: Some(0.88),
        hook_confidence: Some(0.92),
        opening_context_score: Some(0.88),
        payoff_completion: Some(true),
        requires_extension: Some(false),
        extension_direction: Some("none".into()),
        extension_reason: Some("none".into()),
        confidence_bias: Some(0.0),
        endpoint_state: Some(EndpointState::SentenceComplete),
        closure_signals: None,
    };

    let inserted = db.replace_candidates(&project.id, &[original.clone()]).expect("replace candidates");
    assert_eq!(inserted.len(), 1);

    let candidates = db.list_candidates(&project.id).expect("list candidates");
    assert_eq!(candidates.len(), 1);

    let restored = candidates[0].try_metadata_draft().expect("deserialization should succeed");
    
    // Assert 100% complete struct equality
    assert_eq!(restored, original, "Restored CandidateDraft must match original in every single field");
}

#[test]
fn test_candidate_corrupted_metadata_json_returns_explicit_error() {
    let db = Database::open(std::path::Path::new(":memory:")).expect("open in-memory db");
    let project = db
        .create_project("/tmp/test.mp4", "local", "preset_viral_bold", "adaptive", Some(60.0))
        .expect("create project");

    let draft = CandidateDraft {
        start: 0.0,
        end: 10.0,
        score: 0.8,
        hook: "Test".into(),
        rationale: "Test".into(),
        ..Default::default()
    };
    let inserted = db.replace_candidates(&project.id, &[draft]).expect("insert");
    let cand_id = &inserted[0].id;

    // Deliberately corrupt metadata_json directly in SQLite
    let conn = db.conn.lock().unwrap();
    conn.execute("UPDATE candidates SET metadata_json = '{\"corrupted\": true, invalid_json' WHERE id = ?1", params![cand_id]).unwrap();
    drop(conn);

    let candidates = db.list_candidates(&project.id).expect("list");
    let cand = &candidates[0];
    
    // Explicit error check — must NEVER silently fall back to legacy data!
    let res = cand.try_metadata_draft();
    assert!(res.is_err(), "Corrupted metadata_json must return an Err, not fallback silently");
    match res {
        Err(MetadataError::CorruptedJson(msg)) => {
            assert!(msg.contains("corrupted") || msg.contains("invalid"));
        }
        _ => panic!("Expected MetadataError::CorruptedJson"),
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```powershell
cargo test -p autoshorts --lib -- db::tests::test_candidate_draft_full_persistence_roundtrip_all_fields
```
Expected: FAIL because `CandidateDraft` does not derive `PartialEq`, `try_metadata_draft` does not exist, and `metadata_json` column is not yet implemented.

- [ ] **Step 3: Implement CandidateDraft PartialEq and 3-state metadata deserialization**

1. In `autoshorts/src-tauri/src/models.rs`:
- Add `PartialEq` to `#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]` on `CandidateDraft`. (Also ensure nested structs like `EndpointState` and `ClosureSignals` derive `PartialEq`).
- Define explicit error type:
```rust
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum MetadataError {
    #[error("Corrupted candidate metadata JSON: {0}")]
    CorruptedJson(String),
}
```
- Add `metadata_json: Option<String>` to `Candidate`.
- Implement `try_metadata_draft()`:
```rust
impl Candidate {
    pub fn try_metadata_draft(&self) -> Result<CandidateDraft, MetadataError> {
        match &self.metadata_json {
            None => Ok(self.reconstruct_legacy_draft()),
            Some(raw) if raw.trim().is_empty() => Ok(self.reconstruct_legacy_draft()),
            Some(raw) => serde_json::from_str::<CandidateDraft>(raw)
                .map_err(|e| MetadataError::CorruptedJson(format!("{}: {}", e, raw))),
        }
    }

    pub fn reconstruct_legacy_draft(&self) -> CandidateDraft {
        CandidateDraft {
            start: self.start_sec,
            end: self.end_sec,
            score: self.score,
            hook: self.hook.clone(),
            rationale: self.rationale.clone(),
            hook_start: self.hook_start_sec,
            hook_end: self.hook_end_sec,
            hook_confidence: self.hook_confidence,
            opening_context_score: self.opening_context_score,
            payoff_text: self.payoff_text.clone(),
            payoff_start: self.payoff_start_sec,
            payoff_end: self.payoff_end_sec,
            payoff_score: self.payoff_score,
            payoff_completion: self.payoff_completion,
            ..Default::default()
        }
    }
}
```

2. In `autoshorts/src-tauri/src/db.rs`:
- In `init_db`: add `metadata_json TEXT` to `CREATE TABLE IF NOT EXISTS candidates`.
- Add backward-compatible migration with explicit error handling (propagating any non-duplicate-column SQLite failure):
```rust
match conn.execute("ALTER TABLE candidates ADD COLUMN metadata_json TEXT", []) {
    Ok(_) => {}
    Err(e) => {
        let msg = e.to_string().to_lowercase();
        if !msg.contains("duplicate column") {
            return Err(e.into());
        }
    }
}
```
- In `replace_candidates`: serialize the full `draft` to JSON via `serde_json::to_string(draft).ok()`, and insert it as the 19th column.
- In `candidate_from_row`: read `metadata_json` from the row.
- In `get_candidate_with_project`: include `candidates.metadata_json` in the SELECT query.

- [ ] **Step 4: Run test to verify it passes**

Run:
```powershell
cargo test -p autoshorts --lib -- db::tests
```
Expected: PASS with 100% field equality and verified corruption error rejection.

---

### Task 3: Eliminate Hardcoded Paths and Add Silent Crop Fallback Telemetry (Fixing F8)

**Files:**
- Modify: `autoshorts/src-tauri/scripts/speaker_tracker.py:100-120`
- Modify: `autoshorts/src-tauri/src/media.rs:272-315, 1120-1145`
- Create: `autoshorts/src-tauri/scripts/test_path_resolution.py`

**Interfaces:**
- Consumes: Environment variables, relative script roots, runtime executable location.
- Produces: Guaranteed path resolution and explicit distinction between `optimized speaker tracking` and `emergency center crop` fallback.

- [ ] **Step 1: Write test verifying dynamic model discovery and missing tracker telemetry**

1. Create `autoshorts/src-tauri/scripts/test_path_resolution.py`:
```python
import os
import inspect
import unittest
import speaker_tracker
from speaker_tracker import find_yolo_model

class TestPathResolution(unittest.TestCase):
    def test_no_hardcoded_autoshorts_5_paths(self):
        source = inspect.getsource(speaker_tracker.find_yolo_model)
        self.assertNotIn("Autoshorts 5.0", source, "Hardcoded Autoshorts 5.0 path must be removed")

    def test_env_var_override_yolo_model(self):
        os.environ["AUTOSHORTS_YOLO_MODEL"] = "non_existent_yolo_path.pt"
        self.assertIsNone(find_yolo_model())
        del os.environ["AUTOSHORTS_YOLO_MODEL"]

if __name__ == "__main__":
    unittest.main()
```

2. In `autoshorts/src-tauri/src/media.rs` (under `mod tests`):
```rust
#[test]
fn test_detect_speaker_crop_params_missing_tracker_marks_emergency_fallback() {
    // Point tracker to nonexistent path
    std::env::set_var("AUTOSHORTS_SPEAKER_TRACKER_SCRIPT", "/tmp/nonexistent_tracker.py");
    let plan = detect_speaker_crop_params(
        "/tmp/dummy.mp4",
        0.0,
        5.0,
        1920,
        1080,
        608,
        None,
        "original"
    );
    std::env::remove_var("AUTOSHORTS_SPEAKER_TRACKER_SCRIPT");

    // Must return center crop (x=656) BUT must be flagged as emergency fallback
    assert_eq!(plan.x, "656");
    assert!(plan.is_emergency_fallback, "Must explicitly record is_emergency_fallback = true");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```powershell
& "d:\College\Autoshorts 11.0\.venv\Scripts\python.exe" autoshorts/src-tauri/scripts/test_path_resolution.py
cargo test -p autoshorts --lib -- media::tests::test_detect_speaker_crop_params_missing_tracker_marks_emergency_fallback
```
Expected: FAIL because `is_emergency_fallback` field is missing on `SmartFramingPlan` and `Autoshorts 5.0` is still in Python source.

- [ ] **Step 3: Implement clean path resolution and emergency fallback telemetry**

1. In `autoshorts/src-tauri/scripts/speaker_tracker.py`:
- Remove `d:\College\Autoshorts 5.0` paths in `find_yolo_model()`.
- Search dynamically relative to `__file__` and `%LOCALAPPDATA%/autoshorts/models`.

2. In `autoshorts/src-tauri/src/media.rs`:
- Add `#[serde(default)] pub is_emergency_fallback: bool` and `#[serde(default)] pub fallback_reason: Option<String>` to `SmartFramingPlan`.
- In `SmartFramingPlan::single_fallback()`, set `is_emergency_fallback: true` and `fallback_reason: Some(...)`.
- In `detect_speaker_crop_params()`:
  - If `find_speaker_tracker_script()` is `None` or script execution fails:
    ```rust
    eprintln!(
        "[Smart Framing] EMERGENCY_FALLBACK_ACTIVE: Speaker tracker unavailable or failed for {}. Default center crop X: {} applied. Telemetry flagged as emergency fallback.",
        source_path, default_x
    );
    ```
  - Successful parser returns `is_emergency_fallback: false`.

- [ ] **Step 4: Run test to verify it passes**

Run:
```powershell
& "d:\College\Autoshorts 11.0\.venv\Scripts\python.exe" autoshorts/src-tauri/scripts/test_path_resolution.py
cargo test -p autoshorts --lib -- media::tests
```
Expected: PASS with explicit emergency fallback telemetry.

---

### Task 4: Bind FramingConfig to the Production Tracking Run Loop (Fixing F9)

**Files:**
- Modify: `autoshorts/src-tauri/scripts/speaker_tracker.py:180-235, 5025-5140, 5330-5390`
- Create: `autoshorts/src-tauri/scripts/test_framing_config_suite.py`

**Interfaces:**
- Consumes: `FramingConfig` dataclass instance passed to `run()`.
- Produces: Production tracking execution where loop decision variables (`DISP_THRESH`, `SUSTAINED_DUR`, `REFRAME_COOLDOWN`, `SCALE_DEADBAND`) are governed by `FramingConfig`.

- [ ] **Step 1: Write test exercising production tracking loop with custom FramingConfig**

Create `autoshorts/src-tauri/scripts/test_framing_config_suite.py`:
```python
import unittest
from unittest.mock import patch
import numpy as np
from speaker_tracker import FramingConfig, DEFAULT_FRAMING_CONFIG, run, apply_scale_deadband

class TestFramingConfigProductionBinding(unittest.TestCase):
    def test_apply_scale_deadband_honors_config(self):
        scale_def, reason_def = apply_scale_deadband(1.0, 1.10, deadband=0.08)
        self.assertEqual(scale_def, 1.10)
        self.assertEqual(reason_def, "SCALE_UP")

        scale_cust, reason_cust = apply_scale_deadband(1.0, 1.10, deadband=0.15)
        self.assertEqual(scale_cust, 1.0)
        self.assertEqual(reason_cust, "DEADBAND")

    @patch("speaker_tracker.sample_video_and_detect_shots")
    def test_production_run_loop_respects_framing_config(self, mock_sample):
        # Synthetic single shot with a sudden face jump at t=0.5s from x=800 to x=1200
        # Face detection: (fx, fy, fw, fh)
        face_left = {"box": (700, 200, 200, 200), "conf": 0.9, "area": 40000, "landmarks": []}
        face_right = {"box": (1200, 200, 200, 200), "conf": 0.9, "area": 40000, "landmarks": []}
        
        # 10 frames from 0.0 to 1.0s
        samples = []
        for t in np.linspace(0.0, 1.0, 9):
            face = face_left if t < 0.4 else face_right
            samples.append((t, np.zeros((1080, 1920, 3), dtype=np.uint8), [face]))
        
        mock_sample.return_value = ([(0.0, 1.0)], samples)

        # Config A: Very high displacement threshold (ratio 0.8) -> ignores the face movement
        cfg_high_thresh = FramingConfig(displacement_thresh_ratio=0.8, reframe_cooldown=0.1, sustained_displacement_dur=0.1)
        plan_high = run("mock.mp4", 0.0, 1000.0, 608, 1312, 656, config=cfg_high_thresh)

        # Config B: Very sensitive displacement threshold (ratio 0.02) -> triggers reframing
        cfg_sensitive = FramingConfig(displacement_thresh_ratio=0.02, reframe_cooldown=0.1, sustained_displacement_dur=0.1)
        plan_sens = run("mock.mp4", 0.0, 1000.0, 608, 1312, 656, config=cfg_sensitive)

        self.assertNotEqual(plan_high.x, plan_sens.x, "Production run() loop must produce different trajectories when FramingConfig changes")

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```powershell
& "d:\College\Autoshorts 11.0\.venv\Scripts\python.exe" autoshorts/src-tauri/scripts/test_framing_config_suite.py
```
Expected: FAIL because `run()` does not accept `config` and hardcodes `DISP_THRESH`, `SUSTAINED_DUR`, and `REFRAME_COOLDOWN`.

- [ ] **Step 3: Wire FramingConfig into `run()` and eliminate shadowing**

1. In `autoshorts/src-tauri/scripts/speaker_tracker.py`:
- Update `apply_scale_deadband`:
```python
def apply_scale_deadband(
    current_scale: float,
    ideal_scale: float,
    min_scale: float = MIN_SCALE,
    max_scale: float = MAX_SCALE,
    previous_reason: str = "",
    deadband: Optional[float] = None
):
    eff_deadband = deadband if deadband is not None else DEFAULT_FRAMING_CONFIG.scale_deadband
    if abs(current_scale - ideal_scale) < eff_deadband:
        return current_scale, "DEADBAND"
    ...
```
- Update `def run(..., config: Optional[FramingConfig] = None)`:
```python
    cfg = config or DEFAULT_FRAMING_CONFIG
    DISP_THRESH = max(70.0, crop_w_baseline * cfg.displacement_thresh_ratio)
    SUSTAINED_DUR = cfg.sustained_displacement_dur
    REFRAME_COOLDOWN = cfg.reframe_cooldown
    SCALE_DEADBAND = cfg.scale_deadband
```
- Use `cfg` when calling `compute_visual_prominence(..., config=cfg)`, `resolve_visual_subject(..., config=cfg)`, and `solve_framing(..., config=cfg)`.
- Connect dead constants (`FAST_TRANSITION_DUR`, `DRIFT_THRESH_PX`, `SALIENCY_*`) or document them as explicit aliases of `DEFAULT_FRAMING_CONFIG`.

- [ ] **Step 4: Run test to verify it passes**

Run:
```powershell
& "d:\College\Autoshorts 11.0\.venv\Scripts\python.exe" autoshorts/src-tauri/scripts/test_framing_config_suite.py
```
Expected: `Ran 2 tests in ...s. OK`

---

### Task 5: Resolution-Independent Geometry Thresholds with 4K and 9:16 Portrait Tests (Fixing F7)

**Files:**
- Modify: `autoshorts/src-tauri/scripts/speaker_tracker.py:445-455`
- Create: `autoshorts/src-tauri/scripts/test_geometry_scaling_suite.py`

**Interfaces:**
- Consumes: Face bounding box `(fx, fy, fw, fh)` and frame dimensions `(w, h)`.
- Produces: Resolution-invariant back-of-head classification matching 1080p pixel areas exactly on 1080p, while scaling accurately across 720p, 4K, and 1080×1920 portrait without axis confusion.

- [ ] **Step 1: Write test verifying 720p, 1080p, 4K, and 9:16 portrait scaling with boundary cases**

Create `autoshorts/src-tauri/scripts/test_geometry_scaling_suite.py`:
```python
import unittest
from speaker_tracker import is_back_of_head_candidate

class TestGeometryScalingSuite(unittest.TestCase):
    def test_1080p_exact_equivalence_and_boundary(self):
        w, h = 1920, 1080
        # Baseline threshold is 18000 px^2 for bottom foreground (fy > 1080*0.48 = 518.4)
        # Just below threshold: 134 * 134 = 17956
        self.assertFalse(is_back_of_head_candidate(500, 600, 134, 134, w, h, 0.90, 0.5, 0.1))
        # Just above threshold: 135 * 135 = 18225
        self.assertTrue(is_back_of_head_candidate(500, 600, 135, 135, w, h, 0.90, 0.5, 0.1))

    def test_720p_proportional_scaling_and_boundary(self):
        w, h = 1280, 720
        # At 720p, frame area is 921,600. Equiv threshold = 18000 * (921600 / 2073600) = 8000 px^2
        # Just below: 89 * 89 = 7921
        self.assertFalse(is_back_of_head_candidate(300, 400, 89, 89, w, h, 0.90, 0.5, 0.1))
        # Just above: 90 * 90 = 8100
        self.assertTrue(is_back_of_head_candidate(300, 400, 90, 90, w, h, 0.90, 0.5, 0.1))

    def test_4k_proportional_scaling_and_boundary(self):
        w, h = 3840, 2160
        # At 4K, frame area is 8,294,400 (4x of 1080p). Equiv threshold = 18000 * 4 = 72,000 px^2
        # Just below: 268 * 268 = 71824
        self.assertFalse(is_back_of_head_candidate(1000, 1200, 268, 268, w, h, 0.90, 0.5, 0.1))
        # Just above: 269 * 269 = 72361
        self.assertTrue(is_back_of_head_candidate(1000, 1200, 269, 269, w, h, 0.90, 0.5, 0.1))

    def test_portrait_9_16_scaling_and_axis_orientation(self):
        # 1080x1920 (vertical video)
        w, h = 1080, 1920
        # Frame area is 2,073,600 (identical to 1080p landscape). Equiv threshold = 18000 px^2
        # Bottom threshold is fy > h * 0.48 = 1920 * 0.48 = 921.6 px.
        # Test a face at fy=700 (which would be bottom in landscape h=1080, but is UPPER-MIDDLE in portrait h=1920):
        # Even with area > 18000, fy=700 < 921.6, so it must NOT trigger bottom foreground!
        self.assertFalse(is_back_of_head_candidate(300, 700, 150, 150, w, h, 0.90, 0.5, 0.1))
        
        # Test a face at fy=1000 (fy > 921.6, genuinely bottom in portrait) with area > 18000:
        self.assertTrue(is_back_of_head_candidate(300, 1000, 150, 150, w, h, 0.90, 0.5, 0.1))

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```powershell
& "d:\College\Autoshorts 11.0\.venv\Scripts\python.exe" autoshorts/src-tauri/scripts/test_geometry_scaling_suite.py
```
Expected: FAIL because `is_back_of_head_candidate` is not factored and fixed pixel areas fail on 720p, 4K, and portrait.

- [ ] **Step 3: Implement normalized area fractions in `speaker_tracker.py`**

Define normalized area fractions relative to standard 1080p frame area (`1920 * 1080 = 2,073,600`):
```python
# Resolution-normalized face area thresholds (calibrated to 1080p baseline)
REF_FRAME_AREA_1080P   = 1920.0 * 1080.0
FRAC_BOTTOM_FOREGROUND = 18000.0 / REF_FRAME_AREA_1080P   # ~0.00868
FRAC_LOW_FRONT_LARGE   = 14000.0 / REF_FRAME_AREA_1080P   # ~0.00675
FRAC_EXTREME_EDGE_HUGE = 35000.0 / REF_FRAME_AREA_1080P   # ~0.01688

def is_back_of_head_candidate(
    fx: float, fy: float, fw: float, fh: float,
    w: int, h: int,
    conf: float, skin_ratio: float, sym_err: float
) -> bool:
    frame_area = max(1.0, float(w * h))
    face_area = fw * fh
    
    is_bottom_foreground = (fy > h * 0.48) and (face_area > FRAC_BOTTOM_FOREGROUND * frame_area)
    is_low_front_large = (conf < 0.78) and (skin_ratio < 0.35 or sym_err > 0.32) and (face_area > FRAC_LOW_FRONT_LARGE * frame_area)
    is_extreme_edge_huge = (fx < w * 0.15 or fx + fw > w * 0.85) and (face_area > FRAC_EXTREME_EDGE_HUGE * frame_area) and (conf < 0.80)

    return bool(is_bottom_foreground or is_low_front_large or is_extreme_edge_huge)
```
Update line 448 in `speaker_tracker.py` to use `is_back_of_head_candidate(fx, fy, fw, fh, w, h, conf, skin_ratio, sym_err)`.

- [ ] **Step 4: Run test to verify it passes**

Run:
```powershell
& "d:\College\Autoshorts 11.0\.venv\Scripts\python.exe" autoshorts/src-tauri/scripts/test_geometry_scaling_suite.py
```
Expected: `Ran 4 tests in ...s. OK`

---

### Task 6: Full Verification, Complete 16+ Suite Regression, and Real-Media Render Validation

**Files:**
- Touch: Entire test surface & representative clips (`clip-01_flat.mp4`, `clip-02_flat.mp4`, `clip-04_flat.mp4`).

- [ ] **Step 1: Run complete Rust test suite**

Run:
```powershell
cargo test -p autoshorts --lib -- --test-threads=1
```
Expected: All 282+ tests passing.

- [ ] **Step 2: Run all 16 existing Python test suites + 4 new Phase 0 suites**

Enumerate and run the full inventory of 20 verified test suites:
1. `test_applied_features_suite.py`
2. `test_audio_intelligence_suite.py`
3. `test_caption_qa_suite.py`
4. `test_hook_closure_suite.py`
5. `test_hook_ending_optimization_suite.py`
6. `test_smart_pacing_2_suite.py`
7. `test_smart_pacing_suite.py`
8. `autoshorts/src-tauri/scripts/test_active_speaker_suite.py`
9. `autoshorts/src-tauri/scripts/test_caption_intelligence_suite.py`
10. `autoshorts/src-tauri/scripts/test_caption_templates_suite.py`
11. `autoshorts/src-tauri/scripts/test_dualframe_real_render.py`
12. `autoshorts/src-tauri/scripts/test_dualframe_suite.py`
13. `autoshorts/src-tauri/scripts/test_framing_regression_suite.py`
14. `autoshorts/src-tauri/scripts/test_multimodal_hook_suite.py`
15. `autoshorts/src-tauri/scripts/test_sampling_memory_suite.py`
16. `autoshorts/src-tauri/scripts/test_ultralytics_adapter_suite.py`
17. *(New)* `autoshorts/src-tauri/scripts/test_multimodal_acoustic_suite.py`
18. *(New)* `autoshorts/src-tauri/scripts/test_path_resolution.py`
19. *(New)* `autoshorts/src-tauri/scripts/test_framing_config_suite.py`
20. *(New)* `autoshorts/src-tauri/scripts/test_geometry_scaling_suite.py`

Run:
```powershell
& "d:\College\Autoshorts 11.0\.venv\Scripts\python.exe" -m unittest `
    test_applied_features_suite.py `
    test_audio_intelligence_suite.py `
    test_caption_qa_suite.py `
    test_hook_closure_suite.py `
    test_hook_ending_optimization_suite.py `
    test_smart_pacing_2_suite.py `
    test_smart_pacing_suite.py `
    autoshorts/src-tauri/scripts/test_active_speaker_suite.py `
    autoshorts/src-tauri/scripts/test_caption_intelligence_suite.py `
    autoshorts/src-tauri/scripts/test_caption_templates_suite.py `
    autoshorts/src-tauri/scripts/test_dualframe_suite.py `
    autoshorts/src-tauri/scripts/test_framing_regression_suite.py `
    autoshorts/src-tauri/scripts/test_multimodal_hook_suite.py `
    autoshorts/src-tauri/scripts/test_sampling_memory_suite.py `
    autoshorts/src-tauri/scripts/test_ultralytics_adapter_suite.py `
    autoshorts/src-tauri/scripts/test_multimodal_acoustic_suite.py `
    autoshorts/src-tauri/scripts/test_path_resolution.py `
    autoshorts/src-tauri/scripts/test_framing_config_suite.py `
    autoshorts/src-tauri/scripts/test_geometry_scaling_suite.py
```
Expected: Discovered: 20 suites. Executed: 20 suites. Passed: 20. Failed: 0. Skipped: 0.

- [ ] **Step 3: Run frontend production build**

Run:
```powershell
cd autoshorts; npm run build
```
Expected: Exit code 0, 0 TypeScript errors.

- [ ] **Step 4: Real-Media Dry-Run Render Regression Validation**

Execute dry-run render tests across representative real clips in `d:\College\Autoshorts 11.0`:
1. **Adaptive Framing Two-Person Invariant on `clip-02_flat.mp4`**:
   - Run framing solver with `AUTOSHORTS_FRAMING_MODE=adaptive`.
   - Confirm stderr outputs `[Framing] adaptive=on dualframe_disabled=true`.
   - Verify returned `SmartFramingPlan` mode is `single` (square 1080×1080 inside 9:16).
   - Verify both subjects are retained in the shared composition with zero camera jumping/follow and zero DualFrame split.
   - Verify T7 caption layout is positioned face-safe below both subjects with 0.00% face overlap.
2. **Original 9:16 Invariants on `clip-01_flat.mp4`**:
   - Confirm normal 9:16 crop geometry.
   - Confirm payoff endpoint matches transcript words without duration-budget truncation.
   - Confirm Smart Pacing preserves timeline monotonically.
3. **Telemetry & Path Resolution**:
   - Verify stdout/stderr logs `[Smart Framing] dynamic crop trajectory ... (mode: single/multi-speaker)`.
   - Verify `is_emergency_fallback` is FALSE for valid runs and TRUE only when deliberately pointing to missing trackers.

- [ ] **Step 5: Authoritative Project Brain Update**

Update `AUTOSHORTS_11_0_BRAIN.md` in `d:\College\Autoshorts 11.0`:
- Record Phase 0 completion.
- Document exact DSP acoustic feature metrics and burstiness naming.
- Document full `CandidateDraft` SQLite roundtrip persistence.
- Document resolution-invariant geometry scaling and authoritative `FramingConfig` run loop binding.
- Unblock Phase 1 implementation.
