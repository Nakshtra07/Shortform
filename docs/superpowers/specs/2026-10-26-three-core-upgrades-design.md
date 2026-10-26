# Specification: Three Core Architectural Upgrades

**Date:** 2026-10-26  
**Status:** Approved  
**Author:** AutoShorts Engineering Team  
**Path:** `docs/superpowers/specs/2026-10-26-three-core-upgrades-design.md`  

---

## 1. Executive Summary

This specification defines the architecture, data structures, algorithms, and verification plan for three core architectural upgrades in AutoShorts 11.0:

1. **Upgrade 1: Deepgram Diarization & Zero-Redundancy Speaker Turn Propagation** — Eliminate duplicate network audio uploads to Deepgram by directly deriving word-level and turn-level `SpeakerDiarizationResult` in Rust from the primary transcription output, while preserving full fallback capability for offline or non-diarized transcripts.
2. **Upgrade 2: Resolution-Independent Framing in `speaker_tracker.py`** — Normalize all remaining fixed pixel thresholds (matching distances, area bounds, speaker jump deltas, pan limits, and centering deadbands) using 1080p frame diagonal scaling (`hypot(w, h) / 2202.9`) and frame area ratios, ensuring identical tracking behavior across 4K UHD, 1080p, 720p, and non-16:9 aspect ratios.
3. **Upgrade 3: AGPL-3.0 License Decoupling & First-Run Weights Downloader** — Decouple distributed application binaries from AGPL-3.0 copyleft obligations by excluding `yolo11n.pt` and `yolo11n-seg.pt` from installer bundles, shipping only Apache-2.0 OpenCV YuNet, and providing a robust, atomic Rust downloader into `%LOCALAPPDATA%/autoshorts/models/` with seamless YuNet fallback.

---

## 2. Upgrade 1: Deepgram Diarization & Zero-Redundancy Turn Propagation

### 2.1 Problem Statement
When AutoShorts transcribes audio via Deepgram Nova-3 in `transcription.rs`, the endpoint URL already includes `diarize=true`. The response contains word-level speaker tags (`speaker: 0`, `speaker: 1`, etc.), which `normalize_deepgram()` maps into `TranscriptWord.speaker: Some("S1")` and `Some("S2")`.

However, downstream in `speaker_intelligence.rs`, `process_source_video()` executes `speaker_diarization.py`. If a Deepgram API key is configured, this sidecar extracts the audio and uploads the file to Deepgram a **second time**. This results in:
* 5–30 seconds of wasted processing latency.
* Double Deepgram billing usage per video.
* Redundant network failure risks.

### 2.2 Rust Inline Diarization Extractor
In `autoshorts/src-tauri/src/speaker_intelligence.rs`, a new constructor function will parse `transcript_words`:

```rust
pub fn build_diarization_from_transcript_words(
    words: &[TranscriptWord],
    model_label: &str,
) -> Option<SpeakerDiarizationResult>
```

#### Algorithm:
1. **Diarization Detection:** Scan `words`. If `words.iter().all(|w| w.speaker.is_none())`, return `None` (triggers fallback).
2. **Segment Aggregation:** Iterate sequentially through `words`:
   * Group consecutive words sharing the same `speaker` into a `DiarizedSegment`.
   * If a gap between words of the same speaker is `<= 0.45` seconds, keep them within the same segment.
   * If the speaker changes or gap exceeds `0.45` seconds, close the current segment and start a new one.
   * Assign confidence `0.95` (Deepgram Nova-3 word confidence default).
3. **Speaker Statistics Calculation:**
   * Collect unique speaker IDs (e.g. `S1`, `S2`).
   * For each speaker, sum `total_speech_sec = sum(segment.end - segment.start)`.
   * Record `segment_count` and compute average confidence.
4. **Assembly:** Construct `SpeakerDiarizationResult`:
   * `model`: `model_label.to_string()` (e.g. `"deepgram-nova3-transcription"`).
   * `version`: `"inline-v1".to_string()`.
   * `speakers`: aggregated `Vec<DiarizedSpeaker>`.
   * `segments`: contiguous `Vec<DiarizedSegment>`.
   * `overall_confidence`: `0.95`.

### 2.3 Integration in `process_source_video`
In `SpeakerIntelligenceEngine::process_source_video()`:
```rust
let diarization = if let Some(inline_diar) = Self::build_diarization_from_transcript_words(
    transcript_words,
    "deepgram-nova3-transcription",
) {
    println!(
        "[SpeakerIntel] Zero-redundancy diarization extracted from transcript ({} speakers, {} segments)",
        inline_diar.speakers.len(),
        inline_diar.segments.len()
    );
    inline_diar
} else {
    self.run_diarization(source_path, &source_hash)?
};
```

If `transcript_words` contains no speaker tags (such as local Whisper), `build_diarization_from_transcript_words` returns `None`, and the engine proceeds cleanly to `run_diarization()` via the Python sidecar.

---

## 3. Upgrade 2: Resolution-Independent Framing in `speaker_tracker.py`

### 3.1 Problem Statement
In `autoshorts/src-tauri/scripts/speaker_tracker.py`, visual subject tracking and smart framing thresholds were calibrated for 1080p (`1920×1080`). While OTS back-of-head thresholds (`FRAC_BOTTOM_FOREGROUND`, etc.) were normalized to area fractions, several core tracking and framing algorithms still use hardcoded pixel thresholds:
* Line 3462: `fa < 1800.0` (distant face area rejection).
* Lines 2328 & 2336: `dist_thresh=180.0` and `dist_thresh=140.0` (YuNet track-to-detection matching distances).
* Line 2599: `hypot(tr["cx"] - fcx, tr["cy"] - fcy) < 150.0` (shot re-association distance).
* Line 3253: `abs(target_cx - last_confirmed_cx_by_spk[spk]) > 250.0` (speaker horizontal center jump).
* Line 4966: `_deadband = max(17.5, eff_crop_w * 0.045)` (minimum crop deadband).
* Line 685: `AUTFLIP_MAX_PAN_SPEED = 720.0` (pan speed in px/s).

On 4K footage (`3840×2160`), 180px is only ~4.7% of width, causing tracking tracks to break under normal motion. On 720p footage (`1280×720`), 1800px² rejects valid faces.

### 3.2 Diagonal Normalization Model
A new `ResolutionScaler` is implemented in `speaker_tracker.py`:

```python
REF_FRAME_W = 1920.0
REF_FRAME_H = 1080.0
REF_DIAGONAL_1080P = math.hypot(REF_FRAME_W, REF_FRAME_H)  # ≈ 2202.907 px
REF_AREA_1080P = REF_FRAME_W * REF_FRAME_H                 # 2,073,600 px²

class ResolutionScaler:
    def __init__(self, w: float, h: float):
        self.w = max(1.0, float(w))
        self.h = max(1.0, float(h))
        self.diag = math.hypot(self.w, self.h)
        self.area = self.w * self.h
        self.scale_diag = self.diag / REF_DIAGONAL_1080P
        self.scale_area = self.area / REF_AREA_1080P

    def dist(self, px_1080p: float) -> float:
        """Scales linear Euclidean distances based on frame diagonal."""
        return px_1080p * self.scale_diag

    def area_val(self, area_1080p: float) -> float:
        """Scales 2D pixel area thresholds based on total frame area."""
        return area_1080p * self.scale_area
```

### 3.3 Target Threshold Calibrations
| Metric | 1080p Baseline | 4K UHD (`3840×2160`) | 720p HD (`1280×720`) | Normalization Call |
|---|:---:|:---:|:---:|---|
| **Stage 1 Match Dist** | `180.0 px` | `360.0 px` | `120.0 px` | `scaler.dist(180.0)` |
| **Stage 2 Match Dist** | `140.0 px` | `280.0 px` | `93.3 px` | `scaler.dist(140.0)` |
| **Shot Reconnect Dist** | `150.0 px` | `300.0 px` | `100.0 px` | `scaler.dist(150.0)` |
| **Speaker Jump Delta** | `250.0 px` | `500.0 px` | `166.7 px` | `scaler.dist(250.0)` |
| **Min Deadband** | `17.5 px` | `35.0 px` | `11.7 px` | `scaler.dist(17.5)` |
| **Max Pan Speed** | `720.0 px/s` | `1440.0 px/s` | `480.0 px/s` | `scaler.dist(720.0)` |
| **Min Face Area** | `1800.0 px²` | `7200.0 px²` | `800.0 px²` | `scaler.area_val(1800.0)` |

### 3.4 Numerical Equivalence Guarantee
At `1920×1080`:
`self.scale_diag == 1.0` and `self.scale_area == 1.0`. All threshold outputs are identical to the baseline constants. Zero regression on all existing 1080p tests.

---

## 4. Upgrade 3: AGPL-3.0 License Decoupling & YOLO11 Weights Downloader

### 4.1 Problem Statement
`yolo11n.pt` and `yolo11n-seg.pt` are licensed under GNU AGPL-3.0 by Ultralytics LLC. Bundling these weights inside the Tauri desktop installer creates copyleft exposure. OpenCV YuNet (`face_detection_yunet_2023mar.onnx`), by contrast, is licensed under Apache-2.0.

### 4.2 Decoupled Packaging & Storage Architecture
1. **Tauri Bundler:** The application installer bundles only Apache-2.0 and MIT assets (`face_detection_yunet_2023mar.onnx`). No `.pt` files are included in the distributed installer.
2. **Local Storage Location:**
   * Windows: `%LOCALAPPDATA%/autoshorts/models/yolo11n.pt`
   * macOS / Linux: `~/.local/share/autoshorts/models/yolo11n.pt`
3. **Rust Model Manager (`model_manager.rs`):**
   * Upstream URL: `https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt`
   * Method: `pub async fn ensure_yolo_weights(app: Option<&tauri::AppHandle>) -> Result<PathBuf>`
   * Download flow:
     1. Check if destination file exists and has size > 1,000,000 bytes. If valid, return path immediately.
     2. Stream download chunks to `yolo11n.pt.part`.
     3. Emit `model_download_progress` event to Tauri frontend.
     4. On complete download, atomically rename `yolo11n.pt.part` to `yolo11n.pt`.
     5. If download fails (network offline or proxy error), log warning and return error.
4. **Resilient Python Fallback:**
   In `speaker_tracker.py`, `find_yolo_model()` already checks `%LOCALAPPDATA%/autoshorts/models/yolo11n.pt`. If missing or invalid, it returns `None`, and the tracker automatically runs `NativeTracker` (YuNet + ByteTrack). AutoShorts executes with zero crashes even if offline.

---

## 5. Verification Plan

### 5.1 Rust Tests
* `speaker_intelligence.rs`:
  * `test_build_diarization_from_transcript_words_multi_speaker`: Verify segments, speech duration, and speaker count.
  * `test_build_diarization_from_transcript_words_single_speaker`: Verify 1 speaker handling.
  * `test_build_diarization_from_transcript_words_empty_returns_none`: Verify safe fallback trigger.
* `model_manager.rs`:
  * `test_get_models_dir`: Verify path structure.
  * `test_is_yolo_installed_missing`: Verify detection of absent model.

### 5.2 Python Tests
* `test_resolution_scaling_suite.py`:
  * Test `ResolutionScaler` at `1920x1080` (scale = 1.0).
  * Test `ResolutionScaler` at `3840x2160` (scale = 2.0).
  * Test `ResolutionScaler` at `1280x720` (scale = 2/3).
  * Verify `min_face_area` and distance thresholds match expectations.
* `test_autoclip_adapter.py`: Run all 54 existing tests to confirm 100% green status.
* `test_botsort_regression_suite.py`: Verify framing pipeline integrity.

### 5.3 System Tests
* `cargo check --manifest-path autoshorts/src-tauri/Cargo.toml`
* `cargo test --manifest-path autoshorts/src-tauri/Cargo.toml --lib -j 2`
* `npm --prefix autoshorts run build`
