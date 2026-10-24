# GPU-Accelerated Scene Intelligence & UI Status Indicator Specification

## 1. Overview & Goals
AutoShorts Phase 3 currently executes full-video scene detection via PySceneDetect on the CPU. On long (1.5+ hour) videos, this CPU-bound process takes 50–70 minutes, which stalls pipeline throughput and leaves users uncertain about background progress.

This specification details:
1. **GPU Acceleration Engine:** Accelerating full-video scene detection via **PyTorch CUDA batch tensor processing** in `scene_intelligence.py`, dropping scan times from ~60 minutes on CPU to **~2 minutes on GPU**.
2. **Hardware Ingestion Prioritization:** Updating `youtube.rs` to prioritize **VP9 and H.264** hardware-decodable streams over AV1 for the NVIDIA GeForce GTX 1650 (Turing TU117, 4 GB VRAM).
3. **VRAM Concurrency Protection:** Bounding VRAM allocation strictly under 400 MB to avoid contention with face detection and Re-ID models.
4. **Tauri IPC & Real-time Progress:** Adding Tauri event streaming (`scene-intel-status`) and a persistent state query command (`get_scene_intelligence_status`).
5. **Frontend Indicator:** Displaying a compact, non-intrusive status pill directly beside the "Find Viral Moments" button in the candidate controls bar.

---

## 2. Core Architecture & GPU Engine

### 2.1 PyTorch CUDA Adaptive Detector (`scene_intelligence.py`)
* **Sampling & Batching:**
  * Video frames are sampled and downscaled to 360p (640x360).
  * Frames are loaded into pinned CPU memory and transferred to CUDA in fixed batches of 128 frames as `torch.uint8` tensors.
* **Algorithmic Parity with AdaptiveDetector:**
  * For each frame pair $(F_{t}, F_{t-1})$, compute the L1 luminance/channel difference.
  * A 1D sliding convolution computes a rolling 2-frame window mean ($W=2$) to establish adaptive background lighting thresholds.
  * A scene cut is recorded when the frame delta exceeds `adaptive_threshold * rolling_mean` (with minimum debounce distance of 12 frames), exactly matching PySceneDetect's `AdaptiveDetector` logic.
* **Schema Contract:**
  The output schema strictly preserves the existing contract consumed by `speaker_tracker.py` and Rust:
  ```json
  {
    "sourceHash": "<sha256>",
    "detector": "PyTorch-CUDA-AdaptiveDetector",
    "detectorVersion": "1.0.0",
    "configVersion": "gpu-cuda-v1",
    "fallback": false,
    "scenes": [
      { "sceneId": 0, "start": 0.0, "end": 4.12 },
      { "sceneId": 1, "start": 4.12, "end": 12.85 }
    ],
    "elapsedSec": 118.4,
    "createdAt": "2026-10-24T22:50:00Z"
  }
  ```
* **VRAM Safety:**
  * 128 frames at 640x360x3 in uint8 consume ~88 MB in GPU memory.
  * Explicit `torch.cuda.empty_cache()` calls run after each batch.
  * Total peak VRAM footprint is capped under **350 MB**, leaving >3.5 GB free for YuNet face tracking and Re-ID.
* **Fail-Soft Contract:**
  * If CUDA is unavailable or decoding encounters corrupt packets, the engine falls back to CPU detection or probe-based single-scene fallback without throwing uncaught exceptions.

---

## 3. Hardware Ingestion: `youtube.rs`
* **GTX 1650 NVDEC Constraints:**
  * Turing TU117 supports hardware decoding for **H.264 (AVC), H.265 (HEVC), and VP9**.
  * It lacks AV1 hardware decoding.
* **Format Selector Update:**
  * In `autoshorts/src-tauri/src/youtube.rs`, update `build_probe_args` to prefer hardware-accelerated codecs:
    ```bash
    --format-sort "res,fps,+codec:vp9,+codec:avc,size,br"
    ```
  * Ensures YouTube imports are delivered in streams that NVDEC can process at 800–1200+ FPS.

---

## 4. Backend State & Tauri IPC

### 4.1 State Structure
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SceneIntelStatus {
    pub project_id: String,
    pub status: String, // "idle" | "running" | "completed" | "error"
    pub progress: u8,   // 0 to 100
    pub scene_count: usize,
}
```

### 4.2 State Registry & Lifecycle
* A thread-safe global registry `Arc<Mutex<HashMap<String, SceneIntelStatus>>>` tracks project progress.
* When `transcribe_project` initiates background pre-warming:
  * Emits event `scene-intel-status` with `status: "running", progress: 0`.
  * Intercepts `[SceneIntel] PROGRESS <pct>` from the Python sidecar's stderr and emits updated percentages.
  * On completion, writes `scenes_<hash>.json` and emits `status: "completed", progress: 100, scene_count: N`.
  * On error or timeout, writes the fallback document and emits `status: "error"`.
* Exposes command `get_scene_intelligence_status(project_id: String) -> Result<SceneIntelStatus, String>` to instantly hydrate the UI on mount or tab navigation.

---

## 5. Frontend Indicator UI

### 5.1 Placement
Directly inside `.candidate-controls-right` in `autoshorts/src/main.tsx`, immediately to the left of the "Find Viral Moments" button:

```tsx
<div className="candidate-controls-right">
  {sceneIntelStatus && sceneIntelStatus.status !== "idle" && (
    <div 
      className={`scene-intel-pill status-${sceneIntelStatus.status}`}
      title={getSceneIntelTooltip(sceneIntelStatus)}
    >
      {sceneIntelStatus.status === "running" && (
        <>
          <Loader2 className="spin" size={13} />
          <span>Scene Intel {sceneIntelStatus.progress > 0 ? `${sceneIntelStatus.progress}%` : "Scanning..."}</span>
        </>
      )}
      {sceneIntelStatus.status === "completed" && (
        <>
          <Check size={13} className="text-emerald-400" />
          <span>Scene Intel Ready ({sceneIntelStatus.sceneCount})</span>
        </>
      )}
      {sceneIntelStatus.status === "error" && (
        <>
          <AlertCircle size={13} className="text-amber-400" />
          <span>Scene Intel (Fallback)</span>
        </>
      )}
    </div>
  )}
  <select value={discoveryMode} ... />
  <button className="find-moments-btn" ... />
</div>
```

### 5.2 Visual Styling (`styles.css`)
* Compact pill with rounded corners (`border-radius: 9999px` or `6px`).
* Height matching standard controls (32px).
* Background: dark slate / dark zinc with subtle border.
* Running: blue/cyan pulsing border with spinning icon.
* Completed: emerald green border with checkmark icon.

---

## 6. Testing & TDD Strategy

### 6.1 Test Suite Plan
1. **Rust Tests (`scene_intelligence.rs` & `youtube.rs`):**
   * `test_build_probe_args_prefers_hardware_codecs`: Asserts `--format-sort` contains `+codec:vp9,+codec:avc`.
   * `test_get_scene_intelligence_status_idle`: Asserts unanalyzed project returns `status: "idle"`.
   * `test_get_scene_intelligence_status_cached`: Asserts cached project returns `status: "completed"`.
   * `test_scene_timeout_for_gpu_bound`: Asserts watchdog timeout ceiling is calibrated to 600 seconds.
2. **Python Tests (`test_scene_intelligence_gpu.py`):**
   * `test_pytorch_cuda_detector_detects_synthetic_cuts`: Validates 3 synthetic cut transitions within 100ms tolerance.
   * `test_pytorch_cuda_failsoft_fallback`: Validates fallback output format on invalid files.
   * `test_vram_budget_cleanup`: Asserts memory cleanup after batch processing.
3. **Frontend Verification:**
   * `npm --prefix autoshorts run build` passes with 0 TypeScript compilation errors.
