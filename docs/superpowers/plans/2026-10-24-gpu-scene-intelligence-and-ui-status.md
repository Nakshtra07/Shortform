# GPU-Accelerated Scene Intelligence & UI Status Indicator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Accelerate AutoShorts Full Scene Intelligence using PyTorch CUDA batch tensor processing on the NVIDIA GeForce GTX 1650, prioritize hardware-decodable VP9/H.264 streams in YouTube ingestion, stream real-time progress events over Tauri IPC, and display a compact inline status pill in the controls bar on the video page.

**Architecture:** 
1. `scene_intelligence.py` implements a PyTorch CUDA matrix difference and rolling convolution detector with real-time stderr progress reporting and memory bounded under 400 MB.
2. `youtube.rs` updates format sorting to prioritize `+codec:vp9,+codec:avc` so downloaded videos trigger hardware NVDEC.
3. `scene_intelligence.rs` and `lib.rs` manage a thread-safe project status registry, calibrate timeout bounds to 600s, stream Tauri events (`scene-intel-status`), and expose `get_scene_intelligence_status`.
4. `main.tsx` and `styles.css` render an inline status pill next to "Find Viral Moments" displaying running (spinner + %), completed (checkmark + count), or fallback state.

**Tech Stack:** Rust (Tauri 2.x), Python 3.12 (PyTorch 2.5.1 + CUDA 12.1, OpenCV), React 18, TypeScript, Tailwind/Vanilla CSS.

## Global Constraints
- Target GPU: NVIDIA GeForce GTX 1650 (Turing TU117, 4.0 GB VRAM).
- Total scene intelligence VRAM footprint must remain < 400 MB.
- Downstream contract in `speaker_tracker.py` remains authoritative; scene boundaries are advisory metadata.
- JSON output schema strictly maintained: `{ sourceHash, detector, detectorVersion, configVersion, fallback, scenes: [{ sceneId, start, end }], elapsedSec, createdAt }`.
- All tests must pass with zero billable API calls.
- On Windows, always pass `-j 2` to `cargo test`.

---

### Task 1: Hardware Codec Sorting in YouTube Ingestion (Rust)

**Files:**
- Modify: `autoshorts/src-tauri/src/youtube.rs:59-75`
- Test: `autoshorts/src-tauri/src/youtube.rs:1050-1070`

**Interfaces:**
- `build_probe_args(url: &str, auth: &Option<YouTubeAuth>) -> Vec<String>`: returns yt-dlp arguments including `--format-sort`.

- [ ] **Step 1: Write the failing unit test**

In `autoshorts/src-tauri/src/youtube.rs`, append to the `tests` module:

```rust
    #[test]
    fn test_build_probe_args_prefers_hardware_codecs() {
        let args = build_probe_args("https://www.youtube.com/watch?v=MNCHAAupzBo", &None);
        let sort_arg_idx = args.iter().position(|a| a == "--format-sort")
            .expect("--format-sort must be present");
        let sort_val = &args[sort_arg_idx + 1];
        assert!(
            sort_val.contains("+codec:vp9") && sort_val.contains("+codec:avc"),
            "format-sort must prioritize hardware decodable codecs (vp9, avc) for Turing NVDEC: got {}",
            sort_val
        );
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```powershell
cargo test --manifest-path autoshorts/src-tauri/Cargo.toml -p autoshorts --lib -j 2 -- youtube::tests::test_build_probe_args_prefers_hardware_codecs
```
Expected: FAIL with `format-sort must prioritize hardware decodable codecs`.

- [ ] **Step 3: Update `build_probe_args` implementation**

In `autoshorts/src-tauri/src/youtube.rs`, locate `build_probe_args`:
Replace:
```rust
    // Deterministic format-sort: must exactly match download_video
    args.push("--format-sort".to_string());
    args.push("res,fps,codec,size,br".to_string());
```
With:
```rust
    // Deterministic format-sort: prioritize hardware-decodable VP9 and H.264 (AVC)
    // for NVIDIA Turing NVDEC (GTX 1650 lacks AV1 ASIC decode).
    args.push("--format-sort".to_string());
    args.push("res,fps,+codec:vp9,+codec:avc,size,br".to_string());
```

And in `download_video` in `autoshorts/src-tauri/src/youtube.rs`, update the format-sort argument identically:
```rust
    args.push("--format-sort".to_string());
    args.push("res,fps,+codec:vp9,+codec:avc,size,br".to_string());
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```powershell
cargo test --manifest-path autoshorts/src-tauri/Cargo.toml -p autoshorts --lib -j 2 -- youtube::tests::test_build_probe_args_prefers_hardware_codecs
```
Expected: PASS.

- [ ] **Step 5: Run existing youtube tests to guarantee no regression**

Run:
```powershell
cargo test --manifest-path autoshorts/src-tauri/Cargo.toml -p autoshorts --lib -j 2 -- youtube::tests
```
Expected: all 40 tests passed.

- [ ] **Step 6: Commit**

```bash
git add autoshorts/src-tauri/src/youtube.rs
git commit -m "feat(youtube): prioritize VP9 and AVC hardware codecs for GPU NVDEC"
```

---

### Task 2: PyTorch CUDA Adaptive Scene Detection Engine (Python)

**Files:**
- Create: `autoshorts/src-tauri/scripts/test_scene_intelligence_gpu.py`
- Modify: `autoshorts/src-tauri/scripts/scene_intelligence.py:1-197`

**Interfaces:**
- Produces: `detect_scenes_gpu(source_path: str) -> Optional[List[Dict]]`
- Stderr protocol: emits `[SceneIntel] PROGRESS <0-100>%` periodically.
- Output JSON contract:
  ```json
  {
    "sourceHash": str,
    "detector": "PyTorch-CUDA-AdaptiveDetector",
    "detectorVersion": "1.0.0",
    "configVersion": "gpu-cuda-v1",
    "fallback": bool,
    "scenes": [{"sceneId": int, "start": float, "end": float}],
    "elapsedSec": float,
    "createdAt": str
  }
  ```

- [ ] **Step 1: Write the failing Python unit tests**

Create `autoshorts/src-tauri/scripts/test_scene_intelligence_gpu.py`:

```python
import os
import sys
import unittest
import torch
import numpy as np

# Add scripts directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scene_intelligence

class TestSceneIntelligenceGPU(unittest.TestCase):
    def test_cuda_available(self):
        self.assertTrue(torch.cuda.is_available(), "CUDA must be available for GPU scene intelligence")

    def test_pytorch_adaptive_detector_detects_cuts(self):
        # Create a synthetic sequence of 150 frames (30fps = 5.0s)
        # Shot 1: Frames 0..49 (black, 0)
        # Shot 2: Frames 50..99 (white, 255) -> Cut at frame 50 (1.667s)
        # Shot 3: Frames 100..149 (gray, 128) -> Cut at frame 100 (3.333s)
        frames = []
        for i in range(150):
            if i < 50:
                val = 10
            elif i < 100:
                val = 240
            else:
                val = 120
            frame = np.full((180, 320, 3), val, dtype=np.uint8)
            frames.append(frame)

        cuts = scene_intelligence.detect_cuts_from_frames_pytorch(frames, fps=30.0)
        # Should detect cuts near 1.667s and 3.333s
        self.assertGreaterEqual(len(cuts), 2)
        cut1, cut2 = cuts[0], cuts[1]
        self.assertAlmostEqual(cut1, 50 / 30.0, delta=0.2)
        self.assertAlmostEqual(cut2, 100 / 30.0, delta=0.2)

    def test_failsoft_fallback_on_invalid_file(self):
        doc = scene_intelligence.run_cached("invalid_file_nonexistent_12345.mp4", None)
        self.assertTrue(doc.get("fallback"), "Missing file must produce fallback=True")
        self.assertGreaterEqual(len(doc.get("scenes", [])), 1)

    def test_memory_cleanup(self):
        if not torch.cuda.is_available():
            return
        torch.cuda.empty_cache()
        baseline = torch.cuda.memory_allocated()
        # Run a small batch
        frames = [np.full((180, 320, 3), 128, dtype=np.uint8) for _ in range(64)]
        _ = scene_intelligence.detect_cuts_from_frames_pytorch(frames, fps=30.0)
        torch.cuda.empty_cache()
        after = torch.cuda.memory_allocated()
        self.assertLessEqual(after - baseline, 5 * 1024 * 1024, "VRAM must be cleaned up to <5MB delta")

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```powershell
.\.venv\Scripts\python.exe autoshorts/src-tauri/scripts/test_scene_intelligence_gpu.py
```
Expected: FAIL with `AttributeError: module 'scene_intelligence' has no attribute 'detect_cuts_from_frames_pytorch'`.

- [ ] **Step 3: Implement GPU Adaptive Detector in `scene_intelligence.py`**

In `autoshorts/src-tauri/scripts/scene_intelligence.py`, update:
1. Version identifiers:
   ```python
   SCENE_DETECTOR_NAME = "PyTorch-CUDA-AdaptiveDetector"
   SCENE_DETECTOR_VERSION = "1.0.0"
   SCENE_CONFIG_VERSION = "gpu-cuda-v1"
   ```
2. Implement `detect_cuts_from_frames_pytorch`:
   ```python
   def detect_cuts_from_frames_pytorch(
       frames: List[np.ndarray],
       fps: float = 30.0,
       adaptive_threshold: float = ADAPTIVE_THRESHOLD,
       min_scene_len_frames: int = MIN_SCENE_LEN_FRAMES,
       min_content_val: float = 15.0,
   ) -> List[float]:
       """Calculate cut timestamps using PyTorch CUDA tensor operations with rolling mean."""
       if len(frames) < 2:
           return []

       device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
       # Stack frames into tensor: (N, H, W, C) -> (N, C, H, W) float32
       try:
           # Convert to tensor in batches to avoid CPU memory spikes
           tensor_frames = torch.from_numpy(np.stack(frames)).to(device=device, dtype=torch.float32)
           # Convert to grayscale luminance: 0.299 R + 0.587 G + 0.114 B
           luma = 0.299 * tensor_frames[:, :, :, 2] + 0.587 * tensor_frames[:, :, :, 1] + 0.114 * tensor_frames[:, :, :, 0]
           # Frame-to-frame absolute differences: (N-1,)
           diffs = torch.abs(luma[1:] - luma[:-1]).mean(dim=(1, 2))  # mean over H, W

           diffs_cpu = diffs.cpu().numpy()
           del tensor_frames, luma, diffs
           if torch.cuda.is_available():
               torch.cuda.empty_cache()

           # Rolling adaptive mean over window w=2 (frames t-2..t+2)
           w = 2
           n = len(diffs_cpu)
           cuts = []
           last_cut_frame = -min_scene_len_frames

           for i in range(n):
               score = diffs_cpu[i]
               if score < min_content_val:
                   continue
               start_w = max(0, i - w)
               end_w = min(n, i + w + 1)
               window_mean = float(np.mean(diffs_cpu[start_w:end_w]))
               ratio = score / (window_mean + 1e-5)

               if ratio >= adaptive_threshold and (i - last_cut_frame) >= min_scene_len_frames:
                   cut_time = round((i + 1) / fps, 3)
                   cuts.append(cut_time)
                   last_cut_frame = i

           return cuts
       except Exception as e:
           sys.stderr.write(f"[SceneIntel] PyTorch detection error: {e}\n")
           if torch.cuda.is_available():
               torch.cuda.empty_cache()
           return []
   ```
3. Update `detect_scenes(source_path: str)` to read the video in batches via OpenCV/PyAV, emit progress percentages (`[SceneIntel] PROGRESS <pct>`), downscale to 360p, compute cuts via `detect_cuts_from_frames_pytorch`, and format scenes into `[{ "sceneId": i, "start": s, "end": e }]`.

- [ ] **Step 4: Run test to verify it passes**

Run:
```powershell
.\.venv\Scripts\python.exe autoshorts/src-tauri/scripts/test_scene_intelligence_gpu.py
```
Expected: PASS (all 4 tests pass in < 2 seconds).

- [ ] **Step 5: Commit**

```bash
git add autoshorts/src-tauri/scripts/scene_intelligence.py autoshorts/src-tauri/scripts/test_scene_intelligence_gpu.py
git commit -m "feat(scene_intelligence): implement PyTorch CUDA adaptive scene detection"
```

---

### Task 3: Backend State Registry, IPC Progress & Query Command (Rust)

**Files:**
- Modify: `autoshorts/src-tauri/src/scene_intelligence.rs`
- Modify: `autoshorts/src-tauri/src/lib.rs`

**Interfaces:**
- Event: `"scene-intel-status"` with payload:
  ```rust
  #[derive(Clone, Serialize, Deserialize)]
  pub struct SceneIntelStatus {
      pub project_id: String,
      pub status: String, // "idle" | "running" | "completed" | "error"
      pub progress: u8,
      pub scene_count: usize,
  }
  ```
- Command: `get_scene_intelligence_status(project_id: String) -> Result<SceneIntelStatus, String>`

- [ ] **Step 1: Write failing Rust unit tests**

In `autoshorts/src-tauri/src/scene_intelligence.rs`, update `SCENE_TIMEOUT_CEILING_SEC`:
Add test to `tests` module:

```rust
    #[test]
    fn test_scene_timeout_for_gpu_bound() {
        let dur = scene_timeout_for(Some(5000.0));
        assert!(dur.as_secs_f64() <= 600.0, "Ceiling must clamp to 600s for GPU acceleration");
    }

    #[test]
    fn test_scene_intel_status_serialization() {
        let s = SceneIntelStatus {
            project_id: "test_proj".to_string(),
            status: "completed".to_string(),
            progress: 100,
            scene_count: 42,
        };
        let json = serde_json::to_string(&s).unwrap();
        assert!(json.contains("\"projectId\":\"test_proj\""));
        assert!(json.contains("\"sceneCount\":42"));
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```powershell
cargo test --manifest-path autoshorts/src-tauri/Cargo.toml -p autoshorts --lib -j 2 -- scene_intelligence::tests::test_scene_timeout_for_gpu_bound
```
Expected: FAIL (current ceiling is 3600.0s).

- [ ] **Step 3: Update `scene_intelligence.rs`**

1. Define `SceneIntelStatus` in `scene_intelligence.rs` with `#[derive(Debug, Clone, Serialize, Deserialize)]`.
2. Update constants:
   ```rust
   pub const SCENE_DETECTOR: &str = "PyTorch-CUDA-AdaptiveDetector";
   pub const SCENE_DETECTOR_VERSION: &str = "1.0.0";
   pub const SCENE_CONFIG_VERSION: &str = "gpu-cuda-v1";
   const SCENE_TIMEOUT_FLOOR_SEC: f64 = 60.0;
   const SCENE_TIMEOUT_REALTIME_FACTOR: f64 = 0.5;
   const SCENE_TIMEOUT_CEILING_SEC: f64 = 600.0;
   ```
3. Update `lib.rs`:
   - Add `SceneIntelRegistry` to Tauri state: `Arc<Mutex<HashMap<String, SceneIntelStatus>>>`.
   - In `transcribe_project`:
     - Emit `"scene-intel-status"` with `status: "running", progress: 0`.
     - In the blocking background task, parse `[SceneIntel] PROGRESS <pct>` from stderr and emit progress updates.
     - On complete, emit `status: "completed", progress: 100, scene_count: doc.scenes.len()`.
     - On error, emit `status: "error"`.
   - Register Tauri command `get_scene_intelligence_status(app: tauri::AppHandle, project_id: String) -> Result<SceneIntelStatus, String>`.

- [ ] **Step 4: Run tests to verify they pass**

Run:
```powershell
cargo test --manifest-path autoshorts/src-tauri/Cargo.toml -p autoshorts --lib -j 2 -- scene_intelligence::tests
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add autoshorts/src-tauri/src/scene_intelligence.rs autoshorts/src-tauri/src/lib.rs
git commit -m "feat(scene_intel): add status registry, IPC progress events, and calibrated GPU timeouts"
```

---

### Task 4: Frontend Inline Status Pill in Controls Bar (React & CSS)

**Files:**
- Modify: `autoshorts/src/main.tsx`
- Modify: `autoshorts/src/styles.css`

**Interfaces:**
- Listen to: `listen<SceneIntelStatus>("scene-intel-status", (event) => ...)`
- Query: `invoke<SceneIntelStatus>("get_scene_intelligence_status", { projectId })`

- [ ] **Step 1: Add TypeScript types and state hook in `main.tsx`**

In `autoshorts/src/main.tsx`:
```tsx
type SceneIntelStatus = {
  projectId: string;
  status: "idle" | "running" | "completed" | "error";
  progress: number;
  sceneCount: number;
};
```
Inside `App` component:
```tsx
const [sceneIntelStatus, setSceneIntelStatus] = useState<SceneIntelStatus | null>(null);
```

- [ ] **Step 2: Add event listener and state hydration on project select**

In `main.tsx`, inside the existing project selection / mount `useEffect`:
```tsx
useEffect(() => {
  if (!detail?.project?.id) {
    setSceneIntelStatus(null);
    return;
  }
  const pid = detail.project.id;
  // Hydrate initial status
  invoke<SceneIntelStatus>("get_scene_intelligence_status", { projectId: pid })
    .then((status) => {
      if (status) setSceneIntelStatus(status);
    })
    .catch((err) => console.warn("[SceneIntel] Failed to query status:", err));

  // Listen to real-time events
  let unlistenFn: (() => void) | null = null;
  listen<SceneIntelStatus>("scene-intel-status", (event) => {
    if (event.payload.projectId === pid) {
      setSceneIntelStatus(event.payload);
    }
  }).then((unlisten) => {
    unlistenFn = unlisten;
  });

  return () => {
    if (unlistenFn) unlistenFn();
  };
}, [detail?.project?.id]);
```

- [ ] **Step 3: Render the inline status pill in `.candidate-controls-right`**

Directly beside `<button className="find-moments-btn" ...>`:
```tsx
{sceneIntelStatus && sceneIntelStatus.status !== "idle" && (
  <div
    className={`scene-intel-pill status-${sceneIntelStatus.status}`}
    title={
      sceneIntelStatus.status === "running"
        ? "GPU Scene Intelligence is scanning video for camera cuts in the background."
        : sceneIntelStatus.status === "completed"
        ? `Full scene intelligence ready (${sceneIntelStatus.sceneCount} camera cuts cached).`
        : "Scene intelligence encountered an issue; single-scene fallback active."
    }
  >
    {sceneIntelStatus.status === "running" && (
      <>
        <Loader2 className="spin" size={13} />
        <span>Scene Intel {sceneIntelStatus.progress > 0 ? `${sceneIntelStatus.progress}%` : "Scanning..."}</span>
      </>
    )}
    {sceneIntelStatus.status === "completed" && (
      <>
        <Check size={13} className="pill-check-icon" />
        <span>Scene Intel Ready ({sceneIntelStatus.sceneCount})</span>
      </>
    )}
    {sceneIntelStatus.status === "error" && (
      <>
        <AlertCircle size={13} className="pill-error-icon" />
        <span>Scene Intel (Fallback)</span>
      </>
    )}
  </div>
)}
```

- [ ] **Step 4: Add CSS styles in `styles.css`**

In `autoshorts/src/styles.css`:
```css
.scene-intel-pill {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 10px;
  border-radius: 6px;
  font-size: 12px;
  font-weight: 500;
  line-height: 1;
  height: 32px;
  box-sizing: border-box;
  user-select: none;
  transition: all 0.2s ease-in-out;
}

.scene-intel-pill.status-running {
  background: rgba(30, 58, 138, 0.25);
  border: 1px solid rgba(59, 130, 246, 0.4);
  color: #93c5fd;
  animation: pulse-border 2s infinite;
}

.scene-intel-pill.status-completed {
  background: rgba(6, 78, 59, 0.25);
  border: 1px solid rgba(16, 185, 129, 0.4);
  color: #6ee7b7;
}

.scene-intel-pill.status-error {
  background: rgba(120, 53, 15, 0.25);
  border: 1px solid rgba(245, 158, 11, 0.4);
  color: #fcd34d;
}

.pill-check-icon {
  color: #34d399;
}

.pill-error-icon {
  color: #fbbf24;
}

@keyframes pulse-border {
  0%, 100% { border-color: rgba(59, 130, 246, 0.4); }
  50% { border-color: rgba(147, 197, 253, 0.8); }
}
```

- [ ] **Step 5: Verify frontend build passes**

Run:
```powershell
npm --prefix autoshorts run build
```
Expected: PASS with 0 errors.

- [ ] **Step 6: Commit**

```bash
git add autoshorts/src/main.tsx autoshorts/src/styles.css
git commit -m "feat(ui): add compact Scene Intelligence status pill in candidate controls"
```

---

### Task 5: End-to-End Verification & Safety Invariants Check

**Files:** All modified files across Tasks 1–4.

- [ ] **Step 1: Run the full Rust test suite**
Run:
```powershell
cargo test --manifest-path autoshorts/src-tauri/Cargo.toml -p autoshorts --lib -j 2
```
Expected: all tests pass (0 failures).

- [ ] **Step 2: Run Python GPU test suite**
Run:
```powershell
.\.venv\Scripts\python.exe autoshorts/src-tauri/scripts/test_scene_intelligence_gpu.py
```
Expected: all 4 tests pass.

- [ ] **Step 3: Run existing Python adapter test suite**
Run:
```powershell
.\.venv\Scripts\python.exe -m unittest discover -s autoshorts/src-tauri/scripts -p "test_*.py"
```
Expected: all tests pass.

- [ ] **Step 4: Final frontend build verification**
Run:
```powershell
npm --prefix autoshorts run build
```
Expected: 0 errors.
