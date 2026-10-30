# AutoShorts 11.0 — Mode 2 Watcher Rebuild Loop Root Cause Diagnosis and Remediation Report

**Date:** 2026-10-13  
**Status:** RESOLVED & VERIFIED  
**Severity:** P0 Bug (Development-mode process rebuild loop interrupting user render)  
**Workspace:** `D:\College\Autoshorts 11.0`  
**Branch:** `feature/autoclip-scoring-topn`

---

## 1. Executive Summary

During testing of **AutoClip Studio (Presentation Mode 2)**, triggering the **Cut** action on a clip caused `cargo-tauri dev` to enter a rebuild loop, displaying messages such as:
```text
Info File src-tauri\scripts\autoclip_studio\data\projects changed. Rebuilding application...
Info File src-tauri\scripts\autoclip_studio\metadata\input.srt changed.
```
The desktop application automatically reloaded, briefly displayed an empty project list, and returned to the workspace. The underlying render was interrupted, and user actions had to be repeated.

This audit definitively diagnosed the root cause, implemented an architecture-compliant, minimal surgical fix isolating all mutable runtime paths outside `src-tauri`, and verified that zero watcher triggers occur during Mode 2 execution while preserving all immutable assets, unit tests, and Mode 1 presentation behavior.

---

## 2. Root Cause Analysis

### 2.1 Tauri Development Watcher Architecture
`cargo-tauri dev` monitors the entire `src-tauri/` directory recursively for file changes. Any file write or directory creation inside `src-tauri/` (including under `src-tauri/scripts/`) triggers an immediate Rust recompilation and webview reload.

### 2.2 Unconfigured Runtime Data Directory in AutoClip
In `autoshorts/src-tauri/scripts/autoclip_adapter.py`, the default data directory logic was implemented as:
```python
def setup_data_directory(project_id: str) -> Path:
    if "AUTOCLIP_DATA_DIR" not in os.environ:
        if os.getenv("AUTOSHORTS_DATA_DIR"):
            os.environ["AUTOCLIP_DATA_DIR"] = os.environ["AUTOSHORTS_DATA_DIR"]
        else:
            os.environ["AUTOCLIP_DATA_DIR"] = str((STUDIO_DIR / "data").resolve())
```
Where `STUDIO_DIR` resolved to `autoshorts/src-tauri/scripts/autoclip_studio`.

Similarly, in `project_cancellation.py`:
```python
def project_directory(project_id: str) -> Path:
    data_dir = os.getenv('AUTOCLIP_DATA_DIR')
    if data_dir:
        base = Path(data_dir) / 'projects'
    else:
        base = Path(__file__).resolve().parent / 'data' / 'projects'
    return base / project_id
```
And in `framing.py`:
```python
def _data_dir() -> Path:
    d = os.getenv("AUTOCLIP_DATA_DIR")
    return Path(d) if d else Path(__file__).resolve().parent / "data"
```

### 2.3 The Rebuild Loop Execution Path
1. The user clicks **Cut** on a Mode 2 candidate in `main.tsx`.
2. Rust Tauri IPC invokes `lib.rs:render_flat_clip_for_candidate_impl`, which dispatches to `media.rs:render_autoclip_clip`.
3. `media.rs` spawned `python autoclip_adapter.py --payload <temp_payload_path>` without providing `AUTOCLIP_DATA_DIR` or `AUTOSHORTS_DATA_DIR`.
4. The Python process defaulted `AUTOCLIP_DATA_DIR` to `src-tauri\scripts\autoclip_studio\data`.
5. `write_srt_file` created `src-tauri\scripts\autoclip_studio\data\projects\<id>\metadata\input.srt`.
6. `render_draft` executed ffmpeg and wrote intermediate outputs into `src-tauri\scripts\autoclip_studio\data\projects\<id>\output\studio\<job_id>.mp4`.
7. Tauri's file watcher detected file creation and modification inside `src-tauri`, emitted `Info File ... changed. Rebuilding application...`, killed the dev backend process, rebuilt, and refreshed the browser.
8. The webview reload reset React state in `main.tsx`, briefly showing an empty list while SQLite queries resolved, creating a jarring UX and failing the render in flight.

---

## 3. Immutable Assets vs Mutable State Boundary

To ensure strict separation:

| Category | Path | Writable at Runtime? | Status |
| :--- | :--- | :--- | :--- |
| **YuNet ONNX Model** | `autoclip_studio/assets/models/face_detection_yunet_2023mar.onnx` | No (Read-Only) | Unchanged in `src-tauri` |
| **OFL Fonts** | `autoclip_studio/assets/fonts/*.ttf` | No (Read-Only) | Unchanged in `src-tauri` |
| **Python Modules** | `autoclip_studio/*.py`, `autoclip_adapter.py` | No (Read-Only) | Unchanged in `src-tauri` |
| **Project Working Dirs** | `<AUTOCLIP_DATA_DIR>/projects/<project_id>/` | Yes (Runtime) | **Relocated outside `src-tauri`** |
| **SRT Subtitles** | `<AUTOCLIP_DATA_DIR>/projects/<project_id>/metadata/input.srt` | Yes (Runtime) | **Relocated outside `src-tauri`** |
| **Cancellation Markers**| `<AUTOCLIP_DATA_DIR>/projects/<project_id>/metadata/cancel.*` | Yes (Runtime) | **Relocated outside `src-tauri`** |
| **Intermediate Renders**| `<AUTOCLIP_DATA_DIR>/projects/<project_id>/output/studio/*.mp4` | Yes (Runtime) | **Relocated outside `src-tauri`** |

---

## 4. Remediation Implementation

The smallest safe and architecture-compliant fix was applied across both the Rust orchestrator and the Python runtime:

### 4.1 Rust Orchestrator (`media.rs` & `lib.rs`)
- In `media.rs:render_autoclip_clip`, added `data_dir: Option<&Path>` to the function signature.
- Before spawning the Python adapter child process, explicitly sets `AUTOCLIP_DATA_DIR` on the `Command`:
  ```rust
  if let Some(dir) = data_dir {
      let autoclip_data = dir.join("autoclip_data");
      let _ = std::fs::create_dir_all(&autoclip_data);
      cmd.env("AUTOCLIP_DATA_DIR", &autoclip_data);
  } else if let Some(appdata) = dirs::data_dir() {
      let autoclip_data = appdata.join("autoshorts").join("autoclip_data");
      let _ = std::fs::create_dir_all(&autoclip_data);
      cmd.env("AUTOCLIP_DATA_DIR", &autoclip_data);
  }
  ```
- In `lib.rs:render_flat_clip_for_candidate_impl`, passed `Some(&data_dir)` directly into `media::render_autoclip_clip`. This aligns Mode 2 with AutoShorts' persistent data directory (`app_data_dir()`, typically `%LOCALAPPDATA%\com.autoshorts.desktop` or `%APPDATA%`).

### 4.2 Python Adapter Fallback (`autoclip_adapter.py`)
- Updated `setup_data_directory` fallback so that even if invoked standalone without environment variables, it defaults to `LOCALAPPDATA/autoshorts/autoclip_data` (or system temporary directory), never `STUDIO_DIR / "data"`:
  ```python
  if "AUTOCLIP_DATA_DIR" not in os.environ:
      if os.getenv("AUTOSHORTS_DATA_DIR"):
          os.environ["AUTOCLIP_DATA_DIR"] = os.environ["AUTOSHORTS_DATA_DIR"]
      elif os.getenv("LOCALAPPDATA"):
          os.environ["AUTOCLIP_DATA_DIR"] = str((Path(os.environ["LOCALAPPDATA"]) / "autoshorts" / "autoclip_data").resolve())
      else:
          import tempfile
          os.environ["AUTOCLIP_DATA_DIR"] = str((Path(tempfile.gettempdir()) / "autoshorts" / "autoclip_data").resolve())
  ```

### 4.3 Python Sidecar Fallbacks (`project_cancellation.py` & `framing.py`)
- Updated fallback in `project_directory(project_id)` in `project_cancellation.py` and `_data_dir()` in `framing.py` to route to `LOCALAPPDATA` / `tempfile` when `AUTOCLIP_DATA_DIR` is unset.
- Cleaned up the untracked `autoshorts/src-tauri/scripts/autoclip_studio/data` folder from disk.

---

## 5. Verification Results

### 5.1 End-to-End Mode 2 Cut Execution (No Watcher Rebuild)
Executed a full synthetic Mode 2 cut run (`ffmpeg` synthetic fixture -> `autoclip_adapter.py` subprocess -> draft framing -> render -> output relocation):
- **Result:** Return code 0, status `success`.
- **Output:** Produced cleanly at target destination.
- **External Data Path:** `input.srt` and project metadata written to `%LOCALAPPDATA%\autoshorts\autoclip_data\projects\test_loop_proj\metadata\input.srt`.
- **`src-tauri` Cleanliness:** Verified `Test-Path autoshorts\src-tauri\scripts\autoclip_studio\data` returned `False`. Zero files written to `src-tauri`. Zero Tauri dev watcher rebuild triggers.

### 5.2 Python Test Suites
- `pytest autoshorts/src-tauri/scripts/test_autoclip_adapter.py`: **36 passed** (including new `test_default_data_directory_fallback_outside_src_tauri`).
- `pytest test_m1_challenger2_suite.py test_m5_challenger2_suite.py test_adversarial_m1_presentation.py test_adversarial_m5_offline_framing.py`: **295 passed**.
- `unittest test_t7_prosody_suite.py test_pause_intel_suite.py`: **32 passed**.

### 5.3 Rust Test Suite
- `cargo test -p autoshorts --lib -j 2`: **556 passed, 0 failed**.
- `cargo test -p autoshorts --lib media::tests:: -j 2`: **31 passed, 0 failed**.

### 5.4 Frontend Compilation
- `npm run build`: **Built in 2.42s** with 0 errors.

---

## 6. Conclusion

The Mode 2 Cut rebuild loop bug is completely resolved. All mutable runtime paths are relocated outside `src-tauri` into designated application data directories. No watcher triggers occur, and Mode 1, Mode 2, and scoring pipelines operate cleanly with zero regression.
