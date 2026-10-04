# AutoShorts 11.0: Production-Integrity Fixes Design Specification

- **Date**: 2026-10-04
- **Author**: Antigravity (Advanced Agentic Coding) & Engineering Team
- **Status**: Approved
- **Target Repository**: `https://github.com/Nakshtra07/Shortform.git`

---

## 1. Executive Summary & Core Invariants

During a deep production-integrity audit of AutoShorts 11.0, ten specific defects across database safety, execution observability, error semantics, subprocess lifecycles, and intelligence handoffs were identified.

This design specification details the minimal-scope, production-grade resolutions for issues **D1, D2, D3, D4, N1, N2, N3, N4, L1, L2, and L3**.

### Absolute Engineering Invariants
1. **Never Force Features to Run**: Do not alter thresholds, feature flags, activation conditions, model requirements, media selection, prompts, or test inputs to manufacture feature execution or forced passes.
2. **Untouched Blocked & Unproven Stages (B1–B4, U1–U7)**: Do not invent fake ML models, download unvetted replacements, or alter execution criteria solely to make unproven or blocked features appear active.
3. **Fail-Closed Production Guardrails**: If mandatory validation (such as Render QA) cannot establish the integrity of an artifact, completion must be blocked.
4. **Truthful & Observational Telemetry**: Feature telemetry observes actual pipeline decisions and distinguishes blocked/fallback/skipped from executed; it never forces execution or fabricates success.
5. **Independent-Clip & Cross-Video Isolation**: Caching and state must never leak between clips or across different media sources.

---

## 2. D1: Database Transaction Safety

### 2.1 Problem Analysis
`db.rs::replace_candidates` performs a raw `DELETE FROM candidates WHERE project_id = ?1` directly on `conn` before looping through candidate and clip `INSERT` statements. Because `clips.candidate_id` references `candidates(id) ON DELETE CASCADE`, if any subsequent `INSERT` fails (e.g., constraint error, duplicate key, process crash), the previous candidates and all associated rendered clip rows are already permanently deleted.

### 2.2 Architecture & Implementation
Wrap candidate replacement into an atomic SQLite transaction with rollback semantics:
1. Acquire the database lock: `let mut conn = self.conn.lock().expect("database mutex poisoned");`
2. Open a transaction: `let tx = conn.transaction()?;`
3. Execute candidate deletion on `tx`:
   ```rust
   tx.execute("DELETE FROM candidates WHERE project_id = ?1", params![project_id])?;
   ```
4. Execute all candidate and clip `INSERT` statements on `tx`:
   ```rust
   for (index, draft) in drafts.iter().enumerate() {
       tx.execute("INSERT INTO candidates (...) VALUES (...)", params![...])?;
       tx.execute("INSERT INTO clips (id, candidate_id, status) VALUES (?1, ?2, 'pending')", params![...])?;
   }
   ```
5. Commit atomically: `tx.commit()?;`
6. **Rollback Behavior**: If any `execute` fails or an error is returned, `tx` drops without commit. SQLite automatically rolls back the entire transaction. The pre-existing candidates and their associated clips remain 100% intact.

### 2.3 Verification Strategy
- **Success Case**: New candidate drafts cleanly replace old candidates and pending clips.
- **Fault Injection Case**: Deliberately inject an error into the second candidate insert (e.g., duplicate primary key). Verify that the transaction aborts and the pre-existing candidate set and clip records are completely preserved in SQLite.

---

## 3. N3: Database Schema Versioning & Inspection-Backed Migrations

### 3.1 Problem Analysis
`db.rs::migrate()` runs unversioned `ALTER TABLE` statements masked by `let _ = conn.execute(...)`. While intended to ignore "duplicate column" errors on development databases, this pattern silently swallows genuine errors (syntax mistakes, locked tables, constraint violations) and leaves `PRAGMA user_version = 0`.

### 3.2 Architecture & Implementation
Schema evolution is made explicit, inspection-backed, and failure-aware:
1. **Schema Inspection**: Before altering a table, query `PRAGMA table_info(<table>)` to inspect existing column names dynamically.
2. **Apply Only Missing Migrations**:
   ```rust
   fn column_exists(conn: &Connection, table: &str, column: &str) -> Result<bool> {
       let mut stmt = conn.prepare(&format!("PRAGMA table_info({table})"))?;
       let rows = stmt.query_map([], |row| row.get::<_, String>(1))?;
       for col in rows {
           if col?.eq_ignore_ascii_case(column) {
               return Ok(true);
           }
       }
       Ok(false)
   }
   ```
3. **Strict Execution**: If a column is missing, run `conn.execute(&format!("ALTER TABLE {table} ADD COLUMN {def}"), [])?`. If this fails, return `Err(e)` immediately.
4. **Post-Condition Verification**: Verify that all expected columns now exist.
5. **Version Advancement**: Only after all table definitions and missing columns are verified, advance `PRAGMA user_version`:
   ```rust
   conn.execute_batch(&format!("PRAGMA user_version = {TARGET_SCHEMA_VERSION};"))?;
   ```

### 3.3 Verification Strategy
- **Fresh Database**: Initializes cleanly to `TARGET_SCHEMA_VERSION`.
- **Existing Database**: Accurately detects existing columns, adds only missing ones, and advances version.
- **Repeat Initialization**: Subsequent calls detect `user_version >= TARGET_SCHEMA_VERSION` and no-op idempotently.
- **Genuine Error**: Injected syntax or permission error aborts `Database::open` with an explicit error.

---

## 4. D3 & N1: Render QA Guardrail & Fail-Closed Semantics

### 4.1 Problem Analysis
`run_render_qa()` in `render_qa.rs` returns `None` on errors or when disabled. In `lib.rs:2596-2617`, `qa_rejected` evaluated to `false` when `qa_report` was `None`. Consequently, an unvalidated or errored render silently **failed open** and was marked `"done"`. Additionally, `RenderQaConfig::default()` had `enabled: false`, contradicting `render_qa_enabled()` which defaults to `true`.

### 4.2 Architecture & Implementation
1. **Unified Configuration Default (N1)**:
   - Update `RenderQaConfig::default().enabled = true`.
   - Both `engine.is_enabled()` and `render_qa_enabled()` standardize on the production contract: Render QA is **enabled by default**, with `AUTOSHORTS_RENDER_QA=0|false|off` serving as the development opt-out.
2. **Explicit Outcome Enum (D3)**:
   ```rust
   #[derive(Debug)]
   pub enum RenderQaOutcome {
       Pass(RenderQaReport),
       Fail(RenderQaReport),
       Error(String),
       Disabled,
   }
   ```
3. **Fail-Closed Production Gate in `lib.rs`**:
   - `RenderQaOutcome::Pass(report)`: Persist summary, mark clip as `"done"`.
   - `RenderQaOutcome::Fail(report)`: Critical check failed. Log details, mark clip as `"error"`.
   - `RenderQaOutcome::Error(err)`: **Fail Closed**. Log `[Render QA] Validation error — rejecting clip: {err}`, mark clip as `"error"`. **Never mark as `"done"`**.
   - `RenderQaOutcome::Disabled`: Permitted to complete only when `AUTOSHORTS_RENDER_QA` is explicitly disabled.

### 4.3 Verification Strategy
- **Pass**: Valid render marked `"done"`.
- **Fail**: Critical check failure (e.g. 0-byte file, resolution mismatch) marked `"error"`.
- **Error**: Injected QA crash or ffprobe error marked `"error"`, blocking successful completion.

---

## 5. N2, L2, & L3: Watchdogs, Accounting & Process Lifecycle

### 5.1 N2: Authoritative PANNs Reaction Timeout
- **Problem**: `PannsConfig.timeout_sec` defaulted to `300s`, while `panns_reactions.rs:361` hardcoded `PANNS_TIMEOUT_SEC = 900.0s`, ignoring the configuration.
- **Resolution**:
  - `self.config.timeout_sec` becomes the sole authoritative budget passed to `proc_guard::run_bounded`.
  - Standardize `PannsConfig::default().timeout_sec` to `300s` (5 minutes). Support an optional `AUTOSHORTS_PANNS_TIMEOUT_SEC` environment variable override for long media.
  - Remove the detached constant `PANNS_TIMEOUT_SEC = 900.0`.

### 5.2 L2: Explicit Accounting for Conditional QA Checks
- **Problem**: Render QA reported 13 checks while 14 were defined because conditional checks (`check_face_containment`, `duration_consistency`) were silently omitted when prerequisites were absent.
- **Resolution**:
  - Every defined QA check must always be accounted for in `report.checks`.
  - If a check's prerequisite is missing (e.g., `framing_plan` is `None`), explicitly add the check with `QaStatus::Skipped` and an explanatory message (`"No framing plan provided"`).

### 5.3 L3: Process-Tree Termination in `proc_guard`
- **Problem**: Abnormal exits from `child.try_wait()` or pipe failures could leave orphaned grandchild processes on Windows.
- **Resolution**:
  - Guarantee that any still-running process tree owned by the guarded operation is terminated on every abnormal termination path (timeout, wait error, cancellation, or pipe failure).
  - Ensure processes that have already exited are correctly reaped without unnecessary termination attempts.
  - Centralize cleanup to prevent orphaned background processes from holding file locks or CPU.

---

## 6. D2 & L1: Intelligence Wiring & Handoffs

### 6.1 D2: Scene Intelligence Framing Handoff
- **Problem**: Scene Intelligence boundary outputs were generated and saved to a temp file, but the third slot of `speaker_intel_inputs` was hardcoded to `None`, and `sidecar_inputs` was `None` if speaker intelligence was disabled.
- **Resolution**:
  - Decouple the intelligence input struct in `media.rs`:
    ```rust
    pub struct SpeakerIntelSidecarInputs<'a> {
        pub diarization_json: Option<&'a str>,
        pub gallery_json: Option<&'a str>,
        pub scene_cuts_json: Option<&'a str>,
    }
    ```
  - In `lib.rs`: pass `scene_cuts_json.as_deref()` whenever Scene Intelligence produced usable scene cuts.
  - Do not automatically reinterpret 0 cuts as failure; pass scene output when genuinely usable, otherwise retain the normal detector fallback.
  - In `media.rs`: forward `--scene-cuts-json` to `speaker_tracker.py` whenever present.
  - Verify actual consumption: confirm `speaker_tracker.py::load_scene_cuts_json` successfully loads the JSON file and merges external boundaries.

### 6.2 L1: Wiring `speaker_map` to Framing Metadata
- **Problem**: `si.speaker_map` (Host/Guest mapping) was computed and stored in SQLite, but omitted from the sidecar JSON. `speaker_tracker.py` hardcoded `applicationRole: None`.
- **Resolution**:
  - Include the canonical speaker role mapping in `diar_doc`:
    ```rust
    let diar_doc = serde_json::json!({
        "model": si.diarization.model,
        "segments": ...,
        "speaker_map": si.speaker_map.mappings.iter().map(|m| {
            serde_json::json!({
                "speaker_id": m.diarization_id,
                "role": m.application_role, // canonical serde snake_case: "host", "guest", "unknown"
            })
        }).collect::<Vec<_>>(),
    });
    ```
  - In `speaker_tracker.py`: parse `speaker_map` in `load_diarization_sidecar_json()` and populate `applicationRole` in fusion intervals.
  - Verify actual consumption: assert that fusion intervals contain the mapped `applicationRole`.

---

## 7. D4: Truthful Telemetry Data Model (`applied_features`)

### 7.1 Schema & Backwards Compatibility
Preserve all 5 legacy top-level booleans and introduce the observational `stages` object:

```json
{
  "smartPacing": true,
  "hookEndingOptimization": false,
  "audioIntelligence": false,
  "captionIntelligence": true,
  "framingFallback": false,
  "stages": {
    "smartPacing": {
      "status": "executed",
      "variant": "sp2",
      "fallbackUsed": false,
      "outputProduced": true,
      "outputConsumed": true,
      "details": { "cutsCount": 3, "removedSec": 0.42 }
    },
    "sceneIntelligence": {
      "status": "executed",
      "fallbackUsed": false,
      "outputProduced": true,
      "outputConsumed": true,
      "details": { "sceneBoundariesCount": 8 }
    },
    "framing": {
      "status": "executed",
      "variant": "dualframe",
      "fallbackUsed": false,
      "outputProduced": true,
      "outputConsumed": true
    },
    "t7Prosody": {
      "status": "blocked",
      "reason": "model_missing",
      "fallbackUsed": true,
      "outputProduced": false,
      "outputConsumed": false
    },
    "audioIntelligence": {
      "status": "skipped",
      "reason": "no_filtering_needed",
      "fallbackUsed": false,
      "outputProduced": false,
      "outputConsumed": false
    },
    "panns": {
      "status": "disabled",
      "reason": "feature_flag_off",
      "fallbackUsed": false,
      "outputProduced": null,
      "outputConsumed": null
    }
  }
}
```

### 7.2 Semantic Status Definitions
- `disabled`: Feature flag or environment variable is turned off.
- `blocked`: Prerequisite model, binary, or key is unavailable.
- `skipped`: Algorithm was bypassed or did not run due to upstream gating.
- `executed`: Algorithm ran, even if it made zero changes (e.g. Smart Pacing ran analysis and made 0 cuts).
- `fallback`: Fallback path actually executed.
- `fallbackUsed`: Boolean distinguishing blocked-primary + fallback-success vs pure unhandled block.
- `outputProduced` / `outputConsumed`: Boolean or null when not applicable.

---

## 8. N4: Unified Canonical Caching for Framing & Pacing

### 8.1 Core Architecture
Replace unbounded execution with disk-backed caching based on cryptographic fingerprints of canonical input descriptors:

```text
Actual function inputs
        ↓
Canonical Descriptor Struct
        ↓
SHA-256 Hash Fingerprint
        ↓
Disk Lookup: %APPDATA%/com.autoshorts.desktop/<cache_dir>/<fingerprint>.json
        ↓
valid cache? ──► [YES] ──► Deserialize & return semantic plan (bypass subprocess)
        │
      [NO]
        ▼
Execute existing pipeline
        ▼
Validate output plan
        ▼
Atomic Cache Write: Temp File ──► Flush/Close ──► Atomic Rename
```

### 8.2 Framing Cache Descriptor
```rust
#[derive(Serialize)]
struct FramingCacheDescriptor<'a> {
    source_hash: &'a str,           // Canonical compute_source_hash()
    start_ms: i64,                  // Quantized candidate bounds
    end_ms: i64,
    crop_w: i64,
    iw: u32,
    ih: u32,
    framing_mode: &'a str,          // "original" vs "adaptive"
    diarization_hash: Option<&'a str>,
    gallery_hash: Option<&'a str>,
    speaker_map_hash: Option<&'a str>,
    scene_cuts_hash: Option<&'a str>,
    algorithm_version: &'a str,
    cache_schema_version: &'a str,
}
```

### 8.3 Pacing Cache Descriptor
```rust
#[derive(Serialize)]
struct PacingCacheDescriptor<'a> {
    source_hash: &'a str,           // Canonical compute_source_hash()
    start_ms: i64,
    end_ms: i64,
    words_fingerprint: &'a str,     // Hash of word timings in candidate range
    pacing_mode: &'a str,           // "sp1" vs "sp2"
    effective_config_hash: &'a str, // Effective thresholds & protection rules
    algorithm_version: &'a str,
    cache_schema_version: &'a str,
}
```

### 8.4 Invariants & Guarantees
1. **Source Identity**: Reuses the repository's canonical `compute_source_hash` (SHA-256 of size + mtime + first 4MB).
2. **No-Op Caching**: Valid plans resulting in 0 cuts or default framing are valid and fully cacheable.
3. **Semantic Equality**: Cache hit verification compares semantic plan structures rather than fragile raw byte equality.
4. **Subprocess Bypass**: A cache hit avoids launching Python or FFmpeg subprocesses.
5. **Atomic Concurrency**: Write to temp file $\rightarrow$ flush/close $\rightarrow$ atomic rename guarantees zero partial or corrupted cache files.
6. **Strict Invalidation**: Incrementing `algorithm_version` or `cache_schema_version` naturally invalidates older cache entries.

---

## 9. Implementation Plan & Staging Order

1. **Phase 1: Database Safety (D1, N3)**
   - Atomic transaction in `replace_candidates` (`db.rs`).
   - Inspection-backed schema evolution (`db.rs`).
2. **Phase 2: Error Semantics & Lifecycles (D3, N1, N2, L2, L3)**
   - `RenderQaOutcome` fail-closed gate in `render_qa.rs` and `lib.rs`.
   - Align `RenderQaConfig::default()` and `PannsConfig.timeout_sec`.
   - Complete 14-check accounting with `QaStatus::Skipped`.
   - Full process-tree termination in `proc_guard.rs`.
3. **Phase 3: Intelligence Wiring (D2, L1)**
   - Pass Scene Intelligence cuts to `SpeakerIntelSidecarInputs`.
   - Wire canonical `speaker_map` roles to `speaker_tracker.py`.
4. **Phase 4: Truthful Telemetry (D4)**
   - Implement `applied_features` hybrid schema and semantic stage statuses in `lib.rs`.
5. **Phase 5: Canonical Reusable Caches (N4)**
   - Implement canonical descriptor hashing and atomic disk caches for framing (`media.rs`) and pacing (`pacing.rs`).
6. **Phase 6: Full Regression Verification**
   - Run existing and new test suites.
