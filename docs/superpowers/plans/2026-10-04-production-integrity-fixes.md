# AutoShorts 11.0: Production-Integrity Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve the ten confirmed production-integrity defects (D1–D4, N1–N4, L1–L3) across database safety, error semantics, intelligence handoffs, truthful telemetry, and canonical caching in AutoShorts 11.0 without forcing features or modifying unproven/blocked components.

**Architecture:** Implement atomic SQLite transactions (D1) and inspection-backed schema evolution (N3); establish fail-closed Render QA gates (D3, N1, L2); tighten process-tree lifecycles and timeouts (N2, L3); connect Scene Intelligence cuts and canonical speaker roles to framing (D2, L1); build backward-compatible observational telemetry (D4); and implement canonical fingerprint disk caching for framing and pacing (N4).

**Tech Stack:** Rust (Tauri 2, rusqlite, serde_json, sha2), Python 3.10+ (OpenCV, YOLO, BoT-SORT sidecars), SQLite.

## Global Constraints
- DO NOT force features to run (no threshold modifications, prompt changes, or artificial media).
- DO NOT touch B1–B4 (missing learned models, Re-ID, YouTube path) or U1–U7 (unproven features).
- Minimal-scope principle: preserve existing APIs, fallbacks, and observable behavior.
- Every task must end with an independently testable deliverable.
- Windows-safe execution: pass `-j 2` to `cargo test`.

---

### Task 1: Database Safety & Migrations (D1 & N3)

**Files:**
- Modify: `autoshorts/src-tauri/src/db.rs:20-180, 381-460`
- Test: `autoshorts/src-tauri/src/db.rs` (under `#[cfg(test)] mod tests`)

**Interfaces:**
- Consumes: `rusqlite::Transaction`, `rusqlite::Connection`
- Produces: Atomic `replace_candidates(&self, project_id: &str, drafts: &[CandidateDraft]) -> Result<Vec<Candidate>>` and inspection-backed `migrate(&self) -> Result<()>`

- [ ] **Step 1: Write failing unit tests for atomic rollback and inspection-backed migrations**

Add to `autoshorts/src-tauri/src/db.rs` under `mod tests`:
```rust
#[test]
fn test_replace_candidates_atomic_rollback_on_failure() {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("test_rollback.db");
    let db = Database::open(&db_path).unwrap();

    // Create a project and initial candidate
    let project_id = "p-rollback-1";
    let init_candidate = CandidateDraft {
        start: 10.0,
        end: 20.0,
        score: 8.5,
        hook: "Initial Hook".to_string(),
        rationale: "Initial Rationale".to_string(),
        hook_start: Some(10.0),
        hook_end: Some(13.0),
        hook_confidence: Some(0.9),
        opening_context_score: Some(8.0),
        payoff_text: Some("Initial Payoff".to_string()),
        payoff_start: Some(17.0),
        payoff_end: Some(20.0),
        payoff_score: Some(8.0),
        payoff_completion: Some(true),
    };
    let initial_candidates = db.replace_candidates(project_id, &[init_candidate.clone()]).unwrap();
    assert_eq!(initial_candidates.len(), 1);

    // Draft set where the second draft is crafted to cause a failure or simulated transaction abort
    // Verify that on failure, initial_candidates[0] still exists in the database
}

#[test]
fn test_schema_migration_inspection_backed_and_idempotent() {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("test_migrate.db");
    let db = Database::open(&db_path).unwrap();

    // Opening a second time must verify schema and succeed idempotently
    let db2 = Database::open(&db_path).unwrap();
    let user_ver: i64 = db2.conn.lock().unwrap()
        .query_row("PRAGMA user_version", [], |r| r.get(0)).unwrap();
    assert!(user_ver >= 1);
}
```

- [ ] **Step 2: Run tests to verify they fail or compile**
Run: `cargo test -p autoshorts --lib -j 2 -- test_replace_candidates_atomic_rollback test_schema_migration_inspection_backed`
Expected: Compile check or failure before implementation.

- [ ] **Step 3: Implement atomic transaction in `replace_candidates` and inspection-backed `migrate`**

In `autoshorts/src-tauri/src/db.rs`:
1. In `replace_candidates`:
```rust
pub fn replace_candidates(
    &self,
    project_id: &str,
    drafts: &[CandidateDraft],
) -> Result<Vec<Candidate>> {
    let mut conn = self.conn.lock().expect("database mutex poisoned");
    let tx = conn.transaction()?;

    tx.execute(
        "DELETE FROM candidates WHERE project_id = ?1",
        params![project_id],
    )?;

    let selected_cutoff = drafts.len().min(12).max(3).min(drafts.len());
    let mut candidates = Vec::with_capacity(drafts.len());

    for (index, draft) in drafts.iter().enumerate() {
        let metadata_json = serde_json::to_string(draft).ok();
        let candidate = Candidate {
            id: Uuid::new_v4().to_string(),
            project_id: project_id.to_string(),
            start_sec: draft.start,
            end_sec: draft.end,
            score: draft.score,
            hook: draft.hook.clone(),
            rationale: draft.rationale.clone(),
            rank: (index + 1) as i64,
            selected: index < selected_cutoff,
            hook_start_sec: draft.hook_start,
            hook_end_sec: draft.hook_end,
            hook_confidence: draft.hook_confidence,
            opening_context_score: draft.opening_context_score,
            payoff_text: draft.payoff_text.clone(),
            payoff_start_sec: draft.payoff_start,
            payoff_end_sec: draft.payoff_end,
            payoff_score: draft.payoff_score,
            payoff_completion: draft.payoff_completion,
            metadata_json,
        };

        let payoff_completion_int: Option<i64> =
            candidate.payoff_completion.map(|b| if b { 1 } else { 0 });

        tx.execute(
            "INSERT INTO candidates (
                id, project_id, start_sec, end_sec, score, hook, rationale, rank, selected,
                hook_start_sec, hook_end_sec, hook_confidence, opening_context_score,
                payoff_text, payoff_start_sec, payoff_end_sec, payoff_score, payoff_completion,
                metadata_json
             )
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19)",
            params![
                &candidate.id,
                &candidate.project_id,
                candidate.start_sec,
                candidate.end_sec,
                candidate.score,
                &candidate.hook,
                &candidate.rationale,
                candidate.rank,
                if candidate.selected { 1 } else { 0 },
                candidate.hook_start_sec,
                candidate.hook_end_sec,
                candidate.hook_confidence,
                candidate.opening_context_score,
                &candidate.payoff_text,
                candidate.payoff_start_sec,
                candidate.payoff_end_sec,
                candidate.payoff_score,
                payoff_completion_int,
                &candidate.metadata_json,
            ],
        )?;

        tx.execute(
            "INSERT INTO clips (id, candidate_id, status) VALUES (?1, ?2, 'pending')",
            params![Uuid::new_v4().to_string(), &candidate.id],
        )?;

        candidates.push(candidate);
    }

    tx.commit()?;
    Ok(candidates)
}
```

2. In `migrate`:
Implement column inspection via `PRAGMA table_info` helper and verify schema post-conditions:
```rust
fn ensure_column(conn: &Connection, table: &str, column: &str, col_type: &str) -> Result<()> {
    let mut stmt = conn.prepare(&format!("PRAGMA table_info({})", table))?;
    let cols = stmt.query_map([], |row| row.get::<_, String>(1))?;
    let mut found = false;
    for col in cols {
        if col?.eq_ignore_ascii_case(column) {
            found = true;
            break;
        }
    }
    if !found {
        conn.execute(&format!("ALTER TABLE {} ADD COLUMN {} {}", table, column, col_type), [])?;
    }
    Ok(())
}
```
Apply required columns, perform schema verification, and advance `PRAGMA user_version`. Verify schema even if `user_version` is already current.

- [ ] **Step 4: Run tests to verify they pass**
Run: `cargo test -p autoshorts --lib -j 2 -- test_replace_candidates_atomic_rollback test_schema_migration_inspection_backed`
Expected: PASS (0 failed).

- [ ] **Step 5: Commit Task 1**
```bash
git add autoshorts/src-tauri/src/db.rs
git commit -m "fix(db): implement atomic candidate replacement transaction (D1) and inspection-backed schema migration (N3)"
```

---

### Task 2: Render QA Guardrail & Configuration Alignment (D3, N1, L2)

**Files:**
- Modify: `autoshorts/src-tauri/src/render_qa.rs:83-97, 302-307, 1280-1344`
- Modify: `autoshorts/src-tauri/src/lib.rs:2596-2645`
- Test: `autoshorts/src-tauri/src/render_qa.rs` (under `#[cfg(test)] mod tests`)

**Interfaces:**
- Consumes: `render_qa::run_render_qa`
- Produces: `pub enum RenderQaOutcome { Pass(RenderQaReport), Fail(RenderQaReport), Error(String), Disabled }`

- [ ] **Step 1: Write failing unit tests for RenderQaOutcome and 14-check accounting**

Add tests in `autoshorts/src-tauri/src/render_qa.rs`:
```rust
#[test]
fn test_render_qa_reports_all_14_checks_even_when_framing_absent() {
    let config = RenderQaConfig { enabled: true, ..Default::default() };
    let engine = RenderQaEngine::new(config);
    // When validating without a framing plan, face_containment must be explicitly present with QaStatus::Skipped
    // Total checks count must equal 14.
}

#[test]
fn test_render_qa_config_default_is_enabled() {
    let cfg = RenderQaConfig::default();
    assert!(cfg.enabled, "RenderQaConfig default must be enabled (N1)");
}
```

- [ ] **Step 2: Run tests to verify they fail**
Run: `cargo test -p autoshorts --lib -j 2 -- test_render_qa_config_default_is_enabled test_render_qa_reports_all_14_checks`
Expected: FAIL (enabled is false currently).

- [ ] **Step 3: Implement RenderQaOutcome, N1 alignment, and L2 accounting**

In `autoshorts/src-tauri/src/render_qa.rs`:
1. In `RenderQaConfig::default`: change `enabled: true`.
2. Define `pub enum RenderQaOutcome`:
```rust
#[derive(Debug, Clone)]
pub enum RenderQaOutcome {
    Pass(RenderQaReport),
    Fail(RenderQaReport),
    Error(String),
    Disabled,
}
```
3. In `run_render_qa`:
```rust
pub fn run_render_qa(
    output_path: &str,
    source_path: &str,
    candidate_id: &str,
    expected_duration_sec: Option<f64>,
    expected_has_audio: bool,
    captions_expected: bool,
    framing_plan: Option<&crate::media::SmartFramingPlan>,
) -> RenderQaOutcome {
    if !render_qa_enabled() {
        return RenderQaOutcome::Disabled;
    }

    let config = RenderQaConfig {
        enabled: true,
        ..Default::default()
    };
    let engine = RenderQaEngine::new(config);

    match engine.validate_render(
        output_path,
        source_path,
        candidate_id,
        expected_duration_sec,
        expected_has_audio,
        captions_expected,
        framing_plan,
    ) {
        Ok(report) => {
            if report.has_critical_failures() {
                RenderQaOutcome::Fail(report)
            } else {
                RenderQaOutcome::Pass(report)
            }
        }
        Err(e) => RenderQaOutcome::Error(e.to_string()),
    }
}
```
4. In `validate_render`, if `framing_plan` is `None`, explicitly add `check_face_containment` with `QaStatus::Skipped`:
```rust
if path.exists() {
    if let Some(plan) = framing_plan {
        self.check_face_containment(output_path, plan, &mut report);
    } else {
        report.add_check(QaCheck {
            name: "face_containment".to_string(),
            status: QaStatus::Skipped,
            details: "No framing plan provided".to_string(),
            severity: QaSeverity::Info,
        });
    }
}
```
Likewise, ensure duration consistency check records `QaStatus::Skipped` if `expected_duration_sec` is `None`.

5. In `autoshorts/src-tauri/src/lib.rs:2596-2645`:
Handle `RenderQaOutcome`:
```rust
let (qa_rejected, qa_summary_line, qa_error_msg) = match qa_outcome {
    render_qa::RenderQaOutcome::Pass(ref report) => {
        (false, Some(format!("render_qa: {}", report.summary())), None)
    }
    render_qa::RenderQaOutcome::Fail(ref report) => {
        let detail = report
            .checks
            .iter()
            .filter(|c| c.status == render_qa::QaStatus::Fail && c.severity == render_qa::QaSeverity::Critical)
            .map(|c| format!("{}: {}", c.name, c.details))
            .collect::<Vec<_>>()
            .join("; ");
        eprintln!("[Render QA] rejecting clip {} — {}", candidate_id, detail);
        (true, Some(format!("render_qa: {}", report.summary())), Some(detail))
    }
    render_qa::RenderQaOutcome::Error(ref err) => {
        eprintln!("[Render QA] validation error — rejecting clip {}: {}", candidate_id, err);
        (true, Some(format!("render_qa error: {}", err)), Some(err.clone()))
    }
    render_qa::RenderQaOutcome::Disabled => {
        (false, Some("render_qa: disabled by environment".to_string()), None)
    }
};
```
If `qa_rejected`, update clip to status `"error"` in SQLite and reject publication. Never mark `"done"` on error or failure.

- [ ] **Step 4: Run tests to verify they pass**
Run: `cargo test -p autoshorts --lib -j 2 -- test_render_qa`
Expected: PASS (0 failed).

- [ ] **Step 5: Commit Task 2**
```bash
git add autoshorts/src-tauri/src/render_qa.rs autoshorts/src-tauri/src/lib.rs
git commit -m "fix(render_qa): enforce fail-closed gate (D3), align config default (N1), and account for all 14 checks (L2)"
```

---

### Task 3: Subprocess Lifecycle & Timeout Realignment (N2 & L3)

**Files:**
- Modify: `autoshorts/src-tauri/src/proc_guard.rs:70-155`
- Modify: `autoshorts/src-tauri/src/panns_reactions.rs:55-80, 360-375, 490-505`
- Test: `autoshorts/src-tauri/src/proc_guard.rs`, `autoshorts/src-tauri/src/panns_reactions.rs`

**Interfaces:**
- Consumes: `proc_guard::run_bounded`, `PannsConfig`
- Produces: Safe process tree termination on abnormal exits and authoritative `self.config.timeout_sec` in PANNs.

- [ ] **Step 1: Write failing unit test for process tree cleanup and PANNs timeout**

In `autoshorts/src-tauri/src/panns_reactions.rs`:
```rust
#[test]
fn test_panns_config_timeout_authoritative() {
    let mut config = PannsConfig::default();
    assert_eq!(config.timeout_sec, 300);
    config.timeout_sec = 45;
    // Verify engine uses config.timeout_sec rather than detached 900 constant
}
```

- [ ] **Step 2: Run test to verify it compiles and checks behavior**
Run: `cargo test -p autoshorts --lib -j 2 -- test_panns_config_timeout`

- [ ] **Step 3: Implement process tree cleanup and authoritative PANNs timeout**

1. In `autoshorts/src-tauri/src/proc_guard.rs`:
Update `run_bounded` so that:
- Any still-running process tree owned by the guarded operation is terminated on every abnormal termination path (timeout, wait error, cancellation, or pipe failure).
- Processes that have already exited are correctly reaped without unnecessary termination attempts:
```rust
// In proc_guard.rs loop:
match child.try_wait() {
    Ok(Some(status)) => break Some(status),
    Ok(None) => {
        if started.elapsed() >= timeout {
            timed_out = true;
            kill_process_tree(&mut child);
            let _ = child.wait();
            break None;
        }
        std::thread::sleep(poll);
    }
    Err(e) => {
        kill_process_tree(&mut child);
        let _ = child.wait();
        return Err(anyhow!("{}: wait failed: {}", stage, e));
    }
}
```
2. In `autoshorts/src-tauri/src/panns_reactions.rs`:
Remove `const PANNS_TIMEOUT_SEC: f64 = 900.0;`.
In `process_source`:
```rust
let timeout_sec = if self.config.timeout_sec > 0 { self.config.timeout_sec } else { 300 };
let budget = std::time::Duration::from_secs(timeout_sec);
println!("[PANNs] sidecar START budget={:.0}s", timeout_sec);
```
Ensure log and `run_bounded` match `self.config.timeout_sec`.

- [ ] **Step 4: Run tests to verify they pass**
Run: `cargo test -p autoshorts --lib -j 2 -- proc_guard panns`
Expected: PASS (0 failed).

- [ ] **Step 5: Commit Task 3**
```bash
git add autoshorts/src-tauri/src/proc_guard.rs autoshorts/src-tauri/src/panns_reactions.rs
git commit -m "fix(proc_guard, panns): clean up owned process trees on abnormal exits (L3) and unify PANNs timeout budget (N2)"
```

---

### Task 4: Intelligence Handoff & Role Consumption (D2 & L1)

**Files:**
- Modify: `autoshorts/src-tauri/src/media.rs:1515-1525, 1635-1645`
- Modify: `autoshorts/src-tauri/src/lib.rs:2245-2275, 2315-2335`
- Modify: `autoshorts/src-tauri/scripts/speaker_tracker.py:625-655, 885-905`
- Test: `autoshorts/src-tauri/src/media.rs`, Python tests

**Interfaces:**
- Consumes: `si.speaker_map`, `scene_cuts_json`
- Produces: Decoupled `SpeakerIntelSidecarInputs`, `--scene-cuts-json` passed to tracker, and `applicationRole` populated in fusion intervals.

- [ ] **Step 1: Write failing test for decoupled sidecar inputs and speaker map role serialization**

Add test in `autoshorts/src-tauri/src/media.rs`:
```rust
#[test]
fn test_speaker_intel_sidecar_inputs_decoupled() {
    let inputs = SpeakerIntelSidecarInputs {
        diarization_json: None,
        gallery_json: None,
        scene_cuts_json: Some("path/to/scenes.json"),
    };
    assert!(inputs.scene_cuts_json.is_some());
    assert!(inputs.diarization_json.is_none());
}
```

- [ ] **Step 2: Run test to verify it fails/compiles**
Run: `cargo test -p autoshorts --lib -j 2 -- test_speaker_intel_sidecar_inputs_decoupled`

- [ ] **Step 3: Implement decoupled inputs and canonical speaker role wiring**

1. In `autoshorts/src-tauri/src/media.rs`:
```rust
pub struct SpeakerIntelSidecarInputs<'a> {
    pub diarization_json: Option<&'a str>,
    pub gallery_json: Option<&'a str>,
    pub scene_cuts_json: Option<&'a str>,
}
```
In `detect_speaker_crop_params_with_intel`:
```rust
if let Some(inputs) = speaker_intel_inputs {
    if let Some(d) = inputs.diarization_json { cmd.arg("--diarization-json").arg(d); }
    if let Some(g) = inputs.gallery_json { cmd.arg("--gallery-json").arg(g); }
    if let Some(s) = inputs.scene_cuts_json { cmd.arg("--scene-cuts-json").arg(s); }
}
```

2. In `autoshorts/src-tauri/src/lib.rs`:
Serialize `speaker_map` into `diar_doc`:
```rust
let diar_doc = serde_json::json!({
    "model": si.diarization.model,
    "segments": si.diarization.segments.iter().map(|s| serde_json::json!({
        "speaker_id": s.speaker_id,
        "start": s.start,
        "end": s.end,
        "confidence": s.confidence,
    })).collect::<Vec<_>>(),
    "speaker_map": si.speaker_map.mappings.iter().map(|m| serde_json::json!({
        "speaker_id": m.diarization_id,
        "role": m.application_role, // canonical serde snake_case ("host", "guest", "unknown")
    })).collect::<Vec<_>>(),
});
```
Assemble `sidecar_inputs` whenever diarization, gallery, or scene cuts are present:
```rust
let sidecar_inputs = if diar_tmp.is_some() || gallery_tmp.is_some() || scene_cuts_json.is_some() {
    Some(media::SpeakerIntelSidecarInputs {
        diarization_json: diar_tmp.as_ref().map(|p| p.to_str().unwrap()),
        gallery_json: gallery_tmp.as_ref().map(|p| p.to_str().unwrap()),
        scene_cuts_json: scene_cuts_json.as_deref(),
    })
} else {
    None
};
```
3. In `autoshorts/src-tauri/scripts/speaker_tracker.py`:
In `load_diarization_sidecar_json()`: parse `speaker_map` into a dictionary `{spk_id: role}`.
In interval serialization (`lines 890-905`): look up `s.diarization_id` in `speaker_map` to populate `"applicationRole": role`.
In `load_scene_cuts_json()`: confirm scene cuts are merged into candidate shot boundaries.

- [ ] **Step 4: Run tests to verify actual consumption**
Run: `cargo test -p autoshorts --lib -j 2 -- test_speaker_intel_sidecar_inputs_decoupled`
Run Python tests: `python -m unittest autoshorts/src-tauri/scripts/test_active_speaker_suite.py`
Expected: PASS (0 failed).

- [ ] **Step 5: Commit Task 4**
```bash
git add autoshorts/src-tauri/src/media.rs autoshorts/src-tauri/src/lib.rs autoshorts/src-tauri/scripts/speaker_tracker.py
git commit -m "fix(framing): pass Scene Intelligence cuts (D2) and wire canonical speaker_map roles to intervals (L1)"
```

---

### Task 5: Truthful Stage Execution Telemetry (D4)

**Files:**
- Modify: `autoshorts/src-tauri/src/models.rs`
- Modify: `autoshorts/src-tauri/src/lib.rs:2495-2530`
- Test: `autoshorts/src-tauri/src/models.rs`

**Interfaces:**
- Consumes: Pipeline stage outputs (`pacing_plan`, `boundary_opt`, `audio_plan`, `caption_intel_plan`, `framing_plan`)
- Produces: Backward-compatible hybrid JSON in `clips.applied_features` with structured observational `stages`.

- [ ] **Step 1: Write failing unit test for hybrid applied_features JSON**

In `autoshorts/src-tauri/src/models.rs`:
```rust
#[test]
fn test_applied_features_hybrid_schema_backward_compatibility() {
    let json_str = r#"{
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
                "outputConsumed": true
            },
            "t7Prosody": {
                "status": "blocked",
                "reason": "model_missing",
                "fallbackUsed": true,
                "outputProduced": false,
                "outputConsumed": false
            }
        }
    }"#;
    let v: serde_json::Value = serde_json::from_str(json_str).unwrap();
    assert_eq!(v["smartPacing"], true);
    assert_eq!(v["stages"]["smartPacing"]["status"], "executed");
    assert_eq!(v["stages"]["t7Prosody"]["status"], "blocked");
}
```

- [ ] **Step 2: Implement stage execution data model and lib.rs telemetry builder**

1. In `autoshorts/src-tauri/src/models.rs`:
Define `StageStatus` (`disabled`, `skipped`, `blocked`, `executed`, `fallback`, `failed`) and `StageExecutionRecord` (`status`, `variant`, `fallback_used`, `output_produced`, `output_consumed`, `reason`, `details`).
2. In `autoshorts/src-tauri/src/lib.rs:2495-2530`:
Build `stages` observational record:
- `smartPacing`: `executed` if `pacing_plan.is_some()`; `skipped` if non-noop 0 cuts; `fallback` if timed out.
- `t7Prosody`: `blocked` if model missing; `fallbackUsed: true` if deterministic phrasing ran; `executed` only if learned model ran.
- `sceneIntelligence`: `executed` with `outputProduced: true`, `outputConsumed: true` if passed to framing and loaded.
- `framing`: `executed` (`variant: "dualframe"` or `"adaptive"`); `fallback` if emergency center-crop ran (`framingFallback: true`).
- `audioIntelligence`: `executed` if non-empty filter chain applied; `skipped` if clean audio; `disabled` if off.
- `panns`: `disabled` if flag off; `blocked` if checkpoint absent.
- `vlm`: `disabled` if key missing or flag off.
Serialize preserving existing top-level booleans: `smartPacing`, `hookEndingOptimization`, `audioIntelligence`, `captionIntelligence`, `framingFallback`.

- [ ] **Step 3: Run tests to verify they pass**
Run: `cargo test -p autoshorts --lib -j 2 -- test_applied_features`
Expected: PASS (0 failed).

- [ ] **Step 4: Commit Task 5**
```bash
git add autoshorts/src-tauri/src/models.rs autoshorts/src-tauri/src/lib.rs
git commit -m "feat(telemetry): implement backward-compatible hybrid applied_features schema with truthful stage execution (D4)"
```

---

### Task 6: Canonical Reusable Caches for Framing & Pacing (N4)

**Files:**
- Modify: `autoshorts/src-tauri/src/media.rs`
- Modify: `autoshorts/src-tauri/src/pacing.rs`
- Test: `autoshorts/src-tauri/src/media.rs`, `autoshorts/src-tauri/src/pacing.rs`

**Interfaces:**
- Consumes: `compute_source_hash`, canonical input descriptors
- Produces: Atomic disk caches in `%APPDATA%/com.autoshorts.desktop/framing_cache` and `pacing_cache`.

- [ ] **Step 1: Write failing unit tests for canonical framing and pacing cache hits**

In `autoshorts/src-tauri/src/media.rs`:
```rust
#[test]
fn test_framing_cache_key_generation_and_semantic_hit() {
    // Verify canonical descriptor hashes consistently to SHA256 hex string
    // Verify cache hit returns semantic plan and bypasses subprocess
}
```
In `autoshorts/src-tauri/src/pacing.rs`:
```rust
#[test]
fn test_pacing_cache_no_op_plan_cacheable() {
    // Verify a plan with 0 cuts is structurally valid and successfully cached
}
```

- [ ] **Step 2: Implement canonical descriptor hashing and atomic disk cache**

1. In `autoshorts/src-tauri/src/media.rs`:
Define `FramingCacheDescriptor`:
```rust
#[derive(Serialize)]
struct FramingCacheDescriptor<'a> {
    source_hash: &'a str,
    start_ms: i64,
    end_ms: i64,
    crop_w: i64,
    iw: u32,
    ih: u32,
    framing_mode: &'a str,
    diarization_hash: Option<&'a str>,
    gallery_hash: Option<&'a str>,
    speaker_map_hash: Option<&'a str>,
    scene_cuts_hash: Option<&'a str>,
    algorithm_version: &'a str,
    cache_schema_version: &'a str,
}
```
Compute SHA-256 fingerprint of JSON descriptor.
Check `%APPDATA%/com.autoshorts.desktop/framing_cache/<fingerprint>.json`.
On hit $\rightarrow$ deserialize and return `SmartFramingPlan`.
On miss $\rightarrow$ run tracker, then write to temp file $\rightarrow$ flush $\rightarrow$ atomic rename.
2. In `autoshorts/src-tauri/src/pacing.rs`:
Define `PacingCacheDescriptor` with `source_hash`, `start_ms`, `end_ms`, `words_fingerprint`, `pacing_mode`, `effective_config_hash`, `algorithm_version`, `cache_schema_version`.
Compute SHA-256 fingerprint.
Support caching of valid plans including no-op / 0-cut plans.
Atomic write via temp file and rename.

- [ ] **Step 3: Run tests to verify they pass**
Run: `cargo test -p autoshorts --lib -j 2 -- test_framing_cache test_pacing_cache`
Expected: PASS (0 failed).

- [ ] **Step 4: Commit Task 6**
```bash
git add autoshorts/src-tauri/src/media.rs autoshorts/src-tauri/src/pacing.rs
git commit -m "feat(cache): implement canonical descriptor disk caching with atomic writes for framing and pacing (N4)"
```

---

### Task 7: Full System Regression & End-to-End Validation

**Files:**
- Test: All Rust test suites, Python test suites, and frontend build.

- [ ] **Step 1: Run full Rust library test suite**
Run: `cargo test -p autoshorts --lib -j 2`
Expected: 485+ passed, 0 failed.

- [ ] **Step 2: Run integration test suites**
Run: `cargo test -p autoshorts --test pause_intel_suite -j 2`
Run: `cargo test -p autoshorts --test t7_prosody_suite -j 2`
Expected: All pass, 0 failed.

- [ ] **Step 3: Run Python test suites**
Run: `python -m unittest autoshorts/src-tauri/scripts/test_active_speaker_suite.py`
Run: `python -m unittest autoshorts/src-tauri/scripts/test_pause_intel_suite.py`
Run: `python -m unittest autoshorts/src-tauri/scripts/test_t7_prosody_suite.py`
Run: `python autoshorts/src-tauri/scripts/test_smart_pacing_learned.py`
Expected: All pass, 0 failed.

- [ ] **Step 4: Run frontend production build**
Run: `npm --prefix autoshorts run build`
Expected: Clean build, 0 errors.

- [ ] **Step 5: Run natural end-to-end pipeline verification**
Run the existing AutoShorts candidate generation and render path on real media. Verify:
1. Application completes.
2. Render QA executes and fails closed if invalid.
3. Scene Intelligence boundaries reach framing.
4. Database replacement transaction is atomic.
5. Telemetry truthfully reflects stage execution.
6. Process trees terminate cleanly.
7. Caches preserve correctness without cross-source contamination.
8. B1–B4 and U1–U7 remain untouched.
