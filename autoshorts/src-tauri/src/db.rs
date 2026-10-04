use std::{
    path::Path,
    sync::{Arc, Mutex},
};

use anyhow::{bail, Context, Result};
use base64::Engine as _;
use chrono::Utc;
use rusqlite::{params, Connection, OptionalExtension};
use sha2::Digest;
use uuid::Uuid;

use crate::models::{
    ActiveSpeakerState, ApplicationSpeakerMap, ApplicationSpeakerRole, Candidate, CandidateDraft,
    Clip, ClipCopy, MappingEvidence, Project, ProjectDetail, SpeakerDiarizationResult,
    SpeakerMapping, TrackReIdEmbedding, Transcript,
};

#[derive(Clone)]
pub struct Database {
    conn: Arc<Mutex<Connection>>,
}

impl Database {
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let conn = Connection::open(path).context("opening SQLite database")?;
        let db = Self {
            conn: Arc::new(Mutex::new(conn)),
        };
        db.migrate()?;
        Ok(db)
    }

    fn migrate(&self) -> Result<()> {
        let conn = self.conn.lock().expect("database mutex poisoned");
        conn.execute_batch(
            "
            PRAGMA foreign_keys = ON;

            CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY,
                name TEXT,
                source_path TEXT NOT NULL,
                source_duration REAL,
                status TEXT NOT NULL,
                transcription_mode TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS transcripts (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                engine TEXT NOT NULL,
                raw_json TEXT NOT NULL,
                raw_transcript_json TEXT,
                language TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS candidates (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                start_sec REAL NOT NULL,
                end_sec REAL NOT NULL,
                score REAL NOT NULL,
                hook TEXT NOT NULL,
                rationale TEXT NOT NULL,
                rank INTEGER NOT NULL,
                selected INTEGER NOT NULL DEFAULT 0,
                hook_start_sec REAL,
                hook_end_sec REAL,
                hook_confidence REAL,
                opening_context_score REAL,
                payoff_text TEXT,
                payoff_start_sec REAL,
                payoff_end_sec REAL,
                payoff_score REAL,
                payoff_completion INTEGER,
                metadata_json TEXT
            );

            CREATE TABLE IF NOT EXISTS clips (
                id TEXT PRIMARY KEY,
                candidate_id TEXT NOT NULL REFERENCES candidates(id) ON DELETE CASCADE,
                status TEXT NOT NULL,
                output_path TEXT,
                face_track_json TEXT,
                caption_ass_path TEXT,
                render_log TEXT,
                applied_features TEXT
            );

            CREATE TABLE IF NOT EXISTS clip_copy (
                id TEXT PRIMARY KEY,
                clip_id TEXT NOT NULL REFERENCES clips(id) ON DELETE CASCADE,
                platform TEXT NOT NULL,
                hook_text TEXT,
                caption_text TEXT,
                hashtags TEXT
            );

            ",
        )?;
        let _ = conn.execute("ALTER TABLE projects ADD COLUMN name TEXT", []);
        let _ = conn.execute("ALTER TABLE projects ADD COLUMN caption_style TEXT", []);
        let _ = conn.execute("ALTER TABLE projects ADD COLUMN framing_mode TEXT", []);
        let _ = conn.execute("ALTER TABLE candidates ADD COLUMN hook_start_sec REAL", []);
        let _ = conn.execute("ALTER TABLE candidates ADD COLUMN hook_end_sec REAL", []);
        let _ = conn.execute("ALTER TABLE candidates ADD COLUMN hook_confidence REAL", []);
        let _ = conn.execute(
            "ALTER TABLE candidates ADD COLUMN opening_context_score REAL",
            [],
        );
        let _ = conn.execute("ALTER TABLE candidates ADD COLUMN payoff_text TEXT", []);
        let _ = conn.execute(
            "ALTER TABLE candidates ADD COLUMN payoff_start_sec REAL",
            [],
        );
        let _ = conn.execute("ALTER TABLE candidates ADD COLUMN payoff_end_sec REAL", []);
        let _ = conn.execute("ALTER TABLE candidates ADD COLUMN payoff_score REAL", []);
        let _ = conn.execute(
            "ALTER TABLE candidates ADD COLUMN payoff_completion INTEGER",
            [],
        );
        match conn.execute("ALTER TABLE candidates ADD COLUMN metadata_json TEXT", []) {
            Ok(_) => {}
            Err(e) => {
                let msg = e.to_string().to_lowercase();
                if !msg.contains("duplicate column") {
                    return Err(e.into());
                }
            }
        }
        let _ = conn.execute(
            "ALTER TABLE transcripts ADD COLUMN raw_transcript_json TEXT",
            [],
        );
        // AutoShorts 8.0 — structured per-clip feature-application status.
        // NULL = legacy clip (no feature-status information). Non-NULL = complete
        // boolean truth table for a clip generated after this migration.
        let _ = conn.execute("ALTER TABLE clips ADD COLUMN applied_features TEXT", []);

        // Phase 2 migration addenda: visual_track_identity columns added after
        // the initial Phase 2 migration (existing dev databases). The `let _ =`
        // pattern ignores "duplicate column" errors, consistent with above.
        let _ = conn.execute(
            "ALTER TABLE visual_track_identity ADD COLUMN source_hash TEXT",
            [],
        );
        let _ = conn.execute(
            "ALTER TABLE visual_track_identity ADD COLUMN shot_idx INTEGER DEFAULT 0",
            [],
        );

        // Phase 2: Speaker Intelligence tables
        conn.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS speaker_diarization (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                source_hash TEXT NOT NULL,
                model TEXT NOT NULL,
                version TEXT NOT NULL,
                speakers_json TEXT NOT NULL,
                segments_json TEXT NOT NULL,
                confidence REAL,
                created_at TEXT NOT NULL,
                UNIQUE(project_id, source_hash, model, version)
            );

            CREATE TABLE IF NOT EXISTS speaker_mapping (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                diarization_id TEXT NOT NULL,
                application_role TEXT NOT NULL,
                confidence REAL NOT NULL,
                evidence_json TEXT,
                created_at TEXT NOT NULL,
                UNIQUE(project_id, diarization_id)
            );

            CREATE TABLE IF NOT EXISTS visual_track_identity (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                source_hash TEXT,
                track_id INTEGER NOT NULL,
                shot_idx INTEGER DEFAULT 0,
                reid_embedding BLOB,
                embedding_model TEXT,
                first_seen_sec REAL,
                last_seen_sec REAL,
                total_detections INTEGER,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS active_speaker_fusion (
                id TEXT PRIMARY KEY,
                candidate_id TEXT NOT NULL REFERENCES candidates(id) ON DELETE CASCADE,
                intervals_json TEXT NOT NULL,
                model_version TEXT,
                created_at TEXT NOT NULL
            );
            ",
        )?;
        Ok(())
    }

    pub fn create_project(
        &self,
        source_path: &str,
        transcription_mode: &str,
        caption_style: &str,
        framing_mode: &str,
        source_duration: Option<f64>,
    ) -> Result<Project> {
        let now = Utc::now().to_rfc3339();
        let project = Project {
            id: Uuid::new_v4().to_string(),
            name: None,
            source_path: source_path.to_string(),
            source_duration,
            status: "ingest".to_string(),
            transcription_mode: transcription_mode.to_string(),
            caption_style: Some(caption_style.to_string()),
            framing_mode: Some(framing_mode.to_string()),
            created_at: now.clone(),
            updated_at: now,
        };

        let conn = self.conn.lock().expect("database mutex poisoned");
        conn.execute(
            "INSERT INTO projects (id, name, source_path, source_duration, status, transcription_mode, caption_style, framing_mode, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                project.id,
                project.name,
                project.source_path,
                project.source_duration,
                project.status,
                project.transcription_mode,
                project.caption_style,
                project.framing_mode,
                project.created_at,
                project.updated_at
            ],
        )?;

        Ok(project)
    }

    pub fn list_projects(&self) -> Result<Vec<Project>> {
        let conn = self.conn.lock().expect("database mutex poisoned");
        let mut stmt = conn.prepare(
            "SELECT id, name, source_path, source_duration, status, transcription_mode, created_at, updated_at, caption_style, framing_mode
             FROM projects ORDER BY updated_at DESC",
        )?;

        let rows = stmt.query_map([], project_from_row)?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(Into::into)
    }

    pub fn get_project(&self, project_id: &str) -> Result<Project> {
        let conn = self.conn.lock().expect("database mutex poisoned");
        conn.query_row(
            "SELECT id, name, source_path, source_duration, status, transcription_mode, created_at, updated_at, caption_style, framing_mode
             FROM projects WHERE id = ?1",
            params![project_id],
            project_from_row,
        )
        .map_err(Into::into)
    }

    pub fn project_detail(&self, project_id: &str) -> Result<ProjectDetail> {
        let project = self.get_project(project_id)?;
        let transcript = self.latest_transcript(project_id)?;
        let candidates = self.list_candidates(project_id)?;
        let clips = self.list_clips_for_project(project_id)?;
        let copy = self.list_copy_for_project(project_id)?;

        Ok(ProjectDetail {
            project,
            transcript,
            candidates,
            clips,
            copy,
            multimodal_status: None,
        })
    }

    pub fn update_project_status(
        &self,
        project_id: &str,
        status: &str,
        source_duration: Option<f64>,
    ) -> Result<()> {
        let now = Utc::now().to_rfc3339();
        let conn = self.conn.lock().expect("database mutex poisoned");
        conn.execute(
            "UPDATE projects SET status = ?1, source_duration = COALESCE(?2, source_duration), updated_at = ?3 WHERE id = ?4",
            params![status, source_duration, now, project_id],
        )?;
        Ok(())
    }

    pub fn save_transcript(
        &self,
        project_id: &str,
        engine: &str,
        raw_json: &str,
        language: Option<&str>,
        raw_transcript_json: Option<&str>,
    ) -> Result<Transcript> {
        let transcript = Transcript {
            id: Uuid::new_v4().to_string(),
            project_id: project_id.to_string(),
            engine: engine.to_string(),
            raw_json: raw_json.to_string(),
            raw_transcript_json: raw_transcript_json.map(ToOwned::to_owned),
            language: language.map(ToOwned::to_owned),
            created_at: Utc::now().to_rfc3339(),
        };

        let conn = self.conn.lock().expect("database mutex poisoned");
        conn.execute(
            "DELETE FROM transcripts WHERE project_id = ?1",
            params![project_id],
        )?;
        conn.execute(
            "INSERT INTO transcripts (id, project_id, engine, raw_json, raw_transcript_json, language, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                transcript.id,
                transcript.project_id,
                transcript.engine,
                transcript.raw_json,
                transcript.raw_transcript_json,
                transcript.language,
                transcript.created_at
            ],
        )?;
        Ok(transcript)
    }

    pub fn latest_transcript(&self, project_id: &str) -> Result<Option<Transcript>> {
        let conn = self.conn.lock().expect("database mutex poisoned");
        conn.query_row(
            "SELECT id, project_id, engine, raw_json, raw_transcript_json, language, created_at
             FROM transcripts WHERE project_id = ?1 ORDER BY created_at DESC LIMIT 1",
            params![project_id],
            |row| {
                Ok(Transcript {
                    id: row.get(0)?,
                    project_id: row.get(1)?,
                    engine: row.get(2)?,
                    raw_json: row.get(3)?,
                    raw_transcript_json: row.get(4)?,
                    language: row.get(5)?,
                    created_at: row.get(6)?,
                })
            },
        )
        .optional()
        .map_err(Into::into)
    }

    pub fn clear_candidates(&self, project_id: &str) -> Result<()> {
        let conn = self.conn.lock().expect("database mutex poisoned");
        conn.execute(
            "DELETE FROM candidates WHERE project_id = ?1",
            params![project_id],
        )?;
        Ok(())
    }

    pub fn replace_candidates(
        &self,
        project_id: &str,
        drafts: &[CandidateDraft],
    ) -> Result<Vec<Candidate>> {
        let conn = self.conn.lock().expect("database mutex poisoned");
        conn.execute(
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

            conn.execute(
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

            conn.execute(
                "INSERT INTO clips (id, candidate_id, status) VALUES (?1, ?2, 'pending')",
                params![Uuid::new_v4().to_string(), &candidate.id],
            )?;

            candidates.push(candidate);
        }

        Ok(candidates)
    }

    pub fn list_candidates(&self, project_id: &str) -> Result<Vec<Candidate>> {
        let conn = self.conn.lock().expect("database mutex poisoned");
        let mut stmt = conn.prepare(
            "SELECT id, project_id, start_sec, end_sec, score, hook, rationale, rank, selected,
                    hook_start_sec, hook_end_sec, hook_confidence, opening_context_score,
                    payoff_text, payoff_start_sec, payoff_end_sec, payoff_score, payoff_completion,
                    metadata_json
             FROM candidates WHERE project_id = ?1 ORDER BY rank ASC",
        )?;
        let rows = stmt.query_map(params![project_id], candidate_from_row)?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(Into::into)
    }

    pub fn get_candidate_with_project(&self, candidate_id: &str) -> Result<(Candidate, Project)> {
        let conn = self.conn.lock().expect("database mutex poisoned");
        conn.query_row(
            "SELECT
                candidates.id, candidates.project_id, candidates.start_sec, candidates.end_sec,
                candidates.score, candidates.hook, candidates.rationale, candidates.rank, candidates.selected,
                candidates.hook_start_sec, candidates.hook_end_sec, candidates.hook_confidence, candidates.opening_context_score,
                candidates.payoff_text, candidates.payoff_start_sec, candidates.payoff_end_sec, candidates.payoff_score, candidates.payoff_completion,
                candidates.metadata_json,
                projects.id, projects.name, projects.source_path, projects.source_duration, projects.status,
                projects.transcription_mode, projects.created_at, projects.updated_at, projects.caption_style, projects.framing_mode
             FROM candidates
             INNER JOIN projects ON projects.id = candidates.project_id
             WHERE candidates.id = ?1",
            params![candidate_id],
            |row| {
                let selected: i64 = row.get(8)?;
                let payoff_completion: Option<bool> = match row.get::<_, Option<i64>>(17)? {
                    Some(1) => Some(true),
                    Some(0) => Some(false),
                    Some(v) => Some(v != 0),
                    None => None,
                };
                let candidate = Candidate {
                    id: row.get(0)?,
                    project_id: row.get(1)?,
                    start_sec: row.get(2)?,
                    end_sec: row.get(3)?,
                    score: row.get(4)?,
                    hook: row.get(5)?,
                    rationale: row.get(6)?,
                    rank: row.get(7)?,
                    selected: selected == 1,
                    hook_start_sec: row.get(9)?,
                    hook_end_sec: row.get(10)?,
                    hook_confidence: row.get(11)?,
                    opening_context_score: row.get(12)?,
                    payoff_text: row.get(13)?,
                    payoff_start_sec: row.get(14)?,
                    payoff_end_sec: row.get(15)?,
                    payoff_score: row.get(16)?,
                    payoff_completion,
                    metadata_json: row.get(18)?,
                };
                let project = Project {
                    id: row.get(19)?,
                    name: row.get(20)?,
                    source_path: row.get(21)?,
                    source_duration: row.get(22)?,
                    status: row.get(23)?,
                    transcription_mode: row.get(24)?,
                    created_at: row.get(25)?,
                    updated_at: row.get(26)?,
                    caption_style: row.get(27)?,
                    framing_mode: row.get(28)?,
                };
                Ok((candidate, project))
            },
        )
        .map_err(Into::into)
    }

    pub fn update_clip_for_candidate(
        &self,
        candidate_id: &str,
        status: &str,
        output_path: Option<&str>,
        caption_ass_path: Option<&str>,
        render_log: Option<&str>,
        applied_features: Option<&str>,
    ) -> Result<()> {
        let conn = self.conn.lock().expect("database mutex poisoned");
        conn.execute(
            "UPDATE clips
             SET status = ?1,
                 output_path = COALESCE(?2, output_path),
                 caption_ass_path = COALESCE(?3, caption_ass_path),
                 render_log = COALESCE(?4, render_log),
                 applied_features = COALESCE(?5, applied_features)
             WHERE candidate_id = ?6",
            params![
                status,
                output_path,
                caption_ass_path,
                render_log,
                applied_features,
                candidate_id
            ],
        )?;
        Ok(())
    }

    pub fn set_selected_clip_count(
        &self,
        project_id: &str,
        count: usize,
    ) -> Result<Vec<Candidate>> {
        let conn = self.conn.lock().expect("database mutex poisoned");
        conn.execute(
            "UPDATE candidates SET selected = CASE WHEN rank <= ?1 THEN 1 ELSE 0 END WHERE project_id = ?2",
            params![count as i64, project_id],
        )?;
        drop(conn);
        self.list_candidates(project_id)
    }

    /// Range-based candidate selection: marks candidates with `start_rank <= rank <= end_rank`
    /// (inclusive, 1-based ranks) as selected and all others as unselected.
    ///
    /// Validation:
    /// - `start_rank >= 1`
    /// - `end_rank` must not exceed the total candidate count for the project
    /// - `start_rank <= end_rank`
    ///
    /// Returns the full ordered candidate list (rank ASC) reflecting the new selection.
    pub fn set_selected_rank_range(
        &self,
        project_id: &str,
        start_rank: i64,
        end_rank: i64,
    ) -> Result<Vec<Candidate>> {
        if start_rank < 1 {
            bail!(
                "Invalid selection range: start rank must be >= 1 (got {})",
                start_rank
            );
        }
        if end_rank < start_rank {
            bail!(
                "Invalid selection range: start rank ({}) must not exceed end rank ({})",
                start_rank,
                end_rank
            );
        }

        let conn = self.conn.lock().expect("database mutex poisoned");
        let total: i64 = conn.query_row(
            "SELECT COUNT(*) FROM candidates WHERE project_id = ?1",
            params![project_id],
            |row| row.get(0),
        )?;
        if end_rank > total {
            bail!(
                "Invalid selection range: end rank ({}) exceeds total candidate count ({})",
                end_rank,
                total
            );
        }

        conn.execute(
            "UPDATE candidates
             SET selected = CASE WHEN rank >= ?1 AND rank <= ?2 THEN 1 ELSE 0 END
             WHERE project_id = ?3",
            params![start_rank, end_rank, project_id],
        )?;
        drop(conn);
        self.list_candidates(project_id)
    }

    fn list_clips_for_project(&self, project_id: &str) -> Result<Vec<Clip>> {
        let conn = self.conn.lock().expect("database mutex poisoned");
        let mut stmt = conn.prepare(
            "SELECT clips.id, clips.candidate_id, clips.status, clips.output_path, clips.face_track_json, clips.caption_ass_path, clips.render_log, clips.applied_features
             FROM clips
             INNER JOIN candidates ON candidates.id = clips.candidate_id
             WHERE candidates.project_id = ?1
             ORDER BY candidates.rank ASC",
        )?;
        let rows = stmt.query_map(params![project_id], |row| {
            Ok(Clip {
                id: row.get(0)?,
                candidate_id: row.get(1)?,
                status: row.get(2)?,
                output_path: row.get(3)?,
                face_track_json: row.get(4)?,
                caption_ass_path: row.get(5)?,
                render_log: row.get(6)?,
                applied_features: row.get(7)?,
            })
        })?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(Into::into)
    }

    fn list_copy_for_project(&self, project_id: &str) -> Result<Vec<ClipCopy>> {
        let conn = self.conn.lock().expect("database mutex poisoned");
        let mut stmt = conn.prepare(
            "SELECT clip_copy.id, clip_copy.clip_id, clip_copy.platform, clip_copy.hook_text, clip_copy.caption_text, clip_copy.hashtags
             FROM clip_copy
             INNER JOIN clips ON clips.id = clip_copy.clip_id
             INNER JOIN candidates ON candidates.id = clips.candidate_id
             WHERE candidates.project_id = ?1",
        )?;
        let rows = stmt.query_map(params![project_id], |row| {
            Ok(ClipCopy {
                id: row.get(0)?,
                clip_id: row.get(1)?,
                platform: row.get(2)?,
                hook_text: row.get(3)?,
                caption_text: row.get(4)?,
                hashtags: row.get(5)?,
            })
        })?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(Into::into)
    }

    pub fn delete_project(&self, project_id: &str) -> Result<()> {
        let conn = self.conn.lock().expect("database mutex poisoned");
        conn.execute("DELETE FROM projects WHERE id = ?1", params![project_id])?;
        Ok(())
    }

    pub fn rename_project(&self, project_id: &str, name: &str) -> Result<()> {
        let conn = self.conn.lock().expect("database mutex poisoned");
        let now = Utc::now().to_rfc3339();
        conn.execute(
            "UPDATE projects SET name = ?1, updated_at = ?2 WHERE id = ?3",
            params![name, now, project_id],
        )?;
        Ok(())
    }

    // ─── Phase 2: Speaker Intelligence Queries ────────────────────────────────

    /// Save or update a speaker diarization result for a project.
    pub fn save_speaker_diarization(
        &self,
        project_id: &str,
        result: &SpeakerDiarizationResult,
    ) -> Result<()> {
        let conn = self.conn.lock().expect("database mutex poisoned");
        let speakers_json = serde_json::to_string(&result.speakers)?;
        let segments_json = serde_json::to_string(&result.segments)?;
        let id = Uuid::new_v4().to_string();
        conn.execute(
            "INSERT OR REPLACE INTO speaker_diarization (
                id, project_id, source_hash, model, version,
                speakers_json, segments_json, confidence, created_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                &id,
                project_id,
                &result.source_hash,
                &result.model,
                &result.version,
                &speakers_json,
                &segments_json,
                result.confidence,
                &result.created_at,
            ],
        )?;
        Ok(())
    }

    /// Load a speaker diarization result by source hash.
    pub fn load_speaker_diarization(
        &self,
        project_id: &str,
        source_hash: &str,
        model: &str,
        version: &str,
    ) -> Result<Option<SpeakerDiarizationResult>> {
        let conn = self.conn.lock().expect("database mutex poisoned");
        let mut stmt = conn.prepare(
            "SELECT id, project_id, source_hash, model, version,
                    speakers_json, segments_json, confidence, created_at
             FROM speaker_diarization
             WHERE project_id = ?1 AND source_hash = ?2 AND model = ?3 AND version = ?4",
        )?;
        let result = stmt
            .query_row(params![project_id, source_hash, model, version], |row| {
                let speakers_json: String = row.get(5)?;
                let segments_json: String = row.get(6)?;
                let speakers: Vec<crate::models::DiarizedSpeaker> =
                    serde_json::from_str(&speakers_json).unwrap_or_default();
                let segments: Vec<crate::models::DiarizedSegment> =
                    serde_json::from_str(&segments_json).unwrap_or_default();
                Ok(SpeakerDiarizationResult {
                    source_hash: row.get(2)?,
                    model: row.get(3)?,
                    version: row.get(4)?,
                    speakers,
                    segments,
                    confidence: row.get(7)?,
                    created_at: row.get(8)?,
                })
            })
            .optional()?;
        Ok(result)
    }

    /// Save or update the application speaker mapping for a project.
    pub fn save_speaker_mapping(
        &self,
        project_id: &str,
        mapping: &ApplicationSpeakerMap,
    ) -> Result<()> {
        let conn = self.conn.lock().expect("database mutex poisoned");
        for m in &mapping.mappings {
            let evidence_json = serde_json::to_string(&m.evidence)?;
            let id = Uuid::new_v4().to_string();
            let now = Utc::now().to_rfc3339();
            conn.execute(
                "INSERT OR REPLACE INTO speaker_mapping (
                    id, project_id, diarization_id, application_role,
                    confidence, evidence_json, created_at
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                params![
                    &id,
                    project_id,
                    &m.diarization_id,
                    &m.application_role.to_string(),
                    m.confidence,
                    &evidence_json,
                    &now,
                ],
            )?;
        }
        Ok(())
    }

    /// Load the application speaker mapping for a project.
    pub fn load_speaker_mapping(&self, project_id: &str) -> Result<Option<ApplicationSpeakerMap>> {
        let conn = self.conn.lock().expect("database mutex poisoned");
        let mut stmt = conn.prepare(
            "SELECT id, project_id, diarization_id, application_role,
                    confidence, evidence_json, created_at
             FROM speaker_mapping
             WHERE project_id = ?1",
        )?;
        let rows = stmt.query_map(params![project_id], |row| {
            let evidence_json: String = row.get(5)?;
            let evidence: MappingEvidence =
                serde_json::from_str(&evidence_json).unwrap_or_default();
            Ok(SpeakerMapping {
                diarization_id: row.get(2)?,
                application_role: match row.get::<_, String>(3)?.as_str() {
                    "host" => ApplicationSpeakerRole::Host,
                    "guest" => ApplicationSpeakerRole::Guest,
                    _ => ApplicationSpeakerRole::Unknown,
                },
                confidence: row.get(4)?,
                evidence,
            })
        })?;
        let mut mappings = Vec::new();
        for m in rows {
            mappings.push(m?);
        }
        if mappings.is_empty() {
            return Ok(None);
        }
        Ok(Some(ApplicationSpeakerMap {
            source_hash: String::new(), // Will be filled by caller if needed
            mappings,
        }))
    }

    /// Save a visual track's Re-ID embedding.
    ///
    /// Dedup: the row id is a deterministic hash of the natural key
    /// (project_id | source_hash | shot_idx | track_id | model) so repeated
    /// saves REPLACE the same row instead of accumulating duplicates.
    /// Stale-model cleanup: entries for the same project+source written by a
    /// DIFFERENT embedding model are removed so stale embeddings never
    /// survive a model change.
    pub fn save_track_reid_embedding(
        &self,
        project_id: &str,
        source_hash: &str,
        embedding: &TrackReIdEmbedding,
    ) -> Result<()> {
        let conn = self.conn.lock().expect("database mutex poisoned");
        // embedding_b64 is base64-encoded float32 bytes (matches reid.py's
        // embedding_to_b64); decode to the raw BLOB for storage.
        let embedding_bytes = base64::engine::general_purpose::STANDARD
            .decode(&embedding.embedding_b64)
            .unwrap_or_default();
        if embedding_bytes.is_empty() || embedding_bytes.len() % 4 != 0 {
            // Malformed embedding: refuse to persist.
            anyhow::bail!("malformed Re-ID embedding for track {}", embedding.track_id);
        }

        // Deterministic natural-key row id.
        let mut hasher = sha2::Sha256::new();
        hasher.update(project_id.as_bytes());
        hasher.update(b"|");
        hasher.update(source_hash.as_bytes());
        hasher.update(b"|");
        hasher.update(embedding.shot_idx.to_le_bytes());
        hasher.update(b"|");
        hasher.update(embedding.track_id.to_le_bytes());
        hasher.update(b"|");
        hasher.update(embedding.model.as_bytes());
        let id = format!("{:x}", hasher.finalize());
        let now = Utc::now().to_rfc3339();

        // Remove stale-model entries for this project+source before writing.
        conn.execute(
            "DELETE FROM visual_track_identity
             WHERE project_id = ?1 AND source_hash = ?2
               AND embedding_model IS NOT NULL AND embedding_model != ?3",
            params![project_id, source_hash, &embedding.model],
        )?;

        conn.execute(
            "INSERT OR REPLACE INTO visual_track_identity (
                id, project_id, source_hash, track_id, shot_idx, reid_embedding,
                embedding_model, first_seen_sec, last_seen_sec, total_detections, created_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
            params![
                &id,
                project_id,
                source_hash,
                embedding.track_id,
                embedding.shot_idx,
                &embedding_bytes,
                &embedding.model,
                embedding.first_seen_sec,
                embedding.last_seen_sec,
                embedding.detection_count as i64,
                &now,
            ],
        )?;
        Ok(())
    }

    /// Load a visual track's Re-ID embedding.
    pub fn load_track_reid_embedding(
        &self,
        project_id: &str,
        source_hash: &str,
        track_id: i32,
        shot_idx: i32,
    ) -> Result<Option<TrackReIdEmbedding>> {
        let conn = self.conn.lock().expect("database mutex poisoned");
        let mut stmt = conn.prepare(
            "SELECT track_id, shot_idx, reid_embedding, embedding_model,
                    first_seen_sec, last_seen_sec, total_detections, created_at
             FROM visual_track_identity
             WHERE project_id = ?1 AND source_hash = ?2 AND track_id = ?3 AND shot_idx = ?4",
        )?;
        let result = stmt
            .query_row(
                params![project_id, source_hash, track_id, shot_idx],
                |row| {
                    let embedding_bytes: Vec<u8> = row.get(2)?;
                    // Convert back to base64 for JSON serialization
                    let embedding_b64 =
                        base64::engine::general_purpose::STANDARD.encode(&embedding_bytes);
                    Ok(TrackReIdEmbedding {
                        track_id: row.get(0)?,
                        shot_idx: row.get::<_, Option<i32>>(1)?.unwrap_or(0),
                        embedding_b64,
                        model: row.get::<_, Option<String>>(3)?.unwrap_or_default(),
                        first_seen_sec: row.get::<_, Option<f64>>(4)?.unwrap_or(0.0),
                        last_seen_sec: row.get::<_, Option<f64>>(5)?.unwrap_or(0.0),
                        detection_count: row.get::<_, Option<i64>>(6)?.unwrap_or(0) as usize,
                        updated_at: row.get(7)?,
                    })
                },
            )
            .optional()?;
        Ok(result)
    }

    /// Load all Re-ID embeddings for a project and source (current model only;
    /// stale-model entries are removed on save).
    pub fn load_all_track_reid_embeddings(
        &self,
        project_id: &str,
        source_hash: &str,
    ) -> Result<Vec<TrackReIdEmbedding>> {
        let conn = self.conn.lock().expect("database mutex poisoned");
        let mut stmt = conn.prepare(
            "SELECT track_id, shot_idx, reid_embedding, embedding_model,
                    first_seen_sec, last_seen_sec, total_detections, created_at
             FROM visual_track_identity
             WHERE project_id = ?1 AND source_hash = ?2",
        )?;
        let rows = stmt.query_map(params![project_id, source_hash], |row| {
            let embedding_bytes: Vec<u8> = row.get(2)?;
            let embedding_b64 = base64::engine::general_purpose::STANDARD.encode(&embedding_bytes);
            Ok(TrackReIdEmbedding {
                track_id: row.get(0)?,
                shot_idx: row.get::<_, Option<i32>>(1)?.unwrap_or(0),
                embedding_b64,
                model: row.get::<_, Option<String>>(3)?.unwrap_or_default(),
                first_seen_sec: row.get::<_, Option<f64>>(4)?.unwrap_or(0.0),
                last_seen_sec: row.get::<_, Option<f64>>(5)?.unwrap_or(0.0),
                detection_count: row.get::<_, Option<i64>>(6)?.unwrap_or(0) as usize,
                updated_at: row.get(7)?,
            })
        })?;
        let mut embeddings = Vec::new();
        for e in rows {
            embeddings.push(e?);
        }
        Ok(embeddings)
    }

    /// Save active speaker fusion result for a candidate.
    pub fn save_active_speaker_fusion(
        &self,
        candidate_id: &str,
        intervals: &[ActiveSpeakerState],
        model_version: &str,
    ) -> Result<()> {
        let conn = self.conn.lock().expect("database mutex poisoned");
        let intervals_json = serde_json::to_string(intervals)?;
        let id = Uuid::new_v4().to_string();
        let now = Utc::now().to_rfc3339();
        conn.execute(
            "INSERT OR REPLACE INTO active_speaker_fusion (
                id, candidate_id, intervals_json, model_version, created_at
            ) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![&id, candidate_id, &intervals_json, model_version, &now],
        )?;
        Ok(())
    }

    /// Load active speaker fusion result for a candidate.
    pub fn load_active_speaker_fusion(
        &self,
        candidate_id: &str,
    ) -> Result<Option<Vec<ActiveSpeakerState>>> {
        let conn = self.conn.lock().expect("database mutex poisoned");
        let mut stmt = conn.prepare(
            "SELECT intervals_json, model_version, created_at
             FROM active_speaker_fusion
             WHERE candidate_id = ?1",
        )?;
        let result = stmt
            .query_row(params![candidate_id], |row| {
                let intervals_json: String = row.get(0)?;
                let intervals: Vec<ActiveSpeakerState> =
                    serde_json::from_str(&intervals_json).unwrap_or_default();
                Ok(intervals)
            })
            .optional()?;
        Ok(result)
    }
}

fn project_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<Project> {
    Ok(Project {
        id: row.get(0)?,
        name: row.get(1)?,
        source_path: row.get(2)?,
        source_duration: row.get(3)?,
        status: row.get(4)?,
        transcription_mode: row.get(5)?,
        created_at: row.get(6)?,
        updated_at: row.get(7)?,
        caption_style: row.get(8)?,
        framing_mode: row.get(9)?,
    })
}

fn candidate_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<Candidate> {
    let selected: i64 = row.get(8)?;
    let payoff_completion: Option<bool> = match row.get::<_, Option<i64>>(17)? {
        Some(1) => Some(true),
        Some(0) => Some(false),
        Some(v) => Some(v != 0),
        None => None,
    };
    Ok(Candidate {
        id: row.get(0)?,
        project_id: row.get(1)?,
        start_sec: row.get(2)?,
        end_sec: row.get(3)?,
        score: row.get(4)?,
        hook: row.get(5)?,
        rationale: row.get(6)?,
        rank: row.get(7)?,
        selected: selected == 1,
        hook_start_sec: row.get(9)?,
        hook_end_sec: row.get(10)?,
        hook_confidence: row.get(11)?,
        opening_context_score: row.get(12)?,
        payoff_text: row.get(13)?,
        payoff_start_sec: row.get(14)?,
        payoff_end_sec: row.get(15)?,
        payoff_score: row.get(16)?,
        payoff_completion,
        metadata_json: row.get(18)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::MetadataError;

    fn test_db_with_candidates(count: usize) -> (Database, String) {
        let db = Database::open(&std::path::Path::new(":memory:")).expect("open in-memory db");
        let project = db
            .create_project(
                "/tmp/test-video.mp4",
                "local",
                "preset_viral_bold",
                "original",
                Some(600.0),
            )
            .expect("create project");
        let drafts: Vec<CandidateDraft> = (0..count)
            .map(|i| CandidateDraft {
                start: i as f64 * 30.0,
                end: i as f64 * 30.0 + 30.0,
                score: 0.9 - (i as f64 * 0.01),
                hook: format!("Hook {}", i + 1),
                rationale: format!("Rationale {}", i + 1),
                ..Default::default()
            })
            .collect();
        db.replace_candidates(&project.id, &drafts)
            .expect("seed candidates");
        (db, project.id)
    }

    fn selected_ranks(db: &Database, project_id: &str) -> Vec<i64> {
        db.list_candidates(project_id)
            .expect("list candidates")
            .iter()
            .filter(|c| c.selected)
            .map(|c| c.rank)
            .collect()
    }

    #[test]
    fn test_range_selection_one_to_five() {
        let (db, project_id) = test_db_with_candidates(12);
        db.set_selected_rank_range(&project_id, 1, 5)
            .expect("apply range 1-5");
        assert_eq!(selected_ranks(&db, &project_id), vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_range_selection_five_to_ten() {
        let (db, project_id) = test_db_with_candidates(12);
        db.set_selected_rank_range(&project_id, 5, 10)
            .expect("apply range 5-10");
        assert_eq!(selected_ranks(&db, &project_id), vec![5, 6, 7, 8, 9, 10]);
    }

    #[test]
    fn test_range_selection_seven_to_last() {
        let (db, project_id) = test_db_with_candidates(12);
        db.set_selected_rank_range(&project_id, 7, 12)
            .expect("apply range 7-last");
        assert_eq!(selected_ranks(&db, &project_id), vec![7, 8, 9, 10, 11, 12]);
    }

    #[test]
    fn test_range_selection_replaces_previous_selection() {
        let (db, project_id) = test_db_with_candidates(12);
        db.set_selected_rank_range(&project_id, 1, 5)
            .expect("apply range 1-5");
        db.set_selected_rank_range(&project_id, 8, 10)
            .expect("apply range 8-10");
        assert_eq!(selected_ranks(&db, &project_id), vec![8, 9, 10]);
    }

    #[test]
    fn test_range_selection_rejects_reversed_range() {
        let (db, project_id) = test_db_with_candidates(12);
        let err = db
            .set_selected_rank_range(&project_id, 10, 7)
            .expect_err("reversed range must be rejected");
        assert!(
            err.to_string().contains("must not exceed"),
            "unexpected error: {}",
            err
        );
    }

    #[test]
    fn test_range_selection_rejects_start_below_one() {
        let (db, project_id) = test_db_with_candidates(12);
        let err = db
            .set_selected_rank_range(&project_id, 0, 5)
            .expect_err("start < 1 must be rejected");
        assert!(
            err.to_string().contains(">= 1"),
            "unexpected error: {}",
            err
        );
    }

    #[test]
    fn test_range_selection_rejects_end_beyond_total() {
        let (db, project_id) = test_db_with_candidates(12);
        let err = db
            .set_selected_rank_range(&project_id, 7, 13)
            .expect_err("end > total must be rejected");
        assert!(
            err.to_string().contains("exceeds total"),
            "unexpected error: {}",
            err
        );
    }

    #[test]
    fn test_range_selection_single_candidate() {
        let (db, project_id) = test_db_with_candidates(12);
        db.set_selected_rank_range(&project_id, 4, 4)
            .expect("apply single-candidate range");
        assert_eq!(selected_ranks(&db, &project_id), vec![4]);
    }

    #[test]
    fn test_range_selection_full_range() {
        let (db, project_id) = test_db_with_candidates(12);
        db.set_selected_rank_range(&project_id, 1, 12)
            .expect("apply full range");
        assert_eq!(
            selected_ranks(&db, &project_id),
            (1..=12).collect::<Vec<_>>()
        );
    }

    // ── Applied Features tests (Cases A–O from the spec) ───────────────────────

    /// Helper: retrieve the single clip for the first candidate in a project.
    fn first_clip(db: &Database, project_id: &str) -> Clip {
        db.list_clips_for_project(project_id)
            .expect("list clips")
            .into_iter()
            .next()
            .expect("expected at least one clip")
    }

    /// Helper: build a one-candidate DB and return (db, project_id, candidate_id).
    fn test_db_single_candidate() -> (Database, String, String) {
        let db = Database::open(std::path::Path::new(":memory:")).expect("open in-memory db");
        let project = db
            .create_project(
                "/tmp/test.mp4",
                "local",
                "preset_viral_bold",
                "original",
                Some(60.0),
            )
            .expect("create project");
        let candidates = db
            .replace_candidates(
                &project.id,
                &[CandidateDraft {
                    start: 10.0,
                    end: 40.0,
                    score: 0.90,
                    hook: "Test hook".into(),
                    rationale: "Test rationale".into(),
                    ..Default::default()
                }],
            )
            .expect("seed candidates");
        let candidate_id = candidates[0].id.clone();
        (db, project.id, candidate_id)
    }

    /// Persist applied_features JSON and reload; verify round-trip fidelity.
    fn assert_applied_features_round_trip(
        db: &Database,
        project_id: &str,
        candidate_id: &str,
        json: &str,
    ) -> Clip {
        db.update_clip_for_candidate(
            candidate_id,
            "done",
            Some("/tmp/out.mp4"),
            Some("/tmp/out.ass"),
            Some("test log"),
            Some(json),
        )
        .expect("update clip");
        let clip = first_clip(db, project_id);
        assert_eq!(
            clip.applied_features.as_deref(),
            Some(json),
            "applied_features round-trip mismatch"
        );
        clip
    }

    // Case A — all three features NOT applied
    #[test]
    fn test_applied_none() {
        let (db, project_id, candidate_id) = test_db_single_candidate();
        let json =
            r#"{"smartPacing":false,"hookEndingOptimization":false,"audioIntelligence":false}"#;
        let clip = assert_applied_features_round_trip(&db, &project_id, &candidate_id, json);
        let parsed: serde_json::Value =
            serde_json::from_str(clip.applied_features.unwrap().as_str()).unwrap();
        assert_eq!(parsed["smartPacing"], false);
        assert_eq!(parsed["hookEndingOptimization"], false);
        assert_eq!(parsed["audioIntelligence"], false);
    }

    // Case B — Smart Pacing only
    #[test]
    fn test_applied_smart_pacing_only() {
        let (db, project_id, candidate_id) = test_db_single_candidate();
        let json =
            r#"{"smartPacing":true,"hookEndingOptimization":false,"audioIntelligence":false}"#;
        let clip = assert_applied_features_round_trip(&db, &project_id, &candidate_id, json);
        let parsed: serde_json::Value =
            serde_json::from_str(clip.applied_features.unwrap().as_str()).unwrap();
        assert_eq!(parsed["smartPacing"], true);
        assert_eq!(parsed["hookEndingOptimization"], false);
        assert_eq!(parsed["audioIntelligence"], false);
    }

    // Case C — Hook & Ending only
    #[test]
    fn test_applied_hook_ending_only() {
        let (db, project_id, candidate_id) = test_db_single_candidate();
        let json =
            r#"{"smartPacing":false,"hookEndingOptimization":true,"audioIntelligence":false}"#;
        let clip = assert_applied_features_round_trip(&db, &project_id, &candidate_id, json);
        let parsed: serde_json::Value =
            serde_json::from_str(clip.applied_features.unwrap().as_str()).unwrap();
        assert_eq!(parsed["smartPacing"], false);
        assert_eq!(parsed["hookEndingOptimization"], true);
        assert_eq!(parsed["audioIntelligence"], false);
    }

    // Case D — Audio Intelligence only
    #[test]
    fn test_applied_audio_only() {
        let (db, project_id, candidate_id) = test_db_single_candidate();
        let json =
            r#"{"smartPacing":false,"hookEndingOptimization":false,"audioIntelligence":true}"#;
        let clip = assert_applied_features_round_trip(&db, &project_id, &candidate_id, json);
        let parsed: serde_json::Value =
            serde_json::from_str(clip.applied_features.unwrap().as_str()).unwrap();
        assert_eq!(parsed["smartPacing"], false);
        assert_eq!(parsed["hookEndingOptimization"], false);
        assert_eq!(parsed["audioIntelligence"], true);
    }

    // Case E — Smart Pacing + Hook & Ending
    #[test]
    fn test_applied_smart_and_hook() {
        let (db, project_id, candidate_id) = test_db_single_candidate();
        let json =
            r#"{"smartPacing":true,"hookEndingOptimization":true,"audioIntelligence":false}"#;
        let clip = assert_applied_features_round_trip(&db, &project_id, &candidate_id, json);
        let parsed: serde_json::Value =
            serde_json::from_str(clip.applied_features.unwrap().as_str()).unwrap();
        assert_eq!(parsed["smartPacing"], true);
        assert_eq!(parsed["hookEndingOptimization"], true);
        assert_eq!(parsed["audioIntelligence"], false);
    }

    // Case F — Smart Pacing + Audio Intelligence
    #[test]
    fn test_applied_smart_and_audio() {
        let (db, project_id, candidate_id) = test_db_single_candidate();
        let json =
            r#"{"smartPacing":true,"hookEndingOptimization":false,"audioIntelligence":true}"#;
        let clip = assert_applied_features_round_trip(&db, &project_id, &candidate_id, json);
        let parsed: serde_json::Value =
            serde_json::from_str(clip.applied_features.unwrap().as_str()).unwrap();
        assert_eq!(parsed["smartPacing"], true);
        assert_eq!(parsed["hookEndingOptimization"], false);
        assert_eq!(parsed["audioIntelligence"], true);
    }

    // Case G — Hook & Ending + Audio Intelligence
    #[test]
    fn test_applied_hook_and_audio() {
        let (db, project_id, candidate_id) = test_db_single_candidate();
        let json =
            r#"{"smartPacing":false,"hookEndingOptimization":true,"audioIntelligence":true}"#;
        let clip = assert_applied_features_round_trip(&db, &project_id, &candidate_id, json);
        let parsed: serde_json::Value =
            serde_json::from_str(clip.applied_features.unwrap().as_str()).unwrap();
        assert_eq!(parsed["smartPacing"], false);
        assert_eq!(parsed["hookEndingOptimization"], true);
        assert_eq!(parsed["audioIntelligence"], true);
    }

    // Case H — All three applied
    #[test]
    fn test_applied_all_three() {
        let (db, project_id, candidate_id) = test_db_single_candidate();
        let json = r#"{"smartPacing":true,"hookEndingOptimization":true,"audioIntelligence":true}"#;
        let clip = assert_applied_features_round_trip(&db, &project_id, &candidate_id, json);
        let parsed: serde_json::Value =
            serde_json::from_str(clip.applied_features.unwrap().as_str()).unwrap();
        assert_eq!(parsed["smartPacing"], true);
        assert_eq!(parsed["hookEndingOptimization"], true);
        assert_eq!(parsed["audioIntelligence"], true);
    }

    // Case I — All four applied (including Caption Intelligence 2.0)
    #[test]
    fn test_applied_all_four() {
        let (db, project_id, candidate_id) = test_db_single_candidate();
        let json = r#"{"smartPacing":true,"hookEndingOptimization":true,"audioIntelligence":true,"captionIntelligence":true}"#;
        let clip = assert_applied_features_round_trip(&db, &project_id, &candidate_id, json);
        let parsed: serde_json::Value =
            serde_json::from_str(clip.applied_features.unwrap().as_str()).unwrap();
        assert_eq!(parsed["smartPacing"], true);
        assert_eq!(parsed["hookEndingOptimization"], true);
        assert_eq!(parsed["audioIntelligence"], true);
        assert_eq!(parsed["captionIntelligence"], true);
    }

    // Case J — Caption Intelligence only
    #[test]
    fn test_applied_caption_intel_only() {
        let (db, project_id, candidate_id) = test_db_single_candidate();
        let json = r#"{"smartPacing":false,"hookEndingOptimization":false,"audioIntelligence":false,"captionIntelligence":true}"#;
        let clip = assert_applied_features_round_trip(&db, &project_id, &candidate_id, json);
        let parsed: serde_json::Value =
            serde_json::from_str(clip.applied_features.unwrap().as_str()).unwrap();
        assert_eq!(parsed["smartPacing"], false);
        assert_eq!(parsed["hookEndingOptimization"], false);
        assert_eq!(parsed["audioIntelligence"], false);
        assert_eq!(parsed["captionIntelligence"], true);
    }

    // Case K — Caption Intelligence false with other features true
    #[test]
    fn test_applied_caption_intel_false() {
        let (db, project_id, candidate_id) = test_db_single_candidate();
        let json = r#"{"smartPacing":true,"hookEndingOptimization":false,"audioIntelligence":true,"captionIntelligence":false}"#;
        let clip = assert_applied_features_round_trip(&db, &project_id, &candidate_id, json);
        let parsed: serde_json::Value =
            serde_json::from_str(clip.applied_features.unwrap().as_str()).unwrap();
        assert_eq!(parsed["smartPacing"], true);
        assert_eq!(parsed["hookEndingOptimization"], false);
        assert_eq!(parsed["audioIntelligence"], true);
        assert_eq!(parsed["captionIntelligence"], false);
    }

    // Case L — Backward compatibility: legacy 3-key applied_features JSON remains valid
    #[test]
    fn test_applied_features_legacy_three_keys_compatibility() {
        let (db, project_id, candidate_id) = test_db_single_candidate();
        let legacy_json =
            r#"{"smartPacing":true,"hookEndingOptimization":true,"audioIntelligence":true}"#;
        let clip = assert_applied_features_round_trip(&db, &project_id, &candidate_id, legacy_json);
        let parsed: serde_json::Value =
            serde_json::from_str(clip.applied_features.unwrap().as_str()).unwrap();
        // Preserves the existing three keys
        assert_eq!(parsed["smartPacing"], true);
        assert_eq!(parsed["hookEndingOptimization"], true);
        assert_eq!(parsed["audioIntelligence"], true);
        // captionIntelligence is not present in legacy JSON (treated as null/missing)
        assert!(parsed.get("captionIntelligence").is_none());
    }

    // Case M — Full persistence round-trip: save → reload → identical
    #[test]
    fn test_applied_features_round_trip_identity() {
        let (db, project_id, candidate_id) = test_db_single_candidate();
        let json =
            r#"{"smartPacing":true,"hookEndingOptimization":false,"audioIntelligence":true}"#;
        // First write
        db.update_clip_for_candidate(
            &candidate_id,
            "done",
            Some("/out.mp4"),
            None,
            None,
            Some(json),
        )
        .expect("write applied_features");
        let clip1 = first_clip(&db, &project_id);
        // Second read — values must be identical
        let clip2 = first_clip(&db, &project_id);
        assert_eq!(
            clip1.applied_features, clip2.applied_features,
            "applied_features must be stable across reads"
        );
        assert_eq!(clip2.applied_features.as_deref(), Some(json));
    }

    // Case O — Backward compatibility: legacy clip with NULL applied_features is readable
    #[test]
    fn test_applied_features_backward_compat_null() {
        let (db, project_id, candidate_id) = test_db_single_candidate();
        // Simulate a legacy clip: update status without providing applied_features
        db.update_clip_for_candidate(&candidate_id, "done", Some("/out.mp4"), None, None, None)
            .expect("update without applied_features");
        let clip = first_clip(&db, &project_id);
        // Field must be None (NULL) — not a crash, not a default
        assert!(
            clip.applied_features.is_none(),
            "legacy clip with NULL applied_features must deserialize as None, got {:?}",
            clip.applied_features
        );
    }

    // COALESCE preservation: a second update without applied_features must NOT overwrite an existing value
    #[test]
    fn test_applied_features_coalesce_preservation() {
        let (db, project_id, candidate_id) = test_db_single_candidate();
        let json =
            r#"{"smartPacing":true,"hookEndingOptimization":true,"audioIntelligence":false}"#;
        // Write with features
        db.update_clip_for_candidate(
            &candidate_id,
            "done",
            Some("/out.mp4"),
            None,
            None,
            Some(json),
        )
        .expect("first update");
        // Second update (e.g. error recovery) without applied_features
        db.update_clip_for_candidate(&candidate_id, "done", Some("/out2.mp4"), None, None, None)
            .expect("second update");
        let clip = first_clip(&db, &project_id);
        // applied_features must still hold the original JSON (COALESCE semantics)
        assert_eq!(
            clip.applied_features.as_deref(),
            Some(json),
            "COALESCE must preserve existing applied_features when None is passed"
        );
    }

    // ── Payoff Metadata Persistence Tests (prerequisite for Caption Intelligence) ────

    #[test]
    fn test_payoff_metadata_full_round_trip() {
        let db = Database::open(std::path::Path::new(":memory:")).expect("open in-memory db");
        let project = db
            .create_project(
                "/tmp/test.mp4",
                "local",
                "preset_viral_bold",
                "original",
                Some(60.0),
            )
            .expect("create project");

        let draft = CandidateDraft {
            start: 10.0,
            end: 40.0,
            score: 0.95,
            hook: "Test hook".into(),
            rationale: "Test rationale".into(),
            hook_start: Some(10.0),
            hook_end: Some(13.5),
            hook_confidence: Some(0.88),
            opening_context_score: Some(0.91),
            payoff_text: Some("And that is how we solved it.".into()),
            payoff_start: Some(35.0),
            payoff_end: Some(39.5),
            payoff_score: Some(0.92),
            payoff_completion: Some(true),
            ..Default::default()
        };

        let inserted = db
            .replace_candidates(&project.id, &[draft])
            .expect("replace candidates");
        assert_eq!(inserted.len(), 1);
        assert_eq!(
            inserted[0].payoff_text.as_deref(),
            Some("And that is how we solved it.")
        );
        assert_eq!(inserted[0].payoff_start_sec, Some(35.0));
        assert_eq!(inserted[0].payoff_end_sec, Some(39.5));
        assert_eq!(inserted[0].payoff_score, Some(0.92));
        assert_eq!(inserted[0].payoff_completion, Some(true));

        let loaded = db.list_candidates(&project.id).expect("list candidates");
        assert_eq!(loaded.len(), 1);
        assert_eq!(
            loaded[0].payoff_text.as_deref(),
            Some("And that is how we solved it.")
        );
        assert_eq!(loaded[0].payoff_start_sec, Some(35.0));
        assert_eq!(loaded[0].payoff_end_sec, Some(39.5));
        assert_eq!(loaded[0].payoff_score, Some(0.92));
        assert_eq!(loaded[0].payoff_completion, Some(true));
    }

    #[test]
    fn test_payoff_metadata_null_compatibility() {
        let db = Database::open(std::path::Path::new(":memory:")).expect("open in-memory db");
        let project = db
            .create_project(
                "/tmp/test.mp4",
                "local",
                "preset_viral_bold",
                "original",
                Some(60.0),
            )
            .expect("create project");

        // Draft with all payoff fields unset (None)
        let draft = CandidateDraft {
            start: 5.0,
            end: 25.0,
            score: 0.80,
            hook: "Null payoff hook".into(),
            rationale: "Null payoff rationale".into(),
            payoff_text: None,
            payoff_start: None,
            payoff_end: None,
            payoff_score: None,
            payoff_completion: None,
            ..Default::default()
        };

        db.replace_candidates(&project.id, &[draft])
            .expect("replace candidates with None payoff");

        let loaded = db.list_candidates(&project.id).expect("list candidates");
        assert_eq!(loaded.len(), 1);
        assert!(loaded[0].payoff_text.is_none());
        assert!(loaded[0].payoff_start_sec.is_none());
        assert!(loaded[0].payoff_end_sec.is_none());
        assert!(loaded[0].payoff_score.is_none());
        assert!(loaded[0].payoff_completion.is_none());

        // Also test legacy row with NULLs directly in SQLite
        let legacy_id = Uuid::new_v4().to_string();
        {
            let conn = db.conn.lock().unwrap();
            conn.execute(
                "INSERT INTO candidates (id, project_id, start_sec, end_sec, score, hook, rationale, rank, selected)
                 VALUES (?1, ?2, 0.0, 10.0, 0.7, 'legacy hook', 'legacy rat', 2, 0)",
                params![&legacy_id, &project.id],
            ).expect("insert legacy candidate");
        }

        let all_candidates = db
            .list_candidates(&project.id)
            .expect("list all candidates");
        let legacy_candidate = all_candidates
            .iter()
            .find(|c| c.id == legacy_id)
            .expect("find legacy");
        assert!(legacy_candidate.payoff_text.is_none());
        assert!(legacy_candidate.payoff_start_sec.is_none());
        assert!(legacy_candidate.payoff_end_sec.is_none());
        assert!(legacy_candidate.payoff_score.is_none());
        assert!(legacy_candidate.payoff_completion.is_none());
    }

    #[test]
    fn test_payoff_metadata_render_time_read_path() {
        let db = Database::open(std::path::Path::new(":memory:")).expect("open in-memory db");
        let project = db
            .create_project(
                "/tmp/render-source.mp4",
                "local",
                "preset_viral_bold",
                "adaptive",
                Some(120.0),
            )
            .expect("create project");

        let draft = CandidateDraft {
            start: 15.0,
            end: 45.0,
            score: 0.94,
            hook: "Render test hook".into(),
            rationale: "Render test rationale".into(),
            hook_start: Some(15.0),
            hook_end: Some(18.0),
            hook_confidence: Some(0.95),
            opening_context_score: Some(0.89),
            payoff_text: Some("Final punchline delivered perfectly".into()),
            payoff_start: Some(40.0),
            payoff_end: Some(44.5),
            payoff_score: Some(0.96),
            payoff_completion: Some(true),
            ..Default::default()
        };

        let inserted = db
            .replace_candidates(&project.id, &[draft])
            .expect("replace candidates");
        let candidate_id = &inserted[0].id;

        // Verify get_candidate_with_project returns candidate with payoff fields AND intact project fields
        let (loaded_candidate, loaded_project) = db
            .get_candidate_with_project(candidate_id)
            .expect("get candidate with project");

        assert_eq!(loaded_candidate.id, *candidate_id);
        assert_eq!(
            loaded_candidate.payoff_text.as_deref(),
            Some("Final punchline delivered perfectly")
        );
        assert_eq!(loaded_candidate.payoff_start_sec, Some(40.0));
        assert_eq!(loaded_candidate.payoff_end_sec, Some(44.5));
        assert_eq!(loaded_candidate.payoff_score, Some(0.96));
        assert_eq!(loaded_candidate.payoff_completion, Some(true));

        // Verify project fields are not corrupted by the join column shift
        assert_eq!(loaded_project.id, project.id);
        assert_eq!(loaded_project.source_path, "/tmp/render-source.mp4");
        assert_eq!(
            loaded_project.caption_style.as_deref(),
            Some("preset_viral_bold")
        );
        assert_eq!(loaded_project.framing_mode.as_deref(), Some("adaptive"));
        assert_eq!(loaded_project.source_duration, Some(120.0));
    }

    #[test]
    fn test_payoff_metadata_boolean_mapping() {
        let db = Database::open(std::path::Path::new(":memory:")).expect("open in-memory db");
        let project = db
            .create_project(
                "/tmp/test.mp4",
                "local",
                "preset_viral_bold",
                "original",
                Some(60.0),
            )
            .expect("create project");

        let drafts = vec![
            CandidateDraft {
                start: 0.0,
                end: 10.0,
                score: 0.9,
                hook: "Hook True".into(),
                rationale: "Rationale".into(),
                payoff_completion: Some(true),
                ..Default::default()
            },
            CandidateDraft {
                start: 10.0,
                end: 20.0,
                score: 0.8,
                hook: "Hook False".into(),
                rationale: "Rationale".into(),
                payoff_completion: Some(false),
                ..Default::default()
            },
            CandidateDraft {
                start: 20.0,
                end: 30.0,
                score: 0.7,
                hook: "Hook None".into(),
                rationale: "Rationale".into(),
                payoff_completion: None,
                ..Default::default()
            },
        ];

        let inserted = db
            .replace_candidates(&project.id, &drafts)
            .expect("replace candidates");
        assert_eq!(inserted[0].payoff_completion, Some(true));
        assert_eq!(inserted[1].payoff_completion, Some(false));
        assert_eq!(inserted[2].payoff_completion, None);

        let loaded = db.list_candidates(&project.id).expect("list candidates");
        assert_eq!(loaded[0].payoff_completion, Some(true));
        assert_eq!(loaded[1].payoff_completion, Some(false));
        assert_eq!(loaded[2].payoff_completion, None);

        // Verify non-standard non-zero integer in SQLite maps safely to Some(true)
        {
            let conn = db.conn.lock().unwrap();
            conn.execute(
                "UPDATE candidates SET payoff_completion = 2 WHERE id = ?1",
                params![&inserted[2].id],
            )
            .expect("set payoff_completion = 2");
        }
        let reloaded = db.list_candidates(&project.id).expect("reload candidates");
        assert_eq!(reloaded[2].payoff_completion, Some(true));
    }

    #[test]
    fn test_candidate_draft_full_persistence_roundtrip_all_fields() {
        let db = Database::open(std::path::Path::new(":memory:")).expect("open in-memory db");
        let project = db
            .create_project(
                "/tmp/test-persistence.mp4",
                "local",
                "preset_viral_bold",
                "original",
                Some(180.0),
            )
            .expect("create project");

        let draft = CandidateDraft {
            start: 12.5,
            end: 45.2,
            score: 0.94,
            hook: "You won't believe what happened next".into(),
            rationale: "High tension narrative arc with clear payoff".into(),
            hook_start: Some(12.5),
            hook_end: Some(15.0),
            payoff_text: Some("And that is how we survived.".into()),
            payoff_start: Some(40.0),
            payoff_end: Some(45.2),
            structure: Some("story_arc".into()),
            hook_score: Some(0.95),
            coherence_score: Some(0.88),
            payoff_score: Some(0.92),
            question_hook_used: Some(true),
            question_start: Some(12.5),
            question_end: Some(14.0),
            question_text: Some("Why did this happen?".into()),
            question_hook_score: Some(0.91),
            question_relevance_score: Some(0.89),
            answer_start: Some(14.5),
            answer_end: Some(45.0),
            answer_text: Some("Because the system overloaded.".into()),
            answer_strength_score: Some(0.93),
            language: Some("en".into()),
            script: Some("Latin".into()),
            hook_type: Some("curiosity_gap".into()),
            speakers: Some(vec!["Speaker 1".into(), "Speaker 2".into()]),
            summary: Some("Detailed summary of the clip".into()),
            curiosity_score: Some(0.87),
            emotional_impact_score: Some(0.85),
            surprise_score: Some(0.82),
            value_score: Some(0.90),
            story_quality_score: Some(0.86),
            context_completeness_score: Some(0.89),
            shareability_score: Some(0.91),
            hook_speaker: Some("Speaker 1".into()),
            context_text: Some("Earlier they discussed the problem".into()),
            conversation_type: Some("interview".into()),
            start_reason: Some("clear_question".into()),
            end_reason: Some("complete_thought".into()),
            semantic_complete: Some(true),
            interviewer_hook_strength: Some(0.88),
            guest_hook_strength: Some(0.92),
            answer_completeness: Some(0.95),
            semantic_closure: Some(0.90),
            multimodal_verified: true,
            multimodal_status: Some("SUCCESS".into()),
            fallback_label: Some("none".into()),
            multimodal_score: Some(0.93),
            visual_score: Some(0.89),
            audio_score: Some(0.91),
            temporal_score: Some(0.87),
            semantic_score: Some(0.94),
            visual_scene_change: Some(0.75),
            visual_reaction_strength: Some(0.81),
            visual_expression_change: Some(0.83),
            visual_gesture_strength: Some(0.79),
            visual_framing_change: Some(0.72),
            visual_saliency: Some(0.88),
            audio_energy_change: Some(0.85),
            audio_peak_strength: Some(0.90),
            audio_pitch_change: Some(0.78),
            audio_pause_emphasis: Some(0.82),
            audio_laughter: Some(0.0),
            temporal_escalation: Some(0.86),
            temporal_turning_point: Some(0.84),
            temporal_narrative_progression: Some(0.89),
            temporal_payoff_alignment: Some(0.91),
            multimodal_evidence: Some(vec![
                "Scene cut at 12.5s".into(),
                "Pitch spike at 13.2s".into(),
            ]),
            hook_topic_clarity: Some(0.92),
            hook_curiosity: Some(0.94),
            hook_relevance: Some(0.90),
            hook_contrast: Some(0.85),
            hook_confusion_penalty: Some(0.0),
            hook_delay_penalty: Some(0.0),
            hook_irrelevance_penalty: Some(0.0),
            hook_disinterest_penalty: Some(0.0),
            continuation_probability: Some(0.12),
            breath_pause_risk: Some(0.05),
            mid_thought_risk: Some(0.08),
            closure_confidence: Some(0.96),
            answer_complete: Some(true),
            story_completeness: Some(true),
            payoff_completion: Some(true),
            ending_naturalness: Some(0.94),
            ending_type: Some("resolution".into()),
            endpoint_state: Some(crate::models::EndpointState::PayoffComplete),
            raw_llm_end: None,
            hook_confidence: Some(0.95),
            opening_context_score: Some(0.92),
            opening_unresolved_reference: None,
            opening_continuation_marker: None,
            vlm_quality_score: None,
            vlm_visual_engagement: None,
            vlm_semantic_coherence: None,
            vlm_production_quality: None,
            vlm_highlight_relevance: None,
            vlm_model: None,
            vlm_model_version: None,
            vlm_scored_at: None,
            vlm_evidence: None,
            reactions_json: None,
        };

        // 1. Verify replace_candidates returns candidate with metadata_json and try_metadata_draft succeeds
        let inserted = db
            .replace_candidates(&project.id, &[draft.clone()])
            .expect("replace candidates");
        assert_eq!(inserted.len(), 1);
        let candidate = &inserted[0];
        assert!(candidate.metadata_json.is_some());
        let restored_from_insert = candidate
            .try_metadata_draft()
            .expect("parse from inserted candidate");
        assert_eq!(restored_from_insert, draft);

        // 2. Verify list_candidates retrieves metadata_json and try_metadata_draft restores all fields
        let listed = db.list_candidates(&project.id).expect("list candidates");
        assert_eq!(listed.len(), 1);
        assert!(listed[0].metadata_json.is_some());
        let restored_from_list = listed[0]
            .try_metadata_draft()
            .expect("parse from listed candidate");
        assert_eq!(restored_from_list, draft);

        // 3. Verify get_candidate_with_project retrieves metadata_json and project fields accurately
        let (cand_with_proj, proj) = db
            .get_candidate_with_project(&candidate.id)
            .expect("get candidate with project");
        assert_eq!(proj.id, project.id);
        assert_eq!(proj.source_path, "/tmp/test-persistence.mp4");
        assert_eq!(proj.caption_style.as_deref(), Some("preset_viral_bold"));
        assert_eq!(proj.framing_mode.as_deref(), Some("original"));
        assert_eq!(proj.source_duration, Some(180.0));
        assert!(cand_with_proj.metadata_json.is_some());
        let restored_from_proj = cand_with_proj
            .try_metadata_draft()
            .expect("parse from get_candidate_with_project");
        assert_eq!(restored_from_proj, draft);
    }

    /// Phase 4 wiring: prove the advisory VLM + PANNs fields actually survive
    /// the CandidateDraft -> metadata_json -> DB -> CandidateDraft round trip.
    ///
    /// The forensic audit found the VLM fields were written by
    /// `enhance_candidates_with_vlm` but never verified to reach persistence.
    /// This closes requirement J (fields survive to the database) with a real
    /// round trip against an in-memory SQLite database.
    #[test]
    fn test_phase4_advisory_fields_survive_persistence_roundtrip() {
        let db = Database::open(std::path::Path::new(":memory:")).expect("open in-memory db");
        let project = db
            .create_project(
                "/tmp/test-phase4.mp4",
                "local",
                "preset_t7",
                "original",
                Some(120.0),
            )
            .expect("create project");

        let draft = CandidateDraft {
            start: 10.0,
            end: 55.0,
            score: 0.78,
            hook: "phase four hook".into(),
            rationale: "phase four rationale".into(),
            // Advisory VLM scores
            vlm_quality_score: Some(0.81),
            vlm_visual_engagement: Some(0.72),
            vlm_semantic_coherence: Some(0.66),
            vlm_production_quality: Some(0.90),
            vlm_highlight_relevance: Some(0.58),
            vlm_model: Some("qwen/qwen3.8-27b:free".into()),
            vlm_model_version: Some("1.0".into()),
            vlm_scored_at: Some("2026-09-29T12:00:00Z".into()),
            vlm_evidence: Some(vec!["clear hook".into(), "good framing".into()]),
            // Advisory PANNs reaction metadata
            reactions_json: Some(
                r#"{"count":1,"events":[{"type":"Laughter","start":20.0,"end":21.5,"confidence":0.4}]}"#
                    .to_string(),
            ),
            // Authoritative endpoints must round trip untouched.
            payoff_end: Some(55.0),
            ..Default::default()
        };

        let inserted = db
            .replace_candidates(&project.id, &[draft.clone()])
            .expect("persist");

        let restored = inserted[0]
            .try_metadata_draft()
            .expect("restore from metadata_json");

        assert_eq!(restored.vlm_quality_score, Some(0.81));
        assert_eq!(restored.vlm_visual_engagement, Some(0.72));
        assert_eq!(restored.vlm_semantic_coherence, Some(0.66));
        assert_eq!(restored.vlm_production_quality, Some(0.90));
        assert_eq!(restored.vlm_highlight_relevance, Some(0.58));
        assert_eq!(restored.vlm_model.as_deref(), Some("qwen/qwen3.8-27b:free"));
        assert_eq!(restored.vlm_model_version.as_deref(), Some("1.0"));
        assert_eq!(
            restored.vlm_scored_at.as_deref(),
            Some("2026-09-29T12:00:00Z")
        );
        assert_eq!(
            restored.vlm_evidence,
            Some(vec!["clear hook".to_string(), "good framing".to_string()])
        );
        assert!(restored
            .reactions_json
            .as_deref()
            .expect("reactions_json persisted")
            .contains("Laughter"));

        // Full equality proves nothing was silently dropped or coerced.
        assert_eq!(restored, draft);

        // And re-reading from the DB (not just the insert return) works too.
        let listed = db.list_candidates(&project.id).expect("list");
        let listed_restored = listed[0].try_metadata_draft().expect("restore from list");
        assert_eq!(listed_restored.vlm_quality_score, Some(0.81));
        assert_eq!(listed_restored.reactions_json, draft.reactions_json);
    }

    #[test]
    fn test_candidate_corrupted_metadata_json_returns_explicit_error() {
        let db = Database::open(std::path::Path::new(":memory:")).expect("open in-memory db");
        let project = db
            .create_project(
                "/tmp/test-corrupt.mp4",
                "local",
                "preset_viral_bold",
                "original",
                Some(60.0),
            )
            .expect("create project");

        let draft = CandidateDraft {
            start: 5.0,
            end: 25.0,
            score: 0.85,
            hook: "Corrupt test hook".into(),
            rationale: "Corrupt test rationale".into(),
            ..Default::default()
        };

        let inserted = db
            .replace_candidates(&project.id, &[draft])
            .expect("replace candidates");
        let cand_id = &inserted[0].id;

        // Corrupt the metadata JSON directly in SQLite
        {
            let conn = db.conn.lock().unwrap();
            conn.execute(
                "UPDATE candidates SET metadata_json = 'INVALID_MALFORMED_JSON_{{{' WHERE id = ?1",
                params![cand_id],
            )
            .expect("corrupt metadata_json");
        }

        // list_candidates must load the corrupted string, and try_metadata_draft must fail explicitly
        let listed = db.list_candidates(&project.id).expect("list candidates");
        let result = listed[0].try_metadata_draft();
        match result {
            Err(MetadataError::CorruptedJson(err_msg)) => {
                assert!(err_msg.contains("INVALID_MALFORMED_JSON_"));
            }
            Ok(_) => panic!("Expected Err(MetadataError::CorruptedJson), got Ok"),
        }

        // Test empty string / whitespace: should fall back to legacy reconstruction
        {
            let conn = db.conn.lock().unwrap();
            conn.execute(
                "UPDATE candidates SET metadata_json = '   ' WHERE id = ?1",
                params![cand_id],
            )
            .expect("set whitespace metadata_json");
        }
        let listed_whitespace = db.list_candidates(&project.id).expect("list candidates");
        let whitespace_draft = listed_whitespace[0]
            .try_metadata_draft()
            .expect("fallback on whitespace");
        assert_eq!(whitespace_draft.start, 5.0);
        assert_eq!(whitespace_draft.end, 25.0);
        assert_eq!(whitespace_draft.hook, "Corrupt test hook");
        assert_eq!(whitespace_draft.rationale, "Corrupt test rationale");

        // Test NULL: should fall back to legacy reconstruction
        {
            let conn = db.conn.lock().unwrap();
            conn.execute(
                "UPDATE candidates SET metadata_json = NULL WHERE id = ?1",
                params![cand_id],
            )
            .expect("set NULL metadata_json");
        }
        let listed_null = db.list_candidates(&project.id).expect("list candidates");
        let null_draft = listed_null[0]
            .try_metadata_draft()
            .expect("fallback on NULL");
        assert_eq!(null_draft.start, 5.0);
        assert_eq!(null_draft.end, 25.0);
        assert_eq!(null_draft.hook, "Corrupt test hook");
        assert_eq!(null_draft.rationale, "Corrupt test rationale");
    }

    #[test]
    fn test_migration_explicit_error_handling() {
        let db = Database::open(std::path::Path::new(":memory:")).expect("open in-memory db");
        // Running migrate again on the same DB connection should tolerate duplicate columns
        let second_migration_result = db.migrate();
        assert!(
            second_migration_result.is_ok(),
            "Second migration failed: {:?}",
            second_migration_result
        );

        // Verify that unexpected SQLite errors propagate rather than being swallowed
        let conn = db.conn.lock().unwrap();
        let bad_alter = conn.execute("ALTER TABLE nonexistent_table ADD COLUMN test_col TEXT", []);
        let err = bad_alter.unwrap_err();
        let msg = err.to_string().to_lowercase();
        assert!(!msg.contains("duplicate column"));
    }
}
