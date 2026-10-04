# AutoShorts 11.0 Phase 2: Speaker Intelligence — Implementation Plan

## Current Architecture Assessment

### What Already Exists
1. **Diarization via Deepgram**: `diarize=true` in API call → word-level speaker labels (S1, S2, ...)
2. **Transcript Structure**: `TranscriptWord.speaker: Option<String>` populated from Deepgram
3. **SpeakerIdentity Framework**: `speaker_tracker.py` lines 279-315 has `SpeakerIdentity` class with position history
4. **Speaker Turns**: `build_speaker_turns()` creates contiguous speaking segments from diarized words
5. **Visual Tracking**: YOLO11n + BoT-SORT + YuNet with per-shot scale state
6. **Audio Intelligence**: `audio_intelligence.py` with real DSP (RMS, YIN pitch, onset burstiness)

### What's Missing / Incomplete
1. **Diarization Robustness**: No fallback when Deepgram fails/unavailable; no local diarization
2. **Host/Guest Mapping**: No deterministic mapping from S1/S2 → Host/Guest
3. **Speaker Re-ID**: No appearance embeddings (OSNet) for visual track persistence
4. **Active Speaker Fusion**: Visual mouth-motion only; no audio-visual fusion
5. **Caching**: No per-source-video speaker intelligence cache
6. **Schema**: Database lacks speaker intelligence tables

---

## Phase 2A: Diarization Layer

### Design Decisions

**Primary: Deepgram Nova-3 (existing)**
- Already integrated, supports `diarize=true`, `language=multi`
- Cloud API, requires `DEEPGRAM_API_KEY`

**Fallback: pyannote.audio (local)**
- Open-source speaker diarization
- Requires: `torch`, `pyannote.audio`, `huggingface_hub` token for pretrained models
- Runs on CPU (slow) or GPU
- License: MIT for code, models have separate licenses

**Fallback: WhisperX (local)**
- Whisper + forced alignment + diarization
- Requires: `whisperx`, `torch`
- Can use `whisper.cpp` for faster CPU inference

**Selection**: **Deepgram primary, pyannote.audio fallback**
- Deepgram is already working and provides best accuracy
- pyannote is the standard open-source diarization
- Both produce word-level speaker labels compatible with existing `TranscriptWord.speaker`

### Architecture

```
Source Audio
    ↓
[Deepgram Primary] → success → Diarized Transcript
    ↓ fail
[pyannote Fallback] → success → Diarized Transcript
    ↓ fail
[No Diarization] → single speaker "S1" (legacy behavior)
    ↓
[Cache Result] → SpeakerDiarizationResult
    ↓
[Host/Guest Mapping] → ApplicationSpeakerMap
    ↓
[Visual Re-ID + Active Speaker Fusion] → Unified Speaker Intelligence
```

### Data Structures

```rust
// In models.rs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerDiarizationResult {
    pub source_hash: String,          // SHA256 of source audio
    pub model: String,                // "deepgram-nova3" | "pyannote-3.1"
    pub version: String,              // model version
    pub speakers: Vec<DiarizedSpeaker>,
    pub segments: Vec<DiarizedSegment>,
    pub created_at: String,
    pub confidence: f64,              // overall confidence
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiarizedSpeaker {
    pub diarization_id: String,       // "S1", "S2", ...
    pub total_speech_sec: f64,
    pub segment_count: usize,
    pub avg_confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiarizedSegment {
    pub speaker_id: String,           // matches DiarizedSpeaker.diarization_id
    pub start: f64,
    pub end: f64,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApplicationSpeakerMap {
    pub source_hash: String,
    pub mappings: Vec<SpeakerMapping>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerMapping {
    pub diarization_id: String,       // "S1"
    pub application_role: ApplicationSpeakerRole,
    pub confidence: f64,
    pub evidence: MappingEvidence,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ApplicationSpeakerRole {
    Host,
    Guest,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MappingEvidence {
    pub visual_track_id: Option<i32>,
    pub speaking_time_ratio: f64,
    pub first_speaker: bool,
    pub question_asker: bool,
    pub user_override: bool,
}
```

---

## Phase 2B: Host/Guest Mapping

### Mapping Logic (Deterministic, Evidence-Based)

**Priority Order:**
1. **User Override** (explicit UI designation) — highest confidence
2. **Visual Track Identity** — if diarization speaker consistently maps to a visual track
3. **Question Asker** — the speaker who asks questions is likely Host
4. **First Speaker** — in interview format, first speaker often Host
5. **Speaking Time Ratio** — Host typically speaks less in guest-driven content
6. **Unknown** — if insufficient evidence

### Implementation

```rust
fn map_diarization_to_application(
    diarization: &SpeakerDiarizationResult,
    transcript: &NormalizedTranscript,
    visual_tracks: Option<&VisualTrackData>,
    user_override: Option<HashMap<String, ApplicationSpeakerRole>>,
) -> ApplicationSpeakerMap {
    // 1. Apply user override
    // 2. Cross-reference with visual tracks (speaker_tracker.py track IDs)
    // 3. Analyze transcript for question patterns
    // 4. Apply heuristics
    // 5. Return map with confidence per mapping
}
```

---

## Phase 2C: Speaker Re-Identification (OSNet)

### Model Selection: OSNet (torchreid)

**Why OSNet:**
- Proven person re-identification architecture
- Available in `torchreid` library (MIT license)
- Lightweight: ~2.2M parameters (osnet_x1_0)
- ONNX exportable for production
- Pre-trained on Market1501, DukeMTMC-reID, MSMT17

**Integration Strategy:**
- **Not a replacement** for positional tracking (BoT-SORT)
- **Additive signal**: embedding distance combined with spatial/temporal cost
- **Per-track embedding**: compute once per track, update with EMA
- **Gallery management**: maintain embedding gallery per source video

### Architecture

```python
# In speaker_tracker.py (new module: reid.py)

class ReIDManager:
    def __init__(self, model_name="osnet_x1_0", device="cpu"):
        self.model = self._load_model(model_name, device)
        self.gallery = {}  # track_id -> embedding (EMA)
        self.embedding_dim = 512
    
    def extract_embedding(self, frame, bbox) -> np.ndarray:
        # crop, resize to 256x128, normalize, forward pass
        pass
    
    def update_track(self, track_id: int, frame, bbox):
        # extract embedding, EMA update gallery
        pass
    
    def match_tracks(self, track_a: int, track_b: int) -> float:
        # cosine similarity between gallery embeddings
        pass
    
    def resolve_identity(self, new_track_id: int, candidate_tracks: List[int]) -> Optional[int]:
        # find best match in gallery above threshold
        pass
```

### Fusion with BoT-SORT

```python
# In solve_dual_frame_trajectories / resolve_visual_subject:
# Existing: spatial IoU + Kalman prediction
# New: add embedding distance as cost term

association_cost = spatial_cost * 0.7 + embedding_cost * 0.3
# Only use embedding when spatial association is ambiguous
```

---

## Phase 2D: Active Speaker Fusion

### Current State
- Visual: mouth-motion score per face track (YuNet landmarks)
- Audio: None (no diarization integration)

### Fusion Architecture

```
For each time interval:
    Audio Evidence:
        - Diarization speaker active? (from SpeakerDiarizationResult)
        - Speech activity detection (silencedetect / RMS)
        - Confidence from diarization
    
    Visual Evidence:
        - Mouth motion score (YuNet landmarks)
        - Face visibility / frontality
        - Face track identity
    
    Identity Evidence:
        - Persistent Re-ID track ID
        - Track continuity
    
    Temporal Evidence:
        - Previous active speaker
        - Switch latency / hysteresis
    
    Fusion:
        - Weighted combination with confidence gating
        - Disagreement handling (audio vs visual)
        - Output: active speaker probability per track
```

### Implementation

```python
# In speaker_tracker.py (new function)
def compute_active_speaker_fusion(
    diarization_segments: List[DiarizedSegment],
    visual_tracks: Dict[int, TrackData],
    reid_gallery: ReIDGallery,
    config: FusionConfig,
) -> List[ActiveSpeakerState]:
    # 1. Align time grids
    # 2. For each track, compute audio evidence (overlap with diarization)
    # 3. For each track, compute visual evidence (mouth motion)
    # 4. Combine with temporal hysteresis
    # 5. Return active speaker state per track per interval
```

---

## Data Schema Extensions

### Database Migrations (db.rs)

```sql
-- Speaker diarization cache (per source video)
CREATE TABLE IF NOT EXISTS speaker_diarization (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    source_hash TEXT NOT NULL,
    model TEXT NOT NULL,
    version TEXT NOT NULL,
    speakers_json TEXT NOT NULL,      -- Vec<DiarizedSpeaker>
    segments_json TEXT NOT NULL,      -- Vec<DiarizedSegment>
    confidence REAL,
    created_at TEXT NOT NULL,
    UNIQUE(project_id, source_hash, model, version)
);

-- Host/Guest mapping
CREATE TABLE IF NOT EXISTS speaker_mapping (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    diarization_id TEXT NOT NULL,
    application_role TEXT NOT NULL,  -- "host", "guest", "unknown"
    confidence REAL NOT NULL,
    evidence_json TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(project_id, diarization_id)
);

-- Visual track identity with Re-ID embeddings
CREATE TABLE IF NOT EXISTS visual_track_identity (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    track_id INTEGER NOT NULL,
    reid_embedding BLOB,             -- serialized f32 array (512 dims)
    embedding_model TEXT,            -- "osnet_x1_0"
    first_seen_sec REAL,
    last_seen_sec REAL,
    total_detections INTEGER,
    created_at TEXT NOT NULL
);

-- Active speaker fusion results (per candidate clip)
CREATE TABLE IF NOT EXISTS active_speaker_fusion (
    id TEXT PRIMARY KEY,
    candidate_id TEXT NOT NULL REFERENCES candidates(id) ON DELETE CASCADE,
    intervals_json TEXT NOT NULL,    -- Vec<ActiveSpeakerInterval>
    model_version TEXT,
    created_at TEXT NOT NULL
);
```

---

## Caching Strategy

### Cache Key Composition
```
speaker_intelligence_v1:
  source_sha256: <SHA256 of source video>
  diarization_model: "deepgram-nova3"
  diarization_version: "2024-01"
  reid_model: "osnet_x1_0"
  fusion_version: "1.0"
  config_hash: <hash of FramingConfig + FusionConfig>
```

### Cache Behavior
- **Write-once**: computed on first render/analysis, stored in SQLite
- **Read-many**: reused for all candidates from same source
- **Invalidation**: any config/model change → new cache key
- **Size bound**: embeddings ~512 floats × max 50 tracks = ~100KB per video

---

## Integration Points

### 1. Transcription Pipeline (transcription.rs)
- After Deepgram normalization, run diarization cache check
- If cache miss, run Deepgram (already does diarize) or pyannote fallback
- Store `SpeakerDiarizationResult` in DB

### 2. Candidate Generation (lib.rs → generate_candidates)
- Load diarization result
- Run Host/Guest mapping
- Attach `application_role` to `CandidateDraft.hook_speaker` / `conversation_type`

### 3. Visual Tracking (speaker_tracker.py)
- Initialize ReID manager
- Extract embeddings per track per detection
- Update gallery with EMA
- Use embeddings in association cost

### 4. Active Speaker Fusion
- New Python sidecar: `active_speaker_fusion.py`
- Input: diarization segments + visual tracks + ReID gallery
- Output: `ActiveSpeakerState` per track per interval
- Consumed by framing solver for identity-aware decisions

### 5. Render Pipeline (lib.rs → render_flat_clip_for_candidate)
- Load fusion result for candidate
- Pass to caption generator for speaker-aware metadata
- Store fusion result in DB

---

## Model Licensing & Availability

| Model | License | Commercial Use | Weights Source |
|-------|---------|----------------|----------------|
| Deepgram Nova-3 | Proprietary (API) | Yes (paid) | Cloud API |
| pyannote.audio 3.1 | MIT (code) / Model-specific | Check per model | HuggingFace (pyannote/segmentation, pyannote/speaker-diarization) |
| OSNet (torchreid) | MIT | Yes | torchreid model zoo (Market1501 pretrained) |
| YuNet (existing) | Apache-2.0 | Yes | OpenCV Zoo |
| YOLO11n (existing) | AGPL-3.0 | Requires Enterprise license | Ultralytics |

**Action Required:**
- pyannote: Verify model licenses (pyannote/segmentation-3.0, pyannote/speaker-diarization-3.1)
- OSNet: MIT license confirmed safe
- Deepgram: Already in use
- Document all in `docs/license_audit/phase2_license_audit.md`

---

## Development Sequence

### Step 1: Core Data Structures & Database (Week 1)
- [ ] Add `SpeakerDiarizationResult`, `ApplicationSpeakerMap`, `SpeakerMapping` to `models.rs`
- [ ] Add database migrations in `db.rs` for new tables
- [ ] Add cache key utilities

### Step 2: Diarization Engine (Week 1-2)
- [ ] Create `speaker_diarization.py` sidecar with Deepgram + pyannote
- [ ] Implement cache check/load/save in Rust
- [ ] Integrate into `transcription.rs` pipeline
- [ ] Add fallback logic

### Step 3: Host/Guest Mapping (Week 2)
- [ ] Implement deterministic mapping algorithm in Rust
- [ ] Cross-reference with visual tracks from `speaker_tracker.py`
- [ ] Store `ApplicationSpeakerMap` in DB

### Step 4: Re-ID Integration (Week 2-3)
- [ ] Add `torchreid` + OSNet to Python requirements
- [ ] Create `reid.py` module with `ReIDManager`
- [ ] Integrate embedding extraction in `speaker_tracker.py` detection loop
- [ ] Add embedding gallery and association cost fusion

### Step 5: Active Speaker Fusion (Week 3)
- [ ] Create `active_speaker_fusion.py` sidecar
- [ ] Implement fusion algorithm with hysteresis
- [ ] Output `ActiveSpeakerState` per interval
- [ ] Integrate with `speaker_tracker.py` for identity-aware framing

### Step 6: Pipeline Integration (Week 3-4)
- [ ] Wire diarization into `lib.rs` transcription flow
- [ ] Wire Host/Guest mapping into candidate generation
- [ ] Wire Re-ID into `speaker_tracker.py` run()
- [ ] Wire fusion into framing/render path
- [ ] Add feature flags: `AUTOSHORTS_DIARIZATION`, `AUTOSHORTS_REID`, `AUTOSHORTS_ACTIVE_SPEAKER_FUSION`

### Step 7: Testing & Validation (Week 4)
- [ ] Unit tests for each component
- [ ] Real-media validation on multi-speaker content
- [ ] Regression test suite (all existing tests must pass)
- [ ] Performance benchmarking

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| pyannote model license issues | Verify HuggingFace model cards; have WhisperX as alternative |
| OSNet GPU dependency | Ensure CPU fallback; ONNX export for DirectML |
| Re-ID embedding drift | EMA gallery update; confidence threshold for matching |
| Fusion disagreement | Default to visual (existing behavior); log disagreements |
| Cache invalidation | Config hash in cache key; explicit version fields |
| Performance | Lazy load models; cache per source video; single pass per video |

---

## Acceptance Criteria

### Diarization
- [ ] Deepgram path works (existing)
- [ ] pyannote fallback activates on Deepgram failure
- [ ] Cache hit avoids re-running diarization
- [ ] Single-speaker fallback when all diarization fails

### Host/Guest Mapping
- [ ] User override works
- [ ] Visual track cross-reference works
- [ ] Question-asker heuristic identifies Host
- [ ] Unknown state preserved when uncertain

### Re-ID
- [ ] OSNet loads and extracts 512-d embeddings
- [ ] Gallery maintains per-track embeddings
- [ ] Embedding distance improves track association in ambiguous cases
- [ ] No regression when Re-ID disabled/unavailable

### Active Speaker Fusion
- [ ] Audio diarization + visual mouth motion combined
- [ ] Hysteresis prevents rapid switching
- [ ] Disagreement logged but doesn't crash
- [ ] Falls back to visual-only when diarization unavailable

### Regression
- [ ] All 321 Rust tests pass
- [ ] All 20+ Python test suites pass
- [ ] npm build passes
- [ ] Real-media validation on Jensen Huang interview (2 speakers) + at least 2 more multi-speaker sources

---

## File Changes Summary

### New Files
- `autoshorts/src-tauri/src/speaker_intelligence.rs` — Core Rust types + logic
- `autoshorts/src-tauri/scripts/speaker_diarization.py` — Deepgram + pyannote sidecar
- `autoshorts/src-tauri/scripts/reid.py` — OSNet embedding extraction
- `autoshorts/src-tauri/scripts/active_speaker_fusion.py` — Fusion sidecar

### Modified Files
- `autoshorts/src-tauri/src/models.rs` — New structs
- `autoshorts/src-tauri/src/db.rs` — Migrations + queries
- `autoshorts/src-tauri/src/transcription.rs` — Diarization integration
- `autoshorts/src-tauri/src/lib.rs` — Pipeline wiring
- `autoshorts/src-tauri/src/media.rs` — Multimodal + render integration
- `autoshorts/src-tauri/scripts/speaker_tracker.py` — Re-ID + fusion integration
- `autoshorts/src-tauri/Cargo.toml` — Any new deps (minimal)

### Documentation
- `docs/license_audit/phase2_license_audit.md`
- `autoshorts_11_phase_2_implementation_and_validation_report.md`
- `AUTOSHORTS_11_0_BRAIN.md` (append Phase 2 section)

---

## Next Steps

1. **Start with Step 1**: Add core data structures to `models.rs` and database migrations to `db.rs`
2. **Implement diarization sidecar** with Deepgram primary / pyannote fallback
3. **Build incrementally** — each step must pass regression tests before proceeding