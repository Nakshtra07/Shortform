"""
AutoShorts Smart Reframing Engine v12.0 — True Shot-Aware Scale + Position + Composition Optimization

Core Architectural Invariants:

1. THREE SEPARATE DECISIONS:
   A. Subject Tracking     — Who is speaking and where? (YOLO / YuNet, unchanged)
   B. Crop Position        — Where should the portrait window be? (x, y)
   C. Crop Scale           — How large should the subject appear? (crop_w, crop_h)

   Face tracking does NOT automatically determine scale.
   Speaker movement does NOT cause scale changes.

2. REAL DYNAMIC SCALE:
   crop_w(t) and crop_h(t) are genuine rendering parameters.
   scale = 1.0  -> baseline portrait crop (ih * 9/16)
   scale = 1.15 -> crop_w /= 1.15 (tighter real crop, more zoom)
   scale = 0.90 -> crop_w /= 0.90 (wider real crop, less zoom)
   Aspect ratio 9:16 always preserved. Output always scaled to 1080x1920.

3. SHOT-LOCAL SCALE STATE:
   Each shot computes its own scale independently.
   At shot change: RESET scale state entirely.
   Previous shot scale is NEVER authoritative for the next shot.

4. JOINT (POSITION, SCALE) SOLVING:
   Candidate (crop_x, crop_y, scale) combinations scored together.

5. SUBJECT CONTAINMENT != SUBJECT CENTERING:
   Containment is a HARD CONSTRAINT.
   Centering is a SOFT PREFERENCE with LOW WEIGHT.

6. COMPOSITIONAL DEADBAND:
   If current position + scale produce a valid composition:
   camera remains STATIONARY (zero artificial movement).

7. ANTI-OVERZOOM:
   Closer is NOT better. Prefer MINIMUM scale satisfying composition.

8. OUTPUT FORMAT:
   JSON line: {"x": "<ffmpeg_expr>", "y": "<y_expr>", "w": "<w_expr>", "h": "<h_expr>"}
   media.rs parses this JSON to populate all four crop parameters.
"""

import sys
import os
import traceback
import time
import json
import math
import argparse
import subprocess
try:
    import cv2
except ImportError:
    sys.stderr.write(
        "[Smart Framing] Smart Framing dependency error: OpenCV (cv2) is not installed in the Python environment used by AutoShorts.\n"
        f"[Smart Framing] Interpreter: {sys.executable}\n"
        "[Smart Framing] Please run: python -m pip install -r requirements.txt (or python -m pip install opencv-python)\n"
    )
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    sys.stderr.write(
        "[Smart Framing] Smart Framing dependency error: NumPy is not installed in the Python environment used by AutoShorts.\n"
        f"[Smart Framing] Interpreter: {sys.executable}\n"
        "[Smart Framing] Please run: python -m pip install -r requirements.txt (or python -m pip install numpy)\n"
    )
    sys.exit(1)

try:
    from scipy.signal import savgol_filter
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ─── Dual-Engine Vision Layer Bridge ──────────────────────────────────────────
HAS_ULTRALYTICS = False
HAS_TORCH = False
from dataclasses import dataclass, field
from typing import Optional, Any, Dict, List, Sequence, Tuple

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import ultralytics
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False

# ─── Phase 2: Active Speaker Fusion bridge (optional, fallback-safe) ─────────
# active_speaker_fusion.py provides the diarization + mouth-motion + temporal
# fusion. If it is missing or fails to import, the deterministic mouth-motion
# heuristic remains the sole active-speaker evidence (documented fallback).
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
HAS_ACTIVE_SPEAKER_FUSION = False
_asf = None
try:
    import active_speaker_fusion as _asf
    HAS_ACTIVE_SPEAKER_FUSION = True
except Exception as _e:
    sys.stderr.write(f"[SpeakerIntel] fusion module unavailable ({_e}); heuristic fallback active\n")

_ULTRALYTICS_YOLO_MODEL = None

def find_yolo_model():
    """Locate yolo11n.pt weights in project or standard locations."""
    if "AUTOSHORTS_YOLO_MODEL" in os.environ:
        p = os.environ["AUTOSHORTS_YOLO_MODEL"].strip()
        if os.path.exists(p):
            return p

    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        # Relative to scripts dir
        os.path.join(base_dir, "..", "models", "yolo11n.pt"),
        os.path.join(base_dir, "..", "..", "models", "yolo11n.pt"),
        os.path.join(base_dir, "..", "..", "..", "models", "yolo11n.pt"),
        os.path.join(base_dir, "yolo11n.pt"),
    ]

    # Discover in %LOCALAPPDATA%/autoshorts/models
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        candidates.append(os.path.join(local_app_data, "autoshorts", "models", "yolo11n.pt"))

    # Cross-platform user profile fallbacks
    user_home = os.path.expanduser("~")
    candidates.append(os.path.join(user_home, "AppData", "Local", "autoshorts", "models", "yolo11n.pt"))
    candidates.append(os.path.join(user_home, ".local", "share", "autoshorts", "models", "yolo11n.pt"))

    for c in candidates:
        norm = os.path.normpath(c)
        if os.path.exists(norm) and os.path.getsize(norm) > 1000000:
            return norm
    return None


def get_ultralytics_model():
    """Returns singleton YOLO11n instance or None."""
    global _ULTRALYTICS_YOLO_MODEL
    if _ULTRALYTICS_YOLO_MODEL is None and HAS_ULTRALYTICS:
        model_path = find_yolo_model()
        if model_path:
            try:
                _ULTRALYTICS_YOLO_MODEL = YOLO(model_path)
            except Exception as e:
                sys.stderr.write(f"[CV Backend] Failed to load Ultralytics YOLO model from {model_path}: {e}\n")
                _ULTRALYTICS_YOLO_MODEL = False
        else:
            sys.stderr.write("[CV Backend] yolo11n.pt weights not found; Ultralytics tracker disabled\n")
            _ULTRALYTICS_YOLO_MODEL = False
    return _ULTRALYTICS_YOLO_MODEL if _ULTRALYTICS_YOLO_MODEL else None


# ─── Re-ID Integration (OSNet) ────────────────────────────────────────────────
HAS_TORCHREID = False
try:
    import torchreid
    HAS_TORCHREID = True
except ImportError:
    HAS_TORCHREID = False

# ─── Re-ID Model Configuration ────────────────────────────────────────────────
REID_MODEL_NAME = "osnet_x1_0"
REID_INPUT_SIZE = (256, 128)
REID_MEAN = [0.485, 0.456, 0.406]
REID_STD = [0.229, 0.224, 0.225]
REID_DEVICE = "cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu"
REID_BATCH_SIZE = 16
EMA_ALPHA = 0.9
MIN_DETECTIONS_FOR_GALLERY = 3

# ─── Re-ID Model Loading ──────────────────────────────────────────────────────
_HAS_TORCHREID = False
_REID_MODEL = None

def _load_reid_model():
    """Load OSNet model from torchreid."""
    global _HAS_TORCHREID, _REID_MODEL
    if _REID_MODEL is not None:
        return _REID_MODEL
    if not HAS_TORCHREID:
        return None
    try:
        import torchreid
        _HAS_TORCHREID = True
    except ImportError:
        return None
    
    print(f"[ReID] Loading {REID_MODEL_NAME} on {REID_DEVICE}...", file=sys.stderr)
    
    model = torchreid.models.build_model(
        name=REID_MODEL_NAME,
        num_classes=1000,
        loss="softmax",
        pretrained=True
    )
    model.eval()
    model.to(REID_DEVICE)
    
    _REID_MODEL = model
    print(f"[ReID] Model loaded successfully", file=sys.stderr)
    return model

def preprocess_person_crop(crop: np.ndarray, target_size: Tuple[int, int] = REID_INPUT_SIZE) -> Optional[torch.Tensor]:
    """Preprocess a person crop for OSNet input."""
    if crop is None or crop.size == 0:
        return None
    
    h, w = crop.shape[:2]
    if h == 0 or w == 0:
        return None
    
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    
    target_h, target_w = REID_INPUT_SIZE
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(crop_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    pad_top = (target_h - new_h) // 2
    pad_left = (target_w - new_w) // 2
    padded[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
    
    normalized = padded.astype(np.float32) / 255.0
    mean = np.array(REID_MEAN, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(REID_STD, dtype=np.float32).reshape(1, 1, 3)
    normalized = (normalized - mean) / std
    
    tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0).float()
    return tensor

def extract_embedding_batch(model: torch.nn.Module, crops: List[np.ndarray], device: str) -> List[Optional[np.ndarray]]:
    """Extract embeddings for a batch of person crops."""
    if not crops:
        return []
    
    tensors = []
    valid_indices = []
    
    for i, crop in enumerate(crops):
        tensor = preprocess_person_crop(crop)
        if tensor is not None:
            tensors.append(tensor)
            valid_indices.append(i)
    
    if not tensors:
        return [None] * len(crops)
    
    batch = torch.cat(tensors, dim=0).to(REID_DEVICE)
    
    with torch.no_grad():
        features = model(batch)
        if features.dim() == 4:
            features = F.adaptive_avg_pool2d(features, (1, 1))
            features = features.view(features.size(0), -1)
        features = F.normalize(features, p=2, dim=1)
    
    results = [None] * len(crops)
    for idx, feat_idx in enumerate(valid_indices):
        results[feat_idx] = features[idx].cpu().numpy().astype(np.float32)
    
    return results

# ─── Integer crop-window helper (float-slice regression guard) ───────────────

def int_crop_window_xyxy(frame: np.ndarray, bbox: Sequence[float]):
    """Convert an XYXY bounding box into an in-bounds INTEGER (x, y, w, h) crop window.

    Every crop in this module slices ``frame[y:y + h, x:x + w]``. Python's
    ``slice`` rejects ``float`` bounds with
    ``TypeError: slice indices must be integers or None or have an __index__
    method``, and numpy scalars do not reliably implement ``__index__`` either.
    Detections (Ultralytics ``boxes.xyxy``, YuNet faces) are float-valued, so
    bounds MUST be coerced with ``int()`` before slicing.

    Returns ``(x, y, w, h)`` as plain Python ints, clamped to the frame, or
    ``None`` when the box is degenerate (zero/negative extent) or falls entirely
    outside the frame.

    This is the single sanctioned entry point for building crop windows; all
    call sites must route through it so the float-slice TypeError cannot recur.
    """
    if frame is None or bbox is None:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return None
    if not all(map(math.isfinite, (x1, y1, x2, y2))):
        return None

    fh, fw = int(frame.shape[0]), int(frame.shape[1])
    # Normalise so inverted boxes still yield a positive-extent window.
    lo_x, hi_x = min(x1, x2), max(x1, x2)
    lo_y, hi_y = min(y1, y2), max(y1, y2)
    if hi_x <= lo_x or hi_y <= lo_y:
        return None

    x = int(max(0, min(lo_x, fw - 1)))
    y = int(max(0, min(lo_y, fh - 1)))
    # Clamp the far edge to the frame, then force an integer extent of >= 1.
    w = int(max(1, min(int(math.ceil(hi_x)), fw - x)))
    h = int(max(1, min(int(math.ceil(hi_y)), fh - y)))
    return x, y, w, h


# ─── Re-ID Manager ─────────────────────────────────────────────────────────────

@dataclass
class TrackEmbedding:
    track_id: int
    embedding: np.ndarray
    detection_count: int
    first_seen: float
    last_seen: float
    updated_at: str

class ReIDManager:
    """Manages OSNet embeddings for visual tracks."""
    
    def __init__(self, model_name: str = REID_MODEL_NAME, device: str = REID_DEVICE, ema_alpha: float = EMA_ALPHA):
        self.model = None
        if HAS_TORCHREID:
            self.model = _load_reid_model()
        self.device = REID_DEVICE
        self.ema_alpha = EMA_ALPHA
        self.gallery: Dict[int, TrackEmbedding] = {}
        self.model_name = REID_MODEL_NAME
    
    def extract_embeddings(self, frame: np.ndarray, tracks: List[Dict]) -> List[Optional[np.ndarray]]:
        """Extract embeddings for a list of tracks in a frame.

        Each track's ``bbox`` is an XYXY box (x1, y1, x2, y2) — the same
        convention Ultralytics ``boxes.xyxy`` and YuNet faces use. Crop windows
        go through :func:`int_crop_window_xyxy`, which guarantees integer slice
        bounds; passing floats here raises ``TypeError: slice indices must be
        integers`` and aborts the whole shot.
        """
        if self.model is None:
            return [None] * len(tracks)
        
        crops = []
        for track in tracks:
            bbox = track.get("bbox")
            if bbox is None:
                crops.append(None)
                continue
            window = int_crop_window_xyxy(frame, bbox)
            if window is None:
                crops.append(None)
                continue
            x, y, w, h = window
            crop = frame[y:y + h, x:x + w]
            crops.append(crop if crop.size else None)
        
        if not any(c is not None for c in crops):
            return [None] * len(tracks)
        
        embeddings = extract_embedding_batch(self.model, crops, REID_DEVICE)
        return embeddings
    
    def update_gallery(self, track_id: int, embedding: np.ndarray, timestamp: float):
        """Update track embedding with EMA."""
        if track_id in self.gallery:
            entry = self.gallery[track_id]
            entry.embedding = self.ema_alpha * entry.embedding + (1 - self.ema_alpha) * embedding
            entry.embedding = entry.embedding / np.linalg.norm(entry.embedding)
            entry.detection_count += 1
            entry.last_seen = timestamp
            entry.updated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        else:
            self.gallery[track_id] = TrackEmbedding(
                track_id=track_id,
                embedding=embedding,
                detection_count=1,
                first_seen=timestamp,
                last_seen=timestamp,
                updated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            )
    
    def get_embedding(self, track_id: int) -> Optional[np.ndarray]:
        if track_id in self.gallery:
            return self.gallery[track_id].embedding
        return None
    
    def compute_similarity(self, track_id_a: int, track_id_b: int) -> float:
        emb_a = self.get_embedding(track_id_a)
        emb_b = self.get_embedding(track_id_b)
        if emb_a is None or emb_b is None:
            return 0.0
        return float(np.dot(emb_a, emb_b))
    
    def find_best_match(self, track_id: int, candidate_ids: List[int], threshold: float = 0.7) -> Optional[int]:
        if track_id not in self.gallery:
            return None
        best_id = None
        best_sim = threshold
        for cand_id in candidate_ids:
            if cand_id == track_id:
                continue
            sim = self.compute_similarity(track_id, cand_id)
            if sim > best_sim:
                best_sim = sim
                best_id = cand_id
        return best_id
    
    def get_gallery_embeddings(self) -> Dict[int, TrackEmbedding]:
        return self.gallery.copy()

# ─── Re-ID Manager Instance ────────────────────────────────────────────────────
_REID_MANAGER = None

def get_reid_manager() -> ReIDManager:
    global _REID_MANAGER
    if _REID_MANAGER is None:
        _REID_MANAGER = ReIDManager()
    return _REID_MANAGER


# ─── Phase 2: Speaker Intelligence runtime helpers ────────────────────────────
# Active-speaker fusion consumes DIARIZATION + MOUTH MOTION + RE-ID IDENTITY +
# TEMPORAL CONTEXT and produces per-track ActiveSpeakerState intervals. The
# deterministic mouth-motion heuristic stays available: every helper fails
# soft (returns None / empty) and every consumer falls back to the heuristic
# when fusion evidence is missing, uncertain, or disabled.

FUSION_ENV_MASTER = "AUTOSHORTS_SPEAKER_INTELLIGENCE"
FUSION_ENV_FLAG = "AUTOSHORTS_ACTIVE_SPEAKER_FUSION"
REID_ENV_FLAG = "AUTOSHORTS_REID"
# Mouth-motion scores are normalized into 0..1 for the fusion module by
# dividing by the heuristic's own activity threshold (documented calibration
# reuse — no new magic constant).
FUSION_MOTION_NORM = 4.0
# Track ids restart at 0 every shot; gallery keys are namespaced as
# shot_idx * SHOT_TRACK_ID_STRIDE + local_tid so different people tracked
# under the same local id in different shots never share an EMA entry.
SHOT_TRACK_ID_STRIDE = 10000


def shot_gallery_key(shot_idx: int, local_tid: int) -> int:
    return int(shot_idx) * SHOT_TRACK_ID_STRIDE + int(local_tid)


# ─── Phase 3: Scene intelligence + smoothing + crop scorer helpers ───────────
# All advisory: the deterministic framing engine, its hard containment, and
# its invariants remain authoritative. Every helper fails soft.

AUTFLIP_MAX_PAN_SPEED = 720.0  # px/s bound for smoothstep transitions
SCENE_STATIC_DIFF = 6.0        # mean 160x90 gray diff below which a shot is static
SCENE_STATIC_SCALE = 1.25      # reframe-threshold scale for static shots
SCENE_STATIC_SCALE_CAP = 1.5   # hard cap on the static-shot scaling


def framing_smoothing_enabled() -> bool:
    return _env_flag_enabled("AUTOSHORTS_FRAMING_SMOOTHING")


def crop_scorer_enabled() -> bool:
    """Opt-in experimental advisory (default OFF): the composition scorer is
    config-independent, so enabling it must remain an explicit choice — it
    must never silently flatten FramingConfig-driven behavior (guarded by
    test_framing_config_suite)."""
    v = os.environ.get("AUTOSHORTS_CROP_SCORER")
    if v is None:
        return False
    return v.strip().lower() not in ("0", "false", "off")


CROP_SCORER_MIN_GAIN = 1.0  # required score improvement before overriding the solved x


def scene_intelligence_enabled_py() -> bool:
    return _env_flag_enabled("AUTOSHORTS_SCENE_INTELLIGENCE")


def load_scene_cuts_json(path, start_sec: float, clip_dur: float, pixel_cuts):
    """Merge cached source-level PySceneDetect boundaries (absolute source
    seconds) with the tracker's own pixel-diff cuts. Returns the merged,
    debounced, clip-relative cut list (first element 0.0) — the exact contract
    of the internal detector. Missing/corrupt file -> pixel cuts unchanged."""
    merged = sorted({0.0} | {float(c) for c in (pixel_cuts or [])})
    if not path:
        return merged
    try:
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)
        external = []
        for s in doc.get("scenes", []) or []:
            rel = float(s.get("start", 0.0)) - float(start_sec)
            if 0.0 < rel < float(clip_dur) - 0.35:
                external.append(rel)
        merged = sorted({0.0} | {float(c) for c in (pixel_cuts or [])} | set(external))
        # Same 0.35 s debounce as the internal detector.
        debounced = [merged[0]]
        for c in merged[1:]:
            if c - debounced[-1] > 0.35:
                debounced.append(c)
        if external:
            sys.stderr.write(f"[SceneIntel] merged {len(external)} cached scene boundary(ies)\n")
        return debounced
    except Exception as e:
        sys.stderr.write(f"[SceneIntel] scene cuts load failed ({e}); internal cuts only\n")
        return merged


def compute_shot_motion_profile(all_samples, shots):
    """Mean inter-sample gray diff per shot (160x90 downscale, cheap). Used as
    the per-scene motion advisory for the Camera Lock Operator. Deterministic."""
    profile = {}
    try:
        for idx, (shot_s, shot_e) in enumerate(shots):
            samples = [s for s in all_samples if shot_s <= s[0] < shot_e and s[1] is not None]
            diffs = []
            prev_small = None
            for _, frame, _faces in samples:
                small = cv2.resize(frame, (160, 90))
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
                if prev_small is not None:
                    diffs.append(float(np.mean(cv2.absdiff(gray, prev_small))))
                prev_small = gray
            profile[idx] = float(np.mean(diffs)) if diffs else 0.0
    except Exception:
        return {}
    return profile


def rank_crop_candidates(candidates, crop_y, eff_crop_w, eff_crop_h, scale,
                         target_face, all_faces, source_w, source_h,
                         current_crop_x):
    """Advisory crop ranking (Phase 3): score containment-legal x candidates
    with the existing deterministic composition score and return the best.
    Adds no new legality rules; the corridor already guarantees head
    containment, and hard containment is re-checked downstream."""
    best_x, best_score = None, float("-inf")
    for cx in candidates:
        motion = abs(cx - (current_crop_x if current_crop_x is not None else cx))
        try:
            s = evaluate_crop_composition_score(
                cx, crop_y, eff_crop_w, eff_crop_h, scale,
                target_face, all_faces, source_w, source_h,
                camera_motion_px=motion, scale_motion=0.0,
            )
        except Exception:
            continue
        if s > best_score:
            best_score, best_x = s, cx
    return best_x, best_score


def _env_flag_enabled(name: str) -> bool:
    """Matches the Rust env_flag semantics: unset/any value enables, except
    0/false/off (case-insensitive) which disables."""
    v = os.environ.get(name)
    if v is None:
        return True
    return v.strip().lower() not in ("0", "false", "off")


def speaker_fusion_enabled() -> bool:
    return _env_flag_enabled(FUSION_ENV_MASTER) and _env_flag_enabled(FUSION_ENV_FLAG)


def speaker_reid_enabled() -> bool:
    return _env_flag_enabled(FUSION_ENV_MASTER) and _env_flag_enabled(REID_ENV_FLAG)


def build_diarization_segments_from_words(words, clip_start: float, clip_end: float):
    """Derive diarization segments from word-level speaker labels.

    Contiguous same-speaker words form one segment (audio evidence the
    transcription backend already produced). Words without speaker labels
    contribute no evidence. Times are absolute source seconds.
    """
    segments = []
    if not words:
        return segments
    cur_speaker, cur_start, cur_end = None, None, None
    for w in words:
        spk = w.get("speaker") if isinstance(w, dict) else getattr(w, "speaker", None)
        try:
            ws = float(w.get("start", 0.0) if isinstance(w, dict) else w.start)
            we = float(w.get("end", 0.0) if isinstance(w, dict) else w.end)
        except (TypeError, ValueError):
            continue
        if we <= clip_start or ws >= clip_end:
            continue
        if spk is None:
            continue
        if cur_speaker == spk and cur_end is not None and ws <= cur_end + 0.75:
            cur_end = max(cur_end, we)
        else:
            if cur_speaker is not None and cur_end is not None:
                segments.append(_asf.DiarizedSegment(
                    speaker_id=str(cur_speaker),
                    start=max(cur_start, clip_start),
                    end=min(cur_end, clip_end),
                    confidence=0.85,
                ))
            cur_speaker, cur_start, cur_end = str(spk), ws, we
    if cur_speaker is not None and cur_end is not None:
        segments.append(_asf.DiarizedSegment(
            speaker_id=str(cur_speaker),
            start=max(cur_start, clip_start),
            end=min(cur_end, clip_end),
            confidence=0.85,
        ))
    return segments


def load_diarization_sidecar_json(path):
    """Load cached source-level diarization (absolute source seconds) written
    by the Rust Speaker Intelligence engine (speaker_diarization.py output).

    Returns a list of DiarizedSegment, or [] when the file is missing,
    unreadable, malformed, or has no segments (fallback: transcript labels).
    """
    segments = []
    if not path:
        return segments
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for seg in data.get("segments", []) or []:
            sid = seg.get("speaker_id")
            s, e = seg.get("start"), seg.get("end")
            if sid is None or s is None or e is None:
                continue
            segments.append(_asf.DiarizedSegment(
                speaker_id=str(sid),
                start=float(s),
                end=float(e),
                confidence=float(seg.get("confidence", 0.8) or 0.8),
            ))
        if segments:
            sys.stderr.write(f"[SpeakerIntel] loaded {len(segments)} cached diarization segments\n")
    except Exception as e:
        sys.stderr.write(f"[SpeakerIntel] diarization sidecar load failed ({e}); transcript fallback\n")
        return []
    return segments


def load_persisted_gallery_json(path):
    """Load persisted Re-ID gallery entries (Rust TrackReIdEmbedding JSON,
    camelCase, base64 float32 embeddings) for cross-render track association.

    Returns {persistent_id: {"embedding": np.ndarray(L2-normed), "model": str,
    "detection_count": int}} or {} on any failure (association then falls back
    to positional matching only).
    """
    import base64 as _b64
    gallery = {}
    if not path:
        return gallery
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        entries = data.get("entries", data if isinstance(data, list) else [])
        for ent in entries or []:
            emb_b64 = ent.get("embeddingB64") or ent.get("embedding_b64")
            model = ent.get("model") or "osnet_x1_0"
            pid = ent.get("persistentId") or ent.get("persistent_id")
            if not emb_b64:
                continue
            raw = _b64.b64decode(emb_b64)
            if not raw or len(raw) % 4 != 0:
                continue
            emb = np.frombuffer(raw, dtype=np.float32).astype(np.float32)
            n = float(np.linalg.norm(emb))
            if n <= 0.0 or not np.isfinite(emb).all():
                continue
            key = str(pid) if pid else f"{model}:{int(ent.get('trackId', ent.get('track_id', -1)))}:{int(ent.get('shotIdx', ent.get('shot_idx', 0)))}"
            gallery[key] = {
                "embedding": emb / n,
                "model": str(model),
                "detection_count": int(ent.get("detectionCount", ent.get("detection_count", 0)) or 0),
            }
        if gallery:
            sys.stderr.write(f"[SpeakerIntel] loaded {len(gallery)} persisted gallery identities\n")
    except Exception as e:
        sys.stderr.write(f"[SpeakerIntel] gallery load failed ({e}); positional association only\n")
        return {}
    return gallery


def attach_reid_embeddings(shot_tracks, shot_samples, clip_start_sec, shot_idx=0,
                           face_space_ratio=1.0):
    """Extract OSNet embeddings for completed shot tracks and EMA-merge them
    into the Re-ID gallery (gallery keys namespaced per shot). Detections'
    face crops are sampled from the shot's stored frames; track dicts gain a
    `reid_embedding` marker (L2-normed np array or None). Fails soft per shot.

    `face_space_ratio` is stored_w / true_w: track dicts are in TRUE source
    space while the sampled frames are the memory-budgeted STORED frames, so
    crop boxes must be scaled into stored space before indexing (no-op when
    frames were stored at full resolution)."""
    if not speaker_reid_enabled():
        return
    mgr = get_reid_manager()
    if mgr.model is None:
        return
    try:
        frames_by_t = [(s[0], s[1]) for s in shot_samples if s[1] is not None]
        if not frames_by_t:
            return
        for tid, tr in shot_tracks.items():
            dets = tr.get("detections", []) or []
            # Best faces first: highest-confidence crops with a usable bbox.
            dets_sorted = sorted(
                (d for d in dets if isinstance(d[1], dict) and d[1].get("bbox")),
                key=lambda d: float(d[1].get("conf", 0.0)), reverse=True,
            )[:8]
            if not dets_sorted:
                continue
            crops, stamps = [], []
            for rel_t, face, _mdiff in dets_sorted:
                frame = next((fr for t, fr in frames_by_t if abs(t - rel_t) <= 0.05), None)
                if frame is None:
                    continue
                x, y, w, h = face["bbox"]
                if face_space_ratio != 1.0:
                    x, y, w, h = (x * face_space_ratio, y * face_space_ratio,
                                  w * face_space_ratio, h * face_space_ratio)
                fh, fw = frame.shape[0], frame.shape[1]
                xi, yi = int(max(0, min(x, fw - 1))), int(max(0, min(y, fh - 1)))
                wi, hi = int(max(1, min(w, fw - xi))), int(max(1, min(h, fh - yi)))
                if wi < 8 or hi < 16:
                    continue
                crops.append(frame[yi:yi + hi, xi:xi + wi])
                stamps.append(clip_start_sec + float(rel_t))
            if not crops:
                continue
            # Extract in time order so EMA gallery first/last_seen stay monotonic.
            order = sorted(range(len(stamps)), key=lambda i: stamps[i])
            crops = [crops[i] for i in order]
            stamps = [stamps[i] for i in order]
            embeddings = extract_embedding_batch(mgr.model, crops, REID_DEVICE)
            for emb, ts in zip(embeddings, stamps):
                if emb is None:
                    continue
                mgr.update_gallery(shot_gallery_key(shot_idx, tid), emb, ts)
                if tr.get("reid_embedding") is None:
                    tr["reid_embedding"] = np.array(emb, dtype=np.float32)
    except Exception as e:
        sys.stderr.write(f"[SpeakerIntel] reid embedding attach failed ({e}); continuing without\n")


def associate_with_persisted_gallery(shot_tracks, persisted_gallery, threshold=None):
    """Associate shot tracks with persisted gallery identities via cosine
    similarity of OSNet embeddings (Re-ID assisted identity persistence).
    Adds `persistent_identity_id` to each matched track. Without embeddings
    or a persisted gallery this is a no-op (positional association unchanged)."""
    if not persisted_gallery or not speaker_reid_enabled():
        return
    thr = float(threshold if threshold is not None else 0.7)
    for tid, tr in shot_tracks.items():
        emb = tr.get("reid_embedding")
        if emb is None:
            continue
        n = float(np.linalg.norm(emb))
        if n <= 0.0:
            continue
        e = emb / n
        best_id, best_sim = None, thr
        for pid, entry in persisted_gallery.items():
            sim = float(np.dot(e, entry["embedding"]))
            if sim > best_sim:
                best_sim, best_id = sim, pid
        if best_id is not None:
            tr["persistent_identity_id"] = best_id
            tr["persistent_identity_sim"] = best_sim


def run_fusion_for_shot(shot_tracks, shot_samples, diar_segments,
                        clip_start_sec, shot_s_rel, shot_e_rel,
                        interval_sec=1.0):
    """Run active-speaker fusion over one shot's completed tracks.

    Returns a list of fused ActiveSpeakerState intervals in ABSOLUTE source
    seconds, or [] when fusion is disabled/unavailable or produced nothing.
    Never raises.
    """
    if not speaker_fusion_enabled() or _asf is None:
        return []
    try:
        visual_tracks = []
        for tid, tr in shot_tracks.items():
            motion, visibility = [], []
            for d in tr.get("detections", []) or []:
                rel_t, face, mdiff = d
                if not isinstance(face, dict):
                    continue
                t_abs = clip_start_sec + float(rel_t)
                if face.get("conf") is not None:
                    visibility.append((t_abs, float(face["conf"])))
                if mdiff is None:
                    continue
                motion.append((t_abs, min(1.0, max(0.0, float(mdiff) / FUSION_MOTION_NORM))))
            if not motion and not visibility:
                continue
            visual_tracks.append(_asf.VisualTrack(
                track_id=int(tid),
                mouth_motion_scores=motion,
                face_visibility=visibility,
            ))
        if not visual_tracks:
            return []
        shot_dur = max(0.1, float(shot_e_rel) - float(shot_s_rel))
        span_start = clip_start_sec + float(shot_s_rel)
        states = _asf.run_active_speaker_fusion(
            list(diar_segments or []), visual_tracks, {},
            interval_duration=float(interval_sec),
            source_duration=span_start + shot_dur,
            interval_start_sec=span_start,
        )
        # Clamp to the shot's own span so per-shot results cannot leak outside.
        return [s for s in states if s.end > span_start and s.start < span_start + shot_dur]
    except Exception as e:
        sys.stderr.write(f"[SpeakerIntel] fusion failed ({e}); heuristic fallback active\n")
        return []


def fusion_states_for_window(fusion_states, rel_ts, rel_te, clip_start_sec,
                             min_confidence=0.55):
    """Fused speaking states overlapping [rel_ts, rel_te] (clip-relative),
    keyed by track id. Only states with confidence >= min_confidence and a
    diarization match count as fusion evidence; anything weaker falls back
    to the deterministic heuristic."""
    out = {}
    if not fusion_states:
        return out
    a = clip_start_sec + rel_ts
    b = clip_start_sec + rel_te
    for s in fusion_states:
        if s.confidence is None or s.confidence < min_confidence:
            continue
        if s.speaking_probability is None or s.speaking_probability < 0.5:
            continue
        if s.end <= a or s.start >= b:
            continue
        prev = out.get(s.track_id)
        if prev is None or s.confidence > prev.confidence:
            out[s.track_id] = s
    return out


def build_speaker_intel_block(gallery_manager, all_fusion_states, clip_start_sec,
                              fusion_method="fusion"):
    """Emit the speakerIntel plan block consumed by the Rust engine:
    Re-ID gallery entries (EMA embeddings, base64 float32) + fused active
    speaker intervals (absolute source seconds)."""
    import base64 as _b64
    gallery_entries = []
    try:
        for gkey, ent in (gallery_manager.get_gallery_embeddings() if gallery_manager else {}).items():
            emb = np.asarray(ent.embedding, dtype=np.float32)
            if emb.size == 0 or not np.isfinite(emb).all():
                continue
            n = float(np.linalg.norm(emb))
            if n <= 0.0:
                continue
            emb = emb / n
            gallery_entries.append({
                "trackId": int(gkey) % SHOT_TRACK_ID_STRIDE,
                "shotIdx": int(gkey) // SHOT_TRACK_ID_STRIDE,
                "embeddingB64": _b64.b64encode(emb.astype(np.float32).tobytes()).decode("ascii"),
                "model": getattr(gallery_manager, "model_name", "osnet_x1_0") or "osnet_x1_0",
                "detectionCount": int(ent.detection_count),
                "firstSeenSec": float(ent.first_seen),
                "lastSeenSec": float(ent.last_seen),
                "updatedAt": ent.updated_at,
            })
    except Exception as e:
        sys.stderr.write(f"[SpeakerIntel] gallery serialization failed ({e}); emitting without gallery\n")
        gallery_entries = []

    intervals = []
    for s in all_fusion_states or []:
        try:
            intervals.append({
                "trackId": int(s.track_id),
                "applicationRole": None,
                "diarizationId": s.diarization_id,
                "start": round(float(s.start), 3),
                "end": round(float(s.end), 3),
                "audioEvidence": round(float(s.audio_evidence), 4),
                "visualEvidence": round(float(s.visual_evidence), 4),
                "speakingProbability": round(float(s.speaking_probability), 4),
                "confidence": round(float(s.confidence), 4),
                "evidenceSources": list(s.evidence_sources or []),
            })
        except Exception:
            continue

    return {
        "method": fusion_method,
        "fusionVersion": "1.0",
        "gallery": gallery_entries,
        "fusion": {"intervals": intervals},
    }


def get_ultralytics_model():
    """Returns singleton YOLO11n instance or None."""
    global _ULTRALYTICS_YOLO_MODEL
    if _ULTRALYTICS_YOLO_MODEL is None and HAS_ULTRALYTICS:
        model_path = find_yolo_model()
        if model_path:
            try:
                _ULTRALYTICS_YOLO_MODEL = YOLO(model_path)
            except Exception as e:
                sys.stderr.write(f"[CV Backend] Failed to load Ultralytics YOLO model from {model_path}: {e}\n")
                _ULTRALYTICS_YOLO_MODEL = False
        else:
            sys.stderr.write("[CV Backend] yolo11n.pt weights not found; Ultralytics tracker disabled\n")
            _ULTRALYTICS_YOLO_MODEL = False
    return _ULTRALYTICS_YOLO_MODEL if _ULTRALYTICS_YOLO_MODEL else None

# ─── Centralized Universal Framing Configuration ──────────────────────────────

@dataclass(frozen=True)
class FramingConfig:
    """
    Centralized configuration for AutoShorts Smart Reframing and Visual Subject Tracking.
    All detection, tracking, visual prominence, and framing thresholds are defined here.
    """
    # Detection & Tracking
    high_conf_threshold: float = 0.55
    low_conf_threshold: float = 0.30
    track_max_lost_seconds: float = 0.60
    gmc_min_features: int = 6

    # Visual Prominence Weights
    w_area: float = 0.30
    w_frontality: float = 0.25
    w_conf: float = 0.20
    w_centrality: float = 0.15
    w_stability: float = 0.10

    # Visual Subject Resolver
    mouth_motion_active_thresh: float = 4.0
    mouth_motion_dominance_ratio: float = 1.20
    prominence_dominance_diff: float = 0.08
    prominence_dominance_ratio: float = 1.20
    listener_reaction_min_dwell: float = 0.80

    # Camera Lock & Movement
    displacement_thresh_ratio: float = 0.14
    sustained_displacement_dur: float = 0.90
    reframe_cooldown: float = 1.50
    scale_deadband: float = 0.08

DEFAULT_FRAMING_CONFIG = FramingConfig()

# Multi-Factor Visual Prominence Configuration (Configurable Signal)
DEFAULT_VISUAL_PROMINENCE_WEIGHTS = {
    "area": DEFAULT_FRAMING_CONFIG.w_area,
    "frontality": DEFAULT_FRAMING_CONFIG.w_frontality,
    "confidence": DEFAULT_FRAMING_CONFIG.w_conf,
    "centrality": DEFAULT_FRAMING_CONFIG.w_centrality,
    "stability": DEFAULT_FRAMING_CONFIG.w_stability,
}

# ─── Configuration Constants ──────────────────────────────────────────────────
TRANSITION_DUR        = 0.30   # 300ms smoothstep transition for normal intra-shot drift
FAST_TRANSITION_DUR   = 0.15   # 150ms fast corrective transition when approaching boundary
DRIFT_THRESH_PX       = 14     # px -- reframe only if active speaker leaves valid corridor by > 14px
STALE_TARGET_THRESH   = 24     # px -- force immediate recomputation if displacement > 24px
SCENE_CUT_DIFF        = 24.0   # pixel difference threshold for scene change
SCENE_CUT_HIST        = 0.85   # histogram correlation threshold
FACE_CONF_THRESH      = 0.35   # YuNet face confidence threshold
ANALYSIS_FPS          = 8.0    # Visual analysis sampling rate (8 fps)

# ─── Person-detection inference scale ──────────────────────────────────────────
# MEASURED (1920x1080 source, YOLO11n + BoT-SORT, CPU):
#     1920px -> 766 ms/frame   (639 frames for a 79.9s candidate = 490s)
#      960px -> 339 ms/frame   (217s)
#      640px -> 324 ms/frame   (207s)
# YOLO letterboxes to imgsz=640 internally, so feeding 1920px frames buys NO
# extra detection quality while paying ~2.3x the preprocessing cost: the
# resize-to-imgsz work is proportional to the INPUT pixel count. Person boxes
# are converted back to true source space afterwards, so crop geometry,
# containment and composition are unaffected.
#
# Detection at the model's native working scale is the correct fix; it changes
# no thresholds, no tracking parameters, and no output semantics.
PERSON_DETECT_INFER_WIDTH = 960   # 2.3x faster, within 5% of the 640px floor
PERSON_DETECT_MIN_WIDTH   = 640   # never below the model's native imgsz
SNAP_CENTER_THRESH    = 0.08   # 8% of source width for snap-to-center
SAFETY_MARGIN_RATIO   = 0.08   # 8% of crop width (~48px) horizontal safety margin
TWO_SHOT_MAX_SPAN     = 0.65   # Max face span (as ratio of crop_w) to allow joint two-shot

# ─── Scale Constants ───────────────────────────────────────────────────────────
SCALE_DEADBAND        = DEFAULT_FRAMING_CONFIG.scale_deadband
SCALE_MAX_RATE        = 0.15   # max 15% scale change per 0.5s sub-interval
MIN_SCALE_IMPROVEMENT = 0.08   # minimum meaningful improvement to justify a scale change
MIN_SCALE             = 0.80   # never crop wider than 25% beyond baseline portrait width
MAX_SCALE             = 1.35   # never zoom tighter than 35% above baseline portrait width
SCALE_VELOCITY_CAP    = 0.12   # max scale change per sub-interval during PAN / gradual

# ─── Close-Speaker Panel Isolation Constants (DualFrame) ──────────────────────
# When two speakers are physically close, each 9:8 panel crop may contain a
# portion of the OTHER speaker ("cross-contamination"). Isolation is decided by
# PROJECTED CROP-vs-OTHER-SUBJECT contamination (never raw center distance):
# a panel is contaminated only if NO legal crop placement can contain the
# intended subject while excluding the other speaker's subject-safe box.
ISOLATION_CLEARANCE_RATIO = 0.02    # clearance between panel crop edge and the other speaker's safe box (× source_w; 38px @1920)
ISOLATION_SCALE_STEP       = 0.02   # deterministic pinch-zoom search granularity (scale units)
MAX_ISOLATION_SCALE        = 1.80   # bounded pinch-zoom ceiling: never zoom a panel tighter than this for isolation
ISOLATION_PAIR_TOL_SEC     = 0.25   # max |Δt| when pairing a detection with the other speaker's nearest detection
ISOLATION_BOUNDARY_HYSTERESIS_PX = 5.0  # isolation boundary hysteresis: other-speaker/safe-box jitter below this many px must NOT move the camera

# ─── Conditional Subject-Tight Framing Constants (DualFrame, §27) ─────────────
# NORMAL DualFrame panels only (NOT close-speaker isolation). When a panel's
# assigned speaker body occupies too little of the projected baseline crop,
# the panel is background-heavy and the crop is tightened MODERATELY so the
# speaker gains prominence. Already well-composed panels are kept UNCHANGED
# (zero tightening). Requires YOLO person-body evidence (person_bbox); panels
# without body evidence pass through with the existing composition. Isolation
# (cross-speaker contamination) always takes precedence and is never stacked.
TIGHTEN_ENTER_OCCUPANCY  = 0.32  # body width < 32% of baseline panel width → background-heavy candidate
TIGHTEN_TARGET_OCCUPANCY = 0.44  # tightening aims for body width ≈ 44% of the tightened panel width
TIGHTEN_SCALE_STEP       = 0.02  # deterministic tightening search granularity (scale units)
MAX_TIGHTEN_SCALE        = 1.50  # moderate tightening ceiling (well below MAX_ISOLATION_SCALE 1.80)
TIGHTEN_MIN_SCALE_GAIN   = 0.06  # gains below this are negligible: keep the existing composition

# Subject Importance & Saliency Weights (AutoFlip Model)
SALIENCY_ACTIVE_FACE    = 1.00
SALIENCY_ACTIVE_CORE    = 1.00
SALIENCY_ACTIVE_BODY    = 0.80
SALIENCY_SECONDARY_SPK  = 0.55
SALIENCY_OTS_LISTENER   = 0.20
SALIENCY_BACKGROUND_FACE = 0.05
SALIENCY_FURNITURE      = 0.00
SALIENCY_EMPTY_CENTER   = 0.00


# ─── Per-Shot Scale State ──────────────────────────────────────────────────────

class ShotScaleState:
    """
    Independent scale state for each shot.
    Reset at every shot cut. Never carried across shots.
    """
    def __init__(self, shot_id: int, shot_type: str, ideal_scale: float,
                 min_valid_scale: float, max_valid_scale: float):
        self.shot_id = shot_id
        self.shot_type = shot_type
        self.ideal_scale = ideal_scale
        self.min_valid_scale = min_valid_scale
        self.max_valid_scale = max_valid_scale
        self.current_scale = ideal_scale
        self.scale_history = []
        self.scale_change_reason = "SHOT_RESET"

    def record(self, t: float, scale: float, reason: str):
        self.current_scale = scale
        self.scale_change_reason = reason
        self.scale_history.append((t, scale))
        if len(self.scale_history) > 60:
            self.scale_history = self.scale_history[-60:]


# ─── Persistent Speaker Identity ──────────────────────────────────────────────

class SpeakerIdentity:
    """Persistent spatial and temporal identity state for one diarized speaker."""

    def __init__(self, spk: str):
        self.spk = spk
        self.position_history = []  # [(cx, cy, timestamp), ...]
        self.confirmed_frames = 0
        self.last_confirmed_cx = None
        self.last_confirmed_cy = None

    def record_detection(self, cx: float, cy: float, t_sec: float) -> None:
        self.last_confirmed_cx = cx
        self.last_confirmed_cy = cy
        self.position_history.append((cx, cy, t_sec))
        if len(self.position_history) > 50:
            self.position_history = self.position_history[-50:]
        self.confirmed_frames = min(self.confirmed_frames + 1, 25)

    def get_last_position(self, fallback_cx: float) -> float:
        if self.last_confirmed_cx is not None:
            return self.last_confirmed_cx
        if self.position_history:
            return self.position_history[-1][0]
        return fallback_cx

    def predict_position(self, target_t: float, fallback_cx: float) -> float:
        """Linear motion extrapolation from recent detections."""
        if len(self.position_history) < 2:
            return self.get_last_position(fallback_cx)
        p1 = self.position_history[-2]
        p2 = self.position_history[-1]
        dt = p2[2] - p1[2]
        if 0.05 < dt < 2.0:
            vx = (p2[0] - p1[0]) / dt
            time_diff = min(1.0, max(0.0, target_t - p2[2]))
            return float(p2[0] + vx * time_diff)
        return p2[0]


speaker_identities = {}

def get_or_create_identity(spk: str) -> SpeakerIdentity:
    global speaker_identities
    if spk not in speaker_identities:
        speaker_identities[spk] = SpeakerIdentity(spk)
    return speaker_identities[spk]


# ─── 1. Diarization Engine ───────────────────────────────────────────────────

def build_speaker_turns(words, clip_start: float, clip_end: float):
    """
    Constructs contiguous speaking turns from word-level diarized transcript.
    """
    if not words:
        return []
    turns = []
    cur_spk = cur_s = cur_e = None
    for w in words:
        ws = max(float(w.get("start", 0)), clip_start)
        we = min(float(w.get("end",   0)), clip_end)
        spk = w.get("speaker") or "S1"
        if ws >= clip_end or we <= clip_start:
            continue
        if spk == cur_spk:
            cur_e = we
        else:
            if cur_spk is not None and cur_s < cur_e:
                turns.append((cur_s, cur_e, cur_spk))
            cur_spk, cur_s, cur_e = spk, ws, we
    if cur_spk is not None and cur_s < cur_e:
        turns.append((cur_s, cur_e, cur_spk))
    return turns


# ─── 2. Face & Subject Detection Engine ───────────────────────────────────────

def find_yunet_model():
    candidates = [
        os.path.join(os.path.dirname(__file__), "face_detection_yunet_2023mar.onnx"),
        os.path.join(os.path.dirname(__file__), "..", "models", "face_detection_yunet_2023mar.onnx"),
        os.path.join(os.path.dirname(__file__), "..", "..", "src-tauri", "scripts", "face_detection_yunet_2023mar.onnx"),
        os.path.expanduser(r"~\AppData\Roaming\com.autoshorts.desktop\models\face_detection_yunet_2023mar.onnx"),
        os.path.expanduser(r"~\Downloads\face_detection_yunet_2023mar.onnx"),
    ]
    for c in candidates:
        if os.path.exists(c) and os.path.getsize(c) > 100000:
            return c
    return None


def create_detector(yunet_path):
    if not yunet_path:
        return None
    try:
        return cv2.FaceDetectorYN_create(
            model=yunet_path,
            config="",
            input_size=(1280, 720),
            score_threshold=FACE_CONF_THRESH,
            nms_threshold=0.3,
        )
    except Exception:
        return None


def detect_letterboxing(frame):
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    top = 0
    while top < h // 4 and np.mean(gray[top, :]) < 12.0:
        top += 1
    bottom = h
    while bottom > h * 3 // 4 and np.mean(gray[bottom - 1, :]) < 12.0:
        bottom -= 1
    left = 0
    while left < w // 4 and np.mean(gray[:, left]) < 12.0:
        left += 1
    right = w
    while right > w * 3 // 4 and np.mean(gray[:, right - 1]) < 12.0:
        right -= 1
    return left, top, right, bottom


# ─── Resolution-Independent Head Geometry Thresholds (F7) ───────────────────
REF_FRAME_AREA_1080P = 1920.0 * 1080.0
FRAC_BOTTOM_FOREGROUND = 18000.0 / REF_FRAME_AREA_1080P
FRAC_LOW_FRONT_LARGE   = 14000.0 / REF_FRAME_AREA_1080P
FRAC_EXTREME_EDGE_HUGE = 35000.0 / REF_FRAME_AREA_1080P


def is_back_of_head_candidate(
    fx: float, fy: float, fw: float, fh: float,
    w: int, h: int,
    conf: float, skin_ratio: float, sym_err: float
) -> bool:
    """
    Evaluates whether a detected face box represents a back-of-head / OTS foreground candidate.
    Uses normalized area fractions relative to standard 1080p area (1920x1080) for resolution invariance.
    """
    if w <= 0 or h <= 0:
        return False

    frame_area = max(1.0, float(w * h))
    face_area = fw * fh

    is_bottom_foreground = (fy > h * 0.48) and (face_area > FRAC_BOTTOM_FOREGROUND * frame_area)
    is_low_front_large = (conf < 0.78) and (skin_ratio < 0.35 or sym_err > 0.32) and (face_area > FRAC_LOW_FRONT_LARGE * frame_area)
    is_extreme_edge_huge = (fx < w * 0.15 or fx + fw > w * 0.85) and (face_area > FRAC_EXTREME_EDGE_HUGE * frame_area) and (conf < 0.80)

    return bool(is_bottom_foreground or is_low_front_large or is_extreme_edge_huge)


def detect_faces_and_subjects(detector, frame):
    """
    Detects faces with landmarks and evaluates complete subject geometry.
    Uses downscaled 640px analysis stream for 16x faster inference speed.
    """
    h, w = frame.shape[:2]
    faces = []
    if detector is not None:
        try:
            if w > 640:
                scale = 640.0 / float(w)
                dw, dh = 640, int(round(h * scale))
                det_img = cv2.resize(frame, (dw, dh))
                inv_s = float(w) / 640.0
            else:
                det_img = frame
                dw, dh = w, h
                inv_s = 1.0

            detector.setInputSize((dw, dh))
            _, raw = detector.detect(det_img)
            if raw is not None and len(raw) > 0:
                for f in raw:
                    fx = float(f[0]) * inv_s
                    fy = float(f[1]) * inv_s
                    fw = float(f[2]) * inv_s
                    fh = float(f[3]) * inv_s
                    conf = float(f[14])
                    if conf < FACE_CONF_THRESH or fw < 16 or fh < 16:
                        continue

                    re = (float(f[4]) * inv_s, float(f[5]) * inv_s)
                    le = (float(f[6]) * inv_s, float(f[7]) * inv_s)
                    nose = (float(f[8]) * inv_s, float(f[9]) * inv_s)
                    rm = (float(f[10]) * inv_s, float(f[11]) * inv_s)
                    lm = (float(f[12]) * inv_s, float(f[13]) * inv_s)

                    eye_dist = math.hypot(le[0] - re[0], le[1] - re[1])
                    mid_eye_x = (re[0] + le[0]) / 2.0 if (re[0] > 0 and le[0] > 0) else fx + fw / 2.0
                    mid_eye_y = (re[1] + le[1]) / 2.0 if (re[1] > 0 and le[1] > 0) else fy + 0.35 * fh

                    gaze_offset_x = (nose[0] - mid_eye_x) / max(1.0, eye_dist)
                    sym_err = abs(gaze_offset_x)
                    frontality = max(0.0, 1.0 - min(1.0, sym_err))

                    y1, y2 = max(0, int(fy)), min(h, int(fy + fh))
                    x1, x2 = max(0, int(fx)), min(w, int(fx + fw))
                    face_crop = frame[y1:y2, x1:x2]
                    skin_ratio = 0.5
                    if face_crop.size > 0:
                        try:
                            hsv_f = cv2.cvtColor(face_crop, cv2.COLOR_BGR2HSV)
                            skin_mask = cv2.inRange(hsv_f, np.array([0, 18, 35]), np.array([28, 230, 255]))
                            skin_ratio = float(np.count_nonzero(skin_mask)) / float(face_crop.size / 3)
                        except Exception:
                            pass

                    is_back = is_back_of_head_candidate(
                        fx, fy, fw, fh, w, h, conf, skin_ratio, sym_err
                    )

                    head_top_y = max(0.0, fy - 0.22 * fh)
                    shoulder_w = fw * 2.2
                    shoulder_left = max(0.0, fx + fw / 2.0 - shoulder_w / 2.0)
                    shoulder_right = min(float(w), fx + fw / 2.0 + shoulder_w / 2.0)
                    torso_top = fy + fh * 0.85

                    mouth_cx = (rm[0] + lm[0]) / 2.0 if (rm[0] > 0 and lm[0] > 0) else fx + fw / 2.0
                    mouth_cy = (rm[1] + lm[1]) / 2.0 if (rm[1] > 0 and lm[1] > 0) else fy + 0.75 * fh
                    mw = max(15.0, fw * 0.4)
                    mh = max(10.0, fh * 0.25)
                    mx1 = int(max(0, mouth_cx - mw / 2))
                    mx2 = int(min(w, mouth_cx + mw / 2))
                    my1 = int(max(0, mouth_cy - mh / 2))
                    my2 = int(min(h, mouth_cy + mh / 2))
                    mouth_crop = frame[my1:my2, mx1:mx2]
                    mouth_gray = cv2.cvtColor(mouth_crop, cv2.COLOR_BGR2GRAY) if mouth_crop.size > 0 else None

                    faces.append({
                        "bbox": (fx, fy, fw, fh),
                        "center": (fx + fw / 2.0, fy + fh / 2.0),
                        "conf": conf,
                        "area": fw * fh,
                        "frontality": frontality,
                        "gaze_offset_x": gaze_offset_x,
                        "skin_ratio": skin_ratio,
                        "is_back": is_back,
                        "head_top_y": head_top_y,
                        "eye_line_y": mid_eye_y,
                        "shoulder_span": (shoulder_left, shoulder_right),
                        "torso_top": torso_top,
                        "mouth_gray": mouth_gray,
                        "landmarks": (re, le, nose, rm, lm),
                    })
        except Exception:
            pass
    return faces


# ─── 3. Subject-Safe Box ──────────────────────────────────────────────────────

def nearest_genuine_face(genuine: list, crop_x: float, crop_w: float) -> dict:
    """
    Recovery target for trajectory validation: the genuine face NEAREST the
    crop's current center, prominence as tie-break.

    The validator fires only when the crop contains no face; the correct
    recovery is the subject the camera was already moving toward. Selecting by
    global prominence instead can contradict a deliberate speaker handoff and
    lock the crop onto the previous speaker for seconds.
    """
    crop_cx = crop_x + crop_w / 2.0
    return max(
        genuine,
        key=lambda f: (
            -abs(f["center"][0] - crop_cx),
            f["conf"] * f["frontality"] * f["area"],
        ),
    )


def sanitize_face_bbox_for_containment(
    target_face: dict, crop_w_baseline: float, source_w: float
) -> dict:
    """
    Face-geometry sanitizer (source-camera-zoom guard).

    When the SOURCE camera zooms in, YuNet (running on the 640px analysis
    stream) can merge the zoomed subject cluster into one giant box spanning a
    third of the source frame. Used raw, its derived subject safe box
    (shoulders = 2.2 x face width) is wider than the portrait crop can ever
    contain, so the containment corridor collapses to a point and the face is
    guaranteed to be clipped -- the crop "compounds" the source zoom.

    The portrait crop can contain a head (1.24 x face width, from
    compute_subject_safe_box) with SAFETY_MARGIN_RATIO margins on both sides
    only while face width <= crop_w_baseline x (1 - 2*SAFETY_MARGIN_RATIO) / 1.24.
    Cap the reported box at that bound, preserving its center, so the existing
    geometry stays the authority and the crop can center on the real subject.
    """
    if target_face is None:
        return target_face
    fx, fy, fw, fh = target_face["bbox"]
    max_fw = crop_w_baseline * (1.0 - 2.0 * SAFETY_MARGIN_RATIO) / 1.24
    if fw <= max_fw or max_fw <= 0.0:
        return target_face
    scale = max_fw / fw
    cx, cy = fx + fw / 2.0, fy + fh / 2.0
    repaired = dict(target_face)
    repaired["bbox"] = (cx - max_fw / 2.0, cy - fh * scale / 2.0, max_fw, fh * scale)
    return repaired


def _pair_subject_safe_box(l_track: dict, r_track: dict, source_w: float, source_h: float) -> dict:
    """
    Union of both subjects' safe boxes — the composition unit for a two-shot.
    Returns None if either track has no usable detection.
    """
    def _best_face(tr):
        dets = [d[1] for d in tr["detections"]]
        return max(dets, key=lambda f: f["conf"]) if dets else None

    lf, rf = _best_face(l_track), _best_face(r_track)
    if lf is None or rf is None:
        return None
    lb = compute_subject_safe_box(lf, source_w, source_h)
    rb = compute_subject_safe_box(rf, source_w, source_h)
    left = min(lb["left"], rb["left"])
    right = max(lb["right"], rb["right"])
    top = min(lb["top"], rb["top"])
    bottom = max(lb["bottom"], rb["bottom"])
    return {
        "left": left,
        "right": right,
        "head_left": min(lb["head_left"], rb["head_left"]),
        "head_right": max(lb["head_right"], rb["head_right"]),
        "top": top,
        "bottom": bottom,
        "width": right - left,
        "height": bottom - top,
        "center_x": (left + right) / 2.0,
        "gaze_offset_x": 0.0,  # pair composition: no gaze bias
        "face_bbox": None,
    }


def _pair_composition_fits(
    l_track: dict, r_track: dict, crop_w_baseline: float, source_w: float, source_h: float,
    min_scale: float = MIN_SCALE, max_scale: float = 1.10,
    adaptive_framing: bool = False,
):
    """
    General close-pair composition criterion (Adaptive Framing).

    A two-person shot is a CLOSE PAIR iff BOTH subjects' head extents fit
    inside the renderable inner composition — i.e. the pair can be held in
    ONE natural two-shot that clips neither head. In Original 9:16 the
    renderable window is the widest true-9:16 crop (source_h*9/16, full
    height, capped by the source width). In Adaptive Framing the renderable
    window is the SQUARE inner composition (side = min(source_h, source_w)),
    which is what the reference video composes: a full-width square video
    centered in the 9:16 canvas.

    The test is a single comparison of the pair's own head span against that
    renderable width, so it is independent of source resolution, subject size
    and camera distance: no raw-pixel span threshold, no per-video tuning.

    needed_scale is the LARGEST scale that still keeps both heads inside the
    crop (renderable_w / head_span), clamped to the solver's scale ceiling. It
    is a ceiling on the two-shot scale, not a target: the solver may shoot
    below it.

    Returns (fits: bool, needed_scale: float or None).
    """
    pair_box = _pair_subject_safe_box(l_track, r_track, source_w, source_h)
    if pair_box is None:
        return False, None

    if adaptive_framing:
        renderable_w = min(source_h, source_w)
    else:
        renderable_w = min(source_h * 9.0 / 16.0, source_w)
    head_span = pair_box["head_right"] - pair_box["head_left"]
    if head_span <= 0.0:
        return False, None

    if adaptive_framing:
        # In 1080x1080 Adaptive Framing, both subjects must be kept together
        # in the shared composition. Do not allow artificial margins to reject
        # the pair and trigger active-speaker fallback. Spans up to canvas width
        # (e.g. 1037-1092px in 1080px) are accepted with documented numerical tolerance.
        usable_w = float(renderable_w) * 1.05
    else:
        # The solver keeps a SAFETY_MARGIN_RATIO horizontal corridor on both sides
        # of the crop; a pair whose head span leaves no such corridor renders with
        # heads flush against the composition edges (verified on the far-pair
        # control: head span 1032 of a 1080 square -> 2-5px edge margins). Apply
        # the same margin here so "fits" means "fits with the solver's own margin".
        usable_w = renderable_w * (1.0 - 2.0 * SAFETY_MARGIN_RATIO)
    needed = min(float(max_scale), max(1.0, renderable_w / head_span))
    return head_span <= usable_w, needed


def compute_subject_safe_box(target_face: dict, source_w: float, source_h: float) -> dict:
    fx, fy, fw, fh = target_face["bbox"]
    head_top = max(0.0, fy - 0.22 * fh)
    head_left = max(0.0, fx - 0.12 * fw)
    head_right = min(source_w, fx + 1.12 * fw)
    shoulder_w = fw * 2.2
    shoulder_left = max(0.0, fx + fw / 2.0 - shoulder_w / 2.0)
    shoulder_right = min(source_w, fx + fw / 2.0 + shoulder_w / 2.0)
    torso_bottom = min(source_h, fy + 2.6 * fh)
    subj_left = min(head_left, shoulder_left)
    subj_right = max(head_right, shoulder_right)
    return {
        "left": subj_left,
        "right": subj_right,
        "head_left": head_left,
        "head_right": head_right,
        "top": head_top,
        "bottom": torso_bottom,
        "width": subj_right - subj_left,
        "height": torso_bottom - head_top,
        "center_x": (subj_left + subj_right) / 2.0,
        "gaze_offset_x": target_face.get("gaze_offset_x", 0.0),
        "face_bbox": (fx, fy, fw, fh)
    }


# ─── 4. Shot-Local Scale Estimator ────────────────────────────────────────────

def classify_shot_and_estimate_scale(
    target_face,
    shot_samples,
    crop_w_baseline: float,
    source_w: float,
    source_h: float,
    shot_type: str = "unknown",
    adaptive_framing: bool = False,
    pair_needed_scale: Optional[float] = None,
):
    """
    Shot-local scale estimator. Anti-overzoom: prefer MINIMUM satisfying scale.
    Returns (ideal_scale, min_valid_scale, max_valid_scale, shot_class).
    scale > 1.0 = zoom in (narrower crop), scale < 1.0 = zoom out (wider crop).
    """
    if target_face is None:
        return 0.92, MIN_SCALE, 1.05, "no_face"

    fx, fy, fw, fh = target_face["bbox"]
    face_w_ratio = fw / max(1.0, source_w)

    if face_w_ratio >= 0.30:
        shot_class, ideal_scale, min_s, max_s = "extreme_close_up", 1.00, 1.00, 1.10
    elif face_w_ratio >= 0.15:
        shot_class, ideal_scale, min_s, max_s = "close_up", 1.00, 1.00, 1.15
    elif face_w_ratio >= 0.08:
        shot_class, ideal_scale, min_s, max_s = "medium_close_up", 1.05, 0.95, 1.20
    elif face_w_ratio >= 0.04:
        shot_class, ideal_scale, min_s, max_s = "medium", 1.05, 0.95, 1.15
    elif face_w_ratio >= 0.01:
        shot_class, ideal_scale, min_s, max_s = "wide", 0.95, MIN_SCALE, 1.05
    else:
        shot_class, ideal_scale, min_s, max_s = "very_wide", 0.90, MIN_SCALE, 1.00

    if shot_type in ("two_shot_both_fit",):
        shot_class = "two_shot"
        ideal_scale = min(ideal_scale, 1.00)
        if adaptive_framing:
            # Adaptive: pair_needed_scale is the LARGEST scale that still keeps
            # both heads inside the widest renderable 9:16 crop — a CEILING on
            # the two-shot scale, never a floor. A scale below 1.0 would widen
            # the crop past the renderable 9:16 width (which would need a
            # taller-than-source crop or an anisotropic stretch), so the valid
            # range is [1.0, needed_scale] and the ideal is clamped inside it —
            # a "wide" subject classification can otherwise propose 0.95 and
            # open the shot at a non-renderable width.
            min_s = 1.00
            if pair_needed_scale is not None:
                max_s = float(np.clip(pair_needed_scale, 1.00, 1.10))
            else:
                max_s = 1.10
            ideal_scale = float(np.clip(ideal_scale, min_s, max_s))
        else:
            min_s, max_s = 0.88, 1.10
    elif shot_type in ("ots_over_the_shoulder",):
        shot_class = "ots"
        ideal_scale = min(ideal_scale, 1.05)
        min_s, max_s = 0.88, 1.10
    elif shot_type in ("wide_two_shot",):
        shot_class = "solo_speaker_wide"
        ideal_scale = ideal_scale
        min_s, max_s = MIN_SCALE, 1.15

    ideal_scale = float(np.clip(ideal_scale, MIN_SCALE, MAX_SCALE))
    min_s = float(np.clip(min_s, MIN_SCALE, MAX_SCALE))
    max_s = float(np.clip(max_s, MIN_SCALE, MAX_SCALE))
    if min_s > max_s:
        min_s, max_s = max_s, min_s

    return ideal_scale, min_s, max_s, shot_class


def apply_scale_deadband(
    current_scale: float,
    ideal_scale: float,
    min_valid_scale: float = MIN_SCALE,
    max_valid_scale: float = MAX_SCALE,
    prev_reason: str = "",
    deadband: Optional[float] = None,
):
    """Applies scale deadband hysteresis and velocity limit."""
    eff_deadband = deadband if deadband is not None else DEFAULT_FRAMING_CONFIG.scale_deadband
    if min_valid_scale <= current_scale <= max_valid_scale:
        if abs(current_scale - ideal_scale) < eff_deadband:
            return current_scale, "DEADBAND"

    raw_delta = ideal_scale - current_scale
    capped_delta = float(np.clip(raw_delta, -SCALE_VELOCITY_CAP, SCALE_VELOCITY_CAP))
    new_scale = float(np.clip(current_scale + capped_delta, min_valid_scale, max_valid_scale))
    new_scale = float(np.clip(new_scale, MIN_SCALE, MAX_SCALE))

    if abs(new_scale - current_scale) < 0.005:
        return current_scale, "DEADBAND"

    if current_scale < min_valid_scale or current_scale > max_valid_scale:
        reason = "BOUNDARY"
    elif new_scale > current_scale:
        reason = "SCALE_UP"
    elif new_scale < current_scale:
        reason = "SCALE_DOWN"
    else:
        reason = "COMPOSITION"
    return new_scale, reason


def compute_effective_crop_dims(scale: float, source_h: float, source_w: float, adaptive_framing: bool = False):
    """
    Real dynamic crop dimensions for a given scale.
    Original 9:16: scale=1.0 -> baseline 9:16 portrait (source_h*9/16 wide, full
    height). scale>1.0: zoom in (narrower). scale<1.0: zoom out (wider).
    Adaptive Framing: the inner composition is a SQUARE (side = min(source_h,
    source_w)) — the full-width square video the reference composes inside the
    9:16 canvas. Zoom scales the square uniformly.
    """
    if adaptive_framing:
        baseline_side = round(min(source_h, source_w))
        if baseline_side % 2 != 0:
            baseline_side -= 1
        eff = baseline_side / scale
        eff = min(eff, float(min(source_h, source_w)))
        eff = max(2, round(eff))
        if eff % 2 != 0:
            eff -= 1
        return float(eff), float(eff)

    baseline_crop_w = round(source_h * 9.0 / 16.0)
    if baseline_crop_w % 2 != 0:
        baseline_crop_w -= 1

    eff_w = baseline_crop_w / scale
    eff_w = min(eff_w, source_w)

    if scale >= 1.0:
        eff_h = eff_w * 16.0 / 9.0
    else:
        eff_h = min(source_h, eff_w * 16.0 / 9.0)

    eff_w = max(2, round(eff_w))
    if eff_w % 2 != 0:
        eff_w -= 1
    eff_h = max(2, round(eff_h))
    if eff_h % 2 != 0:
        eff_h -= 1
    return float(eff_w), float(eff_h)


# ─── 5. Joint Position + Scale Containment Solver ────────────────────────────

def solve_containment_crop_x(
    subject_safe_box: dict,
    crop_w: float,
    max_x: float,
    source_w: float,
    safety_margin_ratio: float = SAFETY_MARGIN_RATIO,
    current_crop_x=None,
):
    """
    Separates Hard Subject Containment from Soft Centering.
    Compositional Deadband: camera stays stationary if already inside corridor.
    """
    margin_x = crop_w * safety_margin_ratio
    subj_left = subject_safe_box["left"]
    subj_right = subject_safe_box["right"]
    subj_cx = subject_safe_box["center_x"]
    subj_w = subject_safe_box["width"]
    gaze_dir = subject_safe_box.get("gaze_offset_x", 0.0)

    eff_left = max(0.0, min(source_w, subj_left))
    eff_right = max(0.0, min(source_w, subj_right))
    head_left = max(0.0, min(source_w, subject_safe_box.get("head_left", eff_left)))
    head_right = max(0.0, min(source_w, subject_safe_box.get("head_right", eff_right)))

    if subj_w + 2.0 * margin_x <= crop_w:
        min_allowed_x = max(0.0, subj_right + margin_x - crop_w)
        max_allowed_x = min(float(max_x), subj_left - margin_x)
    else:
        min_allowed_x = max(0.0, head_right + margin_x - crop_w)
        max_allowed_x = min(float(max_x), head_left - margin_x)

    if min_allowed_x > max_allowed_x:
        min_allowed_x = max_allowed_x = float(np.clip(subj_cx - crop_w / 2.0, 0.0, float(max_x)))

    corridor_info = {
        "min_valid_x": min_allowed_x,
        "max_valid_x": max_allowed_x,
        "corridor_width": max(0.0, max_allowed_x - min_allowed_x),
    }

    pref = 0.54 if gaze_dir < -0.10 else (0.46 if gaze_dir > 0.10 else 0.50)
    ideal_crop_x = subj_cx - crop_w * pref
    clamped_ideal_x = float(np.clip(ideal_crop_x, min_allowed_x, max_allowed_x))
    clamped_ideal_x = float(np.clip(clamped_ideal_x, 0.0, float(max_x)))

    deadband_x = max(35.0, crop_w * 0.09)

    if current_crop_x is not None:
        curr_x = float(np.clip(current_crop_x, 0.0, float(max_x)))
        head_contained = (head_left >= curr_x - 10.0) and (head_right <= curr_x + crop_w + 10.0)
        within_corridor = (curr_x >= min_allowed_x - 15.0) and (curr_x <= max_allowed_x + 15.0)
        if (head_contained or within_corridor) and abs(curr_x - clamped_ideal_x) <= deadband_x:
            return curr_x, True, corridor_info
        final_crop_x = clamped_ideal_x
    else:
        final_crop_x = clamped_ideal_x

    final_crop_x = float(np.clip(final_crop_x, 0.0, float(max_x)))

    if subj_w <= crop_w:
        is_contained = (eff_left >= final_crop_x - 2.0) and (eff_right <= final_crop_x + crop_w + 2.0)
    else:
        is_contained = (head_left >= final_crop_x - 2.0) and (head_right <= final_crop_x + crop_w + 2.0)

    if not is_contained:
        if head_right > final_crop_x + crop_w:
            final_crop_x = min(float(max_x), max(final_crop_x, head_right - crop_w + margin_x))
        elif head_left < final_crop_x:
            final_crop_x = max(0.0, min(final_crop_x, head_left - margin_x))
        final_crop_x = float(np.clip(final_crop_x, 0.0, float(max_x)))
        is_contained = (head_left >= final_crop_x - 2.0) and (head_right <= final_crop_x + crop_w + 2.0)

    return final_crop_x, is_contained, corridor_info


def solve_crop_y(target_face, crop_h: float, source_h: float, current_crop_y=None):
    """Solves vertical crop position. Eye-line ~30% from top. Vertical deadband applied."""
    if target_face is None:
        ideal_y = max(0.0, (source_h - crop_h) / 2.0)
        if current_crop_y is not None and abs(current_crop_y - ideal_y) < crop_h * 0.15:
            return float(np.clip(current_crop_y, 0.0, max(0.0, source_h - crop_h)))
        return float(np.clip(ideal_y, 0.0, max(0.0, source_h - crop_h)))

    fx, fy, fw, fh = target_face["bbox"]
    head_top_y = max(0.0, fy - 0.22 * fh)
    eye_line_y = target_face.get("eye_line_y", fy + 0.35 * fh)
    ideal_crop_y = min(eye_line_y - crop_h * 0.30, head_top_y - crop_h * 0.08)
    max_valid_y = max(0.0, source_h - crop_h)
    ideal_crop_y = float(np.clip(ideal_crop_y, 0.0, max_valid_y))

    if current_crop_y is not None:
        curr_y = float(np.clip(current_crop_y, 0.0, max_valid_y))
        head_in = head_top_y >= curr_y and head_top_y <= curr_y + crop_h
        eye_ok = abs((eye_line_y - curr_y) / max(1.0, crop_h) - 0.30) < 0.12
        if head_in and eye_ok:
            return curr_y

    return ideal_crop_y


# ─── 6. Composition Quality Scoring (Joint Position + Scale) ─────────────────

def evaluate_crop_composition_score(
    crop_x: float, crop_y: float, crop_w: float, crop_h: float, scale: float,
    active_face, all_faces: list, source_w: float, source_h: float,
    camera_motion_px: float = 0.0, scale_motion: float = 0.0, shot_scale_state=None,
) -> float:
    """
    Joint (position, scale) composition quality scorer.
    HARD: containment. SOFT: headroom, eye-line, scale quality, anti-furniture, motion penalties.
    """
    if active_face is None:
        return 0.50

    fx, fy, fw, fh = active_face["bbox"]
    cx, cy = active_face["center"]
    crop_right = crop_x + crop_w
    crop_bottom = crop_y + crop_h
    score = 0.0

    # Horizontal containment (hard)
    score += 5.0 if fx >= crop_x and fx + fw <= crop_right else -5.0
    # Vertical containment (hard)
    score += 1.0 if fy >= crop_y and fy + fh <= crop_bottom else -3.0

    # Shoulder margin
    min_margin = min(fx - crop_x, crop_right - (fx + fw))
    score += 2.0 if min_margin >= crop_w * 0.08 else (-3.0 if min_margin < 0 else 0.0)

    # Soft centering [38%, 62%]
    fpr = (cx - crop_x) / max(1.0, crop_w)
    if 0.38 <= fpr <= 0.62:
        score += 1.0
    elif fpr < 0.38:
        score -= 2.5 * (0.38 - fpr)
    else:
        score -= 2.5 * (fpr - 0.62)

    # Headroom
    head_top_y = active_face.get("head_top_y", fy - 0.22 * fh)
    if crop_h > 0:
        hpct = (head_top_y - crop_y) / crop_h
        if 0.08 <= hpct <= 0.20:
            score += 2.0
        elif hpct < 0.04:
            score -= 3.0
        elif hpct > 0.28:
            score -= 1.0

    # Eye-line
    ely = active_face.get("eye_line_y", fy + 0.35 * fh)
    if crop_h > 0:
        epct = (ely - crop_y) / crop_h
        score += 2.0 if 0.25 <= epct <= 0.38 else (-1.5 if epct < 0.18 or epct > 0.50 else 0.0)

    # Scale quality: face ~5-25% of portrait
    fac = (fw * fh) / max(1.0, crop_w * crop_h)
    if 0.05 <= fac <= 0.25:
        score += 2.0
    elif fac > 0.40:
        score -= 3.0 * (fac - 0.40) / 0.40
    elif fac < 0.02:
        score -= 2.0

    # Overzoom penalty
    if shot_scale_state is not None:
        if scale > shot_scale_state.max_valid_scale:
            score -= 4.0 * (scale - shot_scale_state.max_valid_scale)
        if scale > shot_scale_state.ideal_scale + SCALE_DEADBAND:
            score -= 1.5 * (scale - shot_scale_state.ideal_scale)

    # Anti-empty center
    clr = crop_x + crop_w * 0.30
    crr = crop_x + crop_w * 0.70
    if not any(clr <= f["center"][0] <= crr for f in all_faces if f.get("conf", 0) >= 0.40) and len(all_faces) >= 2:
        score -= 4.0

    # Motion penalties
    score -= min(3.0, 0.02 * camera_motion_px)
    score -= min(4.0, 3.0 * abs(scale_motion))
    return score


# ─── 7. Crash-Proof Video Ingestion & Decoder Fallback ───────────────────────

def decode_frames_via_ffmpeg_pipe(source_path: str, start_sec: float, dur_sec: float, target_fps: float = ANALYSIS_FPS):
    dw, dh = 640, 360
    cap = None
    try:
        cap = cv2.VideoCapture(str(source_path))
        w = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920.0)
        h = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080.0)
        if w > 0:
            dh = int(round(h * (640.0 / w)))
            if dh % 2 != 0:
                dh -= 1
    except Exception:
        dh = 360
    finally:
        if cap is not None:
            try: cap.release()
            except Exception: pass

    dh = max(180, dh)
    frame_bytes = dw * dh * 3
    if not os.path.exists(str(source_path)):
        return []

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error",
           "-ss", f"{start_sec:.3f}", "-t", f"{dur_sec:.3f}",
           "-i", str(source_path), "-vf", f"fps={target_fps},scale={dw}:{dh}",
           "-f", "image2pipe", "-vcodec", "rawvideo", "-pix_fmt", "bgr24", "-"]

    frames = []
    proc = None
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        t = 0.0
        dt = 1.0 / target_fps
        while True:
            raw = proc.stdout.read(frame_bytes)
            if not raw or len(raw) < frame_bytes:
                break
            frames.append((t, np.frombuffer(raw, dtype=np.uint8).reshape((dh, dw, 3))))
            t += dt
    except Exception as e:
        sys.stderr.write(f"[Smart Framing] FFmpeg pipe decode error: {e}\n")
    finally:
        if proc is not None:
            try:
                if proc.stdout: proc.stdout.close()
                if proc.stderr: proc.stderr.close()
                proc.terminate()
                proc.wait(timeout=1.0)
            except Exception:
                pass
    return frames


# ─── 7b. Memory-Budgeted Sample Storage ──────────────────────────────────────
# INVARIANT (v12.2): The analysis sampler MUST cover the FULL clip range.
# Storing full-resolution frames for long 4K clips exhausts memory mid-clip,
# silently truncating sampling (OpenCV read OOM -> loop break -> partial
# samples still pass the >=3 check). Any late-scene change in the unsampled
# tail is then invisible to the framing pipeline and the crop freezes on a
# stale target. To guarantee full coverage, stored frames are downscaled to
# fit a memory budget; detection coordinates are kept in TRUE source space.

# Default total memory budget for stored analysis frames (bytes).
_DEFAULT_SAMPLE_MEM_BUDGET_BYTES = 3200 * 1024 * 1024  # ~3.2 GB
# Minimum stored-frame width (Ultralytics letterboxes to 640 anyway; YuNet
# internally analyzes at <=640 width, so tracking quality is unaffected).
_MIN_STORED_FRAME_WIDTH = 640

# True source geometry recorded by the sampler (true_w, true_h).
# None when sampling was mocked/absent (tests) -> consumers fall back to
# the first stored frame's shape, and no coordinate rescaling occurs.
_SOURCE_GEOMETRY = None


def _get_sample_mem_budget_bytes():
    """Total stored-frame memory budget, overridable via env for testing."""
    raw = os.environ.get("AUTOSHORTS_SAMPLE_MEM_BUDGET_MB")
    if raw:
        try:
            mb = float(raw)
            if mb > 0:
                return int(mb * 1024 * 1024)
        except ValueError:
            pass
    return _DEFAULT_SAMPLE_MEM_BUDGET_BYTES


def _compute_stored_frame_width(true_w: float, true_h: float, n_frames_est: float) -> int:
    """
    Largest even stored-frame width (<= true_w) such that n_frames_est frames
    of (store_w x store_w*true_h/true_w x 3) fit within the memory budget.
    Never below _MIN_STORED_FRAME_WIDTH; equals true_w when the budget allows.
    """
    true_w = max(1.0, float(true_w))
    true_h = max(1.0, float(true_h))
    n_frames_est = max(1.0, float(n_frames_est))

    budget = _get_sample_mem_budget_bytes()
    # bytes per frame at width W: W * (W * true_h / true_w) * 3
    # W^2 <= budget * true_w / (3 * true_h * n_frames_est)
    max_w = math.sqrt((budget * true_w) / (3.0 * true_h * n_frames_est))
    store_w = int(min(true_w, math.floor(max_w)))
    store_w = max(_MIN_STORED_FRAME_WIDTH, min(int(true_w), store_w))
    if store_w % 2 != 0:
        store_w -= 1
    return max(2, store_w)


def _scale_face_to_space(face: dict, ratio: float) -> dict:
    """Return a copy of `face` with pixel geometry scaled by `ratio`."""
    if ratio == 1.0:
        return face
    f = dict(face)
    fx, fy, fw, fh = f["bbox"]
    f["bbox"] = (fx * ratio, fy * ratio, fw * ratio, fh * ratio)
    if "center" in f:
        cx, cy = f["center"]
        f["center"] = (cx * ratio, cy * ratio)
    if "area" in f:
        f["area"] = float(f["area"]) * ratio * ratio
    if "landmarks" in f and f["landmarks"] is not None:
        f["landmarks"] = tuple((x * ratio, y * ratio) for (x, y) in f["landmarks"])
    if "shoulder_span" in f and f["shoulder_span"] is not None:
        sl, sr = f["shoulder_span"]
        f["shoulder_span"] = (sl * ratio, sr * ratio)
    for k in ("head_top_y", "eye_line_y", "torso_top"):
        if k in f and f[k] is not None:
            f[k] = float(f[k]) * ratio
    if "person_bbox" in f and f["person_bbox"] is not None:
        px1, py1, px2, py2 = f["person_bbox"]
        f["person_bbox"] = (px1 * ratio, py1 * ratio, px2 * ratio, py2 * ratio)
    # mouth_gray: pixel image crop; kept unscaled (only used for 32x20-resized
    # temporal mouth-motion diffs, which are scale-invariant by construction).
    return f


def _scale_shot_tracks_to_source_space(shot_tracks: dict, ratio: float) -> dict:
    """
    Scale track geometry produced on stored (downscaled) frames back to TRUE
    source space. Returns the same dict with scaled values.
    """
    if ratio == 1.0 or not shot_tracks:
        return shot_tracks
    for tr in shot_tracks.values():
        for key in ("cx", "cy", "vx", "vy"):
            if key in tr and tr[key] is not None:
                tr[key] = float(tr[key]) * ratio
        if tr.get("person_bbox") is not None:
            px1, py1, px2, py2 = tr["person_bbox"]
            tr["person_bbox"] = (px1 * ratio, py1 * ratio, px2 * ratio, py2 * ratio)
        dets = tr.get("detections")
        if dets:
            tr["detections"] = [(d[0], _scale_face_to_space(d[1], ratio), d[2]) for d in dets]
    return shot_tracks


def sample_video_and_detect_shots(source_path, start_ms, end_ms=None):
    if isinstance(start_ms, (list, np.ndarray)):
        sample_times_ms = start_ms
        actual_start_ms = float(end_ms) if end_ms is not None else float(sample_times_ms[0])
        actual_end_ms = float(sample_times_ms[-1]) if len(sample_times_ms) > 0 else actual_start_ms + 30000.0
    else:
        actual_start_ms = float(start_ms)
        actual_end_ms = float(end_ms) if end_ms is not None else actual_start_ms + 30000.0

    start_sec = actual_start_ms / 1000.0
    end_sec = actual_end_ms / 1000.0
    dur_sec = max(0.1, end_sec - start_sec)

    global _SOURCE_GEOMETRY
    _SOURCE_GEOMETRY = None

    yunet = find_yunet_model()
    detector = create_detector(yunet)
    scene_cuts = [0.0]
    all_samples = []
    prev_small = None
    cv_success = False

    cap = None
    try:
        if isinstance(source_path, cv2.VideoCapture):
            cap = source_path
        else:
            cap = cv2.VideoCapture(str(source_path))

        if cap.isOpened():
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
            cap.set(cv2.CAP_PROP_POS_MSEC, actual_start_ms)
            frame_step = max(1, int(round(fps / ANALYSIS_FPS)))
            frame_count = 0

            # Memory-budgeted storage: learn true geometry from the first
            # frame, then pick a stored-frame width so the ENTIRE clip's
            # samples fit the budget (full-coverage invariant).
            store_w = None
            store_ratio = 1.0  # stored_width / true_width

            while True:
                try:
                    ret, frame = cap.read()
                except (SystemError, cv2.error, Exception) as read_err:
                    sys.stderr.write(f"[Smart Framing] OpenCV read error: {read_err} -> FFmpeg fallback\n")
                    break

                if not ret or frame is None:
                    break

                rel_t = frame_count / fps
                if rel_t > dur_sec + 0.1:
                    break

                if frame_count % frame_step == 0:
                    if store_w is None:
                        true_h, true_w = frame.shape[:2]
                        n_est = max(1.0, (dur_sec * ANALYSIS_FPS) + 4.0)
                        store_w = _compute_stored_frame_width(true_w, true_h, n_est)
                        store_ratio = float(store_w) / float(true_w)
                        _SOURCE_GEOMETRY = (float(true_w), float(true_h))
                        if store_w < true_w:
                            sys.stderr.write(
                                f"[Smart Framing] memory-budgeted sampling: "
                                f"storing {store_w}px frames (source {true_w}x{true_h}, "
                                f"ratio={store_ratio:.4f})\n"
                            )
                            sys.stderr.flush()

                    # Face detection ALWAYS runs on the TRUE-resolution frame:
                    # absolute-area heuristics (is_back etc.) and all downstream
                    # consumers operate in true source space.
                    faces = detect_faces_and_subjects(detector, frame)

                    small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (160, 90))
                    if prev_small is not None:
                        diff = float(np.mean(cv2.absdiff(small, prev_small)))
                        h1 = cv2.calcHist([small], [0], None, [32], [0, 256])
                        h2 = cv2.calcHist([prev_small], [0], None, [32], [0, 256])
                        corr = float(cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL))
                        if diff > SCENE_CUT_DIFF or (diff > 16.0 and corr < SCENE_CUT_HIST):
                            if rel_t - scene_cuts[-1] > 0.35:
                                scene_cuts.append(rel_t)
                    prev_small = small

                    if store_ratio < 1.0:
                        stored_frame = cv2.resize(frame, (store_w, max(2, int(round(true_h * store_ratio)))))
                    else:
                        stored_frame = frame

                    all_samples.append((rel_t, stored_frame, faces))

                frame_count += 1

            cv_success = len(all_samples) >= 3
    except Exception as e:
        sys.stderr.write(f"[Smart Framing] OpenCV init error: {e}\n")
    finally:
        if cap is not None and not isinstance(source_path, cv2.VideoCapture):
            try: cap.release()
            except Exception: pass

    if not cv_success or len(all_samples) < 3:
        sys.stderr.write("[Smart Framing] Running FFmpeg fallback decoder...\n")
        # The pipe decoder emits 640px frames and faces are detected in that
        # pipe-space; downstream consumers use the stored frame shape as the
        # source geometry (legacy behavior). Never mix pipe-space samples
        # with true-source geometry recorded above.
        _SOURCE_GEOMETRY = None
        pipe_frames = decode_frames_via_ffmpeg_pipe(source_path, start_sec, dur_sec, ANALYSIS_FPS)
        if pipe_frames:
            scene_cuts = [0.0]
            all_samples = []
            prev_small = None
            for rel_t, frame in pipe_frames:
                small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (160, 90))
                if prev_small is not None:
                    diff = float(np.mean(cv2.absdiff(small, prev_small)))
                    h1 = cv2.calcHist([small], [0], None, [32], [0, 256])
                    h2 = cv2.calcHist([prev_small], [0], None, [32], [0, 256])
                    corr = float(cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL))
                    if diff > SCENE_CUT_DIFF or (diff > 16.0 and corr < SCENE_CUT_HIST):
                        if rel_t - scene_cuts[-1] > 0.35:
                            scene_cuts.append(rel_t)
                prev_small = small
                all_samples.append((rel_t, frame, detect_faces_and_subjects(detector, frame)))

    # Full-coverage invariant: warn loudly if sampling stopped early. A
    # truncated sample set freezes the crop on a stale target for the
    # unsampled tail (the exact failure mode this guard exists for).
    if all_samples and dur_sec > 1.0:
        last_rel_t = all_samples[-1][0]
        if last_rel_t < dur_sec - 1.0:
            sys.stderr.write(
                f"[Smart Framing] WARNING: sampling truncated at t={last_rel_t:.2f}s "
                f"of {dur_sec:.2f}s clip; late-scene framing may be stale\n"
            )
            sys.stderr.flush()

    return scene_cuts if scene_cuts else [0.0], all_samples


# ─── 8. Scene-Level Subject Tracking ──────────────────────────────────────────

def compute_camera_motion_compensation(prev_frame, curr_frame):
    """
    Global Motion Compensation (GMC) affine camera translation (dx, dy).
    Tracks optical flow of background corner features on downscaled 160x90 frames.
    Decouples camera panning from true subject motion.
    """
    if prev_frame is None or curr_frame is None:
        return 0.0, 0.0
    try:
        def _to_small_gray(img):
            if len(img.shape) == 3:
                g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                g = img
            h, w = g.shape[:2]
            if w != 480 or h != 270:
                g = cv2.resize(g, (480, 270))
            return g, float(w) / 480.0

        g1, s1 = _to_small_gray(prev_frame)
        g2, _ = _to_small_gray(curr_frame)

        p0 = cv2.goodFeaturesToTrack(g1, maxCorners=100, qualityLevel=0.01, minDistance=6)
        if p0 is None or len(p0) < 6:
            return 0.0, 0.0

        p1, st, _ = cv2.calcOpticalFlowPyrLK(g1, g2, p0, None, winSize=(15, 15), maxLevel=2)
        if p1 is None or st is None:
            return 0.0, 0.0

        good_p0 = p0[st == 1]
        good_p1 = p1[st == 1]
        if len(good_p0) < 6:
            return 0.0, 0.0

        M, inliers = cv2.estimateAffinePartial2D(good_p0, good_p1)
        if M is None:
            return 0.0, 0.0

        dx = float(M[0, 2]) * s1
        dy = float(M[1, 2]) * s1
        return dx, dy
    except Exception:
        return 0.0, 0.0


class NativeTracker:
    """
    AutoShorts 4.x Native Computer Vision Tracking Backend.
    Uses Lucas-Kanade optical flow Global Motion Compensation (GMC) + 
    2-stage ByteTrack face association with Kalman/velocity motion prediction.
    """
    @staticmethod
    def track_shot(shot_samples):
        shot_tracks = {}
        next_tid = 0
        prev_frame = None

        for sample in shot_samples:
            rel_t = sample[0]
            frame = sample[1] if len(sample) > 1 else None
            faces = sample[2] if len(sample) > 2 else []

            # 1. Global Motion Compensation (GMC)
            dx, dy = 0.0, 0.0
            if prev_frame is not None and frame is not None:
                dx, dy = compute_camera_motion_compensation(prev_frame, frame)
            prev_frame = frame

            # Apply GMC camera translation to all active tracks
            if abs(dx) > 0.1 or abs(dy) > 0.1:
                for tr in shot_tracks.values():
                    if rel_t - tr["last_t"] <= 1.2:
                        tr["cx"] += dx
                        tr["cy"] += dy

            # Separate detections into high and low confidence
            high_dets = []
            low_dets = []
            for face in faces:
                conf = face.get("conf", 0.0)
                if conf >= 0.55:
                    high_dets.append(face)
                elif conf >= 0.30:
                    low_dets.append(face)

            active_track_ids = [
                tid for tid, tr in shot_tracks.items()
                if rel_t - tr["last_t"] <= 1.2
            ]

            matched_tracks = set()
            matched_high_dets = set()

            def _match_tracks_to_dets(track_ids, det_list, dist_thresh):
                if not track_ids or not det_list:
                    return []
                cost_matrix = np.zeros((len(track_ids), len(det_list)), dtype=np.float32)
                for i, tid in enumerate(track_ids):
                    tr = shot_tracks[tid]
                    dt = max(0.01, rel_t - tr["last_t"])
                    pred_cx = tr["cx"] + tr.get("vx", 0.0) * dt
                    pred_cy = tr["cy"] + tr.get("vy", 0.0) * dt
                    for j, det in enumerate(det_list):
                        dcx, dcy = det["center"]
                        cost_matrix[i, j] = math.hypot(pred_cx - dcx, pred_cy - dcy)

                if HAS_SCIPY:
                    row_ind, col_ind = linear_sum_assignment(cost_matrix)
                    pairs = []
                    for r, c in zip(row_ind, col_ind):
                        if cost_matrix[r, c] <= dist_thresh:
                            pairs.append((track_ids[r], c))
                    return pairs
                else:
                    pairs = []
                    used_c = set()
                    for r in range(len(track_ids)):
                        best_c, best_d = None, dist_thresh
                        for c in range(len(det_list)):
                            if c not in used_c and cost_matrix[r, c] < best_d:
                                best_d, best_c = cost_matrix[r, c], c
                        if best_c is not None:
                            pairs.append((track_ids[r], best_c))
                            used_c.add(best_c)
                    return pairs

            def _update_tr(tr, face):
                cx, cy = face["center"]
                dt = max(0.01, rel_t - tr["last_t"])
                inst_vx = (cx - tr["cx"]) / dt
                inst_vy = (cy - tr["cy"]) / dt
                tr["vx"] = 0.40 * inst_vx + 0.60 * tr.get("vx", 0.0)
                tr["vy"] = 0.40 * inst_vy + 0.60 * tr.get("vy", 0.0)
                tr["cx"] = 0.35 * cx + 0.65 * tr["cx"]
                tr["cy"] = 0.35 * cy + 0.65 * tr["cy"]

                m_diff = 0.0
                if tr["last_mouth"] is not None and face.get("mouth_gray") is not None:
                    try:
                        p_r = cv2.resize(tr["last_mouth"], (32, 20))
                        c_r = cv2.resize(face["mouth_gray"], (32, 20))
                        m_diff = float(np.mean(cv2.absdiff(p_r, c_r)))
                    except Exception:
                        pass

                tr["last_t"] = rel_t
                tr["last_mouth"] = face.get("mouth_gray")
                tr["detections"].append((rel_t, face, m_diff))
                tr["hits"] = tr.get("hits", 0) + 1
                tr["age"] = tr.get("age", 0) + 1
                tr["is_occluded"] = False

            # Stage 1: High confidence detections
            stage1_matches = _match_tracks_to_dets(active_track_ids, high_dets, dist_thresh=180.0)
            for tid, det_idx in stage1_matches:
                matched_tracks.add(tid)
                matched_high_dets.add(det_idx)
                _update_tr(shot_tracks[tid], high_dets[det_idx])

            # Stage 2: Low confidence detections for remaining tracks
            unmatched_track_ids = [tid for tid in active_track_ids if tid not in matched_tracks]
            stage2_matches = _match_tracks_to_dets(unmatched_track_ids, low_dets, dist_thresh=140.0)
            for tid, det_idx in stage2_matches:
                matched_tracks.add(tid)
                _update_tr(shot_tracks[tid], low_dets[det_idx])

            # Occlusion handling for unmatched tracks
            for tid in active_track_ids:
                if tid not in matched_tracks:
                    shot_tracks[tid]["is_occluded"] = True
                    dt = max(0.01, rel_t - shot_tracks[tid]["last_t"])
                    if dt <= 0.60:
                        shot_tracks[tid]["cx"] += shot_tracks[tid].get("vx", 0.0) * dt
                        shot_tracks[tid]["cy"] += shot_tracks[tid].get("vy", 0.0) * dt

            # New tracks from unmatched high confidence detections
            for det_idx, face in enumerate(high_dets):
                if det_idx not in matched_high_dets:
                    cx, cy = face["center"]
                    tid = next_tid
                    next_tid += 1
                    shot_tracks[tid] = {
                        "cx": cx, "cy": cy, "vx": 0.0, "vy": 0.0, "last_t": rel_t,
                        "last_mouth": face.get("mouth_gray"), "detections": [(rel_t, face, 0.0)],
                        "hits": 1, "age": 1, "is_occluded": False,
                        "provenance": "native_face",
                        "has_person_body": True,
                    }

        # Extract Re-ID embeddings for all tracks after processing
        reid_manager = get_reid_manager()
        if reid_manager.model is not None and shot_tracks:
            # Extract embeddings for all tracks
            tracks_list = list(shot_tracks.values())
            if tracks_list:
                embeddings = reid_manager.extract_embeddings(frame, tracks_list)
                for i, emb in enumerate(embeddings):
                    if i < len(tracks_list) and emb is not None:
                        tracks_list[i]["reid_embedding"] = emb

        return shot_tracks


class UltralyticsTracker:
    """
    AutoShorts 5.0 Official Ultralytics CV Tracking Backend.
    Uses official Ultralytics YOLO11n person detection + BoT-SORT multi-object tracker
    with native sparseOptFlow Global Motion Compensation (GMC).
    Fuses tracked person bodies with YuNet facial landmarks and mouth dynamics.
    External Lucas-Kanade GMC is bypassed in this backend to avoid double-compensation.
    """
    def __init__(self, model):
        self.model = model
        self._reset_tracker()

    def _reset_tracker(self):
        """Reset internal BoT-SORT tracker state at shot boundaries."""
        if hasattr(self.model, 'predictor') and hasattr(self.model.predictor, 'trackers'):
            for t in self.model.predictor.trackers:
                if hasattr(t, 'reset'):
                    t.reset()

    def track_shot(self, shot_samples):
        # Reset BoT-SORT tracker so tracking IDs never bleed across shot cuts
        self._reset_tracker()
        sys.stderr.write("[CV Runtime] BoT-SORT tracking started\n")
        sys.stderr.flush()
        shot_tracks = {}
        fallback_next_tid = 1000
        total_detections = 0

        # Initialize Re-ID manager for this shot
        reid_manager = get_reid_manager()

        for sample in shot_samples:
            rel_t = sample[0]
            frame = sample[1] if len(sample) > 1 else None
            faces = sample[2] if len(sample) > 2 else []

            if frame is None:
                continue

            # Run official Ultralytics BoT-SORT tracking for person class (classes=[0])
            # sparseOptFlow GMC is performed inside BoT-SORT; external GMC is bypassed
            #
            # Detection runs at the model's working scale, not the source scale. YOLO
            # letterboxes to imgsz=640 internally, so a 1920px input costs ~2.3x the
            # preprocessing for zero detection-quality gain (measured: 766ms -> 339ms
            # per frame). Boxes are scaled back to TRUE source space immediately, so
            # every downstream consumer (face matching, Re-ID crops, composition,
            # containment) keeps working in the same coordinate system as before.
            infer_frame = frame
            infer_scale = 1.0
            fw = frame.shape[1]
            if fw > PERSON_DETECT_INFER_WIDTH:
                target_w = PERSON_DETECT_INFER_WIDTH
                if target_w < PERSON_DETECT_MIN_WIDTH:
                    target_w = PERSON_DETECT_MIN_WIDTH
                if target_w < fw:
                    infer_h = max(2, int(round(frame.shape[0] * (target_w / float(fw)))))
                    if infer_h % 2 != 0:
                        infer_h -= 1
                    infer_frame = cv2.resize(frame, (target_w, infer_h),
                                             interpolation=cv2.INTER_AREA)
                    infer_scale = float(fw) / float(target_w)

            results = self.model.track(
                source=infer_frame,
                persist=True,
                tracker="botsort.yaml",
                classes=[0],
                conf=0.25,
                verbose=False,
            )

            person_tracks = []
            if results and len(results) > 0:
                boxes = results[0].boxes
                if boxes is not None and boxes.xyxy is not None and len(boxes.xyxy) > 0:
                    xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, 'cpu') else np.array(boxes.xyxy)
                    confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, 'cpu') else np.array(boxes.conf)
                    if boxes.id is not None:
                        tids = boxes.id.cpu().numpy().astype(int) if hasattr(boxes.id, 'cpu') else np.array(boxes.id, dtype=int)
                    else:
                        tids = [-1] * len(xyxy)

                    for i in range(len(xyxy)):
                        box = xyxy[i]
                        conf = float(confs[i]) if i < len(confs) else 0.5
                        tid = int(tids[i]) if i < len(tids) else -1
                        # Map the detection from inference space back to TRUE
                        # source space so every consumer below (YuNet face
                        # matching, Re-ID crops, composition, containment) sees
                        # the same coordinate system it did before the
                        # detection-scale change. No-op when unscaled.
                        bx1 = float(box[0]) * infer_scale
                        by1 = float(box[1]) * infer_scale
                        bx2 = float(box[2]) * infer_scale
                        by2 = float(box[3]) * infer_scale
                        person_tracks.append({
                            "track_id": tid,
                            "bbox": (bx1, by1, bx2, by2),
                            "conf": conf
                        })

            total_detections += len(person_tracks)

            # Re-ID embeddings are deliberately NOT extracted per frame here.
            # This loop runs once per analysis frame (8 fps => ~640 calls for a
            # 80s candidate) and each call ran a batched OSNet forward pass over
            # EVERY person in that frame. attach_reid_embeddings() then extracted
            # them AGAIN per shot (up to 8 best-confidence detections per track),
            # so the identical crops were embedded twice: once wastefully per
            # frame, once usefully per shot. The per-frame pass is the pure
            # duplicate and its removal changes no downstream value -- Re-ID
            # embeddings still reach the gallery and the identity association,
            # because attach_reid_embeddings() is the sole consumer of
            # `reid_embedding` on these tracks.

            # Match YuNet faces to YOLO person tracks
            used_faces = set()
            for p in person_tracks:
                px1, py1, px2, py2 = p["bbox"]
                pw = px2 - px1
                ph = py2 - py1

                # Find candidate YuNet faces whose center is within the upper 65% of the person box
                best_face_idx = None
                best_score = -1.0
                for f_idx, face in enumerate(faces):
                    if f_idx in used_faces:
                        continue
                    fcx, fcy = face["center"]
                    if (px1 - 0.15 * pw) <= fcx <= (px2 + 0.15 * pw):
                        if (py1 - 0.20 * ph) <= fcy <= (py1 + 0.65 * ph):
                            dist_to_head = abs(fcx - (px1 + px2) / 2.0) / max(1.0, pw) + abs(fcy - (py1 + ph * 0.20)) / max(1.0, ph)
                            score = face.get("conf", 0.5) - 0.3 * dist_to_head
                            if score > best_score:
                                best_score = score
                                best_face_idx = f_idx

                if best_face_idx is not None:
                    matched_face = faces[best_face_idx]
                    used_faces.add(best_face_idx)
                    matched_face["person_bbox"] = (px1, py1, px2, py2)
                else:
                    head_w = max(32.0, pw * 0.35)
                    head_h = max(32.0, ph * 0.25)
                    hfx = (px1 + px2) / 2.0 - head_w / 2.0
                    hfy = max(0.0, py1)
                    matched_face = {
                        "bbox": (hfx, hfy, head_w, head_h),
                        "center": (hfx + head_w / 2.0, hfy + head_h / 2.0),
                        "conf": float(p["conf"] * 0.70),
                        "area": head_w * head_h,
                        "frontality": 0.20,
                        "gaze_offset_x": 0.0,
                        "skin_ratio": 0.30,
                        "is_back": True,
                        "head_top_y": max(0.0, py1),
                        "eye_line_y": py1 + 0.35 * head_h,
                        "shoulder_span": (px1, px2),
                        "torso_top": py1 + head_h,
                        "mouth_gray": None,
                        "landmarks": None,
                        "person_bbox": (px1, py1, px2, py2)
                    }

                tid = p["track_id"]
                if tid < 0:
                    tid = fallback_next_tid
                    fallback_next_tid += 1

                cx, cy = matched_face["center"]
                m_diff = 0.0
                if tid in shot_tracks:
                    tr = shot_tracks[tid]
                    if tr["last_mouth"] is not None and matched_face.get("mouth_gray") is not None:
                        try:
                            p_r = cv2.resize(tr["last_mouth"], (32, 20))
                            c_r = cv2.resize(matched_face["mouth_gray"], (32, 20))
                            m_diff = float(np.mean(cv2.absdiff(p_r, c_r)))
                        except Exception:
                            pass
                    dt = max(0.01, rel_t - tr["last_t"])
                    inst_vx = (cx - tr["cx"]) / dt
                    inst_vy = (cy - tr["cy"]) / dt
                    tr["vx"] = 0.40 * inst_vx + 0.60 * tr.get("vx", 0.0)
                    tr["vy"] = 0.40 * inst_vy + 0.60 * tr.get("vy", 0.0)
                    tr["cx"] = 0.35 * cx + 0.65 * tr["cx"]
                    tr["cy"] = 0.35 * cy + 0.65 * tr["cy"]
                    tr["last_t"] = rel_t
                    tr["last_mouth"] = matched_face.get("mouth_gray")
                    tr["detections"].append((rel_t, matched_face, m_diff))
                    tr["hits"] = tr.get("hits", 0) + 1
                    tr["age"] = tr.get("age", 0) + 1
                    tr["is_occluded"] = False
                    tr["person_bbox"] = (px1, py1, px2, py2)
                    tr["provenance"] = "yolo_person"
                    tr["has_person_body"] = True
                else:
                    shot_tracks[tid] = {
                        "cx": cx,
                        "cy": cy,
                        "vx": 0.0,
                        "vy": 0.0,
                        "last_t": rel_t,
                        "last_mouth": matched_face.get("mouth_gray"),
                        "detections": [(rel_t, matched_face, 0.0)],
                        "hits": 1,
                        "age": 1,
                        "is_occluded": False,
                        "person_bbox": (px1, py1, px2, py2),
                        "provenance": "yolo_person",
                        "has_person_body": True,
                    }

            # Safeguard: if there are high-confidence YuNet faces not covered by any YOLO person box
            # (e.g. extreme close-up headshots where body is outside frame), include them
            for f_idx, face in enumerate(faces):
                if f_idx not in used_faces and face.get("conf", 0.0) >= 0.50:
                    fcx, fcy = face["center"]
                    matched_tid = None
                    for tid, tr in shot_tracks.items():
                        if rel_t - tr["last_t"] <= 0.60 and math.hypot(tr["cx"] - fcx, tr["cy"] - fcy) < 150.0:
                            matched_tid = tid
                            break
                    if matched_tid is not None:
                        tr = shot_tracks[matched_tid]
                        tr["cx"] = 0.35 * fcx + 0.65 * tr["cx"]
                        tr["cy"] = 0.35 * fcy + 0.65 * tr["cy"]
                        tr["last_t"] = rel_t
                        tr["detections"].append((rel_t, face, 0.0))
                        tr["hits"] = tr.get("hits", 0) + 1
                    else:
                        tid = fallback_next_tid
                        fallback_next_tid += 1
                        shot_tracks[tid] = {
                            "cx": fcx,
                            "cy": fcy,
                            "vx": 0.0,
                            "vy": 0.0,
                            "last_t": rel_t,
                            "last_mouth": face.get("mouth_gray"),
                            "detections": [(rel_t, face, 0.0)],
                            "hits": 1,
                            "age": 1,
                            "is_occluded": False,
                            "person_bbox": None,
                            "provenance": "yunet_face_only",
                            "has_person_body": False,
                        }

        active_tracks = len(shot_tracks)
        sys.stderr.write(f"[CV Runtime] detections={total_detections}\n")
        sys.stderr.write(f"[CV Runtime] active_tracks={active_tracks}\n")
        sys.stderr.flush()
        return shot_tracks


_CURRENT_BACKEND = None
_BACKEND_HEADER_PRINTED = False


def build_shot_tracks(shot_samples):
    """
    Unified Multi-Object Subject Tracking Entry Point.
    Supports dual backends:
      - 'ultralytics' (Default): YOLO11n + BoT-SORT with native sparseOptFlow GMC.
      - 'native': OpenCV Lucas-Kanade GMC + 2-stage ByteTrack face association.
    Selected via AUTOSHORTS_CV_BACKEND environment variable.
    """
    global _CURRENT_BACKEND, _BACKEND_HEADER_PRINTED
    backend = os.environ.get("AUTOSHORTS_CV_BACKEND", "ultralytics").strip().lower()
    if backend != _CURRENT_BACKEND:
        _CURRENT_BACKEND = backend
        _BACKEND_HEADER_PRINTED = False

    t_start = time.time()

    if backend != "ultralytics":
        shot_tracks = NativeTracker.track_shot(shot_samples)
        if not _BACKEND_HEADER_PRINTED:
            sys.stderr.write(f"[CV Backend] backend={backend}\n")
            sys.stderr.flush()
            _BACKEND_HEADER_PRINTED = True
        return shot_tracks

    # Ultralytics mode: requires available model, runtime, and real video frames
    has_frames = any(sample[1] is not None for sample in shot_samples) if shot_samples else False
    if has_frames and HAS_ULTRALYTICS:
        try:
            model = get_ultralytics_model()
            if model:
                device_str = "cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu"
                if not _BACKEND_HEADER_PRINTED:
                    sys.stderr.write(
                        "[CV Backend] backend=ultralytics\n"
                        "[CV Backend] model=yolo11n.pt\n"
                        "[CV Backend] tracker=botsort\n"
                        "[CV Backend] gmc=sparseOptFlow\n"
                        f"[CV Backend] device={device_str}\n"
                        "[CV Runtime] YOLO model loaded\n"
                    )
                    sys.stderr.flush()
                    _BACKEND_HEADER_PRINTED = True

                tracker = UltralyticsTracker(model)
                shot_tracks = tracker.track_shot(shot_samples)
                return shot_tracks
        except Exception as e:
            # Genuine last-resort fallback only. The concrete error type, message
            # and traceback are surfaced so a real regression is never hidden
            # behind a silent downgrade to the native tracker.
            sys.stderr.write(
                f"[CV Backend] Ultralytics tracking error: {type(e).__name__}: {e}; "
                "falling back to native\n"
            )
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()

    # Fallback to NativeTracker (for synthetic unit tests where frame is None or when Ultralytics is disabled)
    shot_tracks = NativeTracker.track_shot(shot_samples)
    if not _BACKEND_HEADER_PRINTED:
        sys.stderr.write("[CV Backend] backend=native\n")
        sys.stderr.flush()
        _BACKEND_HEADER_PRINTED = True
    return shot_tracks


def evaluate_camera_mode(track_positions, source_w: float) -> str:
    if len(track_positions) < 3:
        return "STATIC"
    positions = np.array(track_positions)
    std_dev = float(np.std(positions))
    total_drift = float(abs(positions[-1] - positions[0]))
    if std_dev <= source_w * 0.015 and total_drift <= source_w * 0.025:
        return "STATIC"
    diffs = np.diff(positions)
    if (np.mean(diffs > 0) > 0.75 or np.mean(diffs < 0) > 0.75) and total_drift > source_w * 0.025:
        return "PAN"
    return "TRACK"


def compute_visual_prominence(
    track_or_face: dict,
    source_w: float = 1920.0,
    source_h: float = 1080.0,
    max_area_in_shot: float = None,
    weights: dict = None,
    config: FramingConfig = None,
) -> float:
    """
    Computes a multi-factor visual prominence score in [0.0, 1.0].
    
    Factors considered:
    1. Relative bounding box area (face / subject area relative to frame and shot)
    2. Face visibility / frontality (facing camera vs profile vs back of head)
    3. Detection / track confidence
    4. Screen composition / centrality (distance from center, avoiding extreme edge cutoffs)
    5. Temporal stability (persistent presence in shot / track length)
    """
    cfg = config or DEFAULT_FRAMING_CONFIG
    w = {
        "area": cfg.w_area,
        "frontality": cfg.w_frontality,
        "confidence": cfg.w_conf,
        "centrality": cfg.w_centrality,
        "stability": cfg.w_stability,
    }
    if weights:
        w.update(weights)

    if "detections" in track_or_face:
        dets = [d[1] for d in track_or_face["detections"]]
        area = float(np.mean([d["area"] for d in dets])) if dets else 0.0
        conf = float(np.mean([d["conf"] for d in dets])) if dets else 0.0
        front = float(np.mean([d["frontality"] for d in dets])) if dets else 0.5
        cx = track_or_face.get("cx", source_w / 2.0)
        n_dets = len(track_or_face["detections"])
    else:
        area = float(track_or_face.get("area", 0.0))
        conf = float(track_or_face.get("conf", 0.0))
        front = float(track_or_face.get("frontality", 0.5))
        cx = float(track_or_face.get("center", (source_w / 2.0, source_h / 2.0))[0])
        n_dets = 1

    frame_area = max(1.0, source_w * source_h)
    norm_frame_area = min(1.0, area / (0.08 * frame_area))
    if max_area_in_shot and max_area_in_shot > 0:
        norm_shot_area = min(1.0, area / max_area_in_shot)
        norm_area = 0.5 * norm_frame_area + 0.5 * norm_shot_area
    else:
        norm_area = norm_frame_area

    dist_from_center = abs(cx - source_w / 2.0) / (source_w / 2.0)
    centrality = max(0.0, 1.0 - min(1.0, dist_from_center))
    stability = min(1.0, n_dets / 4.0)

    score = (
        w["area"] * norm_area +
        w["frontality"] * front +
        w["confidence"] * conf +
        w["centrality"] * centrality +
        w["stability"] * stability
    )
    return float(np.clip(score, 0.0, 1.0))


def _has_recent_detection(tr, rel_ts: float, rel_te: float, max_lost: float) -> bool:
    """True if the track was detected recently enough to be a live selection candidate.

    Tracks whose last detection predates rel_te by more than max_lost seconds are
    stale: they may still win motion/prominence comparisons from historical context,
    but their cached face geometry no longer describes who is on screen.
    """
    dets = tr.get("detections") or []
    if not dets:
        return False
    last_t = max(d[0] for d in dets)
    return (rel_te - last_t) <= max_lost


# ─── Same-Person Track Fragment Merge (Adaptive two-person integrity) ─────────
# BoT-SORT occasionally splits ONE physical person into 2+ track ids (ID
# switch on occlusion/motion). Both fragments then carry real face detections,
# pass the genuine-track filters, and — left unmerged — fabricate a
# "two-person" composition for a SOLO shot (two-shot centered between a face
# and its own duplicate). Distinct people standing in frame are separated by
# well over a face width; fragments of one person either overlap in time at
# nearly the same position or hand off to each other within a small gap.

_SAME_PERSON_PROXIMITY_RATIO = 0.60  # center distance as fraction of face width
_SAME_PERSON_TIME_TOL = 0.30         # |Δt| for "simultaneous" detections
_SAME_PERSON_MAX_GAP = 1.00          # max temporal gap for a continuation hand-off


def _track_valid_dets(tr):
    return [d for d in (tr.get("detections") or [])
            if isinstance(d[1], dict) and d[1].get("bbox") and d[1].get("center")]


def _mean_face_width(dets):
    if not dets:
        return 0.0
    return float(np.mean([float(d[1]["bbox"][2]) for d in dets]))


def _tracks_same_person(tr_a, tr_b):
    """True when two genuine tracks are spatial-temporal fragments of one person."""
    da = _track_valid_dets(tr_a)
    db = _track_valid_dets(tr_b)
    if not da or not db:
        return False
    w = max(_mean_face_width(da), _mean_face_width(db))
    if w <= 1.0:
        return False
    prox = _SAME_PERSON_PROXIMITY_RATIO * w
    ta = [(float(d[0]), float(d[1]["center"][0]), float(d[1]["center"][1])) for d in da]
    tb = [(float(d[0]), float(d[1]["center"][0]), float(d[1]["center"][1])) for d in db]

    # (a) Simultaneous detections at the same head region: one person twice.
    ia = ib = 0
    while ia < len(ta) and ib < len(tb):
        dt = ta[ia][0] - tb[ib][0]
        if abs(dt) <= _SAME_PERSON_TIME_TOL:
            if math.hypot(ta[ia][1] - tb[ib][1], ta[ia][2] - tb[ib][2]) <= prox:
                return True
            if dt <= 0:
                ia += 1
            else:
                ib += 1
        elif dt < 0:
            ia += 1
        else:
            ib += 1

    # (b) Continuation hand-off: one track's detections stop and the other's
    # resume shortly after, at the same head region (ID-switch fragment).
    a_last = max(da, key=lambda d: float(d[0]))
    b_first = min(db, key=lambda d: float(d[0]))
    if 0.0 <= float(b_first[0]) - float(a_last[0]) <= _SAME_PERSON_MAX_GAP:
        if math.hypot(b_first[1]["center"][0] - a_last[1]["center"][0],
                      b_first[1]["center"][1] - a_last[1]["center"][1]) <= prox:
            return True
    b_last = max(db, key=lambda d: float(d[0]))
    a_first = min(da, key=lambda d: float(d[0]))
    if 0.0 <= float(a_first[0]) - float(b_last[0]) <= _SAME_PERSON_MAX_GAP:
        if math.hypot(a_first[1]["center"][0] - b_last[1]["center"][0],
                      a_first[1]["center"][1] - b_last[1]["center"][1]) <= prox:
            return True
    return False


def merge_same_person_tracks(tracks):
    """Union-find merge of genuine tracks that are fragments of one person.

    The surviving track keeps the richer detection set and absorbs the
    fragment's detections at timestamps it does not already cover, so mean
    geometry, prominence and Re-ID markers describe the full person. Merging
    is what keeps ONE-person shots one-person: without it a duplicate track
    pair passes the pair-composition test and the crop is centered between a
    subject and its own fragment. No-ops for genuinely distinct people.
    """
    n = len(tracks)
    if n < 2:
        return tracks
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            ri, rj = find(i), find(j)
            if ri != rj and _tracks_same_person(tracks[i], tracks[j]):
                parent[rj] = ri

    groups = {}
    for i, tr in enumerate(tracks):
        groups.setdefault(find(i), []).append(tr)
    if len(groups) == n:
        return tracks

    merged = []
    for members in groups.values():
        if len(members) == 1:
            merged.append(members[0])
            continue
        base = max(members, key=lambda t: len(_track_valid_dets(t)))
        seen_t = {round(float(d[0]), 3) for d in base.get("detections", [])}
        for other in members:
            if other is base:
                continue
            for d in other.get("detections", []):
                try:
                    key = round(float(d[0]), 3)
                except (TypeError, ValueError):
                    continue
                if key not in seen_t:
                    base["detections"].append(d)
                    seen_t.add(key)
            for k in ("persistent_identity_id", "persistent_identity_sim"):
                if k in other and k not in base:
                    base[k] = other[k]
            if other.get("reid_embedding") is not None and base.get("reid_embedding") is None:
                base["reid_embedding"] = other["reid_embedding"]
        vd = _track_valid_dets(base)
        if vd:
            base["cx"] = float(np.mean([d[1]["center"][0] for d in vd]))
            base["cy"] = float(np.mean([d[1]["center"][1] for d in vd]))
        base["detections"].sort(key=lambda d: d[0])
        merged.append(base)
    return merged


def resolve_visual_subject(
    genuine_tracks: list,
    ots_listener_tracks: list,
    pers_tracks: list,
    spk: str,
    rel_ts: float,
    rel_te: float,
    is_shot_cut: bool,
    last_confirmed_cx_by_spk: dict,
    default_cx: float,
    current_crop_x: float,
    crop_w_baseline: float,
    source_w: float = 1920.0,
    source_h: float = 1080.0,
    prominence_weights: dict = None,
    identity=None,
    config: FramingConfig = None,
    adaptive_framing: bool = False,
    fusion_by_track: dict = None,
):
    """
    Visual Subject Resolver (VSR) v12.1.
    Speech is evidence of visual importance, NOT the definition of visual importance.
    Evaluates arbitrary N >= 1 tracks dynamically without binary top_two truncation.
    Derives crop dynamically from tracked subject geometry.

    Phase 2: `fusion_by_track` (track_id -> fused ActiveSpeakerState) refines
    WHICH track the active diarized speaker maps to, using diarization + mouth
    motion + temporal hysteresis. It is consumed ONLY in active-speaker track
    selection below; two-shot compositions (adaptive shared framing, DualFrame)
    never consult it, so no active-speaker camera follow is introduced. When it
    is empty or uncertain the deterministic mouth-motion heuristic decides.
    """
    cfg = config or DEFAULT_FRAMING_CONFIG
    max_area_in_shot = max(
        (float(np.mean([d[1]["area"] for d in tr["detections"]])) for tr in genuine_tracks),
        default=0.0
    )

    ctx_w = 1.25
    for tr in genuine_tracks:
        tr["visual_prominence"] = compute_visual_prominence(
            tr, source_w, source_h, max_area_in_shot, prominence_weights, config=cfg
        )
        ctx_dets = [d for d in tr["detections"] if rel_ts - ctx_w <= d[0] <= rel_te + ctx_w]
        ctx_mot = float(np.mean([d[2] for d in ctx_dets])) if ctx_dets else 0.0
        sub_dets = [d for d in tr["detections"] if rel_ts <= d[0] < rel_te]
        sub_mot = float(np.mean([d[2] for d in sub_dets])) if sub_dets else 0.0
        tr["mot"] = 0.70 * ctx_mot + 0.30 * sub_mot if sub_dets else ctx_mot

    target_track = None
    target_face = None
    target_cx = None
    shot_type = "unknown"
    subject_role = "speaker"
    track_positions_override = None

    if len(genuine_tracks) >= 3:
        genuine_tracks_sorted = sorted(genuine_tracks, key=lambda tr: tr["cx"])
        l_tr, r_tr = genuine_tracks_sorted[0], genuine_tracks_sorted[-1]
        if adaptive_framing:
            # Adaptive Framing (user-directed, 10.0 behavior): when 2+ persons
            # are simultaneously in frame the composition is ALWAYS the shared
            # centered outer-pair two-shot — no fit gate, no active-speaker
            # follow, no panel mode. DualFrame split-screens/panels remain
            # strictly OFF in Adaptive Framing.
            shot_type = "two_shot_both_fit"
            subject_role = "two_shot"
            target_cx = (l_tr["cx"] + r_tr["cx"]) / 2.0
            target_track = l_tr
            dets = [d[1] for d in l_tr["detections"]]
            if dets:
                target_face = max(dets, key=lambda f: f["conf"])
            mn = min(len(l_tr["detections"]), len(r_tr["detections"]))
            track_positions_override = [
                (l_tr["detections"][i][1]["center"][0] + r_tr["detections"][i][1]["center"][0]) / 2.0
                for i in range(mn)
            ]
        else:
            keep_outer_pair, _ = _pair_composition_fits(l_tr, r_tr, crop_w_baseline, source_w, source_h, adaptive_framing=adaptive_framing)
            if keep_outer_pair:
                # Original 9:16: compose the visible outer pair as two-shot
                # when it fits the renderable window (unchanged behavior).
                shot_type = "two_shot_both_fit"
                subject_role = "two_shot"
                target_cx = (l_tr["cx"] + r_tr["cx"]) / 2.0
                target_track = l_tr
                dets = [d[1] for d in l_tr["detections"]]
                if dets:
                    target_face = max(dets, key=lambda f: f["conf"])
                mn = min(len(l_tr["detections"]), len(r_tr["detections"]))
                track_positions_override = [
                    (l_tr["detections"][i][1]["center"][0] + r_tr["detections"][i][1]["center"][0]) / 2.0
                    for i in range(mn)
                ]
            else:
                # Multi-person panel / roundtable mode: NO top_two truncation
                shot_type = "multi_person_panel"
            live_tracks = [tr for tr in genuine_tracks if _has_recent_detection(tr, rel_ts, rel_te, cfg.track_max_lost_seconds)]
            tracks_by_mot = sorted(live_tracks, key=lambda tr: tr.get("mot", 0.0), reverse=True)
            best_mot_tr = tracks_by_mot[0] if tracks_by_mot else None
            second_mot = tracks_by_mot[1].get("mot", 0.0) if len(tracks_by_mot) > 1 else 0.0

            # Phase 2 fusion-first selection: when exactly one live track is
            # confidently fused-speaking (diarization + mouth motion + temporal
            # hysteresis), it is the active speaker. With several fused tracks
            # (overlapping speech / diarization bleed) require the same
            # dominance ratio the heuristic uses before overriding it.
            fusion_pick = None
            if fusion_by_track:
                fused = [
                    (fusion_by_track[tr.get("track_id")], tr)
                    for tr in live_tracks
                    if tr.get("track_id") is not None and fusion_by_track.get(tr.get("track_id"))
                ]
                if len(fused) == 1:
                    fusion_pick = fused[0][1]
                elif len(fused) > 1:
                    best_state, best_f = max(
                        fused, key=lambda p: (p[0].speaking_probability, p[0].confidence)
                    )
                    others_p = [p[0].speaking_probability for p in fused if p[1] is not best_f]
                    if best_state.speaking_probability >= max(others_p) * cfg.mouth_motion_dominance_ratio:
                        fusion_pick = best_f
            if fusion_pick is not None:
                target_track = fusion_pick
                subject_role = "speaker"
            elif live_tracks and best_mot_tr.get("mot", 0.0) >= cfg.mouth_motion_active_thresh and best_mot_tr.get("mot", 0.0) >= second_mot * cfg.mouth_motion_dominance_ratio:
                target_track = best_mot_tr
                subject_role = "speaker"
            elif is_shot_cut:
                tracks_by_prom = sorted(live_tracks or genuine_tracks, key=lambda tr: tr.get("visual_prominence", 0.0), reverse=True)
                best_prom_tr = tracks_by_prom[0]
                second_prom = tracks_by_prom[1].get("visual_prominence", 0.0) if len(tracks_by_prom) > 1 else 0.0
                if (best_prom_tr.get("visual_prominence", 0.0) - second_prom >= cfg.prominence_dominance_diff or
                    best_prom_tr.get("visual_prominence", 0.0) / max(0.01, second_prom) >= cfg.prominence_dominance_ratio):
                    target_track = best_prom_tr
                    subject_role = "listener_reaction"
                else:
                    target_track = best_prom_tr
                    subject_role = "visual_subject"
            elif live_tracks and spk and spk in last_confirmed_cx_by_spk:
                last_spk_cx = last_confirmed_cx_by_spk[spk]
                target_track = min(live_tracks, key=lambda tr: abs(tr["cx"] - last_spk_cx))
                subject_role = "speaker"
            elif live_tracks and current_crop_x is not None:
                curr_frame_cx = current_crop_x + crop_w_baseline / 2.0
                target_track = min(live_tracks, key=lambda tr: abs(tr["cx"] - curr_frame_cx))
                subject_role = "continuity_hold"
            elif live_tracks:
                target_track = max(live_tracks, key=lambda tr: tr.get("visual_prominence", 0.0))
                subject_role = "visual_subject"

            if target_track is not None:
                ld = [d[1] for d in target_track["detections"] if abs(d[0] - rel_ts) <= 0.35]
                rd = [d[1] for d in target_track["detections"]]
                src = ld or rd
                if src:
                    target_face = max(src, key=lambda f: f["conf"])
                    target_cx = target_face["center"][0]
                else:
                    target_cx = target_track["cx"]
            else:
                # All panel tracks are stale: hold the current crop rather than reframing
                # toward a subject whose last detection no longer describes the frame.
                target_cx = (current_crop_x + crop_w_baseline / 2.0) if current_crop_x is not None else default_cx
                subject_role = "continuity_hold"

    elif len(genuine_tracks) == 2:
        genuine_tracks_sorted = sorted(genuine_tracks, key=lambda tr: tr["cx"])
        l_tr, r_tr = genuine_tracks_sorted[0], genuine_tracks_sorted[1]
        keep_pair, _ = _pair_composition_fits(l_tr, r_tr, crop_w_baseline, source_w, source_h, adaptive_framing=adaptive_framing)

        if adaptive_framing:
            # Adaptive Framing (10.0 behavior, user-directed): when TWO persons
            # are simultaneously in frame, the composition is ALWAYS the one
            # shared centered two-shot — both people retained, NO active-speaker
            # follow. The pair-fit test no longer gates the centering: when the
            # pair span exceeds the square, centering still places both heads as
            # far inside the canvas as a legal x allows, and the downstream
            # containment validator remains the hard guarantee. Non-adaptive
            # (Original 9:16) keeps the fit-gated behavior below, unchanged.
            shot_type = "two_shot_both_fit"
            subject_role = "two_shot"
            target_cx = (l_tr["cx"] + r_tr["cx"]) / 2.0
            target_track = l_tr
            dets = [d[1] for d in l_tr["detections"]]
            if dets:
                target_face = max(dets, key=lambda f: f["conf"])
            mn = min(len(l_tr["detections"]), len(r_tr["detections"]))
            track_positions_override = [
                (l_tr["detections"][i][1]["center"][0] + r_tr["detections"][i][1]["center"][0]) / 2.0
                for i in range(mn)
            ]
        else:
            face_span = (r_tr["cx"] + r_tr["detections"][-1][1]["bbox"][2] / 2.0) - (l_tr["cx"] - l_tr["detections"][-1][1]["bbox"][2] / 2.0)
            center_dist = r_tr["cx"] - l_tr["cx"]
            l_area = float(np.mean([d[1]["area"] for d in l_tr["detections"]]))
            r_area = float(np.mean([d[1]["area"] for d in r_tr["detections"]]))
            comparable_area = (min(l_area, r_area) / max(l_area, r_area, 1.0)) >= 0.15

            l_mot = l_tr.get("mot", 0.0)
            r_mot = r_tr.get("mot", 0.0)
            l_prom = l_tr.get("visual_prominence", 0.5)
            r_prom = r_tr.get("visual_prominence", 0.5)
            l_live = _has_recent_detection(l_tr, rel_ts, rel_te, cfg.track_max_lost_seconds)
            r_live = _has_recent_detection(r_tr, rel_ts, rel_te, cfg.track_max_lost_seconds)

            def _nearest_live_two(last_cx: float):
                if l_live != r_live:
                    return l_tr if l_live else r_tr
                return l_tr if abs(l_tr["cx"] - last_cx) < abs(r_tr["cx"] - last_cx) else r_tr

            active_speaker_found = False
            # Phase 2 fusion-first selection (non-adaptive two-person): one
            # confidently fused track is the active speaker; two fused tracks
            # need the dominance ratio to pick between them. The adaptive
            # shared two-shot above never reaches this branch.
            if fusion_by_track:
                fused = [
                    (fusion_by_track[tr.get("track_id")], tr)
                    for tr in (l_tr, r_tr)
                    if tr.get("track_id") is not None and fusion_by_track.get(tr.get("track_id"))
                ]
                if len(fused) == 1:
                    target_track = fused[0][1]
                    active_speaker_found = True
                    subject_role = "speaker"
                elif len(fused) == 2:
                    (st_a, tr_a), (st_b, tr_b) = fused
                    if st_a.speaking_probability >= st_b.speaking_probability * cfg.mouth_motion_dominance_ratio:
                        target_track = tr_a
                        active_speaker_found = True
                        subject_role = "speaker"
                    elif st_b.speaking_probability >= st_a.speaking_probability * cfg.mouth_motion_dominance_ratio:
                        target_track = tr_b
                        active_speaker_found = True
                        subject_role = "speaker"
            if not active_speaker_found and r_live and r_mot > l_mot * cfg.mouth_motion_dominance_ratio and r_mot >= cfg.mouth_motion_active_thresh:
                target_track = r_tr
                active_speaker_found = True
                subject_role = "speaker"
            elif not active_speaker_found and l_live and l_mot > r_mot * cfg.mouth_motion_dominance_ratio and l_mot >= cfg.mouth_motion_active_thresh:
                target_track = l_tr
                active_speaker_found = True
                subject_role = "speaker"
            elif not active_speaker_found and is_shot_cut:
                if l_live != r_live:
                    target_track = l_tr if l_live else r_tr
                    active_speaker_found = True
                    subject_role = "listener_reaction"
                else:
                    prom_diff = abs(r_prom - l_prom)
                    prom_ratio = max(r_prom, l_prom) / max(0.01, min(r_prom, l_prom))
                    if prom_diff >= cfg.prominence_dominance_diff or prom_ratio >= cfg.prominence_dominance_ratio:
                        target_track = r_tr if r_prom > l_prom else l_tr
                        active_speaker_found = True
                        subject_role = "listener_reaction"
            elif not active_speaker_found and spk and spk in last_confirmed_cx_by_spk:
                last_spk_cx = last_confirmed_cx_by_spk[spk]
                target_track = _nearest_live_two(last_spk_cx)
                active_speaker_found = True
                subject_role = "speaker"
            elif not active_speaker_found and current_crop_x is not None:
                curr_frame_cx = current_crop_x + crop_w_baseline / 2.0
                target_track = _nearest_live_two(curr_frame_cx)
                active_speaker_found = True
                subject_role = "continuity_hold"

            keep_pair = (
                (not active_speaker_found or subject_role not in ("listener_reaction", "speaker"))
                and face_span <= crop_w_baseline * TWO_SHOT_MAX_SPAN
            )

            if not adaptive_framing and keep_pair and center_dist >= source_w * 0.05 and comparable_area:
                shot_type = "two_shot_both_fit"
                subject_role = "two_shot"
                target_cx = (l_tr["cx"] + r_tr["cx"]) / 2.0
                target_track = l_tr
                dets = [d[1] for d in l_tr["detections"]]
                if dets:
                    target_face = max(dets, key=lambda f: f["conf"])
                mn = min(len(l_tr["detections"]), len(r_tr["detections"]))
                track_positions_override = [
                    (l_tr["detections"][i][1]["center"][0] + r_tr["detections"][i][1]["center"][0]) / 2.0
                    for i in range(mn)
                ]
            else:
                if shot_type == "unknown":
                    shot_type = "wide_two_shot"
                if target_track is None:
                    if identity and spk:
                        last_spk_cx = identity.get_last_position(default_cx)
                        target_track = l_tr if abs(l_tr["cx"] - last_spk_cx) < abs(r_tr["cx"] - last_spk_cx) else r_tr
                    else:
                        target_track = r_tr if r_prom > l_prom else l_tr
                    subject_role = "visual_subject"
                ld = [d[1] for d in target_track["detections"] if abs(d[0] - rel_ts) <= 0.35]
                rd = [d[1] for d in target_track["detections"]]
                src = ld or rd
                if src:
                    target_face = max(src, key=lambda f: f["conf"])
                    target_cx = target_face["center"][0]
                else:
                    target_cx = target_track["cx"]

    elif len(genuine_tracks) == 1:
        target_track = genuine_tracks[0]
        shot_type = "ots_over_the_shoulder" if ots_listener_tracks else "solo_close_up"
        ld = [d[1] for d in target_track["detections"] if abs(d[0] - rel_ts) <= 0.35]
        rd = [d[1] for d in target_track["detections"]]
        src = ld or rd
        if src:
            target_face = max(src, key=lambda f: f["conf"])
            target_cx = target_face["center"][0]
        else:
            target_cx = target_track["cx"]

        m_mot = target_track.get("mot", 0.0)
        if m_mot < cfg.mouth_motion_active_thresh and (is_shot_cut or (spk and spk in last_confirmed_cx_by_spk and abs(target_cx - last_confirmed_cx_by_spk[spk]) > 250.0)):
            subject_role = "listener_reaction"
        else:
            subject_role = "speaker"

    elif pers_tracks:
        shot_type = "fallback_persistent_track"
        target_track = max(pers_tracks, key=lambda tr: np.mean([d[1]["conf"] * d[1]["frontality"] for d in tr["detections"]]))
        rd = [d[1] for d in target_track["detections"]]
        if rd:
            target_face = max(rd, key=lambda f: f["conf"])
            target_cx = target_face["center"][0]
        else:
            target_cx = target_track["cx"]
        subject_role = "fallback"
    else:
        shot_type = "fallback_recovery_hierarchy"
        target_cx = last_confirmed_cx_by_spk.get(spk, identity.predict_position(rel_ts, default_cx) if identity else default_cx)
        subject_role = "fallback"

    return target_track, target_face, target_cx, shot_type, subject_role, track_positions_override


# ─── 8B. DualFrame Visual-Layout Configuration & Resolver ─────────────────────

def _is_dualframe_env_enabled() -> bool:
    val = os.environ.get("AUTOSHORTS_DUALFRAME_ENABLED", "true").strip().lower()
    return val not in ("false", "0", "no", "off")


@dataclass(frozen=True)
class DualFrameConfig:
    """
    Configuration parameters for DualFrame vertical stack layout resolution.
    """
    enabled: bool = field(default_factory=_is_dualframe_env_enabled)
    dual_entry_persistence_sec: float = 0.60
    dual_exit_persistence_sec: float = 0.75
    min_segment_dur_sec: float = 1.00
    min_track_age_sec: float = 0.50
    max_lost_sec: float = 0.60
    panel_aspect_ratio: float = 9.0 / 8.0  # 1.125


@dataclass
class DualFrameDecision:
    """
    Output resolution from DualFrameLayoutResolver for a single analysis step.
    Supports attribute access as well as tuple unpacking: mode, top_id, bottom_id = decision.
    """
    mode: str = "single"
    top_track_id: Optional[Any] = None
    bottom_track_id: Optional[Any] = None
    is_dual: bool = False
    top_track: Optional[dict] = None
    bottom_track: Optional[dict] = None

    def __iter__(self):
        return iter((self.mode, self.top_track_id, self.bottom_track_id))


class DualFrameLayoutResolver:
    """
    DualFrameLayoutResolver (AutoShorts 6.0).
    Evaluates visual subject tracks post-VSR with anti-flicker hysteresis,
    left-to-right entry slot assignment, and slot identity locking.
    """
    def __init__(self, config: Optional[DualFrameConfig] = None):
        self.config = config if config is not None else DualFrameConfig()
        self.mode: str = "single"
        self.locked_top_track_id: Optional[Any] = None
        self.locked_bottom_track_id: Optional[Any] = None
        self.candidate_dual_start: Optional[float] = None
        self.candidate_tracks: Optional[set] = None
        self.last_dual_seen: Optional[float] = None
        self.segment_start_t: float = 0.0

    @property
    def top_track_id(self) -> Optional[Any]:
        return self.locked_top_track_id

    @property
    def bottom_track_id(self) -> Optional[Any]:
        return self.locked_bottom_track_id

    @property
    def is_dual(self) -> bool:
        return self.mode == "dual_stack"

    def reset(self, rel_t: Optional[float] = None):
        """
        Complete state reset on camera cuts.
        """
        self.mode = "single"
        self.locked_top_track_id = None
        self.locked_bottom_track_id = None
        self.candidate_dual_start = None
        self.candidate_tracks = None
        self.last_dual_seen = None
        self.segment_start_t = rel_t if rel_t is not None else 0.0
        t_str = f"{rel_t:.2f}" if rel_t is not None else "0.00"
        sys.stderr.write(f"[DualFrame] shot_reset t={t_str}\n")
        sys.stderr.flush()

    def is_subject_eligible(
        self,
        track: dict,
        source_w: float = 1920.0,
        source_h: float = 1080.0,
    ) -> bool:
        """
        DualFrame Subject Eligibility Criteria:
        1. Track Provenance & Physical Human Body: Genuine YOLO person body or verified prominent face-only shot.
        2. Track Persistence & Stability: >= min_track_age_sec, low jitter.
        3. VSR Relevance & Prominence: visual_prominence >= threshold, not micro background detail.
        4. Visibility: not occluded, not back of head, valid geometry.
        5. Crop Feasibility: within source dimensions for 9:8 panel crop.
        """
        if not isinstance(track, dict):
            return False

        # Explicit eligibility override (for mocking or fast paths)
        if track.get("is_eligible") is not None:
            return bool(track["is_eligible"])

        # 1. Track Provenance Hierarchy:
        # A YuNet-only synthetic fallback track that has no associated YOLO person body
        # must NOT automatically qualify as a DualFrame subject.
        # - Genuine YOLO-associated person track = valid candidate.
        # - YuNet-only fallback track = reject for DualFrame by default.
        # - Allow a YuNet-only exception ONLY when existing evidence strongly indicates
        #   a genuine prominent human face in a legitimate face-only shot.
        provenance = track.get("provenance")
        has_person_body = track.get("has_person_body")

        if provenance == "yunet_face_only" or has_person_body is False:
            is_valid_face_only_exception = False
            detections = track.get("detections", [])
            if detections:
                latest_det = detections[-1]
                face = latest_det[1] if len(latest_det) > 1 and isinstance(latest_det[1], dict) else {}
                face_conf = float(face.get("conf", 0.0))
                face_bbox = face.get("bbox", (0, 0, 0, 0))
                face_w = float(face_bbox[2]) if len(face_bbox) > 2 else 0.0
                face_h = float(face_bbox[3]) if len(face_bbox) > 3 else 0.0
                face_area = float(face.get("area", face_w * face_h))
                frontality = float(face.get("frontality", 0.0))

                # Legitimate close-up face-only shot criteria (e.g. ECU shot where body is out of frame)
                min_ecu_w = source_w * 0.065   # ~125px in 1080p
                min_ecu_h = source_h * 0.10    # ~108px in 1080p
                min_ecu_area = min_ecu_w * min_ecu_h

                if (face_conf >= 0.70 and
                    face_w >= min_ecu_w and
                    face_h >= min_ecu_h and
                    face_area >= min_ecu_area and
                    frontality >= 0.40 and
                    not face.get("is_back", False)):
                    is_valid_face_only_exception = True

            if not is_valid_face_only_exception:
                return False

        # 2. Occlusion & back-of-head check
        if track.get("is_occluded", False):
            return False
        if track.get("is_back", False):
            return False

        # 3. Geometry check & bounds
        cx = track.get("cx")
        cy = track.get("cy")
        detections = track.get("detections", [])

        if cx is None:
            if detections and len(detections) > 0:
                cx = detections[-1][1].get("center", (None, None))[0]
            if cx is None:
                return False
        track["cx"] = cx

        if cy is None:
            if detections and len(detections) > 0:
                cy = detections[-1][1].get("center", (None, None))[1]
            if cy is None:
                cy = source_h / 2.0
        track["cy"] = cy

        if cx < 0.0 or cx > source_w or cy < 0.0 or cy > source_h:
            return False

        # 4. Detection confidence, face characteristics, and micro-artifact filter
        if detections:
            latest_det = detections[-1]
            face = latest_det[1] if len(latest_det) > 1 and isinstance(latest_det[1], dict) else {}
            if face.get("is_back", False):
                return False
            if face.get("conf", 1.0) < 0.35:
                return False

            face_bbox = face.get("bbox")
            if face_bbox and len(face_bbox) >= 4:
                fw = float(face_bbox[2])
                fh = float(face_bbox[3])
                fa = float(face.get("area", fw * fh))
                # Reject micro-artifacts and distant background specs (< ~38px width/height or < 1800px area in 1080p)
                min_w = source_w * 0.02   # ~38px
                min_h = source_h * 0.035  # ~38px
                if fw < min_w or fh < min_h or fa < 1800.0:
                    return False

        # 5. Prominence check (not distant background passerby)
        prom = track.get("visual_prominence")
        if prom is not None and prom < 0.15:
            return False

        # 6. Track age / duration check
        age_sec = track.get("age_sec")
        if age_sec is None and detections:
            age_sec = max(0.0, float(detections[-1][0] - detections[0][0]))
        if age_sec is not None and age_sec < self.config.min_track_age_sec:
            return False

        return True

    def update(
        self,
        rel_t: float,
        active_tracks,
        is_shot_cut: bool = False,
        active_spk: Optional[str] = None,
        source_w: float = 1920.0,
        source_h: float = 1080.0,
    ) -> DualFrameDecision:
        """
        Updates the DualFrame state machine with anti-flicker hysteresis,
        left-to-right entry slot assignment, and slot identity locking.
        """
        # 1. Check feature toggle
        if not self.config.enabled:
            if self.mode != "single":
                self.reset(rel_t)
            return DualFrameDecision(mode="single", top_track_id=None, bottom_track_id=None, is_dual=False)

        # 2. Complete state reset on camera cut
        if is_shot_cut:
            self.reset(rel_t)

        # Normalize active_tracks into {tid: tr}
        tracks_dict = {}
        if isinstance(active_tracks, dict):
            for tid, tr in active_tracks.items():
                tr_copy = dict(tr)
                if "track_id" not in tr_copy:
                    tr_copy["track_id"] = tid
                tracks_dict[tid] = tr_copy
        elif isinstance(active_tracks, (list, tuple)):
            for idx, tr in enumerate(active_tracks):
                tr_copy = dict(tr)
                tid = tr_copy.get("track_id", tr_copy.get("id", idx))
                tr_copy["track_id"] = tid
                tracks_dict[tid] = tr_copy

        # Filter eligible tracks
        eligible_tracks = {
            tid: tr for tid, tr in tracks_dict.items()
            if self.is_subject_eligible(tr, source_w=source_w, source_h=source_h)
        }
        n_eligible = len(eligible_tracks)

        # 3. State machine transitions
        if self.mode == "single":
            if n_eligible == 2:
                # Pair validation invariants:
                # 1. Distinct track identities: left and right cannot be the same track ID
                sorted_subjects = sorted(eligible_tracks.values(), key=lambda tr: tr.get("cx", 0.0))
                left_subj = sorted_subjects[0]
                right_subj = sorted_subjects[1]
                left_id = left_subj.get("track_id")
                right_id = right_subj.get("track_id")

                is_valid_pair = (left_id is not None and right_id is not None and left_id != right_id)

                # 2. Scale / Area ratio: neither subject can be vastly disproportionate (< 15% area ratio)
                if is_valid_pair:
                    def _get_subj_area(tr):
                        dets = tr.get("detections", [])
                        if dets and isinstance(dets[-1][1], dict):
                            return float(dets[-1][1].get("area", 0.0))
                        return float(tr.get("area", 0.0))

                    area_l = _get_subj_area(left_subj)
                    area_r = _get_subj_area(right_subj)
                    if area_l > 0.0 and area_r > 0.0:
                        min_a, max_a = min(area_l, area_r), max(area_l, area_r)
                        if min_a < 0.15 * max_a:
                            is_valid_pair = False

                # 3. Spatial clearance: two distinct subjects cannot sit at the exact same coordinate
                if is_valid_pair:
                    cx_l = float(left_subj.get("cx", 0.0))
                    cx_r = float(right_subj.get("cx", 0.0))
                    if abs(cx_r - cx_l) < source_w * 0.12:
                        is_valid_pair = False

                if is_valid_pair:
                    current_candidates = set(eligible_tracks.keys())
                    if self.candidate_dual_start is None or self.candidate_tracks != current_candidates:
                        self.candidate_dual_start = rel_t
                        self.candidate_tracks = current_candidates

                    elapsed = rel_t - self.candidate_dual_start
                    # Entry persistence requirement: >= dual_entry_persistence_sec
                    if elapsed >= self.config.dual_entry_persistence_sec:
                        self.locked_top_track_id = left_subj["track_id"]
                        self.locked_bottom_track_id = right_subj["track_id"]
                        self.mode = "dual_stack"
                        self.segment_start_t = rel_t
                        self.last_dual_seen = rel_t
                        self.candidate_dual_start = None
                        self.candidate_tracks = None

                        sys.stderr.write("[DualFrame] mode=dual_stack\n")
                        sys.stderr.write(
                            f"[DualFrame] enter t={rel_t:.2f} top_track={self.locked_top_track_id} bottom_track={self.locked_bottom_track_id}\n"
                        )
                        sys.stderr.flush()
                else:
                    self.candidate_dual_start = None
                    self.candidate_tracks = None
            else:
                # Not exactly 2 eligible subjects -> reset candidate timer
                self.candidate_dual_start = None
                self.candidate_tracks = None

        elif self.mode == "dual_stack":
            # If 3 or more subjects appear, or duplicate IDs, immediately exit to single
            if n_eligible >= 3 or (self.locked_top_track_id is not None and self.locked_top_track_id == self.locked_bottom_track_id):
                self.mode = "single"
                self.locked_top_track_id = None
                self.locked_bottom_track_id = None
                self.last_dual_seen = None
                self.candidate_dual_start = None
                self.candidate_tracks = None
                self.segment_start_t = rel_t
                sys.stderr.write("[DualFrame] mode=single\n")
                sys.stderr.write(f"[DualFrame] exit t={rel_t:.2f} reason=too_many_subjects\n")
                sys.stderr.flush()
            else:
                top_present = self.locked_top_track_id in eligible_tracks
                bot_present = self.locked_bottom_track_id in eligible_tracks

                if top_present and bot_present:
                    self.last_dual_seen = rel_t
                else:
                    # One or both subjects missing/occluded -> test exit grace buffer
                    absence_dur = rel_t - (self.last_dual_seen if self.last_dual_seen is not None else rel_t)
                    if absence_dur > self.config.dual_exit_persistence_sec:
                        self.mode = "single"
                        self.locked_top_track_id = None
                        self.locked_bottom_track_id = None
                        self.last_dual_seen = None
                        self.candidate_dual_start = None
                        self.candidate_tracks = None
                        self.segment_start_t = rel_t
                        sys.stderr.write("[DualFrame] mode=single\n")
                        sys.stderr.write(f"[DualFrame] exit t={rel_t:.2f} reason=subject_lost\n")
                        sys.stderr.flush()

        # Build decision
        top_tr = tracks_dict.get(self.locked_top_track_id) if self.locked_top_track_id is not None else None
        bot_tr = tracks_dict.get(self.locked_bottom_track_id) if self.locked_bottom_track_id is not None else None

        return DualFrameDecision(
            mode=self.mode,
            top_track_id=self.locked_top_track_id,
            bottom_track_id=self.locked_bottom_track_id,
            is_dual=(self.mode == "dual_stack"),
            top_track=top_tr,
            bottom_track=bot_tr,
        )

def classify_and_frame_subject(
    shot_tracks: dict,
    shot_samples: list,
    spk: str,
    rel_ts: float,
    rel_te: float,
    last_confirmed_cx_by_spk: dict,
    default_cx: float,
    crop_w_baseline: float,
    max_x_baseline: float,
    source_h: float = 1080.0,
    source_w: float = 1920.0,
    current_crop_x=None,
    current_crop_y=None,
    current_scale: float = 1.0,
    shot_scale_state=None,
    shot_id: int = 0,
    is_shot_cut: bool = False,
    prominence_weights: dict = None,
    config: FramingConfig = None,
    adaptive_framing: bool = False,
    fusion_by_track: dict = None,
):
    """
    Joint (Position + Scale) Framing Solver v12.1 with Universal Visual Subject Tracking.
    Returns (clamped_x, crop_y, crop_w, crop_h, target_cx, shot_type, camera_mode, diag, shot_scale_state).
    crop_w and crop_h are REAL dynamic pixel dimensions.
    """
    cfg = config or DEFAULT_FRAMING_CONFIG
    identity = get_or_create_identity(spk) if spk else None
    local_samples, all_local_faces = [], []
    if shot_samples and isinstance(shot_samples[0], (list, tuple)) and len(shot_samples[0]) >= 3:
        local_samples = [s for s in shot_samples if rel_ts - 0.25 <= s[0] <= rel_te + 0.25] or shot_samples
        for s in local_samples:
            all_local_faces.extend(s[2])

    min_dets = max(1, int(0.15 * len(shot_samples))) if shot_samples else 1
    pers_tracks = [tr for tr in shot_tracks.values() if len(tr["detections"]) >= min_dets]
    max_area_in_shot = max((float(np.mean([d[1]["area"] for d in tr["detections"]])) for tr in pers_tracks), default=0.0)
    max_conf_in_shot = max((float(np.mean([d[1]["conf"] for d in tr["detections"]])) for tr in pers_tracks), default=0.0)

    genuine_tracks, ots_listener_tracks = [], []
    # Per-track base quality (the original, non-adaptive genuine test),
    # recorded so the Adaptive two-person rescue below can still promote a
    # REAL person that only failed the adaptive decoration gate (wide shots
    # make the far face small/low-area, which used to leave a single genuine
    # track and shifted the crop onto the active speaker).
    base_quality = {}
    for tr in pers_tracks:
        avg_conf = float(np.mean([d[1]["conf"] for d in tr["detections"]]))
        back_ratio = float(np.mean([d[1]["is_back"] for d in tr["detections"]]))
        avg_front = float(np.mean([d[1]["frontality"] for d in tr["detections"]]))
        avg_w = float(np.mean([d[1]["bbox"][2] for d in tr["detections"]]))
        avg_h = float(np.mean([d[1]["bbox"][3] for d in tr["detections"]]))
        avg_a = float(np.mean([d[1]["area"] for d in tr["detections"]]))

        if avg_a < max_area_in_shot * 0.15 and (avg_w < 50 or avg_h < 55):
            continue

        is_credible_conf = avg_conf >= 0.50 and avg_conf >= max_conf_in_shot * 0.70
        is_strong_profile = avg_conf >= 0.70 and avg_a >= max_area_in_shot * 0.25
        is_base_genuine = bool(
            is_credible_conf and back_ratio < 0.40 and (avg_front >= 0.35 or is_strong_profile)
        )
        base_quality[id(tr)] = (is_base_genuine, float(avg_conf), float(avg_a))

        if adaptive_framing:
            # Filter low-confidence false-positive background decoration tracks (confidence >= 0.70, area >= 0.005)
            frame_area = float(source_w * source_h)
            if avg_conf < 0.70 or avg_a < (0.005 * frame_area):
                continue

        if is_credible_conf and back_ratio < 0.40 and (avg_front >= 0.35 or is_strong_profile):
            genuine_tracks.append(tr)
        elif back_ratio >= 0.40 or avg_front < 0.35:
            ots_listener_tracks.append(tr)

    if adaptive_framing and not genuine_tracks:
        # Fallback if strict decoration filter dropped all tracks
        for tr in pers_tracks:
            avg_conf = float(np.mean([d[1]["conf"] for d in tr["detections"]]))
            back_ratio = float(np.mean([d[1]["is_back"] for d in tr["detections"]]))
            avg_front = float(np.mean([d[1]["frontality"] for d in tr["detections"]]))
            if avg_conf >= 0.60 and back_ratio < 0.40 and avg_front >= 0.35:
                genuine_tracks.append(tr)

    if adaptive_framing and len(genuine_tracks) == 1:
        # Adaptive two-person rescue (user-directed): when TWO persons are in
        # frame but the decoration gate dropped the second one (wide shots
        # shrink the far face below the area/conf gate), promote the best
        # remaining REAL-person track — it passed the original genuine test
        # and only failed the decoration gate — so the composition centers on
        # the pair instead of following the active speaker. Exactly ONE track
        # is promoted (the strongest by conf×area). Promotion requires YOLO
        # person-body evidence (has_person_body): face-only tracks (the
        # YuNet safeguard range) include wall photos / posters and must never
        # complete a two-person composition.
        anchor = genuine_tracks[0]
        anchor_id = id(anchor)
        best_tr, best_score = None, -1.0
        for tr in pers_tracks:
            if id(tr) == anchor_id:
                continue
            if not tr.get("has_person_body"):
                continue
            q = base_quality.get(id(tr))
            if not q or not q[0]:
                continue
            score = q[1] * q[2]
            if score > best_score:
                best_score, best_tr = score, tr
        if best_tr is not None:
            genuine_tracks.append(best_tr)

    # Same-person fragment merge: BoT-SORT ID switches can split ONE person
    # into 2+ genuine tracks; without this merge a solo shot passes the pair
    # composition test and produces a false two_shot_both_fit composition
    # centered between a subject and its own duplicate. Distinct people are
    # never merged (they are separated by more than a face width). A rescued
    # same-person fragment merges back here, correctly restoring the solo read.
    genuine_tracks = merge_same_person_tracks(genuine_tracks)

    genuine_tracks.sort(key=lambda tr: tr["cx"])

    # Resolve visual subject using multi-factor prominence and speech evidence
    (target_track, target_face, target_cx, shot_type,
     subject_role, track_positions_override) = resolve_visual_subject(
        genuine_tracks, ots_listener_tracks, pers_tracks,
        spk, rel_ts, rel_te, is_shot_cut,
        last_confirmed_cx_by_spk, default_cx, current_crop_x,
        crop_w_baseline, source_w, source_h, prominence_weights, identity,
        config=cfg,
        adaptive_framing=adaptive_framing,
        fusion_by_track=fusion_by_track,
    )

    track_positions = track_positions_override or ([d[1]["center"][0] for d in target_track["detections"]] if target_track else [target_cx])
    camera_mode = evaluate_camera_mode(track_positions, source_w)

    if camera_mode == "STATIC" and len(track_positions) >= 3 and shot_type != "wide_two_shot":
        target_cx = float(np.median(track_positions))

    if target_cx is not None and target_face is not None and identity is not None and spk is not None:
        if shot_type in ("solo_close_up", "ots_over_the_shoulder"):
            if target_face.get("conf", 0.0) >= 0.50 and subject_role != "listener_reaction":
                identity.record_detection(target_cx, target_face["center"][1], rel_ts)
                last_confirmed_cx_by_spk[spk] = target_cx
        elif shot_type in ("wide_two_shot", "multi_person_panel"):
            if target_face.get("conf", 0.0) >= 0.50 and subject_role == "speaker":
                identity.record_detection(target_cx, target_face["center"][1], rel_ts)
                last_confirmed_cx_by_spk[spk] = target_cx

    # ── Scale Solver ──────────────────────────────────────────────────────────
    # Source-camera-zoom guard runs BEFORE the scale estimator: a detector box
    # wider than the crop can contain (merged subject cluster under a source
    # zoom) must not be read as a genuine extreme close-up, or the estimator
    # pins min_scale at 1.00 and compounds the source zoom (task: source zoom
    # is not AutoShorts framing zoom). See sanitize_face_bbox_for_containment.
    if target_face is not None:
        target_face = sanitize_face_bbox_for_containment(target_face, crop_w_baseline, source_w)
    pair_needed_scale = None
    if shot_type == "two_shot_both_fit" and adaptive_framing and len(genuine_tracks) >= 2:
        _, pair_needed_scale = _pair_composition_fits(
            genuine_tracks[0], genuine_tracks[-1], crop_w_baseline, source_w, source_h,
            adaptive_framing=adaptive_framing,
        )
    ideal_scale, min_s, max_s, shot_class = classify_shot_and_estimate_scale(
        target_face, shot_samples, crop_w_baseline, source_w, source_h, shot_type,
        adaptive_framing=adaptive_framing, pair_needed_scale=pair_needed_scale,
    )

    if shot_scale_state is None:
        shot_scale_state = ShotScaleState(shot_id, shot_class, ideal_scale, min_s, max_s)
        resolved_scale = ideal_scale
        scale_change_reason = "SHOT_RESET"
    else:
        resolved_scale, scale_change_reason = apply_scale_deadband(
            shot_scale_state.current_scale, ideal_scale, min_s, max_s, shot_scale_state.scale_change_reason,
            deadband=cfg.scale_deadband,
        )

    shot_scale_state.ideal_scale = ideal_scale
    shot_scale_state.min_valid_scale = min_s
    shot_scale_state.max_valid_scale = max_s
    shot_scale_state.record(rel_ts, resolved_scale, scale_change_reason)

    # Real crop dimensions derived from scale
    eff_crop_w, eff_crop_h = compute_effective_crop_dims(resolved_scale, source_h, source_w, adaptive_framing=adaptive_framing)
    eff_max_x = max(0.0, source_w - eff_crop_w)

    # ── Position Solver ───────────────────────────────────────────────────────
    subject_safe_box, is_contained, corridor_info = None, True, {}

    if target_face is not None:
        if shot_type == "two_shot_both_fit" and len(genuine_tracks) >= 2 and adaptive_framing:
            pair_cx = (genuine_tracks[0]["cx"] + genuine_tracks[-1]["cx"]) / 2.0
            raw_x = pair_cx - eff_crop_w / 2.0
            clamped_x = float(np.clip(raw_x, 0.0, eff_max_x))
            target_cx = pair_cx
            is_contained = True
            subject_safe_box = _pair_subject_safe_box(genuine_tracks[0], genuine_tracks[-1], source_w, source_h)
            if subject_safe_box is None:
                subject_safe_box = compute_subject_safe_box(target_face, source_w, source_h)
            corridor_info = {"min_valid_x": clamped_x, "max_valid_x": clamped_x, "corridor_width": 0.0}
        else:
            subject_safe_box = compute_subject_safe_box(target_face, source_w, source_h)
            clamped_x, is_contained, corridor_info = solve_containment_crop_x(
                subject_safe_box, eff_crop_w, eff_max_x, source_w, SAFETY_MARGIN_RATIO, current_crop_x
            )
    else:
        # Fallback hierarchy:
        # 1. If we have a trusted prior target_cx that differs from default_cx, move towards it.
        if abs(target_cx - default_cx) > 1.0:
            raw_x = target_cx - eff_crop_w / 2.0
            clamped_x = float(np.clip(raw_x, 0.0, eff_max_x))
        # 2. Hold current valid crop position
        elif current_crop_x is not None and 0.0 <= current_crop_x <= eff_max_x:
            clamped_x = float(current_crop_x)
        # 3. Safe center fallback
        else:
            raw_x = (source_w - eff_crop_w) / 2.0
            clamped_x = float(np.clip(raw_x, 0.0, eff_max_x))

    clamped_x = float(np.clip(clamped_x, 0.0, eff_max_x))
    crop_y = solve_crop_y(target_face, eff_crop_h, source_h, current_crop_y)

    # ── Phase 3: advisory crop ranking (containment-legal candidates only) ──
    # The corridor from solve_containment_crop_x is exactly the set of x
    # positions that keep the head contained; we only RANK a few corridor-legal
    # candidates with the existing deterministic composition score. Movement is
    # bounded per decision (current x ± half-deadband) to prevent jitter, and
    # the trajectory validator remains the hard containment guarantee.
    if (crop_scorer_enabled() and target_face is not None
            and float(corridor_info.get("corridor_width", 0.0)) > 8.0):
        try:
            _lo = float(corridor_info.get("min_valid_x", clamped_x))
            _hi = float(corridor_info.get("max_valid_x", clamped_x))
            _deadband = max(17.5, eff_crop_w * 0.045)
            _cands = sorted({
                float(np.clip(clamped_x, _lo, _hi)),
                float(np.clip(clamped_x - _deadband, _lo, _hi)),
                float(np.clip(clamped_x + _deadband, _lo, _hi)),
            })
            _scored_x, _score = rank_crop_candidates(
                _cands, crop_y, eff_crop_w, eff_crop_h, resolved_scale,
                target_face, all_local_faces, source_w, source_h, current_crop_x,
            )
            _solved_score = evaluate_crop_composition_score(
                clamped_x, crop_y, eff_crop_w, eff_crop_h, resolved_scale,
                target_face, all_local_faces, source_w, source_h,
                camera_motion_px=abs(clamped_x - (current_crop_x or clamped_x)),
                scale_motion=0.0,
            )
            _scored_x, _score = rank_crop_candidates(
                _cands, crop_y, eff_crop_w, eff_crop_h, resolved_scale,
                target_face, all_local_faces, source_w, source_h, current_crop_x,
            )
            if _scored_x is not None and _score >= _solved_score + CROP_SCORER_MIN_GAIN:
                clamped_x = float(np.clip(_scored_x, 0.0, eff_max_x))
                sys.stderr.write(
                    f"[CropScorer] t={rel_ts:.2f} candidates={len(_cands)} "
                    f"chosen_x={clamped_x:.0f} score={_score:.3f} (solved {_solved_score:.3f})\n")
        except Exception as _e:
            sys.stderr.write(f"[CropScorer] ranking failed ({_e}); containment solve kept\n")

    camera_motion_px = abs(clamped_x - (current_crop_x or clamped_x))
    scale_motion = abs(resolved_scale - current_scale)
    framing_score = evaluate_crop_composition_score(
        clamped_x, crop_y, eff_crop_w, eff_crop_h, resolved_scale,
        target_face, all_local_faces, source_w, source_h,
        camera_motion_px=camera_motion_px, scale_motion=scale_motion, shot_scale_state=shot_scale_state
    )

    diag = {
        "shot_id": shot_id,
        "shot_type": shot_type,
        "shot_class": shot_class,
        "camera_mode": camera_mode,
        "speaker": spk,
        "target_cx": target_cx,
        "clamped_x": clamped_x,
        "crop_y": crop_y,
        "crop_w": eff_crop_w,
        "crop_h": eff_crop_h,
        "crop_scale": resolved_scale,
        "scale_min": min_s,
        "scale_max": max_s,
        "ideal_scale": ideal_scale,
        "scale_change_reason": scale_change_reason,
        "pair_needed_scale": pair_needed_scale,
        "face_detected": target_face is not None,
        "headroom_pct": None,
        "eye_line_pct": None,
        "containment_pass": is_contained,
        "framing_score": framing_score,
        "subject_safe_box": subject_safe_box,
        "corridor_info": corridor_info,
        "camera_motion_px": camera_motion_px,
        "scale_motion": scale_motion,
        "subject_role": subject_role,
        "visual_prominence": target_track.get("visual_prominence", 0.0) if target_track else 0.0,
        "config": cfg,
    }

    if target_face is not None:
        if "head_top_y" in target_face:
            diag["headroom_pct"] = (target_face["head_top_y"] - crop_y) / max(1.0, eff_crop_h) * 100.0
        if "eye_line_y" in target_face:
            diag["eye_line_pct"] = (target_face["eye_line_y"] - crop_y) / max(1.0, eff_crop_h) * 100.0
        diag["face_bbox"] = target_face.get("bbox")

    return clamped_x, crop_y, eff_crop_w, eff_crop_h, target_cx, shot_type, camera_mode, diag, shot_scale_state


detect_scene_cuts = sample_video_and_detect_shots
track_faces_in_shot = build_shot_tracks


# ─── 10. Global Trajectory Optimization & Pre-Render Validation ──────────────

def optimize_camera_trajectory(raw_keyframes, max_x: float):
    if len(raw_keyframes) <= 2:
        return raw_keyframes

    times = [kf[0] for kf in raw_keyframes]
    xs = np.array([kf[1] for kf in raw_keyframes], dtype=float)
    cuts = [kf[2] for kf in raw_keyframes]

    shot_segments, curr_seg = [], [0]
    for i in range(1, len(raw_keyframes)):
        if cuts[i]:
            shot_segments.append(curr_seg)
            curr_seg = [i]
        else:
            curr_seg.append(i)
    shot_segments.append(curr_seg)

    optimized_xs = xs.copy()
    for seg in shot_segments:
        if len(seg) >= 4 and HAS_SCIPY:
            seg_xs = xs[seg]
            wl = min(len(seg), 7)
            if wl % 2 == 0: wl -= 1
            if wl >= 3:
                try:
                    optimized_xs[seg] = np.clip(savgol_filter(seg_xs, window_length=wl, polyorder=2), 0.0, float(max_x))
                except Exception:
                    pass

    return [(times[i], float(optimized_xs[i]), cuts[i]) for i in range(len(raw_keyframes))]


def validate_and_correct_trajectory(
    keyframes, scale_keyframes, all_samples, start_sec: float,
    crop_w_baseline: float, source_w: float, source_h: float,
    adaptive_framing: bool = False,
):
    corrected = list(keyframes) if keyframes else []
    n_violations = n_corrections = n_insertions = 0
    pending_violation = None  # first sample of a suspected transient-free violation

    def _min_keyframe_distance(abs_t):
        return min((abs(kf[0] - abs_t) for kf in corrected), default=float("inf"))

    if all_samples and corrected:
        for rel_t, frame, faces in all_samples:
            genuine = [f for f in faces if f["conf"] >= 0.50 and not f["is_back"]]
            if not genuine:
                continue
            abs_t = start_sec + rel_t

            # Active scale keyframe at abs_t (latest skf with skf[0] <= abs_t)
            active_skf_idx = 0
            if scale_keyframes:
                for i, skf in enumerate(scale_keyframes):
                    if skf[0] <= abs_t + 0.001:
                        active_skf_idx = i
                    else:
                        break
            scale_at_t = scale_keyframes[active_skf_idx][1] if scale_keyframes else 1.0
            eff_crop_w, _ = compute_effective_crop_dims(scale_at_t, source_h, source_w, adaptive_framing=adaptive_framing)
            eff_max_x = max(0.0, source_w - eff_crop_w)

            # Active position keyframe at abs_t (latest kf with kf[0] <= abs_t)
            # CRITICAL INVARIANT: Keyframe i governs the interval [kf[i][0], kf[i+1][0]).
            # A sample frame before a shot cut (e.g. t < 22.80s) must NEVER match or overwrite
            # a post-cut keyframe (e.g. t = 22.80s).
            bki = 0
            for i, kf in enumerate(corrected):
                if kf[0] <= abs_t + 0.001:
                    bki = i
                else:
                    break

            kf_t, kf_x, is_cut = corrected[bki]

            primary_face = nearest_genuine_face(genuine, kf_x, eff_crop_w)

            sb = compute_subject_safe_box(
                sanitize_face_bbox_for_containment(primary_face, crop_w_baseline, source_w),
                source_w, source_h,
            )
            target_crop_x, _, _ = solve_containment_crop_x(sb, eff_crop_w, eff_max_x, source_w)

            is_any_face_contained = any(
                (f["center"][0] >= kf_x - 20.0 and f["center"][0] <= kf_x + eff_crop_w + 20.0)
                for f in genuine
            )
            if not is_any_face_contained:
                n_violations += 1
                clipped_x = float(np.clip(target_crop_x, 0.0, eff_max_x))
                # Case 1: the keyframe is wrong from its very own start → fix in place.
                if abs(abs_t - kf_t) <= 0.001:
                    corrected[bki] = (kf_t, clipped_x, is_cut)
                    n_corrections += 1
                    pending_violation = None
                    continue

                # Case 2: the keyframe was valid when it started, but the visible subject
                # changed at some point inside its interval. Rewriting the keyframe in place
                # would retroactively break the (correct) earlier part of the interval, so we
                # instead SPLIT the interval: keep the original keyframe intact and insert a
                # new hard-cut keyframe at the first sample where the old crop stopped
                # containing any face. Require the violation to persist across ≥2 consecutive
                # samples so a single-frame detection dropout does not churn the trajectory.
                if pending_violation is not None and pending_violation["bki"] == bki:
                    ins_t = pending_violation["abs_t"]
                    if _min_keyframe_distance(ins_t) >= 0.25:
                        insert_at = bki + 1
                        while insert_at < len(corrected) and corrected[insert_at][0] < ins_t:
                            insert_at += 1
                        corrected.insert(insert_at, (ins_t, pending_violation["x"], True))
                        n_insertions += 1
                    pending_violation = None
                else:
                    pending_violation = {"bki": bki, "abs_t": abs_t, "x": clipped_x}
            else:
                pending_violation = None

    scales = [kf[1] for kf in scale_keyframes] if scale_keyframes else [1.0]
    avg_scale = float(np.mean(scales))
    max_scale = float(np.max(scales))

    scale_escalation = False
    if len(scales) >= 5:
        tn = max(2, len(scales) // 5)
        body_avg = float(np.mean(scales[:-tn]))
        tail_avg = float(np.mean(scales[-tn:]))
        if tail_avg > body_avg + 0.08:
            scale_escalation = True
            sys.stderr.write(f"[Smart Framing] WARNING: scale escalation body={body_avg:.3f} tail={tail_avg:.3f}\n")

    return corrected, scale_keyframes, {
        "n_frames": len(all_samples) if all_samples else 0, "n_violations": n_violations,
        "n_corrections": n_corrections, "n_insertions": n_insertions,
        "scale_escalation_detected": scale_escalation,
        "avg_scale": avg_scale, "max_scale": max_scale
    }


# ─── 11. FFmpeg Expression Builders (Position + Scale + Y) ───────────────────

def build_balanced_binary_tree(intervals, fmt_func):
    """
    Recursively builds a balanced binary search tree over piecewise time intervals.
    Reduces nested if() expression depth from O(N) to O(log2 N).
    Prevents overflowing FFmpeg's internal expression recursion depth limit (max ~98).
    """
    if not intervals:
        return "0"
    if len(intervals) == 1:
        return intervals[0][1]
    mid = len(intervals) // 2
    split_t = intervals[mid - 1][0]
    left_part = build_balanced_binary_tree(intervals[:mid], fmt_func)
    right_part = build_balanced_binary_tree(intervals[mid:], fmt_func)
    return f"if(lt(t,{fmt_func(split_t)}),{left_part},{right_part})"


def build_ffmpeg_expr(keyframes: list, clip_start: float, default_x: float, max_legal_x: float = None) -> str:
    """
    Builds FFmpeg crop x-coordinate expression.
    Guarantees:
    1. Every coordinate is clamped strictly to [0, max_legal_x] (preventing x + w > source_w).
    2. Cubic Hermite smoothing (3u^2 - 2u^3) is preserved with zero overshoot.
    3. Expression depth is O(log N) <= 15 via balanced binary tree.

    Phase 3 (AutoFlip-inspired): the smoothstep window widens with move size
    so large pans keep a bounded peak speed instead of a fixed 0.30 s sweep.
    Endpoints (the containment-validated keyframe positions) are untouched;
    only the interpolation window between two legal positions changes, and
    hard-cut keyframes still jump instantly.
    """
    if not keyframes:
        clamped_def = int(round(default_x))
        if max_legal_x is not None:
            clamped_def = int(max(0, min(int(max_legal_x), clamped_def)))
        return str(clamped_def)

    def clamp(v):
        iv = int(round(v))
        if max_legal_x is not None:
            return int(max(0, min(int(max_legal_x), iv)))
        return iv

    if len(keyframes) == 1:
        return str(clamp(keyframes[0][1]))

    def fmt(v): return f"{v:.4f}".rstrip("0").rstrip(".")

    def ss(t0, dur, x0, x1):
        dur_str = fmt(max(0.001, dur))
        u = f"((t-{fmt(t0)})/{dur_str})"
        return f"({x0}+({x1}-{x0})*({u}*{u}*(3-2*{u})))"

    intervals = []
    for i in range(len(keyframes) - 1):
        tc, xc, _ = keyframes[i]
        tn, xn, is_cut = keyframes[i + 1]
        rn = tn - clip_start
        rc = tc - clip_start
        c_xc = clamp(xc)
        c_xn = clamp(xn)
        if is_cut or abs(c_xc - c_xn) < 1:
            intervals.append((rn, str(c_xc)))
        else:
            # AutoFlip-inspired motion-aware transition window (Phase 3):
            # dur = clamp(|Δx| / MAX_PAN_SPEED, TRANSITION_DUR, 3x). Bounded
            # peak pan speed; small moves keep the historical 0.30 s.
            move_px = abs(c_xc - c_xn)
            dur_want = move_px / max(1.0, AUTFLIP_MAX_PAN_SPEED)
            dur_cap = TRANSITION_DUR if not framing_smoothing_enabled() else 3.0 * TRANSITION_DUR
            trans_dur = float(np.clip(dur_want, TRANSITION_DUR, dur_cap))
            ts = max(rc, rn - trans_dur)
            dur = max(0.01, rn - ts)
            smooth_expr = ss(ts, dur, c_xc, c_xn)
            if ts > rc + 0.001:
                intervals.append((ts, str(c_xc)))
            intervals.append((rn, smooth_expr))

    intervals.append((float("inf"), str(clamp(keyframes[-1][1]))))
    return build_balanced_binary_tree(intervals, fmt)


def build_ffmpeg_scale_dim_expr(scale_keyframes, clip_start: float, baseline_crop_w: float, source_h: float, source_w: float, dim: str = "w"):
    """
    Legacy helper for scale dimension expressions, generating balanced binary tree.
    """
    if not scale_keyframes:
        w, h = compute_effective_crop_dims(1.0, source_h, source_w)
        return str(int(round(w))) if dim == "w" else str(int(round(h)))

    def fmt(v): return f"{v:.4f}".rstrip("0").rstrip(".")

    scales = [kf[1] for kf in scale_keyframes]
    if max(scales) - min(scales) < 0.01:
        w, h = compute_effective_crop_dims(scales[0], source_h, source_w)
        return str(int(round(w))) if dim == "w" else str(int(round(h)))

    intervals = []
    for i in range(len(scale_keyframes) - 1):
        tc, sc, _ = scale_keyframes[i]
        tn, _, _ = scale_keyframes[i + 1]
        rn = tn - clip_start
        cw, ch = compute_effective_crop_dims(sc, source_h, source_w)
        val = int(round(cw)) if dim == "w" else int(round(ch))
        intervals.append((rn, str(val)))

    lw, lh = compute_effective_crop_dims(scale_keyframes[-1][1], source_h, source_w)
    intervals.append((float("inf"), str(int(round(lw)) if dim == "w" else int(round(lh)))))
    return build_balanced_binary_tree(intervals, fmt)


def build_ffmpeg_y_expr(y_keyframes, clip_start: float, source_h: float, max_legal_y: float = 0) -> str:
    """
    Builds FFmpeg crop y-coordinate expression with balanced binary tree and clamping.
    """
    if not y_keyframes or max_legal_y <= 0:
        return "0"

    def clamp(v):
        iv = int(round(v))
        return int(max(0, min(int(max_legal_y), iv)))

    ys = [kf[1] for kf in y_keyframes]
    if max(ys) - min(ys) < 2.0:
        return str(clamp(ys[0]))

    def fmt(v): return f"{v:.4f}".rstrip("0").rstrip(".")

    def ss(t0, dur, y0, y1):
        dur_str = fmt(max(0.001, dur))
        u = f"((t-{fmt(t0)})/{dur_str})"
        return f"({y0}+({y1}-{y0})*({u}*{u}*(3-2*{u})))"

    intervals = []
    for i in range(len(y_keyframes) - 1):
        tc, yc, _ = y_keyframes[i]
        tn, yn, is_cut = y_keyframes[i + 1]
        rn = tn - clip_start
        rc = tc - clip_start
        c_yc = clamp(yc)
        c_yn = clamp(yn)
        if is_cut or abs(c_yc - c_yn) < 2:
            intervals.append((rn, str(c_yc)))
        else:
            ts = max(rc, rn - TRANSITION_DUR)
            dur = max(0.01, rn - ts)
            smooth_expr = ss(ts, dur, c_yc, c_yn)
            if ts > rc + 0.001:
                intervals.append((ts, str(c_yc)))
            intervals.append((rn, smooth_expr))

    intervals.append((float("inf"), str(clamp(y_keyframes[-1][1]))))
    return build_balanced_binary_tree(intervals, fmt)


# ─── 11B. DualFrame Independent Panel Trajectory Solver ───────────────────────

@dataclass
class CropRectExpr:
    """
    Independent crop rectangle expression for a single panel.
    Provides attribute access (x, y, w, h) and dictionary-style access (['x'], ['y'], etc.)
    for seamless integration with both Python and JSON/Rust pipelines.
    """
    x: str
    y: str
    w: str
    h: str

    def to_dict(self) -> Dict[str, str]:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}

    def __getitem__(self, key: str) -> str:
        if key in ("x", "y", "w", "h"):
            return getattr(self, key)
        raise KeyError(f"Invalid crop parameter key: {key}")

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        return getattr(self, key, default)

    def __iter__(self):
        return iter((self.x, self.y, self.w, self.h))


@dataclass
class DualFrameTrajectories:
    """
    Dual-panel crop trajectory container holding independent expressions
    for top and bottom 9:8 panels.
    """
    top_crop: CropRectExpr
    bottom_crop: CropRectExpr

    def to_dict(self) -> Dict[str, Dict[str, str]]:
        return {
            "top_crop": self.top_crop.to_dict(),
            "bottom_crop": self.bottom_crop.to_dict(),
        }

    def __getitem__(self, key: str) -> CropRectExpr:
        if key == "top_crop":
            return self.top_crop
        elif key == "bottom_crop":
            return self.bottom_crop
        raise KeyError(f"Invalid trajectory key: {key}")

    def get(self, key: str, default: Any = None) -> Any:
        if key == "top_crop":
            return self.top_crop
        elif key == "bottom_crop":
            return self.bottom_crop
        return default


@dataclass
class LayoutSegment:
    """
    Temporal layout segment descriptor for either 'single' (9:16) or 'dual_stack' (stacked 9:8) layout.
    """
    mode: str
    start: float
    end: float
    crop: Optional[Any] = None
    face_bounds: Optional[Dict[str, float]] = None
    top_track_id: Optional[Any] = None
    bottom_track_id: Optional[Any] = None
    top_crop: Optional[Any] = None
    bottom_crop: Optional[Any] = None
    top_face_bounds: Optional[Dict[str, float]] = None
    bottom_face_bounds: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        if self.mode == "single":
            c = self.crop
            if hasattr(c, "to_dict"):
                c = c.to_dict()
            elif isinstance(c, dict):
                pass
            else:
                c = {}
            d = {
                "mode": "single",
                "start": round(float(self.start), 3),
                "end": round(float(self.end), 3),
                "crop": c,
            }
            if self.face_bounds is not None:
                d["face_bounds"] = self.face_bounds
            return d
        else:
            tc = self.top_crop.to_dict() if hasattr(self.top_crop, "to_dict") else (self.top_crop if isinstance(self.top_crop, dict) else {})
            bc = self.bottom_crop.to_dict() if hasattr(self.bottom_crop, "to_dict") else (self.bottom_crop if isinstance(self.bottom_crop, dict) else {})
            d = {
                "mode": "dual_stack",
                "start": round(float(self.start), 3),
                "end": round(float(self.end), 3),
                "top_track_id": self.top_track_id,
                "bottom_track_id": self.bottom_track_id,
                "top_crop": tc,
                "bottom_crop": bc,
            }
            if self.top_face_bounds is not None:
                d["top_face_bounds"] = self.top_face_bounds
            if self.bottom_face_bounds is not None:
                d["bottom_face_bounds"] = self.bottom_face_bounds
            return d


@dataclass
class SmartFramingPlan:
    """
    Complete Smart Framing Execution Plan contract.
    Contains overall layout mode, backward-compatible single 9:16 fallback parameters (x, y, w, h),
    a contiguous sequence of LayoutSegments covering the entire clip duration, and the framing mode
    ("original" full-bleed 9:16 vs "adaptive" square inner composition) that the renderer keys on.
    """
    mode: str
    x: str
    y: str
    w: str
    h: str
    segments: List[LayoutSegment] = field(default_factory=list)
    face_bounds: Optional[Dict[str, float]] = None
    framing: str = "original"
    is_emergency_fallback: bool = False
    fallback_reason: Optional[str] = None
    # Phase 2 Speaker Intelligence: Re-ID gallery + fused active-speaker
    # intervals emitted for the Rust engine to persist. Purely additive —
    # the deterministic plan fields are never derived from this block.
    speaker_intel: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "mode": self.mode,
            "x": str(self.x),
            "y": str(self.y),
            "w": str(self.w),
            "h": str(self.h),
            "segments": [s.to_dict() for s in self.segments],
            "framing": self.framing,
            "is_emergency_fallback": self.is_emergency_fallback,
            "fallback_reason": self.fallback_reason,
        }
        if self.face_bounds is not None:
            d["face_bounds"] = self.face_bounds
        if self.speaker_intel is not None:
            d["speakerIntel"] = self.speaker_intel
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


def compute_panel_crop_dimensions(
    subject_box: Any = None,
    source_w: float = 1920.0,
    source_h: float = 1080.0,
    current_scale: float = 1.0,
) -> Tuple[int, int]:
    """
    Computes independent 9:8 aspect ratio panel crop dimensions (crop_w, crop_h).
    Enforces:
    1. crop_w / crop_h == 9.0 / 8.0 = 1.125 (abs(crop_w - round(crop_h * 1.125)) <= 1).
    2. Even pixel dimensions for both crop_w and crop_h (required for YUV420p / H.264 / HEVC encoding).
    3. Strict bounding within source frame dimensions (2 <= crop_w <= source_w, 2 <= crop_h <= source_h).
    4. Dynamic scaling proportional to subject size or explicit current_scale.
    """
    scale = float(current_scale)

    if subject_box is not None:
        if isinstance(subject_box, (int, float)):
            scale = float(subject_box)
        elif isinstance(subject_box, dict):
            if "scale" in subject_box:
                scale = float(subject_box["scale"])
            elif current_scale == 1.0:
                fw = None
                bbox = subject_box.get("bbox")
                if bbox and len(bbox) >= 3:
                    fw = bbox[2]
                elif "width" in subject_box:
                    fw = subject_box["width"]
                elif "w" in subject_box:
                    fw = subject_box["w"]

                if fw is not None and source_w > 0:
                    face_ratio = fw / source_w
                    if face_ratio >= 0.20:
                        scale = 0.90
                    elif face_ratio >= 0.12:
                        scale = 1.00
                    elif face_ratio >= 0.06:
                        scale = 1.12
                    elif face_ratio >= 0.03:
                        scale = 1.22
                    else:
                        scale = 1.30
        elif isinstance(subject_box, (list, tuple)) and len(subject_box) >= 3 and current_scale == 1.0:
            fw = subject_box[2]
            if source_w > 0:
                face_ratio = fw / source_w
                if face_ratio >= 0.20:
                    scale = 0.90
                elif face_ratio >= 0.12:
                    scale = 1.00
                elif face_ratio >= 0.06:
                    scale = 1.12
                elif face_ratio >= 0.03:
                    scale = 1.22
                else:
                    scale = 1.30

    scale = max(0.50, min(2.50, scale))

    panel_ratio = 9.0 / 8.0

    if source_w / max(1.0, source_h) >= panel_ratio:
        baseline_h = float(source_h)
        baseline_w = baseline_h * panel_ratio
    else:
        baseline_w = float(source_w)
        baseline_h = baseline_w / panel_ratio

    eff_h = baseline_h / scale
    eff_w = baseline_w / scale

    # Clamp effective dimensions to source bounds
    if eff_h > source_h:
        eff_h = float(source_h)
        eff_w = eff_h * panel_ratio

    if eff_w > source_w:
        eff_w = float(source_w)
        eff_h = eff_w / panel_ratio

    # Convert to even integers while strictly preserving 9:8
    crop_h = int(round(eff_h))
    if crop_h > int(source_h):
        crop_h = int(source_h)
    if crop_h % 2 != 0:
        crop_h -= 1
    crop_h = max(2, min(int(source_h), crop_h))

    crop_w = int(round(crop_h * panel_ratio))
    if crop_w > int(source_w):
        crop_w = int(source_w)
        if crop_w % 2 != 0:
            crop_w -= 1
        crop_h = int(round(crop_w / panel_ratio))
        if crop_h % 2 != 0:
            crop_h -= 1
        crop_h = max(2, min(int(source_h), crop_h))
        crop_w = int(round(crop_h * panel_ratio))

    # Make crop_w even while maintaining abs(crop_w - round(crop_h * 1.125)) <= 1
    if crop_w % 2 != 0:
        if crop_w + 1 <= int(source_w):
            crop_w += 1
        else:
            crop_w -= 1

    even_max_w = int(source_w) // 2 * 2
    even_max_h = int(source_h) // 2 * 2
    crop_w = max(2, min(even_max_w, crop_w))
    crop_h = max(2, min(even_max_h, crop_h))

    return int(crop_w), int(crop_h)


# ─── Close-Speaker Panel Isolation (DualFrame) ────────────────────────────────

def _panel_crop_fits_subject(
    crop_w: float,
    crop_h: float,
    subject_safe_box: dict,
    source_w: float,
    source_h: float,
) -> bool:
    """
    Returns True if a crop_w x crop_h panel can legally contain the intended
    subject's safe box (head + shoulders) somewhere within the source frame.
    Uses the same head-priority containment rule as solve_containment_crop_x:
    if the full safe box cannot fit, the head box alone must fit.
    """
    margin_x = crop_w * SAFETY_MARGIN_RATIO
    subj_w = subject_safe_box["width"]
    head_left = subject_safe_box.get("head_left", subject_safe_box["left"])
    head_right = subject_safe_box.get("head_right", subject_safe_box["right"])

    if subj_w + 2.0 * margin_x <= crop_w:
        need_w = subj_w + 2.0 * margin_x
    else:
        need_w = (head_right - head_left) + 2.0 * margin_x

    if need_w > crop_w:
        return False
    # Horizontal placement must exist within [0, source_w - crop_w]
    if crop_w > source_w:
        return False
    return True


def _crop_excludes_other(
    crop_x: float,
    crop_w: float,
    other_safe_box: dict,
    source_w: float,
    clearance_px: float,
) -> bool:
    """
    Returns True if the candidate panel crop [crop_x, crop_x + crop_w] excludes
    the other speaker's subject-safe box with the required clearance.
    The other speaker is excluded when their safe box lies fully outside the
    crop window with at least clearance_px gap on the near side.
    """
    other_left = other_safe_box["left"]
    other_right = other_safe_box["right"]
    crop_right = crop_x + crop_w
    # Other speaker entirely left of the crop with clearance
    if other_right <= crop_x - clearance_px:
        return True
    # Other speaker entirely right of the crop with clearance
    if other_left >= crop_right + clearance_px:
        return True
    return False


def _find_isolated_crop_x(
    subject_safe_box: dict,
    other_safe_box: dict,
    crop_w: float,
    max_x: float,
    source_w: float,
    clearance_px: float,
) -> Optional[float]:
    """
    Searches for a legal crop_x that contains the intended subject (hard
    containment, same corridor rule as solve_containment_crop_x) while
    excluding the other speaker's safe box with clearance.
    Returns the placement closest to the subject's compositional center, or
    None if no legal isolated placement exists at this crop width.
    """
    margin_x = crop_w * SAFETY_MARGIN_RATIO
    subj_left = subject_safe_box["left"]
    subj_right = subject_safe_box["right"]
    subj_cx = subject_safe_box["center_x"]
    subj_w = subject_safe_box["width"]
    head_left = subject_safe_box.get("head_left", subj_left)
    head_right = subject_safe_box.get("head_right", subj_right)

    # Subject containment corridor (identical rule to solve_containment_crop_x)
    if subj_w + 2.0 * margin_x <= crop_w:
        min_allowed_x = max(0.0, subj_right + margin_x - crop_w)
        max_allowed_x = min(float(max_x), subj_left - margin_x)
    else:
        min_allowed_x = max(0.0, head_right + margin_x - crop_w)
        max_allowed_x = min(float(max_x), head_left - margin_x)

    if min_allowed_x > max_allowed_x:
        # Subject cannot be placed with full safety margins (near frame edge
        # or oversized safe box): fall back to the same center-clamped
        # placement rule as solve_containment_crop_x. The subject remains
        # contained (reduced margin), so isolation zones are still evaluated.
        min_allowed_x = max_allowed_x = float(np.clip(subj_cx - crop_w / 2.0, 0.0, float(max_x)))
        # If even the head cannot fit within the crop, no contained placement exists
        if (head_right - head_left) > crop_w:
            return None

    # Compositional ideal (gaze-neutral 0.50, matching the deadband center)
    ideal_x = float(np.clip(subj_cx - crop_w * 0.50, min_allowed_x, max_allowed_x))

    # Collect isolated placements: corridor positions excluding the other speaker
    candidates = []
    other_left = other_safe_box["left"]
    other_right = other_safe_box["right"]

    # Rounding buffer: final crop_x is rounded to int; guarantee the clearance
    # survives rounding by tightening zone bounds by up to 1px.
    round_buf = 1.0

    # Zone A: crop entirely right of the other speaker
    # crop_x >= other_right + clearance
    zone_a_min = max(min_allowed_x, other_right + clearance_px + round_buf)
    zone_a_max = max_allowed_x
    # Zone B: crop entirely left of the other speaker
    # crop_x + crop_w <= other_left - clearance
    zone_b_max = min(max_allowed_x, other_left - clearance_px - round_buf - crop_w)
    zone_b_min = min_allowed_x

    for zone_min, zone_max in ((zone_a_min, zone_a_max), (zone_b_min, zone_b_max)):
        if zone_min <= zone_max:
            lo = float(np.clip(ideal_x, zone_min, zone_max))
            candidates.append(lo)

    if not candidates:
        return None

    # Deterministic: placement closest to compositional ideal
    return min(candidates, key=lambda c: abs(c - ideal_x))


def _nearest_other_face_at(
    other_track: Optional[dict],
    t_sec: float,
    tol_sec: float = ISOLATION_PAIR_TOL_SEC,
) -> Optional[dict]:
    """
    Returns the other speaker's face dict nearest in time to t_sec (within
    tol_sec), or None. Used to project the other speaker's position at the
    intended subject's keyframe times.
    """
    if not other_track:
        return None
    dets = other_track.get("detections", [])
    best = None
    best_dt = None
    for det in dets:
        dt = abs(det[0] - t_sec)
        if dt <= tol_sec and (best_dt is None or dt < best_dt):
            best = det
            best_dt = dt
    if best is None or len(best) < 2 or not isinstance(best[1], dict):
        return None
    face = best[1]
    return face if "bbox" in face else None


def _solve_panel_isolation_scale(
    own_track: Optional[dict],
    other_track: Optional[dict],
    base_scale: float,
    source_w: float,
    source_h: float,
) -> Tuple[float, Dict[str, Any]]:
    """
    Computes the close-speaker isolation (pinch-zoom) scale for ONE DualFrame
    panel, given the OTHER speaker's track.

    Decision principle — projected crop-vs-other-subject contamination:
    1. Pair each of the panel subject's detections with the other speaker's
       nearest-in-time detection (|Δt| <= ISOLATION_PAIR_TOL_SEC).
    2. For each pair, test whether the BASELINE panel crop (base_scale) admits
       a legal placement that contains the subject AND excludes the other
       speaker's safe box with ISOLATION_CLEARANCE_RATIO clearance.
    3. If any paired moment is contaminated (no such placement exists), search
       deterministic pinch-zoom scales in ISOLATION_SCALE_STEP increments up
       to MAX_ISOLATION_SCALE until every paired moment admits an isolated
       placement. The FIRST (smallest) scale that isolates all paired moments
       is chosen — minimal zoom that achieves isolation.
    4. If no scale <= MAX_ISOLATION_SCALE isolates every paired moment, the
       panel keeps the largest scale that isolates the most paired moments
       (deterministic bounded fallback: report, never overzoom, never disable
       DualFrame).

    Returns (isolation_scale, report) where report carries the decision
    telemetry (contaminated pair count, chosen scale, fallback flag).
    """
    report: Dict[str, Any] = {
        "base_scale": round(float(base_scale), 3),
        "isolation_scale": round(float(base_scale), 3),
        "paired_moments": 0,
        "contaminated_moments": 0,
        "isolation_active": False,
        "fallback": False,
    }

    if not own_track or not other_track:
        return float(base_scale), report

    own_dets = own_track.get("detections", [])
    if not own_dets:
        return float(base_scale), report

    clearance_px = ISOLATION_CLEARANCE_RATIO * source_w

    # Pair own detections with the other speaker's nearest detections
    pairs = []
    for det in own_dets:
        face = det[1] if len(det) > 1 and isinstance(det[1], dict) else None
        if not face or "bbox" not in face:
            continue
        other_face = _nearest_other_face_at(other_track, det[0])
        if other_face is None:
            continue
        pairs.append((face, other_face))

    report["paired_moments"] = len(pairs)
    if not pairs:
        return float(base_scale), report

    def isolated_at_scale(scale: float) -> Tuple[bool, int]:
        crop_w, crop_h = compute_panel_crop_dimensions(
            subject_box=None,
            source_w=source_w,
            source_h=source_h,
            current_scale=scale,
        )
        max_x = max(0.0, source_w - crop_w)
        ok_count = 0
        for own_face, other_face in pairs:
            own_box = compute_subject_safe_box(own_face, source_w, source_h)
            other_box = compute_subject_safe_box(other_face, source_w, source_h)
            if not _panel_crop_fits_subject(crop_w, crop_h, own_box, source_w, source_h):
                continue
            pos = _find_isolated_crop_x(own_box, other_box, crop_w, max_x, source_w, clearance_px)
            if pos is not None:
                ok_count += 1
        return ok_count == len(pairs), ok_count

    fully_isolated, _ = isolated_at_scale(float(base_scale))
    if fully_isolated:
        return float(base_scale), report

    report["contaminated_moments"] = sum(
        1 for _ in pairs
    )
    report["isolation_active"] = True

    # Deterministic pinch-zoom search: smallest scale that isolates all pairs
    best_scale = float(base_scale)
    best_ok = -1
    s = float(base_scale)
    while s <= MAX_ISOLATION_SCALE + 1e-9:
        ok, ok_count = isolated_at_scale(s)
        if ok:
            report["isolation_scale"] = round(s, 3)
            report["fallback"] = False
            return s, report
        if ok_count > best_ok:
            best_ok = ok_count
            best_scale = s
        s += ISOLATION_SCALE_STEP

    # Bounded fallback: geometrically impossible to fully isolate within the
    # zoom ceiling. Keep the best-effort scale (most isolated moments), never
    # overzoom past MAX_ISOLATION_SCALE, never disable DualFrame.
    report["isolation_scale"] = round(best_scale, 3)
    report["fallback"] = True
    return best_scale, report


def _isolation_safe_region(
    subject_safe_box: dict,
    other_safe_box: dict,
    crop_w: float,
    max_x: float,
    source_w: float,
    clearance_px: float,
) -> Optional[Tuple[float, float]]:
    """
    Computes the isolation-safe crop_x region [min_safe_x, max_safe_x] for the
    intended subject: the union of legal placements that contain the subject
    (same corridor rule as solve_containment_crop_x) while excluding the other
    speaker's safe box with clearance.

    The safe region is the union of two disjoint zones:
      Zone A: crop entirely right of the other speaker (crop_x >= other_right + clearance)
      Zone B: crop entirely left of the other speaker  (crop_x + crop_w <= other_left - clearance)

    Head-priority corridor relaxation: when the full subject safe box cannot
    be placed inside any exclusion zone but the subject's HEAD still can —
    the same relaxation solve_containment_crop_x applies when the full box
    cannot fit the crop — the safe region is derived from the head corridor.
    Isolation stays ACTIVE right up to true geometric impossibility so the
    constrained trajectory remains continuous across genuine boundary
    changes (no jump through the unconstrained fallback).

    Returns (min_safe_x, max_safe_x) spanning the union, or None when no legal
    isolated placement exists at this crop width (bounded fallback applies).
    """
    margin_x = crop_w * SAFETY_MARGIN_RATIO
    subj_left = subject_safe_box["left"]
    subj_right = subject_safe_box["right"]
    subj_cx = subject_safe_box["center_x"]
    subj_w = subject_safe_box["width"]
    head_left = subject_safe_box.get("head_left", subj_left)
    head_right = subject_safe_box.get("head_right", subj_right)

    other_left = other_safe_box["left"]
    other_right = other_safe_box["right"]

    # Rounding buffer: final crop_x is rounded to int; guarantee the clearance
    # survives rounding by tightening zone bounds by up to 1px (same rule as
    # _find_isolated_crop_x).
    round_buf = 1.0

    def _corridor_zones(
        corr_min: float, corr_max: float
    ) -> List[Tuple[float, float]]:
        # Exclusion zones for one subject-containment corridor: the legal
        # placements inside [corr_min, corr_max] that exclude the other
        # speaker's safe box with clearance.
        # Zone A: crop entirely right of the other speaker
        zone_a_min = max(corr_min, other_right + clearance_px + round_buf)
        zone_a_max = corr_max
        # Zone B: crop entirely left of the other speaker
        zone_b_min = corr_min
        zone_b_max = min(corr_max, other_left - clearance_px - round_buf - crop_w)
        zs = []
        if zone_a_min <= zone_a_max:
            zs.append((zone_a_min, zone_a_max))
        if zone_b_min <= zone_b_max:
            zs.append((zone_b_min, zone_b_max))
        return zs

    # Subject containment corridor (identical rule to solve_containment_crop_x)
    full_box_fits = subj_w + 2.0 * margin_x <= crop_w
    if full_box_fits:
        min_allowed_x = max(0.0, subj_right + margin_x - crop_w)
        max_allowed_x = min(float(max_x), subj_left - margin_x)
    else:
        min_allowed_x = max(0.0, head_right + margin_x - crop_w)
        max_allowed_x = min(float(max_x), head_left - margin_x)

    if min_allowed_x > max_allowed_x:
        # Subject cannot be placed with full safety margins: same center-clamped
        # fallback corridor as solve_containment_crop_x / _find_isolated_crop_x.
        min_allowed_x = max_allowed_x = float(np.clip(subj_cx - crop_w / 2.0, 0.0, float(max_x)))
        if (head_right - head_left) > crop_w:
            return None

    zones = _corridor_zones(min_allowed_x, max_allowed_x)

    if not zones and full_box_fits:
        # Head-priority corridor relaxation (smooth constrained→fallback
        # transition): the full subject safe box cannot be placed while
        # excluding the other speaker, but the subject's HEAD still can —
        # the exact relaxation solve_containment_crop_x itself applies when
        # the full box cannot fit the crop. Keep isolation ACTIVE on the
        # head corridor instead of dropping to the unconstrained placement,
        # which would jump the camera away from the previous constrained
        # position in a single step.
        head_min_x = max(0.0, head_right + margin_x - crop_w)
        head_max_x = min(float(max_x), head_left - margin_x)
        if head_min_x <= head_max_x:
            zones = _corridor_zones(head_min_x, head_max_x)

    if not zones:
        return None

    # Union of disjoint zones: [min(zone mins), max(zone maxes)]. When both
    # zones exist they are separated by the other speaker's forbidden gap; the
    # caller clamps into this union region and the resulting placement always
    # lands inside one of the zones (never inside the gap), because the
    # forbidden gap is excluded by the per-zone clamp below.
    region_min = min(z[0] for z in zones)
    region_max = max(z[1] for z in zones)
    return (float(region_min), float(region_max))


def _clamp_into_isolation_safe_region(
    desired_x: float,
    safe_region: Tuple[float, float],
    subject_safe_box: dict,
    other_safe_box: dict,
    crop_w: float,
    max_x: float,
    source_w: float,
    clearance_px: float,
    hysteresis_px: float = 0.0,
) -> float:
    """
    Clamps desired_x into the isolation-safe region while keeping the subject
    contained. The safe region may be a union of two disjoint zones (A: right
    of the other speaker, B: left of the other speaker) separated by the other
    speaker's forbidden gap. The result is the position inside the union that
    is closest to desired_x (deterministic, no oscillation: ties resolved by
    preferring the zone containing desired_x's side of the subject center).

    hysteresis_px: temporal-stability inset applied to the other-speaker-
    derived zone edges (zone_a_min / zone_b_max). The clamped placement is
    kept at least hysteresis_px away from the other speaker's exclusion
    boundary, so boundary jitter below hysteresis_px cannot move the camera
    and the placement stays legal under any sub-threshold boundary change.
    When the inset empties a zone, the zone collapses to its subject-derived
    edge (the legal placement farthest from the other speaker) instead of
    falling back toward the jitter-sensitive boundary.
    """
    region_min, region_max = safe_region
    margin_x = crop_w * SAFETY_MARGIN_RATIO
    subj_left = subject_safe_box["left"]
    subj_right = subject_safe_box["right"]
    subj_cx = subject_safe_box["center_x"]
    subj_w = subject_safe_box["width"]
    head_left = subject_safe_box.get("head_left", subj_left)
    head_right = subject_safe_box.get("head_right", subj_right)

    other_left = other_safe_box["left"]
    other_right = other_safe_box["right"]
    round_buf = 1.0

    def _corridor_zones(
        corr_min: float, corr_max: float
    ) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
        # Exclusion zones for one subject-containment corridor, intersected
        # with the safe-region union bounds. The zone_a min edge and zone_b
        # max edge are the other-speaker-derived edges the hysteresis inset
        # below pulls away from.
        # Zone A: crop entirely right of the other speaker
        za_min = max(corr_min, other_right + clearance_px + round_buf)
        za_max = min(corr_max, region_max)
        # Zone B: crop entirely left of the other speaker
        zb_min = max(region_min, corr_min)
        zb_max = min(corr_max, other_left - clearance_px - round_buf - crop_w)
        zone_a = (za_min, za_max) if za_min <= za_max else None
        zone_b = (zb_min, zb_max) if zb_min <= zb_max else None
        return zone_a, zone_b

    # Subject containment corridor (identical rule to solve_containment_crop_x)
    full_box_fits = subj_w + 2.0 * margin_x <= crop_w
    if full_box_fits:
        min_allowed_x = max(0.0, subj_right + margin_x - crop_w)
        max_allowed_x = min(float(max_x), subj_left - margin_x)
    else:
        min_allowed_x = max(0.0, head_right + margin_x - crop_w)
        max_allowed_x = min(float(max_x), head_left - margin_x)

    if min_allowed_x > max_allowed_x:
        min_allowed_x = max_allowed_x = float(np.clip(subj_cx - crop_w / 2.0, 0.0, float(max_x)))

    zone_a, zone_b = _corridor_zones(min_allowed_x, max_allowed_x)

    if zone_a is None and zone_b is None and full_box_fits:
        # Head-priority corridor relaxation — MUST mirror
        # _isolation_safe_region exactly: when the safe region was derived
        # from the head corridor (full safe box unplaceable but the head
        # placeable), the full-box zones here would be empty and the clamp
        # would degrade to an unconstrained placement, reintroducing the
        # constrained→fallback transition jump this way.
        head_min_x = max(0.0, head_right + margin_x - crop_w)
        head_max_x = min(float(max_x), head_left - margin_x)
        if head_min_x <= head_max_x:
            zone_a, zone_b = _corridor_zones(head_min_x, head_max_x)

    zones = []
    # Zone A: crop entirely right of the other speaker. zone_a_min is the
    # other-speaker-derived edge; the hysteresis inset pulls the placement
    # right, away from the other speaker.
    if zone_a is not None:
        zone_a_min, zone_a_max = zone_a
        inset_a_min = zone_a_min + hysteresis_px
        if inset_a_min <= zone_a_max:
            zones.append((inset_a_min, zone_a_max))
        else:
            # Inset empties the zone: collapse to the subject-derived edge
            # (the legal placement farthest from the other speaker).
            zones.append((zone_a_max, zone_a_max))
    # Zone B: crop entirely left of the other speaker. zone_b_max is the
    # other-speaker-derived edge; the hysteresis inset pulls the placement
    # left, away from the other speaker.
    if zone_b is not None:
        zone_b_min, zone_b_max = zone_b
        inset_b_max = zone_b_max - hysteresis_px
        if zone_b_min <= inset_b_max:
            zones.append((zone_b_min, inset_b_max))
        else:
            # Inset empties the zone: collapse to the subject-derived edge
            # (the legal placement farthest from the other speaker).
            zones.append((zone_b_min, zone_b_min))
    if not zones:
        # No isolated placement at this width: keep the desired position
        # (bounded fallback — same behavior as the pre-existing code path).
        return float(np.clip(desired_x, 0.0, float(max_x)))

    # Closest legal point in the union of zones to desired_x
    candidates = []
    for z_min, z_max in zones:
        candidates.append(float(np.clip(desired_x, z_min, z_max)))
    return min(candidates, key=lambda c: abs(c - desired_x))


def _solve_isolation_aware_crop_x(
    face_det: dict,
    other_track: Optional[dict],
    t_sec: float,
    crop_w: float,
    max_legal_x: float,
    source_w: float,
    source_h: float,
    current_crop_x=None,
    iso_state: Optional[dict] = None,
) -> float:
    """
    Solves the panel crop_x for one detection with close-speaker isolation
    awareness. Isolation is CONDITIONAL and acts as a CONSTRAINT on the normal
    containment trajectory, never a replacement for it:

    PART 1 — Clean pass-through: when the normal containment placement
    (base_x, with its gaze preference, deadband and trajectory continuity via
    current_crop_x) already excludes the other speaker's safe box with the
    required clearance, base_x is returned UNCHANGED. No isolation solve runs,
    so clean moments keep the exact normal DualFrame camera behavior.

    PART 2 — Constrained tracking under contamination: when base_x would
    include the other speaker, the normal desired trajectory is clamped into
    the isolation-safe region (subject contained + other speaker excluded with
    clearance), inset ISOLATION_BOUNDARY_HYSTERESIS_PX away from the
    other-speaker-derived zone edges. The camera keeps tracking the intended
    speaker normally inside that region; only the boundary constrains it.
    Temporal hysteresis (iso_state) holds the previous CONSTRAINED position
    when it is still legal under the current geometry and the CONSTRAINED
    desired position moved less than ISOLATION_BOUNDARY_HYSTERESIS_PX —
    suppressing camera movement caused by small other-speaker / safe-box
    jitter while genuine target movement and genuine boundary changes still
    reframe the camera.

    Head-priority relaxation: when the full subject safe box cannot be
    placed inside an exclusion zone but the subject's head still can, the
    safe region relaxes to the head corridor (the same relaxation the normal
    solver applies when the full box cannot fit the crop). Isolation stays
    ACTIVE and the camera stays constrained near its previous position
    instead of jumping to the unconstrained placement.

    Bounded fallback: when no isolated placement exists even head-priority
    (geometrically impossible), the normal containment placement is kept
    (subject stays contained; other speaker may intrude — never overzoom,
    never disable), handed off CONTINUOUSLY from the previously constrained
    position: the first release step is capped at the compositional deadband
    scale and subsequent keyframes keep converging toward the normal
    trajectory — no single-step jump, no frozen crop. Hard subject
    containment always wins over smoothness.
    """
    subject_safe_box = compute_subject_safe_box(face_det, source_w, source_h)
    base_x, _, _ = solve_containment_crop_x(
        subject_safe_box, float(crop_w), float(max_legal_x), float(source_w),
        current_crop_x=current_crop_x,
    )

    if other_track is None:
        return base_x

    other_face = _nearest_other_face_at(other_track, t_sec)
    if other_face is None:
        return base_x

    other_safe_box = compute_subject_safe_box(other_face, source_w, source_h)
    clearance_px = ISOLATION_CLEARANCE_RATIO * source_w

    # ── PART 1: Clean pass-through ────────────────────────────────────────
    # The normal placement already excludes the other speaker with clearance:
    # normal DualFrame behavior passes through completely unchanged.
    if _crop_excludes_other(base_x, float(crop_w), other_safe_box, float(source_w), clearance_px):
        if iso_state is not None:
            iso_state["last_x"] = base_x
            iso_state["last_mode"] = "clean"
        return base_x

    # ── PART 2: Constrained tracking inside the isolation-safe region ────
    safe_region = _isolation_safe_region(
        subject_safe_box, other_safe_box, float(crop_w), float(max_legal_x),
        float(source_w), clearance_px,
    )
    if safe_region is None:
        # Bounded fallback: no isolated placement exists at this crop width,
        # not even head-priority. Keep the normal containment placement
        # (subject contained; the other speaker may intrude — never overzoom,
        # never disable DualFrame). The handoff from the previously
        # CONSTRAINED position is continuous: the first release step is
        # capped at the compositional deadband scale (the system's own
        # movement-significance threshold) and subsequent keyframes keep
        # converging toward the normal trajectory — no single-step jump, no
        # frozen crop. Hard subject containment always wins over smoothness.
        fallback_x = base_x
        if (
            iso_state is not None
            and iso_state.get("last_x") is not None
            and iso_state.get("last_mode") in ("constrained", "release")
        ):
            last_x = float(np.clip(float(iso_state["last_x"]), 0.0, float(max_legal_x)))
            gap = base_x - last_x
            step_cap = max(35.0, float(crop_w) * 0.09)
            if abs(gap) > step_cap:
                released_x = last_x + step_cap if gap > 0.0 else last_x - step_cap
                # Hard containment wins over smoothness: never let the eased
                # step push the subject's head out of the crop (same
                # correction rule as solve_containment_crop_x).
                head_left = subject_safe_box.get("head_left", subject_safe_box["left"])
                head_right = subject_safe_box.get("head_right", subject_safe_box["right"])
                margin_x = float(crop_w) * SAFETY_MARGIN_RATIO
                if head_right > released_x + float(crop_w):
                    released_x = min(float(max_legal_x), max(released_x, head_right - float(crop_w) + margin_x))
                elif head_left < released_x:
                    released_x = max(0.0, min(released_x, head_left - margin_x))
                fallback_x = float(np.clip(released_x, 0.0, float(max_legal_x)))
        if iso_state is not None:
            iso_state["last_x"] = fallback_x
            iso_state["last_mode"] = "release" if fallback_x != base_x else "fallback"
        return fallback_x

    # Constrained desired position: the normal trajectory (base_x) clamped
    # into the safe region, inset ISOLATION_BOUNDARY_HYSTERESIS_PX away from
    # the other-speaker-derived zone edges. The inset makes this position
    # immune to sub-threshold boundary jitter while keeping it legal.
    desired_x = _clamp_into_isolation_safe_region(
        base_x, safe_region, subject_safe_box, other_safe_box,
        float(crop_w), float(max_legal_x), float(source_w), clearance_px,
        hysteresis_px=ISOLATION_BOUNDARY_HYSTERESIS_PX,
    )

    # Temporal hysteresis on the EFFECTIVE CONSTRAINED TRAJECTORY: hold the
    # previous constrained position when (a) it is still semantically legal
    # under the CURRENT geometry — the other speaker stays excluded with
    # clearance and the subject's head stays contained — and (b) the
    # CONSTRAINED desired position has not moved beyond the hysteresis
    # threshold. Comparing the constrained desired position against the
    # previously CONSTRAINED position (never the unconstrained base_x, which
    # differs from it by the constraint offset under active isolation) is
    # what lets the hold engage:
    #   - other-speaker / safe-box jitter below the threshold shifts the
    #     insetted desired position by at most the jitter -> hold (no move);
    #   - genuine target-speaker movement shifts the constrained desired
    #     position beyond the threshold -> camera follows naturally;
    #   - a genuine boundary change either makes the held position illegal
    #     or shifts the desired position beyond the threshold -> the camera
    #     smoothly re-constrains.
    if iso_state is not None and iso_state.get("last_x") is not None:
        last_x = float(np.clip(float(iso_state["last_x"]), 0.0, float(max_legal_x)))
        last_excludes_other = _crop_excludes_other(
            last_x, float(crop_w), other_safe_box, float(source_w), clearance_px
        )
        head_left = subject_safe_box.get("head_left", subject_safe_box["left"])
        head_right = subject_safe_box.get("head_right", subject_safe_box["right"])
        last_contains_subject = (
            head_left >= last_x - 2.0
            and head_right <= last_x + float(crop_w) + 2.0
        )
        if (
            last_excludes_other
            and last_contains_subject
            and abs(desired_x - last_x) <= ISOLATION_BOUNDARY_HYSTERESIS_PX
        ):
            iso_state["last_x"] = last_x
            iso_state["last_mode"] = "constrained"
            return last_x

    final_x = float(np.clip(desired_x, 0.0, float(max_legal_x)))
    if iso_state is not None:
        iso_state["last_x"] = final_x
        iso_state["last_mode"] = "constrained"
    return final_x


# ─── Conditional Subject-Tight Framing (Normal DualFrame, §27) ────────────────

def _solve_panel_composition_tightening(
    track: Optional[dict],
    base_crop_w: int,
    base_crop_h: int,
    source_w: float,
    source_h: float,
    panel_role: str = "top",
) -> Tuple[Optional[float], Dict[str, Any]]:
    """
    Computes the conditional subject-tight framing scale for ONE normal
    DualFrame panel (§27). This is a MODERATE composition tightening for
    background-heavy panels — it is NOT close-speaker isolation and must never
    be used to solve cross-speaker contamination (the sealed isolation system
    owns that exclusively).

    Decision principle — body-evidence occupancy:
    1. Body evidence: gather per-detection YOLO person bodies (person_bbox,
       corner format (px1, py1, px2, py2)) from the panel subject's face
       detections, falling back to the track-level person_bbox. Panels with
       NO body evidence keep the existing composition unchanged (face width
       already drives the baseline scale; without body evidence the amount
       of non-speaker content cannot be measured reliably).
    2. Occupancy: median person-body width / baseline panel crop width.
       occupancy >= TIGHTEN_ENTER_OCCUPANCY → the panel is ALREADY WELL
       COMPOSED: zero tightening (true pass-through).
    3. Background-heavy (occupancy < TIGHTEN_ENTER_OCCUPANCY): search
       deterministic tightening scales on the global TIGHTEN_SCALE_STEP grid
       (anchored at 1.0) above the baseline scale, capped at
       MAX_TIGHTEN_SCALE. A candidate scale is feasible only if EVERY
       detection's subject safe box still fits the tightened crop (head +
       shoulders + meaningful torso preserved; no edge clipping). The first
       (smallest) feasible scale reaching TIGHTEN_TARGET_OCCUPANCY is chosen
       — minimal tightening that achieves the target. If no feasible scale
       reaches the target, the largest feasible scale is kept (bounded,
       moderate). Gains below TIGHTEN_MIN_SCALE_GAIN are negligible: keep
       the existing composition.
    4. Stability: the decision is made ONCE per panel solve from the MEDIAN
       body width (never per keyframe — this is not a second camera
       tracker). Quantized scale steps + the enter/target gap provide
       hysteresis against CV noise, so tiny person-box jitter cannot zoom
       or oscillate the crop.

    Returns (tighten_scale, report); tighten_scale is None when the panel
    must keep its existing composition.
    """
    report: Dict[str, Any] = {
        "reason": None,
        "base_scale": round(float(source_h) / float(base_crop_h), 3) if base_crop_h > 0 else 1.0,
        "tighten_scale": None,
        "occupancy": None,
        "body_w": None,
    }

    if not track:
        report["reason"] = "no_track"
        return None, report

    detections = track.get("detections", [])
    if not detections:
        report["reason"] = "no_detections"
        return None, report

    # 1. Body evidence: per-detection person_bbox, falling back to the
    #    track-level person_bbox (production Ultralytics tracks carry both).
    body_widths: List[float] = []
    for det in detections:
        face = det[1] if len(det) > 1 and isinstance(det[1], dict) else None
        if not face:
            continue
        pb = face.get("person_bbox")
        if pb is not None and len(pb) >= 4:
            body_widths.append(max(0.0, float(pb[2]) - float(pb[0])))
    if not body_widths:
        tb = track.get("person_bbox")
        if tb is not None and len(tb) >= 4:
            body_widths.append(max(0.0, float(tb[2]) - float(tb[0])))
    if not body_widths:
        report["reason"] = "no_body_evidence"
        return None, report

    body_w = float(np.median(body_widths))
    report["body_w"] = round(body_w, 1)

    if base_crop_w <= 0:
        report["reason"] = "invalid_base_crop"
        return None, report

    base_scale = float(source_h) / float(base_crop_h) if base_crop_h > 0 else 1.0
    occupancy = body_w / float(base_crop_w)
    report["occupancy"] = round(occupancy, 4)

    # 2. Already well composed → zero tightening (true pass-through).
    if occupancy >= TIGHTEN_ENTER_OCCUPANCY:
        report["reason"] = "well_framed"
        return None, report

    # 3. Background-heavy → moderate tightening search on the global grid.
    #    At EVERY detection the tightened crop must still contain the
    #    subject's full safe box (head + shoulders + meaningful torso):
    #    horizontally via the production containment check, vertically via
    #    the production solve_crop_y placement (eye-line ~30% + standard
    #    headroom) — the same framing metadata the normal solver uses.
    face_boxes = []
    for det in detections:
        face = det[1] if len(det) > 1 and isinstance(det[1], dict) else None
        if not face or "bbox" not in face:
            continue
        face_boxes.append((face, compute_subject_safe_box(face, source_w, source_h)))
    if not face_boxes:
        report["reason"] = "no_face_geometry"
        return None, report

    def feasible_at_scale(s: float) -> Tuple[bool, int]:
        cw, ch = compute_panel_crop_dimensions(
            subject_box=float(s),
            source_w=source_w,
            source_h=source_h,
        )
        for face, sb in face_boxes:
            if not _panel_crop_fits_subject(float(cw), float(ch), sb, source_w, source_h):
                return False, cw
            # Vertical: the actual production placement must keep the full
            # safe box (head .. meaningful-torso bottom) inside the crop.
            y = solve_crop_y(face, float(ch), float(source_h))
            if sb["bottom"] > y + float(ch) + 2.0:
                return False, cw
            if sb["top"] < y - 2.0:
                return False, cw
        return True, cw

    # Global grid anchored at 1.0: first candidate strictly above base_scale.
    first_step = math.floor(base_scale / TIGHTEN_SCALE_STEP) * TIGHTEN_SCALE_STEP + TIGHTEN_SCALE_STEP
    first_step = min(first_step, MAX_TIGHTEN_SCALE)

    chosen_scale = None
    chosen_w = None
    best_feasible_scale = None
    best_feasible_w = None
    s = first_step
    while s <= MAX_TIGHTEN_SCALE + 1e-9:
        ok, cw = feasible_at_scale(s)
        if ok:
            occ = body_w / float(cw)
            best_feasible_scale = s
            best_feasible_w = cw
            if occ >= TIGHTEN_TARGET_OCCUPANCY:
                chosen_scale = s
                chosen_w = cw
                break
        s += TIGHTEN_SCALE_STEP

    if chosen_scale is None:
        # No feasible scale reached the target: keep the largest feasible
        # bounded tightening (moderate, never past MAX_TIGHTEN_SCALE).
        chosen_scale = best_feasible_scale
        chosen_w = best_feasible_w

    if chosen_scale is None:
        report["reason"] = "infeasible"
        return None, report

    # Negligible gain: keep the existing composition.
    if chosen_scale - base_scale < TIGHTEN_MIN_SCALE_GAIN:
        report["reason"] = "negligible_gain"
        return None, report

    report["reason"] = "background_heavy"
    report["tighten_scale"] = round(float(chosen_scale), 3)
    report["tighten_crop_w"] = int(chosen_w)
    return float(chosen_scale), report


def _solve_single_panel_trajectory(
    track: Optional[dict],
    source_w: float,
    source_h: float,
    start_sec: float = 0.0,
    panel_role: str = "top",
    other_track: Optional[dict] = None,
) -> CropRectExpr:
    """
    Solves trajectory and crop expressions for a single 9:8 panel.
    When other_track is provided (DualFrame close-speaker isolation), the
    panel's crop scale is first raised via _solve_panel_isolation_scale so the
    projected crop excludes the other speaker wherever geometrically possible.
    """
    if not track:
        fallback_cx = source_w * 0.30 if panel_role == "top" else source_w * 0.70
        fallback_cy = source_h * 0.35
        crop_w, crop_h = compute_panel_crop_dimensions(None, source_w, source_h)
        max_legal_x = max(0, int(source_w - crop_w))
        max_legal_y = max(0, int(source_h - crop_h))
        fx = int(round(np.clip(fallback_cx - crop_w / 2.0, 0.0, float(max_legal_x))))
        fy = int(round(np.clip(fallback_cy - crop_h * 0.30, 0.0, float(max_legal_y))))
        return CropRectExpr(x=str(fx), y=str(fy), w=str(crop_w), h=str(crop_h))

    detections = track.get("detections", [])
    rep_face = None
    if detections:
        latest_det = detections[-1]
        if len(latest_det) > 1 and isinstance(latest_det[1], dict):
            rep_face = latest_det[1]
    elif "face" in track and isinstance(track["face"], dict):
        rep_face = track["face"]

    # Close-speaker isolation: raise the panel scale (pinch-zoom) if the
    # projected baseline crop would contain the other speaker. No-op when the
    # other track is absent or speakers are sufficiently separated.
    isolation_scale = None
    isolation_active = False
    if other_track is not None:
        base_crop_w, base_crop_h = compute_panel_crop_dimensions(
            subject_box=rep_face or track,
            source_w=source_w,
            source_h=source_h,
        )
        base_scale = (float(source_h) / float(base_crop_h)) if base_crop_h > 0 else 1.0
        isolation_scale, iso_report = _solve_panel_isolation_scale(
            own_track=track,
            other_track=other_track,
            base_scale=base_scale,
            source_w=source_w,
            source_h=source_h,
        )
        isolation_active = bool(iso_report.get("isolation_active"))
        if iso_report.get("isolation_active"):
            sys.stderr.write(
                "[DualFrame] close-speaker isolation {}: base_scale={:.2f} -> isolation_scale={:.2f} "
                "paired={} contaminated={} fallback={}\n".format(
                    panel_role,
                    float(iso_report.get("base_scale", base_scale)),
                    float(iso_report.get("isolation_scale", base_scale)),
                    int(iso_report.get("paired_moments", 0)),
                    int(iso_report.get("contaminated_moments", 0)),
                    bool(iso_report.get("fallback", False)),
                )
            )
            sys.stderr.flush()

    if isolation_scale is not None and isolation_scale > 1.0 + 1e-9:
        crop_w, crop_h = compute_panel_crop_dimensions(
            subject_box=float(isolation_scale),
            source_w=source_w,
            source_h=source_h,
        )
    else:
        crop_w, crop_h = compute_panel_crop_dimensions(
            subject_box=rep_face or track,
            source_w=source_w,
            source_h=source_h,
        )

    # ─── Conditional subject-tight framing (§27, normal panels only) ──────
    # Applies ONLY when close-speaker isolation is NOT active: isolation is
    # the sole authority under cross-speaker contamination and tightening
    # never stacks on top of it. Already well-composed panels keep their
    # composition exactly (zero tightening).
    if not isolation_active:
        base_crop_w, base_crop_h = compute_panel_crop_dimensions(
            subject_box=rep_face or track,
            source_w=source_w,
            source_h=source_h,
        )
        tighten_scale, tighten_report = _solve_panel_composition_tightening(
            track=track,
            base_crop_w=base_crop_w,
            base_crop_h=base_crop_h,
            source_w=source_w,
            source_h=source_h,
            panel_role=panel_role,
        )
        if tighten_scale is not None:
            crop_w, crop_h = compute_panel_crop_dimensions(
                subject_box=float(tighten_scale),
                source_w=source_w,
                source_h=source_h,
            )
            sys.stderr.write(
                "[DualFrame] composition tightening {}: base_scale={:.2f} -> tight_scale={:.2f} "
                "occupancy={:.3f} -> target={:.2f} body_w={:.0f}px\n".format(
                    panel_role,
                    float(tighten_report.get("base_scale", 1.0)),
                    float(tighten_scale),
                    float(tighten_report.get("occupancy", 0.0)),
                    float(TIGHTEN_TARGET_OCCUPANCY),
                    float(tighten_report.get("body_w", 0.0)),
                )
            )
            sys.stderr.flush()

    max_legal_x = max(0, int(source_w - crop_w))
    max_legal_y = max(0, int(source_h - crop_h))

    if not detections or len(detections) <= 1:
        cx = track.get("cx")
        cy = track.get("cy", source_h / 2.0)
        if rep_face and "center" in rep_face:
            cx = rep_face["center"][0]
            cy = rep_face["center"][1]
        if cx is None:
            cx = source_w * 0.30 if panel_role == "top" else source_w * 0.70
        if cy is None:
            cy = source_h / 2.0

        if rep_face and "bbox" in rep_face:
            subject_safe_box = compute_subject_safe_box(rep_face, source_w, source_h)
            final_x = _solve_isolation_aware_crop_x(
                rep_face, other_track, 0.0,
                float(crop_w), float(max_legal_x), float(source_w), float(source_h),
            )
            final_y = solve_crop_y(rep_face, float(crop_h), float(source_h))
        else:
            final_x = float(np.clip(cx - crop_w / 2.0, 0.0, float(max_legal_x)))
            final_y = float(np.clip(cy - crop_h * 0.30, 0.0, float(max_legal_y)))

        clamped_x = int(round(np.clip(final_x, 0.0, float(max_legal_x))))
        clamped_y = int(round(np.clip(final_y, 0.0, float(max_legal_y))))
        return CropRectExpr(
            x=str(clamped_x),
            y=str(clamped_y),
            w=str(crop_w),
            h=str(crop_h),
        )

    # Multiple detections over time: build position keyframes
    x_keyframes = []
    y_keyframes = []
    curr_crop_x = None
    curr_crop_y = None
    # Per-panel isolation hysteresis state: carries the last position and
    # the mode that produced it ("clean" / "constrained" / "release" /
    # "fallback") across keyframes so jitter below
    # ISOLATION_BOUNDARY_HYSTERESIS_PX cannot move the camera and the
    # constrained→fallback handoff stays continuous. Fresh state per panel
    # solve (symmetric for top and bottom panels).
    iso_state = {"last_x": None, "last_mode": None}

    for det in detections:
        t_sec = det[0]
        face_det = det[1] if len(det) > 1 and isinstance(det[1], dict) else None
        if face_det and "bbox" in face_det:
            pos_x = _solve_isolation_aware_crop_x(
                face_det, other_track, t_sec,
                float(crop_w), float(max_legal_x), float(source_w), float(source_h),
                current_crop_x=curr_crop_x,
                iso_state=iso_state,
            )
            pos_y = solve_crop_y(face_det, float(crop_h), float(source_h), current_crop_y=curr_crop_y)
        else:
            if isinstance(det[1], dict) and "center" in det[1]:
                cx, cy = det[1]["center"]
            else:
                cx = det[1].get("cx") if isinstance(det[1], dict) else None
                cy = det[1].get("cy") if isinstance(det[1], dict) else None
            if cx is None:
                cx = track.get("cx")
            if cy is None:
                cy = track.get("cy")
            if cx is None:
                cx = source_w * 0.30 if panel_role == "top" else source_w * 0.70
            if cy is None:
                cy = source_h / 2.0
            pos_x = float(np.clip(cx - crop_w / 2.0, 0.0, float(max_legal_x)))
            pos_y = float(np.clip(cy - crop_h * 0.30, 0.0, float(max_legal_y)))

        abs_t = start_sec + t_sec
        x_keyframes.append((abs_t, pos_x, False))
        y_keyframes.append((abs_t, pos_y, False))
        curr_crop_x = pos_x
        curr_crop_y = pos_y

    xs = [kf[1] for kf in x_keyframes]
    ys = [kf[1] for kf in y_keyframes]

    if max(xs) - min(xs) < 3.0:
        x_expr = str(int(round(np.clip(xs[-1], 0.0, float(max_legal_x)))))
    else:
        ckf = [x_keyframes[0]]
        for kf in x_keyframes[1:]:
            prev = ckf[-1]
            dt = kf[0] - prev[0]
            dx = abs(kf[1] - prev[1])
            if dt < 0.05:
                ckf[-1] = kf
            elif dx >= 15.0 or (dt >= 0.50 and dx >= 5.0):
                ckf.append(kf)
        if len(ckf) == 1 and x_keyframes:
            ckf.append(x_keyframes[-1])
        opt_kf = optimize_camera_trajectory(ckf, float(max_legal_x))
        x_expr = build_ffmpeg_expr(opt_kf, start_sec, default_x=xs[0], max_legal_x=max_legal_x)

    if max(ys) - min(ys) < 3.0:
        y_expr = str(int(round(np.clip(ys[-1], 0.0, float(max_legal_y)))))
    else:
        y_expr = build_ffmpeg_y_expr(y_keyframes, start_sec, source_h=source_h, max_legal_y=max_legal_y)

    return CropRectExpr(
        x=x_expr,
        y=y_expr,
        w=str(crop_w),
        h=str(crop_h),
    )


def solve_dual_frame_trajectories(
    top_track: Optional[dict],
    bottom_track: Optional[dict],
    source_w: float = 1920.0,
    source_h: float = 1080.0,
    start_sec: float = 0.0,
    **kwargs,
) -> DualFrameTrajectories:
    """
    Generates independent 9:8 panel crop trajectory expressions for top and bottom subjects.
    Enforces:
    1. Strict 9:8 aspect ratio per panel.
    2. Independent horizontal and vertical camera trajectories centered on each subject.
    3. Eye-line positioning (~30% from top) and headroom preservation.
    4. Strict clamping within legal frame boundaries (0 <= x <= source_w - crop_w, 0 <= y <= source_h - crop_h).
    5. Close-speaker isolation: each panel's crop scale is raised (pinch-zoom)
       when the projected baseline crop would contain the OTHER speaker, so
       each panel shows only its intended speaker wherever geometrically
       possible. Isolation is decided by projected crop-vs-other-subject
       contamination, never raw center distance.
    """
    top_crop = _solve_single_panel_trajectory(
        top_track,
        source_w=source_w,
        source_h=source_h,
        start_sec=start_sec,
        panel_role="top",
        other_track=bottom_track,
    )
    bottom_crop = _solve_single_panel_trajectory(
        bottom_track,
        source_w=source_w,
        source_h=source_h,
        start_sec=start_sec,
        panel_role="bottom",
        other_track=top_track,
    )
    return DualFrameTrajectories(top_crop=top_crop, bottom_crop=bottom_crop)


def derive_single_segment_crop(
    val_kf: list,
    y_keyframes: list,
    seg_start: float,
    seg_end: float,
    start_sec: float,
    default_x: float,
    max_legal_x: int,
    max_legal_y: int,
    final_crop_w: int,
    final_crop_h: int,
    x_expr: str,
    y_expr: str,
    w_expr: str,
    h_expr: str,
    clip_dur: float,
    source_h: float = 1080.0,
) -> CropRectExpr:
    if seg_start <= 0.001 and seg_end >= clip_dur - 0.01:
        return CropRectExpr(x=x_expr, y=y_expr, w=w_expr, h=h_expr)

    # 1. Determine X expression for segment
    if not val_kf:
        seg_x = str(int(max(0, min(max_legal_x, round(default_x)))))
    else:
        base_kf = val_kf[0]
        for kf in val_kf:
            kf_rel = kf[0] - start_sec
            if kf_rel <= seg_start + 0.001:
                base_kf = kf
            else:
                break
        in_seg_kfs = [kf for kf in val_kf if seg_start < (kf[0] - start_sec) <= seg_end]
        seg_kfs = [base_kf] + in_seg_kfs

        xs = [kf[1] for kf in seg_kfs]
        if len(seg_kfs) == 1 or max(xs) - min(xs) < 1.0:
            seg_x = str(int(max(0, min(max_legal_x, round(seg_kfs[0][1])))))
        else:
            seg_x = build_ffmpeg_expr(seg_kfs, start_sec + seg_start, default_x=seg_kfs[0][1], max_legal_x=max_legal_x)

    # 2. Determine Y expression for segment
    if not y_keyframes or max_legal_y <= 0:
        seg_y = "0"
    else:
        base_ykf = y_keyframes[0]
        for ykf in y_keyframes:
            ykf_rel = ykf[0] - start_sec
            if ykf_rel <= seg_start + 0.001:
                base_ykf = ykf
            else:
                break
        in_seg_ykfs = [ykf for ykf in y_keyframes if seg_start < (ykf[0] - start_sec) <= seg_end]
        seg_ykfs = [base_ykf] + in_seg_ykfs

        ys = [ykf[1] for ykf in seg_ykfs]
        if len(seg_ykfs) == 1 or max(ys) - min(ys) < 2.0:
            seg_y = str(int(max(0, min(max_legal_y, round(seg_ykfs[0][1])))))
        else:
            seg_y = build_ffmpeg_y_expr(seg_ykfs, start_sec + seg_start, source_h=source_h, max_legal_y=max_legal_y)

    return CropRectExpr(
        x=seg_x,
        y=seg_y,
        w=str(final_crop_w),
        h=str(final_crop_h),
    )


def compute_single_segment_face_bounds(
    timeline_decisions: list,
    s_start: float,
    s_end: float,
    adaptive_framing: bool = False,
) -> Optional[Dict[str, float]]:
    """
    Computes normalized face bounds [0.0, 1.0] relative to the final 1080x1920 single 9:16 canvas,
    scoped STRICTLY to face detections within the segment time interval [s_start, s_end].
    Does NOT mix face coordinates across shots or other segments.

    In Adaptive Framing the crop maps to a full-width SQUARE video vertically
    centered in the canvas (canvas y = 420..1500 of 1920), so crop-relative
    vertical coordinates are remapped into final canvas coordinates; the
    horizontal mapping is unchanged (the crop spans the full canvas width).
    """
    all_bounds = []
    for step in timeline_decisions:
        st_start = step.get("start", 0.0)
        st_end = step.get("end", 0.0)
        if st_end <= s_start or st_start >= s_end:
            continue
        face_bbox = step.get("face_bbox")
        single_crop = step.get("single_crop")
        if face_bbox is not None and single_crop is not None:
            try:
                fx, fy, fw, fh = [float(v) for v in face_bbox]
                clamped_x, crop_y, eff_crop_w, eff_crop_h = single_crop
                cur_cw = max(1.0, float(eff_crop_w))
                cur_ch = max(1.0, float(eff_crop_h))
                cur_cx = float(clamped_x)
                cur_cy = float(crop_y)
                ntop = max(0.0, min(1.0, (fy - cur_cy) / cur_ch))
                nbot = max(0.0, min(1.0, (fy + fh - cur_cy) / cur_ch))
                nleft = max(0.0, min(1.0, (fx - cur_cx) / cur_cw))
                nright = max(0.0, min(1.0, (fx + fw - cur_cx) / cur_cw))
                if adaptive_framing:
                    # crop square -> canvas: full width, y in [420, 1500] of 1920
                    ntop = max(0.0, min(1.0, (ntop * 1080.0 + 420.0) / 1920.0))
                    nbot = max(0.0, min(1.0, (nbot * 1080.0 + 420.0) / 1920.0))
                all_bounds.append((ntop, nbot, nleft, nright))
            except Exception:
                pass

    if not all_bounds:
        return None

    try:
        tops = [b[0] for b in all_bounds]
        bots = [b[1] for b in all_bounds]
        lefts = [b[2] for b in all_bounds]
        rights = [b[3] for b in all_bounds]
        return {
            "top": round(float(np.percentile(tops, 10)), 4),
            "bottom": round(float(np.percentile(bots, 98)), 4),
            "left": round(float(np.percentile(lefts, 10)), 4),
            "right": round(float(np.percentile(rights, 90)), 4),
        }
    except Exception:
        return None


def compute_dual_segment_face_bounds(
    track: Optional[dict],
    s_start: float,
    s_end: float,
    panel_crop: Any,
    source_w: float,
    source_h: float,
    is_bottom_panel: bool = False,
) -> Optional[Dict[str, float]]:
    """
    Computes normalized face bounds [0.0, 1.0] relative to the final 1080x1920 stacked canvas
    for a single speaker in a DualFrame panel:
    - Top panel: maps to Y in [0.0, 0.50] (0..960px)
    - Bottom panel: maps to Y in [0.50, 1.00] (960..1920px with +0.50 vstack offset)
    Scoped STRICTLY to face detections within the segment time interval [s_start, s_end].
    """
    if not track or not isinstance(track, dict):
        return None

    dets = track.get("detections", [])
    if not dets:
        return None

    try:
        crop_w = float(panel_crop.w if hasattr(panel_crop, "w") else panel_crop.get("w", 1216.0))
        crop_h = float(panel_crop.h if hasattr(panel_crop, "h") else panel_crop.get("h", 1080.0))
    except Exception:
        crop_w, crop_h = 1216.0, 1080.0

    try:
        crop_x_val = panel_crop.x if hasattr(panel_crop, "x") else panel_crop.get("x", "0")
        if isinstance(crop_x_val, str) and not crop_x_val.isdigit() and not crop_x_val.replace('.', '', 1).isdigit():
            crop_x_base = None
        else:
            crop_x_base = float(crop_x_val)
    except Exception:
        crop_x_base = None

    try:
        crop_y_val = panel_crop.y if hasattr(panel_crop, "y") else panel_crop.get("y", "0")
        if isinstance(crop_y_val, str) and not crop_y_val.isdigit() and not crop_y_val.replace('.', '', 1).isdigit():
            crop_y_base = None
        else:
            crop_y_base = float(crop_y_val)
    except Exception:
        crop_y_base = None

    all_bounds = []
    for det in dets:
        t_rel = det[0]
        if t_rel < s_start - 0.25 or t_rel > s_end + 0.25:
            continue
        face_det = det[1] if len(det) > 1 and isinstance(det[1], dict) else None
        if not face_det or "bbox" not in face_det:
            continue

        try:
            fx, fy, fw, fh = [float(v) for v in face_det["bbox"]]
            fcx = fx + fw / 2.0
            fcy = fy + fh / 2.0

            if crop_x_base is not None:
                cx_origin = crop_x_base
            else:
                cx_origin = float(np.clip(fcx - crop_w / 2.0, 0.0, max(0.0, source_w - crop_w)))

            if crop_y_base is not None:
                cy_origin = crop_y_base
            else:
                cy_origin = float(np.clip(fcy - crop_h * 0.30, 0.0, max(0.0, source_h - crop_h)))

            x_rel = fx - cx_origin
            y_rel = fy - cy_origin

            nleft = max(0.0, min(1.0, x_rel / max(1.0, crop_w)))
            nright = max(0.0, min(1.0, (x_rel + fw) / max(1.0, crop_w)))

            if not is_bottom_panel:
                ntop = max(0.0, min(0.50, y_rel / (2.0 * max(1.0, crop_h))))
                nbot = max(0.0, min(0.50, (y_rel + fh) / (2.0 * max(1.0, crop_h))))
            else:
                ntop = max(0.50, min(1.00, 0.50 + y_rel / (2.0 * max(1.0, crop_h))))
                nbot = max(0.50, min(1.00, 0.50 + (y_rel + fh) / (2.0 * max(1.0, crop_h))))

            all_bounds.append((ntop, nbot, nleft, nright))
        except Exception:
            pass

    if not all_bounds:
        return None

    try:
        tops = [b[0] for b in all_bounds]
        bots = [b[1] for b in all_bounds]
        lefts = [b[2] for b in all_bounds]
        rights = [b[3] for b in all_bounds]
        return {
            "top": round(float(np.percentile(tops, 10)), 4),
            "bottom": round(float(np.percentile(bots, 90)), 4),
            "left": round(float(np.percentile(lefts, 10)), 4),
            "right": round(float(np.percentile(rights, 90)), 4),
        }
    except Exception:
        return None


def build_layout_plan(
    timeline_decisions: list,
    clip_dur: float,
    start_sec: float,
    source_w: float,
    source_h: float,
    crop_w_baseline: float,
    val_kf: list,
    y_keyframes: list,
    default_x: float,
    max_legal_x: int,
    max_legal_y: int,
    final_crop_w: int,
    final_crop_h: int,
    x_expr: str,
    y_expr: str,
    w_expr: str,
    h_expr: str,
    config: Optional[DualFrameConfig] = None,
    face_bounds: Optional[Dict[str, float]] = None,
    framing: str = "original",
) -> SmartFramingPlan:
    """
    Builds the full SmartFramingPlan JSON payload from temporal decisions.
    Enforces:
    1. Backward-compatible root fallback parameters (x, y, w, h).
    2. Zero-gap contiguous coverage from 0.0 to clip_dur.
    3. Proper single vs dual_stack layout segment boundaries.
    4. Anti-flicker minimum duration enforcement (demotes short dual segments < min_segment_dur_sec).
    5. Structured telemetry reporting to stderr.
    """
    cfg = config if config is not None else DualFrameConfig()

    if not timeline_decisions:
        seg_face_b = compute_single_segment_face_bounds(
            timeline_decisions, 0.0, clip_dur, adaptive_framing=framing == "adaptive",
        ) or face_bounds
        single_seg = LayoutSegment(
            mode="single",
            start=0.0,
            end=round(clip_dur, 3),
            crop=CropRectExpr(x=x_expr, y=y_expr, w=w_expr, h=h_expr),
            face_bounds=seg_face_b,
        )
        plan = SmartFramingPlan(
            mode="single",
            x=x_expr,
            y=y_expr,
            w=w_expr,
            h=h_expr,
            segments=[single_seg],
            face_bounds=seg_face_b,
            framing=framing,
        )
        sys.stderr.write("[DualFrame] layout_plan total_segments=1 dual_segments=0\n")
        sys.stderr.flush()
        return plan

    if not cfg.enabled:
        # DualFrame is disabled (e.g. in Adaptive Framing).
        # Preserve shot cuts as individual LayoutSegment::Single segments with their own face_bounds.
        shot_groups = []
        for step in timeline_decisions:
            s_idx = step.get("shot_idx")
            if not shot_groups or (s_idx is not None and shot_groups[-1]["shot_idx"] != s_idx):
                shot_groups.append({
                    "shot_idx": s_idx,
                    "start": step["start"],
                    "end": step["end"],
                })
            else:
                shot_groups[-1]["end"] = step["end"]

        if shot_groups:
            shot_groups[0]["start"] = 0.0
            for i in range(len(shot_groups) - 1):
                shot_groups[i + 1]["start"] = shot_groups[i]["end"]
            shot_groups[-1]["end"] = clip_dur

        segments = []
        for g in shot_groups:
            s_start = g["start"]
            s_end = g["end"]
            crop = derive_single_segment_crop(
                val_kf=val_kf,
                y_keyframes=y_keyframes,
                seg_start=s_start,
                seg_end=s_end,
                start_sec=start_sec,
                default_x=default_x,
                max_legal_x=max_legal_x,
                max_legal_y=max_legal_y,
                final_crop_w=final_crop_w,
                final_crop_h=final_crop_h,
                x_expr=x_expr,
                y_expr=y_expr,
                w_expr=w_expr,
                h_expr=h_expr,
                clip_dur=clip_dur,
                source_h=source_h,
            )
            seg_fb = compute_single_segment_face_bounds(
                timeline_decisions, s_start, s_end, adaptive_framing=framing == "adaptive"
            )
            segments.append(LayoutSegment(
                mode="single",
                start=s_start,
                end=s_end,
                crop=crop,
                face_bounds=seg_fb,
            ))

        legacy_face_bounds = None
        for s in segments:
            if s.face_bounds is not None:
                legacy_face_bounds = s.face_bounds
                break
        if legacy_face_bounds is None:
            legacy_face_bounds = face_bounds

        plan = SmartFramingPlan(
            mode="single",
            x=x_expr,
            y=y_expr,
            w=w_expr,
            h=h_expr,
            segments=segments,
            face_bounds=legacy_face_bounds,
            framing=framing,
        )
        total_segments = len(segments)
        sys.stderr.write(f"[DualFrame] layout_plan total_segments={total_segments} dual_segments=0\n")
        sys.stderr.flush()
        return plan

    # 1. Group continuous decisions into raw runs
    raw_runs = []
    for step in timeline_decisions:
        sub_s = step["start"]
        sub_e = step["end"]
        dec = step["decision"]
        mode = dec.mode if cfg.enabled else "single"
        top_id = dec.top_track_id if (mode == "dual_stack" and cfg.enabled) else None
        bot_id = dec.bottom_track_id if (mode == "dual_stack" and cfg.enabled) else None
        tracks = step.get("tracks", {})

        if not raw_runs:
            raw_runs.append({
                "mode": mode,
                "start": sub_s,
                "end": sub_e,
                "top_track_id": top_id,
                "bottom_track_id": bot_id,
                "tracks": tracks,
            })
        else:
            prev = raw_runs[-1]
            same_mode = (prev["mode"] == mode)
            same_slots = (mode != "dual_stack" or (prev["top_track_id"] == top_id and prev["bottom_track_id"] == bot_id))
            if same_mode and same_slots:
                prev["end"] = sub_e
                if tracks and not prev.get("tracks"):
                    prev["tracks"] = tracks
            else:
                raw_runs.append({
                    "mode": mode,
                    "start": sub_s,
                    "end": sub_e,
                    "top_track_id": top_id,
                    "bottom_track_id": bot_id,
                    "tracks": tracks,
                })

    # 2. Continuous boundary coverage
    if raw_runs:
        raw_runs[0]["start"] = 0.0
        for i in range(len(raw_runs) - 1):
            raw_runs[i + 1]["start"] = raw_runs[i]["end"]
        raw_runs[-1]["end"] = clip_dur

    # 3. Minimum segment duration hysteresis (demote short dual runs to single)
    min_dur = cfg.min_segment_dur_sec
    filtered_runs = []
    for r in raw_runs:
        dur = r["end"] - r["start"]
        if r["mode"] == "dual_stack" and dur < min_dur and len(raw_runs) > 1:
            r["mode"] = "single"
            r["top_track_id"] = None
            r["bottom_track_id"] = None
        filtered_runs.append(r)

    # 4. Merge adjacent runs with same mode and slots
    merged_runs = []
    for r in filtered_runs:
        if not merged_runs:
            merged_runs.append(r)
        else:
            prev = merged_runs[-1]
            same_mode = (prev["mode"] == r["mode"])
            same_slots = (r["mode"] != "dual_stack" or (prev["top_track_id"] == r["top_track_id"] and prev["bottom_track_id"] == r["bottom_track_id"]))
            if same_mode and same_slots:
                prev["end"] = r["end"]
                if r.get("tracks") and not prev.get("tracks"):
                    prev["tracks"] = r["tracks"]
            else:
                merged_runs.append(r)

    if merged_runs:
        merged_runs[0]["start"] = 0.0
        for i in range(len(merged_runs) - 1):
            merged_runs[i + 1]["start"] = merged_runs[i]["end"]
        merged_runs[-1]["end"] = clip_dur

    # 5. Build segment objects with crop trajectory expressions
    segments = []
    for r in merged_runs:
        s_start = r["start"]
        s_end = r["end"]
        if r["mode"] == "single":
            crop = derive_single_segment_crop(
                val_kf=val_kf,
                y_keyframes=y_keyframes,
                seg_start=s_start,
                seg_end=s_end,
                start_sec=start_sec,
                default_x=default_x,
                max_legal_x=max_legal_x,
                max_legal_y=max_legal_y,
                final_crop_w=final_crop_w,
                final_crop_h=final_crop_h,
                x_expr=x_expr,
                y_expr=y_expr,
                w_expr=w_expr,
                h_expr=h_expr,
                clip_dur=clip_dur,
                source_h=source_h,
            )
            seg_fb = compute_single_segment_face_bounds(
                timeline_decisions, s_start, s_end, adaptive_framing=framing == "adaptive"
            )
            segments.append(LayoutSegment(
                mode="single",
                start=s_start,
                end=s_end,
                crop=crop,
                face_bounds=seg_fb,
            ))
        else:
            tracks = r.get("tracks", {})
            top_id = r.get("top_track_id")
            bot_id = r.get("bottom_track_id")
            top_tr = tracks.get(top_id) if top_id is not None else None
            bot_tr = tracks.get(bot_id) if bot_id is not None else None

            # Enforce architectural invariant:
            # 1. Exactly two distinct tracks: top_id != bot_id and neither is None
            # 2. Both tracks must exist in current shot tracks
            if top_id is None or bot_id is None or top_id == bot_id or top_tr is None or bot_tr is None:
                sys.stderr.write(
                    f"[DualFrame] Demoting segment [{s_start:.2f}-{s_end:.2f}] to single: invalid or missing track pair (top={top_id}, bot={bot_id})\n"
                )
                crop = derive_single_segment_crop(
                    val_kf=val_kf,
                    y_keyframes=y_keyframes,
                    seg_start=s_start,
                    seg_end=s_end,
                    start_sec=start_sec,
                    default_x=default_x,
                    max_legal_x=max_legal_x,
                    max_legal_y=max_legal_y,
                    final_crop_w=final_crop_w,
                    final_crop_h=final_crop_h,
                    x_expr=x_expr,
                    y_expr=y_expr,
                    w_expr=w_expr,
                    h_expr=h_expr,
                    clip_dur=clip_dur,
                    source_h=source_h,
                )
                seg_fb = compute_single_segment_face_bounds(
                    timeline_decisions, s_start, s_end, adaptive_framing=framing == "adaptive"
                )
                segments.append(LayoutSegment(
                    mode="single",
                    start=s_start,
                    end=s_end,
                    crop=crop,
                    face_bounds=seg_fb,
                ))
                continue

            top_tr_slice = dict(top_tr) if top_tr else {}
            bot_tr_slice = dict(bot_tr) if bot_tr else {}
            if "detections" in top_tr_slice and top_tr_slice["detections"]:
                matching_dets = [
                    (d[0] - s_start, d[1]) + tuple(d[2:])
                    for d in top_tr_slice["detections"]
                    if s_start - 0.5 <= d[0] <= s_end + 0.5
                ]
                top_tr_slice["detections"] = matching_dets or [
                    (0.0, top_tr_slice["detections"][-1][1]) + tuple(top_tr_slice["detections"][-1][2:])
                ]
            if "detections" in bot_tr_slice and bot_tr_slice["detections"]:
                matching_dets = [
                    (d[0] - s_start, d[1]) + tuple(d[2:])
                    for d in bot_tr_slice["detections"]
                    if s_start - 0.5 <= d[0] <= s_end + 0.5
                ]
                bot_tr_slice["detections"] = matching_dets or [
                    (0.0, bot_tr_slice["detections"][-1][1]) + tuple(bot_tr_slice["detections"][-1][2:])
                ]

            dual_traj = solve_dual_frame_trajectories(
                top_track=top_tr_slice,
                bottom_track=bot_tr_slice,
                source_w=source_w,
                source_h=source_h,
                start_sec=0.0,
            )
            top_fb = compute_dual_segment_face_bounds(
                track=top_tr,
                s_start=s_start,
                s_end=s_end,
                panel_crop=dual_traj.top_crop,
                source_w=source_w,
                source_h=source_h,
                is_bottom_panel=False,
            )
            bot_fb = compute_dual_segment_face_bounds(
                track=bot_tr,
                s_start=s_start,
                s_end=s_end,
                panel_crop=dual_traj.bottom_crop,
                source_w=source_w,
                source_h=source_h,
                is_bottom_panel=True,
            )
            segments.append(LayoutSegment(
                mode="dual_stack",
                start=s_start,
                end=s_end,
                top_track_id=r["top_track_id"],
                bottom_track_id=r["bottom_track_id"],
                top_crop=dual_traj.top_crop,
                bottom_crop=dual_traj.bottom_crop,
                top_face_bounds=top_fb,
                bottom_face_bounds=bot_fb,
            ))

    has_dual = any(s.mode == "dual_stack" for s in segments)
    overall_mode = "dual_stack" if has_dual else "single"

    # Derive legacy root fallback face_bounds strictly from first single segment
    legacy_face_bounds = None
    for s in segments:
        if s.mode == "single" and s.face_bounds is not None:
            legacy_face_bounds = s.face_bounds
            break
    if legacy_face_bounds is None:
        legacy_face_bounds = face_bounds

    plan = SmartFramingPlan(
        mode=overall_mode,
        x=x_expr,
        y=y_expr,
        w=w_expr,
        h=h_expr,
        segments=segments,
        face_bounds=legacy_face_bounds,
        framing=framing,
    )

    total_segments = len(segments)
    dual_segments = sum(1 for s in segments if s.mode == "dual_stack")
    sys.stderr.write(f"[DualFrame] layout_plan total_segments={total_segments} dual_segments={dual_segments}\n")
    sys.stderr.flush()

    return plan


# ─── 12. Debug Visualization Overlay ─────────────────────────────────────────

def save_debug_visualization(frame, rel_t: float, diag: dict, output_dir: str):
    if not os.path.exists(output_dir):
        try: os.makedirs(output_dir, exist_ok=True)
        except Exception: return

    h, w = frame.shape[:2]
    dbg = frame.copy()
    cx_d = int(round(diag.get("clamped_x", 0)))
    cw_d = int(round(diag.get("crop_w", w * 9 // 16)))
    ch_d = int(round(diag.get("crop_h", h)))
    cy_d = int(round(diag.get("crop_y", 0)))
    sc = diag.get("crop_scale", 1.0)

    cv2.rectangle(dbg, (max(0, cx_d), max(0, cy_d)), (min(w, cx_d + cw_d), min(h, cy_d + ch_d)), (0, 255, 255), 3)

    sb = diag.get("subject_safe_box")
    if sb:
        cv2.rectangle(dbg, (int(sb["left"]), int(sb["top"])), (int(sb["right"]), int(sb["bottom"])), (255, 255, 0), 2)

    if "face_bbox" in diag and diag["face_bbox"]:
        fx, fy, fw, fh = [int(v) for v in diag["face_bbox"]]
        cv2.rectangle(dbg, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)

    cv2.rectangle(dbg, (10, 10), (640, 165), (0, 0, 0), -1)
    cont = "PASS" if diag.get("containment_pass") else "ADJUSTED"
    cv2.putText(dbg, f"AUTOSHORTS v12.0 | {cont}", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 0) if cont == "PASS" else (0, 0, 255), 2)
    cv2.putText(dbg, f"t={rel_t:.2f}s Spk={diag.get('speaker', '?')} Mode={diag.get('camera_mode', '?')}", (20, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(dbg, f"Shot={diag.get('shot_type', '?')} Scale={sc:.3f} [{diag.get('scale_change_reason', '?')}]", (20, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(dbg, f"X={cx_d} Y={cy_d} W={cw_d} H={ch_d}", (20, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1)
    cv2.putText(dbg, f"Score={diag.get('framing_score', 0):.2f}", (20, 114), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(dbg, f"ScaleRange=[{diag.get('scale_min', 0.8):.2f}-{diag.get('scale_max', 1.35):.2f}] ideal={diag.get('ideal_scale', 1.0):.2f}", (20, 134), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 255, 180), 1)

    cv2.imwrite(os.path.join(output_dir, f"debug_frame_t{rel_t:.2f}s.jpg"), dbg)


# ─── 13. Main Pipeline Execution ──────────────────────────────────────────────

def run(
    source_path: str, start_ms: float, end_ms: float,
    crop_w: float, max_x: float, default_x: float,
    transcript_words=None, debug_vis_dir=None,
    config: Optional[FramingConfig] = None,
    diarization_json_path: Optional[str] = None,
    gallery_json_path: Optional[str] = None,
    scene_cuts_json_path: Optional[str] = None,
):
    cfg = config or DEFAULT_FRAMING_CONFIG
    start_sec = start_ms / 1000.0
    end_sec = end_ms / 1000.0
    clip_dur = max(0.1, end_sec - start_sec)

    # Adaptive Framing (10.0): per-project framing mode injected by the Rust
    # caller via the environment. "adaptive" disables DualFrame (never
    # split-screen) and relaxes the two-shot fit threshold so two people who
    # fit comfortably stay in one natural composition; the final canvas stays
    # 9:16. Any other value (or unset) preserves the original behavior.
    adaptive_framing = os.environ.get("AUTOSHORTS_FRAMING_MODE", "original").strip().lower() == "adaptive"
    # [Framing] START + heartbeat for long tracking runs
    sys.stderr.write(f"[Framing] START adaptive={adaptive_framing} clip_dur={clip_dur:.1f}s\n")
    sys.stderr.flush()
    last_heartbeat = time.time()

    # Reset recorded geometry so a mocked/absent sampler cannot leak stale
    # geometry from a previous real invocation (tests patch the sampler).
    global _SOURCE_GEOMETRY
    _SOURCE_GEOMETRY = None

    scene_cuts, all_samples = sample_video_and_detect_shots(source_path, start_ms, end_ms)

    # ── Phase 3: merge cached source-level scene boundaries (advisory) ─────
    # The tracker's own pixel-diff detector stays authoritative for anything
    # it found; PySceneDetect boundaries only ADD cut points (same debounce).
    if scene_cuts_json_path and scene_intelligence_enabled_py():
        scene_cuts = load_scene_cuts_json(scene_cuts_json_path, start_sec, clip_dur, scene_cuts)

    # True source geometry: recorded by the sampler when it actually decoded
    # frames (stored frames may be memory-budget downscaled). Falls back to
    # the first stored frame's shape when the sampler was mocked (tests).
    source_h, source_w = 1080.0, 1920.0
    stored_ratio = 1.0  # stored frame width / true source width
    if _SOURCE_GEOMETRY is not None:
        source_w, source_h = float(_SOURCE_GEOMETRY[0]), float(_SOURCE_GEOMETRY[1])
        if all_samples and all_samples[0][1] is not None:
            stored_ratio = float(all_samples[0][1].shape[1]) / source_w
    elif all_samples:
        source_h = float(all_samples[0][1].shape[0])
        source_w = float(all_samples[0][1].shape[1])

    # Adaptive Framing composes a SQUARE inner video (side = min(source_h,
    # source_w)) rendered full-width and vertically centered inside the 9:16
    # canvas, mirroring the reference composition; Original 9:16 keeps the
    # widest true-9:16 crop (source_h*9/16, full height) stretched full-bleed.
    if adaptive_framing:
        crop_w_baseline = round(min(source_h, source_w))
    else:
        crop_w_baseline = round(source_h * 9.0 / 16.0)
    if crop_w_baseline % 2 != 0:
        crop_w_baseline -= 1

    if (not adaptive_framing) and crop_w_baseline >= source_w:
        plan = SmartFramingPlan(
            mode="single",
            x="0",
            y="0",
            w=str(int(source_w)),
            h=str(int(source_h)),
            segments=[
                LayoutSegment(
                    mode="single",
                    start=0.0,
                    end=round(clip_dur, 3),
                    crop=CropRectExpr(x="0", y="0", w=str(int(source_w)), h=str(int(source_h))),
                )
            ],
            framing="original",
        )
        sys.stderr.write("[DualFrame] mode=single\n")
        sys.stderr.write("[DualFrame] layout_plan total_segments=1 dual_segments=0\n")
        sys.stderr.flush()
        print(plan.to_json())
        return plan

    shots = []
    for i in range(len(scene_cuts)):
        shots.append((scene_cuts[i], scene_cuts[i + 1] if i + 1 < len(scene_cuts) else clip_dur))

    # ── Phase 3: per-scene motion advisory (Camera Lock tuning, advisory) ──
    # Static scenes (locked-off interview shots) tolerate slightly less
    # frequent reframing; dynamic scenes keep the historical responsiveness.
    # Bounded scaling of the EXISTING deterministic thresholds only.
    shot_motion = compute_shot_motion_profile(all_samples, shots) if scene_intelligence_enabled_py() else {}

    speaker_turns = build_speaker_turns(transcript_words, start_sec, end_sec)

    # ── Phase 2: Speaker Intelligence inputs (all optional, all fallback-safe) ─
    # Diarization evidence: cached source-level sidecar result when the Rust
    # engine provides one; otherwise derived from word-level speaker labels in
    # the transcript; otherwise empty (fusion then runs visual-only and the
    # deterministic heuristic remains authoritative).
    diar_segments = []
    if speaker_fusion_enabled() and _asf is not None:
        diar_segments = load_diarization_sidecar_json(diarization_json_path)
        if not diar_segments:
            diar_segments = build_diarization_segments_from_words(transcript_words or [], start_sec, end_sec)
    persisted_gallery = load_persisted_gallery_json(gallery_json_path) if speaker_reid_enabled() else {}
    try:
        fusion_interval_sec = float(os.environ.get("AUTOSHORTS_FUSION_INTERVAL_SEC", "1.0") or 1.0)
    except ValueError:
        fusion_interval_sec = 1.0
    fusion_interval_sec = max(0.25, fusion_interval_sec)
    all_fusion_states = []

    global speaker_identities
    speaker_identities = {}
    last_confirmed_cx_by_spk = {}

    raw_keyframes, scale_keyframes, y_keyframes = [], [], []
    current_crop_x = default_x
    current_crop_y = 0.0
    current_scale = 1.0
    last_scale_kf_time = -1.0
    last_y_kf_time = -1.0
    debug_frames_saved = 0

    # ── Camera Lock Operator State (Portrait Camera Hold & Target Lock) ───────
    locked_speaker = None
    locked_role = None
    locked_anchor_x = None
    locked_anchor_y = None
    displacement_start_time = None
    last_reframe_time = -1.0
    last_role_change_time = -1.0

    DISP_THRESH = max(70.0, crop_w_baseline * cfg.displacement_thresh_ratio)
    SUSTAINED_DUR = cfg.sustained_displacement_dur
    REFRAME_COOLDOWN = cfg.reframe_cooldown
    SCALE_DEADBAND = cfg.scale_deadband
    # Per-shot overrides are computed at the top of the shot loop from
    # shot_motion (Phase 3 advisory); the base constants stay untouched.

    # Adaptive Framing (10.0) never uses DualFrame; original mode still honors
    # the AUTOSHORTS_DUALFRAME_ENABLED kill switch.
    dual_enabled = (not adaptive_framing) and _is_dualframe_env_enabled()
    dual_config = DualFrameConfig(enabled=dual_enabled)
    dual_resolver = DualFrameLayoutResolver(dual_config)
    timeline_decisions = []
    sys.stderr.write(f"[DualFrame] mode={dual_resolver.mode}\n")
    if adaptive_framing:
        sys.stderr.write("[Framing] adaptive=on dualframe_disabled=true\n")
    sys.stderr.flush()

    for shot_idx, (shot_s_rel, shot_e_rel) in enumerate(shots):
        shot_samples = [s for s in all_samples if shot_s_rel <= s[0] < shot_e_rel]
        is_shot_cut = shot_idx > 0

        # ── Phase 3 per-scene camera advisory: inside static shots, tolerate
        # slightly less frequent reframing (bounded scaling of the EXISTING
        # deterministic thresholds; containment/validator unchanged).
        _mv = shot_motion.get(shot_idx)
        _ss = min(SCENE_STATIC_SCALE, SCENE_STATIC_SCALE_CAP) if (
            _mv is not None and _mv < SCENE_STATIC_DIFF) else 1.0
        shot_disp_thresh = DISP_THRESH * _ss
        shot_sustained_dur = SUSTAINED_DUR * _ss
        shot_reframe_cooldown = REFRAME_COOLDOWN * _ss
        if _ss != 1.0:
            sys.stderr.write(f"[SceneIntel] shot {shot_idx} static (diff={_mv:.1f}); reframe thresholds x{_ss:.2f}\n")

        # Tracking runs in STORED-frame space (frames + faces scaled to the
        # stored resolution so YOLO boxes and YuNet geometry share one space).
        # Track results are scaled back to TRUE source space afterwards, so
        # every downstream consumer (framing solver, DualFrame, Camera Lock,
        # trajectory validation) sees true-space coordinates.
        # stored_ratio = stored_w / true_w: true->stored multiplies by it,
        # stored->true divides by it.
        if stored_ratio != 1.0 and shot_samples:
            track_samples = [
                (s[0], s[1], [_scale_face_to_space(f, stored_ratio) for f in s[2]])
                for s in shot_samples
            ]
        else:
            track_samples = shot_samples
        shot_tracks = _scale_shot_tracks_to_source_space(
            build_shot_tracks(track_samples), 1.0 / stored_ratio
        )
        shot_scale_state = None  # KEY INVARIANT: reset per shot

        # ── Phase 2: Speaker Intelligence per-shot hooks (fallback-safe) ─────
        # 1. Tag local track ids so fused states can be attributed per track.
        for _tid, _tr in shot_tracks.items():
            _tr["track_id"] = _tid
        # 2. Re-ID: extract OSNet embeddings (EMA gallery) + associate with the
        #    persisted DB gallery. No-ops when Re-ID is disabled or the model
        #    is unavailable (positional association unchanged). Crops come from
        #    the STORED frames, so the true-space track boxes are scaled by
        #    stored_ratio before indexing.
        attach_reid_embeddings(shot_tracks, shot_samples, start_sec, shot_idx,
                               face_space_ratio=stored_ratio)
        associate_with_persisted_gallery(shot_tracks, persisted_gallery)
        # 3. Active-speaker fusion: diarization + mouth motion + identity +
        #    temporal hysteresis -> per-track speaking states for this shot.
        shot_fusion_states = run_fusion_for_shot(
            shot_tracks, shot_samples, diar_segments,
            start_sec, shot_s_rel, shot_e_rel,
            interval_sec=fusion_interval_sec,
        )
        all_fusion_states.extend(shot_fusion_states)

        # Reset camera anchor at every genuine shot cut
        locked_anchor_x = None
        locked_anchor_y = None
        locked_speaker = None
        locked_role = None
        displacement_start_time = None
        last_reframe_time = -1.0
        last_role_change_time = -1.0

        cur_t = shot_s_rel
        # Heartbeat every ~15s so a long shot loop is observable
        while cur_t < shot_e_rel:
            if time.time() - last_heartbeat > 15.0:
                sys.stderr.write(f"[Framing] HEARTBEAT shot_idx={shot_idx} cur_t={cur_t:.1f} elap={time.time()-last_heartbeat:.0f}s\n")
                sys.stderr.flush()
                last_heartbeat = time.time()
            sub_s = cur_t
            sub_e = min(shot_e_rel, cur_t + 0.50)
            sub_abs = start_sec + sub_s

            active_spk = None
            for (ts, te, st) in (speaker_turns or []):
                if ts <= sub_abs < te or ts <= sub_abs + 0.25 <= te:
                    active_spk = st
                    break

            local_s = [s for s in shot_samples if sub_s - 0.25 <= s[0] <= sub_e + 0.25] or shot_samples
            fusion_by_track = fusion_states_for_window(
                all_fusion_states, sub_s, sub_e, start_sec,
            )
            (clamped_x, crop_y, eff_crop_w, eff_crop_h, target_cx,
             shot_type, camera_mode, diag, shot_scale_state) = classify_and_frame_subject(
                shot_tracks, local_s, active_spk, sub_s, sub_e,
                last_confirmed_cx_by_spk, default_x + crop_w_baseline / 2.0,
                crop_w_baseline, max(0.0, source_w - crop_w_baseline),
                source_h, source_w,
                current_crop_x=locked_anchor_x,
                current_crop_y=locked_anchor_y,
                current_scale=current_scale if not is_shot_cut else 1.0,
                shot_scale_state=shot_scale_state, shot_id=shot_idx,
                is_shot_cut=is_shot_cut,
                config=cfg,
                adaptive_framing=adaptive_framing,
                fusion_by_track=fusion_by_track,
            )

            resolved_scale = diag["crop_scale"]

            # ── DualFrame Layout Resolution ──────────────────────────────────
            active_tracks_at_t = {}
            for tid, tr in shot_tracks.items():
                dets = [d for d in tr.get("detections", []) if d[0] <= sub_e]
                if not dets and "detections" in tr and tr["detections"]:
                    continue
                last_det_t = dets[-1][0] if dets else sub_s
                if sub_s - last_det_t <= dual_config.max_lost_sec:
                    tr_copy = dict(tr)
                    tr_copy["detections"] = dets or tr.get("detections", [])
                    tr_copy["age_sec"] = tr.get("age_sec") if tr.get("age_sec") is not None else (
                        max(0.0, float(last_det_t - dets[0][0])) if dets else 0.0
                    )
                    if dets:
                        last_det = dets[-1][1]
                        if isinstance(last_det, dict):
                            if "center" in last_det:
                                tr_copy["cx"] = last_det["center"][0]
                                tr_copy["cy"] = last_det["center"][1]
                            elif "bbox" in last_det:
                                tr_copy["cx"] = last_det["bbox"][0] + last_det["bbox"][2] / 2.0
                                tr_copy["cy"] = last_det["bbox"][1] + last_det["bbox"][3] / 2.0
                    if "cx" not in tr_copy:
                        tr_copy["cx"] = tr.get("cx", source_w / 2.0)
                    if "cy" not in tr_copy:
                        tr_copy["cy"] = tr.get("cy", source_h / 2.0)
                    if dets:
                        tr_copy["is_occluded"] = (sub_s - last_det_t > 0.35)
                    active_tracks_at_t[tid] = tr_copy

            dual_dec = dual_resolver.update(
                rel_t=sub_s,
                active_tracks=active_tracks_at_t,
                is_shot_cut=is_shot_cut,
                active_spk=active_spk,
                source_w=source_w,
                source_h=source_h,
            )
            timeline_decisions.append({
                "start": sub_s,
                "end": sub_e,
                "decision": dual_dec,
                "shot_idx": shot_idx,
                "shot_type": shot_type,
                "pair_needed_scale": diag.get("pair_needed_scale"),
                "tracks": shot_tracks,
                "face_bbox": diag.get("face_bbox"),
                "single_crop": (clamped_x, crop_y, eff_crop_w, eff_crop_h),
            })

            if debug_vis_dir and local_s and debug_frames_saved < 24:
                if debug_frames_saved == 0 or sub_abs - last_reframe_time >= 1.5 or is_shot_cut:
                    save_debug_visualization(min(local_s, key=lambda s: abs(s[0] - sub_s))[1], sub_s, diag, debug_vis_dir)
                    debug_frames_saved += 1

            # ── Camera Lock Operator: Target Lock + Stable Framing Anchor ────
            face_bbox = diag.get("face_bbox")
            current_role = diag.get("subject_role", "speaker")
            framing_cfg = diag.get("config", DEFAULT_FRAMING_CONFIG)
            should_reframe = False
            is_cut_event = False
            # Deliberate subject-change reframes (speaker handoff, role change,
            # face-leaving-crop correction). These are HARD transitions: the crop
            # must jump to the new subject, never be smoothed through the empty
            # space between two speakers (optimize_camera_trajectory and
            # build_ffmpeg_expr both key off the cut flag).
            hard_transition = False

            if locked_anchor_x is None or is_shot_cut:
                # 1. Shot cut or initial shot framing: establish anchor immediately
                should_reframe = True
                is_cut_event = is_shot_cut
                hard_transition = True
                is_shot_cut = False
                locked_role = current_role
                last_role_change_time = sub_abs
                locked_speaker = active_spk
            elif locked_role == "listener_reaction":
                target_dist = abs(clamped_x - locked_anchor_x)
                is_same_subject = target_dist < max(25.0, crop_w_baseline * 0.05)
                time_in_reaction = sub_abs - last_role_change_time

                if current_role == "speaker" and is_same_subject:
                    # Framed listener themselves started speaking: immediate role transition to speaker
                    locked_role = "speaker"
                    last_role_change_time = sub_abs
                    locked_speaker = active_spk
                elif (current_role == "speaker" or (active_spk is not None and active_spk != locked_speaker)) and not is_same_subject:
                    # Off-screen speaker or switch to different subject: enforce reaction dwell hysteresis
                    if time_in_reaction >= framing_cfg.listener_reaction_min_dwell:
                        should_reframe = True
                        hard_transition = True
                        locked_role = current_role
                        last_role_change_time = sub_abs
                        locked_speaker = active_spk
            elif locked_role == "two_shot":
                if current_role != "two_shot":
                    should_reframe = True
                    hard_transition = True
                    locked_role = current_role
                    last_role_change_time = sub_abs
                    locked_speaker = active_spk
            elif active_spk is not None and active_spk != locked_speaker:
                # 2. Speaker handoff: responsive transition to new active speaker
                spk_dist = abs(clamped_x - locked_anchor_x)
                if spk_dist >= max(25.0, crop_w_baseline * 0.05):
                    should_reframe = True
                    hard_transition = True
                    locked_role = current_role
                    last_role_change_time = sub_abs
                    locked_speaker = active_spk
                else:
                    locked_speaker = active_spk

            if not should_reframe:
                if locked_role == "two_shot":
                    pass
                elif face_bbox is not None:
                    # 3. Same speaker / dwelling subject: evaluate framing stability vs meaningful movement
                    fx, fy, fw, fh = face_bbox
                    fcx = fx + fw / 2.0
                    crop_l = locked_anchor_x
                    crop_r = locked_anchor_x + eff_crop_w
                    crop_cx = crop_l + eff_crop_w / 2.0

                    avail_margin = max(60.0, eff_crop_w - fw)
                    max_center_dev = max(45.0, min(eff_crop_w * 0.24, avail_margin * 0.75))

                    severe_clip = (fx < crop_l - 20.0) or (fx + fw > crop_r + 20.0)
                    off_center = abs(fcx - crop_cx) > max_center_dev

                    time_since_reframe = sub_abs - last_reframe_time
                    can_reframe_offcenter = time_since_reframe >= shot_reframe_cooldown
                    can_reframe_clip = time_since_reframe >= 0.40

                    if severe_clip and can_reframe_clip:
                        # Subject is leaving the portrait boundary: hard corrective reframe
                        should_reframe = True
                        hard_transition = True
                    elif off_center and can_reframe_offcenter:
                        # Subject is drifting off-center: smooth corrective reframe
                        should_reframe = True
                    else:
                        # 4. Sustained displacement check (temporal hysteresis)
                        disp = abs(clamped_x - locked_anchor_x)
                        if disp >= shot_disp_thresh and time_since_reframe >= shot_reframe_cooldown:
                            if displacement_start_time is None:
                                displacement_start_time = sub_abs
                            elif sub_abs - displacement_start_time >= shot_sustained_dur:
                                should_reframe = True
                        else:
                            displacement_start_time = None
                else:
                    # Fallback check when face not detected in local sub-window
                    disp = abs(clamped_x - locked_anchor_x)
                    time_since_reframe = sub_abs - last_reframe_time
                    if disp >= shot_disp_thresh * 1.5 and time_since_reframe >= shot_reframe_cooldown * 1.5:
                        should_reframe = True

            if should_reframe:
                raw_keyframes.append((sub_abs, clamped_x, is_cut_event or hard_transition))
                locked_anchor_x = clamped_x
                locked_anchor_y = crop_y
                if active_spk is not None:
                    locked_speaker = active_spk
                if locked_role != current_role:
                    last_role_change_time = sub_abs
                locked_role = current_role
                last_reframe_time = sub_abs
                displacement_start_time = None

            # Scale keyframes: hold shot-local scale state
            is_sc = is_cut_event
            if not scale_keyframes:
                scale_keyframes.append((sub_abs, resolved_scale, False))
                current_scale = resolved_scale
                last_scale_kf_time = sub_abs
            elif is_sc:
                scale_keyframes.append((sub_abs, resolved_scale, True))
                current_scale = resolved_scale
                last_scale_kf_time = sub_abs
            elif abs(resolved_scale - current_scale) >= SCALE_DEADBAND and sub_abs - last_scale_kf_time >= 1.5:
                scale_keyframes.append((sub_abs, resolved_scale, False))
                current_scale = resolved_scale
                last_scale_kf_time = sub_abs

            # Y keyframes: hold stable eye-line, only update on shot cut or large sustained shift
            is_yc = is_cut_event
            if not y_keyframes:
                y_keyframes.append((sub_abs, crop_y, False))
                current_crop_y = crop_y
                last_y_kf_time = sub_abs
            elif is_yc:
                y_keyframes.append((sub_abs, crop_y, True))
                current_crop_y = crop_y
                last_y_kf_time = sub_abs
            elif abs(crop_y - current_crop_y) >= max(30.0, eff_crop_h * 0.08) and sub_abs - last_y_kf_time >= 1.5:
                y_keyframes.append((sub_abs, crop_y, False))
                current_crop_y = crop_y
                last_y_kf_time = sub_abs

            cur_t += 0.50

    if not raw_keyframes:
        fb = list(last_confirmed_cx_by_spk.values())[0] if last_confirmed_cx_by_spk else default_x
        fx = int(round(np.clip(fb - crop_w_baseline / 2.0, 0.0, max(0.0, source_w - crop_w_baseline))))
        bw, bh = compute_effective_crop_dims(1.0, source_h, source_w, adaptive_framing=adaptive_framing)
        fallback_crop = CropRectExpr(
            x=str(fx),
            y="0",
            w=str(int(round(bw))),
            h=str(int(round(bh))),
        )
        seg_face_b = compute_single_segment_face_bounds(
            timeline_decisions, 0.0, clip_dur, adaptive_framing=adaptive_framing,
        )
        plan = SmartFramingPlan(
            mode="single",
            x=fallback_crop.x,
            y=fallback_crop.y,
            w=fallback_crop.w,
            h=fallback_crop.h,
            segments=[
                LayoutSegment(
                    mode="single",
                    start=0.0,
                    end=round(clip_dur, 3),
                    crop=fallback_crop,
                    face_bounds=seg_face_b,
                )
            ],
            face_bounds=seg_face_b,
            framing="adaptive" if adaptive_framing else "original",
            speaker_intel=build_speaker_intel_block(
                get_reid_manager(), all_fusion_states, start_sec,
                fusion_method="fusion" if all_fusion_states else "heuristic_fallback",
            ) if (speaker_fusion_enabled() or speaker_reid_enabled()) else None,
        )
        sys.stderr.write("[DualFrame] layout_plan total_segments=1 dual_segments=0\n")
        sys.stderr.flush()
        print(plan.to_json())
        return plan

    # Clean position keyframes
    ckf = [raw_keyframes[0]]
    for kf in raw_keyframes[1:]:
        prev = ckf[-1]
        if abs(kf[0] - prev[0]) < 0.05:
            ckf[-1] = kf
        elif abs(kf[1] - prev[1]) < 3 and not kf[2]:
            continue
        else:
            ckf.append(kf)

    # Clean scale keyframes
    cskf = [scale_keyframes[0]]
    for kf in scale_keyframes[1:]:
        prev = cskf[-1]
        if abs(kf[0] - prev[0]) < 0.05:
            cskf[-1] = kf
        elif abs(kf[1] - prev[1]) < 0.01 and not kf[2]:
            continue
        else:
            cskf.append(kf)

    opt_kf = optimize_camera_trajectory(ckf, max(0.0, source_w - crop_w_baseline))
    val_kf, val_skf, audit = validate_and_correct_trajectory(
        opt_kf, cskf, all_samples, start_sec, crop_w_baseline, source_w, source_h,
        adaptive_framing=adaptive_framing,
    )

    sys.stderr.write(
        f"[Smart Framing v12] shots={len(shots)} pos_kf={len(val_kf)} scale_kf={len(val_skf)} "
        f"y_kf={len(y_keyframes)} corr={audit.get('n_corrections', 0)} "
        f"avg_scale={audit.get('avg_scale', 1.0):.3f} max_scale={audit.get('max_scale', 1.0):.3f} "
        f"escalation={audit.get('scale_escalation_detected', False)}\n"
    )

    # Final legal bounds clamping for crop output. The rendered crop is ALWAYS
    # the baseline: in Original 9:16 that is the widest true-9:16 crop
    # (source_h*9/16, full height) — a wider crop is not renderable as 9:16
    # without an anisotropic stretch, and the pair is kept via HEAD containment
    # at the renderable width, so no width escalation is ever needed. In
    # Adaptive Framing the baseline is the SQUARE inner composition
    # (min(source_h, source_w) per side), rendered full-width and vertically
    # centered in the 9:16 canvas. Original 9:16 is untouched by construction.
    final_crop_w = int(round(crop_w_baseline))
    if final_crop_w % 2 != 0:
        final_crop_w -= 1
    final_crop_w = max(2, min(int(source_w), final_crop_w))

    if adaptive_framing:
        # The inner composition is a square: the crop height equals its width.
        final_crop_h = int(round(crop_w_baseline))
    else:
        final_crop_h = int(round(source_h))
    if final_crop_h % 2 != 0:
        final_crop_h -= 1
    final_crop_h = max(2, min(int(source_h), final_crop_h))

    max_legal_x = max(0, int(source_w - final_crop_w))
    max_legal_y = max(0, int(source_h - final_crop_h))

    x_expr = str(int(max(0, min(max_legal_x, round(val_kf[0][1]))))) if len(val_kf) == 1 else build_ffmpeg_expr(val_kf, start_sec, default_x, max_legal_x)
    w_expr = str(final_crop_w)
    h_expr = str(final_crop_h)
    y_expr = str(int(max(0, min(max_legal_y, round(y_keyframes[0][1]))))) if len(y_keyframes) == 1 else build_ffmpeg_y_expr(y_keyframes, start_sec, source_h, max_legal_y)
    plan = build_layout_plan(
        timeline_decisions=timeline_decisions,
        clip_dur=clip_dur,
        start_sec=start_sec,
        source_w=source_w,
        source_h=source_h,
        crop_w_baseline=crop_w_baseline,
        val_kf=val_kf,
        y_keyframes=y_keyframes,
        default_x=default_x,
        max_legal_x=max_legal_x,
        max_legal_y=max_legal_y,
        final_crop_w=final_crop_w,
        final_crop_h=final_crop_h,
        x_expr=x_expr,
        y_expr=y_expr,
        w_expr=w_expr,
        h_expr=h_expr,
        config=dual_config,
        framing="adaptive" if adaptive_framing else "original",
    )
    plan.speaker_intel = build_speaker_intel_block(
        get_reid_manager(), all_fusion_states, start_sec,
        fusion_method="fusion" if all_fusion_states else "heuristic_fallback",
    ) if (speaker_fusion_enabled() or speaker_reid_enabled()) else None

    print(plan.to_json())
    return plan


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoShorts Smart Reframing Engine v12.0")
    parser.add_argument("source_path")
    parser.add_argument("start_ms", type=float)
    parser.add_argument("end_ms", type=float)
    parser.add_argument("crop_w", type=float)
    parser.add_argument("max_x", type=float)
    parser.add_argument("default_x", type=float)
    parser.add_argument("transcript_file", nargs="?", default=None)
    parser.add_argument("--debug-vis", default=None)
    parser.add_argument("--diarization-json", default=None,
                        help="Cached source-level diarization JSON (Rust Speaker Intelligence engine)")
    parser.add_argument("--gallery-json", default=None,
                        help="Persisted Re-ID gallery JSON (Rust Speaker Intelligence engine)")
    parser.add_argument("--scene-cuts-json", default=None,
                        help="Cached source-level PySceneDetect boundaries (Phase 3 scene intelligence)")

    args, _ = parser.parse_known_args()

    transcript_words = None
    if args.transcript_file and os.path.exists(args.transcript_file):
        try:
            with open(args.transcript_file, "r", encoding="utf-8") as f:
                transcript_words = json.load(f)
        except Exception:
            pass

    run(
        args.source_path, args.start_ms, args.end_ms,
        args.crop_w, args.max_x, args.default_x,
        transcript_words, debug_vis_dir=args.debug_vis,
        diarization_json_path=args.diarization_json,
        gallery_json_path=args.gallery_json,
        scene_cuts_json_path=args.scene_cuts_json,
    )
