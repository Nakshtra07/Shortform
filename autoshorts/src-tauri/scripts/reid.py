#!/usr/bin/env python3
"""
AutoShorts 11.0 Phase 2 — Re-ID Embedding Extraction (OSNet)

Extracts appearance embeddings for visual tracks using OSNet.
Designed to be integrated into speaker_tracker.py detection loop.

Model: OSNet (torchreid) — MIT license, lightweight (~2.2M params)
"""

import sys
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
import base64
import time

# ─── Configuration ─────────────────────────────────────────────────────────────

REID_MODEL_NAME = "osnet_x1_0"  # Options: osnet_x1_0, osnet_x0_75, osnet_x0_5, osnet_x0_25
REID_INPUT_SIZE = (256, 128)  # (height, width) - OSNet standard
REID_MEAN = [0.485, 0.456, 0.406]
REID_STD = [0.229, 0.224, 0.225]
REID_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REID_BATCH_SIZE = 16

# EMA update parameters
EMA_ALPHA = 0.9  # gallery = alpha * gallery + (1 - alpha) * new_embedding
MIN_DETECTIONS_FOR_GALLERY = 3  # min detections before embedding is reliable

# ─── OSNet Model Loading ───────────────────────────────────────────────────────

def load_osnet_model(model_name: str = REID_MODEL_NAME, device: str = REID_DEVICE):
    """Load OSNet model from torchreid or local weights."""
    try:
        import torchreid
    except ImportError:
        raise RuntimeError("torchreid not installed. Run: pip install torchreid")
    
    print(f"[ReID] Loading {model_name} on {device}...", file=sys.stderr)
    
    model = torchreid.models.build_model(
        name=model_name,
        num_classes=1000,  # placeholder, we use features only
        loss="softmax",
        pretrained=True
    )
    model.eval()
    model.to(device)
    
    print(f"[ReID] Model loaded successfully", file=sys.stderr)
    return model

# ─── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess_person_crop(crop: np.ndarray, target_size: Tuple[int, int] = REID_INPUT_SIZE) -> torch.Tensor:
    """Preprocess a person crop for OSNet input."""
    # crop is BGR from OpenCV
    if crop is None or crop.size == 0:
        return None
    
    # Resize
    h, w = crop.shape[:2]
    if h == 0 or w == 0:
        return None
    
    # Convert BGR to RGB
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    
    # Resize with letterbox to maintain aspect ratio
    target_h, target_w = target_size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(crop_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Pad to target size
    padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    pad_top = (target_h - new_h) // 2
    pad_left = (target_w - new_w) // 2
    padded[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
    
    # Normalize
    normalized = padded.astype(np.float32) / 255.0
    mean = np.array(REID_MEAN, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(REID_STD, dtype=np.float32).reshape(1, 1, 3)
    normalized = (normalized - mean) / std
    
    # HWC to CHW
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
    
    # Batch inference
    batch = torch.cat(tensors, dim=0).to(device)
    
    with torch.no_grad():
        features = model(batch)
        # Global average pooling if needed
        if features.dim() == 4:
            features = F.adaptive_avg_pool2d(features, (1, 1))
            features = features.view(features.size(0), -1)
        # L2 normalize
        features = F.normalize(features, p=2, dim=1)
    
    results = [None] * len(crops)
    for idx, feat_idx in enumerate(valid_indices):
        results[feat_idx] = features[idx].cpu().numpy().astype(np.float32)
    
    return results

# ─── ReID Manager ──────────────────────────────────────────────────────────────

@dataclass
class TrackEmbedding:
    track_id: int
    embedding: np.ndarray  # (512,) float32, L2 normalized
    detection_count: int
    first_seen: float
    last_seen: float
    updated_at: str

class ReIDManager:
    """
    Manages OSNet embeddings for visual tracks.
    Maintains an embedding gallery with EMA updates.
    """
    
    def __init__(self, model_name: str = REID_MODEL_NAME, device: str = REID_DEVICE, ema_alpha: float = EMA_ALPHA):
        self.model = load_osnet_model(model_name, device)
        self.device = device
        self.ema_alpha = ema_alpha
        self.gallery: Dict[int, TrackEmbedding] = {}
        self.model_name = model_name
    
    def extract_embeddings(self, frame: np.ndarray, tracks: List[Dict]) -> List[Optional[np.ndarray]]:
        """
        Extract embeddings for a list of tracks in a frame.
        
        Args:
            frame: Source frame (BGR)
            tracks: List of track dicts with 'bbox' (x, y, w, h) in source coordinates
            
        Returns:
            List of embeddings (512,) or None for each track
        """
        crops = []
        for track in tracks:
            bbox = track.get("bbox")
            if bbox is None:
                crops.append(None)
                continue
            x, y, w, h = bbox
            if w <= 0 or h <= 0:
                crops.append(None)
                continue
            # Clamp to frame
            h, h_frame = frame.shape[:2]
            x = max(0, min(x, h_frame - 1))
            y = max(0, min(y, h_frame - 1))
            w = max(1, min(w, h_frame - x))
            h = max(1, min(h, h - y))
            crop = frame[y:y+h, x:x+w]
            crops.append(crop)
        
        return extract_embedding_batch(self.model, crops, self.device)
    
    def update_gallery(self, track_id: int, embedding: np.ndarray, timestamp: float):
        """Update track embedding with EMA."""
        if track_id in self.gallery:
            entry = self.gallery[track_id]
            # EMA update
            entry.embedding = self.ema_alpha * entry.embedding + (1 - self.ema_alpha) * embedding
            entry.embedding = entry.embedding / np.linalg.norm(entry.embedding)
            entry.detection_count += 1
            entry.last_seen = timestamp
            entry.updated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        else:
            # New track
            self.gallery[track_id] = TrackEmbedding(
                track_id=track_id,
                embedding=embedding,
                detection_count=1,
                first_seen=timestamp,
                last_seen=timestamp,
                updated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            )
    
    def get_embedding(self, track_id: int) -> Optional[np.ndarray]:
        """Get current embedding for a track."""
        if track_id in self.gallery:
            return self.gallery[track_id].embedding
        return None
    
    def compute_similarity(self, track_id_a: int, track_id_b: int) -> float:
        """Compute cosine similarity between two tracks."""
        emb_a = self.get_embedding(track_id_a)
        emb_b = self.get_embedding(track_id_b)
        if emb_a is None or emb_b is None:
            return 0.0
        return float(np.dot(emb_a, emb_b))  # Already L2 normalized
    
    def find_best_match(self, track_id: int, candidate_ids: List[int], threshold: float = 0.7) -> Optional[int]:
        """Find best matching track from candidates."""
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
        """Get all gallery embeddings for serialization."""
        return self.gallery.copy()
    
    def load_gallery(self, gallery: Dict[int, TrackEmbedding]):
        """Load gallery from storage."""
        self.gallery = gallery

# ─── Utility Functions ─────────────────────────────────────────────────────────

def embedding_to_b64(embedding: np.ndarray) -> str:
    """Serialize float32 embedding to base64."""
    if embedding.dtype != np.float32:
        embedding = embedding.astype(np.float32)
    return base64.b64encode(embedding.tobytes()).decode('ascii')

def b64_to_embedding(b64_str: str, dim: int = 512) -> np.ndarray:
    """Deserialize base64 to float32 embedding."""
    data = base64.b64decode(b64_str)
    arr = np.frombuffer(data, dtype=np.float32)
    if arr.shape[0] != dim:
        arr = arr[:dim]  # truncate or pad
    return arr

def save_reid_gallery(gallery: Dict[int, TrackEmbedding], filepath: str):
    """Save ReID gallery to JSON file."""
    data = {}
    for track_id, entry in gallery.items():
        data[str(track_id)] = {
            "track_id": entry.track_id,
            "embedding_b64": embedding_to_b64(entry.embedding),
            "detection_count": entry.detection_count,
            "first_seen": entry.first_seen,
            "last_seen": entry.last_seen,
            "updated_at": entry.updated_at
        }
    with open(filepath, 'w') as f:
        json.dump({
            "model": REID_MODEL_NAME,
            "gallery": data,
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }, f)

def load_reid_gallery(filepath: str) -> Dict[int, TrackEmbedding]:
    """Load ReID gallery from JSON file."""
    with open(filepath) as f:
        data = json.load(f)
    gallery = {}
    for track_id_str, entry in data["gallery"].items():
        track_id = int(track_id_str)
        gallery[track_id] = TrackEmbedding(
            track_id=entry["track_id"],
            embedding=b64_to_embedding(entry["embedding_b64"]),
            detection_count=entry["detection_count"],
            first_seen=entry["first_seen"],
            last_seen=entry["last_seen"],
            updated_at=entry["updated_at"]
        )
    return gallery

# ─── CLI for Testing ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="OSNet Re-ID Embedding Extraction")
    parser.add_argument("--source", required=True, help="Source video file")
    parser.add_argument("--tracks-json", required=True, help="JSON file with track bboxes per frame")
    parser.add_argument("--output", required=True, help="Output gallery JSON file")
    parser.add_argument("--model", default=REID_MODEL_NAME, help="OSNet model name")
    parser.add_argument("--device", default=REID_DEVICE, help="Device: cuda or cpu")
    args = parser.parse_args()
    
    # Load tracks JSON (format: {frame_idx: [track_bboxes...]})
    with open(args.tracks_json) as f:
        tracks_data = json.load(f)
    
    # Load model
    model = load_osnet_model(args.model, args.device)
    
    # Process frames
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"[ReID] Failed to open video: {args.source}", file=sys.stderr)
        sys.exit(1)
    
    gallery = {}
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if str(frame_idx) in tracks_data:
            tracks = tracks_data[str(frame_idx)]
            if tracks:
                crops = []
                valid_tracks = []
                for track in tracks:
                    bbox = track.get("bbox")
                    if bbox and bbox[2] > 0 and bbox[3] > 0:
                        x, y, w, h = bbox
                        crop = frame[y:y+h, x:x+w]
                        if crop.size > 0:
                            crops.append(crop)
                            valid_tracks.append(track)
                
                if crops:
                    embeddings = extract_embedding_batch(model, crops, args.device)
                    for i, (track, emb) in enumerate(zip(valid_tracks, embeddings)):
                        if emb is not None:
                            track_id = track.get("track_id")
                            if track_id is not None:
                                if track_id in gallery:
                                    g = gallery[track_id]
                                    g.embedding = EMA_ALPHA * g.embedding + (1 - EMA_ALPHA) * emb
                                    g.embedding = g.embedding / np.linalg.norm(g.embedding)
                                    g.detection_count += 1
                                    g.last_seen = frame_idx / cap.get(cv2.CAP_PROP_FPS)
                                else:
                                    gallery[track_id] = TrackEmbedding(
                                        track_id=track_id,
                                        embedding=emb,
                                        detection_count=1,
                                        first_seen=frame_idx / cap.get(cv2.CAP_PROP_FPS),
                                        last_seen=frame_idx / cap.get(cv2.CAP_PROP_FPS),
                                        updated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                                    )
        
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"[ReID] Processed {frame_idx} frames, gallery size: {len(gallery)}", file=sys.stderr)
    
    cap.release()
    
    # Save gallery
    save_reid_gallery(gallery, args.output)
    print(f"[ReID] Gallery saved to {args.output} ({len(gallery)} tracks)", file=sys.stderr)

if __name__ == "__main__":
    import argparse
    import base64
    import json
    import time
    main()