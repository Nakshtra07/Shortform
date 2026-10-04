#!/usr/bin/env python3
"""
AutoShorts 11.0 Phase 2 — Active Speaker Fusion

Fuses audio diarization evidence with visual mouth-motion evidence
to determine active speaker per track per time interval.

Input: 
- Diarization segments (from speaker_diarization.py)
- Visual tracks with mouth motion scores (from speaker_tracker.py)
- Re-ID gallery (from reid.py)

Output: ActiveSpeakerState intervals for each track.
"""

import sys
import os
import json
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict

# ─── Configuration ─────────────────────────────────────────────────────────────

# Fusion weights
FUSION_AUDIO_WEIGHT = 0.6
FUSION_VISUAL_WEIGHT = 0.4

# Thresholds
AUDIO_EVIDENCE_THRESHOLD = 0.5
VISUAL_EVIDENCE_THRESHOLD = 0.3
SPEAKING_PROBABILITY_THRESHOLD = 0.5

# Temporal smoothing
HYSTERESIS_FRAMES = 3  # frames to wait before switching
MIN_SPEAKING_DURATION = 0.5  # minimum speaking duration (seconds)
SWITCH_COOLDOWN = 1.0  # minimum time between speaker switches (seconds)

# Evidence source weights
EVIDENCE_WEIGHTS = {
    "diarization": 1.0,
    "visual_mouth_motion": 0.8,
    "reid_match": 0.6,
    "temporal_continuity": 0.5,
}

# ─── Data Classes ──────────────────────────────────────────────────────────────

@dataclass
class DiarizedSegment:
    speaker_id: str
    start: float
    end: float
    confidence: float

@dataclass
class VisualTrack:
    track_id: int
    mouth_motion_scores: List[Tuple[float, float]]  # (time, score)
    face_visibility: List[Tuple[float, float]]  # (time, visibility)
    reid_embedding: Optional[List[float]] = None

@dataclass
class ActiveSpeakerInterval:
    track_id: int
    diarization_id: Optional[str]
    application_role: Optional[str]
    start: float
    end: float
    audio_evidence: float
    visual_evidence: float
    speaking_probability: float
    confidence: float
    evidence_sources: List[str]

@dataclass
class ActiveSpeakerState:
    track_id: int
    diarization_id: Optional[str]
    application_role: Optional[str]
    start: float
    end: float
    audio_evidence: float
    visual_evidence: float
    speaking_probability: float
    confidence: float
    evidence_sources: List[str]

# ─── Core Fusion Logic ─────────────────────────────────────────────────────────

def compute_audio_evidence(
    diarization_segments: List[DiarizedSegment],
    track_id: int,
    diarization_to_track: Dict[str, int],
    interval_start: float,
    interval_end: float
) -> Tuple[float, bool]:
    """
    Compute audio evidence for a track in a time interval.
    Returns (evidence_score, has_evidence).
    """
    if not diarization_segments:
        return 0.0, False
    
    # Find which diarization speaker maps to this track
    speaker_id = None
    for spk_id, trk_id in diarization_to_track.items():
        if trk_id == track_id:
            speaker_id = spk_id
            break
    
    if speaker_id is None:
        return 0.0, False
    
    # Check overlap with diarization segments for this speaker
    total_overlap = 0.0
    max_confidence = 0.0
    
    for seg in diarization_segments:
        if seg.speaker_id != speaker_id:
            continue
        # Compute overlap
        overlap_start = max(seg.start, interval_start)
        overlap_end = min(seg.end, interval_end)
        if overlap_end > overlap_start:
            total_overlap += overlap_end - overlap_start
            max_confidence = max(max_confidence, seg.confidence)
    
    interval_duration = interval_end - interval_start
    if interval_duration <= 0:
        return 0.0, False
    
    # Evidence = overlap ratio * confidence
    overlap_ratio = total_overlap / interval_duration
    evidence = overlap_ratio * max_confidence
    
    return min(1.0, evidence), total_overlap > 0

def compute_visual_evidence(
    track: VisualTrack,
    interval_start: float,
    interval_end: float
) -> Tuple[float, bool]:
    """
    Compute visual evidence from mouth motion scores.
    Returns (evidence_score, has_evidence).
    """
    if not track.mouth_motion_scores:
        return 0.0, False
    
    # Filter scores in interval
    interval_scores = [
        score for time, score in track.mouth_motion_scores
        if interval_start <= time <= interval_end
    ]
    
    if not interval_scores:
        return 0.0, False
    
    # Use max score in interval (peak mouth motion)
    max_score = max(interval_scores)
    avg_score = sum(interval_scores) / len(interval_scores)
    
    # Evidence = weighted combination of max and avg
    evidence = 0.7 * max_score + 0.3 * avg_score
    
    return min(1.0, evidence), max_score > 0.1

def fuse_evidence(
    audio_evidence: float,
    visual_evidence: float,
    has_audio: bool,
    has_visual: bool,
    prev_speaking: bool,
    track_id: int,
    last_switch_time: float,
    current_time: float,
    any_audio_speaking: bool = False
) -> Tuple[float, float, bool, List[str]]:
    """
    Fuse audio and visual evidence into speaking probability.

    When audio IS present in the interval (some diarized speaker is active)
    but THIS track has no audio evidence of its own, the visual-only evidence
    is strongly discounted: a laughing/moving listener must not out-evidence
    the diarized speaker's track. With NO diarization at all, visual-only
    fusion is unchanged.

    Returns (speaking_probability, confidence, is_speaking, evidence_sources).
    """
    sources = []
    total_weight = 0.0
    weighted_sum = 0.0

    if has_audio:
        weighted_sum += audio_evidence * FUSION_AUDIO_WEIGHT
        total_weight += FUSION_AUDIO_WEIGHT
        sources.append("diarization")

    if has_visual:
        weighted_sum += visual_evidence * FUSION_VISUAL_WEIGHT
        total_weight += FUSION_VISUAL_WEIGHT
        sources.append("visual_mouth_motion")

    if total_weight == 0:
        return 0.0, 0.0, False, []

    base_probability = weighted_sum / total_weight

    # Audio-conflict discount: someone else is diarized-speaking in this
    # interval and this track carries no audio evidence of its own.
    if any_audio_speaking and not has_audio:
        base_probability *= 0.3

    # Temporal continuity: boost if was speaking recently
    temporal_boost = 0.0
    if prev_speaking:
        temporal_boost = 0.1
        sources.append("temporal_continuity")

    # Hysteresis: require higher threshold to start speaking than to continue
    if not prev_speaking:
        # Need stronger evidence to start speaking
        threshold = SPEAKING_PROBABILITY_THRESHOLD + 0.15
    else:
        threshold = SPEAKING_PROBABILITY_THRESHOLD - 0.1

    speaking_probability = min(1.0, base_probability + temporal_boost)
    is_speaking = speaking_probability >= threshold
    
    # Switch cooldown
    if is_speaking != prev_speaking:
        # Check cooldown
        pass  # handled at interval level
    
    confidence = min(1.0, (base_probability + temporal_boost) * (1.0 if (has_audio or has_visual) else 0.5))
    
    return speaking_probability, confidence, is_speaking, sources

def build_diarization_to_track_map(
    diarization_segments: List[DiarizedSegment],
    visual_tracks: List[VisualTrack],
    reid_gallery: Dict[int, Any]  # track_id -> embedding
) -> Dict[str, int]:
    """
    Map diarization speaker IDs to visual track IDs using mouth-motion overlap.

    The assignment is ONE-TO-ONE (greedy by score, descending): each track is
    bound to at most one speaker and each speaker to at most one track. Without
    this, alternating speakers with comparable motion all bind to the same
    dominant track and every track inherits every speaker's audio evidence.
    """
    # Score every (speaker, track) pair by mouth-motion overlap.
    scores: List[Tuple[float, str, int]] = []
    speaker_ids = set(seg.speaker_id for seg in diarization_segments)
    for speaker_id in speaker_ids:
        speaker_segments = [s for s in diarization_segments if s.speaker_id == speaker_id]
        if not speaker_segments:
            continue
        for track in visual_tracks:
            overlap_score = 0.0
            for seg in speaker_segments:
                for mv_start, mv_score in track.mouth_motion_scores:
                    if seg.start <= mv_start <= seg.end:
                        overlap_score += mv_score
            if overlap_score > 0.1:
                scores.append((overlap_score, speaker_id, track.track_id))

    # Greedy one-to-one assignment by descending score. Ties break
    # deterministically on (speaker_id, track_id).
    scores.sort(key=lambda s: (-s[0], s[1], s[2]))
    mapping: Dict[str, int] = {}
    used_speakers = set()
    used_tracks = set()
    for score, speaker_id, track_id in scores:
        if speaker_id in used_speakers or track_id in used_tracks:
            continue
        mapping[speaker_id] = track_id
        used_speakers.add(speaker_id)
        used_tracks.add(track_id)
    return mapping

# ─── Main Fusion Pipeline ──────────────────────────────────────────────────────

def run_active_speaker_fusion(
    diarization_segments: List[DiarizedSegment],
    visual_tracks: List[VisualTrack],
    reid_gallery: Dict[int, Any],
    interval_duration: float = 1.0,
    source_duration: float = 0.0,
    speaker_to_track: Optional[Dict[str, int]] = None,
    interval_start_sec: float = 0.0
) -> List[ActiveSpeakerState]:
    """
    Main fusion pipeline.

    Args:
        diarization_segments: Audio speaker segments with timestamps
        visual_tracks: Visual tracks with mouth motion scores
        reid_gallery: Track ID -> embedding for Re-ID
        interval_duration: Time interval for output states (seconds)
        source_duration: Total source duration (for clamping)
        speaker_to_track: Optional pre-bound diarization-speaker -> track map
            (e.g. learned by the runtime from confirmed detections or Re-ID).
            When None, the map is computed from mouth-motion overlap only.
        interval_start_sec: Origin of the interval grid. Per-shot callers pass
            the shot's absolute span start so the loop covers ONLY the span
            with visual evidence; the legacy 0.0 origin burned one interval
            per second of pre-candidate source (a candidate at 400s produced
            ~400 no-evidence intervals per shot).

    Returns:
        List of ActiveSpeakerState intervals
    """
    if source_duration <= 0:
        source_duration = max(
            max((seg.end for seg in diarization_segments), default=0),
            max(
                (max((t for t, _ in track.mouth_motion_scores), default=0) for track in visual_tracks),
                default=0,
            ),
        )
    
    # Build diarization to track mapping
    diarization_to_track = dict(speaker_to_track) if speaker_to_track else build_diarization_to_track_map(
        diarization_segments, visual_tracks, {}
    )
    
    # Create track lookup
    tracks_by_id = {t.track_id: t for t in visual_tracks}
    
    # Generate time intervals (origin at the caller's span start, default 0.0)
    intervals = []
    grid_start = max(0.0, float(interval_start_sec or 0.0))
    if source_duration > grid_start:
        current_time = grid_start
        while current_time < source_duration:
            interval_end = min(current_time + interval_duration, source_duration)
            intervals.append((current_time, interval_end))
            current_time = interval_end
    
    # Track previous speaking state per track
    prev_speaking = {t.track_id: False for t in visual_tracks}
    last_switch_time = {t.track_id: -SWITCH_COOLDOWN for t in visual_tracks}

    results = []

    for interval_start, interval_end in intervals:
        # Does ANY diarized speaker overlap this interval? (audio present)
        any_audio_speaking = any(
            seg.end > interval_start and seg.start < interval_end
            for seg in diarization_segments
        )
        for track in visual_tracks:
            track_id = track.track_id
            
            # Compute audio evidence
            audio_evidence, has_audio = compute_audio_evidence(
                diarization_segments, track_id, diarization_to_track,
                interval_start, interval_end
            )
            
            # Compute visual evidence
            visual_evidence, has_visual = compute_visual_evidence(
                track, interval_start, interval_end
            )
            
            # Fuse
            speaking_prob, confidence, is_speaking, sources = fuse_evidence(
                audio_evidence, visual_evidence, has_audio, has_visual,
                prev_speaking[track_id], track_id, last_switch_time[track_id],
                interval_start,
                any_audio_speaking=any_audio_speaking
            )
            
            # Check switch cooldown
            if is_speaking != prev_speaking[track_id]:
                if interval_start - last_switch_time[track_id] < SWITCH_COOLDOWN:
                    # Revert to previous state
                    is_speaking = prev_speaking[track_id]
                    speaking_prob = 0.8 if is_speaking else 0.2
                else:
                    last_switch_time[track_id] = interval_start
            
            prev_speaking[track_id] = is_speaking
            
            # Only output if speaking or high confidence
            if is_speaking or confidence > 0.3:
                # Determine diarization ID and application role
                diarization_id = None
                for spk_id, trk_id in diarization_to_track.items():
                    if trk_id == track_id:
                        diarization_id = spk_id
                        break
                
                state = ActiveSpeakerState(
                    track_id=track_id,
                    diarization_id=diarization_id,
                    application_role=None,  # Filled by caller
                    start=interval_start,
                    end=interval_end,
                    audio_evidence=audio_evidence,
                    visual_evidence=visual_evidence,
                    speaking_probability=speaking_prob if is_speaking else 0.0,
                    confidence=confidence,
                    evidence_sources=sources
                )
                results.append(state)
    
    # Merge adjacent intervals with same state
    merged = merge_adjacent_states(results)
    return merged

def merge_adjacent_states(states: List[ActiveSpeakerState]) -> List[ActiveSpeakerState]:
    """Merge adjacent intervals with same track and speaking state."""
    if not states:
        return []
    
    # Sort by track then time
    states.sort(key=lambda s: (s.track_id, s.start))
    
    merged = []
    for track_id in set(s.track_id for s in states):
        track_states = [s for s in states if s.track_id == track_id]
        if not track_states:
            continue
        
        current = track_states[0]
        for state in track_states[1:]:
            # Same speaking state and adjacent
            if (state.speaking_probability > 0.5) == (current.speaking_probability > 0.5) \
               and abs(state.start - current.end) < 0.1:
                # Merge
                current.end = state.end
                current.confidence = (current.confidence + state.confidence) / 2
                # Merge evidence sources
                for src in state.evidence_sources:
                    if src not in current.evidence_sources:
                        current.evidence_sources.append(src)
            else:
                merged.append(current)
                current = state
        merged.append(current)
    
    return merged

# ─── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Active Speaker Fusion")
    parser.add_argument("--diarization", required=True, help="Diarization JSON file")
    parser.add_argument("--visual-tracks", required=True, help="Visual tracks JSON file")
    parser.add_argument("--reid-gallery", help="Re-ID gallery JSON file")
    parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument("--interval", type=float, default=1.0, help="Interval duration (seconds)")
    parser.add_argument("--source-duration", type=float, default=0.0, help="Source duration (seconds)")
    args = parser.parse_args()
    
    # Load inputs
    with open(args.diarization) as f:
        diarization_data = json.load(f)
    
    with open(args.visual_tracks) as f:
        visual_data = json.load(f)
    
    reid_gallery = {}
    if args.reid_gallery and os.path.exists(args.reid_gallery):
        with open(args.reid_gallery) as f:
            reid_gallery = json.load(f)
    
    # Parse diarization segments
    diarization_segments = [
        DiarizedSegment(**seg) for seg in diarization_data.get("segments", [])
    ]
    
    # Parse visual tracks
    visual_tracks = []
    for track_data in visual_data.get("tracks", []):
        visual_tracks.append(VisualTrack(
            track_id=track_data["track_id"],
            mouth_motion_scores=track_data.get("mouth_motion_scores", []),
            face_visibility=track_data.get("face_visibility", []),
            reid_embedding=track_data.get("reid_embedding")
        ))
    
    # Run fusion
    results = run_active_speaker_fusion(
        diarization_segments,
        visual_tracks,
        reid_gallery,
        interval_duration=args.interval,
        source_duration=args.source_duration
    )
    
    # Output
    output = {
        "status": "success",
        "intervals": [asdict(s) for s in results],
        "processed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"[Fusion] Generated {len(results)} active speaker intervals")

if __name__ == "__main__":
    import sys
    import os
    import time
    main()