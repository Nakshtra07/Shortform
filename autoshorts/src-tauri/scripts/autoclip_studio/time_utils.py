"""Time utilities and SRT parsing for autoclip_studio.

Independent implementation decoupled from AutoClip backend and pysrt.
"""
from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


def to_seconds(t: str) -> float:
    """'HH:MM:SS,mmm' / 'HH:MM:SS.mmm' / 'MM:SS' -> seconds. Format error raises ValueError."""
    s = str(t).strip().replace(",", ".")
    parts = s.split(":")
    if len(parts) == 3:
        h, m, sec = parts
    elif len(parts) == 2:
        h, (m, sec) = "0", parts
    else:
        raise ValueError(f"Invalid time format: {t}")
    return int(h) * 3600 + int(m) * 60 + float(sec)


def to_srt_time(sec: float) -> str:
    """Seconds -> 'HH:MM:SS,mmm' with round-to-millisecond rollover protection."""
    total_ms = int(round(max(0.0, float(sec)) * 1000))
    h, rest = divmod(total_ms, 3600_000)
    m, rest = divmod(rest, 60_000)
    s, ms = divmod(rest, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def parse_srt(srt_path: Path) -> List[Dict[str, Any]]:
    """Parse an SRT subtitle file into standard dictionaries without external dependencies."""
    srt_path = Path(srt_path)
    if not srt_path.exists() or srt_path.stat().st_size == 0:
        return []

    content = ""
    for enc in ("utf-8", "utf-8-sig", "gbk", "latin-1"):
        try:
            content = srt_path.read_text(encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    if not content:
        return []

    blocks = re.split(r"\n\s*\n", content.strip().replace("\r\n", "\n"))
    subtitles = []
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 2:
            continue
        time_line_idx = 1 if re.match(r"^\d+$", lines[0]) else 0
        if time_line_idx >= len(lines):
            continue
        match = re.search(
            r"(\d{1,2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}[,.]\d{3})",
            lines[time_line_idx],
        )
        if not match:
            continue
        start_time = match.group(1).replace(".", ",")
        end_time = match.group(2).replace(".", ",")
        text = "\n".join(lines[time_line_idx + 1 :]).strip()
        subtitles.append({
            "start_time": start_time,
            "end_time": end_time,
            "text": text,
            "index": len(subtitles) + 1,
        })
    return subtitles


def load_srt_chunks(metadata_dir: Path) -> List[Dict[str, Any]]:
    """Load step1 SRT chunks into complete cue list."""
    chunks_dir = Path(metadata_dir) / "step1_srt_chunks"
    if not chunks_dir.exists():
        return []
    entries: List[Dict[str, Any]] = []
    manifest = chunks_dir / "manifest.json"
    if manifest.exists():
        try:
            names = json.loads(manifest.read_text(encoding="utf-8"))
            if not isinstance(names, list) or any(
                not isinstance(name, str) or not re.fullmatch(r"chunk_\d+\.json", name)
                for name in names
            ):
                raise ValueError("invalid chunk manifest")
            files = [chunks_dir / name for name in names]
        except (ValueError, OSError) as error:
            logger.warning("Failed to read chunk manifest: %s", error)
            return []
    else:
        files = sorted(
            chunks_dir.glob("chunk_*.json"),
            key=lambda p: int(p.stem.split("_")[1]) if p.stem.split("_")[1].isdigit() else 0,
        )
    for f in files:
        try:
            entries.extend(json.loads(f.read_text(encoding="utf-8")))
        except Exception as e:
            logger.warning("Failed to read chunk %s: %s", f, e)
    return entries


def excerpt_between(
    srt_entries: Sequence[Dict[str, Any]], start: float, end: float, max_chars: int = 600
) -> str:
    """Excerpt transcript text between start and end seconds."""
    parts: List[str] = []
    for e in srt_entries:
        try:
            s, t = to_seconds(e["start_time"]), to_seconds(e["end_time"])
        except (KeyError, ValueError):
            continue
        if t >= start and s <= end:
            text = str(e.get("text") or "").strip()
            if text:
                parts.append(text)
    return " ".join(parts)[:max_chars]
