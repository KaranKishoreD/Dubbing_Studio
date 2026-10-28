"""
Where each project's files live: cache/<project id>/, with one file per line
under segments/. Everything a stage makes is saved here, which is what lets
a re-run or an edit reuse finished work.
"""

import hashlib
import re
from pathlib import Path

CACHE_ROOT = Path("cache")


def get_cache_dir(video_id: str) -> Path:
    d = CACHE_ROOT / video_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_segments_dir(video_id: str) -> Path:
    d = get_cache_dir(video_id) / "segments"
    d.mkdir(exist_ok=True)
    return d


def instruction_hash(instruction: str) -> str:
    return hashlib.md5(instruction.encode()).hexdigest()[:8]


def video_id_from_url(url: str) -> str:
    patterns = [
        r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})",
        r"(?:embed/)([A-Za-z0-9_-]{11})",
        r"(?:shorts/)([A-Za-z0-9_-]{11})",
    ]
    for pattern in patterns:
        m = re.search(pattern, url)
        if m:
            return m.group(1)
    return hashlib.md5(url.encode()).hexdigest()[:11]


def clip_id(video_id: str, start: float | None, end: float | None) -> str:
    """Project id for part of a video: each range gets its own cache, so a
    0:00–0:49 dub never reuses a full-length transcript or vice versa."""
    if start is None and end is None:
        return video_id
    a = f"{start:g}" if start is not None else "0"
    b = f"{end:g}" if end is not None else "end"
    return f"{video_id}_{a}-{b}"
