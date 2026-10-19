from pathlib import Path
import hashlib
import re

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
