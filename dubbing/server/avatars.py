"""
A face picture for each speaker, cut from the video.

A speaker only gets a face when the evidence is clear:
  - at least two of their lines show the same face visibly talking
    (video/faces.py talking_faces), and that face is at least 60% of their
    lines that show one;
  - no other speaker has the same face.
Anything less (voice-over, off-screen speaker, crowd, no faces at all, no
face analysis yet) gives no picture, and the page shows the speaker's
coloured initial instead. A wrong face is worse than none.
"""

import hashlib
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np

from pipeline.state import Segment

_MIN_LINES = 2
_MIN_SHARE = 0.6


def speaker_faces(project_dir: Path, segments: list[dict]) -> dict[int, str]:
    """{speaker: file name of their face picture} for speakers with a clear face."""
    faces_file = project_dir / "faces.npz"
    video = project_dir / "video.mp4"
    if not faces_file.exists() or not video.exists() or not segments:
        return {}
    try:
        from pipeline.video import faces

        a = dict(np.load(faces_file))
        segs = [Segment.from_dict(d) for d in segments]
        talking = faces.talking_faces(segs, a)
        ident = faces._identities(a)
    except Exception:
        return {}

    votes: dict[int, Counter] = {}
    for s in segs:
        if s.id in talking:
            votes.setdefault(s.speaker or 0, Counter())[talking[s.id]] += 1
    best = {}
    for spk, c in votes.items():
        face, n = c.most_common(1)[0]
        if n >= _MIN_LINES and n / sum(c.values()) >= _MIN_SHARE:
            best[spk] = face
    owners = Counter(best.values())
    best = {spk: f for spk, f in best.items() if owners[f] == 1}

    out = {}
    for spk, face in best.items():
        lines = sorted(s.id for s in segs if (s.speaker or 0) == spk)
        key = hashlib.sha1(repr((face, lines)).encode()).hexdigest()[:8]
        name = f"face_s{spk}_{key}.jpg"
        path = project_dir / name
        if path.exists() or _crop(video, a, ident, face, segs, spk, path):
            out[spk] = name
    return out


def _crop(video: Path, a: dict, ident, face: int, segs, spk: int, out: Path) -> bool:
    """Save the clearest view of `face` during this speaker's lines."""
    spans = [(s.start, s.end) for s in segs if (s.speaker or 0) == spk]
    t, bbox = a["t"], a["bbox"]
    during = np.array([any(x <= tt < y for x, y in spans) for tt in t])
    rows = np.where((ident == face) & during)[0]
    if len(rows) == 0:
        return False
    # Prefer a face that is large and turned towards the camera.
    size = (bbox[rows, 2] - bbox[rows, 0]) * (bbox[rows, 3] - bbox[rows, 1])
    turn = np.abs(a["yaw"][rows]) if "yaw" in a else np.zeros(len(rows))
    k = rows[int(np.argmax(size / (1 + turn)))]
    x0, y0, x1, y1 = bbox[k]
    try:
        from pipeline.video.compose import _video_size

        w, h = _video_size(str(video))
    except Exception:
        return False
    # A square around the face, a little wider than it, kept inside the frame.
    side = min(max(x1 - x0, y1 - y0) * 1.6, w, h)
    x = min(max(0, (x0 + x1) / 2 - side / 2), w - side)
    y = min(max(0, (y0 + y1) / 2 - side / 2), h - side)
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-ss",
                f"{t[k]:.2f}",
                "-i",
                str(video),
                "-frames:v",
                "1",
                "-vf",
                f"crop={int(side)}:{int(side)}:{int(x)}:{int(y)}:exact=1,scale=160:160",
                "-q:v",
                "3",
                str(out),
                "-y",
            ],
            capture_output=True,
            timeout=30,
        )
    except Exception:
        return False
    return out.exists()
