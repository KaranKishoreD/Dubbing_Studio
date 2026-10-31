"""
Smart lip sync: sync only the face whose lips really move with the line in
the ORIGINAL video.

Each line is cut into shots at camera cuts. In every shot, each face on
screen (turned at most ~40°) is cropped and scored with SyncNet against the
original voice of that moment (syncnet_worker.py). A face that is saying the
line lines up with the audio at an offset near zero; a listener, a reaction
shot or a photo does not. The best-matching face is synced; if no face
matches, the shot keeps its original frames ("when unsure, don't sync").

Measured on 3 Idiots: speaking faces scored offset −1…−3 frames, confidence
1.7–2.5; every listener hit the ±15-frame search limit or scored below 1.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

_LS_DIR = Path(os.environ.get("LATENTSYNC_DIR", "LatentSync"))
MAX_OFFSET = int(os.environ.get("SMART_SYNC_MAX_OFFSET", "4"))  # frames at 25 fps
MIN_CONF = float(os.environ.get("SMART_SYNC_MIN_CONF", "1.5"))
# One face alone in a long shot (a lecturer, a reporter): a small face scores
# lower (Arun sir at 74 px: 0.9–1.5, offset −1/−2), but a listener does not
# line up with the voice for seconds by chance (3 Idiots listeners: ±15
# frames or confidence < 0.3), so less confidence is needed.
MIN_CONF_LONE = float(os.environ.get("SMART_SYNC_MIN_CONF_LONE", "0.8"))
LONE_SHOT = float(os.environ.get("SMART_SYNC_LONE_SHOT", "3.0"))  # seconds
MIN_SHOT = float(
    os.environ.get("SMART_SYNC_MIN_SHOT", "0.9")
)  # seconds: shorter gives SyncNet too small a search window
MAX_YAW = float(os.environ.get("LIPSYNC_MAX_YAW", "0.7"))


def smart_plan(a: dict, segments, spans: dict, video: str, voice: str, cache_dir: Path):
    """{seg id: [(start, end, cx, cy)]} and a log note, like plan_sync_shots."""
    from .faces import _identities

    ident = _identities(a)
    t, bbox, yaw = a["t"], a["bbox"], a["yaw"]
    cuts = sorted(float(c) for c in a.get("cuts", []))

    shots = []  # (seg id, start, end, {face: detection indices})
    for s in segments:
        x, y = spans.get(s.id, (s.start, s.end))
        edges = [x] + [c for c in cuts if x < c < y] + [y]
        for p, q in zip(edges, edges[1:]):
            if q - p < MIN_SHOT:
                continue
            k = np.where((t >= p) & (t < q) & (ident >= 0) & (yaw <= MAX_YAW))[0]
            faces = {}
            for f in set(ident[k].tolist()):
                kk = k[ident[k] == f]
                if len(kk) >= max(4, 0.3 * (q - p) * 12.5):  # on screen ≥ 30% of the shot
                    faces[f] = kk
            if faces:
                shots.append((s.id, p, q, faces))

    scores = _scores(shots, t, bbox, video, voice, cache_dir)
    plan: dict[int, list] = {}
    synced = skipped = 0
    for sid, p, q, faces in shots:
        lone = len(faces) == 1 and q - p >= LONE_SHOT
        best, best_conf = None, MIN_CONF_LONE if lone else MIN_CONF
        for f in faces:
            off, conf = scores.get(_key(sid, p, q, f), (99, 0.0))
            if abs(off) <= MAX_OFFSET and conf >= best_conf:
                best, best_conf = f, conf
        if best is None:
            skipped += 1
            continue
        b = bbox[faces[best]]
        cx, cy = float(np.mean((b[:, 0] + b[:, 2]) / 2)), float(np.mean((b[:, 1] + b[:, 3]) / 2))
        plan.setdefault(sid, []).append((p, q, cx, cy))
        synced += 1
    note = (
        f"smart lip sync: {synced} shot(s) where a face's lips match the original voice; "
        f"{skipped} shot(s) kept original (no face speaking the line)"
    )
    return plan, note


def _key(sid, p, q, f) -> str:
    return f"{sid}|{p:.2f}|{q:.2f}|{f}"


def _scores(shots, t, bbox, video, voice, cache_dir: Path) -> dict:
    """{key: (offset, confidence)}, cached in syncnet_scores.json."""
    path = cache_dir / "syncnet_scores.json"
    try:
        cache = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cache = {}
    jobs = []
    for sid, p, q, faces in shots:
        for f, kk in faces.items():
            key = _key(sid, p, q, f)
            if key not in cache:
                track = [[float(t[i]), *map(float, bbox[i])] for i in kk]
                jobs.append({"key": key, "a": p, "b": q, "track": track})
    if jobs:
        py = _LS_DIR / ".venv-ls" / "bin" / "python"
        cfg, out = cache_dir / "syncnet_jobs.json", cache_dir / "syncnet_out.json"
        cfg.write_text(
            json.dumps(
                {
                    "video": str(Path(video).resolve()),
                    "audio": str(Path(voice).resolve()),
                    "jobs": jobs,
                }
            )
        )
        env = dict(os.environ)
        env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")  # ffmpeg
        r = subprocess.run(
            [
                os.path.abspath(py),  # not resolve(): that follows the venv symlink out of the venv
                str(Path(__file__).with_name("syncnet_worker.py")),
                str(cfg.resolve()),
                str(out.resolve()),
            ],
            cwd=str(_LS_DIR),
            capture_output=True,
            text=True,
            env=env,
        )
        if r.returncode != 0 or not out.exists():
            raise RuntimeError((r.stderr or "syncnet worker failed")[-300:])
        cache.update(json.loads(out.read_text()))
        path.write_text(json.dumps(cache))
    return {k: tuple(v) for k, v in cache.items()}
