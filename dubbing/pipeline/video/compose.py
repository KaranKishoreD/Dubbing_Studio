"""
Put the lip-synced shots back into the original video (used by
steps/assemble.py).

Each lip-synced clip is laid over the original video only for its own time
span, so everything else stays the untouched original frames. LatentSync
renders at 25 fps; clips are converted to the video's frame rate first.
"""

import subprocess
from pathlib import Path

# How many lip-synced clips go into one ffmpeg overlay pass.
_OVERLAY_BATCH = 24


def overlay_synced(video_path: str, synced: list[dict], cache_dir: Path) -> str:
    """
    Lay each lip-synced clip over the original video at its own time span.
    Everything outside those spans is the original frames, so timing cannot
    drift the way concatenating dozens of re-encoded pieces does. LatentSync
    renders at 25 fps; each clip is converted to the source frame rate first.
    """
    fps = _video_fps(video_path)
    w, h = _video_size(video_path)
    synced = _retime(synced, fps, cache_dir)
    src = video_path
    batches = [synced[i : i + _OVERLAY_BATCH] for i in range(0, len(synced), _OVERLAY_BATCH)]
    for b, batch in enumerate(batches):
        out = cache_dir / f"lipsync_video_{b}.mp4"
        cmd = ["ffmpeg", "-i", src]
        for piece in batch:
            cmd += ["-i", piece["path"]]
        parts, last = [], "0:v"
        for k, piece in enumerate(batch, 1):
            a, b = piece["start"], piece["end"]
            parts.append(
                f"[{k}:v]{piece.get('filter') or _to_fps(fps)},scale={w}:{h},format=yuv420p,"
                f"setpts=PTS-STARTPTS+{a:.3f}/TB[c{k}]"
            )
            parts.append(
                f"[{last}][c{k}]overlay=eof_action=pass:enable='between(t,{a:.3f},{b:.3f})'[v{k}]"
            )
            last = f"v{k}"
        cmd += [
            "-filter_complex",
            ";".join(parts),
            "-map",
            f"[{last}]",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            str(out),
            "-y",
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        src = str(out)
    return src


# LatentSync renders at 25 fps. Converting that to a 30/60 fps video by
# repeating frames stutters visibly (one duplicate in every five at 30 fps),
# so other rates are rebuilt with motion-compensated interpolation. 25 fps
# videos are untouched by this.
_LS_FPS = 25.0


def _rate(fps: str) -> float:
    try:
        num, _, den = fps.partition("/")
        return float(num) / float(den or 1)
    except ValueError:
        return _LS_FPS


def _retime(synced: list[dict], fps: str, cache_dir: Path) -> list[dict]:
    """
    Bring each 25 fps lip-synced clip to the video's frame rate. With a free
    GPU, in-between frames are warped from optical flow (frame_interp.py);
    otherwise ffmpeg blends neighbouring frames, which is near-instant and
    fine for talking heads. CPU minterpolate took minutes per minute.
    """
    rate = _rate(fps)
    if abs(rate - _LS_FPS) < 0.01:
        return synced
    from ..video import frame_interp

    out = []
    for piece in synced:
        p = Path(piece["path"])
        conv = p.with_name(f"{p.stem}_{rate:g}fps.mp4")
        if conv.exists() or frame_interp.convert(str(p), str(conv), rate, _LS_FPS):
            out.append({**piece, "path": str(conv), "filter": f"fps={fps}"})
        else:
            out.append({**piece, "filter": f"framerate=fps={fps}"})
    return out


def _to_fps(fps: str) -> str:
    try:
        num, _, den = fps.partition("/")
        rate = float(num) / float(den or 1)
    except ValueError:
        rate = _LS_FPS
    if abs(rate - _LS_FPS) < 0.01:
        return f"fps={fps}"
    return f"minterpolate=fps={fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1"


def _ffprobe_stream(path: str, entries: str) -> str:
    return subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            f"stream={entries}",
            "-of",
            "default=nw=1:nk=1",
            path,
        ],
        capture_output=True,
        text=True,
    ).stdout.strip()


def video_codec(path: str) -> str:
    return _ffprobe_stream(path, "codec_name")


def _video_fps(path: str) -> str:
    return _ffprobe_stream(path, "r_frame_rate") or "25"


def _video_size(path: str) -> tuple[int, int]:
    w, h = _ffprobe_stream(path, "width,height").split()[:2]
    return int(w), int(h)
