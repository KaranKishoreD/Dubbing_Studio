"""
Stage 9 — Lip sync: make the speakers' mouths match the dubbed audio.

  1. Analyse faces once per video (who is on screen, whose mouth moves,
     camera cuts) — video/latentsync_worker.py --analyze.
  2. Plan shots: only shots where the speaker's own face is visible and
     talking are synced (video/shot_planner.py). Others keep their frames.
  3. Render each planned shot with LatentSync 1.6 on the GPU, several
     workers in parallel when memory allows (video/latentsync_runner.py).

Backends (LIPSYNC_BACKEND = latentsync | wav2lip | fal | none). LatentSync is
used whenever its checkout and venv exist; the older Wav2Lip and fal.ai
paths live in video/legacy_lipsync.py.
"""

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import ffmpeg
import numpy as np

from ..cache import get_segments_dir, instruction_hash
from ..state import PipelineState, Segment
from ..video.latentsync_runner import _LS_CKPT, _LS_PYTHON, run_parallel, run_worker
from ..video.legacy_lipsync import _FAL_ENABLED, _USE_LOCAL, _fal_key, lipsync_fal, lipsync_wav2lip
from ..video.shot_planner import plan_sync_shots


def _backend() -> str:
    b = os.environ.get("LIPSYNC_BACKEND", "").lower()
    if b:
        return b
    if _LS_PYTHON.exists() and _LS_CKPT.exists():
        return "latentsync"
    if _USE_LOCAL:
        return "wav2lip"
    if _fal_key() and _FAL_ENABLED:
        return "fal"
    return "none"


def lipsync_node(state: PipelineState) -> dict:
    segments = [Segment.from_dict(s) for s in state["segments"]]
    edit_ops = state.get("edit_ops", [])
    edit_ids = {sid for op in edit_ops for sid in op["segment_ids"]}
    video_path = state["video_path"]
    segs_dir = get_segments_dir(state["video_id"])

    to_sync = [seg for seg in segments if seg.aligned_audio is not None]
    if not to_sync:
        return {
            "segments": [s.to_dict() for s in segments],
            "log": state.get("log", []) + ["✓ Lip sync: no segments to process"],
        }

    backend = _backend()
    if backend == "none":
        # Nothing to do: the assemble stage keeps the original video as is.
        return {
            "segments": [s.to_dict() for s in segments],
            "log": state.get("log", [])
            + [
                "⚠ Lip sync skipped — install LatentSync (scripts/setup_gpu.sh) "
                "or set LIPSYNC_BACKEND"
            ],
        }
    if backend == "latentsync":
        res = _lipsync_latentsync(segments, to_sync, edit_ids, segs_dir, video_path, state)
        res["log"] = state.get("log", []) + res["log"]
        return res

    # Keyed by time range: segment ids are renumbered when sentences merge.
    def _ensure_clip(seg: Segment) -> None:
        orig = segs_dir / f"clip_{seg.start:.3f}_{seg.end:.3f}.mp4"
        if not orig.exists():
            _extract_clip(video_path, str(orig), seg.start, seg.end)
        seg.original_clip = str(orig)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(_ensure_clip, to_sync))

    if backend == "wav2lip":
        res = lipsync_wav2lip(segments, to_sync, edit_ids, segs_dir)
    else:
        res = lipsync_fal(segments, to_sync, edit_ids, segs_dir)
    res["log"] = state.get("log", []) + res["log"]
    return res


# ── LatentSync (local GPU) ────────────────────────────────────────────────────


def _lipsync_latentsync(
    segments: list[Segment],
    to_sync: list[Segment],
    edit_ids: set[int],
    segs_dir: Path,
    video_path: str,
    state: PipelineState,
) -> dict:
    import soundfile as sf

    from .align import GAP_GUARD

    cache_dir = segs_dir.parent
    total = _probe_duration(video_path)
    ordered = sorted(segments, key=lambda s: s.start)
    next_start = {
        s.id: (ordered[k + 1].start if k + 1 < len(ordered) else total)
        for k, s in enumerate(ordered)
    }

    # Span of each line: the whole dub — it may run into the pause after the
    # line — but never into the next line, which gets its own clips.
    spans: dict[int, tuple[float, float]] = {}
    for seg in to_sync:
        end = max(seg.end, seg.start + sf.info(seg.aligned_audio).duration)
        end = min(end, next_start[seg.id] - GAP_GUARD, total)
        spans[seg.id] = (seg.start, max(end, seg.start + 0.2))

    # Who is on screen, whose mouth moves, where the cuts are — once per video.
    faces = cache_dir / "faces.npz"
    # An analysis saved without head angle or face gender is out of date:
    # delete it so it is made again.
    if faces.exists() and not {"yaw", "male"} <= set(np.load(faces).files):
        faces.unlink()
    if not faces.exists():
        run_worker(
            ["--analyze", str(Path(video_path).resolve()), str(faces.resolve())],
            cache_dir / "face_analysis.log",
        )
    if not faces.exists():
        return {
            "segments": [s.to_dict() for s in segments],
            "log": [
                f"⚠ Lip sync skipped — face analysis failed (see {cache_dir / 'face_analysis.log'})"
            ],
        }
    plan, plan_note = plan_sync_shots(dict(np.load(faces)), ordered, spans)

    jobs, pending = [], []
    for seg in to_sync:
        seg.original_clip = None
        seg.lip_synced_clip = None
        seg.sync_clips = []
        y, sr = sf.read(seg.aligned_audio, dtype="float32", always_2d=True)
        y = y.mean(axis=1)
        for a, b, cx, cy in plan.get(seg.id, []):
            key = instruction_hash(f"{Path(seg.aligned_audio).name}|{a:.3f}|{b:.3f}|{cx:.0f}")
            out = segs_dir / f"s{seg.id:04d}_ls_{key}.mp4"
            piece = {"start": a, "end": b, "path": str(out)}
            if out.exists() and seg.id not in edit_ids:
                seg.sync_clips.append(piece)
                continue
            clip = segs_dir / f"lsin_{a:.3f}_{b:.3f}.mp4"
            if not clip.exists():
                _extract_clip(video_path, str(clip), a, b)
            # This shot's slice of the dub, exactly as long as the shot:
            # LatentSync stops at whichever is shorter.
            i0, n = int(round((a - seg.start) * sr)), int(round((b - a) * sr))
            part = y[i0 : i0 + n]
            wav = segs_dir / f"lsin_{key}.wav"
            sf.write(str(wav), np.pad(part, (0, max(0, n - len(part)))), sr)
            jobs.append(
                {
                    "video": str(clip.resolve()),
                    "audio": str(wav.resolve()),
                    "out": str(out.resolve()),
                    "cx": cx,
                    "cy": cy,
                }
            )
            pending.append((seg, piece, out))

    failed = 0
    n_workers = 0
    if jobs:
        results, n_workers = run_parallel(jobs, pending, cache_dir)
        for seg, piece, out in pending:
            r = results.get(str(out.resolve()), "worker crashed — see latentsync.log")
            if r == "ok" and out.exists():
                seg.sync_clips.append(piece)
            else:
                failed += 1
                print(f"  ⚠ LatentSync seg {seg.id} {piece['start']:.1f}s: {r}")
    for seg in to_sync:
        seg.sync_clips.sort(key=lambda p: p["start"])

    shots = sum(len(s.sync_clips) for s in to_sync)
    secs = sum(p["end"] - p["start"] for s in to_sync for p in s.sync_clips)
    speech = sum(s.duration for s in to_sync)
    note = (
        f"✓ Lip-synced {shots} shot(s), {secs:.0f}s of {speech:.0f}s speech, with "
        f"LatentSync 1.6 ({len(jobs)} rendered"
        + (f" by {n_workers} parallel workers" if n_workers > 1 else "")
        + ") — only shots showing the "
        f"speaker's face; {plan_note}"
    )
    if failed:
        note += f"; {failed} shot(s) kept original frames (face lost / error)"
    return {"segments": [s.to_dict() for s in segments], "log": [note]}


def _probe_duration(path: str) -> float:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
            path,
        ],
        capture_output=True,
        text=True,
    )
    return float(out.stdout.strip())


# ── Shared helpers ────────────────────────────────────────────────────────────


def _extract_clip(video_path: str, out_path: str, start: float, end: float) -> None:
    duration = round(end - start, 3)
    (
        ffmpeg.input(video_path, ss=start, t=duration)
        .output(out_path, vcodec="libx264", acodec="aac", avoid_negative_ts="make_zero")
        .overwrite_output()
        .run(quiet=True)
    )
