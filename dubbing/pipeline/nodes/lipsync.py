"""
Lip sync node.

Backends (LIPSYNC_BACKEND = latentsync | wav2lip | fal | none):
  0. LatentSync 1.6 (default when its checkout and venv exist) — ByteDance's
     audio-conditioned latent diffusion model, 512 px faces, tracked and
     re-aligned every frame. Runs locally on the GPU in its own venv
     (dubbing/LatentSync/.venv-ls); see pipeline/latentsync_worker.py.
     Each line's clip spans the dubbed audio, which may run into the silence
     after the line, so the mouth keeps moving for as long as the voice does.
     Shots with no detectable face keep the original frames.
  1. Local Wav2Lip + MediaPipe — set WAV2LIP_USE_LOCAL=1 in .env
     MediaPipe detects face accurately → crop → Wav2Lip → composite back
     Fixes the "lips on forehead" bug from bad S3FD face detection.
  2. fal.ai cloud GPU  — set FAL_KEY + FAL_ENABLED=1 in .env (needs billing)
  3. Skip (fallback)   — audio replaced at assembly; no mouth movement

Setup for local Wav2Lip:
  git clone https://github.com/Rudrabha/Wav2Lip
  # download wav2lip_gan.pth into Wav2Lip/checkpoints/
  pip install mediapipe opencv-python
  Set WAV2LIP_USE_LOCAL=1 in .env
"""

import os
import sys
import urllib.request
import subprocess
import tempfile
import ffmpeg
import numpy as np
import cv2
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from ..state import PipelineState, Segment
from ..cache import get_segments_dir, instruction_hash

# ── Backend flags (evaluated at call time) ────────────────────────────────────
def _fal_key() -> str | None:
    return os.environ.get("FAL_KEY") or None

_USE_LOCAL  = os.environ.get("WAV2LIP_USE_LOCAL", "0") == "1"
_FAL_ENABLED = os.environ.get("FAL_ENABLED", "0") == "1"

LATENTSYNC_DIR = Path(os.environ.get("LATENTSYNC_DIR", "LatentSync"))
_LS_PYTHON = LATENTSYNC_DIR / ".venv-ls" / "bin" / "python"
_LS_CKPT = LATENTSYNC_DIR / "checkpoints" / "latentsync_unet.pt"
_WORKER = Path(__file__).resolve().parent.parent / "latentsync_worker.py"


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

WAV2LIP_DIR        = Path("Wav2Lip")
_ckpt_pth = WAV2LIP_DIR / "checkpoints" / "wav2lip_gan.pth"
_ckpt_pt  = WAV2LIP_DIR / "checkpoints" / "wav2lip_gan.pt"
WAV2LIP_CHECKPOINT = _ckpt_pth if _ckpt_pth.exists() else _ckpt_pt
_FAL_MODEL         = "fal-ai/video-retalking"

# Padding around detected face bbox (fraction of bbox size)
_FACE_PAD = 0.35


def lipsync_node(state: PipelineState) -> dict:
    segments  = [Segment.from_dict(s) for s in state["segments"]]
    edit_ops  = state.get("edit_ops", [])
    edit_ids  = {sid for op in edit_ops for sid in op["segment_ids"]}
    video_path = state["video_path"]
    segs_dir  = get_segments_dir(state["video_id"])

    to_sync = [seg for seg in segments if seg.aligned_audio is not None]
    if not to_sync:
        return {
            "segments": [s.to_dict() for s in segments],
            "log": state.get("log", []) + ["✓ Lip sync: no segments to process"],
        }

    backend = _backend()
    if backend == "none":
        # No clip extraction: assemble's audio-overlay path never touches
        # per-segment video, and re-encoding every clip just to discard it
        # was pure wasted time.
        return {
            "segments": [s.to_dict() for s in segments],
            "log": state.get("log", []) + [
                "⚠ Lip sync skipped — install LatentSync (scripts/setup_gpu.sh) "
                "or set LIPSYNC_BACKEND"
            ],
        }
    if backend == "latentsync":
        res = _lipsync_latentsync(segments, to_sync, edit_ids, segs_dir,
                                  video_path, state)
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
        res = _lipsync_local(segments, to_sync, edit_ids, segs_dir)
    else:
        res = _lipsync_fal(segments, to_sync, edit_ids, segs_dir)
    res["log"] = state.get("log", []) + res["log"]
    return res


# ── LatentSync (local GPU) ────────────────────────────────────────────────────

def _lipsync_latentsync(
    segments: list[Segment], to_sync: list[Segment], edit_ids: set[int],
    segs_dir: Path, video_path: str, state: PipelineState,
) -> dict:
    import json
    import soundfile as sf
    from .align import GAP_GUARD

    cache_dir = segs_dir.parent
    total = _probe_duration(video_path)
    ordered = sorted(segments, key=lambda s: s.start)
    next_start = {s.id: (ordered[k + 1].start if k + 1 < len(ordered) else total)
                  for k, s in enumerate(ordered)}

    # Span of each line: the whole dub — it may run into the pause after the
    # line — but never into the next line, which gets its own clips.
    spans: dict[int, tuple[float, float]] = {}
    for seg in to_sync:
        end = max(seg.end, seg.start + sf.info(seg.aligned_audio).duration)
        end = min(end, next_start[seg.id] - GAP_GUARD, total)
        spans[seg.id] = (seg.start, max(end, seg.start + 0.2))

    # Who is on screen, whose mouth moves, where the cuts are — once per video.
    faces = cache_dir / "faces.npz"
    if not faces.exists():
        _run_worker(["--analyze", str(Path(video_path).resolve()), str(faces.resolve())],
                    cache_dir / "face_analysis.log")
    if not faces.exists():
        return {"segments": [s.to_dict() for s in segments],
                "log": [f"⚠ Lip sync skipped — face analysis failed "
                        f"(see {cache_dir / 'face_analysis.log'})"]}
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
            part = y[i0:i0 + n]
            wav = segs_dir / f"lsin_{key}.wav"
            sf.write(str(wav), np.pad(part, (0, max(0, n - len(part)))), sr)
            jobs.append({"video": str(clip.resolve()), "audio": str(wav.resolve()),
                         "out": str(out.resolve()), "cx": cx, "cy": cy})
            pending.append((seg, piece, out))

    failed = 0
    n_workers = 0
    if jobs:
        results, n_workers = _run_parallel(jobs, pending, cache_dir)
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
    note = (f"✓ Lip-synced {shots} shot(s), {secs:.0f}s of {speech:.0f}s speech, with "
            f"LatentSync 1.6 ({len(jobs)} rendered"
            + (f" by {n_workers} parallel workers" if n_workers > 1 else "")
            + ") — only shots showing the "
            f"speaker's face; {plan_note}")
    if failed:
        note += f"; {failed} shot(s) kept original frames (face lost / error)"
    return {"segments": [s.to_dict() for s in segments], "log": [note]}


# Measured peak per LatentSync worker with VAE slicing: ~8.3 GB. Workers are
# sized from FREE memory at the moment lip sync starts, with headroom kept.
_WORKER_GB = float(os.environ.get("LATENTSYNC_WORKER_GB", "8.5"))
_GPU_RESERVE_GB = float(os.environ.get("LATENTSYNC_RESERVE_GB", "1.5"))
_MAX_WORKERS = int(os.environ.get("LATENTSYNC_MAX_WORKERS", "4"))


def _free_gpu_gb() -> float:
    """Free memory on GPU 0 right now, as the driver reports it (0 if unknown)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free",
                              "--format=csv,noheader,nounits", "-i", "0"],
                             capture_output=True, text=True, timeout=10).stdout
        return float(out.strip().splitlines()[0]) / 1024
    except Exception:
        return 0.0


def _run_parallel(jobs: list[dict], pending: list, cache_dir: Path) -> tuple[dict, int]:
    """
    Render with as many LatentSync workers as free GPU memory allows (at
    least 1, at most _MAX_WORKERS). Shots are dealt longest-first to the
    least-loaded worker so all workers finish at about the same time.
    """
    import json
    free = _free_gpu_gb()
    n = max(1, min(_MAX_WORKERS, int((free - _GPU_RESERVE_GB) // _WORKER_GB), len(jobs)))
    print(f"  LatentSync: {free:.1f} GB GPU free → {n} worker(s) for {len(jobs)} shot(s)",
          flush=True)

    dur = [p[1]["end"] - p[1]["start"] for p in pending]
    buckets: list[list[dict]] = [[] for _ in range(n)]
    load = [0.0] * n
    for k in sorted(range(len(jobs)), key=lambda k: -dur[k]):
        w = load.index(min(load))
        buckets[w].append(jobs[k])
        load[w] += dur[k]

    procs = []
    for w, bucket in enumerate(buckets):
        jp = cache_dir / f"latentsync_jobs_{w}.json"
        rp = cache_dir / f"latentsync_results_{w}.json"
        jp.write_text(json.dumps(bucket), encoding="utf-8")
        rp.unlink(missing_ok=True)
        log = open(cache_dir / ("latentsync.log" if w == 0 else f"latentsync_{w}.log"), "w")
        procs.append((subprocess.Popen(
            [str(_LS_PYTHON.absolute()), str(_WORKER), str(jp.resolve()), str(rp.resolve())],
            cwd=str(LATENTSYNC_DIR), stdout=log, stderr=subprocess.STDOUT), log, rp))
    results: dict = {}
    for proc, log, rp in procs:
        proc.wait()
        log.close()
        try:
            results.update(json.loads(rp.read_text(encoding="utf-8")))
        except Exception:
            pass
    return results, n


def _run_worker(args: list[str], log_path: Path) -> None:
    with open(log_path, "w") as log:
        subprocess.run(
            # absolute(), not resolve(): resolving follows the venv symlink to
            # the base interpreter, which has none of LatentSync's packages.
            [str(_LS_PYTHON.absolute()), str(_WORKER), *args],
            cwd=str(LATENTSYNC_DIR), stdout=log, stderr=subprocess.STDOUT,
        )


def _probe_duration(path: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", path], capture_output=True, text=True)
    return float(out.stdout.strip())


# ── Local Wav2Lip + MediaPipe ─────────────────────────────────────────────────

def _lipsync_local(
    segments: list[Segment],
    to_sync: list[Segment],
    edit_ids: set[int],
    segs_dir: Path,
) -> dict:
    if not WAV2LIP_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Wav2Lip checkpoint missing: {WAV2LIP_CHECKPOINT}\n"
            "git clone https://github.com/Rudrabha/Wav2Lip && "
            "download wav2lip_gan.pth into Wav2Lip/checkpoints/"
        )

    seg_map = {s.id: s for s in segments}
    processed = 0

    for seg in to_sync:
        h = instruction_hash(seg.edit_instruction) if seg.edit_instruction else "default"
        clip_path = segs_dir / f"s{seg.id:04d}_lipsynced_{h}.mp4"

        if not clip_path.exists() or seg.id in edit_ids:
            try:
                _run_wav2lip_mediapipe(seg.original_clip, seg.aligned_audio, str(clip_path))
            except Exception as exc:
                print(f"  ⚠ Wav2Lip failed seg {seg.id}: {exc}. Using original clip.")
                clip_path = Path(seg.original_clip)

        seg_map[seg.id].lip_synced_clip = str(clip_path)
        processed += 1

    return {
        "segments": [s.to_dict() for s in segments],
        "log": [f"✓ Lip-synced {processed} segments (Wav2Lip + MediaPipe)"],
    }


def _run_wav2lip_mediapipe(face_path: str, audio_path: str, output_path: str) -> None:
    """
    Wav2Lip with OpenCV face detection.

    Pipeline:
      1. OpenCV Haar cascade detects face bbox on first reliable frame
      2. Crop face region (+ padding) from all frames → temp video
      3. Run Wav2Lip on cropped face video
      4. Composite synced face crop back into original frames
      5. Re-attach audio
    """
    import cv2
    import soundfile as sf

    # ── Read source frames ────────────────────────────────────────────────────
    cap = cv2.VideoCapture(face_path)
    fps  = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames: list[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        raise RuntimeError(f"Could not read frames from {face_path}")

    # ── MediaPipe face detection ──────────────────────────────────────────────
    bbox = _detect_face(frames, W, H)

    if bbox is None:
        # No face found — run Wav2Lip on full frame (may misplace, but best effort)
        print(f"  ⚠ No face detected in {Path(face_path).name}; using full frame")
        _run_wav2lip_direct(face_path, audio_path, output_path)
        return

    x1, y1, x2, y2 = bbox

    with tempfile.TemporaryDirectory() as tmp:
        crop_path   = os.path.join(tmp, "crop.mp4")
        synced_crop = os.path.join(tmp, "synced_crop.mp4")
        composite   = os.path.join(tmp, "composite.mp4")

        # ── Write face crop video (no audio needed for Wav2Lip face input) ──
        cw, ch = x2 - x1, y2 - y1
        writer = cv2.VideoWriter(
            crop_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (cw, ch)
        )
        for frame in frames:
            writer.write(frame[y1:y2, x1:x2])
        writer.release()

        # ── Run Wav2Lip on the cropped face ──────────────────────────────────
        _run_wav2lip_direct(crop_path, audio_path, synced_crop)

        # ── Read synced crop frames ───────────────────────────────────────────
        cap2 = cv2.VideoCapture(synced_crop)
        synced_frames: list[np.ndarray] = []
        while True:
            ret, f = cap2.read()
            if not ret:
                break
            synced_frames.append(f)
        cap2.release()

        # ── Composite synced crop back into originals ─────────────────────────
        writer2 = cv2.VideoWriter(
            composite, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H)
        )
        for i, orig in enumerate(frames):
            out_frame = orig.copy()
            if i < len(synced_frames):
                patch = cv2.resize(synced_frames[i], (cw, ch))
                out_frame[y1:y2, x1:x2] = patch
            writer2.write(out_frame)
        writer2.release()

        # ── Re-attach dubbed audio to composite video ─────────────────────────
        (
            ffmpeg
            .output(
                ffmpeg.input(composite).video,
                ffmpeg.input(audio_path).audio,
                output_path,
                vcodec="libx264", acodec="aac", strict="experimental",
            )
            .overwrite_output()
            .run(quiet=True)
        )


def _detect_face(
    frames: list, W: int, H: int
) -> tuple[int, int, int, int] | None:
    """
    Detect face bbox with padding.
    Tries MTCNN first (handles sideways/angled faces up to ~90°),
    falls back to OpenCV Haar cascade for frontal faces.
    Returns (x1,y1,x2,y2) or None if no face found in first 20 frames.
    """
    # ── MTCNN (rotation-robust) ───────────────────────────────────────────────
    try:
        from mtcnn import MTCNN
        detector = MTCNN()
        for frame in frames[:20]:
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.detect_faces(rgb)
            if not results:
                continue
            # Pick the highest-confidence detection
            best = max(results, key=lambda r: r["confidence"])
            if best["confidence"] < 0.85:
                continue
            x, y, w, h = best["box"]
            x, y = max(0, x), max(0, y)   # MTCNN can return negatives
            pad_x = int(_FACE_PAD * w)
            pad_y = int(_FACE_PAD * h)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(W, x + w + pad_x)
            y2 = min(H, y + h + pad_y)
            return x1, y1, x2, y2
    except Exception:
        pass  # fall through to Haar

    # ── Haar cascade fallback (frontal faces only) ────────────────────────────
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    for frame in frames[:20]:
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        if len(faces) == 0:
            continue
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        pad_x = int(_FACE_PAD * w)
        pad_y = int(_FACE_PAD * h)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(W, x + w + pad_x)
        y2 = min(H, y + h + pad_y)
        return x1, y1, x2, y2

    return None


def _run_wav2lip_direct(face_path: str, audio_path: str, output_path: str) -> None:
    """Call Wav2Lip inference.py directly (no face pre-crop)."""
    subprocess.run(
        [
            sys.executable, "inference.py",
            "--checkpoint_path", str(WAV2LIP_CHECKPOINT.resolve()),
            "--face",   face_path,
            "--audio",  audio_path,
            "--outfile", output_path,
            "--resize_factor", "2",
            "--wav2lip_batch_size", "64",
            "--nosmooth",
        ],
        check=True,
        cwd=str(WAV2LIP_DIR),
    )


# ── fal.ai cloud backend ──────────────────────────────────────────────────────

def _lipsync_fal(
    segments: list[Segment],
    to_sync: list[Segment],
    edit_ids: set[int],
    segs_dir: Path,
) -> dict:
    import fal_client

    seg_map   = {s.id: s for s in segments}
    processed = 0
    skipped   = 0

    def _fal_one(seg: Segment) -> tuple[int, str | None]:
        h = instruction_hash(seg.edit_instruction) if seg.edit_instruction else "default"
        clip_path = segs_dir / f"s{seg.id:04d}_lipsynced_{h}.mp4"
        if clip_path.exists() and seg.id not in edit_ids:
            return seg.id, str(clip_path)
        try:
            face_url  = fal_client.upload_file(seg.original_clip)
            audio_url = fal_client.upload_file(seg.aligned_audio)
            result    = fal_client.subscribe(
                _FAL_MODEL,
                arguments={"video_url": face_url, "audio_url": audio_url},
            )
            out_url = (
                (result.get("video") or {}).get("url")
                or result.get("output_video")
                or result.get("url")
            )
            if not out_url:
                raise RuntimeError(f"Unexpected fal.ai response: {list(result.keys())}")
            urllib.request.urlretrieve(out_url, str(clip_path))
            return seg.id, str(clip_path)
        except Exception as exc:
            print(f"  ⚠ fal.ai failed seg {seg.id}: {exc}")
            return seg.id, None

    with ThreadPoolExecutor(max_workers=min(len(to_sync), 4)) as pool:
        futures = {pool.submit(_fal_one, seg): seg for seg in to_sync}
        for future in as_completed(futures):
            seg_id, clip_path = future.result()
            if clip_path:
                seg_map[seg_id].lip_synced_clip = clip_path
                processed += 1
            else:
                seg_map[seg_id].lip_synced_clip = seg_map[seg_id].original_clip
                skipped += 1

    note = f"✓ Lip-synced {processed} segments via fal.ai"
    if skipped:
        note += f" ({skipped} fell back to original)"
    return {"segments": [s.to_dict() for s in segments], "log": [note]}


# ── Shared helpers ────────────────────────────────────────────────────────────

def _extract_clip(video_path: str, out_path: str, start: float, end: float) -> None:
    duration = round(end - start, 3)
    (
        ffmpeg
        .input(video_path, ss=start, t=duration)
        .output(out_path, vcodec="libx264", acodec="aac", avoid_negative_ts="make_zero")
        .overwrite_output()
        .run(quiet=True)
    )


# ── Shot planning: sync only shots that show the speaker's own face ──────────

# Same person if face embeddings (ArcFace) are at least this similar.
_FACE_SAME_SIM = float(os.environ.get("FACE_SAME_SIM", "0.40"))
# Share of a shot's samples in which the speaker must be the main frontal face.
_SHOT_FACE_FRAC = float(os.environ.get("SHOT_FACE_FRAC", "0.85"))
_MIN_SHOT_SECS = float(os.environ.get("MIN_SYNC_SHOT_SECS", "0.4"))


def plan_sync_shots(analysis: dict, segments: list[Segment],
                    spans: dict[int, tuple[float, float]]) -> tuple[dict, str]:
    """
    {segment id: [(start, end, face_cx, face_cy), ...]} — the parts of each line's span where
    the camera is on the speaker. Faces are clustered into identities, and
    each identity is matched to the diarized speaker whose lines it moves its
    mouth during (lip motion, not screen time: in shot/reverse-shot the
    listener is on screen about as often as the speaker).
    """
    from sklearn.cluster import AgglomerativeClustering
    from scipy.optimize import linear_sum_assignment

    t, bbox, frontal = analysis["t"], analysis["bbox"], analysis["frontal"]
    mouth, emb, cuts = analysis["mouth"], analysis["emb"], list(analysis["cuts"])
    if len(t) == 0:
        return {}, "no faces in the video"
    area = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])

    # Identities: cluster the reasonably sized faces; tiny background faces
    # are never lip-synced and only add noise.
    big = area >= np.percentile(area, 20)
    ident = np.full(len(t), -1)
    if big.sum() >= 2:
        lab = AgglomerativeClustering(
            n_clusters=None, metric="cosine", linkage="average",
            distance_threshold=1 - _FACE_SAME_SIM).fit_predict(emb[big])
        ident[big] = lab
    elif big.sum() == 1:
        ident[big] = 0

    # Who is speaking at each sample.
    def speaker_at(x: float) -> int | None:
        for s in segments:
            if s.start <= x <= s.end:
                return s.speaker if s.speaker is not None else 0
        return None
    spk_at = np.array([(-1 if (v := speaker_at(x)) is None else v) for x in t])

    # Only faces seen mostly frontal can be lip-synced; the big blurred
    # over-the-shoulder heads in a reverse shot are not candidates (and camera
    # motion makes their "mouth" look busy).
    ids = [i for i in set(ident.tolist())
           if i >= 0 and (ident == i).sum() >= 10 and frontal[ident == i].mean() >= 0.5]
    spks = sorted({s.speaker if s.speaker is not None else 0 for s in segments})
    if not ids:
        return {}, "no recurring faces"

    # Score[id][spk]: how much more this face's mouth moves while spk talks
    # than while anyone else talks.
    score = np.zeros((len(ids), len(spks)))
    for a, i in enumerate(ids):
        m = (ident == i) & np.isfinite(mouth) & frontal
        for b, s in enumerate(spks):
            on, off = m & (spk_at == s), m & (spk_at != s) & (spk_at >= 0)
            if on.sum() >= 3:
                score[a, b] = mouth[on].mean() - (mouth[off].mean() if off.sum() >= 3 else 0)
    rows, cols = linear_sum_assignment(-score)
    face_of = {spks[c]: ids[r] for r, c in zip(rows, cols) if score[r, c] > 0}

    plan: dict[int, list[tuple[float, float]]] = {}
    for seg in segments:
        spk = seg.speaker if seg.speaker is not None else 0
        if seg.id not in spans or spk not in face_of:
            continue
        a, b = spans[seg.id]
        edges = [a] + [c for c in cuts if a < c < b] + [b]
        for x, y in zip(edges, edges[1:]):
            if y - x < _MIN_SHOT_SECS:
                continue
            inside = np.where((t >= x) & (t < y))[0]
            if len(inside) == 0:
                continue
            # Is the speaker's own face on screen, frontal, through the shot?
            mine = inside[(ident[inside] == face_of[spk]) & frontal[inside]]
            n_frames = max(1, int(round((y - x) * float(analysis["fps"]) / int(analysis["step"]))))
            if len(np.unique(t[mine])) >= _SHOT_FACE_FRAC * n_frames:
                b = bbox[mine]
                # Where the face sits, so LatentSync syncs this face and not
                # the larger blurred head of the listener in the foreground.
                cx = float(np.median((b[:, 0] + b[:, 2]) / 2))
                cy = float(np.median((b[:, 1] + b[:, 3]) / 2))
                plan.setdefault(seg.id, []).append((round(float(x), 3), round(float(y), 3), cx, cy))

    note = ", ".join(f"S{s + 1}→face {face_of[s]}" for s in sorted(face_of))
    return plan, f"faces matched by lip motion: {note or 'none'}"
