"""
Older lip-sync backends, kept selectable through LIPSYNC_BACKEND.

  wav2lip  local Wav2Lip with MediaPipe face detection (WAV2LIP_USE_LOCAL=1;
           needs a Wav2Lip checkout and wav2lip_gan.pth)
  fal      fal.ai cloud "video-retalking" (FAL_KEY + FAL_ENABLED=1)

LatentSync (steps/lipsync.py) is the default and gives much better results.
"""

import json
import os
import subprocess
import sys
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import ffmpeg
import numpy as np

from ..cache import get_segments_dir, instruction_hash
from ..state import PipelineState, Segment


# ── Backend flags (evaluated at call time) ────────────────────────────────────
def _fal_key() -> str | None:
    return os.environ.get("FAL_KEY") or None


_USE_LOCAL = os.environ.get("WAV2LIP_USE_LOCAL", "0") == "1"


_FAL_ENABLED = os.environ.get("FAL_ENABLED", "0") == "1"


WAV2LIP_DIR = Path("Wav2Lip")


_ckpt_pth = WAV2LIP_DIR / "checkpoints" / "wav2lip_gan.pth"


_ckpt_pt = WAV2LIP_DIR / "checkpoints" / "wav2lip_gan.pt"


WAV2LIP_CHECKPOINT = _ckpt_pth if _ckpt_pth.exists() else _ckpt_pt


_FAL_MODEL = "fal-ai/video-retalking"


# Padding around detected face bbox (fraction of bbox size)
_FACE_PAD = 0.35


def lipsync_wav2lip(
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

    # ── Read source frames ────────────────────────────────────────────────────
    cap = cv2.VideoCapture(face_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
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
        crop_path = os.path.join(tmp, "crop.mp4")
        synced_crop = os.path.join(tmp, "synced_crop.mp4")
        composite = os.path.join(tmp, "composite.mp4")

        # ── Write face crop video (no audio needed for Wav2Lip face input) ──
        cw, ch = x2 - x1, y2 - y1
        writer = cv2.VideoWriter(crop_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (cw, ch))
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
        writer2 = cv2.VideoWriter(composite, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        for i, orig in enumerate(frames):
            out_frame = orig.copy()
            if i < len(synced_frames):
                patch = cv2.resize(synced_frames[i], (cw, ch))
                out_frame[y1:y2, x1:x2] = patch
            writer2.write(out_frame)
        writer2.release()

        # ── Re-attach dubbed audio to composite video ─────────────────────────
        (
            ffmpeg.output(
                ffmpeg.input(composite).video,
                ffmpeg.input(audio_path).audio,
                output_path,
                vcodec="libx264",
                acodec="aac",
                strict="experimental",
            )
            .overwrite_output()
            .run(quiet=True)
        )


def _detect_face(frames: list, W: int, H: int) -> tuple[int, int, int, int] | None:
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
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.detect_faces(rgb)
            if not results:
                continue
            # Pick the highest-confidence detection
            best = max(results, key=lambda r: r["confidence"])
            if best["confidence"] < 0.85:
                continue
            x, y, w, h = best["box"]
            x, y = max(0, x), max(0, y)  # MTCNN can return negatives
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
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
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
            sys.executable,
            "inference.py",
            "--checkpoint_path",
            str(WAV2LIP_CHECKPOINT.resolve()),
            "--face",
            face_path,
            "--audio",
            audio_path,
            "--outfile",
            output_path,
            "--resize_factor",
            "2",
            "--wav2lip_batch_size",
            "64",
            "--nosmooth",
        ],
        check=True,
        cwd=str(WAV2LIP_DIR),
    )


def lipsync_fal(
    segments: list[Segment],
    to_sync: list[Segment],
    edit_ids: set[int],
    segs_dir: Path,
) -> dict:
    import fal_client

    seg_map = {s.id: s for s in segments}
    processed = 0
    skipped = 0

    def _fal_one(seg: Segment) -> tuple[int, str | None]:
        h = instruction_hash(seg.edit_instruction) if seg.edit_instruction else "default"
        clip_path = segs_dir / f"s{seg.id:04d}_lipsynced_{h}.mp4"
        if clip_path.exists() and seg.id not in edit_ids:
            return seg.id, str(clip_path)
        try:
            face_url = fal_client.upload_file(seg.original_clip)
            audio_url = fal_client.upload_file(seg.aligned_audio)
            result = fal_client.subscribe(
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


# ── Assembly for these backends: stitch whole clips end to end ───────────────


def assemble_with_lipsync(
    segments: list[Segment], cache_dir: Path, video_path: str, state: PipelineState
) -> dict:
    """
    Stitch lip-synced clips + original gap clips.
    Re-encode everything to a consistent format so concat timestamps are stable.
    """
    segs_dir = get_segments_dir(state["video_id"])
    total_duration = _video_duration(video_path)
    clips: list[str] = []
    prev_end = 0.0

    for seg in segments:
        video_clip = seg.lip_synced_clip or seg.original_clip
        if not video_clip or not Path(video_clip).exists():
            continue

        if seg.start > prev_end + 0.05:
            gap = segs_dir / f"gap_{prev_end:.3f}_{seg.start:.3f}.mp4"
            if not gap.exists():
                _extract_reencoded(video_path, str(gap), prev_end, seg.start)
            if gap.exists():
                clips.append(str(gap))

        if seg.aligned_audio and Path(seg.aligned_audio).exists():
            out = segs_dir / f"s{seg.id:04d}_final.mp4"
            if not out.exists():
                _replace_audio(video_clip, seg.aligned_audio, str(out))
            clips.append(str(out))
        else:
            clips.append(video_clip)

        prev_end = seg.end

    if total_duration > prev_end + 0.05:
        tail = segs_dir / f"gap_{prev_end:.3f}_{total_duration:.3f}.mp4"
        if not tail.exists():
            _extract_reencoded(video_path, str(tail), prev_end, total_duration)
        if tail.exists():
            clips.append(str(tail))

    concat_file = cache_dir / "concat_list.txt"
    with open(concat_file, "w") as f:
        for clip in clips:
            f.write(f"file '{Path(clip).resolve()}'\n")

    final_path = cache_dir / f"dubbed_{state['target_lang']}.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            str(final_path),
            "-y",
        ],
        check=True,
        capture_output=True,
    )

    return {
        "final_video": str(final_path),
        "log": state.get("log", []) + [f"✓ Assembled {len(clips)} clips (lip-sync mode)"],
    }


def _video_duration(video_path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", video_path],
        capture_output=True,
        text=True,
    )
    return float(json.loads(result.stdout)["format"]["duration"])


def _extract_reencoded(video_path: str, out_path: str, start: float, end: float) -> None:
    """Extract with output-seek (accurate) and re-encode for stable timestamps."""
    duration = round(end - start, 3)
    subprocess.run(
        [
            "ffmpeg",
            "-i",
            video_path,
            "-ss",
            str(start),
            "-t",
            str(duration),
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-avoid_negative_ts",
            "make_zero",
            out_path,
            "-y",
        ],
        check=True,
        capture_output=True,
    )


def _replace_audio(video_path: str, audio_path: str, out_path: str) -> None:
    (
        ffmpeg.output(
            ffmpeg.input(video_path).video,
            ffmpeg.input(audio_path).audio,
            out_path,
            vcodec="libx264",
            acodec="aac",
            strict="experimental",
        )
        .overwrite_output()
        .run(quiet=True)
    )
