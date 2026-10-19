import shutil
import subprocess
import sys
import soundfile as sf
from pathlib import Path
from ..state import PipelineState, Segment
from ..cache import get_segments_dir, instruction_hash

# If ratio is within this tolerance, skip stretching (avoids audible artifacts)
_TOLERANCE = 0.12


def align_node(state: PipelineState) -> dict:
    """Stretch/compress dubbed audio to match original segment duration."""
    segments = [Segment.from_dict(s) for s in state["segments"]]
    edit_ops = state.get("edit_ops", [])
    edit_ids = {sid for op in edit_ops for sid in op["segment_ids"]}
    segs_dir = get_segments_dir(state["video_id"])
    processed = 0

    for seg in segments:
        if seg.dubbed_audio is None:
            continue

        h = instruction_hash(seg.edit_instruction) if seg.edit_instruction else "default"
        aligned_path = segs_dir / f"s{seg.id:04d}_aligned_{h}.wav"

        if aligned_path.exists() and seg.id not in edit_ids:
            seg.aligned_audio = str(aligned_path)
            continue

        original_duration = seg.end - seg.start
        audio, sr = sf.read(seg.dubbed_audio)
        dubbed_duration = len(audio) / sr
        ratio = dubbed_duration / original_duration  # >1 means dubbed is longer → speed up

        if abs(ratio - 1.0) <= _TOLERANCE:
            shutil.copy(seg.dubbed_audio, str(aligned_path))
        else:
            try:
                import pyrubberband as pyrb
                stretched = pyrb.time_stretch(audio, sr, rate=ratio)
                sf.write(str(aligned_path), stretched, sr)
            except (ImportError, RuntimeError):
                _ffmpeg_atempo(seg.dubbed_audio, str(aligned_path), ratio)

        seg.aligned_audio = str(aligned_path)
        processed += 1

    return {
        "segments": [s.to_dict() for s in segments],
        "log": state.get("log", []) + [f"✓ Duration-aligned {processed} segments"],
    }


def _ffmpeg_atempo(input_path: str, output_path: str, ratio: float) -> None:
    # atempo supports 0.5–2.0; chain two filters for extreme ratios
    if ratio < 0.5:
        tempo_filter = f"atempo=0.5,atempo={ratio / 0.5:.4f}"
    elif ratio > 2.0:
        tempo_filter = f"atempo=2.0,atempo={ratio / 2.0:.4f}"
    else:
        tempo_filter = f"atempo={ratio:.4f}"

    subprocess.run(
        ["ffmpeg", "-i", input_path, "-filter:a", tempo_filter, output_path, "-y"],
        check=True,
        capture_output=True,
    )
