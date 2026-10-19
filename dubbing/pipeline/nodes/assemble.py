"""
Assembly node.

Strategy (no lip sync):
  Keep original video completely untouched.
  Build a single dubbed audio track using pydub:
    - Start with full original audio (preserves applause, gaps, ambient)
    - Duck original speech -18 dB at each speech segment (kills original voice,
      keeps ambient/background sounds)
    - Overlay dubbed TTS at each segment position
  Apply mixed audio to original video with one FFmpeg pass (video copy = no re-encode).

Strategy (with lip sync):
  Wav2Lip produces lip-synced clips per segment.
  Stitch those clips + original gap clips with re-encoding for consistent timestamps.
"""

import json
import subprocess
from pathlib import Path
from pydub import AudioSegment
import ffmpeg
from ..state import PipelineState, Segment
from ..cache import get_cache_dir, get_segments_dir


def assemble_node(state: PipelineState) -> dict:
    segments = sorted(
        [Segment.from_dict(s) for s in state["segments"]], key=lambda s: s.start
    )
    cache_dir = get_cache_dir(state["video_id"])
    video_path = state["video_path"]

    # Check if real lip sync was applied (clip differs from original)
    has_lipsync = any(
        seg.lip_synced_clip
        and seg.original_clip
        and seg.lip_synced_clip != seg.original_clip
        for seg in segments
    )

    if has_lipsync:
        return _assemble_with_lipsync(segments, cache_dir, video_path, state)
    else:
        return _assemble_audio_overlay(segments, cache_dir, video_path, state)


# ── Audio-overlay path (no lip sync) ─────────────────────────────────────────

def _assemble_audio_overlay(
    segments: list[Segment], cache_dir: Path, video_path: str, state: PipelineState
) -> dict:
    """
    Keep original video intact. Build dubbed audio by overlaying TTS onto
    the original audio track, ducking the original speech during dubbed segments.
    """
    audio_path = state["audio_path"]
    original = AudioSegment.from_file(audio_path)
    result = original

    dubbed_count = 0
    for seg in segments:
        if not seg.aligned_audio or not Path(seg.aligned_audio).exists():
            continue

        start_ms = int(seg.start * 1000)
        end_ms   = min(int(seg.end * 1000), len(result))
        if start_ms >= end_ms:
            continue

        # Duck original speech: -18 dB kills voice, leaves ambient/music audible
        result = (
            result[:start_ms]
            + result[start_ms:end_ms].apply_gain(-18)
            + result[end_ms:]
        )

        # Overlay dubbed TTS at this position
        dubbed = AudioSegment.from_file(seg.aligned_audio)
        result = result.overlay(dubbed, position=start_ms)
        dubbed_count += 1

    mixed_audio = cache_dir / "mixed_audio.wav"
    result.export(str(mixed_audio), format="wav")

    final_path = cache_dir / f"dubbed_{state['target_lang']}.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-i", video_path,
            "-i", str(mixed_audio),
            "-c:v", "copy",        # original video untouched — no re-encode
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0",
            str(final_path), "-y",
        ],
        check=True,
        capture_output=True,
    )

    return {
        "final_video": str(final_path),
        "log": state.get("log", []) + [
            f"✓ Assembled: {dubbed_count} segments dubbed over original audio"
        ],
    }


# ── Lip-sync path ─────────────────────────────────────────────────────────────

def _assemble_with_lipsync(
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
        ["ffmpeg", "-f", "concat", "-safe", "0",
         "-i", str(concat_file), "-c:v", "libx264", "-c:a", "aac",
         str(final_path), "-y"],
        check=True, capture_output=True,
    )

    return {
        "final_video": str(final_path),
        "log": state.get("log", []) + [f"✓ Assembled {len(clips)} clips (lip-sync mode)"],
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _video_duration(video_path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", video_path],
        capture_output=True, text=True,
    )
    return float(json.loads(result.stdout)["format"]["duration"])


def _extract_reencoded(video_path: str, out_path: str, start: float, end: float) -> None:
    """Extract with output-seek (accurate) and re-encode for stable timestamps."""
    duration = round(end - start, 3)
    subprocess.run(
        ["ffmpeg", "-i", video_path,
         "-ss", str(start), "-t", str(duration),
         "-c:v", "libx264", "-c:a", "aac",
         "-avoid_negative_ts", "make_zero",
         out_path, "-y"],
        check=True, capture_output=True,
    )


def _replace_audio(video_path: str, audio_path: str, out_path: str) -> None:
    (
        ffmpeg
        .output(
            ffmpeg.input(video_path).video,
            ffmpeg.input(audio_path).audio,
            out_path,
            vcodec="libx264", acodec="aac", strict="experimental",
        )
        .overwrite_output()
        .run(quiet=True)
    )
