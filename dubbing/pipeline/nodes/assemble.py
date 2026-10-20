"""
Assembly node.

Strategy (no lip sync):
  Keep the original video untouched (stream copy, no re-encode) and rebuild
  the soundtrack at full quality: original audio outside dialogue, the Demucs
  background bed inside it (crossfaded), dubbed lines on top, then loudness
  normalisation. See _assemble_audio_overlay.

Strategy (with lip sync):
  Wav2Lip produces lip-synced clips per segment.
  Stitch those clips + original gap clips with re-encoding for consistent timestamps.
"""

import json
import os
import subprocess
from pathlib import Path
from pydub import AudioSegment
import ffmpeg
from ..state import PipelineState, Segment
from ..cache import get_cache_dir, get_segments_dir

# How far to duck the original speech under the dub. -18 dB still leaves the
# original voice clearly audible underneath the dub, so the default is deep
# enough to remove it. Set DUCK_DB=-100 for literal silence, or raise it
# (e.g. -25) to keep a trace of the original performance.
_DUCK_DB = float(os.environ.get("DUCK_DB", "-45"))

# Padding either side of a dubbed span. Whisper's VAD clips the attack and
# tail of speech, so ducking exactly [start, end] leaves audible fragments.
_DUCK_PAD = float(os.environ.get("DUCK_PAD", "0.25"))

# Gaps between dubbed spans shorter than this are ducked too. A sub-second
# gap mid-sentence is trimmed speech, not ambience worth keeping.
_BRIDGE_GAP = float(os.environ.get("DUCK_BRIDGE_GAP", "1.2"))

# Crossfade between original audio and the background bed at each dialogue
# boundary. Short enough to be inaudible, long enough to avoid a click.
_FADE_S = float(os.environ.get("BED_FADE_SECS", "0.01"))

# Broadcast-style loudness target. Without this, output level drifts with
# whatever the TTS happened to return and sounds amateurish next to the source.
_LOUDNORM = os.environ.get("LOUDNORM", "1") == "1"
_LUFS_TARGET = os.environ.get("LUFS_TARGET", "-14")   # streaming standard
_TRUE_PEAK   = os.environ.get("TRUE_PEAK", "-1.0")    # dBTP ceiling


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
    Keep the video untouched; rebuild the soundtrack.

    Outside dialogue: the original audio, untouched.
    Inside dialogue:  the Demucs background bed (music, applause, room tone,
                      no voice), crossfaded in over _FADE_S — so the original
                      voice is removed outright, not just turned down.
    Then the dubbed lines are laid on top. All at full quality (44.1 kHz
    stereo); the 16 kHz mono ASR track is never used for output.

    Without a bed (SEPARATE=none or Demucs failed) it falls back to ducking
    the original by DUCK_DB inside dialogue.
    """
    import numpy as np
    import soundfile as sf

    base_path = state.get("audio_hq_path")
    if not base_path or not Path(base_path).exists():
        base_path = state["audio_path"]
    base, sr = _read_stereo(base_path)
    total = len(base)

    bed = None
    bed_path = state.get("background_path")
    if bed_path and Path(bed_path).exists():
        bed = _fit_length(_read_stereo(bed_path, target_sr=sr)[0], total)

    # ── Dialogue spans ───────────────────────────────────────────────────────
    # Padded either side because Whisper's VAD clips word attacks and tails,
    # and bridged across short gaps because a sub-second gap mid-sentence is
    # trimmed speech, not ambience.
    pad = int(_DUCK_PAD * sr)
    spans: list[tuple[int, int]] = []
    dubs: list[tuple[int, np.ndarray]] = []
    for seg in segments:
        if not seg.aligned_audio or not Path(seg.aligned_audio).exists():
            continue
        y = _read_mono(seg.aligned_audio, target_sr=sr)
        a = max(0, int(seg.start * sr))
        b = min(total, max(int(seg.end * sr), a + len(y)))
        if a >= b:
            continue
        spans.append((max(0, a - pad), min(total, b + pad)))
        dubs.append((a, y))

    merged: list[list[int]] = []
    for s, e in sorted(spans):
        if merged and s - merged[-1][1] <= int(_BRIDGE_GAP * sr):
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])

    # ── Crossfaded dialogue mask: 0 = original, 1 = bed ──────────────────────
    mask = np.zeros(total, dtype=np.float32)
    fade = int(_FADE_S * sr)
    for s, e in merged:
        env = np.ones(e - s, dtype=np.float32)
        f = min(fade, (e - s) // 2)
        if f > 0:
            env[:f] = np.linspace(0.0, 1.0, f, dtype=np.float32)
            env[-f:] = np.linspace(1.0, 0.0, f, dtype=np.float32)
        mask[s:e] = np.maximum(mask[s:e], env)
    m = mask[:, None]

    if bed is not None:
        mix = base * (1.0 - m) + bed * m
        how = "replaced by the Demucs background bed"
    else:
        g = 10 ** (_DUCK_DB / 20)
        mix = base * (1.0 - m * (1.0 - g))
        how = f"ducked {_DUCK_DB:.0f} dB (no background bed available)"

    for a, y in dubs:
        n = min(len(y), total - a)
        mix[a:a + n] += y[:n, None]

    dubbed_count = len(dubs)
    ducked_s = float(mask.sum()) / sr

    mixed_audio = cache_dir / "mixed_audio.wav"
    sf.write(str(mixed_audio), mix, sr, subtype="FLOAT")

    final_path = cache_dir / f"dubbed_{state['target_lang']}.mp4"
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-i", str(mixed_audio),
        "-c:v", "copy",        # original video untouched — no re-encode
        "-c:a", "aac", "-b:a", "192k",
        "-map", "0:v:0",
        "-map", "1:a:0",
    ]
    if _LOUDNORM:
        cmd += ["-af", f"loudnorm=I={_LUFS_TARGET}:TP={_TRUE_PEAK}:LRA=11"]
    cmd += [str(final_path), "-y"]

    subprocess.run(cmd, check=True, capture_output=True)

    note = (
        f"✓ Assembled {dubbed_count} lines at {sr/1000:.1f} kHz stereo; original "
        f"voice {how} across {ducked_s:.1f}s ({len(merged)} span(s)), "
        f"{total/sr - ducked_s:.1f}s of original audio untouched"
    )
    if _LOUDNORM:
        note += f"; normalized to {_LUFS_TARGET} LUFS"

    return {
        "final_video": str(final_path),
        "log": state.get("log", []) + [note],
    }


def _read_stereo(path: str, target_sr: int | None = None):
    import numpy as np
    import soundfile as sf
    y, sr = sf.read(path, dtype="float32", always_2d=True)
    if y.shape[1] == 1:
        y = np.repeat(y, 2, axis=1)
    elif y.shape[1] > 2:
        y = y[:, :2]
    if target_sr and sr != target_sr:
        import librosa
        y = librosa.resample(y.T, orig_sr=sr, target_sr=target_sr).T.astype(np.float32)
        sr = target_sr
    return y, sr


def _read_mono(path: str, target_sr: int):
    import soundfile as sf
    y, sr = sf.read(path, dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    if sr != target_sr:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    return y


def _fit_length(y, n: int):
    import numpy as np
    if len(y) >= n:
        return y[:n]
    return np.pad(y, ((0, n - len(y)), (0, 0)))


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
