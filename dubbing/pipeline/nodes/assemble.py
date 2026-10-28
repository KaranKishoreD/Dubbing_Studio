"""
Stage 10 — Assemble: the final soundtrack and the final video.

The original video is kept; only the soundtrack is rebuilt, at full quality
(44.1 kHz stereo):

  1. Outside dialogue: the original audio, untouched.
  2. Inside dialogue: the background track from Demucs (music, effects, room
     tone, no voice), crossfaded in, so the original voice is removed.
  3. Song mode only: re-sung lyrics go in where the original singing was.
  4. Each dubbed line is placed at its time, lifted over loud music when
     needed (audio/mixing.py).
  5. Loudness is normalised to -14 LUFS and saved as AAC.
  6. Lip-synced shots are laid over the original video (video/compose.py),
     and video + new soundtrack are joined into the final MP4.

The old Wav2Lip/fal backends produce whole lip-synced clips instead; they
are stitched together by video/legacy_lipsync.py.
"""

import os
import subprocess
from pathlib import Path

from ..audio.mixing import balance_dialogue, fit_length, read_mono, read_stereo, render_songs
from ..cache import get_cache_dir
from ..state import PipelineState, Segment
from ..video.compose import overlay_synced, video_codec
from ..video.legacy_lipsync import assemble_with_lipsync

# Only used when there is no background track: how far the original speech
# is turned down under the dub. -45 dB removes it; -18 dB would still leave
# it clearly audible. -100 is silence, -25 keeps a trace of the original.
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
# whatever level the TTS returned.
_LOUDNORM = os.environ.get("LOUDNORM", "1") == "1"
_LUFS_TARGET = os.environ.get("LUFS_TARGET", "-14")  # streaming standard
_TRUE_PEAK = os.environ.get("TRUE_PEAK", "-1.0")  # dBTP ceiling


def assemble_node(state: PipelineState) -> dict:
    segments = sorted([Segment.from_dict(s) for s in state["segments"]], key=lambda s: s.start)
    cache_dir = get_cache_dir(state["video_id"])
    video_path = state["video_path"]

    # LatentSync pieces are overlaid onto the original video at their own
    # times; the soundtrack is always the full-quality mix below.
    synced = [p for s in segments for p in (s.sync_clips or []) if Path(p["path"]).exists()]
    legacy = any(
        seg.lip_synced_clip and seg.original_clip and seg.lip_synced_clip != seg.original_clip
        for seg in segments
    )

    if legacy and not synced:
        return assemble_with_lipsync(segments, cache_dir, video_path, state)
    return _assemble_audio_overlay(segments, cache_dir, video_path, state, synced)


def _assemble_audio_overlay(
    segments: list[Segment],
    cache_dir: Path,
    video_path: str,
    state: PipelineState,
    synced: list[dict] | None = None,
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
    base, sr = read_stereo(base_path)
    total = len(base)

    bed = None
    bed_path = state.get("background_path")
    if bed_path and Path(bed_path).exists():
        bed = fit_length(read_stereo(bed_path, target_sr=sr)[0], total)

    # ── Dialogue spans ───────────────────────────────────────────────────────
    # Padded either side because Whisper's VAD clips word attacks and tails,
    # and bridged across short gaps because a sub-second gap mid-sentence is
    # trimmed speech, not ambience.
    pad = int(_DUCK_PAD * sr)
    spans: list[tuple[int, int]] = []
    dubs: list[tuple[int, np.ndarray, Segment]] = []
    for seg in segments:
        if not seg.aligned_audio or not Path(seg.aligned_audio).exists():
            continue
        y = read_mono(seg.aligned_audio, target_sr=sr)
        a = max(0, int(seg.start * sr))
        b = min(total, max(int(seg.end * sr), a + len(y)))
        if a >= b:
            continue
        spans.append((max(0, a - pad), min(total, b + pad)))
        dubs.append((a, y, seg))

    # Song mode: sung lyrics re-sung in the target language (song_mode.py).
    # Their span joins the dialogue mask (original voice → score) and the new
    # vocal goes in at the original singer's level.
    sung_parts = render_songs(segments, state, cache_dir, sr, total) if bed is not None else []
    for a, y in sung_parts:
        spans.append((max(0, a - int(0.03 * sr)), min(total, a + len(y) + int(0.03 * sr))))

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

    balanced = balance_dialogue(mix, dubs, state, sr)
    for a, y, _seg in dubs:
        n = min(len(y), total - a)
        mix[a : a + n] += y[:n, None]
    for a, y in sung_parts:
        n = min(len(y), total - a)
        mix[a : a + n] += y[:n, None]

    dubbed_count = len(dubs)
    ducked_s = float(mask.sum()) / sr

    mixed_audio = cache_dir / "mixed_audio.wav"
    sf.write(str(mixed_audio), mix, sr, subtype="FLOAT")

    # The soundtrack on its own, as AAC: plays in any browser, so the UI can
    # offer it directly instead of only inside the video.
    lang = state["target_lang"]
    final_audio = cache_dir / f"dubbed_{lang}.m4a"
    acmd = ["ffmpeg", "-i", str(mixed_audio)]
    if _LOUDNORM:
        acmd += ["-af", f"loudnorm=I={_LUFS_TARGET}:TP={_TRUE_PEAK}:LRA=11"]
    acmd += ["-ar", "48000", "-c:a", "aac", "-b:a", "192k", str(final_audio), "-y"]
    subprocess.run(acmd, check=True, capture_output=True)

    final_path = cache_dir / f"dubbed_{lang}.mp4"
    lip_note = ""
    if synced:
        video_src = overlay_synced(video_path, synced, cache_dir)
        lip_note = f"; {len(synced)} lip-synced shot(s) composited"
    else:
        video_src = video_path
    vcodec = video_codec(video_src)
    cmd = [
        "ffmpeg",
        "-i",
        video_src,
        "-i",
        str(final_audio),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        # Browsers play H.264 + AAC everywhere; anything else is re-encoded.
        *(
            ["-c:v", "copy"]
            if vcodec == "h264"
            else ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p"]
        ),
        "-c:a",
        "copy",
        # Index at the front so the player starts at once and can seek
        # without downloading the whole file first.
        "-movflags",
        "+faststart",
        "-shortest",
        str(final_path),
        "-y",
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    note = (
        f"✓ Assembled {dubbed_count} lines at {sr / 1000:.1f} kHz stereo; original "
        f"voice {how} across {ducked_s:.1f}s ({len(merged)} span(s)), "
        f"{total / sr - ducked_s:.1f}s of original audio untouched"
    )
    if balanced:
        note += f"; {balanced} line(s) lifted over a loud score (dialogue kept clear)"
    if sung_parts:
        note += f"; {len(sung_parts)} sung phrase(s) re-sung in the target language"
    if _LOUDNORM:
        note += f"; normalized to {_LUFS_TARGET} LUFS"
    note += lip_note

    return {
        "final_video": str(final_path),
        "final_audio": str(final_audio),
        "log": state.get("log", []) + [note],
    }
