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

    # LatentSync pieces are overlaid onto the original video at their own
    # times; the soundtrack is always the full-quality mix below.
    synced = [p for s in segments for p in (s.sync_clips or []) if Path(p["path"]).exists()]
    legacy = any(
        seg.lip_synced_clip
        and seg.original_clip
        and seg.lip_synced_clip != seg.original_clip
        for seg in segments
    )

    if legacy and not synced:
        return _assemble_with_lipsync(segments, cache_dir, video_path, state)
    return _assemble_audio_overlay(segments, cache_dir, video_path, state, synced)


# ── Audio-overlay path (no lip sync) ─────────────────────────────────────────

def _assemble_audio_overlay(
    segments: list[Segment], cache_dir: Path, video_path: str, state: PipelineState,
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
    dubs: list[tuple[int, np.ndarray, Segment]] = []
    for seg in segments:
        if not seg.aligned_audio or not Path(seg.aligned_audio).exists():
            continue
        y = _read_mono(seg.aligned_audio, target_sr=sr)
        a = max(0, int(seg.start * sr))
        b = min(total, max(int(seg.end * sr), a + len(y)))
        if a >= b:
            continue
        spans.append((max(0, a - pad), min(total, b + pad)))
        dubs.append((a, y, seg))

    # Song mode: sung lyrics re-sung in the target language (song_mode.py).
    # Their span joins the dialogue mask (original voice → score) and the new
    # vocal goes in at the original singer's level.
    sung_parts = _render_songs(segments, state, cache_dir, sr, total) if bed is not None else []
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

    balanced = _balance_dialogue(mix, dubs, state, sr)
    for a, y, _seg in dubs:
        n = min(len(y), total - a)
        mix[a:a + n] += y[:n, None]
    for a, y in sung_parts:
        n = min(len(y), total - a)
        mix[a:a + n] += y[:n, None]

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
        video_src = _overlay_synced(video_path, synced, cache_dir)
        lip_note = f"; {len(synced)} lip-synced shot(s) composited"
    else:
        video_src = video_path
    vcodec = _video_codec(video_src)
    cmd = [
        "ffmpeg", "-i", video_src, "-i", str(final_audio),
        "-map", "0:v:0", "-map", "1:a:0",
        # Browsers play H.264 + AAC everywhere; anything else is re-encoded.
        *(["-c:v", "copy"] if vcodec == "h264" else
          ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p"]),
        "-c:a", "copy",
        # Index at the front so the player starts at once and can seek
        # without downloading the whole file first.
        "-movflags", "+faststart", "-shortest",
        str(final_path), "-y",
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    note = (
        f"✓ Assembled {dubbed_count} lines at {sr/1000:.1f} kHz stereo; original "
        f"voice {how} across {ducked_s:.1f}s ({len(merged)} span(s)), "
        f"{total/sr - ducked_s:.1f}s of original audio untouched"
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


def _render_songs(segments: list[Segment], state: PipelineState, cache_dir: Path,
                  sr: int, total: int) -> list:
    """[(start sample, mono vocal)] for every sung piece song mode could render."""
    import numpy as np
    from .. import song_mode
    sung = [s for s in segments if s.sung and s.keep_original and s.dubbed_text]
    if not sung or not song_mode.available() or not state.get("vocals_path"):
        return []
    voc = _read_mono(state["vocals_path"], target_sr=sr)
    out = []
    for seg in sung:
        y = song_mode.render(seg, state, cache_dir, state["target_lang"].split("-")[0], sr)
        if y is None or not len(y):
            continue
        a = int(seg.start * sr)
        ref = voc[a:a + len(y)]
        g = np.sqrt(np.mean(ref ** 2) + 1e-12) / (np.sqrt(np.mean(y ** 2)) + 1e-9)
        out.append((a, (y * min(g, 8.0)).astype(np.float32)))
    return out


def _balance_dialogue(mix, dubs: list, state: PipelineState, sr: int) -> int:
    """
    Keep every dubbed line as clear over the background as the original
    voice was (and at least _MIN_MARGIN dB clear). A dub laid at its own
    level was buried under a trailer's score in action scenes. Where a line
    falls short, the dub is raised (≤ _MAX_LIFT dB) and the background is
    dipped just around it (≤ _MAX_DUCK dB, 100 ms ramps). Lines already clear
    — speech over quiet rooms — are untouched. Edits `dubs` gains in place,
    `mix` (background) in place; returns lines changed.
    """
    import numpy as np
    vocals_path = state.get("vocals_path")
    if not vocals_path or not Path(vocals_path).exists() or not dubs:
        return 0
    voc = _read_mono(vocals_path, target_sr=sr)
    bg = mix.mean(axis=1)
    hop = int(0.05 * sr)

    def frames_db(x):
        n = len(x) // hop
        if n < 3:
            return None
        r = np.sqrt((x[:n * hop].reshape(n, hop) ** 2).mean(1)) + 1e-9
        return 20 * np.log10(r)

    duck = np.zeros(len(mix), dtype=np.float32)        # dB of dip wanted
    changed = 0
    for k, (a, y, seg) in enumerate(dubs):
        b = min(len(mix), a + len(y))
        ov = frames_db(voc[int(seg.start * sr):int(seg.end * sr)])
        ob = frames_db(bg[int(seg.start * sr):int(seg.end * sr)])
        dv, db = frames_db(y[:b - a]), frames_db(bg[a:b])
        if ov is None or ob is None or dv is None or db is None:
            continue
        orig_margin = np.percentile(ov, 90) - np.median(ob)
        dub_margin = np.percentile(dv, 90) - np.median(db)
        short = max(orig_margin + 1.0, _MIN_MARGIN) - dub_margin
        if short <= 0.5:
            continue
        lift = min(short, _MAX_LIFT)
        y *= 10 ** (lift / 20)
        dip = min(short - lift, _MAX_DUCK)
        if dip > 0.5:
            p = int(0.15 * sr)
            duck[max(0, a - p):min(len(mix), b + p)] = np.maximum(
                duck[max(0, a - p):min(len(mix), b + p)], dip)
        changed += 1
        dubs[k] = (a, y, seg)
    if duck.any():
        ramp = int(0.1 * sr)
        kernel = np.ones(ramp, dtype=np.float32) / ramp
        duck = np.convolve(duck, kernel, mode="same")
        mix *= (10 ** (-duck / 20))[:, None]
    return changed


# Dialogue over the background (see _balance_dialogue).
_MIN_MARGIN = float(os.environ.get("DIALOGUE_MIN_MARGIN_DB", "8"))
_MAX_LIFT = float(os.environ.get("DIALOGUE_MAX_LIFT_DB", "6"))
_MAX_DUCK = float(os.environ.get("BED_MAX_DUCK_DB", "8"))


# How many lip-synced clips go into one ffmpeg overlay pass.
_OVERLAY_BATCH = 24


def _overlay_synced(video_path: str, synced: list[dict], cache_dir: Path) -> str:
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
    batches = [synced[i:i + _OVERLAY_BATCH] for i in range(0, len(synced), _OVERLAY_BATCH)]
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
                f"setpts=PTS-STARTPTS+{a:.3f}/TB[c{k}]")
            parts.append(
                f"[{last}][c{k}]overlay=eof_action=pass:"
                f"enable='between(t,{a:.3f},{b:.3f})'[v{k}]")
            last = f"v{k}"
        cmd += ["-filter_complex", ";".join(parts), "-map", f"[{last}]", "-an",
                "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-pix_fmt", "yuv420p", str(out), "-y"]
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
    from .. import frame_interp
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
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         f"stream={entries}", "-of", "default=nw=1:nk=1", path],
        capture_output=True, text=True,
    ).stdout.strip()


def _video_codec(path: str) -> str:
    return _ffprobe_stream(path, "codec_name")


def _video_fps(path: str) -> str:
    return _ffprobe_stream(path, "r_frame_rate") or "25"


def _video_size(path: str) -> tuple[int, int]:
    w, h = _ffprobe_stream(path, "width,height").split()[:2]
    return int(w), int(h)


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
