"""
Stage 8 — Align: fit each dubbed line into the time the speaker talks.

  1. Re-voice every line in the original speaker's own voice with Seed-VC
     (voice/voice_clone.py). If that is unavailable the TTS voice is kept.
  2. Trim the silence TTS engines leave at both ends of a clip.
  3. Work out the line's slot: from its start until just before the next
     line, borrowing up to GAP_BORROW_MAX seconds of following silence.
     Two dubbed lines never overlap.
  4. need = clip length / slot length
       need <= 1 and the clip is much shorter than the original
                     -> slow it down gently (not below MIN_RATE), so the
                        voice does not stop while the mouth still moves
       need <= 1 otherwise -> leave it as is
       need > 1      -> speed it up by exactly what it needs. Words are
                        never cut off; lines needing a lot were already
                        rewritten shorter in the synthesize stage.
  5. Smooth the tempo between close lines of the same speaker, so one fast
     line between two normal ones does not jump out.

The output file name includes the source clip name and the rate, so a
re-cast or re-translated line is always aligned again.
"""

import os
import subprocess

import numpy as np
import soundfile as sf

from ..cache import get_segments_dir
from ..state import PipelineState, Segment

# Speed-up the ear does not notice. Lines needing more are rewritten shorter
# and re-voiced by synthesize_node first, so beyond this is rare.
MAX_STRETCH = float(os.environ.get("MAX_STRETCH", "1.10"))
# Lines that need more speed-up than this are still sped up (never cut),
# but are reported in the log, since the change is audible.
HARD_MAX_STRETCH = float(os.environ.get("HARD_MAX_STRETCH", "1.25"))
MIN_RATE = float(os.environ.get("MIN_RATE", "0.90"))  # slowest a short line is played
GAP_GUARD = float(os.environ.get("GAP_GUARD_SECS", "0.08"))  # silence kept before the next line
# How far a dub may run past the end of the speaker's line into silence.
# Much more than a second keeps the voice going after the mouth has closed,
# or into the next shot.
GAP_BORROW_MAX = float(os.environ.get("GAP_BORROW_MAX_SECS", "1.0"))
_UNDERRUN = 0.95  # a clip under 95% of the original line is slowed
_EDGE_FADE_MS = 15  # fade in/out on every clip, so none starts with a click
# Speed smoothing between close lines of one speaker (see _smooth_rates).
_MAX_JUMP = float(os.environ.get("MAX_SPEED_JUMP", "0.15"))
_SMOOTH_GAP = float(os.environ.get("SMOOTH_GAP_SECS", "1.0"))
_SMOOTH_MIN_FILL = 0.85
_TRIM_TOP_DB = 40  # quieter than this (dB below peak) counts as silence


def slot_seconds(segments: list[Segment], i: int, total: float | None) -> float:
    """Usable time for segments[i] (sorted by start)."""
    seg = segments[i]
    if i + 1 < len(segments):
        limit = segments[i + 1].start - GAP_GUARD
    else:
        limit = total if total else seg.end + GAP_BORROW_MAX
    end = min(limit, seg.end + GAP_BORROW_MAX)
    return max(seg.duration, end - seg.start)


def trimmed_duration(path: str) -> float:
    y, sr = _load_trimmed(path)
    return len(y) / sr


def align_node(state: PipelineState) -> dict:
    segments = sorted([Segment.from_dict(s) for s in state["segments"]], key=lambda s: s.start)
    segs_dir = get_segments_dir(state["video_id"])
    total = _audio_len(state.get("audio_path"))

    counts = {"fits": 0, "slowed": 0, "sped": 0, "trimmed": 0}
    worst = 1.0
    new_audio: list[int] = []
    # Lines an edit did not touch keep the fitted audio they have: it was
    # made from their current TTS clip, and refitting them (or easing their
    # speed toward a changed neighbour) would only redo their lip sync.
    keep = {
        s.id: s.aligned_audio
        for s in segments
        if s.dubbed_audio
        and s.aligned_audio
        and os.path.exists(s.aligned_audio)
        and os.path.basename(s.aligned_audio).startswith(
            os.path.splitext(os.path.basename(s.dubbed_audio))[0]
        )
    }
    # Each TTS line re-voiced in its original speaker's voice (same length).
    cloned, clone_note = _clone_voices(segments, state, segs_dir)

    plan = []  # [seg, y, sr, rate, need, out]
    for i, seg in enumerate(segments):
        if not seg.dubbed_audio:
            continue
        source = cloned.get(seg.id, seg.dubbed_audio)
        y, sr = _load_trimmed(source)
        dur = len(y) / sr
        slot = slot_seconds(segments, i, total)
        need = dur / slot if slot > 0 else 1.0
        worst = max(worst, need)
        if need <= 1.0:
            rate = max(dur / seg.duration, MIN_RATE) if dur < _UNDERRUN * seg.duration else 1.0
        else:
            # Never cut words off: a line still too long after rewriting is
            # sped up exactly as much as it needs, even past HARD_MAX_STRETCH.
            rate = need
        plan.append([seg, y, sr, rate, need, source])

    smoothed = _smooth_rates(plan)

    for seg, y, sr, rate, need, source in plan:
        if seg.id in keep:
            seg.aligned_audio = keep[seg.id]
            counts["kept"] = counts.get("kept", 0) + 1
            continue
        stem = os.path.splitext(os.path.basename(source))[0]
        out = segs_dir / f"{stem}_fit3_{rate:.3f}.wav"
        seg.synced = False  # new audio: its lip sync is still to be done
        new_audio.append(seg.id)
        if not out.exists():
            y2 = _stretch(y, sr, rate) if abs(rate - 1.0) > 0.005 else y
            # Short fades so no line starts or stops with a click.
            f = min(int(sr * _EDGE_FADE_MS / 1000), len(y2) // 4)
            if f > 0:
                y2 = y2.copy()
                y2[:f] *= np.linspace(0.0, 1.0, f, dtype=np.float32)
                y2[-f:] *= np.linspace(1.0, 0.0, f, dtype=np.float32)
            sf.write(str(out), y2, sr)
        if need > HARD_MAX_STRETCH:
            status = "trimmed"  # counted as "over the limit", nothing is cut
            print(f"  ⚠ line {seg.id}: needs {need:.2f}× to fit — sped up, no words cut")
        else:
            status = "sped" if rate > 1.005 else "slowed" if rate < 0.995 else "fits"
        counts[status] += 1
        seg.aligned_audio = str(out)

    kept = counts.pop("kept", 0)
    note = (
        (f"✓ Kept {kept} unchanged line(s) as they were; " if kept else "✓ ")
        + f"Fitted {sum(counts.values())} clips — {counts['fits']} as-is, "
        f"{counts['slowed']} slowed, {counts['sped']} sped up (≤{HARD_MAX_STRETCH}×), "
        f"{counts['trimmed']} over {HARD_MAX_STRETCH}× (sped up, not cut); worst need {worst:.2f}×"
        + (
            f"; {smoothed} neighbouring line(s) eased toward a fast line's speed"
            if smoothed
            else ""
        )
    )
    return {
        "segments": [s.to_dict() for s in segments],
        "new_audio": new_audio,
        "log": state.get("log", []) + ([clone_note] if clone_note else []) + [note],
    }


def _smooth_rates(plan: list) -> int:
    """
    One line at 1.4× between two at 1.0× is a jarring jump in tempo. Lines of
    the same speaker that follow closely (gap < _SMOOTH_GAP) are kept within
    _MAX_JUMP of each other by speeding the slower one up — never slowing the
    fast one (it would overrun) and never so much that the slower line stops
    filling its own speech (≥ _SMOOTH_MIN_FILL of it). Returns lines changed.
    """
    changed = set()
    by_spk: dict = {}
    for row in plan:
        by_spk.setdefault(row[0].speaker, []).append(row)
    pairs = [(x, y) for rows in by_spk.values() for x, y in zip(rows, rows[1:])]
    for _ in range(3):  # let an adjustment ripple a little
        for a, b in pairs + [(y, x) for x, y in pairs]:
            sa, sb = a[0], b[0]
            first, second = (sa, sb) if sa.start < sb.start else (sb, sa)
            if second.start - first.end > _SMOOTH_GAP:
                continue
            target = a[3] - _MAX_JUMP
            if b[3] >= target:
                continue
            dur = len(b[1]) / b[2]
            cap = dur / (_SMOOTH_MIN_FILL * sb.duration) if sb.duration > 0 else target
            new = min(target, cap)
            if new > b[3] + 0.005:
                b[3] = new
                changed.add(sb.id)
    return len(changed)


def _clone_voices(segments: list[Segment], state: PipelineState, segs_dir) -> tuple[dict, str]:
    """Seed-VC voice cloning (voice_clone.py); TTS voices if unavailable."""
    from ..voice import voice_clone as vcl

    vocals_path = state.get("vocals_path")
    if not vcl.available() or not vocals_path or not os.path.exists(vocals_path):
        return {}, ""
    try:
        # Speakers the user gave a voice keep it (state voice_map holds only
        # explicit choices).
        chosen = {int(k) for k in (state.get("voice_map") or {}) if str(k).isdigit()}
        return vcl.clone_lines(segments, vocals_path, segs_dir.parent, segs_dir, chosen)
    except Exception as exc:
        print(f"  ⚠ voice cloning skipped: {exc}")
        return {}, "⚠ Voice cloning skipped — TTS voices used (see Server output)"


# ── Helpers ───────────────────────────────────────────────────────────────────


def _load_trimmed(path: str) -> tuple[np.ndarray, int]:
    import librosa

    y, sr = sf.read(path, dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    if y.size == 0:
        return y, sr
    yt, idx = librosa.effects.trim(y, top_db=_TRIM_TOP_DB)
    # Keep a few ms either side so consonant onsets/offsets are not clipped.
    pad = int(0.03 * sr)
    a, b = max(0, idx[0] - pad), min(len(y), idx[1] + pad)
    return y[a:b], sr


def _audio_len(path: str | None) -> float | None:
    try:
        return sf.info(path).duration if path else None
    except Exception:
        return None


def _stretch(y: np.ndarray, sr: int, rate: float) -> np.ndarray:
    """rate > 1 = faster/shorter. Rubberband if available, else ffmpeg atempo."""
    try:
        import pyrubberband as pyrb

        return pyrb.time_stretch(y, sr, rate=rate).astype(np.float32)
    except (ImportError, RuntimeError, OSError):
        pass
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        src, dst = os.path.join(tmp, "in.wav"), os.path.join(tmp, "out.wav")
        sf.write(src, y, sr)
        f = (
            f"atempo=0.5,atempo={rate / 0.5:.4f}"
            if rate < 0.5
            else f"atempo=2.0,atempo={rate / 2.0:.4f}"
            if rate > 2.0
            else f"atempo={rate:.4f}"
        )
        subprocess.run(
            ["ffmpeg", "-i", src, "-filter:a", f, dst, "-y"], check=True, capture_output=True
        )
        out, _ = sf.read(dst, dtype="float32", always_2d=True)
        return out.mean(axis=1)
