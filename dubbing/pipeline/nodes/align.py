"""
Fit each dubbed clip into its slot on the original timeline.

Modelled on VoiceStudio's fit planner:

  1. Trim the TTS clip's leading/trailing silence. TTS engines pad both ends,
     which inflates the length and pushes the voice late.
  2. Slot = from the segment's start up to just before the next segment
     starts, borrowing at most GAP_BORROW_MAX seconds of following silence.
     The dub never runs into the next line, so two voices never overlap.
  3. need = clip / slot
       need ≤ 1, clip much shorter than the original line → slow gently
                  (≥ MIN_RATE) so the voice does not finish while the speaker
                  is visibly still talking
       need ≤ 1 otherwise → untouched
       1 < need          → speed up, capped at MAX_STRETCH
  4. Anything still longer than the slot is trimmed with a short fade. The
     synthesize step shortens the text of such lines first, so this is rare.

The cache key is the dubbed clip's own filename (which encodes voice, style
and text), so a re-cast or re-translation always re-aligns.
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
# Last resort for lines that still overrun after rewriting: audibly faster,
# but better than cutting words off. Only past this is a clip trimmed.
HARD_MAX_STRETCH = float(os.environ.get("HARD_MAX_STRETCH", "1.25"))
MIN_RATE = float(os.environ.get("MIN_RATE", "0.90"))
GAP_GUARD = float(os.environ.get("GAP_GUARD_SECS", "0.08"))
GAP_BORROW_MAX = float(os.environ.get("GAP_BORROW_MAX_SECS", "3.0"))
_UNDERRUN = 0.95
_FADE_MS = 80
_TRIM_TOP_DB = 40


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
    segments = sorted(
        [Segment.from_dict(s) for s in state["segments"]], key=lambda s: s.start
    )
    segs_dir = get_segments_dir(state["video_id"])
    total = _audio_len(state.get("audio_path"))

    counts = {"fits": 0, "slowed": 0, "sped": 0, "trimmed": 0}
    worst = 1.0

    for i, seg in enumerate(segments):
        if not seg.dubbed_audio:
            continue
        stem = os.path.splitext(os.path.basename(seg.dubbed_audio))[0]
        out = segs_dir / f"{stem}_fit_{MAX_STRETCH}_{HARD_MAX_STRETCH}_{MIN_RATE}.wav"

        y, sr = _load_trimmed(seg.dubbed_audio)
        dur = len(y) / sr
        slot = slot_seconds(segments, i, total)
        need = dur / slot if slot > 0 else 1.0
        worst = max(worst, need)

        if need <= 1.0:
            if dur < _UNDERRUN * seg.duration:
                rate = max(dur / seg.duration, MIN_RATE)
                status = "slowed" if rate < 0.995 else "fits"
            else:
                rate, status = 1.0, "fits"
        else:
            rate = min(need, HARD_MAX_STRETCH)
            status = "sped"

        if not out.exists():
            y2 = _stretch(y, sr, rate) if abs(rate - 1.0) > 0.005 else y
            max_len = int(slot * sr)
            if len(y2) > max_len:
                y2 = y2[:max_len].copy()
                fade = min(int(sr * _FADE_MS / 1000), len(y2))
                y2[-fade:] *= np.linspace(1.0, 0.0, fade)
            sf.write(str(out), y2, sr)

        if need > HARD_MAX_STRETCH:
            status = "trimmed"
        counts[status] += 1
        seg.aligned_audio = str(out)

    note = (
        f"✓ Fitted {sum(counts.values())} clips — {counts['fits']} as-is, "
        f"{counts['slowed']} slowed, {counts['sped']} sped up (≤{HARD_MAX_STRETCH}×), "
        f"{counts['trimmed']} trimmed; worst need {worst:.2f}×"
    )
    notes = [note]
    adapt_note = _voice_adapt(segments, state, segs_dir)
    if adapt_note:
        notes.append(adapt_note)
    return {
        "segments": [s.to_dict() for s in segments],
        "log": state.get("log", []) + notes,
    }


def _voice_adapt(segments: list[Segment], state: PipelineState, segs_dir) -> str:
    """Pitch / loudness / mic-colour matching of each fitted clip (voice_adapt.py)."""
    from .. import voice_adapt as va
    vocals_path = state.get("vocals_path")
    if not va.ENABLED or not vocals_path or not os.path.exists(vocals_path):
        return ""
    try:
        voc, vsr = sf.read(vocals_path, dtype="float32", always_2d=True)
        voc = voc.mean(axis=1)
        models = va.build_models(segments, voc, vsr)
        # What each speaker's DUB voice is like, from all their fitted clips.
        by_spk: dict[int, list[np.ndarray]] = {}
        sr0 = None
        for s in segments:
            if s.aligned_audio and os.path.exists(s.aligned_audio):
                y, sr0 = sf.read(s.aligned_audio, dtype="float32", always_2d=True)
                by_spk.setdefault(s.speaker or 0, []).append(y.mean(axis=1))
        dub = {spk: (va._ltas(c, sr0), va._median_f0(np.concatenate(c), sr0))
               for spk, c in by_spk.items()}
        shifted = 0
        for s in segments:
            if not s.aligned_audio or not os.path.exists(s.aligned_audio):
                continue
            spk = s.speaker or 0
            stem = os.path.splitext(os.path.basename(s.aligned_audio))[0]
            out = segs_dir / f"{stem}_va.wav"
            if not out.exists():
                y, sr = sf.read(s.aligned_audio, dtype="float32", always_2d=True)
                y2, info = va.adapt(y.mean(axis=1), sr, s, models[spk], *dub[spk])
                sf.write(str(out), y2, sr)
                shifted += int("pitch_st" in info)
            s.aligned_audio = str(out)
        return (f"✓ Voice adapted to each speaker (pitch ≤±{va._MAX_ST:g} st, loudness, "
                f"mic colour)")
    except Exception as exc:
        print(f"  ⚠ voice adaptation skipped: {exc}")
        return "⚠ Voice adaptation skipped (see Server output)"


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
        f = (f"atempo=0.5,atempo={rate / 0.5:.4f}" if rate < 0.5 else
             f"atempo=2.0,atempo={rate / 2.0:.4f}" if rate > 2.0 else
             f"atempo={rate:.4f}")
        subprocess.run(["ffmpeg", "-i", src, "-filter:a", f, dst, "-y"],
                       check=True, capture_output=True)
        out, _ = sf.read(dst, dtype="float32", always_2d=True)
        return out.mean(axis=1)
