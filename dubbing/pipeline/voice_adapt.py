"""
Voice adaptation: make each dubbed line sit like the original performance.

Applied to the fitted clip of every line (after align, before lip sync),
each step independently switchable:

  pitch     shift the dub toward the original SPEAKER's median pitch, at
            most ±VA_MAX_SEMITONES. Per speaker, not per line, so a voice
            keeps one register; line-to-line emotion comes from the TTS.
  loudness  each original line's loudness relative to its speaker's
            average (a shout, an aside) is re-applied to the dub, within
            ±VA_MAX_GAIN_DB. Without it every dubbed line has equal level.
  eq        the dub's long-term spectrum is matched to the speaker's
            original voice (mic, room, distance), smoothed and limited to
            ±VA_MAX_EQ_DB, so the dub does not sound like a studio insert.

Everything is measured on the Demucs vocal track, so music and effects do
not skew it. VOICE_ADAPT=1 enables it.
"""

import os

import numpy as np

ENABLED = os.environ.get("VOICE_ADAPT", "1") == "1"
_PITCH = os.environ.get("VA_PITCH", "1") == "1"
_LOUD = os.environ.get("VA_LOUDNESS", "1") == "1"
_EQ = os.environ.get("VA_EQ", "1") == "1"
# ±2 st sounded processed on a male voice (+2.0 needed); +0.8 on a female
# voice sounded natural. Larger gaps are a casting problem, not a pitch one.
_MAX_ST = float(os.environ.get("VA_MAX_SEMITONES", "1"))
_MAX_GAIN = float(os.environ.get("VA_MAX_GAIN_DB", "6"))
_MAX_EQ = float(os.environ.get("VA_MAX_EQ_DB", "6"))
_SR = 16000          # analysis rate
_NFFT = 1024
_BANDS = 24          # log-spaced EQ bands


def _median_f0(y: np.ndarray, sr: int) -> float | None:
    import librosa
    if sr != _SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=_SR)
    if len(y) < _SR * 0.4:
        return None
    f0, voiced, _ = librosa.pyin(y, fmin=60, fmax=400, sr=_SR, frame_length=1024)
    v = f0[voiced & ~np.isnan(f0)]
    return float(np.median(v)) if v.size >= 10 else None


def _rms_db(y: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(y ** 2)) + 1e-9))


def _band_edges(sr: int) -> np.ndarray:
    return np.geomspace(80, min(8000, sr / 2 - 1), _BANDS + 1)


def _ltas(clips: list[np.ndarray], sr: int) -> np.ndarray | None:
    """Long-term average spectrum in log bands (dB)."""
    import librosa
    mags = [np.abs(librosa.stft(c, n_fft=_NFFT)) for c in clips if len(c) > _NFFT]
    if not mags:
        return None
    spec = np.concatenate(mags, axis=1)
    # Only frames with signal: silence would flatten the average.
    energy = spec.sum(axis=0)
    spec = spec[:, energy > np.percentile(energy, 40)]
    freqs = librosa.fft_frequencies(sr=sr, n_fft=_NFFT)
    edges = _band_edges(sr)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        k = (freqs >= a) & (freqs < b)
        out.append(20 * np.log10(spec[k].mean() + 1e-9) if k.any() else 0.0)
    return np.array(out)


class SpeakerModel:
    """What a speaker's original voice is like, measured once per run."""

    def __init__(self):
        self.f0: float | None = None
        self.mean_db: float = 0.0
        self.ltas: np.ndarray | None = None
        self.line_db: dict[int, float] = {}


def build_models(segments, vocals: np.ndarray, sr: int) -> dict[int, SpeakerModel]:
    models: dict[int, SpeakerModel] = {}
    clips: dict[int, list[np.ndarray]] = {}
    for s in segments:
        spk = s.speaker or 0
        m = models.setdefault(spk, SpeakerModel())
        y = vocals[int(s.start * sr):int(s.end * sr)]
        if len(y) < sr * 0.3:
            continue
        m.line_db[s.id] = _rms_db(y)
        clips.setdefault(spk, []).append(y)
    for spk, m in models.items():
        f0s = [s.f0 for s in segments if (s.speaker or 0) == spk and s.f0]
        m.f0 = float(np.median(f0s)) if f0s else None
        if m.line_db:
            m.mean_db = float(np.median(list(m.line_db.values())))
        m.ltas = _ltas(clips.get(spk, []), sr)
    return models


def adapt(y: np.ndarray, sr: int, seg, model: SpeakerModel, dub_ltas: np.ndarray | None,
          dub_f0: float | None) -> tuple[np.ndarray, dict]:
    """Return the adapted clip and what was done."""
    info = {}
    if _PITCH and model.f0 and dub_f0:
        st = float(np.clip(12 * np.log2(model.f0 / dub_f0), -_MAX_ST, _MAX_ST))
        if abs(st) >= 0.25:
            import pyrubberband as pyrb
            y = pyrb.pitch_shift(y, sr, st).astype(np.float32)
            info["pitch_st"] = round(st, 2)
    if _EQ and model.ltas is not None and dub_ltas is not None:
        y = _apply_eq(y, sr, model.ltas - dub_ltas)
        info["eq"] = True
    if _LOUD and seg.id in model.line_db:
        rel = float(np.clip(model.line_db[seg.id] - model.mean_db, -_MAX_GAIN, _MAX_GAIN))
        if abs(rel) >= 0.5:
            y = y * (10 ** (rel / 20))
            info["gain_db"] = round(rel, 1)
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    if peak > 0.99:
        y = y * (0.99 / peak)
    return y.astype(np.float32), info


def _apply_eq(y: np.ndarray, sr: int, diff_db: np.ndarray) -> np.ndarray:
    import librosa
    # Remove overall level difference (that is loudness, handled separately),
    # smooth across bands, and limit.
    d = diff_db - np.median(diff_db)
    d = np.convolve(np.pad(d, 2, mode="edge"), np.ones(5) / 5, mode="valid")
    d = np.clip(d, -_MAX_EQ, _MAX_EQ)
    edges = _band_edges(sr)
    centers = np.sqrt(edges[:-1] * edges[1:])
    freqs = librosa.fft_frequencies(sr=sr, n_fft=_NFFT)
    gain_db = np.interp(np.log(np.maximum(freqs, 1)), np.log(centers), d,
                        left=d[0], right=d[-1])
    S = librosa.stft(y, n_fft=_NFFT)
    S *= (10 ** (gain_db / 20))[:, None]
    out = librosa.istft(S, length=len(y))
    return out.astype(np.float32)
