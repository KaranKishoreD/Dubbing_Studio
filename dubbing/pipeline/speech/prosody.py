"""
Delivery matching: how was each line said, relative to that speaker's norm?

Adapted from VoiceStudio's prosody_mirror. Per line we measure

  pitch      median F0, in semitones re 100 Hz
  spread     10th-90th percentile pitch range, semitones (how animated)
  loud       RMS level of active frames, dB
  rate       syllable nuclei per second (energy peaks >=3 dB, >=80 ms apart)
  voicing    share of active frames that are voiced (low = whisper/breath)

then express each as a robust z-score against the SPEAKER'S OWN baseline
(median, 1.4826*MAD, floored). A naturally loud speaker is not "energetic"
on every line — only lines that are loud for them.

Rules then turn the z-scores into a delivery label. Emotion is deliberately
not inferred: four scalar features cannot tell anger from excitement.
"""

import numpy as np

_SR = 16000
_MIN_ACTIVE_S = 0.30
_MIN_BASELINE_LINES = 3
_VOICE_OUTLIER_Z = 4.0
_FLOORS = {"pitch": 1.0, "spread": 1.0, "loud": 2.0, "rate": 0.5, "voicing": 0.08}


def measure(y: np.ndarray, f0_track: np.ndarray, voiced: np.ndarray) -> dict | None:
    """
    y: 16 kHz mono clip. f0_track / voiced: librosa.pyin output for the same
    clip (frame_length=1024, default hop=256), reused so pitch is not
    estimated twice.
    """
    import librosa
    from scipy.signal import find_peaks

    if y.size < _SR * _MIN_ACTIVE_S:
        return None

    # Activity on the pyin grid, so voicing ratio lines up frame-for-frame.
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]
    n = min(len(rms), len(voiced))
    rms, voiced, f0_track = rms[:n], voiced[:n], f0_track[:n]
    db = 20 * np.log10(rms + 1e-9)
    active = db > max(db.max() - 35.0, -60.0)
    active_s = active.sum() * 256 / _SR
    if active_s < _MIN_ACTIVE_S:
        return None

    f = f0_track[voiced & active & ~np.isnan(f0_track)]
    f = f[f > 70]
    pitch = spread = None
    if f.size >= 5:
        st = 12 * np.log2(f / 100.0)
        pitch = float(np.median(st))
        spread = float(np.percentile(st, 90) - np.percentile(st, 10))

    loud = float(20 * np.log10(np.mean(rms[active]) + 1e-9))
    voicing = float(np.mean(voiced[active])) if active.any() else 0.0

    # Syllable rate on a finer 40 ms / 10 ms grid.
    env = librosa.feature.rms(y=y, frame_length=640, hop_length=160)[0]
    env_db = 20 * np.log10(env + 1e-9)
    floor = env_db.max() - 35.0
    peaks, _ = find_peaks(env_db, prominence=3.0, distance=8, height=floor)
    rate = len(peaks) / active_s

    return {"pitch": pitch, "spread": spread, "loud": loud, "rate": rate, "voicing": voicing}


def label_lines(
    feats: dict[int, dict], speaker_of: dict[int, int]
) -> dict[int, tuple[str, str | None]]:
    """
    feats: {seg_id: measure() output}. speaker_of: {seg_id: speaker}.
    Returns {seg_id: (delivery, pace)}.
    """
    by_spk: dict[int, list[int]] = {}
    for sid in feats:
        by_spk.setdefault(speaker_of.get(sid, 0), []).append(sid)

    pooled = list(feats)
    n_speakers = len(by_spk)
    out: dict[int, tuple[str, str | None]] = {}

    for spk, ids in by_spk.items():
        own = len(ids) >= _MIN_BASELINE_LINES
        base_ids = ids if own else pooled
        # Pitch differs wildly between people, so a pooled pitch baseline is
        # only meaningful when the pool is a single speaker.
        use_pitch = own or n_speakers == 1
        base = _baseline([feats[i] for i in base_ids])

        for sid in ids:
            z = _zscores(feats[sid], base)
            if not use_pitch or (z.get("pitch") is not None and abs(z["pitch"]) > _VOICE_OUTLIER_Z):
                z["pitch"] = z["spread"] = None
            out[sid] = (_delivery(z), _pace(z))
    return out


def _baseline(rows: list[dict]) -> dict:
    base = {}
    for k, floor in _FLOORS.items():
        vals = np.array([r[k] for r in rows if r.get(k) is not None], dtype=float)
        if vals.size == 0:
            base[k] = None
            continue
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med))) * 1.4826
        base[k] = (med, max(mad, floor))
    return base


def _zscores(row: dict, base: dict) -> dict:
    z = {}
    for k in _FLOORS:
        if row.get(k) is None or base.get(k) is None:
            z[k] = None
        else:
            med, sd = base[k]
            z[k] = (row[k] - med) / sd
    return z


def _delivery(z: dict) -> str:
    loud, rate, pitch = z.get("loud") or 0.0, z.get("rate") or 0.0, z.get("pitch")
    voicing = z.get("voicing") or 0.0

    if voicing <= -2.0 and loud <= -1.0:
        return "whispered"
    if loud >= 1.0 and rate >= 1.0 and (pitch is None or pitch >= 0.5):
        return "urgent"
    if loud >= 1.5 and rate <= -0.5:
        return "announcing"
    arousal = [v for v in (z.get("loud"), pitch, z.get("spread")) if v is not None]
    a = float(np.mean(arousal)) if arousal else 0.0
    if a >= 0.8 and loud >= 0.3:
        return "energetic"
    if a <= -0.8 and loud <= -0.3:
        return "calm"
    return "neutral"


def _pace(z: dict) -> str | None:
    r = z.get("rate")
    if r is None:
        return None
    return "quick" if r >= 1.0 else "slow" if r <= -1.0 else None
