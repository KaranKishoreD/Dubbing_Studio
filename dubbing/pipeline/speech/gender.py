"""
Speaker gender, used to cast a matching voice.

  - pitch: median pitch (F0) of a line, plus the pitch track that
    speech/prosody.py reuses to judge delivery.
  - classify_genders: two wav2vec2 voice classifiers, averaged over all of
    a speaker's lines (the default; see GENDER_METHOD).
  - assign_genders: the pitch-only fallback.

nodes/diarize.py then lets face analysis overrule these when faces are clear.
"""

import os

import numpy as np

from ..state import Segment
from .speaker_turns import SR, cuda_available

_F0_MIN, _F0_MAX = 60.0, 400.0  # pitch search range, Hz

# Pitch fallback: a speaker at or above this median pitch is female. If all
# speakers land on the same side but the highest is this many times the
# lowest, they are a man/woman pair and are split at the midpoint.
_FEMALE_F0_HZ = float(os.environ.get("FEMALE_F0_HZ", "180"))
_GENDER_RATIO = float(os.environ.get("DIARIZE_GENDER_RATIO", "1.55"))


def pitch(y: np.ndarray) -> tuple[float, np.ndarray | None, np.ndarray | None]:
    """(median F0, pyin f0 track, voiced flags). Track is reused by prosody."""
    if y.size < SR * 0.3 or float(np.max(np.abs(y))) < 1e-3:
        return 0.0, None, None
    import librosa

    try:
        f0, voiced, _ = librosa.pyin(y, fmin=_F0_MIN, fmax=_F0_MAX, sr=SR, frame_length=1024)
    except Exception:
        return 0.0, None, None
    v = f0[voiced & ~np.isnan(f0)]
    v = v[v > _F0_MIN * 1.15]  # frames pinned to the floor are creak/noise
    return (float(np.median(v)) if v.size >= 5 else 0.0), f0, voiced


# ── Gender from voice timbre ──────────────────────────────────────────────────
# Pitch alone mislabels raised voices: a man shouting can sit above 200 Hz,
# in the usual female range. Two wav2vec2 gender classifiers judge the tone
# of the voice instead. Each makes mistakes on different single lines, so
# their average over all of a speaker's lines is used. Pitch is the fallback.
GENDER_METHOD = os.environ.get("GENDER_METHOD", "classifier").lower()

_GENDER_MODELS = [
    m.strip()
    for m in os.environ.get(
        "GENDER_MODELS",
        "prithivMLmods/Common-Voice-Gender-Detection,"
        "alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech",
    ).split(",")
    if m.strip()
]


def classify_genders(audio: np.ndarray, segments: list[Segment]) -> dict[int, float]:
    """{speaker: P(female)} — ensemble mean, weighted by line duration."""
    import torch
    from transformers import pipeline

    clips: dict[int, list[tuple[np.ndarray, float]]] = {}
    for seg in segments:
        y = audio[int(seg.start * SR) : int(min(seg.end, seg.start + 12) * SR)]
        if len(y) >= SR * 0.7:
            clips.setdefault(seg.speaker or 0, []).append((y, seg.duration))
    if not clips:
        raise RuntimeError("no line long enough to classify")
    per_model: list[dict[int, float]] = []
    device = 0 if cuda_available() else -1
    for name in _GENDER_MODELS:
        clf = pipeline("audio-classification", model=name, device=device)
        out = {}
        for spk, items in clips.items():
            num = den = 0.0
            for y, w in items:
                r = {x["label"].lower(): x["score"] for x in clf({"raw": y, "sampling_rate": SR})}
                num += w * r.get("female", 0.0)
                den += w
            out[spk] = num / den
        per_model.append(out)
        del clf
        if cuda_available():
            torch.cuda.empty_cache()
    return {spk: float(np.mean([m[spk] for m in per_model])) for spk in clips}


def assign_genders(profiles: dict[str, dict]) -> bool:
    """
    Absolute threshold first. Only if every measured speaker lands on the same
    side AND the highest is at least _GENDER_RATIO × the lowest is the top end
    relabelled — a gap that size is a man/woman pair, not two men.
    """
    fallback = os.environ.get("UNKNOWN_GENDER_FALLBACK", "male")
    measured = {k: p for k, p in profiles.items() if p["f0"]}
    for p in profiles.values():
        p["gender"] = ("female" if p["f0"] >= _FEMALE_F0_HZ else "male") if p["f0"] else fallback
    if len(measured) < 2 or len({p["gender"] for p in measured.values()}) > 1:
        return False
    lo = min(measured.values(), key=lambda p: p["f0"])
    hi = max(measured.values(), key=lambda p: p["f0"])
    if hi["f0"] / lo["f0"] < _GENDER_RATIO:
        return False
    mid = (lo["f0"] * hi["f0"]) ** 0.5
    for p in measured.values():
        p["gender"] = "female" if p["f0"] >= mid else "male"
    return True
