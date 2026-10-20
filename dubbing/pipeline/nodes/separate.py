"""
Vocal separation with Demucs (htdemucs), run in-process.

Produces two stems from the full-quality 44.1 kHz stereo audio:
  vocals → speech only. Cleaner input for diarization, pitch and onset work.
  bed    → drums + bass + other. Everything except the voice: music, applause,
           room tone. assemble_node splices this in under the dub, so the
           original voice is removed outright instead of merely turned down.

Audio I/O goes through soundfile, not torchaudio, so a torch/torchaudio
version mismatch cannot break it. Falls back to the raw audio (and the old
ducking mix) if Demucs is unavailable or SEPARATE=none.
"""

import os
import shutil
from pathlib import Path

import ffmpeg
import numpy as np
import soundfile as sf

from ..cache import get_cache_dir
from ..state import PipelineState
from .download import ensure_hq_audio

_MODE = os.environ.get("SEPARATE", "demucs").lower()
_MODEL = os.environ.get("DEMUCS_MODEL", "htdemucs")
_model_cache = None


def separate_node(state: PipelineState) -> dict:
    cache_dir = get_cache_dir(state["video_id"])
    sep_dir = cache_dir / "separated"
    sep_dir.mkdir(exist_ok=True)
    log = state.get("log", [])

    hq = Path(state.get("audio_hq_path") or ensure_hq_audio(Path(state["video_path"]), cache_dir))
    vocals = sep_dir / f"{_MODEL}_vocals.wav"
    bed = sep_dir / f"{_MODEL}_bed.wav"

    note = ""
    if _MODE != "none" and not (vocals.exists() and bed.exists()):
        try:
            secs = _run_demucs(hq, vocals, bed)
            note = f"✓ Separated voice from background with Demucs {_MODEL} ({secs:.0f}s)"
        except Exception as exc:
            note = f"⚠ Demucs failed ({type(exc).__name__}: {exc}) — using raw audio"
    elif vocals.exists() and bed.exists():
        note = f"✓ Separation loaded from cache (Demucs {_MODEL})"

    if vocals.exists() and bed.exists():
        vocals_path, background_path = vocals, bed
    else:
        # Fallback: raw mix as "vocals", no bed. assemble then ducks instead.
        vocals_path = sep_dir / "vocals.wav"
        if not vocals_path.exists():
            shutil.copy(str(state["audio_path"]), str(vocals_path))
        background_path = None
        note = note or "✓ Separation off (SEPARATE=none) — using raw audio"

    voice_ref = cache_dir / "voice_ref.wav"
    if not voice_ref.exists():
        (
            ffmpeg.input(str(vocals_path), ss=0, t=10)
            .output(str(voice_ref), ar=16000, ac=1)
            .overwrite_output()
            .run(quiet=True)
        )

    return {
        "audio_hq_path": str(hq),
        "vocals_path": str(vocals_path),
        "background_path": str(background_path) if background_path else None,
        "voice_ref_path": str(voice_ref),
        "log": log + [note],
    }


def _load_model():
    global _model_cache
    if _model_cache is None:
        from demucs.pretrained import get_model
        _model_cache = get_model(_MODEL)
        _model_cache.eval()
    return _model_cache


def _run_demucs(src: Path, vocals_out: Path, bed_out: Path) -> float:
    import time
    import torch
    from demucs.apply import apply_model

    t0 = time.time()
    model = _load_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    y, sr = sf.read(str(src), dtype="float32", always_2d=True)     # (n, ch)
    if sr != model.samplerate:
        import librosa
        y = librosa.resample(y.T, orig_sr=sr, target_sr=model.samplerate).T
        sr = model.samplerate
    if y.shape[1] == 1:
        y = np.repeat(y, 2, axis=1)

    wav = torch.from_numpy(y.T.copy())                             # (2, n)
    # Same normalisation the demucs CLI applies.
    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    wav = (wav - mean) / std

    with torch.no_grad():
        out = apply_model(
            model, wav[None], device=device, shifts=0, split=True,
            overlap=0.25, progress=False,
        )[0]                                                       # (sources, 2, n)
    out = out * std + mean

    idx = model.sources.index("vocals")
    voice = out[idx]
    rest = out.sum(0) - voice
    sf.write(str(vocals_out), voice.T.cpu().numpy(), sr, subtype="FLOAT")
    sf.write(str(bed_out), rest.T.cpu().numpy(), sr, subtype="FLOAT")
    return time.time() - t0
