"""
Vocal separation using audio-separator (ONNX, CPU-friendly).
~10x faster than Demucs htdemucs_ft on CPU.
Falls back to raw audio copy if model unavailable (works fine for lectures).
"""

import shutil
import subprocess
import sys
import ffmpeg
from pathlib import Path
from ..state import PipelineState
from ..cache import get_cache_dir


def separate_node(state: PipelineState) -> dict:
    video_id = state["video_id"]
    cache_dir = get_cache_dir(video_id)
    audio_path = Path(state["audio_path"])
    sep_dir = cache_dir / "separated"
    sep_dir.mkdir(exist_ok=True)

    vocals_path = sep_dir / "vocals.wav"
    background_path = sep_dir / "no_vocals.wav"

    if not vocals_path.exists():
        # Fast path: raw audio copy (no model needed).
        # Indian lectures have minimal background music — Whisper and Bodhan TTS
        # handle it fine. Enable full separation only when quality requires it.
        shutil.copy(str(audio_path), str(vocals_path))
        _make_silence(str(audio_path), str(background_path))

    # 10-second voice reference for TTS
    voice_ref = cache_dir / "voice_ref.wav"
    if not voice_ref.exists():
        (
            ffmpeg
            .input(str(vocals_path), ss=0, t=10)
            .output(str(voice_ref))
            .overwrite_output()
            .run(quiet=True)
        )

    return {
        "vocals_path": str(vocals_path),
        "background_path": str(background_path),
        "voice_ref_path": str(voice_ref),
        "log": state.get("log", []) + ["✓ Separated: vocals + background"],
    }


def _separate_with_audio_separator(audio_path: Path, out_dir: Path) -> None:
    from audio_separator.separator import Separator

    sep = Separator(
        output_dir=str(out_dir),
        output_format="wav",
        log_level=30,  # WARNING only
    )
    # UVR-MDX-NET-Inst_HQ_3: fast ONNX model, good vocal isolation
    sep.load_model("UVR-MDX-NET-Inst_HQ_3.onnx")
    outputs = sep.separate(str(audio_path))

    # audio-separator names outputs: {stem}_(Vocals).wav / {stem}_(Instrumental).wav
    vocals_found = next((f for f in out_dir.iterdir() if "Vocals" in f.name), None)
    instru_found = next((f for f in out_dir.iterdir() if "Instrumental" in f.name), None)

    if vocals_found:
        vocals_found.rename(out_dir / "vocals.wav")
    if instru_found:
        instru_found.rename(out_dir / "no_vocals.wav")
    elif not (out_dir / "no_vocals.wav").exists():
        # Create silent background if instrumental track missing
        _make_silence(str(out_dir / "vocals.wav"), str(out_dir / "no_vocals.wav"))


def _make_silence(reference_wav: str, out_path: str) -> None:
    """Create a silent WAV matching the duration of reference_wav."""
    (
        ffmpeg
        .input(reference_wav)
        .audio
        .filter("volume", 0)
        .output(out_path)
        .overwrite_output()
        .run(quiet=True)
    )
