# RAGA — YouTube Dubbing Pipeline

YouTube URL → lip-synced video dubbed into Indian languages, with
prompt-driven editing of individual segments.

Built on LangGraph. Translation and TTS via [Bodhan AI](https://console.bodhan.ai);
transcription via Whisper; lip sync via LatentSync (GPU) or Wav2Lip (fallback).

## Pipeline

```
download → separate → transcribe → translate → synthesize → align → lipsync → assemble
```

Two graphs share these nodes: `FULL_PIPELINE` (whole video) and `EDIT_PIPELINE`
(re-runs only the segments a natural-language edit touches).

## Setup — GPU machine

```bash
git clone <your-repo-url> RAGA_Project
cd RAGA_Project
bash scripts/setup_gpu.sh
```

The script creates `.venv`, installs base + GPU requirements, verifies CUDA can
actually launch a kernel on your card, optionally clones LatentSync and
downloads its weights, then pre-warms Whisper and Demucs.

Afterwards fill in `.env` (copied from `dubbing/.env.example`) and run:

```bash
source .venv/bin/activate
cd dubbing && python -m ui.app
```

### RTX 5090 / Blackwell

The 5090 is `sm_120` and needs **CUDA 12.8+ with torch ≥ 2.7**. Older
`cu121`/`cu124` wheels install without complaint but fail at runtime with
`CUDA error: no kernel image is available for execution on the device`.
`requirements-gpu.txt` pins the `cu128` index. Verify with:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
```

## Setup — CPU machine

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r dubbing/requirements.txt
cp dubbing/.env.example .env     # then fill in keys
cd dubbing && python -m ui.app
```

Lip sync is skipped on CPU unless `WAV2LIP_USE_LOCAL=1`; audio is dubbed over
the untouched original video. Needs `ffmpeg` on PATH (and `rubberband` for
better time-stretching, else it falls back to ffmpeg's `atempo`).

## What is not in the repo

`.gitignore` excludes anything large or machine-specific. `scripts/setup_gpu.sh`
restores all of it:

| Excluded | Restored by |
|---|---|
| `.venv/` | `python -m venv` + `pip install` |
| `*.pt` `*.pth` `*.safetensors`, `checkpoints/` | auto-download on first use, or the setup script |
| `dubbing/Wav2Lip/`, `dubbing/LatentSync/` | `git clone` in the setup script |
| `dubbing/cache/`, `outputs/`, `data/` | regenerated per run |
| `.env` | copied from `dubbing/.env.example` |

## Keys

| Variable | Needed for |
|---|---|
| `BODHAN_TRANSLATE_KEY` | translation (`indic-translate`) |
| `BODHAN_SPEAK_KEY` | TTS (`indic-speak`) |
| `OPENROUTER_API_KEY` | edit resolver — maps a prompt to segment IDs |
| `HF_TOKEN` | pyannote speaker diarization (GPU) |
| `FAL_KEY` | fal.ai cloud lip sync (optional, needs billing) |

`HF_TOKEN` also requires accepting the gated terms once on the
[speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
and [segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) model
pages, or loading fails with a 401.
