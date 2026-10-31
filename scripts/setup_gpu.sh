#!/usr/bin/env bash
# Bootstrap the dubbing pipeline on a fresh GPU machine.
#
#   git clone <your-repo-url> RAGA_Project
#   cd RAGA_Project
#   bash scripts/setup_gpu.sh
#
# Re-fetches everything .gitignore excludes: venv, model weights, vendored repos.
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"
echo "▶ Project root: $ROOT"

# ── 1. System deps ────────────────────────────────────────────────────────────
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "✗ ffmpeg not found. Install it first:"
  echo "    Ubuntu/Debian : sudo apt update && sudo apt install -y ffmpeg rubberband-cli"
  echo "    macOS         : brew install ffmpeg rubberband"
  echo "    Windows       : winget install Gyan.FFmpeg"
  exit 1
fi
echo "✓ ffmpeg present"

command -v rubberband >/dev/null 2>&1 \
  && echo "✓ rubberband present (high-quality time-stretch)" \
  || echo "⚠ rubberband missing — will fall back to ffmpeg atempo (lower quality)"

# ── 2. Python env ─────────────────────────────────────────────────────────────
if [ ! -d .venv ]; then
  echo "▶ Creating .venv"
  # Python 3.11: several ML deps (mediapipe, pyannote, insightface) have no
  # wheels for 3.13+ yet.
  PY311=$(command -v python3.11 || echo python3)
  "$PY311" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate 2>/dev/null || source .venv/Scripts/activate
python -m pip install --upgrade pip wheel

echo "▶ Installing base requirements"
pip install -r dubbing/requirements.txt
python -m unidic download   # Japanese dictionary for International mode (~500 MB)

echo "▶ Installing GPU requirements (CUDA 12.8 / torch>=2.7 for Blackwell)"
pip install -r dubbing/requirements-gpu.txt

# ── 3. Verify CUDA actually works on this GPU ─────────────────────────────────
echo "▶ Verifying CUDA"
python - <<'PY'
import sys
try:
    import torch
except ImportError:
    sys.exit("✗ torch not installed")
print(f"  torch {torch.__version__}  CUDA available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    sys.exit("✗ CUDA unavailable — check driver, then reinstall torch from the cu128 index")
name = torch.cuda.get_device_name(0)
cap  = torch.cuda.get_device_capability(0)
print(f"  device: {name}  sm_{cap[0]}{cap[1]}")
try:
    (torch.zeros(8, device="cuda") + 1).sum().item()
    print("  ✓ kernel launch OK")
except Exception as exc:
    sys.exit(
        f"✗ kernel launch failed: {exc}\n"
        f"  sm_{cap[0]}{cap[1]} unsupported by this torch build.\n"
        "  Fix: pip install --force-reinstall --extra-index-url "
        "https://download.pytorch.org/whl/cu128 'torch>=2.7.0' 'torchaudio>=2.7.0'"
    )
PY

# ── 4. .env ───────────────────────────────────────────────────────────────────
if [ ! -f .env ]; then
  cp dubbing/.env.example .env
  echo "⚠ Created .env from template — fill in your keys before running:"
  echo "    $ROOT/.env"
else
  echo "✓ .env already present"
fi

# ── 5. Lip sync backend: LatentSync (recommended) ─────────────────────────────
read -r -p "Clone LatentSync for lip sync? (~5GB of weights) [y/N] " ans
if [[ "${ans:-N}" =~ ^[Yy]$ ]]; then
  if [ ! -d dubbing/LatentSync ]; then
    git clone https://github.com/bytedance/LatentSync dubbing/LatentSync
  fi
  # LatentSync pins torch 2.5.1/cu121 (no Blackwell kernels) and old
  # diffusers/transformers that clash with the pipeline, so it gets its own
  # venv with a cu128 torch. lipsync.py runs it out of process.
  LSV=dubbing/LatentSync/.venv-ls
  [ -d "$LSV" ] || python3 -m venv "$LSV"
  "$LSV/bin/pip" install --upgrade pip wheel
  "$LSV/bin/pip" install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
      --index-url https://download.pytorch.org/whl/cu128
  grep -vE "^(torch|torchvision|--extra-index-url|gradio|mediapipe|lpips|face-alignment)" \
      dubbing/LatentSync/requirements.txt > /tmp/latentsync-reqs.txt
  "$LSV/bin/pip" install -r /tmp/latentsync-reqs.txt soundfile || \
    echo "⚠ Some LatentSync deps failed — check its README"
  python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    "ByteDance/LatentSync-1.6",
    local_dir="dubbing/LatentSync/checkpoints",
    allow_patterns=["*.pt", "*.ckpt", "*.json", "*.safetensors"],
)
print("✓ LatentSync checkpoints downloaded")
PY
fi

# ── 5b. Voice cloning: Seed-VC (GPL-3, run out of process) ────────────────────
read -r -p "Clone Seed-VC for speaker voice cloning? (~1GB of weights) [y/N] " ans
if [[ "${ans:-N}" =~ ^[Yy]$ ]]; then
  [ -d dubbing/SeedVC ] || git clone --depth 1 https://github.com/Plachtaa/seed-vc dubbing/SeedVC
  # Newer huggingface_hub no longer passes these to BigVGAN's loader.
  sed -i -E 's/^(\s*)proxies: Optional\[Dict\],$/\1proxies: Optional[Dict] = None,/;
             s/^(\s*)resume_download: bool,$/\1resume_download: bool = False,/' \
      dubbing/SeedVC/modules/bigvgan/bigvgan.py
  # Overlay venv: reuses the main venv's torch/transformers via a .pth file,
  # adds only what Seed-VC needs on top. voice_clone.py runs it as a worker.
  SVV=dubbing/SeedVC/.venv-vc
  [ -d "$SVV" ] || python3 -m venv "$SVV"
  python -c "import site;print(site.getsitepackages()[0])" \
      > "$("$SVV/bin/python" -c "import site;print(site.getsitepackages()[0])")/zz_main_venv.pth"
  "$SVV/bin/pip" install munch==4.0.0 hydra-core==1.3.2
  # Only DAC's quantizer is imported; the full package pins old audio libs.
  DAC="$("$SVV/bin/python" -c "import site;print(site.getsitepackages()[0])")/dac"
  mkdir -p "$DAC/nn" && touch "$DAC/__init__.py" "$DAC/nn/__init__.py"
  for f in quantize layers; do
    curl -sL "https://raw.githubusercontent.com/descriptinc/descript-audio-codec/1.0.0/dac/nn/$f.py" \
        -o "$DAC/nn/$f.py"
  done
  echo "✓ Seed-VC ready (weights download on first use)"
fi

# ── 5c. Song mode: vocal separator (BS-RoFormer via audio-separator) ──────────
# audio-separator is already in the main venv (requirements.txt). song_mode.py
# runs it from its own folder so its model files stay in dubbing/Separator/;
# this overlay venv just points back at the main venv's packages.
SEPV=dubbing/Separator/.venv-sep
mkdir -p dubbing/Separator/models
[ -d "$SEPV" ] || python3 -m venv "$SEPV"
python -c "import site;print(site.getsitepackages()[0])" \
    > "$("$SEPV/bin/python" -c "import site;print(site.getsitepackages()[0])")/zz_main_venv.pth"
echo "✓ Separator ready (model downloads on first use)"

# ── 6. Pre-warm the models that auto-download ─────────────────────────────────
echo "▶ Pre-warming Whisper + Demucs (first run would otherwise stall)"
python - <<'PY'
try:
    from faster_whisper import WhisperModel
    WhisperModel("large-v3", device="cuda", compute_type="float16")
    print("✓ faster-whisper large-v3 cached")
except Exception as exc:
    print(f"⚠ whisper prewarm skipped: {exc}")
try:
    from demucs.pretrained import get_model
    get_model("htdemucs_ft")
    print("✓ demucs htdemucs_ft cached")
except Exception as exc:
    print(f"⚠ demucs prewarm skipped: {exc}")
PY

cat <<'EOF'

─────────────────────────────────────────────────────────────
Setup complete.

Next:
  1. Fill in .env  (BODHAN_TRANSLATE_KEY, BODHAN_SPEAK_KEY,
                    OPENROUTER_API_KEY, HF_TOKEN)
  2. For diarization, accept the gated model terms once:
       https://huggingface.co/pyannote/speaker-diarization-3.1
       https://huggingface.co/pyannote/segmentation-3.0
  3. Run:
       source .venv/bin/activate
       cd dubbing && python -m server      # then open http://127.0.0.1:8000
─────────────────────────────────────────────────────────────
EOF
