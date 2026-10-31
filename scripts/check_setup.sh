#!/usr/bin/env bash
# Verify an install in seconds, without changing anything or printing secrets.
#
#   bash scripts/check_setup.sh
#
# Exit code 0 when every REQUIRED item passes.
cd "$(dirname "$0")/.."
export PATH="$PWD/.venv/bin:$PATH"   # ffmpeg may live in the venv

fail=0
ok()   { printf '  \033[32m✓\033[0m %s\n' "$1"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$1"; }
bad()  { printf '  \033[31m✗\033[0m %s\n' "$1"; fail=1; }

echo "System"
for c in git ffmpeg; do
  command -v "$c" >/dev/null && ok "$c" || bad "$c missing"
done
command -v rubberband >/dev/null && ok "rubberband" || warn "rubberband missing (ffmpeg atempo is used instead)"

echo "Python environment"
PY=.venv/bin/python
if [ -x "$PY" ]; then
  ok "$($PY --version) in .venv"
  "$PY" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null \
    && ok "GPU: $($PY -c 'import torch; print(torch.cuda.get_device_name(0))')" \
    || warn "no usable GPU (dubbing is slow and lip sync is skipped)"
else
  bad ".venv missing (run: bash scripts/setup_gpu.sh)"
fi

echo "Optional models"
[ -x dubbing/LatentSync/.venv-ls/bin/python ] && ok "LatentSync (lip sync)" || warn "LatentSync not installed"
[ -x dubbing/SeedVC/.venv-vc/bin/python ]     && ok "Seed-VC (voice cloning)" || warn "Seed-VC not installed"
[ -d dubbing/Separator ]                      && ok "Separator (songs)" || warn "Separator not installed"

echo "API keys in .env (names only, values are never shown)"
if [ -f .env ]; then
  for k in BODHAN_TRANSLATE_KEY BODHAN_SPEAK_KEY OPENROUTER_API_KEY; do
    grep -Eq "^$k=.+" .env && ok "$k" || bad "$k not set"
  done
  for k in BODHAN_TRANSCRIBE_KEY HF_TOKEN SARVAM_API_KEY; do
    grep -Eq "^$k=.+" .env && ok "$k" || warn "$k not set (optional)"
  done
else
  bad ".env missing (copy dubbing/.env.example to .env)"
fi

echo "Code"
if [ -x "$PY" ]; then
  (cd dubbing && "../$PY" -m pytest -q 2>&1 | tail -1 | sed 's/^/  /')
fi

echo
[ "$fail" = 0 ] && echo "Ready: cd dubbing && python -m server" || echo "Fix the ✗ items above, then run this again."
exit "$fail"
