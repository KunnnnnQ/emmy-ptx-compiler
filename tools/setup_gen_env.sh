#!/usr/bin/env bash
# Linux/WSL environment for GENERATING extra test problems (emmy-ml 0.3.11 + CPU torch),
# the same versions the task repository's problems were generated with.
#   bash tools/setup_gen_env.sh            -> ~/venvs/emmy311
# Set UV_DEFAULT_INDEX to a PyPI mirror if the default index is slow.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
VENV="${VENV:-$HOME/venvs/emmy311}"
[ -f "$VENV/bin/activate" ] || uv venv -q --python 3.12 "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
UV_DEFAULT_INDEX=https://download.pytorch.org/whl/cpu uv pip install -q torch
uv pip install -q "emmy-ml[compile]==0.3.11" numpy requests
emmy --version
python -c "import torch; print('torch', torch.__version__)"
