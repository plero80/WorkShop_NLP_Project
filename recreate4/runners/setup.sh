#!/usr/bin/env bash
set -e
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
if [[ "${1:-}" == --help ]]; then
    printf 'Usage: bash recreate3/runners/setup.sh\nCreates the dedicated recreate3 Conda environment. Does not launch training.\n'
    exit 0
fi
if (( $# )); then printf 'setup.sh takes no arguments.\n' >&2; exit 2; fi
init_paths setup
require_tools
if [[ ! -f "$CONDA_EXEC" ]]; then printf 'Missing Conda initialization: %s\n' "$CONDA_EXEC" >&2; exit 2; fi
source "$CONDA_EXEC"
if ! conda activate "$RECREATE3_ENV" >/dev/null 2>&1; then
    conda create -y -n "$RECREATE3_ENV" python=3.11 pip
    conda activate "$RECREATE3_ENV"
fi
python -m pip install 'torch==2.8.0' --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
python -m pip install -r "$PROJECT_ROOT/requirements.txt"
python -m pip check
python -c 'import torch, transformers, peft, datasets; print("torch", torch.__version__, "transformers", transformers.__version__, "peft", peft.__version__, "datasets", datasets.__version__)'
printf 'Environment ready. Next, run runners/run.sh EXPERIMENT_ID on an allocated GPU node.\n'
