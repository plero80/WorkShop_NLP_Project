#!/usr/bin/env bash
# Cluster filesystem conventions taken from recreate2.

init_paths() {
    if (( BASH_VERSINFO[0] < 5 || BASH_VERSINFO[0] == 5 && BASH_VERSINFO[1] < 1 )); then
        printf 'Bash 5.1 or newer is required.\n' >&2; return 2
    fi
    if [[ "$(uname -s)" != Linux ]]; then
        printf 'Run these scripts on your allocated Linux node.\n' >&2; return 2
    fi
    if [[ "${SLURM_JOB_NUM_NODES:-1}" != 1 ]]; then
        printf 'Use a single-node allocation.\n' >&2; return 2
    fi
    EXPERIMENT_ID="${1:?Pass an experiment ID}"
    if [[ ! "$EXPERIMENT_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
        printf 'Experiment ID must be a simple directory name.\n' >&2; return 2
    fi
    PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
    SCRATCH="${SCRATCH:-/vol/scratch/${USER:?USER must be set}}"
    TMP_ROOT="${TMP_ROOT:-/tmp/$USER}"
    CONDA_EXEC="${CONDA_EXEC:-$SCRATCH/miniconda3/etc/profile.d/conda.sh}"
    RECREATE3_ENV="${RECREATE3_ENV:-recreate3}"
    LOCAL_EXPERIMENT="$TMP_ROOT/recreate3/$EXPERIMENT_ID"
    SCRATCH_EXPERIMENT="$SCRATCH/checkpoints/recreate3/$EXPERIMENT_ID"
    LOGS_ROOT="$SCRATCH/logs/recreate3/$EXPERIMENT_ID"
    MODEL_CACHE="$SCRATCH/weights/recreate3/huggingface"
    DATA_CACHE="$SCRATCH/datasets/recreate3/huggingface"
    DATA_METADATA="$SCRATCH/datasets/recreate3/$EXPERIMENT_ID"
    HELPERS="$PROJECT_ROOT/python_helper"
    export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
    export HF_HUB_DISABLE_PROGRESS_BARS=1 TQDM_DISABLE=1 NO_COLOR=1 TERM=dumb
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
}

select_run_profile() {
    VRAM_GB=24
    RUN_ARGUMENTS=()
    while (( $# )); do
        case "$1" in
            --vram-gb)
                if (( $# < 2 )); then printf '%s\n' '--vram-gb requires 12, 24, or 48.' >&2; return 2; fi
                VRAM_GB="$2"; shift 2 ;;
            --vram-gb=*) VRAM_GB="${1#*=}"; shift ;;
            --recipe)
                if (( $# < 2 )) || [[ -z "$2" ]]; then printf '%s\n' '--recipe requires a YAML path.' >&2; return 2; fi
                RECIPE="$2"; shift 2 ;;
            --recipe=*)
                RECIPE="${1#*=}"
                if [[ -z "$RECIPE" ]]; then printf '%s\n' '--recipe requires a YAML path.' >&2; return 2; fi
                shift ;;
            *) RUN_ARGUMENTS+=("$1"); shift ;;
        esac
    done
    case "$VRAM_GB" in
        12|24|48) ;;
        *) printf '%s\n' '--vram-gb must be 12, 24, or 48.' >&2; return 2 ;;
    esac
    RECIPE="${RECIPE:-$PROJECT_ROOT/configs/gsm8k-${VRAM_GB}gb.yaml}"
}

activate_env() {
    if ! conda activate "$RECREATE3_ENV" >/dev/null 2>&1; then
        if [[ ! -f "$CONDA_EXEC" ]]; then
            printf 'Missing Conda initialization: %s\n' "$CONDA_EXEC" >&2; return 2
        fi
        source "$CONDA_EXEC"
        conda activate "$RECREATE3_ENV"
    fi
}

require_tools() {
    local program
    for program in rsync flock; do
        command -v "$program" >/dev/null || { printf 'Missing program: %s\n' "$program" >&2; return 2; }
    done
}

local_cache_env() {
    export HF_HOME="$LOCAL_EXPERIMENT/cache/huggingface"
    export HF_HUB_CACHE="$HF_HOME/hub"
    export HF_DATASETS_CACHE="$LOCAL_EXPERIMENT/cache/datasets"
    export TRITON_CACHE_DIR="$LOCAL_EXPERIMENT/cache/triton"
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
}
