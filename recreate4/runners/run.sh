#!/usr/bin/env bash
set -e
set +m
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
if [[ "${1:-}" == --help || $# == 0 ]]; then
    printf 'Usage: bash recreate3/runners/run.sh EXPERIMENT_ID [--dry-run|--check] [--vram-gb 12|24|48] [--recipe PATH] [--stage prepare|pilot|full] [--parallelism auto|seeds|arms] [--gpus 0,1,2] [--seeds 42] [--base-seed 42]\nDefault VRAM profile: 24. Profile 12 parks inactive graders in host RAM; budget roughly 32 GB RAM per concurrent job (64 GB gives more headroom for one job). This is unmeasured planning guidance. Custom recipes for profiles 12/48 must match their batching/offload settings; 24 retains legacy recipe validation.\nDefault auto: 1-2 GPUs run independent seeds; 3+ GPUs run one seed across its three arms on the first three GPUs. Extra GPUs are unused. Use --parallelism seeds for one complete seed per GPU, or arms with at least three GPUs and one seed. Full stage: 400 updates per arm. Auto preserves existing scheduling on resume; use a fresh ID to change scheduling or VRAM profile.\n'
    if [[ "${1:-}" == --help ]]; then exit 0; else exit 2; fi
fi
init_paths "$1"
shift
select_run_profile "$@"
set -- "${RUN_ARGUMENTS[@]}"
activate_env
INSPECT_ONLY=0
for argument in "$@"; do if [[ "$argument" == --dry-run || "$argument" == --check ]]; then INSPECT_ONLY=1; fi; done
if (( INSPECT_ONLY )); then
    exec python -u "$HELPERS/launch_seeds.py" --runtime "$PROJECT_ROOT/runtime" --recipe "$RECIPE" \
        --output-root "$LOCAL_EXPERIMENT/work/runs" --state-root "$SCRATCH_EXPERIMENT" --log-root "$LOGS_ROOT" --vram-gb "$VRAM_GB" "$@"
fi
require_tools
mkdir -p -m 700 "$TMP_ROOT"
mkdir -p "$LOCAL_EXPERIMENT" "$SCRATCH_EXPERIMENT" "$LOGS_ROOT" "$MODEL_CACHE" "$DATA_CACHE" "$DATA_METADATA"
exec 9>"$SCRATCH_EXPERIMENT/.pipeline.lock"
if ! flock -n 9; then printf 'This experiment already has a running pipeline.\n' >&2; exit 2; fi
python -u "$HELPERS/package_state.py" verify --package "$PROJECT_ROOT"
python -u "$HELPERS/package_state.py" stage --package "$PROJECT_ROOT" --destination "$LOCAL_EXPERIMENT/runtime"
# Read-only CLI/GPU validation before any download or training.
python -u "$HELPERS/launch_seeds.py" --runtime "$PROJECT_ROOT/runtime" --recipe "$RECIPE" \
    --output-root "$LOCAL_EXPERIMENT/work/runs" --state-root "$SCRATCH_EXPERIMENT" --log-root "$LOGS_ROOT" --vram-gb "$VRAM_GB" --check "$@"
printf 'Preparing pinned models and data; follow %s/prepare.log\n' "$LOGS_ROOT"
if ! (
    exec 8>"$MODEL_CACHE/.prepare.lock" || exit $?
    flock 8 || exit $?
    export HF_HOME="$MODEL_CACHE" HF_DATASETS_CACHE="$DATA_CACHE"
    export HF_HUB_CACHE="$HF_HOME/hub"
    unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE
    python -u "$HELPERS/prepare_assets.py" --runtime "$PROJECT_ROOT/runtime" --recipe "$RECIPE" --metadata "$DATA_METADATA" || exit $?
    mkdir -p "$LOCAL_EXPERIMENT/cache/huggingface" "$LOCAL_EXPERIMENT/cache/datasets" || exit $?
    rsync -a --exclude=.prepare.lock -- "$MODEL_CACHE/" "$LOCAL_EXPERIMENT/cache/huggingface/" || exit $?
    rsync -a -- "$DATA_CACHE/" "$LOCAL_EXPERIMENT/cache/datasets/" || exit $?
) >> "$LOGS_ROOT/prepare.log" 2>&1; then
    printf 'Asset preparation failed. See %s/prepare.log\n' "$LOGS_ROOT" >&2
    exit 1
fi
local_cache_env
export RECREATE3_ARCHIVE_ROOT="$SCRATCH_EXPERIMENT/archive"
python -u "$HELPERS/package_state.py" restore --runtime "$LOCAL_EXPERIMENT/runtime" \
    --archive-root "$RECREATE3_ARCHIVE_ROOT" --output-root "$LOCAL_EXPERIMENT/work/runs"
printf 'Asset preparation log: %s/prepare.log\nJob logs: %s/seed_<N>*.log\nSnapshots: %s/archive/<job>/latest.json\nArm scheduling jobs: seed_<N>_prepare, seed_<N>_proxy, seed_<N>_judge, seed_<N>_knn_static; final results: seed_<N>.\n' "$LOGS_ROOT" "$LOGS_ROOT" "$SCRATCH_EXPERIMENT"
exec python -u "$HELPERS/launch_seeds.py" --runtime "$LOCAL_EXPERIMENT/runtime" --recipe "$RECIPE" \
    --output-root "$LOCAL_EXPERIMENT/work/runs" --state-root "$SCRATCH_EXPERIMENT" --log-root "$LOGS_ROOT" --vram-gb "$VRAM_GB" "$@"
