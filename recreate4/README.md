# Repeat the local GSM8K experiment

Same training code and 24 GB profile as `local-seed42-v2`: proxy, judge, and static-kNN PPO, 400 updates each. The completed run used an RTX 5090 (32 GB). Run commands from this folder.

## Windows

Requires PowerShell 7, Python 3.12, and an NVIDIA GPU with BF16 support. Setup is needed once; change `-PythonBase` if your Python is elsewhere.

```powershell
pwsh -File .\windows\Setup.ps1 -PythonBase "C:\Users\$env:USERNAME\anaconda3\envs\RHC\python.exe"
pwsh -File .\windows\Run.ps1 -ExperimentId seed42-rerun -Gpus 0 -Seeds 42 -VramGB 24 -Background
pwsh -File .\windows\Status.ps1 -ExperimentId seed42-rerun
# After it finishes: another seed, with a new ID.
pwsh -File .\windows\Run.ps1 -ExperimentId seed43 -Gpus 0 -Seeds 43 -VramGB 24 -Background
```

Results: `windows/work/runs/<ID>/suite_report.md`.

## Linux

Requires Bash 5.1+, Conda, `rsync`, `flock`, and an allocated NVIDIA GPU with BF16 support. Set these paths for your cluster; the examples use the original filesystem layout.

```bash
export SCRATCH="/vol/scratch/$USER" TMP_ROOT="/tmp/$USER"
export CONDA_EXEC="$SCRATCH/miniconda3/etc/profile.d/conda.sh" RECREATE3_ENV=recreate4
bash runners/setup.sh
bash runners/run.sh linux-seed42 --gpus 0 --seeds 42 --vram-gb 24 --check
bash runners/run.sh linux-seed42 --gpus 0 --seeds 42 --vram-gb 24
# After it finishes: another seed, with a new ID.
bash runners/run.sh linux-seed43 --gpus 0 --seeds 43 --vram-gb 24
```

For three allocated GPUs, run one reward pipeline per GPU with the same seed:

```bash
bash runners/run.sh linux-seed42-parallel --gpus 0,1,2 --parallelism arms --seeds 42 --vram-gb 24
```

GPU 0 runs proxy PPO, GPU 1 judge PPO, and GPU 2 static-kNN-corrected PPO, each for 400 updates. Preparation runs once, training runs concurrently, and final evaluation follows. GPU order controls assignment: `--gpus 2,0,1` assigns proxy to 2, judge to 0, and kNN to 1. Use a new experiment ID when switching scheduling; append `--check` for a preflight without training.

Results: `$SCRATCH/checkpoints/recreate3/<ID>/suite_report.md`. Logs: `$SCRATCH/logs/recreate3/<ID>/`. The Linux scripts retain the `recreate3` storage subfolder name. Linux setup uses Python 3.11 and the shared pinned requirements; the full Windows dependency lock is only used on Windows. Shell checks passed here; Linux GPU training has not been run here.

Change the seed and experiment ID for a new run. Repeat the same command to resume; use a new ID to start over. Data splits stay fixed at seed 42. Keep the 24 GB profile to preserve batching. Setup downloads dependencies; the first run downloads pinned models/data.

The completed run's settings/results are in `reference_run/`; they are not used as checkpoints. Allow roughly 23 hours on this RTX 5090, plus initial downloads. This reproduces the training code/settings; identical numerical results are not guaranteed.
