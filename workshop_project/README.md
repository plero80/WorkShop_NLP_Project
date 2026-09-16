# Reward-gap showcase

The active project contains **Best-of-N, exploration, kNN distillation,
follow-up, and GSM8K**. The four retained families preserve their original source,
notebook, settings, and scientific-result bytes. The new
[GSM8K integration](docs/gsm8k/README.md) uses the existing shared PPO and kNN code;
its generated outputs are kept locally.

| Folder | Contents |
|---|---|
| [notebooks/](notebooks/README.md) | The four selected groups plus GSM8K |
| [code/](code/README.md) | Shared implementation, experiment adapters, and tests |
| [configs/](configs/README.md) | Settings and requirements for retained experiments |
| [data/](data/README.md) | Original inputs and saved prerequisite studies |
| [results/](results/README.md) | Results for the four selected families |
| [docs/](docs/README.md) | Experiment and result guides |
| [reproducibility/](reproducibility/README.md) | Checksums, file mapping, and runtime restoration |

Start with [EXPERIMENTS.md](docs/EXPERIMENTS.md), or open the saved
[cluster explorer](results/exploration/analysis_6ee2efa375535c11/cluster_explorer.html).
See [EXPERIMENT_DATA_COUNTS.md](docs/EXPERIMENT_DATA_COUNTS.md) for each experiment's
prompt counts, kNN memory, seeds, and training/evaluation budget.

## Running the code

Run the new HH-RLHF ridge baseline and memory-size ablation on saved data:

```bash
python hh_offline.py run --dry-run
python hh_offline.py run
```

This is a CPU experiment with a [YAML recipe and run guide](docs/hh_offline/README.md).
It uses the saved HH-RLHF embeddings and judge scores without additional PPO.

Run GSM8K from this folder using [YAML recipes and the CLI](docs/EXPERIMENT_CLI.md):

```bash
python gsm8k.py run gsm8k-b200 --dry-run
python gsm8k.py run gsm8k-b200
# Continue a completed pilot to 400 total attempts per arm:
python gsm8k.py run gsm8k-b200 --stage full
```

The launcher imports this project's `code/core` and `code/experiments` directly.
New outputs are saved in `gsm8k_outputs/`; no runtime source copy is created.
Install dependencies with `python -m pip install -r requirements-gsm8k.txt` in your
CUDA Python environment. Dry runs create no files and start no experiment.

An existing Runpod experiment in `../run/gsm8k` keeps using that original runtime
and its saved checkpoints. The launcher recognizes its outputs and routes resume,
status and export commands there. Leave a currently active run alone until it finishes.
Use an absolute `--output` to select a run if both layouts contain that run name.

For submission, use [submission.py](submission.py) to package the source once and
optionally include completed results. See [SUBMISSION.md](docs/SUBMISSION.md).

The other retained experiments and notebooks use their original runtime paths. From
`workshop_project/`, restore them into a new directory:

```text
python -B reproducibility/manage.py restore --destination ../run/reward_gap_followup
```

Restore preserves runtime paths and verifies checksums. It does not launch
experiments or overwrite an existing destination. Removed experiment sources
and notebooks are excluded from restoration.

Best-of-N needs saved teacher-comparison checkpoints and memory. Distillation
and the second-refresh explorer need saved memory-refresh outputs. These live
in [data/prerequisites/](data/prerequisites/README.md) and are restored to their
original paths. Supporting cohorts remain available for prompt-exclusion checks.

## Verification

```text
python -B reproducibility/manage.py verify --code-only
python -B reproducibility/manage.py verify
```

`--code-only` checks retained source, settings, notebooks, and original input
files. The full check includes saved results and prerequisites.
The earlier workspace snapshot and submission ZIPs remain in
`../original_project/`; they describe the previous, larger project.
