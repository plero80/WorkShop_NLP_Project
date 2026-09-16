# Reward-gap showcase

The active project contains **Best-of-N, exploration, kNN distillation,
follow-up, and GSM8K**. The four retained families preserve their original source,
notebook, settings, and scientific-result bytes. The new
[GSM8K integration](docs/gsm8k/README.md) uses the existing shared PPO and kNN code;
it has no GPU training results yet.

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

## Running the code

GSM8K now supports [YAML recipes and a CLI](docs/EXPERIMENT_CLI.md): after
restoring below, use `python -m experiment_cli run gsm8k`. The notebook calls
the same command.

The notebooks and Python files use the original runtime paths. From
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
