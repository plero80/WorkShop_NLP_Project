# Selected experiments

| Experiment | Notebooks | Implementation | Settings | Results |
|---|---|---|---|---|
| Best-of-N | [best_of_n](../notebooks/best_of_n/) | [best_of_n](../code/experiments/best_of_n/) | [settings.json](../configs/best_of_n/settings.json) | [best_of_n](../results/best_of_n/) |
| Exploration | [exploration](../notebooks/exploration/) | Analysis code embedded in the notebooks | Notebook configuration cells | [exploration](../results/exploration/) |
| kNN distillation | [distillation](../notebooks/distillation/) | [knn_distillation](../code/experiments/knn_distillation/) | [settings.json](../configs/knn_distillation/settings.json) | [distillation](../results/distillation/) |
| Follow-up | [followup](../notebooks/followup/) | [core](../code/core/) | [config.json](../configs/config.json) | [followup](../results/followup/) |
| GSM8K | [gsm8k](../notebooks/gsm8k/) | [gsm8k_experiment](../code/experiments/gsm8k_experiment/) using shared core PPO | [settings.json](../configs/gsm8k/settings.json) | Not run on GPU; [guide](gsm8k/README.md) |

Use the [restored runtime layout](../README.md#running-the-code) for execution.
Retained detailed protocols are in [original/](original/).

## Saved prerequisites

Best-of-N loads its fixed generator checkpoints and memory from the completed
teacher-comparison study. Distillation loads full parent checkpoints and frozen
memory from the completed second-refresh study. The second-refresh explorer
reads that study's saved answers and vectors. These prerequisite outputs remain
in [data/prerequisites/](../data/prerequisites/README.md); their experiment
notebooks and training implementations have been removed from the active project.

Saved cohorts in the retained results and prerequisites remain available to
the retained code's prompt-exclusion checks.
