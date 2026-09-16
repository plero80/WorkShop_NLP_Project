# Saved prerequisites

| Directory | Why retained | Restored runtime path |
|---|---|---|
| [teacher_comparison/](teacher_comparison/) | Fixed generator checkpoints, frozen memory, and source manifests used by Best-of-N | `next_studies_outputs/` |
| [memory_refresh/](memory_refresh/) | Parent checkpoints and memory for distillation; saved responses and vectors for the second cluster explorer | `refresh2_outputs/` |
| [distillation_recovery/](distillation_recovery/) | Original recovery settings for the retained distillation workflow | `knn_distillation_recovery/` |

These are existing input artifacts for the four retained workflows. The separate
teacher-comparison and memory-refresh notebooks and training modules are no
longer active. Keep these directories to continue or inspect the retained runs.
