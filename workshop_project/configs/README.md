# Retained configuration

- [config.json](config.json): shared follow-up experiment settings.
- [best_of_n/](best_of_n/): Best-of-N settings.
- [knn_distillation/](knn_distillation/): distillation settings and notebook defaults.
- [environment/](environment/): shared dependency requirements.
- [gsm8k/](gsm8k/): new GSM8K profiles, shared source hashes, and extra dependencies.
- [experiments/](experiments/): YAML launch presets for the new experiment CLI.
- [experiments/hh-offline.yaml](experiments/hh-offline.yaml): saved-data HH-RLHF
  ridge/kNN comparison and memory-budget ablation, run with `hh_offline.py`.

Exploration settings are embedded in its notebooks. Retained settings are
unchanged. Source guards are under
[original_manifests/](../reproducibility/original_manifests/); saved study
manifests record the settings used for historical results.
