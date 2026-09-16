# Saved runs for the selected experiments

Recorded status describes saved artifacts, not a live process check. Source hashes refer to the retained source bytes.

| Saved manifest | Recorded status | Listed source hashes |
|---|---|---|
| [outputs/study_cfdbaf579047d418/manifest.json](../results/followup/study_cfdbaf579047d418/manifest.json) | study_cfdbaf579047d418: complete | 13/13 match |
| [knn_distillation_outputs/study_30a10b1428017bde/manifest.json](../results/distillation/study_30a10b1428017bde/manifest.json) | study_30a10b1428017bde: failed | 15/15 match |
| [knn_distillation_outputs/study_93c278980369efa3/manifest.json](../results/distillation/study_93c278980369efa3/manifest.json) | study_93c278980369efa3: failed | 15/15 match |
| [knn_distillation_outputs/study_c18793a5593485e3/manifest.json](../results/distillation/study_c18793a5593485e3/manifest.json) | study_c18793a5593485e3: complete | 15/15 match |
| [best_of_n_outputs/study_40100cfcb78f1920/manifest.json](../results/best_of_n/study_40100cfcb78f1920/manifest.json) | study_40100cfcb78f1920: complete | 18/18 match |

The two earlier distillation studies have failed status. The completed distillation, Best-of-N development, and follow-up runs remain available. Cluster exploration is a separate descriptive analysis in [results/exploration/](../results/exploration/).

Saved parent studies needed by the retained experiments are in [data/prerequisites/](../data/prerequisites/README.md). The original full audit is retained in [prior_validation/run_source_audit.json](../reproducibility/prior_validation/run_source_audit.json); its removed source paths refer to the previous project snapshot.

## Existing reproduction limitation

`common.py` hashes eligible Python files in the runtime root. `RESUME_KNN_DISTILLATION.py` is an extra root script relative to the saved follow-up manifest, so a fresh run can have a different identity. Saved manifests and identity checks remain unchanged.

Historical package versions are in [recorded_environments.json](../reproducibility/prior_validation/recorded_environments.json).
