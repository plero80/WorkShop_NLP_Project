# Documentation

- [EXPERIMENTS.md](EXPERIMENTS.md): notebook, source, settings, and result folder
  for each experiment.
- [EXPERIMENT_DATA_COUNTS.md](EXPERIMENT_DATA_COUNTS.md): prompt allocations,
  answer counts, kNN memory sizes, seeds, and training/evaluation budgets.
- [RESULTS.md](RESULTS.md): direct links to saved reports and tables.
- [SAVED_RUNS.md](SAVED_RUNS.md): recorded completion states and source-hash checks.
- [hh_offline/README.md](hh_offline/README.md): HH-RLHF ridge baseline, matched kNN
  comparison, memory-size ablation, and CPU run commands.
- [hh_ridge_ppo/README.md](hh_ridge_ppo/README.md): matched M2 ridge-reward PPO
  continuation, verified saved controls, complete reporting, and RTX PRO 6000 commands.
- [hh_fresh/README.md](hh_fresh/README.md): full fresh HH pipeline on a new pod;
  downloads data/models, rebuilds both refreshes, trains all controls and ridge.
- [gsm8k/README.md](gsm8k/README.md): new GSM8K experiment, shared PPO integration, and run commands.
- [gsm8k/B200_RESULTS_ANALYSIS.md](gsm8k/B200_RESULTS_ANALYSIS.md): verified results
  from the completed seed-42 run, comparisons, and diagnostic limitations.
- [EXPERIMENT_CLI.md](EXPERIMENT_CLI.md): YAML presets and common notebook/terminal commands.
- [original/](original/): original project documentation, copied unchanged.

Paths in the original documentation describe the original execution layout.
Use the guides above to browse this organized copy, and the
[restore instructions](../README.md#running-the-code) to execute it.
