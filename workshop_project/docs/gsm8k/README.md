# GSM8K with the existing PPO engine

This experiment adapts `gsm8k_knn_ppo_h200_suite.zip` to the project's shared
`ppo_engine.PPOActor`, `PPOTrainer`, masked GAE, and `knn_core.top_neighbors`.
The original shared source files are unchanged. The experiment's `ppo.py` is
an adapter for sampled responses, rewards, diagnostics, and checkpoints; it
does not implement a second PPO optimization loop.

The default comparison trains the same Qwen2.5 0.5B policy with four rewards:
proxy, direct 4B judge, static kNN correction from the 4B judge, and static
kNN correction from a 30B teacher. The 30B arm keeps the same memory questions,
responses, and proxy embeddings, changing the teacher labels. Optional settings
add refreshed memory and an exact-answer oracle. Evaluation includes numeric
answer accuracy, judge scores, saved responses, paired comparisons, and reports.

No GSM8K training results are included. Integration tests use tiny randomly
initialized local models; they establish code behavior, not math performance.

## Deliberate changes from the ZIP

| Component | Integrated behavior |
|---|---|
| PPO | Existing shared trainer owns clipping, value loss, KL control, GAE, minibatch ordering, and gradient updates |
| Sampling | Temperature **1.0**, matching shared policy log probabilities; ZIP used 0.7 |
| LoRA | q/k/v/o attention projections, matching the shared actor; ZIP also adapted feed-forward projections |
| Optimizer | Shared AdamW epsilon 1e-5, rather than the ZIP's 1e-8 |
| Advantages | Shared masked normalization `sqrt(variance + 1e-8)` |
| kNN | Shared exact cosine search, with GSM8K question-group exclusion and temperature weights |
| Resume | New engine identity, source hashes, and checkpoint checksums; standalone ZIP checkpoints/migrations are rejected |
| Files | Separate `gsm8k_outputs/` and package settings, leaving the existing follow-up config untouched |

The GSM8K prompt, grading rubric/retries, frozen proxy encoder, disjoint question
cohorts, calibration, memory selection, matched 30B teacher, and numeric checker
are retained. This is an adapted experiment, so its results should not be
described as an exact rerun of the standalone ZIP.

## Run from a restored project

For YAML presets, dry runs, and shorter commands, use the
[experiment CLI](../EXPERIMENT_CLI.md). For example, `python -m experiment_cli run
gsm8k` runs the pilot and `python -m experiment_cli run gsm8k --stage full`
continues it. The GSM8K notebook uses this same CLI.

From the repository root, create a fresh runtime containing shared code plus this
experiment (existing saved studies are unnecessary for GSM8K):

```text
python workshop_project/reproducibility/manage.py restore --code-only --destination run/gsm8k
cd run/gsm8k
python -m pip install -r experiment_cli/requirements.txt
python -m gsm8k_experiment.run --stage pilot
python -m gsm8k_experiment.run --stage full
python -m gsm8k_experiment.status
python -m gsm8k_experiment.export
```

Use the project's existing CUDA PyTorch environment. Real runs require CUDA;
the imported batch sizes and 30B teacher target a large-memory GPU. Installation,
model downloads, and training happen only when you run these commands.
Default pilot/full targets are 100/400 updates. Pilot uses the monitor cohort;
full opens the held-out final cohort and fixes its target and arm list.
The same output directory resumes an unchanged protocol. Changes to configuration,
code, versions, or model revisions require a new output directory.

To launch a sequential multi-seed suite:

```text
python -m gsm8k_experiment.suite --seeds 42 43 44
```

See `python -m gsm8k_experiment.suite --help` for stage/target options. Profiles
are in `gsm8k_experiment/configs/`; pass one with `--config`. The default is
`gsm8k_experiment/settings.json`. The 30B revision starts as `main` and is resolved
to an exact commit on first preparation, then shared across suite seeds.

The notebook `RUN_GSM8K_PPO.ipynb` offers the same explicit commands. Reports go
to `gsm8k_outputs/main/report.md`; export omits checkpoint weights. Keep the
original output directory to resume training.

## Offline validation

```text
python -m pip install pytest
python -m pytest tests/gsm8k -q
```

The integration passed **76 offline tests**. Tests cover generation/log-probability agreement, unequal sequence padding,
microbatch equivalence, frozen reference weights, exact optimizer continuation,
grading and memory behavior, and a tiny-model pilot/full/resume run. Model
fixtures are created locally without downloading pretrained weights.
