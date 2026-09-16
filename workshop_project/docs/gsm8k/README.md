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

## Run from workshop_project

From `workshop_project/`, use the project launcher:

```bash
python gsm8k.py run gsm8k-b200 --dry-run
python gsm8k.py run gsm8k-b200
python gsm8k.py run gsm8k-b200 --stage full
python gsm8k.py status gsm8k-b200
python gsm8k.py export gsm8k-b200
```

It prepares the required files in `../run/gsm8k` automatically and reuses an existing
runtime. The full-stage command continues existing checkpoints to the full target.
It uses the same resolved configuration and output path as launching inside that
runtime. Source or checkpoint identities are not amended by this launcher.

For a fresh environment, `python gsm8k.py setup` prepares only the required code
and configuration and prints the requirements installation command. Install those
dependencies using the existing CUDA Python environment. Setup starts no training.

## Direct runtime commands

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
Default pilot/full targets are 100/400 rollout attempts (successful updates and
attempts skipped for missing rewards are reported separately). Pilot uses the monitor cohort;
full opens the held-out final cohort and fixes its target and arm list.
The same output directory resumes an unchanged protocol. Changes to configuration,
code, versions, or model revisions require a new output directory unless an explicit,
audited upgrade below supports that exact source transition.

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

Offline tests cover generation/log-probability agreement, unequal sequence padding,
microbatch equivalence, frozen reference weights, exact optimizer continuation,
grading and memory behavior, and a tiny-model pilot/full/resume run. Model
fixtures are created locally without downloading pretrained weights.

## Continue past ungradable examples

The reply `Judgement: Correctness_score: 5` contains an explicit rating but the
original parser required the score on its own line. The `grading_inline_score_v1`
fix accepts this exact complete inline form for ratings 1 through 5. It rejects
truncated replies, competing scores, quotes, ranges, and inferred ratings.

If grading still fails after bounded retries, the run saves the full question,
reference, candidate answer, and grader replies under `review/ungraded/`. The
score stays `null`. Failed cases are cached, so resuming does not endlessly retry
the same example. No made-up reward is substituted.

- PPO excludes examples without valid rewards. An entirely ungraded batch skips
  that attempt, then continues; checkpoints and reports record successful and skipped updates.
- Calibration and memory use valid pairs. If a 30B memory label is missing, both
  static memories use the same remaining subset. If too few labels remain for a
  valid calibration or the configured k, affected arms are reported as unavailable;
  other stages continue.
- Evaluation saves every answer and keeps every question in accuracy denominators.
  Grade diagnostics disclose their available-label counts; absent metrics are `null`.
- `ungraded_examples.csv` is generated with the report for convenient review.
  Editing review files does not change stored rewards or training.

To update an existing Runpod runtime, first let any active run exit (or interrupt
it with Ctrl+C). Pull and apply this upgrade from the Git checkout, then resume:

```bash
cd /workspace/WorkShop_NLP_Project
git pull --ff-only
source .venv/bin/activate
python workshop_project/reproducibility/upgrade_gsm8k_ungraded.py --runtime run/gsm8k
cd run/gsm8k
export HF_HOME=/workspace/hf-cache
export TMPDIR=/workspace/tmp
mkdir -p "$TMPDIR"
python -m experiment_cli run gsm8k-b200
```

The upgrade supports both published source versions (before and after the inline
parser fix), including runs with checkpoints. It locks the output and checks exact
source hashes, unchanged shared-engine hashes, configuration, and checkpoint checksums.
It records the parent manifest and original checkpoints under
`source_amendments/ungraded_review_v1/`, updates runtime sources, and changes checkpoint
identity metadata. Weights, optimizer state, RNG state, progress, generated answers,
and cached grades are preserved. Repeating an interrupted upgrade completes it;
repeating a completed upgrade is a no-op. Checkpoint engine and identity checks remain enforced.

This is a recorded change to missing-grade handling, not an unchanged scientific
protocol. The original `repair_gsm8k_inline_score.py` remains available for the specific
historical pre-training parser repair, with its original patch payload preserved.

For a different output, pass `--output gsm8k_outputs/NAME` (relative to the runtime).
Newly restored runtimes already include nonblocking grading. Review records for the
B200 preset are in `run/gsm8k/gsm8k_outputs/b200/review/ungraded/` from the Git checkout.
