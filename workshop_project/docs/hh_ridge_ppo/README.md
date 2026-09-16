# Matched M2 ridge-reward PPO continuation

This implements the additional HH-RLHF experiment: fit ridge directly to the
**actual proxy-minus-judge gaps in the same twice-refreshed M2 memory** used by
the saved kNN control, then continue PPO from update 300 to 400 for seeds
**42, 43 and 44**. It imports the existing PPO trainer, continuation loop,
reward scorers, evaluator, cosine search, and human-review generator. There is
one PPO implementation.

The earlier [offline comparison](../hh_offline/RESULTS.md) used the original
7,990-row memory. Its ridge coefficients and numerical results are not the
results of this new M2 experiment. **The new ridge PPO arm has not yet been run.**

## Run on your RTX PRO 6000

Use Linux with the saved HH environment and CUDA. The saved controls used
`torch==2.8.0+cu128`, `transformers==5.17.0`, and `peft==0.20.0`; the launcher
checks these exact recorded versions before reusing the controls. Keep an
existing matching environment. For a fresh environment, install its CUDA torch
build first, then the additional requirements:

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-hh-ridge-ppo.txt
```

The CUDA command follows the [official PyTorch 2.8 instructions](https://pytorch.org/get-started/previous-versions/#v280).

From `workshop_project/`, with the saved artifacts in this organized layout:

```bash
python hh_ridge_ppo.py audit
python hh_ridge_ppo.py run --dry-run
python hh_ridge_ppo.py preflight
python hh_ridge_ppo.py run
```

`audit` works without a GPU and writes the verified existing-control table.
`--dry-run` verifies inputs and prints the plan without starting an experiment.
`preflight` checks the encoder against saved memory and restores all three
parent policy/value/optimizer/RNG states; it performs no PPO updates. `run`
performs the preflight too, selects ridge, trains the new arm, and creates the
reports. Run experiments sequentially on the GPU.

If your saved HH studies are still in their **original runtime layout**, use
the same commands with `--runtime` pointing to that directory, for example:

```bash
python hh_ridge_ppo.py run --runtime /workspace/reward_gap_followup --dry-run
python hh_ridge_ppo.py preflight --runtime /workspace/reward_gap_followup
python hh_ridge_ppo.py run --runtime /workspace/reward_gap_followup
```

That directory must contain `inputs/`, `outputs/study_cfdbaf579047d418/`,
`refresh2_outputs/study_e7a7994106548cb9/`, and
`knn_distillation_outputs/study_c18793a5593485e3/`.
The source is always imported from this checkout; no second code tree is made.
Use `--output /workspace/hh_ridge_results` to select a different output root.

A source-only Git clone is insufficient. In the organized layout the required
saved artifacts are under `data/inputs/`,
`results/followup/study_cfdbaf579047d418/`,
`data/prerequisites/memory_refresh/study_e7a7994106548cb9/`, and
`results/distillation/study_c18793a5593485e3/`. Keep the full update-300 parent
checkpoints and saved update-400 control checkpoints, not only LoRA adapters.
The local audit found them and verified all six control runs. Missing files
produce a specific error before any model is loaded.
The [saved audit and control table](verified_controls/policy_report.md) is
included in Git; its [provenance record](verified_controls/source_audit.json)
also identifies all nine recovered second-refresh prediction sets.

The [YAML recipe](../../configs/experiments/hh-ridge-ppo.yaml) defaults to cached,
pinned models. If they are not cached, set `allow_downloads: true` or set
`extra_hf_cache` to your existing Hugging Face hub cache. Training settings come
from the saved study, rather than a new GPU preset.

## Exact comparison

| Condition | PPO reward | Training |
|---|---|---|
| Proxy | normalized proxy score | Reuse verified update-300 to update-400 control |
| kNN M2 | proxy score minus signed kNN gap | Reuse verified matching continuation |
| Ridge M2 | proxy score minus ridge gap | New 100-update continuation per seed |

For every control, the audit verifies the checkpoint hash, the parent
policy/value/optimizer/RNG fingerprint, source files, training prompt order,
update count, reward memory, fixed normalization, evaluation prompt texts,
generation settings, and saved output checksums. A mismatch stops reuse.
The new arm restores the identical parent state using the original trainer.
PPO rollout/minibatch seeds are independent of reward-arm names in that trainer.

| Role | Per-seed budget |
|---|---|
| M2 fitting labels | **10,038** answers: original 7,990 + first refresh 1,024 + second refresh 1,024 |
| Ridge validation | 400 prompts, base and parent answers: **800** saved answers |
| Predictor evaluation | Another 400 prompts, base and parent answers: **800** saved answers |
| PPO | 2,400-prompt pool; 100 updates x 32 rollouts = **3,200** newly generated answers |
| Monitoring | 512 shared prompts at updates 350 and 400 |
| Final policy evaluation | 512 shared prompts at update 400 |

Every M2 row is used by both ridge and kNN. Ridge fits an intercept on the same
unit proxy embeddings. Alpha is selected by conversation-weighted validation
MSE over the declared grid; there is no refit on validation. kNN retains
`k=31`, temperature `0.05`, and its signed correction, matching the completed
control. All three ridge models are frozen before offline evaluation and new
final policy evaluation. The original high-gap cutoff remains fixed.
The fit uses scikit-learn's [Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
with its direct Cholesky solver, as in the earlier offline baseline.

Validation and offline answers exist, but their proxy embeddings were not
saved. Validation also lacks actual judge labels. The runner recovers features
from the exact answer texts, verifies proxy-score parity, and acquires only the
missing judge labels. **This adds 2,400 validation judge scores across three
seeds.** Those are a ridge-selection cost, separate from the identical M2 fitting
budget. Existing offline/control judge scores are reused. PPO makes zero judge
calls; monitoring/final evaluation adds 4,608 judge-scored answers.

The training pool, validation and offline sets are disjoint from M2 and from
each other. Validation/offline prompts may have been seen by the update-300
parent during its earlier PPO training; they are held out from the new fitting
and continuation. Final policy prompts are the saved fresh HH test groups.
This is a retrospective extension to already inspected experiments.

## Outputs and resumption

Outputs go to `results/hh_ridge_ppo/study_<identity>/`. The identity includes
input/source hashes, settings, and analysis-library versions. Rerun the same
command to resume. Scoring batches, selected ridge bundles, PPO checkpoints,
and evaluation batches are reused after their dependency checks. Ctrl+C requests
a pause at a completed batch/update. A `PAUSE` file also pauses; remove it before
resuming. A completed run verifies its saved artifacts and exits.

The two main reports are:

- `reports/predictor_report.md`: AUROC, average precision, predictive R2,
  Pearson, Spearman, MSE and MAE on identical held-out answers; individual seeds
  and seed mean/SD CSVs. Additional tables cover shared post-PPO answers and
  the recovered second-refresh evaluations.
- `reports/policy_report.md`: mean judge score, high-gap rate, response length,
  EOS completion, length-capped fraction and refusal diagnostic; individual
  seeds, seed means/SD, paired differences and confidence intervals.

`reports/all_predictions.csv.gz` includes prompt ID/text, answer, seed, predictor,
proxy and judge scores, actual gap, predicted gap, corrected reward, and high-gap
label. Policy CSVs retain the original answer-level generation diagnostics.
`reports/new_work_costs.csv` and `label_budget.json` record scoring, fitting and
PPO costs. Times across different GPUs are descriptive, not equal-compute claims.

Nine original second-refresh prediction/feature sets are available locally;
the runner verifies and reads them rather than repeating that PPO training.
Their transfer table evaluates each seed's M2 predictors on identical saved
answers; it does not relabel every source policy's deployed reward as M2.

Policy intervals resample shared prompts with methods paired; the combined
interval averages the three seed results within each prompt first. They are
conditional on those three trained seeds. Seed SD is also reported, and no
multiple-comparison correction or broad seed-population claim is implied.

EOS is an operational completion measure. The refusal diagnostic is a fixed
phrase heuristic, not proof of inappropriate refusal. The unchanged review
generator prepares **120 randomly sampled blinded pairs** (20 per seed and
comparison: ridge/kNN and proxy/kNN). Human ratings remain empty. Share only
`review/ridge_m2_blinded/ridge_m2_blinded_BLINDED.zip` with reviewers, keeping its
private key separate.

If ridge is similar or better, report that outcome. Stronger offline gap
prediction does not guarantee stronger PPO or human-preference outcomes.
