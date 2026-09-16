# Full HH-RLHF run from scratch

Use **`hh_fresh.py`** for a new pod with no previous experiments. It downloads
the pinned pretrained models and Anthropic HH-RLHF, reserves fresh data splits,
generates new answers, grades them, builds M0/M1/M2, trains all controls and the
ridge PPO arm, and evaluates all three seeds: **42, 43, 44**.

No old checkpoints, candidate-bank labels, embeddings, memories, result ZIPs,
or cached controls are required. This starts PPO from the pretrained Qwen base
policy; it is not language-model pretraining. The original `PPOTrainer`, reward
scorers, kNN search, checkpointing, continuation loop and evaluator are reused.

## New RTX PRO 6000 pod

In the pod terminal:

```bash
cd /workspace
git clone https://github.com/plero80/WorkShop_NLP_Project.git
cd WorkShop_NLP_Project/workshop_project

python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-hh-fresh.txt

export HF_HOME=/workspace/hf-cache
python hh_fresh.py run --dry-run
python hh_fresh.py run
```

The PyTorch command uses the [official CUDA 12.8 wheel installation](https://pytorch.org/get-started/previous-versions/#v280).
The run checks CUDA and BF16 support before loading models. Models and data
download automatically: `allow_downloads: true` is the default in the
[YAML recipe](../../configs/experiments/hh-fresh.yaml). The cache and outputs
stay under `/workspace`. Ensure that workspace storage has room for the model
cache, Python environment and generated results; 50 GB of free space is a
practical starting allowance, not a measured peak for this run.

The final `run` command executes the complete pipeline. There is no `--stage full`
flag to add. `--dry-run` prints budgets without downloading or training.
If the checkout already exists, use `git pull` there instead of cloning again.

For a terminal-independent launch, use this **instead of** the foreground
`python hh_fresh.py run` command:

```bash
nohup python -u hh_fresh.py run > /workspace/hh-fresh.log 2>&1 < /dev/null &
tail -f /workspace/hh-fresh.log
```

Rerun the same command to resume. Generation/scoring batches, completed
memories, selected ridge models, PPO checkpoints and evaluation batches are
checked and reused. Keep the same code, recipe and package versions while
resuming: the run identity includes them. To begin a separate fresh replicate
with the same settings, use a new `--output /workspace/another_hh_run` root.
The runner also supports Ctrl+C or a `PAUSE` file in its study folder, stopping
at a completed batch/update. Remove a `PAUSE` file before resuming.

## What runs

| Stage | Memory / parent | New training per seed |
|---|---|---:|
| Calibration and M0 | Fresh base-policy answers; shared across seeds | No PPO |
| First parent | M0, pretrained policy at update 0 | 100 kNN PPO updates |
| First refresh | Append 1,024 newly judged answers from that parent | No PPO |
| First matched comparison | Same update-100 parent; static M0 versus refreshed M1 | 100 updates per arm |
| Second refresh | Append 1,024 newly judged answers from the refreshed update-200 parent | No PPO |
| Second matched comparison | Same update-200 parent; static M1 versus refreshed M2 | 100 updates per arm |
| Fit ridge | Same M2 embeddings and actual gaps as kNN; separate validation | No PPO |
| Final matched comparison | Same refreshed update-300 parent; proxy, kNN M2, ridge M2 | 100 updates per arm |

All matched forks restore the same policy, value head, optimizer and RNG state.
They share training prompt order and the trainer's arm-independent rollout
seeds. Final-arm order rotates across seeds. Ridge reward is normalized proxy
score minus predicted signed gap; the judge is not called during PPO.

This totals **800 PPO updates per seed**, **2,400 updates across three seeds**,
and **76,800 PPO rollout answers** at 32 answers per update. The final policies
are at update 400; the larger total includes independently trained controls.

## Data and label budgets

| Cohort | Distinct prompts | Answers / use |
|---|---:|---|
| Calibration | 400 | Two base answers each: 800; fit normalization and freeze 95th-percentile high-gap cutoff |
| Initial M0 | 2,000 | Four base answers each: 8,000 labels, shared across seeds |
| Parent training M0 / M1 / M2 | 3,200 per stage | Three disjoint pools; matched arms share each stage's pool |
| Refresh 1 | 1,024 | One current-parent answer per prompt per seed |
| Refresh 2 | 1,024 other prompts | One current-parent answer per prompt per seed |
| Ridge validation | 400 | Base and update-300 parent answers: 800 per seed |
| Offline predictor evaluation | 400 other prompts | Base and parent answers: 800 per seed |
| Final comparison training | 2,400 | 3,200 rollout slots per arm, repeating 800 prompts with new answers |
| Monitoring | 512 | Each final arm at updates 350 and 400 |
| Refresh policy evaluation | 2,048 HH test prompts | Shared across the five first-/second-refresh policy checkpoints per seed |
| Final policy evaluation | 512 other HH test prompts | Shared across proxy, kNN and ridge, all seeds |

Splits are reserved before scoring, using normalized conversation openings to
prevent leakage across multi-turn variants. HH test openings are excluded from
training-source cohorts. Each cohort is disjoint from memory, training and other
evaluation cohorts. Multiple answers to a prompt stay in the same cohort.
Unlike the historical continuation, these new validation/offline prompts are
also excluded from all newly trained parent policies.

Memory grows **8,000 -> 9,024 -> 10,048 rows per seed**. Ridge and kNN receive
identical M2 rows. The new initial count is 8,000, not the old saved 7,990; the
fresh calibration and generated answers also differ. Compare against this run's
newly trained controls rather than combining its outcomes with old controls.

The planned full run scores **64,288 answers with the judge**, including initial
memory, calibration, refreshes, ridge validation, offline evaluation, policy
evaluation and monitoring. PPO adds zero judge calls. Reports verify realized
counts against the declared budget. The recipe fixes kNN at k=31 and temperature
0.05; ridge alpha is selected by validation MSE, never AUROC or final outcomes.

## Outputs

The launcher prints `results/hh_fresh/study_<identity>/`. It contains the new
checkpoints, fitted ridge coefficients, all memories, scores, features, data
split hashes and model/runtime metadata. Under `reports/`:

- `predictor_report.md`, `predictor_by_seed.csv`, `predictor_seed_summary.csv`:
  **Pearson, Spearman, R2, MSE, MAE, AUROC and average precision**, including
  every refresh-2 policy's answers. All methods score the same held-out answers.
- `final_policy_report.md`: proxy/kNN/ridge mean judge score, high-gap rate,
  response length, EOS completion and refusal diagnostic.
- `refresh1_policy_report.md`, `refresh2_policy_report.md`: both newly trained
  memory-refresh comparisons and their controls.
- Per-seed metrics, seed means/SD, paired differences and conditional bootstrap
  intervals. Cross-seed intervals average seeds within prompt before resampling.
- `all_predictions.csv.gz`: prompt, answer, seed, predictor, raw/normalized scores,
  actual/predicted gap, corrected reward and high-gap label.
- `budget.json`, `costs.csv`: completed work and timing by stage. Discarded work
  and model loading are excluded; resumed evaluation time is its last invocation.

The blinded review ZIP under `review/fresh_ridge_blinded/` contains **120 random
answer pairs**. Human ratings remain empty. EOS completion and phrase-based
refusal detection are diagnostics, not human assessments of completeness or
inappropriate refusal.

The pipeline and report orchestration are tested locally with simulated GPU
interfaces. No full fresh GPU training or new scientific outcome is claimed
until your pod completes it.
