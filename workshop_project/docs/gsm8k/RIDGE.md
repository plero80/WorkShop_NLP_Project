# Ridge on GSM8K

The ridge extension reads a completed GSM8K run, fits on its exact static 4B
kNN memory, and trains **only one new ridge PPO arm**. It uses the existing
`gsm8k_experiment.run.train_arm` and shared `PPOTrainer`. It does not restart
the saved proxy/kNN controls or change their files.

## Run

From `workshop_project/`, in the matching CUDA environment:

```bash
python -m pip install -r requirements-gsm8k-ridge.txt
python gsm8k.py ridge run --source /workspace/WorkShop_NLP_Project/run/gsm8k/gsm8k_outputs/b200 --dry-run
python gsm8k.py ridge run --source /workspace/WorkShop_NLP_Project/run/gsm8k/gsm8k_outputs/b200
```

Replace `--source` with the actual **completed output folder**, not a code folder
or ZIP. For a native run, it may instead be
`/workspace/WorkShop_NLP_Project/workshop_project/gsm8k_outputs/b200`.
Use the **full original output**, including `reward_cache.sqlite`,
`initial_trainable.pt`, prepared memories, splits and control evaluations.
The small shareable result export does not contain all of these files.
If using your original `gsm8k_outputs.zip`, extract it and point to its `b200/`.
No control checkpoint weights are needed: ridge starts at update zero using
the saved initial adapter and value head, just as those controls did.

The [YAML recipe](../../configs/experiments/gsm8k-ridge.yaml) sets the sources,
separate output root, alpha grid and CPU threads. Override it with `--config`.
The update target, seed, normalization, prompt schedule, generation settings,
reward rubric and PPO settings come from the source controls. They cannot be
silently changed. The original runtime package versions are checked before GPU
training; the requirements above match the archived B200 run (torch 2.8.0).
Hardware may differ, so bitwise equality across GPUs is not claimed.

For an offline fit and report **without GPU training or downloads**:

```bash
python gsm8k.py ridge prepare --source /path/to/b200
```

For three matched saved seeds, point `--source` to their completed suite folder
(the one containing `suite_protocol.json`), or repeat `--source` for individual
seed folders. Each ridge arm uses that seed's own saved memory and initial state.
Supplying one source produces one seed, not three independent training seeds.

Rerunning the same command resumes ridge checkpoints. `--output` selects a new
result directory; it must be separate from the saved source. Do not change code,
inputs or settings while resuming. The extension uses current project code even
when its source is an older runtime; the old runner's sealed controls are not
reopened for training.

## Comparison

- Fit on the same 512 memory questions / up to 1,024 valid answer labels as the
  saved kNN control, including its shared valid subset when 30B grading excluded
  examples. No validation labels are added to fitting memory.
- Select alpha by question-weighted gap MSE on the separate 128 selection
  questions / up to 256 answers. The target is actual normalized proxy minus
  4B judge score, not a kNN prediction.
- Freeze selections before reading evaluation answers. Both predictors use the
  existing kNN validation rule's high-gap definition. It may favor kNN on
  validation; ridge cannot select a different target to improve its own AUROC.
  Original 95th-percentile metrics are retained as legacy diagnostics; discrete
  grades can leave that definition with no positive examples.
- Ridge PPO reward is normalized proxy minus ridge-predicted gap (or the saved
  positive-only correction if that was the source protocol). Training calls
  only the proxy. The 4B judge is used for monitoring and final evaluation.
- Match the completed controls' target: for the saved full seed-42 run this is
  **400 rollout attempts / 6,400 sampled answers**, plus monitoring every 25
  attempts and evaluation on **1,319 official test questions**. A pilot source
  stays a pilot comparison; it is not mislabeled as final evaluation.
- Missing grades retain the existing review/exclusion behavior. They do not
  become zero/default rewards. Reports show grading coverage and actual updates.

## Outputs

Default: `gsm8k_outputs/ridge/`.

- `seed_<seed>/fitted/`: coefficients, selected alpha, validation grid, common
  high-gap definition, fitting counts and input hashes.
- `seed_<seed>/arms/ridge/`: adapter, resumable checkpoint and per-attempt logs.
- `seed_<seed>/evaluations/`, `review/ungraded/`: answers, scores and review cases.
- `report.md`, `predictor_metrics.csv`: MSE, RMSE, MAE, R2, Pearson, Spearman,
  AUROC and AP for **both predictors on identical answers from every policy**.
- `policy_metrics.csv`, `policy_seed_summary.csv`: accuracy, numeric accuracy,
  judge score, high-gap rate, length, grading coverage, seed means and sample SD.
- `paired_comparisons.json`: ridge minus proxy/kNN question-bootstrap intervals.
- `all_predictions.jsonl.gz`: answers, actual/predicted gaps, corrected rewards
  and high-gap labels. `new_judge_budget.csv` records new scoring costs.

Fitting ridge makes zero new judge calls. A completed 400-attempt extension
normally evaluates 16 monitoring checkpoints x 128 questions plus 1,319 final
answers, excluding retries/probe calls. Saved control answers are reused.
An offline report contains no ridge-policy accuracy until PPO is run.
This is a follow-up on an already inspected benchmark, not a newly untouched
test. Question-bootstrap intervals do not measure training-seed uncertainty.

The [saved seed-42 offline report](b200_seed42/ridge/report.md) contains the first
calculation from your output ZIP. Alpha 0.01 was selected. On the saved kNN
policy's final answers, ridge versus kNN gives MSE 1.011 versus 1.055, R2 0.171
versus 0.135, Pearson 0.428 versus 0.392, and AUROC 0.700 versus 0.691 under the
same validated target. These are predictor results on existing answers;
ridge-policy accuracy has not yet been measured.
