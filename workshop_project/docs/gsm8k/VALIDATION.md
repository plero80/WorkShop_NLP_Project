# Validation of the GSM8K gap predictor

New runs automatically validate the saved gap predictor after preparation and
before PPO. Final reporting applies the frozen validation choices to saved
final answers. The existing selection cohort serves as validation: **128
questions, two answers each** in the default B200 recipe. No additional answers,
grader calls, GPU inference, or PPO training are required for these diagnostics.

To validate a completed run from `workshop_project/`:

```bash
python gsm8k.py validate gsm8k-b200
```

The launcher detects an existing legacy output under `../run/gsm8k`. Validation
uses the current project's CPU analysis code and leaves the historical training
source, config, checkpoints, normalization and scores intact. For another run:

```bash
python gsm8k.py validate gsm8k-b200 --output /path/to/gsm8k_outputs/b200
```

For the imported seed-42 data in this repository, from the repository root:

```bash
python workshop_project/gsm8k.py validate gsm8k-b200 \
  --output results/gsm8k/b200 \
  --destination workshop_project/docs/gsm8k/b200_seed42/validation_v2
```

The default destination is `<run>/validation/gap_validation_v2/`. Version 2 adds
reward/correctness AUROCs; previous v1 reports remain preserved. Outputs are
`report.md`, `metrics.csv`, `summary.json`, teacher-specific selection locks,
label-cutoff and prediction-cutoff candidate tables, and final diagnostic JSONs.
Existing selection locks are reused only when input hashes, analysis source,
settings and bootstrap choices match. For an intentional new analysis, choose
a new `--destination`; do not edit the old lock. Validate each suite seed
separately with its output path and the single-seed recipe.

## Which quantities are optimized

| Quantity | How it is chosen or measured |
|---|---|
| High-gap **label cutoff** | Maximum selection AUROC among supported upper-tail definitions derived from calibration |
| **Prediction cutoff** for reviewing answers | Maximum selection balanced accuracy by default, with the label definition held fixed; F1 is also supported |
| AUROC / average precision (AP) | Continuous predicted gap ranked against the selected binary labels |
| Reward/correctness AUROC | Proxy and corrected reward (`proxy_z - predicted_gap`) ranked against strict answer correctness on identical answers; numeric matching is also reported separately |
| Gap MSE / RMSE / MAE | Continuous predicted gap compared with the observed normalized proxy-minus-judge gap |
| Gap R² | `1 - sum((gap - prediction)^2) / sum((gap - mean(gap))^2)` |
| Corrected-judge R² | The same R² formula, comparing `proxy_z - predicted_gap` with `judge_z` |
| Pearson / Spearman | Linear correlation / rank correlation of continuous predicted and observed gaps |
| Precision / recall / specificity / F1 / balanced accuracy | Computed at the separate frozen prediction cutoff |

**MSE and R² do not have a threshold to optimize.** Changing a binary cutoff
does not change continuous predictions. The existing `knn.k_grid` and
`knn.temperature_grid` select a predictor by minimum selection MSE; for the
same nonconstant target cohort that also maximizes R². The B200 default fixes
k = 32 and temperature = 0.05. This validation step evaluates that saved
predictor and does not refit it or alter PPO rewards.

Reward/correctness AUROC has its own target and no selected cutoff. It can be
strong even when high-gap AUROC or gap R2 is modest: a small correction can
separate correct and incorrect answers tied by the proxy. These metrics are
reported together to prevent that distinction from being lost. They do not
replace each other or measure the fraction of questions answered correctly.
See the [saved-run reward audit](B200_REWARD_BUG_AUDIT.md) for the reproduced
scores, training reward checks and pairwise examples. Correctness comparisons
count missing/nonfinite rewards and unavailable verification labels; an absent
judge grade does not remove an otherwise usable proxy/corrected comparison.

R² is not squared Pearson correlation. It can be negative when predictions
have higher squared error than the evaluated cohort's mean. The report also
includes a zero-gap baseline and the calibration-mean baseline. The evaluation
mean is the mathematical R² reference, not a deployable fitted predictor. With
the same valid rows, corrected-judge MSE equals gap MSE algebraically; their
R² values can differ because the two target variances differ.

Metric definitions follow [scikit-learn R²](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html),
[average precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html),
and [prediction-threshold tuning](https://scikit-learn.org/stable/modules/classification_threshold.html).

## Validation protocol

Candidate label cutoffs include configured calibration quantiles, each distinct
observed calibration gap, zero, and Q95 with inclusive `>=` to handle tied
maximum grades. The selected definition must use a nonnegative cutoff, include
at most half the valid calibration answers, and have sufficient positive and
negative selection examples **and question groups**. The defaults require 20
examples and 20 questions per class. All candidates, including unsupported
ones, remain in the CSV. Ties prefer strict `>`, then the highest configured
quantile. This restricts the target to relatively large positive gaps instead
of relabeling most answers positive just to increase AUROC.

The prediction cutoff is selected separately over observed prediction values,
using `predicted_gap >= cutoff`. Ties prefer the higher cutoff, which flags
fewer examples. A high-recall cutoff may still generate many false positives;
the confusion matrix and flagged fraction expose that tradeoff.

For new runs, the selection files are frozen before final responses are opened.
For already completed runs, `posthoc: true` records that final artifacts already
exist. Selection and final cohorts are checked for question-ID overlap. The
4B and 30B teacher definitions and predictions stay separate. Final 30B metrics
are produced only if 30B-graded final responses exist; a policy trained with
30B-memory rewards normally still has common 4B final diagnostics.

Rows with unavailable/nonfinite gaps or predictions are excluded from these
diagnostics and counted. Single-class AUROC, constant-target R², insufficient
class support, and missing optional teacher outputs remain unavailable. These
conditions do not stop training or invent grades. Original response files
and grading-review records remain available for inspection.

MSE, R² and AUROC intervals resample whole questions, retaining their associated
answers, with the existing `evaluation.bootstrap_samples` and training seed.
They condition on the fitted predictor, normalization and selected definitions;
they do not adjust for threshold selection or estimate training-seed variation.
Selection scores are tuning diagnostics, not independent performance evidence.
No claim of an untouched benchmark follows from adding this analysis.

## YAML settings for new experiments

```yaml
settings:
  validation:
    label_quantiles: [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    minimum_class_examples: 20
    minimum_class_questions: 20
    decision_metric: balanced_accuracy  # alternatively: f1
```

Old saved configs without this section use these defaults for CPU validation.
Changing a training run's config or source still requires a new training output
directory under the existing provenance rules. The standalone `validate`
command can analyze old outputs without resuming their training.
