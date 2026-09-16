# GSM8K B200: exploratory high-gap thresholds

For the **4B high-gap diagnostic**, the best tested upper-tail definition on
the selection set is **`gap > 1.2097379800099946`**, the calibration 90th
percentile: **AUROC 0.731**, with a question-bootstrap 95% interval of
**[0.662, 0.796]**. The 80th and 85th percentiles produce exactly the same
cutoff in this run. This is the highest observed selection result among the
tested upper-tail definitions, not an established universal optimum.

An equally effective definition here is **`gap >= 1.9198088541157206`**:
include ties at the original 95th percentile. Both definitions identify
exactly the same score pair: **proxy grade 5, judge grade 1**. This equivalence
was checked over all 25 possible integer grade pairs, not just observed rows.

The saved run used strict `>` at that maximum value, so it had no positive
labels and undefined AUROC. The problem is saturation and the treatment of
ties, not evidence that kNN has no ranking ability. The percentile is not a
confidence level.

## What is being selected

There are two different cutoffs:

| Cutoff | Purpose | Effect |
|---|---|---|
| Label cutoff, `gap > tau` | Defines which observed proxy-minus-judge gaps count as high | Changes the classification target, so AUROCs refer to different tasks |
| Prediction cutoff, `predicted_gap >= t` | Flags an answer using its kNN prediction | Changes precision/recall; continuous-score AUROC stays the same |

AUROC uses continuous `predicted_gap`, with larger values predicting a higher
gap. It integrates over prediction cutoffs. The distinction between continuous
scores and thresholded decisions follows the
[scikit-learn threshold guide](https://scikit-learn.org/stable/modules/classification_threshold.html)
and [AUROC definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html).

This is a **post hoc diagnostic analysis of one completed seed**, using
existing answers and predictions. Threshold candidates come from 256
calibration answers to 128 questions. Ranking uses the separate **256 selection
answers to 128 questions**. The 1,024 memory answers to 512 questions are
disjoint from both cohorts. Final answers do not select the cutoff.

The search includes calibration Q05 through Q95 in five-percentile steps,
every distinct observed calibration gap, zero, and inclusive Q95. A candidate
needs at least 20 positive and 20 negative selection answers, each class
covering at least 20 questions. The primary upper-tail comparison additionally
requires a nonnegative cutoff and at most 50% positive calibration answers.
This restriction preserves a meaning of unusually large positive gaps; it
is an exploratory choice, not part of the original protocol.

Candidates are ranked by selection AUROC. Equal results prefer strict `>`,
then the highest named percentile. Percentiles can coincide because grades
are discrete. The complete search, including unsupported cutoffs and all
aliases, is in [threshold_candidates.csv](b200_seed42/threshold_candidates.csv).

## 4B selection results

| Calibration percentile | Label cutoff | Positive / 256 | AUROC | Conditional 95% interval |
|---|---|---:|---:|---|
| Q50 / Q55 | `gap > 0.095124` | 108 | 0.628 | See full CSV |
| Q60 / Q65 | `gap > 0.499667` | 86 | 0.666 | See full CSV |
| Q70 / Q75 | `gap > 0.805195` | 67 | 0.687 | [0.623, 0.749] |
| **Q80 / Q85 / Q90** | **`gap > 1.209738`** | **27** | **0.731** | **[0.662, 0.796]** |
| Q95, original | `gap > 1.919809` | 0 | Undefined | Undefined |
| Q95, including ties | `gap >= 1.919809` | 27 | 0.731 | [0.662, 0.796] |

Table cutoffs are rounded for display; reproduce with the full precision in
the first paragraph or the JSON. The selected definition has 20/256 calibration
positives (7.81%) and 27/256 selection positives (10.55%, across 26 questions).
Ties mean that a percentile does not guarantee the nominal positive fraction.

The AUROC difference from the next distinct upper-tail definition is +0.043,
with paired question-bootstrap interval **[-0.025, +0.115]**. The sample does
not establish that this cutoff is better than its runner-up.

## Why maximizing AUROC without a target definition is misleading

The unrestricted search obtains larger AUROCs by changing most answers into
positive cases:

| Teacher | Unrestricted winning cutoff | Selection positives | Selection AUROC | Conditional 95% interval |
|---|---|---:|---:|---|
| 4B | `gap > -0.920475` (Q30 alias) | 181/256 = 70.70% | 0.757 | [0.682, 0.825] |
| 30B | `gap > -0.958872` (Q25 alias) | 191/256 = 74.61% | 0.801 | [0.729, 0.863] |

These are valid ranking statistics for those alternative targets. They include
negative and ordinary gaps and do not measure detection of only severe proxy
overestimation. Report their different label definitions if using them; the
higher number is not an improvement to the fitted predictor.

## 30B selection is a separate diagnostic

| Calibration definition | Cutoff | Selection positives / 256 | AUROC | Conditional 95% interval |
|---|---|---:|---:|---|
| Q50 / Q55 / Q60 / Q65, strict | `gap > 0.321031` | 74 | **0.676** | [0.612, 0.739] |
| Q70, strict | `gap > 0.837016` | 66 | 0.671 | [0.602, 0.737] |
| Q75 through Q95, strict | `gap > 1.435646` | 0 | Undefined | Undefined |
| Q95, including ties | `gap >= 1.435646` | 66 | 0.671 | [0.602, 0.737] |

The best tested upper-tail cutoff is `0.3210313895764224`. Its advantage over
the extreme-grade definition is only **0.005 AUROC**, with interval
**[-0.025, +0.034]**. Lowering the percentile to 90 alone does not fix the
30B diagnostic. Inclusive Q95 selects proxy grade 5 / teacher grade 1 here
too, but includes 68/256 calibration answers (26.56%) because the teacher
often gives the minimum grade.

For a future comparison specifically about maximum disagreement, an explicit
**proxy 5 / teacher 1** label is interpretable across teachers. The two
teacher-specific AUROCs still measure prediction of different teachers'
judgments. Choose and freeze the intended target before collecting new results;
do not present the maximum among different targets as a model improvement.

## Final evaluation using the fixed 4B cutoff

The following uses `gap > 1.2097379800099946` for every policy. Each policy
answers the same 1,319 test questions. All saved final diagnostic gaps and
predictions use the **common 4B judge and 4B memory**, including the policy
trained with 30B-memory rewards.

| Policy producing the answers | High-gap answers / 1,319 | High-gap rate | AUROC | Conditional 95% interval |
|---|---:|---:|---:|---|
| Base | 72 | 5.46% | 0.661 | [0.611, 0.718] |
| Proxy PPO | 131 | 9.93% | 0.698 | [0.655, 0.743] |
| Judge PPO | 15 | 1.14% | 0.667 | [0.494, 0.812] |
| kNN PPO, 4B memory | 22 | 1.67% | **0.691** | **[0.556, 0.811]** |
| kNN PPO, 30B memory | 16 | 1.21% | 0.708 | [0.585, 0.820] |

These are detector diagnostics on different policies' answer distributions,
not a ranking of policy accuracy or a comparison of 4B versus 30B predictors.
The few positives for the trained judge/kNN policies produce wide intervals.
There are no saved 30B final grades in this run, so no 30B-labeled final AUROC
is claimed. See [final diagnostic CSV](b200_seed42/threshold_final_metrics.csv).

## If the goal is to flag answers for review

With the selected 4B label definition held fixed, the prediction cutoff
`predicted_gap >= 0.09989431500434875` maximizes balanced accuracy on selection:
**0.723**, with 27 true positives, 127 false positives, 102 true negatives,
and no false negatives. Its recall is 100%, specificity 44.54%, and precision
17.53% on that same selection set. It flags 154/256 answers, so it is a costly
review rule. This separately tuned operating point does not increase AUROC
and has not been validated as a deployment rule. A review budget or desired
precision would imply a different objective and cutoff.

## Reproduction and limits

From the repository root, using the local imported output directory:

```bash
python -B workshop_project/reproducibility/analyze_gsm8k_thresholds.py
```

For another output directory, supply `--input PATH` and a separate `--output
PATH`. The helper requires NumPy and imports the existing project's AUROC
function. It launches no model, changes no original score, and writes only
separate analysis JSON/CSV files. Training configuration, normalization,
memory contents, rewards, checkpoints, and original reported metrics retain
their recorded values. This diagnostic reanalysis requires no PPO rerun.

[Selection settings](b200_seed42/threshold_selection.json) record the choices
before the script opens final responses. The [audit](b200_seed42/threshold_audit.json)
records input and code hashes, recomputation of saved gaps/Q95, disjoint-cohort
checks, and checks that inputs stayed unchanged. Source archive provenance is
in the [original analysis](B200_RESULTS_ANALYSIS.md#provenance-and-checks).
An independent check against scikit-learn reproduced all 70 defined AUROC
entries in the two CSV tables; the other 10 correctly had no defined AUROC.
Both selection operating points were checked against balanced accuracy, and
the project's preservation check passed for all 153 tracked source-map files.

All intervals use 2,000 bootstrap resamples of questions, seed 42, retaining
both answers for a sampled selection question. They condition on the fitted
memory, normalization, and chosen label definition. They do not account for
searching thresholds, calibration uncertainty, training-seed variation, or
multiple comparisons. Selection scores are optimistically selected; intervals
are descriptive. The benchmark and completed run had already been inspected,
so this remains exploratory even though final scores do not pick the cutoff.
