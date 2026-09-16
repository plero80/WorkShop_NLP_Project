# GSM8K B200: reward and metric bug audit

The saved data support the accuracy improvement. This audit found no reward-sign,
normalization, cache alignment, teacher-routing, or kNN prediction error in the
checks below. It did identify a reporting omission: the previous summary showed
**high-gap detection AUROC**, without showing how well the **actual corrected
reward ranks correct answers**. Those measure different targets.

The gap metrics remain modest after independent recomputation; they have not
been replaced or improved by changing the metric's name. All results below
describe the same completed seed-42 run, analyzed after completion.

## What was checked

The [independent CPU audit](../../reproducibility/audit_gsm8k_rewards.py) reads
the original uploaded ZIP, looks up each cached score/embedding using its
model/protocol identity, question, reference and response, and reconstructs
cosine neighbors, weights and rewards. It imports no project training or
metric functions and loads no model or checkpoint. Its machine-readable
evidence is [reward_bug_audit.json](b200_seed42/reward_bug_audit.json).

| Check | Result |
|---|---|
| Calibration means and standard deviations, both teachers | Reproduced exactly |
| Both 1,024-row memories' gap labels and question ordering | Reproduced exactly at saved float32 precision |
| Proxy embeddings versus the original answer-keyed cache | Exact equality for all memory rows |
| 4B/30B memory geometry | Same embeddings and question IDs; different teacher gap labels |
| kNN replay: 512 selection, 12,800 training, 6,595 final predictions | All 19,907 agree within 0.00000024; every saved neighbor set agrees |
| All four arms' terminal rewards | All 25,600 agree exactly with the intended formulas |
| Per-update mean rewards and optimizer activity | All 1,600 updates checked; each recorded optimizer activity |
| Question partitions and training schedule | All cohorts disjoint by ID; all arms use the same training question schedule |
| Final policy comparisons | All five policies cover the same 1,319 test IDs |
| Cache roles and grades | Saved score and rationale match the corresponding scorer's cache entry, including the 30B preparation and 4B final judge |
| Executed source versus inspected source | Eight experiment modules and all 15 shared core files match historical hashes; seven relevant run/adapter functions match historical syntax trees |

The PPO adapter passes `proxy_z - predicted_gap` as the terminal reward into
the original shared PPO trainer. Its placeholder logging fields are not the
reward tensor. The policy prompt contains the question and a fixed worked
example; the reference is supplied to reward grading, not to policy generation.
The saved GPU preflight reports zero old/current log-probability discrepancy
and zero frozen-reference drift. These source/log checks do not independently
reproduce the complete GPU optimizer trajectory.

This audit rechecked **32,707 strict correctness labels** across training,
selection and final answers, plus **7,107 numeric labels** where that protocol
was recorded. All **224 pre-existing validation metric values** remain identical;
the 42 new strict/numeric reward AUROCs agree independently with scikit-learn.
All 1,782 imported original files retain their recorded SHA-256 hashes. These
checks are recorded in [reward_audit_verification.json](b200_seed42/reward_audit_verification.json).
The previous audit also reproduced the original five final metric sets. See the
[original accuracy audit](B200_RESULTS_ANALYSIS.md).

## The missing comparison

Each row compares proxy and corrected rewards on **identical answers**, with
strict answer correctness as the binary target. Higher reward predicts a
correct answer. No high-gap cutoff enters this calculation.

| Answer cohort | Answers | Proxy reward correctness AUROC | Corrected reward correctness AUROC |
|---|---:|---:|---:|
| Validation, 4B memory | 256 | 0.7178 | **0.8788** |
| Validation, 30B memory | 256 | 0.7178 | **0.8775** |
| Base policy's final answers, 4B memory | 1,319 | 0.6595 | **0.8720** |
| 4B-kNN policy's training answers, 4B memory | 6,400 | 0.6548 | **0.8271** |
| 4B-kNN policy's final answers, 4B memory | 1,319 | 0.6432 | **0.8262** |
| 30B-kNN policy's training answers, 30B memory | 6,400 | 0.6446 | **0.8080** |

The former **0.7307** validation AUROC instead ranks `predicted_gap` against
`observed_gap > 1.2097379800099946`. Correctness AUROC ranks
`proxy_z - predicted_gap` against `correct`. Both numbers are valid, and the
second is more directly relevant to rewarding correct answers during PPO.
Neither is the percentage of questions a policy answers correctly.

AUROC uses continuous scores and a specified binary target; a prediction
cutoff is not required. See the official
[scikit-learn AUROC definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html).
The new correctness AUROC must not be presented as an improved high-gap
AUROC, or as the same metric as the HH-RLHF high-gap result.

## Direct evidence from training: breaking proxy ties

The small proxy frequently awards the maximum grade to wrong answers. On the
validation set, **132 answers received 5/5, and 87 of those were strictly
incorrect**. A reward of 5/5 alone cannot distinguish these from correct answers.

In the 4B-kNN arm's training rollouts, there were **948 same-question pairs**
with one strictly correct and one incorrect answer. The proxy tied **501** of
those pairs. Subtracting the predicted gap broke those ties in favor of the
correct answer in **439/501 = 87.6%**; it favored the incorrect answer in 62.
The correction also reversed 9 incorrect proxy preferences and reversed none
of the 344 already-correct proxy preferences on this cohort.

This is observed reward ordering on recorded training examples. It supports
an explanation for why PPO can benefit from the correction even when exact
gap prediction is weak. It does not isolate how much of the final gain comes
from tie-breaking, response length, style, or other learned behavior.

## Why MSE and R2 are still modest

On 4B validation answers, predicting zero gap has MSE **1.4004**; the kNN
prediction has MSE **1.2621**, a **9.9%** reduction. The gap's standard deviation
is **1.1811**, while the prediction's is only **0.3825**. Predictions average
32 neighbors and do not reproduce most of the per-answer gap variation.
The independent R2 calculation is **0.0953**, using the cohort mean baseline
with MSE **1.3951**. This is a real limitation, not an arithmetic error.

A small correction can order tied proxy rewards correctly without accurately
recovering each numerical gap. R2 measures squared error against a mean
baseline; it does not measure tie-breaking or PPO improvement. See
[scikit-learn R2](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html).

The final strict accuracy comparison is **28.28% base**, **21.15% proxy PPO**,
and **43.82% 4B-kNN PPO**. The +22.67 percentage points over proxy PPO therefore
includes avoiding the proxy arm's 7.13-point decline below the base. The gain
over the base is **15.54 points**. One seed cannot establish reproducibility
across training seeds.

## Confirmed diagnostic issue and report changes

The original strict Q95 high-gap rule had zero positives because Q95 equaled
the largest possible gap. Its AUROC was correctly unavailable. The previously
added validation handles supported alternative label definitions. That cutoff
is diagnostic only: it was never a gate on this run's continuous PPO reward.

Validation v2 adds clearly named strict/numeric **reward-correctness AUROCs**
beside the retained high-gap and regression metrics. The
[new report](b200_seed42/validation_v2/report.md) preserves the
[v1 report](b200_seed42/validation/report.md) and the original run files.
The selected cutoffs and all pre-existing numerical metrics are unchanged.
Two regression tests cover useful reward ranking with poor gap R2 and missing
grades/independent numeric labels. All 156 GSM8K/CLI/launcher tests passed.

Reproduce the independent cache/reward replay from the repository root:

```bash
python workshop_project/reproducibility/audit_gsm8k_rewards.py /path/to/gsm8k_outputs.zip /path/to/new_audit.json
```

This replay needs NumPy, scikit-learn, threadpoolctl, Python 3.11+, and this
Git checkout with its historical source commit. Validation alone needs neither
the reward cache nor Git history; see [VALIDATION.md](VALIDATION.md).
