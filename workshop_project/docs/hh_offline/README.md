# HH-RLHF offline baselines

This experiment adds a ridge-regression baseline and a memory-size ablation to
the saved HH-RLHF follow-up data. It compares predictors of the **actual
normalized proxy-minus-judge gap** using the same frozen proxy embeddings and
judge-label budget. It does not rerun PPO or generate answers.

The completed run and its interpretation are in [RESULTS.md](RESULTS.md).

From `workshop_project/`:

```bash
python -m pip install -r requirements-hh-offline.txt
python hh_offline.py run --dry-run
python hh_offline.py run
```

The YAML recipe is [hh-offline.yaml](../../configs/experiments/hh-offline.yaml).
Paths in recipes are relative to `workshop_project/`, regardless of the working
directory. To run only the primary follow-up comparison:

```bash
python hh_offline.py run --skip-transfer
```

You can pass `--config /path/to/recipe.yaml` and `--output /path/to/new_output`.
The default output is `results/hh_offline/comparison_<identity>/`. Inputs,
settings, source and library versions determine the identity. A completed run
is reused after verifying its output checksums; changed inputs require a new
output. No GPU, model download or Hugging Face credentials are needed.

These commands require the saved data, not just a source-only Git clone:

- Original candidate bank and memory under `data/inputs/`.
- Saved follow-up development-validation/offline-test CSVs, feature NPZs and
  completion records under `results/followup/study_cfdbaf579047d418/`.
- For the optional transfer evaluation, second-refresh final features and
  predictions under `data/prerequisites/memory_refresh/study_e7a7994106548cb9/`.

The [notebook](../../notebooks/followup/HH_OFFLINE_BASELINES.ipynb) calls the
same launcher in this checkout. It keeps one implementation of the experiment.

## Comparison protocol

| Role | Data and use |
|---|---|
| Full fitting memory | 7,990 labeled answers, 2,000 full prompt texts, 1,885 normalized first-human-turn groups |
| Validation | 512 distinct groups, two saved policy answers each: 1,024 answers |
| Primary evaluation | Another 512 disjoint groups, two saved policy answers each: 1,024 answers |
| Transfer evaluation | 2,048 other groups; parent/static/refreshed policies with seeds 42/43/44: 18,432 answers total |

Memory, validation, primary evaluation and transfer conversations are disjoint.
Different policies within a cohort intentionally answer the same conversations.
The original normalization and high-gap threshold are shared and frozen. The
original memory's historical calibration uses additional labels; that common
calibration budget is not counted as fitting rows in the ablation.

There are five methods:

1. **Uncorrected proxy:** predict zero gap.
2. **Training mean:** predict the mean of the selected training gap labels.
3. **Original kNN:** cosine retrieval with k=31 and temperature=0.05, using
   the existing `knn_core.top_neighbors` implementation.
4. **Tuned kNN:** choose k and temperature by validation gap MSE, using the
   same selected memory rows and same validation answers as ridge.
5. **Ridge:** fit an intercept and L2-regularized linear predictor of the
   actual gap from the same unit embeddings. Choose alpha by validation MSE.

The primary full-memory comparison is ridge against tuned kNN. Original kNN
remains an explicit reference. All methods use the exact same answers for
evaluation. Neither validation nor test examples are appended to fitting data.
The ridge implementation uses scikit-learn's direct Cholesky solver, with no
feature-wise scaling, learned representation or test-dependent preprocessing.
See [Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
and [data leakage](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage).

## Memory-size ablation

The default budgets are **12.5%, 25%, 50%, and 100%** of conversation groups.
Seeds 42, 43 and 44 produce nested group samples: every answer belonging to a
chosen group is retained. Ridge and both kNN variants share each exact sample.
Report actual answer counts because group sizes vary. Full memory is identical
across these sampling seeds and is evaluated once.

These are **memory-sampling seeds**, not three new PPO runs. The figure's error
bars are standard deviations across memory samples. Hyperparameters are selected
separately for each memory budget using the same validation set; the fixed-kNN
curve keeps k and temperature unchanged. These choices are made before any
evaluation data are parsed. File bytes are hashed before fitting for provenance.

## Outputs and interpretation

Each run saves `report.md`, `metrics.csv`, `summary.json`, validation candidate
tables, fitted ridge coefficients and subset indices, a selection-completion
record, and per-answer predictions in `predictions.csv.gz`. The figure is
exported as PNG, SVG and PDF. Original scientific inputs remain untouched.

Metrics include gap MSE/RMSE/MAE, predictive R2, Pearson/Spearman correlation,
high-gap AUROC/AP, corrected-reward agreement with the judge, and agreement with
the judge's ordering of answers within the same conversation. No HH answer is
assigned a fabricated binary correctness label. Paired full-memory intervals
resample whole conversation groups and compare ridge with tuned kNN on identical
examples; they are conditional on fitted models and the existing saved policies.

The transfer table evaluates **original-memory predictors** on later policy
answers. It does not substitute the policies' refreshed memories or estimate
how those policies would perform if retrained with ridge rewards.

This is a retrospective addition to already inspected experiments. The primary
held-out conversations come from reserved HH training-source data rather than
the official HH test split. The transfer cohort uses the saved second-refresh
final data. Offline prediction improvements do not establish an improvement in
PPO or human preference.

The original memory retains its historical reward labels, which used the earlier
1,024-token scoring limit; later saved evaluations grade complete answers under
their recorded context guard. Both methods receive the same original labels.
The results therefore include transfer across the saved policy and scoring
conditions, rather than a claim of identical training/evaluation distributions.
