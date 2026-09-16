# GSM8K gap-predictor validation

Selection chooses cutoffs; final evaluation uses those fixed cutoffs. These are diagnostic metrics, not policy-answer accuracy. Original run artifacts are unchanged.

## judge

Label definition: `{'quantile': 0.9, 'threshold': 1.2097379800099946, 'comparator': '>'}`.

Prediction cutoff: `{'threshold': 0.09989431500434875, 'comparator': '>=', 'objective': 'balanced_accuracy', 'selection_objective_value': 0.722707423580786}`.

Post hoc on a run with final artifacts already present: **True**.

## judge30b

Label definition: `{'quantile': 0.65, 'threshold': 0.3210313895764224, 'comparator': '>'}`.

Prediction cutoff: `{'threshold': 0.1288985311985016, 'comparator': '>=', 'objective': 'balanced_accuracy', 'selection_objective_value': 0.6638699138699138}`.

Post hoc on a run with final artifacts already present: **True**.

| Cohort | Teacher | Policy | Scored / all | AUROC | AP | Gap MSE | RMSE | MAE | Gap R2 | Corrected-judge R2 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| selection | judge | preparation | 256 / 256 | 0.7307 | 0.1789 | 1.2621 | 1.1234 | 0.8784 | 0.0953 | -0.2770 |
| selection | judge30b | preparation | 256 / 256 | 0.6761 | 0.3903 | 1.1942 | 1.0928 | 0.8511 | 0.0883 | -0.3704 |
| final | judge | base | 1319 / 1319 | 0.6614 | 0.0883 | 1.2946 | 1.1378 | 0.8794 | 0.1343 | -0.3231 |
| final | judge | judge | 1319 / 1319 | 0.6671 | 0.0319 | 1.0021 | 1.0011 | 0.8018 | 0.0765 | -0.0613 |
| final | judge | knn_static | 1319 / 1319 | 0.6911 | 0.0436 | 1.0550 | 1.0271 | 0.7855 | 0.1350 | -0.0509 |
| final | judge | knn_static_30b | 1319 / 1319 | 0.7083 | 0.0307 | 1.0274 | 1.0136 | 0.7978 | 0.0829 | -0.0600 |
| final | judge | proxy | 1319 / 1319 | 0.6981 | 0.1991 | 1.2616 | 1.1232 | 0.8922 | 0.1490 | -0.4911 |

MSE, RMSE, MAE and R2 use continuous normalized gaps. R2 can be negative; it is not squared correlation. Corrected-judge R2 instead compares proxy_z - predicted_gap with judge_z. AUROC/AP use continuous predicted_gap; the separate prediction cutoff determines precision/recall/F1 and balanced accuracy.

Missing/nonfinite pairs are excluded and counted. Single-class AUROC, constant-target R2 and unsupported threshold searches remain unavailable. No fabricated labels are used. Lack of diagnostic support does not stop training.

Intervals in summary.json resample whole questions conditional on the fitted predictor and chosen cutoffs. They do not account for threshold selection, calibration uncertainty or training-seed variability. Selection values are tuning diagnostics. The final benchmark may already have been inspected.

k and temperature remain those of the saved predictor. Existing kNN grid selection minimizes validation MSE; on one fixed target cohort this also maximizes R2. Threshold search does not refit that predictor.
