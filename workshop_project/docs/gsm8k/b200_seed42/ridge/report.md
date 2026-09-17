# GSM8K ridge comparison

Ridge fits actual signed 4B gaps on exactly the saved kNN memory. Alpha is selected by question-weighted validation MSE. All selections are frozen before this report reads policy outcomes. No validation labels are added to the fitting memory.

This is a follow-up on previously inspected GSM8K results. Saved controls are reused; only ridge receives new PPO training. The new policy starts from the saved initial actor/value state, with the same seed, prompt schedule, optimizer settings and target. Different GPU hardware can still introduce numerical differences.

Both predictors use the same diagnostic target selected by the existing kNN validation rule on calibration/selection only. That existing rule chooses the supported upper-tail target with highest validation kNN AUROC, so it can favor kNN on validation. Ridge does not tune its own label cutoff. The legacy 95th-percentile threshold is also recorded; it can have no positives with discrete grades. AUROC ranks continuous predicted gaps. Selection metrics are tuning diagnostics, not held-out evidence.

| Seed | Answers from | Predictor | Pairs | MSE | R2 | Pearson | Spearman | AUROC | AP |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 42 | selection | knn | 256 | 1.2621 | 0.0953 | 0.3114 | 0.3341 | 0.7307 | 0.1789 |
| 42 | selection | ridge | 256 | 1.1802 | 0.1540 | 0.4000 | 0.4072 | 0.6733 | 0.2321 |
| 42 | final/base | knn | 1319 | 1.2946 | 0.1343 | 0.3972 | 0.4245 | 0.6614 | 0.0883 |
| 42 | final/base | ridge | 1319 | 1.2005 | 0.1973 | 0.4499 | 0.4918 | 0.7216 | 0.1280 |
| 42 | final/proxy | knn | 1319 | 1.2616 | 0.1490 | 0.4027 | 0.4153 | 0.6981 | 0.1991 |
| 42 | final/proxy | ridge | 1319 | 1.2161 | 0.1796 | 0.4272 | 0.4525 | 0.7165 | 0.2096 |
| 42 | final/knn_static | knn | 1319 | 1.0550 | 0.1350 | 0.3919 | 0.4502 | 0.6911 | 0.0436 |
| 42 | final/knn_static | ridge | 1319 | 1.0114 | 0.1708 | 0.4276 | 0.4799 | 0.6995 | 0.0453 |

| Seed | Cohort | Policy | Updates | Strict accuracy | Numeric accuracy | Mean judge | High-gap rate | Mean tokens |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 42 | final | base | 0 | 0.2828 | 0.2858 | 2.8506 | 0.0546 | 105.3556 |
| 42 | final | proxy | 400 | 0.2115 | 0.2115 | 2.5436 | 0.0993 | 72.3442 |
| 42 | final | knn_static | 400 | 0.4382 | 0.4382 | 3.4155 | 0.0167 | 173.9522 |

Ridge policy results appear only after its GPU run finishes. Missing grades remain unscored and use the existing review queue; no replacement reward is invented. Accuracy retains every question. See each seed's `review/ungraded/` for failed grades.

`all_predictions.jsonl.gz` contains the underlying answers and both predictions. CSV files contain all metrics, including MAE. `paired_comparisons.json` contains question-bootstrap intervals for ridge minus each control. These intervals condition on the trained policies; `policy_seed_summary.csv` reports seed means and sample SD separately. `new_judge_budget.csv` records new requests/cache hits/retries; ridge fitting itself uses zero new judge calls.
