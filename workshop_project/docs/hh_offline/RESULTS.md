# HH-RLHF offline comparison: completed results

**Ridge outperformed the kNN configurations tested in this offline comparison.**
Both use the same original 7,990 judge-labeled answers and 1,024-dimensional
proxy embeddings. Parameters were selected on the same 1,024 validation answers;
the table below uses another **1,024 answers from 512 disjoint conversations**.
All original data and PPO implementations remain unchanged.

| Gap predictor | MSE (lower better) | Predictive R2 | Pearson r | High-gap AUROC | High-gap AP |
|---|---:|---:|---:|---:|---:|
| Zero gap / uncorrected proxy | 0.2705 | -0.0259 | Undefined | 0.5000 | 0.0303 |
| kNN, original k=31 / temperature=0.05 | 0.1568 | 0.4052 | 0.6383 | 0.9114 | 0.2729 |
| kNN, validation-selected k=63 / temperature=0.01 | 0.1559 | 0.4086 | 0.6460 | 0.9162 | 0.2787 |
| Ridge, validation-selected alpha=0.1 | **0.1282** | **0.5138** | **0.7187** | **0.9440** | **0.3917** |

Ridge's test MSE is **17.8% lower than tuned kNN's** and **52.6% lower than
the uncorrected-proxy baseline**. The paired conversation-bootstrap difference,
ridge minus tuned kNN, is **-0.0277 MSE**, with a conditional 95% interval
**[-0.0369, -0.0186]**. The AUROC difference is **+0.0278**, interval
**[+0.0036, +0.0584]**. There are 31 high-gap answers, so AUROC/AP describe a
relatively small upper tail. This does not establish superiority over every
possible kNN configuration: the selected k and temperature are at boundaries
of the declared search grid.

These intervals keep every answer for a sampled conversation together. They
condition on the fitted predictors and saved policies. They do not estimate
variability from new PPO training or undo the retrospective nature of this
additional analysis.

## Memory-size result

The sampling unit is a normalized conversation group. All answers from the
selected group enter both models. Each partial budget uses three nested samples;
full memory is fitted once because it is identical across sampling seeds.

| Memory fraction | Actual answer budget | Original kNN MSE | Tuned kNN MSE | Ridge MSE |
|---|---:|---:|---:|---:|
| 12.5% | 973–1,016 | 0.1722 | 0.1697 | **0.1538** |
| 25% | 1,985–2,026 | 0.1671 | 0.1662 | **0.1430** |
| 50% | 3,962–4,005 | 0.1599 | 0.1595 | **0.1352** |
| 100% | 7,990 | 0.1568 | 0.1559 | **0.1282** |

Partial-budget entries are means across the three memory samples. More labels
reduce mean MSE for both methods in this experiment. These samples are not three
new PPO training seeds. The figure shows their variation:

![HH-RLHF memory-size ablation](completed/memory_ablation.png)

Downloads: [SVG](completed/memory_ablation.svg), [PDF](completed/memory_ablation.pdf).

## Additional transfer evaluations

The same full original-memory predictors were evaluated on the saved second-refresh
parent, static-memory and refreshed-memory policies. Each condition contains
6,144 answers: 2,048 conversations with outputs from policy seeds 42, 43 and 44.

| Saved policy answers | Tuned kNN MSE / AUROC | Ridge MSE / AUROC |
|---|---:|---:|
| Parent | 0.1553 / 0.8909 | **0.1265 / 0.9197** |
| Static memory | 0.1565 / 0.8917 | **0.1320 / 0.9222** |
| Refreshed memory | 0.1545 / 0.8827 | **0.1297 / 0.9204** |

These are offline transfer results. They do not replace the policies' actual
refreshed-memory rewards and do not show what PPO would do with ridge rewards.
Original-memory labels retain their historical scoring protocol; subsequent
policy answers were graded under the later complete-answer protocol.

## Consequence for the project

This strengthens the evidence that frozen proxy representations contain useful
information about reward disagreement, and that reusing more judge labels helps.
It also changes the comparative claim: **the tested retrieval method does not
beat a simple linear predictor on offline gap fidelity here**. The existing kNN
PPO outcomes remain valid; a ridge-reward PPO comparison would be a separate
experiment. Neither judge agreement nor high-gap detection proves improved
human preference or prevention of reward hacking.

## Verification and reproducibility

- Seven new tests cover grouped sampling, library agreement, input corruption,
  overlap rejection, missing/constant metrics and whole-conversation bootstrapping.
- An end-to-end test changes only test labels and checks that selected model
  coefficients remain identical; it also verifies selection files exist before
  the test loader is called.
- All **325 independently recomputed metrics across 65 method/cohort/budget
  combinations** match the saved report. A separate centered normal-equation
  solution matches full-memory ridge coefficients within **9.1e-14**.
- All **44 input files**, **13 analysis/dependency source files**, and **30 output
  artifacts** passed their checksum checks.
- Two historical kNN predictions differ due to float32 near-ties at the 31st
  neighbor. A single boundary-neighbor substitution reproduces each saved value
  within 1.1e-7. Historical versus recomputed MSE is 0.15681485 versus 0.15682550.
  Both are retained; no archived prediction was rewritten.

Evidence: [verification](verification.json), [neighbor replay audit](neighbor_replay_audit.json),
[full generated report](completed/report.md), [metrics](completed/metrics.csv),
[input/source manifest](completed/manifest.json), and [selection candidates](completed/validation_candidates.csv).
The complete local run also contains compressed per-answer predictions under
`results/hh_offline/comparison_83789341f53dcd52/`.

Run it with `python hh_offline.py run` from `workshop_project/`; see the
[run guide](README.md) for dependencies and required saved inputs.
