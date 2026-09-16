# Showcase results

| Study | Saved report or table |
|---|---|
| Best-of-N development | [interpretation](../results/best_of_n/study_40100cfcb78f1920/development/reports/INTERPRETATION.md), [primary result](../results/best_of_n/study_40100cfcb78f1920/development/reports/primary_result.json), [summary](../results/best_of_n/study_40100cfcb78f1920/development/reports/summary.csv) |
| Cluster exploration | [interactive explorer](../results/exploration/analysis_6ee2efa375535c11/cluster_explorer.html), [geometry summary](../results/exploration/analysis_6ee2efa375535c11/geometry_summary.json), [cluster summary](../results/exploration/analysis_6ee2efa375535c11/cluster_summary.csv) |
| kNN distillation | [student fidelity](../results/distillation/study_c18793a5593485e3/reports/post_ppo_student_teacher_fidelity.json), [final seed means](../results/distillation/study_c18793a5593485e3/reports/final_seed_means.csv), [paired deltas](../results/distillation/study_c18793a5593485e3/reports/paired_deltas_by_seed.csv) |
| Follow-up | [HTML report](../results/followup/study_cfdbaf579047d418/reports/report.html), [seed summary](../results/followup/study_cfdbaf579047d418/reports/seed_summary.csv), [paired differences](../results/followup/study_cfdbaf579047d418/reports/paired_differences.csv) |

Best-of-N's primary development comparison is a judge-score difference of
+0.035679 for kNN versus proxy selection, with a conditional 95% paired-prompt
interval [0.015389, 0.057550]. No completed confirmation run is supplied.

The saved exploratory geometry analysis reports raw proxy-judge MSE 0.259860
and kNN-corrected MSE 0.159575. It describes the saved vectors and responses;
it is not a PPO improvement test or human-confirmed reward-hacking assessment.

The distilled reward closely reproduces the kNN teacher, with correlations
around 0.99. Fidelity to that reward does not establish improved policy quality;
retain the final policy comparisons and their uncertainty when presenting it.

The selected results and supporting parent artifacts have not been recomputed.
Supporting parent studies are under
[data/prerequisites/](../data/prerequisites/README.md). Earlier evaluation reports
remain in the separate `original_project/` snapshot.
See [SAVED_RUNS.md](SAVED_RUNS.md) for completion status and source provenance.
