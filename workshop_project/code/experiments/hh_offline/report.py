"""Portable tables and publication-exportable memory-budget figures."""
import numpy as np
import pandas as pd

NAMES = {"proxy": "Proxy (zero gap)", "mean_gap": "Training mean gap",
         "knn_fixed": "kNN, original settings", "knn_tuned": "kNN, validation tuned", "ridge": "Ridge"}


def report(out, result, manifest):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = pd.DataFrame(result["results"])
    primary = rows[rows.cohort == "test"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.1), layout="constrained")
    colors = {"knn_fixed": "#007f86", "knn_tuned": "#2658a6", "ridge": "#be552d", "mean_gap": "#777777"}
    for ax, metric, title in zip(axes, ["gap_mse", "high_gap_auroc"], ["Gap prediction error (lower is better)", "High-gap AUROC (higher is better)"]):
        for method in colors:
            frame = primary[primary.method == method]
            grouped = frame.groupby("fraction", sort=True)
            x = grouped.memory_rows.mean()
            mean, std = grouped[metric].mean(), grouped[metric].std().fillna(0.)
            ax.errorbar(x, mean, yerr=std, marker="o", linewidth=1.8, capsize=3,
                        color=colors[method], label=NAMES[method])
        baseline = primary[primary.method == "proxy"][metric].iloc[0]
        ax.axhline(baseline, color="#333333", linestyle=":", linewidth=1., label="Uncorrected proxy")
        ax.set(xlabel="Judge-labeled memory answers (mean across samples)", title=title)
        ax.grid(alpha=.2)
        ax.set_xscale("log", base=2)
        ticks = primary.groupby("fraction").memory_rows.mean().to_numpy()
        ax.set_xticks(ticks, labels=[f"{n:,.0f}" for n in ticks])
    axes[0].set_ylabel("MSE in normalized gap units")
    axes[1].set_ylabel("AUROC")
    axes[1].legend(loc="best", fontsize=8)
    fig.suptitle("HH-RLHF offline memory-size ablation", fontsize=14)
    fig.savefig(out / "memory_ablation.png", dpi=180)
    fig.savefig(out / "memory_ablation.svg")
    fig.savefig(out / "memory_ablation.pdf")
    plt.close(fig)
    fmt = lambda x: "unavailable" if x is None or pd.isna(x) else f"{x:.4f}"
    lines = ["# HH-RLHF: ridge baseline and memory-size ablation", "",
             "This is a retrospective offline comparison using saved proxy embeddings and actual proxy-minus-judge labels. "
             "It adds no PPO runs, model inference, generated answers or grader calls. "
             "Ridge and tuned kNN minimize MSE on the same validation answers and are frozen before evaluation.", "",
             "## Label budget and partitions", "",
             f"The full memory contains **{manifest['training_rows']:,} answers** from **{manifest['training_groups']:,} normalized "
             f"conversation groups**. Validation has **{manifest['validation_rows']:,} answers** from "
             f"**{manifest['validation_groups']:,} groups**. Primary evaluation has "
             f"**{result['cohorts']['test']['answers']:,} answers** from **{result['cohorts']['test']['groups']:,} groups**. "
             "The two held-out answers per group come from the saved raw and signed-kNN policies. "
             "These are reserved HH train-source conversations, not the official HH test split.", "",
             "All methods receive identical training rows, embeddings, target labels and evaluation answers at each memory size. "
             "The proxy control uses no fitted gap labels. Ridge fits an intercept with L2 regularization on the same unit vectors, "
             "without feature-wise rescaling or test-dependent preprocessing. Validation examples are never added to fitting data.", "",
             "The original normalization and high-gap threshold are inherited and fixed. Their historical calibration budget is shared "
             "by all methods and is additional to the memory-fitting budget. The upper-tail target is never changed to maximize test AUROC.", "",
             "## Primary results at full memory", "",
             "| Predictor | Gap MSE | Gap R2 | Gap Pearson r | High-gap AUROC | High-gap AP | Agreement with judge's pair preference |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    full = primary[primary.fraction == 1]
    for _, r in full.iterrows():
        lines.append("| " + NAMES[r.method] + " | " + " | ".join(fmt(r[k]) for k in
                     ["gap_mse", "gap_r2", "gap_pearson", "high_gap_auroc", "high_gap_ap", "judge_preference_agreement"]) + " |")
    lines += ["", "Gap MSE equals the corrected reward's MSE against normalized judge scores. "
              "R2 uses the evaluated cohort's mean as its reference and is not squared correlation. "
              "High-gap AUROC ranks predicted gaps; pair agreement instead ranks proxy_z minus predicted_gap within each conversation. "
              "Judge ties are omitted and reward ties receive half credit. Judge agreement is not verified human preference or answer correctness.", "",
              "## Paired full-memory comparison", "",
              "The prespecified comparison is **ridge minus validation-tuned kNN**. Negative MSE differences favor ridge; positive "
              "AUROC differences favor ridge. Intervals resample whole conversation groups, retaining every method/policy answer for each group.", "",
              "| Evaluation | MSE difference | 95% interval | AUROC difference | 95% interval |", "|---|---:|---|---:|---|"]
    for cohort, interval in result["paired_full_memory_intervals"].items():
        subset = rows[(rows.cohort == cohort) & (rows.fraction == 1)].set_index("method")
        dm = subset.loc["ridge", "gap_mse"] - subset.loc["knn_tuned", "gap_mse"]
        da = subset.loc["ridge", "high_gap_auroc"] - subset.loc["knn_tuned", "high_gap_auroc"]
        def bounds(key):
            value = interval[key]["ci95"]
            return "unavailable" if value is None else "[" + ", ".join(fmt(v) for v in value) + "]"
        lines.append(f"| {cohort} | {fmt(dm)} | {bounds('ridge_minus_knn_gap_mse')} | {fmt(da)} | {bounds('ridge_minus_knn_high_gap_auroc')} |")
    lines += ["", "These are conditional evaluation intervals. They do not include uncertainty from fitting, hyperparameter selection, "
              "training new policies, or choosing experiments after inspecting earlier results. Transfer comparisons are exploratory and have "
              "no multiple-comparison adjustment.", "", "## Memory-size ablation", "",
              "![Memory-size ablation](memory_ablation.png)", "",
              "Export: [SVG](memory_ablation.svg), [PDF](memory_ablation.pdf). Error bars show one standard deviation across "
              "three nested conversation samples, not confidence intervals and not three new PPO seeds. Full memory is identical across "
              "samples and is fitted once. Horizontal positions use actual mean answer counts; whole-group sampling makes these counts vary.", "",
              "| Memory fraction | Groups per sample | Answer-count range | Predictor | Mean test MSE | Mean test R2 | Mean test AUROC |",
              "|---|---:|---|---|---:|---:|---:|"]
    for (fraction, method), frame in primary[primary.method.isin(["knn_fixed", "knn_tuned", "ridge"])].groupby(["fraction", "method"], sort=True):
        lines.append(f"| {fraction:.1%} | {frame.memory_groups.iloc[0]} | {frame.memory_rows.min()}–{frame.memory_rows.max()} | "
                     f"{NAMES[method]} | {fmt(frame.gap_mse.mean())} | {fmt(frame.gap_r2.mean())} | {fmt(frame.high_gap_auroc.mean())} |")
    if len(result["cohorts"]) > 1:
        lines += ["", "## Transfer to saved second-refresh policies", "",
                  "Only full-memory predictors are evaluated here. These are still the predictors fitted to the **original memory**; "
                  "they are not each policy's deployed refreshed-memory reward. Each condition combines its saved seeds 42, 43 and 44, "
                  "with all answers for the same conversation kept together in uncertainty calculations. This measures transfer of an "
                  "offline predictor, not the outcome of replacing a reward model and rerunning PPO.", "",
                  "| Policy-answer cohort | Answers / groups | Predictor | MSE | R2 | AUROC |", "|---|---:|---|---:|---:|---:|"]
        for _, r in rows[(rows.cohort != "test") & (rows.method.isin(["knn_fixed", "knn_tuned", "ridge"]))].iterrows():
            lines.append(f"| {r.cohort} | {r.answers} / {r.conversation_groups} | {NAMES[r.method]} | "
                         f"{fmt(r.gap_mse)} | {fmt(r.gap_r2)} | {fmt(r.high_gap_auroc)} |")
    lines += ["", "## Reproducibility and limits", ""]
    replay = result["original_knn_replay"].get("test")
    if replay:
        lines += [f"Historical fixed-kNN MSE: **{replay['historical_mse']:.8f}**; CPU recomputation: "
                  f"**{replay['recomputed_mse']:.8f}**. There are **{replay['answers_over_1e_5']}** predictions differing "
                  f"by more than 1e-5; maximum absolute difference **{replay['max_abs_error']:.8f}**. "
                  "The shared search uses float32 cosine similarities; near-ties at the kth neighbor can change across "
                  "BLAS/batching implementations. The analysis retains both the historical and recomputed values rather than "
                  "silently replacing the archived predictions.", ""]
    lines += [
              "[Metrics](metrics.csv), [summary and paired intervals](summary.json), [validation candidates](validation_candidates.csv), "
              "and [input/source manifest](manifest.json). Model choices and coefficients are saved under models/ before any "
              "evaluation CSV or feature matrix is parsed. Input bytes are hashed for provenance before fitting; test metrics never select a model. "
              "Original scientific files and the shared PPO/kNN implementation are unchanged.", "",
              "The runner rejects missing, changed, misaligned or overlapping inputs. It validates memory-to-bank indices, saved "
              "score normalization, conversation identities, full-answer scoring and completion hashes. Historical results may already "
              "have been inspected; this is not a newly untouched benchmark. Better offline reward prediction does not establish better PPO.", "",
              "Ridge follows the [scikit-learn Ridge definition](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html). "
              "Selection follows the [train/test separation guidance](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage).", ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
