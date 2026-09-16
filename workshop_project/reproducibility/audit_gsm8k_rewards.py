"""Independently audit archived GSM8K scores, embeddings and terminal rewards.

No project training or metric functions are imported. No model/checkpoint is
loaded. The SQLite cache is copied into a temporary directory and opened read
only. All other inputs are read directly from the ZIP. Output is a new JSON
report, never an edited scientific artifact.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import tempfile
import zipfile

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from threadpoolctl import threadpool_limits

PREFIX = "gsm8k_outputs/b200/"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def auc(y, score):
    return float(roc_auc_score(y, score)) if len(set(y)) == 2 else None


def reward_quality(rows, norm):
    """Compare rewards on identical answers; correctness is not a high-gap label."""
    x = lambda key: np.array([r[key] for r in rows])
    proxy = (x("proxy_score") - norm["proxy_mean"]) / norm["proxy_std"]
    pred, correct = x("predicted_gap"), x("correct").astype(bool)
    corrected = proxy - pred
    saturated = x("proxy_score") == 5
    result = {"n": len(rows), "correct": int(correct.sum()),
              "proxy_correctness_auroc": auc(correct, proxy),
              "corrected_correctness_auroc": auc(correct, corrected),
              "proxy_max_score_answers": int(saturated.sum()),
              "incorrect_at_proxy_max_score": int((saturated & ~correct).sum()),
              "corrected_auroc_within_proxy_max_score": auc(correct[saturated], corrected[saturated]),
              "prediction_std": float(pred.std()),
              "prediction_length_correlation": float(np.corrcoef(pred, x("response_tokens"))[0, 1])}
    if all("judge_score" in r for r in rows):
        judge = (x("judge_score") - norm["judge_mean"]) / norm["judge_std"]
        gap = proxy - judge
        result.update(judge_correctness_auroc=auc(correct, judge),
                      gap_mse=float(mean_squared_error(gap, pred)),
                      gap_r2=float(r2_score(gap, pred)), gap_std=float(gap.std()),
                      proxy_judge_mse=float(mean_squared_error(judge, proxy)),
                      corrected_judge_mse=float(mean_squared_error(judge, corrected)))
    pairs = defaultdict(list)
    for i, r in enumerate(rows):
        pairs[r.get("update", 0), r["id"]].append(i)
    counts = Counter()
    for indices in pairs.values():
        if len(indices) != 2 or correct[indices[0]] == correct[indices[1]]:
            continue
        good, bad = sorted(indices, key=lambda i: not correct[i])
        first, second = int(np.sign(proxy[good] - proxy[bad])), int(np.sign(corrected[good] - corrected[bad]))
        counts[f"proxy_{first}_corrected_{second}"] += 1
    result["same_question_correct_incorrect_pairs"] = dict(counts)
    return result


def audit(archive, destination):
    checks, numerical, quality = {}, {}, {}
    with zipfile.ZipFile(archive) as z, tempfile.TemporaryDirectory(prefix="gsm8k_reward_audit_") as tmp:
        read = lambda name: json.loads(z.read(PREFIX + name))
        def rows(name):
            return [json.loads(line) for line in z.read(PREFIX + name).split(b"\n") if line.strip()]
        manifest = read("manifest.json")
        patch = read("source_amendments/ungraded_review_v1/upgrade.json")["patch"]
        project = Path(__file__).resolve().parents[1]
        experiment = project / "code/experiments/gsm8k_experiment"
        checked_sources = ["models.py", "answers.py", "numeric.py", "memory.py", "metrics.py", "ppo.py", "teacher_memory.py", "data.py"]
        for name in checked_sources:
            assert hashlib.sha256((experiment / name).read_bytes()).hexdigest() == patch["after"][name]
        for name, expected in patch["shared_sources"].items():
            assert hashlib.sha256((project / "code/core" / name).read_bytes()).hexdigest() == expected
        checks["unchanged_experiment_sources"] = checked_sources
        checks["unchanged_shared_core_sources"] = list(patch["shared_sources"])
        # Read pinned historical code as text, without executing archive code.
        function_checks = {}
        for name, names in {"run.py": ["reward_for_arm", "train_arm", "annotated", "evaluate", "prepare_memory"],
                            "shared.py": ["pack_items", "flat_config"]}.items():
            old = subprocess.check_output(["git", "-C", str(project.parent), "-c", "safe.directory=" + project.parent.as_posix(),
                "show", f"f4ad4f5ecf059cea0c80729a2ac92ab05fc4dd0c:workshop_project/code/experiments/gsm8k_experiment/{name}"])
            assert hashlib.sha256(old).hexdigest() == patch["after"][name]
            def functions(source):
                return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)}
            before, after = functions(old), functions((experiment / name).read_bytes())
            assert all(before[n] == after[n] for n in names)
            function_checks[name] = names
        checks["historical_training_and_evaluation_functions_unchanged"] = function_checks
        norms = {t: read(f"{p}/normalization.json") for t, p in (("judge", "prepared"), ("judge30b", "prepared_30b"))}
        memories = {}
        for teacher, folder in (("judge", "prepared"), ("judge30b", "prepared_30b")):
            with np.load(io.BytesIO(z.read(PREFIX + folder + "/memory_initial.npz")), allow_pickle=False) as f:
                memories[teacher] = {k: f[k] for k in f.files}
            norm = norms[teacher]
            cal = rows(folder + "/calibration_raw.jsonl")
            for role in ("proxy", "judge"):
                scores = np.array([r[role + "_score"] for r in cal])
                assert scores.mean() == norm[role + "_mean"]
                assert scores.std() == norm[role + "_std"]
            raw = rows(folder + "/memory_raw.jsonl")
            gap = np.array([(r["proxy_score"]-norm["proxy_mean"])/norm["proxy_std"] -
                            (r["judge_score"]-norm["judge_mean"])/norm["judge_std"] for r in raw], np.float32)
            assert np.array_equal(gap, memories[teacher]["gaps"])
            assert [r["id"] for r in raw] == memories[teacher]["group_ids"].tolist()
        checks["normalization_and_both_memory_label_arrays"] = True
        assert all(np.array_equal(memories["judge"][k], memories["judge30b"][k]) for k in ("embeddings", "group_ids"))
        checks["matched_teacher_geometry_and_question_ids"] = True
        cache_path = Path(tmp) / "cache.sqlite"
        with z.open(PREFIX + "reward_cache.sqlite") as src, cache_path.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        db = sqlite3.connect(cache_path.as_uri() + "?mode=ro", uri=True)
        db.execute("PRAGMA query_only=ON")
        cache = {k: (json.loads(r), np.frombuffer(e, np.float32) if e else None)
                 for k, r, e in db.execute("SELECT key,result,embedding FROM scores")}
        db.close()
        identity = memories["judge"]["encoder_identity"].item()
        # Execute only this repository's inspected, hash-verified prompt formatter,
        # avoiding GPU model imports. It reproduces cache identities, not metrics.
        tree = ast.parse((experiment / "models.py").read_bytes())
        formatter = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "judge_messages")
        namespace = {"json": json}
        exec(compile(ast.Module(body=[formatter], type_ignores=[]), "verified_prompt_formatter", "exec"), namespace)
        config = manifest["identity"]["config"]
        protocol = namespace["judge_messages"]({"question": "", "reference": "", "response": ""}, config["scoring"]["mode"])
        grader_ids = {}
        for role in ("proxy", "judge", "judge30b"):
            scoring = dict(config["scoring"])
            if role == "judge30b":
                scoring["batch_size"] = config["teacher30b"]["batch_size"]
            grader_ids[role] = digest({"model": config["models"][role], "revision": manifest["identity"]["resolved"][role],
                "scoring": scoring, "runtime": config["runtime"], "protocol": protocol,
                "encoder": "last_input_token_after_final_norm_l2_v1"})
        assert grader_ids["proxy"] == identity
        assert grader_ids["judge30b"] == read("prepared_30b/complete.json")["teacher_scorer_identity"]
        grade_counts = Counter()
        def check_grade(row, role):
            field = "proxy" if role == "proxy" else "judge"
            result, embedding = cache[digest([grader_ids[role], row["question"], row["reference"], row["response"]])]
            assert result["score"] == row[field + "_score"]
            if field + "_judgement" in row:
                assert result["judge_output"] == row[field + "_judgement"]
            grade_counts[role] += 1
            return embedding
        def proxy_cache(row):
            return check_grade(row, "proxy")
        for teacher, folder in (("judge", "prepared"), ("judge30b", "prepared_30b")):
            for cohort in ("calibration", "memory", "selection"):
                for row in rows(f"{folder}/{cohort}_raw.jsonl"):
                    proxy_cache(row)
                    check_grade(row, teacher)
        raw = rows("prepared/memory_raw.jsonl")
        assert np.array_equal(np.stack([proxy_cache(r) for r in raw]), memories["judge"]["embeddings"])
        checks["memory_embeddings_match_correct_response_cache_keys"] = True
        cohorts = read("data/splits.json")["cohorts"]
        groups = {k: {r["id"] for r in v} for k, v in cohorts.items()}
        assert not any(a & b for i, a in enumerate(groups.values()) for b in list(groups.values())[:i])
        checks["all_question_cohorts_disjoint"] = True
        numerical["nearest_neighbor_replay"] = {}
        def predictions(name, data, teacher):
            mem = memories[teacher]
            ref, gaps = mem["embeddings"], mem["gaps"]
            errors, saved_neighbor_errors, maxima, kth_margins = [], [], [], []
            nearest_sets_changed = 0
            for r in data:
                q = proxy_cache(r)
                similarities = np.clip(q[None, :] @ ref.T, -1, 1)[0]
                similarities[mem["group_ids"] == r["id"]] = -np.inf
                order = np.argsort(-similarities, kind="stable")
                k, temperature = mem["k"].item(), mem["temperature"].item()
                idx = order[:k]
                w = np.exp((similarities[idx] - similarities[idx].max()) / temperature)
                w /= w.sum()
                pred = float(w @ gaps[idx])
                errors.append(abs(pred - r["predicted_gap"]))
                maxima.append(float(w.max()))
                kth_margins.append(float(similarities[order[k-1]] - similarities[order[k]]))
                if "neighbor_indices" in r:
                    saved = np.array(r["neighbor_indices"])
                    assert not any(mem["group_ids"][saved] == r["id"])
                    nearest_sets_changed += set(idx) != set(saved)
                    sw = np.exp((similarities[saved] - similarities[saved].max()) / temperature)
                    sw /= sw.sum()
                    saved_neighbor_errors.append(abs(float(sw @ gaps[saved])-r["predicted_gap"]))
            numerical["nearest_neighbor_replay"][name] = {
                "n": len(data), "max_prediction_error": max(errors),
                "errors_over_1e_5": sum(e > 1e-5 for e in errors),
                "changed_saved_neighbor_sets": nearest_sets_changed,
                "max_error_using_saved_neighbor_indices": max(saved_neighbor_errors, default=None),
                "median_max_neighbor_weight": float(np.median(maxima)),
                "minimum_kth_similarity_margin": min(kth_margins)}
            print(name, numerical["nearest_neighbor_replay"][name], flush=True)
        for teacher, folder in (("judge", "prepared"), ("judge30b", "prepared_30b")):
            data = rows(folder + "/selection_scored.jsonl")
            norm = norms[teacher]
            for row in data:
                assert row["gap"] == ((row["proxy_score"]-norm["proxy_mean"])/norm["proxy_std"] -
                                      (row["judge_score"]-norm["judge_mean"])/norm["judge_std"])
            predictions(folder, data, teacher)
            quality[folder] = reward_quality(data, norms[teacher])
        training = {}
        schedules = []
        for arm in ("proxy", "judge", "knn_static", "knn_static_30b"):
            data, schedule, errors = [], [], []
            norm = norms["judge30b" if arm.endswith("30b") else "judge"]
            files = sorted(n for n in z.namelist() if n.startswith(PREFIX + f"arms/{arm}/rollouts/") and n.endswith(".jsonl"))
            for filename in files:
                chunk = rows(filename.removeprefix(PREFIX))
                update = int(Path(filename).stem.removeprefix("step_"))
                for r in chunk:
                    r["update"] = update
                    assert r["used_for_ppo"] and r["format_penalty"] == r["incomplete_penalty"] == 0
                    assert r["id"] in groups["ppo"]
                    if arm == "judge":
                        check_grade(r, "judge")
                        expected = (r["judge_score"] - norm["judge_mean"]) / norm["judge_std"]
                    else:
                        proxy_cache(r)
                        expected = (r["proxy_score"] - norm["proxy_mean"]) / norm["proxy_std"]
                        if arm.startswith("knn"):
                            expected -= r["predicted_gap"]
                    errors.extend([abs(expected-r["task_reward"]), abs(expected-r["optimization_reward"])])
                    schedule.append(r["id"])
                stats = read(f"arms/{arm}/training/step_{update:06d}.json")
                assert np.isclose(stats["mean_reward"], np.mean([r["optimization_reward"] for r in chunk]), atol=1e-12)
                assert stats["optimizer_steps"] > 0 and stats["graded_responses"] == len(chunk)
                data.extend(chunk)
            schedules.append(schedule)
            training[arm] = {"rows": len(data), "updates": len(files), "maximum_terminal_reward_error": max(errors)}
            assert max(errors) < 1e-12
            if arm.startswith("knn"):
                teacher = "judge30b" if arm.endswith("30b") else "judge"
                predictions("training/" + arm, data, teacher)
                quality["training/" + arm] = reward_quality(data, norm)
        assert all(s == schedules[0] for s in schedules)
        checks["all_training_terminal_rewards_and_update_means"] = training
        checks["matched_training_question_schedule"] = True
        for filename in sorted(n for n in z.namelist() if n.startswith(PREFIX + "evaluations/final/") and n.endswith("/responses.jsonl")):
            data = rows(filename.removeprefix(PREFIX))
            arm = filename.split("/")[-3]
            assert {r["id"] for r in data} == groups["final"]
            assert all(r["diagnostic_teacher"] == r["diagnostic_memory_teacher"] == "judge" for r in data)
            norm = norms["judge"]
            for row in data:
                check_grade(row, "judge")
                assert row["proxy_z"] == (row["proxy_score"]-norm["proxy_mean"])/norm["proxy_std"]
                assert row["judge_z"] == (row["judge_score"]-norm["judge_mean"])/norm["judge_std"]
                assert row["gap"] == row["proxy_z"]-row["judge_z"]
            predictions("final/" + arm, data, "judge")
            quality["final/" + arm] = reward_quality(data, norms["judge"])
        checks["final_cohort_complete_and_common_4b_diagnostic_identity"] = True
        checks["cache_score_and_rationale_matches_by_question_reference_response_and_scorer"] = dict(grade_counts)
        with Path(archive).open("rb") as handle:
            archive_sha = hashlib.file_digest(handle, "sha256").hexdigest()
        report = {"source_archive_sha256": archive_sha,
                  "audit_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "run_fingerprint": manifest["fingerprint"], "checks": checks,
                  "numerical_replay": numerical, "reward_quality": quality,
                  "limits": ["CPU replay from cached embeddings, not a fresh GPU model forward or PPO rerun",
                             "Final 30B-policy diagnostics use the 4B memory; 30B training is replayed with its own memory",
                             "Reward/correctness AUROC is descriptive and not a controlled causal decomposition",
                             "Floating point near-ties can change nearest neighbors across BLAS implementations"]}
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        audit(args.archive, args.destination)
