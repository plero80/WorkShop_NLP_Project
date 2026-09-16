"""CPU-only provenance checks. Saved controls are reused only after verification."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import canonical_hash as digest, file_hash as sha, write_json as write
from experiment_cli.cli import yaml_value
from knn_distillation.data import group, groups, schedule, validate_cohorts
from hh_offline.data import vectors


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def resolve(project, config=None, runtime=None, output=None):
    project = Path(project).resolve()
    cfg = yaml_value(Path(config or project / "configs/experiments/hh-ridge-ppo.yaml").read_text(encoding="utf-8"))
    expected = {"version", "experiment", "distillation_study", "refresh_study", "followup_study", "output", "seeds",
                "ridge_alphas", "cpu_threads", "bootstrap_samples", "bootstrap_seed", "allow_downloads", "extra_hf_cache"}
    require(isinstance(cfg, dict) and set(cfg) == expected and cfg["version"] == 1 and cfg["experiment"] == "hh_ridge_ppo", "Invalid HH ridge PPO recipe")
    require(cfg["seeds"] == [42, 43, 44], "This matched study requires all three seeds [42, 43, 44]")
    alphas = cfg["ridge_alphas"]
    require(isinstance(alphas, list) and alphas and len(set(alphas)) == len(alphas) and
            all(type(a) in (int, float) and np.isfinite(a) and a > 0 for a in alphas), "Invalid ridge alpha grid")
    for key in ("cpu_threads", "bootstrap_samples", "bootstrap_seed"):
        require(type(cfg[key]) is int and cfg[key] >= (0 if key == "bootstrap_seed" else 1), "Invalid " + key)
    require(type(cfg["allow_downloads"]) is bool, "allow_downloads must be boolean")
    for key in ("distillation_study", "refresh_study", "followup_study"):
        require(isinstance(cfg[key], str) and Path(cfg[key]).name == cfg[key] and cfg[key].startswith("study_"), "Invalid study name")
    base = Path(runtime).resolve() if runtime else project
    plan = {"project": project, "recipe": cfg,
            "source": base / ("knn_distillation_outputs" if runtime else "results/distillation") / cfg["distillation_study"],
            "refresh": base / ("refresh2_outputs" if runtime else "data/prerequisites/memory_refresh") / cfg["refresh_study"],
            "followup": base / ("outputs" if runtime else "results/followup") / cfg["followup_study"],
            "inputs": base / ("inputs" if runtime else "data/inputs")}
    destination = Path(output) if output else project / cfg["output"]
    plan["output"] = (destination if destination.is_absolute() else project / destination).resolve()
    for key in ("source", "refresh", "followup", "inputs"):
        root = plan[key].resolve()
        require(not plan["output"].is_relative_to(root) and not root.is_relative_to(plan["output"]), "Output must be separate from saved inputs")
    return plan


def checkpoint(path, expected):
    path = Path(path)
    meta = read(path.with_suffix(".json"))
    require(path.is_file(), "Full PPO checkpoint required: " + str(path))
    require(sha(path) == meta["sha256"], "Checkpoint checksum mismatch: " + str(path))
    require(all(meta.get(k) == v for k, v in expected.items()), "Checkpoint metadata does not match the saved study")
    return {**meta, "path": str(path.resolve())}


def load_evaluation(folder):
    folder = Path(folder)
    done = read(folder / "complete.json")
    require(done["identity"] == digest(done["signature"]) and sha(folder / "predictions.csv") == done["csv_sha256"], "Evaluation checksum mismatch")
    frame = pd.read_csv(folder / "predictions.csv", keep_default_na=False, dtype={"prompt_id": str})
    require(len(frame) == done["rows"] and not frame.prompt_id.duplicated().any(), "Invalid evaluation rows")
    return frame, done


def labels(plan, seed, cohort):
    folder = plan["source"] / "labels" / f"seed_{seed}" / cohort
    done, rows = read(folder / "complete.json"), read(folder / "examples.json")
    require(done["rows_hash"] == digest(rows) and len(rows) == done["rows"], "Saved answer checksum mismatch")
    require(len({r["example_id"] for r in rows}) == len(rows), "Duplicate saved answer IDs")
    sig = done["signature"]
    cohort_rows = plan["data"]["distill_" + cohort]
    require(sig["study"] == plan["manifest"]["identity"] and sig["seed"] == seed and
            sig["parent"] == plan["parents"][seed]["sha256"] and sig["memory"] == plan["memory_hashes"][str(seed)] and
            sig["cohort"] == cohort and sig["prompts"] == digest(cohort_rows) and sig["cap"] == 256 and
            sig["batch"] == plan["config"]["generation_batch_size"], "Saved answer provenance mismatch")
    expected = {(r["prompt_id"], origin): r["prompt"] for r in cohort_rows for origin in ("base", "parent")}
    require({(r["prompt_id"], r["origin"]): r["prompt"] for r in rows} == expected and len(rows) == len(expected), "Saved answers do not match the reserved cohort")
    return rows


def audit(plan):
    """Hash checkpoints without deserializing them; verify controls and prompt schedules."""
    source, refresh, followup, inputs = (plan[k] for k in ("source", "refresh", "followup", "inputs"))
    tracked = set()
    def record(path):
        tracked.add(Path(path))
        return read(path)
    m = record(source / "manifest.json")
    require(m["identity"] == digest({k: v for k, v in m.items() if k != "identity"}), "Distillation manifest identity mismatch")
    c, options = m["config"], m["options"]
    require(options["source_kind"] == "refresh2" and options["ppo_updates"] == 100 and options["seeds"] == [42, 43, 44], "Expected the saved 100-update, three-seed M2 continuation")
    require(c["max_new_tokens"] == 256 and c["reward_max_tokens"] == 4096, "Scoring/generation protocol changed")
    # The continuation, router and evaluator below are imported unchanged.
    for name, expected in m["source_sha256"].items():
        p = plan["project"] / "code/experiments/knn_distillation" / name
        require(sha(p) == expected, "Original distillation implementation changed: " + name)
    original_map = read(plan["project"] / "reproducibility/file_map.json")
    for entry in original_map["files"]:
        if entry["organized"].startswith("code/core/") and entry["organized"].endswith(".py"):
            require(sha(plan["project"] / entry["organized"]) == entry["sha256"], "Original PPO/core implementation changed")
    dseal = record(source / "data/complete.json")
    data = {}
    for name, expected in dseal["sha256"].items():
        p = source / "data" / name
        require(sha(p) == expected, "Reserved prompt file changed: " + name)
        data[Path(name).stem] = record(p)
    validate_cohorts(data)
    require(dseal["counts"] == {k: len(v) for k, v in data.items()}, "Prompt counts changed")
    original = inputs / "memory/detectors/gap_knn.npz"
    bank_path = inputs / "candidate_bank.csv"
    base_protocol = record(inputs / "memory/protocol.json")
    require(sha(bank_path) == base_protocol["bank_sha256"], "Candidate bank changed")
    tracked.update([original, bank_path])
    with np.load(original, allow_pickle=False) as z:
        base_x, base_y, bank_ids = vectors(z["vectors"]), z["gaps"], z["bank_ids"]
    base_frame = pd.read_csv(bank_path, keep_default_na=False).iloc[bank_ids]
    np.testing.assert_allclose(base_y, base_frame.proxy_z - base_frame.judge_z, atol=1e-12, rtol=0)
    parents, fingerprints, memory_hashes, counts, control_rows = {}, {}, {}, [], []
    memory_groups = set(base_frame.prompt.map(group))
    preflight = record(source / "preflight.json")
    require(preflight["passed"], "Original preflight did not pass")
    for seed in plan["recipe"]["seeds"]:
        parent_folder = refresh / "runs" / f"seed_{seed}" / "refresh_M2"
        parent_path = parent_folder / "checkpoints/checkpoint_000300.pt"
        parents[seed] = checkpoint(parent_path, m["parents"][str(seed)])
        tracked.update([parent_path, parent_path.with_suffix(".json")])
        parent_done = record(parent_folder / "complete.json")
        memdir = refresh / "memories" / f"seed_{seed}"
        lock = record(memdir / "locked_reward.json")
        memory_hashes[str(seed)] = sha(memdir / "locked_reward.json")
        require(memory_hashes[str(seed)] == m["memories"][str(seed)] == parent_done["dependency"]["memory_lock"], "M2 memory differs from the parent/control memory")
        require(parent_done["checkpoint_sha256"] == parents[seed]["sha256"], "Parent completion hash mismatch")
        require(lock["k"] == 31 and lock["temperature"] == .05 and lock["calibration"] == base_protocol["calibration"], "M2 reward protocol changed")
        npz = memdir / "refreshed_memory.npz"
        require(sha(npz) == lock["memory_sha256"], "M2 checksum mismatch")
        tracked.add(npz)
        with np.load(npz, allow_pickle=False) as z:
            x, y = vectors(z["vectors"]), z["gaps"]
        require(y.shape == (len(x),) and np.isfinite(y).all(), "Invalid M2 labels")
        require(np.array_equal(x[:len(base_x)], base_x) and np.array_equal(y[:len(base_y)], base_y), "M2 does not retain the original memory prefix")
        # M1 additions are the original refresh evaluation, M2 additions the saved CSV.
        first = followup / "evaluations/refresh/refresh_round1/cap_256" / f"round1_knn_signed_s{seed}"
        first_done = record(first / "complete.json")
        require(sha(first / "predictions.csv") == first_done["csv_sha256"], "First-refresh answers changed")
        added_paths = [first / "predictions.csv", memdir / "added_examples.csv"]
        added = [pd.read_csv(p, keep_default_na=False) for p in added_paths]
        tracked.update(added_paths)
        labels_y = np.concatenate([base_y, *[f.actual_proxy_judge_gap.to_numpy(float) for f in added]])
        require(y.shape == labels_y.shape and np.allclose(y, labels_y, atol=1e-12, rtol=0), "M2 labels differ from actual saved proxy-minus-judge gaps")
        for frame in added:
            memory_groups.update(frame.prompt.map(group))
        counts.append({"seed": seed, "memory_answers": len(y), "original_answers": len(base_y), "refresh1_answers": len(added[0]), "refresh2_answers": len(added[1])})
        rows = schedule(data["distill_train"], 100, c["rollout_batch_size"], options["data_seed"] + seed)
        fingerprint = preflight["source_state_fingerprints"][str(seed)]
        fingerprints[seed] = fingerprint
        for branch in ("proxy", "knn"):
            run = source / "runs" / f"seed_{seed}" / branch
            segment, fork = record(run / "segment.json"), record(run / "fork_start.json")
            expected_reward = memory_hashes[str(seed)] if branch == "knn" else digest(["frozen_proxy", lock["calibration"]])
            require(segment == {"identity": m["identity"], "parent_sha256": parents[seed]["sha256"], "parent_identity": parents[seed]["identity"], "seed": seed, "branch": branch,
                                "reward_identity": expected_reward, "training_rows": digest(rows), "start": 300}, "Control does not match the parent/reward/prompt schedule")
            require(fork == {"parent_sha256": parents[seed]["sha256"], "state_fingerprint": fingerprint, "policy_value_optimizer_rng_restored": True}, "Control optimizer/RNG fork differs")
            endpoint = checkpoint(run / "checkpoints/checkpoint_000400.pt", {"identity": m["identity"], "seed": seed, "branch": branch, "update": 400})
            tracked.update([Path(endpoint["path"]), Path(endpoint["path"]).with_suffix(".json")])
            history = record(run / "history.json")
            tail = [h for h in history if h.get("segment_start") == 300 and h.get("reward_source") == branch]
            require([h["update"] for h in tail] == list(range(301, 401)), "Control continuation is incomplete")
            require([pid for h in tail for pid in h["prompt_ids"]] == [r["prompt_id"] for r in rows], "Control training prompt order differs")
            folder = source / "evaluations/final" / f"seed_{seed}" / branch / "update_000400"
            frame, done = load_evaluation(folder)
            sig = done["signature"]
            tracked.update([folder / "complete.json", folder / "predictions.csv"])
            expected_sig = {"experiment": m["identity"], "seed": seed, "branch": branch, "checkpoint": endpoint["sha256"], "cohort": "final", "update": 400,
                            "prompts": digest(data["final"]), "cap": 256, "reward_guard": 4096, "batch": c["generation_batch_size"], "eval_seed": c["eval_seed"], "memory": memory_hashes[str(seed)]}
            require(all(sig.get(k) == v for k, v in expected_sig.items()), "Control final evaluation settings differ")
            require(frame.prompt_id.tolist() == [r["prompt_id"] for r in data["final"]] and frame.prompt.tolist() == [r["prompt"] for r in data["final"]], "Control final contexts differ")
            require(not frame.proxy_truncated.any() and not frame.judge_truncated.any(), "Control rewards were truncated")
            control_rows.append({"seed": seed, "branch": branch, "answers": len(frame), "PPO_updates": len(tail), "PPO_seconds": sum(h["seconds"] for h in tail), "parent_sha256": parents[seed]["sha256"], "endpoint_sha256": endpoint["sha256"]})
    for name, rows in data.items():
        require(not groups(rows) & memory_groups, "M2 prompt overlap with " + name)
    transfer_records = []
    for folder in sorted((refresh / "evaluations/final3").glob("seed_*/*")):
        if not (folder / "complete.json").is_file():
            continue
        frame, done = load_evaluation(folder)
        require(sha(folder / "features.npz") == done["features_sha256"], "Second-refresh features changed")
        require(not set(frame.prompt.map(group)) & memory_groups, "Second-refresh evaluation overlaps M2")
        tracked.update([folder / n for n in ("complete.json", "predictions.csv", "features.npz")])
        transfer_records.append({"folder": folder.relative_to(refresh).as_posix(), "answers": len(frame), "csv_sha256": done["csv_sha256"], "features_sha256": done["features_sha256"]})
    plan.update(manifest=m, config=c, data=data, parents=parents, fingerprints=fingerprints, memory_hashes=memory_hashes, calibration=base_protocol["calibration"])
    for seed in plan["recipe"]["seeds"]:
        for cohort in ("validation", "offline"):
            labels(plan, seed, cohort)
            tracked.update([source / "labels" / f"seed_{seed}" / cohort / n for n in ("complete.json", "examples.json")])
    input_hashes = {str(p.resolve()): sha(p) for p in sorted(tracked)}
    report = {"passed": True, "control_reuse_verified": True, "new_arm": "ridge", "start_update": 300, "end_update": 400,
              "seeds": plan["recipe"]["seeds"], "memory_budgets": counts, "prompt_counts": dseal["counts"], "controls": control_rows,
              "parent_state_fingerprints": fingerprints, "input_sha256": input_hashes,
              "validation_new_judge_answers": sum(len(labels(plan, s, "validation")) for s in plan["recipe"]["seeds"]),
              "second_refresh_predictions_present": len(transfer_records), "second_refresh_recovered": transfer_records,
              "runtime_versions_required": {k: m["runtime_versions"][k] for k in ("torch", "transformers", "peft")}}
    plan["audit"] = report
    return report
