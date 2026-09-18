"""CPU fixtures for isolated arm clones, verified budgets and atomic seed merges."""
from __future__ import annotations

import copy
from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python_helper"))
import parallel_state as state


class ParallelStateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="parallel state ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.runtime = self.root / "runtime"
        self.runtime.mkdir()
        self.prepared = self.root / "seed_42_prepare"
        self.prepared.mkdir()
        self.destination = self.root / "seed_42"
        self.jobs = {arm: self.root / f"seed_42_{arm}" for arm in state.ARMS}
        names = ("calibration", "memory", "selection", "monitor", "refresh", "ppo", "final")
        self.settings = {"arms": list(state.ARMS), "seed": 42, "data_seed": 42,
                         "revisions": dict(zip(("policy", "proxy", "judge", "dataset"), [x * 40 for x in "abcd"])),
                         "dataset": {name: 2 for name in names}, "scoring": {"minimum_std": .05},
                         "knn": {"k_grid": [1]}, "ppo": {"pilot_updates": 1, "full_updates": 2}}
        cohorts = {name: [{"id": f"{name}-{i}", "question": "Q", "reference": "#### 1"}
                          for i in range(2)] for name in names}
        self.split = {"cohorts": cohorts, "audit": {"seed": 42, "sizes": {name: 2 for name in names}}}
        self.split["fingerprint"] = state._digest(self.split)
        self.split["hf_revision"] = self.settings["revisions"]["dataset"]
        identity = {"config": self.settings, "resolved": self.settings["revisions"],
                    "split_fingerprint": self.split["fingerprint"], "source": "fixture"}
        self.fingerprint = state._digest(identity)
        for name in state.REQUIRED:
            path = self.prepared / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if name.endswith(".json"):
                self.write(path, {})
            elif name.endswith(".jsonl"):
                path.write_text('{"fixture":true}\n')
        self.write(self.prepared / "config.json", self.settings)
        self.write(self.prepared / "manifest.json", {"identity": identity, "fingerprint": self.fingerprint})
        self.write(self.prepared / "resolved_assets.json", self.settings["revisions"])
        self.write(self.prepared / "data/splits.json", self.split)
        self.write(self.prepared / "status.json", {"stage": "prepared"})
        self.write(self.prepared / "preflight_ppo.json", {"passed": True, "updates_discarded": True})
        self.write(self.prepared / "prepared/normalization.json",
                   dict(proxy_mean=3, proxy_std=1, judge_mean=3, judge_std=1, threshold=1))
        self.write(self.prepared / "prepared/complete.json", {"encoder_identity": "fixture", "n_memory": 2})
        np.savez_compressed(self.prepared / "prepared/memory_initial.npz", embeddings=np.eye(2),
                            gaps=np.zeros(2), group_ids=np.array(["memory-0", "memory-1"]),
                            k=1, temperature=.05, encoder_identity="fixture")
        torch.save({"adapter": {"weight": torch.zeros(1)}}, self.prepared / "initial_trainable.pt")
        self.evaluation(self.prepared, "base", 0)
        (self.prepared / "judge_calls.jsonl").write_text('{"event":"prepare"}\n')
        (self.prepared / "run.lock").write_bytes(b" ")
        self.cache = sqlite3.connect(self.prepared / "reward_cache.sqlite")
        self.addCleanup(self.cache.close)
        self.cache.execute("PRAGMA journal_mode=WAL")
        self.cache.execute("CREATE TABLE scores (key TEXT PRIMARY KEY, result TEXT, embedding BLOB)")
        self.cache.execute("INSERT INTO scores VALUES ('prepared','{}',?)", (b"embedding",))
        self.cache.commit()

    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")

    def evaluation(self, root, arm, update):
        path = root / "evaluations/monitor" / arm / f"step_{update:06d}"
        self.write(path / "metrics.json", {"arm": arm, "update": update, "cohort": "monitor", "n": 2})
        (path / "responses.jsonl").write_text("".join(json.dumps(row) + "\n" for row in self.split["cohorts"]["monitor"]))
        self.write(root / "generations/monitor" / arm / f"{update:06d}" / "000000.json", {"fixture": arm})

    def checkpoint(self, arm, step, target=None):
        folder = self.jobs[arm] / "arms" / arm
        folder.mkdir(parents=True, exist_ok=True)
        checkpoint = folder / "checkpoint.pt"
        torch.save({"engine": state.ENGINE, "step": step, "arm": arm,
                    "fingerprint": self.fingerprint, "extra": {"successful_updates": step},
                    "trainable": {"tensor": torch.zeros(1)}}, checkpoint)
        self.write(checkpoint.with_suffix(".sha256.json"), {"engine": state.ENGINE, "sha256": state._sha(checkpoint)})

    def finish_jobs(self, target):
        for arm, job in self.jobs.items():
            state.clone_preparation(self.prepared, job, arm, self.settings)
            self.checkpoint(arm, target)
            self.write(job / "arms" / arm / "completed.json",
                       {"update": target, "fingerprint": self.fingerprint, "successful_updates": target,
                        "skipped_updates": 0, "last_refresh": 0})
            self.write(job / "status.json", {"stage": "complete", "run_stage": "pilot",
                                            "updates": target, "arms": [arm], "skipped_arms": {}})
            self.write(job / "skipped_arms.json", {})
            for index in range(1, target + 1):
                folder = job / "arms" / arm
                self.write(folder / "training" / f"step_{index:06d}.json", {"update": index, "arm": arm})
                rollouts = folder / "rollouts" / f"step_{index:06d}.jsonl"
                rollouts.parent.mkdir(parents=True, exist_ok=True)
                rollouts.write_text('{"fixture":"rollout"}\n')
            self.evaluation(job, arm, target)
            (job / "judge_calls.jsonl").write_bytes((self.prepared / "judge_calls.jsonl").read_bytes() +
                                                    json.dumps({"event": arm, "target": target}).encode() + b"\n")

    def merge(self, stage="full"):
        return state.merge_seed(self.runtime, self.prepared, self.jobs, self.destination, self.settings, stage)

    def test_readiness_requires_complete_preparation_and_matching_identity(self):
        self.assertTrue(state.preparation_ready(self.prepared, self.settings))
        (self.prepared / "status.json").unlink()
        self.assertFalse(state.preparation_ready(self.prepared, self.settings))
        altered = copy.deepcopy(self.settings)
        altered["seed"] = 99
        with self.assertRaisesRegex(ValueError, "configuration differs"):
            state.preparation_ready(self.prepared, altered)

    def test_clone_uses_committed_wal_and_cache_is_independent_on_resume(self):
        self.cache.execute("INSERT INTO scores VALUES ('wal-only','{}',NULL)")
        self.cache.commit()
        path = state.clone_preparation(self.prepared, self.jobs["proxy"], "proxy", self.settings)
        with closing(sqlite3.connect(path / "reward_cache.sqlite")) as clone:
            self.assertEqual(clone.execute("SELECT count(*) FROM scores").fetchone()[0], 2)
            clone.execute("INSERT INTO scores VALUES ('arm-new','{}',NULL)")
            clone.commit()
        self.assertEqual(self.cache.execute("SELECT count(*) FROM scores").fetchone()[0], 2)
        before = (path / "initial_trainable.pt").read_bytes()
        self.assertEqual(state.clone_preparation(self.prepared, path, "proxy", self.settings), path)
        self.assertEqual((path / "initial_trainable.pt").read_bytes(), before)
        self.assertFalse((path / "reward_cache.sqlite-wal").exists())

    def test_active_writer_untracked_output_and_wrong_arm_are_rejected(self):
        with state._paused(self.prepared), self.assertRaisesRegex(ValueError, "active writer"):
            state.clone_preparation(self.prepared, self.jobs["proxy"], "proxy", self.settings)
        self.jobs["proxy"].mkdir()
        with self.assertRaisesRegex(ValueError, "Untracked"):
            state.clone_preparation(self.prepared, self.jobs["proxy"], "proxy", self.settings)
        self.jobs["proxy"].rmdir()
        state.clone_preparation(self.prepared, self.jobs["proxy"], "proxy", self.settings)
        with self.assertRaisesRegex(ValueError, "origin"):
            state.clone_preparation(self.prepared, self.jobs["proxy"], "judge", self.settings)

    def test_clone_failure_never_publishes_partial_destination(self):
        original = state._copy_file
        def fail(source, destination, **kwargs):
            if source.name == "initial_trainable.pt":
                raise OSError("Synthetic copy interruption")
            return original(source, destination, **kwargs)
        with mock.patch.object(state, "_copy_file", side_effect=fail), self.assertRaises(OSError):
            state.clone_preparation(self.prepared, self.jobs["proxy"], "proxy", self.settings)
        self.assertFalse(self.jobs["proxy"].exists())
        self.assertFalse(list(self.root.glob(".seed_42_proxy.clone-*")))
        self.assertTrue(state.preparation_ready(self.prepared, self.settings))

    def test_merge_keeps_all_checkpoints_and_counts_preparation_log_once(self):
        self.finish_jobs(2)
        self.assertEqual(self.merge(), self.destination)
        self.assertEqual(state._rows(self.destination / "judge_calls.jsonl"),
                         [{"event": "prepare"}, *[{"event": a, "target": 2} for a in state.ARMS]])
        for arm, job in self.jobs.items():
            relative = Path("arms") / arm / "checkpoint.pt"
            self.assertEqual(state._sha(self.destination / relative), state._sha(job / relative))
        self.assertFalse((self.destination / "final_protocol.json").exists())
        self.assertFalse((self.destination / "summary.json").exists())
        self.assertEqual(state._json(self.destination / "status.json")["stage"], "parallel_merged")

    def test_same_merge_preserves_original_finalizer_results(self):
        self.finish_jobs(2)
        self.merge()
        self.write(self.destination / "final_protocol.json", {"updates": 2, "arms": list(state.ARMS),
                                                              "fingerprint": self.fingerprint})
        self.write(self.destination / "summary.json", {"original_finalizer": True})
        self.merge()
        self.assertEqual(state._json(self.destination / "summary.json"), {"original_finalizer": True})
        self.write(self.destination / "final_protocol.json", {"updates": 1, "arms": list(state.ARMS),
                                                              "fingerprint": self.fingerprint})
        with self.assertRaisesRegex(ValueError, "final protocol"):
            self.merge()

    def test_pilot_to_full_retains_old_output_and_never_moves_arm_jobs(self):
        self.finish_jobs(1)
        self.merge("pilot")
        (self.destination / "run.lock").write_bytes(b" ")
        previous = state._sha(self.destination / "arms/proxy/checkpoint.pt")
        self.finish_jobs(2)
        self.merge("full")
        retained = list(self.root.glob(".seed_42.retained-*"))
        self.assertEqual(len(retained), 1)
        self.assertEqual(state._sha(retained[0] / "arms/proxy/checkpoint.pt"), previous)
        self.assertTrue(all(path.exists() for path in self.jobs.values()))
        self.assertEqual(state._json(self.destination / "parallel_merge.json")["identity"]["target_updates"], 2)

    def test_actual_checkpoint_metadata_must_match_claimed_budget(self):
        self.finish_jobs(1)
        self.merge("pilot")
        before = state._sha(self.destination / "parallel_merge.json")
        self.finish_jobs(2)
        self.checkpoint("judge", 1)  # Correct checksum, stale actual checkpoint.
        with self.assertRaisesRegex(ValueError, "Checkpoint identity/budget"):
            self.merge()
        self.assertEqual(state._sha(self.destination / "parallel_merge.json"), before)

    def test_child_final_evaluation_and_modified_prepared_input_are_rejected(self):
        self.finish_jobs(2)
        self.write(self.jobs["judge"] / "final_protocol.json", {"updates": 2})
        with self.assertRaisesRegex(ValueError, "final evaluation unopened"):
            self.merge()
        (self.jobs["judge"] / "final_protocol.json").unlink()
        (self.jobs["judge"] / "initial_trainable.pt").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Prepared input changed"):
            self.merge()
        self.assertFalse(self.destination.exists())

    def test_worker_log_without_original_prefix_is_rejected(self):
        self.finish_jobs(2)
        (self.jobs["knn_static"] / "judge_calls.jsonl").write_text('{"wrong":"prefix"}\n')
        with self.assertRaisesRegex(ValueError, "exact preparation prefix"):
            self.merge()
        self.assertFalse(self.destination.exists())

    def test_publish_failure_restores_existing_canonical(self):
        self.finish_jobs(1)
        self.merge("pilot")
        before = state._sha(self.destination / "parallel_merge.json")
        self.finish_jobs(2)
        original = Path.rename
        def fail_staging(path, target):
            if path.name.startswith(".seed_42.merge-"):
                raise OSError("Synthetic publish interruption")
            return original(path, target)
        with mock.patch.object(Path, "rename", fail_staging), self.assertRaisesRegex(OSError, "Synthetic publish"):
            self.merge()
        self.assertEqual(state._sha(self.destination / "parallel_merge.json"), before)
        self.assertFalse(list(self.root.glob(".seed_42.merge-*")))
        self.assertTrue(all(path.exists() for path in self.jobs.values()))

    def test_linked_preparation_entry_is_rejected_before_copy(self):
        linked = self.prepared / "linked.txt"
        linked.write_text("fixture")
        original = state._linked
        with mock.patch.object(state, "_linked", side_effect=lambda p: p == linked or original(p)):
            with self.assertRaisesRegex(ValueError, "Linked output entry"):
                state.clone_preparation(self.prepared, self.jobs["proxy"], "proxy", self.settings)
        self.assertFalse(self.jobs["proxy"].exists())


if __name__ == "__main__":
    unittest.main()
