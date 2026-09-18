"""Memory profiles route all phases consistently without allocating a GPU."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / "python_helper"))
import launch_seeds as launch
from memory_profiles import profile_for


class VramLaunchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.args = argparse.Namespace(runtime=str(PACKAGE / "runtime"), recipe=None,
            output_root=str(self.root / "runs"), state_root=str(self.root / "state"),
            log_root=str(self.root / "logs"), gpus="0", dry_run=True, base_seed=42,
            seeds=None, stage="full", parallelism="auto", vram_gb=24)
        self.settings = json.loads((PACKAGE / "runtime/gsm8k_experiment/settings.json").read_text())
        self.settings["arms"] = launch.ARMS[:]

    def plan(self, devices=None):
        settings = copy.deepcopy(self.settings)
        profile = profile_for(self.args.vram_gb)
        settings["generation"]["batch_size"] = profile["generation_batch_size"]
        settings["scoring"]["batch_size"] = profile["scoring_batch_size"]
        for key in ("ppo_microbatch_size", "rollout_stats_batch_size"):
            settings["runtime"][key] = profile[key]
        return launch.make_plan(self.args, {}, probe=lambda env: devices,
            resolver=lambda *args: {**settings, "seed": args[4]})

    def test_legacy_and_explicit_24_keep_old_identity_and_cli(self):
        explicit = self.plan()
        del self.args.vram_gb
        self.args.recipe = str(PACKAGE / "configs/gsm8k-24gb.yaml")
        legacy = launch.make_plan(self.args, {}, resolver=lambda *args:
            explicit["identity"]["settings_by_seed"]["42"])
        self.assertEqual(explicit["identity"], legacy["identity"])
        self.assertEqual(explicit["workers"][0]["command"], legacy["workers"][0]["command"])
        self.assertEqual(explicit["identity"]["version"], 1)
        self.assertNotIn("memory_profile", explicit["identity"])

    def test_nondefault_profiles_route_every_phase_and_bind_helpers(self):
        for vram in (12, 48):
            for gpus in ("0", "0,1,2"):
                with self.subTest(vram=vram, gpus=gpus):
                    self.args.vram_gb, self.args.gpus = vram, gpus
                    plan = self.plan()
                    self.assertTrue(plan["recipe"].endswith(f"gsm8k-{vram}gb.yaml"))
                    self.assertEqual(plan["identity"]["version"], 3)
                    identity = plan["identity"]["memory_profile"]
                    self.assertEqual(identity["profile"]["vram_gb"], vram)
                    self.assertEqual(set(identity["helper_sha256"]),
                                     {"run_profile.py", "gpu_residency.py", "memory_profiles.py", "windows_io.py"})
                    workers = plan["workers"] + plan.get("preparation_workers", []) + plan.get("finalization_workers", [])
                    for worker in workers:
                        command = worker["command"]
                        self.assertTrue(command[2].endswith("run_profile.py"))
                        self.assertEqual(command[command.index("--vram-gb") + 1], str(vram))
                        self.assertEqual(command[command.index("--recipe") + 1], plan["recipe"])
                    if gpus.count(",") == 2:
                        self.assertEqual([w["arm"] for w in plan["workers"]], launch.ARMS)
                        self.assertEqual(plan["identity"]["seeds"], [42])
                    self.assertFalse(self.root.joinpath("runs").exists())

    def test_capacity_validation_tracks_selected_profile(self):
        self.args.dry_run = False
        for profile in (12, 24, 48):
            self.args.vram_gb = profile
            with self.subTest(profile=profile):
                self.plan([{"bf16": True, "total_memory": profile * 1024**3}])
                with self.assertRaisesRegex(ValueError, f"nominal {profile} GB"):
                    self.plan([{"bf16": True, "total_memory": (profile // 2) * 1024**3}])
                with self.assertRaisesRegex(ValueError, "BF16"):
                    self.plan([{"bf16": False, "total_memory": profile * 1024**3}])

    def test_profile_changes_cannot_resume_existing_run(self):
        self.args.vram_gb = 12
        saved = self.plan()
        state = Path(self.args.state_root)
        state.mkdir()
        launch.atomic_json(state / "launch_manifest.json", launch.check_manifest(saved))
        self.args.stage = "full"
        launch.check_manifest(self.plan())
        self.args.vram_gb = 48
        with self.assertRaisesRegex(ValueError, "manifest differs"):
            launch.check_manifest(self.plan())

    def test_profile_change_on_helper_mutation_is_detected(self):
        self.args.vram_gb = 12
        plan = self.plan()
        state = Path(self.args.state_root)
        state.mkdir()
        old = launch.check_manifest(plan)
        old["identity"]["memory_profile"]["helper_sha256"]["gpu_residency.py"] = "different"
        launch.atomic_json(state / "launch_manifest.json", old)
        with self.assertRaisesRegex(ValueError, "manifest differs"):
            launch.check_manifest(self.plan())


if __name__ == "__main__":
    unittest.main()
