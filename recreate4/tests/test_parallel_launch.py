"""Scheduling and phase boundaries without loading models or allocating GPUs."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / "python_helper"))
spec = importlib.util.spec_from_file_location("parallel_launch_tests", PACKAGE / "python_helper/launch_seeds.py")
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)


class ParallelPlanTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.runtime = self.root / "runtime"
        (self.runtime / "experiment_cli").mkdir(parents=True)
        (self.runtime / "experiment_cli/__main__.py").write_text("# CPU fixture\n")
        self.recipe = self.root / "recipe.yaml"
        self.recipe.write_text("version: 1\n")
        self.args = argparse.Namespace(runtime=str(self.runtime), recipe=str(self.recipe),
            output_root=str(self.root / "runs"), state_root=str(self.root / "state"),
            log_root=str(self.root / "logs"), gpus="0,1,2", dry_run=True,
            base_seed=42, seeds=None, stage="full", parallelism="auto")

    def plan(self, environment=None):
        return launch.make_plan(self.args, environment or {}, resolver=lambda *args: {
            "seed": args[4], "data_seed": 42, "arms": launch.ARMS,
            "ppo": {"full_updates": 400, "pilot_updates": 100}})

    def test_auto_boundaries_and_first_three_tokens(self):
        for count in (1, 2, 3, 4, 6):
            with self.subTest(count=count):
                selected = [f"GPU-{i}" for i in range(count)]
                self.args.gpus = ",".join(selected)
                plan = self.plan({"CUDA_VISIBLE_DEVICES": self.args.gpus})
                if count < 3:
                    self.assertEqual(plan["parallelism"], "seeds")
                    self.assertEqual(plan["identity"]["version"], 1)
                    self.assertEqual(plan["identity"]["seeds"], list(range(42, 42 + count)))
                    self.assertNotIn("orchestration_sha256", plan["identity"])
                else:
                    self.assertEqual(plan["parallelism"], "arms")
                    self.assertEqual(plan["identity"]["seeds"], [42])
                    self.assertEqual([w["arm"] for w in plan["workers"]], launch.ARMS)
                    self.assertEqual([w["gpu_token"] for w in plan["workers"]], selected[:3])
                    self.assertEqual(plan["unused_gpu_tokens"], selected[3:])
                    self.assertEqual(len({w["output"] for w in plan["workers"]}), 3)
                    self.assertEqual(len({w["log"] for w in plan["workers"]}), 3)
                    self.assertEqual(plan["identity"]["settings_by_seed"]["42"]["arms"], launch.ARMS)
        self.assertFalse(Path(self.args.output_root).exists())
        self.assertFalse(Path(self.args.state_root).exists())

    def test_explicit_modes_seed_validation_and_gpu_order(self):
        self.args.parallelism = "seeds"
        self.assertEqual(self.plan()["identity"]["seeds"], [42, 43, 44])
        self.args.parallelism = "arms"
        self.args.seeds = "71"
        self.args.gpus = "5,3,9"
        plan = self.plan()
        self.assertEqual(plan["identity"]["seeds"], [71])
        self.assertEqual([w["gpu_token"] for w in plan["workers"]], ["5", "3", "9"])
        self.args.seeds = "42,43"
        with self.assertRaisesRegex(ValueError, "exactly one seed"):
            self.plan()
        self.args.seeds = None
        self.args.gpus = "0,1"
        with self.assertRaisesRegex(ValueError, "at least three"):
            self.plan()

    def test_unused_gpu_does_not_have_to_meet_training_requirements(self):
        self.args.gpus = "0,1,2,3"
        self.args.dry_run = False
        devices = [{"bf16": True, "total_memory": 24 * 1024**3}] * 3 + [{"bf16": False, "total_memory": 8 * 1024**3}]
        resolver = lambda *args: {"seed": args[4], "data_seed": 42, "arms": launch.ARMS, "ppo": {"full_updates": 400}}
        plan = launch.make_plan(self.args, {}, probe=lambda env: devices, resolver=resolver)
        self.assertEqual(plan["unused_gpu_tokens"], ["3"])
        self.args.parallelism = "seeds"
        with self.assertRaisesRegex(ValueError, "GPU 3"):
            launch.make_plan(self.args, {}, probe=lambda env: devices, resolver=resolver)

    def test_auto_preserves_legacy_three_gpu_identity(self):
        self.args.parallelism = "seeds"
        old = self.plan()
        root = Path(self.args.state_root)
        root.mkdir()
        launch.atomic_json(root / "launch_manifest.json", launch.check_manifest(old))
        self.args.parallelism = "auto"
        resumed = self.plan()
        self.assertEqual(resumed["identity"], old["identity"])
        launch.check_manifest(resumed)
        self.args.parallelism = "arms"
        with self.assertRaisesRegex(ValueError, "manifest differs"):
            launch.check_manifest(self.plan())

    def test_parallel_resume_allows_gpu_reassignment_and_stage_upgrade(self):
        self.args.stage = "prepare"
        prepared = self.plan()
        root = Path(self.args.state_root)
        root.mkdir()
        manifest = launch.check_manifest(prepared)
        self.assertEqual(manifest["initial_gpu_assignment"], {"42/proxy": "0", "42/judge": "1", "42/knn_static": "2"})
        launch.atomic_json(root / "launch_manifest.json", manifest)
        self.args.stage = "full"
        self.args.gpus = "GPU-b,GPU-c,GPU-a,GPU-idle"
        full = self.plan()
        self.assertEqual(full["identity"], prepared["identity"])
        launch.check_manifest(full)

    def test_full_result_validation_targets_canonical_seed(self):
        plan = self.plan()
        root = Path(plan["finalization_workers"][0]["output"])
        root.mkdir(parents=True)
        launch.atomic_json(root / "final_protocol.json", {"arms": launch.ARMS, "updates": 400})
        launch.atomic_json(root / "summary.json", {"seed": 42, "target_updates": 400,
            "arms": launch.ARMS, "metrics": [{"cohort": "final", "arm": a} for a in ["base", *launch.ARMS]],
            "training_completion": {a: {"update": 400} for a in launch.ARMS}})
        launch.verify_full_results(plan)
        self.assertFalse(Path(plan["workers"][0]["output"]).exists())

    def run_phases(self, plan, *, ready=(False, True), failing_phase=None, merge_error=None):
        Path(plan["state_root"]).mkdir(exist_ok=True)
        phases = []
        io = SimpleNamespace(preparation_ready=mock.Mock(side_effect=ready),
                             clone_preparation=mock.Mock(), merge_seed=mock.Mock(side_effect=merge_error))
        def workers(phase_plan, environment):
            phase = phase_plan["phase"]
            phases.append((phase, copy.deepcopy(phase_plan["workers"])))
            launch.atomic_json(Path(plan["state_root"]) / "launch_status.json", {"state": "phase_complete"})
            return 19 if phase == failing_phase else 0
        with mock.patch.dict(sys.modules, {"parallel_state": io}), mock.patch.object(launch, "run_workers", side_effect=workers):
            result = launch.run_parallel_arms(plan, {})
        return result, phases, io

    def test_preparation_then_three_jobs_then_single_finalizer(self):
        plan = self.plan()
        result, phases, io = self.run_phases(plan)
        self.assertEqual(result, 0)
        self.assertEqual([p for p, _ in phases], ["preparation", "ppo_arms", "evaluation"])
        self.assertEqual([len(w) for _, w in phases], [1, 3, 1])
        self.assertEqual(io.clone_preparation.call_count, 3)
        io.merge_seed.assert_called_once()
        final = phases[-1][1][0]
        self.assertEqual(final["output"], str(Path(plan["output_root"]) / "seed_42"))
        if os.name == "nt":
            self.assertTrue(final["command"][2].endswith("run_profile.py"))
        else:
            self.assertIn("experiment_cli", final["command"])
        self.assertEqual(final["command"][final["command"].index("--stage") + 1], "full")

    def test_prepare_stage_cannot_start_training_and_reuses_completed_preparation(self):
        self.args.stage = "prepare"
        result, phases, io = self.run_phases(self.plan(), ready=(True, True))
        self.assertEqual(result, 0)
        self.assertEqual(phases, [])
        io.clone_preparation.assert_not_called()
        io.merge_seed.assert_not_called()

    def test_failed_arm_never_merges_or_opens_final_test(self):
        result, phases, io = self.run_phases(self.plan(), failing_phase="ppo_arms")
        self.assertEqual(result, 19)
        self.assertEqual([p for p, _ in phases], ["preparation", "ppo_arms"])
        io.merge_seed.assert_not_called()

    def test_failed_merge_never_finalizes(self):
        plan = self.plan()
        with self.assertRaisesRegex(ValueError, "checkpoint mismatch"):
            self.run_phases(plan, merge_error=ValueError("checkpoint mismatch"))
        status = json.loads((Path(plan["state_root"]) / "launch_status.json").read_text())
        self.assertEqual(status["phase"], "merge_checkpoints")
        self.assertEqual(status["state"], "failed")

    def test_final_evaluation_resume_does_not_rerun_completed_arm_workers(self):
        plan = self.plan()
        for worker in plan["workers"]:
            root = Path(worker["output"])
            (root / "arms" / worker["arm"]).mkdir(parents=True)
            launch.atomic_json(root / "status.json", {"stage": "complete", "run_stage": "pilot",
                "arms": [worker["arm"]], "updates": 400, "skipped_arms": {}})
            launch.atomic_json(root / "arms" / worker["arm"] / "completed.json", {"update": 400})
        result, phases, io = self.run_phases(plan, ready=(True, True))
        self.assertEqual(result, 0)
        self.assertEqual([p for p, _ in phases], ["evaluation"])
        io.merge_seed.assert_called_once()  # Still independently validates every checkpoint.
        worker = plan["workers"][1]
        launch.atomic_json(Path(worker["output"]) / "arms" / worker["arm"] / "completed.json", {"update": 100})
        result, phases, io = self.run_phases(plan, ready=(True, True))
        self.assertEqual([(phase, len(jobs)) for phase, jobs in phases], [("ppo_arms", 1), ("evaluation", 1)])
        self.assertEqual(phases[0][1][0]["arm"], "judge")

    def test_three_real_cpu_workers_start_concurrently_with_distinct_gpu_masks(self):
        import os
        plan = self.plan()
        Path(plan["state_root"]).mkdir()
        marker_root = self.root / "barrier"
        marker_root.mkdir()
        code = ("import json,os,sys,time;from pathlib import Path;root=Path(sys.argv[1]);"
                "(root/(sys.argv[2]+'.started')).write_text(json.dumps({'gpu':os.environ['CUDA_VISIBLE_DEVICES'],'seed':int(sys.argv[3])}));"
                "deadline=time.monotonic()+8\n"
                "while len(list(root.glob('*.started')))<3 and time.monotonic()<deadline: time.sleep(.02)\n"
                "assert len(list(root.glob('*.started')))==3, 'Workers were serialized'\n")
        for worker in plan["workers"]:
            worker["command"] = [sys.executable, "-u", "-c", code, str(marker_root), worker["arm"], str(worker["seed"])]
        self.assertEqual(launch.run_workers({**plan, "phase": "ppo_arms"}, dict(os.environ), poll_seconds=.01), 0)
        for index, arm in enumerate(launch.ARMS):
            self.assertEqual(json.loads((marker_root / f"{arm}.started").read_text()), {"gpu": str(index), "seed": 42})
        status = json.loads((Path(plan["state_root"]) / "launch_status.json").read_text())
        self.assertEqual([worker["arm"] for worker in status["workers"]], launch.ARMS)
        self.assertEqual(status["state"], "phase_complete")


class OriginalCliBridgeTests(unittest.TestCase):
    def test_real_cli_resolves_full_config_and_keeps_final_data_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = root / "runtime"
            shutil.copytree(PACKAGE / "runtime", runtime, ignore=shutil.ignore_patterns("__pycache__", ".experiment_cli"))
            recipe = root / "recipe.yaml"
            shutil.copy2(PACKAGE / "configs/gsm8k-24gb.yaml", recipe)
            code = "import json,sys;sys.path.insert(0,sys.argv[1]);from run_arm import command_for;print(json.dumps(command_for(*sys.argv[2:5],int(sys.argv[5]),sys.argv[6],sys.argv[7])))"
            for stage, target in (("pilot", 100), ("full", 400)):
                output = root / stage
                result = subprocess.run([sys.executable, "-B", "-c", code, str(PACKAGE / "python_helper"),
                    str(runtime), str(recipe), str(output), "71", stage, "judge"], capture_output=True, text=True, check=True)
                command = json.loads(result.stdout)
                self.assertIn("gsm8k_experiment.run", command)
                self.assertEqual(command[command.index("--stage") + 1], "pilot")
                self.assertEqual(command[command.index("--updates") + 1], str(target))
                self.assertEqual(command[command.index("--arms") + 1], "judge")
                config = json.loads(Path(command[command.index("--config") + 1]).read_text())
                self.assertEqual(config["arms"], launch.ARMS)
                self.assertEqual(config["seed"], 71)
                self.assertEqual(config["data_seed"], 42)
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
