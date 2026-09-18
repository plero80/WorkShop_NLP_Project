"""CPU-only launcher tests; no model, dataset, or CUDA imports are required."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from unittest import mock


SOURCE = Path(__file__).resolve().parents[1] / "python_helper" / "launch_seeds.py"
sys.path.insert(0, str(SOURCE.parent))
SPEC = importlib.util.spec_from_file_location("launch_seeds", SOURCE)
launch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launch)


def gpu(**changes):
    return {"name": "test device", "total_memory": 24 * 1024**3, "bf16": True, **changes}


class VisibilityTests(unittest.TestCase):
    def test_mask_tokens_preserve_order_and_worker_native_ids(self):
        probe = mock.Mock(return_value=[gpu(), gpu()])
        selected, _ = launch.discover_gpus({"CUDA_VISIBLE_DEVICES": "3,1"}, probe=probe)
        self.assertEqual(selected, ["3", "1"])
        self.assertEqual(probe.call_args.args[0]["CUDA_VISIBLE_DEVICES"], "3,1")

    def test_uuid_subset_and_slurm_fallback(self):
        mask = "GPU-abc,GPU-def"
        self.assertEqual(launch.discover_gpus({"CUDA_VISIBLE_DEVICES": mask}, "GPU-def", True)[0], ["GPU-def"])
        self.assertEqual(launch.discover_gpus({"SLURM_STEP_GPUS": "2,4", "SLURM_JOB_GPUS": "0,1,2,4"}, dry_run=True)[0], ["2", "4"])
        self.assertEqual(launch.discover_gpus({"CUDA_VISIBLE_DEVICES": "0", "SLURM_JOB_GPUS": "7"}, dry_run=True)[0], ["0"])

    def test_allocations_cannot_be_expanded_or_hidden_mask_overridden(self):
        for environment, requested in [({"CUDA_VISIBLE_DEVICES": "1"}, "0"),
                                       ({"SLURM_JOB_GPUS": "1"}, "2"),
                                       ({"CUDA_VISIBLE_DEVICES": ""}, "0"),
                                       ({"CUDA_VISIBLE_DEVICES": "-1"}, "0")]:
            with self.subTest(environment=environment), self.assertRaises(ValueError):
                launch.discover_gpus(environment, requested, True)

    def test_dry_run_never_probes_and_requires_declared_devices(self):
        probe = mock.Mock(side_effect=AssertionError("must not probe"))
        self.assertEqual(launch.discover_gpus({}, "GPU-abc,1", True, probe), (["GPU-abc", "1"], None))
        with self.assertRaises(ValueError):
            launch.discover_gpus({}, dry_run=True, probe=probe)
        probe.assert_not_called()

    def test_unmasked_device_discovery_and_hardware_checks(self):
        self.assertEqual(launch.discover_gpus({}, probe=lambda env: [gpu(), gpu()])[0], ["0", "1"])
        for devices in [[], [gpu(bf16=False)], [gpu(total_memory=21 * 1024**3)], [gpu(), gpu()]]:
            with self.subTest(devices=devices), self.assertRaises(ValueError):
                launch.discover_gpus({"CUDA_VISIBLE_DEVICES": "0"}, probe=lambda env: devices)

    def test_invalid_tokens_and_seeds(self):
        for bad in ["", "0,0", "0,", "all", "-2"]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                launch.discover_gpus({}, bad, True)
        self.assertEqual(launch.select_seeds(3), [42, 43, 44])
        self.assertEqual(launch.select_seeds(2, explicit="9,7"), [9, 7])
        for count, explicit in [(2, "42"), (2, "1,1"), (1, "-1"), (1, str(2**32)), (1, "x")]:
            with self.subTest(explicit=explicit), self.assertRaises(ValueError):
                launch.select_seeds(count, explicit=explicit)


class TemporaryProject(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.runtime = self.root / "runtime"
        package = self.runtime / "experiment_cli"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("", encoding="utf-8")
        (package / "__main__.py").write_text("# fixture\n", encoding="utf-8")
        self.recipe = self.root / "recipe.yaml"
        self.recipe.write_text("version: 1\n", encoding="utf-8")
        self.args = argparse.Namespace(runtime=str(self.runtime), recipe=str(self.recipe),
            output_root=str(self.root / "runs"), log_root=str(self.root / "logs"),
            state_root=str(self.root / "state"), gpus="3,1", dry_run=True,
            seeds=None, base_seed=42, stage="pilot")

    @staticmethod
    def resolver(runtime, recipe, output, stage, seed, environment):
        return {"seed": seed, "data_seed": 42, "arms": launch.ARMS, "ppo": {"full_updates": 8}}

    def plan(self):
        return launch.make_plan(self.args, {}, resolver=self.resolver)


class PlanAndResumeTests(TemporaryProject):
    def test_plan_is_read_only_and_uses_platform_worker(self):
        plan = self.plan()
        self.assertFalse(Path(self.args.state_root).exists())
        self.assertFalse(Path(self.args.output_root).exists())
        self.assertFalse(Path(self.args.log_root).exists())
        self.assertEqual(plan["identity"]["seeds"], [42, 43])
        self.assertEqual(plan["gpu_validation"], "skipped (dry run)")
        if os.name == "nt":
            self.assertEqual(plan["workers"][0]["command"], [sys.executable, "-u",
                str(SOURCE.with_name("run_profile.py")), "--runtime", str(self.runtime),
                "--recipe", str(self.recipe), "--output", str(self.root / "runs" / "seed_42"),
                "--seed", "42", "--stage", "pilot", "--vram-gb", "24"])
            self.assertEqual(plan["io_policy"]["name"], "windows_atomic_replace_retry_v1")
        else:
            self.assertEqual(plan["workers"][0]["command"], [sys.executable, "-u", "-m", "experiment_cli", "run",
                str(self.recipe), "--output", str(self.root / "runs" / "seed_42"), "--stage", "pilot",
                "--set", "seed=42", "--set", "data_seed=42"])

    def test_check_probes_hardware_without_writing_or_launching(self):
        probe = mock.Mock(return_value=[gpu(), gpu()])
        arguments = ["--runtime", str(self.runtime), "--recipe", str(self.recipe),
            "--output-root", self.args.output_root, "--log-root", self.args.log_root,
            "--state-root", self.args.state_root, "--gpus", "3,1", "--check"]
        with mock.patch.object(launch, "probe_gpus", probe), mock.patch.object(launch, "resolve_settings", self.resolver), \
             mock.patch.dict(os.environ, {}, clear=True), redirect_stdout(io.StringIO()) as output:
            self.assertEqual(launch.main(arguments), 0)
        probe.assert_called_once()
        self.assertEqual(json.loads(output.getvalue())["gpu_validation"], "passed")
        for path in (self.args.state_root, self.args.output_root, self.args.log_root):
            self.assertFalse(Path(path).exists())

    def test_resume_allows_gpu_reassignment_and_stage_upgrade(self):
        plan = self.plan()
        state = Path(plan["state_root"])
        state.mkdir()
        launch.atomic_json(state / "launch_manifest.json", launch.check_manifest(plan))
        self.args.gpus = "GPU-new1,GPU-new2"
        self.args.stage = "full"
        resumed = self.plan()
        updated = launch.check_manifest(resumed)
        self.assertEqual(updated["highest_stage"], "full")
        self.assertEqual(updated["initial_gpu_assignment"], {"42": "3", "43": "1"})
        launch.atomic_json(state / "launch_manifest.json", updated)
        with self.assertRaisesRegex(ValueError, "backwards"):
            launch.check_manifest(plan)

    def test_changed_seed_recipe_runtime_settings_or_count_rejected(self):
        plan = self.plan()
        state = Path(plan["state_root"])
        state.mkdir()
        launch.atomic_json(state / "launch_manifest.json", launch.check_manifest(plan))
        for key, value in [("seeds", [42, 44]), ("recipe_sha256", "changed"),
                           ("runtime_sha256", "changed"), ("workers", 3), ("settings_by_seed", {})]:
            changed = copy.deepcopy(plan)
            changed["identity"][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "manifest differs"):
                launch.check_manifest(changed)

    def test_untracked_existing_outputs_rejected_without_mutation(self):
        plan = self.plan()
        output = Path(plan["output_root"])
        output.mkdir()
        (output / "seed_42").mkdir()
        with self.assertRaisesRegex(ValueError, "untracked"):
            launch.check_manifest(plan)
        self.assertFalse(Path(plan["state_root"]).exists())

    def test_first_launch_accepts_only_known_orchestrator_locks(self):
        plan = self.plan()
        state = Path(plan["state_root"])
        state.mkdir()
        for name in ("launcher.lock", ".pipeline.lock"):
            (state / name).touch()
        self.assertEqual(launch.check_manifest(plan)["highest_stage"], "pilot")
        (state / "seed_42").mkdir()
        with self.assertRaisesRegex(ValueError, "untracked"):
            launch.check_manifest(plan)

    def test_exclusive_lock_rejects_second_owner_and_releases(self):
        state = self.root / "state"
        with launch.exclusive_lock(state):
            with self.assertRaisesRegex(ValueError, "Another launcher"):
                with launch.exclusive_lock(state):
                    self.fail("second owner acquired lock")
        with launch.exclusive_lock(state):
            pass

    def test_original_cli_show_rejects_nested_suite_and_wrong_arms(self):
        settings = self.resolver(None, None, None, None, 42, None)
        for alterations in [{"seeds": [42, 43]}, {"settings": {**settings, "arms": ["proxy"]}},
                            {"settings": {**settings, "data_seed": 43}}]:
            document = {"settings": settings, "seeds": None, **alterations}
            response = subprocess.CompletedProcess([], 0, json.dumps(document), "")
            with self.subTest(alterations=alterations), mock.patch.object(launch.subprocess, "run", return_value=response), self.assertRaises(ValueError):
                launch.resolve_settings(self.runtime, self.recipe, self.root / "out", "full", 42, {})

    def test_real_subprocess_dry_run_needs_no_torch_and_writes_nothing(self):
        fixture = ("import json,sys\n"
                   "seed = int(next(a.split('=',1)[1] for a in sys.argv if a.startswith('seed=')))\n"
                   "print(json.dumps({'seeds':None,'settings':{'seed':seed,'data_seed':42,"
                   "'arms':['proxy','judge','knn_static'],'ppo':{'full_updates':8}}}))\n")
        (self.runtime / "experiment_cli" / "__main__.py").write_text(fixture, encoding="utf-8")
        command = [sys.executable, "-B", str(SOURCE), "--runtime", str(self.runtime),
                   "--recipe", str(self.recipe), "--output-root", self.args.output_root,
                   "--log-root", self.args.log_root, "--state-root", self.args.state_root,
                   "--gpus", "0,1", "--dry-run"]
        env = {k: v for k, v in os.environ.items() if k not in ("CUDA_VISIBLE_DEVICES", "SLURM_JOB_GPUS", "SLURM_STEP_GPUS")}
        result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["identity"]["seeds"], [42, 43])
        for path in (self.args.state_root, self.args.output_root, self.args.log_root):
            self.assertFalse(Path(path).exists())
        self.assertFalse(list(self.runtime.rglob("__pycache__")))


class WorkerTests(TemporaryProject):
    def worker_plan(self, programs):
        plan = self.plan()
        Path(plan["state_root"]).mkdir()
        for worker, program in zip(plan["workers"], programs):
            worker["command"] = [sys.executable, "-u", "-c", program]
        return plan

    def run_quietly(self, plan, **kwargs):
        with redirect_stdout(io.StringIO()):
            return launch.run_workers(plan, dict(os.environ), poll_seconds=.02, grace=.2, **kwargs)

    def test_complete_workers_have_isolated_environment_and_tee_logs(self):
        program = "import os; print('visible=' + os.environ['CUDA_VISIBLE_DEVICES']); print('cwd=' + os.getcwd())"
        plan = self.worker_plan([program, program])
        self.assertEqual(self.run_quietly(plan), 0)
        status = json.loads((Path(plan["state_root"]) / "launch_status.json").read_text())
        self.assertEqual(status["state"], "complete")
        for worker in plan["workers"]:
            log = Path(worker["log"]).read_text()
            self.assertIn("visible=" + worker["gpu_token"], log)
            self.assertIn("cwd=" + str(self.runtime), log)
        self.assertTrue(all(w["returncode"] == 0 for w in status["workers"]))

    def test_one_failure_stops_other_worker_and_reports_failure(self):
        plan = self.worker_plan(["import time; print('waiting'); time.sleep(30)",
                                 "import time; time.sleep(.2); raise SystemExit(7)"])
        self.assertEqual(self.run_quietly(plan), 7)
        status = json.loads((Path(plan["state_root"]) / "launch_status.json").read_text())
        self.assertEqual(status["state"], "failed")
        self.assertTrue(all(w["returncode"] is not None for w in status["workers"]))
        self.assertNotEqual(status["workers"][0]["returncode"], 0)

    def test_cancellation_cleans_up_children(self):
        plan = self.worker_plan(["import time; time.sleep(30)"] * 2)
        event = threading.Event()
        timer = threading.Timer(.2, event.set)
        timer.start()
        self.addCleanup(timer.cancel)
        self.assertEqual(self.run_quietly(plan, stop_event=event), 128 + signal.SIGTERM)
        status = json.loads((Path(plan["state_root"]) / "launch_status.json").read_text())
        self.assertEqual(status["state"], "interrupted")
        self.assertTrue(all(w["returncode"] is not None for w in status["workers"]))

    @unittest.skipUnless(os.name == "posix", "Linux process-group behavior")
    def test_groups_are_signalled_even_when_cli_leader_already_exited(self):
        process = mock.Mock(pid=12345)
        process.poll.return_value = 0
        with mock.patch.object(launch.os, "killpg") as killpg:
            launch.stop_workers([{"process": process}], grace=0)
        self.assertEqual(killpg.call_args_list, [mock.call(12345, signal.SIGTERM), mock.call(12345, signal.SIGKILL)])
        process.wait.assert_called_once()


class CompletionTests(TemporaryProject):
    def complete(self):
        plan = self.plan()
        for worker in plan["workers"]:
            output = Path(worker["output"])
            output.mkdir(parents=True)
            launch.atomic_json(output / "final_protocol.json", {"arms": launch.ARMS, "updates": 8})
            launch.atomic_json(output / "summary.json", {"seed": worker["seed"], "arms": launch.ARMS,
                "target_updates": 8, "skipped_arms": {},
                "metrics": [{"arm": a, "cohort": "final"} for a in ["base", *launch.ARMS]],
                "training_completion": {a: {"update": 8} for a in launch.ARMS}})
        return plan

    def test_verification_rejects_missing_or_skipped_arms_and_wrong_budget(self):
        plan = self.complete()
        launch.verify_full_results(plan)
        summary = Path(plan["workers"][0]["output"]) / "summary.json"
        original = json.loads(summary.read_text())
        for changes in [{"metrics": []}, {"skipped_arms": {"judge": "missing grades"}},
                        {"training_completion": {a: {"update": 7} for a in launch.ARMS}},
                        {"seed": 99}]:
            launch.atomic_json(summary, {**original, **changes})
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                launch.verify_full_results(plan)

    def test_aggregation_uses_original_function_in_cuda_hidden_child(self):
        plan = self.complete()
        Path(plan["state_root"]).mkdir()
        for name in ("suite_summary.json", "suite_report.md"):
            (Path(plan["output_root"]) / name).write_text("fixture", encoding="utf-8")
        response = subprocess.CompletedProcess([], 0)
        with mock.patch.object(launch.subprocess, "run", return_value=response) as runner, redirect_stdout(io.StringIO()):
            launch.aggregate_results(plan, {"CUDA_VISIBLE_DEVICES": "3,1"})
        self.assertIn("from gsm8k_experiment.suite import aggregate", runner.call_args.args[0][3])
        self.assertEqual(runner.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(runner.call_args.kwargs["cwd"], plan["runtime"])
        for name in ("suite_summary.json", "suite_report.md"):
            self.assertEqual((Path(plan["state_root"]) / name).read_text(), "fixture")


if __name__ == "__main__":
    unittest.main()
