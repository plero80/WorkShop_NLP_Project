"""Original-CLI bridge tests using standard-library mocks only."""
from __future__ import annotations

import importlib.util
import json
from contextlib import contextmanager
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python_helper"))
import run_profile
from memory_profiles import profile_for, validate_settings


class ProfileWorkerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.runtime = Path(self.temporary.name)
        self.recipe, self.output = self.runtime / "profile.yaml", self.runtime / "seed_42"
        self.plan = {"seeds": None, "settings": {"arms": list(run_profile.ARMS),
                     "ppo": {"pilot_updates": 100, "full_updates": 400}}}
        self.cli = SimpleNamespace(resolve=mock.Mock(return_value=self.plan),
                                   command=mock.Mock(), materialize=mock.Mock())
        self.cli.command.side_effect = lambda plan, action: [sys.executable, "-u", "-m",
            "gsm8k_experiment.run", "--config", "original.json", "--stage",
            self.cli.resolve.call_args.kwargs["stage"], "--output", str(self.output)]
        self.validator = mock.patch.object(run_profile, "validate_settings").start()
        self.addCleanup(mock.patch.stopall)

    def command(self, stage="full", vram=12, arm=None):
        return run_profile.command_for(self.runtime, self.recipe, self.output, 42,
                                       stage, vram, arm, cli=self.cli)

    def test_prepare_and_full_keep_original_stages_and_seed_overrides(self):
        for stage in ("prepare", "pilot", "full"):
            with self.subTest(stage=stage):
                cmd = self.command(stage=stage)
                self.assertEqual(cmd[cmd.index("--stage") + 1], stage)
                self.assertNotIn("--updates", cmd)
                self.cli.resolve.assert_called_with(str(self.recipe), stage=stage,
                    output=str(self.output), overrides=("seed=42", "data_seed=42"))
                self.validator.assert_called_with(self.plan["settings"], 12)
                self.cli.materialize.assert_called_with(self.plan)

    def test_arm_workers_use_pilot_stage_with_requested_budget_and_full_arm_config(self):
        for stage, expected in (("pilot", "100"), ("full", "400")):
            with self.subTest(stage=stage):
                cmd = self.command(stage=stage, arm="judge")
                self.assertEqual(cmd[cmd.index("--stage") + 1], "pilot")
                self.assertEqual(cmd[-4:], ["--updates", expected, "--arms", "judge"])
                self.assertEqual(self.plan["settings"]["arms"], run_profile.ARMS)

    def test_validation_failure_prevents_materialization(self):
        self.validator.side_effect = ValueError("profile mismatch")
        with self.assertRaisesRegex(ValueError, "profile mismatch"):
            self.command()
        self.cli.materialize.assert_not_called()

    def test_suite_and_incomplete_arm_configuration_rejected(self):
        self.plan["seeds"] = [42, 43]
        with self.assertRaisesRegex(ValueError, "single-seed"):
            self.command()
        self.plan["seeds"] = None
        self.plan["settings"]["arms"] = ["proxy"]
        with self.assertRaisesRegex(ValueError, "three-arm"):
            self.command()
        self.cli.materialize.assert_not_called()

    def test_arm_prepare_and_unexpected_runner_rejected(self):
        with self.assertRaisesRegex(ValueError, "pilot/full"):
            self.command(stage="prepare", arm="proxy")
        self.cli.command.side_effect = None
        self.cli.command.return_value = [sys.executable, "-u", "-m", "unexpected.module"]
        with self.assertRaisesRegex(ValueError, "expected single-seed"):
            self.command()
        self.cli.materialize.assert_not_called()

    def worker(self, vram, runner, install):
        command = [sys.executable, "-u", "-m", "gsm8k_experiment.run",
                   "--config", "original.json", "--stage", "full"]
        with mock.patch.object(run_profile, "command_for", return_value=command) as build, \
             mock.patch.object(run_profile.importlib, "import_module", return_value=runner), \
             mock.patch.dict(sys.modules, {"gpu_residency": SimpleNamespace(install=install)}):
            result = run_profile.main(["--runtime", str(self.runtime), "--recipe", str(self.recipe),
                "--output", str(self.output), "--seed", "42", "--stage", "full",
                "--vram-gb", str(vram)])
        build.assert_called_once_with(self.runtime, self.recipe, self.output, 42, "full", vram, None)
        return result

    def test_12_installs_before_runner_and_cleans_up_after_in_process_call(self):
        events, previous = [], Path.cwd()
        controller = SimpleNamespace(close=mock.Mock(side_effect=lambda: events.append("close")))
        install = mock.Mock(side_effect=lambda: events.append("install") or controller)
        def main(argv):
            events.append("main")
            self.assertEqual(Path.cwd(), self.runtime)
            self.assertEqual(argv, ["--config", "original.json", "--stage", "full"])
            return 0
        self.assertEqual(self.worker(12, SimpleNamespace(main=main), install), 0)
        self.assertEqual(events, ["install", "main", "close"])
        self.assertEqual(Path.cwd(), previous)

    def test_other_profiles_run_without_residency_hooks(self):
        for vram in (24, 48):
            install, runner = mock.Mock(), SimpleNamespace(main=mock.Mock(return_value=0))
            self.assertEqual(self.worker(vram, runner, install), 0)
            install.assert_not_called()
            runner.main.assert_called_once()

    def test_runner_failure_cleans_up_and_restores_working_directory(self):
        previous = Path.cwd()
        controller = SimpleNamespace(close=mock.Mock())
        runner = SimpleNamespace(main=mock.Mock(side_effect=RuntimeError("synthetic run failure")))
        with self.assertRaisesRegex(RuntimeError, "synthetic"):
            self.worker(12, runner, mock.Mock(return_value=controller))
        controller.close.assert_called_once()
        self.assertEqual(Path.cwd(), previous)

    def test_windows_io_context_covers_materialization_and_runner_for_all_profiles(self):
        for vram in (12, 24, 48):
            events = []
            @contextmanager
            def protection():
                events.append("enter")
                try:
                    yield
                finally:
                    events.append("exit")
            original_run = run_profile.run
            def run(*args):
                self.assertEqual(events, ["enter"])
                events.append("materialize and run")
                return original_run(*args)
            runner = SimpleNamespace(main=mock.Mock(return_value=0))
            controller = SimpleNamespace(close=mock.Mock())
            with self.subTest(vram=vram), \
                 mock.patch.object(run_profile, "replacement_retries", protection), \
                 mock.patch.object(run_profile, "run", side_effect=run):
                self.assertEqual(self.worker(vram, runner, mock.Mock(return_value=controller)), 0)
            self.assertEqual(events, ["enter", "materialize and run", "exit"])

    def test_real_original_cli_resolves_and_materializes_only_under_temporary_runtime(self):
        source_runtime = Path(__file__).resolve().parents[1] / "runtime"
        spec = importlib.util.spec_from_file_location("profile_test_original_cli",
                                                     source_runtime / "experiment_cli" / "cli.py")
        original = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(original)
        layout = {"root": self.runtime, "settings": source_runtime / "gsm8k_experiment" / "settings.json",
                  "presets": self.runtime}
        cli = SimpleNamespace(resolve=lambda *a, **kw: original.resolve(*a, **kw, layout=layout),
                              command=original.command, materialize=original.materialize)
        for vram in (12, 24, 48):
            with self.subTest(vram=vram):
                profile = profile_for(vram)
                recipe = {"version": 1, "experiment": "gsm8k", "stage": "full", "settings": {
                    "arms": list(run_profile.ARMS),
                    "runtime": {"ppo_microbatch_size": profile["ppo_microbatch_size"],
                                "rollout_stats_batch_size": profile["rollout_stats_batch_size"]},
                    "generation": {"batch_size": profile["generation_batch_size"]},
                    "scoring": {"batch_size": profile["scoring_batch_size"]},
                    "teacher30b": {"evaluate_final": False}}}
                self.recipe.write_text(json.dumps(recipe), encoding="utf-8")
                with mock.patch.object(run_profile, "validate_settings", side_effect=validate_settings):
                    cmd = run_profile.command_for(self.runtime, self.recipe, self.output, 43,
                                                  "full", vram, "knn_static", cli=cli)
                config_path = Path(cmd[cmd.index("--config") + 1])
                self.assertTrue(config_path.is_relative_to(self.runtime))
                resolved = json.loads(config_path.read_text(encoding="utf-8"))
                self.assertEqual((resolved["seed"], resolved["data_seed"]), (43, 42))
                self.assertEqual(resolved["arms"], run_profile.ARMS)
                self.assertEqual(resolved["generation"]["max_new_tokens"], 768)
                self.assertEqual(cmd[cmd.index("--stage") + 1], "pilot")
                self.assertEqual(cmd[-4:], ["--updates", "400", "--arms", "knn_static"])


if __name__ == "__main__":
    unittest.main()
