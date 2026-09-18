"""CPU-only VRAM recipe selection and forwarding through the actual wrappers."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


PACKAGE = Path(__file__).resolve().parents[1]


def load_fixture(name):
    spec = importlib.util.spec_from_file_location("vram_fixture_" + name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


shell_fixture = load_fixture("test_shell")
parallel_fixture = load_fixture("test_parallel_wrappers")


@unittest.skipUnless(shell_fixture.BASH, "Bash unavailable")
class LinuxVramTests(unittest.TestCase):
    def setUp(self):
        self.fixture = shell_fixture.ShellRunnerTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def calls(self, name):
        return [r for r in self.fixture.records() if self.fixture.helper(r, name)]

    def test_default_24_recipe_and_profile_are_forwarded(self):
        result = self.fixture.run_runner("--dry-run", "--gpus", "0")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        record, = self.calls("launch_seeds.py")
        self.assertEqual(self.fixture.option(record, "--vram-gb"), "24")
        self.assertTrue(self.fixture.option(record, "--recipe").endswith("/configs/gsm8k-24gb.yaml"))

    def test_12_recipe_is_identical_for_preflight_prewarm_and_launch(self):
        result = self.fixture.run_runner("--vram-gb", "12", "--gpus", "0")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        launchers = self.calls("launch_seeds.py")
        prepare, = self.calls("prepare_assets.py")
        self.assertEqual(len(launchers), 2)
        recipe = self.fixture.option(prepare, "--recipe")
        self.assertTrue(recipe.endswith("/configs/gsm8k-12gb.yaml"))
        for call in launchers:
            self.assertEqual(self.fixture.option(call, "--recipe"), recipe)
            self.assertEqual(self.fixture.option(call, "--vram-gb"), "12")

    def test_check_resolves_equals_profile_without_asset_preparation(self):
        result = self.fixture.run_runner("--vram-gb=48", "--check", "--gpus", "0")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        record, = self.fixture.records()
        self.assertTrue(self.fixture.helper(record, "launch_seeds.py"))
        self.assertEqual(self.fixture.option(record, "--vram-gb"), "48")
        self.assertTrue(self.fixture.option(record, "--recipe").endswith("/configs/gsm8k-48gb.yaml"))
        self.assertIn("--check", record["args"])
        self.assertFalse(self.fixture.scratch.exists())
        self.assertFalse((self.fixture.local / "recreate3").exists())

    def test_custom_recipe_is_used_by_preparation_and_both_launcher_calls(self):
        custom = shell_fixture.shell_path(self.fixture.root / "custom profile with spaces.yaml")
        result = self.fixture.run_runner("--recipe", custom, "--vram-gb", "48", "--gpus", "0")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        for record in self.calls("launch_seeds.py") + self.calls("prepare_assets.py"):
            self.assertEqual(self.fixture.option(record, "--recipe"), custom)
            self.assertEqual(record["args"].count("--recipe"), 1)

    def test_environment_recipe_override_is_retained(self):
        custom = shell_fixture.shell_path(self.fixture.root / "environment recipe.yaml")
        result = self.fixture.run_runner("--vram-gb", "12", "--dry-run", "--gpus", "0", RECIPE=custom)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        record, = self.calls("launch_seeds.py")
        self.assertEqual(self.fixture.option(record, "--recipe"), custom)

    def test_invalid_or_missing_profile_fails_before_commands_or_outputs(self):
        for arguments in [("--dry-run", "--vram-gb", "16"), ("--dry-run", "--vram-gb")]:
            with self.subTest(arguments=arguments):
                result = self.fixture.run_runner(*arguments)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(self.fixture.records(), [])
                self.assertFalse(self.fixture.scratch.exists())
                self.assertFalse(self.fixture.local.exists())


@unittest.skipUnless(parallel_fixture.PWSH, "PowerShell 7 unavailable")
class PowerShellVramTests(unittest.TestCase):
    def setUp(self):
        self.fixture = parallel_fixture.PowerShellParallelForwardingTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        # Use actual profile/path resolution, overriding only the Python executable
        # and side effects. This fixture never loads a model or probes a GPU.
        common = (PACKAGE / "windows/Common.ps1").read_text(encoding="utf-8")
        common = common.replace("    return $Result", "    $Result.Python = $env:WRAPPER_PYTHON\n    return $Result")
        stubs = "function Assert-LocalPython" + parallel_fixture.COMMON_FIXTURE.split("function Assert-LocalPython", 1)[1]
        (self.fixture.windows / "Common.ps1").write_text(common + "\n" + stubs, encoding="utf-8")
        self.fixture.env["WRAPPER_PYTHON"] = sys.executable
        self.helpers = self.fixture.root / "python_helper"
        self.helpers.mkdir()
        (self.fixture.root / "runtime/experiment_cli").mkdir(parents=True)
        (self.helpers / "prepare_assets.py").write_text(
            "import json,os,sys\n"
            "with open(os.environ['WRAPPER_TRACE'], 'a', encoding='utf-8') as f:\n"
            "    f.write(json.dumps({'kind':'prepare','args':sys.argv[1:]}) + '\\n')\n", encoding="utf-8")
        self.work = self.fixture.root / "selected work"
        (self.work / "logs/fixture").mkdir(parents=True)

    def run_script(self, *args):
        return self.fixture.run_script("-WorkRoot", str(self.work), *args)

    def launchers(self):
        return [r for r in self.fixture.records() if r["kind"] == "native" and r["args"][0].endswith("launch_seeds.py")]

    def test_default_profile_24_and_real_common_recipe_resolution(self):
        result = self.run_script("-DryRun", "-Gpus", "0")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        record, = self.launchers()
        self.assertEqual(self.fixture.option(record, "--vram-gb"), "24")
        self.assertEqual(Path(self.fixture.option(record, "--recipe")).name, "gsm8k-24gb.yaml")
        self.assertFalse((self.work / "state").exists())
        self.assertFalse((self.work / "runs").exists())

    def test_check_selects_12_before_launcher(self):
        result = self.run_script("-Check", "-VramGB", "12", "-Gpus", "0")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        record, = self.launchers()
        self.assertEqual(self.fixture.option(record, "--vram-gb"), "12")
        self.assertEqual(Path(self.fixture.option(record, "--recipe")).name, "gsm8k-12gb.yaml")
        self.assertIn("--check", record["args"])

    def test_hidden_child_receives_profile_and_selected_recipe(self):
        result = self.run_script("-Background", "-VramGB", "48", "-Gpus", "0")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        start = next(r for r in self.fixture.records() if r["kind"] == "start")
        self.assertEqual(self.fixture.option(start, "-VramGB"), "48")
        self.assertEqual(Path(self.fixture.option(start, "-Recipe")).name, "gsm8k-48gb.yaml")
        self.assertEqual(start["window"], "Hidden")

    def test_worker_prewarm_and_launch_share_selected_12_recipe(self):
        result = self.run_script("-Worker", "-VramGB", "12", "-Gpus", "0")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        preparation, = [r for r in self.fixture.records() if r["kind"] == "prepare"]
        expected = self.fixture.option(preparation, "--recipe")
        self.assertEqual(Path(expected).name, "gsm8k-12gb.yaml")
        self.assertEqual(len(self.launchers()), 2)
        for record in self.launchers():
            self.assertEqual(self.fixture.option(record, "--recipe"), expected)
            self.assertEqual(self.fixture.option(record, "--vram-gb"), "12")

    def test_custom_recipe_is_not_replaced_by_profile(self):
        custom = self.fixture.root / "custom recipe.yaml"
        result = self.run_script("-Worker", "-VramGB", "48", "-Recipe", str(custom), "-Gpus", "0")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = self.launchers() + [r for r in self.fixture.records() if r["kind"] == "prepare"]
        self.assertEqual(len(calls), 3)
        for record in calls:
            self.assertEqual(self.fixture.option(record, "--recipe"), str(custom))

    def test_unsupported_profile_fails_parameter_binding(self):
        result = self.run_script("-DryRun", "-VramGB", "16", "-Gpus", "0")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.fixture.records(), [])


if __name__ == "__main__":
    unittest.main()
