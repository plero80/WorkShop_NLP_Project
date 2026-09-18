"""Scheduling options cross the real wrappers; all experiment commands are stubbed."""
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
PWSH = shutil.which("pwsh")
_shell_spec = importlib.util.spec_from_file_location("parallel_shell_fixture", Path(__file__).with_name("test_shell.py"))
_shell = importlib.util.module_from_spec(_shell_spec)
_shell_spec.loader.exec_module(_shell)


@unittest.skipUnless(_shell.BASH, "Bash unavailable")
class LinuxParallelForwardingTests(unittest.TestCase):
    def setUp(self):
        # Reuse executable stubs, not the original class's test methods.
        self.fixture = _shell.ShellRunnerTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_arms_reaches_preflight_and_launch_unchanged(self):
        result = self.fixture.run_runner("--parallelism", "arms", "--gpus", "0,1,2", "--seeds", "42")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        launchers = [r for r in self.fixture.records() if self.fixture.helper(r, "launch_seeds.py")]
        self.assertEqual(len(launchers), 2)
        for record in launchers:
            self.assertEqual(self.fixture.option(record, "--parallelism"), "arms")
            self.assertEqual(self.fixture.option(record, "--seeds"), "42")
            self.assertEqual(self.fixture.option(record, "--gpus"), "0,1,2")
        self.assertIn("--check", launchers[0]["args"])
        self.assertNotIn("--check", launchers[1]["args"])

    def test_seed_mode_dry_run_forwards_full_seed_list_without_outputs(self):
        result = self.fixture.run_runner("--dry-run", "--parallelism", "seeds", "--gpus", "0,1,2", "--seeds", "42,43,44")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        records = self.fixture.records()
        self.assertEqual(len(records), 1)
        self.assertEqual(self.fixture.option(records[0], "--parallelism"), "seeds")
        self.assertEqual(self.fixture.option(records[0], "--seeds"), "42,43,44")
        self.assertFalse(self.fixture.local.exists())
        self.assertFalse(self.fixture.scratch.exists())

    def test_help_describes_the_new_default(self):
        result = subprocess.run([_shell.BASH, "--noprofile", "--norc", "-c",
                                'export PATH="/usr/bin:/bin:$PATH"; exec bash "$@"', "help-test",
                                _shell.shell_path(PACKAGE / "runners/run.sh"), "--help"],
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--parallelism auto|seeds|arms", result.stdout)
        self.assertIn("Extra GPUs are unused", result.stdout)
        self.assertIn("fresh ID", result.stdout)


# The copied real Run.ps1 sources this fixture instead of Common.ps1. No command
# can touch the GPU, real environment, real caches, or non-fixture process tree.
COMMON_FIXTURE = r'''
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
function Get-LocalPaths {
    param($ExperimentId, $WorkRoot, $VenvPath, $Recipe)
    return (Get-Content -LiteralPath $env:WRAPPER_PATHS -Raw | ConvertFrom-Json -AsHashtable)
}
function Assert-LocalPython { param($Paths) }
function Invoke-NativeChecked {
    param([string]$Executable, [string[]]$Arguments)
    Add-Content -LiteralPath $env:WRAPPER_TRACE -Value (@{kind='native'; executable=$Executable; args=$Arguments} | ConvertTo-Json -Compress -Depth 10)
}
function Invoke-WithLocalEnvironment { param($Paths, [scriptblock]$Action); & $Action }
function Open-PipelineLock { param($StateRoot, $WaitSeconds); return [IO.MemoryStream]::new() }
function Get-RecordedProcess { param($RecordPath); return $null }
function Join-NativeArguments { param([string[]]$Values); return ($Values | ConvertTo-Json -Compress) }
function Write-AtomicJson {
    param($Path, $Value)
    Add-Content -LiteralPath $env:WRAPPER_TRACE -Value (@{kind='record'; value=$Value} | ConvertTo-Json -Compress -Depth 10)
}
function Start-Process {
    param($FilePath, $ArgumentList, $PassThru, $WindowStyle, $RedirectStandardOutput, $RedirectStandardError)
    Add-Content -LiteralPath $env:WRAPPER_TRACE -Value (@{kind='start'; args=($ArgumentList | ConvertFrom-Json); window=$WindowStyle} | ConvertTo-Json -Compress -Depth 10)
    return [PSCustomObject]@{Id=12345; StartTime=[DateTime]::Now; HasExited=$true}
}
'''


@unittest.skipUnless(PWSH, "PowerShell 7 unavailable")
class PowerShellParallelForwardingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="recreate3 parallel wrapper ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.windows = self.root / "windows"
        self.windows.mkdir()
        shutil.copy2(PACKAGE / "windows/Run.ps1", self.windows / "Run.ps1")
        (self.windows / "Common.ps1").write_text(COMMON_FIXTURE, encoding="utf-8")
        helpers = self.root / "helpers"
        helpers.mkdir()
        (helpers / "prepare_assets.py").write_text("print('CPU-only preparation fixture')\n", encoding="utf-8")
        logs = self.root / "logs"
        logs.mkdir()
        self.paths = {
            "Python": sys.executable, "Helpers": str(helpers), "Runtime": str(self.root / "runtime"),
            "Recipe": str(self.root / "recipe.yaml"), "Output": str(self.root / "outputs"),
            "State": str(self.root / "state"), "Logs": str(logs), "Package": str(self.root),
            "Work": str(self.root / "work"), "Venv": str(self.root / "venv"),
            "Job": str(logs / "process.json"), "Metadata": str(self.root / "metadata"),
        }
        path_file = self.root / "paths.json"
        path_file.write_text(json.dumps(self.paths), encoding="utf-8")
        self.trace = self.root / "trace.jsonl"
        self.env = dict(os.environ, WRAPPER_PATHS=str(path_file), WRAPPER_TRACE=str(self.trace), PYTHONDONTWRITEBYTECODE="1")

    def run_script(self, *arguments):
        return subprocess.run([PWSH, "-NoLogo", "-NoProfile", "-File", str(self.windows / "Run.ps1"),
                               "-ExperimentId", "fixture", *arguments], env=self.env,
                              capture_output=True, text=True, timeout=20,
                              creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)

    def records(self):
        return [json.loads(line) for line in self.trace.read_text(encoding="utf-8-sig").splitlines()] if self.trace.exists() else []

    @staticmethod
    def option(record, name):
        return record["args"][record["args"].index(name) + 1]

    def test_default_auto_reaches_dry_run_without_creating_outputs(self):
        result = self.run_script("-DryRun", "-Gpus", "0,1,2", "-Seeds", "42")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        record, = self.records()
        self.assertEqual(self.option(record, "--parallelism"), "auto")
        self.assertIn("--dry-run", record["args"])
        self.assertFalse(Path(self.paths["State"]).exists())
        self.assertFalse(Path(self.paths["Output"]).exists())

    def test_explicit_arms_reaches_check(self):
        result = self.run_script("-Check", "-Parallelism", "arms", "-Gpus", "0,1,2", "-Seeds", "73")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        record, = self.records()
        self.assertEqual(self.option(record, "--parallelism"), "arms")
        self.assertEqual(self.option(record, "--seeds"), "73")
        self.assertIn("--check", record["args"])

    def test_background_controller_preserves_seed_mode(self):
        result = self.run_script("-Background", "-Parallelism", "seeds", "-Gpus", "0,1,2", "-Seeds", "42,43,44")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        start = next(r for r in self.records() if r["kind"] == "start")
        self.assertEqual(self.option(start, "-Parallelism"), "seeds")
        self.assertEqual(self.option(start, "-Seeds"), "42,43,44")
        self.assertIn("-Worker", start["args"])
        self.assertEqual(start["window"], "Hidden")
        self.assertFalse(any(r["kind"] == "native" for r in self.records()))

    def test_worker_forwards_arms_to_preflight_and_final_launch(self):
        result = self.run_script("-Worker", "-Parallelism", "arms", "-Gpus", "0,1,2", "-Seeds", "42")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        launchers = [r for r in self.records() if r["kind"] == "native" and r["args"][0].endswith("launch_seeds.py")]
        self.assertEqual(len(launchers), 2)
        for record in launchers:
            self.assertEqual(self.option(record, "--parallelism"), "arms")
            self.assertEqual(self.option(record, "--seeds"), "42")
        self.assertIn("--check", launchers[0]["args"])
        self.assertNotIn("--check", launchers[1]["args"])

    def test_invalid_mode_fails_parameter_binding(self):
        result = self.run_script("-DryRun", "-Parallelism", "ddp")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.records(), [])


if __name__ == "__main__":
    unittest.main()
