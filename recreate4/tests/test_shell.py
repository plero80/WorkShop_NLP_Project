"""Exercise the real Linux runner with executable tool stubs, never a GPU/download.

Git Bash provides the shell on Windows. A temporary path containing spaces and
NUL-delimited records preserve argument boundaries without shell/JSON quoting.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


PACKAGE = Path(__file__).resolve().parents[1]
GIT_BASH = Path(r"C:\Program Files\Git\bin\bash.exe")
BASH = str(GIT_BASH) if os.name == "nt" and GIT_BASH.is_file() else shutil.which("bash")
UNSET = "__UNSET__"
ENVIRONMENT_KEYS = (
    "HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE", "TRITON_CACHE_DIR",
    "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE",
    "CUDA_VISIBLE_DEVICES", "RECREATE3_ARCHIVE_ROOT", "CONDA_DEFAULT_ENV",
)


def shell_path(path):
    value = Path(path).resolve().as_posix()
    if os.name == "nt" and len(value) > 2 and value[1] == ":":
        return "/" + value[0].lower() + value[2:]
    return value


def executable(path, text):
    path.write_text(text, encoding="utf-8", newline="\n")
    path.chmod(0o755)


STUB_TOOL = r'''#!/usr/bin/env bash
set -e
count=0
if [[ -f "$STUB_TRACE/count" ]]; then read -r count < "$STUB_TRACE/count"; fi
count=$((count + 1))
printf '%s\n' "$count" > "$STUB_TRACE/count"
record="$STUB_TRACE/$count"
printf '%s\0' "${0##*/}" "$@" > "$record.args"
for name in HF_HOME HF_HUB_CACHE HF_DATASETS_CACHE TRITON_CACHE_DIR \
            HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE \
            CUDA_VISIBLE_DEVICES RECREATE3_ARCHIVE_ROOT CONDA_DEFAULT_ENV; do
    value=__UNSET__
    if [[ -v "$name" ]]; then value="${!name}"; fi
    printf '%s\0%s\0' "$name" "$value" >> "$record.env"
done
for argument in "$@"; do
    if [[ "$argument" == */prepare_assets.py && "${STUB_FAIL_PREPARE:-0}" == 1 ]]; then
        printf 'Synthetic preparation failure\n' >&2
        exit 37
    fi
    if [[ "$argument" == --check && "${STUB_FAIL_CHECK:-0}" == 1 ]]; then
        printf 'Synthetic GPU validation failure\n' >&2
        exit 38
    fi
done
if [[ "${0##*/}" == rsync && "${STUB_FAIL_RSYNC:-0}" == 1 ]]; then
    printf 'Synthetic cache copy failure\n' >&2
    exit 39
fi
exit 0
'''


@unittest.skipUnless(BASH, "Bash is unavailable")
class ShellRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="recreate3 shell test ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.bin = self.root / "stub bin"
        self.trace = self.root / "trace"
        self.bin.mkdir()
        self.trace.mkdir()
        self.scratch = self.root / "scratch with spaces"
        self.local = self.root / "local with spaces"
        self.experiment = "shell_test-42"
        self.conda = self.root / "conda init.sh"
        executable(self.bin / "uname", "#!/usr/bin/env bash\nprintf 'Linux\\n'\n")
        # Force activate_env to source the explicit initialization file.
        executable(self.bin / "conda", "#!/usr/bin/env bash\nexit 1\n")
        for name in ("python", "rsync", "flock"):
            executable(self.bin / name, STUB_TOOL)
        executable(self.conda, r'''conda() {
    [[ "$1" == activate && "$2" == recreate3_shell_test ]] || return 9
    export PATH="$STUB_BIN:$PATH"
    export CONDA_DEFAULT_ENV="$2"
}
''')
        self.env = {key: value for key, value in os.environ.items()
                    if not (key.startswith(("BASH_FUNC_", "SLURM_", "HF_", "CONDA_")) or
                            key in {"BASH_ENV", "ENV", "RECIPE", "RECREATE3_ARCHIVE_ROOT",
                                    "TRANSFORMERS_OFFLINE", "TRITON_CACHE_DIR"})}
        self.env.update(
            SCRATCH=shell_path(self.scratch), TMP_ROOT=shell_path(self.local),
            CONDA_EXEC=shell_path(self.conda), RECREATE3_ENV="recreate3_shell_test",
            STUB_BIN=shell_path(self.bin), STUB_TRACE=shell_path(self.trace),
            USER="recreate3_shell_test", CUDA_VISIBLE_DEVICES="GPU-first,GPU-second",
            # Deliberately inherited: preparation must clear all three, training restores 1.
            HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
        )

    def run_runner(self, *arguments, **environment):
        env = dict(self.env, **environment)
        if os.name == "nt" and "--dry-run" not in arguments:
            # Git Bash cannot set POSIX mode 700 under some Windows ACLs. An
            # existing TMP_ROOT makes mkdir -p leave its permissions unchanged.
            self.local.mkdir()
        result = subprocess.run(
            [BASH, "--noprofile", "--norc", "-c",
             'export PATH="$STUB_BIN:$PATH"; exec bash "$@"', "shell-test",
             shell_path(PACKAGE / "runners/run.sh"), self.experiment, *arguments],
            env=env, text=True, capture_output=True, timeout=30)
        return result

    def records(self):
        result = []
        for path in sorted(self.trace.glob("*.args"), key=lambda p: int(p.stem)):
            args = path.read_bytes().decode("utf-8").split("\0")[:-1]
            fields = path.with_suffix(".env").read_bytes().decode("utf-8").split("\0")[:-1]
            result.append({"tool": args[0], "args": args[1:], "env": dict(zip(fields[::2], fields[1::2]))})
        return result

    @staticmethod
    def helper(record, name):
        return record["tool"] == "python" and any(x.endswith("/" + name) for x in record["args"])

    @staticmethod
    def option(record, name):
        return record["args"][record["args"].index(name) + 1]

    def test_cache_boundaries_launch_order_and_inherited_gpu_mask(self):
        result = self.run_runner("--stage", "pilot", "--seeds", "42,43")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        records = self.records()
        checks = [r for r in records if self.helper(r, "launch_seeds.py") and "--check" in r["args"]]
        prepares = [r for r in records if self.helper(r, "prepare_assets.py")]
        launches = [r for r in records if self.helper(r, "launch_seeds.py") and "--check" not in r["args"]]
        self.assertEqual((len(checks), len(prepares), len(launches)), (1, 1, 1))
        check, prepare, launch = checks[0], prepares[0], launches[0]
        self.assertLess(records.index(check), records.index(prepare))
        copies = [r for r in records if r["tool"] == "rsync"]
        self.assertEqual(len(copies), 2)
        for copy in copies:
            self.assertLess(records.index(prepare), records.index(copy))
            self.assertLess(records.index(copy), records.index(launch))
        restores = [r for r in records if self.helper(r, "package_state.py") and "restore" in r["args"]]
        self.assertEqual(len(restores), 1)
        self.assertLess(records.index(copies[-1]), records.index(restores[0]))
        self.assertLess(records.index(restores[0]), records.index(launch))

        scratch = shell_path(self.scratch)
        local = shell_path(self.local / "recreate3" / self.experiment)
        self.assertEqual(prepare["env"]["HF_HOME"], scratch + "/weights/recreate3/huggingface")
        self.assertEqual(prepare["env"]["HF_HUB_CACHE"], prepare["env"]["HF_HOME"] + "/hub")
        self.assertEqual(prepare["env"]["HF_DATASETS_CACHE"], scratch + "/datasets/recreate3/huggingface")
        self.assertEqual(launch["env"]["HF_HOME"], local + "/cache/huggingface")
        self.assertEqual(launch["env"]["HF_HUB_CACHE"], local + "/cache/huggingface/hub")
        self.assertEqual(launch["env"]["HF_DATASETS_CACHE"], local + "/cache/datasets")
        self.assertEqual(launch["env"]["TRITON_CACHE_DIR"], local + "/cache/triton")
        for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
            self.assertEqual(prepare["env"][key], UNSET)
            self.assertEqual(launch["env"][key], "1")
        for record in (check, prepare, launch):
            self.assertEqual(record["env"]["CUDA_VISIBLE_DEVICES"], "GPU-first,GPU-second")
            self.assertEqual(record["env"]["CONDA_DEFAULT_ENV"], "recreate3_shell_test")
        self.assertEqual(self.option(launch, "--runtime"), local + "/runtime")
        self.assertEqual(self.option(launch, "--output-root"), local + "/work/runs")
        self.assertEqual(self.option(launch, "--state-root"), scratch + "/checkpoints/recreate3/" + self.experiment)
        self.assertEqual(self.option(launch, "--log-root"), scratch + "/logs/recreate3/" + self.experiment)
        self.assertEqual(launch["env"]["RECREATE3_ARCHIVE_ROOT"], self.option(launch, "--state-root") + "/archive")
        self.assertEqual(self.option(launch, "--stage"), "pilot")
        self.assertEqual(self.option(launch, "--seeds"), "42,43")

    def test_preparation_failure_prevents_copy_restore_and_training(self):
        result = self.run_runner(STUB_FAIL_PREPARE="1")
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        records = self.records()
        self.assertTrue(any(self.helper(r, "prepare_assets.py") for r in records))
        self.assertFalse(any(r["tool"] == "rsync" for r in records))
        self.assertFalse(any(self.helper(r, "package_state.py") and "restore" in r["args"] for r in records))
        self.assertFalse(any(self.helper(r, "launch_seeds.py") and "--check" not in r["args"] for r in records))

    def test_failed_gpu_check_prevents_preparation(self):
        result = self.run_runner(STUB_FAIL_CHECK="1")
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        records = self.records()
        self.assertTrue(any(self.helper(r, "launch_seeds.py") and "--check" in r["args"] for r in records))
        self.assertFalse(any(self.helper(r, "prepare_assets.py") for r in records))
        self.assertFalse(any(r["tool"] == "rsync" for r in records))

    def test_cache_copy_failure_prevents_restore_and_training(self):
        result = self.run_runner(STUB_FAIL_RSYNC="1")
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        records = self.records()
        self.assertEqual(sum(r["tool"] == "rsync" for r in records), 1)
        self.assertFalse(any(self.helper(r, "package_state.py") and "restore" in r["args"] for r in records))
        self.assertFalse(any(self.helper(r, "launch_seeds.py") and "--check" not in r["args"] for r in records))

    def test_dry_run_only_invokes_launcher_and_creates_no_experiment_files(self):
        result = self.run_runner("--dry-run", "--gpus", "GPU-second", "--seeds", "43")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        records = self.records()
        self.assertEqual(len(records), 1)
        self.assertTrue(self.helper(records[0], "launch_seeds.py"))
        self.assertIn("--dry-run", records[0]["args"])
        self.assertNotIn("--check", records[0]["args"])
        self.assertEqual(self.option(records[0], "--gpus"), "GPU-second")
        self.assertEqual(self.option(records[0], "--seeds"), "43")
        self.assertFalse(self.scratch.exists())
        self.assertFalse(self.local.exists())


if __name__ == "__main__":
    unittest.main()
