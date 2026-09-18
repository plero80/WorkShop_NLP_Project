"""Native Windows, CPU-only process-tree checks; never launch models or CUDA."""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stdout
from unittest import mock

HELPERS = Path(__file__).resolve().parents[1] / "python_helper"
sys.path.insert(0, str(HELPERS))
import windows_job
import launch_seeds


@unittest.skipUnless(os.name == "nt", "Native Windows Job Object integration")
class WindowsJobTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="recreate3 job test ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.kernel.OpenProcess.restype = wintypes.HANDLE
        self.kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        self.kernel.WaitForSingleObject.restype = wintypes.DWORD
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.kernel.CloseHandle.restype = wintypes.BOOL

    def watch(self, pid):
        handle = self.kernel.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        self.addCleanup(self.kernel.CloseHandle, handle)
        return handle

    def assert_alive(self, handle):
        self.assertEqual(self.kernel.WaitForSingleObject(handle, 0), 0x102)

    def assert_exited(self, handle):
        self.assertEqual(self.kernel.WaitForSingleObject(handle, 5000), 0)

    def read_ready(self, path):
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            try:
                return json.loads(path.read_text())
            except (OSError, ValueError):
                time.sleep(.02)
        self.fail(f"Fixture did not become ready: {path}")

    def unrelated_process(self):
        process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"],
                                   creationflags=subprocess.CREATE_NO_WINDOW)
        def cleanup():
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
        self.addCleanup(cleanup)
        return process

    def tree_code(self, pid_file, release_file=None):
        code = ("import json,os,subprocess,sys,time; from pathlib import Path; "
                "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
                f"Path({str(pid_file)!r}).write_text(json.dumps({{'parent':os.getpid(),'child':child.pid}})); ")
        if release_file is not None:
            code += f"\nwhile not Path({str(release_file)!r}).exists(): time.sleep(.02)\n"
        return code

    def test_native_structure_layout_and_kill_after_leader_exits(self):
        if ctypes.sizeof(ctypes.c_void_p) == 8:
            self.assertEqual(ctypes.sizeof(windows_job._BASIC_LIMIT_INFORMATION), 64)
            self.assertEqual(ctypes.sizeof(windows_job._EXTENDED_LIMIT_INFORMATION), 144)
        control = self.unrelated_process()
        ready = self.root / "tree.json"
        process, job = windows_job.start_worker(
            [sys.executable, "-u", "-c", self.tree_code(ready)], cwd=self.root,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.addCleanup(job.close)
        info = self.read_ready(ready)
        child = self.watch(info["child"])
        self.assertEqual(process.wait(timeout=5), 0)
        self.assert_alive(child)
        job.close()
        job.close()  # Idempotent; cannot close a reused OS handle.
        self.assert_exited(child)
        self.assertIsNone(control.poll(), "Unrelated process was affected")

    def test_assignment_failure_never_releases_user_code(self):
        marker = self.root / "must_not_execute"
        startup_marker = self.root / "site_must_not_execute"
        (self.root / "sitecustomize.py").write_text(
            f"from pathlib import Path; Path({str(startup_marker)!r}).write_text('ran')\n")
        spawned = []
        original_popen = subprocess.Popen
        def capture(*args, **kwargs):
            process = original_popen(*args, **kwargs)
            spawned.append(process)
            return process
        def reject(_job, _pid):
            time.sleep(.15)
            self.assertFalse(marker.exists())
            self.assertFalse(startup_marker.exists())
            raise OSError("Synthetic job assignment failure")
        with mock.patch.object(windows_job.WindowsJob, "assign", reject), \
             mock.patch.object(windows_job.subprocess, "Popen", capture):
            with self.assertRaisesRegex(OSError, "Synthetic job assignment"):
                windows_job.start_worker([sys.executable, "-c",
                    f"from pathlib import Path; Path({str(marker)!r}).write_text('ran')"],
                    env={**os.environ, "PYTHONPATH": str(self.root)},
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.assertEqual(len(spawned), 1)
        self.assertIsNotNone(spawned[0].poll())
        self.assertFalse(marker.exists())
        self.assertFalse(startup_marker.exists())

    def test_module_entry_point_preserves_arguments_interpreter_and_environment(self):
        package = self.root / "entry_fixture"
        package.mkdir()
        (package / "__init__.py").write_text("")
        (package / "__main__.py").write_text(
            "import json,os,sys; print(json.dumps({'argv':sys.argv,'cwd':os.getcwd(),"
            "'prefix':sys.prefix,'executable':sys.executable,"
            "'visible':os.environ['CUDA_VISIBLE_DEVICES']}))\n")
        command = [sys.executable, "-u", "-m", "entry_fixture", "--set", "seed=42", "path with spaces"]
        environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "GPU-fixture"}
        expected = subprocess.run(command, cwd=self.root, env=environment, capture_output=True,
                                  text=True, check=True)
        process, job = windows_job.start_worker(command, cwd=self.root, env=environment,
                                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.addCleanup(job.close)
        stdout, stderr = process.communicate(timeout=5)
        self.assertEqual(process.returncode, 0, stderr)
        self.assertEqual(json.loads(stdout), json.loads(expected.stdout))

    def test_gate_eof_without_assignment_exits_without_user_code(self):
        marker = self.root / "must_not_execute"
        args = ["-c", f"from pathlib import Path; Path({str(marker)!r}).write_text('ran')"]
        process = subprocess.Popen([sys.executable, "-u", "-S", str(HELPERS / "windows_job.py"),
                                    "--worker", json.dumps(args)], stdin=subprocess.PIPE,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        process.stdin.close()
        self.assertEqual(process.wait(timeout=5), 125)
        self.assertFalse(marker.exists())

    def test_coordinator_death_closes_job_and_kills_owned_tree_only(self):
        control = self.unrelated_process()
        ready = self.root / "tree.json"
        release = self.root / "never_release"
        code = (f"import sys,time; sys.path.insert(0,{str(HELPERS)!r}); "
                "from windows_job import start_worker; "
                f"worker,job=start_worker([sys.executable,'-c',{self.tree_code(ready, release)!r}], "
                "stdout=-3,stderr=-3); time.sleep(30)")
        coordinator = subprocess.Popen([sys.executable, "-c", code],
                                       creationflags=subprocess.CREATE_NO_WINDOW)
        def cleanup():
            if coordinator.poll() is None:
                coordinator.kill()
            coordinator.wait(timeout=5)
        self.addCleanup(cleanup)
        info = self.read_ready(ready)
        parent, child = self.watch(info["parent"]), self.watch(info["child"])
        self.assert_alive(parent)
        self.assert_alive(child)
        coordinator.kill()  # No Python finally/close handler can run.
        coordinator.wait(timeout=5)
        self.assert_exited(parent)
        self.assert_exited(child)
        self.assertIsNone(control.poll())

    def launcher_plan(self, code):
        return {"runtime": str(self.root), "output_root": str(self.root),
                "state_root": str(self.root), "log_root": str(self.root / "logs"),
                "stage": "pilot", "workers": [{"seed": 42, "gpu_token": "0",
                "output": str(self.root / "seed_42"), "log": str(self.root / "logs/seed_42.log"),
                "command": [sys.executable, "-u", "-c", code]}]}

    def test_launcher_normal_exit_reaps_grandchild(self):
        ready, release = self.root / "tree.json", self.root / "release"
        handles, errors = [], []
        def watch_then_release():
            try:
                info = self.read_ready(ready)
                handles.append(self.watch(info["child"]))
                release.touch()
            except BaseException as error:
                errors.append(error)
                release.touch()
        watcher = threading.Thread(target=watch_then_release)
        watcher.start()
        with redirect_stdout(io.StringIO()):
            result = launch_seeds.run_workers(self.launcher_plan(self.tree_code(ready, release)),
                                              dict(os.environ), poll_seconds=.02, grace=.1)
        watcher.join(timeout=10)
        self.assertFalse(watcher.is_alive())
        self.assertFalse(errors, errors)
        self.assertEqual(result, 0)
        self.assert_exited(handles[0])

    def test_launcher_sigint_reaps_worker_and_grandchild(self):
        control = self.unrelated_process()
        ready, release = self.root / "tree.json", self.root / "never_release"
        handles, errors = [], []
        cancellation = threading.Event()
        def watch_then_interrupt():
            try:
                info = self.read_ready(ready)
                handles.extend([self.watch(info["parent"]), self.watch(info["child"])])
                signal.raise_signal(signal.SIGINT)
            except BaseException as error:
                errors.append(error)
                cancellation.set()
        watcher = threading.Thread(target=watch_then_interrupt)
        watcher.start()
        with redirect_stdout(io.StringIO()):
            result = launch_seeds.run_workers(self.launcher_plan(self.tree_code(ready, release)),
                                              dict(os.environ), stop_event=cancellation,
                                              poll_seconds=.02, grace=.1)
        watcher.join(timeout=10)
        self.assertFalse(watcher.is_alive())
        self.assertFalse(errors, errors)
        self.assertEqual(result, 128 + signal.SIGINT)
        for handle in handles:
            self.assert_exited(handle)
        self.assertIsNone(control.poll())


if __name__ == "__main__":
    unittest.main()
