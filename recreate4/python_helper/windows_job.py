"""Own a Windows worker process tree, with a gate before any experiment code.

The unnamed, non-inheritable Job Object terminates all members when its last
handle closes, including if the coordinator dies. The child starts with -S so
site initialization cannot run before it is assigned to its job. Its stdin gate
also fails closed if the coordinator dies before assignment/release.
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import subprocess
import sys


class _IO_COUNTERS(ctypes.Structure):
    _fields_ = [(name, ctypes.c_ulonglong) for name in (
        "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
        "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]


class _BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_longlong),
        ("PerJobUserTimeLimit", ctypes.c_longlong),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class _EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _BASIC_LIMIT_INFORMATION),
        ("IoInfo", _IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class WindowsJob:
    """One owned kill-on-close job; no breakaway or inheritable handles."""

    def __init__(self):
        if os.name != "nt":
            raise OSError("Windows Job Objects require native Windows.")
        self.handle = None
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel.CreateJobObjectW.restype = wintypes.HANDLE
        kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int,
                                                   ctypes.c_void_p, wintypes.DWORD]
        kernel.SetInformationJobObject.restype = wintypes.BOOL
        kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        kernel.TerminateJobObject.restype = wintypes.BOOL
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        self.kernel = kernel
        self.handle = kernel.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = _EXTENDED_LIMIT_INFORMATION()
        limits.BasicLimitInformation.LimitFlags = 0x00002000  # KILL_ON_JOB_CLOSE
        if not kernel.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            error = ctypes.WinError(ctypes.get_last_error())
            self.close()
            raise error

    def assign(self, pid):
        if self.handle is None:
            raise RuntimeError("Cannot assign a process to a closed job.")
        process = self.kernel.OpenProcess(0x0100 | 0x0001, False, pid)  # SET_QUOTA | TERMINATE
        if not process:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            if not self.kernel.AssignProcessToJobObject(self.handle, process):
                raise ctypes.WinError(ctypes.get_last_error())
        finally:
            self.kernel.CloseHandle(process)

    def close(self):
        if self.handle is not None:
            # Explicit shutdown gives live workers a failing exit status. The
            # kill-on-close limit independently protects coordinator crashes.
            error = None
            if not self.kernel.TerminateJobObject(self.handle, 1):
                error = ctypes.WinError(ctypes.get_last_error())
            if not self.kernel.CloseHandle(self.handle):
                raise ctypes.WinError(ctypes.get_last_error())
            self.handle = None
            if error is not None:
                raise error

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def start_worker(command, **popen_options):
    """Return (Popen, job) only after assignment; fail before running user code.

    ``command`` retains the original Python CLI command for manifests/plans.
    The bootstrap runs its -m, -c or script entry point in the assigned process.
    """
    job = WindowsJob()
    process = None
    try:
        bootstrap = [command[0], "-u", "-S", str(Path(__file__).resolve()),
                     "--worker", json.dumps(command[1:])]
        process = subprocess.Popen(bootstrap, stdin=subprocess.PIPE,
                                   creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                                   **popen_options)
        job.assign(process.pid)
        process.stdin.write("G" if popen_options.get("text") else b"G")
        process.stdin.flush()
        process.stdin.close()
        process.stdin = None
        return process, job
    except BaseException:
        # Assignment failures leave the child gated; closing stdin alone would
        # stop it, but kill/wait ensures no unowned process survives this call.
        try:
            job.close()
        finally:
            if process is not None:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=10)
                for stream in (process.stdin, process.stdout, process.stderr):
                    if stream is not None:
                        stream.close()
        raise


def _worker(arguments):
    if sys.stdin.buffer.read(1) != b"G":
        return 125
    # Match normal Python startup only after the job owns this process.
    import site
    site.main()
    import runpy
    args = json.loads(arguments)
    while args and args[0] in ("-u", "-B"):
        args = args[1:]
    if not args:
        raise ValueError("Missing worker entry point.")
    sys.path[0] = os.getcwd()
    if args[0] == "-m" and len(args) >= 2:
        sys.argv = args[1:]
        runpy.run_module(args[1], run_name="__main__", alter_sys=True)
    elif args[0] == "-c" and len(args) >= 2:
        sys.argv = ["-c", *args[2:]]
        exec(compile(args[1], "<string>", "exec"), {"__name__": "__main__", "__package__": None})
    elif not args[0].startswith("-"):
        sys.argv = args
        sys.path[0] = str(Path(args[0]).resolve().parent)
        runpy.run_path(args[0], run_name="__main__")
    else:
        raise ValueError("Unsupported Python worker entry point.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--worker":
        raise SystemExit("This module is an internal gated Windows worker bootstrap.")
    raise SystemExit(_worker(sys.argv[2]))
