#!/usr/bin/env python3
"""Read-only run status; no GPU/model imports and no process changes."""
from gsm8k_experiment.common import DEFAULT_CONFIG, OUTPUT_ROOT
import argparse
from collections import deque
import json
import os
from pathlib import Path

from .common import ROOT


def process_alive(pid):
    """Query a PID without sending a signal on Windows."""
    if pid <= 0:
        raise ValueError("A recorded PID must be positive.")
    if os.name != 'nt':
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False

    # os.kill(pid, 0) calls TerminateProcess on Windows; it is not a probe.
    import ctypes
    from ctypes import wintypes
    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel32.GetExitCodeProcess.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        error = ctypes.get_last_error()
        if error == 87:  # ERROR_INVALID_PARAMETER: the PID no longer exists.
            return False
        raise ctypes.WinError(error)
    try:
        code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            raise ctypes.WinError(ctypes.get_last_error())
        return code.value == 259  # STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


def show(folder, lines):
    print(f'\nOutput: {folder}')
    pidfile = folder / 'pid.txt'
    if pidfile.exists():
        try:
            pid = int(pidfile.read_text(encoding='utf-8'))
            if process_alive(pid):
                print(f'PID {pid} exists; check the stage/log below for actual progress.')
            else:
                print('The recorded process is no longer running.')
        except (ValueError, ProcessLookupError):
            print('The recorded process is no longer running.')
        except OSError as error:
            print(f'Could not verify the recorded PID: {error}')
    for name in ('suite_status.json', 'status.json'):
        if (folder / name).exists():
            print(json.dumps(json.loads((folder / name).read_text(encoding='utf-8')), indent=2))
    for path in sorted((folder / 'arms').glob('*/completed.json')):
        print(path.parent.name, 'completed target', json.loads(path.read_text(encoding='utf-8'))['update'])
    for name in ('experiment.log', 'suite.log'):
        path = folder / name
        if path.exists() and lines:
            print(f'Last {lines} lines of {name}:')
            with path.open(encoding='utf-8', errors='replace') as f:
                print(''.join(deque(f, maxlen=lines)), end='')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT_ROOT / 'main')
    parser.add_argument('--lines', type=int, default=8)
    args = parser.parse_args()
    show(args.output.resolve(), max(0, args.lines))
    state = args.output / 'suite_status.json'
    if state.exists():
        current = json.loads(state.read_text(encoding='utf-8')).get('seed')
        if current is not None:
            show(args.output.resolve() / f'seed_{current}', max(0, args.lines))
