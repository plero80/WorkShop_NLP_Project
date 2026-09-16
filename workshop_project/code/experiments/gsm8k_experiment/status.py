#!/usr/bin/env python3
"""Read-only run status; no GPU/model imports and no process changes."""
from gsm8k_experiment.common import DEFAULT_CONFIG, OUTPUT_ROOT
import argparse
from collections import deque
import json
import os
from pathlib import Path

from .common import ROOT


def show(folder, lines):
    print(f'\nOutput: {folder}')
    pidfile = folder / 'pid.txt'
    if pidfile.exists():
        try:
            pid = int(pidfile.read_text())
            os.kill(pid, 0)
            print(f'PID {pid} exists; check the stage/log below for actual progress.')
        except (ValueError, ProcessLookupError):
            print('The recorded process is no longer running.')
    for name in ('suite_status.json', 'status.json'):
        if (folder / name).exists():
            print(json.dumps(json.loads((folder / name).read_text()), indent=2))
    for path in sorted((folder / 'arms').glob('*/completed.json')):
        print(path.parent.name, 'completed target', json.loads(path.read_text())['update'])
    for name in ('experiment.log', 'suite.log'):
        path = folder / name
        if path.exists() and lines:
            print(f'Last {lines} lines of {name}:')
            with path.open(errors='replace') as f:
                print(''.join(deque(f, maxlen=lines)), end='')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT_ROOT / 'main')
    parser.add_argument('--lines', type=int, default=8)
    args = parser.parse_args()
    show(args.output.resolve(), max(0, args.lines))
    state = args.output / 'suite_status.json'
    if state.exists():
        current = json.loads(state.read_text()).get('seed')
        if current is not None:
            show(args.output.resolve() / f'seed_{current}', max(0, args.lines))
