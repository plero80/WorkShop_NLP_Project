#!/usr/bin/env python3
"""Export a single run or a multi-seed suite for analysis, without checkpoint weights."""
from __future__ import annotations
from gsm8k_experiment.common import DEFAULT_CONFIG, OUTPUT_ROOT
import argparse
from pathlib import Path
import zipfile

from .common import ROOT
from .shared import shared_sources


def export_results(output, destination):
    output, destination = Path(output).resolve(), Path(destination).resolve()
    if not (output / 'config.json').exists() and not (output / 'suite_protocol.json').exists():
        raise ValueError(f'No run or suite found at {output}')
    if destination.exists():
        raise ValueError(f'Export already exists: {destination}. Choose another --destination to preserve it.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(destination.name + '.tmp')
    try:
        with zipfile.ZipFile(temp, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted([*Path(__file__).parent.rglob('*'), ROOT / 'requirements.txt',
                                *[ROOT / name for name in shared_sources()]]):
                if not path.is_file():
                    continue
                rel = path.relative_to(ROOT)
                if any(x in rel.parts for x in ('.venv', '__pycache__', '.pytest_cache', 'outputs', '.git', '.ipynb_checkpoints')):
                    continue
                if path.name == 'runtime_environment.json':
                    continue
                if path.suffix in ('.py', '.json', '.md', '.txt', '.sh', '.ipynb') or path.name == '.gitignore':
                    archive.write(path, Path('code') / rel)
            for path in sorted(output.rglob('*')):
                if not path.is_file() or path in (destination, temp):
                    continue
                rel = path.relative_to(output)
                if any(x in rel.parts for x in ('generations', 'adapter', 'memories', '__pycache__')):
                    continue
                if path.suffix in ('.json', '.jsonl', '.csv', '.md', '.png', '.log', '.npz'):
                    archive.write(path, Path('outcomes') / rel)
        temp.replace(destination)
    finally:
        if temp.exists():
            temp.unlink()
    return destination


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=OUTPUT_ROOT / 'main')
    p.add_argument('--destination', type=Path, default=ROOT / 'important_outcomes_gsm8k.zip')
    args = p.parse_args()
    path = export_results(args.output, args.destination)
    print(f'Created {path} ({path.stat().st_size / 2**20:.1f} MiB)')
    print('Analysis export: code, reports, responses, call logs, and initial memory arrays. Keep original outputs to resume.')


if __name__ == '__main__':
    main()
