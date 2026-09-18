"""Verify the portable package, stage immutable runtime code, and restore snapshots."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
import uuid


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def safe_path(root, name):
    root = Path(root).resolve()
    path = (root / name).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError(f'Path escapes its directory: {name}')
    return path


def verify(package):
    package = Path(package).resolve()
    manifest = json.loads((package / 'provenance/package_manifest.json').read_text(encoding='utf-8'))
    relocated_windows = not os.path.lexists(package / 'windows')
    for name, expected in manifest['files'].items():
        path = safe_path(package, name)
        if relocated_windows and name.startswith('windows/'):
            # Windows may be moved beside recreate3 with its environment and
            # work tree. Only this fixed sibling is allowed; never fall back
            # for individual missing files in an existing embedded directory.
            path = safe_path(package.parent / 'windows', name.removeprefix('windows/'))
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f'Package file differs from its manifest: {name}')
    actual_runtime = {p.relative_to(package).as_posix() for p in (package / 'runtime').rglob('*')
                      if p.is_file() and '__pycache__' not in p.parts and '.experiment_cli' not in p.parts}
    expected_runtime = {name for name in manifest['files'] if name.startswith('runtime/')}
    if actual_runtime != expected_runtime:
        raise ValueError('Runtime contains unexpected or missing files. Use a clean package.')
    print(f'Verified {len(manifest["files"])} package files.', flush=True)
    return manifest


def stage(package, destination):
    package, destination = Path(package).resolve(), Path(destination).resolve()
    manifest = verify(package)
    names = {name.removeprefix('runtime/'): checksum for name, checksum in manifest['files'].items()
             if name.startswith('runtime/')}
    if destination.exists():
        for name, checksum in names.items():
            target = safe_path(destination, name)
            if not target.is_file() or sha256(target) != checksum:
                raise ValueError(f'Existing local runtime changed at {name}. Use a new experiment ID.')
        actual = {p.relative_to(destination).as_posix() for p in destination.rglob('*')
                  if p.is_file() and '__pycache__' not in p.parts and '.experiment_cli' not in p.parts}
        if actual != set(names):
            raise ValueError('Unexpected files in existing runtime; use a new experiment ID.')
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.runtime-', dir=destination.parent) as temp:
        tree = Path(temp) / 'runtime'
        shutil.copytree(package / 'runtime', tree, ignore=shutil.ignore_patterns('__pycache__', '.experiment_cli'))
        tree.rename(destination)
    print(f'Runtime: {destination}', flush=True)


def restore(runtime, archive_root, output_root):
    sys.path.insert(0, str(Path(runtime).resolve()))
    from gsm8k_experiment.archive import restore_output
    archive_root, output_root = Path(archive_root).resolve(), Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if not archive_root.exists():
        print('No archived snapshots yet; starting or continuing local preparation.', flush=True)
        return
    for folder in sorted(archive_root.iterdir()):
        if not folder.is_dir() or not re.fullmatch(r'seed_\d+(?:_(?:prepare|proxy|judge|knn_static))?', folder.name) or not (folder / 'latest.json').is_file():
            continue
        # Validate and restore into a new directory BEFORE touching any local work.
        temporary = output_root / f'.restore-{folder.name}-{uuid.uuid4().hex}'
        restore_output(folder, temporary)
        target = safe_path(output_root, folder.name)
        if target.exists():
            retained = output_root.parent / 'retained_local' / f'{folder.name}-{time.time_ns()}'
            retained.parent.mkdir(parents=True, exist_ok=True)
            target.rename(retained)
            print(f'Previous local work retained at {retained}', flush=True)
        temporary.rename(target)
        print(f'Restored published snapshot: {target}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest='action', required=True)
    v = commands.add_parser('verify')
    v.add_argument('--package', type=Path, required=True)
    s = commands.add_parser('stage')
    s.add_argument('--package', type=Path, required=True)
    s.add_argument('--destination', type=Path, required=True)
    r = commands.add_parser('restore')
    r.add_argument('--runtime', type=Path, required=True)
    r.add_argument('--archive-root', type=Path, required=True)
    r.add_argument('--output-root', type=Path, required=True)
    a = p.parse_args()
    try:
        if a.action == 'verify': verify(a.package)
        elif a.action == 'stage': stage(a.package, a.destination)
        else: restore(a.runtime, a.archive_root, a.output_root)
    except (ValueError, OSError) as error:
        p.error(str(error))


if __name__ == '__main__':
    main()
