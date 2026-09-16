"""Build a submission with one source tree and optional completed run results.

    python submission.py --destination ../submission.zip
    python submission.py --destination ../submission-with-results.zip --results ../run/gsm8k/gsm8k_outputs/b200

Only Git-tracked project files enter the source tree, including reference input
adapters. Runtime copies, backups and environments are excluded. Added results
contain no generated checkpoints or copied source.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import zipfile

from reproducibility.repair_gsm8k_inline_score import lock

PROJECT = Path(__file__).resolve().parent
REPO = PROJECT.parent
RESULT_SUFFIXES = {".json", ".jsonl", ".csv", ".md", ".png", ".npz", ".log"}
SKIP_PARTS = {"generations", "adapter", "memories", "__pycache__", "original_checkpoints"}


def git(*args):
    return subprocess.check_output(["git", "-C", str(REPO), "-c", "safe.directory=" + REPO.as_posix(), *args])


def source_files():
    names = git("ls-files", "-z", "--", "README.md", ".gitignore", ".gitattributes", "workshop_project").decode().split("\0")
    for name in names:
        if not name or name.startswith("workshop_project/gsm8k_outputs/"):
            continue
        path = REPO / name
        if path.is_symlink():
            raise ValueError(f"Submission source cannot be a link: {name}")
        if path.is_file():
            yield path, name


def build(destination, results=()):
    destination = Path(destination).resolve()
    if destination.exists():
        raise ValueError("Submission already exists; choose a new destination.")
    folders = [Path(p).resolve() for p in results]
    if len({p.name for p in folders}) != len(folders):
        raise ValueError("Result folder names must be distinct.")
    with ExitStack() as stack:
        for folder in folders:
            state = next((folder / name for name in ("status.json", "suite_status.json") if (folder / name).is_file()), None)
            if state is None or json.loads(state.read_text())["stage"] != "complete":
                raise ValueError(f"Wait for this run to finish before packaging its results: {folder}")
            stack.enter_context(lock(folder))
            if json.loads(state.read_text())["stage"] != "complete":
                raise ValueError(f"Run state changed before packaging: {folder}")
        files = list(source_files())
        if not files:
            raise ValueError("No tracked project files. Build a submission from the Git checkout.")
        for folder in folders:
            for path in sorted(folder.rglob('*')):
                rel = path.relative_to(folder)
                if path.is_symlink():
                    raise ValueError(f"Result files cannot be links: {path}")
                if path.is_file() and not (set(rel.parts) & SKIP_PARTS) and path.suffix in RESULT_SUFFIXES:
                    files.append((path, (Path('workshop_project/results/gsm8k') / folder.name / rel).as_posix()))
        if len({name for _, name in files}) != len(files):
            raise ValueError("Submission paths overlap; choose distinct result folders.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=destination.name + '.', suffix='.tmp', dir=destination.parent)
        os.close(descriptor)
        temp = Path(temporary)
        try:
            hashes = {}
            with zipfile.ZipFile(temp, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
                for path, name in files:
                    data = path.read_bytes()
                    archive.writestr(name, data)
                    hashes[name] = hashlib.sha256(data).hexdigest()
                archive.writestr('SUBMISSION_MANIFEST.json', json.dumps({
                    'git_commit': git('rev-parse', 'HEAD').decode().strip(),
                    'source': 'tracked project files at packaging time; exact bytes pinned below',
                    'results_included': [folder.name for folder in folders],
                    'runtime_copies_included': False, 'sha256': hashes}, indent=2))
            temp.replace(destination)
        finally:
            if temp.exists():
                temp.unlink()
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--results', type=Path, action='append', default=[])
    args = parser.parse_args()
    try:
        path = build(args.destination, args.results)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.error(str(error))
    print(f'Created {path}. Source code appears once; result folders contain no copied code or checkpoints.')


if __name__ == '__main__':
    main()
