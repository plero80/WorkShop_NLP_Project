#!/usr/bin/env python3
"""Recover an existing kNN-distillation study without reserving a new test set.

Place this file beside ppo_engine.py, OUTSIDE the knn_distillation package.
  python RESUME_KNN_DISTILLATION.py --project . --resume
  python RESUME_KNN_DISTILLATION.py --project . --list
  python RESUME_KNN_DISTILLATION.py --project . --study study_<id> --resume

The launcher restores the saved scientific settings, including the full seed
list. It checks the exact study identity before invoking the original runner.
It never deletes reservations, edits manifests, reduces the test set, merges
studies, or changes an existing scientific source file. If the package changed,
it can use an exact, hash-verified source copy from the selected study's own
important_outcomes_knn_distillation.zip (or --code-archive PATH).

Without --resume it inspects and validates only. A running experiment retains
the original project GPU lock; this launcher will not run over another job.
"""
from pathlib import Path
from importlib import metadata
import argparse
import hashlib
import importlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
import zipfile


def read(path):
    return json.loads(Path(path).read_text())


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def scientific(obj):
    return {k: v for k, v in obj.items()
            if k not in ('allow_downloads', 'extra_hf_cache', 'max_wall_hours')}


def scan(project):
    records = []
    for path in sorted((project / 'knn_distillation_outputs').glob('study_*/manifest.json')):
        out = path.parent
        try:
            m = read(path)
            if digest({k: v for k, v in m.items() if k != 'identity'}) != m['identity']:
                raise ValueError('Saved manifest checksum mismatch')
            if out.name != 'study_' + m['identity'][:16]:
                raise ValueError('Study folder does not match its manifest')
            complete = sorted(out.glob('students/seed_*/*/complete.json'))
            student_states = sorted(out.glob('students/seed_*/*/recovery/latest.json'))
            ppo = sorted(out.glob('runs/seed_*/*/checkpoints/*.pt'))
            labels = sorted(out.glob('labels/seed_*/*/shards/*.json'))
            state = read(out / 'status.json') if (out / 'status.json').exists() else {}
            records.append({'out': out, 'manifest': m, 'stage': state.get('stage', 'unknown'),
                            'reserved': (out / 'data/complete.json').is_file(),
                            'complete_students': [str(p.parent.relative_to(out / 'students')) for p in complete],
                            'saved_student_states': len(student_states), 'ppo_checkpoints': len(ppo),
                            'label_shards': len(labels),
                            'progress': bool(complete or student_states or ppo or labels)})
        except (ValueError, KeyError, OSError) as error:
            print(f'Skipping invalid study {out.name}: {error}', flush=True)
    return records


def select(records, wanted=None):
    if wanted:
        candidates = [r for r in records if r['out'].name == Path(wanted).name]
    else:
        candidates = [r for r in records if r['reserved'] and r['progress'] and r['stage'] != 'complete']
        if not candidates:
            candidates = [r for r in records if r['reserved'] and r['progress']]
        if not candidates:
            candidates = [r for r in records if r['reserved']]
    if len(candidates) != 1:
        raise ValueError('Cannot choose one saved study unambiguously. Use --study study_<id> '
                         'from the printed inventory; do not reduce FINAL_PROMPTS or delete old data.')
    if not candidates[0]['reserved']:
        raise ValueError('That attempt has no completed data reservation. Select the earlier study with saved work.')
    return candidates[0]


def validate_data(out, manifest):
    done = read(out / 'data/complete.json')
    options = {k: manifest['options'][k] for k in
               ('train_prompts', 'validation_prompts', 'offline_prompts', 'final_prompts', 'data_seed')}
    if done['options'] != options:
        raise ValueError('Saved cohort options do not match the study.')
    for name, expected in done['sha256'].items():
        if Path(name).name != name or sha(out / 'data' / name) != expected:
            raise ValueError('Saved cohort checksum mismatch: ' + name)
    final = read(out / 'data/final.json')
    if len(final) != manifest['options']['final_prompts']:
        raise ValueError('Saved final cohort size does not match the study.')


def source_hashes(package):
    return {p.name: sha(p) for p in sorted(package.glob('*.py'))}


def code_payload(project, out, manifest, archive=None):
    """Return None for identical installed code, or validated original bytes."""
    expected = manifest['source_sha256']
    for name in expected:
        if Path(name).name != name or not name.endswith('.py'):
            raise ValueError('Invalid saved source filename.')
    current = source_hashes(project / 'knn_distillation')
    if current == expected:
        return None
    changed = sorted(n for n in current.keys() | expected.keys() if current.get(n) != expected.get(n))
    print('Installed code differs from this study:', ', '.join(changed), flush=True)
    archive = Path(archive) if archive else out / 'important_outcomes_knn_distillation.zip'
    if not archive.is_file():
        raise ValueError('Exact original source code is required to resume. No saved code archive was found at '
                         + str(archive) + '. Supply the earlier outcomes ZIP using --code-archive PATH. '
                         'No source hashes or checkpoints were altered.')
    prefix = 'code/knn_distillation/'
    payload = {}
    with zipfile.ZipFile(archive) as z:
        for name, expected_hash in expected.items():
            content = z.read(prefix + name)
            if hashlib.sha256(content).hexdigest() != expected_hash:
                raise ValueError('Archived source does not match the selected study: ' + name)
            payload[name] = content
        payload['expected_sources.json'] = z.read(prefix + 'expected_sources.json')
    original = json.loads(payload['expected_sources.json'])
    for name, expected_hash in original.items():
        if Path(name).name != name or sha(project / name) != expected_hash:
            raise ValueError('Original project source changed: ' + name)
    print('Found an exact source copy in:', archive, flush=True)
    return payload


def find_manifest(project, root, identity):
    found = [p.parent for p in (project / root).glob('*/manifest.json')
             if read(p).get('identity') == identity]
    if len(found) != 1:
        raise ValueError(f'Expected one source study {identity} under {root}; found {len(found)}.')
    return found[0]


def source_paths(project, manifest):
    kind = manifest['options']['source_kind']
    if kind == 'refresh34':
        refresh34 = find_manifest(project, 'refresh34_outputs', manifest['source_identity'])
        refresh2_id = read(refresh34 / 'manifest.json')['refresh2_identity']
    elif kind == 'refresh2':
        refresh34 = None
        refresh2_id = manifest['source_identity']
    else:
        raise ValueError('Unsupported source_kind: ' + str(kind))
    refresh2 = find_manifest(project, 'refresh2_outputs', refresh2_id)
    followup = find_manifest(project, 'outputs', read(refresh2 / 'manifest.json')['parent_identity'])
    return followup, refresh2, refresh34


def options_for_resume(project, manifest):
    o = dict(manifest['options'])
    active = project / 'knn_distillation/notebook_settings.json'
    if not active.exists():
        active = project / 'knn_distillation/settings.json'
    current = read(active) if active.exists() else {}
    differences = sorted(k for k in set(o) | set(scientific(current))
                         if o.get(k) != scientific(current).get(k))
    if differences:
        print('Restoring saved settings for:', ', '.join(differences), flush=True)
        for name in differences:
            print(f'  {name}: current={current.get(name)!r}; saved={o.get(name)!r}', flush=True)
    o['allow_downloads'] = current.get('allow_downloads', True)
    o['extra_hf_cache'] = current.get('extra_hf_cache')
    return o


def verify_identity(project, package, m, o, paths):
    """Recompute the unmodified runner's identity before allowing it to launch."""
    current_versions = {name: metadata.version(name) for name in m['runtime_versions']}
    if current_versions != m['runtime_versions']:
        changes = {k: {'saved': v, 'current': current_versions[k]}
                   for k, v in m['runtime_versions'].items() if current_versions[k] != v}
        raise ValueError('Runtime versions changed; use the saved environment: ' + json.dumps(changes))
    sys.path.insert(0, str(project))
    sys.path.insert(0, str(package.parent))
    importlib.invalidate_caches()
    io = importlib.import_module('knn_distillation.io')
    if Path(io.ROOT).resolve() != package.resolve():
        raise ValueError('Unexpected imported package path.')
    io.check_config(o)
    _, _, _, source, parent_config, parents, memories = io.inspect_source(project, o, *paths)
    c = {**parent_config, 'seeds': o['seeds'], 'allow_downloads': o['allow_downloads'],
         'extra_hf_cache': o['extra_hf_cache'], 'eval_seed': 2026091233, 'review_seed': 2026091234,
         'review_pairs_per_stratum': o['review_pairs_per_seed']}
    record = {'protocol': 'Frozen proxy+kNN distillation followed by matched frozen-reward PPO',
              'source_identity': source['identity'], 'options': scientific(o), 'config': scientific(c),
              'runtime_versions': current_versions,
              'parents': {str(s): {k: v for k, v in p.items() if k != 'path'} for s, p in parents.items()},
              'memories': {str(s): sha(p / 'locked_reward.json') for s, p in memories.items()},
              'source_sha256': source_hashes(package)}
    if digest(record) != m['identity']:
        different = sorted(k for k in record if record[k] != m.get(k))
        raise ValueError('Resume would create a different study; stopped before reserving data. '
                         'Mismatched fields: ' + ', '.join(different))


def write_payload(package, payload):
    package.mkdir(parents=True, exist_ok=True)
    for name, content in payload.items():
        path = package / name
        if path.exists() and path.read_bytes() != content:
            raise ValueError('Existing recovery copy differs: ' + str(path))
        path.write_bytes(content)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--project', type=Path, default=Path.cwd())
    parser.add_argument('--study')
    parser.add_argument('--code-archive', type=Path)
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    project = args.project.resolve()
    if Path(__file__).resolve().parent == project / 'knn_distillation':
        raise ValueError('Place this launcher beside ppo_engine.py, outside knn_distillation, '
                         'so adding it does not change the scientific source identity.')
    records = scan(project)
    for r in records:
        print(f'{r["out"].name}: stage={r["stage"]}; seeds={r["manifest"]["options"]["seeds"]}; '
              f'final={r["manifest"]["options"]["final_prompts"]}; reserved={r["reserved"]}; '
              f'completed_students={r["complete_students"]}; student_checkpoints={r["saved_student_states"]}; '
              f'PPO_checkpoints={r["ppo_checkpoints"]}; label_shards={r["label_shards"]}', flush=True)
    if args.list:
        return 0
    r = select(records, args.study)
    out, m = r['out'], r['manifest']
    print('Selected existing study:', out, flush=True)
    validate_data(out, m)
    o = options_for_resume(project, m)
    paths = source_paths(project, m)
    payload = code_payload(project, out, m, args.code_archive)
    with tempfile.TemporaryDirectory(prefix='knn_resume_check_') as tmp:
        package = project / 'knn_distillation'
        if payload is not None:
            package = Path(tmp) / 'knn_distillation'
            write_payload(package, payload)
        verify_identity(project, package, m, o, paths)
    print('Verified: same study ID, saved cohorts, source checkpoints, code, runtime and full seed list.', flush=True)
    if not args.resume:
        print('No experiment was started. Add --resume to continue this saved study.', flush=True)
        return 0
    import fcntl
    with (project / 'outputs/runner.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('Another GPU study is running. Wait for it to pause before resuming.')
        recovery = project / 'knn_distillation_recovery' / out.name
        recovery.mkdir(parents=True, exist_ok=True)
        config = recovery / 'resume_settings.json'
        pending = config.with_suffix('.pending')
        pending.write_text(json.dumps(o, indent=2) + '\n')
        pending.replace(config)
        package = project / 'knn_distillation'
        if payload is not None:
            package = recovery / 'code/knn_distillation'
            write_payload(package, payload)
        if source_hashes(package) != m['source_sha256']:
            raise ValueError('Source changed during preparation.')
    # The original runner acquires the project lock again before any GPU work.
    bootstrap = 'import runpy,sys; sys.path.insert(0,sys.argv.pop(1)); runpy.run_module("knn_distillation.run",run_name="__main__")'
    cmd = [sys.executable, '-u', '-c', bootstrap, str(package.parent), '--project', str(project), '--config', str(config)]
    for flag, path in zip(('--followup', '--refresh2', '--refresh34'), paths):
        if path is not None:
            cmd.extend([flag, str(path)])
    print('Resuming original settings: seeds=', o['seeds'], 'final_prompts=', o['final_prompts'], flush=True)
    print('Runner command:', shlex.join(cmd), flush=True)
    return subprocess.call(cmd, cwd=project)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, OSError, zipfile.BadZipFile, metadata.PackageNotFoundError) as error:
        print('Recovery stopped:', error, file=sys.stderr, flush=True)
        raise SystemExit(2)
