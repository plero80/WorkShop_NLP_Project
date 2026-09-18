"""Atomic artifacts and a protocol identity shared by all stages."""
from pathlib import Path
from importlib import metadata
import hashlib
import json
import os
import random
import time

ROOT = Path(__file__).resolve().parent


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical_hash(obj):
    return digest(json.dumps(obj, sort_keys=True, ensure_ascii=False, allow_nan=False).encode())


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.pending')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def seed_for(*parts):
    return int(digest('|'.join(map(str, parts)).encode())[:8], 16) % (2**31 - 1)


def set_seed(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def runtime_versions():
    names = ['torch', 'transformers', 'peft', 'numpy', 'pandas', 'scikit-learn', 'huggingface_hub']
    return {n: metadata.version(n) for n in names}


def validate_config(c):
    for key in ['updates','rollout_batch_size','generation_batch_size','reward_batch_size','mini_batch_size',
                'micro_batch_size','ppo_epochs','checkpoint_every','max_new_tokens','review_pairs_per_stratum']:
        if type(c[key]) is not int or c[key] < 1:
            raise ValueError('Positive integer required: '+key)
    if c['seeds'] != [42,43,44] or c['branches'] != ['raw','knn_signed','iterative_knn','iterative_capped']:
        raise ValueError('Keep the packaged seeds and raw/static/iterative/capped branches.')
    if not 0 < c['round1_updates'] < c['updates']:
        raise ValueError('Round one must end strictly before the final update.')
    if c['updates']*c['rollout_batch_size'] > 6400:
        raise ValueError('The packaged matched schedule covers 6400 responses per seed.')
    if c['rollout_batch_size']%c['mini_batch_size'] or c['mini_batch_size']%c['micro_batch_size']:
        raise ValueError('Rollout/minibatch/microbatch sizes must divide exactly.')
    if c['max_new_tokens'] != 256 or c['recheck_caps'] != [128,256]:
        raise ValueError('This follow-up pairs 128 and 256 evaluation; new PPO uses 256.')
    if c['reward_max_tokens'] < 1024:
        raise ValueError('Full scoring requires a context guard of at least 1024 tokens.')
    if not 0 <= c['gae_lambda'] <= 1 or not 0 <= c['gamma'] <= 1:
        raise ValueError('Invalid GAE parameters.')
    if any(x < 0 or x > 1 for x in c['selection_alphas']) or any(x < 0 for x in c['selection_bonus_caps']):
        raise ValueError('Invalid shrinkage/bonus-cap search.')
    if c['review_pairs_per_stratum']*12 > 2000:
        raise ValueError('Review needs distinct prompts across 12 strata.')


def experiment_manifest(c):
    # Execution controls do not change the statistical experiment.
    protocol = {k: v for k, v in c.items() if k not in
                ('allow_downloads', 'extra_hf_cache', 'max_wall_hours', 'run_new_ppo')}
    sources = {p.name: file_hash(p) for p in sorted(ROOT.glob('*.py'))
               if not p.name.startswith('test_') and p.name != 'setup_environment.py'}
    inputs = {str(p.relative_to(ROOT)): file_hash(p) for p in sorted((ROOT / 'inputs').rglob('*')) if p.is_file()}
    record = {'config': protocol, 'source_sha256': sources, 'input_sha256': inputs,
              'runtime_versions': runtime_versions(), 'protocol_version': 1}
    return {**record, 'identity': canonical_hash(record)}


def output_root(c):
    manifest = experiment_manifest(c)
    output = ROOT / 'outputs' / ('study_' + manifest['identity'][:16])
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'manifest.json'
    if path.exists() and read_json(path) != manifest:
        raise ValueError('Experiment identity collision or changed manifest.')
    write_json(path, manifest)
    write_json(ROOT / 'outputs/latest.json', {'relative_output': str(output.relative_to(ROOT))})
    return output


def status(output, stage, **details):
    write_json(output / 'status.json', {'stage': stage, 'updated_unix': time.time(), **details})


class StopRequested(Exception):
    pass
