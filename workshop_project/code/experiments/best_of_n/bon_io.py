"""Atomic, checksummed experiment state; CPU-only imports."""
import hashlib, json, os
from pathlib import Path

HERE = Path(__file__).resolve().parent

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()

def write(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.pending')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    tmp.replace(path)

def seal(path, obj):
    if Path(path).exists() and read(path) != obj:
        raise ValueError('Changed experiment dependency: ' + str(path))
    write(path, obj)

def save_shard(path, signature, rows):
    write(path, {'signature': signature, 'rows_sha256': digest(rows), 'rows': rows})

def load_shard(path, signature):
    obj = read(path)
    if obj['signature'] != signature or obj['rows_sha256'] != digest(obj['rows']):
        raise ValueError('Changed or corrupt shard: ' + str(path))
    return obj['rows']

def locate_oracle(project, supplied=None):
    if supplied:
        root = Path(supplied).resolve()
    else:
        latest = read(project / 'next_studies_outputs/latest.json')
        suite = Path(latest['suite'])
        if not suite.exists(): suite = project / 'next_studies_outputs' / suite.name
        root = suite / 'oracle'
    if read(root / 'status.json')['stage'] != 'complete':
        raise ValueError('The selected teacher-access comparison must be complete.')
    return root

def checkpoint(root, policy, seed):
    if policy == 'initial':
        path = root / 'initializations' / f'seed_{seed}' / 'checkpoint_000000.pt'
    else:
        end = read(root.parent / 'manifest.json')['options']['oracle_updates']
        path = root / 'runs' / f'seed_{seed}' / policy / 'checkpoints' / f'checkpoint_{end:06d}.pt'
    if not path.is_file():
        raise FileNotFoundError('Full RunPod checkpoint required (outcome ZIPs omit it): ' + str(path))
    meta = read(path.with_suffix('.json'))
    if sha(path) != meta['sha256'] or meta['seed'] != seed or meta['branch'] != policy:
        raise ValueError('Checkpoint provenance mismatch: ' + str(path))
    if meta['identity'] != read(root / 'manifest.json')['identity']:
        raise ValueError('Checkpoint belongs to a different oracle experiment.')
    if policy != 'initial':
        lock = read(root / 'final_lock.json')['endpoint_hashes']
        if lock[f'{seed}/{policy}'] != meta['sha256']: raise ValueError('Not the locked final endpoint.')
    return {**meta, 'path': str(path)}

def validate_settings(s):
    for key in ('development_prompts', 'confirmation_prompts', 'candidates', 'candidate_batch_size', 'answer_cap', 'bootstrap_draws'):
        if type(s[key]) is not int or s[key] < 1: raise ValueError('Positive integer required: ' + key)
    if s['candidates'] != max(s['pool_sizes']) or s['pool_sizes'] != sorted(set(s['pool_sizes'])):
        raise ValueError('Pool sizes must be sorted, unique, and end at candidates.')
    if any(type(n) is not int or n < 1 for n in s['pool_sizes']) or 1 not in s['pool_sizes']:
        raise ValueError('Positive pool sizes including N=1 required.')
    if s['primary_n'] not in s['pool_sizes']: raise ValueError('Primary N must be evaluated.')
    if not s['policies'] or len(set(s['policies'])) != len(s['policies']) or any(p not in ('initial','proxy','knn','judge') for p in s['policies']):
        raise ValueError('Unknown or duplicate generator policy.')
    if s['primary_policy'] not in s['policies']: raise ValueError('Primary generator missing.')
    if s['dataset_split'] not in ('train', 'test'): raise ValueError('Use an explicit HH dataset split.')
    if s['policy_seed'] not in (42,43,44): raise ValueError('No such original training seed.')
