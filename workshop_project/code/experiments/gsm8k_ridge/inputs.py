"""Read completed controls and cached proxy features without changing their files."""
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np

from gsm8k_experiment.common import digest, read_json, read_jsonl
from gsm8k_experiment.memory import GapMemory, Normalization


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def inspect(source):
    source = Path(source).resolve()
    config = read_json(source/'config.json')
    manifest = read_json(source/'manifest.json')
    identity = manifest['identity']
    require(digest(identity) == manifest['fingerprint'] and identity['config'] == config,
            'Saved configuration or manifest changed')
    resolved = read_json(source/'resolved_assets.json')
    split = read_json(source/'data/splits.json')
    require(resolved == identity['resolved'] and split['fingerprint'] == identity['split_fingerprint'], 'Saved split/model identity differs')
    split_content = {k:v for k,v in split.items() if k not in ('fingerprint','hf_revision')}
    require(digest(split_content) == split['fingerprint'], 'Saved question splits changed')
    seen = set()
    for rows in split['cohorts'].values():
        ids = {r['id'] for r in rows}
        require(len(ids) == len(rows) and not ids & seen, 'Question groups overlap across saved cohorts')
        seen.update(ids)
    controls = {a:read_json(source/'arms'/a/'completed.json') for a in ('proxy','knn_static')}
    updates = controls['proxy']['update']
    require(updates > 0 and all(r['update'] == updates and r['fingerprint'] == manifest['fingerprint'] for r in controls.values()),
            'Need completed proxy and static kNN controls at the same target and identity')
    marker = source/'final_protocol.json'
    kind = 'final' if marker.exists() else 'monitor'
    if marker.exists():
        final = read_json(marker)
        require(final['updates'] == updates and {'proxy','knn_static'} <= set(final['arms']), 'Saved final protocol differs from controls')
    norm = Normalization.load(source/'prepared/normalization.json')
    memory = GapMemory.load(source/'prepared/memory_initial.npz')
    matched = source/'prepared_30b/complete.json'
    if matched.exists():
        indices = read_json(matched).get('shared_memory_indices')
        if indices is not None:
            memory = GapMemory(memory.embeddings[indices], memory.gaps[indices], memory.group_ids[indices],
                               memory.k, memory.temperature, memory.encoder_identity)
    require(set(memory.group_ids) <= {r['id'] for r in split['cohorts']['memory']}, 'Memory contains non-memory questions')
    for step in (1,updates):
        stats = read_json(source/'arms/knn_static/training'/f'step_{step:06d}.json')
        require(stats['memory_examples'] == len(memory.gaps), 'Saved kNN control used a different memory size')
    paths = ['config.json','manifest.json','resolved_assets.json','data/splits.json','prepared/normalization.json',
             'prepared/memory_initial.npz','prepared/selection_raw.jsonl','prepared/calibration_raw.jsonl','prepared/encoder_probe.json','reward_cache.sqlite',
             *[f'arms/{a}/completed.json' for a in controls],
             *[f'arms/knn_static/training/step_{s:06d}.json' for s in (1,updates)]]
    paths += [p for p in ('initial_trainable.pt','final_protocol.json','prepared_30b/complete.json') if (source/p).is_file()]
    for a in ('base','proxy','knn_static'):
        path = f'evaluations/{kind}/{a}/step_{0 if a == "base" else updates:06d}'
        meta = read_json(source/path/'metrics.json')
        require(meta['arm'] == a and meta['update'] == (0 if a == 'base' else updates) and meta['cohort'] == kind,
                'Saved evaluation metadata differs')
        paths.extend([path+'/metrics.json',path+'/responses.jsonl'])
    require((source/'reward_cache.sqlite').is_file(), 'Use the full GSM8K output, including reward_cache.sqlite; the small results export omits it')
    return {'source':source,'config':config,'resolved':resolved,'split':split,'manifest':manifest,'norm':norm,
            'memory':memory,'updates':updates,'kind':kind,'seed':config['seed'],
            'files':{p:sha(source/p) for p in sorted(set(paths))}}


def features(rows, identity, databases):
    """Exact cache keys and score checks prevent mismatched answer embeddings."""
    connections = [sqlite3.connect(Path(p).resolve().as_uri()+'?mode=ro', uri=True) for p in databases if Path(p).is_file()]
    try:
        result = []
        for row in rows:
            key = digest([identity,row['question'],row['reference'],row['response']])
            hit = next((value for db in connections if (value := db.execute('SELECT result,embedding FROM scores WHERE key=?',(key,)).fetchone()) is not None),None)
            require(hit is not None and hit[1] is not None, 'Missing cached proxy embedding for '+row['id']+'; use the original complete reward cache')
            require(json.loads(hit[0])['score'] == row['proxy_score'], 'Cached proxy score differs from saved answer')
            result.append(np.frombuffer(hit[1],dtype=np.float32).copy())
        x = np.stack(result)
        require(np.isfinite(x).all() and np.allclose(np.linalg.norm(x,axis=1),1,atol=1e-4), 'Invalid proxy feature vectors')
        return x
    finally:
        for db in connections:
            db.close()


def selection(plan):
    rows = read_jsonl(plan['source']/'prepared/selection_raw.jsonl')
    expected = {r['id'] for r in plan['split']['cohorts']['selection']}
    require(rows and {r['id'] for r in rows} <= expected and not expected & set(plan['memory'].group_ids), 'Selection/memory split mismatch')
    valid = [r for r in rows if all(r.get(k) is not None and np.isfinite(r[k]) for k in ('proxy_score','judge_score'))]
    require(valid, 'No valid validation grades; ridge cannot be selected')
    x = features(valid,plan['memory'].encoder_identity,[plan['source']/'reward_cache.sqlite'])
    y = plan['norm'].gap([r['proxy_score'] for r in valid],[r['judge_score'] for r in valid])
    return valid,x,y,{'answers':len(rows),'valid_pairs':len(valid),'excluded':len(rows)-len(valid)}
