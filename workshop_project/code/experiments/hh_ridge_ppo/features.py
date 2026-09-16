"""Recover features and missing judge labels for saved answers, without regeneration."""
from pathlib import Path
import time
import numpy as np
import pandas as pd

from evaluation import save_npz
from hh_offline.data import vectors
from knn_distillation.data import group
from knn_distillation.io import should_stop
from .protocol import read, write, sha, digest, require


def load(folder):
    folder = Path(folder)
    done = read(folder / 'complete.json')
    for name, expected in done['artifacts'].items():
        require(sha(folder / name) == expected, 'Recovered scoring artifact changed: ' + name)
    rows = read(folder / 'examples.json')
    with np.load(folder / 'features.npz', allow_pickle=False) as z:
        x = vectors(z['vectors'])
    require(len(rows) == len(x) == done['answers'], 'Recovered feature alignment changed')
    return pd.DataFrame(rows), x


def recover(folder, rows, router, identity, batch_size, stop_out):
    """Reuse recorded normalized scores; validate the recovered proxy against them."""
    from common import StopRequested
    folder = Path(folder)
    signature = {'experiment': identity, 'answers': digest(rows), 'batch_size': batch_size,
                 'calibration': router.calibration, 'protocol': 'saved_answers_full_context_v1'}
    if (folder / 'complete.json').exists():
        require(read(folder / 'complete.json')['signature'] == signature, 'Recovery dependencies changed')
        return load(folder)
    if (folder / 'signature.json').exists():
        require(read(folder / 'signature.json') == signature, 'Recovery dependencies changed')
    write(folder / 'signature.json', signature)
    output, features, costs = [], [], []
    for start in range(0, len(rows), batch_size):
        if should_stop(stop_out):
            raise StopRequested('Paused before saved-answer scoring batch')
        batch = rows[start:start+batch_size]
        shard = folder / 'shards' / f'{start:06d}.json'
        npz = shard.with_suffix('.npz')
        if shard.exists():
            saved = read(shard)
            require(saved['signature'] == digest(signature) and saved['input'] == digest(batch) and
                    saved['rows_hash'] == digest(saved['rows']) and sha(npz) == saved['features_sha256'], 'Recovered scoring shard changed')
        else:
            began = time.perf_counter()
            prompts, answers = [r['prompt'] for r in batch], [r['answer'] for r in batch]
            router.reset_cost()
            p = router.proxy_values(prompts, answers, features=True)
            require(not np.asarray(p['truncated']).any(), 'Recovered proxy input was truncated')
            cal = router.calibration
            zp = (p['raw'] - cal['proxy_mean']) / cal['proxy_std']
            expected = np.asarray([r['proxy_z'] for r in batch])
            error = abs(zp - expected) * cal['proxy_std']
            require(error.mean() <= .03 and error.max() <= .3, 'Recovered proxy differs from saved scores; check model revision, precision and formatting')
            missing = [i for i, r in enumerate(batch) if 'judge_z' not in r]
            extra = {}
            if missing:
                j = router.judge_values([prompts[i] for i in missing], [answers[i] for i in missing])
                require(not np.asarray(j['truncated']).any(), 'Recovered judge input was truncated')
                extra = {i: float((raw-cal['judge_mean'])/cal['judge_std']) for i, raw in zip(missing, j['raw'])}
            records = []
            for i, row in enumerate(batch):
                zj = float(row['judge_z']) if 'judge_z' in row else extra[i]
                records.append({**row, 'judge_z': zj, 'group': group(row['prompt']),
                                'gap': float(expected[i] - zj), 'recovered_proxy_z': float(zp[i]),
                                'judge_label_reused': i not in extra})
            save_npz(npz, vectors=vectors(p['features']))
            saved = {'signature': digest(signature), 'input': digest(batch), 'rows': records, 'rows_hash': digest(records),
                     'features_sha256': sha(npz), 'cost': dict(router.cost), 'seconds': time.perf_counter()-began}
            write(shard, saved)
            print(f'Recover {folder.name}: {start+len(batch)}/{len(rows)} saved answers', flush=True)
        with np.load(npz, allow_pickle=False) as z:
            features.append(vectors(z['vectors']))
        output.extend(saved['rows'])
        costs.append(saved)
    write(folder / 'examples.json', output)
    save_npz(folder / 'features.npz', vectors=np.concatenate(features))
    write(folder / 'complete.json', {'signature': signature, 'answers': len(output),
          'new_judge_answers': sum(c['cost']['teacher_answers'] for c in costs),
          'proxy_answers': sum(c['cost']['proxy_answers'] for c in costs),
          'seconds': sum(c['seconds'] for c in costs),
          'artifacts': {n: sha(folder / n) for n in ('examples.json', 'features.npz')}})
    return load(folder)
