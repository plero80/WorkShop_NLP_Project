"""Resumable pseudo-labels from a frozen proxy+kNN teacher."""
from pathlib import Path
from knn_distillation.io import read, write, digest, should_stop, status


def load_labels(folder):
    folder = Path(folder)
    done, rows = read(folder / 'complete.json'), read(folder / 'examples.json')
    if done['rows_hash'] != digest(rows) or len(rows) != done['rows']:
        raise ValueError('Distillation examples changed.')
    if len({r['example_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate distillation example IDs.')
    return rows


def generate_labels(out, c, o, assets, reward, prompts, seed, parent, identity, cohort, include_judge):
    from run_study import load_actor, checkpoint_actor, release
    from common import seed_for, StopRequested
    folder = out / 'labels' / f'seed_{seed}' / cohort
    signature = {'study': identity, 'seed': seed, 'parent': parent['sha256'], 'memory': reward.memory_hash,
                 'prompts': digest(prompts), 'cohort': cohort, 'origins': ['base', 'parent'],
                 'judge_labels': include_judge, 'cap': c['max_new_tokens'], 'batch': c['generation_batch_size']}
    sid = digest(signature)
    if (folder / 'complete.json').exists():
        if read(folder / 'complete.json')['identity'] != sid:
            raise ValueError('Pseudo-label dependencies changed.')
        load_labels(folder)
        return folder
    rows, costs = [], []
    for origin in ('base', 'parent'):
        actor = None
        try:
            for start in range(0, len(prompts), c['generation_batch_size']):
                if should_stop(out):
                    raise StopRequested('Paused before pseudo-label batch.')
                batch = prompts[start:start + c['generation_batch_size']]
                path = folder / 'shards' / f'{origin}_{start:06d}.json'
                if path.exists():
                    shard = read(path)
                    if shard['identity'] != sid or shard['hash'] != digest(shard['rows']):
                        raise ValueError('Pseudo-label shard changed.')
                    if [r['prompt_id'] for r in shard['rows']] != [r['prompt_id'] for r in batch]:
                        raise ValueError('Pseudo-label prompt alignment changed.')
                else:
                    if actor is None:
                        actor = (load_actor(assets['policy'], c, seed) if origin == 'base' else
                                 checkpoint_actor(assets['policy'], c, seed, parent['path'], parent['identity'], parent['branch']))
                    parts = actor.generate([r['prompt'] for r in batch], seed_for(o['data_seed'], cohort, origin, start))
                    answers = [a for p in parts for a in p['answers']]
                    reward.reset_cost()
                    target = reward.targets([r['prompt'] for r in batch], answers, include_judge)
                    records = []
                    for i, row in enumerate(batch):
                        record = {**row, 'example_id': digest([row['prompt_id'], origin])[:32], 'origin': origin,
                                  'answer': answers[i], 'teacher_z': float(target['teacher_z'][i]),
                                  'proxy_z': float(target['proxy_z'][i]), 'gap_hat': float(target['gap_hat'][i]),
                                  'proxy_input_tokens': int(target['tokens'][i])}
                        if include_judge:
                            record['judge_z'] = float(target['judge_z'][i])
                        records.append(record)
                    if not include_judge and reward.cost['teacher_answers']:
                        raise RuntimeError('Main distillation labels unexpectedly called the large judge.')
                    shard = {'identity': sid, 'rows': records, 'hash': digest(records), 'model_calls': reward.cost.copy()}
                    write(path, shard)
                rows.extend(shard['rows']); costs.append(shard['model_calls'])
                status(out, 'labeling', seed=seed, cohort=cohort, origin=origin, completed=start + len(batch), total=len(prompts))
                print(f'Labels seed={seed} {cohort}/{origin}: {start + len(batch)}/{len(prompts)}', flush=True)
        finally:
            if actor is not None:
                del actor
                release()
    write(folder / 'examples.json', rows)
    write(folder / 'complete.json', {'identity': sid, 'signature': signature, 'rows_hash': digest(rows), 'rows': len(rows),
                                    'costs': {key: sum(c.get(key, 0) for c in costs) for key in costs[0]}})
    return folder
