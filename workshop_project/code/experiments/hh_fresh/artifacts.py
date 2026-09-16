"""New answers, calibration and append-only memories; no legacy result inputs."""
import time
import numpy as np

from common import seed_for, StopRequested
from evaluation import save_npz
from hh_offline.data import vectors
from hh_ridge_ppo.features import load
from hh_ridge_ppo.protocol import read, write, sha, digest, require
from knn_distillation.io import should_stop


def seal(path, value):
    if path.exists():
        require(digest(read(path)) == digest(value), 'Changed dependency: '+str(path))
    else:
        write(path, value)


def collect(out, folder, rows, origins, c, recipe, assets, proxy, judge, identity, cohort):
    """Generate/scalar-grade saved batches with the original actor and scorers.

    origins: (name, policy seed, checkpoint or None, answer repetitions).
    Validation/offline can each combine a base and parent answer per prompt.
    """
    from run_study import load_actor, checkpoint_actor, release
    signature = {'identity': identity, 'prompts': digest(rows), 'cohort': cohort,
                 'origins': [[n, s, ck['sha256'] if ck else 'base', repeats] for n, s, ck, repeats in origins]}
    if (folder / 'complete.json').exists():
        require(read(folder / 'complete.json')['signature'] == signature, 'Generated answer dependencies changed')
        return load(folder)
    seal(folder / 'signature.json', signature)
    records, features, costs = [], [], []
    for origin, seed, checkpoint, repeats in origins:
        actor = None
        try:
            for sample in range(repeats):
                for start in range(0, len(rows), c['generation_batch_size']):
                    if should_stop(out):
                        raise StopRequested('Paused before a generation/scoring batch')
                    batch = rows[start:start+c['generation_batch_size']]
                    shard = folder / 'shards' / f'{origin}_{sample}_{start:06d}.json'
                    npz = shard.with_suffix('.npz')
                    if shard.exists():
                        saved = read(shard)
                        require(saved['signature'] == digest(signature) and saved['input'] == digest(batch) and saved['row_hash'] == digest(saved['rows']) and sha(npz) == saved['features_sha256'], 'Generated scoring shard changed')
                    else:
                        if actor is None:
                            actor = (load_actor(assets['policy'], c, seed) if checkpoint is None else
                                     checkpoint_actor(assets['policy'], c, seed, checkpoint['path'], checkpoint['identity'], checkpoint['branch']))
                        began = time.perf_counter()
                        prompts = [r['prompt'] for r in batch]
                        parts = actor.generate(prompts, seed_for(recipe['data_seed'], cohort, origin, seed, sample, start))
                        answers = [a for p in parts for a in p['answers']]
                        lengths = [int(n) for p in parts for n in p['response_mask'].sum(1)]
                        eos = [bool(v) for p in parts for v in p['ended_eos']]
                        p = proxy.score(prompts, answers, features=True)
                        j = judge.score(prompts, answers)
                        require(not np.asarray(p['truncated']).any() and not np.asarray(j['truncated']).any(), 'Fresh reward input was truncated')
                        scored = [{**row, 'example_id': digest([row['prompt_id'], origin, seed, sample]),
                                   'origin': origin, 'seed': seed, 'answer_sample': sample, 'answer': answers[i],
                                   'proxy_raw': float(p['raw'][i]), 'judge_raw': float(j['raw'][i]),
                                   'proxy_input_tokens': int(p['tokens'][i]), 'judge_input_tokens': int(j['tokens'][i]),
                                   'response_tokens': lengths[i], 'ended_eos': eos[i]} for i, row in enumerate(batch)]
                        save_npz(npz, vectors=vectors(p['features']))
                        saved = {'signature': digest(signature), 'input': digest(batch), 'rows': scored, 'row_hash': digest(scored),
                                 'features_sha256': sha(npz), 'seconds': time.perf_counter()-began,
                                 'proxy_answers': len(scored), 'judge_answers': len(scored)}
                        write(shard, saved)
                        if start % 256 == 0 or start+len(batch) == len(rows):
                            print(f'{cohort} {origin} sample {sample+1}: {start+len(batch)}/{len(rows)}', flush=True)
                    with np.load(npz, allow_pickle=False) as z:
                        features.append(vectors(z['vectors']))
                    records.extend(saved['rows'])
                    costs.append(saved)
        finally:
            if actor is not None:
                del actor
                release()
    write(folder / 'examples.json', records)
    save_npz(folder / 'features.npz', vectors=np.concatenate(features))
    write(folder / 'complete.json', {'signature': signature, 'answers': len(records),
          'seconds': sum(r['seconds'] for r in costs), 'new_judge_answers': sum(r['judge_answers'] for r in costs),
          'proxy_answers': sum(r['proxy_answers'] for r in costs),
          'artifacts': {n: sha(folder / n) for n in ('examples.json', 'features.npz')}})
    return load(folder)


def calibrate(frame, quantile):
    calibration = {'format_version': 1}
    for role in ('proxy', 'judge'):
        raw = frame[role+'_raw'].to_numpy(float)
        require(np.isfinite(raw).all() and np.std(raw) > 1e-12, 'Degenerate calibration scores')
        calibration[role+'_mean'], calibration[role+'_std'] = float(raw.mean()), float(raw.std())
    gap = ((frame.proxy_raw-calibration['proxy_mean'])/calibration['proxy_std'] -
           (frame.judge_raw-calibration['judge_mean'])/calibration['judge_std'])
    calibration['theta'] = float(np.quantile(gap, quantile))
    return calibration


def normalize(frame, calibration):
    f = frame.copy()
    for role in ('proxy', 'judge'):
        f[role+'_z'] = (f[role+'_raw']-calibration[role+'_mean'])/calibration[role+'_std']
    f['gap'] = f.proxy_z-f.judge_z
    f['actual_proxy_judge_gap'] = f.gap
    f['high_gap'] = f.gap > calibration['theta']
    f['group'] = f.conversation_group
    return f


def memory(folder, frame, x, calibration, dependency, previous=None):
    dependency = {**dependency, 'previous_lock': sha(previous/'locked_reward.json') if previous else None}
    marker = folder / 'locked_reward.json'
    if marker.exists():
        lock = read(marker)
        require(lock['dependency'] == dependency and sha(folder/'refreshed_memory.npz') == lock['memory_sha256'], 'Fresh memory dependencies changed')
        return folder
    gap = normalize(frame, calibration).gap.to_numpy(float)
    x = vectors(x)
    require(len(x) == len(gap), 'Memory label/feature alignment differs')
    old_count = 0
    if previous:
        lock = read(previous/'locked_reward.json')
        require(sha(previous/'refreshed_memory.npz') == lock['memory_sha256'] and lock['calibration'] == calibration, 'Previous memory changed')
        with np.load(previous/'refreshed_memory.npz', allow_pickle=False) as z:
            old_count = len(z['gaps'])
            x, gap = np.concatenate([z['vectors'], x]), np.concatenate([z['gaps'], gap])
    save_npz(folder/'refreshed_memory.npz', vectors=x, gaps=gap)
    write(marker, {'dependency': dependency, 'memory_sha256': sha(folder/'refreshed_memory.npz'),
          'calibration': calibration, 'k': 31, 'temperature': .05, 'old_memory_rows': old_count,
          'new_memory_rows': len(frame), 'total_memory_rows': len(gap), 'test_labels_used': False,
          'target': 'actual normalized proxy minus judge; all signed labels retained'})
    return folder
