"""Held-out student fidelity, judge agreement and scoring timings."""
from pathlib import Path
import time
import numpy as np
import pandas as pd
from knn_distillation.io import read, write, digest, should_stop, status


def regression_metrics(prediction, target):
    from scipy.stats import spearmanr
    x, y = np.asarray(prediction, float), np.asarray(target, float)
    if x.shape != y.shape or x.ndim != 1 or not len(x) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Invalid reward metric arrays.')
    variable = np.std(x) > 1e-12 and np.std(y) > 1e-12
    return {'n': len(x), 'mse': float(np.mean((x - y) ** 2)), 'mae': float(np.mean(abs(x - y))),
            'pearson': float(np.corrcoef(x, y)[0, 1]) if variable else None,
            'spearman': float(spearmanr(x, y).statistic) if variable else None}


def pair_agreement(frame, prediction, target):
    records = []
    for _, group in frame.groupby('prompt_id'):
        if len(group) != 2:
            raise ValueError('Offline pairwise metrics require two answers per prompt.')
        target_delta = float(group[target].iloc[0] - group[target].iloc[1])
        if abs(target_delta) < 1e-8:
            continue
        pred_delta = float(group[prediction].iloc[0] - group[prediction].iloc[1])
        records.append(.5 if abs(pred_delta) < 1e-8 else float(np.sign(target_delta) == np.sign(pred_delta)))
    return {'non_tied_teacher_pairs': len(records), 'ordering_agreement': float(np.mean(records)) if records else None}


def score_offline(out, seed, students, rows, batch_size):
    from common import StopRequested
    folder = out / 'offline' / f'seed_{seed}'
    identity = digest({'examples': digest(rows), 'students': {k: s.identity for k, s in students.items()}, 'batch': batch_size})
    if (folder / 'complete.json').exists():
        if read(folder / 'complete.json')['identity'] != identity:
            raise ValueError('Offline evaluation dependencies changed.')
        return
    allrows = []
    for start in range(0, len(rows), batch_size):
        if should_stop(out):
            raise StopRequested('Paused before offline student batch.')
        batch = rows[start:start + batch_size]; path = folder / 'shards' / f'{start:06d}.json'
        if path.exists():
            saved = read(path)
            if saved['identity'] != identity or saved['hash'] != digest(saved['rows']):
                raise ValueError('Offline shard changed.')
            result = saved['rows']
        else:
            result = [dict(r) for r in batch]
            for key, student in students.items():
                values = student.score([r['prompt'] for r in batch], [r['answer'] for r in batch])
                z = (values['raw'] - student.calibration['proxy_mean']) / student.calibration['proxy_std']
                for r, value in zip(result, z):
                    r[key + '_z'] = float(value)
            write(path, {'identity': identity, 'rows': result, 'hash': digest(result)})
        allrows.extend(result)
        status(out, 'offline_evaluation', seed=seed, completed=start + len(batch), total=len(rows))
    frame = pd.DataFrame(allrows); frame.to_csv(folder / 'predictions.csv', index=False)
    metrics = []
    for name in ('proxy', 'teacher', *students):
        col = name + '_z'
        for target in ('teacher_z', 'judge_z'):
            metrics.append({'method': name, 'target': target, **regression_metrics(frame[col], frame[target]),
                            **pair_agreement(frame, col, target)})
    slices = []
    boundaries = frame.gap_hat.quantile([.25, .5, .75]).to_numpy()
    bins = np.searchsorted(boundaries, frame.gap_hat.to_numpy(), side='right')
    for key in students:
        for label in range(4):
            subset = frame[bins == label]
            if len(subset):
                slices.append({'method': key, 'predicted_gap_quartile': label + 1,
                               **regression_metrics(subset[key + '_z'], subset.teacher_z)})
    write(folder / 'metrics.json', {'records': metrics, 'teacher_gap_quartiles': slices,
          'checkpoint_selection_used_these_rows': False, 'source_policy_may_have_seen_these_prompts': True})
    write(folder / 'complete.json', {'identity': identity, 'rows': len(frame)})


def latency_benchmark(out, seed, router, rows, o, bundle_paths, memory_path):
    import torch
    folder = out / 'latency' / f'seed_{seed}'
    batch = rows[:o['latency_examples']]
    signature = {'students': router.student_ids, 'memory': router.memory_hash,
                 'examples': digest(batch), 'repeats': o['latency_repeats']}
    if (folder / 'results.json').exists():
        if read(folder / 'results.json')['signature'] != signature:
            raise ValueError('Latency benchmark dependencies changed.')
        return
    prompts, answers = [r['prompt'] for r in batch], [r['answer'] for r in batch]
    records = []
    methods = ['proxy', 'knn', *router.students]
    methods = methods[seed % len(methods):] + methods[:seed % len(methods)]
    for method in methods:
        for _ in range(2):
            router.score(prompts, answers, method)
        times = []; peaks = []; baselines = []
        for _ in range(o['latency_repeats']):
            if torch.cuda.is_available():
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                baselines.append(torch.cuda.memory_allocated())
            start = time.perf_counter()
            router.score(prompts, answers, method)
            if torch.cuda.is_available():
                torch.cuda.synchronize(); peaks.append(torch.cuda.max_memory_allocated())
            times.append(time.perf_counter() - start)
        records.append({'method': method, 'answers_per_timed_call': len(batch),
                        'median_seconds': float(np.median(times)), 'p95_seconds': float(np.quantile(times, .95)),
                        'answers_per_second': len(batch) / float(np.median(times)),
                        'incremental_peak_cuda_bytes': max(p - b for p, b in zip(peaks, baselines)) if peaks else None,
                        'resident_cuda_bytes_before': min(baselines) if baselines else None})
    write(folder / 'results.json', {'signature': signature, 'records': records,
          'teacher_memory_array_bytes': sum(x.nbytes for x in router.memory.values()),
          'teacher_memory_file_bytes': (Path(memory_path) / 'refreshed_memory.npz').stat().st_size,
          'student_adapter_bytes': {key: sum(p.stat().st_size for p in Path(path).glob('*.safetensors')) for key, path in bundle_paths.items()},
          'note': 'Same texts/batch settings, two warmups. All study scorers remain resident. Incremental CUDA peaks are activation overhead, not isolated deployment footprint. Judge is never timed. Compilation/preflight/repeated work excluded.'})
