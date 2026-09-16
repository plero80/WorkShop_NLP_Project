"""Separate predictor fidelity from PPO outcomes, with paired prompt intervals."""
from pathlib import Path
import re
import numpy as np
import pandas as pd

from hh_offline.metrics import summarize, paired_intervals
from knn_distillation.maths import predict_memory
from .protocol import read, write, require, load_evaluation

# Descriptive language diagnostic, not a human rating of inappropriate refusal.
REFUSAL = re.compile(r"\b(?:i (?:am sorry|apologize|cannot|can't|won't|am unable)|i'm (?:sorry|unable)|"
                     r"i (?:can not|will not) (?:help|assist|provide)|as an ai)\b", re.I)


def annotate(frame, cap=256):
    frame = frame.copy()
    frame['refusal_diagnostic'] = frame.answer.map(lambda a: bool(REFUSAL.search(str(a)[:300])))
    frame['completion_eos'] = frame.ended_eos.astype(bool)
    frame['length_capped'] = (~frame.completion_eos) & (frame.response_tokens >= cap)
    return frame


def policy_rows(frames):
    rows = []
    for (seed, branch), raw in frames.items():
        f = annotate(raw)
        rows.append({'seed': seed, 'branch': branch, 'answers': len(f), 'mean_judge_z': float(f.judge_z.mean()),
                     'mean_proxy_z': float(f.proxy_z.mean()), 'high_gap_rate': float(f.high_gap.mean()),
                     'mean_response_tokens': float(f.response_tokens.mean()), 'completion_eos_rate': float(f.completion_eos.mean()),
                     'length_capped_rate': float(f.length_capped.mean()), 'refusal_diagnostic_rate': float(f.refusal_diagnostic.mean())})
    return pd.DataFrame(rows)


def aggregate(table, keys):
    metrics = [c for c in table.select_dtypes(include='number').columns if c not in [*keys, 'seed']]
    means = table.groupby(keys)[metrics].mean().add_suffix('_mean')
    stds = table.groupby(keys)[metrics].std(ddof=1).add_suffix('_seed_sd')
    result = means.join(stds)
    result['seeds'] = table.groupby(keys).seed.nunique()
    return result.reset_index()


def policy_intervals(frames, draws, seed):
    metrics = {'judge_z': 'judge_delta', 'high_gap': 'high_gap_delta', 'response_tokens': 'length_delta',
               'completion_eos': 'completion_delta', 'refusal_diagnostic': 'refusal_diagnostic_delta'}
    rng = np.random.default_rng(seed)
    records, paired = [], []
    for alternative, reference in [('knn', 'proxy'), ('ridge', 'proxy'), ('ridge', 'knn')]:
        seeds = sorted(s for s, b in frames if b == alternative and (s, reference) in frames)
        differences = []
        for s in seeds:
            a, b = (annotate(frames[(s, arm)]).set_index('prompt_id') for arm in (reference, alternative))
            require(set(a.index) == set(b.index) and a.index.is_unique, 'Policy comparison prompt mismatch')
            b = b.loc[a.index]
            require(a.prompt.equals(b.prompt), 'Policy comparison context mismatch')
            delta = pd.DataFrame({name: b[key].to_numpy(float)-a[key].to_numpy(float) for key, name in metrics.items()}, index=a.index)
            delta['seed'], delta['comparison'] = s, alternative + '_minus_' + reference
            differences.append(delta)
        if not differences:
            continue
        combined = pd.concat(differences)
        paired.append(combined.reset_index())
        require((combined.groupby(level=0).size() == len(seeds)).all(), 'Seeds have different prompt sets')
        # Retain cross-seed dependence: average seeds within prompt, then resample prompts.
        scopes = [(str(s), delta[list(metrics.values())]) for s, delta in zip(seeds, differences)]
        scopes.append(('seed_mean', combined.groupby(level=0)[list(metrics.values())].mean()))
        for scope, values in scopes:
            x = values.to_numpy(float)
            boot = np.array([x[rng.integers(len(x), size=len(x))].mean(axis=0) for _ in range(draws)])
            for i, metric in enumerate(values.columns):
                lo, hi = np.quantile(boot[:, i], [.025, .975])
                records.append({'comparison': alternative + '_minus_' + reference, 'scope': scope, 'metric': metric,
                                'mean': float(x[:, i].mean()), 'low': float(lo), 'high': float(hi), 'seeds': seeds})
    return records, pd.concat(paired, ignore_index=True) if paired else pd.DataFrame()


def fidelity(frame, x, memory, coef, intercept, calibration, seed, cohort, threads):
    f = frame.copy()
    if 'group' not in f:
        from knn_distillation.data import group
        f['group'] = f.prompt.map(group)
    f['gap'] = f.proxy_z - f.judge_z
    prediction = {'proxy': np.zeros(len(f)), 'knn': predict_memory(x, memory, threads=threads)[0], 'ridge': x @ coef + intercept}
    rows, exports = [], []
    for method, gap in prediction.items():
        rows.append({'seed': seed, 'cohort': cohort, 'predictor': method, **summarize(f, gap, calibration['theta'])})
        record = f[[k for k in ['prompt_id', 'prompt', 'answer', 'origin', 'branch', 'example_id'] if k in f]].copy()
        record['seed'], record['cohort'], record['predictor'] = seed, cohort, method
        record['proxy_z'], record['judge_z'] = f.proxy_z.to_numpy(), f.judge_z.to_numpy()
        record['proxy_raw'] = record.proxy_z * calibration['proxy_std'] + calibration['proxy_mean']
        record['judge_raw'] = record.judge_z * calibration['judge_std'] + calibration['judge_mean']
        record['actual_gap'], record['predicted_gap'] = f.gap.to_numpy(), gap
        record['corrected_reward'] = record.proxy_z - gap
        record['high_gap'] = f.gap.to_numpy() > calibration['theta']
        exports.append(record)
    return rows, pd.concat(exports, ignore_index=True), prediction


def controls(plan):
    return {(seed, branch): load_evaluation(plan['source'] / 'evaluations/final' / f'seed_{seed}' / branch / 'update_000400')[0]
            for seed in plan['recipe']['seeds'] for branch in ('proxy', 'knn')}


def markdown(table):
    def value(v):
        if isinstance(v, (float, np.floating)):
            return f'{v:.4f}' if np.isfinite(v) else 'undefined'
        return str(v)
    return '\n'.join(['| ' + ' | '.join(table.columns) + ' |', '| ' + ' | '.join(['---']*len(table.columns)) + ' |'] +
                     ['| ' + ' | '.join(value(v) for v in row) + ' |' for row in table.itertuples(index=False, name=None)])


def save_policy_report(plan, frames, destination, complete):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    by_seed = policy_rows(frames)
    by_seed.to_csv(destination / 'policy_by_seed.csv', index=False)
    means = aggregate(by_seed, ['branch'])
    means.to_csv(destination / 'policy_seed_summary.csv', index=False)
    ci, pairs = policy_intervals(frames, plan['recipe']['bootstrap_samples'], plan['recipe']['bootstrap_seed'])
    write(destination / 'policy_paired_intervals.json', {'records': ci, 'scope': 'Paired prompt bootstrap conditional on these three trained seeds; seed-mean intervals average seeds within prompt first. No multiplicity adjustment.'})
    pairs.to_csv(destination / 'policy_paired_answers.csv', index=False)
    status = 'Complete three-arm evaluation.' if complete else '**RIDGE PPO NOT RUN. These are the verified existing controls only.**'
    text = '# Matched HH-RLHF M2 continuation\n\n' + status + '\n\n'
    text += 'Start: update 300; endpoint: update 400. Seeds: 42, 43, 44. Shared final prompts: 512.\n\n'
    text += markdown(by_seed) + '\n\nSeed means and sample standard deviations are in `policy_seed_summary.csv`.\n\n'
    text += 'Completion is the EOS fraction, not a semantic completeness rating. Refusal is a fixed phrase heuristic over the first 300 answer characters, not a measure of inappropriate refusals. Inspect answers or complete a blinded human review before making those claims.\n\n'
    text += 'Controls were checked for parent checkpoint, policy/value/optimizer/RNG fingerprint, exact training prompt schedule, update count, source code, scoring protocol, and evaluation prompts. The ridge arm uses the unchanged PPO trainer.\n'
    (destination / 'policy_report.md').write_text(text, encoding='utf-8')


def final_report(plan, out, bundles, recovered):
    from .features import load
    from .protocol import sha
    from hh_offline.data import evaluation as load_refresh
    dest = out / 'reports'
    frames = controls(plan)
    metric_rows, predictions, fidelity_ci = [], [], []
    for seed in plan['recipe']['seeds']:
        bundle = bundles[seed]
        with np.load(plan['refresh'] / 'memories' / f'seed_{seed}' / 'refreshed_memory.npz', allow_pickle=False) as z:
            memory = {k: z[k] for k in ('vectors', 'gaps')}
        targets = [('offline', *load(recovered[seed]['offline']))]
        for branch in ('proxy', 'knn'):
            targets.append(('post_ppo_' + branch, *load(recovered[seed][branch])))
        folder = out / 'evaluations/final' / f'seed_{seed}/ridge/update_000400'
        frame, done = load_evaluation(folder)
        frames[(seed, 'ridge')] = frame
        require(sha(folder / 'features.npz') == done['features_sha256'], 'Ridge evaluation features changed')
        with np.load(folder / 'features.npz', allow_pickle=False) as z:
            targets.append(('post_ppo_ridge', frame, z['vectors']))
        # Restore the available second-refresh feature/prediction files, no policy training.
        for branch in ('parent_pi2', 'static_M1', 'refresh_M2'):
            folder = plan['refresh'] / 'evaluations/final3' / f'seed_{seed}' / branch
            if folder.exists():
                targets.append(('second_refresh_' + branch, *load_refresh([folder], plan['calibration'])))
        for cohort, f, x in targets:
            rows, export, preds = fidelity(f, x, memory, bundle['coef'], bundle['intercept'], plan['calibration'], seed, cohort, plan['recipe']['cpu_threads'])
            metric_rows.extend(rows)
            predictions.append(export)
            if cohort == 'offline':
                f = f.copy()
                f['gap'] = f.proxy_z - f.judge_z
                fidelity_ci.append({'seed': seed, 'cohort': cohort, **paired_intervals(f, preds['knn'], preds['ridge'], plan['calibration']['theta'], plan['recipe']['bootstrap_samples'], plan['recipe']['bootstrap_seed'])})
    save_policy_report(plan, frames, dest, complete=True)
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(dest / 'predictor_by_seed.csv', index=False)
    aggregate(metrics, ['cohort', 'predictor']).to_csv(dest / 'predictor_seed_summary.csv', index=False)
    pd.concat(predictions, ignore_index=True).to_csv(dest / 'all_predictions.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
    write(dest / 'predictor_paired_intervals.json', {'records': fidelity_ci, 'scope': 'Conversation bootstrap conditional on fitted predictors; per-seed intervals. Offline responses excluded from alpha selection.'})
    cols = ['seed', 'predictor', 'gap_mse', 'gap_mae', 'gap_r2', 'gap_pearson', 'gap_spearman', 'high_gap_auroc', 'high_gap_ap']
    (dest / 'predictor_report.md').write_text('# M2 gap prediction on identical held-out answers\n\n' + markdown(metrics[metrics.cohort == 'offline'][cols]) +
        '\n\nEvery predictor sees the same 800 answers per seed (400 prompts, base and parent answers). Alpha was selected on separate validation answers. M2 contains 10,038 actual gap labels per seed.\n\n'
        'Post-PPO and second-refresh transfer metrics are in the complete CSV tables; the second-refresh table evaluates each predictor on the saved answers, not the reward actually used to train every policy.\n', encoding='utf-8')
    costs = []
    for seed in plan['recipe']['seeds']:
        for cohort, folder in recovered[seed].items():
            done = read(folder / 'complete.json')
            costs.append({'seed': seed, 'stage': 'recover_' + cohort, **{k: done[k] for k in ('answers', 'new_judge_answers', 'proxy_answers', 'seconds')}})
        costs.append({'seed': seed, 'stage': 'ridge_fit', 'seconds': bundles[seed]['metadata']['fit_seconds'], 'new_judge_answers': 0})
        history = read(out / 'runs' / f'seed_{seed}/ridge/history.json')
        tail = [h for h in history if h.get('segment_start') == 300 and h.get('reward_source') == 'ridge']
        require(len(tail) == 100 and sum(h['model_calls']['teacher_answers'] for h in tail) == 0, 'Incomplete ridge continuation or unexpected PPO judge calls')
        costs.append({'seed': seed, 'stage': 'ridge_PPO', 'seconds': sum(h['seconds'] for h in tail), 'new_judge_answers': 0, 'proxy_answers': sum(h['model_calls']['proxy_answers'] for h in tail)})
        for cohort in ('monitor', 'final'):
            for p in (out / 'evaluations' / cohort / f'seed_{seed}').glob('*/update_*/complete.json'):
                done = read(p)
                costs.append({'seed': seed, 'stage': cohort + '_' + str(done['signature']['update']), 'answers': done['rows'], 'new_judge_answers': done['rows'], 'proxy_answers': done['rows'], 'seconds': done['last_invocation_seconds']})
    pd.DataFrame(costs).to_csv(dest / 'new_work_costs.csv', index=False)
    write(dest / 'label_budget.json', {'memory': plan['audit']['memory_budgets'], 'shared_original_labels': 7990,
          'label_rows_counting_common_original_once': 7990+3*2048, 'ridge_validation_new_judge_answers': 2400,
          'fitting_labels_identical_for_knn_and_ridge': True, 'validation_is_not_added_to_memory': True,
          'note': 'Historical shared normalization/calibration labels are additional common costs. Memory labels are reused. Judge labels for ridge validation are an additional selection cost, not hidden inside the fitting budget. Timing across different GPU models is descriptive, not a matched speed comparison. Evaluation seconds cover the last invocation when resumed; failed/discarded work and model loading are excluded.'})
    import review
    original_root = review.ROOT
    try:
        review.ROOT = plan['project'] / 'code/templates'
        review_frames = {('raw' if branch == 'knn' else branch, seed, 256): frame for (seed, branch), frame in frames.items()}
        folder = review.make_pack(out, 'ridge_m2_blinded', review_frames, ['proxy', 'ridge'], plan['config'])
        key = read(folder / 'private/key.json')
        key['reference_method'] = 'knn'
        write(folder / 'private/key.json', key)
    finally:
        review.ROOT = original_root
