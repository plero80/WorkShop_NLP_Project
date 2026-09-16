"""Audit or run one new ridge PPO arm against verified saved HH controls."""
import argparse
from importlib import metadata
import json
import os
from pathlib import Path
import signal
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from evaluation import save_npz
from hh_offline.run import fit
from knn_distillation.data import schedule
from .protocol import resolve, audit, read, write, sha, digest, require, labels


def identify(plan):
    project = plan['project']
    sources = [*Path(__file__).parent.glob('*.py'), *[project / 'code/experiments/hh_offline' / n for n in ('data.py', 'metrics.py', 'run.py')],
               *[project / 'code/experiments/knn_distillation' / n for n in plan['manifest']['source_sha256']],
               *(project / 'code/core').glob('*.py'), project / 'hh_ridge_ppo.py']
    recipe = {k: v for k, v in plan['recipe'].items() if k not in ('allow_downloads', 'extra_hf_cache', 'output')}
    # Use logical root names, so the organized layout and original runtime identify the same inputs.
    def logical(p):
        p = Path(p)
        for name in ('source', 'refresh', 'followup', 'inputs'):
            if p.is_relative_to(plan[name]):
                return name + '/' + p.relative_to(plan[name]).as_posix()
        raise ValueError('Unknown input root')
    record = {'protocol': 'hh_m2_ridge_ppo_v1', 'recipe': recipe, 'source_study': plan['manifest']['identity'],
              'input_sha256': {logical(p): h for p, h in plan['audit']['input_sha256'].items()},
              'source_sha256': {p.relative_to(project).as_posix(): sha(p) for p in sorted(set(sources))},
              'analysis_versions': {n: metadata.version(n) for n in ('numpy', 'pandas', 'scipy', 'scikit-learn', 'threadpoolctl')},
              'primary_predictor_cohort': 'distill_offline', 'primary_policy_comparison': 'ridge minus knn at update 400',
              'retrospective_addition': True, 'new_PPO_arms': ['ridge']}
    return digest(record), record


def fit_bundle(plan, out, seed, validation, val_x, identity):
    """Only M2 and validation enter fitting. Neither offline nor final data are passed."""
    folder = out / 'models' / f'seed_{seed}'
    dependencies = {'experiment': identity, 'memory_lock': plan['memory_hashes'][str(seed)],
                    'validation': sha(out / 'recovered' / f'seed_{seed}/validation/complete.json'),
                    'alpha_grid': plan['recipe']['ridge_alphas']}
    if (folder / 'complete.json').exists():
        complete = read(folder / 'complete.json')
        require(complete['dependencies'] == dependencies, 'Ridge selection dependencies changed')
        for name, expected in complete['artifacts'].items():
            require(sha(folder / name) == expected, 'Frozen ridge artifact changed')
        with np.load(folder / 'ridge.npz', allow_pickle=False) as z:
            return {'coef': z['coef'], 'intercept': float(z['intercept']), 'identity': digest(complete), 'metadata': read(folder / 'selection.json')}
    with np.load(plan['refresh'] / 'memories' / f'seed_{seed}' / 'refreshed_memory.npz', allow_pickle=False) as z:
        x, y = z['vectors'], z['gaps']
    began = time.perf_counter()
    settings = {'ridge_alphas': plan['recipe']['ridge_alphas'], 'knn_k': 31, 'knn_temperature': .05,
                'knn_k_grid': [31], 'knn_temperature_grid': [.05]}
    with threadpool_limits(limits=plan['recipe']['cpu_threads']):
        fitted = fit(x, y, val_x, validation, settings)
    selection = {'seed': seed, 'memory_answers': len(y), 'alpha': fitted['ridge']['alpha'],
                 'fit_seconds': time.perf_counter()-began, 'candidates': fitted['candidates'],
                 'target': 'actual normalized proxy-minus-judge gap', 'validation_answers': len(validation),
                 'objective': 'conversation-weighted validation MSE', 'refit_on_validation': False,
                 'test_labels_used': False, 'knn_parameters': {'k': 31, 'temperature': .05}}
    write(folder / 'selection.json', selection)
    save_npz(folder / 'ridge.npz', coef=fitted['ridge']['coef'], intercept=np.array(fitted['ridge']['intercept']))
    complete = {'dependencies': dependencies, 'artifacts': {n: sha(folder / n) for n in ('selection.json', 'ridge.npz')}}
    write(folder / 'complete.json', complete)
    return {'coef': fitted['ridge']['coef'], 'intercept': fitted['ridge']['intercept'], 'identity': digest(complete), 'metadata': selection}


def encoder_parity(plan, proxy):
    with np.load(plan['inputs'] / 'memory/detectors/gap_knn.npz', allow_pickle=False) as z:
        ids, reference = z['bank_ids'], z['vectors']
    bank = pd.read_csv(plan['inputs'] / 'candidate_bank.csv', keep_default_na=False)
    positions, eligible = [], []
    for pos in np.linspace(0, len(ids)-1, min(128, len(ids)), dtype=int):
        row = bank.iloc[int(ids[pos])]
        try:
            _, lengths = proxy.encode_full([row.prompt], [row.answer])
        except ValueError as error:
            if str(error).startswith('Full reward input needs'):
                continue
            raise
        if int(lengths[0]) <= 1024:
            positions.append(pos)
            eligible.append(int(ids[pos]))
        if len(positions) == 32:
            break
    require(len(positions) >= 8, 'Too few encoder parity references')
    frame = bank.iloc[eligible]
    actual = proxy.score(frame.prompt.tolist(), frame.answer.tolist(), features=True)
    error = abs(actual['raw'] - frame.proxy_raw.to_numpy())
    cosine = (actual['features'] * reference[positions]).sum(1)
    require(error.mean() <= .03 and error.max() <= .3 and cosine.min() >= .999, 'Frozen encoder does not match saved memory')
    return {'answers': len(frame), 'max_raw_error': float(error.max()), 'min_cosine': float(cosine.min())}


def runtime_check(plan):
    require(os.name == 'posix', 'GPU continuation uses the original Linux CUDA environment. Run audit on Windows, and run/preflight on your RTX PRO 6000 Linux pod.')
    for name, expected in plan['audit']['runtime_versions_required'].items():
        require(metadata.version(name) == expected, f'Matched control requires {name}=={expected}; found {metadata.version(name)}. Use the original HH environment.')


def evaluate_ridge(plan, out, assets, router, seed, ck, identity, cohort):
    from run_study import checkpoint_actor, release
    from knn_distillation.policy_eval import evaluate
    actor = checkpoint_actor(assets['policy'], plan['config'], seed, ck['path'], ck['identity'], ck['branch'])
    try:
        return evaluate(actor, router, plan['data'][cohort], out / 'evaluations' / cohort / f'seed_{seed}/ridge/update_{ck["update"]:06d}',
                        plan['config'], identity, seed, 'ridge', ck['sha256'], cohort, ck['update'], features=True, monitor_kl=True, stop_out=out)
    finally:
        del actor
        release()


def execute(plan, out, identity, preflight_only=False):
    import torch
    import assets as asset_module
    from run_study import configure, load_actor, release
    from ppo_engine import PPOTrainer
    from reward_bridge import RewardScorer
    from knn_distillation.ppo_training import train_to
    from knn_distillation.io import state_fingerprint, should_stop
    from common import StopRequested
    from .reward import RidgeRouter
    from .features import recover
    from .reports import controls, final_report
    c = {**plan['config'], 'allow_downloads': plan['recipe']['allow_downloads'], 'extra_hf_cache': plan['recipe']['extra_hf_cache']}
    configure(c)
    # Only the download-status destination changes; imports use one shared source tree.
    asset_module.ROOT = out
    assets = asset_module.resolve_all(c)
    write(out / 'environment.json', {'gpu': torch.cuda.get_device_name(), 'cuda': torch.version.cuda,
          'runtime': {n: metadata.version(n) for n in ('torch', 'transformers', 'peft')}, 'model_revisions': asset_module.MODELS})
    proxy = RewardScorer(assets['proxy'], c['reward_batch_size'], c['reward_max_tokens'])
    judge = RewardScorer(assets['judge'], c['reward_batch_size'], c['reward_max_tokens'])
    router = RidgeRouter(proxy, judge, plan['calibration'], c['cpu_threads'])
    write(out / 'encoder_parity.json', encoder_parity(plan, proxy))
    for seed in plan['recipe']['seeds']:
        actor = load_actor(assets['policy'], c, seed)
        trainer = PPOTrainer(actor, None, c)
        try:
            parent = plan['parents'][seed]
            step, _ = trainer.restore(parent['path'], parent['identity'], seed, parent['branch'], optimizer=True)
            require(step == 300 and state_fingerprint(trainer) == plan['fingerprints'][seed], 'Restored parent state does not match the saved controls')
        finally:
            del trainer, actor
            release()
    write(out / 'preflight.json', {'passed': True, 'matched_parent_states': plan['fingerprints'], 'gpu': torch.cuda.get_device_name()})
    if preflight_only:
        print('Preflight passed. No PPO updates or validation judge calls were made.', flush=True)
        return
    bundles, recovered = {}, {}
    # Select every seed's alpha before opening offline labels or running final evaluation.
    for seed in plan['recipe']['seeds']:
        router.load_memory(plan['refresh'] / 'memories' / f'seed_{seed}')
        folder = out / 'recovered' / f'seed_{seed}/validation'
        validation, x = recover(folder, labels(plan, seed, 'validation'), router, identity, c['reward_batch_size'], out)
        bundles[seed] = fit_bundle(plan, out, seed, validation, x, identity)
        recovered[seed] = {'validation': folder}
    write(out / 'selection_complete.json', {'identity': identity, 'bundles': {str(s): b['identity'] for s, b in bundles.items()}, 'test_used_for_selection': False})
    endpoints = {}
    for seed in plan['recipe']['seeds']:
        router.load_memory(plan['refresh'] / 'memories' / f'seed_{seed}')
        bundle = bundles[seed]
        router.set_ridge(bundle['coef'], bundle['intercept'], bundle['identity'])
        rows = schedule(plan['data']['distill_train'], 100, c['rollout_batch_size'], plan['manifest']['options']['data_seed']+seed)
        for target in (350, 400):
            ck = train_to(out, out / 'runs' / f'seed_{seed}/ridge', c, assets, router, rows, seed, 'ridge', plan['parents'][seed],
                          identity, target, 50, plan['fingerprints'][seed])
            evaluate_ridge(plan, out, assets, router, seed, ck, identity, 'monitor')
        endpoints[seed] = ck
    write(out / 'final_lock.json', {'endpoints': {str(s): ck['sha256'] for s, ck in endpoints.items()}, 'final_prompts': digest(plan['data']['final']), 'all_training_complete': True})
    control_frames = controls(plan)
    for seed in plan['recipe']['seeds']:
        router.load_memory(plan['refresh'] / 'memories' / f'seed_{seed}')
        bundle = bundles[seed]
        router.set_ridge(bundle['coef'], bundle['intercept'], bundle['identity'])
        evaluate_ridge(plan, out, assets, router, seed, endpoints[seed], identity, 'final')
        folder = out / 'recovered' / f'seed_{seed}/offline'
        recover(folder, labels(plan, seed, 'offline'), router, identity, c['reward_batch_size'], out)
        recovered[seed]['offline'] = folder
        for branch in ('proxy', 'knn'):
            folder = out / 'recovered' / f'seed_{seed}' / branch
            recover(folder, control_frames[(seed, branch)].to_dict('records'), router, identity, c['reward_batch_size'], out)
            recovered[seed][branch] = folder
    if should_stop(out):
        raise StopRequested('Paused before report generation')
    with threadpool_limits(limits=c['cpu_threads']):
        final_report(plan, out, bundles, recovered)
    write(out / 'complete.json', {'identity': identity, 'new_PPO_updates': 300,
          'artifacts': {p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob('*')) if p.is_file() and
                        p != out / 'complete.json' and p.name not in ('status.json', 'runner.lock') and p.suffix not in ('.pending', '.pt')}})
    print('Completed reports: ' + str(out / 'reports'), flush=True)


def main(argv=None, project=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['audit', 'preflight', 'run'])
    parser.add_argument('--config', type=Path)
    parser.add_argument('--runtime', type=Path, help='Original runtime containing inputs/, outputs/, refresh2_outputs/ and knn_distillation_outputs/')
    parser.add_argument('--output', type=Path, help='New output root; never an existing input study')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    try:
        plan = resolve(project or Path(__file__).resolve().parents[3], args.config, args.runtime, args.output)
        audit(plan)
        identity, record = identify(plan)
        out = plan['output'] / ('study_' + identity[:16])
        summary = {k: plan['audit'][k] for k in ('control_reuse_verified', 'seeds', 'start_update', 'end_update', 'memory_budgets', 'prompt_counts', 'validation_new_judge_answers', 'runtime_versions_required')}
        print(json.dumps({**summary, 'output': str(out), 'new_PPO_arm': 'ridge', 'controls_retrained': False}, indent=2), flush=True)
        if args.dry_run:
            return 0
        if args.action == 'audit':
            from .reports import controls, save_policy_report
            write(out / 'source_audit.json', plan['audit'])
            save_policy_report(plan, controls(plan), out / 'audit', complete=False)
            print('Verified controls: ' + str(out / 'audit/policy_report.md'))
            return 0
        runtime_check(plan)
        import fcntl
        from knn_distillation.io import pause
        from common import StopRequested
        out.mkdir(parents=True, exist_ok=True)
        # Shared with other invocations of this new runner in the same project.
        with open(plan['project'] / 'hh_ridge_ppo.lock', 'a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if (out / 'manifest.json').exists():
                require(read(out / 'manifest.json') == {'identity': identity, **record}, 'Run identity changed')
            write(out / 'manifest.json', {'identity': identity, **record})
            write(out / 'source_audit.json', plan['audit'])
            if (out / 'complete.json').exists():
                for name, expected in read(out / 'complete.json')['artifacts'].items():
                    require(sha(out / name) == expected, 'Completed output changed: ' + name)
                print('Already complete: ' + str(out / 'reports'))
                return 0
            require(not (out / 'PAUSE').exists(), 'Remove the PAUSE file to resume this run')
            signal.signal(signal.SIGTERM, pause)
            signal.signal(signal.SIGINT, pause)
            write(out / 'status.json', {'stage': 'running', 'identity': identity})
            try:
                execute(plan, out, identity, preflight_only=args.action == 'preflight')
                write(out / 'status.json', {'stage': 'ready' if args.action == 'preflight' else 'complete', 'identity': identity})
            except StopRequested as error:
                write(out / 'status.json', {'stage': 'paused', 'reason': str(error)})
                print(str(error))
            except Exception as error:
                write(out / 'status.json', {'stage': 'failed', 'error': str(error), 'type': type(error).__name__})
                raise
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    return 0
