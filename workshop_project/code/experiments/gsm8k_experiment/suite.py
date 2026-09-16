"""Sequential seeds on one GPU; each seed retains an isolated experiment directory."""
from __future__ import annotations
from gsm8k_experiment.common import DEFAULT_CONFIG, OUTPUT_ROOT

import argparse
import copy
import statistics
import subprocess
import sys
from pathlib import Path

from .common import ROOT, atomic_json, load_config, read_json, run_lock
from .report import LABELS


def aggregate(output, seeds, arms):
    records, differences = [], []
    for seed in seeds:
        folder = Path(output) / f'seed_{seed}'
        marker = read_json(folder / 'final_protocol.json')
        summary = read_json(folder / 'summary.json')
        if marker['arms'] != arms:
            raise ValueError('Cannot aggregate different arm lists.')
        by_arm = {r['arm']: r for r in summary['metrics'] if r['cohort'] == 'final'}
        missing = {'base', *arms} - set(by_arm)
        if missing - set(summary.get('skipped_arms', {})) or set(by_arm) - {'base', *arms}:
            raise ValueError('A seed is missing final evaluations.')
        records.append({'seed': seed, 'metrics': by_arm, 'skipped_arms': summary.get('skipped_arms', {})})
        if {'knn_static', 'knn_static_30b'} <= set(by_arm):
            differences.append({'seed': seed,
                'strict_difference': by_arm['knn_static_30b']['accuracy'] - by_arm['knn_static']['accuracy'],
                'numeric_difference': by_arm['knn_static_30b']['numeric_accuracy'] - by_arm['knn_static']['numeric_accuracy']})
    def stats(values):
        return {'mean': statistics.mean(values) if values else None, 'sample_sd': statistics.stdev(values) if len(values) > 1 else None, 'values': values, 'n_seeds': len(values)}
    summary = {arm: {key: stats([r['metrics'][arm][key] for r in records if arm in r['metrics']])
                    for key in ('accuracy', 'numeric_accuracy', 'numeric_unresolved_rate', 'format_valid_rate', 'length_cap_rate')}
               for arm in ['base', *arms]}
    paired = {key: stats([r[key] for r in differences]) for key in ('strict_difference', 'numeric_difference')} if differences else {}
    atomic_json(Path(output) / 'suite_summary.json', {'seeds': seeds, 'arms': summary, 'per_seed': records, 'paired_teacher_differences': paired})
    lines = ['# GSM8K matched-seed suite', '', f'Seeds: {seeds}. Values below are mean ± sample standard deviation across training seeds, not confidence intervals.', '',
             '| Policy | Strict accuracy | Numeric matches / all | Valid box | Unresolved |', '|---|---:|---:|---:|---:|']
    def fmt(v):
        if v['mean'] is None:
            return 'unavailable (no graded training data)'
        return f"{v['mean']:.2%} ± {v['sample_sd']:.2%}" if v['sample_sd'] is not None else f"{v['mean']:.2%} (one seed)"
    for arm, values in summary.items():
        lines.append(f"| {LABELS.get(arm, arm)} | {fmt(values['accuracy'])} | {fmt(values['numeric_accuracy'])} | {fmt(values['format_valid_rate'])} | {fmt(values['numeric_unresolved_rate'])} |")
    lines += ['', 'Within-seed 30B-memory minus 4B-memory differences:', '']
    for key, value in paired.items():
        lines.append(f"- {key}: {fmt(value)}; per-seed values {value['values']}.")
    lines += ['', 'The data partition is held fixed across seeds. Policy initialization, sampled candidates, and PPO randomness vary by seed. '
              'Each seed uses matched memory responses across its two teachers. Three seeds provide limited evidence about variability; inspect every seed.', '']
    (Path(output) / 'suite_report.md').write_text('\n'.join(lines), encoding='utf-8')


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    p.add_argument('--output', type=Path, default=OUTPUT_ROOT / 'seeds')
    p.add_argument('--seeds', type=int, nargs='+', required=True)
    p.add_argument('--stage', choices=['pilot', 'full', 'prepare'], default='full')
    args = p.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds):
        p.error('Seeds must be distinct.')
    config = load_config(args.config)
    output = args.output.resolve()
    with run_lock(output):
        declaration = {'seeds': args.seeds, 'config': config}
        path = output / 'suite_protocol.json'
        if path.exists() and read_json(path) != declaration:
            raise ValueError('Suite seeds or configuration changed. Use another output directory.')
        atomic_json(path, declaration)
        first = output / f'seed_{args.seeds[0]}'
        for seed in args.seeds:
            current = copy.deepcopy(config)
            current['seed'] = seed
            current['data_seed'] = config.get('data_seed', config['seed'])
            config_path = output / 'configs' / f'seed_{seed}.json'
            atomic_json(config_path, current)
            destination = output / f'seed_{seed}'
            # Resolve once in the first seed and reuse exact model revisions in every seed.
            resolved = first / 'resolved_assets.json'
            if destination != first and resolved.exists():
                pinned = destination / 'resolved_assets.json'
                if pinned.exists() and read_json(pinned) != read_json(resolved):
                    raise ValueError('Seeds must use identical resolved model revisions.')
                atomic_json(pinned, read_json(resolved))
            destination.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, '-u', '-m', 'gsm8k_experiment.run', '--config', str(config_path),
                       '--output', str(destination), '--stage', args.stage]
            print(f'Start/resume seed {seed}: {destination}', flush=True)
            atomic_json(output / 'suite_status.json', {'stage': 'running', 'seed': seed, 'log': str(destination / 'experiment.log')})
            with (destination / 'experiment.log').open('a') as log:
                code = subprocess.call(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            if code:
                atomic_json(output / 'suite_status.json', {'stage': 'failed', 'seed': seed, 'returncode': code})
                raise SystemExit(f'Seed {seed} stopped. See {destination / "experiment.log"}; other seeds were not launched.')
        if args.stage == 'full':
            aggregate(output, args.seeds, config['arms'])
        atomic_json(output / 'suite_status.json', {'stage': 'complete', 'run_stage': args.stage, 'seeds': args.seeds})


if __name__ == '__main__':
    main()
