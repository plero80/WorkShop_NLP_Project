from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from .common import atomic_json, read_json, read_jsonl
from .metrics import paired_bootstrap

LABELS = {
    'base': 'Base policy (no PPO)', 'proxy': 'Proxy PPO (1.5B grader)',
    'judge': 'Judge PPO (4B grader)', 'knn_static': 'Static kNN PPO (4B memory)',
    'knn_static_30b': 'Static kNN PPO — new (30B memory)',
    'knn_refresh': 'Refreshing kNN PPO (4B memory)', 'oracle': 'Oracle PPO (strict 0/1)',
}


def display(value):
    return f"{value:.3f}" if value is not None else "unavailable"


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)


def call_budgets(output):
    budgets = defaultdict(lambda: defaultdict(float))
    for e in read_jsonl(Path(output) / 'judge_calls.jsonl'):
        b = budgets[e['stage'], e['role']]
        if e['kind'] == 'request':
            b['requested_examples'] += e['examples']
            b['cache_hits'] += e['cache_hits']
        elif e['kind'] == 'forward_started':
            b['forward_started_examples'] += e['examples']
        elif e['kind'] == 'forward_completed':
            for key in ('input_tokens', 'output_tokens', 'seconds', 'invalid_scores'):
                b[key] += e[key]
            b['forward_completed_examples'] += e['examples']
            b['retry_examples'] += e['examples'] if e['retry'] else 0
        elif e['kind'] in ('extended_grade_recovered', 'score_format_recovered', 'unscored'):
            b[e['kind'] + '_examples'] += e['examples']
    return [{'stage': stage, 'model_role': role, **dict(v)} for (stage, role), v in sorted(budgets.items())]


def make_report(output, target=None, arms=None):
    output = Path(output)
    config = read_json(output / 'config.json')
    marker = read_json(output / 'final_protocol.json') if (output / 'final_protocol.json').exists() else None
    if arms is None:
        arms = marker['arms'] if marker else [a for a in config['arms'] if (output / 'arms' / a / 'completed.json').exists()]
    if target is None:
        steps = [read_json(output / 'arms' / a / 'completed.json')['update'] for a in arms]
        target = marker['updates'] if marker else min(steps, default=0)
    records, files, teacher_records = [], {}, []
    for kind in ('monitor', 'final'):
        for arm in ['base', *arms]:
            folder = output / 'evaluations' / kind / arm / f"step_{0 if arm == 'base' else target:06d}"
            if (folder / 'metrics.json').exists():
                records.append(read_json(folder / 'metrics.json'))
                files[kind, arm] = folder / 'responses.jsonl'
            if (folder / 'teacher30b_metrics.json').exists():
                teacher_records.append(read_json(folder / 'teacher30b_metrics.json'))
    comparisons = {}
    pairs = [(a, 'proxy') for a in arms if a != 'proxy']
    if 'knn_static_30b' in arms and 'knn_static' in arms:
        pairs.append(('knn_static_30b', 'knn_static'))
    for kind in ('monitor', 'final'):
        for left, right in pairs:
            if (kind, left) not in files or (kind, right) not in files:
                continue
            lhs, rhs = read_jsonl(files[kind, left]), read_jsonl(files[kind, right])
            key = f'{kind}/{left}_minus_{right}'
            comparisons[key] = paired_bootstrap(lhs, rhs, config['evaluation']['bootstrap_samples'], config['seed'])
            if all('numeric_match' in x for x in lhs + rhs):
                comparisons[key + '/numeric'] = paired_bootstrap(lhs, rhs, config['evaluation']['bootstrap_samples'], config['seed'], metric='numeric_match')
    budgets = call_budgets(output)
    review = [{"case_id": r['case_id'], "role": r['role'], "question_id": r['question_id'],
               "question": r['question'], "reference": r['reference'], "response": r['response'],
               "judge_output": r['judge_output'], "path": str(p.relative_to(output))}
              for p in sorted((output / 'review' / 'ungraded').glob('*.json')) for r in [read_json(p)]]
    write_csv(output / 'ungraded_examples.csv', review)
    skipped = read_json(output / 'skipped_arms.json') if (output / 'skipped_arms.json').exists() else {}
    training = {a: read_json(output / 'arms' / a / 'completed.json') for a in arms
                if a not in skipped and (output / 'arms' / a / 'completed.json').exists()}
    write_csv(output / 'metrics.csv', records)
    write_csv(output / 'teacher30b_metrics.csv', teacher_records)
    write_csv(output / 'judge_budget.csv', budgets)
    atomic_json(output / 'comparisons.json', comparisons)
    atomic_json(output / 'summary.json', {'target_updates': target, 'arms': arms, 'seed': config['seed'],
                'metrics': records, 'teacher30b_metrics': teacher_records,
                'paired_comparisons': comparisons, 'judge_budgets': budgets,
                'ungraded_cases': len(review), 'skipped_arms': skipped, 'training_completion': training})
    lines = ['# GSM8K PPO experiment', '', f"Training seed **{config['seed']}**, data split seed **{config.get('data_seed', config['seed'])}**, declared PPO target **{target}**.", '',
             'Every row evaluates the same 0.5B policy architecture. Judge PPO uses the 4B model as its training reward. '
             'The new static kNN arm uses the 1.5B proxy during PPO and a frozen memory labeled by the 30B model.', '',
             'Strict accuracy requires one parseable numeric box matching the reference. The secondary numeric check '
             'extracts a clear final value without using the reference during extraction. All questions remain in its denominator; '
             'unresolved responses are reported separately. Neither numeric check verifies the reasoning or unit semantics.', '',
             'The numeric extraction protocol is frozen before this suite runs. It was developed after inspecting the earlier experiment, '
             'so this suite is a follow-up on a previously inspected benchmark, not a fresh untouched benchmark test.', '',
             f"Common completion penalties: {config.get('completion_reward', {})}. Zero means the original task reward is retained.", '']
    lines += ['## Grading coverage and review', '',
              f"Ungraded scorer/answer cases: **{len(review)}**. Inspect [review files](review/README.md) "
              "or `ungraded_examples.csv` for questions, references, candidate answers and failed grader replies.", '',
              'Missing grades are excluded from reward fitting and PPO. Every evaluation answer remains in the accuracy denominator. '
              'Grade means use the available labels for that grader; gap diagnostics use available pairs. Missing metrics are unavailable.', '',
              'The declared target counts rollout attempts. An attempt with no valid rewards makes no optimizer update.', '']
    for arm, completion in training.items():
        lines.append(f"- {arm}: {completion.get('successful_updates', completion['update'])} successful updates; "
                     f"{completion.get('skipped_updates', 0)} skipped attempts.")
    for arm, reason in skipped.items():
        lines.append(f"- {arm}: unavailable ({reason}).")
    lines.append('')
    for kind in ('monitor', 'final'):
        rows = [r for r in records if r['cohort'] == kind]
        if not rows:
            continue
        if kind == 'final' and {r['arm'] for r in rows} != {'base', *arms}:
            lines += ['Final evaluation is incomplete: some declared policies do not yet have final metrics.', '']
        lines += [f'## {kind.title()} cohort', '',
                  '| Policy | Questions | Strict accuracy | Confirmed numeric matches / all | Unresolved | Valid box | Length capped |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for r in rows:
            numeric = f"{r['numeric_accuracy']:.2%} ({r['numeric_matches']}/{r['n']})" if 'numeric_accuracy' in r else 'not measured'
            lines.append(f"| {LABELS.get(r['arm'], r['arm'])} | {r['n']} | {r['accuracy']:.2%} | {numeric} | {r.get('numeric_unresolved', 'not measured')} | {r['format_valid_rate']:.2%} | {r['length_cap_rate']:.2%} |")
        lines += ['', 'Paired differences (percentage points; question-bootstrap 95% intervals):', '']
        for key, v in comparisons.items():
            if key.startswith(kind + '/'):
                lines.append(f"- {key.removeprefix(kind + '/')}: {100*v['accuracy_difference']:+.2f} [{100*v['ci95'][0]:+.2f}, {100*v['ci95'][1]:+.2f}].")
        lines += ['', 'Common 4B grading diagnostics for the same saved policy answers. Gap prediction here uses the 4B memory '
                  'for the static comparison policies, including the 30B-memory policy; it does not describe that new arm’s actual training reward. The optional refreshing branch uses its refreshed 4B memory.', '',
                  '| Policy | Proxy labels | 4B labels | Paired / all | Mean proxy score | Mean 4B score | 4B gap MSE | Zero-gap MSE | High-gap AUROC |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for r in rows:
            auc = f"{r['high_gap_auroc']:.3f}" if r['high_gap_auroc'] is not None else 'undefined'
            lines.append(f"| {r['arm']} | {r.get('n_proxy_scored', r['n'])} | {r.get('n_judge_scored', r['n'])} | {r.get('n_pair_scored', r['n'])}/{r['n']} | {display(r['mean_proxy_score'])} | {display(r['mean_judge_score'])} | {display(r['gap_mse'])} | {display(r['zero_gap_baseline_mse'])} | {auc} |")
        lines += ['']
    if teacher_records:
        lines += ['## Common 30B evaluation of final answers', '',
                  'These use the same saved answers and the 30B memory/normalization for every row. '
                  'No answers were regenerated. The 30B grader is also the new arm’s memory-label source, '
                  'so its score is a related-teacher diagnostic, not an independent correctness oracle.', '',
                  '| Policy | 30B labels | Paired / all | Mean 30B grade | 30B gap MSE | Zero-gap MSE | High-gap AUROC |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for r in teacher_records:
            auc = f"{r['high_gap_auroc']:.3f}" if r['high_gap_auroc'] is not None else 'undefined'
            lines.append(f"| {r['arm']} | {r.get('n_judge_scored', r['n'])} | {r.get('n_pair_scored', r['n'])}/{r['n']} | {display(r['mean_judge_score'])} | {display(r['gap_mse'])} | {display(r['zero_gap_baseline_mse'])} | {auc} |")
        lines += ['']
    if not any(r['cohort'] == 'final' for r in records):
        lines += ['The final test cohort has not been evaluated. Monitor results are exploratory.', '']
    lines += ['## What to notice', '',
              '- The primary teacher comparison is `knn_static_30b − knn_static`. Bigger teachers are candidates for improvement, not guaranteed better graders.',
              '- Report strict accuracy, numeric matches, unresolved responses, formatting, and truncation together. Missing a box does not establish that an answer is absent or incorrect.',
              '- Question-bootstrap intervals do not measure training-seed variability. Use the optional matched three-seed run for that.',
              '- Memory labels are signed normalized proxy–teacher gaps. A high gap is disagreement, not independently established reward hacking.',
              '- Memory prompts, proxy embeddings, neighbor settings, and PPO settings are shared. Each teacher has its own frozen calibration statistics.',
              '- Every PPO arm starts from the same initial actor/value state within a seed. The final checkpoint is the declared target, not the best monitor checkpoint.',
              '- Missing grades enter the review queue after bounded retries. No zero/default score is substituted. If missing calibration or memory labels leave insufficient data, affected arms are marked unavailable and other stages continue.',
              '- If a 30B memory label is missing, both static memories use the same remaining valid subset. Preparation coverage is recorded in prepared/grading_coverage.json and prepared_30b/complete.json.',
              '- Judge calls, retries and cache hits are itemized in judge_budget.csv. 30B preparation and optional final grading are distinct from PPO training.', '']
    (output / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    make_plot(output, arms)
    print(f"Report: {output / 'report.md'}", flush=True)


def make_plot(output, arms):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    base_path = output / 'evaluations' / 'monitor' / 'base' / 'step_000000' / 'metrics.json'
    base = read_json(base_path) if base_path.exists() else None
    for arm in arms:
        records = ([base] if base else []) + [read_json(p) for p in sorted((output / 'evaluations' / 'monitor' / arm).glob('step_*/metrics.json'))]
        for ax, key, title in zip(axes, ('accuracy', 'numeric_accuracy', 'format_valid_rate'),
                                 ('Strict boxed accuracy', 'Confirmed numeric matches / all', 'Valid-box rate')):
            selected = [r for r in records if key in r]
            if selected:
                ax.plot([r['update'] for r in selected], [r[key] for r in selected], marker='o', label=arm)
            ax.set(title=title, xlabel='PPO rollout attempts', ylim=(0, 1))
            ax.grid(alpha=.2)
    if axes[0].lines:
        axes[0].legend(fontsize=7)
    fig.suptitle('GSM8K monitor cohort — one training seed')
    fig.tight_layout()
    fig.savefig(output / 'learning_curves.png', dpi=160)
    plt.close(fig)
