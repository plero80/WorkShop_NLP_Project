"""Evaluate both predictors on identical saved answers and compare policy outcomes."""
from collections import defaultdict
import gzip
import json
import statistics

import numpy as np
from sklearn.metrics import average_precision_score

from gsm8k_experiment.common import atomic_json, read_json, read_jsonl
from gsm8k_experiment.grading import number
from gsm8k_experiment.memory import corrected_reward
from gsm8k_experiment.metrics import auroc, paired_bootstrap, summarize_rows
from gsm8k_experiment.validation import regression, csv_file, label
from .inputs import features, require, selection


def pair_metrics(left, right, count, seed):
    require({r['id'] for r in left} == {r['id'] for r in right}, 'Policy evaluation questions differ')
    result = {}
    for metric in ('correct','numeric_match','judge_score','high_gap','response_tokens'):
        lhs = {r['id']:r for r in left}
        rhs = {r['id']:r for r in right}
        ids = sorted(k for k in lhs if lhs[k].get(metric) is not None and rhs[k].get(metric) is not None)
        if ids:
            value = paired_bootstrap([lhs[k] for k in ids],[rhs[k] for k in ids],count,seed,metric=metric)
            result[metric] = {'mean_difference':value['accuracy_difference'],'ci95':value['ci95'],'paired_questions':len(ids)}
    return result


def report(resolved, models):
    out = resolved['output']
    predictors,policies,comparisons = [],[],{}
    with gzip.open(out/'all_predictions.jsonl.gz','wt',encoding='utf-8') as stream:
        for p in resolved['plans']:
            seed,model = p['seed'],models[p['seed']]
            definition = model.label_definition
            folder = out/f'seed_{seed}'
            selection_rows,x,y,_ = selection(p)
            cohorts = [('selection',selection_rows,x)]
            policy_rows = {}
            for arm in ('base','proxy','knn_static','ridge'):
                root = folder if arm == 'ridge' else p['source']
                path = root/'evaluations'/p['kind']/arm/f"step_{0 if arm == 'base' else p['updates']:06d}/responses.jsonl"
                if arm == 'ridge' and not path.exists():
                    continue
                rows = read_jsonl(path)
                expected = {r['id']:r for r in p['split']['cohorts'][p['kind']]}
                require(len(rows) == len(expected) and {r['id'] for r in rows} == set(expected), 'Evaluation does not cover the exact reserved questions')
                require(all(all(r[k] == expected[r['id']][k] for k in ('question','reference')) for r in rows), 'Evaluation question text changed')
                x = features(rows,p['memory'].encoder_identity,[root/'reward_cache.sqlite',p['source']/'reward_cache.sqlite'])
                cohorts.append((p['kind']+'/'+arm,rows,x))
                pred = (model if arm == 'ridge' else p['memory']).predict(x,[r['id'] for r in rows],p['memory'].encoder_identity)[0]
                annotated = []
                for r,guess in zip(rows,pred):
                    gap = number(p['norm'].gap(r['proxy_score'],r['judge_score']))
                    annotated.append({**r,'gap':gap,'predicted_gap':float(guess),
                                      'high_gap':None if gap is None or definition is None else bool(label(np.array([gap]),definition)[0])})
                policy_rows[arm] = annotated
                summary = summarize_rows(annotated,p['norm'].threshold)
                flags = [r['high_gap'] for r in annotated if r['high_gap'] is not None]
                summary.update(legacy_high_gap_rate=summary['high_gap_rate'],high_gap_rate=float(np.mean(flags)) if flags else None,
                               common_label_threshold=definition['threshold'] if definition else None,
                               common_label_comparator=definition['comparator'] if definition else None)
                policies.append({'seed':seed,'cohort':p['kind'],'arm':arm,'updates':0 if arm == 'base' else p['updates'],
                                 **summary})
            for cohort,rows,x in cohorts:
                valid = np.array([all(r.get(k) is not None and np.isfinite(r[k]) for k in ('proxy_score','judge_score')) for r in rows])
                gaps = p['norm'].gap([r['proxy_score'] for r in rows],[r['judge_score'] for r in rows])
                for name,predictor in (('knn',p['memory']),('ridge',model)):
                    predicted = predictor.predict(x,[r['id'] for r in rows],p['memory'].encoder_identity)[0]
                    high = label(gaps[valid],definition) if definition else None
                    predictors.append({'seed':seed,'cohort':cohort,'predictor':name,'answers':len(rows),'paired_answers':int(valid.sum()),
                       'selection_diagnostic':cohort == 'selection','threshold':definition['threshold'] if definition else None,
                       'comparator':definition['comparator'] if definition else None,'legacy_threshold':p['norm'].threshold,
                       **regression(gaps[valid],predicted[valid]),'high_gap_auroc':auroc(high,predicted[valid]) if high is not None else None,
                       'high_gap_ap':float(average_precision_score(high,predicted[valid])) if high is not None and high.any() else None})
                    corrected = corrected_reward(p['norm'].proxy_z([r['proxy_score'] for r in rows]),predicted,p['config']['knn']['correction'])
                    for i,r in enumerate(rows):
                        item = {**r,'seed':seed,'cohort':cohort,'predictor':name,'gap':number(gaps[i]),
                                'predicted_gap':number(predicted[i]),'corrected_reward':number(corrected[i]),
                                'high_gap':bool(label(gaps[i:i+1],definition)[0]) if valid[i] and definition else None}
                        stream.write(json.dumps(item,ensure_ascii=False,allow_nan=False)+'\n')
            if 'ridge' in policy_rows:
                for control in ('proxy','knn_static'):
                    comparisons[f'seed_{seed}/ridge_minus_{control}'] = pair_metrics(policy_rows['ridge'],policy_rows[control],
                            p['config']['evaluation']['bootstrap_samples'],seed)
    csv_file(out/'predictor_metrics.csv',predictors)
    csv_file(out/'policy_metrics.csv',policies)
    atomic_json(out/'paired_comparisons.json',comparisons)
    groups = defaultdict(list)
    for r in policies:
        groups[(r['cohort'],r['arm'],r['updates'])].append(r)
    means = []
    for (cohort,arm,updates),rows in groups.items():
        for key in ('accuracy','numeric_accuracy','mean_judge_score','high_gap_rate','mean_response_tokens'):
            values = [r[key] for r in rows if r.get(key) is not None]
            means.append({'cohort':cohort,'arm':arm,'updates':updates,'metric':key,'seeds':len(values),
                          'mean':statistics.mean(values) if values else None,'sample_sd':statistics.stdev(values) if len(values)>1 else None})
    csv_file(out/'policy_seed_summary.csv',means)
    from gsm8k_experiment.report import call_budgets
    costs = [{'seed':p['seed'],**r} for p in resolved['plans'] for r in call_budgets(out/f"seed_{p['seed']}")]
    csv_file(out/'new_judge_budget.csv',costs)
    lines = ['# GSM8K ridge comparison','',
      'Ridge fits actual signed 4B gaps on exactly the saved kNN memory. Alpha is selected by question-weighted validation MSE. '
      'All selections are frozen before this report reads policy outcomes. No validation labels are added to the fitting memory.','',
      'This is a follow-up on previously inspected GSM8K results. Saved controls are reused; only ridge receives new PPO training. '
      'The new policy starts from the saved initial actor/value state, with the same seed, prompt schedule, optimizer settings and target. '
      'Different GPU hardware can still introduce numerical differences.','',
      'Both predictors use the same diagnostic target selected by the existing kNN validation rule on calibration/selection only. '
      'That existing rule chooses the supported upper-tail target with highest validation kNN AUROC, so it can favor kNN on validation. '
      'Ridge does not tune its own label cutoff. The legacy 95th-percentile threshold is also recorded; it can have no positives with discrete grades. '
      'AUROC ranks continuous predicted gaps. Selection metrics are tuning diagnostics, not held-out evidence.','',
      '| Seed | Answers from | Predictor | Pairs | MSE | R2 | Pearson | Spearman | AUROC | AP |',
      '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    fmt = lambda x:'unavailable' if x is None else f'{x:.4f}'
    for r in predictors:
        lines.append('| '+ ' | '.join([str(r['seed']),r['cohort'],r['predictor'],str(r['paired_answers']),
          *[fmt(r[k]) for k in ('gap_mse','gap_r2','gap_pearson','gap_spearman','high_gap_auroc','high_gap_ap')]])+' |')
    lines += ['','| Seed | Cohort | Policy | Updates | Strict accuracy | Numeric accuracy | Mean judge | High-gap rate | Mean tokens |',
              '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in policies:
        lines.append('| '+' | '.join([str(r['seed']),r['cohort'],r['arm'],str(r['updates']),
                    *[fmt(r.get(k)) for k in ('accuracy','numeric_accuracy','mean_judge_score','high_gap_rate','mean_response_tokens')]])+' |')
    lines += ['','Ridge policy results appear only after its GPU run finishes. Missing grades remain unscored and use the existing review queue; '
              'no replacement reward is invented. Accuracy retains every question. See each seed\'s `review/ungraded/` for failed grades.',
              '', '`all_predictions.jsonl.gz` contains the underlying answers and both predictions. CSV files contain all metrics, including MAE. '
              '`paired_comparisons.json` contains question-bootstrap intervals for ridge minus each control. These intervals condition on the trained policies; '
              '`policy_seed_summary.csv` reports seed means and sample SD separately. '
              '`new_judge_budget.csv` records new requests/cache hits/retries; ridge fitting itself uses zero new judge calls.','']
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')
