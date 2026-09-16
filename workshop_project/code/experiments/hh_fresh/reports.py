"""Fresh-run prediction and policy tables, including both refresh comparisons."""
import numpy as np
import pandas as pd

from hh_ridge_ppo.features import load
from hh_ridge_ppo.protocol import read,write,sha,require,load_evaluation
from hh_ridge_ppo.reports import policy_rows,aggregate,policy_intervals,fidelity,markdown
from hh_offline.metrics import paired_intervals
from .artifacts import normalize


def table(plan,out,cohort,frames,comparisons):
    dest=out/'reports';dest.mkdir(parents=True,exist_ok=True)
    per_seed=policy_rows(frames)
    per_seed.to_csv(dest/(cohort+'_policy_by_seed.csv'),index=False)
    aggregate(per_seed,['branch']).to_csv(dest/(cohort+'_policy_seed_summary.csv'),index=False)
    intervals,pairs=policy_intervals(frames,plan['recipe']['bootstrap_samples'],plan['recipe']['review_seed'],comparisons)
    pairs.to_csv(dest/(cohort+'_policy_paired_answers.csv'),index=False)
    write(dest/(cohort+'_policy_intervals.json'),{'records':intervals,'scope':'Paired prompt bootstrap; seed-mean intervals average seeds within prompt first. Conditional on the three trained seeds; not multiplicity adjusted.'})
    (dest/(cohort+'_policy_report.md')).write_text('# '+cohort+' policy outcomes\n\n'+markdown(per_seed)+
        '\n\nAll policies and controls were trained in this fresh run. Individual seeds and mean/SD tables are saved separately. '
        'EOS completion and the refusal phrase diagnostic are automatic measures, not human ratings.\n',encoding='utf-8')


def report(plan,out,cal,bundles,memories):
    dest=out/'reports';dest.mkdir(parents=True,exist_ok=True)
    metrics,exports,intervals=[],[],[]
    policy_frames={name:{} for name in ('final','refresh1','refresh2')}
    u=plan['recipe']['stage_updates']
    labels={'final':{'proxy':4*u,'knn':4*u,'ridge':4*u},'refresh1':{'static_M0':2*u,'refresh_M1':2*u},
            'refresh2':{'parent_pi2':2*u,'static_M1':3*u,'refresh_M2':3*u}}
    for seed in plan['recipe']['seeds']:
        with np.load(memories[seed]/'refreshed_memory.npz',allow_pickle=False) as z:
            memory={k:z[k] for k in ('vectors','gaps')}
        f,x=load(out/'samples'/f'seed_{seed}/offline')
        targets=[('offline',normalize(f,cal),x)]
        for cohort,arms in labels.items():
            for label,update in arms.items():
                folder=out/'evaluations'/cohort/f'seed_{seed}'/label/f'update_{update:06d}'
                frame,done=load_evaluation(folder)
                require(sha(folder/'features.npz')==done['features_sha256'],'Evaluation features changed')
                frame['branch']=label
                policy_frames[cohort][(seed,label)]=frame
                # The full M2 predictors are compared on identical answers for each policy.
                with np.load(folder/'features.npz',allow_pickle=False) as z:
                    targets.append((cohort+'_'+label,frame,z['vectors']))
        b=bundles[seed]
        for cohort,frame,x in targets:
            rows,export,predictions=fidelity(frame,x,memory,b['coef'],b['intercept'],cal,seed,cohort,plan['config']['cpu_threads'])
            metrics.extend(rows);exports.append(export)
            if cohort=='offline':
                intervals.append({'seed':seed,**paired_intervals(frame,predictions['knn'],predictions['ridge'],cal['theta'],plan['recipe']['bootstrap_samples'],plan['recipe']['data_seed'])})
    for cohort,comparisons in [('final',[('knn','proxy'),('ridge','proxy'),('ridge','knn')]),
                               ('refresh1',[('refresh_M1','static_M0')]),('refresh2',[('refresh_M2','static_M1')])]:
        table(plan,out,cohort,policy_frames[cohort],comparisons)
    metric_table=pd.DataFrame(metrics)
    metric_table.to_csv(dest/'predictor_by_seed.csv',index=False)
    aggregate(metric_table,['cohort','predictor']).to_csv(dest/'predictor_seed_summary.csv',index=False)
    pd.concat(exports,ignore_index=True).to_csv(dest/'all_predictions.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    write(dest/'predictor_intervals.json',{'records':intervals,'scope':'Per-seed paired conversation bootstrap on the offline cohort; all ridge settings selected on separate validation data.'})
    cols=['seed','predictor','gap_mse','gap_mae','gap_r2','gap_pearson','gap_spearman','high_gap_auroc','high_gap_ap']
    (dest/'predictor_report.md').write_text('# Fresh M2 gap prediction\n\n'+markdown(metric_table[metric_table.cohort=='offline'][cols])+
        '\n\nThe complete CSV includes Pearson and all other metrics on refresh-2 answers, final-policy answers, and first-refresh answers. '
        'These compare the final M2 predictors on the same saved answers; they do not claim every earlier policy was trained with M2 or ridge. '
        'The independent calibration cutoff was frozen before any PPO.\n',encoding='utf-8')
    costs=[]
    for p in sorted((out/'samples').glob('**/complete.json')):
        done=read(p)
        costs.append({'stage':p.parent.relative_to(out).as_posix(),'kind':'generation_and_scoring',
                      'answers':done['answers'],'judge_answers':done['new_judge_answers'],'proxy_answers':done['proxy_answers'],'seconds':done['seconds']})
    for p in sorted((out/'runs').glob('seed_*/*/history.json')):
        segment=read(p.parent/'segment.json')
        tail=[row for row in read(p) if row.get('segment_start')==segment['start'] and row.get('reward_source')==segment['branch']]
        require(len(tail)==u,'A PPO segment has not completed its declared budget')
        require(sum(row['model_calls']['teacher_answers'] for row in tail)==0,'PPO unexpectedly called the judge')
        costs.append({'stage':p.parent.relative_to(out).as_posix(),'kind':'PPO','updates':len(tail),
                      'judge_answers':0,'proxy_answers':sum(row['model_calls']['proxy_answers'] for row in tail),'seconds':sum(row['seconds'] for row in tail)})
    for p in sorted((out/'evaluations').glob('**/complete.json')):
        done=read(p)
        costs.append({'stage':p.parent.relative_to(out).as_posix(),'kind':'evaluation','answers':done['rows'],
                      'judge_answers':done['rows'],'proxy_answers':done['rows'],'seconds':done['last_invocation_seconds']})
    for seed in plan['recipe']['seeds']:
        selected=read(out/'ridge'/f'seed_{seed}/selection.json')
        costs.append({'stage':f'ridge/seed_{seed}','kind':'ridge_fit','judge_answers':0,'seconds':selected['fit_seconds']})
    pd.DataFrame(costs).to_csv(dest/'costs.csv',index=False)
    from .run import budget
    expected=budget(plan)
    realized_judge=int(sum(row.get('judge_answers',0) for row in costs))
    realized_updates=int(sum(row.get('updates',0) for row in costs))
    require(realized_updates==expected['total_PPO_updates_all_branches'],'Training budget does not match the recipe')
    require(realized_judge==expected['new_judge_answers'],'Judge budget does not match the completed stages')
    write(dest/'budget.json',{'planned':expected,'realized_judge_answers':realized_judge,'realized_PPO_updates':realized_updates,
          'timing':'Completed work only; interrupted/discarded batches and model loading excluded. Evaluation time records the last invocation after resumption.',
          'fitting_budget':'Ridge and kNN receive the exact same M2 embeddings and actual gap labels. Validation is additional ridge-selection cost and is never appended to memory.'})
    import review
    original_root=review.ROOT
    try:
        review.ROOT=plan['project']/'code/templates'
        frames={('raw' if arm=='knn' else arm,seed,256):frame for (seed,arm),frame in policy_frames['final'].items()}
        folder=review.make_pack(out,'fresh_ridge_blinded',frames,['proxy','ridge'],plan['config'])
        key=read(folder/'private/key.json');key['reference_method']='knn';write(folder/'private/key.json',key)
    finally:review.ROOT=original_root
    (dest/'README.md').write_text('# Full fresh HH-RLHF results\n\n'
        'All policies, controls, labels, calibration and memories were produced in this run from pinned pretrained models and HH-RLHF. '
        'The original PPO implementation is shared with the existing project.\n\n'
        '- [Prediction metrics](predictor_report.md), including Pearson; all cohorts in `predictor_by_seed.csv`.\n'
        '- [Final proxy/kNN/ridge PPO comparison](final_policy_report.md).\n'
        '- [First refresh comparison](refresh1_policy_report.md).\n'
        '- [Second refresh comparison](refresh2_policy_report.md).\n'
        '- Underlying scores: `all_predictions.csv.gz`; per-seed and mean/SD CSVs; paired interval JSONs.\n'
        '- Measured budgets and timing: `budget.json`, `costs.csv`.\n'
        '- Blinded review: `../review/fresh_ridge_blinded/`; ratings remain manual.\n\n'
        'Offline gap fidelity, PPO judge scores, and human preference are distinct outcomes. '
        'This new protocol uses 8,000 initial memory rows by default, rather than replaying the old 7,990-row memory. '
        'Use this run\'s own control comparisons; do not combine its outcomes with historical controls as if they were matched.\n',encoding='utf-8')
