"""Completed-case reports, paired prompt uncertainty and explicit pending stages."""
from pathlib import Path
import argparse
import html
import zipfile
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
from common import ROOT,read_json,write_json,seed_for
from evaluation import load_completed


def metric_row(df):
    gap=df.actual_proxy_judge_gap;positive=df.high_gap.astype(bool)
    return {'n':len(df),'judge_mean_z':df.judge_z.mean(),'proxy_mean_z':df.proxy_z.mean(),
        'high_gap_rate':positive.mean(),'mean_gap_given_high':gap[positive].mean() if positive.any() else np.nan,
        'gap_q95':gap.quantile(.95),'gap_q99':gap.quantile(.99),'mean_actual_gap':gap.mean(),
        'raw_mse':np.mean(gap**2),'corrected_mse':np.mean(df.corrected_judge_residual**2),
        'response_tokens':df.response_tokens.mean(),'eos_fraction':df.ended_eos.mean(),
        'generic_refusal_fraction':df.answer.str.strip().isin(["I'm sorry, but I can't assist with that.",
            "Sorry, but I can't assist with that.","I'm sorry, but I cannot assist with that."]).mean(),
        'auroc':roc_auc_score(positive,df.gap_hat) if positive.nunique()==2 else np.nan,
        'high_gap_count':int(positive.sum()),
        'average_precision':average_precision_score(positive,df.gap_hat) if positive.any() else np.nan,
        'max_proxy_input_tokens':df.proxy_input_tokens.max(),'max_judge_input_tokens':df.judge_input_tokens.max(),
        'truncated_reward_inputs':int(df.proxy_truncated.sum()+df.judge_truncated.sum())}


def interval(delta,seed,draws):
    delta=np.asarray(delta,float);rng=np.random.default_rng(seed);means=[]
    for start in range(0,draws,100):
        ix=rng.integers(0,len(delta),(min(100,draws-start),len(delta)));means.extend(delta[ix].mean(1))
    return np.quantile(means,[.025,.975]) if draws else [np.nan,np.nan]


def create_report(output,full=False):
    output=Path(output);folder=output/'reports';folder.mkdir(parents=True,exist_ok=True)
    c=read_json(output/'manifest.json')['config'];rows=[];frames={};cost=[]
    for marker in sorted((output/'evaluations').rglob('complete.json')):
        sig=read_json(marker)['signature'];df=load_completed(marker.parent)
        key=tuple(sig[k] for k in ('family','cohort','cap','branch','seed','update'))
        if key in frames:raise ValueError('Duplicate evaluation identity in report.')
        frames[key]=df;rows.append({k:sig[k] for k in ('family','cohort','cap','branch','seed','update')}|metric_row(df))
        cost.append({'family':sig['family'],'cohort':sig['cohort'],'cap':sig['cap'],'teacher_answers_scored':len(df),
                     'teacher_input_tokens':int(df.judge_input_tokens.sum())})
    summary=pd.DataFrame(rows);summary.to_csv(folder/'all_evaluations.csv',index=False)
    comparisons=[];subgroups=[]
    targets=[]
    for key,other in frames.items():
        family,cohort,cap,branch,seed,update=key
        if family in ('checkpoint_recheck','round2','legacy_fresh','round1') and branch not in ('raw','initial'):
            targets.append((key,(family,cohort,cap,'raw',seed,update),'method'))
        if family=='round2' and branch in ('iterative_knn','iterative_capped'):
            targets.append((key,(family,cohort,cap,'knn_signed',seed,update),'main_iterative_vs_static' if branch=='iterative_knn' else 'capped_ablation_vs_static'))
        if family=='round2':
            parent_branch='raw' if branch=='raw' else 'knn_signed'
            targets.append((key,('round1',cohort,cap,parent_branch,seed,c['round1_updates']),'round2_minus_round1'))
        if family=='checkpoint_recheck' and cap==256:
            targets.append((key,(family,cohort,128,branch,seed,update),'cap_256_minus_128'))
    for other_key,base_key,kind in targets:
        if base_key not in frames:continue
        other,base=frames[other_key],frames[base_key]
        if set(other.prompt_id)!=set(base.prompt_id):raise ValueError('Paired prompt set differs.')
        p=other.merge(base,on='prompt_id',suffixes=('_other','_base'),validate='one_to_one')
        family,cohort,cap,branch,seed,update=other_key
        label=f'{branch} minus {base_key[3]}' if kind!='cap_256_minus_128' else f'{branch}: 256 minus 128'
        if kind=='round2_minus_round1':label=f'{branch} round2 minus {base_key[3]} round1'
        for metric in ['judge_z','high_gap','response_tokens','ended_eos','actual_proxy_judge_gap']:
            delta=p[metric+'_other'].astype(float)-p[metric+'_base'].astype(float)
            lo,hi=interval(delta,seed_for(family,cohort,cap,label,seed,metric),c['bootstrap_draws'] if full else 0)
            comparisons.append({'family':family,'cohort':cohort,'cap':cap,'comparison':label,'kind':kind,
                'seed':seed,'metric':metric,'n_prompts':len(p),'difference':delta.mean(),
                'prompt_bootstrap_low':lo,'prompt_bootstrap_high':hi})
        if kind=='cap_256_minus_128':continue
        for oe,be in [(True,True),(True,False),(False,True),(False,False)]:
            group=p[(p.ended_eos_other==oe)&(p.ended_eos_base==be)]
            if len(group):subgroups.append({'family':family,'cap':cap,'comparison':label,'seed':seed,
                'other_eos':oe,'base_eos':be,'n':len(group),
                'judge_delta':(group.judge_z_other-group.judge_z_base).mean(),
                'note':'Post-treatment descriptive subgroup, not a causal adjustment.'})
    paired=pd.DataFrame(comparisons);paired.to_csv(folder/'paired_differences.csv',index=False)
    seed_summary=pd.DataFrame()
    if len(paired):
        seed_summary=paired.groupby(['family','cohort','cap','comparison','kind','metric'],as_index=False).agg(
            seeds=('seed','nunique'),mean_difference=('difference','mean'),training_seed_sd=('difference','std'),
            minimum=('difference','min'),maximum=('difference','max'))
    seed_summary.to_csv(folder/'seed_summary.csv',index=False)
    pd.DataFrame(subgroups).to_csv(folder/'completion_subgroups.csv',index=False)
    if cost:pd.DataFrame(cost).groupby(['family','cohort','cap'],as_index=False).sum().to_csv(folder/'teacher_query_counts.csv',index=False)
    def table(df):return df.to_html(index=False,border=0,float_format=lambda x:f'{x:.4f}') if len(df) else '<p>Pending.</p>'
    state=read_json(output/'status.json') if (output/'status.json').is_file() else {'stage':'pending'}
    main=summary[(summary.family=='round2')&(summary.cohort=='fresh_final')] if len(summary) else summary
    short=summary[summary.family=='checkpoint_recheck'] if len(summary) else summary
    human=[]
    for path in sorted((output/'review').glob('*/analysis/human_summary.csv')):
        human.append('<h3>'+html.escape(path.parent.parent.name)+'</h3>'+table(pd.read_csv(path)))
    text='''<!doctype html><html><meta charset="utf-8"><title>Reward-gap follow-up</title><style>
    body{font:16px system-ui;max-width:1400px;margin:30px;color:#20303c}table{font-size:12px;border-collapse:collapse;display:block;overflow:auto}
    td,th{padding:7px;border-bottom:1px solid #ddd;text-align:right}th{background:#eef3f7}p{line-height:1.6}</style>
    <h1>Reward-gap follow-up</h1>'''
    text+='<p><b>State: '+html.escape(state['stage'])+'</b>. '+('Prompt intervals computed.' if full else 'Provisional summary; final prompt intervals pending.')+'</p>'
    text+='<p>Main comparison: iterative kNN versus static kNN after two 256-token PPO rounds. They share the exact round-one policy, value head and optimizer state. Only the main iterative memory changes; normalization, theta and signed correction stay fixed. Capped correction is a separate ablation.</p>'
    text+='<p>Labels represent proxy overestimation relative to a model judge, not human-confirmed hacking. Final conversation groups never enter either memory. Prompt intervals condition on trained policies; seed variability is separate. No checkpoint is selected using final results.</p>'
    text+='<h2>Fresh final results</h2>'+table(main)+'<h2>Paired effects</h2>'+table(seed_summary)
    text+='<h2>Saved-checkpoint 128/256 evaluation</h2>'+table(short)
    text+='<h2>Human review</h2>'+(''.join(human) if human else '<p>Awaiting manual blinded ratings. No human results are inferred from model scores.</p>')
    text+='<p>Exact refusal-template counts are diagnostics, not judgments that a refusal was inappropriate. Inspect human review and output-limit effects alongside reward scores. Teacher query counts exclude preflight reference checks.</p></html>'
    temporary=folder/'report.html.pending';temporary.write_text(text);temporary.replace(folder/'report.html')
    write_json(folder/'report_status.json',{'stage':state['stage'],'completed_evaluations':len(rows),
        'completed_main_final_policies':len(main),'full_prompt_intervals':full})
    return folder/'report.html'


def export_results(output):
    output=Path(output);archive=output/'important_outcomes_followup.zip';temp=archive.with_suffix('.pending')
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(output.rglob('*')):
            if not p.is_file() or p in (archive,temp) or '.pending' in p.name or p.suffix=='.pt':continue
            rel=p.relative_to(output)
            if 'checkpoints' in rel.parts or 'shards' in rel.parts or p.suffix=='.zip':continue
            z.write(p,Path('results')/rel)
        for p in sorted(ROOT.iterdir()):
            if p.is_file() and p.suffix in ('.py','.html','.md','.json','.txt','.ipynb'):z.write(p,Path('code')/p.name)
        for p in sorted((ROOT/'inputs/data').glob('*.json')):z.write(p,Path('inputs/data')/p.name)
        for p in sorted((ROOT/'inputs/memory').rglob('*')):
            if p.is_file():z.write(p,Path('inputs/memory')/p.relative_to(ROOT/'inputs/memory'))
    temp.replace(archive);return archive


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path);p.add_argument('--export',action='store_true');a=p.parse_args()
    out=a.output or ROOT/read_json(ROOT/'outputs/latest.json')['relative_output']
    print(create_report(out,full=True))
    if a.export:print(export_results(out))
