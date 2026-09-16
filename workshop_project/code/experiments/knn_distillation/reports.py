"""Descriptive curves and prespecified final comparisons; never chooses training settings."""
from pathlib import Path
import numpy as np,pandas as pd
from knn_distillation.io import read,write
from knn_distillation.policy_eval import load_completed

def summarize(f):
    gap=f.actual_proxy_judge_gap.to_numpy(float);high=f.high_gap.to_numpy(bool)
    result={'responses':len(f),'proxy_z':float(f.proxy_z.mean()),'judge_z':float(f.judge_z.mean()),
            'proxy_std':float(f.proxy_z.std()),'judge_std':float(f.judge_z.std()),'gap_mean':float(gap.mean()),'high_gap_rate':float(high.mean()),
            'gap_q95':float(np.quantile(gap,.95)),'gap_q99':float(np.quantile(gap,.99)),
            'high_gap_severity':float(gap[high].mean()) if high.any() else None,'response_tokens':float(f.response_tokens.mean()),
            'eos_rate':float(f.ended_eos.mean()),'corrected_judge_mse':float((f.corrected_judge_residual**2).mean())}
    if 'sequence_kl_sample' in f:
        x=pd.to_numeric(f.sequence_kl_sample,errors='coerce');result['sampled_sequence_kl']=float(x.mean()) if x.notna().any() else None
    return result

def frames(out,cohort):
    result={}
    for p in sorted((Path(out)/'evaluations'/cohort).glob('seed_*/*/update_*/complete.json')):
        d=read(p);s=d['signature'];f=load_completed(p.parent)
        result[(s['seed'],s['branch'],s['update'])]=f
    return result

def plotting(table,dest,name):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11,'svg.fonttype':'none','pdf.fonttype':42})
    methods=list(dict.fromkeys(table.branch));fig,axs=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    for metric,ax,title in zip(['proxy_z','judge_z','high_gap_rate','response_tokens','eos_rate','sampled_sequence_kl'],axs.flat,
                              ['Proxy score','Judge score','High-gap rate','Response tokens','EOS rate','Sampled sequence KL to base']):
        for branch in methods:
            df=table[table.branch==branch].groupby('update')[metric].agg(['mean','std','count'])
            if not df['mean'].notna().any():continue
            ax.plot(df.index,df['mean'],marker='o',ms=3,label=branch)
            if (df['count']>1).any():ax.fill_between(df.index,df['mean']-df['std'],df['mean']+df['std'],alpha=.12)
        ax.set_title(title);ax.set_xlabel('PPO updates (not compute)');ax.grid(alpha=.2)
    axs[0,0].legend(fontsize=9);fig.suptitle('Same monitoring prompts at each checkpoint; shading = ±1 seed SD, not a confidence interval')
    for ext in ['png','pdf','svg']:fig.savefig(dest/(name+'.'+ext),dpi=180)
    plt.close(fig)
    # Proxy and judge share a panel per arm, to expose divergence directly.
    fig,axes=plt.subplots(1,len(methods),figsize=(5*len(methods),4),squeeze=False,layout='constrained')
    for branch,ax in zip(methods,axes.flat):
        d=table[table.branch==branch].groupby('update')[['proxy_z','judge_z']].mean()
        ax.plot(d.index,d.proxy_z,'--o',label='proxy z',ms=3);ax.plot(d.index,d.judge_z,'-o',label='judge z',ms=3)
        ax.set(title=branch,xlabel='PPO updates',ylabel='Fixed-calibration score');ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Observed proxy–judge trajectories; no assumed turning point')
    for ext in ['png','pdf','svg']:fig.savefig(dest/('proxy_judge_trajectories.'+ext),dpi=180)
    plt.close(fig)

def monitoring_report(out,cohort):
    out=Path(out);rows=[]
    for (seed,branch,update),f in frames(out,cohort).items():rows.append({'seed':seed,'branch':branch,'update':update,**summarize(f)})
    if not rows:return
    dest=out/'reports';dest.mkdir(exist_ok=True);table=pd.DataFrame(rows);table.to_csv(dest/'monitoring_by_seed.csv',index=False)
    table.groupby(['branch','update']).mean(numeric_only=True).reset_index().to_csv(dest/'monitoring_seed_means.csv',index=False)
    plotting(table,dest,'monitoring_curves')
    # A descriptive sampled maximum is not a formal C2 estimate or early-stop rule.
    peaks=[]
    for branch,df in table.groupby('branch'):
        avg=df.groupby('update')[['judge_z','proxy_z']].mean();idx=int(avg.judge_z.idxmax())
        peaks.append({'branch':branch,'best_observed_monitor_update':idx,'judge_at_sampled_maximum':float(avg.loc[idx,'judge_z']),
                      'later_checkpoints_observed':int((avg.index>idx).sum()),'formal_C2_established':False})
    write(dest/'descriptive_peaks.json',{'records':peaks,'note':'A noisy sampled maximum, especially at an endpoint, is not an established overoptimization threshold. Monitoring does not select training duration or final-test endpoints.'})

def costs(out):
    records=[]
    for p in sorted(Path(out).glob('runs/**/history.json')):
        dep=read(p.parent/'segment.json');h=[r for r in read(p) if r.get('segment_start')==dep['start'] and r.get('reward_source')==dep['branch']]
        if not h:continue
        rows={'seed':dep['seed'],'branch':dep['branch'],'segment_start':dep['start'],'completed_PPO_updates':len(h),
              'PPO_seconds':sum(x['seconds'] for x in h),'PPO_generated_tokens':sum(x['mean_response_tokens']*len(x['prompt_ids']) for x in h)}
        for metric in ['teacher_answers','proxy_answers','student_answers','knn_queries','teacher_input_tokens','proxy_input_tokens','student_input_tokens']:rows['PPO_'+metric]=sum(x['model_calls'].get(metric,0) for x in h)
        records.append(rows)
    evals=[]
    for p in (Path(out)/'evaluations').glob('**/complete.json'):
        d=read(p);s=d['signature'];evals.append({'seed':s['seed'],'branch':s['branch'],'cohort':s['cohort'],'update':s['update'],'teacher_answers':d['rows'],'proxy_answers':d['rows']})
    pd.DataFrame(records).to_csv(Path(out)/'reports/training_costs.csv',index=False)
    pd.DataFrame(evals).to_csv(Path(out)/'reports/scoring_counts.csv',index=False)
    write(Path(out)/'reports/cost_interpretation.json',{'matched_budget':'PPO update and rollout counts, not equal GPU compute',
      'counts':'Completed work only. Preflight, discarded/repeated work and teacher model pretraining are excluded.',
      'reward_routing':'Student PPO makes only student forwards. Proxy and kNN arms use their named frozen reward. The large judge is never called in PPO.',
      'memory_labels':'Label-generation counts are in labels/*/*/complete.json. Main train/validation pseudo-labels use zero new large-judge calls unless the optional direct-judge control is enabled.',
      'historical_memory':'The frozen source memory and its prior large-judge costs precede this study and are not included in new-work counters.'})

def paired_stats(a,b,seed,branch,update,draws):
    a=a.set_index('prompt_id');b=b.set_index('prompt_id')
    if set(a.index)!=set(b.index):raise ValueError('Final prompt sets differ.')
    b=b.loc[a.index]
    if not a.prompt.equals(b.prompt):raise ValueError('Final prompt contexts differ.')
    return pd.DataFrame({'prompt_id':a.index,'seed':seed,'comparison':branch,'update':update,
            'judge_delta':b.judge_z.to_numpy()-a.judge_z.to_numpy(),
            'proxy_delta':b.proxy_z.to_numpy()-a.proxy_z.to_numpy(),
            'high_gap_delta':b.high_gap.astype(float).to_numpy()-a.high_gap.astype(float).to_numpy(),
            'length_delta':b.response_tokens.to_numpy()-a.response_tokens.to_numpy()})

def final_report(out,c,cohort,reference,others,o,oracle=False):
    from review import make_pack
    out=Path(out);dest=out/'reports';dest.mkdir(exist_ok=True)
    fs=frames(out,cohort);rows=[];pairs=[]
    for (seed,branch,update),f in fs.items():rows.append({'seed':seed,'branch':branch,'update':update,**summarize(f)})
    summary=pd.DataFrame(rows);summary.to_csv(dest/'final_by_seed.csv',index=False)
    summary.groupby(['branch','update']).mean(numeric_only=True).reset_index().to_csv(dest/'final_seed_means.csv',index=False)
    for seed in c['seeds']:
        updates=sorted({u for s,b,u in fs if s==seed and b==reference})
        for u in updates:
            for other in others:
                if (seed,other,u) in fs:pairs.append(paired_stats(fs[(seed,reference,u)],fs[(seed,other,u)],seed,other,u,c['bootstrap_draws']))
    paired=pd.concat(pairs,ignore_index=True);paired.to_csv(dest/'paired_answers.csv',index=False)
    metrics=['judge_delta','proxy_delta','high_gap_delta','length_delta'];paired.groupby(['comparison','update','seed'])[metrics].mean().reset_index().to_csv(dest/'paired_deltas_by_seed.csv',index=False)
    records=[];rng=np.random.default_rng(91727)
    for (comparison,u),df in paired.groupby(['comparison','update']):
        avg=df.groupby('prompt_id')[metrics].mean()
        for metric in metrics:
            values=avg[metric].to_numpy();boot=[]
            for _ in range(c['bootstrap_draws']):boot.append(values[rng.integers(0,len(values),len(values))].mean())
            lo,hi=np.quantile(boot,[.025,.975]);records.append({'comparison':comparison+' minus '+reference,'update':int(u),'metric':metric,'mean':float(values.mean()),'low':float(lo),'high':float(hi)})
    write(dest/'conditional_intervals.json',{'records':records,'note':'Seeds averaged within prompt before prompt bootstrap. Conditional on the configured trained seeds; intervals are not adjusted for multiple metrics/endpoints. Prespecified final update is primary.'})
    last=max(u for s,b,u in fs if b==reference);review_frames={('raw' if b==reference else b,s,256):f for (s,b,u),f in fs.items() if u==last and b in [reference]+others}
    review=make_pack(out,'final_blinded_review',review_frames,others,c)
    key=read(review/'private/key.json');key['reference_method']=reference;write(review/'private/key.json',key)
    if oracle:
        means=summary[summary['update']==last].groupby('branch').judge_z.mean();den=float(means['judge']-means['proxy'])
        value=float((means['knn']-means['proxy'])/den) if den>.01 else None
        write(dest/'teacher_gain_fraction_exploratory.json',{'ratio':value,'denominator_judge_minus_proxy':den,
             'note':'Exploratory point estimate only when teacher-minus-proxy exceeds .01 z. Can be negative or exceed one; not a guaranteed bound, cost-normalized benefit, or uncertainty claim.'})
    costs(out)
    (dest/'READ_ME.txt').write_text('Primary comparison: student minus knn at the final configured PPO update. All arms begin at the same selected source policy/value/optimizer/RNG state. Do not choose the best final-test checkpoint. High gap is proxy–judge disagreement, not verified reward hacking. Complete the blinded usefulness/refusal review. Monitoring plots use held-out prompts; their x-axis is PPO updates, not compute. Seed bands are descriptive SD. Clusters do not affect any policy or reward.\n')
