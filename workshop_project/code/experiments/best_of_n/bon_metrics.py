"""Shared-pool selection and prompt-paired inference. No model dependencies."""
import csv
import numpy as np
from bon_io import write

SELECTORS = ('proxy', 'knn', 'judge')

def selection_rows(rows, sizes, epsilon=1e-8):
    rows = sorted(rows, key=lambda r:r['candidate_id'])
    if [r['candidate_id'] for r in rows] != list(range(max(sizes))):
        raise ValueError('Incomplete or duplicate candidate IDs.')
    if len({r['prompt_id'] for r in rows}) != 1: raise ValueError('Mixed prompt pool.')
    p = np.array([r['proxy_z'] for r in rows], float)
    j = np.array([r['judge_z'] for r in rows], float)
    k = np.array([r['corrected_z'] for r in rows], float)
    if not np.isfinite([p,j,k]).all(): raise ValueError('Nonfinite candidate rewards.')
    records = []
    for n in sizes:
        ix = {'proxy':int(p[:n].argmax()), 'knn':int(k[:n].argmax()), 'judge':int(j[:n].argmax())}
        # Stable lowest candidate index breaks ties, identically for every selector.
        delta = float(j[ix['knn']] - j[ix['proxy']])
        errors = p[:n]-j[:n]; corrected_errors = k[:n]-j[:n]
        left, right = np.triu_indices(n, 1)
        jdiff = j[left]-j[right]; valid = np.abs(jdiff) > epsilon
        def agreement(values):
            diff = values[left]-values[right]
            return float(np.mean(np.where(np.abs(diff[valid]) <= epsilon, .5,
                                (diff[valid]*jdiff[valid] > 0).astype(float)))) if valid.any() else None
        for selector, idx in ix.items():
            r = rows[idx]
            records.append({'prompt_id':r['prompt_id'], 'generator':r['generator'], 'n':n,
                'selector':selector, 'candidate_id':idx, 'judge_z':float(j[idx]), 'proxy_z':float(p[idx]),
                'corrected_z':float(k[idx]), 'judge_regret':float(j[:n].max()-j[idx]),
                'high_gap':int(r['actual_gap'] > r['theta']), 'response_tokens':r['response_tokens'],
                'answer_cap':int(r['answer_cap']), 'unique_answer_fraction':len({x['answer'] for x in rows[:n]})/n,
                'knn_changes_choice':int(ix['knn'] != ix['proxy']),
                'knn_changes_text':int(rows[ix['knn']]['answer'] != rows[ix['proxy']]['answer']),
                'knn_win':int(delta > epsilon), 'knn_loss':int(delta < -epsilon), 'knn_tie':int(abs(delta) <= epsilon),
                'proxy_judge_top_choice_agreement':int(ix['proxy'] == ix['judge']),
                'knn_judge_top_choice_agreement':int(ix['knn'] == ix['judge']),
                'proxy_pairwise_agreement':agreement(p), 'knn_pairwise_agreement':agreement(k),
                'raw_mse':float(np.mean(errors**2)), 'corrected_mse':float(np.mean(corrected_errors**2)),
                'raw_centered_mse':float(np.mean((errors-errors.mean())**2)),
                'corrected_centered_mse':float(np.mean((corrected_errors-corrected_errors.mean())**2))})
    return records

def interval(values, draws, seed):
    v = np.asarray(values, float)
    if v.ndim != 1 or not len(v) or not np.isfinite(v).all(): raise ValueError('Invalid bootstrap values.')
    rng = np.random.default_rng(seed); means = []
    for start in range(0, draws, 100):
        ix = rng.integers(0, len(v), size=(min(100, draws-start),len(v)))
        means.extend(v[ix].mean(axis=1))
    low, high = np.quantile(means, [.025,.975])
    return {'mean':float(v.mean()), 'ci_low':float(low), 'ci_high':float(high)}

def csv_write(path, rows):
    if not rows: raise ValueError('No report rows.')
    with open(path,'w',newline='',encoding='utf-8') as f:
        w = csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def report(output, settings, candidate_rows):
    from collections import defaultdict
    groups = defaultdict(list)
    for r in candidate_rows: groups[(r['generator'],r['prompt_id'])].append(r)
    selections = []
    for rows in groups.values(): selections.extend(selection_rows(rows, settings['pool_sizes']))
    folder = output/'reports'; folder.mkdir(parents=True, exist_ok=True)
    csv_write(folder/'selected_answers.csv', selections)
    tables = defaultdict(list)
    for r in selections: tables[(r['generator'],r['n'],r['selector'])].append(r)
    summary = []
    metrics = ['judge_z','proxy_z','judge_regret','high_gap','response_tokens','answer_cap','unique_answer_fraction',
               'knn_changes_choice','knn_changes_text','knn_win','knn_loss','knn_tie',
               'proxy_judge_top_choice_agreement','knn_judge_top_choice_agreement',
               'proxy_pairwise_agreement','knn_pairwise_agreement','raw_mse','corrected_mse',
               'raw_centered_mse','corrected_centered_mse']
    for (gen,n,selector), rs in sorted(tables.items()):
        item = {'generator':gen,'n':n,'selector':selector,'prompts':len(rs)}
        for m in metrics:
            vals = [r[m] for r in rs if r[m] is not None]
            item[m] = float(np.mean(vals)) if vals else None
        summary.append(item)
    csv_write(folder/'summary.csv',summary)
    paired=[]; diagnostic=[]
    for gen in settings['policies']:
        for n in settings['pool_sizes']:
            maps={s:{r['prompt_id']:r for r in tables[(gen,n,s)]} for s in SELECTORS}
            ids=sorted(maps['proxy'])
            if any(set(maps[s]) != set(ids) for s in SELECTORS): raise ValueError('Unpaired selector prompts.')
            headroom=np.array([maps['judge'][i]['judge_z']-maps['proxy'][i]['judge_z'] for i in ids])
            gain=np.array([maps['knn'][i]['judge_z']-maps['proxy'][i]['judge_z'] for i in ids])
            for comparator in ('knn','judge'):
                for metric in ('judge_z','judge_regret','high_gap','response_tokens'):
                    values=[maps[comparator][i][metric]-maps['proxy'][i][metric] for i in ids]
                    paired.append({'generator':gen,'n':n,'comparison':comparator+' minus proxy','metric':metric,
                        'prompts':len(ids),**interval(values,settings['bootstrap_draws'],settings['bootstrap_seed']),
                        'primary':gen==settings['primary_policy'] and n==settings['primary_n'] and comparator=='knn' and metric=='judge_z'})
            diagnostic.append({'generator':gen,'n':n,'judge_selection_headroom':float(headroom.mean()),
                'knn_judge_gain':float(gain.mean()),
                'headroom_recovered_fraction':float(gain.mean()/headroom.mean()) if headroom.mean() > .01 else None,
                'note':'Exploratory ratio; omitted if denominator <= 0.01. Can be negative. Judge ceiling is within this pool only.'})
    csv_write(folder/'paired_comparisons.csv',paired);write(folder/'headroom.json',diagnostic)
    primary=next(r for r in paired if r['primary']);write(folder/'primary_result.json',primary)
    # Save full readable selections for qualitative inspection without choosing examples by result.
    lookup={(r['generator'],r['prompt_id'],r['candidate_id']):r for r in candidate_rows}
    selected_text=[]
    for r in selections:
        if r['n']==settings['primary_n']:
            raw=lookup[(r['generator'],r['prompt_id'],r['candidate_id'])]
            selected_text.append({**r,'prompt':raw['prompt'],'answer':raw['answer']})
    csv_write(folder/'selected_text.csv',selected_text)
    (folder/'INTERPRETATION.md').write_text(
        '# Best-of-N selection evaluation\n\n'
        f"Primary: {settings['primary_policy']} generator, N={settings['primary_n']}, kNN minus proxy selected judge score.\n\n"
        f"Observed difference: {primary['mean']:.6f}; 95% paired prompt bootstrap interval "
        f"[{primary['ci_low']:.6f}, {primary['ci_high']:.6f}].\n\n"
        'All selectors see identical nested candidate pools. Selection ties use the first candidate. '
        'Judge selection is an exact ceiling only for this pool under this judge; it is not human ground truth. '
        'Judge evaluation calls for all candidates are research costs; proxy/kNN selectors do not consume these scores. '
        'The memory is frozen. No PPO, fitting, or refresh occurs in this experiment.\n\n'
        'Intervals are conditional on the fixed trained checkpoints and one sampled candidate pool per prompt. '
        'They do not estimate variability across all policy training seeds. Secondary N, generators, and metrics '
        'are exploratory; intervals are not multiplicity-adjusted. Positive lower primary bound supports better '
        'judge-based selection under this protocol, not automatically improved human usefulness or PPO.\n\n'
        'Centered MSE removes each prompt\'s mean residual before squaring, revealing whether improvement is '
        'mostly a prompt-level offset. Pairwise agreement excludes judge ties and gives predicted ties half credit; '
        'it is macro-averaged over prompts with at least one non-tied judge pair. N=1 has no pairwise metric.\n\n'
        'Development results may guide a new protocol; reserve confirmation for a locked protocol. '
        'Never tune on confirmation, cherry-pick favorable N, or treat candidate answers as independent prompts.\n',encoding='utf-8')
    plot(folder,settings,summary,paired)
    return primary

def plot(folder, settings, summary, paired):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(len(settings['policies']),2,figsize=(11,4*len(settings['policies'])),squeeze=False)
    colors={'proxy':'#B05B3B','knn':'#267E80','judge':'#5D5DAD'}
    for row,gen in enumerate(settings['policies']):
        for selector in SELECTORS:
            rs=sorted([r for r in summary if r['generator']==gen and r['selector']==selector],key=lambda r:r['n'])
            axes[row,0].plot([r['n'] for r in rs],[r['judge_z'] for r in rs],'-o',label=selector,color=colors[selector])
        rs=[r for r in paired if r['generator']==gen and r['metric']=='judge_z' and r['comparison']=='knn minus proxy']
        x=np.array([r['n'] for r in rs]);m=np.array([r['mean'] for r in rs]);lo=np.array([r['ci_low'] for r in rs]);hi=np.array([r['ci_high'] for r in rs])
        axes[row,1].plot(x,m,'-o',color=colors['knn']);axes[row,1].fill_between(x,lo,hi,alpha=.18,color=colors['knn'])
        axes[row,1].axhline(0,color='gray',lw=1);axes[row,0].legend()
        axes[row,0].set_ylabel('Selected answer: judge z');axes[row,1].set_ylabel('kNN minus proxy: judge z')
        for ax in axes[row]:
            ax.set_xscale('log',base=2);ax.set_xticks(settings['pool_sizes'],labels=settings['pool_sizes'])
            ax.set_xlabel('Candidate pool size N');ax.set_title(gen+' generator');ax.grid(alpha=.15)
    fig.suptitle('Frozen-memory best-of-N | paired prompt intervals',fontsize=14)
    fig.tight_layout();fig.savefig(folder/'selection_curves.png',dpi=180);fig.savefig(folder/'selection_curves.pdf');plt.close(fig)
