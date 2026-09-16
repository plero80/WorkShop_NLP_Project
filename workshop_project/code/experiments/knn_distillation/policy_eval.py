"""Matched, resumable evaluation. Monitoring prompts never enter memory."""
from pathlib import Path
import time
import numpy as np,pandas as pd
from knn_distillation.io import read,write,sha,digest,should_stop,status

def load_completed(folder,features=False):
    folder=Path(folder);done=read(folder/'complete.json')
    if done['identity']!=digest(done['signature']) or sha(folder/'predictions.csv')!=done['csv_sha256']:raise ValueError('Evaluation checksum mismatch.')
    f=pd.read_csv(folder/'predictions.csv',keep_default_na=False,dtype={'prompt_id':str})
    if len(f)!=done['rows'] or f.prompt_id.duplicated().any():raise ValueError('Evaluation row alignment error.')
    if not features:return f
    if sha(folder/'features.npz')!=done['features_sha256']:raise ValueError('Feature checksum mismatch.')
    with np.load(folder/'features.npz',allow_pickle=False) as z:v=z['vectors'].copy()
    if len(v)!=len(f):raise ValueError('Features are not aligned.')
    return f,v

def sample_kl(actor,parts,c):
    import torch
    vals=[]
    with torch.inference_mode():
        for p in parts:
            for s in range(0,len(p['ids']),c['micro_batch_size']):
                sl=slice(s,s+c['micro_batch_size']);ids=p['ids'][sl].to(actor.device_name);att=p['attention'][sl].to(actor.device_name)
                a,_=actor.statistics(ids,att,p['prompt_width'],with_values=False)
                b,_=actor.statistics(ids,att,p['prompt_width'],reference=True,with_values=False)
                mask=p['response_mask'][sl].to(actor.device_name)
                vals.extend(((a-b)*mask).sum(1).cpu().tolist())
    return vals

def evaluate(actor,reward,rows,folder,c,identity,seed,branch,checkpoint,cohort,update,features=False,monitor_kl=True,stop_out=None):
    from common import seed_for,StopRequested
    from evaluation import save_npz
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    signature={'experiment':identity,'seed':seed,'branch':branch,'checkpoint':checkpoint,'cohort':cohort,'update':update,
        'prompts':digest(rows),'cap':c['max_new_tokens'],'reward_guard':c['reward_max_tokens'],'batch':c['generation_batch_size'],'eval_seed':c['eval_seed'],
        'features':features,'kl':monitor_kl,'memory':reward.memory_hash,'students':reward.student_ids}
    sid=digest(signature);marker=folder/'complete.json'
    if marker.exists():
        if read(marker)['identity']!=sid:raise ValueError('Changed evaluation identity.')
        load_completed(folder,features);return folder
    if (folder/'identity.json').exists() and read(folder/'identity.json')!=signature:raise ValueError('Changed evaluation dependencies.')
    write(folder/'identity.json',signature);allrows=[];vectors=[];start_time=time.monotonic()
    for start in range(0,len(rows),c['generation_batch_size']):
        if stop_out is not None and should_stop(stop_out):raise StopRequested('Paused at an evaluation batch boundary.')
        batch=rows[start:start+c['generation_batch_size']];shard=folder/'shards'/f'{start:06d}.json';vf=shard.with_suffix('.npz')
        if shard.exists():
            s=read(shard)
            if s['identity']!=sid or s['row_hash']!=digest(s['rows']) or [r['prompt_id'] for r in s['rows']]!=[r['prompt_id'] for r in batch]:raise ValueError('Evaluation shard alignment changed.')
            records=s['rows']
            if features:
                if sha(vf)!=s['features_sha256']:raise ValueError('Feature shard changed.')
                with np.load(vf,allow_pickle=False) as f:vectors.append(f['vectors'])
        else:
            prompts=[r['prompt'] for r in batch]
            # Seed is independent of method, policy seed and checkpoint: common random numbers.
            parts=actor.generate(prompts,seed_for(c['eval_seed'],cohort,start))
            answers=[a for p in parts for a in p['answers']]
            lengths=[int(x) for p in parts for x in p['response_mask'].sum(1)]
            eos=[bool(x) for p in parts for x in p['ended_eos']]
            kl=sample_kl(actor,parts,c) if monitor_kl else [None]*len(batch)
            proxy,judge,scores=reward.evaluate_all(prompts,answers,branch)
            zj=(judge['raw']-reward.calibration['judge_mean'])/reward.calibration['judge_std'];gap=scores['proxy_z']-zj
            records=[]
            for i,row in enumerate(batch):
                records.append({**row,'answer':answers[i],'branch':branch,'seed':seed,'update':update,'cohort':cohort,
                    'proxy_raw':float(proxy['raw'][i]),'judge_raw':float(judge['raw'][i]),'proxy_z':float(scores['proxy_z'][i]),'judge_z':float(zj[i]),
                    'actual_proxy_judge_gap':float(gap[i]),'high_gap':bool(gap[i]>reward.calibration['theta']),
                    'optimization_reward_z':float(scores['reward'][i]),'gap_hat':float(scores['gap_hat'][i]),
                    'knn_z':float(scores['knn_z'][i]),
                    'student_z':float(scores['student_z'][i]) if scores['student_z'] is not None else None,
                    'judge_student_z':float(scores['judge_student_z'][i]) if scores['judge_student_z'] is not None else None,
                    'corrected_judge_residual':float(scores['reward'][i]-zj[i]),'mean_neighbor_distance':float(scores['mean_neighbor_distance'][i]),
                    'response_tokens':lengths[i],'ended_eos':eos[i],'sequence_kl_sample':kl[i],
                    'proxy_input_tokens':int(proxy['tokens'][i]),'judge_input_tokens':int(judge['tokens'][i]),'proxy_truncated':False,'judge_truncated':False})
            s={'identity':sid,'rows':records,'row_hash':digest(records)}
            if features:
                save_npz(vf,vectors=proxy['features']);s['features_sha256']=sha(vf);vectors.append(proxy['features'])
            write(shard,s)
            print(f'{cohort} {branch} s{seed} update {update}: {start+len(batch)}/{len(rows)}',flush=True)
        allrows.extend(records)
        if stop_out is not None:status(stop_out,'evaluating',cohort=cohort,branch=branch,seed=seed,update=update,answers=start+len(batch),total=len(rows))
    tmp=folder/'predictions.csv.pending';pd.DataFrame(allrows).to_csv(tmp,index=False);tmp.replace(folder/'predictions.csv')
    done={'identity':sid,'signature':signature,'rows':len(allrows),'csv_sha256':sha(folder/'predictions.csv'),'last_invocation_seconds':time.monotonic()-start_time}
    if features:save_npz(folder/'features.npz',vectors=np.concatenate(vectors));done['features_sha256']=sha(folder/'features.npz')
    write(marker,done);return folder
