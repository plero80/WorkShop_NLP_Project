"""Resumable matched evaluation; every score covers the entire generated answer."""
from pathlib import Path
import time
import numpy as np
import pandas as pd
from common import canonical_hash,read_json,write_json,file_hash,seed_for,StopRequested


def save_npz(path,**arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.pending')
    with open(temporary,'wb') as f:np.savez_compressed(f,**arrays)
    temporary.replace(path)


def evaluate(actor,reward,judge,rows,folder,c,identity,policy_id,branch,seed,checkpoint_hash,
             cohort,family,cap,update=200,save_features=False,should_stop=lambda:False,progress=lambda **kw:None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    signature={'experiment':identity,'policy_id':policy_id,'branch':branch,'seed':seed,'checkpoint':checkpoint_hash,
        'cohort':cohort,'family':family,'cap':cap,'update':update,'prompts':canonical_hash(rows),
        'batch_size':c['generation_batch_size'],'eval_seed':c['eval_seed'],'full_reward_scoring':True,
        'reward_context_guard':c['reward_max_tokens'],'save_features':save_features,
        'reward_lock':reward.lock_hash if branch in ('refreshed_stable','iterative_knn','iterative_capped') else None}
    sid=canonical_hash(signature)
    marker=folder/'complete.json'
    if marker.is_file():
        done=read_json(marker)
        if done['identity']!=sid or file_hash(folder/'predictions.csv')!=done['csv_sha256']:
            raise ValueError('Changed evaluation identity or predictions: '+str(folder))
        if save_features and file_hash(folder/'features.npz')!=done['features_sha256']:
            raise ValueError('Changed development embeddings.')
        return folder/'predictions.csv'
    if (folder/'identity.json').is_file() and read_json(folder/'identity.json')!=signature:
        raise ValueError('Cannot combine batches from a changed policy, cohort or configuration.')
    write_json(folder/'identity.json',signature)
    all_rows=[];all_features=[];old_cap=actor.c['max_new_tokens'];actor.c={**actor.c,'max_new_tokens':cap}
    try:
        for start in range(0,len(rows),c['generation_batch_size']):
            if should_stop():raise StopRequested('Paused at an evaluation batch boundary.')
            batch=rows[start:start+c['generation_batch_size']]
            shard=folder/'shards'/f'{start:06d}.json';feat=shard.with_suffix('.npz')
            if shard.is_file():
                saved=read_json(shard)
                if saved['identity']!=sid or saved['content_sha256']!=canonical_hash(saved['rows']):
                    raise ValueError('Changed evaluation batch.')
                records=saved['rows']
                if [r['prompt_id'] for r in records]!=[r['prompt_id'] for r in batch]:
                    raise ValueError('Evaluation batch prompt alignment changed.')
                if save_features:
                    if file_hash(feat)!=saved['features_sha256']:raise ValueError('Changed feature batch.')
                    with np.load(feat,allow_pickle=False) as f:all_features.append(f['vectors'])
            else:
                prompts=[r['prompt'] for r in batch]
                # Cap and method deliberately excluded: matching evaluation starts
                # from the same random stream within each cohort and batch.
                generated=actor.generate(prompts,seed_for(c['eval_seed'],cohort,start))
                answers=[a for p in generated for a in p['answers']]
                lengths=[int(n) for p in generated for n in p['response_mask'].sum(1)]
                eos=[v for p in generated for v in p['ended_eos']]
                scores=reward.score(prompts,answers,branch,features=save_features)
                j=judge.score(prompts,answers)
                if np.any(scores['reward_truncated']) or np.any(j['truncated']):
                    raise RuntimeError('Truncated reward inputs cannot enter this experiment.')
                zj=(j['raw']-reward.calibration['judge_mean'])/reward.calibration['judge_std']
                gap=scores['proxy_z']-zj;records=[]
                for i,row in enumerate(batch):
                    records.append({**row,'answer':answers[i],'policy_id':policy_id,'branch':branch,'seed':seed,
                        'update':update,'family':family,'cohort':cohort,'cap':cap,
                        'proxy_raw':float(scores['proxy_raw'][i]),'proxy_z':float(scores['proxy_z'][i]),
                        'judge_raw':float(j['raw'][i]),'judge_z':float(zj[i]),
                        'actual_proxy_judge_gap':float(gap[i]),'high_gap':bool(gap[i]>reward.calibration['theta']),
                        'gap_hat':float(scores['gap_hat'][i]),'original_gap_hat':float(scores['original_gap_hat'][i]),
                        'applied_gap':float(scores['applied_gap'][i]),'optimization_reward_z':float(scores['reward'][i]),
                        'corrected_judge_residual':float(scores['reward'][i]-zj[i]),
                        'mean_neighbor_distance':float(scores['mean_neighbor_distance'][i]),
                        'within_old_distance_gate':bool(scores['within_distance_gate'][i]),
                        'response_tokens':lengths[i],'ended_eos':bool(eos[i]),
                        'proxy_input_tokens':int(scores['reward_tokens'][i]),'judge_input_tokens':int(j['tokens'][i]),
                        'proxy_truncated':False,'judge_truncated':False})
                payload={'identity':sid,'rows':records,'content_sha256':canonical_hash(records)}
                if save_features:
                    save_npz(feat,vectors=scores['features']);payload['features_sha256']=file_hash(feat)
                    all_features.append(scores['features'])
                write_json(shard,payload)
                print(f'{family} {policy_id} {cohort} cap={cap}: {start+len(batch)}/{len(rows)}',flush=True)
            all_rows.extend(records)
            progress(policy=policy_id,cohort=cohort,cap=cap,done=start+len(batch),total=len(rows))
    finally:actor.c={**actor.c,'max_new_tokens':old_cap}
    temporary=folder/'predictions.csv.pending';pd.DataFrame(all_rows).to_csv(temporary,index=False)
    temporary.replace(folder/'predictions.csv')
    done={'identity':sid,'signature':signature,'rows':len(all_rows),'csv_sha256':file_hash(folder/'predictions.csv'),
          'finished_unix':time.time()}
    if save_features:
        save_npz(folder/'features.npz',vectors=np.concatenate(all_features))
        done['features_sha256']=file_hash(folder/'features.npz')
    write_json(marker,done)
    return folder/'predictions.csv'


def load_completed(folder,features=False):
    folder=Path(folder);done=read_json(folder/'complete.json')
    if done['identity']!=canonical_hash(done['signature']) or file_hash(folder/'predictions.csv')!=done['csv_sha256']:
        raise ValueError('Evaluation completion/CSV checksum mismatch.')
    frame=pd.read_csv(folder/'predictions.csv',keep_default_na=False)
    if not features:return frame
    if file_hash(folder/'features.npz')!=done['features_sha256']:raise ValueError('Feature checksum mismatch.')
    with np.load(folder/'features.npz',allow_pickle=False) as f:vec=f['vectors'].copy()
    if len(frame)!=len(vec):raise ValueError('Features and rows do not align.')
    return frame,vec
