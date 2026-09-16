"""One development-only memory refresh; parameters locked before test/PPO."""
from pathlib import Path
import itertools
import numpy as np
import pandas as pd
from common import read_json,write_json,file_hash,canonical_hash
from evaluation import load_completed,save_npz
from reward_bridge import predict_memory,bounded_gap


def source_bank(output,cohort,seed):
    parts=[];vectors=[];identities=[]
    for branch in ['raw','knn_signed']:
        folder=Path(output)/'evaluations/development'/cohort/f'cap_256/legacy_{branch}_s{seed}'
        frame,vec=load_completed(folder,features=True);parts.append(frame);vectors.append(vec)
        identities.append(read_json(folder/'complete.json')['identity'])
    frame=pd.concat(parts,ignore_index=True)
    if frame.groupby('prompt_id').size().nunique()!=1 or frame.groupby('prompt_id').size().iloc[0]!=2:
        raise ValueError('Development bank needs exactly two policy outputs per prompt.')
    return frame,np.concatenate(vectors),identities


def prompt_mse(frame,residual):
    return float(pd.Series(np.asarray(residual)**2).groupby(frame.prompt_id.to_numpy()).mean().mean())


def controls(frame):
    return np.column_stack([np.ones(len(frame)),frame.proxy_z,np.log1p(frame.response_tokens),frame.ended_eos.astype(float)])


def fit_lock(output,reward,c):
    folder=Path(output)/'improvement';folder.mkdir(parents=True,exist_ok=True)
    fit,fv,fi=source_bank(output,'development_fit',c['development_policy_seed'])
    val,vv,vi=source_bank(output,'development_validation',c['development_policy_seed'])
    if set(fit.conversation_group)&set(val.conversation_group):raise ValueError('Development group leakage.')
    dependency={'fit':fi,'validation':vi,'alphas':c['selection_alphas'],'caps':c['selection_bonus_caps'],
                'gates':c['selection_distance_gates'],'old_memory':file_hash(reward.corrector.result_dir/'detectors/gap_knn.npz')}
    lock_path=folder/'locked_reward.json'
    if lock_path.is_file():
        if read_json(lock_path)['dependency']!=dependency:raise ValueError('Locked reward dependencies changed.')
        reward.load_update(folder);return folder
    base=reward.corrector.model
    memory={'vectors':np.concatenate([base['vectors'],fv]),
            'gaps':np.concatenate([base['gaps'],fit.actual_proxy_judge_gap.to_numpy(float)])}
    save_npz(folder/'refreshed_memory.npz',**memory)
    gh,dist=predict_memory(vv,memory,threads=c['cpu_threads'])
    candidates=[{'alpha':0.,'bonus_cap':0.,'distance_gate':False}]
    candidates += [dict(alpha=a,bonus_cap=b,distance_gate=g) for a,b,g in itertools.product(
        c['selection_alphas'],c['selection_bonus_caps'],c['selection_distance_gates'])]
    records=[]
    for parameters in candidates:
        applied=bounded_gap(gh,dist,parameters,reward.corrector.cutoff)
        residual=val.actual_proxy_judge_gap.to_numpy()-applied
        records.append({**parameters,'validation_mse':prompt_mse(val,residual),
            'validation_mean_boost':float(np.maximum(-applied,0).mean()),
            'high_gap_mean_boost':float(np.maximum(-applied[val.high_gap.to_numpy()],0).mean()) if val.high_gap.any() else None})
    # Deterministic ties favor smaller coefficient and smaller bonus.
    selected=min(records,key=lambda r:(r['validation_mse'],r['alpha'],r['bonus_cap'],r['distance_gate']))
    parameters={k:selected[k] for k in ('alpha','bonus_cap','distance_gate')}
    x=controls(fit);y=fit.actual_proxy_judge_gap.to_numpy()
    # These are diagnostic baselines; neither competes for the PPO arm.
    affine=np.linalg.lstsq(x[:,:2],y,rcond=None)[0]
    penalty=np.eye(4)*1e-3;penalty[0,0]=0
    length_eos=np.linalg.solve(x.T@x+penalty,x.T@y)
    pd.DataFrame(records).to_csv(folder/'validation_search.csv',index=False)
    write_json(lock_path,{'dependency':dependency,'selected':parameters,'selection_metric':'validation prompt-weighted MSE',
        'validation_mse':selected['validation_mse'],'memory_sha256':file_hash(folder/'refreshed_memory.npz'),
        'old_memory_rows':len(base['gaps']),'new_memory_rows':len(fit),'fit_prompts':int(fit.prompt_id.nunique()),
        'validation_prompts':int(val.prompt_id.nunique()),'k':31,'temperature':.05,
        'distance_cutoff':reward.corrector.cutoff,'affine_coefficients':affine.tolist(),'length_eos_coefficients':length_eos.tolist(),
        'test_labels_used':False,'note':'No tuning on offline_test, fresh_final, legacy_eval or PPO outcomes.'})
    reward.load_update(folder)
    return folder


def offline_audit(output,reward,c):
    folder=Path(output)/'improvement';lock=read_json(folder/'locked_reward.json')
    test,features,ids=source_bank(output,'offline_test',c['development_policy_seed'])
    gh,dist=predict_memory(features,reward.updated,threads=c['cpu_threads'])
    gap=test.actual_proxy_judge_gap.to_numpy();x=controls(test)
    predictions={'raw':np.zeros(len(test)),'frozen_signed':test.original_gap_hat.to_numpy(),
        'refreshed_signed_unbounded':gh,
        'refreshed_stable':bounded_gap(gh,dist,lock['selected'],reward.corrector.cutoff),
        'affine_proxy':x[:,:2]@np.asarray(lock['affine_coefficients']),
        'proxy_length_eos':x@np.asarray(lock['length_eos_coefficients'])}
    records=[];details=test[['prompt_id','policy_id','proxy_z','judge_z','high_gap','response_tokens','ended_eos']].copy()
    for name,applied in predictions.items():
        error=gap-applied;details[name+'_residual']=error
        records.append({'method':name,'prompts':int(test.prompt_id.nunique()),'responses':len(test),
            'judge_mse':prompt_mse(test,error),'mean_reward_boost':float(np.maximum(-applied,0).mean()),
            'boosted_fraction':float((applied<0).mean()),
            'high_gap_boosted_fraction':float((applied[test.high_gap.to_numpy()]<0).mean()) if test.high_gap.any() else None})
    pd.DataFrame(records).to_csv(folder/'offline_test_summary.csv',index=False)
    details.to_csv(folder/'offline_test_predictions.csv',index=False)
    write_json(folder/'offline_test_audit.json',{'lock_sha256':file_hash(folder/'locked_reward.json'),
        'test_evaluation_identities':ids,'no_parameter_changes':True})


def round_refresh(output,seed,reward,c,parent_checkpoint_hash):
    """Append teacher-labeled outputs of the exact shared round-one policy.

    The main iterative arm uses unbounded signed correction, identical to static
    apart from memory contents. Validation selects ONLY the capped ablation.
    """
    output=Path(output);policy=f'round1_knn_signed_s{seed}'
    fit_folder=output/'evaluations/refresh/refresh_round1/cap_256'/policy
    val_folder=output/'evaluations/refresh_validation/development_validation/cap_256'/policy
    fit,fv=load_completed(fit_folder,features=True);val,vv=load_completed(val_folder,features=True)
    if set(fit.conversation_group)&set(val.conversation_group):raise ValueError('Refresh/validation group overlap.')
    dependency={'fit':read_json(fit_folder/'complete.json')['identity'],
                'validation':read_json(val_folder/'complete.json')['identity'],
                'parent_checkpoint':parent_checkpoint_hash,'seed':seed,
                'old_memory':file_hash(reward.corrector.result_dir/'detectors/gap_knn.npz')}
    folder=output/'refresh'/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'locked_reward.json').is_file():
        if read_json(folder/'locked_reward.json')['dependency']!=dependency:raise ValueError('Refresh dependencies changed.')
        reward.load_update(folder);return folder
    base=reward.corrector.model
    memory={'vectors':np.concatenate([base['vectors'],fv]),
            'gaps':np.concatenate([base['gaps'],fit.actual_proxy_judge_gap.to_numpy(float)])}
    save_npz(folder/'refreshed_memory.npz',**memory)
    gh,distance=predict_memory(vv,memory,threads=c['cpu_threads'])
    params=[{'alpha':0.,'bonus_cap':0.,'distance_gate':False}]+[
        dict(alpha=a,bonus_cap=b,distance_gate=g) for a,b,g in itertools.product(
            c['selection_alphas'],c['selection_bonus_caps'],c['selection_distance_gates'])]
    records=[]
    for p in params:
        applied=bounded_gap(gh,distance,p,reward.corrector.cutoff)
        records.append({**p,'validation_mse':prompt_mse(val,val.actual_proxy_judge_gap.to_numpy()-applied)})
    selected=min(records,key=lambda r:(r['validation_mse'],r['alpha'],r['bonus_cap'],r['distance_gate']))
    pd.DataFrame(records).to_csv(folder/'capped_validation_search.csv',index=False)
    write_json(folder/'locked_reward.json',{'dependency':dependency,'selected':{k:selected[k] for k in ['alpha','bonus_cap','distance_gate']},
        'memory_sha256':file_hash(folder/'refreshed_memory.npz'),'k':31,'temperature':.05,
        'old_memory_rows':len(base['gaps']),'new_memory_rows':len(fit),'refresh_high_gap_rows':int(fit.high_gap.sum()),
        'refresh_negative_gap_rows':int((fit.actual_proxy_judge_gap<0).sum()),
        'distance_cutoff':reward.corrector.cutoff,'calibration':reward.calibration,
        'main_iterative_correction':{'alpha':1.,'bonus_cap':None,'distance_gate':False},
        'main_parameters_changed':False,'capped_parameters_selection':'Separate validation prompts only',
        'teacher_query_selection':'All outputs on fixed random refresh prompts, not only positive gaps',
        'test_labels_used':False})
    reward.load_update(folder);return folder
