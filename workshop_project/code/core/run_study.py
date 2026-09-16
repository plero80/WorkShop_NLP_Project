"""Checkpoint recheck and two-round static-versus-iterative kNN PPO study."""
import argparse
import fcntl
import gc
import hashlib
import os
from pathlib import Path
import signal
import time
import traceback
import numpy as np
import torch
from assets import resolve_all,MODELS
from common import ROOT,read_json,write_json,file_hash,canonical_hash,output_root,validate_config,status,StopRequested,runtime_versions
from ppo_engine import PPOActor,PPOTrainer
from reward_bridge import RewardScorer,FrozenGapReward
from evaluation import evaluate,load_completed
from improvement import fit_lock,offline_audit,round_refresh
from reporting import create_report,export_results
from review import make_pack

STOP=False


def request_stop(signum,frame):
    global STOP
    STOP=True
    print('Pause requested. Saving at the next complete evaluation batch or PPO update.',flush=True)


def fingerprint(actor):
    from peft import get_peft_model_state_dict
    h=hashlib.sha256()
    for name,tensor in sorted(get_peft_model_state_dict(actor.policy).items()):
        h.update(name.encode());h.update(tensor.detach().float().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def load_actor(snapshot,c,seed,adapter=None):
    actor=PPOActor.load(snapshot,c,seed)
    if adapter is not None:
        from safetensors.torch import load_file
        from peft import set_peft_model_state_dict,get_peft_model_state_dict
        adapter=Path(adapter);config=read_json(adapter/'adapter_config.json')
        if config['r']!=c['lora_rank'] or config['lora_alpha']!=c['lora_alpha']:
            raise ValueError('Saved adapter architecture differs from the experiment.')
        state=load_file(str(adapter/'adapter_model.safetensors'),device='cpu')
        before=get_peft_model_state_dict(actor.policy)
        if set(state)!=set(before) or any(state[k].shape!=before[k].shape for k in state):
            raise ValueError('Saved adapter keys/shapes do not match the pinned base policy.')
        set_peft_model_state_dict(actor.policy,state)
        after=get_peft_model_state_dict(actor.policy)
        for k,v in state.items():torch.testing.assert_close(after[k].detach().cpu().float(),v.float(),atol=0,rtol=0)
    return actor


def release(*objects):
    # Caller must drop its references first; this clears allocator caches.
    gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()


def configure(c):
    torch.set_num_threads(c['cpu_threads'])
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable. Use the RTX PRO 6000 Python kernel.')
    if torch.cuda.get_device_capability()[0]>=12 and tuple(map(int,torch.version.cuda.split('.')[:2]))<(12,8):
        raise RuntimeError('Blackwell requires a compatible CUDA build. Use the notebook repair option once.')
    if not torch.cuda.is_bf16_supported():raise RuntimeError('BF16 support is required by the frozen scoring protocol.')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    x=torch.ones((32,32),device='cuda',dtype=torch.bfloat16)
    if not torch.isfinite(x@x).all():raise RuntimeError('GPU BF16 smoke test failed.')
    print('GPU:',torch.cuda.get_device_name(),'VRAM GiB:',round(torch.cuda.get_device_properties(0).total_memory/2**30,1),flush=True)


def load_data():
    # Input data is sealed independently of user-editable execution settings.
    seal=read_json(ROOT/'PACKAGE_MANIFEST.json')
    for name,expected in seal['input_sha256'].items():
        if file_hash(ROOT/name)!=expected:raise ValueError('Packaged input changed: '+name)
    data={p.stem:read_json(p) for p in (ROOT/'inputs/data').glob('*.json')}
    fresh=['development_fit','development_validation','offline_test','refresh_round1','fresh_final']
    from chat_format import prompt_messages
    import unicodedata
    for name in fresh:
        for row in data[name]:
            first=next(m['content'] for m in prompt_messages(row['prompt']) if m['role']=='user')
            computed=canonical_hash(' '.join(unicodedata.normalize('NFKC',first).casefold().split()))
            if computed!=row['conversation_group']:raise ValueError('Recorded conversation group differs from prompt text.')
    groups=[r['conversation_group'] for name in fresh for r in data[name]]
    forbidden=set(read_json(ROOT/'inputs/forbidden_opening_groups.json'))
    if len(groups)!=len(set(groups)) or set(groups)&forbidden:raise ValueError('Fresh conversation-group separation failed.')
    train={seed:read_json(ROOT/f'inputs/training/seed_{seed}.json') for seed in [42,43,44]}
    final_ids={r['prompt_id'] for r in data['fresh_final']}
    if any(final_ids&{r['prompt_id'] for r in rows} for rows in train.values()):raise ValueError('Final/PPO overlap.')
    return data,train


def load_rewards(c,assets):
    proxy=RewardScorer(assets['proxy'],c['reward_batch_size'],c['reward_max_tokens'])
    reward=FrozenGapReward(proxy,c['cpu_threads'])
    judge=RewardScorer(assets['judge'],c['reward_batch_size'],c['reward_max_tokens'])
    return reward,judge


def preflight(c,output,assets,data,train):
    import pandas as pd
    reward,judge=load_rewards(c,assets)
    parity=reward.validate_memory_encoder()
    bank=pd.read_csv(ROOT/'inputs/candidate_bank.csv',keep_default_na=False)
    eligible=[]
    for i in parity['bank_ids']:
        row=bank.iloc[i]
        try:_,n=judge.encode_full([row.prompt],[row.answer])
        except ValueError as e:
            if str(e).startswith('Full reward input needs'):continue
            raise
        if int(n[0])<=1024:eligible.append(i)
        if len(eligible)==8:break
    if len(eligible)<4:raise ValueError('Insufficient whole-answer judge references.')
    rows=bank.iloc[eligible];j=judge.score(rows.prompt.tolist(),rows.answer.tolist())
    diff=abs(j['raw']-rows.judge_raw.to_numpy())
    if diff.mean()>.05 or diff.max()>.5:raise RuntimeError('Pinned judge reference mismatch.')
    smoke={**c,'max_new_tokens':16,'generation_batch_size':4,'mini_batch_size':4,'micro_batch_size':2,'ppo_epochs':1}
    actor=load_actor(assets['policy'],smoke,42);trainer=PPOTrainer(actor,reward,smoke)
    before=fingerprint(actor);diagnostic=trainer.update(train[42][:4],'knn_signed',42,1)
    if fingerprint(actor)==before:raise RuntimeError('GPU PPO smoke update did not change the adapter.')
    del trainer,actor;release()
    actor=load_actor(assets['policy'],smoke,42,ROOT/'inputs/adapters/seed_42/knn_signed')
    _=actor.generate([r['prompt'] for r in train[42][:2]],19)
    write_json(output/'preflight.json',{'passed':True,'identity':read_json(output/'manifest.json')['identity'],
        'runtime_versions':runtime_versions(),'resolved_assets':assets,'memory_parity':parity,
        'judge_max_absolute_error':float(diff.max()),'discarded_gpu_smoke_update':diagnostic,
        'saved_adapter_loaded_and_generated':True,'whole_answer_scoring':True,'finished_unix':time.time()})
    del actor,reward,judge;release()
    print('PREFLIGHT PASSED: GPU, saved adapter, whole-answer scoring, memory/judge parity and PPO gradients.',flush=True)


def checked_checkpoint(path):
    path=Path(path);meta=read_json(path.with_suffix('.json'))
    if file_hash(path)!=meta['sha256']:raise ValueError('Checkpoint checksum changed: '+str(path))
    return meta


def checkpoint_actor(snapshot,c,seed,path,identity,branch):
    actor=load_actor(snapshot,c,seed);trainer=PPOTrainer(actor,None,c)
    trainer.restore(path,identity,seed,branch,optimizer=False)
    del trainer
    return actor


def train_segment(output,c,assets,train,reward,judge,seed,branch,start,end,should_stop,parent=None):
    identity=read_json(output/'manifest.json')['identity']
    round_name='round1' if start==0 else 'round2'
    folder=output/'runs'/round_name/f'seed_{seed}'/branch;ck=folder/'checkpoints';ck.mkdir(parents=True,exist_ok=True)
    parent_info=checked_checkpoint(parent) if parent is not None else None
    segment={'identity':identity,'seed':seed,'branch':branch,'start':start,'end':end,
             'parent_sha256':parent_info['sha256'] if parent_info else None,
             'reward_lock':reward.lock_hash if branch.startswith('iterative_') else None}
    marker=folder/'segment.json'
    if marker.is_file() and read_json(marker)!=segment:raise ValueError('Training fork/reward dependency changed.')
    write_json(marker,segment)
    final=ck/f'checkpoint_{end:06d}.pt'
    if (folder/'complete.json').is_file():
        done=read_json(folder/'complete.json');meta=checked_checkpoint(final)
        if done['checkpoint_sha256']!=meta['sha256'] or done['segment']!=segment:raise ValueError('Changed final training checkpoint.')
        return final
    actor=load_actor(assets['policy'],c,seed);trainer=PPOTrainer(actor,reward,c)
    candidates=sorted(ck.glob('checkpoint_*.json'))
    history=[];current=start
    if candidates:
        latest=read_json(candidates[-1]);current,history=trainer.restore(ck/latest['name'],identity,seed,branch)
    elif parent is not None:
        current,history=trainer.restore(parent,identity,seed,parent_info['branch'])
        if current!=start:raise ValueError('Round-two parent is not the declared round-one endpoint.')
        steps=sorted({int(v['step'].item()) for v in trainer.optimizer.state.values() if 'step' in v})
        write_json(folder/'fork_start.json',{'parent_checkpoint_sha256':parent_info['sha256'],
            'adapter_fingerprint':fingerprint(actor),'optimizer_step_counts':steps,
            'policy_value_optimizer_restored':True,'reference_policy':'unchanged original base with adapter disabled'})
    else:
        init=output/'initializations'/f'seed_{seed}.json';record={'seed':seed,'fingerprint':fingerprint(actor)}
        if init.is_file() and read_json(init)!=record:raise ValueError('Round-one initial policies differ.')
        write_json(init,record)
    if not start<=current<=end:raise ValueError('Resumed update is outside the segment.')
    try:
        for update in range(current+1,end+1):
            if should_stop():
                if current>start:trainer.checkpoint(ck,current,identity,seed,branch,history)
                raise StopRequested('Paused before the next PPO update.')
            rows=train[seed][(update-1)*c['rollout_batch_size']:update*c['rollout_batch_size']]
            if len(rows)!=c['rollout_batch_size']:raise ValueError('Insufficient matched PPO prompts.')
            record=trainer.update(rows,branch,seed,update)
            record['prompt_ids']=[r['prompt_id'] for r in rows];history.append(record);current=update
            write_json(folder/'history.json',history)
            if update%c['checkpoint_every']==0 or update==end or should_stop():
                trainer.checkpoint(ck,update,identity,seed,branch,history)
                previous=sorted(ck.glob('checkpoint_*.json'))
                for old in previous[:-2]:
                    info=read_json(old);(ck/info['name']).unlink();old.unlink()
            status(output,'training',round=round_name,branch=branch,seed=seed,update=update,target=end,
                   seconds_per_update=record['seconds'])
            print(f'{round_name} {branch} seed={seed} update={update}/{end} reward={record["mean_base_reward"]:.4f} seconds={record["seconds"]:.1f}',flush=True)
        actor.policy.save_pretrained(folder/'final_adapter',safe_serialization=True)
        final_meta=checked_checkpoint(final)
        write_json(folder/'final_adapter/provenance.json',segment|{'checkpoint_sha256':final_meta['sha256']})
        write_json(folder/'complete.json',{'segment':segment,'checkpoint_sha256':final_meta['sha256'],'finished_unix':time.time()})
    finally:
        del trainer,actor;release()
    return final


def run(c,output,assets,data,train):
    identity=read_json(output/'manifest.json')['identity'];start=time.monotonic()
    should_stop=lambda:STOP or (c['max_wall_hours']>0 and time.monotonic()-start>=c['max_wall_hours']*3600)
    reward,judge=load_rewards(c,assets)
    def ev(actor,policy,branch,seed,checkpoint,cohort,family,cap=256,features=False,update=200):
        folder=output/'evaluations'/family/cohort/f'cap_{cap}'/policy
        def progress(**kw):status(output,'evaluating',family=family,**kw)
        return evaluate(actor,reward,judge,data[cohort],folder,c,identity,policy,branch,seed,checkpoint,
                        cohort,family,cap,update,features,should_stop,progress)
    def legacy(branch,seed):
        path=ROOT/f'inputs/adapters/seed_{seed}/{branch}'
        return load_actor(assets['policy'],c,seed,path),file_hash(path/'adapter_model.safetensors')
    # Repeat both caps in this runtime; previous 128-token scores remain historical.
    for seed in c['seeds']:
        for branch in ['raw','knn_positive','knn_signed']:
            if should_stop():raise StopRequested('Paused before checkpoint evaluation.')
            actor,checksum=legacy(branch,seed)
            for cap in c['recheck_caps']:ev(actor,f'legacy_{branch}_s{seed}',branch,seed,checksum,'legacy_eval','checkpoint_recheck',cap)
            del actor;release();create_report(output)
    actor=load_actor(assets['policy'],c,42)
    for cap in c['recheck_caps']:ev(actor,'initial','initial',-1,MODELS['policy'][1],'legacy_eval','checkpoint_recheck',cap,update=0)
    del actor;release()
    frames={(b,s,cap):load_completed(output/'evaluations/checkpoint_recheck/legacy_eval'/f'cap_{cap}'/f'legacy_{b}_s{s}')
            for b in ['raw','knn_positive','knn_signed'] for s in c['seeds'] for cap in c['recheck_caps']}
    make_pack(output,'checkpoint_review',frames,['knn_positive','knn_signed'],c)
    del frames;create_report(output)
    # Extra offline controls: a development-only refresh and non-geometric baselines.
    for cohort in ['development_fit','development_validation']:
        for branch in ['raw','knn_signed']:
            actor,checksum=legacy(branch,c['development_policy_seed'])
            ev(actor,f'legacy_{branch}_s{c["development_policy_seed"]}',branch,c['development_policy_seed'],checksum,
               cohort,'development',features=True)
            del actor;release()
    fit_lock(output,reward,c)
    for branch in ['raw','knn_signed']:
        actor,checksum=legacy(branch,c['development_policy_seed'])
        ev(actor,f'legacy_{branch}_s{c["development_policy_seed"]}',branch,c['development_policy_seed'],checksum,
           'offline_test','development',features=True)
        del actor;release()
    offline_audit(output,reward,c)
    if not c['run_new_ppo']:
        status(output,'evaluation_complete',note='Checkpoint review/offline diagnostics complete; two-round PPO remains pending.')
        create_report(output,full=True);export_results(output);return
    # First round: static and iterative conditions share one trained kNN policy.
    parents={}
    for seed in c['seeds']:
        for branch in ['raw','knn_signed']:
            parents[(seed,branch)]=train_segment(output,c,assets,train,reward,judge,seed,branch,0,c['round1_updates'],should_stop)
        actor=checkpoint_actor(assets['policy'],c,seed,parents[(seed,'knn_signed')],identity,'knn_signed')
        parent_hash=file_hash(parents[(seed,'knn_signed')])
        ev(actor,f'round1_knn_signed_s{seed}','knn_signed',seed,parent_hash,'refresh_round1','refresh',features=True,update=c['round1_updates'])
        ev(actor,f'round1_knn_signed_s{seed}','knn_signed',seed,parent_hash,'development_validation','refresh_validation',features=True,update=c['round1_updates'])
        del actor;release();round_refresh(output,seed,reward,c,parent_hash)
        create_report(output)
    branches=[b for b in c['branches'] if c['run_capped_ablation'] or b!='iterative_capped']
    finals={}
    for seed in c['seeds']:
        reward.load_update(output/'refresh'/f'seed_{seed}')
        for branch in branches:
            parent=parents[(seed,'raw' if branch=='raw' else 'knn_signed')]
            finals[(seed,branch)]=train_segment(output,c,assets,train,reward,judge,seed,branch,c['round1_updates'],c['updates'],should_stop,parent)
    # No fresh-final scores were computed before all memory and policy updates ended.
    write_json(output/'final_evaluation_lock.json',{'all_training_complete':True,
        'checkpoint_sha256':{f'{s}/{b}':file_hash(p) for (s,b),p in finals.items()},
        'memory_locks':{str(s):file_hash(output/'refresh'/f'seed_{s}/locked_reward.json') for s in c['seeds']},
        'final_prompt_hash':canonical_hash(data['fresh_final'])})
    for seed in c['seeds']:
        reward.load_update(output/'refresh'/f'seed_{seed}')
        for branch in branches:
            path=finals[(seed,branch)];actor=checkpoint_actor(assets['policy'],c,seed,path,identity,branch)
            ev(actor,f'round2_{branch}_s{seed}',branch,seed,file_hash(path),'fresh_final','round2')
            del actor;release();create_report(output)
        for branch in ['raw','knn_signed']:
            path=parents[(seed,branch)];actor=checkpoint_actor(assets['policy'],c,seed,path,identity,branch)
            ev(actor,f'round1_{branch}_s{seed}',branch,seed,file_hash(path),'fresh_final','round1',update=c['round1_updates'])
            del actor;release()
        for branch in ['raw','knn_positive','knn_signed']:
            actor,checksum=legacy(branch,seed)
            ev(actor,f'legacy_{branch}_s{seed}',branch,seed,checksum,'fresh_final','legacy_fresh')
            del actor;release()
    frames={(b,s,256):load_completed(output/'evaluations/round2/fresh_final/cap_256'/f'round2_{b}_s{s}')
            for b in branches for s in c['seeds']}
    # Main human comparison needs static as reference, represented under the raw
    # lookup slot; the private mapping retains its true method name separately.
    primary={('raw' if b=='knn_signed' else b,s,cap):df for (b,s,cap),df in frames.items() if b!='raw'}
    folder=make_pack(output,'iterative_vs_static_review',primary,['iterative_knn'],c)
    # Explicit reference label for unblinding/reporting, never sent to reviewers.
    key=read_json(folder/'private/key.json');key['reference_method']='knn_signed';write_json(folder/'private/key.json',key)
    used_ids={r['prompt_id'] for r in key['pairs']}
    remaining={k:v[~v.prompt_id.isin(used_ids)] for k,v in frames.items()}
    make_pack(output,'new_policy_vs_raw_review',remaining,[b for b in branches if b!='raw'],c)
    status(output,'complete',elapsed_seconds=time.monotonic()-start,final_policy_runs=len(finals),
           note='GPU study complete. Blinded human ratings are a separate manual step.')
    create_report(output,full=True);archive=export_results(output)
    print('STUDY COMPLETE:',output/'reports/report.html','\nRESULTS ZIP:',archive,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--preflight',action='store_true');p.add_argument('--config',type=Path,default=ROOT/'config.json');a=p.parse_args()
    c=read_json(a.config);validate_config(c);(ROOT/'outputs').mkdir(exist_ok=True)
    lock=open(ROOT/'outputs/runner.lock','a+')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('A worker is already running in this project.')
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    output=None
    try:
        data,train=load_data();configure(c)
        if c['allow_downloads']:
            os.environ.pop('HF_HUB_OFFLINE',None);os.environ.pop('TRANSFORMERS_OFFLINE',None)
        output=output_root(c);status(output,'preparing_assets');assets=resolve_all(c)
        marker=output/'preflight.json'
        if a.preflight or not marker.is_file():preflight(c,output,assets,data,train)
        elif not read_json(marker).get('passed') or read_json(marker).get('identity')!=read_json(output/'manifest.json')['identity']:raise ValueError('Preflight identity differs.')
        if a.preflight:status(output,'ready')
        else:run(c,output,assets,data,train)
    except StopRequested as e:
        status(output,'paused',reason=str(e));create_report(output);export_results(output);print(str(e),flush=True)
    except Exception as e:
        if output is not None:status(output,'failed',error_type=type(e).__name__,message=str(e))
        traceback.print_exc()
        if output is not None:
            try:create_report(output);export_results(output)
            except Exception:traceback.print_exc()
        raise
    finally:lock.close()


if __name__=='__main__':main()
