"""Continue to fixed milestones and preserve each curve checkpoint."""
from pathlib import Path
import time
from knn_distillation.io import read,write,sealed,sha,digest,checkpoint_info,state_fingerprint,should_stop,status

def train_to(out,folder,c,assets,reward,rows,seed,branch,parent,identity,target,interval,expected_start=None):
    from run_study import load_actor,release
    from ppo_engine import PPOTrainer
    from common import StopRequested
    folder=Path(folder);ck=folder/'checkpoints';ck.mkdir(parents=True,exist_ok=True)
    start=parent['update']
    dependency={'identity':identity,'parent_sha256':parent['sha256'],'parent_identity':parent['identity'],'seed':seed,'branch':branch,
                'reward_identity':reward.route_identity(branch),'training_rows':digest(rows),'start':start}
    sealed(folder/'segment.json',dependency)
    targetfile=ck/f'checkpoint_{target:06d}.pt'
    if targetfile.exists() and targetfile.with_suffix('.json').exists():
        m=checkpoint_info(targetfile)
        if (m['identity'],m['seed'],m['branch'])!=(identity,seed,branch):raise ValueError('Scheduled checkpoint provenance changed.')
        return m
    actor=load_actor(assets['policy'],c,seed);trainer=PPOTrainer(actor,reward,c)
    try:
        candidates=sorted(ck.glob('checkpoint_*.json'))
        if candidates:
            m=read(candidates[-1]);current,history=trainer.restore(ck/m['name'],identity,seed,branch)
        else:
            current,history=trainer.restore(parent['path'],parent['identity'],seed,parent['branch'],optimizer=True)
            if current!=start:raise ValueError('Checkpoint payload update differs from metadata.')
            fingerprint=state_fingerprint(trainer)
            if expected_start is not None and fingerprint!=expected_start:raise ValueError('Starting policy/value/optimizer/RNG differs across reward arms.')
            sealed(folder/'fork_start.json',{'parent_sha256':parent['sha256'],'state_fingerprint':fingerprint,'policy_value_optimizer_rng_restored':True})
        if not start<=current<target:raise ValueError('Resumed checkpoint is outside scheduled segment.')
        for update in range(current+1,target+1):
            if should_stop(out):
                if current>start:trainer.checkpoint(ck,current,identity,seed,branch,history)
                raise StopRequested('Paused before PPO update.')
            off=(update-start-1)*c['rollout_batch_size'];batch=rows[off:off+c['rollout_batch_size']]
            if len(batch)!=c['rollout_batch_size']:raise ValueError('Matched prompt schedule exhausted.')
            reward.reset_cost();record=trainer.update(batch,branch,seed,update)
            if branch in ('student', 'judge_student'):
                for key in ('mean_proxy_z', 'mean_predicted_gap', 'mean_applied_gap', 'correction_fraction', 'within_old_distance_gate_fraction'):
                    record[key] = None
                record['proxy_diagnostics_measured'] = False
                if reward.cost['proxy_answers'] or reward.cost['teacher_answers'] or reward.cost['knn_queries']:
                    raise RuntimeError('The student PPO route accessed the teacher or kNN memory.')
            record.update(prompt_ids=[x['prompt_id'] for x in batch],reward_source=branch,model_calls=reward.cost.copy(),segment_start=start)
            history.append(record);current=update;write(folder/'history.json',history)
            if update%c['checkpoint_every']==0 or update==target or should_stop(out):
                trainer.checkpoint(ck,update,identity,seed,branch,history)
                # Keep ALL scheduled monitoring checkpoints and latest two recovery files.
                checkpoints=sorted(ck.glob('checkpoint_*.json'));recent=set(checkpoints[-2:])
                for p in checkpoints:
                    meta=read(p)
                    if p not in recent and (meta['update']-start)%interval!=0:
                        (ck/meta['name']).unlink();p.unlink()
            status(out,'training',branch=branch,seed=seed,update=update,target=target)
            print(f'{branch} seed={seed} update={update} reward={record["mean_base_reward"]:.4f} sec={record["seconds"]:.1f}',flush=True)
        return checkpoint_info(targetfile)
    finally:del trainer,actor;release()
