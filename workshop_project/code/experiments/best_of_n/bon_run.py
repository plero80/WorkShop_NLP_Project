"""Run an independent, resumable best-of-N diagnostic on original RunPod assets."""
import argparse, fcntl, os, signal, sys, time, traceback, zipfile
from pathlib import Path
from bon_io import *

STOP = False
class Paused(Exception): pass

def signal_stop(*args):
    global STOP
    STOP = True
    print('Pause requested; finishing the current candidate batch.', flush=True)

def check_stop(out):
    if STOP or (out/'PAUSE').exists(): raise Paused('Paused safely at a batch boundary.')

def state(out, stage, **details):
    write(out/'status.json', {'stage':stage,'updated_unix':time.time(),**details})

def setup(project, oracle_arg, settings_path):
    s=read(settings_path);validate_settings(s)
    expected=read(HERE/'expected_sources.json')
    for name,h in expected.items():
        if not (project/name).is_file() or sha(project/name)!=h:
            raise ValueError('Project source differs from the inspected archive: '+name+'. Review the difference before updating expected_sources.json.')
    oracle=locate_oracle(project,oracle_arg);suite=read(oracle.parent/'manifest.json')
    if digest({k:v for k,v in suite.items() if k!='identity'})!=suite['identity']:
        raise ValueError('Oracle suite manifest identity mismatch.')
    if read(oracle/'manifest.json')['suite_identity']!=suite['identity']: raise ValueError('Wrong oracle suite.')
    for name,h in read(project/'PACKAGE_MANIFEST.json')['input_sha256'].items():
        if sha(project/name)!=h: raise ValueError('Original project input changed: '+name)
    ck={p:checkpoint(oracle,p,s['policy_seed']) for p in s['policies']}
    lock=read(oracle/'memory/locked_reward.json')
    if sha(oracle/'memory/refreshed_memory.npz')!=lock['memory_sha256']:
        raise ValueError('Oracle memory checksum mismatch.')
    if lock.get('k')!=31 or lock.get('temperature')!=.05 or lock.get('monitor_or_test_labels_used') is not False:
        raise ValueError('Expected frozen oracle k=31, temperature=.05 training memory.')
    sys.path.insert(0,str(project))
    from common import runtime_versions
    versions=runtime_versions()
    for lib in ('torch','transformers','peft'):
        if versions[lib]!=suite['runtime_versions'][lib]:
            raise ValueError(f'Use the successful oracle environment: {lib}={suite["runtime_versions"][lib]}, found {versions[lib]}.')
    c={**suite['parent_config'],'allow_downloads':s['allow_downloads'],'extra_hf_cache':s['extra_hf_cache'],
       'max_new_tokens':s['answer_cap'],'generation_batch_size':s['candidate_batch_size'],
       'reward_batch_size':s['candidate_batch_size']}
    if c['reward_max_tokens']<4096: raise ValueError('Full-answer scoring guard must be at least 4096.')
    metadata={'protocol_version':1,'settings':{k:v for k,v in s.items() if k not in ('allow_downloads','extra_hf_cache')},
        'oracle_suite_identity':suite['identity'],'oracle_identity':read(oracle/'manifest.json')['identity'],
        'checkpoints':{p:{k:v for k,v in m.items() if k!='path'} for p,m in ck.items()},
        'memory_lock':lock,'memory_examples_sha256':sha(oracle/'memory/added_examples.csv'),
        'project_source_sha256':expected,'addon_sha256':{p.name:sha(p) for p in sorted(HERE.glob('*.py'))},
        'runtime_versions':versions,'generation':{'temperature':1.0,'top_p':1.0,'top_k':0,'answer_strip':'original PPOActor .strip()'},
        'primary':{'generator':s['primary_policy'],'n':s['primary_n'],'metric':'selected judge z: knn minus proxy'},
        'inference_scope':'Fixed checkpoints; prompt bootstrap. No training-seed generalization.'}
    identity=digest(metadata);out=project/'best_of_n_outputs'/('study_'+identity[:16]);out.mkdir(parents=True,exist_ok=True)
    seal(out/'manifest.json',{'identity':identity,**metadata})
    write(project/'best_of_n_outputs/latest.json',{'output':str(out),'oracle':str(oracle)})
    return s,c,oracle,ck,out,identity

def memory_and_rewards(c,assets,oracle):
    import numpy as np
    from run_study import load_rewards
    frozen,judge=load_rewards(c,assets)
    legacy_parity=frozen.validate_memory_encoder()
    lock=read(oracle/'memory/locked_reward.json')
    if lock['calibration']!=frozen.calibration: raise ValueError('Memory normalization differs from proxy/judge calibration.')
    with np.load(oracle/'memory/refreshed_memory.npz',allow_pickle=False) as z:
        memory={k:z[k].copy() for k in ('vectors','gaps')}
    v,g=memory['vectors'],memory['gaps']
    if v.ndim!=2 or g.shape!=(len(v),) or len(v)<31 or not np.isfinite(v).all() or not np.isfinite(g).all() or not np.allclose(np.linalg.norm(v,axis=1),1,atol=1e-4):
        raise ValueError('Invalid frozen memory.')
    if len(v)!=lock['total_rows']: raise ValueError('Memory row count changed.')
    # Check the ACTUAL oracle memory, as well as the project's legacy encoder reference.
    import pandas as pd
    bank=pd.read_csv(oracle/'memory/added_examples.csv',keep_default_na=False)
    if len(bank)!=len(v) or lock['old_rows']!=0: raise ValueError('Expected complete fresh oracle memory references.')
    indices=np.linspace(0,len(v)-1,8,dtype=int);rows=bank.iloc[indices]
    p=frozen.scorer.score(rows.prompt.tolist(),rows.answer.tolist(),features=True)
    j=judge.score(rows.prompt.tolist(),rows.answer.tolist())
    pdiff=np.abs(p['raw']-rows.proxy_raw.to_numpy(float));jdiff=np.abs(j['raw']-rows.judge_raw.to_numpy(float))
    cos=(p['features']*v[indices]).sum(axis=1)
    if pdiff.mean()>.03 or pdiff.max()>.3 or jdiff.mean()>.05 or jdiff.max()>.5 or cos.min()<.999:
        raise ValueError('Oracle memory / judge parity failed; no candidates generated.')
    if not np.allclose(bank.actual_proxy_judge_gap.to_numpy(float),g,atol=1e-6): raise ValueError('Memory gaps differ from saved labels.')
    return frozen,judge,memory,{'legacy':legacy_parity,'oracle_rows':8,'proxy_max_error':float(pdiff.max()),
        'judge_max_error':float(jdiff.max()),'minimum_cosine':float(cos.min()),
        'preflight_teacher_answers':8,'preflight_proxy_answers':8+legacy_parity['checked_rows']}

def evaluate_generator(project,out,identity,phase,rows,gen,ck,c,s,assets,frozen,judge,memory):
    import numpy as np
    from common import seed_for
    from run_study import checkpoint_actor,release
    from reward_bridge import predict_memory
    root=out/phase;folder=root/'runs'/gen;all_rows=[];actor=None
    try:
        for pi,row in enumerate(rows):
            for start in range(0,s['candidates'],s['candidate_batch_size']):
                check_stop(out);stop=min(start+s['candidate_batch_size'],s['candidates']);ids=list(range(start,stop))
                signature={'identity':identity,'phase':phase,'generator':gen,'prompt':digest(row),'candidate_ids':ids}
                filename=f'{pi:05d}_{start:03d}.json'
                generated_path=folder/'generated'/filename;scored_path=folder/'scored'/filename
                if scored_path.exists():
                    scored=load_shard(scored_path,signature)
                    generated=load_shard(generated_path,signature)
                    if len(scored)!=len(ids) or [x['candidate_id'] for x in scored]!=ids: raise ValueError('Invalid scored candidate batch.')
                    for a,b in zip(scored,generated):
                        if any(a[k]!=v for k,v in b.items()): raise ValueError('Scoring no longer matches generation.')
                else:
                    if generated_path.exists():
                        generated=load_shard(generated_path,signature)
                    else:
                        if actor is None:
                            meta=ck[gen]
                            actor=checkpoint_actor(assets['policy'],c,s['policy_seed'],meta['path'],meta['identity'],meta['branch'])
                        seed=seed_for(s['generation_seed'],phase,gen,row['prompt_id'],start)
                        t=time.monotonic();parts=actor.generate([row['prompt']]*len(ids),seed,batch_size=len(ids));elapsed=time.monotonic()-t
                        answers=[a for p in parts for a in p['answers']]
                        lengths=[int(n) for p in parts for n in p['response_mask'].sum(1)]
                        eos=[bool(x) for p in parts for x in p['ended_eos']]
                        if len(answers)!=len(ids): raise ValueError('Incomplete generation batch.')
                        generated=[{**row,'generator':gen,'candidate_id':i,'answer':a,'response_tokens':n,
                            'ended_eos':e,'answer_cap':not e and n==s['answer_cap'],
                            'generation_seed':seed,'generation_seconds_allocated':elapsed/len(ids)}
                            for i,a,n,e in zip(ids,answers,lengths,eos)]
                        save_shard(generated_path,signature,generated)
                        del parts
                    if len(generated)!=len(ids) or [x['candidate_id'] for x in generated]!=ids: raise ValueError('Invalid generated candidate batch.')
                    check_stop(out)
                    prompts=[x['prompt'] for x in generated];answers=[x['answer'] for x in generated]
                    t=time.monotonic();p=frozen.scorer.score(prompts,answers,features=True)
                    gap,distance=predict_memory(p['features'],memory,threads=c['cpu_threads'])
                    proxy_seconds=time.monotonic()-t
                    cal=frozen.calibration;zp=(p['raw']-cal['proxy_mean'])/cal['proxy_std']
                    corrected=zp-gap  # kNN selection reward is fixed BEFORE judge scores are obtained.
                    t=time.monotonic();j=judge.score(prompts,answers);judge_seconds=time.monotonic()-t
                    zj=(j['raw']-cal['judge_mean'])/cal['judge_std']
                    if not np.isfinite([zp,zj,corrected,gap,distance]).all(): raise ValueError('Nonfinite scores.')
                    if np.any(p['truncated']) or np.any(j['truncated']): raise ValueError('Whole-answer scoring required.')
                    scored=[{**r,'proxy_raw':float(p['raw'][i]),'judge_raw':float(j['raw'][i]),
                        'proxy_z':float(zp[i]),'judge_z':float(zj[i]),'gap_hat':float(gap[i]),'corrected_z':float(corrected[i]),
                        'actual_gap':float(zp[i]-zj[i]),'theta':cal['theta'],'mean_neighbor_distance':float(distance[i]),
                        'proxy_input_tokens':int(p['tokens'][i]),'judge_input_tokens':int(j['tokens'][i]),
                        'proxy_knn_seconds_allocated':proxy_seconds/len(ids),'judge_seconds_allocated':judge_seconds/len(ids)}
                        for i,r in enumerate(generated)]
                    save_shard(scored_path,signature,scored)
                all_rows.extend(scored)
                state(out,'running',phase=phase,generator=gen,prompt=pi+1,prompts=len(rows),candidate_batch_end=stop)
            print(f'{phase} {gen}: {pi+1}/{len(rows)} prompts; {s["candidates"]} candidates each',flush=True)
        write(folder/'complete.json',{'identity':identity,'phase':phase,'generator':gen,'prompts':len(rows),
              'candidates':len(all_rows),'rows_sha256':digest(all_rows)})
        return all_rows
    finally:
        if actor is not None: del actor
        release()

def export(out,phase):
    dest=out/f'important_outcomes_best_of_n_{phase}.zip';tmp=dest.with_suffix('.pending')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted((out/phase).rglob('*')):
            if p.is_file() and p.suffix!='.pending':z.write(p,'results/'+str(p.relative_to(out/phase)))
        for name in ('manifest.json','status.json','preflight.json'):
            if (out/name).exists():z.write(out/name,name)
        for name in ('complete.json',phase+'.json'):
            if (out/'data'/name).exists():z.write(out/'data'/name,'data/'+name)
        for p in sorted(HERE.glob('*')):
            if p.is_file():z.write(p,'code/'+p.name)
    tmp.replace(dest);return dest

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,required=True);parser.add_argument('--oracle',type=Path)
    parser.add_argument('--settings',type=Path,default=HERE/'settings.json')
    parser.add_argument('--phase',choices=['development','confirmation'],default='development')
    parser.add_argument('--preflight-only',action='store_true');args=parser.parse_args()
    project=args.project.resolve();(project/'outputs').mkdir(exist_ok=True)
    # Share the original project's GPU lock, including preflight and data reservation.
    lock=open(project/'outputs/runner.lock','a+')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('Another project experiment is active. Run sequentially on this GPU.')
    out=None
    signal.signal(signal.SIGTERM,signal_stop);signal.signal(signal.SIGINT,signal_stop)
    try:
        s,c,oracle,ck,out,identity=setup(project,args.oracle,args.settings)
        if (out/'PAUSE').exists():raise Paused('PAUSE exists. Use launch_best_of_n.py start to resume.')
        from run_study import configure
        from assets import resolve_all
        from transformers import AutoTokenizer
        from bon_data import reserve
        from bon_metrics import report
        state(out,'preflight',phase=args.phase);configure(c);assets=resolve_all(c)
        tok=AutoTokenizer.from_pretrained(assets['policy'],local_files_only=True)
        data=reserve(project,out,c,s,tok)
        frozen,judge,memory,parity=memory_and_rewards(c,assets,oracle)
        write(out/'preflight.json',{'passed':True,'identity':identity,'parity':parity,
            'full_answer_scoring':True,'no_ppo_updates':True,'memory_frozen':True,'updated_unix':time.time()})
        print('PREFLIGHT PASSED:',out,flush=True)
        if args.preflight_only:
            state(out,'ready',phase=args.phase);return
        phase=args.phase;root=out/phase;root.mkdir(exist_ok=True)
        seal(root/'protocol_lock.json',{'identity':identity,'phase':phase,'prompts_sha256':digest(data[phase]),
              'primary_policy':s['primary_policy'],'primary_n':s['primary_n'],'locked_before_scoring':True})
        all_rows=[]
        for gen in s['policies']:
            all_rows.extend(evaluate_generator(project,out,identity,phase,data[phase],gen,ck,c,s,assets,frozen,judge,memory))
        expected=len(data[phase])*s['candidates']*len(s['policies'])
        if len(all_rows)!=expected:raise ValueError('Incomplete candidate pool.')
        primary=report(root,s,all_rows)
        write(root/'costs.json',{'completed_candidate_answers':len(all_rows),
            'candidate_evaluation_teacher_answers':len(all_rows),'candidate_proxy_answers':len(all_rows),
            'teacher_answers_consumed_by_knn_selector':0,'teacher_memory_labels_preexisting':len(memory['gaps']),
            'original_memory_build_time_included':False,
            'generated_tokens':sum(r['response_tokens'] for r in all_rows),
            'proxy_input_tokens':sum(r['proxy_input_tokens'] for r in all_rows),'judge_input_tokens':sum(r['judge_input_tokens'] for r in all_rows),
            'generation_seconds_completed':sum(r['generation_seconds_allocated'] for r in all_rows),
            'proxy_knn_seconds_completed':sum(r['proxy_knn_seconds_allocated'] for r in all_rows),
            'judge_seconds_completed':sum(r['judge_seconds_allocated'] for r in all_rows),
            'note':'Completed batch work only; preflight, repeated/discarded work and original memory creation excluded. Every invocation rechecks parity.'})
        write(root/'complete.json',{'identity':identity,'prompts':len(data[phase]),'candidate_answers':len(all_rows),'primary_result':primary})
        state(out,'complete',phase=phase);print('RESULTS:',export(out,phase),flush=True)
    except Paused as error:
        if out:state(out,'paused',phase=args.phase,reason=str(error))
        print(str(error),flush=True)
    except Exception as error:
        if out:state(out,'failed',phase=args.phase,error_type=type(error).__name__,message=str(error))
        traceback.print_exc();raise
    finally:lock.close()

if __name__=='__main__':main()
