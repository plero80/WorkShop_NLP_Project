"""Run the whole HH memory-refresh/ridge comparison from pretrained models."""
import argparse
from importlib import metadata
import os
from pathlib import Path
import signal
import time

import numpy as np
from threadpoolctl import threadpool_limits

from common import StopRequested
from evaluation import save_npz
from experiment_cli.cli import yaml_value
from hh_offline.run import fit
from hh_ridge_ppo.protocol import read, write, sha, digest, require
from hh_ridge_ppo.reward import RidgeRouter
from knn_distillation.data import schedule
from knn_distillation.io import checkpoint_info, should_stop
from .artifacts import collect, calibrate, normalize, memory, seal
from .data import prepare

COUNTS = {'calibration','memory','train_M0','train_M1','train_M2','train_comparison','refresh1','refresh2','validation','offline','monitor','refresh2_eval','final'}


def resolve(project, config=None, output=None):
    project = Path(project).resolve()
    recipe = yaml_value(Path(config or project/'configs/experiments/hh-fresh.yaml').read_text(encoding='utf-8'))
    expected = {'version','experiment','output','seeds','data_seed','eval_seed','review_seed','allow_downloads','extra_hf_cache',
                'stage_updates','monitor_every','calibration_answers_per_prompt','memory_answers_per_prompt','high_gap_quantile','ridge_alphas','bootstrap_samples','counts'}
    require(isinstance(recipe, dict) and set(recipe) == expected and recipe['version'] == 1 and recipe['experiment'] == 'hh_fresh', 'Invalid fresh HH recipe')
    require(recipe['seeds'] == [42,43,44], 'The full comparison uses seeds [42, 43, 44]')
    for name in ('stage_updates','monitor_every','calibration_answers_per_prompt','memory_answers_per_prompt','bootstrap_samples','data_seed','eval_seed','review_seed'):
        require(type(recipe[name]) is int and recipe[name] > 0, 'Positive integer required: '+name)
    require(recipe['stage_updates'] % recipe['monitor_every'] == 0, 'monitor_every must divide stage_updates')
    require(type(recipe['allow_downloads']) is bool and (recipe['extra_hf_cache'] is None or isinstance(recipe['extra_hf_cache'], str)), 'Invalid model-cache settings')
    require(type(recipe['high_gap_quantile']) in (int,float) and 0 < recipe['high_gap_quantile'] < 1, 'Invalid calibration quantile')
    require(isinstance(recipe['counts'],dict) and set(recipe['counts']) == COUNTS and all(type(n) is int and n>0 for n in recipe['counts'].values()), 'Invalid prompt counts')
    alphas = recipe['ridge_alphas']
    require(isinstance(alphas,list) and alphas and len(set(alphas)) == len(alphas) and all(type(a) in (float,int) and np.isfinite(a) and a>0 for a in alphas), 'Invalid ridge grid')
    shared = read(project/'configs/config.json')
    ppo_keys = ('rollout_batch_size','generation_batch_size','reward_batch_size','mini_batch_size','micro_batch_size',
                'ppo_epochs','learning_rate','value_learning_rate','clip_range','value_clip_range','value_coefficient',
                'kl_coefficient','target_update_kl','gamma','gae_lambda','max_grad_norm','max_prompt_tokens','max_new_tokens',
                'reward_max_tokens','lora_rank','lora_alpha','checkpoint_every','cpu_threads','review_pairs_per_stratum')
    c = {k:shared[k] for k in ppo_keys}
    c.update(seeds=recipe['seeds'], allow_downloads=recipe['allow_downloads'], extra_hf_cache=recipe['extra_hf_cache'],
             eval_seed=recipe['eval_seed'], review_seed=recipe['review_seed'])
    require(recipe['counts']['memory']*recipe['memory_answers_per_prompt'] >= 31, 'Initial memory is too small for k=31')
    require(all(recipe['counts'][n] >= c['rollout_batch_size'] for n in ('train_M0','train_M1','train_M2','train_comparison')), 'Training pool is smaller than a rollout batch')
    require(recipe['counts']['final'] >= 6*c['review_pairs_per_stratum'], 'Final cohort is too small for the declared blinded review')
    destination = Path(output) if output else project/recipe['output']
    destination = (destination if destination.is_absolute() else project/destination).resolve()
    protected = [project/'code',project/'configs',project/'data',project/'docs',project/'reproducibility']
    require(not any(destination.is_relative_to(p) or p.is_relative_to(destination) for p in protected), 'Choose a separate result directory')
    return {'project':project,'recipe':recipe,'config':c,'output':destination}


def identify(plan):
    from assets import MODELS, DATASET
    project=plan['project']
    sources=[*Path(__file__).parent.glob('*.py'),*(project/'code/core').glob('*.py'),
             *[project/'code/experiments/knn_distillation'/n for n in ('data.py','io.py','maths.py','policy_eval.py','ppo_training.py','reward.py')],
             *[project/'code/experiments/hh_ridge_ppo'/n for n in ('features.py','protocol.py','reports.py','reward.py')],
             *[project/'code/experiments/hh_offline'/n for n in ('data.py','metrics.py','run.py')],
             project/'code/experiments/experiment_cli/cli.py',project/'hh_fresh.py',project/'code/templates/review_form.html']
    operational={'allow_downloads','extra_hf_cache','output'}
    names=('numpy','pandas','scipy','scikit-learn','threadpoolctl','torch','transformers','peft','huggingface_hub')
    versions={}
    for name in names:
        try: versions[name]=metadata.version(name)
        except metadata.PackageNotFoundError: versions[name]=None
    record={'protocol':'hh_fresh_two_refreshes_v1','recipe':{k:v for k,v in plan['recipe'].items() if k not in operational},
            'ppo_config':{k:v for k,v in plan['config'].items() if k not in operational},'models':MODELS,'dataset':DATASET,
            'sources':{p.relative_to(project).as_posix():sha(p) for p in sorted(set(sources))},'runtime_versions':versions,
            'old_checkpoints_used':False,'old_reward_labels_used':False,'initial_memory_shared_across_seeds':True,
            'primary':'ridge minus knn after matched M2-parent continuation; final prompts never select settings'}
    return digest(record),record


def budget(plan):
    r,c=plan['recipe'],plan['config'];n=r['counts'];s=len(r['seeds']);u=r['stage_updates']
    m0=n['memory']*r['memory_answers_per_prompt']
    # Parent M0 (1), paired M0/M1 (2), paired M1/M2 (2), final three arms (3).
    updates=8*u*s
    return {'seeds':r['seeds'],'old_experiment_files_required':False,'models_and_dataset_downloaded':r['allow_downloads'],
            'memory_rows':{'M0':m0,'M1':m0+n['refresh1'],'M2':m0+n['refresh1']+n['refresh2']},
            'parent_updates':[u,2*u,3*u],'final_update':4*u,'total_PPO_updates_all_branches':updates,
            'PPO_rollout_answers':updates*c['rollout_batch_size'],'prompt_counts':n,
            'new_judge_answers':n['calibration']*r['calibration_answers_per_prompt']+m0+s*(n['refresh1']+n['refresh2']+2*n['validation']+2*n['offline']+
                5*n['refresh2_eval']+3*n['final']+3*(u//r['monitor_every'])*n['monitor'])}


def initial_checkpoint(out,c,assets,seed,identity):
    from run_study import load_actor,release
    from ppo_engine import PPOTrainer
    folder=out/'parents'/f'seed_{seed}'
    path=folder/'checkpoints/checkpoint_000000.pt'
    if path.exists() and path.with_suffix('.json').exists():
        ck=checkpoint_info(path)
        require((ck['identity'],ck['seed'],ck['branch'],ck['update'])==(identity,seed,'initial',0),'Initial checkpoint changed')
        return ck
    actor=load_actor(assets['policy'],c,seed);trainer=PPOTrainer(actor,None,c)
    try:
        trainer.checkpoint(folder/'checkpoints',0,identity,seed,'initial',[])
    finally:
        del trainer,actor
        release()
    return checkpoint_info(path)


def train_stage(out,c,recipe,assets,router,seed,name,branch,parent,pool,identity,target,expected_start=None):
    from knn_distillation.ppo_training import train_to
    start=parent['update']
    require(target>start and target-start<=recipe['stage_updates'],'Invalid fresh training segment')
    # The full segment schedule is identical across its comparison arms and resumes.
    rows=schedule(pool,recipe['stage_updates'],c['rollout_batch_size'],recipe['data_seed']+seed+start)
    ck=train_to(out,out/'runs'/f'seed_{seed}'/name,c,assets,router,rows,seed,branch,parent,identity,target,
                recipe['monitor_every'],expected_start)
    require(ck['update']==target,'Training returned the wrong endpoint')
    return ck


def fork_fingerprint(out,seed,name):
    return read(out/'runs'/f'seed_{seed}'/name/'fork_start.json')['state_fingerprint']


def fit_ridge(out,seed,mem,validation,x,plan,identity):
    folder=out/'ridge'/f'seed_{seed}'
    dep={'run':identity,'memory_lock':sha(mem/'locked_reward.json'),'validation':sha(out/'samples'/f'seed_{seed}/validation/complete.json')}
    if (folder/'complete.json').exists():
        done=read(folder/'complete.json')
        require(done['dependency']==dep,'Ridge dependencies changed')
        for name,value in done['artifacts'].items():require(sha(folder/name)==value,'Ridge artifact changed')
        with np.load(folder/'ridge.npz',allow_pickle=False) as z:
            return {'coef':z['coef'],'intercept':float(z['intercept']),'identity':digest(done)}
    with np.load(mem/'refreshed_memory.npz',allow_pickle=False) as z:
        memory_x,memory_y=z['vectors'],z['gaps']
    began=time.perf_counter()
    with threadpool_limits(limits=plan['config']['cpu_threads']):
        fitted=fit(memory_x,memory_y,x,validation,{'ridge_alphas':plan['recipe']['ridge_alphas'],'knn_k':31,'knn_temperature':.05,'knn_k_grid':[31],'knn_temperature_grid':[.05]})
    save_npz(folder/'ridge.npz',coef=fitted['ridge']['coef'],intercept=np.array(fitted['ridge']['intercept']))
    write(folder/'selection.json',{'alpha':fitted['ridge']['alpha'],'candidates':fitted['candidates'],'memory_rows':len(memory_y),
          'validation_answers':len(validation),'test_used':False,'refit_on_validation':False,'fit_seconds':time.perf_counter()-began})
    done={'dependency':dep,'artifacts':{n:sha(folder/n) for n in ('ridge.npz','selection.json')}}
    write(folder/'complete.json',done)
    return {'coef':fitted['ridge']['coef'],'intercept':fitted['ridge']['intercept'],'identity':digest(done)}


def evaluate_policy(out,plan,assets,router,rows,seed,label,route,checkpoint,identity,cohort):
    from run_study import checkpoint_actor,release
    from knn_distillation.policy_eval import evaluate
    ck=checkpoint
    actor=checkpoint_actor(assets['policy'],plan['config'],seed,ck['path'],ck['identity'],ck['branch'])
    try:
        return evaluate(actor,router,rows,out/'evaluations'/cohort/f'seed_{seed}'/label/f'update_{ck["update"]:06d}',
                        plan['config'],identity,seed,route,ck['sha256'],cohort,ck['update'],features=True,monitor_kl=True,stop_out=out)
    finally:
        del actor
        release()


def execute(plan,out,assets,proxy,judge,identity):
    """Stage coordinator; all policy optimization is delegated to the existing trainer."""
    r,c=plan['recipe'],plan['config'];u=r['stage_updates'];seeds=r['seeds']
    data=prepare(out,c,r,assets)
    calibration_rows,_=collect(out,out/'samples/calibration',data['calibration'],[('base',seeds[0],None,r['calibration_answers_per_prompt'])],c,r,assets,proxy,judge,identity,'calibration')
    cal=calibrate(calibration_rows,r['high_gap_quantile'])
    seal(out/'calibration.json',cal)
    rows,x=collect(out,out/'samples/initial_memory',data['memory'],[('base',seeds[0],None,r['memory_answers_per_prompt'])],c,r,assets,proxy,judge,identity,'initial_memory')
    m0=memory(out/'memories/M0',rows,x,cal,{'examples':sha(out/'samples/initial_memory/complete.json')})
    router=RidgeRouter(proxy,judge,cal,c['cpu_threads'])
    parents,memories,bundles,refresh_controls={},{},{},{}
    for seed in seeds:
        base=initial_checkpoint(out,c,assets,seed,identity)
        router.load_memory(m0)
        pi1=train_stage(out,c,r,assets,router,seed,'parent_M0','knn',base,data['train_M0'],identity,u)
        folder=out/'samples'/f'seed_{seed}/refresh1'
        rows,x=collect(out,folder,data['refresh1'],[('parent',seed,pi1,1)],c,r,assets,proxy,judge,identity,'refresh1')
        m1=memory(out/'memories'/f'seed_{seed}/M1',rows,x,cal,{'examples':sha(folder/'complete.json'),'parent':pi1['sha256']},m0)
        router.load_memory(m0)
        static0=train_stage(out,c,r,assets,router,seed,'static_M0','knn',pi1,data['train_M1'],identity,2*u)
        router.load_memory(m1)
        pi2=train_stage(out,c,r,assets,router,seed,'refresh_M1','knn',pi1,data['train_M1'],identity,2*u,fork_fingerprint(out,seed,'static_M0'))
        folder=out/'samples'/f'seed_{seed}/refresh2'
        rows,x=collect(out,folder,data['refresh2'],[('parent',seed,pi2,1)],c,r,assets,proxy,judge,identity,'refresh2')
        m2=memory(out/'memories'/f'seed_{seed}/M2',rows,x,cal,{'examples':sha(folder/'complete.json'),'parent':pi2['sha256']},m1)
        router.load_memory(m1)
        static1=train_stage(out,c,r,assets,router,seed,'static_M1','knn',pi2,data['train_M2'],identity,3*u)
        router.load_memory(m2)
        pi3=train_stage(out,c,r,assets,router,seed,'refresh_M2','knn',pi2,data['train_M2'],identity,3*u,fork_fingerprint(out,seed,'static_M1'))
        parents[seed],memories[seed]=pi3,m2
        refresh_controls[seed]={'static_M0':static0,'refresh_M1':pi2,'parent_pi2':pi2,'static_M1':static1,'refresh_M2':pi3,'M1':m1}
        validation,x=collect(out,out/'samples'/f'seed_{seed}/validation',data['validation'],[('base',seed,None,1),('parent',seed,pi3,1)],c,r,assets,proxy,judge,identity,'validation')
        bundles[seed]=fit_ridge(out,seed,m2,normalize(validation,cal),x,plan,identity)
    seal(out/'selection_complete.json',{'models':{str(s):b['identity'] for s,b in bundles.items()},'test_used':False,'all_ridge_models_frozen':True})
    endpoints={}
    for seed in seeds:
        router.load_memory(memories[seed]);b=bundles[seed]
        router.set_ridge(b['coef'],b['intercept'],b['identity'])
        expected=None
        # Rotate the arm order across seeds; all have the same parent and schedule.
        order=['proxy','knn','ridge'];offset=seed%3;order=order[offset:]+order[:offset]
        for arm in order:
            for delta in range(r['monitor_every'],u+1,r['monitor_every']):
                ck=train_stage(out,c,r,assets,router,seed,arm,arm,parents[seed],data['train_comparison'],identity,3*u+delta,expected)
                if expected is None:expected=fork_fingerprint(out,seed,arm)
                evaluate_policy(out,plan,assets,router,data['monitor'],seed,arm,arm,ck,identity,'monitor')
            endpoints[(seed,arm)]=ck
    seal(out/'final_lock.json',{'all_training_complete':True,'endpoints':{f'{s}/{a}':ck['sha256'] for (s,a),ck in endpoints.items()},'final_prompts':digest(data['final']),'refresh_eval_prompts':digest(data['refresh2_eval'])})
    for seed in seeds:
        router.load_memory(memories[seed]);b=bundles[seed]
        router.set_ridge(b['coef'],b['intercept'],b['identity'])
        for arm in ('proxy','knn','ridge'):
            evaluate_policy(out,plan,assets,router,data['final'],seed,arm,arm,endpoints[(seed,arm)],identity,'final')
        collect(out,out/'samples'/f'seed_{seed}/offline',data['offline'],[('base',seed,None,1),('parent',seed,parents[seed],1)],c,r,assets,proxy,judge,identity,'offline')
        controls=refresh_controls[seed]
        for label in ('static_M0','refresh_M1','parent_pi2','static_M1','refresh_M2'):
            router.load_memory(m0 if label=='static_M0' else memories[seed] if label=='refresh_M2' else controls['M1'])
            evaluate_policy(out,plan,assets,router,data['refresh2_eval'],seed,label,'knn',controls[label],identity,'refresh1' if label in ('static_M0','refresh_M1') else 'refresh2')
    if should_stop(out):raise StopRequested('Paused before reports')
    from .reports import report
    with threadpool_limits(limits=c['cpu_threads']):report(plan,out,cal,bundles,memories)


def main(argv=None,project=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['run'])
    parser.add_argument('--config',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args(argv)
    try:
        plan=resolve(project or Path(__file__).resolve().parents[3],args.config,args.output)
        identity,record=identify(plan);out=plan['output']/('study_'+identity[:16])
        import json
        print(json.dumps({**budget(plan),'output':str(out)},indent=2),flush=True)
        if args.dry_run:return 0
        require(os.name=='posix','Run fresh GPU training on Linux; --dry-run also works locally')
        import fcntl
        import torch
        import assets as asset_module
        from run_study import configure
        from reward_bridge import RewardScorer
        from knn_distillation.io import pause
        c=plan['config']
        configure(c)
        plan['output'].mkdir(parents=True,exist_ok=True)
        with open(plan['project']/'hh_fresh.lock','a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if out.exists() and any(out.iterdir()) and not (out/'manifest.json').exists():raise ValueError('Output is not a fresh HH run')
            seal(out/'manifest.json',{'identity':identity,**record})
            if (out/'complete.json').exists():
                for name,value in read(out/'complete.json')['artifacts'].items():require(sha(out/name)==value,'Completed result changed: '+name)
                print('Already complete: '+str(out/'reports'));return 0
            require(not (out/'PAUSE').exists(),'Remove the PAUSE file to resume')
            signal.signal(signal.SIGINT,pause);signal.signal(signal.SIGTERM,pause)
            write(out/'status.json',{'stage':'preparing','identity':identity})
            try:
                if c['allow_downloads']:
                    os.environ.pop('HF_HUB_OFFLINE',None);os.environ.pop('TRANSFORMERS_OFFLINE',None)
                asset_module.ROOT=out
                assets=asset_module.resolve_all(c)
                write(out/'environment.json',{'gpu':torch.cuda.get_device_name(),'cuda':torch.version.cuda,'versions':record['runtime_versions'],'model_revisions':asset_module.MODELS})
                proxy=RewardScorer(assets['proxy'],c['reward_batch_size'],c['reward_max_tokens'])
                judge=RewardScorer(assets['judge'],c['reward_batch_size'],c['reward_max_tokens'])
                execute(plan,out,assets,proxy,judge,identity)
                write(out/'complete.json',{'identity':identity,'budget':budget(plan),'artifacts':{p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p!=out/'complete.json' and p.name!='status.json' and p.suffix not in ('.pt','.pending')}})
                write(out/'status.json',{'stage':'complete','identity':identity})
                print('FULL FRESH RUN COMPLETE: '+str(out/'reports'),flush=True)
            except StopRequested as error:
                write(out/'status.json',{'stage':'paused','reason':str(error)});print(str(error),flush=True)
            except Exception as error:
                write(out/'status.json',{'stage':'failed','error_type':type(error).__name__,'message':str(error)})
                raise
    except (ValueError,OSError,KeyError) as error:
        parser.error(str(error))
    return 0
