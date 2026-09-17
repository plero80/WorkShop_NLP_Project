"""Add a ridge PPO arm to completed GSM8K controls; never retrain those controls."""
import argparse
import copy
from importlib import metadata
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from experiment_cli.cli import yaml_value
from gsm8k_experiment.common import atomic_json, digest, read_json, run_lock, seed_all, status
from .inputs import inspect, require, sha
from .model import fit


def resolve(project, recipe_path=None, sources=None, output=None):
    project = Path(project).resolve()
    recipe = yaml_value(Path(recipe_path or project/'configs/experiments/gsm8k-ridge.yaml').read_text(encoding='utf-8'))
    require(set(recipe) == {'version','experiment','sources','output','ridge_alphas','cpu_threads'} and
            recipe['version'] == 1 and recipe['experiment'] == 'gsm8k_ridge', 'Invalid GSM8K ridge recipe')
    alphas = recipe['ridge_alphas']
    require(isinstance(alphas,list) and alphas and all(type(a) in (int,float) and np.isfinite(a) and a>0 for a in alphas)
            and len(set(alphas)) == len(alphas), 'Ridge alphas must be distinct finite positive numbers')
    require(type(recipe['cpu_threads']) is int and recipe['cpu_threads'] > 0, 'cpu_threads must be positive')
    def path(value):
        p = Path(value).expanduser()
        return (p if p.is_absolute() else project/p).resolve()
    requested = sources or recipe['sources']
    require(isinstance(requested,list) and requested, 'Specify at least one saved source run')
    expanded = []
    for name in requested:
        source = path(name)
        if (source/'suite_protocol.json').is_file():
            expanded.extend(source/f'seed_{s}' for s in read_json(source/'suite_protocol.json')['seeds'])
        else:
            expanded.append(source)
    destination = path(output or recipe['output'])
    require(all(not destination.is_relative_to(p) and not p.is_relative_to(destination) for p in expanded), 'Use an output separate from every saved source')
    require(all(not destination.is_relative_to(project/p) and not (project/p).is_relative_to(destination)
                for p in ('code','configs','data','docs','reproducibility')), 'Choose a separate results directory')
    plans = [inspect(p) for p in expanded]
    require(len({p['seed'] for p in plans}) == len(plans), 'Each source must have a different training seed')
    paths = [*Path(__file__).parent.glob('*.py'),Path(__file__).with_name('compatibility.json'),project/'gsm8k.py',
             *[project/'code/experiments/gsm8k_experiment'/n for n in ('run.py','models.py','memory.py','ppo.py','shared.py','grading.py','validation.py','metrics.py')]]
    from gsm8k_experiment.shared import shared_sources
    from gsm8k_experiment.assets import source_fingerprint
    compatibility = read_json(Path(__file__).with_name('compatibility.json'))
    for name,expected in compatibility['critical_sources'].items():
        require(sha(project/'code/experiments/gsm8k_experiment'/name) == expected, 'GSM8K scoring/PPO implementation changed: '+name)
    accepted = {*compatibility['source_fingerprints'],source_fingerprint()}
    for p in plans:
        require(p['manifest']['identity']['source'] in accepted, 'Saved controls use an unverified implementation; cannot claim a matched PPO comparison')
    identity = {'protocol':'gsm8k_ridge_extension_v1','ridge_alphas':alphas,'cpu_threads':recipe['cpu_threads'],
                'controls':{str(p['seed']):p['files'] for p in plans},'sources':{p.relative_to(project).as_posix():sha(p) for p in paths},
                'shared_PPO':shared_sources(),'fit_versions':{n:metadata.version(n) for n in ('numpy','scipy','scikit-learn')},
                'target':'actual 4B normalized gap','test_role':'follow-up on an already inspected benchmark',
                'control_training_repeated':False}
    return {'project':project,'recipe':recipe,'plans':plans,'output':destination,'identity':identity,'fingerprint':digest(identity)}


def seal(path, value):
    if path.exists():
        require(read_json(path) == value, 'Inputs, code or settings changed; choose a new ridge output: '+str(path))
    else:
        atomic_json(path,value)


def prepare(resolved):
    out = resolved['output']
    require(not out.exists() or not any(out.iterdir()) or (out/'ridge_manifest.json').exists(), 'Output already contains unrelated files')
    seal(out/'ridge_manifest.json',{'fingerprint':resolved['fingerprint'],'identity':resolved['identity']})
    models = {}
    with threadpool_limits(limits=resolved['recipe']['cpu_threads']):
        for p in resolved['plans']:
            folder = out/f"seed_{p['seed']}"
            models[p['seed']] = fit(p,folder/'fitted',resolved['recipe']['ridge_alphas'])
            # Existing train_arm/checkpoint code validates this identity when resuming.
            seal(folder/'manifest.json',{'fingerprint':resolved['fingerprint'],'identity':resolved['identity']})
            seal(folder/'config.json',p['config'])
    seal(out/'selection_complete.json',{'models':{str(s):sha(out/f'seed_{s}/fitted/selection.json') for s in models},'all_models_frozen_before_evaluation':True})
    return models


def preflight(plan):
    require((plan['source']/'initial_trainable.pt').is_file(), 'Need initial_trainable.pt from the full original run to match its initial policy/value state')
    saved = plan['manifest']['identity']
    versions = {name:metadata.version(name) for name in saved['versions']}
    changed = {k:{'saved':v,'current':versions[k]} for k,v in saved['versions'].items() if versions[k] != v}
    import torch
    if torch.__version__.split('+')[0] != saved['torch_version']:
        changed['torch'] = {'saved':saved['torch_version'],'current':torch.__version__}
    require(not changed,'Match the original runtime before comparing saved controls: '+str(changed))


def train(plan, folder, model, fingerprint):
    """Only the new reward predictor/arm is added; use the original train_arm."""
    import torch
    from gsm8k_experiment.assets import check_runtime, resolve_assets
    from gsm8k_experiment.models import Policy, RewardScorer, ScoreCache
    from gsm8k_experiment.run import smoke_ppo, train_arm, evaluate
    from gsm8k_experiment.ppo import optimizer_for, load_checkpoint
    preflight(plan)
    c = plan['config']
    runtime = check_runtime(c)
    seal(folder/'environment.json',{'runtime':runtime,'source_runtime':plan['manifest']['runtime_at_creation'],
         'same_initial_trainable_sha256':sha(plan['source']/'initial_trainable.pt')}) if not (folder/'environment.json').exists() else None
    # Keep exact original settings for scoring identities. Download only used models.
    downloading = copy.deepcopy(c)
    downloading['models'] = {k:c['models'][k] for k in ('policy','proxy','judge')}
    atomic_json(folder/'resolved_assets.json',plan['resolved'])
    resolve_assets(downloading,folder)
    seed_all(c['seed'])
    policy = Policy(c,plan['resolved'])
    initial = torch.load(plan['source']/'initial_trainable.pt',map_location='cpu',weights_only=True)
    policy.restore_trainable(initial)
    smoke_ppo(policy,c,folder,initial)
    cache = ScoreCache(folder)
    try:
        proxy = RewardScorer('proxy',c,plan['resolved'],cache)
        judge = RewardScorer('judge',c,plan['resolved'],cache)
        require(proxy.identity == model.encoder_identity, 'Proxy encoder identity differs from the saved memory')
        probe = read_json(plan['source']/'prepared/encoder_probe.json')
        current = proxy._infer([probe['row']],'encoder_parity',c['scoring']['max_new_tokens'])[0]
        require(float(np.dot(current['embedding'],np.asarray(probe['embedding'],np.float32))) >= .999,
                'Saved proxy features cannot be reproduced on this runtime')
        train_arm(policy,'ridge',plan['updates'],proxy,judge,plan['norm'],model,initial,
                  plan['split'],c,folder,fingerprint)
        if plan['kind'] == 'final':
            seal(folder/'final_protocol.json',{'updates':plan['updates'],'arms':['ridge'],'fingerprint':fingerprint,'selection_frozen':True})
            optimizer = optimizer_for(policy,c)
            load_checkpoint(folder/'arms/ridge/checkpoint.pt',policy,optimizer,fingerprint,'ridge')
            evaluate(policy,plan['split']['cohorts']['final'],proxy,judge,plan['norm'],model,folder,'ridge',plan['updates'],'final')
        status(folder,'complete',arm='ridge',seed=plan['seed'],updates=plan['updates'])
    finally:
        cache.close()


def main(argv=None,project=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run','report'])
    parser.add_argument('--config',type=Path)
    parser.add_argument('--source',action='append',help='Completed single run or seed suite; repeat for multiple seeds')
    parser.add_argument('--output')
    parser.add_argument('--dry-run',action='store_true')
    args = parser.parse_args(argv)
    try:
        resolved = resolve(project or Path(__file__).resolve().parents[3],args.config,args.source,args.output)
        import json
        print(json.dumps({'output':str(resolved['output']),'sources':[{'path':str(p['source']),'seed':p['seed'],
              'PPO_target':p['updates'],'evaluation':p['kind'],'memory_answers':len(p['memory'].gaps)} for p in resolved['plans']],
              'new_arms':['ridge'],'control_arms_retrained':[],'new_judge_calls_to_fit_ridge':0},indent=2),flush=True)
        if args.dry_run:
            return 0
        if args.action == 'run':
            for p in resolved['plans']:
                preflight(p)
        # The lock is beside the result tree so it does not create an unrelated output.
        with run_lock(resolved['output'].parent/(resolved['output'].name+'.ridge-lock')):
            models = prepare(resolved)
            if args.action == 'run':
                for p in resolved['plans']:
                    train(p,resolved['output']/f"seed_{p['seed']}",models[p['seed']],resolved['fingerprint'])
            from .reports import report
            with threadpool_limits(limits=resolved['recipe']['cpu_threads']):
                report(resolved,models)
        print('Ridge report: '+str(resolved['output']/'report.md'),flush=True)
        return 0
    except (ValueError,OSError,KeyError) as error:
        parser.error(str(error))
