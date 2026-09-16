"""Artifacts, source validation and experiment state. No model imports."""
from pathlib import Path
import json,hashlib,os,time,signal
ROOT=Path(__file__).resolve().parent
STOP=False

def check_config(o):
    for name in ('train_prompts', 'validation_prompts', 'offline_prompts', 'final_prompts',
                 'ppo_updates', 'monitor_every', 'review_pairs_per_seed', 'data_seed', 'student_seed',
                 'student_epochs', 'student_batch_size', 'student_accumulation', 'student_lora_rank',
                 'student_lora_alpha', 'student_checkpoint_every', 'latency_examples', 'latency_repeats'):
        if type(o.get(name)) is not int or o[name] < 1:
            raise ValueError('Positive integer required: ' + name)
    if o['ppo_updates'] % o['monitor_every']:
        raise ValueError('monitor_every must divide ppo_updates.')
    if (not isinstance(o.get('seeds'), list) or not o['seeds'] or
        len(set(o['seeds'])) != len(o['seeds']) or
        any(type(s) is not int or s not in (42, 43, 44) for s in o['seeds'])):
        raise ValueError('Use a nonempty unique subset of the existing seeds [42, 43, 44].')
    comparisons = 3 if o['include_direct_judge_student'] else 2
    if o['final_prompts'] < comparisons * len(o['seeds']) * o['review_pairs_per_seed']:
        raise ValueError('Not enough final prompts for the specified blinded review.')
    if type(o.get('allow_downloads')) is not bool:
        raise ValueError('allow_downloads must be a boolean.')
    if o['source_kind'] not in ('refresh2', 'refresh34'):
        raise ValueError('source_kind must be refresh2 or refresh34.')
    for key in ('include_direct_judge_student', 'gradient_checkpointing'):
        if type(o[key]) is not bool:
            raise ValueError('Boolean required: ' + key)
    import math
    for key in ('student_lr', 'student_head_lr', 'student_max_grad_norm'):
        if not math.isfinite(o[key]) or o[key] <= 0:
            raise ValueError('Positive finite value required: ' + key)

def inspect_source(project, o, followup=None, refresh2=None, refresh34=None):
    project, followup, refresh2, f, r, parents, memories = inspect_project(
        project, o['seeds'], followup, refresh2)
    source, manifest = refresh2, r
    if o['source_kind'] == 'refresh34':
        source = resolve_study(project, refresh34, 'refresh34_outputs/latest.json', 'output', 'refresh34_outputs')
        manifest = read(source / 'manifest.json')
        if digest({k: v for k, v in manifest.items() if k != 'identity'}) != manifest['identity']:
            raise ValueError('Refresh34 manifest changed.')
        if manifest['refresh2_identity'] != r['identity'] or read(source / 'status.json')['stage'] != 'complete':
            raise ValueError('Complete the matching refresh3/4 study before selecting it.')
        endpoints = read(source / 'endpoints.json')
        final_lock = read(source / 'final_lock.json')
        parents, memories = {}, {}
        for seed in o['seeds']:
            endpoint = endpoints[f'{seed}/adaptive']
            path = (source / endpoint['checkpoint_relative_path']).resolve()
            path.relative_to(source.resolve())
            ck = checkpoint_info(path)
            if (ck['identity'], ck['seed'], ck['branch'], ck['sha256']) != (
                    manifest['identity'], seed, 'adaptive', endpoint['sha256']):
                raise ValueError('Refresh34 checkpoint provenance mismatch.')
            last_memory = 2 + manifest['options']['rounds']
            expected_update = manifest['parents'][str(seed)]['update'] + manifest['options']['rounds'] * manifest['options']['updates_per_round']
            if ck['update'] != expected_update or final_lock['endpoints'][f'{seed}/adaptive/M{last_memory}'] != ck['sha256']:
                raise ValueError('Refresh34 source is not the locked final adaptive endpoint.')
            memory = source / 'memories' / f'M{2 + manifest["options"]["rounds"]}' / f'seed_{seed}'
            lock = read(memory / 'locked_reward.json')
            if sha(memory / 'locked_reward.json') != endpoint['memory_lock_sha256'] or sha(memory / 'refreshed_memory.npz') != lock['memory_sha256']:
                raise ValueError('Refresh34 endpoint memory changed.')
            parents[seed], memories[seed] = ck, memory
    if len({p['update'] for p in parents.values()}) != 1:
        raise ValueError('All selected source policies must have the same update.')
    return project, refresh2, source, manifest, r['parent_config'], parents, memories

def resolve_study(project, override, latest, key, root_name):
    path = Path(override) if override else Path(read(project / latest)[key])
    if not path.is_absolute():
        path = project / path
    if not path.exists():
        relocated = project / root_name / path.name
        if relocated.exists():
            path = relocated
    if not path.is_dir():
        raise FileNotFoundError('Completed study directory missing: ' + str(path))
    return path.resolve()

def inspect_project(project, seeds, followup=None, refresh2=None):
    project = Path(project).resolve()
    for name, expected in read(ROOT / 'expected_sources.json').items():
        if not (project / name).is_file() or sha(project / name) != expected:
            raise ValueError(name + ' differs from the attached, reviewed source. Inspect the change before using this add-on.')
    if not (project / 'review_form.html').is_file():
        raise FileNotFoundError('Original review_form.html is required in the project.')
    followup = resolve_study(project, followup, 'outputs/latest.json', 'relative_output', 'outputs')
    refresh2 = resolve_study(project, refresh2, 'refresh2_outputs/latest.json', 'output', 'refresh2_outputs')
    f, r = read(followup / 'manifest.json'), read(refresh2 / 'manifest.json')
    if digest({k: v for k, v in f.items() if k != 'identity'}) != f['identity']:
        raise ValueError('Follow-up manifest changed.')
    if digest({k: v for k, v in r.items() if k not in ('identity', 'parent_config')}) != r['identity']:
        raise ValueError('Refresh2 manifest changed.')
    if r['parent_identity'] != f['identity']:
        raise ValueError('Follow-up and refresh2 do not belong to the same experiment.')
    if read(refresh2 / 'status.json')['stage'] != 'complete':
        raise ValueError('Complete RUN_SECOND_REFRESH.ipynb before this continuation.')
    if r['parent_config'] != f['config']:
        # Refresh2 stores execution fields alongside the original configuration.
        scientific = lambda c: {k: v for k, v in c.items()
                                if k not in ('allow_downloads', 'extra_hf_cache', 'max_wall_hours')}
        if scientific(r['parent_config']) != scientific(f['config']):
            raise ValueError('Refresh2 parent configuration differs from its source.')
    for name, expected in f['input_sha256'].items():
        if sha(project / name) != expected:
            raise ValueError('Original input changed: ' + name)
    for name, expected in read(refresh2 / 'data/complete.json')['sha256'].items():
        if sha(refresh2 / 'data' / name) != expected:
            raise ValueError('Refresh2 data changed: ' + name)
    parents, memories = {}, {}
    for seed in seeds:
        folder = refresh2 / 'runs' / f'seed_{seed}' / 'refresh_M2'
        done = read(folder / 'complete.json')
        dep = done['dependency']
        ck = checkpoint_info(folder / 'checkpoints' / f'checkpoint_{dep["end"]:06d}.pt')
        if (ck['identity'], ck['seed'], ck['branch'], ck['update'], ck['sha256']) != (
                r['identity'], seed, 'refresh_M2', dep['end'], done['checkpoint_sha256']):
            raise ValueError('Refresh2 adaptive checkpoint provenance mismatch.')
        memory = refresh2 / 'memories' / f'seed_{seed}'
        lock = read(memory / 'locked_reward.json')
        if sha(memory / 'refreshed_memory.npz') != lock['memory_sha256'] or sha(memory / 'locked_reward.json') != dep['memory_lock']:
            raise ValueError('M2 is not the memory used by the source checkpoint.')
        if lock['k'] != 31 or lock['temperature'] != .05:
            raise ValueError('Unexpected M2 kNN protocol.')
        parents[seed] = ck
        memories[seed] = memory
    if len({p['update'] for p in parents.values()}) != 1:
        raise ValueError('Source seeds must have the same completed PPO update.')
    return project, followup, refresh2, f, r, parents, memories

def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.pending')
    with open(tmp,'w') as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    tmp.replace(p)
def sealed(p,x):
    p=Path(p)
    if p.exists() and read(p)!=x:raise ValueError('Changed dependency: '+str(p))
    write(p,x)
def pause(*args):
    global STOP
    STOP=True
    print('Pause requested; completing current update/batch.',flush=True)
def should_stop(out):return STOP or (Path(out)/'PAUSE').exists()
def status(out,stage,**kw):write(Path(out)/'status.json',{'stage':stage,'updated_unix':time.time(),**kw})
def checkpoint_info(path):
    p=Path(path)
    if not p.is_file():raise FileNotFoundError('Full PPO checkpoint required: '+str(p)+'. Use the original RunPod project; result archives omit optimizer checkpoints.')
    meta=read(p.with_suffix('.json'))
    if sha(p)!=meta['sha256']:raise ValueError('Checkpoint checksum mismatch: '+str(p))
    return {'path':str(p.resolve()),**meta}

def state_fingerprint(trainer):
    import torch
    from peft import get_peft_model_state_dict
    h=hashlib.sha256()
    def walk(v):
        if torch.is_tensor(v):h.update(str((v.dtype,tuple(v.shape))).encode());h.update(v.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(v,dict):
            for k in sorted(v,key=str):h.update(str(k).encode());walk(v[k])
        elif isinstance(v,(list,tuple)):
            for x in v:walk(x)
        else:h.update(repr(v).encode())
    walk({'adapter':get_peft_model_state_dict(trainer.actor.policy),'value':trainer.actor.value_head.state_dict(),'optimizer':trainer.optimizer.state_dict(),
          'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()})
    return h.hexdigest()
