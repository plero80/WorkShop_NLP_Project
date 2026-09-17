import copy
import json
from pathlib import Path
import sqlite3

import numpy as np
import pytest

from gsm8k_experiment.common import DEFAULT_CONFIG, atomic_json, digest, read_json, write_jsonl
from gsm8k_experiment.memory import GapMemory, Normalization
from gsm8k_ridge import inputs, model, run
from gsm8k_ridge.reports import report
from test_tiny_models import tiny_assets, items

PROJECT = Path(__file__).resolve().parents[3]


@pytest.fixture
def saved_run(tmp_path):
    src = tmp_path/'saved'; src.mkdir()
    c = read_json(DEFAULT_CONFIG)
    rows = lambda name,n:[{'id':name+str(i),'question':name+str(i),'reference':'#### 3','source':'test' if name=='final' else 'train'} for i in range(n)]
    cohorts = {n:rows(n,count) for n,count in {'memory':12,'selection':4,'final':3,'monitor':3,'ppo':4,'calibration':3}.items()}
    c['evaluation']['bootstrap_samples'] = 20
    split = {'cohorts':cohorts,'audit':{}}
    split['fingerprint'] = digest(split)
    resolved = {'policy':'p','proxy':'x','judge':'j','dataset':'d'}
    identity = {'config':c,'resolved':resolved,'split_fingerprint':split['fingerprint'],
                'source':read_json(PROJECT/'code/experiments/gsm8k_ridge/compatibility.json')['source_fingerprints'][0],
                'versions':{},'torch_version':'2.8.0'}
    fingerprint = digest(identity)
    for name,value in [('config.json',c),('manifest.json',{'identity':identity,'fingerprint':fingerprint}),
                       ('resolved_assets.json',resolved),('data/splits.json',split),
                       ('final_protocol.json',{'updates':2,'arms':['proxy','knn_static']})]:
        atomic_json(src/name,value)
    norm = Normalization(2,1,2,1,.8); norm.save(src/'prepared/normalization.json')
    rng = np.random.default_rng(42)
    x = rng.normal(size=(12,4)).astype('float32'); x /= np.linalg.norm(x,axis=1,keepdims=True)
    memory = GapMemory(x,x @ np.array([.1,.2,.5,1]),[r['id'] for r in cohorts['memory']],2,.05,'encoder')
    memory.save(src/'prepared/memory_initial.npz')
    atomic_json(src/'prepared/encoder_probe.json',{})
    db = sqlite3.connect(src/'reward_cache.sqlite')
    db.execute('CREATE TABLE scores (key TEXT PRIMARY KEY,result TEXT,embedding BLOB)')
    def answers(pool,arm):
        result = []
        for i,r in enumerate(pool):
            emb = rng.normal(size=4).astype('float32'); emb /= np.linalg.norm(emb)
            row = {**r,'response':arm+' answer '+str(i),'proxy_score':3.,'judge_score':float(3.-emb @ np.array([.1,.2,.5,1])),
                   'correct':i%2==0,'numeric_match':i%2==0,'numeric_unresolved':False,'format_valid':True,
                   'response_tokens':10,'length_capped':False,'ended_with_eos':True}
            key = digest(['encoder',row['question'],row['reference'],row['response']])
            db.execute('INSERT INTO scores VALUES (?,?,?)',(key,json.dumps({'score':row['proxy_score']}),emb.tobytes()))
            result.append(row)
        return result
    write_jsonl(src/'prepared/selection_raw.jsonl',answers(cohorts['selection'],'selection'))
    write_jsonl(src/'prepared/calibration_raw.jsonl',answers(cohorts['calibration'],'calibration'))
    for arm in ('proxy','knn_static'):
        atomic_json(src/f'arms/{arm}/completed.json',{'update':2,'fingerprint':fingerprint})
    for step in (1,2):
        atomic_json(src/f'arms/knn_static/training/step_{step:06d}.json',{'memory_examples':12})
    for arm in ('base','proxy','knn_static'):
        folder = src/f'evaluations/final/{arm}/step_{0 if arm=="base" else 2:06d}'
        atomic_json(folder/'metrics.json',{'arm':arm,'update':0 if arm=='base' else 2,'cohort':'final'})
        write_jsonl(folder/'responses.jsonl',answers(cohorts['final'],arm))
    db.commit();db.close()
    return src


def test_saved_fit_and_reports_use_identical_answers_without_modifying_source(saved_run,tmp_path):
    before = {p.relative_to(saved_run).as_posix():inputs.sha(p) for p in saved_run.rglob('*') if p.is_file()}
    resolved = run.resolve(PROJECT,sources=[str(saved_run)],output=str(tmp_path/'new'))
    fitted = run.prepare(resolved)
    report(resolved,fitted)
    selected = read_json(tmp_path/'new/seed_42/fitted/selection.json')
    assert selected['test_used'] is False and selected['new_judge_calls']==0
    assert selected['memory_answers']==12 and selected['validation']['valid_pairs']==4
    assert selected['alpha'] == min(selected['candidates'],key=lambda r:(r['validation_question_weighted_mse'],-r['alpha']))['alpha']
    assert 'Pearson' in (tmp_path/'new/report.md').read_text()
    assert (tmp_path/'new/all_predictions.jsonl.gz').is_file()
    np.testing.assert_array_equal(run.prepare(resolved)[42].coef,fitted[42].coef)
    assert before == {p.relative_to(saved_run).as_posix():inputs.sha(p) for p in saved_run.rglob('*') if p.is_file()}
    with pytest.raises(ValueError,match='initial_trainable'):
        run.preflight(resolved['plans'][0])


def test_cache_and_saved_control_mismatches_rejected(saved_run):
    p = inputs.inspect(saved_run)
    rows,x,y,_ = inputs.selection(p)
    rows[0]['response'] += ' changed'
    with pytest.raises(ValueError,match='Missing cached'):
        inputs.features(rows,p['memory'].encoder_identity,[saved_run/'reward_cache.sqlite'])
    completion = read_json(saved_run/'arms/proxy/completed.json');completion['update']=1
    atomic_json(saved_run/'arms/proxy/completed.json',completion)
    with pytest.raises(ValueError,match='same target'):
        inputs.inspect(saved_run)


def test_heldout_labels_never_select_ridge_and_memory_overlap_rejected(saved_run,tmp_path):
    p = inputs.inspect(saved_run)
    first = model.fit(p,tmp_path/'first',[.01,.1,1])
    path = saved_run/'evaluations/final/proxy/step_000002/responses.jsonl'
    path.write_text('heldout data is deliberately unreadable')
    second = model.fit(p,tmp_path/'second',[.01,.1,1])
    np.testing.assert_array_equal(first.coef,second.coef)
    with pytest.raises(ValueError,match='overlap'):
        first.predict(p['memory'].embeddings[:1],[p['memory'].group_ids[0]],'encoder')


def test_ridge_reward_preserves_missing_scores_and_never_calls_judge():
    from gsm8k_experiment.run import reward_for_arm
    c = read_json(DEFAULT_CONFIG)
    memory = GapMemory(np.eye(2,dtype='float32'),np.array([.2,.4]),['m1','m2'],1,.1,'encoder')
    predictor = model.RidgeGap([.3,.7],.1,memory)
    class Proxy:
        identity='encoder'
        def score(self,rows,stage):
            return [{'score':s,'embedding':x,'judge_output':'saved'} for s,x in zip([3.,None],np.eye(2))]
    class Judge:
        def score(self,*args):
            raise AssertionError('Ridge PPO must not call the judge')
    reward,details = reward_for_arm(items(),'ridge',Proxy(),Judge(),Normalization(2,1,2,1,1),predictor,c)
    assert reward[0] == pytest.approx(.6) and np.isnan(reward[1])
    assert details[0]['nearest_similarity'] is None and details[1]['used_for_ppo'] is False
    assert details[1]['optimization_reward'] is None


def test_ridge_rewards_reach_the_shared_PPO_trainer(tiny_assets,monkeypatch):
    from gsm8k_experiment.models import Policy
    from gsm8k_experiment.run import reward_for_arm
    from gsm8k_experiment.ppo import optimizer_for,prepare_rollout,update
    from ppo_engine import PPOTrainer
    c,resolved = tiny_assets
    actor = Policy(c,resolved)
    memory = GapMemory(np.eye(2,dtype='float32'),np.array([.2,.4]),['m1','m2'],1,.1,'encoder')
    predictor = model.RidgeGap([.3,.7],.1,memory)
    class Proxy:
        identity='encoder'
        def score(self,rows,stage):
            return [{'score':s,'embedding':x,'judge_output':'saved'} for s,x in zip([3.,2.],np.eye(2))]
    rewards,_ = reward_for_arm(items(),'ridge',Proxy(),None,Normalization(2,1,2,1,1),predictor,c)
    calls=[];original=PPOTrainer.update
    def tracked(self,*args,**kwargs):
        calls.append(True)
        return original(self,*args,**kwargs)
    monkeypatch.setattr(PPOTrainer,'update',tracked)
    result = update(actor,optimizer_for(actor,c),prepare_rollout(actor,items(),rewards,c),c,0)
    assert calls == [True] and result['optimizer_steps'] > 0


def test_complete_ridge_arm_trains_evaluates_and_resumes_with_tiny_models(tiny_assets,tmp_path,monkeypatch):
    import torch
    from gsm8k_experiment import assets
    from gsm8k_experiment.models import Policy,RewardScorer,ScoreCache
    from gsm8k_experiment.common import read_jsonl
    c,resolved = tiny_assets
    c['ppo'].update(prompts_per_update=2,responses_per_prompt=1,checkpoint_every=1,monitor_every=1)
    source=tmp_path/'source';source.mkdir()
    actor=Policy(c,resolved)
    torch.save(actor.trainable_state(),source/'initial_trainable.pt')
    cache=ScoreCache(source)
    proxy=RewardScorer('proxy',c,resolved,cache)
    probe=proxy._infer(items()[:1],'probe',8)[0]['embedding']
    atomic_json(source/'prepared/encoder_probe.json',{'row':items()[0],'embedding':probe.tolist()})
    cache.close()
    x=np.eye(2,len(probe),dtype='float32')
    memory=GapMemory(x,np.array([.2,-.2]),['memory_a','memory_b'],1,.05,proxy.identity)
    ridge=model.RidgeGap(np.linspace(-.1,.1,len(probe)),.1,memory)
    cohort=[{k:r[k] for k in ('id','question','reference')} for r in items()]
    plan={'config':c,'resolved':resolved,'source':source,'seed':42,'updates':1,'kind':'final',
          'norm':Normalization(2,1,2,1,1),'split':{'cohorts':{n:cohort for n in ('ppo','monitor','final')}},
          'manifest':{'runtime_at_creation':{}}}
    monkeypatch.setattr(run,'preflight',lambda p:None)
    monkeypatch.setattr(assets,'check_runtime',lambda c:{'gpu':'tiny CPU integration'})
    monkeypatch.setattr(assets,'resolve_assets',lambda c,o:resolved)
    folder=tmp_path/'ridge';identity={'test':'shared-training-loop'};fingerprint=digest(identity)
    atomic_json(folder/'manifest.json',{'identity':identity,'fingerprint':fingerprint})
    run.train(plan,folder,ridge,fingerprint)
    assert read_json(folder/'arms/ridge/completed.json')['successful_updates']==1
    final=folder/'evaluations/final/ridge/step_000001/responses.jsonl'
    assert len(read_jsonl(final))==2
    checkpoint=inputs.sha(folder/'arms/ridge/checkpoint.pt')
    run.train(plan,folder,ridge,fingerprint)
    assert inputs.sha(folder/'arms/ridge/checkpoint.pt')==checkpoint
    requests=[r for r in read_jsonl(folder/'judge_calls.jsonl') if r['kind']=='request' and r['role']=='judge']
    assert requests and all(not r['stage'].startswith('training/') for r in requests)
