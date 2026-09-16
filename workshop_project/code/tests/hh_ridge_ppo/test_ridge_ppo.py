import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from common import canonical_hash as digest, file_hash as sha, write_json as write
from evaluation import save_npz
from knn_distillation.data import group, schedule
from hh_ridge_ppo import protocol, features, reports
from hh_ridge_ppo.reward import RidgeRouter
from hh_ridge_ppo.run import fit_bundle

PROJECT = Path(__file__).resolve().parents[3]
CAL = dict(proxy_mean=0., proxy_std=1., judge_mean=0., judge_std=1., theta=.2)


@pytest.fixture
def study(tmp_path):
    """Small complete provenance fixture, including mismatched-control rejection."""
    roots = {n: tmp_path/n for n in ('source', 'refresh', 'followup', 'inputs')}
    rng = np.random.default_rng(7)
    x = rng.normal(size=(36, 4)).astype('float32')
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    y = x @ np.array([.4, .1, -.7, .2])
    bank = pd.DataFrame({'prompt': ['memory '+str(i) for i in range(32)], 'proxy_z': y[:32], 'judge_z': 0.})
    roots['inputs'].mkdir()
    bank.to_csv(roots['inputs']/'candidate_bank.csv', index=False)
    write(roots['inputs']/'memory/protocol.json', {'bank_sha256': sha(roots['inputs']/'candidate_bank.csv'), 'calibration': CAL})
    save_npz(roots['inputs']/'memory/detectors/gap_knn.npz', vectors=x[:32], gaps=y[:32], bank_ids=np.arange(32))
    data = {}
    for name, size in [('distill_train',4), ('distill_validation',2), ('distill_offline',2), ('monitor',2), ('final',3)]:
        data[name] = [{'prompt_id': name+str(i), 'prompt': name+' question '+str(i), 'conversation_group': group(name+' question '+str(i))} for i in range(size)]
        write(roots['source']/'data'/f'{name}.json', data[name])
    write(roots['source']/'data/complete.json', {'sha256': {n+'.json': sha(roots['source']/'data'/f'{n}.json') for n in data}, 'counts': {n:len(v) for n,v in data.items()}})
    config = dict(max_new_tokens=256, reward_max_tokens=4096, generation_batch_size=2, rollout_batch_size=2, eval_seed=123)
    options = dict(source_kind='refresh2', ppo_updates=100, seeds=[42,43,44], data_seed=17)
    m = dict(config=config, options=options, parents={}, memories={}, source_sha256={}, runtime_versions={'torch':'fake','transformers':'fake','peft':'fake'})
    def ck(folder, seed, branch, step, identity):
        p = folder/'checkpoints'/f'checkpoint_{step:06d}.pt'
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(f'{seed}/{branch}/{step}'.encode())
        meta = dict(name=p.name, sha256=sha(p), seed=seed, branch=branch, update=step, identity=identity)
        write(p.with_suffix('.json'), meta)
        return meta
    for seed in [42,43,44]:
        parent = roots['refresh']/f'runs/seed_{seed}/refresh_M2'
        m['parents'][str(seed)] = ck(parent,seed,'refresh_M2',300,'refresh')
        folder=roots['refresh']/f'memories/seed_{seed}'
        save_npz(folder/'refreshed_memory.npz',vectors=x,gaps=y)
        lock = dict(k=31,temperature=.05,calibration=CAL,memory_sha256=sha(folder/'refreshed_memory.npz'))
        write(folder/'locked_reward.json',lock)
        m['memories'][str(seed)] = sha(folder/'locked_reward.json')
        write(parent/'complete.json',{'checkpoint_sha256':m['parents'][str(seed)]['sha256'],'dependency':{'memory_lock':m['memories'][str(seed)]}})
        first = roots['followup']/f'evaluations/refresh/refresh_round1/cap_256/round1_knn_signed_s{seed}'
        first.mkdir(parents=True)
        pd.DataFrame({'prompt':['added32','added33'],'actual_proxy_judge_gap':y[32:34]}).to_csv(first/'predictions.csv',index=False)
        write(first/'complete.json',{'csv_sha256':sha(first/'predictions.csv')})
        pd.DataFrame({'prompt':['added34','added35'],'actual_proxy_judge_gap':y[34:]}).to_csv(folder/'added_examples.csv',index=False)
    m={'identity':digest(m),**m}
    write(roots['source']/'manifest.json',m)
    write(roots['source']/'preflight.json',{'passed':True,'source_state_fingerprints':{str(s):'fingerprint'+str(s) for s in [42,43,44]}})
    for seed in [42,43,44]:
        parent=m['parents'][str(seed)]
        rows=schedule(data['distill_train'],100,2,17+seed)
        for arm in ['proxy','knn']:
            folder=roots['source']/f'runs/seed_{seed}/{arm}'
            write(folder/'segment.json',dict(identity=m['identity'],parent_sha256=parent['sha256'],parent_identity='refresh',seed=seed,branch=arm,
                reward_identity=m['memories'][str(seed)] if arm=='knn' else digest(['frozen_proxy',CAL]),training_rows=digest(rows),start=300))
            write(folder/'fork_start.json',{'parent_sha256':parent['sha256'],'state_fingerprint':'fingerprint'+str(seed),'policy_value_optimizer_rng_restored':True})
            endpoint=ck(folder,seed,arm,400,m['identity'])
            write(folder/'history.json',[dict(update=u,segment_start=300,reward_source=arm,prompt_ids=[r['prompt_id'] for r in rows[(u-301)*2:(u-300)*2]],seconds=1.) for u in range(301,401)])
            dest=roots['source']/f'evaluations/final/seed_{seed}/{arm}/update_000400'
            dest.mkdir(parents=True)
            f=pd.DataFrame(data['final']).assign(proxy_truncated=False,judge_truncated=False)
            f.to_csv(dest/'predictions.csv',index=False)
            sig=dict(experiment=m['identity'],seed=seed,branch=arm,checkpoint=endpoint['sha256'],cohort='final',update=400,prompts=digest(data['final']),cap=256,reward_guard=4096,batch=2,eval_seed=123,memory=m['memories'][str(seed)])
            write(dest/'complete.json',dict(identity=digest(sig),signature=sig,csv_sha256=sha(dest/'predictions.csv'),rows=len(f)))
        for cohort in ['validation','offline']:
            rows=[{**r,'origin':origin,'example_id':digest([r['prompt_id'],origin]),'answer':'saved answer','proxy_z':0.} for origin in ['base','parent'] for r in data['distill_'+cohort]]
            sig=dict(study=m['identity'],seed=seed,parent=parent['sha256'],memory=m['memories'][str(seed)],cohort=cohort,prompts=digest(data['distill_'+cohort]),cap=256,batch=2)
            folder=roots['source']/f'labels/seed_{seed}/{cohort}'
            write(folder/'examples.json',rows)
            write(folder/'complete.json',dict(signature=sig,rows=len(rows),rows_hash=digest(rows)))
    return {**roots,'project':PROJECT,'recipe':dict(seeds=[42,43,44],ridge_alphas=[.01,.1,1.],cpu_threads=1)}


def test_audit_checks_actual_gap_budget_and_controls(study):
    result=protocol.audit(study)
    assert result['control_reuse_verified'] and len(result['controls'])==6
    assert all(r['memory_answers']==36 for r in result['memory_budgets'])
    path=study['source']/'runs/seed_43/knn/segment.json'
    altered=protocol.read(path);altered['training_rows']='changed';write(path,altered)
    with pytest.raises(ValueError,match='schedule'):
        protocol.audit(study)


def test_audit_rejects_changed_parent_and_overlap(study):
    path=study['refresh']/'runs/seed_42/refresh_M2/checkpoints/checkpoint_000300.pt'
    path.write_bytes(b'changed')
    with pytest.raises(ValueError,match='checksum'):
        protocol.audit(study)


class Scorer:
    def __init__(self,judge=False):
        self.calls=0;self.judge=judge
    def score(self,prompts,answers,features=False):
        self.calls+=len(prompts)
        return dict(raw=np.ones(len(prompts))*(-.5 if self.judge else 0.),tokens=np.ones(len(prompts),int),truncated=np.zeros(len(prompts),bool),features=np.tile([1.,0.,0.,0.],(len(prompts),1)))


def test_ridge_reward_uses_real_features_without_judge_or_knn():
    proxy,judge=Scorer(),Scorer(True)
    router=RidgeRouter(proxy,judge,CAL,1)
    router.memory={'vectors':np.zeros((31,4))}
    router.set_ridge(np.array([.4,0,0,0]),-.1,'model')
    result=router.score(['q'],['answer'],'ridge')
    np.testing.assert_allclose(result['reward'],[-.3])
    assert proxy.calls==1 and judge.calls==0 and router.cost['knn_queries']==0
    assert router.route_identity('ridge')=='model'


def test_recovery_reuses_labels_and_cache_but_rejects_corruption(tmp_path):
    router=RidgeRouter(Scorer(),Scorer(True),CAL,1)
    rows=[dict(prompt='q'+str(i),prompt_id=str(i),answer='a',proxy_z=0.) for i in range(3)]
    rows[1]['judge_z']=.2
    frame,x=features.recover(tmp_path/'features',rows,router,'study',2,tmp_path)
    assert router.judge.calls==2 and router.proxy.calls==3 and x.shape==(3,4)
    assert frame.judge_label_reused.tolist()==[False,True,False]
    features.recover(tmp_path/'features',rows,router,'study',2,tmp_path)
    assert router.proxy.calls==3
    (tmp_path/'features/features.npz').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='changed'):
        features.load(tmp_path/'features')


def test_fit_is_actual_gap_ridge_selected_without_test_data(study,tmp_path):
    protocol.audit(study)
    rng=np.random.default_rng(5)
    x=rng.normal(size=(12,4)).astype('float32');x/=np.linalg.norm(x,axis=1,keepdims=True)
    val=pd.DataFrame({'group':np.repeat(np.arange(6),2),'gap':x@np.array([.4,.1,-.7,.2])})
    out=tmp_path/'out'
    write(out/'recovered/seed_42/validation/complete.json',{'validated':True})
    bundle=fit_bundle(study,out,42,val,x,'id')
    with np.load(study['refresh']/'memories/seed_42/refreshed_memory.npz') as z:
        model=Ridge(alpha=bundle['metadata']['alpha'],solver='cholesky').fit(z['vectors'].astype(float),z['gaps'])
    np.testing.assert_allclose(bundle['coef'],model.coef_,atol=1e-12)
    for arm in ['proxy','knn']:
        path=study['source']/f'evaluations/final/seed_42/{arm}/update_000400/predictions.csv'
        path.write_text('unused test changed',encoding='utf-8')
    again=fit_bundle(study,out,42,val,x,'id')
    np.testing.assert_array_equal(again['coef'],bundle['coef'])


def test_policy_intervals_are_paired_and_keep_seeds_together():
    frames={}
    for seed in [42,43,44]:
        base=pd.DataFrame(dict(prompt_id=['a','b','c'],prompt=['a','b','c'],answer=['Hello']*3,judge_z=[0.,1.,2.],proxy_z=[0.,0.,0.],high_gap=[False,False,True],response_tokens=[20,25,256],ended_eos=[True,True,False]))
        frames[(seed,'knn')]=base
        frames[(seed,'ridge')]=base.assign(judge_z=base.judge_z+.25).iloc[::-1]
    result,_=reports.policy_intervals(frames,100,7)
    score=next(r for r in result if r['scope']=='seed_mean' and r['metric']=='judge_delta')
    assert score['mean']==score['low']==score['high']==.25
    assert reports.annotate(base).length_capped.tolist()==[False,False,True]
    frames[(43,'ridge')]=frames[(43,'ridge')].iloc[:2]
    with pytest.raises(ValueError,match='prompt mismatch'):
        reports.policy_intervals(frames,100,7)


def test_recovery_rejects_encoder_mismatch(tmp_path):
    router=RidgeRouter(Scorer(),Scorer(True),CAL,1)
    with pytest.raises(ValueError,match='differs from saved'):
        features.recover(tmp_path/'features',[dict(prompt='q',prompt_id='q',answer='a',proxy_z=5.)],router,'id',2,tmp_path)
    assert router.judge.calls==0


def test_recipe_rejects_input_output_collision(tmp_path):
    with pytest.raises(ValueError,match='separate'):
        protocol.resolve(PROJECT,output=PROJECT/'results')


def test_continuation_orchestration_freezes_models_and_only_trains_ridge(study,tmp_path,monkeypatch):
    """Run the full coordinator with fake GPU interfaces and real ridge fitting."""
    import sys
    from types import SimpleNamespace
    import assets
    import knn_distillation.io as io
    import knn_distillation.policy_eval as pe
    from hh_ridge_ppo import run
    protocol.audit(study)
    study['config'].update(cpu_threads=1,reward_batch_size=2)
    study['recipe'].update(allow_downloads=False,extra_hf_cache=None)
    out=tmp_path/'new_run'
    events=[]
    class Trainer:
        def __init__(self,actor,reward,c): self.seed=actor.seed
        def restore(self,path,identity,seed,branch,optimizer=True):
            assert optimizer and branch=='refresh_M2'
            events.append(('restore',seed))
            return 300,[]
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(cuda=SimpleNamespace(get_device_name=lambda:'fake GPU'),version=SimpleNamespace(cuda='test')))
    monkeypatch.setitem(sys.modules,'ppo_engine',SimpleNamespace(PPOTrainer=Trainer))
    monkeypatch.setitem(sys.modules,'run_study',SimpleNamespace(configure=lambda c:None,load_actor=lambda snapshot,c,s:SimpleNamespace(seed=s),
        release=lambda:None,checkpoint_actor=lambda *args:SimpleNamespace()))
    monkeypatch.setitem(sys.modules,'reward_bridge',SimpleNamespace(RewardScorer=lambda snapshot,*args:Scorer(snapshot=='judge')))
    monkeypatch.setattr(assets,'ROOT',out)
    monkeypatch.setattr(assets,'resolve_all',lambda c:{'proxy':'proxy','judge':'judge','policy':'policy'})
    monkeypatch.setattr(run.metadata,'version',lambda name:'test')
    monkeypatch.setattr(run,'encoder_parity',lambda plan,proxy:{'passed':True})
    monkeypatch.setattr(io,'state_fingerprint',lambda trainer:'fingerprint'+str(trainer.seed))
    def recover(folder,rows,router,identity,batch,stop_out):
        if folder.name!='validation':
            assert (out/'selection_complete.json').exists() and (out/'final_lock.json').exists()
        events.append(('recover',folder.name))
        write(folder/'complete.json',{'rows':len(rows)})
        return pd.DataFrame({'group':np.arange(len(rows)), 'gap':np.zeros(len(rows))}),np.tile([1.,0.,0.,0.],(len(rows),1))
    monkeypatch.setattr(features,'recover',recover)
    def train(out,folder,c,assets,router,rows,seed,branch,parent,identity,target,interval,expected):
        assert branch=='ridge' and target in (350,400) and len(rows)==200
        assert expected==study['fingerprints'][seed]
        assert (out/'selection_complete.json').exists()
        router.reset_cost()
        router.score(['q'],['a'],'ridge')
        assert router.cost['teacher_answers']==router.cost['knn_queries']==0
        events.append(('train',seed,target))
        return dict(update=target,path='fake',identity=identity,branch=branch,sha256=str(seed)+str(target))
    monkeypatch.setitem(sys.modules,'knn_distillation.ppo_training',SimpleNamespace(train_to=train))
    def evaluate(actor,router,rows,folder,c,identity,seed,branch,checkpoint,cohort,update,**kwargs):
        assert kwargs['features'] and kwargs['monitor_kl'] and c['eval_seed']==123
        if cohort=='final': assert (out/'final_lock.json').exists()
        events.append(('evaluate',seed,cohort,update))
    monkeypatch.setattr(pe,'evaluate',evaluate)
    monkeypatch.setattr(reports,'final_report',lambda *args:events.append(('report',)))
    run.execute(study,out,'id')
    assert [e for e in events if e[0]=='train']==[('train',s,u) for s in [42,43,44] for u in [350,400]]
    assert len([e for e in events if e[:2]==('recover','validation')])==3
    assert protocol.read(out/'complete.json')['new_PPO_updates']==300
    assert events[-1]==('report',)


def test_complete_reports_export_every_predictor_on_identical_answers(study,tmp_path):
    protocol.audit(study)
    study['recipe'].update(bootstrap_samples=25,bootstrap_seed=8)
    study['config'].update(review_pairs_per_stratum=1,review_seed=18)
    out=tmp_path/'completed'
    bundles,recovered={},{}
    rng=np.random.default_rng(21)
    coef=np.array([.4,.1,-.7,.2])
    for seed in [42,43,44]:
        x=rng.normal(size=(12,4)).astype('float32');x/=np.linalg.norm(x,axis=1,keepdims=True)
        frame=pd.DataFrame(dict(prompt_id=[str(i) for i in range(12)],prompt=['held out '+str(i) for i in range(12)],
            answer=['A saved answer']*12,proxy_z=np.zeros(12),judge_z=-x@coef,response_tokens=20,ended_eos=True))
        frame['group']=frame.prompt.map(group)
        frame['gap']=frame.proxy_z-frame.judge_z
        frame['high_gap']=frame.gap>CAL['theta']
        recovered[seed]={}
        for cohort in ['validation','offline','proxy','knn']:
            folder=out/'recovered'/f'seed_{seed}'/cohort
            write(folder/'examples.json',frame.to_dict('records'))
            save_npz(folder/'features.npz',vectors=x)
            write(folder/'complete.json',dict(answers=12,new_judge_answers=12 if cohort=='validation' else 0,proxy_answers=12,seconds=1.,
                artifacts={n:sha(folder/n) for n in ['examples.json','features.npz']}))
            recovered[seed][cohort]=folder
        for branch in ['proxy','knn','ridge']:
            folder=(out if branch=='ridge' else study['source'])/f'evaluations/final/seed_{seed}/{branch}/update_000400'
            folder.mkdir(parents=True,exist_ok=True)
            frame.to_csv(folder/'predictions.csv',index=False)
            save_npz(folder/'features.npz',vectors=x)
            sig=dict(seed=seed,branch=branch,cohort='final',update=400)
            write(folder/'complete.json',dict(identity=digest(sig),signature=sig,rows=12,csv_sha256=sha(folder/'predictions.csv'),
                features_sha256=sha(folder/'features.npz'),last_invocation_seconds=1.))
        write(out/f'runs/seed_{seed}/ridge/history.json',[dict(segment_start=300,reward_source='ridge',seconds=1.,model_calls=dict(teacher_answers=0,proxy_answers=32)) for _ in range(100)])
        bundles[seed]=dict(coef=coef,intercept=0.,metadata={'fit_seconds':.1})
    reports.final_report(study,out,bundles,recovered)
    table=pd.read_csv(out/'reports/predictor_by_seed.csv')
    assert len(table)==3*4*3
    assert (table[table.predictor=='ridge'].gap_mse<1e-28).all()
    exported=pd.read_csv(out/'reports/all_predictions.csv.gz')
    assert len(exported)==3*4*3*12
    for _,f in exported.groupby(['seed','cohort','prompt_id']):
        assert set(f.predictor)=={'proxy','knn','ridge'}
        assert f.actual_gap.nunique()==1
    assert len(pd.read_csv(out/'reports/policy_by_seed.csv'))==9
    assert protocol.read(out/'review/ridge_m2_blinded/private/key.json')['reference_method']=='knn'
