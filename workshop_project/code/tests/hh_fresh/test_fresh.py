import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from common import seed_for
from evaluation import save_npz
from hh_fresh import artifacts,data,run
from hh_ridge_ppo.protocol import read,write,sha,digest
from knn_distillation.data import group,validate_cohorts

PROJECT=Path(__file__).resolve().parents[3]


def rows(prefix,n):
    return [{'prompt_id':prefix+str(i),'prompt':prefix+' question '+str(i),'conversation_group':group(prefix+' question '+str(i))} for i in range(n)]


def test_fresh_plan_needs_no_saved_artifacts_and_budget_is_complete(tmp_path):
    project=tmp_path/'source';(project/'configs').mkdir(parents=True)
    (project/'configs/config.json').write_bytes((PROJECT/'configs/config.json').read_bytes())
    plan=run.resolve(project,PROJECT/'configs/experiments/hh-fresh.yaml')
    b=run.budget(plan)
    assert b['old_experiment_files_required'] is False
    assert b['memory_rows']=={'M0':8000,'M1':9024,'M2':10048}
    assert b['total_PPO_updates_all_branches']==2400
    assert b['PPO_rollout_answers']==76800 and b['new_judge_answers']==64288
    assert not (project/'data').exists() and not plan['output'].exists()


def test_partitions_exclude_test_openings_and_do_not_depend_on_yaml_order():
    train=rows('train',30);test=rows('test',10)+train[:2]
    counts={'memory':8,'calibration':2,'validation':3,'offline':3,'final':4,'refresh2_eval':3}
    split=data.partition(train,test,counts,23)
    assert split==data.partition(train,test,dict(reversed(list(counts.items()))),23)
    validate_cohorts(split)
    test_groups={r['conversation_group'] for r in test}
    assert all(not {r['conversation_group'] for r in split[k]} & test_groups for k in counts if k not in ('final','refresh2_eval'))
    with pytest.raises(ValueError,match='Insufficient'):
        data.partition(train,test,{'memory':100},23)


def test_calibration_and_memory_preserve_signed_actual_gaps(tmp_path):
    frame=pd.DataFrame({'proxy_raw':[-2.,0.,1.,4.],'judge_raw':[-1.,2.,0.,5.]})
    cal=artifacts.calibrate(frame,.95)
    expected=(frame.proxy_raw-frame.proxy_raw.mean())/frame.proxy_raw.std(ddof=0)-(frame.judge_raw-frame.judge_raw.mean())/frame.judge_raw.std(ddof=0)
    np.testing.assert_allclose(artifacts.normalize(frame.assign(conversation_group=list('abcd')),cal).gap,expected)
    x=np.eye(4,dtype='float32')
    f=frame.assign(conversation_group=list('abcd'))
    first=artifacts.memory(tmp_path/'M0',f,x,cal,{'source':'initial'})
    added=f.iloc[:2].copy()
    second=artifacts.memory(tmp_path/'M1',added,x[:2],cal,{'source':'refresh'},first)
    with np.load(second/'refreshed_memory.npz') as z:
        np.testing.assert_allclose(z['gaps'],np.r_[expected,expected[:2]])
        np.testing.assert_array_equal(z['vectors'][:4],x)
        assert (z['gaps']<0).any()
    with pytest.raises(ValueError,match='dependencies changed'):
        artifacts.memory(second,added,x[:2],cal,{'source':'changed'},first)


class Scorer:
    def __init__(self,judge=False):self.judge=judge;self.calls=0
    def score(self,prompts,answers,features=False):
        self.calls+=len(prompts)
        x=np.tile([1.,0.,0.,0.],(len(prompts),1)).astype('float32')
        return {'raw':np.full(len(prompts),.3 if self.judge else .5),'tokens':np.ones(len(prompts),int),
                'truncated':np.zeros(len(prompts),bool),'features':x}


def test_collection_resumes_without_regenerating_answers(tmp_path,monkeypatch):
    calls=[]
    class Actor:
        def generate(self,prompts,seed):
            calls.append(seed)
            return [{'answers':['answer']*len(prompts),'response_mask':np.ones((len(prompts),2),bool),'ended_eos':[True]*len(prompts)}]
    monkeypatch.setitem(sys.modules,'run_study',SimpleNamespace(load_actor=lambda *args:Actor(),checkpoint_actor=lambda *args:Actor(),release=lambda:None))
    proxy,judge=Scorer(),Scorer(True)
    args=(tmp_path,tmp_path/'samples',rows('cal',5),[('base',42,None,2)],{'generation_batch_size':2},{'data_seed':12},{'policy':'base'},proxy,judge,'id','calibration')
    frame,x=artifacts.collect(*args)
    assert len(frame)==10 and x.shape==(10,4) and len(calls)==6
    artifacts.collect(*args)
    assert len(calls)==6 and proxy.calls==judge.calls==10
    (tmp_path/'samples/features.npz').write_bytes(b'bad')
    with pytest.raises(ValueError,match='changed'):
        artifacts.collect(*args)


def test_full_fresh_coordinator_and_reports_without_old_data(tmp_path,monkeypatch):
    """GPU boundaries are fake; real calibration, memories, ridge, routing and reports run."""
    plan=run.resolve(PROJECT,output=tmp_path/'fresh')
    r,c=plan['recipe'],plan['config']
    r.update(stage_updates=2,monitor_every=1,bootstrap_samples=20,memory_answers_per_prompt=1,calibration_answers_per_prompt=2)
    r['counts']={name:32 if name.startswith('train_') or name=='memory' else 6 for name in run.COUNTS}
    c.update(cpu_threads=1,review_pairs_per_stratum=1)
    cohorts={name:rows(name,n) for name,n in r['counts'].items()}
    events=[];out=tmp_path/'run'
    monkeypatch.setattr(run,'prepare',lambda *args:cohorts)
    def collect(out,folder,prompts,origins,c,r,assets,proxy,judge,identity,cohort):
        if cohort=='offline':assert (out/'final_lock.json').exists()
        rng=np.random.default_rng(seed_for(cohort,str(origins)))
        examples=[]
        for origin,seed,ck,repeats in origins:
            for sample in range(repeats):
                examples.extend([{**row,'origin':origin,'seed':seed,'answer':'A fresh answer','response_tokens':20,'ended_eos':True} for row in prompts])
        f=pd.DataFrame(examples);n=len(f)
        x=rng.normal(size=(n,4)).astype('float32');x/=np.linalg.norm(x,axis=1,keepdims=True)
        f['proxy_raw']=rng.normal(size=n)
        f['judge_raw']=f.proxy_raw-x@np.array([.2,-.5,.6,.9])
        write(folder/'examples.json',f.to_dict('records'));save_npz(folder/'features.npz',vectors=x)
        write(folder/'complete.json',{'answers':n,'seconds':1.,'new_judge_answers':n,'proxy_answers':n,'artifacts':{name:sha(folder/name) for name in ['examples.json','features.npz']}})
        events.append(('samples',cohort))
        return f,x
    monkeypatch.setattr(run,'collect',collect)
    def initial(out,c,assets,seed,identity):
        return {'update':0,'seed':seed,'branch':'initial','identity':identity,'sha256':str(seed)+'base','path':'base'}
    monkeypatch.setattr(run,'initial_checkpoint',initial)
    def train_to(out,folder,c,assets,router,prompts,seed,branch,parent,identity,target,interval,expected):
        name=folder.name;start=parent['update']
        if branch in ('proxy','ridge') or name=='knn':assert (out/'selection_complete.json').exists()
        fingerprint=f'{seed}/{start}'
        assert expected is None or expected==fingerprint
        write(folder/'fork_start.json',{'state_fingerprint':fingerprint})
        write(folder/'segment.json',{'start':start,'branch':branch})
        history=[]
        for update in range(start+1,target+1):
            router.reset_cost();router.score(['q']*32,['a']*32,branch)
            assert router.cost['teacher_answers']==0
            history.append({'segment_start':start,'reward_source':branch,'seconds':1.,'model_calls':copy.deepcopy(router.cost)})
        write(folder/'history.json',history)
        events.append(('train',seed,name,target,len(router.memory['gaps'])))
        return {'update':target,'seed':seed,'branch':branch,'identity':identity,'sha256':digest([seed,name,target]),'path':'fake'}
    monkeypatch.setitem(sys.modules,'knn_distillation.ppo_training',SimpleNamespace(train_to=train_to))
    def evaluate(out,plan,assets,router,prompts,seed,label,route,ck,identity,cohort):
        assert (out/'selection_complete.json').exists()
        if cohort!='monitor':assert (out/'final_lock.json').exists()
        folder=out/'evaluations'/cohort/f'seed_{seed}'/label/f'update_{ck["update"]:06d}'
        rng=np.random.default_rng(seed_for(cohort,label,seed));n=len(prompts)
        x=rng.normal(size=(n,4)).astype('float32');x/=np.linalg.norm(x,axis=1,keepdims=True)
        f=pd.DataFrame(prompts).assign(answer='A fresh answer',response_tokens=20,ended_eos=True,proxy_raw=rng.normal(size=n),judge_raw=rng.normal(size=n))
        f=artifacts.normalize(f,router.calibration)
        folder.mkdir(parents=True,exist_ok=True);f.to_csv(folder/'predictions.csv',index=False);save_npz(folder/'features.npz',vectors=x)
        signature={'cohort':cohort,'seed':seed,'branch':route,'update':ck['update']}
        write(folder/'complete.json',{'rows':n,'last_invocation_seconds':1.,'identity':digest(signature),'signature':signature,
            'csv_sha256':sha(folder/'predictions.csv'),'features_sha256':sha(folder/'features.npz')})
        events.append(('evaluate',cohort,label))
    monkeypatch.setattr(run,'evaluate_policy',evaluate)
    run.execute(plan,out,{'policy':'base'},Scorer(),Scorer(True),'fresh-id')
    for seed in [42,43,44]:
        stages=[e for e in events if e[0]=='train' and e[1]==seed]
        assert [e[2] for e in stages[:5]]==['parent_M0','static_M0','refresh_M1','static_M1','refresh_M2']
        assert [e[4] for e in stages[:5]]==[32,32,38,38,44]
        assert all(e[4]==44 for e in stages[5:])
    budget=read(out/'reports/budget.json')
    assert budget['realized_PPO_updates']==48
    assert budget['realized_judge_answers']==run.budget(plan)['new_judge_answers']
    metrics=pd.read_csv(out/'reports/predictor_by_seed.csv')
    assert len(metrics)==3*3*9
    assert {'gap_pearson','gap_spearman','gap_r2','high_gap_auroc','high_gap_ap'}<=set(metrics)
    refresh=pd.read_csv(out/'reports/refresh2_policy_by_seed.csv')
    assert len(refresh)==9
    assert (out/'review/fresh_ridge_blinded/fresh_ridge_blinded_BLINDED.zip').exists()
