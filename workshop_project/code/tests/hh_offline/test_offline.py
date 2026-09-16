import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score

from common import canonical_hash, file_hash, write_json
from evaluation import save_npz
from hh_offline import data
from hh_offline.metrics import summarize, paired_intervals
from hh_offline.run import fit, predict, resolve, run

PROJECT = Path(__file__).resolve().parents[3]


@pytest.fixture
def saved(tmp_path):
    rng = np.random.default_rng(17)
    coefficient = np.array([.6, -.3, .9, -.7])
    cal = dict(proxy_mean=0., proxy_std=1., judge_mean=0., judge_std=1., theta=.3)
    def make_frame(kind, count):
        x = rng.normal(size=(count, 4)).astype(np.float32)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        proxy = rng.normal(size=count)
        gap = x @ coefficient + rng.normal(scale=.1, size=count)
        prompts = [f'Human: {kind} conversation {i//2}' for i in range(count)]
        frame = pd.DataFrame(dict(prompt=prompts, answer=['answer']*count,
            prompt_id=[f'{kind}{i//2}' for i in range(count)],
            policy_id=[f'p{i%2}' for i in range(count)], conversation_group=[data.group(p) for p in prompts],
            proxy_raw=proxy, proxy_z=proxy, judge_raw=proxy-gap, judge_z=proxy-gap,
            actual_proxy_judge_gap=gap, high_gap=gap>cal['theta'], proxy_truncated=False, judge_truncated=False))
        return frame, x
    train, x = make_frame('train', 40)
    train['split'] = 'training'
    bank = tmp_path/'bank.csv'
    train.to_csv(bank, index=False)
    mem = tmp_path/'memory'
    save_npz(mem/'detectors/gap_knn.npz', vectors=x, gaps=train.actual_proxy_judge_gap.to_numpy(), bank_ids=np.arange(len(train)))
    write_json(mem/'protocol.json', {'bank_sha256': file_hash(bank), 'calibration': cal, 'embedding_identity': 'test-encoder'})
    write_json(mem/'embedding_information.json', {'identity': 'test-encoder'})
    write_json(mem/'complete.json', {'artifact_sha256': {n: file_hash(mem/n) for n in ('protocol.json','embedding_information.json','detectors/gap_knn.npz')}})
    def finish(folder, frame, x):
        folder.mkdir(parents=True, exist_ok=True)
        frame.to_csv(folder/'predictions.csv', index=False)
        save_npz(folder/'features.npz', vectors=x)
        signature = {'cohort': folder.name}
        write_json(folder/'complete.json', {'identity': canonical_hash(signature), 'signature': signature,
            'rows': len(frame), 'csv_sha256': file_hash(folder/'predictions.csv'), 'features_sha256': file_hash(folder/'features.npz')})
    for name in ('validation','test'):
        frame, vx = make_frame(name, 16)
        finish(tmp_path/name, frame, vx)
    config = {'version': 1, 'experiment': 'hh_offline', 'output': str(tmp_path/'outputs'),
        'candidate_bank': str(bank), 'memory': str(mem), 'validation': [str(tmp_path/'validation')],
        'test': [str(tmp_path/'test')], 'transfer': {},
        'settings': {'memory_fractions': [.5,1.], 'subset_seeds': [42,43], 'ridge_alphas': [.01,1.,100.],
            'knn_k': 3, 'knn_temperature': .05, 'knn_k_grid': [1,3,5], 'knn_temperature_grid': [.05,.2],
            'cpu_threads': 1, 'bootstrap_samples': 0, 'bootstrap_seed': 42}}
    recipe = tmp_path/'recipe.yaml'
    recipe.write_text(json.dumps(config), encoding='utf8')
    return recipe, finish


def test_nested_sampling_retains_entire_groups_and_uses_matched_budgets():
    groups = np.repeat(['a','b','c','d','e'], [1,3,2,5,4])
    subsets = data.nested_subsets(groups,[.2,.6,1.],[42,43])
    assert sum(s['fraction']==1 for s in subsets) == 1
    for s in subsets:
        chosen = set(groups[s['indices']])
        assert np.array_equal(s['indices'],np.flatnonzero(np.isin(groups,list(chosen))))
    for seed in [42,43]:
        selected = [set(s['indices']) for s in subsets if s['subset_seed']==seed]
        assert all(a <= b for a,b in zip(selected,selected[1:]))


def test_ridge_matches_library_and_saved_coefficients_predict_without_refitting(saved):
    recipe, _ = saved
    plan = resolve(PROJECT,recipe)
    train,x,protocol=data.memory(plan['bank'],plan['memory'])
    val,vx=data.evaluation(plan['cohorts']['validation'],protocol['calibration'])
    fitted=fit(x,train.gap.to_numpy(),vx,val,plan['settings'])
    losses=[]
    for a in plan['settings']['ridge_alphas']:
        estimator=Ridge(alpha=a,solver='cholesky').fit(x.astype(float),train.gap)
        losses.append((mean_squared_error(val.gap,estimator.predict(vx)),-a,estimator))
    best=min(losses,key=lambda r:r[:2])
    assert fitted['ridge']['alpha']==-best[1]
    predictions=predict(fitted,x,train.gap.to_numpy(),vx)
    np.testing.assert_allclose(predictions['ridge'],best[2].predict(vx),atol=1e-12)
    # Independent brute-force cosine/softmax reference for fixed kNN.
    scores=np.clip(vx@x.T,-1,1)
    ids=np.argsort(-scores,axis=1,kind='stable')[:,:3]
    selected=np.take_along_axis(scores,ids,axis=1)
    w=np.exp((selected-selected[:,:1])/.05)
    expected=(w*train.gap.to_numpy()[ids]).sum(1)/w.sum(1)
    np.testing.assert_allclose(predictions['knn_fixed'],expected,atol=1e-7)


def test_test_labels_never_reach_selection_and_changed_test_does_not_change_models(saved,monkeypatch):
    recipe,finish=saved
    plan=resolve(PROJECT,recipe)
    actual=data.evaluation
    destinations=[]
    def checked(folders,calibration):
        if folders==plan['cohorts']['test']:
            assert (destinations[-1]/'selection_complete.json').is_file()
            assert list((destinations[-1]/'models').glob('*.npz'))
        return actual(folders,calibration)
    monkeypatch.setattr(data,'evaluation',checked)
    destinations.append(plan['output'])
    first=run(plan)
    folder=plan['cohorts']['test'][0]
    frame,x=actual([folder],read_json_cal(plan))
    frame['judge_z']=-frame.judge_z+2
    frame['judge_raw']=frame.judge_z
    frame['actual_proxy_judge_gap']=frame.proxy_z-frame.judge_z
    frame['high_gap']=frame.actual_proxy_judge_gap>.3
    finish(folder,frame,x)
    second_plan=resolve(PROJECT,recipe)
    assert second_plan['output']!=plan['output']
    destinations.append(second_plan['output'])
    second=run(second_plan)
    for p in (plan['output']/'models').glob('*.npz'):
        with np.load(p,allow_pickle=False) as a,np.load(second_plan['output']/'models'/p.name,allow_pickle=False) as b:
            for key in a.files:np.testing.assert_array_equal(a[key],b[key])
    assert first['results'][0]['gap_mse']!=second['results'][0]['gap_mse']


def read_json_cal(plan):
    return json.loads((plan['memory']/'protocol.json').read_text())['calibration']


def test_overlap_and_changed_saved_artifacts_rejected(saved):
    recipe,_=saved
    plan=resolve(PROJECT,recipe)
    frame,_,_=data.memory(plan['bank'],plan['memory'])
    with pytest.raises(ValueError,match='overlap'):data.disjoint(frame,frame)
    f=plan['cohorts']['test'][0]/'features.npz'
    with f.open('ab') as h:h.write(b'corruption')
    with pytest.raises(ValueError,match='checksum'):data.evaluation(plan['cohorts']['test'],read_json_cal(plan))


def test_metrics_and_constant_targets_are_honest():
    f=pd.DataFrame({'group':['a','a','b','b'],'gap':[-1.,.1,.5,1.], 'proxy_z':[1.,0.,1.,0.], 'judge_z':[2.,-.1,.5,-1.]})
    pred=np.array([-.5,.2,.4,.8])
    m=summarize(f,pred,.3)
    assert m['gap_mse']==pytest.approx(mean_squared_error(f.gap,pred))
    assert m['gap_r2']==pytest.approx(r2_score(f.gap,pred))
    assert m['high_gap_auroc']==pytest.approx(roc_auc_score(f.gap>.3,pred))
    assert m['judge_preference_pairs']==2
    f['gap']=1.
    m=summarize(f,np.zeros(4),2.)
    assert m['gap_r2'] is m['high_gap_auroc'] is m['gap_pearson'] is None
    f['gap']=.1
    m=summarize(f,np.full(4,.011916301938982334),2.)
    assert m['gap_r2'] is m['gap_pearson'] is None


def test_bootstrap_preserves_methods_and_grouped_answers():
    f=pd.DataFrame({'group':['a','a','b','b','c','c'],'gap':[-1.,0.,.5,2.,3.,1.]})
    a=np.zeros(6);b=np.asarray(f.gap)*.8
    first=paired_intervals(f,a,b,.3,30,42)
    doubled=pd.concat([f,f],ignore_index=True)
    second=paired_intervals(doubled,np.tile(a,2),np.tile(b,2),.3,30,42)
    for key in first:np.testing.assert_allclose(first[key]['ci95'],second[key]['ci95'])


def test_dry_run_resolve_writes_nothing_and_rejects_unsafe_output(saved):
    recipe,_=saved
    plan=resolve(PROJECT,recipe)
    assert not plan['output'].exists()
    with pytest.raises(ValueError,match='separate directory'):
        resolve(PROJECT,recipe,output=recipe.parent)
