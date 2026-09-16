"""Scientific isolation and error cases for the added teacher-label comparison."""
from gsm8k_experiment.common import DEFAULT_CONFIG, OUTPUT_ROOT
import copy
import numpy as np
import pytest
from gsm8k_experiment.common import ROOT, atomic_json, load_config, read_json, write_jsonl
from gsm8k_experiment.memory import GapMemory, Normalization
from gsm8k_experiment.run import reward_for_arm
from gsm8k_experiment.teacher_memory import assert_matched, prepare_teacher_memory, load_matched_memory, teacher_config
from gsm8k_experiment.numeric import extract_answer, original_check, recheck

class Proxy:
    identity = 'frozen_proxy'
    def score(self, rows, stage):
        return [{'score': x['proxy_score'], 'embedding': np.array(x['embedding'], np.float32),
                 'judge_output': 'Correctness_score: 3'} for x in rows]

class Teacher:
    identity = 'teacher30'
    def __init__(self): self.calls = []
    def score(self, rows, stage):
        self.calls.append(stage)
        return [{'score': 6-x['judge_score'], 'embedding': None, 'judge_output': 'Correctness_score: 3'} for x in rows]

def test_matched_memory_changes_labels_only_and_resume_checks(tmp_path):
    c = load_config(DEFAULT_CONFIG); p = Proxy(); teacher = Teacher()
    base_norm = Normalization.fit([1,2,4,5], [2,1,5,4], .95, .05)
    emb = np.eye(4, dtype=np.float32)
    base = GapMemory(emb, base_norm.gap([1,2,4,5],[2,1,5,4]), ['memory'+str(i) for i in range(4)], 2,.05,p.identity)
    for name in ('calibration','memory','selection'):
        rows = [{'id':name+str(i), 'question':f'{name} math {i}', 'reference':'#### 2',
                 'response':'The answer is 2.', 'response_tokens':8,'length_capped':False,'ended_with_eos':True,
                 'proxy_score':[1,2,4,5][i], 'judge_score':[2,1,5,4][i], 'embedding':emb[i].tolist()} for i in range(4)]
        write_jsonl(tmp_path/'prepared'/f'{name}_raw.jsonl',rows)
    norm,memory = prepare_teacher_memory(p,teacher,base_norm,base,c,tmp_path,{'judge30b':'abc'})
    assert len(teacher.calls)==3
    assert_matched(base_norm,base,norm,memory)
    assert not np.array_equal(memory.gaps,base.gaps)
    norm2,mem2=load_matched_memory(tmp_path,base_norm,base,c,{'judge30b':'abc'})
    np.testing.assert_array_equal(mem2.gaps,memory.gaps);assert norm2==norm
    with pytest.raises(ValueError,match='identity changed'):
        load_matched_memory(tmp_path,base_norm,base,c,{'judge30b':'different'})
    path=tmp_path/'prepared_30b'/'normalization.json'
    changed=read_json(path);changed['judge_mean']+=1;atomic_json(path,changed)
    with pytest.raises(ValueError,match='artifact changed'):
        load_matched_memory(tmp_path,base_norm,base,c,{'judge30b':'abc'})

def test_new_arm_uses_proxy_and_own_teacher_scale_without_judge_calls():
    c=load_config(DEFAULT_CONFIG);norm=Normalization(3,1,4,.5,2)
    memory=GapMemory(np.eye(2),[1.,-1.],['m1','m2'],1,.1,Proxy.identity)
    class NeverCallJudge:
        def score(self,*args): raise AssertionError('Static training must not call a teacher')
    items=[{'id':'q','question':'2+2','reference':'#### 4','response':'The answer is 4.',
            'proxy_score':4,'embedding':[1.,0.],'ended_with_eos':True,'length_capped':False}]
    reward,details=reward_for_arm(items,'knn_static_30b',Proxy(),NeverCallJudge(),norm,memory,c)
    np.testing.assert_allclose(reward,[0.])
    assert details[0]['predicted_gap']==1
    c['completion_reward']={'format_penalty':.25,'incomplete_penalty':.5}
    reward,details=reward_for_arm(items,'knn_static_30b',Proxy(),NeverCallJudge(),norm,memory,c)
    np.testing.assert_allclose(reward,[-.25]);assert details[0]['task_reward']==0
    items[0].update(length_capped=True,ended_with_eos=False)
    reward,_=reward_for_arm(items,'proxy',Proxy(),NeverCallJudge(),norm,memory,c)
    np.testing.assert_allclose(reward,[.25])

def test_teacher_batch_config_cannot_change_proxy_config():
    c=load_config(DEFAULT_CONFIG);before=copy.deepcopy(c);strong=teacher_config(c)
    assert c==before and strong['scoring']['batch_size']==4
    assert strong['scoring']['mode']==c['scoring']['mode']

@pytest.mark.parametrize('text,value,cap,eos',[
    ('The answer is 12.','12',False,True), (r'Answer: \boxed{12}','12',False,True),
    ('The answer is 12.',None,True,False), ('Maybe 12 or 13.',None,False,True),
    ('Answer: 1/2','1/2',False,True), (r'\boxed{3}\nActually the answer is 4.',None,False,True)])
def test_numeric_extraction_independent_of_reference(text,value,cap,eos):
    assert extract_answer(text,length_capped=cap,ended_with_eos=eos)['prediction']==value

def test_rechecker_includes_all_five_policies_and_rejects_missing_arm(tmp_path):
    c=load_config(DEFAULT_CONFIG);c['dataset']['final']=1;output=tmp_path/'run'
    atomic_json(output/'config.json',c)
    atomic_json(output/'final_protocol.json',{'arms':c['arms'],'updates':400})
    for arm in ['base',*c['arms']]:
        text='The answer is 2.' if arm.startswith('knn') else r'\boxed{2}'
        row={'id':'q1','question':'1+1?','reference':'#### 2','response':text,
             'length_capped':False,'ended_with_eos':True,**original_check(text,'#### 2')}
        step=0 if arm=='base' else 400
        write_jsonl(output/'evaluations'/'final'/arm/f'step_{step:06d}'/'responses.jsonl',[row])
    result=recheck(output,tmp_path/'audit');metrics=read_json(result/'summary.json')['metrics']
    assert len(metrics)==5 and all(m['numeric_matches']==1 for m in metrics)
    assert read_json(result/'COMPLETE.json')['arms'][-1]=='knn_static_30b'
    (output/'evaluations'/'final'/'knn_static_30b'/'step_000400'/'responses.jsonl').unlink()
    with pytest.raises(ValueError,match='knn_static_30b'): recheck(output,tmp_path/'audit2')
