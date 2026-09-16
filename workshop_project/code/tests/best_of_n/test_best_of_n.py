"""CPU tests for scientific correctness, leakage exclusion and recovery."""
import json, sys, tempfile, types, unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from bon_io import HERE, read, write, save_shard, load_shard, validate_settings, digest
from bon_metrics import selection_rows, interval, report

def candidates(p, j, k, prompt='p', generator='proxy'):
    return [{'prompt_id':prompt,'prompt':'Question '+prompt,'candidate_id':i,'generator':generator,
        'proxy_z':a,'judge_z':b,'corrected_z':c,'actual_gap':a-b,'theta':.9,
        'response_tokens':10+i,'answer_cap':False,'answer':'Answer '+str(i)} for i,(a,b,c) in enumerate(zip(p,j,k))]

class SelectionTests(unittest.TestCase):
    def test_perfect_correction_recovers_best(self):
        rs=selection_rows(candidates([3,2,1],[0,1,2],[0,1,2]),[1,3])
        n1=[r for r in rs if r['n']==1]
        self.assertEqual(len({r['candidate_id'] for r in n1}),1)
        m={r['selector']:r for r in rs if r['n']==3}
        self.assertEqual(m['proxy']['judge_regret'],2)
        self.assertEqual(m['knn']['judge_regret'],0)
        self.assertEqual(m['knn']['knn_pairwise_agreement'],1)
        self.assertEqual(m['proxy']['proxy_pairwise_agreement'],0)
    def test_offset_reduces_mse_but_not_selection(self):
        rs=selection_rows(candidates([3,2,1],[1,0,-1],[1,0,-1]),[1,3])
        r=next(r for r in rs if r['n']==3 and r['selector']=='knn')
        self.assertEqual(r['raw_mse'],4);self.assertEqual(r['corrected_mse'],0)
        self.assertEqual(r['raw_centered_mse'],0);self.assertEqual(r['knn_changes_choice'],0)
    def test_judge_scores_cannot_change_knn_choice(self):
        for j in ([100,0,-1],[-1,0,100]):
            rs=selection_rows(candidates([3,2,1],j,[1,4,2]),[1,3])
            r=next(r for r in rs if r['n']==3 and r['selector']=='knn')
            self.assertEqual(r['candidate_id'],1)
    def test_ties_duplicates_and_incomplete_pool(self):
        rs=selection_rows(candidates([2,2],[0,0],[1,1]),[1,2])
        self.assertTrue(all(r['candidate_id']==0 for r in rs))
        self.assertTrue(all(r['proxy_pairwise_agreement'] is None for r in rs))
        with self.assertRaises(ValueError):selection_rows(candidates([1],[1],[1]),[1,2])
        bad=candidates([1,2],[1,2],[1,2]);bad[1]['candidate_id']=0
        with self.assertRaises(ValueError):selection_rows(bad,[1,2])
    def test_bootstrap_paired_and_deterministic(self):
        self.assertEqual(interval([2,2,2],200,10),{'mean':2.,'ci_low':2.,'ci_high':2.})
        self.assertEqual(interval([-1,1,2],100,20),interval([-1,1,2],100,20))
    def test_shard_recovery_rejects_corruption_and_changed_protocol(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'shard.json';save_shard(p,{'v':1},[{'id':0}])
            self.assertEqual(load_shard(p,{'v':1}),[{'id':0}])
            with self.assertRaises(ValueError):load_shard(p,{'v':2})
            d=read(p);d['rows'][0]['id']=9;write(p,d)
            with self.assertRaises(ValueError):load_shard(p,{'v':1})
    def test_exclusion_groups_include_previous_memory_and_final(self):
        fake=types.ModuleType('chat_format');fake.prompt_messages=lambda p:[{'role':'user','content':p}]
        with patch.dict(sys.modules,{'chat_format':fake}):
            from bon_data import collect_forbidden,group
            with tempfile.TemporaryDirectory() as t:
                p=Path(t);write(p/'inputs/forbidden_opening_groups.json',[group('Old')])
                write(p/'next_studies_outputs/suite_a/data/oracle_final.json',[{'prompt':'Final'}])
                write(p/'best_of_n_outputs/study_old/data/confirmation.json',[{'prompt':'Reserved'}])
                write(p/'best_of_n_outputs/self/data/development.json',[{'prompt':'Own'}])
                mem=p/'next_studies_outputs/suite_a/oracle/memory';mem.mkdir(parents=True)
                (mem/'added_examples.csv').write_text('prompt,answer\nMemory,answer\n')
                banned,_=collect_forbidden(p,p/'best_of_n_outputs/self')
                self.assertTrue({group('old'),group(' FINAL '),group('reserved'),group('memory')}<=banned)
                self.assertNotIn(group('Own'),banned)
    def test_report_end_to_end(self):
        s=read(HERE/'settings.json');s.update(policies=['proxy'],pool_sizes=[1,3],candidates=3,primary_n=3,bootstrap_draws=100)
        with tempfile.TemporaryDirectory() as t:
            rows=candidates([3,2,1],[0,1,2],[0,1,2],prompt='a')+candidates([3,2,1],[0,1,2],[0,1,2],prompt='b')
            result=report(Path(t),s,rows)
            self.assertEqual(result['mean'],2);self.assertEqual(result['ci_low'],2)
            self.assertEqual(result['prompts'],2)
            self.assertTrue((Path(t)/'reports/selection_curves.png').is_file())
    def test_default_settings_valid(self):validate_settings(read(HERE/'settings.json'))

class RecoveryIntegrationTests(unittest.TestCase):
    def test_runner_reuses_generated_and_scored_batches(self):
        from bon_run import evaluate_generator
        counters={'generate':0,'proxy':0,'judge':0}
        class Mask:
            def sum(self,axis):return [2,2]
        class Actor:
            def generate(self,prompts,seed,batch_size):
                counters['generate']+=1
                return [{'answers':['A','B'],'response_mask':Mask(),'ended_eos':[True,True]}]
        class Scorer:
            def __init__(self,role):self.role=role
            def score(self,prompts,answers,features=False):
                counters[self.role]+=1
                r={'raw':np.array([1.,2.]),'tokens':np.array([5,5]),'truncated':np.array([False,False])}
                if features:r['features']=np.array([[1.,0],[0,1.]])
                return r
        common=types.ModuleType('common');common.seed_for=lambda *x:1
        run=types.ModuleType('run_study');run.checkpoint_actor=lambda *x:Actor();run.release=lambda:None
        bridge=types.ModuleType('reward_bridge');bridge.predict_memory=lambda *x,**kw:(np.array([0.,1.]),np.array([0.,0.]))
        frozen=types.SimpleNamespace(scorer=Scorer('proxy'),calibration={'proxy_mean':0.,'proxy_std':1.,'judge_mean':0.,'judge_std':1.,'theta':.9})
        with patch.dict(sys.modules,{'common':common,'run_study':run,'reward_bridge':bridge}):
            with tempfile.TemporaryDirectory() as t:
                p=Path(t);rows=[{'prompt':'Q','prompt_id':'q','conversation_group':'g'}]
                s={'candidates':2,'candidate_batch_size':2,'policy_seed':42,'generation_seed':1,'answer_cap':256}
                ck={'proxy':{'path':'x','identity':'i','branch':'proxy'}}
                args=(p,p,'id','development',rows,'proxy',ck,{'cpu_threads':1},s,{'policy':'x'},frozen,Scorer('judge'),{})
                first=evaluate_generator(*args);second=evaluate_generator(*args)
                self.assertEqual(first,second);self.assertEqual(counters,{'generate':1,'proxy':1,'judge':1})
                # Interrupt after generation: deleting only scoring must not regenerate answers.
                next((p/'development/runs/proxy/scored').glob('*.json')).unlink()
                third=evaluate_generator(*args)
                self.assertEqual([r['answer'] for r in third],['A','B'])
                self.assertEqual(counters,{'generate':1,'proxy':2,'judge':2})

if __name__=='__main__':unittest.main()
