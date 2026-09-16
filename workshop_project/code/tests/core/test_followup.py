"""Follow-up correctness gates. Synthetic CPU models/data; no GPU outcomes claimed."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import numpy as np
import pandas as pd
import torch
from common import ROOT,read_json,write_json,canonical_hash,StopRequested
from test_engine import toy_actor,adapter_copy,ToyTokenizer,ToyReward
from evaluation import evaluate,load_completed
from reward_bridge import RewardScorer,FrozenGapReward,bounded_gap
from review import make_pack,summarize_review,RATING_FIELDS
from ppo_engine import PPOTrainer


class SyntheticScorer:
    calls=0
    def score(self,prompts,answers,features=False):
        self.calls+=1
        raw=np.array([sum(map(int,a.split()))/100 if a else 0. for a in answers])
        vectors=[]
        for p,a in zip(prompts,answers):
            v=np.random.default_rng(int(canonical_hash([p,a])[:8],16)).normal(size=32)
            vectors.append(v/np.linalg.norm(v))
        result={'raw':raw,'tokens':np.array([len(p)+len(a) for p,a in zip(prompts,answers)]),
                'truncated':np.zeros(len(prompts),bool)}
        if features:result['features']=np.asarray(vectors,np.float32)
        return result


class SyntheticReward(FrozenGapReward):
    def __init__(self,root):
        self.scorer=SyntheticScorer();self.threads=2;self.updated=None;self.parameters=None;self.lock_hash=None
        self.calibration={'proxy_mean':0.,'proxy_std':1.,'judge_mean':0.,'judge_std':1.,'theta':.5}
        rng=np.random.default_rng(71);v=rng.normal(size=(40,32));v=(v/np.linalg.norm(v,axis=1)[:,None]).astype(np.float32)
        self.corrector=SimpleNamespace(model={'vectors':v,'gaps':np.linspace(-1,1,40)},cutoff=.8,result_dir=Path(root))
        (Path(root)/'detectors').mkdir(parents=True,exist_ok=True)
        np.savez(Path(root)/'detectors/gap_knn.npz',**self.corrector.model)
        def correct(raw,features,mode):
            gap=features[:,0]
            return {'predicted_gap':gap,'mean_neighbor_distance':np.full(len(raw),.5)}
        self.corrector.correct_from_embeddings=correct


class SyntheticJudge(SyntheticScorer):
    def score(self,prompts,answers,features=False):
        result=super().score(prompts,answers,features)
        result['raw']=np.asarray([3. if i%2 else 0. for i in range(len(prompts))])
        return result


def rows(prefix,n):
    return [{'prompt':f'{prefix}{i}','prompt_id':canonical_hash(f'{prefix}{i}'),
             'conversation_group':canonical_hash(f'{prefix}{i}')} for i in range(n)]


class FollowupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_whole_answer_scoring_and_guard(self):
        from transformers import Qwen3Config,Qwen3ForSequenceClassification
        config=Qwen3Config(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=1,
                          num_attention_heads=4,num_key_value_heads=2,head_dim=8,num_labels=1,pad_token_id=0)
        scorer=RewardScorer('unused',device='cpu',max_length=4096,
            model=Qwen3ForSequenceClassification(config),tokenizer=ToyTokenizer())
        _,length=scorer.encode_full(['request'],['a'*1200])
        self.assertGreater(int(length[0]),1200)
        scorer.max_length=1024
        with self.assertRaisesRegex(ValueError,'No prompt or answer was truncated'):
            scorer.encode_full(['request'],['a'*1200])

    def test_adapter_load_and_shared_optimizer_fork(self):
        import run_study
        actor,c=toy_actor();trainer=PPOTrainer(actor,ToyReward(),c)
        prompts=rows('train',4);trainer.update(prompts,'knn_signed',42,1)
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td);actor.policy.save_pretrained(folder/'adapter')
            checkpoint=trainer.checkpoint(folder,1,'study',42,'knn_signed',[])
            def load(*args):return toy_actor()[0]
            with patch('run_study.PPOActor.load',side_effect=load):
                imported=run_study.load_actor('unused',c,42,folder/'adapter')
            for k,v in adapter_copy(imported).items():torch.testing.assert_close(v,adapter_copy(actor)[k],atol=0,rtol=0)
            forks=[]
            for branch in ['knn_signed','iterative_knn']:
                fork,_=toy_actor();t=PPOTrainer(fork,ToyReward(),c);step,_=t.restore(checkpoint,'study',42,'knn_signed')
                self.assertEqual(step,1)
                self.assertEqual([int(v['step']) for v in t.optimizer.state.values()],[int(v['step']) for v in trainer.optimizer.state.values()])
                for key,value in trainer.optimizer.state_dict()['state'].items():
                    for name,tensor in value.items():torch.testing.assert_close(t.optimizer.state_dict()['state'][key][name],tensor,atol=0,rtol=0)
                for k,v in adapter_copy(fork).items():torch.testing.assert_close(v,adapter_copy(actor)[k],atol=0,rtol=0)
                forks.append(fork)
            torch.testing.assert_close(forks[0].value_head.weight,forks[1].value_head.weight,atol=0,rtol=0)

    def test_signed_main_and_capped_ablation_are_distinct(self):
        with tempfile.TemporaryDirectory() as td:
            reward=SyntheticReward(td);reward.updated={'vectors':np.eye(32,dtype=np.float32), 'gaps':np.full(32,-1.)}
            reward.parameters={'alpha':.5,'bonus_cap':.1,'distance_gate':False}
            main=reward.score(['hi'],['2'],'iterative_knn');cap=reward.score(['hi'],['2'],'iterative_capped')
            self.assertAlmostEqual(main['applied_gap'][0],-1.,places=6)
            self.assertAlmostEqual(cap['applied_gap'][0],-.1)
            np.testing.assert_allclose(bounded_gap([-2,2],[.2,.9],{'alpha':.5,'bonus_cap':.1,'distance_gate':True},.5),[-.1,0])

    def test_eval_resume_identity_and_full_gap_labels(self):
        actor,c=toy_actor();judge=SyntheticJudge()
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);reward=SyntheticReward(root/'memory');folder=root/'eval';prompts=rows('eval',3)
            kwargs=dict(actor=actor,reward=reward,judge=judge,rows=prompts,folder=folder,c=c,identity='study',
                        policy_id='static',branch='knn_signed',seed=42,checkpoint_hash='weights',cohort='test',
                        family='round2',cap=6,save_features=True)
            stop=iter([False,True])
            with self.assertRaises(StopRequested):evaluate(**kwargs,should_stop=lambda:next(stop))
            self.assertEqual(judge.calls,1)
            evaluate(**kwargs);self.assertEqual(judge.calls,2)
            frame,vec=load_completed(folder,features=True)
            self.assertEqual(vec.shape,(3,32))
            np.testing.assert_array_equal(frame.high_gap,(frame.proxy_z-frame.judge_z)>.5)
            np.testing.assert_allclose(frame.corrected_judge_residual,frame.proxy_z-frame.applied_gap-frame.judge_z)
            evaluate(**kwargs);self.assertEqual(judge.calls,2)
            with self.assertRaises(ValueError):evaluate(**(kwargs|{'cap':7}))
            self.assertEqual(actor.c['max_new_tokens'],6)

    def test_review_blinding_orientation_validation_and_unblinding(self):
        data=pd.DataFrame(rows('review',8));data['answer']='raw answer </script>'
        other=data.copy();other['answer']='other answer'
        frames={('raw',42,256):data,('iterative_knn',42,256):other}
        c={'review_pairs_per_stratum':4,'review_seed':77}
        with tempfile.TemporaryDirectory() as td:
            folder=make_pack(td,'blind',frames,['iterative_knn'],c);key=read_json(folder/'private/key.json')
            key['reference_method']='knn_signed';write_json(folder/'private/key.json',key)
            public=read_json(folder/'reviewer/pairs.json')
            self.assertEqual(len({p['pair_id'] for p in public['pairs']}),4)
            self.assertEqual(sum(p['other_side']=='A' for p in key['pairs']),2)
            for pair in public['pairs']:self.assertEqual(set(pair),{'pair_id','prompt','answers'})
            with zipfile.ZipFile(folder/'blind_BLINDED.zip') as z:
                self.assertEqual(set(z.namelist()),{'README.txt','REVIEW.html','pairs.json'})
                self.assertNotIn('raw answer </script>',z.read('REVIEW.html').decode())
            self.assertEqual(make_pack(td,'blind',frames,['iterative_knn'],c),folder)
            ratings=[]
            for p in key['pairs']:
                r={field:sorted(choices)[0] for field,choices in RATING_FIELDS.items()}
                r.update(review_batch_id=key['review_batch_id'],pair_id=p['pair_id'],reviewer_id='r1',
                    usefulness_A='2',usefulness_B='2',preference=p['other_side'],inappropriate_refusal_A='no',inappropriate_refusal_B='no')
                r['usefulness_'+p['other_side']]='4';ratings.append(r)
            path=Path(td)/'ratings.json';write_json(path,{'review_batch_id':key['review_batch_id'],'ratings':ratings})
            summary=summarize_review(folder,[path],draws=30);df=pd.read_csv(summary/'human_summary.csv')
            self.assertEqual(df.loc[df.metric=='usefulness_delta','mean'].iloc[0],2.)
            self.assertTrue((df.reference_method=='knn_signed').all())
            self.assertTrue(read_json(summary/'review_status.json')['complete_pair_coverage'])
            ratings[0]['pair_id']='unknown';write_json(path,{'review_batch_id':key['review_batch_id'],'ratings':ratings})
            with self.assertRaises(ValueError):summarize_review(folder,[path],draws=30)

    def test_packaged_inputs_and_conversation_groups(self):
        import run_study
        from chat_format import prompt_messages
        data,train=run_study.load_data()
        def opening(row):return canonical_hash(next(m['content'].strip() for m in prompt_messages(row['prompt']) if m['role']=='user'))
        # Recompute normalized opening strings independently of recorded hashes.
        def normalized(row):return ' '.join(next(m['content'] for m in prompt_messages(row['prompt']) if m['role']=='user').casefold().split())
        all_new=[]
        for role in ['development_fit','development_validation','offline_test','refresh_round1','fresh_final']:
            all_new.extend(normalized(r) for r in data[role])
        self.assertEqual(len(all_new),len(set(all_new)))
        self.assertFalse(set(all_new)&{normalized(r) for rs in train.values() for r in rs})

    def test_tiny_two_round_pipeline_real_ppo_and_resume(self):
        import run_study
        actor,c=toy_actor();del actor
        c.update(seeds=[42],updates=2,round1_updates=1,rollout_batch_size=4,checkpoint_every=1,
                 bootstrap_draws=20,review_pairs_per_stratum=1,selection_alphas=[.5],selection_bonus_caps=[.1],
                 selection_distance_gates=[False],run_new_ppo=True,run_capped_ablation=True)
        data={role:rows(role,8 if role in ['legacy_eval','fresh_final'] else 2)
              for role in ['legacy_eval','fresh_final','development_fit','development_validation','offline_test','refresh_round1']}
        train={42:rows('train',8)}
        def load(snapshot,config,seed,adapter=None):
            actor,_=toy_actor();actor.c={**config,'max_new_tokens':6}
            original=actor.generate
            def short_generate(prompts,seed):
                cap=actor.c['max_new_tokens'];actor.c['max_new_tokens']=6
                try:return original(prompts,seed)
                finally:actor.c['max_new_tokens']=cap
            actor.generate=short_generate
            return actor
        with tempfile.TemporaryDirectory() as td:
            output=Path(td);reward=SyntheticReward(output/'synthetic_memory');judge=SyntheticJudge()
            write_json(output/'manifest.json',{'identity':'tiny_followup','config':c})
            with (patch('run_study.load_actor',side_effect=load),patch('run_study.load_rewards',return_value=(reward,judge)),
                  patch('run_study.export_results',return_value=output/'mock_export.zip')):
                run_study.run(c,output,{'policy':'unused'},data,train)
                self.assertEqual(read_json(output/'status.json')['stage'],'complete')
                self.assertEqual(read_json(output/'reports/report_status.json')['completed_main_final_policies'],4)
                locks=read_json(output/'refresh/seed_42/locked_reward.json')
                self.assertEqual(locks['new_memory_rows'],2);self.assertFalse(locks['main_parameters_changed'])
                with np.load(output/'refresh/seed_42/refreshed_memory.npz') as mem:self.assertEqual(len(mem['gaps']),42)
                forks=[read_json(output/'runs/round2/seed_42'/b/'fork_start.json') for b in ['knn_signed','iterative_knn','iterative_capped']]
                self.assertEqual(forks[0],forks[1]);self.assertEqual(forks[0],forks[2])
                oldcalls=judge.calls
                run_study.run(c,output,{'policy':'unused'},data,train)
                self.assertEqual(judge.calls,oldcalls)
                self.assertTrue(read_json(output/'final_evaluation_lock.json')['all_training_complete'])
                paired=pd.read_csv(output/'reports/paired_differences.csv')
                self.assertIn('iterative_knn minus knn_signed',set(paired.comparison))
                main=read_json(output/'review/iterative_vs_static_review/private/key.json')
                second=read_json(output/'review/new_policy_vs_raw_review/private/key.json')
                self.assertFalse({r['prompt_id'] for r in main['pairs']}&{r['prompt_id'] for r in second['pairs']})


if __name__=='__main__':unittest.main()
