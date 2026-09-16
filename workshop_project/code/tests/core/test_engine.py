"""Small, offline correctness tests. No downloaded models and no GPU required."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM, BatchEncoding
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from common import ROOT, read_json, write_json, set_seed, StopRequested
from ppo_engine import PPOActor, PPOTrainer, response_mask, advantages_and_returns, combine_generation
from gap_correction import apply_gap_correction, GapCorrector
from evaluation import evaluate
from common import canonical_hash
prompt_key = canonical_hash


class ToyTokenizer:
    pad_token_id=0; eos_token_id=2; eos_token='EOS'; bos_token=None; padding_side='left'
    def apply_chat_template(self,messages,**kwargs):
        return ' '.join(m['content'] for m in messages)
    def __call__(self,texts,**kwargs):
        seqs=[[1]+[3+ord(ch)%57 for ch in text] for text in texts]
        if kwargs.get('truncation'):seqs=[s[-kwargs.get('max_length',32):] for s in seqs]
        width=max(map(len,seqs))
        if self.padding_side=='right':
            return BatchEncoding({'input_ids':torch.tensor([s+[0]*(width-len(s)) for s in seqs]),
                                  'attention_mask':torch.tensor([[1]*len(s)+[0]*(width-len(s)) for s in seqs])})
        return BatchEncoding({'input_ids':torch.tensor([[0]*(width-len(s))+s for s in seqs]),
                              'attention_mask':torch.tensor([[0]*(width-len(s))+[1]*len(s) for s in seqs])})
    def decode(self,ids,**kwargs):
        return ' '.join(str(int(v)) for v in ids if int(v) not in (0,1,2))


class ToyReward:
    calibration={'proxy_mean':0.,'proxy_std':1.,'judge_mean':0.,'judge_std':1.,'theta':.5}
    calls=0
    def score(self,prompts,answers,branch):
        self.calls+=1
        raw=np.array([sum(map(int,a.split()))/100 if a else 0. for a in answers])
        gap=np.full(len(raw),.2);applied=np.zeros(len(raw)) if branch=='raw' else gap
        return {'reward':raw-applied,'proxy_z':raw,'proxy_raw':raw,'gap_hat':gap,'applied_gap':applied,
                'within_distance_gate':np.ones(len(raw),bool),'mean_neighbor_distance':np.zeros(len(raw)),
                'reward_tokens':np.full(len(raw),20)}


class ToyJudge:
    calls=0
    def score(self,prompts,answers):
        self.calls+=1
        return {'raw':np.zeros(len(answers)), 'tokens':np.full(len(answers),20)}


def toy_actor():
    c=read_json(ROOT/'config.json')
    c.update(max_prompt_tokens=32,max_new_tokens=6,generation_batch_size=2,mini_batch_size=4,
             micro_batch_size=2,ppo_epochs=2,learning_rate=.0003,value_learning_rate=.001,lora_rank=2,lora_alpha=4)
    set_seed(42)
    config=Qwen2Config(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
        num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=1024,
        bos_token_id=1,eos_token_id=2,pad_token_id=0,attention_dropout=0.)
    base=Qwen2ForCausalLM(config)
    policy=get_peft_model(base,LoraConfig(task_type='CAUSAL_LM',r=2,lora_alpha=4,lora_dropout=0.,
                                        target_modules=['q_proj','k_proj','v_proj','o_proj']))
    return PPOActor(policy,ToyTokenizer(),c,device='cpu'),c


def adapter_copy(actor):
    return {k:v.detach().clone() for k,v in get_peft_model_state_dict(actor.policy).items()}


class PPOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_first_eos_is_included_and_padding_masked(self):
        seq=torch.tensor([[9,2,2,2],[9,8,7,6],[2,2,2,2]])
        self.assertEqual(response_mask(seq,[2]).tolist(),[[True,True,False,False],[True]*4,[True,False,False,False]])

    def test_gae_terminals_and_padding_do_not_bootstrap(self):
        values=torch.tensor([[.2,.4,999.],[.1,.3,.5]])
        rewards=torch.tensor([[0.,2.,999.],[0.,0.,1.]])
        mask=torch.tensor([[1,1,0],[1,1,1]],dtype=torch.bool)
        advantage,returns=advantages_and_returns(values,rewards,mask,1.,1.)
        torch.testing.assert_close(returns,torch.tensor([[2.,2.,0.],[1.,1.,1.]]))
        self.assertAlmostEqual(float(advantage[mask].mean()),0.,places=6)
        self.assertEqual(float(advantage[~mask].sum()),0.)

    def test_response_alignment_and_left_padding(self):
        actor,c=toy_actor()
        ids=torch.tensor([[0,1,7,9,4],[1,5,7,9,4]])
        attention=torch.tensor([[0,1,1,1,1],[1,1,1,1,1]])
        logp,_=actor.statistics(ids,attention,3)
        full=actor.policy(input_ids=ids,attention_mask=attention,
            position_ids=(attention.cumsum(1)-1).clamp_min(0),use_cache=False).logits
        expected=full[:,2:4].log_softmax(-1).gather(-1,ids[:,3:,None]).squeeze(-1)
        torch.testing.assert_close(logp,expected)
        unpadded,_=actor.statistics(ids[:1,1:],attention[:1,1:],2)
        torch.testing.assert_close(logp[:1],unpadded,atol=1e-6,rtol=1e-6)
        generated=actor.generate(['a','longer prompt','bb'],19)
        combined=combine_generation(generated,0)
        self.assertEqual(len(combined['answers']),3)
        self.assertTrue(torch.equal(combined['attention'][:,combined['prompt_width']:].bool(),combined['mask']))

    def test_real_ppo_update_frozen_reference_and_exact_resume(self):
        actor,c=toy_actor();reward=ToyReward();trainer=PPOTrainer(actor,reward,c)
        rows=[{'prompt':p} for p in ['one','two','three','four']]
        ids=torch.tensor([[1,4,5,6]]);attention=torch.ones_like(ids)
        initial=adapter_copy(actor)
        with torch.no_grad():
            ref_before,_=actor.statistics(ids,attention,2,reference=True,with_values=False)
        report=trainer.update(rows,'knn_positive',42,1)
        self.assertGreater(report['optimizer_steps'],0)
        self.assertTrue(any(not torch.equal(v,initial[k]) for k,v in adapter_copy(actor).items()))
        with torch.no_grad():
            ref_after,_=actor.statistics(ids,attention,2,reference=True,with_values=False)
        torch.testing.assert_close(ref_before,ref_after,atol=0,rtol=0)
        with tempfile.TemporaryDirectory() as td:
            checkpoint=trainer.checkpoint(td,1,'test_identity',42,'knn_positive',[report])
            trainer.update(rows,'knn_positive',42,2)
            expected=adapter_copy(actor)
            other,c=toy_actor();resumed=PPOTrainer(other,ToyReward(),c)
            step,history=resumed.restore(checkpoint,'test_identity',42,'knn_positive')
            self.assertEqual(step,1);self.assertEqual(len(history),1)
            resumed.update(rows,'knn_positive',42,2)
            for k,v in adapter_copy(other).items():
                torch.testing.assert_close(v,expected[k],atol=0,rtol=0)
            with self.assertRaises(ValueError):
                resumed.restore(checkpoint,'different',42,'knn_positive')

    def test_correction_scales_and_no_positive_reward_increase(self):
        cal={'proxy_mean':2.,'proxy_std':3.}
        signed=apply_gap_correction([5.,5.],[2.,-1.],[0.,1.],cal,.2,'signed')
        positive=apply_gap_correction([5.,5.],[2.,-1.],[0.,1.],cal,.2,'positive')
        np.testing.assert_allclose(signed['reward_z'],[-1.,2.])
        np.testing.assert_allclose(signed['reward_raw'],[-1.,8.])
        np.testing.assert_allclose(positive['reward_raw'],[-1.,5.])
        gated=apply_gap_correction([5.,5.],[2.,-1.],[0.,1.],cal,.2,'signed_gated')
        np.testing.assert_allclose(gated['reward_raw'],[-1.,5.])

    def test_packaged_memory_loads_and_query_is_finite(self):
        corrector=GapCorrector(ROOT/'inputs/memory',cpu_threads=2)
        self.assertEqual(corrector.model['vectors'].shape,(7990,1024))
        result=corrector.correct_from_embeddings(np.zeros(2),corrector.model['vectors'][:2],mode='signed')
        self.assertTrue(np.isfinite(result['reward_z']).all())
        from reward_bridge import predict_memory
        same_gap,same_distance=predict_memory(corrector.model['vectors'][:2],corrector.model,threads=2)
        np.testing.assert_allclose(same_gap,result['predicted_gap'],atol=1e-6,rtol=1e-6)
        np.testing.assert_allclose(same_distance,result['mean_neighbor_distance'],atol=1e-6,rtol=1e-6)
        np.testing.assert_allclose(result['neighbor_weights'].sum(1),np.ones(2),atol=1e-6,rtol=1e-6)

    def test_reward_head_embedding_pooling_matches_scalar_score(self):
        from transformers import Qwen3Config,Qwen3ForSequenceClassification
        from reward_bridge import RewardScorer
        config=Qwen3Config(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
            num_attention_heads=4,num_key_value_heads=2,head_dim=8,num_labels=1,pad_token_id=0)
        scorer=RewardScorer('unused',device='cpu',model=Qwen3ForSequenceClassification(config),tokenizer=ToyTokenizer())
        result=scorer.score(['a','long text'],['','more'],features=True)
        self.assertEqual(result['features'].shape,(2,32))
        self.assertEqual(result['pooling_parity_max_error'],0.)
        np.testing.assert_allclose(np.linalg.norm(result['features'],axis=1),[1.,1.],atol=1e-6)
        single=scorer.score(['a'],[''],features=True)
        np.testing.assert_allclose(single['features'][0],result['features'][0],atol=1e-6)


    def test_local_assets_need_no_auth_or_network(self):
        import assets
        with tempfile.TemporaryDirectory() as td:
            folder=assets.snapshot(Path(td),'org/repo','abc')
            folder.mkdir(parents=True);(folder/'config.json').write_text('{}')
            with (patch('assets.cache_roots',return_value=[Path(td)]),
                  patch('huggingface_hub.get_token',side_effect=AssertionError('auth called')),
                  patch('huggingface_hub.hf_hub_download',side_effect=AssertionError('network called'))):
                self.assertEqual(assets.resolve_files({'allow_downloads':False},'org/repo','abc',['config.json']),folder)
        self.assertEqual(assets.retry_seconds({'Retry-After':'37'}),37)
        self.assertEqual(assets.retry_seconds({'RateLimit':'"resolvers";r=0;t=123'}),123)





