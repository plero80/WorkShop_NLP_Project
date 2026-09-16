"""Whole generated answer scoring, frozen geometry, and bounded corrections."""
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from chat_format import format_prompt_answer
from common import ROOT, read_json, file_hash
from gap_correction import GapCorrector
from knn_core import top_neighbors, unit_vectors


def dtype_argument():
    import transformers
    return 'dtype' if int(transformers.__version__.split('.')[0]) >= 5 else 'torch_dtype'


class RewardScorer:
    def __init__(self,snapshot,batch_size=16,max_length=4096,device='cuda',model=None,tokenizer=None):
        from transformers import AutoModelForSequenceClassification,AutoTokenizer
        self.device,self.batch_size,self.max_length=device,batch_size,max_length
        self.model=model if model is not None else AutoModelForSequenceClassification.from_pretrained(
            str(snapshot),local_files_only=True,attn_implementation='sdpa',
            **{dtype_argument():torch.bfloat16 if device.startswith('cuda') else torch.float32})
        self.tokenizer=tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(str(snapshot),local_files_only=True)
        if self.tokenizer.pad_token_id is None:self.tokenizer.pad_token=self.tokenizer.eos_token
        self.tokenizer.padding_side='right'
        self.model.config.pad_token_id=self.tokenizer.pad_token_id
        self.model.config.use_cache=False
        self.model.to(device).eval().requires_grad_(False)
        if not hasattr(self.model,'score') or self.model.score.out_features != 1:
            raise TypeError('Expected the pinned scalar Qwen reward model.')

    def encode_full(self,prompts,answers):
        if len(prompts)!=len(answers) or not prompts:raise ValueError('Nonempty aligned prompts/answers required.')
        texts=[format_prompt_answer(self.tokenizer,p,a) for p,a in zip(prompts,answers)]
        # Never truncate first and check the shortened length afterwards.
        tokens=self.tokenizer(texts,padding=True,truncation=False,return_tensors='pt')
        lengths=tokens['attention_mask'].sum(1)
        limit=min(self.max_length,getattr(self.model.config,'max_position_embeddings',self.max_length))
        if int(lengths.max())>limit:
            raise ValueError(f'Full reward input needs {int(lengths.max())} tokens, guard is {limit}. '
                             'No prompt or answer was truncated; inspect this cohort before changing the scoring protocol.')
        return tokens,lengths

    @torch.inference_mode()
    def score(self,prompts,answers,features=False):
        if len(prompts)!=len(answers) or not prompts:raise ValueError('Nonempty aligned prompts/answers required.')
        values,vectors,lengths,parities=[],[],[],[]
        for start in range(0,len(prompts),self.batch_size):
            tokens,n=self.encode_full(prompts[start:start+self.batch_size],answers[start:start+self.batch_size])
            tokens=tokens.to(self.device);capture={}
            def hook(module,args,result):
                mask=tokens['attention_mask'].bool();pos=torch.arange(mask.shape[1],device=mask.device)[None,:]
                last=torch.where(mask,pos,-1).max(1).values;rows=torch.arange(len(last),device=last.device)
                if (last<0).any():raise ValueError('Empty reward input.')
                capture['features']=args[0][rows,last].float().cpu().numpy()
                capture['head']=result[rows,last].reshape(-1).float().cpu().numpy()
            handle=self.model.score.register_forward_hook(hook)
            try:raw=self.model(**tokens).logits.reshape(-1).float().cpu().numpy()
            finally:handle.remove()
            error=float(np.max(abs(raw-capture['head'])))
            if error>1e-4:raise RuntimeError('Pooled reward differs from the captured head token.')
            values.extend(raw.tolist());lengths.extend(n.tolist());parities.append(error)
            if features:vectors.append(unit_vectors(capture['features']))
        result={'raw':np.asarray(values,float),'tokens':np.asarray(lengths,int),
                'pooling_parity_max_error':max(parities),'truncated':np.zeros(len(values),bool)}
        if features:result['features']=np.concatenate(vectors)
        if not np.isfinite(result['raw']).all():raise FloatingPointError('Nonfinite reward score.')
        return result


def bounded_gap(gap,distance,parameters,cutoff):
    gap,distance=np.asarray(gap,float),np.asarray(distance,float)
    if gap.shape!=distance.shape or not np.isfinite(gap).all() or not np.isfinite(distance).all():
        raise ValueError('Nonfinite or misaligned gap/distance.')
    a=float(parameters['alpha']);cap=float(parameters['bonus_cap'])
    if a<0 or cap<0:raise ValueError('Negative correction coefficient/cap.')
    applied=a*np.maximum(gap,0)-np.minimum(a*np.maximum(-gap,0),cap)
    return np.where(distance<cutoff,applied,0.) if parameters['distance_gate'] else applied


def predict_memory(vectors,memory,k=31,temperature=.05,threads=8):
    with threadpool_limits(limits=threads):
        sim,ix=top_neighbors(unit_vectors(vectors),memory['vectors'],k)
        w=np.exp((sim-sim[:,:1])/temperature)
        gap=(w*memory['gaps'][ix]).sum(1)/w.sum(1)
    return gap,(1-sim).mean(1)


class FrozenGapReward:
    def __init__(self,scorer,cpu_threads=8):
        self.scorer,self.threads=scorer,cpu_threads
        self.corrector=GapCorrector(ROOT/'inputs/memory',cpu_threads=cpu_threads)
        self.calibration=self.corrector.calibration
        self.updated=None;self.parameters=None;self.lock_hash=None

    def load_update(self,folder):
        folder=Path(folder);lock=read_json(folder/'locked_reward.json')
        if file_hash(folder/'refreshed_memory.npz')!=lock['memory_sha256']:
            raise ValueError('Refreshed memory differs from the locked model.')
        with np.load(folder/'refreshed_memory.npz',allow_pickle=False) as f:
            self.updated={k:f[k].copy() for k in ('vectors','gaps')}
        v,g=self.updated['vectors'],self.updated['gaps']
        if v.ndim!=2 or v.shape[1]!=self.corrector.model['vectors'].shape[1] or len(v)<31 or g.shape!=(len(v),):
            raise ValueError('Refreshed memory has invalid dimensions.')
        if not np.isfinite(v).all() or not np.isfinite(g).all() or not np.allclose(np.linalg.norm(v,axis=1),1.,atol=1e-4):
            raise ValueError('Refreshed memory must contain finite unit vectors and signed gaps.')
        if lock.get('k')!=31 or lock.get('temperature')!=.05 or ('calibration' in lock and lock['calibration']!=self.calibration):
            raise ValueError('Refresh changed the fixed correction/calibration protocol.')
        self.parameters=lock['selected'];self.lock_hash=file_hash(folder/'locked_reward.json')

    def score(self,prompts,answers,branch,features=False):
        values=self.scorer.score(prompts,answers,features=True)
        old=self.corrector.correct_from_embeddings(values['raw'],values['features'],mode='signed')
        gap=old['predicted_gap'];distance=old['mean_neighbor_distance']
        if branch in ('refreshed_stable','iterative_knn','iterative_capped'):
            if self.updated is None:raise RuntimeError('Refreshed reward must be locked before use.')
            gap,distance=predict_memory(values['features'],self.updated,threads=self.threads)
            applied=gap if branch=='iterative_knn' else bounded_gap(gap,distance,self.parameters,self.corrector.cutoff)
        elif branch=='knn_signed':applied=gap
        elif branch=='knn_positive':applied=np.maximum(0,gap)
        elif branch in ('raw','initial'):applied=np.zeros_like(gap)
        else:raise ValueError('Unknown reward branch '+str(branch))
        z=(values['raw']-self.calibration['proxy_mean'])/self.calibration['proxy_std']
        result={'reward':z-applied,'proxy_z':z,'proxy_raw':values['raw'],'gap_hat':gap,'applied_gap':applied,
                'original_gap_hat':old['predicted_gap'],'mean_neighbor_distance':distance,
                'within_distance_gate':distance<self.corrector.cutoff,'reward_tokens':values['tokens'],
                'reward_truncated':values['truncated']}
        if features:result['features']=values['features']
        return result

    def validate_memory_encoder(self):
        import pandas as pd
        bank=pd.read_csv(ROOT/'inputs/candidate_bank.csv',keep_default_na=False)
        ids=self.corrector.model['bank_ids'];eligible=[];positions=[]
        # Old memory included a 1024-token scoring limit. Use reference examples
        # that fit whole to check parity of the unchanged encoder/pooling.
        for i in np.linspace(0,len(ids)-1,min(128,len(ids)),dtype=int):
            row=bank.iloc[int(ids[i])]
            try:_,n=self.scorer.encode_full([row.prompt],[row.answer])
            except ValueError as e:
                if str(e).startswith('Full reward input needs'):continue
                raise
            if int(n[0])<=1024:eligible.append(int(ids[i]));positions.append(i)
            if len(eligible)==32:break
        if len(eligible)<8:raise ValueError('Too few complete legacy references for encoder parity.')
        rows=bank.iloc[eligible];check=self.scorer.score(rows.prompt.tolist(),rows.answer.tolist(),features=True)
        diff=abs(check['raw']-rows.proxy_raw.to_numpy())
        cosine=(check['features']*self.corrector.model['vectors'][positions]).sum(1)
        if diff.mean()>.03 or diff.max()>.3 or cosine.min()<.999:
            raise RuntimeError('Saved-memory encoder reference mismatch.')
        return {'checked_rows':len(rows),'bank_ids':eligible,'mean_absolute_error':float(diff.mean()),
                'max_absolute_error':float(diff.max()),'minimum_cosine':float(cosine.min())}
