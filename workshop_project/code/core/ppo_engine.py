"""Token-level clipped PPO with a learned value head and masked GAE.

Behavior sampling uses temperature 1, top-p 1, top-k 0; likelihood ratios use
the identical distribution. EOS is included once. Length-capped responses are
finite-horizon terminal episodes. The frozen reference is the base policy with
LoRA disabled. Reward encoders never participate in backpropagation.
"""
from contextlib import nullcontext
from pathlib import Path
import inspect
import json
import os
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from common import (ROOT, canonical_hash, file_hash, read_json, write_json,
                    seed_for, set_seed, StopRequested)
from chat_format import format_user_prompt
from reward_bridge import dtype_argument


def response_mask(response, eos_ids):
    eos = torch.zeros_like(response, dtype=torch.bool)
    for token in eos_ids:
        eos |= response == token
    # Number of EOS tokens strictly before this position: the first EOS stays valid.
    return (eos.long().cumsum(1) - eos.long()) == 0


def masked_mean(values, mask):
    return (values * mask).sum() / mask.sum().clamp_min(1)


def advantages_and_returns(values, rewards, mask, gamma, lam):
    advantages = torch.zeros_like(values)
    running = torch.zeros(values.shape[0], device=values.device)
    for t in range(values.shape[1]-1, -1, -1):
        alive = mask[:, t+1].float() if t+1 < values.shape[1] else torch.zeros_like(running)
        next_value = values[:, t+1] if t+1 < values.shape[1] else torch.zeros_like(running)
        delta = rewards[:, t] + gamma * next_value * alive - values[:, t]
        running = (delta + gamma * lam * running * alive) * mask[:, t]
        advantages[:, t] = running
    returns = (advantages + values) * mask
    valid = advantages[mask]
    standardized = (advantages - valid.mean()) / torch.sqrt(valid.var(unbiased=False) + 1e-8)
    return standardized * mask, returns


class PPOActor(nn.Module):
    def __init__(self, policy, tokenizer, config, device='cuda'):
        super().__init__()
        self.policy, self.tokenizer, self.c, self.device_name = policy, tokenizer, config, device
        self.value_head = nn.Linear(policy.config.hidden_size, 1, dtype=torch.float32)
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)
        self.to(device)
        # Disable all dropout during both rollout and update likelihood calculations.
        self.eval()
        for module in self.policy.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.
        tokenizer.padding_side = 'left'
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.policy.config.pad_token_id = tokenizer.pad_token_id
        eos = policy.generation_config.eos_token_id or tokenizer.eos_token_id
        self.eos_ids = list(eos) if isinstance(eos, (list, tuple)) else [int(eos)]
        underlying = policy.get_base_model() if hasattr(policy, 'get_base_model') else policy
        self.keep_logits = 'logits_to_keep' in inspect.signature(underlying.forward).parameters

    @classmethod
    def load(cls, snapshot, c, seed, device='cuda'):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model
        set_seed(seed)
        tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
        base = AutoModelForCausalLM.from_pretrained(str(snapshot), local_files_only=True,
            attn_implementation='sdpa', **{dtype_argument(): torch.bfloat16 if device.startswith('cuda') else torch.float32})
        adapter = LoraConfig(task_type='CAUSAL_LM', r=c['lora_rank'], lora_alpha=c['lora_alpha'],
                             lora_dropout=0., bias='none', target_modules=['q_proj','k_proj','v_proj','o_proj'])
        model = get_peft_model(base, adapter)
        return cls(model, tokenizer, c, device)

    @torch.no_grad()
    def generate(self, prompts, seed, batch_size=None):
        batch_size = batch_size or self.c['generation_batch_size']
        set_seed(seed)
        parts = []
        for start in range(0, len(prompts), batch_size):
            texts = [format_user_prompt(self.tokenizer, p) for p in prompts[start:start+batch_size]]
            encoded = self.tokenizer(texts, padding=True, truncation=False, return_tensors='pt').to(self.device_name)
            if encoded['attention_mask'].sum(1).max() > self.c['max_prompt_tokens']:
                raise ValueError('Prepared prompt exceeds the fixed policy context limit.')
            width = encoded['input_ids'].shape[1]
            ids = self.policy.generate(**encoded, max_new_tokens=self.c['max_new_tokens'],
                do_sample=True, temperature=1., top_p=1., top_k=0, repetition_penalty=1.,
                no_repeat_ngram_size=0, min_new_tokens=0, eos_token_id=self.eos_ids,
                pad_token_id=self.tokenizer.pad_token_id, use_cache=True,
                forced_bos_token_id=None, forced_eos_token_id=None,
                suppress_tokens=None, begin_suppress_tokens=None, renormalize_logits=False)
            response = ids[:, width:]
            mask = response_mask(response, self.eos_ids)
            lengths = mask.sum(1)
            answers = [self.tokenizer.decode(row[:int(n)], skip_special_tokens=True).strip()
                       for row, n in zip(response, lengths)]
            parts.append({'ids': ids.cpu(),
                          'attention': torch.cat((encoded['attention_mask'], mask.long()), 1).cpu(),
                          'response_mask': mask.cpu(), 'prompt_width': width,
                          'answers': answers,
                          'ended_eos': [bool(int(row[int(n)-1]) in self.eos_ids) for row, n in zip(response, lengths)]})
        return parts

    def statistics(self, ids, attention, prompt_width, reference=False, with_values=True):
        response_length = ids.shape[1] - prompt_width
        if response_length < 1:
            raise ValueError('PPO needs at least one response action.')
        position_ids = attention.long().cumsum(1) - 1
        position_ids.masked_fill_(attention == 0, 0)
        context = self.policy.disable_adapter() if reference else nullcontext()
        kwargs = {'logits_to_keep': response_length+1} if self.keep_logits else {}
        with context:
            result = self.policy(input_ids=ids, attention_mask=attention, position_ids=position_ids,
                                 output_hidden_states=with_values, use_cache=False, **kwargs)
        # State immediately before each response token predicts that response token.
        logits = result.logits[:, -response_length-1:-1, :].float()
        targets = ids[:, -response_length:]
        logp = F.log_softmax(logits, dim=-1).gather(-1, targets[..., None]).squeeze(-1)
        values = self.value_head(result.hidden_states[-1][:, -response_length-1:-1, :].float()).squeeze(-1) if with_values else None
        return logp, values


def combine_generation(parts, pad_id):
    # Re-pad variable prompt/response chunk widths without moving response alignment.
    prompt_width = max(p['prompt_width'] for p in parts)
    response_width = max(p['response_mask'].shape[1] for p in parts)
    ids, attention, masks, answers, eos = [], [], [], [], []
    for p in parts:
        left = prompt_width-p['prompt_width']
        right = response_width-p['response_mask'].shape[1]
        ids.append(F.pad(p['ids'], (left, right), value=pad_id))
        attention.append(F.pad(p['attention'], (left, right), value=0))
        masks.append(F.pad(p['response_mask'], (0, right), value=False))
        answers.extend(p['answers'])
        eos.extend(p['ended_eos'])
    return {'ids': torch.cat(ids), 'attention': torch.cat(attention), 'mask': torch.cat(masks),
            'prompt_width': prompt_width, 'answers': answers, 'ended_eos': eos}


class PPOTrainer:
    def __init__(self, actor, reward, config):
        self.actor, self.reward, self.c = actor, reward, config
        self.optimizer = torch.optim.AdamW([
            {'params': [p for p in actor.policy.parameters() if p.requires_grad], 'lr': config['learning_rate']},
            {'params': actor.value_head.parameters(), 'lr': config['value_learning_rate']}],
            betas=(.9, .999), eps=1e-5, weight_decay=0.)

    def update(self, rows, branch, seed, update):
        c, actor = self.c, self.actor
        start = time.perf_counter()
        prompts = [r['prompt'] for r in rows]
        generated = actor.generate(prompts, seed_for(seed, update, 'rollout'))
        batch = combine_generation(generated, actor.tokenizer.pad_token_id)
        ids, attention, mask = [batch[n].to(actor.device_name) for n in ('ids', 'attention', 'mask')]
        width = batch['prompt_width']
        reward = self.reward.score(prompts, batch['answers'], branch)
        scalar = torch.as_tensor(reward['reward'], dtype=torch.float32, device=actor.device_name)
        old_logp, old_values, reference_logp = [], [], []
        with torch.no_grad():
            for s in range(0, len(rows), c['micro_batch_size']):
                sl = slice(s, s+c['micro_batch_size'])
                p, v = actor.statistics(ids[sl], attention[sl], width)
                r, _ = actor.statistics(ids[sl], attention[sl], width, reference=True, with_values=False)
                old_logp.append(p); old_values.append(v); reference_logp.append(r)
            old_logp, old_values, reference_logp = [torch.cat(v) for v in (old_logp, old_values, reference_logp)]
            kl = (old_logp-reference_logp)*mask
            token_rewards = -c['kl_coefficient'] * kl
            last = mask.sum(1)-1
            token_rewards[torch.arange(len(rows), device=ids.device), last] += scalar
            adv, returns = advantages_and_returns(old_values, token_rewards, mask, c['gamma'], c['gae_lambda'])
        if update == 1 and torch.max(torch.abs(kl)).item() > .05:
            raise RuntimeError('Initial LoRA policy should match the frozen reference; initial KL parity failed.')
        metrics = []
        stopped_early = False
        for epoch in range(c['ppo_epochs']):
            permutation = np.random.default_rng(seed_for(seed, update, epoch, 'minibatches')).permutation(len(rows))
            for start_mini in range(0, len(rows), c['mini_batch_size']):
                mini = permutation[start_mini:start_mini+c['mini_batch_size']]
                denominator = mask[mini].sum().clamp_min(1)
                self.optimizer.zero_grad(set_to_none=True)
                accum = {'policy_loss': 0., 'value_loss': 0., 'approx_kl': 0., 'clip_fraction': 0.}
                for s in range(0, len(mini), c['micro_batch_size']):
                    ix = mini[s:s+c['micro_batch_size']]
                    newp, values = actor.statistics(ids[ix], attention[ix], width)
                    log_ratio = newp-old_logp[ix]
                    if not torch.isfinite(log_ratio[mask[ix]]).all() or log_ratio[mask[ix]].abs().max() > 20:
                        raise FloatingPointError('Non-finite or extreme PPO likelihood ratio. Resume from the last complete checkpoint after diagnosis.')
                    ratio = log_ratio.exp()
                    surrogate = torch.minimum(ratio*adv[ix], ratio.clamp(1-c['clip_range'], 1+c['clip_range'])*adv[ix])
                    clipped_v = old_values[ix] + (values-old_values[ix]).clamp(-c['value_clip_range'], c['value_clip_range'])
                    vloss = .5*torch.maximum((values-returns[ix]).square(), (clipped_v-returns[ix]).square())
                    p_loss = -(surrogate*mask[ix]).sum()/denominator
                    v_loss = (vloss*mask[ix]).sum()/denominator
                    loss = p_loss + c['value_coefficient']*v_loss
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Non-finite PPO loss.')
                    loss.backward()
                    accum['policy_loss'] += p_loss.detach().item()
                    accum['value_loss'] += v_loss.detach().item()
                    accum['approx_kl'] += (((ratio-1)-log_ratio)*mask[ix]).sum().detach().item()/denominator.item()
                    accum['clip_fraction'] += (((ratio-1).abs()>c['clip_range'])*mask[ix]).sum().item()/denominator.item()
                if accum['approx_kl'] > c['target_update_kl']:
                    self.optimizer.zero_grad(set_to_none=True)
                    stopped_early = True
                    break
                norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), c['max_grad_norm'])
                if not torch.isfinite(norm):
                    raise FloatingPointError('Non-finite gradient norm.')
                self.optimizer.step()
                metrics.append(accum)
            if stopped_early:
                break
        if not metrics:
            raise RuntimeError('PPO update completed no optimizer step; inspect likelihood/gradient diagnostics.')
        report = {'update': update, 'branch': branch, 'seed': seed,
                  'seconds': time.perf_counter()-start,
                  'mean_proxy_z': float(np.mean(reward['proxy_z'])),
                  'mean_predicted_gap': float(np.mean(reward['gap_hat'])),
                  'mean_applied_gap': float(np.mean(reward['applied_gap'])),
                  'correction_fraction': float(np.mean(np.abs(reward['applied_gap']) > 1e-12)),
                  'mean_base_reward': scalar.mean().item(),
                  'mean_total_return': token_rewards.sum(1).mean().item(),
                  'mean_sequence_kl_to_reference': kl.sum(1).mean().item(),
                  'mean_response_tokens': mask.sum(1).float().mean().item(),
                  'eos_fraction': float(np.mean(batch['ended_eos'])),
                  'reward_context_guard_reached_fraction': float(np.mean(reward['reward_tokens'] >= c['reward_max_tokens'])),
                  'within_old_distance_gate_fraction': float(np.mean(reward['within_distance_gate'])),
                  'optimizer_steps': len(metrics), 'early_stop_update_kl': stopped_early,
                  **{k: float(np.mean([m[k] for m in metrics])) for k in metrics[0]}}
        return report

    def checkpoint(self, folder, update, identity, seed, branch, history):
        from peft import get_peft_model_state_dict
        folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
        path = folder / f'checkpoint_{update:06d}.pt'
        payload = {'update': update, 'identity': identity, 'seed': seed, 'branch': branch,
                   'adapter': {k: v.detach().cpu() for k, v in get_peft_model_state_dict(self.actor.policy).items()},
                   'value_head': {k: v.detach().cpu() for k, v in self.actor.value_head.state_dict().items()},
                   'optimizer': self.optimizer.state_dict(), 'history': history,
                   'torch_rng': torch.get_rng_state(),
                   'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
        temporary = path.with_suffix('.pending')
        torch.save(payload, temporary)
        with open(temporary, 'rb') as f:
            os.fsync(f.fileno())
        temporary.replace(path)
        info = {'name': path.name, 'sha256': file_hash(path), 'update': update,
                'identity': identity, 'seed': seed, 'branch': branch}
        write_json(path.with_suffix('.json'), info)
        write_json(folder / 'latest.json', info)
        return path

    def restore(self, path, identity, seed, branch, optimizer=True):
        from peft import set_peft_model_state_dict
        path = Path(path)
        info = read_json(path.with_suffix('.json'))
        if info['sha256'] != file_hash(path) or info['identity'] != identity or info['seed'] != seed or info['branch'] != branch:
            raise ValueError('Checkpoint identity/hash mismatch.')
        # Only this experiment's locally generated, hash-checked checkpoints are loaded.
        state = torch.load(path, map_location=self.actor.device_name, weights_only=False)
        set_peft_model_state_dict(self.actor.policy, state['adapter'])
        self.actor.value_head.load_state_dict(state['value_head'])
        if optimizer:
            self.optimizer.load_state_dict(state['optimizer'])
            torch.set_rng_state(state['torch_rng'].cpu())
            if state['cuda_rng'] is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda_rng']])
        return state['update'], state['history']
