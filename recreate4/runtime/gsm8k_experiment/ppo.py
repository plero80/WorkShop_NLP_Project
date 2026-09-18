"""GSM8K rollout/checkpoint bridge. All PPO updates use core PPOTrainer."""
from pathlib import Path
import hashlib
import os
import numpy as np
import torch
from ppo_engine import PPOTrainer, advantages_and_returns
from .common import atomic_json, read_json
from .shared import ENGINE_ID, flat_config, pack_items


def optimizer_for(policy, config):
    return PPOTrainer(policy, None, flat_config(config)).optimizer


@torch.no_grad()
def prepare_rollout(policy, items, terminal_rewards, config):
    """Expose shared-engine diagnostics; the trainer owns the update snapshot."""
    policy.eval()
    if not items or len(items) != len(terminal_rewards):
        raise ValueError("Reward/response count mismatch or empty rollout.")
    if not np.isfinite(np.asarray(terminal_rewards)).all():
        raise ValueError("Nonfinite terminal reward.")
    part = pack_items(items, policy.tokenizer.pad_token_id)
    size = config["runtime"].get("rollout_stats_batch_size", 1)
    old, values, refs = [], [], []
    for start in range(0, len(items), size):
        ids = part["ids"][start:start+size].to(policy.device)
        attn = part["attention"][start:start+size].to(policy.device)
        lp, v = policy.statistics(ids, attn, part["prompt_width"])
        ref, _ = policy.statistics(ids, attn, part["prompt_width"], reference=True, with_values=False)
        old.append(lp.cpu()); values.append(v.cpu()); refs.append(ref.cpu())
    old, values, refs = map(torch.cat, (old, values, refs))
    mask = part["response_mask"]
    rewards = -config["ppo"]["kl_coefficient"] * (old - refs) * mask
    rewards[torch.arange(len(items)), mask.sum(1)-1] += torch.tensor(terminal_rewards, dtype=torch.float32)
    adv, returns = advantages_and_returns(values, rewards, mask, config["ppo"]["gamma"], config["ppo"]["gae_lambda"])
    return [{"item": item, "terminal_reward": float(terminal_rewards[i]),
             "old_logprobs": old[i, :len(item["response_ids"])],
             "old_values": values[i, :len(item["response_ids"])],
             "reference_logprobs": refs[i, :len(item["response_ids"])],
             "advantage": adv[i, :len(item["response_ids"])],
             "return": returns[i, :len(item["response_ids"])]} for i, item in enumerate(items)]


class _RecordedActor:
    """Supply already graded samples without generating a different response."""
    def __init__(self, actor, items):
        self.actor, self.items = actor, items

    def __getattr__(self, name):
        return getattr(self.actor, name)

    def generate(self, prompts, seed, batch_size=None):
        if prompts != [x["question"] for x in self.items]:
            raise ValueError("Sampled prompts changed before the PPO update.")
        return [pack_items(self.items, self.tokenizer.pad_token_id)]


class _RecordedRewards:
    def __init__(self, rollout):
        self.rollout = rollout

    def score(self, prompts, answers, branch):
        if answers != [x["item"]["response"] for x in self.rollout]:
            raise ValueError("Graded responses changed before the PPO update.")
        zeros = np.zeros(len(answers))
        # Only scalar reward enters PPO; placeholders satisfy shared logging fields.
        return {"reward": np.array([x["terminal_reward"] for x in self.rollout]),
                **{k: zeros for k in ("proxy_z", "gap_hat", "applied_gap", "reward_tokens", "within_distance_gate")}}


def update(policy, optimizer, rollout, config, update_index, timing_observer=None):
    policy.eval()
    actor = _RecordedActor(policy, [x["item"] for x in rollout])
    trainer = PPOTrainer(actor, _RecordedRewards(rollout), flat_config(config),
                         timing_observer=timing_observer)
    trainer.optimizer = optimizer
    report = trainer.update([{"prompt": x["item"]["question"]} for x in rollout],
                            "gsm8k", config["seed"], update_index + 1)
    optimizer.zero_grad(set_to_none=True)
    return {**{k: report[k] for k in ("policy_loss", "value_loss", "clip_fraction", "optimizer_steps")},
            "old_policy_kl": report["approx_kl"],
            "early_stopped_on_update_kl": report["early_stop_update_kl"],
            "sampled_reference_kl_per_response": report["mean_sequence_kl_to_reference"],
            "rollout_response_tokens": sum(len(x["item"]["response_ids"]) for x in rollout),
            "engine": ENGINE_ID}


def _cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: _cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu_tree(v) for v in value)
    return value


def _hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_checkpoint(path, policy, optimizer, step, fingerprint, arm, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"engine": ENGINE_ID, "step": step, "fingerprint": fingerprint, "arm": arm,
            "trainable": policy.trainable_state(), "optimizer": _cpu_tree(optimizer.state_dict()),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "extra": extra or {}}
    temp = path.with_suffix(".tmp")
    with temp.open("wb") as stream:
        torch.save(data, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    atomic_json(path.with_suffix(".sha256.json"), {"sha256": _hash(path), "engine": ENGINE_ID})


def load_checkpoint(path, policy, optimizer, fingerprint, arm, accepted_parents=()):
    path = Path(path)
    if accepted_parents:
        raise ValueError("Standalone checkpoint ancestry is incompatible with the shared PPO engine.")
    info = read_json(path.with_suffix(".sha256.json"))
    if info.get("engine") != ENGINE_ID or info.get("sha256") != _hash(path):
        raise ValueError("Checkpoint checksum or engine mismatch.")
    data = torch.load(path, map_location="cpu", weights_only=True)
    if data.get("engine") != ENGINE_ID or data["fingerprint"] != fingerprint or data["arm"] != arm:
        raise ValueError("Checkpoint belongs to another experiment, engine or arm.")
    policy.restore_trainable(data["trainable"])
    optimizer.load_state_dict(data["optimizer"])
    for state in optimizer.state.values():
        for key, val in state.items():
            if isinstance(val, torch.Tensor) and key != "step":
                state[key] = val.to(policy.device)
    torch.set_rng_state(data["torch_rng"].cpu())
    if data["cuda_rng"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([x.cpu() for x in data["cuda_rng"]])
    return data
