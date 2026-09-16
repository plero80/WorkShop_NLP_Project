"""Configuration and sampled-token adapters for the unchanged shared engines."""
from pathlib import Path
import hashlib
import json
import torch
from .common import DEFAULT_CONFIG

ENGINE_ID = "reward_gap_knn.shared_ppo.gsm8k.v1"


def shared_sources():
    import ppo_engine
    root = Path(ppo_engine.__file__).resolve().parent
    expected = json.loads(DEFAULT_CONFIG.with_name("shared_sources.json").read_text())
    actual = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in expected}
    if actual != expected:
        raise ValueError("Shared core source hashes changed; review and version the GSM8K integration before running.")
    return actual


def flat_config(config):
    p, g = config["ppo"], config["generation"]
    if g["temperature"] != 1.0:
        raise ValueError("Shared PPO requires sampling temperature 1.0.")
    keys = ("learning_rate", "value_learning_rate", "clip_range", "value_clip_range",
            "value_coefficient", "kl_coefficient", "gamma", "gae_lambda",
            "max_grad_norm", "target_update_kl", "lora_alpha")
    return {**{k: p[k] for k in keys}, "lora_rank": p["lora_r"],
            "generation_batch_size": g["batch_size"], "max_prompt_tokens": g["max_prompt_tokens"],
            "max_new_tokens": g["max_new_tokens"], "ppo_epochs": p["epochs"],
            "mini_batch_size": p["minibatch_size"],
            "micro_batch_size": config["runtime"].get("ppo_microbatch_size", 1),
            "reward_max_tokens": config["scoring"]["max_input_tokens"]}


def pack_items(items, pad_id):
    """Keep response alignment identical to PPOActor/combine_generation."""
    if not items or any(not x["prompt_ids"] or not x["response_ids"] for x in items):
        raise ValueError("Each rollout needs nonempty prompts and responses.")
    width = max(len(x["prompt_ids"]) for x in items)
    response_width = max(len(x["response_ids"]) for x in items)
    ids = torch.full((len(items), width + response_width), pad_id, dtype=torch.long)
    attention = torch.zeros_like(ids)
    mask = torch.zeros((len(items), response_width), dtype=torch.bool)
    for i, x in enumerate(items):
        p, r = x["prompt_ids"], x["response_ids"]
        ids[i, width-len(p):width+len(r)] = torch.tensor(p + r)
        attention[i, width-len(p):width+len(r)] = 1
        mask[i, :len(r)] = True
    return {"ids": ids, "attention": attention, "response_mask": mask, "prompt_width": width,
            "answers": [x.get("response", "") for x in items],
            "ended_eos": [bool(x.get("ended_with_eos", False)) for x in items]}
