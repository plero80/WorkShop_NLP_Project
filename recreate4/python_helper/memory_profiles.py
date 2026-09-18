"""Execution profiles around the unchanged scientific runtime; standard library only.

The 24 GB profile preserves legacy custom-recipe validation and launch identities.
Nondefault profiles support the pinned three-model protocol below, rather than
claiming that arbitrary models, contexts, or precision fit the requested VRAM.
"""
from __future__ import annotations

_GIB = 1024 ** 3
_PROFILES = {
    12: (21 * _GIB // 2, 1, 1, 1, 1, True),
    24: (22 * _GIB, 4, 1, 1, 1, False),
    48: (44 * _GIB, 8, 2, 4, 4, False),
}
_FIELDS = ("min_gpu_bytes", "generation_batch_size", "scoring_batch_size",
           "ppo_microbatch_size", "rollout_stats_batch_size", "grader_offload")
_BATCH_PATHS = {
    "generation.batch_size": "generation_batch_size",
    "scoring.batch_size": "scoring_batch_size",
    "runtime.ppo_microbatch_size": "ppo_microbatch_size",
    "runtime.rollout_stats_batch_size": "rollout_stats_batch_size",
}
_SUPPORTED = {
    "arms": ["proxy", "judge", "knn_static"],
    "models.policy": "Qwen/Qwen2.5-0.5B-Instruct",
    "models.proxy": "Qwen/Qwen2.5-1.5B-Instruct",
    "models.judge": "Qwen/Qwen3-4B-Instruct-2507",
    "revisions.policy": "7ae557604adf67be50417f59c2c2f167def9a775",
    "revisions.proxy": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
    "revisions.judge": "cdbee75f17c01a7cc42f958dc650907174af0554",
    "runtime.device": "cuda:0",
    "runtime.dtype": "bfloat16",
    "runtime.attention": "sdpa",
    "generation.max_prompt_tokens": 1024,
    "generation.max_new_tokens": 768,
    "generation.temperature": 1.0,
    "scoring.mode": "rationale_then_score",
    "scoring.max_input_tokens": 4096,
    "scoring.max_new_tokens": 160,
    "scoring.retry_max_new_tokens": 320,
    "dataset.responses_per_prompt": 2,
    "ppo.prompts_per_update": 8,
    "ppo.responses_per_prompt": 2,
    "ppo.minibatch_size": 8,
    "ppo.epochs": 2,
    "ppo.lora_r": 16,
    "ppo.lora_alpha": 32,
    "teacher30b.evaluate_final": False,
}


def profile_for(vram_gb):
    """Return a fresh profile dictionary; hardware capacity is not a fit guarantee."""
    if type(vram_gb) is not int or vram_gb not in _PROFILES:
        raise ValueError("--vram-gb must be one of 12, 24, or 48.")
    return {"vram_gb": vram_gb, **dict(zip(_FIELDS, _PROFILES[vram_gb]))}


def _expect(settings, path, expected, vram_gb):
    value = settings
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ValueError(f"The {vram_gb} GB profile requires setting {path}={expected!r}; it is missing.")
        value = value[part]
    # In particular, bool is not a valid execution batch size despite True == 1.
    same_type = type(value) is type(expected)
    if type(expected) is float:
        same_type = type(value) in (int, float)
    if not same_type or value != expected:
        raise ValueError(f"The {vram_gb} GB profile requires {path}={expected!r}; got {value!r}. "
                         "Use the matching profile recipe; 24 GB retains legacy custom-recipe behavior.")


def validate_settings(settings, vram_gb):
    """Validate resolved settings without modifying them, returning their profile.

    Seed, training duration, and cohort sizes remain configurable. Nondefault
    profiles fix the supported model/precision/context and effective PPO batches;
    callers still use the original CLI/runtime for general configuration checks.
    """
    profile = profile_for(vram_gb)
    if vram_gb == 24:
        return profile
    for path, field in _BATCH_PATHS.items():
        _expect(settings, path, profile[field], vram_gb)
    for path, expected in _SUPPORTED.items():
        _expect(settings, path, expected, vram_gb)
    return profile
