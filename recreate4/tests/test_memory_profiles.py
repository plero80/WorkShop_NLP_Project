"""CPU-only profile/recipe compatibility and execution-identity checks."""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("memory_profiles_tests", ROOT / "python_helper/memory_profiles.py")
profiles = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profiles)
sys.path.insert(0, str(ROOT / "runtime"))
from experiment_cli.cli import resolve


def settings(vram_gb):
    return resolve(str(ROOT / "configs" / f"gsm8k-{vram_gb}gb.yaml"))["settings"]


@pytest.mark.parametrize("vram_gb,batches,minimum,offload", [
    (12, (1, 1, 1, 1), 21 * 1024 ** 3 // 2, True),
    (24, (4, 1, 1, 1), 22 * 1024 ** 3, False),
    (48, (8, 2, 4, 4), 44 * 1024 ** 3, False),
])
def test_profile_values_and_recipe_batches(vram_gb, batches, minimum, offload):
    expected = dict(zip(("generation_batch_size", "scoring_batch_size",
                         "ppo_microbatch_size", "rollout_stats_batch_size"), batches))
    expected.update(vram_gb=vram_gb, min_gpu_bytes=minimum, grader_offload=offload)
    assert profiles.profile_for(vram_gb) == expected
    configured = settings(vram_gb)
    original = copy.deepcopy(configured)
    assert profiles.validate_settings(configured, vram_gb) == expected
    assert configured == original
    for path, field in profiles._BATCH_PATHS.items():
        section, name = path.split(".")
        assert configured[section][name] == expected[field]


@pytest.mark.parametrize("bad", [None, True, False, 0, 16, 80, 12.0, "12"])
def test_unknown_or_ambiguous_profiles_are_rejected(bad):
    with pytest.raises(ValueError, match="--vram-gb"):
        profiles.profile_for(bad)


def test_returned_profile_cannot_mutate_defaults():
    profile = profiles.profile_for(12)
    profile["generation_batch_size"] = 128
    assert profiles.profile_for(12)["generation_batch_size"] == 1


@pytest.mark.parametrize("vram_gb", [12, 48])
def test_recipes_change_only_execution_batches(vram_gb):
    baseline, configured = settings(24), settings(vram_gb)
    for path in profiles._BATCH_PATHS:
        section, name = path.split(".")
        configured[section][name] = baseline[section][name]
    assert configured == baseline
    assert baseline["ppo"]["prompts_per_update"] * baseline["ppo"]["responses_per_prompt"] == 16
    assert baseline["ppo"]["minibatch_size"] == 8
    assert baseline["ppo"]["epochs"] == 2


@pytest.mark.parametrize("vram_gb", [12, 48])
@pytest.mark.parametrize("path,value", [
    ("generation.batch_size", 64), ("scoring.batch_size", 64),
    ("runtime.ppo_microbatch_size", 8), ("runtime.rollout_stats_batch_size", 16),
    ("runtime.dtype", "float32"), ("runtime.attention", "eager"), ("runtime.device", "cuda:1"),
    ("models.policy", "other/large-model"), ("models.proxy", "other/large-model"),
    ("models.judge", "Qwen/Qwen3-30B-A3B-Instruct-2507"), ("revisions.judge", "main"),
    ("generation.max_prompt_tokens", 2048), ("generation.max_new_tokens", 1536),
    ("generation.temperature", 0.7), ("scoring.max_input_tokens", 8192),
    ("scoring.max_new_tokens", 320), ("scoring.retry_max_new_tokens", 640),
    ("scoring.mode", "expected_digit"), ("ppo.prompts_per_update", 16),
    ("ppo.responses_per_prompt", 4), ("ppo.minibatch_size", 16), ("ppo.epochs", 4),
    ("ppo.lora_r", 64), ("ppo.lora_alpha", 64), ("dataset.responses_per_prompt", 4),
    ("teacher30b.evaluate_final", True),
    ("arms", ["proxy", "judge", "knn_static", "knn_static_30b"]),
])
def test_nondefault_profiles_reject_unsupported_memory_or_protocol_changes(vram_gb, path, value):
    configured = settings(vram_gb)
    parent, _, leaf = path.rpartition(".")
    (configured[parent] if parent else configured)[leaf] = value
    with pytest.raises(ValueError, match=path.replace(".", r"\.")):
        profiles.validate_settings(configured, vram_gb)


@pytest.mark.parametrize("vram_gb", [12, 48])
def test_profiles_allow_seed_duration_and_cohort_changes(vram_gb):
    configured = settings(vram_gb)
    configured["seed"] = 73
    configured["ppo"]["pilot_updates"] = 2
    configured["ppo"]["full_updates"] = 4
    configured["dataset"]["monitor"] = 4
    profiles.validate_settings(configured, vram_gb)


def test_24gb_preserves_legacy_custom_recipe_behavior():
    assert profiles.validate_settings({"custom": "validated by original CLI"}, 24) == profiles.profile_for(24)


def test_missing_setting_and_boolean_batch_fail_closed():
    configured = settings(12)
    del configured["runtime"]["ppo_microbatch_size"]
    with pytest.raises(ValueError, match="runtime.ppo_microbatch_size.*missing"):
        profiles.validate_settings(configured, 12)
    configured = settings(12)
    configured["generation"]["batch_size"] = True
    with pytest.raises(ValueError, match="generation.batch_size"):
        profiles.validate_settings(configured, 12)
