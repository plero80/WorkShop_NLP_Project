"""Offline CPU integration tests using randomly initialized tiny Qwen models.

No Hugging Face downloads and no claims about math performance.
"""
from gsm8k_experiment.common import DEFAULT_CONFIG, OUTPUT_ROOT
import copy

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM,
                          Qwen3Config, Qwen3ForCausalLM, Qwen3MoeConfig, Qwen3MoeForCausalLM)

from gsm8k_experiment.common import ROOT, load_config, seed_all
from gsm8k_experiment.models import Policy, RewardScorer, ScoreCache, generation_settings
from gsm8k_experiment.ppo import (load_checkpoint, optimizer_for, prepare_rollout,
                                  save_checkpoint, update)

torch.set_num_threads(1)


@pytest.fixture
def tiny_assets(tmp_path):
    seed_all(42)
    vocabulary = {t: i for i, t in enumerate(["<pad>", "<unk>", "<s>", "</s>",
                  "1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "user", "assistant", "system", "math", "answer", "good", "bad"])}
    raw = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    raw.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=raw, pad_token="<pad>", unk_token="<unk>", bos_token="<s>", eos_token="</s>")
    tok.chat_template = "{% for message in messages %}{{ message['role'] + ' ' + message['content'] + ' </s> ' }}{% endfor %}{% if add_generation_prompt %}{{ 'assistant ' }}{% endif %}"
    q2 = tmp_path / "qwen2"
    q3 = tmp_path / "qwen3"
    qm = tmp_path / "qwen3_moe"
    for folder, cls, cfg in ((q2, Qwen2ForCausalLM, Qwen2Config), (q3, Qwen3ForCausalLM, Qwen3Config), (qm, Qwen3MoeForCausalLM, Qwen3MoeConfig)):
        kw = dict(vocab_size=len(vocabulary), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=2048,
                  bos_token_id=2, eos_token_id=3, pad_token_id=0, attention_dropout=0.0)
        if cfg in (Qwen3Config, Qwen3MoeConfig):
            kw["head_dim"] = 8
        if cfg is Qwen3MoeConfig:
            kw.update(num_experts=4, num_experts_per_tok=2, moe_intermediate_size=16)
        cls(cfg(**kw)).save_pretrained(folder)
        tok.save_pretrained(folder)
    config = copy.deepcopy(load_config(DEFAULT_CONFIG))
    config["models"] = {"policy": str(q2), "proxy": str(q2), "judge": str(q3), "judge30b": str(qm)}
    config["runtime"].update(device="cpu", dtype="float32", attention="eager")
    config["generation"].update(max_new_tokens=8, batch_size=2)
    config["scoring"].update(mode="expected_digit", batch_size=2)
    config["ppo"].update(lora_r=2, lora_alpha=4, learning_rate=.001, value_learning_rate=.001,
                         minibatch_size=2, epochs=2, target_update_kl=1)
    return config, {"policy": "main", "proxy": "main", "judge": "main", "judge30b": "main"}


def items():
    return [{"id": "a", "question": "1 2", "reference": "#### 3", "response": "3", "prompt_ids": [2, 4, 5], "response_ids": [6, 3]},
            {"id": "b", "question": "4 5", "reference": "#### 9", "response": "2", "prompt_ids": [2, 7], "response_ids": [5, 7, 3]}]


@pytest.mark.parametrize("gradient_checkpointing", [True, False])
def test_actual_qwen_ppo_updates_adapter_and_preserves_reference(tiny_assets, tmp_path, gradient_checkpointing):
    config, resolved = tiny_assets
    config["runtime"]["gradient_checkpointing"] = gradient_checkpointing
    policy = Policy(config, resolved)
    assert policy.lm.is_gradient_checkpointing == gradient_checkpointing
    initial = policy.trainable_state()
    frozen = {n: x.detach().clone() for n, x in policy.lm.named_parameters() if not x.requires_grad}
    optimizer = optimizer_for(policy, config)
    rollout = prepare_rollout(policy, items(), [1., -1.], config)
    assert [len(x["old_logprobs"]) for x in rollout] == [2, 3]
    with torch.no_grad():
        for item, r in zip(items(), rollout):
            lp, _ = policy.token_stats(item)
            torch.testing.assert_close(lp.cpu(), r["old_logprobs"])
            torch.testing.assert_close(r["old_logprobs"], r["reference_logprobs"])
    result = update(policy, optimizer, rollout, config, 0)
    assert result["optimizer_steps"] == 2
    after = policy.trainable_state()
    assert any(not torch.equal(initial["adapter"][k], after["adapter"][k]) for k in initial["adapter"])
    assert not torch.equal(initial["value"]["weight"], after["value"]["weight"])
    for n, x in policy.lm.named_parameters():
        if n in frozen:
            torch.testing.assert_close(x, frozen[n], rtol=0, atol=0)
    with torch.no_grad():
        ref, _ = policy.token_stats(items()[0], reference=True)
    torch.testing.assert_close(ref.cpu(), rollout[0]["reference_logprobs"], atol=1e-6, rtol=1e-6)
    save_checkpoint(tmp_path / "checkpoint.pt", policy, optimizer, 1, "experiment", "proxy")
    policy.restore_trainable(initial)
    restored_opt = optimizer_for(policy, config)
    loaded = load_checkpoint(tmp_path / "checkpoint.pt", policy, restored_opt, "experiment", "proxy")
    assert loaded["step"] == 1
    for k, v in policy.trainable_state()["adapter"].items():
        torch.testing.assert_close(v, after["adapter"][k])
    # Exact continuation through an optimizer checkpoint: same second update.
    next_rollout = prepare_rollout(policy, items(), [-.5, .8], config)
    update(policy, restored_opt, next_rollout, config, 1)
    continued = policy.trainable_state()
    load_checkpoint(tmp_path / "checkpoint.pt", policy, optimizer, "experiment", "proxy")
    next_rollout = prepare_rollout(policy, items(), [-.5, .8], config)
    update(policy, optimizer, next_rollout, config, 1)
    for part, values in policy.trainable_state().items():
        for k, v in values.items():
            torch.testing.assert_close(v, continued[part][k], atol=0, rtol=0)


def test_generation_distribution_matches_ppo_logprobs(tiny_assets):
    config, resolved = tiny_assets
    policy = Policy(config, resolved)
    policy.eval()
    pids = [2, 4, 5]
    settings = generation_settings(policy.tokenizer, policy.lm, 6, sample=True, temperature=config["generation"]["temperature"])
    with torch.no_grad():
        generated = policy.lm.generate(input_ids=torch.tensor([pids]), attention_mask=torch.ones(1, 3, dtype=torch.long),
                    generation_config=settings, return_dict_in_generate=True, output_scores=True)
        response = generated.sequences[0, len(pids):].tolist()
        item = {"prompt_ids": pids, "response_ids": response}
        lp, _ = policy.token_stats(item)
        behavioral = torch.stack([s[0].log_softmax(-1)[token] for s, token in zip(generated.scores, response)])
    torch.testing.assert_close(lp, behavioral, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("attention", ["eager", "sdpa"])
def test_batched_padding_logprobs_values_and_gradients_match_single(tiny_assets, attention):
    config, resolved = tiny_assets
    config["runtime"]["attention"] = attention
    policy = Policy(config, resolved)
    policy.train()
    # Nonzero adapters/value weights exercise the reference and both loss paths.
    with torch.no_grad():
        for name, param in policy.lm.named_parameters():
            if "lora_B" in name:
                param.normal_(0, .01)
        policy.value.weight.normal_(0, .01)
    rows = items() + [{"prompt_ids": [2], "response_ids": [4, 5, 6, 7, 3]}]
    for reference in (False, True):
        with torch.no_grad():
            batch = policy.token_stats_batch(rows, reference)
            singles = [policy.token_stats(x, reference) for x in rows]
        for got, want in zip(batch, singles):
            for a, b in zip(got, want):
                torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-5)
    gradients = []
    for batched in (False, True):
        policy.zero_grad(set_to_none=True)
        stats = policy.token_stats_batch(rows) if batched else [policy.token_stats(x) for x in rows]
        total = sum(len(x["response_ids"]) for x in rows)
        loss = sum((lp * torch.linspace(-1, 1, len(lp))).sum() + v.square().sum()
                   for lp, v in stats) / total
        loss.backward()
        gradients.append({n: p.grad.clone() for n, p in policy.named_parameters() if p.requires_grad and p.grad is not None})
    assert gradients[0].keys() == gradients[1].keys()
    for name in gradients[0]:
        torch.testing.assert_close(gradients[0][name], gradients[1][name], atol=3e-6, rtol=3e-4)


def test_batched_generation_distribution_with_unequal_prompt_lengths(tiny_assets):
    config, resolved = tiny_assets
    config["runtime"]["attention"] = "sdpa"
    policy = Policy(config, resolved).eval()
    prompts = [[2, 4, 5, 7], [2, 6]]
    batch = policy.tokenizer.pad({"input_ids": prompts}, padding=True, return_tensors="pt")
    settings = generation_settings(policy.tokenizer, policy.lm, 6, sample=True,
                                   temperature=config["generation"]["temperature"])
    with torch.no_grad():
        out = policy.lm.generate(**batch, generation_config=settings,
                                return_dict_in_generate=True, output_scores=True)
        rows, expected = [], []
        for i, prompt in enumerate(prompts):
            suffix = out.sequences[i, batch["input_ids"].shape[1]:].tolist()
            end = next((j + 1 for j, token in enumerate(suffix) if token == 3), len(suffix))
            response = suffix[:end]
            rows.append({"prompt_ids": prompt, "response_ids": response})
            expected.append(torch.stack([score[i].log_softmax(-1)[token]
                                         for score, token in zip(out.scores, response)]))
        stats = policy.token_stats_batch(rows)
    for (lp, _), want in zip(stats, expected):
        torch.testing.assert_close(lp, want, atol=3e-6, rtol=3e-5)


def test_batched_rollout_and_optimizer_match_single_response_path(tiny_assets, monkeypatch):
    config, resolved = tiny_assets
    policy = Policy(config, resolved)
    state = policy.trainable_state()
    original_batch = policy.token_stats_batch
    results, trained = [], []
    for single in (True, False):
        policy.restore_trainable(state)
        config["runtime"]["ppo_microbatch_size"] = 1 if single else 8
        monkeypatch.setattr(policy, "token_stats_batch",
                            (lambda rows, reference=False: [policy.token_stats(x, reference) for x in rows])
                            if single else original_batch)
        rollout = prepare_rollout(policy, items(), [.7, -.4], config)
        result = update(policy, optimizer_for(policy, config), rollout, config, 0)
        results.append((rollout, result))
        trained.append(policy.trainable_state())
    for a, b in zip(results[0][0], results[1][0]):
        for key in ("old_logprobs", "old_values", "reference_logprobs", "advantage", "return"):
            torch.testing.assert_close(a[key], b[key], atol=3e-6, rtol=3e-5)
    assert results[0][1]["optimizer_steps"] == results[1][1]["optimizer_steps"] == 2
    for part in trained[0]:
        for name in trained[0][part]:
            torch.testing.assert_close(trained[0][part][name], trained[1][part][name], atol=1e-5, rtol=1e-3)


def test_qwen_judges_embeddings_and_cache(tiny_assets, tmp_path):
    config, resolved = tiny_assets
    cache = ScoreCache(tmp_path)
    proxy = RewardScorer("proxy", config, resolved, cache)
    judge = RewardScorer("judge", config, resolved, cache)
    rows = items()
    ps = proxy.score(rows, "test")
    js = judge.score(rows, "test")
    assert all(1 <= x["score"] <= 5 for x in ps + js)
    assert all(x["embedding"] is None for x in js)
    assert np.stack([x["embedding"] for x in ps]).shape == (2, 32)
    np.testing.assert_allclose([np.linalg.norm(x["embedding"]) for x in ps], [1, 1], atol=1e-6)
    cached = proxy.score(rows, "test_cached")
    assert [x["score"] for x in cached] == [x["score"] for x in ps]
    np.testing.assert_allclose(cached[0]["embedding"], ps[0]["embedding"])
    with pytest.raises(ValueError, match="input tokens"):
        old = config["scoring"]["max_input_tokens"]
        config["scoring"]["max_input_tokens"] = 1
        try:
            proxy._infer(rows, "too_long", 2)
        finally:
            config["scoring"]["max_input_tokens"] = old
    cache.close()


def test_generated_grading_retry_never_uses_fake_reward(tiny_assets, tmp_path, monkeypatch):
    config, resolved = tiny_assets
    config["scoring"]["mode"] = "rationale_then_score"
    cache = ScoreCache(tmp_path)
    proxy = RewardScorer("proxy", config, resolved, cache)
    calls = []

    def bad(rows, stage, max_new_tokens, retry=False):
        calls.append(retry)
        return [{"score": None, "judge_output": "unparseable", "input_tokens": 10,
                 "output_tokens": 2, "embedding": np.ones(32) / np.sqrt(32)} for _ in rows]

    monkeypatch.setattr(proxy, "_infer", bad)
    result = proxy.score(items()[:1], "invalid_test")
    assert result[0]["score"] is None
    assert result[0]["grading_status"] == "unscored"
    assert calls == [False, True, True, True, True]
    from gsm8k_experiment.common import read_jsonl
    assert len(read_jsonl(tmp_path / "invalid_judge_outputs.jsonl")) == 5
    cache.close()


def test_truncated_grade_recovers_and_valid_cache_is_reused(tiny_assets, tmp_path, monkeypatch):
    from gsm8k_experiment.answers import parse_rating
    config, resolved = tiny_assets
    config["scoring"]["mode"] = "rationale_then_score"
    cache = ScoreCache(tmp_path)
    proxy = RewardScorer("proxy", config, resolved, cache)
    calls = []
    def infer(rows, stage, max_new_tokens, retry=False):
        calls.append(max_new_tokens)
        text = "Judgement: Arithmetic is wrong.\nCorrectness_score: "
        if max_new_tokens >= 640:
            text += "2"
        return [{"score": parse_rating(text), "judge_output": text,
                 "input_tokens": 100, "output_tokens": max_new_tokens if max_new_tokens < 640 else 330,
                 "grading_length_capped": max_new_tokens < 640,
                 "embedding": np.ones(32, np.float32)/np.sqrt(32)} for _ in rows]
    monkeypatch.setattr(proxy, "_infer", infer)
    result = proxy.score(items()[:1], "monitor/judge/325")
    assert calls == [160, 320, 640]
    assert result[0]["score"] == 2
    assert result[0]["grading_recovery"]["max_new_tokens"] == 640
    again = proxy.score(items()[:1], "cached")
    assert calls == [160, 320, 640]
    assert again[0]["score"] == 2
    cache.close()


@pytest.mark.parametrize("success_budget", [160, 320])
def test_original_successful_attempts_never_use_extended_retries(tiny_assets, tmp_path, monkeypatch, success_budget):
    config, resolved = tiny_assets
    config["scoring"]["mode"] = "rationale_then_score"
    cache = ScoreCache(tmp_path)
    judge = RewardScorer("judge", config, resolved, cache)
    calls = []
    def infer(rows, stage, max_new_tokens, retry=False):
        calls.append(max_new_tokens)
        return [{"score": 4 if max_new_tokens >= success_budget else None,
                 "judge_output": "Correctness_score: 4" if max_new_tokens >= success_budget else "Judgement:",
                 "input_tokens": 10, "output_tokens": 10, "embedding": None} for _ in rows]
    monkeypatch.setattr(judge, "_infer", infer)
    results = judge.score(items()[:1], "training/judge")
    assert calls == ([160] if success_budget == 160 else [160, 320])
    assert results[0]["score"] == 4 and "grading_recovery" not in results[0]
    cache.close()




def test_pilot_then_full_runner_with_cpu_qwen_fixtures(tiny_assets, tmp_path, monkeypatch):
    from gsm8k_experiment import run
    from gsm8k_experiment.common import atomic_json, digest, read_json
    from gsm8k_experiment.data import partition_rows
    config, resolved = tiny_assets
    resolved["dataset"] = "offline_fixture"
    config["dataset"].update(calibration=3, memory=4, selection=3, monitor=2, refresh=2, ppo=4, final=2,
                             responses_per_prompt=2)
    config["scoring"]["minimum_std"] = 1e-9
    config["teacher30b"]["evaluate_final"] = True
    config["knn"].update(k_grid=[1, 2], temperature_grid=[.1], refresh_every=1, refresh_prompts=2, refresh_responses=1)
    config["ppo"].update(prompts_per_update=1, responses_per_prompt=2, checkpoint_every=1, monitor_every=1)
    config["arms"] = ["proxy", "judge", "knn_static", "knn_static_30b", "knn_refresh", "oracle"]
    configpath = tmp_path / "config.json"
    atomic_json(configpath, config)
    train = [{"question": "math " + " ".join([str(i % 10)] * (i + 1)), "answer": "#### 1"} for i in range(20)]
    test = [{"question": "test math 2", "answer": "#### 2"}, {"question": "test math 3", "answer": "#### 3"}]
    split = partition_rows(train, test, config["dataset"], 42)
    split["fingerprint"] = digest(split)
    monkeypatch.setattr(run, "check_runtime", lambda c: {"gpu": "OFFLINE TEST: random tiny CPU Qwen"})
    monkeypatch.setattr(run, "resolve_assets", lambda c, o: resolved)
    monkeypatch.setattr(run, "prepare_data", lambda c, o, r: split)
    # Fixed scalar labels isolate runner mechanics from random-model grading skill.
    original_score = RewardScorer.score

    def varied_scores(self, rows, stage):
        out = original_score(self, rows, stage)
        for row, value in zip(rows, out):
            n = int(digest([row["id"], row["response"]])[:8], 16)
            value["score"] = float(1 + (n % 5 if self.role == "proxy" else (n // (25 if self.role == "judge30b" else 5)) % 5))
        return out

    monkeypatch.setattr(RewardScorer, "score", varied_scores)
    output = tmp_path / "experiment"
    common = ["--config", str(configpath), "--output", str(output)]
    run.main(common + ["--stage", "pilot", "--updates", "1"])
    assert not (output / "final_protocol.json").exists()
    assert not (output / "evaluations" / "final").exists()
    assert read_json(output / "status.json")["stage"] == "complete"
    run.main(common + ["--stage", "full", "--updates", "2"])
    for arm in config["arms"]:
        assert read_json(output / "arms" / arm / "completed.json")["update"] == 2
        assert (output / "evaluations" / "final" / arm / "step_000002" / "responses.jsonl").exists()
    assert read_json(output / "arms" / "knn_refresh" / "completed.json")["last_refresh"] == 2
    assert (output / "learning_curves.png").exists()
    report = read_json(output / "summary.json")
    assert len(report["metrics"]) == 14
    assert len(report["teacher30b_metrics"]) == 7
    assert "final/knn_static_30b_minus_knn_static/numeric" in report["paired_comparisons"]
    from gsm8k_experiment.memory import GapMemory
    m4 = GapMemory.load(output / "prepared" / "memory_initial.npz")
    m30 = GapMemory.load(output / "prepared_30b" / "memory_initial.npz")
    np.testing.assert_array_equal(m4.embeddings, m30.embeddings)
    np.testing.assert_array_equal(m4.group_ids, m30.group_ids)
    assert (m4.k, m4.temperature) == (m30.k, m30.temperature)
    completed = output / "arms" / "knn_static_30b" / "checkpoint.pt"
    before_bytes = completed.read_bytes()
    run.main(common + ["--stage", "full", "--updates", "2"])
    assert completed.read_bytes() == before_bytes
    from gsm8k_experiment.common import read_jsonl
    events = read_jsonl(output / "judge_calls.jsonl")
    assert not any(e["role"] != "proxy" and e["stage"] == "training/knn_static_30b" for e in events)
    with pytest.raises(ValueError, match="final test set"):
        run.main(common + ["--stage", "pilot", "--updates", "3"])
