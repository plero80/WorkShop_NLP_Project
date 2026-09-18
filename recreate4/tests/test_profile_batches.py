"""CPU regression: profile microbatches preserve the original PPO accumulation."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))
from ppo_engine import PPOTrainer


class TinyActor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.policy = torch.nn.Linear(3, 1, bias=False)
        self.value_head = torch.nn.Linear(3, 1, bias=False)
        torch.nn.init.zeros_(self.policy.weight)
        torch.nn.init.zeros_(self.value_head.weight)
        self.device_name = "cpu"
        self.tokenizer = SimpleNamespace(pad_token_id=0)
        self.batch_sizes = []

    def generate(self, prompts, seed):
        count, width, max_response = len(prompts), 2, 5
        lengths = torch.arange(count) % max_response + 1
        mask = torch.arange(max_response)[None, :] < lengths[:, None]
        ids = torch.zeros(count, width + max_response, dtype=torch.long)
        ids[:, :width] = 1
        for row in range(count):
            ids[row, width:width + lengths[row]] = row + 2
        attention = torch.cat([torch.ones(count, width, dtype=torch.bool), mask], dim=1)
        return [{"ids": ids, "attention": attention.long(), "response_mask": mask,
                 "prompt_width": width, "answers": ["answer"] * count,
                 "ended_eos": [True] * count}]

    def statistics(self, ids, attention, width, reference=False, with_values=True):
        self.batch_sizes.append(len(ids))
        actions = ids[:, width:].float() / 20
        position = torch.arange(actions.shape[1], dtype=torch.float32).expand_as(actions) / 5
        features = torch.stack([actions, position + .1, actions.square() + .2], dim=-1)
        logp = self.policy(features).squeeze(-1) - 1
        if reference:
            logp = torch.full_like(logp, -1)
        value = self.value_head(features).squeeze(-1) if with_values else None
        return logp, value


class TinyRewards:
    def score(self, prompts, answers, branch):
        count = len(prompts)
        zeros = np.zeros(count)
        # Unequal rewards and lengths exercise token weighting and masked GAE.
        rewards = np.sin(np.arange(count) * .7) + np.linspace(-.2, .3, count)
        return {"reward": rewards,
                **{key: zeros for key in ("proxy_z", "gap_hat", "applied_gap",
                                          "reward_tokens", "within_distance_gate")}}


class ProfileBatchTests(unittest.TestCase):
    def check_equivalence(self, count):
        common = dict(learning_rate=1e-5, value_learning_rate=1e-4,
                      mini_batch_size=8, ppo_epochs=2, kl_coefficient=.01,
                      gamma=1., gae_lambda=.95, clip_range=.2, value_clip_range=.2,
                      value_coefficient=.5, max_grad_norm=1., target_update_kl=.05,
                      reward_max_tokens=4096)
        actors = [TinyActor(), TinyActor()]
        trainers = [PPOTrainer(actor, TinyRewards(), {**common, "micro_batch_size": size})
                    for actor, size in zip(actors, (1, 4))]
        rows = [{"prompt": str(i)} for i in range(count)]
        for update in (1, 2):
            reports = [trainer.update(rows, "test", 42, update) for trainer in trainers]
            for report in reports:
                self.assertEqual(report["optimizer_steps"], 4)
                self.assertFalse(report["early_stop_update_kl"])
            for key in reports[0]:
                if key == "seconds":
                    continue
                left, right = reports[0][key], reports[1][key]
                if type(left) is float:
                    self.assertAlmostEqual(left, right, delta=2e-7, msg=key)
                else:
                    self.assertEqual(left, right, key)
            for key, left in actors[0].state_dict().items():
                torch.testing.assert_close(left, actors[1].state_dict()[key], rtol=1e-5, atol=1e-9)
            first, second = [trainer.optimizer.state_dict()["state"] for trainer in trainers]
            self.assertEqual(first.keys(), second.keys())
            for parameter in first:
                for key in first[parameter]:
                    torch.testing.assert_close(first[parameter][key], second[parameter][key],
                                               rtol=1e-5, atol=1e-8)
        self.assertGreater(actors[0].policy.weight.abs().sum().item(), 0)
        self.assertGreater(actors[0].value_head.weight.abs().sum().item(), 0)
        self.assertEqual(set(actors[0].batch_sizes), {1})
        self.assertIn(4, actors[1].batch_sizes)
        if count == 11:
            self.assertIn(3, actors[1].batch_sizes)

    def test_full_sixteen_response_rollout(self):
        self.check_equivalence(16)

    def test_eleven_valid_responses_include_partial_minibatch_and_microbatch(self):
        self.check_equivalence(11)


if __name__ == "__main__":
    unittest.main()
