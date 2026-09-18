"""CPU regression for smoke likelihood checks with BF16-like shape sensitivity."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))
from gsm8k_experiment import run
from gsm8k_experiment.shared import pack_items


class ShapeSensitivePolicy:
    """Different batch sizes/padding/hidden-state flags produce different scores."""

    def __init__(self, *, current_drift=0.0, reference_drift=0.0):
        self.device = torch.device("cpu")
        self.tokenizer = SimpleNamespace(pad_token_id=0)
        self.weight = torch.tensor(0.0)
        self.current_drift = current_drift
        self.reference_drift = reference_drift
        self.calls = []
        self.updated = False
        self.restores = 0

    def eval(self):
        return self

    def sample(self, rows, greedy=False):
        return [dict(rows[0], prompt_ids=[1], response_ids=[10], response="15"),
                dict(rows[1], prompt_ids=[2, 3, 4], response_ids=[20, 21, 22], response="6")]

    def statistics(self, ids, attention, prompt_width, reference=False, with_values=True):
        self.calls.append((tuple(ids.shape), prompt_width, reference, with_values))
        score = .1 * len(ids) + .01 * ids.shape[1] + .001 * prompt_width + .003 * with_values
        lp = torch.full((len(ids), ids.shape[1] - prompt_width), score)
        second = ids[:, prompt_width] == 20
        # prepare_rollout performs 2 forwards per chunk. Current parity follows
        # after its final reference call; reference drift starts after the update.
        if not reference and any(call[2] for call in self.calls[:-1]):
            if len(self.calls) > 4:
                lp[second] += self.current_drift
        if reference and self.updated:
            lp[second] += self.reference_drift
        return lp, torch.zeros_like(lp) if with_values else None

    def token_stats_batch(self, items, reference=False):
        packed = pack_items(items, self.tokenizer.pad_token_id)
        lp, values = self.statistics(packed["ids"], packed["attention"], packed["prompt_width"],
                                     reference=reference)
        return [(lp[i, :len(item["response_ids"])], values[i, :len(item["response_ids"])])
                for i, item in enumerate(items)]

    def trainable_state(self):
        return {"adapter": {"weight": self.weight.clone()}}

    def restore_trainable(self, state):
        self.weight = state["adapter"]["weight"].clone()
        self.updated = False
        self.restores += 1


class SmokeShapeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name)
        self.config = {"seed": 42, "runtime": {"rollout_stats_batch_size": 1},
                       "ppo": {"kl_coefficient": .01, "gamma": 1.0, "gae_lambda": .95}}

    def smoke(self, policy):
        initial = policy.trainable_state()
        def update(actor, optimizer, rollout, config, index):
            actor.weight += 1
            actor.updated = True
            return {"optimizer_steps": 2}
        with mock.patch.object(run, "optimizer_for", return_value=object()), \
             mock.patch.object(run, "update", side_effect=update), \
             mock.patch.object(run, "seed_all"), \
             mock.patch.object(run.torch.cuda, "empty_cache"):
            run.smoke_ppo(policy, self.config, self.output, initial)

    def test_preserves_full_padding_and_configured_batch_size_and_flags(self):
        policy = ShapeSensitivePolicy()
        self.smoke(policy)
        result = json.loads((self.output / "preflight_ppo.json").read_text())
        self.assertTrue(result["passed"])
        self.assertEqual(result["old_current_max_error"], 0)
        self.assertEqual(result["frozen_reference_max_drift"], 0)
        self.assertEqual(result["optimizer_steps"], 2)
        self.assertTrue(result["updates_discarded"])
        self.assertEqual(policy.restores, 1)
        self.assertEqual(float(policy.weight), 0)
        self.assertEqual(len(policy.calls), 8)
        self.assertTrue(all(shape == (1, 6) and width == 3 for shape, width, _, _ in policy.calls))
        self.assertTrue(all(with_values is not reference for _, _, reference, with_values in policy.calls))

    def test_larger_configured_statistics_batch_still_matches(self):
        self.config["runtime"]["rollout_stats_batch_size"] = 2
        policy = ShapeSensitivePolicy()
        self.smoke(policy)
        self.assertEqual(len(policy.calls), 4)
        self.assertTrue(all(shape == (2, 6) for shape, _, _, _ in policy.calls))
        self.assertEqual(float(policy.weight), 0)

    def test_current_mismatch_in_second_answer_is_rejected_at_original_threshold(self):
        policy = ShapeSensitivePolicy(current_drift=.003)
        with self.assertRaisesRegex(RuntimeError, "Old/current log probability mismatch"):
            self.smoke(policy)
        self.assertFalse((self.output / "preflight_ppo.json").exists())
        self.assertEqual(policy.restores, 1)
        self.assertEqual(float(policy.weight), 0)

    def test_reference_drift_in_second_answer_is_rejected_and_weights_restored(self):
        policy = ShapeSensitivePolicy(reference_drift=.003)
        with self.assertRaisesRegex(RuntimeError, "reference_drift"):
            self.smoke(policy)
        self.assertFalse((self.output / "preflight_ppo.json").exists())
        self.assertEqual(policy.restores, 1)
        self.assertEqual(float(policy.weight), 0)

    def test_error_below_original_threshold_is_recorded_across_all_answers(self):
        policy = ShapeSensitivePolicy(current_drift=.001, reference_drift=.001)
        self.smoke(policy)
        result = json.loads((self.output / "preflight_ppo.json").read_text())
        self.assertAlmostEqual(result["old_current_max_error"], .001, places=6)
        self.assertAlmostEqual(result["frozen_reference_max_drift"], .001, places=6)


if __name__ == "__main__":
    unittest.main()
