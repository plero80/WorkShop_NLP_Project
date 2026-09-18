"""CPU-only timing checks; no model assets, transformers, peft, or GPU needed."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))

from gsm8k_experiment.timing import TrainingTiming


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.syncs = 0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds

    def synchronize(self):
        self.syncs += 1


def observer(clock, messages, **overrides):
    options = dict(arm="knn_static", seed=42, pool_questions=5000,
                   prompts_per_update=8, responses_per_prompt=2,
                   minibatch_size=8, ppo_epochs=2, target_updates=400,
                   full_run_updates=400, clock=clock,
                   synchronize=clock.synchronize, emit=messages.append)
    options.update(overrides)
    return TrainingTiming(**options)


class TimingTests(unittest.TestCase):
    def test_third_actual_step_and_epoch_definitions(self):
        clock, messages = FakeClock(), []
        timing = observer(clock, messages)
        clock.advance(1000)  # Setup is excluded.
        timing.rollout_started(1)
        clock.advance(80)  # Generation, grading and old/reference statistics.
        for duration in (2, 4, 6):
            timing.minibatch_started()
            clock.advance(duration)
            timing.optimizer_step_completed()
            self.assertEqual(len(messages), int(timing.successful_optimizer_steps == 3))
        self.assertIsNone(timing.rollout_estimate)
        self.assertEqual(timing.optimizer_estimate["optimizer_only_ppo_epoch_seconds"], 8)
        # Fourth step belongs to the same rollout but must not delay/repeat step-3 output.
        timing.minibatch_started()
        clock.advance(8)
        timing.optimizer_step_completed()
        estimate = timing.rollout_completed(1)
        self.assertEqual(len(messages), 2)
        self.assertEqual(estimate["mean_rollout_seconds"], 100)
        self.assertEqual(estimate["nominal_data_epoch_updates"], 625)
        self.assertEqual(estimate["nominal_data_epoch_optimizer_steps"], 2500)
        self.assertEqual(estimate["estimated_data_epoch_seconds"], 62500)
        self.assertEqual(estimate["estimated_full_run_seconds"], 40000)
        self.assertEqual(estimate["estimated_remaining_target_seconds"], 39900)
        self.assertIn("one pass over 16 responses (2 optimizer steps)", messages[0])
        self.assertIn("Full 400-update run (nominally 1600 optimizer steps)", messages[1])
        # Later training neither resynchronizes for timing nor emits repeated estimates.
        syncs = clock.syncs
        timing.rollout_started(2)
        timing.minibatch_started()
        timing.optimizer_step_completed()
        self.assertIsNone(timing.rollout_completed(2))
        self.assertEqual(clock.syncs, syncs)
        self.assertEqual(len(messages), 2)

    def test_resume_counts_new_steps_and_waits_for_rollout_boundary(self):
        clock, messages = FakeClock(), []
        timing = observer(clock, messages, resumed_update=120)
        timing.rollout_started(121)
        clock.advance(10)  # A skipped rollout: no graded responses, no Adam step.
        self.assertIsNone(timing.rollout_completed(121))
        timing.rollout_started(122)
        for _ in range(2):
            timing.minibatch_started()
            clock.advance(2)
            timing.optimizer_step_completed()
        # A rejected minibatch does not count as an optimizer step or its duration.
        timing.minibatch_started()
        clock.advance(9)
        self.assertIsNone(timing.rollout_completed(122))
        self.assertEqual(messages, [])
        timing.rollout_started(123)
        timing.minibatch_started()
        clock.advance(2)
        timing.optimizer_step_completed()
        self.assertEqual(len(messages), 1)
        self.assertEqual(timing.optimizer_estimate["mean_successful_minibatch_seconds"], 2)
        self.assertIn("3 new successful optimizer steps in this invocation", messages[0])
        self.assertIn("resumed after rollout update 120", messages[0])
        estimate = timing.rollout_completed(123)
        self.assertEqual(estimate["measured_rollout_updates"], 3)
        self.assertEqual(estimate["new_successful_optimizer_steps"], 3)
        self.assertEqual(estimate["mean_rollout_seconds"], 25 / 3)
        self.assertEqual(estimate["remaining_target_updates"], 277)

    def test_synchronization_precedes_end_timestamp(self):
        clock, messages = FakeClock(), []
        def synchronize():
            clock.advance(1)  # Simulate waiting for pending device work.
        timing = observer(clock, messages, synchronize=synchronize)
        for _ in range(3):
            timing.minibatch_started()
            clock.advance(2)
            timing.optimizer_step_completed()
        self.assertEqual(timing.step_seconds, [3, 3, 3])


class EngineTimingTests(unittest.TestCase):
    def test_shared_engine_emits_during_third_adam_step_not_third_microbatch(self):
        import numpy as np
        import torch
        from ppo_engine import PPOTrainer

        class Actor(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.policy = torch.nn.Linear(1, 1, bias=False)
                self.value_head = torch.nn.Linear(1, 1, bias=False)
                torch.nn.init.zeros_(self.policy.weight)
                torch.nn.init.zeros_(self.value_head.weight)
                self.device_name = "cpu"
                self.tokenizer = SimpleNamespace(pad_token_id=0)
                self.statistics_calls = 0

            def generate(self, prompts, seed):
                count = len(prompts)
                return [{"ids": torch.ones(count, 3, dtype=torch.long),
                         "attention": torch.ones(count, 3, dtype=torch.long),
                         "response_mask": torch.ones(count, 2, dtype=torch.bool),
                         "prompt_width": 1, "answers": ["answer"] * count,
                         "ended_eos": [True] * count}]

            def statistics(self, ids, attention, width, reference=False, with_values=True):
                self.statistics_calls += 1
                features = torch.ones(ids.shape[0], ids.shape[1] - width, 1)
                logp = self.policy(features).squeeze(-1) - 1
                if reference:
                    logp = torch.full_like(logp, -1)
                values = self.value_head(features).squeeze(-1) if with_values else None
                return logp, values

        class Reward:
            def score(self, prompts, answers, branch):
                zeros = np.zeros(len(prompts))
                return {"reward": np.linspace(-1, 1, len(prompts)),
                        **{key: zeros for key in ("proxy_z", "gap_hat", "applied_gap",
                                                   "reward_tokens", "within_distance_gate")}}

        config = dict(learning_rate=1e-5, value_learning_rate=1e-5, micro_batch_size=1,
                      mini_batch_size=8, ppo_epochs=2, kl_coefficient=.02, gamma=1,
                      gae_lambda=.95, clip_range=.2, value_clip_range=.2,
                      value_coefficient=.1, max_grad_norm=1, target_update_kl=.03,
                      reward_max_tokens=100)
        actor = Actor()
        messages = []
        timing = observer(FakeClock(), messages)
        trainer = PPOTrainer(actor, Reward(), config, timing_observer=timing)
        real_step = trainer.optimizer.step
        completed = []
        observations = []

        def counted_step(*args, **kwargs):
            result = real_step(*args, **kwargs)
            completed.append(True)
            return result

        def capture(message):
            observations.append((len(completed), actor.statistics_calls))
            messages.append(message)

        trainer.optimizer.step = counted_step
        timing.emit = capture
        report = trainer.update([{"prompt": str(i)} for i in range(16)], "test", 42, 1)
        self.assertEqual(report["optimizer_steps"], 4)
        self.assertEqual(len(completed), 4)
        self.assertEqual(len(messages), 1)
        self.assertEqual(observations, [(3, 56)])  # 32 old/ref + 3 x 8 microbatches.
        self.assertEqual(timing.successful_optimizer_steps, 4)

        # Enabling the observer must not change learned parameters or PPO metrics.
        plain_actor = Actor()
        plain = PPOTrainer(plain_actor, Reward(), config)
        plain_report = plain.update([{"prompt": str(i)} for i in range(16)], "test", 42, 1)
        for name, value in actor.state_dict().items():
            torch.testing.assert_close(value, plain_actor.state_dict()[name], rtol=0, atol=0)
        for key in report:
            if key != "seconds":
                self.assertEqual(report[key], plain_report[key])


if __name__ == "__main__":
    unittest.main()
