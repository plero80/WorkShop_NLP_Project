"""Early runtime estimates from actual training, without changing PPO updates.

Counts belong to this train_arm invocation, including after a checkpoint resume.
A successful optimizer minibatch includes forward/backward, gradient clipping,
and Adam; it is not a microbatch, a rollout, or an entire PPO epoch.
"""
from __future__ import annotations

import math
import time


def _print(message):
    print(message, flush=True)


def _duration(seconds):
    return f"{seconds:.1f} s ({seconds / 60:.1f} min)"


class TrainingTiming:
    """Emit once at optimizer step 3, then at its completed rollout boundary.

    ``synchronize`` must wait for device work; a CPU caller can omit it. Only
    the early measured steps/rollouts synchronize. Timing is deliberately not
    persisted: a resumed process measures three *new* successful Adam steps.
    """

    def __init__(self, *, arm, seed, pool_questions, prompts_per_update,
                 responses_per_prompt, minibatch_size, ppo_epochs,
                 target_updates, full_run_updates, resumed_update=0,
                 synchronize=None, clock=time.perf_counter, emit=_print):
        counts = (pool_questions, prompts_per_update, responses_per_prompt,
                  minibatch_size, ppo_epochs, target_updates, full_run_updates)
        if any(x < 1 for x in counts) or not 0 <= resumed_update <= target_updates:
            raise ValueError("Timing requires positive budgets and a valid resume update.")
        self.arm, self.seed = arm, seed
        self.pool_questions = pool_questions
        self.prompts_per_update = prompts_per_update
        self.responses_per_update = prompts_per_update * responses_per_prompt
        self.steps_per_ppo_epoch = math.ceil(self.responses_per_update / minibatch_size)
        self.ppo_epochs = ppo_epochs
        self.steps_per_rollout = self.steps_per_ppo_epoch * ppo_epochs
        self.data_epoch_updates = math.ceil(pool_questions / prompts_per_update)
        self.target_updates, self.full_run_updates = target_updates, full_run_updates
        self.resumed_update = resumed_update
        self.synchronize = synchronize or (lambda: None)
        self.clock, self.emit = clock, emit
        self.successful_optimizer_steps = 0
        self.step_seconds = []
        self.rollout_seconds = []
        self.optimizer_estimate = None
        self.rollout_estimate = None
        self._minibatch_started_at = None
        self._rollout_started_at = None
        self._rollout_update = None

    def minibatch_started(self):
        if self.successful_optimizer_steps < 3:
            self.synchronize()
            # An earlier minibatch rejected by the KL guard has no completed
            # optimizer step; replace its timer rather than count it.
            self._minibatch_started_at = self.clock()

    def optimizer_step_completed(self):
        if self.successful_optimizer_steps < 3:
            if self._minibatch_started_at is None:
                raise RuntimeError("Optimizer timing requires a started minibatch.")
            self.synchronize()
            self.step_seconds.append(self.clock() - self._minibatch_started_at)
            self._minibatch_started_at = None
        self.successful_optimizer_steps += 1
        if self.successful_optimizer_steps != 3:
            return
        mean = sum(self.step_seconds) / 3
        self.optimizer_estimate = {
            "new_successful_optimizer_steps": 3,
            "resumed_after_rollout_update": self.resumed_update,
            "mean_successful_minibatch_seconds": mean,
            "nominal_steps_per_ppo_epoch": self.steps_per_ppo_epoch,
            "optimizer_only_ppo_epoch_seconds": mean * self.steps_per_ppo_epoch,
        }
        self.emit(
            f"TIMING {self.arm} seed={self.seed}: 3 new successful optimizer steps "
            f"in this invocation (resumed after rollout update {self.resumed_update}); "
            f"mean successful optimizer minibatch={_duration(mean)}. "
            f"Optimizer-only PPO epoch estimate={_duration(mean * self.steps_per_ppo_epoch)} "
            f"for one pass over {self.responses_per_update} responses "
            f"({self.steps_per_ppo_epoch} optimizer steps); "
            f"{self.ppo_epochs} PPO epochs per rollout update. "
            "Includes forward/backward/gradient clipping/Adam; excludes generation, "
            "grading, old/reference statistics and GAE, setup, evaluation, and I/O. "
            "Nominal schedule assumes all responses graded and no KL early stop. "
            "End-to-end estimate follows when this rollout finishes."
        )

    def rollout_started(self, update):
        if self.rollout_estimate is not None:
            return
        self.synchronize()
        self._rollout_started_at = self.clock()
        self._rollout_update = update

    def rollout_completed(self, update):
        if self.rollout_estimate is not None:
            return None
        if self._rollout_started_at is None or update != self._rollout_update:
            raise RuntimeError("Rollout timing start/end must match.")
        self.synchronize()
        self.rollout_seconds.append(self.clock() - self._rollout_started_at)
        self._rollout_started_at = None
        if self.successful_optimizer_steps < 3:
            return None
        mean = sum(self.rollout_seconds) / len(self.rollout_seconds)
        remaining = max(0, self.target_updates - update)
        self.rollout_estimate = {
            "new_successful_optimizer_steps": self.successful_optimizer_steps,
            "resumed_after_rollout_update": self.resumed_update,
            "measured_rollout_updates": len(self.rollout_seconds),
            "through_rollout_update": update,
            "mean_rollout_seconds": mean,
            "nominal_data_epoch_updates": self.data_epoch_updates,
            "nominal_data_epoch_optimizer_steps": self.data_epoch_updates * self.steps_per_rollout,
            "estimated_data_epoch_seconds": mean * self.data_epoch_updates,
            "full_run_updates": self.full_run_updates,
            "estimated_full_run_seconds": mean * self.full_run_updates,
            "target_updates": self.target_updates,
            "remaining_target_updates": remaining,
            "estimated_remaining_target_seconds": mean * remaining,
        }
        self.emit(
            f"TIMING {self.arm} seed={self.seed}: end-to-end estimate from "
            f"{len(self.rollout_seconds)} completed rollout update(s) in this invocation, "
            f"through update {update}: mean={_duration(mean)}. "
            f"One data epoch ({self.pool_questions} questions / "
            f"{self.prompts_per_update} questions per rollout, rounded up) = "
            f"{self.data_epoch_updates} rollout updates, nominally "
            f"{self.data_epoch_updates * self.steps_per_rollout} optimizer steps: "
            f"{_duration(mean * self.data_epoch_updates)}. "
            f"Full {self.full_run_updates}-update run (nominally "
            f"{self.full_run_updates * self.steps_per_rollout} optimizer steps): "
            f"{_duration(mean * self.full_run_updates)}. "
            f"Remaining to requested target {self.target_updates} "
            f"({remaining} updates): {_duration(mean * remaining)}. "
            "Includes generation, reward grading, old/reference statistics and GAE, "
            "and optimization; excludes setup, evaluation, post-rollout checkpoint/artifact I/O "
            "and memory refresh. Early rough extrapolation: response lengths, "
            "grading failures and KL early stops can change throughput."
        )
        return self.rollout_estimate
