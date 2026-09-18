"""CPU-only residency tests: no torch import, model download or CUDA context."""
from __future__ import annotations

import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python_helper"))
from gpu_residency import install


class ResidencyTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.models = []
        events, all_models = self.events, self.models

        class Model:
            def __init__(self, role, trainable=False):
                self.role, self.device, self.dtype = role, "cuda:0", "bfloat16"
                self.parameter = SimpleNamespace(requires_grad=trainable)
                all_models.append(self)

            def parameters(self):
                return iter([self.parameter])

            def to(self, device):
                events.append((self.role, "to", device))
                self.device = device
                if sum(model.device == "cuda:0" for model in all_models) > 1:
                    raise AssertionError("Two graders are resident together")
                return self

        class RewardScorer:
            def __init__(self, role, config, resolved, cache):
                if any(model.device == "cuda:0" for model in all_models):
                    raise AssertionError("Previous grader was not evicted before construction")
                events.append((role, "construct"))
                self.role, self.config, self.device = role, config, "cuda:0"
                self.identity = "unchanged-" + role
                self.model = Model(role, config.get("trainable", False))

            def _infer(self, rows, stage, max_new_tokens, retry=False):
                if self.model.device != self.device:
                    raise AssertionError("Grader forward did not activate CUDA first")
                events.append((self.role, "infer", stage, retry))
                if stage == "fail":
                    raise ValueError("synthetic forward failure")
                return (rows, max_new_tokens, retry)

        class ActorBase:
            def statistics(self, *args, **kwargs):
                self.check()
                return ("statistics", args, kwargs)

        class Policy(ActorBase):
            def __init__(self):
                self.device = "cuda:0"
                self.parameter, self.optimizer_state = object(), object()

            def to(self, device):
                raise AssertionError("The actor must never be moved")

            def check(self):
                if any(model.device == "cuda:0" for model in all_models):
                    raise AssertionError("Actor work started while a grader was resident")

            def sample(self, *args, **kwargs):
                self.check()
                return ("sample", args, kwargs)

            def token_stats_batch(self, *args, **kwargs):
                self.check()
                return self.statistics(*args, **kwargs)

        self.module = SimpleNamespace(RewardScorer=RewardScorer, Policy=Policy)
        self.torch = SimpleNamespace(cuda=SimpleNamespace(
            synchronize=mock.Mock(side_effect=lambda device: events.append(("sync", device))),
            empty_cache=mock.Mock(side_effect=lambda: events.append(("empty_cache",)))))
        self.original_init = RewardScorer.__init__
        self.original_stats = Policy.statistics
        self.controller = install(self.module, self.torch)
        self.addCleanup(self.controller.close)
        self.config = {"runtime": {"device": "cuda:0", "dtype": "bfloat16"}}

    def scorer(self, role):
        return self.module.RewardScorer(role, self.config, {}, None)

    def test_construction_parks_graders_without_changing_identity_or_dtype(self):
        config_before = copy.deepcopy(self.config)
        proxy, judge = self.scorer("proxy"), self.scorer("judge")
        self.assertEqual([proxy.model.device, judge.model.device], ["cpu", "cpu"])
        self.assertEqual(proxy.identity, "unchanged-proxy")
        self.assertEqual(proxy.model.dtype, "bfloat16")
        self.assertEqual(self.config, config_before)
        self.assertEqual(proxy.device, "cuda:0")

    def test_switches_only_frozen_graders_and_reuses_active_model_for_retry(self):
        proxy, judge = self.scorer("proxy"), self.scorer("judge")
        self.events.clear()
        self.assertEqual(proxy._infer(["a"], "score", 160), (["a"], 160, False))
        proxy._infer(["a"], "score", 320, retry=True)
        judge._infer(["b"], "score", 160)
        self.assertEqual([e for e in self.events if len(e) > 1 and e[1] == "to"],
                         [("proxy", "to", "cuda:0"), ("proxy", "to", "cpu"),
                          ("judge", "to", "cuda:0")])
        self.assertLess(self.events.index(("sync", "cuda:0")),
                        self.events.index(("proxy", "to", "cpu")))

    def test_constructor_evicts_a_previously_active_grader_before_loading(self):
        proxy = self.scorer("proxy")
        proxy._infer([], "probe", 160)
        self.events.clear()
        judge = self.scorer("judge")
        self.assertLess(self.events.index(("proxy", "to", "cpu")),
                        self.events.index(("judge", "construct")))
        self.assertEqual(judge.model.device, "cpu")

    def test_each_actor_entry_evicts_graders_without_touching_actor_or_optimizer(self):
        proxy, actor = self.scorer("proxy"), self.module.Policy()
        parameter, state = actor.parameter, actor.optimizer_state
        for name in ("sample", "statistics", "token_stats_batch"):
            with self.subTest(name=name):
                proxy._infer([], "probe", 160)
                result = getattr(actor, name)([1], reference=True)
                self.assertEqual(result[1:], (([1],), {"reference": True}))
                self.assertEqual(proxy.model.device, "cpu")
                self.assertIs(actor.parameter, parameter)
                self.assertIs(actor.optimizer_state, state)
                self.assertEqual(actor.device, "cuda:0")

    def test_forward_failure_releases_guard_and_next_actor_operation_parks_model(self):
        proxy = self.scorer("proxy")
        with self.assertRaisesRegex(ValueError, "synthetic"):
            proxy._infer([], "fail", 160)
        self.module.Policy().sample([])
        self.assertEqual(proxy.model.device, "cpu")

    def test_unknown_teacher_rejected_before_model_loading(self):
        with self.assertRaisesRegex(ValueError, "canonical"):
            self.scorer("judge30b")
        self.assertFalse(self.models)

    def test_trainable_grader_is_parked_then_rejected(self):
        self.config["trainable"] = True
        with self.assertRaisesRegex(ValueError, "frozen"):
            self.scorer("proxy")
        self.assertEqual(self.models[0].device, "cpu")

    def test_close_parks_last_grader_and_restores_exact_class_attributes(self):
        judge = self.scorer("judge")
        judge._infer([], "probe", 160)
        self.controller.close()
        self.assertEqual(judge.model.device, "cpu")
        self.assertIs(self.module.RewardScorer.__init__, self.original_init)
        self.assertIs(self.module.Policy.statistics, self.original_stats)
        self.assertNotIn("statistics", self.module.Policy.__dict__)
        self.controller.close()

    def test_unregistered_scorer_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, "not registered"):
            self.controller.activate(self.module.Policy())


if __name__ == "__main__":
    unittest.main()
