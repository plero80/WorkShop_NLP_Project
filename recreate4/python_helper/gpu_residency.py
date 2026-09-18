"""Keep the actor on CUDA and swap only frozen graders for the 12 GB profile.

This process-local adapter leaves the restored experiment sources, parameter
dtypes, scorer identities, actor parameters, and optimizer state unchanged.
"""
from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
import threading
import weakref


class GraderResidency:
    def __init__(self, torch_module):
        self.torch = torch_module
        self._graders = weakref.WeakSet()
        self._resident = None
        self._lock = threading.RLock()
        self._patches = []

    def _current(self):
        return self._resident() if self._resident is not None else None

    def _park(self, scorer):
        # Synchronize before releasing storage used by an earlier CUDA forward.
        self.torch.cuda.synchronize(scorer.device)
        scorer.model.to("cpu")
        self._resident = None
        self.torch.cuda.empty_cache()

    def evict(self):
        """Park the active grader; never move the actor or its optimizer."""
        with self._lock:
            current = self._current()
            if current is not None:
                self._park(current)

    def register(self, scorer):
        """An original constructor has just placed this frozen model on CUDA."""
        with self._lock:
            if self._current() is not None:
                raise RuntimeError("A grader was constructed without first evicting the active grader.")
            self._graders.add(scorer)
            self._resident = weakref.ref(scorer)
            self._park(scorer)
            if any(parameter.requires_grad for parameter in scorer.model.parameters()):
                raise ValueError("12 GB residency supports only frozen reward models.")

    def activate(self, scorer):
        with self._lock:
            if scorer not in self._graders:
                raise RuntimeError("Reward scorer was not registered with the residency controller.")
            if self._current() is scorer:
                return
            self.evict()
            # Device-only conversion preserves BF16 weights and the scorer's
            # logical CUDA device. Do not mutate config or call model.float().
            scorer.model.to(scorer.device)
            self._resident = weakref.ref(scorer)

    @contextmanager
    def actor_operation(self):
        with self._lock:
            self.evict()
            yield

    @contextmanager
    def grader_operation(self, scorer):
        with self._lock:
            self.activate(scorer)
            yield

    def _replace(self, cls, name, replacement):
        original = getattr(cls, name)
        self._patches.append((cls, name, original, replacement, name in cls.__dict__))
        setattr(cls, name, replacement)

    def patch(self, models):
        scorer_class, policy_class = models.RewardScorer, models.Policy
        original_init = scorer_class.__init__

        @wraps(original_init)
        def initialize(scorer, role, *args, **kwargs):
            if role not in ("proxy", "judge"):
                raise ValueError("The 12 GB profile supports only the canonical proxy and judge graders.")
            with self.actor_operation():
                original_init(scorer, role, *args, **kwargs)
                self.register(scorer)

        original_infer = scorer_class._infer

        @wraps(original_infer)
        def infer(scorer, *args, **kwargs):
            with self.grader_operation(scorer):
                return original_infer(scorer, *args, **kwargs)

        self._replace(scorer_class, "__init__", initialize)
        self._replace(scorer_class, "_infer", infer)
        for name in ("sample", "statistics", "token_stats_batch"):
            original = getattr(policy_class, name)

            def actor_wrapper(original_method):
                @wraps(original_method)
                def call(policy, *args, **kwargs):
                    with self.actor_operation():
                        return original_method(policy, *args, **kwargs)
                return call

            self._replace(policy_class, name, actor_wrapper(original))
        return self

    def close(self):
        try:
            self.evict()
        finally:
            for cls, name, original, replacement, owned in reversed(self._patches):
                if getattr(cls, name) is replacement:
                    if owned:
                        setattr(cls, name, original)
                    else:
                        delattr(cls, name)
            self._patches.clear()


def install(models_module=None, torch_module=None):
    """Install before importing the experiment runner; return a closeable guard."""
    if models_module is None:
        from gsm8k_experiment import models as models_module
    if torch_module is None:
        import torch as torch_module
    controller = GraderResidency(torch_module)
    try:
        return controller.patch(models_module)
    except BaseException:
        controller.close()
        raise
