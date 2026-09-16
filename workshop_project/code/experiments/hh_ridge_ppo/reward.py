"""A ridge route around the existing frozen reward router; PPO stays unchanged."""
import time
import numpy as np
from knn_distillation.reward import RewardRouter
from .protocol import require


class RidgeRouter(RewardRouter):
    def reset_cost(self):
        super().reset_cost()
        self.cost.update(ridge_queries=0, ridge_seconds=0., proxy_seconds=0.)

    def proxy_values(self, prompts, answers, features=False):
        start = time.perf_counter()
        result = super().proxy_values(prompts, answers, features)
        self.cost['proxy_seconds'] += time.perf_counter() - start
        return result

    def set_ridge(self, coef, intercept, identity):
        self.coef = np.asarray(coef, float).copy()
        require(self.memory is not None and self.coef.shape == (self.memory['vectors'].shape[1],) and
                np.isfinite(self.coef).all() and np.isfinite(intercept), "Invalid ridge coefficients")
        self.coef.flags.writeable = False
        self.intercept, self.ridge_identity = float(intercept), identity
        # Included by the unchanged evaluation signature without loading a student.
        self.student_ids = {'ridge_gap': identity}

    def ridge_gap(self, features):
        start = time.perf_counter()
        result = np.asarray(features) @ self.coef + self.intercept
        require(np.isfinite(result).all(), "Nonfinite ridge prediction")
        self.cost['ridge_queries'] += len(result)
        self.cost['ridge_seconds'] += time.perf_counter() - start
        return result

    def route_identity(self, branch):
        return self.ridge_identity if branch == 'ridge' else super().route_identity(branch)

    def score(self, prompts, answers, branch):
        if branch != 'ridge':
            return super().score(prompts, answers, branch)
        p = self.proxy_values(prompts, answers, features=True)
        z = (p['raw'] - self.calibration['proxy_mean']) / self.calibration['proxy_std']
        gap = self.ridge_gap(p['features'])
        return {'reward': z - gap, 'proxy_z': z, 'gap_hat': gap, 'applied_gap': gap,
                'within_distance_gate': np.zeros(len(z), bool), 'reward_tokens': p['tokens'],
                'reward_truncated': p['truncated'], 'diagnostics_available': True}

    def evaluate_all(self, prompts, answers, branch):
        p, j, scores = super().evaluate_all(prompts, answers, 'initial' if branch == 'ridge' else branch)
        gap = self.ridge_gap(p['features'])
        if branch == 'ridge':
            scores['reward'] = scores['proxy_z'] - gap
        return p, j, scores
