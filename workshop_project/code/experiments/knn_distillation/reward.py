"""Frozen composite teacher and a student-only PPO scoring route."""
from pathlib import Path
import numpy as np
from knn_distillation.io import read, sha, digest
from knn_distillation.maths import predict_memory


class RewardRouter:
    def __init__(self, proxy, judge, calibration, threads=8):
        self.proxy, self.judge, self.calibration, self.threads = proxy, judge, calibration, threads
        self.memory = None; self.memory_hash = None; self.students = {}; self.student_ids = {}
        self.reset_cost()

    def reset_cost(self):
        self.cost = {k: 0 for k in ('proxy_answers', 'teacher_answers', 'student_answers', 'knn_queries',
                                   'proxy_input_tokens', 'teacher_input_tokens', 'student_input_tokens')}

    def load_memory(self, folder):
        folder = Path(folder); lock = read(folder / 'locked_reward.json')
        if sha(folder / 'refreshed_memory.npz') != lock['memory_sha256']:
            raise ValueError('Teacher memory checksum changed.')
        if lock['k'] != 31 or lock['temperature'] != .05 or lock.get('calibration', self.calibration) != self.calibration:
            raise ValueError('Teacher memory/normalization protocol changed.')
        with np.load(folder / 'refreshed_memory.npz', allow_pickle=False) as f:
            memory = {k: f[k].copy() for k in ('vectors', 'gaps')}
        v, g = memory['vectors'], memory['gaps']
        if v.ndim != 2 or len(v) < 31 or g.shape != (len(v),) or not np.isfinite(v).all() or not np.isfinite(g).all():
            raise ValueError('Invalid teacher memory dimensions/values.')
        if not np.allclose(np.linalg.norm(v, axis=1), 1., atol=1e-4):
            raise ValueError('Teacher vectors must be unit normalized.')
        for array in memory.values():
            array.flags.writeable = False
        self.memory, self.memory_hash = memory, sha(folder / 'locked_reward.json')

    def set_students(self, students):
        self.students = students
        self.student_ids = {key: scorer.identity for key, scorer in students.items()}

    def proxy_values(self, prompts, answers, features=False):
        v = self.proxy.score(prompts, answers, features=features)
        self.cost['proxy_answers'] += len(answers); self.cost['proxy_input_tokens'] += int(sum(v['tokens']))
        return v

    def judge_values(self, prompts, answers):
        v = self.judge.score(prompts, answers)
        self.cost['teacher_answers'] += len(answers); self.cost['teacher_input_tokens'] += int(sum(v['tokens']))
        return v

    def student_values(self, prompts, answers, key):
        v = self.students[key].score(prompts, answers)
        self.cost['student_answers'] += len(answers); self.cost['student_input_tokens'] += int(sum(v['tokens']))
        return v

    def corrected(self, proxy):
        if self.memory is None:
            raise ValueError('Load the frozen teacher memory first.')
        self.cost['knn_queries'] += len(proxy['raw'])
        gap, distance = predict_memory(proxy['features'], self.memory, threads=self.threads)
        z = (proxy['raw'] - self.calibration['proxy_mean']) / self.calibration['proxy_std']
        return z - gap, gap, distance

    def targets(self, prompts, answers, include_judge=False):
        p = self.proxy_values(prompts, answers, features=True)
        target, gap, distance = self.corrected(p)
        result = {'teacher_z': target, 'proxy_z': (p['raw'] - self.calibration['proxy_mean']) / self.calibration['proxy_std'],
                  'proxy_raw': p['raw'], 'gap_hat': gap, 'tokens': p['tokens']}
        if include_judge:
            j = self.judge_values(prompts, answers)
            result['judge_z'] = (j['raw'] - self.calibration['judge_mean']) / self.calibration['judge_std']
        return result

    def route_identity(self, branch):
        if branch in self.students:
            return self.student_ids[branch]
        return self.memory_hash if branch == 'knn' else digest(['frozen_proxy', self.calibration])

    def score(self, prompts, answers, branch):
        if branch in ('student', 'judge_student'):
            # Only S runs here: no original proxy, judge, embedding extraction or kNN lookup.
            s = self.student_values(prompts, answers, branch)
            z = (s['raw'] - self.calibration['proxy_mean']) / self.calibration['proxy_std']
            zeros = np.zeros(len(z))
            # Required legacy trainer diagnostics are placeholders. ppo_training clears
            # these fields to None before any history/report is written.
            return {'reward': z, 'proxy_z': zeros, 'gap_hat': zeros, 'applied_gap': zeros,
                    'within_distance_gate': np.zeros(len(z), bool), 'reward_tokens': s['tokens'],
                    'reward_truncated': s['truncated'], 'diagnostics_available': False}
        if branch not in ('proxy', 'knn'):
            raise ValueError('Unknown PPO reward: ' + branch)
        p = self.proxy_values(prompts, answers, features=branch == 'knn')
        z = (p['raw'] - self.calibration['proxy_mean']) / self.calibration['proxy_std']
        result, gap = z, np.zeros(len(z))
        if branch == 'knn':
            result, gap, _ = self.corrected(p)
        return {'reward': result, 'proxy_z': z, 'gap_hat': gap, 'applied_gap': gap,
                'within_distance_gate': np.zeros(len(z), bool), 'reward_tokens': p['tokens'],
                'reward_truncated': p['truncated'], 'diagnostics_available': True}

    def evaluate_all(self, prompts, answers, branch):
        p = self.proxy_values(prompts, answers, features=True)
        j = self.judge_values(prompts, answers)
        knn, gap, distance = self.corrected(p)
        cal = self.calibration
        zp, zj = (p['raw'] - cal['proxy_mean']) / cal['proxy_std'], (j['raw'] - cal['judge_mean']) / cal['judge_std']
        scores = {'proxy': zp, 'knn': knn}
        for key in self.students:
            s = self.student_values(prompts, answers, key)
            scores[key] = (s['raw'] - cal['proxy_mean']) / cal['proxy_std']
        return p, j, {'reward': scores['proxy'] if branch == 'initial' else scores[branch],
                     'proxy_z': zp, 'judge_z': zj, 'gap_hat': gap, 'mean_neighbor_distance': distance,
                     'knn_z': knn, 'student_z': scores.get('student'), 'judge_student_z': scores.get('judge_student')}
