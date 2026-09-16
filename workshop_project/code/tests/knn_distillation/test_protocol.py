"""Data boundaries, reward units, label resume and source selection checks."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from knn_distillation import data, io, labels
from knn_distillation.reward import RewardRouter
from knn_distillation.offline import regression_metrics, pair_agreement


def prompts(n, prefix='p'):
    return [{'prompt_id': f'{prefix}{i}', 'prompt': f'question {prefix} {i}'} for i in range(n)]


class ProtocolTests(unittest.TestCase):
    def test_prompt_partition_and_ppo_reuse_never_include_validation_or_test(self):
        o = io.read(io.ROOT / 'settings.json')
        o.update(train_prompts=12, validation_prompts=4, offline_prompts=4)
        split = data.partition_training(prompts(20), o)
        data.validate_cohorts(split)
        sampled = data.schedule(split['distill_train'], 20, 4, 7)
        self.assertEqual(len(sampled), 80)
        self.assertTrue(data.groups(sampled).isdisjoint(data.groups(split['distill_validation'] + split['distill_offline'])))
        self.assertEqual(split, data.partition_training(prompts(20)[::-1], o))
        with self.assertRaisesRegex(ValueError, 'requires'):
            data.partition_training(prompts(19), o)

    def test_normalized_teacher_target_keeps_signed_correction_and_no_judge_calls(self):
        class Proxy:
            def score(self, prompts, answers, features=False):
                return {'raw': np.full(len(prompts), 6.), 'tokens': np.ones(len(prompts), int),
                        'features': np.tile([1., 0.], (len(prompts), 1)), 'truncated': np.zeros(len(prompts), bool)}
        class Forbidden:
            def score(self, *args, **kwargs):
                raise AssertionError('Unexpected teacher/proxy access')
        cal = {'proxy_mean': 2., 'proxy_std': 2.}
        router = RewardRouter(Proxy(), Forbidden(), cal, threads=1)
        router.memory = {'vectors': np.tile([1., 0.], (31, 1)), 'gaps': np.full(31, -.5)}
        target = router.targets(['q'], ['a'])
        np.testing.assert_allclose(target['teacher_z'], [2.5])
        self.assertEqual(router.cost['teacher_answers'], 0)
        class Student:
            identity = 'student'
            def score(self, prompts, answers):
                return {'raw': np.full(len(prompts), 7.), 'tokens': np.ones(len(prompts), int), 'truncated': np.zeros(len(prompts), bool)}
        router.set_students({'student': Student()}); router.proxy = Forbidden(); router.memory = None
        router.reset_cost()
        np.testing.assert_allclose(router.score(['q'], ['a'], 'student')['reward'], [2.5])
        self.assertEqual(router.cost['proxy_answers'], 0)
        self.assertEqual(router.cost['knn_queries'], 0)

    def test_label_pause_resume_preserves_two_answers_per_prompt(self):
        from common import StopRequested
        class Actor:
            calls = 0
            def generate(self, batch, seed):
                Actor.calls += 1
                if Actor.calls == 1:
                    io.STOP = True
                return [{'answers': ['answer'] * len(batch)}]
        class Teacher:
            memory_hash = 'fixed_memory'
            def reset_cost(self):
                self.cost = {'proxy_answers': 0, 'teacher_answers': 0, 'knn_queries': 0}
            def targets(self, prompts, answers, include_judge):
                n = len(prompts); self.cost.update(proxy_answers=n, knn_queries=n)
                return {'teacher_z': np.ones(n), 'proxy_z': np.full(n, 2.), 'gap_hat': np.ones(n), 'tokens': np.ones(n)}
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            args = (out, {'generation_batch_size': 1, 'max_new_tokens': 256}, {'data_seed': 17}, {'policy': 'fixture'},
                    Teacher(), prompts(2), 42, {'sha256': 'source', 'path': 'fixture', 'identity': 'source', 'branch': 'parent'},
                    'experiment', 'train', False)
            try:
                with patch('run_study.load_actor', return_value=Actor()), patch('run_study.checkpoint_actor', return_value=Actor()):
                    with self.assertRaises(StopRequested):
                        labels.generate_labels(*args)
                    io.STOP = False
                    folder = labels.generate_labels(*args)
                    rows = labels.load_labels(folder)
                    self.assertEqual(Actor.calls, 4)
                    self.assertEqual(len(rows), 4)
                    self.assertEqual({r['origin'] for r in rows}, {'base', 'parent'})
                    self.assertEqual(io.read(folder / 'complete.json')['costs']['teacher_answers'], 0)
                    labels.generate_labels(*args)
                    self.assertEqual(Actor.calls, 4)
            finally:
                io.STOP = False

    def test_pairwise_ties_and_degenerate_correlation_are_explicit(self):
        frame = pd.DataFrame({'prompt_id': ['a', 'a', 'b', 'b'], 'teacher_z': [1., 1., 1., 2.], 'student_z': [0., 2., 0., 0.]})
        result = pair_agreement(frame, 'student_z', 'teacher_z')
        self.assertEqual(result, {'non_tied_teacher_pairs': 1, 'ordering_agreement': .5})
        self.assertIsNone(regression_metrics([1., 1.], [2., 3.])['pearson'])

    def test_larger_source_must_be_final_locked_adaptive_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / 'refresh34_outputs/source'
            source.mkdir(parents=True)
            manifest = {'refresh2_identity': 'M2_source', 'options': {'rounds': 2, 'updates_per_round': 100},
                        'parents': {'42': {'update': 300}}}
            manifest['identity'] = io.digest(manifest)
            io.write(source / 'manifest.json', manifest); io.write(source / 'status.json', {'stage': 'complete'})
            cp = source / 'runs/checkpoints/checkpoint_000500.pt'; cp.parent.mkdir(parents=True); cp.write_text('fixture')
            metadata = {'name': cp.name, 'identity': manifest['identity'], 'branch': 'adaptive', 'seed': 42, 'update': 500, 'sha256': io.sha(cp)}
            io.write(cp.with_suffix('.json'), metadata)
            memory = source / 'memories/M4/seed_42'; memory.mkdir(parents=True)
            (memory / 'refreshed_memory.npz').write_bytes(b'fixture')
            io.write(memory / 'locked_reward.json', {'memory_sha256': io.sha(memory / 'refreshed_memory.npz')})
            io.write(source / 'endpoints.json', {'42/adaptive': {**metadata, 'checkpoint_relative_path': str(cp.relative_to(source)),
                                                               'memory_lock_sha256': io.sha(memory / 'locked_reward.json')}})
            io.write(source / 'final_lock.json', {'endpoints': {'42/adaptive/M4': metadata['sha256']}})
            inspection = (root, root / 'followup', root / 'refresh2', {}, {'identity': 'M2_source', 'parent_config': {}}, {}, {})
            with patch.object(io, 'inspect_project', return_value=inspection):
                result = io.inspect_source(root, {'source_kind': 'refresh34', 'seeds': [42]}, refresh34=source)
                self.assertEqual(result[-2][42]['update'], 500)
                self.assertEqual(result[-1][42], memory)
                io.write(source / 'final_lock.json', {'endpoints': {'42/adaptive/M4': 'wrong'}})
                with self.assertRaisesRegex(ValueError, 'locked final'):
                    io.inspect_source(root, {'source_kind': 'refresh34', 'seeds': [42]}, refresh34=source)

    def test_config_rejects_unmatched_ppo_schedule(self):
        o = io.read(io.ROOT / 'settings.json'); io.check_config(o)
        o['ppo_updates'] = 99
        with self.assertRaisesRegex(ValueError, 'divide'):
            io.check_config(o)


if __name__ == '__main__':
    unittest.main(verbosity=2)
