"""Real tiny Qwen tests. All models are generated locally; no downloads."""
from pathlib import Path
import copy
import json
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from transformers import (Qwen3Config, Qwen3ForSequenceClassification, Qwen2Config,
                          Qwen2ForCausalLM, PreTrainedTokenizerFast)
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from safetensors.torch import load_file
from knn_distillation import io, fit
from knn_distillation.student import StudentScorer, verify_bundle
from knn_distillation.reward import RewardRouter


def tiny_snapshot(folder, policy=False):
    folder = Path(folder); folder.mkdir(parents=True)
    words = ['<pad>', '<unk>', '<bos>', '<eos>', 'user', 'assistant', 'question', 'answer', 'short'] + [str(i) for i in range(55)]
    backend = Tokenizer(WordLevel({word: i for i, word in enumerate(words)}, unk_token='<unk>'))
    backend.pre_tokenizer = WhitespaceSplit()
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='<pad>', unk_token='<unk>', bos_token='<bos>', eos_token='<eos>')
    tok.chat_template = "{% for message in messages %}{{ message['role'] }} {{ message['content'] }} {{ eos_token }} {% endfor %}{% if add_generation_prompt %}assistant {% endif %}"
    tok.save_pretrained(folder)
    opts = dict(vocab_size=len(words), hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
                pad_token_id=0, bos_token_id=2, eos_token_id=3)
    torch.manual_seed(17)
    if policy:
        model = Qwen2ForCausalLM(Qwen2Config(**opts))
    else:
        model = Qwen3ForSequenceClassification(Qwen3Config(**opts, head_dim=8, num_labels=1))
    model.save_pretrained(folder)
    return folder


class NeuralTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.snapshot = tiny_snapshot(cls.root / 'reward')
        cls.policy = tiny_snapshot(cls.root / 'policy', policy=True)
        cls.cal = {'proxy_mean': -.3, 'proxy_std': 2., 'judge_mean': 0., 'judge_std': 1., 'theta': 1.}
        cls.o = io.read(io.ROOT / 'settings.json')
        cls.o.update(student_epochs=3, student_batch_size=2, student_accumulation=2,
                     student_lora_rank=2, student_lora_alpha=4, student_lr=.01, student_head_lr=.01,
                     student_checkpoint_every=2, gradient_checkpointing=True)
        cls.c = {'reward_max_tokens': 128}
        cls.base_meta = {'base_model': 'local_tiny_Qwen3', 'base_revision': 'test',
                         'base_config_sha256': io.sha(cls.snapshot / 'config.json')}

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def examples(self, prefix, n=12):
        s = StudentScorer.create(self.snapshot, self.cal, self.c, self.o, 123, 'cpu')
        rows = [{'prompt_id': f'{prefix}{i}', 'prompt': f'question {i}', 'answer': f'answer {i % 4}',
                 'example_id': f'{prefix}{i}'} for i in range(n)]
        raw = s.score([r['prompt'] for r in rows], [r['answer'] for r in rows])['raw']
        for r, value in zip(rows, raw):
            r['teacher_z'] = float((value - self.cal['proxy_mean']) / self.cal['proxy_std'] + .6)
            r['judge_z'] = r['teacher_z'] - .1
        return rows

    def test_initial_proxy_parity_full_answer_guard_and_trainable_parameter_scope(self):
        from reward_bridge import RewardScorer
        original = RewardScorer(self.snapshot, batch_size=2, max_length=128, device='cpu')
        student = StudentScorer.create(self.snapshot, self.cal, self.c, self.o, 123, 'cpu')
        prompts, answers = ['question 1', 'question 2'], ['answer 1', 'answer 2']
        np.testing.assert_allclose(student.score(prompts, answers)['raw'], original.score(prompts, answers)['raw'], atol=1e-7)
        names = [n for n, p in student.model.named_parameters() if p.requires_grad]
        self.assertTrue(any('.score.modules_to_save.' in n for n in names))
        self.assertTrue(all('lora_' in n or '.score.modules_to_save.' in n for n in names))
        with self.assertRaisesRegex(ValueError, 'No truncation'):
            student.score(['question'], ['answer ' * 200])

    def test_actual_regression_save_load_and_exact_optimizer_resume(self):
        train, validation = self.examples('train'), self.examples('validation', 6)
        student = StudentScorer.create(self.snapshot, self.cal, self.c, self.o, self.o['student_seed'] + 42, 'cpu')
        initial = fit.validation_loss(student, validation, 'teacher_z', 2)
        destination = self.root / 'full_fit'
        selected = fit.train_student(destination, destination, self.snapshot, self.cal, self.c, self.o,
                                     42, 'student', train, validation, 'frozen_teacher', self.base_meta, 'cpu')
        reloaded = StudentScorer.load(self.snapshot, selected, device='cpu')
        after = fit.validation_loss(reloaded, validation, 'teacher_z', 2)
        self.assertLess(after, initial)
        self.assertAlmostEqual(after, verify_bundle(selected)['validation_mse'], places=7)
        self.assertFalse(any(p.requires_grad for p in reloaded.model.parameters()))
        self.assertFalse(verify_bundle(selected)['knn_lookup_required'])
        from common import StopRequested
        paused = self.root / 'paused_fit'
        stop = {'value': False}
        real_status = fit.status
        def observer(out, stage, **kw):
            real_status(out, stage, **kw)
            if kw.get('optimizer_steps') == 2:
                stop['value'] = True
        with patch.object(fit, 'should_stop', lambda out: stop['value']), patch.object(fit, 'status', observer):
            with self.assertRaises(StopRequested):
                fit.train_student(paused, paused, self.snapshot, self.cal, self.c, self.o,
                                  42, 'student', train, validation, 'frozen_teacher', self.base_meta, 'cpu')
        recovered = fit.train_student(paused, paused, self.snapshot, self.cal, self.c, self.o,
                                      42, 'student', train, validation, 'frozen_teacher', self.base_meta, 'cpu')
        a, b = load_file(str(selected / 'adapter_model.safetensors')), load_file(str(recovered / 'adapter_model.safetensors'))
        self.assertEqual(a.keys(), b.keys())
        for key in a:
            torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)
        without_times = lambda history: [{k: v for k, v in r.items() if k != 'optimizer_seconds'} for r in history]
        self.assertEqual(without_times(io.read(destination / 'history.json')), without_times(io.read(paused / 'history.json')))

    def test_real_ppo_all_three_routes_from_identical_checkpoint(self):
        import common
        from ppo_engine import PPOActor, PPOTrainer
        from reward_bridge import RewardScorer
        from knn_core import unit_vectors
        from knn_distillation.ppo_training import train_to
        train, validation = self.examples('ppo_train', 8), self.examples('ppo_validation', 4)
        folder = self.root / 'ppo_student_fit'
        bundle = fit.train_student(folder, folder, self.snapshot, self.cal, self.c, {**self.o, 'student_epochs': 1},
                                   42, 'student', train, validation, 'teacher', self.base_meta, 'cpu')
        student = StudentScorer.load(self.snapshot, bundle, device='cpu')
        proxy = RewardScorer(self.snapshot, batch_size=2, max_length=128, device='cpu')
        class ForbiddenJudge:
            def score(self, *args, **kwargs):
                raise AssertionError('Large judge called during PPO')
        router = RewardRouter(proxy, ForbiddenJudge(), self.cal, threads=1)
        rng = np.random.default_rng(4)
        router.memory = {'vectors': unit_vectors(rng.normal(size=(40, 32))), 'gaps': rng.normal(size=40)}
        router.memory_hash = 'teacher_memory'; router.set_students({'student': student})
        c = json.loads((common.ROOT / 'config.json').read_text())
        c.update(max_new_tokens=8, max_prompt_tokens=64, reward_max_tokens=128, rollout_batch_size=4,
                 generation_batch_size=4, mini_batch_size=2, micro_batch_size=2, ppo_epochs=1,
                 lora_rank=2, lora_alpha=4, checkpoint_every=1, cpu_threads=1)
        actor = PPOActor.load(self.policy, c, 42, device='cpu'); trainer = PPOTrainer(actor, router, c)
        checkpoint = trainer.checkpoint(self.root / 'ppo_initial', 0, 'initial_id', 42, 'initial', [])
        parent = io.checkpoint_info(checkpoint); fingerprint = io.state_fingerprint(trainer)
        reference_head = proxy.model.score.weight.detach().clone()
        reference_student = {k: v.detach().clone() for k, v in student.model.named_parameters()}
        with patch('run_study.load_actor', lambda snapshot, config, seed: PPOActor.load(snapshot, config, seed, device='cpu')):
            for branch in ('proxy', 'knn', 'student'):
                dest = self.root / 'ppo_runs' / branch
                done = train_to(self.root, dest, c, {'policy': str(self.policy)}, router, train[:4], 42,
                                branch, parent, 'experiment', 1, 1, fingerprint)
                self.assertEqual(done['update'], 1)
                history = io.read(dest / 'history.json')
                self.assertGreater(history[-1]['optimizer_steps'], 0)
                self.assertEqual(history[-1]['model_calls']['teacher_answers'], 0)
                if branch == 'student':
                    self.assertIsNone(history[-1]['mean_proxy_z'])
                    self.assertEqual(history[-1]['model_calls']['proxy_answers'], 0)
                    self.assertEqual(history[-1]['model_calls']['knn_queries'], 0)
                    self.assertEqual(history[-1]['model_calls']['student_answers'], 4)
        torch.testing.assert_close(proxy.model.score.weight, reference_head, atol=0, rtol=0)
        for name, value in student.model.named_parameters():
            torch.testing.assert_close(value, reference_student[name], atol=0, rtol=0)

    def test_complete_pipeline_with_optional_judge_student_reports_and_export(self):
        import common
        from ppo_engine import PPOActor, PPOTrainer
        from reward_bridge import RewardScorer
        from knn_core import unit_vectors
        from knn_distillation import run
        from knn_distillation.data import group
        c = json.loads((common.ROOT / 'config.json').read_text())
        c.update(seeds=[42], max_new_tokens=8, max_prompt_tokens=64, reward_max_tokens=128, rollout_batch_size=4,
                 generation_batch_size=4, reward_batch_size=2, mini_batch_size=2, micro_batch_size=2, ppo_epochs=1,
                 lora_rank=2, lora_alpha=4, checkpoint_every=1, cpu_threads=1, bootstrap_draws=10,
                 review_pairs_per_stratum=1)
        o = {**self.o, 'student_epochs': 1, 'ppo_updates': 1, 'monitor_every': 1,
             'include_direct_judge_student': True, 'latency_examples': 2, 'latency_repeats': 1}
        out = self.root / 'complete_pipeline'; out.mkdir()
        (out / 'review_form.html').write_text('<script>__PAIR_DATA__</script>')
        proxy = RewardScorer(self.snapshot, batch_size=2, max_length=128, device='cpu')
        judge = RewardScorer(self.snapshot, batch_size=2, max_length=128, device='cpu')
        router = RewardRouter(proxy, judge, self.cal, threads=1)
        memory = out / 'source_memory'; memory.mkdir()
        rng = np.random.default_rng(5)
        np.savez_compressed(memory / 'refreshed_memory.npz', vectors=unit_vectors(rng.normal(size=(40, 32))), gaps=rng.normal(size=40))
        io.write(memory / 'locked_reward.json', {'k': 31, 'temperature': .05, 'calibration': self.cal,
                 'memory_sha256': io.sha(memory / 'refreshed_memory.npz')})
        router.load_memory(memory)
        actor = PPOActor.load(self.policy, c, 42, device='cpu'); trainer = PPOTrainer(actor, router, c)
        parent = io.checkpoint_info(trainer.checkpoint(out / 'source', 0, 'parent', 42, 'initial', []))
        fingerprint = io.state_fingerprint(trainer)
        def cohort(prefix, n):
            return [{'prompt_id': f'{prefix}{i}', 'prompt': f'question {i} {prefix}',
                     'conversation_group': group(f'question {i} {prefix}')} for i in range(n)]
        data = {'distill_train': cohort('train', 4), 'distill_validation': cohort('validation', 2),
                'distill_offline': cohort('offline', 2), 'monitor': cohort('monitor', 2), 'final': cohort('final', 4)}
        assets = {'proxy': str(self.snapshot), 'judge': str(self.snapshot), 'policy': str(self.policy)}
        original_fit = fit.train_student
        events = []
        real_evaluate = run.evaluate_checkpoint
        def evaluate(*args, **kwargs):
            events.append(args[-1])
            return real_evaluate(*args, **kwargs)
        with patch('run_study.load_actor', lambda snapshot, config, seed: PPOActor.load(snapshot, config, seed, device='cpu')), \
             patch('knn_distillation.fit.train_student', lambda *a, **kw: original_fit(*a, **kw, device='cpu')), \
             patch.object(run, 'load_students', lambda assets, c, paths: {k: StudentScorer.load(assets['proxy'], p, c['reward_batch_size'], 'cpu') for k, p in paths.items()}), \
             patch('knn_distillation.reports.monitoring_report'), patch('review.ROOT', out), \
             patch.object(run, 'evaluate_checkpoint', evaluate):
            bundles = run.execute(out, c, o, assets, router, data, 'end_to_end', {42: parent}, {42: memory}, {42: fingerprint}, self.base_meta)
        self.assertEqual(set(bundles[42]), {'student', 'judge_student'})
        self.assertTrue(io.read(out / 'final_lock.json')['all_training_complete'])
        self.assertEqual(events[-5:], ['final'] * 5)
        self.assertTrue((out / 'reports/paired_deltas_by_seed.csv').is_file())
        self.assertTrue((out / 'reports/post_ppo_student_teacher_fidelity.json').is_file())
        self.assertTrue((out / 'review/final_blinded_review/final_blinded_review_BLINDED.zip').is_file())
        archive = run.export(out, common.ROOT, bundles)
        import zipfile
        with zipfile.ZipFile(archive) as z:
            self.assertIsNone(z.testzip())
            self.assertIn('results/selected_students.json', z.namelist())
        with zipfile.ZipFile(out / 'distilled_reward_adapters.zip') as z:
            self.assertIn('students/seed_42/student/adapter_model.safetensors', z.namelist())
            self.assertIn('knn_distillation/score_student.py', z.namelist())
            self.assertFalse(any('refreshed_memory.npz' in n for n in z.namelist()))


if __name__ == '__main__':
    unittest.main(verbosity=2)
