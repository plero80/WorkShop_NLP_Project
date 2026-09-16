"""Frozen proxy+kNN -> proxy student, then a matched PPO comparison."""
from pathlib import Path
import argparse
import fcntl
import signal
import sys
import traceback
import zipfile
from knn_distillation.io import (ROOT, read, write, sha, digest, sealed, check_config, inspect_source,
                                state_fingerprint, pause, should_stop, status)


def adapter_digest(model):
    import hashlib
    from peft import get_peft_model_state_dict
    h = hashlib.sha256()
    for name, tensor in sorted(get_peft_model_state_dict(model).items()):
        h.update(name.encode()); h.update(tensor.detach().float().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def preflight(out, c, o, assets, router, parents, memories, data, base_metadata):
    import numpy as np
    import torch
    from run_study import load_actor, release
    from ppo_engine import PPOTrainer
    from knn_distillation.student import StudentScorer, save_bundle
    from knn_distillation.fit import make_optimizer
    fingerprints = {}
    for seed in c['seeds']:
        actor = load_actor(assets['policy'], c, seed); trainer = PPOTrainer(actor, None, c)
        try:
            ck = parents[seed]
            step, _ = trainer.restore(ck['path'], ck['identity'], seed, ck['branch'], optimizer=True)
            if step != ck['update']:
                raise ValueError('PPO source checkpoint update mismatch.')
            fingerprints[seed] = state_fingerprint(trainer)
        finally:
            del trainer, actor; release()
    seed = c['seeds'][0]; router.load_memory(memories[seed])
    prompts = [r['prompt'] for r in data['distill_train'][:4]]
    answers = ['A short response for the discarded preflight test.'] * len(prompts)
    reference = router.proxy_values(prompts, answers)['raw']
    teacher_before = {n: p.detach().cpu().clone() for n, p in router.proxy.model.named_parameters() if n.endswith('score.weight')}
    student = StudentScorer.create(assets['proxy'], router.calibration, c, o, o['student_seed'] + seed)
    optimizer = make_optimizer(student, o)
    try:
        initial = student.score(prompts, answers)['raw']
        error = abs(initial - reference)
        if error.mean() > .03 or error.max() > .3:
            raise ValueError('Untrained student does not reproduce its proxy initialization.')
        before = adapter_digest(student.model)
        student.model.train(); optimizer.zero_grad(set_to_none=True)
        z = student.normalized_forward(prompts, answers)
        # Deliberate nonzero synthetic target offset: this optimizer update is discarded.
        (z - (z.detach() + .25)).square().mean().backward()
        torch.nn.utils.clip_grad_norm_([p for p in student.model.parameters() if p.requires_grad], 1.)
        optimizer.step()
        if before == adapter_digest(student.model):
            raise ValueError('Student gradient smoke update changed no trainable weights.')
        if any(p.requires_grad for p in router.proxy.model.parameters()):
            raise ValueError('Teacher proxy was unfrozen.')
        for name, tensor in teacher_before.items():
            torch.testing.assert_close(dict(router.proxy.model.named_parameters())[name].detach().cpu(), tensor, atol=0, rtol=0)
        student.model.requires_grad_(False).eval()
        expected = student.score(prompts, answers)['raw']
        smoke_bundle = out / 'preflight_artifacts/student'
        save_bundle(student, smoke_bundle, {**base_metadata, 'purpose': 'discarded preflight student'})
    finally:
        del student, optimizer; release()
    student = StudentScorer.load(assets['proxy'], smoke_bundle, c['reward_batch_size'])
    try:
        actual = student.score(prompts, answers)['raw']
        np.testing.assert_allclose(actual, expected, atol=.03, rtol=.005)
        router.set_students({'student': student})
        smoke = {**c, 'max_new_tokens': 16, 'generation_batch_size': 4, 'mini_batch_size': 4,
                 'micro_batch_size': 2, 'ppo_epochs': 1}
        actor = load_actor(assets['policy'], smoke, seed); trainer = PPOTrainer(actor, router, smoke)
        try:
            before = state_fingerprint(trainer); router.reset_cost()
            rec = trainer.update(data['distill_train'][:4], 'student', seed, 1)
            if state_fingerprint(trainer) == before:
                raise ValueError('Student-reward PPO smoke update changed no policy weights.')
            if any(router.cost[k] for k in ('proxy_answers', 'teacher_answers', 'knn_queries')):
                raise RuntimeError('Student PPO scoring used the teacher or memory.')
            ppo = {'optimizer_steps': rec['optimizer_steps'], 'model_calls': router.cost.copy()}
        finally:
            del trainer, actor; release()
    finally:
        router.set_students({}); del student; release()
    write(out / 'preflight.json', {'passed': True, 'source_state_fingerprints': fingerprints,
          'untrained_student_proxy_max_abs_error': float(error.max()), 'student_adapter_save_load_checked': True,
          'discarded_student_gradient_step': True, 'discarded_student_reward_PPO_step': ppo})
    return fingerprints


def load_students(assets, c, paths):
    from knn_distillation.student import StudentScorer
    return {kind: StudentScorer.load(assets['proxy'], path, c['reward_batch_size']) for kind, path in paths.items()}


def evaluate_checkpoint(out, c, assets, router, rows, seed, branch, ck, identity, cohort):
    from run_study import checkpoint_actor, release
    from knn_distillation.policy_eval import evaluate
    folder = out / 'evaluations' / cohort / f'seed_{seed}' / branch / f'update_{ck["update"]:06d}'
    actor = checkpoint_actor(assets['policy'], c, seed, ck['path'], ck['identity'], ck['branch'])
    try:
        evaluate(actor, router, rows, folder, c, identity, seed, branch, ck['sha256'], cohort,
                 ck['update'], features=False, monitor_kl=True, stop_out=out)
    finally:
        del actor; release()


def execute(out, c, o, assets, router, data, identity, parents, memories, fingerprints, base_metadata):
    from run_study import release
    from knn_distillation.labels import generate_labels, load_labels
    from knn_distillation.fit import train_student
    from knn_distillation.offline import score_offline, latency_benchmark
    from knn_distillation.data import schedule
    from knn_distillation.ppo_training import train_to
    from knn_distillation.reports import monitoring_report, final_report, frames
    kinds = ['student'] + (['judge_student'] if o['include_direct_judge_student'] else [])
    bundles = {}
    for seed in c['seeds']:
        router.load_memory(memories[seed]); parent = parents[seed]
        teacher_id = digest({'memory': router.memory_hash, 'calibration': router.calibration, 'base': base_metadata})
        train = load_labels(generate_labels(out, c, o, assets, router, data['distill_train'], seed, parent,
                                            identity, 'train', o['include_direct_judge_student']))
        validation = load_labels(generate_labels(out, c, o, assets, router, data['distill_validation'], seed, parent,
                                                 identity, 'validation', o['include_direct_judge_student']))
        bundles[seed] = {}
        for kind in kinds:
            bundles[seed][kind] = train_student(out, out / 'students' / f'seed_{seed}' / kind,
                assets['proxy'], router.calibration, c, o, seed, kind, train, validation, teacher_id, base_metadata)
    sealed(out / 'student_selection_lock.json', {'all_students_selected': True,
           'bundles': {f'{s}/{k}': sha(path / 'reward_config.json') for s, paths in bundles.items() for k, path in paths.items()},
           'selection': 'Lowest validation MSE, never offline or final labels.'})
    endpoints = {}
    branches = ['proxy', 'knn', *kinds]
    for seed in c['seeds']:
        router.load_memory(memories[seed]); parent = parents[seed]
        offline = load_labels(generate_labels(out, c, o, assets, router, data['distill_offline'], seed, parent,
                                              identity, 'offline', True))
        students = load_students(assets, c, bundles[seed]); router.set_students(students)
        try:
            score_offline(out, seed, students, offline, c['reward_batch_size'])
            latency_benchmark(out, seed, router, offline, o, bundles[seed], memories[seed])
            evaluate_checkpoint(out, c, assets, router, data['monitor'], seed, 'initial', parent, identity, 'monitor')
            rows = schedule(data['distill_train'], o['ppo_updates'], c['rollout_batch_size'], o['data_seed'] + seed)
            order = branches[seed % len(branches):] + branches[:seed % len(branches)]
            for branch in order:
                for delta in range(o['monitor_every'], o['ppo_updates'] + 1, o['monitor_every']):
                    ck = train_to(out, out / 'runs' / f'seed_{seed}' / branch, c, assets, router, rows,
                                  seed, branch, parent, identity, parent['update'] + delta, o['monitor_every'], fingerprints[seed])
                    evaluate_checkpoint(out, c, assets, router, data['monitor'], seed, branch, ck, identity, 'monitor')
                    monitoring_report(out, 'monitor')
                endpoints[(seed, branch)] = ck
        finally:
            router.set_students({}); del students; release()
    sealed(out / 'final_lock.json', {'all_training_complete': True, 'test_prompts': digest(data['final']),
           'endpoints': {f'{s}/{b}': ck['sha256'] for (s, b), ck in endpoints.items()},
           'primary_comparison': 'student minus knn', 'primary_update': next(iter(parents.values()))['update'] + o['ppo_updates']})
    for seed in c['seeds']:
        router.load_memory(memories[seed]); students = load_students(assets, c, bundles[seed]); router.set_students(students)
        try:
            evaluate_checkpoint(out, c, assets, router, data['final'], seed, 'initial', parents[seed], identity, 'final')
            for branch in branches:
                evaluate_checkpoint(out, c, assets, router, data['final'], seed, branch, endpoints[(seed, branch)], identity, 'final')
        finally:
            router.set_students({}); del students; release()
    final_report(out, c, 'final', 'knn', ['proxy', *kinds], o)
    from knn_distillation.offline import regression_metrics
    fidelity = []
    for (seed, branch, update), frame in frames(out, 'final').items():
        for kind in kinds:
            fidelity.append({'seed': seed, 'policy_branch': branch, 'update': update, 'reward_student': kind,
                             **regression_metrics(frame[kind + '_z'], frame.knn_z)})
    write(out / 'reports/post_ppo_student_teacher_fidelity.json', {'records': fidelity})
    write(out / 'selected_students.json', {str(s): {k: str(p.relative_to(out)) for k, p in paths.items()} for s, paths in bundles.items()})
    write(out / 'endpoints.json', {f'{s}/{b}': {**ck, 'checkpoint_relative_path': str(Path(ck['path']).relative_to(out))}
                                 for (s, b), ck in endpoints.items()})
    return bundles


def export(out, project, bundles=None):
    archive = out / 'important_outcomes_knn_distillation.zip'; temp = archive.with_suffix('.pending')
    with zipfile.ZipFile(temp, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if (not p.is_file() or p.suffix in ('.pt', '.pending', '.safetensors', '.zip') or
                any(part in p.parts for part in ('recovery', 'checkpoints', 'preflight_artifacts')) or
                p.name in ('PAUSE', 'training.log')):
                continue
            z.write(p, 'results/' + str(p.relative_to(out)))
        for p in sorted(ROOT.glob('*')):
            if p.is_file():
                z.write(p, 'code/knn_distillation/' + p.name)
    temp.replace(archive)
    if bundles is not None:
        deployed = out / 'distilled_reward_adapters.zip'; pending = deployed.with_suffix('.pending')
        with zipfile.ZipFile(pending, 'w', zipfile.ZIP_DEFLATED) as z:
            for seed, paths in bundles.items():
                for kind, folder in paths.items():
                    for p in sorted(folder.rglob('*')):
                        if p.is_file():
                            z.write(p, f'students/seed_{seed}/{kind}/' + str(p.relative_to(folder)))
            for name in ('__init__.py', 'student.py', 'io.py', 'score_student.py'):
                z.write(ROOT / name, 'knn_distillation/' + name)
            z.write(project / 'chat_format.py', 'chat_format.py')
            z.writestr('README.txt', 'These are selected reward adapters, scalar heads and tokenizers. Supply the exact pinned proxy base checkpoint named in reward_config.json. No teacher memory, judge, policy checkpoint or original proxy forward is needed. Use python -m knn_distillation.score_student --help.\n')
        pending.replace(deployed)
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path.cwd())
    parser.add_argument('--config', type=Path, default=ROOT / 'settings.json')
    parser.add_argument('--followup', type=Path); parser.add_argument('--refresh2', type=Path); parser.add_argument('--refresh34', type=Path)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args(); o = read(args.config); check_config(o)
    project, refresh2, source, source_manifest, parent_config, parents, memories = inspect_source(
        args.project, o, args.followup, args.refresh2, args.refresh34)
    sys.path.insert(0, str(project))
    from common import runtime_versions, StopRequested
    from run_study import configure, load_rewards
    from assets import resolve_all, MODELS
    from knn_distillation.data import prepare
    from knn_distillation.reward import RewardRouter
    versions = runtime_versions()
    for key in ('torch', 'transformers', 'peft'):
        if versions[key] != source_manifest['runtime_versions'][key]:
            raise ValueError(f'Use original {key}={source_manifest["runtime_versions"][key]}; found {versions[key]}.')
    c = {**parent_config, 'seeds': o['seeds'], 'allow_downloads': o['allow_downloads'], 'extra_hf_cache': o['extra_hf_cache'],
         'eval_seed': 2026091233, 'review_seed': 2026091234, 'review_pairs_per_stratum': o['review_pairs_per_seed']}
    if c['max_new_tokens'] != 256 or c['reward_max_tokens'] < 4096:
        raise ValueError('Preserve the 256-token answer/full-answer reward protocol.')
    scientific = lambda values: {k: v for k, v in values.items() if k not in ('allow_downloads', 'extra_hf_cache', 'max_wall_hours')}
    record = {'protocol': 'Frozen proxy+kNN distillation followed by matched frozen-reward PPO',
              'source_identity': source_manifest['identity'], 'options': scientific(o), 'config': scientific(c),
              'runtime_versions': versions, 'parents': {str(s): {k: v for k, v in p.items() if k != 'path'} for s, p in parents.items()},
              'memories': {str(s): sha(p / 'locked_reward.json') for s, p in memories.items()},
              'source_sha256': {p.name: sha(p) for p in sorted(ROOT.glob('*.py'))}}
    identity = digest(record); out = project / 'knn_distillation_outputs' / ('study_' + identity[:16])
    lock = open(project / 'outputs/runner.lock', 'a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close(); raise SystemExit('Another GPU study is running; run experiments sequentially.')
    try:
        out.mkdir(parents=True, exist_ok=True); sealed(out / 'manifest.json', {'identity': identity, **record})
        write(project / 'knn_distillation_outputs/latest.json', {'output': str(out)})
        if ((out / 'status.json').exists() and read(out / 'status.json')['stage'] == 'complete' and
                (out / 'important_outcomes_knn_distillation.zip').exists() and (out / 'distilled_reward_adapters.zip').exists() and not args.preflight):
            print('Already complete:', out, flush=True); return
        (out / 'PAUSE').unlink(missing_ok=True)
        signal.signal(signal.SIGTERM, pause); signal.signal(signal.SIGINT, pause)
        status(out, 'preflight'); configure(c); assets = resolve_all(c)
        data = prepare(project, refresh2, out, c, o, assets)
        frozen, judge = load_rewards(c, assets)
        write(out / 'encoder_parity.json', frozen.validate_memory_encoder())
        router = RewardRouter(frozen.scorer, judge, frozen.calibration, c['cpu_threads'])
        base_metadata = {'base_model': MODELS['proxy'][0], 'base_revision': MODELS['proxy'][1],
                         'base_config_sha256': sha(Path(assets['proxy']) / 'config.json'),
                         'base_weight_sha256': {p.name: sha(p) for p in sorted(Path(assets['proxy']).glob('model*.safetensors'))}}
        fingerprints = preflight(out, c, o, assets, router, parents, memories, data, base_metadata)
        if args.preflight:
            status(out, 'ready'); print('PREFLIGHT PASSED:', out, flush=True); return
        if should_stop(out):
            raise StopRequested('Pause requested during preflight.')
        bundles = execute(out, c, o, assets, router, data, identity, parents, memories, fingerprints, base_metadata)
        status(out, 'complete', human_review='awaiting manual ratings')
        print('RESULTS:', export(out, project, bundles), flush=True)
        print('STUDENTS:', out / 'distilled_reward_adapters.zip', flush=True)
    except StopRequested as error:
        status(out, 'paused', reason=str(error)); export(out, project); print('PAUSED:', error, flush=True)
    except Exception as error:
        status(out, 'failed', error_type=type(error).__name__, message=str(error)); traceback.print_exc(); raise
    finally:
        lock.close()


if __name__ == '__main__':
    main()
