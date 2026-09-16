"""Normalized score regression with exact optimizer/cursor resume."""
from pathlib import Path
import math
import time
import uuid
import numpy as np
import torch
from knn_distillation.io import read, write, sha, digest, sealed, should_stop, status
from knn_distillation.student import StudentScorer, save_bundle, verify_bundle


def target_values(rows, target_key):
    values = np.asarray([r[target_key] for r in rows], float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('Student targets must be nonempty and finite.')
    return values


def make_optimizer(student, o):
    head, adapter = [], []
    for name, p in student.model.named_parameters():
        if p.requires_grad:
            (head if '.score.modules_to_save.' in name else adapter).append(p)
    if not head or not adapter:
        raise ValueError('Expected trainable LoRA adapters and a scalar head.')
    return torch.optim.AdamW([{'params': adapter, 'lr': o['student_lr']},
                             {'params': head, 'lr': o['student_head_lr']}], weight_decay=0., eps=1e-8)


def save_state(folder, student, optimizer, payload):
    from peft import get_peft_model_state_dict
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    path = folder / f'state_{payload["steps"]:06d}_{uuid.uuid4().hex}.pt'
    state = {**payload, 'adapter': {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(student.model).items()},
             'optimizer': optimizer.state_dict(), 'torch_rng': torch.get_rng_state(),
             'cuda_rng': torch.cuda.get_rng_state_all() if student.device.startswith('cuda') else None}
    pending = path.with_suffix('.pending')
    torch.save(state, pending)
    with open(pending, 'rb') as f:
        import os
        os.fsync(f.fileno())
    pending.replace(path)
    meta = {'name': path.name, 'sha256': sha(path), 'identity': payload['identity'], 'steps': payload['steps']}
    write(path.with_suffix('.json'), meta)
    previous = read(folder / 'latest.json')['name'] if (folder / 'latest.json').exists() else None
    write(folder / 'latest.json', meta)
    keep = {path.name, previous}
    for p in folder.glob('state_*.json'):
        m = read(p)
        if m['name'] not in keep:
            (folder / m['name']).unlink(missing_ok=True); p.unlink()


def restore_state(folder, student, optimizer, identity):
    from peft import set_peft_model_state_dict
    meta = read(Path(folder) / 'latest.json'); path = Path(folder) / meta['name']
    if meta['identity'] != identity or sha(path) != meta['sha256']:
        raise ValueError('Student recovery checkpoint identity/hash mismatch.')
    state = torch.load(path, map_location=student.device, weights_only=True)
    if state['identity'] != identity:
        raise ValueError('Student checkpoint payload identity mismatch.')
    set_peft_model_state_dict(student.model, state['adapter'])
    optimizer.load_state_dict(state['optimizer'])
    torch.set_rng_state(state['torch_rng'].cpu())
    if state['cuda_rng'] is not None:
        torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda_rng']])
    return {k: v for k, v in state.items() if k not in ('adapter', 'optimizer', 'torch_rng', 'cuda_rng')}


@torch.inference_mode()
def validation_loss(student, rows, target_key, batch_size):
    student.model.eval(); loss_sum = 0.
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        pred = student.normalized_forward([r['prompt'] for r in batch], [r['answer'] for r in batch])
        target = torch.as_tensor(target_values(batch, target_key), dtype=torch.float32, device=student.device)
        loss_sum += float(((pred - target) ** 2).sum())
    return loss_sum / len(rows)


def train_student(out, folder, snapshot, calibration, c, o, seed, kind, train, validation,
                  teacher_identity, base_metadata, device='cuda'):
    from common import StopRequested
    from run_study import release
    target_key = 'teacher_z' if kind == 'student' else 'judge_z'
    target_values(train, target_key); target_values(validation, target_key)
    if {r['prompt_id'] for r in train} & {r['prompt_id'] for r in validation}:
        raise ValueError('Student train/validation prompts overlap.')
    options = {k: v for k, v in o.items() if k.startswith('student_') or k == 'gradient_checkpointing'}
    signature = {'teacher_identity': teacher_identity, 'base': base_metadata, 'seed': seed, 'kind': kind,
                 'calibration': calibration, 'options': options, 'train': digest(train), 'validation': digest(validation),
                 'target': target_key, 'loss': 'MSE on fixed proxy-normalized student score'}
    identity = digest(signature); folder = Path(folder)
    sealed(folder / 'manifest.json', {'identity': identity, **signature})
    if (folder / 'complete.json').exists():
        done = read(folder / 'complete.json')
        if done['identity'] != identity:
            raise ValueError('Completed student dependencies changed.')
        bundle = folder / done['selected_bundle']
        if digest(verify_bundle(bundle)) != done['bundle_identity']:
            raise ValueError('Selected student changed.')
        return bundle
    student = StudentScorer.create(snapshot, calibration, c, o, o['student_seed'] + seed, device)
    optimizer = make_optimizer(student, o)
    checkpoint_dir = folder / 'recovery'
    state = {'identity': identity, 'epoch': 0, 'offset': 0, 'steps': 0, 'best_mse': None,
             'best_epoch': None, 'history': [], 'validations': []}
    if (checkpoint_dir / 'latest.json').exists():
        state = restore_state(checkpoint_dir, student, optimizer, identity)
    started = time.monotonic()
    try:
        effective = o['student_batch_size'] * o['student_accumulation']
        while state['epoch'] < o['student_epochs']:
            epoch = state['epoch']
            order = np.random.default_rng(o['student_seed'] + seed + epoch).permutation(len(train))
            while state['offset'] < len(train):
                if should_stop(out):
                    save_state(checkpoint_dir, student, optimizer, state)
                    raise StopRequested('Paused at student optimizer boundary.')
                selected = order[state['offset']:state['offset'] + effective]
                step_started = time.monotonic()
                optimizer.zero_grad(set_to_none=True)
                student.model.train(); loss_sum = 0.
                for start in range(0, len(selected), o['student_batch_size']):
                    batch = [train[int(i)] for i in selected[start:start + o['student_batch_size']]]
                    pred = student.normalized_forward([r['prompt'] for r in batch], [r['answer'] for r in batch])
                    target = torch.as_tensor(target_values(batch, target_key), dtype=torch.float32, device=device)
                    errors = (pred - target).square()
                    if not torch.isfinite(errors).all():
                        raise FloatingPointError('Nonfinite distillation loss.')
                    # Correct denominator also handles the final partial accumulated batch.
                    (errors.sum() / len(selected)).backward()
                    loss_sum += float(errors.detach().sum())
                norm = torch.nn.utils.clip_grad_norm_([p for p in student.model.parameters() if p.requires_grad], o['student_max_grad_norm'])
                if not torch.isfinite(norm):
                    raise FloatingPointError('Nonfinite student gradients.')
                optimizer.step()
                state['steps'] += 1; state['offset'] += len(selected)
                state['history'].append({'step': state['steps'], 'epoch': epoch + 1, 'examples': len(selected),
                                         'mse': loss_sum / len(selected), 'gradient_norm': float(norm),
                                         'optimizer_seconds': time.monotonic() - step_started})
                write(folder / 'history.json', state['history'])
                status(out, 'distilling', seed=seed, kind=kind, epoch=epoch + 1, epochs=o['student_epochs'],
                       completed=state['offset'], total=len(train), optimizer_steps=state['steps'])
                if state['steps'] % o['student_checkpoint_every'] == 0 or should_stop(out):
                    save_state(checkpoint_dir, student, optimizer, state)
                if state['steps'] % 10 == 0:
                    print(f'Distill {kind} seed={seed} epoch={epoch + 1} step={state["steps"]} MSE={loss_sum / len(selected):.5f}', flush=True)
            if should_stop(out):
                save_state(checkpoint_dir, student, optimizer, state)
                raise StopRequested('Paused before student validation.')
            mse = validation_loss(student, validation, target_key, o['student_batch_size'])
            if not math.isfinite(mse):
                raise FloatingPointError('Nonfinite validation MSE.')
            state['validations'].append({'epoch': epoch + 1, 'mse': mse})
            if state['best_mse'] is None or mse < state['best_mse']:
                state['best_mse'], state['best_epoch'] = mse, epoch + 1
                bundle = folder / f'epoch_{epoch + 1}'
                save_bundle(student, bundle, {**base_metadata, 'training_identity': identity, 'kind': kind,
                            'teacher_identity': teacher_identity, 'selected_epoch': epoch + 1, 'validation_mse': mse})
            state['epoch'] += 1; state['offset'] = 0
            write(folder / 'validation.json', state['validations'])
            save_state(checkpoint_dir, student, optimizer, state)
            print(f'{kind} seed={seed} epoch={epoch + 1}: validation MSE={mse:.5f}', flush=True)
        bundle = folder / f'epoch_{state["best_epoch"]}'
        write(folder / 'complete.json', {'identity': identity, 'selected_bundle': bundle.name,
              'bundle_identity': digest(verify_bundle(bundle)), 'selected_epoch': state['best_epoch'],
              'validation_mse': state['best_mse'], 'optimizer_steps': state['steps'],
              'fit_examples': len(train), 'validation_examples': len(validation),
              'completed_optimizer_seconds': sum(r['optimizer_seconds'] for r in state['history']),
              'last_invocation_seconds': time.monotonic() - started, 'selection_used_final_or_offline_labels': False})
        return bundle
    finally:
        del student, optimizer
        release()
