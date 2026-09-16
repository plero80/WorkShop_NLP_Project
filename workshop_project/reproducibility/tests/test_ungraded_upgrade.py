"""Resume upgrade preserves scientific artifacts and actual checkpoint state."""
import importlib.util
from pathlib import Path

import pytest
import torch

from test_inline_score_repair import failed_run, repair, PROJECT

spec = importlib.util.spec_from_file_location('ungraded_upgrade', PROJECT / 'reproducibility/upgrade_gsm8k_ungraded.py')
upgrade = importlib.util.module_from_spec(spec)
spec.loader.exec_module(upgrade)


def checkpoint(output, fingerprint, step=7, arm='proxy'):
    patch = repair.read(PROJECT / 'reproducibility/gsm8k_ungraded_patch.json')
    path = output / 'arms' / arm / 'checkpoint.pt'
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {'engine': patch['engine'], 'fingerprint': fingerprint, 'arm': arm, 'step': step,
            'trainable': {'weights': torch.arange(12).reshape(3, 4)},
            'optimizer': {'state': {0: {'exp_avg': torch.arange(12) / 10, 'step': torch.tensor(8)}}},
            'torch_rng': torch.get_rng_state(), 'cuda_rng': None, 'extra': {'last_refresh': 0}}
    torch.save(data, path)
    repair.atomic_json(path.with_suffix('.sha256.json'), {'engine': patch['engine'], 'sha256': repair.file_hash(path)})
    return path, data


def equal_state(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal_state(left[key], right[key])
    else:
        assert left == right


@pytest.mark.parametrize('inline_applied', [False, True])
def test_upgrade_preserves_artifacts_and_checkpoint_state_from_both_versions(failed_run, inline_applied):
    runtime, output, _, parent = failed_run
    if inline_applied:
        repair.repair(runtime, output)
        parent = repair.read(output / 'manifest.json')
    path, original = checkpoint(output, parent['fingerprint'])
    protected = {p: p.read_bytes() for p in output.rglob('*') if p.is_file() and
                 p.name not in ('manifest.json', 'checkpoint.pt', 'checkpoint.sha256.json')}
    before_bytes = path.read_bytes()
    upgrade.upgrade(runtime, output)
    manifest = repair.read(output / 'manifest.json')
    assert manifest['identity']['config'] == parent['identity']['config']
    assert manifest['fingerprint'] != parent['fingerprint']
    actual = torch.load(path, weights_only=True)
    equal_state(actual, dict(original, fingerprint=manifest['fingerprint']))
    assert all(p.read_bytes() == data for p, data in protected.items())
    audit = output / 'source_amendments/ungraded_review_v1'
    assert (audit / 'original_checkpoints/arms/proxy/checkpoint.pt').read_bytes() == before_bytes
    assert repair.read(audit / 'upgrade.json')['parent_manifest'] == parent
    once = {p: p.read_bytes() for p in output.rglob('*') if p.is_file()}
    upgrade.upgrade(runtime, output)
    assert all(p.read_bytes() == data for p, data in once.items())


def test_upgrade_before_training_and_after_subsequent_training_is_idempotent(failed_run):
    runtime, output, *_ = failed_run
    upgrade.upgrade(runtime, output)
    fingerprint = repair.read(output / 'manifest.json')['fingerprint']
    path, state = checkpoint(output, fingerprint, step=25)
    upgrade.upgrade(runtime, output)
    equal_state(torch.load(path, weights_only=True), state)


@pytest.mark.parametrize('target', ['checkpoint.sha256.json', 'models.py'])
def test_interrupted_upgrade_resumes_from_original_checkpoint(failed_run, monkeypatch, target):
    runtime, output, _, parent = failed_run
    path, original = checkpoint(output, parent['fingerprint'])
    atomic_bytes, atomic_json = upgrade.atomic_bytes, upgrade.atomic_json
    def interrupted_bytes(path, data):
        if path.name == target and 'original_checkpoints' not in path.parts:
            raise OSError('simulated interruption')
        atomic_bytes(path, data)
    def interrupted_json(path, value):
        if path.name == target:
            raise OSError('simulated interruption')
        atomic_json(path, value)
    monkeypatch.setattr(upgrade, 'atomic_bytes', interrupted_bytes)
    monkeypatch.setattr(upgrade, 'atomic_json', interrupted_json)
    with pytest.raises(OSError, match='interruption'):
        upgrade.upgrade(runtime, output)
    monkeypatch.setattr(upgrade, 'atomic_bytes', atomic_bytes)
    monkeypatch.setattr(upgrade, 'atomic_json', atomic_json)
    upgrade.upgrade(runtime, output)
    manifest = repair.read(output / 'manifest.json')
    equal_state(torch.load(path, weights_only=True), dict(original, fingerprint=manifest['fingerprint']))
    assert repair.read(path.with_suffix('.sha256.json'))['sha256'] == repair.file_hash(path)
    backup = output / 'source_amendments/ungraded_review_v1/original_checkpoints/arms/proxy/checkpoint.pt'
    assert repair.read(backup.with_suffix('.sha256.json'))['sha256'] == repair.file_hash(backup)


@pytest.mark.parametrize('change', ['source', 'core', 'config', 'checkpoint', 'foreign_checkpoint'])
def test_unrecognized_changes_refused_before_mutation(failed_run, change):
    runtime, output, _, parent = failed_run
    path, state = checkpoint(output, parent['fingerprint'])
    if change == 'source':
        with (runtime / 'gsm8k_experiment/models.py').open('ab') as stream:
            stream.write(b'\n# unexpected change\n')
    elif change == 'core':
        with (runtime / 'ppo_engine.py').open('ab') as stream:
            stream.write(b'\n# unexpected change\n')
    elif change == 'config':
        repair.atomic_json(output / 'config.json', {'seed': -1})
    elif change == 'checkpoint':
        info = repair.read(path.with_suffix('.sha256.json'))
        info['sha256'] = 'wrong'
        repair.atomic_json(path.with_suffix('.sha256.json'), info)
    else:
        checkpoint(output, 'foreign')
    saved = {p: p.read_bytes() for p in runtime.rglob('*') if p.is_file()}
    with pytest.raises(ValueError):
        upgrade.upgrade(runtime, output)
    assert all(p.read_bytes() == data for p, data in saved.items())
    assert not (output / 'source_amendments/ungraded_review_v1/upgrade.json').exists()


def test_running_experiment_cannot_be_upgraded(failed_run):
    runtime, output, *_ = failed_run
    with repair.lock(output):
        with pytest.raises(ValueError, match='still running'):
            upgrade.upgrade(runtime, output)
