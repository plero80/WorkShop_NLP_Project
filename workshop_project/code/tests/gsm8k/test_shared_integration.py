"""Integration boundaries absent from the standalone ZIP tests."""
import copy
import pytest
import torch
from test_tiny_models import tiny_assets, items
from gsm8k_experiment.common import DEFAULT_CONFIG, atomic_json, load_config, run_lock
from gsm8k_experiment.models import Policy
from gsm8k_experiment.ppo import optimizer_for, prepare_rollout, update, save_checkpoint, load_checkpoint
from gsm8k_experiment.recovery import validate_migration
from gsm8k_experiment.shared import ENGINE_ID, flat_config, shared_sources
from ppo_engine import PPOActor, PPOTrainer


def test_existing_sources_and_fixed_sampling_distribution():
    assert 'ppo_engine.py' in shared_sources()
    config = copy.deepcopy(load_config(DEFAULT_CONFIG))
    config['generation']['temperature'] = .7
    with pytest.raises(ValueError, match='temperature'):
        flat_config(config)


def test_actual_update_routes_through_shared_trainer(tiny_assets, monkeypatch):
    config, resolved = tiny_assets
    actor = Policy(config, resolved)
    assert isinstance(actor, PPOActor)
    assert type(actor).statistics is PPOActor.statistics
    calls = []
    original = PPOTrainer.update
    def tracked(self, rows, branch, seed, update):
        calls.append((len(rows), update))
        return original(self, rows, branch, seed, update)
    monkeypatch.setattr(PPOTrainer, 'update', tracked)
    result = update(actor, optimizer_for(actor, config),
                    prepare_rollout(actor, items(), [1., -1.], config), config, 0)
    assert calls == [(2, 1)]
    assert result['engine'] == ENGINE_ID


def test_checkpoint_corruption_and_foreign_identity_rejected(tiny_assets, tmp_path):
    config, resolved = tiny_assets
    actor = Policy(config, resolved)
    optimizer = optimizer_for(actor, config)
    path = tmp_path / 'checkpoint.pt'
    save_checkpoint(path, actor, optimizer, 0, 'identity', 'proxy')
    with pytest.raises(ValueError, match='another experiment'):
        load_checkpoint(path, actor, optimizer, 'foreign', 'proxy')
    with pytest.raises(ValueError, match='ancestry'):
        load_checkpoint(path, actor, optimizer, 'identity', 'proxy', accepted_parents=('old',))
    with path.open('ab') as stream:
        stream.write(b'corruption')
    with pytest.raises(ValueError, match='checksum'):
        load_checkpoint(path, actor, optimizer, 'identity', 'proxy')


def test_old_zip_migrations_rejected(tmp_path):
    atomic_json(tmp_path / 'recovery/grading_retry_v1/migration.json', {'status': 'ready'})
    with pytest.raises(ValueError, match='Standalone ZIP'):
        validate_migration(tmp_path, {})


def test_output_lock_excludes_concurrent_writer(tmp_path):
    with run_lock(tmp_path):
        with pytest.raises(RuntimeError, match='already holds'):
            with run_lock(tmp_path):
                pass
    with run_lock(tmp_path):
        pass
