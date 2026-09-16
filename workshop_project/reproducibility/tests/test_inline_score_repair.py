"""Repair against the actual source snapshot uploaded before the reported failure."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

import pytest

PROJECT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('inline_repair', PROJECT / 'reproducibility/repair_gsm8k_inline_score.py')
repair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair)


@pytest.fixture
def failed_run(tmp_path):
    runtime = tmp_path / 'runtime'
    package = runtime / 'gsm8k_experiment'
    package.mkdir(parents=True)
    patch = repair.read(PROJECT / 'reproducibility/gsm8k_inline_score_patch.json')
    for name in patch['before']:
        target = package / name
        data = subprocess.check_output(['git', '-c', 'safe.directory=' + PROJECT.parent.as_posix(),
            'show', '5e3983a:workshop_project/code/experiments/gsm8k_experiment/' + name], cwd=PROJECT.parent)
        target.write_bytes(data)
        assert repair.file_hash(target) == patch['before'][name]
    for path in (PROJECT / 'code/core').glob('*.py'):
        shutil.copyfile(path, runtime / path.name)
    shutil.copyfile(PROJECT / 'configs/gsm8k/shared_sources.json', package / 'shared_sources.json')
    output = runtime / 'gsm8k_outputs/b200'
    output.mkdir(parents=True)
    config = repair.read(PROJECT / 'configs/gsm8k/settings.json')
    identity = {'source': patch['source_before'], 'config': config, 'versions': {'test': 'fixture'}}
    manifest = {'identity': identity, 'fingerprint': repair.digest(identity), 'created_at': 10, 'protocol': 'shared PPO'}
    repair.atomic_json(output / 'manifest.json', manifest)
    repair.atomic_json(output / 'config.json', config)
    repair.atomic_json(output / 'status.json', {'stage': 'failed'})
    evidence = {'stage': 'memory', 'length_capped': False, 'question_id': '6469f8eec70b',
                'judge_output': 'Judgement: Correctness_score: 5'}
    (output / 'invalid_judge_outputs.jsonl').write_text(json.dumps(evidence) + '\n')
    for name in ('reward_cache.sqlite', 'generations/memory/000000.json', 'prepared/calibration_raw.jsonl', 'initial_trainable.pt'):
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'preserve this saved artifact exactly')
    return runtime, output, patch, manifest


def test_patch_preserves_data_updates_only_source_identity_and_is_idempotent(failed_run):
    runtime, output, patch, original = failed_run
    saved = {p: p.read_bytes() for p in output.rglob('*') if p.is_file() and p.name != 'manifest.json'}
    repair.repair(runtime, output)
    current = repair.read(output / 'manifest.json')
    assert current['identity'] == dict(original['identity'], source=patch['source_after'])
    assert current['fingerprint'] == repair.digest(current['identity'])
    assert repair.read(output / 'source_amendments/grading_inline_score_v1.json')['parent_manifest'] == original
    assert all(p.read_bytes() == data for p, data in saved.items())
    before = (output / 'manifest.json').read_bytes()
    repair.repair(runtime, output)
    assert (output / 'manifest.json').read_bytes() == before


@pytest.mark.parametrize('change', ['source', 'core', 'training', 'reply', 'capped', 'config'])
def test_refuses_unrecognized_or_unsafe_state_without_patching(failed_run, change):
    runtime, output, patch, original = failed_run
    if change in ('source', 'core'):
        path = runtime / ('gsm8k_experiment/answers.py' if change == 'source' else 'ppo_engine.py')
        with path.open('ab') as stream:
            stream.write(b'\n# unreviewed change\n')
    elif change == 'training':
        (output / 'arms').mkdir()
    elif change == 'config':
        config = repair.read(output / 'config.json')
        config['seed'] = 99
        repair.atomic_json(output / 'config.json', config)
    else:
        value = json.loads((output / 'invalid_judge_outputs.jsonl').read_text())
        if change == 'reply':
            value['judge_output'] = 'No score available'
        else:
            value['length_capped'] = True
        (output / 'invalid_judge_outputs.jsonl').write_text(json.dumps(value) + '\n')
    before = (runtime / 'gsm8k_experiment/models.py').read_bytes()
    with pytest.raises(ValueError):
        repair.repair(runtime, output)
    assert (runtime / 'gsm8k_experiment/models.py').read_bytes() == before
    assert repair.read(output / 'manifest.json') == original


def test_repair_waits_for_runner_lock(failed_run):
    runtime, output, *_ = failed_run
    with repair.lock(output):
        with pytest.raises(ValueError, match='still running'):
            repair.repair(runtime, output)


def test_interrupted_source_copy_can_be_completed(failed_run, monkeypatch):
    runtime, output, patch, original = failed_run
    atomic = repair.atomic_bytes
    def interrupt(path, data):
        if path.name == 'models.py':
            raise OSError('simulated interruption')
        atomic(path, data)
    monkeypatch.setattr(repair, 'atomic_bytes', interrupt)
    with pytest.raises(OSError, match='interruption'):
        repair.repair(runtime, output)
    assert repair.read(output / 'manifest.json') == original
    monkeypatch.setattr(repair, 'atomic_bytes', atomic)
    repair.repair(runtime, output)
    assert repair.read(output / 'manifest.json')['identity']['source'] == patch['source_after']
