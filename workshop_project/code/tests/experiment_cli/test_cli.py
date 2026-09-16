import json
from pathlib import Path
import subprocess
import sys

import pytest

from experiment_cli import cli


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    # Copy only defaults; tests never instantiate models or launch training.
    original = cli.ROOT / 'gsm8k_experiment/settings.json'
    folder = tmp_path / 'runtime'
    (folder / 'gsm8k_experiment').mkdir(parents=True)
    (folder / 'gsm8k_experiment/settings.json').write_bytes(original.read_bytes())
    monkeypatch.setattr(cli, 'ROOT', folder)
    return folder


def recipe(tmp_path, text):
    path = tmp_path / 'custom.yaml'
    path.write_text('version: 1\nexperiment: gsm8k\n' + text, encoding='utf-8')
    return str(path)


def test_default_preset_preserves_entire_scientific_config(runtime):
    plan = cli.resolve('gsm8k')
    assert plan['settings'] == json.loads((runtime / 'gsm8k_experiment/settings.json').read_text())
    assert plan['output'] == str((runtime / 'gsm8k_outputs/main').resolve())
    assert plan['stage'] == 'pilot'


def test_nested_overrides_and_stage_keep_resume_identity(runtime, tmp_path):
    path = recipe(tmp_path, 'settings:\n  ppo:\n    learning_rate: 2.0e-5\n')
    pilot = cli.resolve(path, overrides=['generation.batch_size=4'])
    full = cli.resolve(path, stage='full', overrides=['generation.batch_size=4'])
    assert pilot['config'] == full['config']
    assert full['settings']['ppo']['learning_rate'] == .00002
    assert full['settings']['generation']['batch_size'] == 4
    assert full['settings']['ppo']['full_updates'] == 400
    assert cli.resolve(path)['config'] != full['config']


@pytest.mark.parametrize('body, message', [
    ('surprise: 2\n', 'Unknown recipe'),
    ('settings:\n  ppo:\n    learnig_rate: 1.0e-5\n', 'Unknown setting'),
    ('settings:\n  ppo:\n    learning_rate: 1e-5\n', 'expects float'),
    ('settings:\n  seed: true\n', 'expects int'),
    ('settings:\n  generation:\n    temperature: 0.7\n', 'temperature=1.0'),
    ('settings:\n  arms: [proxy, proxy]\n', 'duplicate reward'),
    ('settings:\n  arms: [not_a_reward]\n', 'Unknown'),
    ('settings:\n  ppo:\n    checkpoint_every: 0\n', 'positive'),
    ('seeds: [42, 42]\n', 'distinct'),
    ('seeds: []\n', 'nonempty'),
    ('seeds: [true]\n', 'distinct'),
    ('settings: []\n', 'mapping'),
    ('output: 23\n', 'path string'),
    ('stage: typo\n', 'Stage'),
    ('settings:\n  seed: 42\n  seed: 43\n', 'Duplicate YAML'),
    ('settings:\n  ppo:\n    learning_rate: .nan\n', 'expects float'),
])
def test_reject_mistakes_before_launch(runtime, tmp_path, body, message):
    with pytest.raises(ValueError, match=message):
        cli.resolve(recipe(tmp_path, body))


def test_yaml_objects_cannot_execute_code(tmp_path):
    marker = tmp_path / 'not-created'
    with pytest.raises(ValueError, match='Invalid YAML'):
        cli.yaml_value(f'!!python/object/apply:pathlib.Path.touch ["{marker.as_posix()}"]')
    assert not marker.exists()


def test_dry_run_has_no_files_or_subprocess(runtime, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail('Dry run started a subprocess')
    monkeypatch.setattr(cli.subprocess, 'call', forbidden)
    assert cli.main(['run', 'gsm8k', '--dry-run']) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan['command'][3] == 'gsm8k_experiment.run'
    assert not Path(plan['config']).exists()
    assert not Path(plan['output']).exists()


def test_run_passes_resolved_json_to_existing_runner_and_propagates_exit(runtime, monkeypatch):
    calls = []
    def execute(command, cwd):
        calls.append(command)
        assert cwd == runtime
        config = Path(command[command.index('--config') + 1])
        assert json.loads(config.read_text())['generation']['batch_size'] == 8
        return 7
    monkeypatch.setattr(cli.subprocess, 'call', execute)
    assert cli.main(['run', 'gsm8k', '--set', 'generation.batch_size=8']) == 7
    assert cli.main(['run', 'gsm8k', '--stage', 'full', '--set', 'generation.batch_size=8']) == 7
    assert calls[0][calls[0].index('--config')+1] == calls[1][calls[1].index('--config')+1]
    assert calls[1][calls[1].index('--stage')+1] == 'full'


def test_suite_dispatch_preserves_seeds_and_partition(runtime):
    plan = cli.resolve('gsm8k-three-seeds')
    command = cli.command(plan, 'run')
    assert command[3] == 'gsm8k_experiment.suite'
    assert command[-4:] == ['--seeds', '42', '43', '44']
    assert plan['settings']['data_seed'] == 42
    with pytest.raises(ValueError, match='single-run'):
        cli.resolve('gsm8k-three-seeds', stage='report')


def test_paths_with_spaces_and_different_cwd(runtime, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = recipe(tmp_path, 'output: outputs/a run.v1\n')
    plan = cli.resolve(path)
    command = cli.command(plan, 'run')
    assert command[command.index('--output')+1] == str((runtime / 'outputs/a run.v1').resolve())
    assert cli.command(plan, 'export')[-1].endswith('a run.v1.zip')


def test_existing_resolved_json_cannot_be_silently_replaced(runtime):
    plan = cli.resolve('gsm8k')
    cli.materialize(plan)
    path = Path(plan['config'])
    path.write_text('{}')
    with pytest.raises(ValueError, match='changed'):
        cli.materialize(plan)
    assert path.read_text() == '{}'


@pytest.mark.parametrize('action, module', [('status', 'gsm8k_experiment.status'), ('export', 'gsm8k_experiment.export')])
def test_inspection_commands_do_not_materialize_configs(runtime, monkeypatch, action, module):
    calls = []
    monkeypatch.setattr(cli.subprocess, 'call', lambda args, cwd: calls.append(args) or 0)
    assert cli.main([action, 'gsm8k']) == 0
    assert calls[0][3] == module
    assert not (runtime / '.experiment_cli').exists()


def test_module_entry_point_in_actual_restored_runtime():
    result = subprocess.run([sys.executable, '-m', 'experiment_cli', 'run', 'gsm8k', '--dry-run'],
                            cwd=cli.ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['experiment'] == 'gsm8k'
