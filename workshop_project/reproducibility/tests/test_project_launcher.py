"""Launching from the organized project must reuse the existing run's identity."""
import importlib.util
import json
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('project_launcher', PROJECT / 'gsm8k.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    folder = tmp_path / 'runtime'
    monkeypatch.setattr(launcher, 'RUNTIME', folder)
    return folder


def test_dry_run_from_project_needs_no_runtime_and_creates_no_files(runtime, capsys):
    assert launcher.main(['run', 'gsm8k-b200', '--stage', 'full', '--dry-run']) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan['stage'] == 'full'
    assert plan['output'] == str(runtime / 'gsm8k_outputs/b200')
    assert plan['settings']['seed'] == 42
    assert plan['settings']['ppo']['full_updates'] == 400
    assert not runtime.exists()


def test_setup_is_minimal_and_repeated_setup_preserves_saved_experiment(runtime):
    assert launcher.main(['setup']) == 0
    assert (runtime / 'gsm8k_experiment/settings.json').is_file()
    assert (runtime / 'ppo_engine.py').is_file()
    assert not list(runtime.rglob('*.ipynb'))
    assert not (runtime / 'inputs').exists()
    state = runtime / 'gsm8k_outputs/b200/arms/proxy/checkpoint.pt'
    state.parent.mkdir(parents=True)
    state.write_bytes(b'existing checkpoint must stay intact')
    before = {p: p.read_bytes() for p in runtime.rglob('*') if p.is_file()}
    assert launcher.main(['setup']) == 0
    assert all(p.read_bytes() == data for p, data in before.items())


def test_project_and_restored_launchers_resolve_identical_config_and_output(runtime):
    launcher.ensure_runtime()
    project_cli = launcher.load_module('source_cli', PROJECT / 'code/experiments/experiment_cli/cli.py')
    restored_cli = launcher.load_module('restored_cli', runtime / 'experiment_cli/cli.py')
    layout = {'root': runtime, 'settings': PROJECT / 'configs/gsm8k/settings.json',
              'presets': PROJECT / 'configs/experiments'}
    project_plan = project_cli.resolve('gsm8k-b200', stage='full', layout=layout)
    restored_plan = restored_cli.resolve('gsm8k-b200', stage='full')
    for key in ('config', 'config_text', 'settings', 'output', 'seeds'):
        assert project_plan[key] == restored_plan[key]
    assert project_cli.command(project_plan, 'run') == restored_cli.command(restored_plan, 'run')


def test_full_run_delegates_to_existing_runtime_without_overwriting_checkpoint(runtime, monkeypatch):
    launcher.ensure_runtime()
    checkpoint = runtime / 'gsm8k_outputs/b200/arms/proxy/checkpoint.pt'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'100 attempts already completed')
    load = launcher.load_module
    calls = []
    def module(name, path):
        result = load(name, path)
        if name == 'gsm8k_project_cli':
            def execute(command, cwd):
                calls.append((command, cwd))
                cfg = json.loads(Path(command[command.index('--config') + 1]).read_text())
                assert cfg['ppo']['full_updates'] == 400
                return 7
            monkeypatch.setattr(result.subprocess, 'call', execute)
        return result
    monkeypatch.setattr(launcher, 'load_module', module)
    assert launcher.main(['run', 'gsm8k-b200', '--stage', 'full']) == 7
    command, cwd = calls[0]
    assert cwd == runtime
    assert command[3] == 'gsm8k_experiment.run'
    assert command[command.index('--output') + 1] == str(runtime / 'gsm8k_outputs/b200')
    assert checkpoint.read_bytes() == b'100 attempts already completed'


@pytest.mark.parametrize('filename', ['ppo_engine.py', 'gsm8k_experiment/models.py', 'gsm8k_experiment/unexpected.py'])
def test_changed_scientific_sources_are_not_silently_overwritten(runtime, filename):
    launcher.ensure_runtime()
    path = runtime / filename
    path.write_text('# local change\n')
    with pytest.raises(ValueError, match='Nothing was overwritten'):
        launcher.ensure_runtime()
    assert path.read_text() == '# local change\n'


def test_custom_recipe_and_export_destination_are_relative_to_callers_directory(runtime, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    recipe = tmp_path / 'custom.yaml'
    recipe.write_text('version: 1\nexperiment: gsm8k\noutput: gsm8k_outputs/seed43\nsettings:\n  seed: 43\n')
    assert launcher.main(['run', 'custom.yaml', '--dry-run']) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan['recipe'] == str(recipe)
    assert plan['output'] == str(runtime / 'gsm8k_outputs/seed43')
    cli = launcher.load_module('export_cli', PROJECT / 'code/experiments/experiment_cli/cli.py')
    assert cli.command(plan, 'export', 'results.zip')[-1] == str(tmp_path / 'results.zip')
    assert not runtime.exists()
