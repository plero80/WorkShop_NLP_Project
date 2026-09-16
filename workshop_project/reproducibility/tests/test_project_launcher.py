"""New runs import the project source; historical runs keep their own code."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

PROJECT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('project_launcher', PROJECT / 'gsm8k.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


@pytest.fixture
def legacy(tmp_path, monkeypatch):
    folder = tmp_path / 'legacy'
    monkeypatch.setattr(launcher, 'LEGACY', folder)
    return folder


def make_legacy(folder):
    settings = folder / 'gsm8k_experiment/settings.json'
    settings.parent.mkdir(parents=True)
    settings.write_bytes((PROJECT / 'configs/gsm8k/settings.json').read_bytes())
    output = folder / 'gsm8k_outputs/b200'
    output.mkdir(parents=True)
    (output / 'config.json').write_bytes(settings.read_bytes())
    (output / 'checkpoint.pt').write_bytes(b'keep the existing running experiment')
    return output


def test_new_run_dry_run_has_no_copies_or_generated_files(legacy, capsys):
    assert launcher.main(['run', 'gsm8k-b200', '--stage', 'full', '--dry-run']) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan['output'] == str(PROJECT / 'gsm8k_outputs/b200')
    assert plan['settings']['ppo']['full_updates'] == 400
    assert not legacy.exists()
    assert not Path(plan['config']).exists()


def test_setup_only_prints_installation_command(legacy, capsys):
    assert launcher.main(['setup']) == 0
    assert 'requirements-gsm8k.txt' in capsys.readouterr().out
    assert not legacy.exists()


def test_native_children_import_actual_project_files_and_config(legacy):
    cli = launcher.load_cli()
    layout = launcher.select_layout(cli, ['run', 'gsm8k-b200'])
    code = ('import json, gsm8k_experiment.common as c, ppo_engine; '
            'from gsm8k_experiment.shared import shared_sources; '
            'from gsm8k_experiment.assets import source_fingerprint; '
            'assert shared_sources(); '
            'print(json.dumps([str(c.ROOT), str(c.DEFAULT_CONFIG), ppo_engine.__file__, source_fingerprint()]))')
    result = subprocess.run([sys.executable, '-c', code], cwd=PROJECT, env=layout['env'],
                            capture_output=True, text=True, check=True)
    root, settings, core, fingerprint = json.loads(result.stdout)
    assert Path(root) == PROJECT
    assert Path(settings) == PROJECT / 'configs/gsm8k/settings.json'
    assert Path(core) == PROJECT / 'code/core/ppo_engine.py'
    assert len(fingerprint) == 64
    assert not legacy.exists()


@pytest.mark.parametrize('action', ['run', 'status', 'export'])
def test_existing_run_routes_to_old_runtime_without_touching_its_files(legacy, action):
    output = make_legacy(legacy)
    before = {p: p.read_bytes() for p in legacy.rglob('*') if p.is_file()}
    cli = launcher.load_cli()
    layout = launcher.select_layout(cli, [action, 'gsm8k-b200'])
    assert layout['root'] == legacy and 'env' not in layout
    plan = cli.resolve('gsm8k-b200', stage='full', layout=layout)
    assert plan['output'] == str(output)
    assert all(p.read_bytes() == data for p, data in before.items())


def test_existing_runtime_does_not_force_new_seeds_to_make_code_copies(legacy):
    make_legacy(legacy)
    layout = launcher.select_layout(launcher.load_cli(), [
        'run', 'gsm8k-b200', '--set', 'seed=43', '--output', 'gsm8k_outputs/seed43'])
    assert layout['root'] == PROJECT
    assert 'env' in layout


def test_explicit_old_output_uses_its_runtime(legacy):
    output = make_legacy(legacy)
    layout = launcher.select_layout(launcher.load_cli(), ['run', 'gsm8k-b200', '--output', str(output)])
    assert layout['root'] == legacy


def test_ambiguous_runs_require_explicit_output(legacy, monkeypatch):
    make_legacy(legacy)
    monkeypatch.setattr(launcher, 'has_run', lambda path: True)
    with pytest.raises(ValueError, match='Both project and legacy'):
        launcher.select_layout(launcher.load_cli(), ['run', 'gsm8k-b200'])


def test_launch_passes_native_import_paths_to_subprocess(legacy, tmp_path, monkeypatch):
    cli = launcher.load_cli()
    captured = []
    def call(command, **kwargs):
        captured.append((command, kwargs))
        return 9
    monkeypatch.setattr(cli.subprocess, 'call', call)
    monkeypatch.setattr(cli, 'materialize', lambda plan: None)
    monkeypatch.setattr(launcher, 'load_cli', lambda: cli)
    assert launcher.main(['run', 'gsm8k-b200', '--output', str(tmp_path / 'new-run')]) == 9
    command, options = captured[0]
    assert options['cwd'] == PROJECT
    assert str(PROJECT / 'code/core') in options['env']['PYTHONPATH'].split(os.pathsep)
    assert command[3] == 'gsm8k_experiment.run'
    assert not legacy.exists()
