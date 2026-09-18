"""Moved Windows scripts keep their environment and validate the same package."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import pytest


ROOT = Path(__file__).resolve().parents[1]
PWSH = shutil.which('pwsh')
SPEC = importlib.util.spec_from_file_location('windows_layout_package_state', ROOT / 'python_helper/package_state.py')
package_state = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(package_state)


def ps_quote(value):
    return "'" + str(value).replace("'", "''") + "'"


def run_pwsh(script):
    script = '[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false); ' + script
    return subprocess.run([PWSH, '-NoLogo', '-NoProfile', '-Command', script],
                          capture_output=True, text=True, encoding='utf-8', timeout=20,
                          creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)


def make_package(parent, *, embedded=True):
    package = parent / 'recreate3'
    files = {
        'runtime/experiment_cli/__init__.py': b'# immutable scientific runtime\n',
        'python_helper/helper.py': b'# helper fixture\n',
        'windows/Common.ps1': (ROOT / 'windows/Common.ps1').read_bytes(),
        'windows/Run.ps1': b'# launcher fixture\n',
    }
    for name, data in files.items():
        target = package / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    manifest = {'files': {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}}
    (package / 'provenance').mkdir()
    (package / 'provenance/package_manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    windows = package / 'windows'
    if not embedded:
        windows.rename(parent / 'windows')
        windows = parent / 'windows'
    return package, windows


def paths_for(windows, options=''):
    result = run_pwsh(f'. {ps_quote(windows / "Common.ps1")}; '
                      f'Get-LocalPaths -ExperimentId example {options} | ConvertTo-Json -Compress')
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout)


@pytest.mark.skipif(not PWSH, reason='PowerShell 7 unavailable')
@pytest.mark.parametrize('embedded', [True, False])
def test_local_paths_keep_state_environment_and_vram_recipe(tmp_path, embedded):
    package, windows = make_package(tmp_path, embedded=embedded)
    before = {p.relative_to(tmp_path) for p in tmp_path.rglob('*')}
    paths = paths_for(windows, '-VramGB 12')
    assert Path(paths['Package']) == package
    assert Path(paths['Runtime']) == package / 'runtime'
    assert Path(paths['Helpers']) == package / 'python_helper'
    assert Path(paths['Recipe']) == package / 'configs/gsm8k-12gb.yaml'
    assert Path(paths['Work']) == windows / 'work'
    assert Path(paths['Venv']) == windows / '.venv'
    assert Path(paths['Output']) == windows / 'work/runs/example'
    assert before == {p.relative_to(tmp_path) for p in tmp_path.rglob('*')}


@pytest.mark.skipif(not PWSH, reason='PowerShell 7 unavailable')
def test_embedded_package_has_precedence_and_overrides_survive(tmp_path):
    package, windows = make_package(tmp_path)
    make_package(package)  # A second plausible candidate must not change the first.
    custom_work, custom_venv, custom_recipe = [tmp_path / name for name in ('work override', 'venv override', 'recipe override.yaml')]
    paths = paths_for(windows, f'-WorkRoot {ps_quote(custom_work)} -VenvPath {ps_quote(custom_venv)} '
                              f'-Recipe {ps_quote(custom_recipe)}')
    assert Path(paths['Package']) == package
    assert Path(paths['Work']) == custom_work
    assert Path(paths['Venv']) == custom_venv
    assert Path(paths['Recipe']) == custom_recipe


@pytest.mark.skipif(not PWSH, reason='PowerShell 7 unavailable')
@pytest.mark.parametrize('partial', ['none', 'runtime', 'helpers', 'custom_name'])
def test_missing_valid_package_fails_without_creating_paths(tmp_path, partial):
    windows = tmp_path / 'windows'
    windows.mkdir()
    shutil.copy2(ROOT / 'windows/Common.ps1', windows / 'Common.ps1')
    if partial == 'runtime':
        (tmp_path / 'runtime/experiment_cli').mkdir(parents=True)
    elif partial == 'helpers':
        (tmp_path / 'python_helper').mkdir()
    elif partial == 'custom_name':
        (tmp_path / 'other_package/runtime/experiment_cli').mkdir(parents=True)
        (tmp_path / 'other_package/python_helper').mkdir()
    before = {p.relative_to(tmp_path) for p in tmp_path.rglob('*')}
    result = run_pwsh(f'. {ps_quote(windows / "Common.ps1")}; Get-LocalPaths -ExperimentId example')
    assert result.returncode != 0
    assert 'Cannot locate recreate3' in result.stderr
    assert before == {p.relative_to(tmp_path) for p in tmp_path.rglob('*')}


def test_relocated_manifest_verifies_and_stages_immutable_runtime(tmp_path):
    package, windows = make_package(tmp_path, embedded=False)
    runtime_before = (package / 'runtime/experiment_cli/__init__.py').read_bytes()
    (windows / 'work').mkdir()
    (windows / 'work/checkpoint.pt').write_bytes(b'existing checkpoint')
    (windows / '.venv').mkdir()
    package_state.verify(package)
    destination = tmp_path / 'staged/runtime'
    package_state.stage(package, destination)
    assert (destination / 'experiment_cli/__init__.py').read_bytes() == runtime_before
    assert (package / 'runtime/experiment_cli/__init__.py').read_bytes() == runtime_before
    assert (windows / 'work/checkpoint.pt').read_bytes() == b'existing checkpoint'
    assert not (package / 'windows').exists()
    assert not (destination / 'windows').exists()


@pytest.mark.parametrize('location', ['windows', 'runtime'])
def test_relocated_verification_rejects_modified_helper_or_runtime(tmp_path, location):
    package, windows = make_package(tmp_path, embedded=False)
    target = windows / 'Run.ps1' if location == 'windows' else package / 'runtime/experiment_cli/__init__.py'
    target.write_bytes(b'tampered')
    with pytest.raises(ValueError, match='differs from its manifest'):
        package_state.verify(package)


def test_existing_embedded_windows_never_falls_back_for_missing_file(tmp_path):
    package, windows = make_package(tmp_path)
    shutil.copytree(windows, tmp_path / 'windows')
    (windows / 'Run.ps1').unlink()
    with pytest.raises(ValueError, match='windows/Run.ps1'):
        package_state.verify(package)


def test_fallback_does_not_search_arbitrary_sibling_names(tmp_path):
    package, windows = make_package(tmp_path, embedded=False)
    windows.rename(tmp_path / 'custom_windows')
    with pytest.raises(ValueError, match='windows/Common.ps1'):
        package_state.verify(package)


@pytest.mark.parametrize('name', ['windows/../../outside.txt', 'windows/../outside.txt'])
def test_relocated_manifest_cannot_escape_windows(tmp_path, name):
    package, _ = make_package(tmp_path, embedded=False)
    manifest_path = package / 'provenance/package_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['files'][name] = hashlib.sha256(b'outside').hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    (tmp_path / 'outside.txt').write_bytes(b'outside')
    (package / 'outside.txt').write_bytes(b'outside')
    with pytest.raises(ValueError, match='escapes its directory'):
        package_state.verify(package)


@pytest.mark.skipif(not PWSH, reason='PowerShell 7 unavailable')
def test_shared_reader_handles_utf8_and_tail(tmp_path):
    source = tmp_path / 'status.log'
    source.write_text('start\nrésumé\n∑ = 5\n', encoding='utf-8')
    result = run_pwsh(f'. {ps_quote(ROOT / "windows/Common.ps1")}; '
                      f'@{{full=(Read-SharedText {ps_quote(source)}); '
                      f'tail=@(Read-SharedText {ps_quote(source)} -Tail 2)}} | ConvertTo-Json -Compress')
    assert result.returncode == 0, result.stderr
    values = json.loads(result.stdout)
    assert values['full'] == source.read_bytes().decode('utf-8')
    assert values['tail'] == ['résumé', '∑ = 5']


@pytest.mark.skipif(not PWSH or os.name != 'nt', reason='Exercises actual Windows file sharing')
def test_status_readers_allow_atomic_replacement(tmp_path):
    target, ready, stop = [tmp_path / name for name in ('status.json', 'ready', 'stop')]
    target.write_text(json.dumps({'stage': 'training', 'padding': 'x' * 100_000}), encoding='utf-8')
    script = (f'. {ps_quote(ROOT / "windows/Common.ps1")}; '
              f'[IO.File]::WriteAllText({ps_quote(ready)}, "ready"); '
              f'while (-not [IO.File]::Exists({ps_quote(stop)})) '
              f'{{ [void](Read-SharedText {ps_quote(target)}) }}')
    child = subprocess.Popen([PWSH, '-NoLogo', '-NoProfile', '-Command', script],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                             creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        deadline = time.monotonic() + 15
        while not ready.exists():
            assert child.poll() is None, child.communicate()
            assert time.monotonic() < deadline, 'Reader did not start'
            time.sleep(0.01)
        for i in range(150):
            temporary = tmp_path / 'replacement.tmp'
            temporary.write_text(json.dumps({'update': i, 'padding': 'x' * 100_000}), encoding='utf-8')
            # MoveFileEx may also encounter a transient scanner or the instant
            # between a reader's close/reopen. No read should hold it blocked.
            replace_deadline = time.monotonic() + 2
            while True:
                try:
                    os.replace(temporary, target)
                    break
                except PermissionError:
                    if time.monotonic() >= replace_deadline:
                        raise
                    time.sleep(0.01)
        stop.write_text('stop')
        stdout, stderr = child.communicate(timeout=10)
        assert child.returncode == 0, stdout + stderr
    finally:
        if child.poll() is None:
            child.terminate()
            child.communicate(timeout=10)
