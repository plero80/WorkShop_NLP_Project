"""A read-only status command must never terminate the training process."""
import os
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime'))
from gsm8k_experiment import status


@pytest.mark.skipif(os.name != 'nt', reason='Exercises real Windows process-query APIs')
def test_live_child_survives_windows_status(tmp_path, monkeypatch, capsys):
    child = subprocess.Popen(
        [sys.executable, '-u', '-c', "import time; print('ready', flush=True); time.sleep(60)"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    try:
        assert child.stdout.readline().strip() == 'ready'
        (tmp_path / 'pid.txt').write_text(str(child.pid), encoding='utf-8')
        monkeypatch.setattr(status.os, 'kill', lambda *_: pytest.fail('Windows status must not call os.kill'))
        status.show(tmp_path, 0)
        assert f'PID {child.pid} exists' in capsys.readouterr().out
        assert child.poll() is None
        assert status.process_alive(child.pid) is True
        child.terminate()
        child.wait(timeout=10)
        assert status.process_alive(child.pid) is False
        status.show(tmp_path, 0)
        assert 'no longer running' in capsys.readouterr().out
    finally:
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=10)
        child.stdout.close()
        child.stderr.close()


@pytest.mark.parametrize('pid', [0, -1])
def test_nonpositive_pid_is_rejected_without_signaling(pid, monkeypatch):
    monkeypatch.setattr(status.os, 'kill', lambda *_: pytest.fail('Invalid PID must not be signaled'))
    with pytest.raises(ValueError, match='positive'):
        status.process_alive(pid)


def test_unqueryable_process_is_not_reported_as_dead(tmp_path, monkeypatch, capsys):
    (tmp_path / 'pid.txt').write_text('123', encoding='utf-8')
    def denied(pid):
        raise PermissionError('process query denied')
    monkeypatch.setattr(status, 'process_alive', denied)
    status.show(tmp_path, 0)
    output = capsys.readouterr().out
    assert 'Could not verify' in output and 'no longer running' not in output


def test_status_reads_utf8_artifacts_and_invalid_pid(tmp_path, capsys):
    (tmp_path / 'pid.txt').write_text('invalid', encoding='utf-8')
    (tmp_path / 'status.json').write_text('{"stage":"résumé"}', encoding='utf-8')
    (tmp_path / 'experiment.log').write_text('∑ = 5\n', encoding='utf-8')
    status.show(tmp_path, 1)
    output = capsys.readouterr().out
    assert 'no longer running' in output and '∑ = 5' in output
