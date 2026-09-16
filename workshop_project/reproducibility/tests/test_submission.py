import importlib.util
import json
from pathlib import Path
import sys
import zipfile

import pytest

PROJECT = Path(__file__).resolve().parents[2]


@pytest.fixture
def submission(monkeypatch):
    monkeypatch.syspath_prepend(str(PROJECT))
    spec = importlib.util.spec_from_file_location('submission', PROJECT / 'submission.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_source_archive_and_completed_results_contain_one_code_copy(submission, tmp_path, monkeypatch):
    source = tmp_path / 'ppo_engine.py'
    source.write_text('# one source copy')
    monkeypatch.setattr(submission, 'source_files', lambda: iter([(source, 'workshop_project/code/core/ppo_engine.py')]))
    results = tmp_path / 'b200'
    results.mkdir()
    (results / 'status.json').write_text(json.dumps({'stage': 'complete', 'run_stage': 'full'}))
    (results / 'report.md').write_text('finished report')
    (results / 'checkpoint.pt').write_bytes(b'not included')
    (results / 'copied_code.py').write_text('# not included')
    destination = submission.build(tmp_path / 'submission.zip', [results])
    with zipfile.ZipFile(destination) as archive:
        names = archive.namelist()
        assert names.count('workshop_project/code/core/ppo_engine.py') == 1
        assert 'workshop_project/results/gsm8k/b200/report.md' in names
        assert not any(n.endswith(('.pt', 'copied_code.py')) for n in names)
        manifest = json.loads(archive.read('SUBMISSION_MANIFEST.json'))
        assert manifest['results_included'] == ['b200']
        assert manifest['runtime_copies_included'] is False


def test_active_run_is_not_packaged_or_modified(submission, tmp_path):
    results = tmp_path / 'active'
    results.mkdir()
    data = json.dumps({'stage': 'ppo', 'update': 150})
    (results / 'status.json').write_text(data)
    with pytest.raises(ValueError, match='Wait for this run'):
        submission.build(tmp_path / 'submission.zip', [results])
    assert (results / 'status.json').read_text() == data
    assert not (tmp_path / 'submission.zip').exists()


def test_existing_archive_is_not_replaced(submission, tmp_path):
    destination = tmp_path / 'submission.zip'
    destination.write_bytes(b'previous submission')
    with pytest.raises(ValueError, match='already exists'):
        submission.build(destination)
    assert destination.read_bytes() == b'previous submission'
