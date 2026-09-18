"""Asset selection and immutable offline resolution, with no Hub/GPU access."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))
from gsm8k_experiment import assets


@pytest.fixture
def config():
    roles = ("policy", "proxy", "judge", "judge30b")
    return {
        "arms": ["proxy", "judge", "knn_static"],
        "models": {role: f"test/{role}" for role in roles},
        "revisions": {**{role: digit * 40 for role, digit in zip(roles, "abcd")},
                      "dataset": "e" * 40},
        "dataset": {"id": "test/math", "config": "main"},
        "teacher30b": {"evaluate_final": False},
    }


@pytest.fixture
def hub(monkeypatch):
    calls = SimpleNamespace(metadata=[], downloads=[], constructed=0)

    class Api:
        def __init__(self, **kwargs):
            calls.constructed += 1
            assert kwargs == {"token": False}

        def model_info(self, name, revision):
            calls.metadata.append(("model", name, revision))
            return SimpleNamespace(sha="f" * 40)

        def dataset_info(self, name, revision):
            calls.metadata.append(("dataset", name, revision))
            return SimpleNamespace(sha="0" * 40)

    def download(name, **kwargs):
        calls.downloads.append((name, kwargs))
        return "/fake/cache/snapshot"

    monkeypatch.setitem(sys.modules, "huggingface_hub",
                        SimpleNamespace(HfApi=Api, snapshot_download=download))
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    return calls


@pytest.mark.parametrize("offline", [False, True])
def test_three_arm_pinned_run_never_touches_30b(config, hub, tmp_path, monkeypatch, offline):
    if offline:
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    resolved = assets.resolve_assets(config, tmp_path)
    assert resolved == {role: config["revisions"][role] for role in ("policy", "proxy", "judge", "dataset")}
    assert hub.constructed == 0
    assert hub.metadata == []
    assert [name for name, _ in hub.downloads] == ["test/policy", "test/proxy", "test/judge"]
    for name, kwargs in hub.downloads:
        assert kwargs["revision"] == config["revisions"][name.split("/")[-1]]
        assert kwargs["local_files_only"] is offline
        assert kwargs["allow_patterns"] == list(assets.MODEL_DOWNLOAD_PATTERNS)
    assert json.loads((tmp_path / "resolved_assets.json").read_text()) == resolved


@pytest.mark.parametrize("reason", ["arm", "evaluation", "both"])
def test_teacher_is_downloaded_when_requested(config, hub, tmp_path, reason):
    if reason in ("arm", "both"):
        config["arms"].append("knn_static_30b")
    config["teacher30b"]["evaluate_final"] = reason in ("evaluation", "both")
    resolved = assets.resolve_assets(config, tmp_path)
    assert resolved["judge30b"] == config["revisions"]["judge30b"]
    assert [name for name, _ in hub.downloads].count("test/judge30b") == 1


def test_symbolic_revisions_resolve_once_then_work_offline(config, hub, tmp_path, monkeypatch):
    config["revisions"] = dict.fromkeys(config["revisions"], "main")
    resolved = assets.resolve_assets(config, tmp_path)
    assert hub.metadata == [("model", f"test/{role}", "main") for role in ("policy", "proxy", "judge")] + [
        ("dataset", "test/math", "main")]
    assert "judge30b" not in resolved
    assert all(kwargs["revision"] == "f" * 40 for _, kwargs in hub.downloads)
    hub.metadata.clear()
    hub.downloads.clear()
    monkeypatch.setenv("HF_HUB_OFFLINE", "YES")
    assert assets.resolve_assets(config, tmp_path) == resolved
    assert hub.metadata == []
    assert all(kwargs["local_files_only"] for _, kwargs in hub.downloads)


def test_offline_unresolved_branch_fails_without_hub_call(config, hub, tmp_path, monkeypatch):
    config["revisions"]["policy"] = "main"
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    with pytest.raises(ValueError, match="Offline policy requires"):
        assets.resolve_assets(config, tmp_path)
    assert hub.constructed == 0 and not hub.metadata and not hub.downloads


def test_existing_optional_teacher_pin_does_not_trigger_download(config, hub, tmp_path, monkeypatch):
    saved = copy.deepcopy(config["revisions"])
    (tmp_path / "resolved_assets.json").write_text(json.dumps(saved))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert assets.resolve_assets(config, tmp_path) == saved
    assert "test/judge30b" not in [name for name, _ in hub.downloads]
    assert hub.constructed == 0


def test_saved_commit_cannot_override_configured_pin(config, hub, tmp_path):
    saved = copy.deepcopy(config["revisions"])
    saved["policy"] = "f" * 40
    (tmp_path / "resolved_assets.json").write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="Saved policy revision differs"):
        assets.resolve_assets(config, tmp_path)
    assert hub.constructed == 0 and not hub.downloads


def test_offline_missing_snapshot_is_not_retried(config, hub, tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    attempts = []

    def missing(name, **kwargs):
        attempts.append(name)
        raise FileNotFoundError("prewarm the model snapshot first")

    monkeypatch.setattr(sys.modules["huggingface_hub"], "snapshot_download", missing)
    monkeypatch.setattr(assets.time, "sleep", lambda _: pytest.fail("Offline cache misses must not sleep"))
    with pytest.raises(FileNotFoundError, match="prewarm"):
        assets.resolve_assets(config, tmp_path)
    assert attempts == ["test/policy"]


def test_requested_teacher_must_be_configured(config):
    config["arms"].append("knn_static_30b")
    del config["models"]["judge30b"]
    with pytest.raises(ValueError, match="models.judge30b"):
        assets.required_model_roles(config)
