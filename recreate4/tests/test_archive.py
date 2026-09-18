"""Small real-file/SQLite tests; no models, tensors, networks, or GPUs."""
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))
from gsm8k_experiment import archive


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def checkpoint(output, payload=b"fake checkpoint bytes"):
    path = output / "arms" / "proxy" / "checkpoint.pt"
    write(path, payload)
    write(path.with_suffix(".sha256.json"), json.dumps({"sha256": hashlib.sha256(payload).hexdigest(),
                                                     "engine": "preserved-engine"}).encode())


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    output = tmp_path / "node" / "runs" / "seed_42"
    output.mkdir(parents=True)
    write(output / "manifest.json", b'{"fingerprint":"unchanged"}')
    write(output / "arms" / "proxy" / "training" / "step_000005.json", b'{"update":5}')
    checkpoint(output)
    archive_root = tmp_path / "scratch" / "archive"
    monkeypatch.setenv("RECREATE3_ARCHIVE_ROOT", str(archive_root))
    return output, archive_root / "seed_42"


def snapshot(seed_dir):
    latest = json.loads((seed_dir / "latest.json").read_text())
    return seed_dir / "snapshots" / latest["snapshot"]


def test_disabled_is_noop(tmp_path, monkeypatch):
    monkeypatch.delenv("RECREATE3_ARCHIVE_ROOT", raising=False)
    assert archive.archive_output(tmp_path / "does-not-exist") is None
    assert not list(tmp_path.iterdir())


def test_roundtrip_includes_committed_wal_and_excludes_locks_and_temps(experiment, tmp_path):
    output, seed_dir = experiment
    db = sqlite3.connect(output / "reward_cache.sqlite")
    try:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA wal_autocheckpoint=0")
        db.execute("CREATE TABLE scores (key TEXT, score INTEGER)")
        db.execute("INSERT INTO scores VALUES ('committed', 5)")
        db.commit()
        assert (output / "reward_cache.sqlite-wal").stat().st_size > 0
        db.execute("INSERT INTO scores VALUES ('uncommitted', 1)")
        for name in ("run.lock", "checkpoint.tmp", "record.abc.tmp", "pending/work.json",
                     "nested/cache.sqlite-shm", "nested/cache.sqlite-wal", "nested/.pending/work"):
            write(output / name, b"must not archive")
        latest = archive.archive_output(output)
        snap = snapshot(seed_dir)
        manifest = json.loads((snap / "snapshot.json").read_text())
        assert latest["inventory_sha256"] == archive._inventory_hash(manifest["files"])
        assert set(manifest["files"]) == {"manifest.json", "reward_cache.sqlite", "arms/proxy/checkpoint.pt",
                                         "arms/proxy/checkpoint.sha256.json", "arms/proxy/training/step_000005.json"}
        destination = tmp_path / "restored" / "seed_42"
        assert archive.restore_output(seed_dir, destination) == destination.resolve()
        assert (destination / "manifest.json").read_bytes() == (output / "manifest.json").read_bytes()
        assert not (destination / "snapshot.json").exists()
        assert not (destination / "reward_cache.sqlite-wal").exists()
        with sqlite3.connect(destination / "reward_cache.sqlite") as restored:
            assert restored.execute("SELECT * FROM scores").fetchall() == [("committed", 5)]
            assert restored.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
        assert db.execute("SELECT * FROM scores").fetchall() == [("committed", 5), ("uncommitted", 1)]
    finally:
        db.close()


@pytest.mark.parametrize("damage", ["hash", "missing_sidecar", "missing_checkpoint"])
def test_invalid_checkpoint_does_not_replace_previous_latest(experiment, damage):
    output, seed_dir = experiment
    archive.archive_output(output)
    previous = (seed_dir / "latest.json").read_bytes()
    path = output / "arms/proxy/checkpoint.pt"
    if damage == "hash":
        path.write_bytes(b"corrupt")
    elif damage == "missing_sidecar":
        path.with_suffix(".sha256.json").unlink()
    else:
        path.unlink()
    with pytest.raises(ValueError, match="Checkpoint SHA256|unpaired checkpoint"):
        archive.archive_output(output)
    assert (seed_dir / "latest.json").read_bytes() == previous
    assert len(list((seed_dir / "snapshots").iterdir())) == 1


def test_invalid_first_snapshot_has_no_latest(experiment):
    output, seed_dir = experiment
    (output / "arms/proxy/checkpoint.pt").write_bytes(b"wrong bytes")
    with pytest.raises(ValueError, match="Checkpoint SHA256"):
        archive.archive_output(output)
    assert not (seed_dir / "latest.json").exists()


@pytest.mark.parametrize("stage", ["copy", "publish"])
def test_archive_failure_rolls_back_to_previous_snapshot(experiment, monkeypatch, stage):
    output, seed_dir = experiment
    archive.archive_output(output)
    previous = (seed_dir / "latest.json").read_bytes()
    if stage == "copy":
        def fail(*args, **kwargs):
            raise OSError("simulated failed copy")
        monkeypatch.setattr(archive.shutil, "copy2", fail)
    else:
        original = archive._atomic_json
        def fail(path, value):
            if path.name == "latest.json":
                raise OSError("simulated failed publication")
            return original(path, value)
        monkeypatch.setattr(archive, "_atomic_json", fail)
    with pytest.raises(OSError, match="simulated"):
        archive.archive_output(output)
    assert (seed_dir / "latest.json").read_bytes() == previous
    assert len(list((seed_dir / "snapshots").iterdir())) == 1


@pytest.mark.parametrize("damage", ["file", "manifest", "pointer", "missing_latest", "extra_file"])
def test_restore_rejects_corruption_before_creating_destination(experiment, tmp_path, damage):
    output, seed_dir = experiment
    archive.archive_output(output)
    snap = snapshot(seed_dir)
    if damage == "file":
        (snap / "output/manifest.json").write_bytes(b"changed")
    elif damage == "manifest":
        (snap / "snapshot.json").write_text("{}")
    elif damage == "pointer":
        latest = json.loads((seed_dir / "latest.json").read_text())
        latest["snapshot"] = "../../outside"
        (seed_dir / "latest.json").write_text(json.dumps(latest))
    elif damage == "missing_latest":
        (seed_dir / "latest.json").unlink()
    else:
        write(snap / "output/extra.bin", b"unexpected")
    destination = tmp_path / "new-node/seed_42"
    with pytest.raises((ValueError, FileNotFoundError)):
        archive.restore_output(seed_dir, destination)
    assert not destination.exists()
    assert not destination.parent.exists()


def test_restore_preserves_nonempty_destination(experiment, tmp_path):
    output, seed_dir = experiment
    archive.archive_output(output)
    destination = tmp_path / "restored"
    write(destination / "keep.txt", b"user data")
    with pytest.raises(ValueError, match="quarantine"):
        archive.restore_output(seed_dir, destination)
    assert (destination / "keep.txt").read_bytes() == b"user data"


def test_restore_accepts_empty_destination_and_resnapshot_does_not_nest_metadata(experiment, tmp_path):
    output, seed_dir = experiment
    archive.archive_output(output)
    destination = tmp_path / "new-node/seed_42"
    destination.mkdir(parents=True)
    archive.restore_output(seed_dir, destination)
    archive.archive_output(destination)
    assert not (snapshot(seed_dir) / "output/snapshot.json").exists()
    assert not (snapshot(seed_dir) / "output/latest.json").exists()


def test_restore_failed_copy_leaves_empty_destination(experiment, tmp_path, monkeypatch):
    output, seed_dir = experiment
    archive.archive_output(output)
    destination = tmp_path / "restore/seed_42"
    destination.mkdir(parents=True)
    def fail(*args, **kwargs):
        raise OSError("copy interrupted")
    monkeypatch.setattr(archive.shutil, "copy2", fail)
    with pytest.raises(OSError, match="interrupted"):
        archive.restore_output(seed_dir, destination)
    assert list(destination.iterdir()) == []
    assert list(destination.parent.iterdir()) == [destination]


def test_restore_failed_rename_restores_empty_destination(experiment, tmp_path, monkeypatch):
    output, seed_dir = experiment
    archive.archive_output(output)
    destination = tmp_path / "restored/seed_42"
    destination.mkdir(parents=True)
    original = archive.os.replace

    def fail(source, target):
        if Path(target) == destination:
            raise OSError("restore rename interrupted")
        return original(source, target)

    monkeypatch.setattr(archive.os, "replace", fail)
    with pytest.raises(OSError, match="rename interrupted"):
        archive.restore_output(seed_dir, destination)
    assert destination.is_dir() and list(destination.iterdir()) == []
    assert list(destination.parent.iterdir()) == [destination]


def test_postpublication_sync_failure_never_deletes_referenced_snapshot(experiment, tmp_path, monkeypatch):
    output, seed_dir = experiment
    archive.archive_output(output)
    checkpoint(output, b"new checkpoint")
    original = archive._fsync_directory

    def fail(path):
        if Path(path) == seed_dir:
            raise OSError("directory sync interrupted after pointer replacement")
        return original(path)

    monkeypatch.setattr(archive, "_fsync_directory", fail)
    with pytest.raises(OSError, match="sync interrupted"):
        archive.archive_output(output)
    assert (snapshot(seed_dir) / "output/arms/proxy/checkpoint.pt").read_bytes() == b"new checkpoint"
    monkeypatch.setattr(archive, "_fsync_directory", original)
    destination = tmp_path / "recovered"
    archive.restore_output(seed_dir, destination)
    assert (destination / "arms/proxy/checkpoint.pt").read_bytes() == b"new checkpoint"


def test_retention_keeps_latest_and_one_previous(experiment):
    output, seed_dir = experiment
    for i in range(4):
        checkpoint(output, f"checkpoint {i}".encode())
        archive.archive_output(output)
    remaining = list((seed_dir / "snapshots").iterdir())
    assert len(remaining) == 2
    assert snapshot(seed_dir) in remaining
    latest_data = snapshot(seed_dir) / "output/arms/proxy/checkpoint.pt"
    assert latest_data.read_bytes() == b"checkpoint 3"


def test_cleanup_rejects_external_and_protected_paths(tmp_path):
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    external = tmp_path / "external"
    write(external / "keep", b"keep")
    protected = snapshots / "latest"
    write(protected / "keep", b"keep")
    for candidate in (external, snapshots, protected):
        with pytest.raises(ValueError, match="Refusing archive cleanup"):
            archive._remove_tree(candidate, snapshots, protected=(protected,))
    assert (external / "keep").exists() and (protected / "keep").exists()


def test_archive_rejects_nested_archive_root(experiment, monkeypatch):
    output, _ = experiment
    monkeypatch.setenv("RECREATE3_ARCHIVE_ROOT", str(output / "archive"))
    with pytest.raises(ValueError, match="must be separate"):
        archive.archive_output(output)
