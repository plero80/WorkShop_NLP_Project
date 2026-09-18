"""Synchronous, verified snapshots of a paused single-writer experiment.

The live SQLite cache stays on the node-local filesystem. Its committed WAL
contents are captured with SQLite's backup API, never by copying the WAL files.
Snapshot metadata lives beside output/, so restoring cannot nest archive metadata.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sqlite3
import tempfile
import uuid


SCHEMA = "recreate3.archive.v1"
_SNAPSHOT_ID = re.compile(r"\d{8}T\d{6}\.\d{6}Z-[0-9a-f]{32}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _inventory_hash(inventory):
    return hashlib.sha256(json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _linked(path):
    return path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction())


def _read_json(path):
    if _linked(path):
        raise ValueError(f"Archive metadata must not be a link: {path}")
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _fsync_file(path):
    # Windows' _commit requires a writable descriptor; no bytes are modified.
    with Path(path).open("r+b") as stream:
        os.fsync(stream.fileno())


def _fsync_directory(path):
    # Windows cannot open directory handles this way. os.replace remains atomic.
    if os.name != "nt":
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _atomic_json(path, value):
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _excluded(name):
    name = name.lower()
    return (name == "run.lock" or name in {"tmp", "pending", ".tmp", ".pending"}
            or name.endswith((".tmp", ".pending", "-wal", "-shm")))


def _files(root, exclude_pending=False):
    """Do not follow links, junctions, devices, or paths outside the output."""
    result = {}
    for directory, dirs, names in os.walk(root, followlinks=False):
        parent = Path(directory)
        for name in list(dirs):
            path = parent / name
            if exclude_pending and _excluded(name):
                dirs.remove(name)
            elif _linked(path):
                raise ValueError(f"Linked output directory cannot be archived: {path}")
        for name in names:
            if exclude_pending and _excluded(name):
                continue
            path = parent / name
            if _linked(path) or not path.is_file():
                raise ValueError(f"Output must contain regular files: {path}")
            if not path.resolve().is_relative_to(root.resolve()):
                raise ValueError(f"Output file escapes its root: {path}")
            result[path.relative_to(root).as_posix()] = path
    return dict(sorted(result.items()))


def _remove_tree(path, parent, protected=()):
    """Only remove a direct child of the explicitly resolved intended directory."""
    parent = Path(parent).resolve()
    path = Path(path)
    resolved = path.resolve()
    if (_linked(path) or resolved.parent != parent or resolved == parent
            or resolved in {Path(p).resolve() for p in protected}):
        raise ValueError(f"Refusing archive cleanup outside its permitted target: {path}")
    shutil.rmtree(resolved)


def _sqlite_backup(source, destination):
    source_db = sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        destination_db = sqlite3.connect(destination)
        try:
            source_db.backup(destination_db)
            # A self-contained snapshot must not depend on destination WAL files.
            destination_db.execute("PRAGMA journal_mode=DELETE")
        finally:
            destination_db.close()
    finally:
        source_db.close()
    _fsync_file(destination)


def _checkpoint_pairs(root, inventory):
    checkpoints = {name for name in inventory if PurePosixPath(name).name == "checkpoint.pt"}
    sidecars = {name for name in inventory if PurePosixPath(name).name == "checkpoint.sha256.json"}
    expected = {str(PurePosixPath(name).with_suffix(".sha256.json")) for name in checkpoints}
    if sidecars != expected:
        raise ValueError("Snapshot has an unpaired checkpoint.pt or checkpoint.sha256.json.")
    for name in sorted(checkpoints):
        sidecar = str(PurePosixPath(name).with_suffix(".sha256.json"))
        record = _read_json(root / sidecar)
        if not isinstance(record, dict) or record.get("sha256") != inventory[name]["sha256"]:
            raise ValueError(f"Checkpoint SHA256 mismatch: {name}")
    return sorted(checkpoints)


def _verify_inventory(root, inventory):
    if not isinstance(inventory, dict):
        raise ValueError("Snapshot inventory must be a mapping.")
    for name, record in inventory.items():
        relative = PurePosixPath(name)
        if (not name or relative.is_absolute() or relative.as_posix() != name
                or any(part in {".", ".."} for part in relative.parts) or "\\" in name or ":" in name):
            raise ValueError(f"Unsafe snapshot inventory path: {name}")
        if (not isinstance(record, dict) or type(record.get("size")) is not int or record["size"] < 0
                or not isinstance(record.get("sha256"), str) or not _SHA256.fullmatch(record["sha256"])):
            raise ValueError(f"Invalid snapshot inventory entry: {name}")
    actual = _files(root)
    if set(actual) != set(inventory):
        raise ValueError("Snapshot files differ from its inventory.")
    for name, path in actual.items():
        if path.stat().st_size != inventory[name]["size"] or _sha256(path) != inventory[name]["sha256"]:
            raise ValueError(f"Snapshot file checksum mismatch: {name}")
    _checkpoint_pairs(root, inventory)


def _prune(snapshots, latest):
    completed = []
    for path in snapshots.iterdir():
        if (_SNAPSHOT_ID.fullmatch(path.name) and not _linked(path)
                and path.is_dir() and (path / "snapshot.json").is_file()):
            completed.append(path)
    previous = sorted((p for p in completed if p != latest), key=lambda p: p.name, reverse=True)
    for obsolete in previous[1:]:
        _remove_tree(obsolete, snapshots, protected=(latest,))


def archive_output(output):
    """Publish one verified snapshot; call only while this seed's writer is paused.

    Returns None when disabled, otherwise the published latest.json record.
    Other seeds may archive concurrently because each has a separate namespace.
    """
    configured_root = os.environ.get("RECREATE3_ARCHIVE_ROOT")
    if not configured_root:
        return None
    output = Path(output).resolve()
    if not output.is_dir():
        raise FileNotFoundError(f"Experiment output does not exist: {output}")
    archive_root = Path(configured_root).expanduser().resolve()
    if archive_root.is_relative_to(output) or output.is_relative_to(archive_root):
        raise ValueError("Archive and live output directories must be separate.")
    seed_dir = archive_root / output.name
    snapshots = seed_dir / "snapshots"
    if _linked(seed_dir) or _linked(snapshots):
        raise ValueError("Archive seed and snapshot directories must not be links.")
    snapshots.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    snapshot_id = f"{stamp}-{uuid.uuid4().hex}"
    staging = snapshots / f".{snapshot_id}.pending"
    completed = snapshots / snapshot_id
    data = staging / "output"
    data.mkdir(parents=True)
    published = False
    try:
        inventory = {}
        for name, source in _files(output, exclude_pending=True).items():
            destination = data / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if name == "reward_cache.sqlite":
                _sqlite_backup(source, destination)
            else:
                shutil.copy2(source, destination)
                _fsync_file(destination)
            inventory[name] = {"size": destination.stat().st_size, "sha256": _sha256(destination)}
        checkpoints = _checkpoint_pairs(data, inventory)
        manifest = {"schema": SCHEMA, "seed": output.name, "snapshot": snapshot_id,
                    "created_at": datetime.now(timezone.utc).isoformat(), "files": inventory,
                    "checkpoints": checkpoints}
        _atomic_json(staging / "snapshot.json", manifest)
        os.replace(staging, completed)
        _fsync_directory(snapshots)
        latest = {"schema": SCHEMA, "snapshot": snapshot_id,
                  "manifest_sha256": _sha256(completed / "snapshot.json"),
                  "inventory_sha256": _inventory_hash(inventory), "files": len(inventory),
                  "bytes": sum(item["size"] for item in inventory.values()),
                  "created_at": manifest["created_at"]}
        _atomic_json(seed_dir / "latest.json", latest)
        published = True
        _prune(snapshots, completed)
        return latest
    finally:
        if staging.exists():
            _remove_tree(staging, snapshots)
        if not published and completed.exists():
            # Publication may have succeeded before a directory-fsync failure.
            # Never remove a snapshot that the actual pointer now references.
            pointer = seed_dir / "latest.json"
            try:
                is_latest = pointer.exists() and _read_json(pointer).get("snapshot") == snapshot_id
            except (OSError, ValueError, TypeError):
                is_latest = True  # Preserve data when publication status is uncertain.
            if not is_latest:
                _remove_tree(completed, snapshots)


def restore_output(archive_seed_dir, output):
    """Verify all archived bytes, then atomically restore into an empty output.

    Existing nonempty outputs are refused; callers may quarantine them separately.
    No checkpoint or experiment fingerprint is rewritten during restoration.
    """
    seed_dir = Path(archive_seed_dir).resolve()
    latest = _read_json(seed_dir / "latest.json")
    if not isinstance(latest, dict):
        raise ValueError("Invalid archive latest.json pointer.")
    snapshot_id = latest.get("snapshot")
    if latest.get("schema") != SCHEMA or not isinstance(snapshot_id, str) or not _SNAPSHOT_ID.fullmatch(snapshot_id):
        raise ValueError("Invalid archive latest.json pointer.")
    snapshots = seed_dir / "snapshots"
    snapshot = snapshots / snapshot_id
    if _linked(snapshots) or _linked(snapshot) or snapshot.resolve().parent != snapshots.resolve():
        raise ValueError("Snapshot pointer escapes the archive.")
    manifest_path = snapshot / "snapshot.json"
    if _linked(manifest_path) or _sha256(manifest_path) != latest.get("manifest_sha256"):
        raise ValueError("Snapshot manifest checksum mismatch.")
    manifest = _read_json(manifest_path)
    if (not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA or manifest.get("snapshot") != snapshot_id
            or manifest.get("seed") != seed_dir.name):
        raise ValueError("Snapshot identity does not match the archive.")
    inventory = manifest.get("files")
    if _inventory_hash(inventory) != latest.get("inventory_sha256"):
        raise ValueError("Snapshot inventory checksum mismatch.")
    source = snapshot / "output"
    if _linked(source) or not source.is_dir():
        raise ValueError("Snapshot output directory is missing or linked.")
    _verify_inventory(source, inventory)
    output = Path(output).expanduser()
    if _linked(output):
        raise ValueError("Restore destination must not be a link.")
    output = output.resolve()
    if output.is_relative_to(seed_dir) or seed_dir.is_relative_to(output):
        raise ValueError("Restore destination must be separate from its archive.")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Restore destination must be absent or empty; quarantine existing output first.")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.restore-{uuid.uuid4().hex}.pending"
    staging.mkdir()
    removed_empty = False
    try:
        for name in inventory:
            destination = staging / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / name, destination)
            _fsync_file(destination)
        # Also detect mutation of a supposedly immutable snapshot during copying.
        _verify_inventory(staging, inventory)
        if output.exists():
            output.rmdir()  # Succeeds only if still empty; never removes user data.
            removed_empty = True
        os.replace(staging, output)
        _fsync_directory(output.parent)
        return output
    finally:
        if staging.exists():
            _remove_tree(staging, output.parent)
        if removed_empty and not output.exists():
            output.mkdir()
