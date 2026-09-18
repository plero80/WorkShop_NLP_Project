"""Verified preparation clones and checkpoint merges for independent arm jobs.

No scientific runtime is modified. Arm jobs stop at the requested PPO budget in
pilot mode; the original runner later finalizes the merged seed. Call these
functions only after the associated worker processes have exited.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import stat
import subprocess
import sys
import tempfile
import uuid


ARMS = ("proxy", "judge", "knn_static")
ENGINE = "reward_gap_knn.shared_ppo.gsm8k.v1"
ORIGIN_SCHEMA = "recreate3.parallel_origin.v1"
MERGE_SCHEMA = "recreate3.parallel_merge.v1"
LOGS = ("judge_calls.jsonl", "invalid_judge_outputs.jsonl")
REQUIRED = (
    "config.json", "manifest.json", "resolved_assets.json", "data/splits.json",
    "initial_trainable.pt", "preflight_ppo.json", "reward_cache.sqlite",
    "prepared/complete.json", "prepared/normalization.json", "prepared/memory_initial.npz",
    "prepared/encoder_probe.json", "prepared/grading_coverage.json",
    "prepared/selection_grid.json", "prepared/selection_metrics.json",
    "prepared/selection_scored.jsonl", "prepared/calibration_raw.jsonl",
    "prepared/memory_raw.jsonl", "prepared/selection_raw.jsonl",
    "evaluations/monitor/base/step_000000/metrics.json",
    "evaluations/monitor/base/step_000000/responses.jsonl", "judge_calls.jsonl",
)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _linked(path):
    return path.is_symlink() or bool(getattr(path.lstat(), "st_file_attributes", 0) & 0x400)


def _root(path):
    path = Path(path).absolute()
    for candidate in (path, *path.parents):
        if (candidate.exists() or candidate.is_symlink()) and _linked(candidate):
            raise ValueError(f"Linked paths are not allowed: {candidate}")
    if path.exists() and not path.is_dir():
        raise ValueError(f"Expected a directory: {path}")
    return path.resolve()


def _transient(name):
    return name == "run.lock" or name.endswith((".tmp", ".pending", "-wal", "-shm"))


def _files(root):
    files = {}
    for directory, dirs, names in os.walk(root, followlinks=False):
        parent = Path(directory)
        for name in [*dirs, *names]:
            path = parent / name
            if _linked(path):
                raise ValueError(f"Linked output entry: {path}")
        dirs[:] = [name for name in dirs if not _transient(name)]
        for name in names:
            path = parent / name
            if not stat.S_ISREG(path.stat().st_mode):
                raise ValueError(f"Nonregular output entry: {path}")
            if not _transient(name):
                files[path.relative_to(root).as_posix()] = path
    return dict(sorted(files.items()))


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@contextmanager
def _paused(root):
    """Cooperate with the original runner's lock without changing its contents."""
    path = root / "run.lock"
    if not path.exists():
        yield
        return
    with path.open("r+b") as stream:
        if os.name == "nt":
            import msvcrt
            lock = lambda: msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            unlock = lambda: (stream.seek(0), msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1))
        else:
            import fcntl
            lock = lambda: fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            unlock = lambda: fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        try:
            lock()
        except OSError as error:
            raise ValueError(f"Output still has an active writer: {root}") from error
        try:
            yield
        finally:
            unlock()


def _cache_identity(path):
    connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    try:
        if connection.execute("PRAGMA quick_check").fetchone() != ("ok",):
            raise ValueError(f"Invalid SQLite cache: {path}")
        columns = [row[1] for row in connection.execute("PRAGMA table_info(scores)")]
        if columns != ["key", "result", "embedding"]:
            raise ValueError(f"Unexpected reward cache schema: {path}")
        digest, count = hashlib.sha256(), 0
        for key, result, embedding in connection.execute("SELECT key,result,embedding FROM scores ORDER BY key"):
            for value in (key.encode(), result.encode(), embedding):
                digest.update(str(-1 if value is None else len(value)).encode() + b":")
                if value is not None:
                    digest.update(value)
            count += 1
        return {"kind": "sqlite_rows", "sha256": digest.hexdigest(), "rows": count}
    finally:
        connection.close()


def _inventory(root):
    return {name: _cache_identity(path) if name == "reward_cache.sqlite" else
            {"kind": "file", "sha256": _sha(path), "size": path.stat().st_size}
            for name, path in _files(root).items()}


def _identity(root, settings):
    if _json(root / "config.json") != settings:
        raise ValueError(f"Configuration differs: {root}")
    manifest = _json(root / "manifest.json")
    identity = manifest.get("identity", {})
    if identity.get("config") != settings or _digest(identity) != manifest.get("fingerprint"):
        raise ValueError(f"Invalid experiment fingerprint: {root}")
    resolved = _json(root / "resolved_assets.json")
    if identity.get("resolved") != resolved:
        raise ValueError(f"Resolved assets differ from manifest: {root}")
    for role in ("policy", "proxy", "judge", "dataset"):
        if resolved.get(role) != settings["revisions"][role]:
            raise ValueError(f"Pinned {role} differs: {root}")
    split = _json(root / "data/splits.json")
    split_hash = _digest({k: v for k, v in split.items() if k not in ("fingerprint", "hf_revision")})
    if split.get("fingerprint") != split_hash or identity.get("split_fingerprint") != split_hash:
        raise ValueError(f"Data split fingerprint differs: {root}")
    if split.get("hf_revision") != resolved["dataset"] or split["audit"]["seed"] != settings["data_seed"]:
        raise ValueError(f"Data split provenance differs: {root}")
    seen = set()
    for name in ("calibration", "memory", "selection", "monitor", "refresh", "ppo", "final"):
        rows = split["cohorts"][name]
        ids = [row["id"] for row in rows]
        if (len(rows) != settings["dataset"][name] or len(set(ids)) != len(ids)
                or seen.intersection(ids) or split["audit"]["sizes"][name] != len(rows)):
            raise ValueError(f"Invalid {name} cohort: {root}")
        seen.update(ids)
    return manifest["fingerprint"], split


def _evaluation(root, arm, update, split):
    folder = root / "evaluations/monitor" / arm / f"step_{update:06d}"
    metrics, rows = _json(folder / "metrics.json"), _rows(folder / "responses.jsonl")
    expected_ids = [row["id"] for row in split["cohorts"]["monitor"]]
    if (metrics.get("arm") != arm or metrics.get("update") != update or metrics.get("cohort") != "monitor"
            or metrics.get("n") != len(expected_ids) or [row["id"] for row in rows] != expected_ids):
        raise ValueError(f"Invalid completed monitor evaluation: {folder}")


def _no_final(root):
    if ((root / "final_protocol.json").exists() or (root / "evaluations/final").exists()
            or (root / "generations/final").exists()):
        raise ValueError(f"Parallel arm/preparation must leave final evaluation unopened: {root}")


def preparation_ready(prepared, settings):
    """False for an incomplete prepare; contradictions/untracked training raise."""
    prepared = _root(prepared)
    if list(settings.get("arms", [])) != list(ARMS):
        raise ValueError("Parallel preparation requires the unchanged three-arm configuration.")
    if not prepared.exists():
        return False
    files = _files(prepared)
    if files and "config.json" not in files:
        raise ValueError("Existing preparation directory is untracked.")
    _no_final(prepared)
    if any(name.startswith("arms/") for name in files) or "parallel_origin.json" in files or "parallel_merge.json" in files:
        raise ValueError("Preparation output contains arm training or a different parallel role.")
    if "config.json" in files and _json(files["config.json"]) != settings:
        raise ValueError("Existing preparation configuration differs.")
    if any(name not in files or files[name].stat().st_size == 0 for name in REQUIRED):
        return False
    fingerprint, split = _identity(prepared, settings)
    if "status.json" not in files or _json(files["status.json"]).get("stage") != "prepared":
        return False
    smoke = _json(prepared / "preflight_ppo.json")
    if smoke.get("passed") is not True or smoke.get("updates_discarded") is not True:
        raise ValueError("Preparation has no successful discarded PPO smoke.")
    norm = _json(prepared / "prepared/normalization.json")
    if (not all(math.isfinite(norm[k]) for k in ("proxy_mean", "proxy_std", "judge_mean", "judge_std", "threshold"))
            or min(norm["proxy_std"], norm["judge_std"]) < settings["scoring"]["minimum_std"]):
        raise ValueError("Preparation normalization is invalid.")
    complete = _json(prepared / "prepared/complete.json")
    if not complete.get("encoder_identity") or complete.get("n_memory", 0) < max(settings["knn"]["k_grid"]):
        raise ValueError("Preparation memory is incomplete.")
    _evaluation(prepared, "base", 0, split)
    _cache_identity(prepared / "reward_cache.sqlite")
    return True


def _copy_file(source, destination, *, cache=False):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if cache:
        src = sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)
        dst = sqlite3.connect(destination)
        try:
            src.backup(dst)
            dst.execute("PRAGMA journal_mode=DELETE")
        finally:
            dst.close()
            src.close()
    else:
        shutil.copy2(source, destination)
    with destination.open("r+b") as stream:
        os.fsync(stream.fileno())


def _copy_tree(source, destination):
    for name, path in _files(source).items():
        _copy_file(path, destination / name, cache=name == "reward_cache.sqlite")


def _discard(path, parent):
    path, parent = Path(path), _root(parent)
    if _linked(path) or path.resolve().parent != parent or not path.name.startswith("."):
        raise ValueError(f"Unsafe staging cleanup: {path}")
    shutil.rmtree(path)


def _immutable(name):
    return name not in (*LOGS, "status.json", "reward_cache.sqlite") and not name.startswith("review/")


def _origin(prepared, arm, fingerprint, inventory):
    return {"schema": ORIGIN_SCHEMA, "arm": arm, "fingerprint": fingerprint,
            "preparation_id": _digest(inventory), "prepared_source": str(prepared),
            "preparation_inventory": inventory}


def _validate_clone(destination, expected, settings):
    files = _files(destination)
    if "parallel_origin.json" not in files:
        raise ValueError(f"Untracked arm output: {destination}")
    actual = _json(files["parallel_origin.json"])
    for key in ("schema", "arm", "fingerprint", "preparation_id", "preparation_inventory"):
        if actual.get(key) != expected[key]:
            raise ValueError(f"Arm output has a different preparation origin: {destination}")
    _identity(destination, settings)
    _no_final(destination)
    for name, record in expected["preparation_inventory"].items():
        if _immutable(name) and (name not in files or _sha(files[name]) != record["sha256"]):
            raise ValueError(f"Prepared input changed in arm output: {name}")
    allowed = expected["arm"]
    if any(name.startswith("arms/") and name.split("/")[1] != allowed for name in files):
        raise ValueError(f"Arm output contains another condition: {destination}")


def clone_preparation(prepared, destination, arm, settings):
    prepared, destination = _root(prepared), _root(destination)
    if arm not in ARMS or destination.is_relative_to(prepared) or prepared.is_relative_to(destination):
        raise ValueError("Arm clone must be separate from preparation and use a declared arm.")
    with _paused(prepared):
        if not preparation_ready(prepared, settings):
            raise ValueError("Preparation has not completed successfully.")
        inventory = _inventory(prepared)
        fingerprint, _ = _identity(prepared, settings)
        expected = _origin(prepared, arm, fingerprint, inventory)
        if destination.exists():
            with _paused(destination):
                _validate_clone(destination, expected, settings)
            return destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.clone-", dir=destination.parent))
        try:
            _copy_tree(prepared, staging)
            if _inventory(staging) != inventory or _inventory(prepared) != inventory:
                raise ValueError("Preparation changed during cloning.")
            _write_json(staging / "parallel_origin.json", expected)
            staging.rename(destination)
        finally:
            if staging.exists():
                _discard(staging, destination.parent)
    return destination


def _checkpoint_metadata(runtime, jobs):
    # mmap avoids materializing checkpoint tensor payloads; only scalar metadata
    # is returned. CUDA remains hidden in this short-lived validation process.
    code = ("import json,sys,torch; result={}; "
            "\nfor arm,path in json.loads(sys.argv[1]).items():"
            "\n data=torch.load(path,map_location='cpu',weights_only=True,mmap=True)"
            "\n result[arm]={k:data.get(k) for k in ('engine','step','fingerprint','arm','extra')}"
            "\nprint(json.dumps(result))")
    paths = {arm: str(path / "arms" / arm / "checkpoint.pt") for arm, path in jobs.items()}
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run([sys.executable, "-c", code, json.dumps(paths)], cwd=runtime,
                            env=env, capture_output=True, text=True, timeout=120)
    if result.returncode:
        raise ValueError("Checkpoint metadata validation failed: " + result.stderr[-2000:])
    return json.loads(result.stdout)


def _job_complete(path, arm, target, settings, origin, metadata, split):
    _validate_clone(path, origin, settings)
    status = _json(path / "status.json")
    completed = _json(path / "arms" / arm / "completed.json")
    if (status.get("stage") != "complete" or status.get("run_stage") != "pilot"
            or status.get("updates") != target or status.get("arms") != [arm]
            or status.get("skipped_arms") or _json(path / "skipped_arms.json")):
        raise ValueError(f"Arm job did not complete its pilot budget: {path}")
    successful, skipped = completed.get("successful_updates"), completed.get("skipped_updates")
    if (completed.get("update") != target or completed.get("fingerprint") != origin["fingerprint"]
            or not isinstance(successful, int) or not isinstance(skipped, int)
            or min(successful, skipped) < 0 or successful + skipped != target):
        raise ValueError(f"Invalid arm completion: {path}")
    if (metadata.get("engine") != ENGINE or metadata.get("step") != target
            or metadata.get("fingerprint") != origin["fingerprint"] or metadata.get("arm") != arm
            or metadata.get("extra", {}).get("successful_updates") != successful):
        raise ValueError(f"Checkpoint identity/budget differs from completion: {path}")
    checkpoint = path / "arms" / arm / "checkpoint.pt"
    sidecar = _json(checkpoint.with_suffix(".sha256.json"))
    if sidecar.get("engine") != ENGINE or sidecar.get("sha256") != _sha(checkpoint):
        raise ValueError(f"Checkpoint sidecar differs: {path}")
    for index in range(1, target + 1):
        step = _json(path / "arms" / arm / "training" / f"step_{index:06d}.json")
        rollout = path / "arms" / arm / "rollouts" / f"step_{index:06d}.jsonl"
        if step.get("update") != index or step.get("arm") != arm or not rollout.is_file():
            raise ValueError(f"Incomplete training history at update {index}: {path}")
    _evaluation(path, arm, target, split)


def _merge_logs(prepared, jobs, staging):
    for name in LOGS:
        source = prepared / name
        prefix = source.read_bytes() if source.exists() else b""
        if prefix and not prefix.endswith(b"\n"):
            raise ValueError(f"Incomplete preparation log: {name}")
        chunks = [prefix]
        for arm in ARMS:
            path = jobs[arm] / name
            content = path.read_bytes() if path.exists() else b""
            if not content.startswith(prefix) or (content and not content.endswith(b"\n")):
                raise ValueError(f"Worker log lost its exact preparation prefix: {arm}/{name}")
            for line in content.splitlines():
                if line.strip():
                    json.loads(line)
            chunks.append(content[len(prefix):])
        if any(chunks):
            with (staging / name).open("wb") as stream:
                stream.write(b"".join(chunks))
                stream.flush()
                os.fsync(stream.fileno())


def _merge_review(jobs, staging):
    for arm in ARMS:
        for source in sorted((jobs[arm] / "review/ungraded").glob("*.json")):
            destination = staging / "review/ungraded" / source.name
            incoming = _json(source)
            if not destination.exists():
                _copy_file(source, destination)
                continue
            current = _json(destination)
            for key in ("case_id", "role", "scorer_identity", "question_id", "question", "reference", "response"):
                if current.get(key) != incoming.get(key):
                    raise ValueError(f"Conflicting review identity: {source.name}")
            comparable = {k: v for k, v in current.items() if k not in ("stages", "parallel_variants")}
            if comparable != {k: v for k, v in incoming.items() if k != "stages"}:
                relative = Path("review/parallel_variants") / arm / source.name
                _copy_file(source, staging / relative)
                current.setdefault("parallel_variants", []).append(relative.as_posix())
            current["stages"] = sorted(set(current.get("stages", [])) | set(incoming.get("stages", [])))
            _write_json(destination, current)


def merge_seed(runtime, prepared, jobs, destination, settings, stage):
    """Merge verified pilot checkpoints; original runner owns final evaluation.

    Repeating the same completed merge preserves any subsequent original-runner
    finalization. A new target may replace a tracked pilot merge, retaining the
    prior directory, but cannot replace a seed that already opened final testing.
    """
    from contextlib import ExitStack
    if stage not in ("pilot", "full") or set(jobs) != set(ARMS):
        raise ValueError("Merge requires all three arms and a pilot/full target.")
    runtime, prepared, destination = _root(runtime), _root(prepared), _root(destination)
    jobs = {arm: _root(jobs[arm]) for arm in ARMS}
    sources = [prepared, *jobs.values()]
    if len(set(sources)) != 4 or any(destination.is_relative_to(p) or p.is_relative_to(destination) for p in sources):
        raise ValueError("Preparation, workers and canonical seed must be separate directories.")
    target = settings["ppo"]["full_updates" if stage == "full" else "pilot_updates"]
    with ExitStack() as stack:
        for source in sources:
            stack.enter_context(_paused(source))
        if not preparation_ready(prepared, settings):
            raise ValueError("Preparation is incomplete.")
        inventory = _inventory(prepared)
        fingerprint, split = _identity(prepared, settings)
        for arm, path in jobs.items():
            _validate_clone(path, _origin(prepared, arm, fingerprint, inventory), settings)
        metadata = _checkpoint_metadata(runtime, jobs)
        for arm, path in jobs.items():
            _job_complete(path, arm, target, settings, _origin(prepared, arm, fingerprint, inventory), metadata[arm], split)
        job_inventories = {arm: _inventory(path) for arm, path in jobs.items()}
        identity = {"fingerprint": fingerprint, "preparation_id": _digest(inventory), "target_updates": target,
                    "arms": list(ARMS), "source_job_hashes": {a: _digest(v) for a, v in job_inventories.items()}}
        destination_lock = ExitStack()
        stack.callback(destination_lock.close)
        if destination.exists():
            destination_lock.enter_context(_paused(destination))
            _files(destination)
            marker = destination / "parallel_merge.json"
            if not marker.is_file():
                raise ValueError(f"Refusing to replace an untracked canonical output: {destination}")
            previous = _json(marker)
            if previous.get("schema") != MERGE_SCHEMA or previous.get("identity", {}).get("fingerprint") != fingerprint:
                raise ValueError("Existing canonical output has a different experiment identity.")
            _identity(destination, settings)
            final_path = destination / "final_protocol.json"
            if final_path.exists():
                final = _json(final_path)
                if (final.get("updates") != target or final.get("arms") != list(ARMS)
                        or final.get("fingerprint") != fingerprint):
                    raise ValueError("Existing final protocol has a different arm list or budget.")
            if previous.get("identity") == identity:
                for arm in ARMS:
                    for filename in ("checkpoint.pt", "checkpoint.sha256.json", "completed.json"):
                        name = f"arms/{arm}/{filename}"
                        if not (destination / name).is_file() or _sha(destination / name) != job_inventories[arm][name]["sha256"]:
                            raise ValueError(f"Canonical checkpoint/completion changed: {name}")
                return destination
            if (destination / "final_protocol.json").exists():
                raise ValueError("Cannot replace a canonical seed after final evaluation was opened.")
            if previous["identity"].get("target_updates", target + 1) > target:
                raise ValueError("Cannot merge a lower PPO budget over an existing canonical seed.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.merge-", dir=destination.parent))
        retained = None
        try:
            _copy_tree(prepared, staging)
            for arm, path in jobs.items():
                for prefix in (f"arms/{arm}/", f"generations/monitor/{arm}/", f"evaluations/monitor/{arm}/"):
                    for name, source in _files(path).items():
                        if name.startswith(prefix):
                            _copy_file(source, staging / name)
            _merge_logs(prepared, jobs, staging)
            _merge_review(jobs, staging)
            _write_json(staging / "skipped_arms.json", {})
            _write_json(staging / "status.json", {"stage": "parallel_merged", "updates": target, "arms": list(ARMS)})
            _write_json(staging / "parallel_merge.json", {
                "schema": MERGE_SCHEMA, "identity": identity, "requested_stage": stage,
                "preparation_source": str(prepared), "preparation_inventory": inventory,
                "source_jobs": {a: str(p) for a, p in jobs.items()}, "source_inventories": job_inventories,
                "final_evaluation": "Unopened; original full runner finalizes the combined seed.",
                "reward_cache": "Independent preparation cache; per-arm caches remain in their source jobs.",
                "judge_calls": "Preparation prefix once, followed by every new event from each arm in declared order.",
            })
            if _inventory(prepared) != inventory or any(_inventory(jobs[a]) != job_inventories[a] for a in ARMS):
                raise ValueError("A merge input changed while copying.")
            # Windows cannot rename a directory while our run.lock descriptor
            # is open. The caller's launch lock serializes publication/finalizing.
            destination_lock.close()
            # Keep old canonical work recoverable across a crash between the two
            # atomic renames. Jobs and preparation are never moved or modified.
            if destination.exists():
                retained = destination.parent / f".{destination.name}.retained-{uuid.uuid4().hex}"
                destination.rename(retained)
            try:
                staging.rename(destination)
            except BaseException:
                if retained is not None and not destination.exists():
                    retained.rename(destination)
                raise
        finally:
            if staging.exists():
                _discard(staging, destination.parent)
    return destination
