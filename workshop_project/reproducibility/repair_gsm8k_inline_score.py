"""Install the audited inline-grade parser fix in a failed, pre-training runtime.

Run from the Git checkout. This preserves generated responses and valid cache
entries, records the exact source amendment, and updates the run's source identity.
It does not launch training or relax checkpoint identity checks.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time

PROJECT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_bytes(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".")
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path, value):
    atomic_bytes(path, (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode())


@contextmanager
def lock(output):
    with (output / "run.lock").open("a+b") as stream:
        if os.name == "nt":
            import msvcrt
            stream.seek(0, os.SEEK_END)
            if stream.tell() == 0:
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            acquire = lambda: msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            release = lambda: (stream.seek(0), msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1))
        else:
            import fcntl
            acquire = lambda: fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            release = lambda: fcntl.flock(stream, fcntl.LOCK_UN)
        try:
            acquire()
        except OSError as error:
            raise ValueError("The experiment is still running. Wait for it to exit before repair.") from error
        try:
            yield
        finally:
            release()


def repair(runtime, output):
    runtime, output = Path(runtime).resolve(), Path(output).resolve()
    output.relative_to(runtime)
    if not output.is_dir():
        raise ValueError(f"Existing failed output directory not found: {output}")
    patch = read(PROJECT / "reproducibility/gsm8k_inline_score_patch.json")
    package = runtime / "gsm8k_experiment"
    with lock(output):
        expected = patch["before"]
        if {p.name for p in package.glob("*.py")} != set(expected):
            raise ValueError("Runtime package file list differs from the audited integration.")
        actual = {name: file_hash(package / name) for name in expected}
        for name, old in expected.items():
            if actual[name] not in (old, patch["after"][name]):
                raise ValueError(f"Unrecognized source change: {name}. Nothing was patched.")
        for name, expected_hash in patch["shared_sources"].items():
            if file_hash(runtime / name) != expected_hash:
                raise ValueError(f"Shared core changed: {name}. Nothing was patched.")
        if read(package / "shared_sources.json") != patch["shared_sources"]:
            raise ValueError("Shared source pins changed.")
        replacements = {}
        for name in patch["changed_files"]:
            source = PROJECT / "reproducibility/patch_payloads/grading_inline_score_v1" / name
            if file_hash(source) != patch["after"][name]:
                raise ValueError(f"Checkout patch source differs: {name}")
            replacements[name] = source.read_bytes()
        manifest_path = output / "manifest.json"
        manifest = read(manifest_path)
        if digest(manifest["identity"]) != manifest["fingerprint"]:
            raise ValueError("Invalid existing manifest fingerprint.")
        if read(output / "config.json") != manifest["identity"]["config"]:
            raise ValueError("Run configuration changed.")
        audit_path = output / "source_amendments" / (patch["id"] + ".json")
        if manifest["identity"]["source"] == patch["source_after"]:
            if actual != patch["after"] or not audit_path.exists():
                raise ValueError("Incomplete or unrecognized source amendment.")
            audit = read(audit_path)
            if audit.get("patch") != patch or audit.get("new_fingerprint") != manifest["fingerprint"]:
                raise ValueError("Source amendment record mismatch.")
            if audit.get("status") != "complete":
                audit["status"] = "complete"
                atomic_json(audit_path, audit)
            print("Inline-score repair is already applied. Resume the same output directory.")
            return
        if manifest["identity"]["source"] != patch["source_before"]:
            raise ValueError("This run is not from the audited parent source version.")
        if (output / "arms").exists() or (output / "final_protocol.json").exists() or (output / "prepared/complete.json").exists():
            raise ValueError("This repair only supports failed initial memory preparation, before training.")
        status = read(output / "status.json")
        if status.get("stage") != "failed":
            raise ValueError("The recorded run has not failed; inspect its state first.")
        evidence = None
        with (output / "invalid_judge_outputs.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    evidence = json.loads(line)
        if (not evidence or evidence.get("stage") not in ("calibration", "memory", "selection")
                or evidence.get("length_capped") is not False
                or not re.fullmatch(r"\s*Judgement:\s*Correctness_score\s*:\s*[1-5]\s*[.!]?\s*", evidence.get("judge_output", ""), re.I)):
            raise ValueError("Saved failed reply does not match this specific inline-score repair.")
        identity = dict(manifest["identity"], source=patch["source_after"])
        fingerprint = digest(identity)
        audit = {"patch": patch, "parent_manifest": manifest, "new_fingerprint": fingerprint,
                 "created_at": time.time(), "status": "installing",
                 "question_id": evidence.get("question_id"), "valid_scores_and_generations": "unchanged"}
        if audit_path.exists():
            saved = read(audit_path)
            if saved.get("patch") != patch or saved.get("parent_manifest") != manifest:
                raise ValueError("Existing amendment record conflicts with this repair.")
            audit = saved
        else:
            atomic_json(audit_path, audit)
        for name, data in replacements.items():
            atomic_bytes(package / name, data)
        updated = {**manifest, "identity": identity, "fingerprint": fingerprint,
                   "source_amendments": [*manifest.get("source_amendments", []), patch["id"]]}
        atomic_json(manifest_path, updated)
        audit["status"] = "complete"
        atomic_json(audit_path, audit)
        print("Applied audited inline-score repair. Generated responses and cached valid scores were preserved.")
        print("Resume with: python -m experiment_cli run gsm8k-b200")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, default=Path("run/gsm8k"))
    parser.add_argument("--output", type=Path, help="Existing output, relative to the runtime; default gsm8k_outputs/b200")
    args = parser.parse_args()
    output = args.output or Path("gsm8k_outputs/b200")
    if not output.is_absolute():
        output = args.runtime / output
    try:
        repair(args.runtime, output)
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
