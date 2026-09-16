"""Install nonblocking grading in an existing, stopped GSM8K runtime.

Audits exact source versions, preserves generations/cache, and rebinds checkpoint
identity metadata without changing tensors, optimizer state, step or RNG state.
Original checkpoints and manifests remain in source_amendments/ungraded_review_v1/.
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from repair_gsm8k_inline_score import PROJECT, atomic_bytes, atomic_json, digest, file_hash, lock, read


def upgrade(runtime, output):
    runtime, output = Path(runtime).resolve(), Path(output).resolve()
    output.relative_to(runtime)
    patch = read(PROJECT / "reproducibility/gsm8k_ungraded_patch.json")
    package = runtime / "gsm8k_experiment"
    source = PROJECT / "code/experiments/gsm8k_experiment"
    if not (output / "manifest.json").exists():
        raise ValueError("No existing experiment manifest. Restore a new runtime for a new experiment.")
    with lock(output):
        # Validate everything before replacing runtime sources or checkpoints.
        if {p.name: file_hash(p) for p in source.glob("*.py")} != patch["after"]:
            raise ValueError("Checkout sources differ from the reviewed upgrade.")
        for name, expected in patch["shared_sources"].items():
            if file_hash(runtime / name) != expected:
                raise ValueError(f"Shared core changed: {name}")
        if read(package / "shared_sources.json") != patch["shared_sources"]:
            raise ValueError("Shared source pins changed.")
        manifest_path = output / "manifest.json"
        current = read(manifest_path)
        if digest(current["identity"]) != current["fingerprint"]:
            raise ValueError("Invalid experiment manifest fingerprint.")
        audit_path = output / "source_amendments" / patch["id"] / "upgrade.json"
        audit = read(audit_path) if audit_path.exists() else None
        if audit is not None and audit["patch"] != patch:
            raise ValueError("Upgrade audit does not match this patch.")
        parent = audit["parent_manifest"] if audit else current
        if digest(parent["identity"]) != parent["fingerprint"]:
            raise ValueError("Invalid parent manifest fingerprint.")
        old = patch["ancestors"].get(parent["identity"]["source"])
        if old is None:
            raise ValueError("Unrecognized parent source version. No upgrade applied.")
        if read(output / "config.json") != parent["identity"]["config"]:
            raise ValueError("Run configuration changed.")
        identity = dict(parent["identity"], source=patch["source_after"])
        fingerprint = digest(identity)
        updated = {**parent, "identity": identity, "fingerprint": fingerprint,
                   "source_amendments": [*parent.get("source_amendments", []), patch["id"]]}
        if current not in (parent, updated):
            raise ValueError("Run manifest changed outside this upgrade.")
        actual = {p.name: file_hash(p) for p in package.glob("*.py")}
        if set(actual) - set(patch["after"]) or set(old) - set(actual):
            raise ValueError("Unrecognized runtime package file list.")
        for name, sha in actual.items():
            if sha not in (old.get(name), patch["after"][name]):
                raise ValueError(f"Unrecognized runtime source change: {name}")
        if any((output / "recovery").glob("*/migration.json")):
            raise ValueError("Standalone PPO migrations are not supported.")
        # This upgrade does not change grading prompts, cache keys or shared PPO.
        checkpoints = sorted((output / "arms").glob("*/checkpoint.pt"))
        if checkpoints:
            import torch
        plans = dict(audit.get("checkpoints", {})) if audit else {}
        for path in checkpoints:
            relative = path.relative_to(output).as_posix()
            data = torch.load(path, map_location="cpu", weights_only=True)
            info = read(path.with_suffix(".sha256.json"))
            sha = file_hash(path)
            entry = plans.get(relative)
            if data.get("engine") != patch["engine"] or data.get("arm") != path.parent.name:
                raise ValueError(f"Checkpoint engine/arm mismatch: {relative}")
            if info.get("engine") != patch["engine"]:
                raise ValueError(f"Checkpoint sidecar engine mismatch: {relative}")
            if audit and audit["status"] == "complete" and info.get("sha256") != sha:
                raise ValueError(f"Checkpoint checksum mismatch: {relative}")
            if entry and info.get("sha256") not in (entry["before_sha256"], entry.get("after_sha256"), sha):
                raise ValueError(f"Checkpoint sidecar changed during upgrade: {relative}")
            if entry:
                if sha not in (entry["before_sha256"], entry.get("after_sha256")):
                    # After an already completed upgrade, continued training can replace the checkpoint.
                    if not (audit["status"] == "complete" and info.get("sha256") == sha and data["fingerprint"] == fingerprint):
                        raise ValueError(f"Checkpoint changed during upgrade: {relative}")
                elif sha == entry["before_sha256"] and data["fingerprint"] != parent["fingerprint"]:
                    raise ValueError(f"Checkpoint parent mismatch: {relative}")
                elif sha == entry.get("after_sha256") and data["fingerprint"] != fingerprint:
                    raise ValueError(f"Upgraded checkpoint identity mismatch: {relative}")
            else:
                if info.get("sha256") != sha or data["fingerprint"] != parent["fingerprint"]:
                    # An arm can first be trained after an already completed upgrade.
                    if not (audit and audit["status"] == "complete" and info.get("sha256") == sha and data["fingerprint"] == fingerprint):
                        raise ValueError(f"Checkpoint checksum or identity mismatch: {relative}")
                else:
                    plans[relative] = {"before_sha256": sha, "step": data["step"]}
            del data
        if audit and audit["status"] == "complete":
            if current != updated or actual != patch["after"]:
                raise ValueError("Completed upgrade sources/manifest changed.")
            print("Nonblocking grading is already installed. Resume the same experiment.")
            return
        if audit is None:
            audit = {"patch": patch, "parent_manifest": parent, "new_fingerprint": fingerprint,
                     "status": "installing", "created_at": time.time(), "checkpoints": plans,
                     "protocol_change": "Missing grades are reviewed and excluded; empty PPO batches skip an attempt.",
                     "preserved": "generated answers, valid scores, checkpoint tensors, optimizer, steps and RNG"}
        atomic_json(audit_path, audit)
        for path in checkpoints:
            relative = path.relative_to(output).as_posix()
            entry = plans[relative]
            backup = audit_path.parent / "original_checkpoints" / relative
            if not backup.exists():
                if file_hash(path) != entry["before_sha256"]:
                    raise ValueError("Original checkpoint backup missing.")
                atomic_bytes(backup, path.read_bytes())
            if not backup.with_suffix(".sha256.json").exists():
                if file_hash(path) != entry["before_sha256"]:
                    raise ValueError("Original checkpoint sidecar backup missing.")
                atomic_bytes(backup.with_suffix(".sha256.json"), path.with_suffix(".sha256.json").read_bytes())
            if file_hash(backup) != entry["before_sha256"]:
                raise ValueError("Original checkpoint backup changed.")
            # Use the immutable original even if an earlier attempt stopped between writes.
            data = torch.load(backup, map_location="cpu", weights_only=True)
            data["fingerprint"] = fingerprint
            stream = io.BytesIO()
            torch.save(data, stream)
            import hashlib
            payload = stream.getvalue()
            entry["after_sha256"] = hashlib.sha256(payload).hexdigest()
            audit["checkpoints"] = plans
            atomic_json(audit_path, audit)
            atomic_bytes(path, payload)
            atomic_json(path.with_suffix(".sha256.json"), {"engine": patch["engine"], "sha256": entry["after_sha256"]})
            del data, stream, payload
        for name in patch["after"]:
            atomic_bytes(package / name, (source / name).read_bytes())
        atomic_json(manifest_path, updated)
        audit["status"] = "complete"
        atomic_json(audit_path, audit)
        print("Nonblocking grading installed. Saved answers, grades and training state were preserved.")
        print(f"Review queue: {output / 'review/ungraded'}")
        print("Resume with: python -m experiment_cli run gsm8k-b200")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, default=Path("run/gsm8k"))
    parser.add_argument("--output", type=Path, default=Path("gsm8k_outputs/b200"), help="Existing output, relative to runtime")
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else args.runtime / args.output
    try:
        upgrade(args.runtime, output)
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
