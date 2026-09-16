"""Organize exact copies, verify them, or restore the original execution layout."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
MAP = ROOT / "reproducibility/file_map.json"
RESULTS = {
    "outputs": "results/followup",
    "knn_distillation_outputs": "results/distillation",
    "best_of_n_outputs": "results/best_of_n",
    "cluster_visualization": "results/exploration",
    "refresh2_outputs": "data/prerequisites/memory_refresh",
    "next_studies_outputs": "data/prerequisites/teacher_comparison",
    "knn_distillation_recovery": "data/prerequisites/distillation_recovery",
}
NOTEBOOKS = {
    "RUN_FOLLOWUP.ipynb": "followup",
    "RUN_KNN_MEMORY_DISTILLATION.ipynb": "distillation",
    "RUN_BEST_OF_N.ipynb": "best_of_n",
    "RUN_BEST_OF_N_CONFIRMATION.ipynb": "best_of_n",
    "EXPLORE_SECOND_REFRESH.ipynb": "exploration",
    "REWARD_GAP_CLUSTER_EXPLORER.ipynb": "exploration",
}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def safe(root, relative):
    relative = PurePosixPath(relative)
    if relative.is_absolute() or ".." in relative.parts or "\\" in str(relative):
        raise ValueError(f"Invalid relative path: {relative}")
    path = (root / relative).resolve()
    path.relative_to(root.resolve())
    return path


def is_artifact(name):
    path = PurePosixPath(name)
    return path.parts[0] in RESULTS or (len(path.parts) == 1 and path.suffix in {".zip", ".log"})


def destination(name):
    p = PurePosixPath(name)
    parts = p.parts
    if "__pycache__" in parts or p.suffix in {".pyc", ".pyo", ".pid", ".lock"}:
        return None
    if parts[0] in RESULTS:
        return str(PurePosixPath(RESULTS[parts[0]], *parts[1:]))
    if p.suffix == ".ipynb":
        return str(PurePosixPath("notebooks", NOTEBOOKS[p.name], p.name))
    if parts[0] == "inputs":
        return str(PurePosixPath("data", *parts))
    if p.name.startswith("test_") and p.suffix == ".py":
        family = "core" if len(parts) == 1 else (p.stem.removeprefix("test_") if parts[0] == "tests" else parts[0])
        return str(PurePosixPath("code", "tests", family, p.name))
    if p.suffix == ".py":
        return str(PurePosixPath("code", "core" if len(parts) == 1 else "experiments", *parts))
    if p.name == "review_form.html":
        return "code/templates/review_form.html"
    if p.suffix == ".md":
        return str(PurePosixPath("docs", "original", *parts))
    if p.name.startswith("requirements"):
        return str(PurePosixPath("configs", "environment", *parts))
    if "MANIFEST" in p.name or "SHA256" in p.name or p.name in {"expected_sources.json", "vendor_manifest.json", "anchor.json"}:
        return str(PurePosixPath("reproducibility", "original_manifests", *parts))
    if p.suffix == ".json":
        return str(PurePosixPath("configs", *parts))
    raise ValueError(f"Unclassified file; review before copying: {name}")


def copy_checked(source, target, row):
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        with source.open("rb") as src, target.open("xb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)
    if target.stat().st_size != row["bytes"] or digest(target) != row["sha256"]:
        raise ValueError(f"Checksum mismatch; refusing to replace or accept {target}")


def organize():
    original = ROOT.parent / "original_project"
    baseline = original / "provenance/original_inventory.json"
    removed = read(MAP).get("excluded_removed_components", []) if MAP.exists() else []
    removed_set = set(removed)
    entries, excluded = [], []
    for row in read(baseline)["files"]:
        if row["path"] in removed_set:
            continue
        target = destination(row["path"])
        if target is None:
            excluded.append(row["path"])
        else:
            entries.append({"original": row["path"], "organized": target,
                            "bytes": row["bytes"], "sha256": row["sha256"]})
    if len({row["organized"].casefold() for row in entries}) != len(entries):
        raise ValueError("Destination path collision.")
    record = {"schema_version": 1, "original_root": "reward_gap_followup",
              "baseline_sha256": digest(baseline), "files": entries,
              "excluded_caches_and_process_markers": excluded}
    if removed:
        record["excluded_removed_components"] = removed
    if MAP.exists() and read(MAP) != record:
        raise ValueError("Existing file map differs; refusing to replace it.")
    for index, row in enumerate(entries, 1):
        copy_checked(safe(original / "reward_gap_followup", row["original"]),
                     safe(ROOT, row["organized"]), row)
        if index % 5000 == 0:
            print(f"Copied and verified {index}/{len(entries)} files", flush=True)
    write(MAP, record)
    summary = {"files_copied_and_sha256_verified": len(entries),
               "original_bytes_preserved": sum(row["bytes"] for row in entries),
               "excluded_caches_and_process_markers": len(excluded),
               "by_directory": dict(Counter(PurePosixPath(row["organized"]).parts[0] for row in entries)),
               "passed": True}
    write(ROOT / "reproducibility/organization_verification.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def mapped_files():
    rows = list(read(MAP)["files"])
    additions = ROOT / "reproducibility/additions.json"
    if additions.exists():
        rows.extend(read(additions)["files"])
    for key in ("original", "organized"):
        if len({row[key].casefold() for row in rows}) != len(rows):
            raise ValueError(f"Duplicate {key} path in original/additional file maps.")
    for row in rows:
        safe(ROOT, row["original"])
        safe(ROOT, row["organized"])
    return rows


def verify(code_only=False):
    rows = [row for row in mapped_files() if not code_only or not is_artifact(row["original"])]
    errors = []
    for index, row in enumerate(rows, 1):
        path = safe(ROOT, row["organized"])
        if not path.is_file() or path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
            errors.append(row["organized"])
        if index % 10000 == 0:
            print(f"Verified {index}/{len(rows)} files", flush=True)
    print(json.dumps({"files_checked": len(rows), "changed_or_missing": errors, "passed": not errors}, indent=2))
    return not errors


def restore(target, code_only=False):
    target = target.resolve()
    if target.exists():
        raise FileExistsError("Restore requires a new directory; existing projects are never overwritten.")
    target.mkdir(parents=True)
    rows = [row for row in mapped_files() if not code_only or not is_artifact(row["original"])]
    for index, row in enumerate(rows, 1):
        copy_checked(safe(ROOT, row["organized"]), safe(target, row["original"]), row)
        if index % 10000 == 0:
            print(f"Restored {index}/{len(rows)} files", flush=True)
    print(f"Restored and checksum-verified {len(rows)} files at {target}")
    print("The original layout is ready. No experiment was launched.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("organize", "verify", "restore"))
    parser.add_argument("--destination", type=Path, help="A new directory for the original execution layout.")
    parser.add_argument("--code-only", action="store_true", help="Exclude saved results and prerequisite studies.")
    args = parser.parse_args()
    if args.action == "organize":
        organize()
    elif args.action == "verify":
        return 0 if verify(args.code_only) else 1
    else:
        if args.destination is None:
            parser.error("restore requires --destination")
        restore(args.destination, args.code_only)
    return 0


if __name__ == "__main__":
    sys.exit(main())
