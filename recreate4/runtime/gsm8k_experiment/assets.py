from __future__ import annotations

import importlib.metadata
import os
import re
import time
from pathlib import Path

from .common import ROOT, atomic_json, digest, read_json, status
from .recovery import RECOVERY_POLICY, validate_migration


MODEL_DOWNLOAD_PATTERNS = (
    "*.json", "*.jinja", "*.safetensors", "*.model", "merges.txt", "vocab.txt",
)


def required_model_roles(config):
    """Skip the optional teacher unless an arm or final evaluation needs it."""
    needs_teacher = ("knn_static_30b" in config["arms"] or
                     config.get("teacher30b", {}).get("evaluate_final", False))
    if needs_teacher and "judge30b" not in config["models"]:
        raise ValueError("The requested 30B teacher is missing from models.judge30b.")
    return [role for role in config["models"] if role != "judge30b" or needs_teacher]


def source_fingerprint():
    from .shared import shared_sources, ENGINE_ID
    package = Path(__file__).parent
    return digest({"engine": ENGINE_ID, "shared": shared_sources(),
                   "package": {p.name: digest(p.read_text(encoding="utf-8"))
                               for p in sorted(package.glob("*.py"))}})


def retry_hub(fn):
    for attempt in range(6):
        try:
            return fn()
        except Exception as e:
            response = getattr(e, "response", None)
            code = getattr(response, "status_code", None)
            if attempt == 5 or (code is not None and code not in (408, 429, 500, 502, 503, 504)):
                raise
            delay = min(30, 2**attempt)
            print(f"Download/metadata retry {attempt + 1}/6 after {type(e).__name__}; waiting {delay}s.", flush=True)
            time.sleep(delay)


def check_runtime(config):
    import torch
    device = config["runtime"]["device"]
    if not device.startswith("cuda"):
        raise ValueError("Real model experiments require a CUDA GPU. CPU is supported only by the offline tests.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use the project GPU environment with a compatible PyTorch installation.")
    torch.cuda.set_device(device)
    if config["runtime"]["dtype"] == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU does not support bfloat16. Use a supported GPU or explicitly select float32.")
    # Execute a CUDA kernel, catching unsupported Blackwell wheel builds early.
    probe = torch.randn((32, 32), device=device)
    assert torch.isfinite(probe @ probe).all()
    free, total = torch.cuda.mem_get_info(device)
    return {"gpu": torch.cuda.get_device_name(device), "free_gib": free / 2**30, "total_gib": total / 2**30,
            "torch": torch.__version__, "cuda": torch.version.cuda}


def resolve_assets(config, output):
    from huggingface_hub import HfApi, snapshot_download
    output = Path(output)
    path = output / "resolved_assets.json"
    roles = required_model_roles(config)
    offline = os.environ.get("HF_HUB_OFFLINE", "").strip().upper() in {"1", "ON", "YES", "TRUE"}
    resolved = read_json(path) if path.exists() else {}
    previous = dict(resolved)
    api = None
    for role in [*roles, "dataset"]:
        revision = config["revisions"][role]
        pinned = bool(re.fullmatch(r"[0-9a-fA-F]{40}", revision))
        if role in resolved:
            if pinned and resolved[role] != revision:
                raise ValueError(f"Saved {role} revision differs from the configured commit; use a new output directory.")
            continue
        # Commit-pinned, prewarmed runs require no Hub metadata request. Branches
        # are resolved once online and their immutable revisions are persisted.
        if pinned:
            resolved[role] = revision
        else:
            if offline:
                raise ValueError(f"Offline {role} requires a 40-character commit revision or saved resolved_assets.json.")
            if api is None:
                # No whoami request; every default repository is public.
                api = HfApi(token=False)
            if role == "dataset":
                resolved[role] = retry_hub(lambda: api.dataset_info(config["dataset"]["id"], revision=revision).sha)
            else:
                resolved[role] = retry_hub(lambda: api.model_info(config["models"][role], revision=revision).sha)
    if not path.exists() or resolved != previous:
        atomic_json(path, resolved)
    for role in roles:
        name = config["models"][role]
        status(output, "download", role=role, model=name, revision=resolved[role])
        download = lambda: snapshot_download(name, revision=resolved[role], token=False,
                    allow_patterns=list(MODEL_DOWNLOAD_PATTERNS), max_workers=2, local_files_only=offline)
        # A missing offline snapshot cannot become available through backoff.
        download() if offline else retry_hub(download)
    return resolved


def bind_experiment(config, output, resolved, split, runtime):
    import torch
    output = Path(output)
    versions = {name: importlib.metadata.version(name) for name in
                ("transformers", "peft", "accelerate", "datasets", "huggingface-hub", "numpy")}
    identity = {"config": config, "resolved": resolved, "split_fingerprint": split["fingerprint"],
                "source": source_fingerprint(), "versions": versions, "torch_version": torch.__version__.split("+")[0]}
    fingerprint = digest(identity)
    manifest = output / "manifest.json"
    validate_migration(output, identity)
    if manifest.exists() and read_json(manifest)["fingerprint"] != fingerprint:
        raise ValueError("Configuration, data, shared engine, models or library versions changed. Use a new output directory.")
    if not manifest.exists():
        atomic_json(manifest, {"fingerprint": fingerprint, "identity": identity, "runtime_at_creation": runtime,
                               "created_at": time.time(), "protocol": "GSM8K generative judges with project shared PPO engine v1; adapted protocol, not ZIP checkpoint compatible"})
    return fingerprint
