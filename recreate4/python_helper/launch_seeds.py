#!/usr/bin/env python3
"""Schedule independent seeds or one seed's three reward arms across GPUs.

Only the standard library is imported here. CUDA discovery and recipe resolution
run in short child processes; the coordinator never creates a CUDA context.
Output and log locations are supplied by the caller (for example shared scratch).
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time

from memory_profiles import profile_for, validate_settings


ARMS = ["proxy", "judge", "knn_static"]
STAGES = {"prepare": 0, "pilot": 1, "full": 2}
MIN_GPU_BYTES = 22 * 1024**3
TOKEN = re.compile(r"(?:\d+|GPU-[A-Za-z0-9-]+|MIG-[A-Za-z0-9/_.-]+)\Z")
PROBE = r"""
import json, torch
devices = []
for i in range(torch.cuda.device_count()):
    with torch.cuda.device(i):
        p = torch.cuda.get_device_properties(i)
        devices.append({'name': p.name, 'total_memory': p.total_memory,
                        'bf16': bool(p.major >= 8 and torch.cuda.is_bf16_supported())})
print(json.dumps({'devices': devices}))
"""


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def tokens(value, label="GPU tokens"):
    result = [part.strip() for part in value.split(",")]
    if not result or any(not p for p in result) or len(set(result)) != len(result):
        raise ValueError(f"{label} must be a nonempty comma-separated list without duplicates.")
    return result


def probe_gpus(environment):
    try:
        result = subprocess.run([sys.executable, "-u", "-c", PROBE], env=environment,
                                text=True, capture_output=True, timeout=45, check=False)
    except subprocess.TimeoutExpired as error:
        raise ValueError("CUDA discovery timed out in the isolated probe process.") from error
    if result.returncode:
        raise ValueError("CUDA probe failed: " + result.stderr.strip()[-2000:])
    try:
        return json.loads(result.stdout)["devices"]
    except (ValueError, KeyError, TypeError) as error:
        raise ValueError("CUDA probe returned invalid device metadata.") from error


def discover_gpus(environment, requested=None, dry_run=False, probe=None, validate_devices=True):
    """Return native visibility tokens, not device indices inside a masked worker.

    CUDA_VISIBLE_DEVICES is authoritative. Slurm allocation IDs are used only
    when CUDA_VISIBLE_DEVICES is absent; explicit selections cannot widen either
    allocation. Every actual worker receives one token and therefore uses cuda:0.
    """
    probe = probe or probe_gpus
    mask = None
    mask_source = None
    for name in ("CUDA_VISIBLE_DEVICES", "SLURM_STEP_GPUS", "SLURM_JOB_GPUS"):
        if name in environment:
            raw = environment[name].strip()
            if not raw or raw == "-1":
                if name == "CUDA_VISIBLE_DEVICES":
                    raise ValueError("CUDA_VISIBLE_DEVICES explicitly hides all GPUs.")
                continue
            mask, mask_source = tokens(raw), name
            break
    selected = tokens(requested) if requested is not None else mask
    if selected is not None and any(not TOKEN.fullmatch(t) for t in selected):
        raise ValueError("GPU selection must use numeric CUDA IDs or GPU/MIG UUID tokens.")
    if mask is not None and selected is not None and not set(selected) <= set(mask):
        raise ValueError(f"--gpus must select tokens from {mask_source}={','.join(mask)}; "
                         "it cannot expand the allocated visibility mask.")
    if dry_run:
        if selected is None:
            raise ValueError("A CUDA-free dry run requires --gpus or an explicit CUDA/Slurm visibility mask.")
        return selected, None
    probe_env = dict(environment)
    if selected is not None:
        probe_env["CUDA_VISIBLE_DEVICES"] = ",".join(selected)
    devices = probe(probe_env)
    if not devices:
        raise ValueError("No CUDA GPUs are visible to this allocation.")
    if selected is None:
        selected = [str(i) for i in range(len(devices))]
    if len(devices) != len(selected):
        raise ValueError("CUDA probe count does not match selected visibility tokens; "
                         "check the CUDA/Slurm allocation before starting.")
    if validate_devices:
        validate_gpu_capabilities(selected, devices)
    return selected, devices


def validate_gpu_capabilities(selected, devices, vram_gb=24):
    minimum = profile_for(vram_gb)["min_gpu_bytes"]
    for token, device in zip(selected, devices):
        if not device.get("bf16"):
            raise ValueError(f"GPU {token} does not report BF16 support.")
        if device.get("total_memory", 0) < minimum:
            raise ValueError(f"GPU {token} has less than {minimum / 1024**3:g} GiB total memory; "
                             f"the selected profile targets nominal {vram_gb} GB GPUs.")


def select_seeds(count, base_seed=42, explicit=None):
    try:
        seeds = [int(x) for x in tokens(explicit, "Seeds")] if explicit is not None else list(range(base_seed, base_seed + count))
    except ValueError as error:
        raise ValueError("Seeds must be distinct integers in [0, 2**32).") from error
    if len(seeds) != count:
        raise ValueError("Exactly one seed is required per selected GPU; select fewer GPUs with --gpus if needed.")
    if len(set(seeds)) != len(seeds) or any(not 0 <= s < 2**32 for s in seeds):
        raise ValueError("Seeds must be distinct integers in [0, 2**32).")
    return seeds


def resolve_recipe(runtime, name):
    path = Path(name).expanduser()
    candidates = [path, runtime / path]
    if path.name == name and not path.suffix:
        candidates.append(runtime / "experiment_cli" / "presets" / (name + ".yaml"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise ValueError(f"Recipe not found: {name}")


def runtime_fingerprint(runtime):
    """Hash shipped sources/configs, excluding the CLI's generated run configs."""
    paths = list(runtime.glob("*.py"))
    for package in ("experiment_cli", "gsm8k_experiment"):
        paths.extend(p for p in (runtime / package).rglob("*")
                     if p.is_file() and p.suffix in (".py", ".json", ".yaml", ".yml")
                     and "__pycache__" not in p.parts)
    if not (runtime / "experiment_cli" / "__main__.py").is_file():
        raise ValueError("--runtime must contain the original experiment_cli package.")
    h = hashlib.sha256()
    for path in sorted(set(paths)):
        h.update(path.relative_to(runtime).as_posix().encode())
        h.update(b"\0")
        h.update(hashlib.sha256(path.read_bytes()).digest())
    return h.hexdigest()


def cli_command(runtime, recipe, output, stage, seed, action="run"):
    return [sys.executable, "-u", "-m", "experiment_cli", action, str(recipe),
            "--output", str(output), "--stage", stage, "--set", f"seed={seed}",
            "--set", "data_seed=42"]


def worker_command(runtime, recipe, output, stage, seed, vram_gb=24, arm=None, platform_name=None):
    """Keep runtime identity intact while installing Windows I/O retries in-process."""
    platform_name = os.name if platform_name is None else platform_name
    profiled = vram_gb != 24 or platform_name == "nt"
    if not profiled and arm is None:
        return cli_command(runtime, recipe, output, stage, seed)
    helper = "run_profile.py" if profiled else "run_arm.py"
    command = [sys.executable, "-u", str(Path(__file__).with_name(helper).resolve()),
               "--runtime", str(runtime), "--recipe", str(recipe), "--output", str(output),
               "--seed", str(seed), "--stage", stage]
    if profiled:
        command.extend(["--vram-gb", str(vram_gb)])
    if arm is not None:
        command.extend(["--arm", arm])
    return command


def memory_profile_identity(vram_gb):
    helper_names = ("memory_profiles.py", "run_profile.py", "gpu_residency.py", "windows_io.py")
    return {"version": 1, "profile": profile_for(vram_gb),
            "helper_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                              for name in helper_names}}


def resolve_settings(runtime, recipe, output, stage, seed, environment):
    """Ask the original CLI to parse/validate its own recipe without any writes."""
    child_env = dict(environment)
    child_env["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(cli_command(runtime, recipe, output, stage, seed, "show"),
                            cwd=runtime, env=child_env, capture_output=True, text=True,
                            timeout=45, check=False)
    if result.returncode:
        raise ValueError("Original CLI recipe validation failed: " + result.stderr.strip()[-2000:])
    try:
        plan = json.loads(result.stdout)
        settings = plan["settings"]
    except (ValueError, KeyError, TypeError) as error:
        raise ValueError("Original CLI show returned invalid JSON settings.") from error
    if plan.get("seeds") is not None:
        raise ValueError("The recipe must not contain top-level seeds: each worker runs one seed, not a nested suite.")
    if settings.get("seed") != seed or settings.get("data_seed") != 42:
        raise ValueError("Original CLI did not preserve the requested seed and common data_seed=42.")
    if settings.get("arms") != ARMS:
        raise ValueError("The recipe must run exactly these arms in order: proxy, judge, knn_static.")
    return settings


def make_plan(args, environment, probe=None, resolver=None):
    runtime = Path(args.runtime).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    log_root = Path(args.log_root).expanduser().resolve()
    state_root = Path(args.state_root or output_root).expanduser().resolve()
    vram_gb = getattr(args, "vram_gb", 24)
    memory_profile = profile_for(vram_gb)
    recipe_name = args.recipe or str(Path(__file__).resolve().parents[1] / "configs" / f"gsm8k-{vram_gb}gb.yaml")
    recipe = resolve_recipe(runtime, recipe_name)
    fingerprint = runtime_fingerprint(runtime)
    gpu_tokens, devices = discover_gpus(environment, args.gpus, args.dry_run, probe, validate_devices=False)
    parallelism = getattr(args, "parallelism", "auto")
    if parallelism == "auto":
        saved = state_root / "launch_manifest.json"
        if saved.is_file():
            parallelism = json.loads(saved.read_text(encoding="utf-8"))["identity"].get("parallelism", "seeds")
        else:
            parallelism = "arms" if len(gpu_tokens) > 2 else "seeds"
    if parallelism not in ("arms", "seeds"):
        raise ValueError("Unknown scheduling mode in the launch manifest.")
    if parallelism == "arms" and len(gpu_tokens) < len(ARMS):
        raise ValueError("Arm parallelism needs at least three selected GPUs, one per reward arm.")
    if devices is not None:
        scheduled_count = len(ARMS) if parallelism == "arms" else len(gpu_tokens)
        validate_gpu_capabilities(gpu_tokens[:scheduled_count], devices[:scheduled_count], vram_gb)
    seed_count = 1 if parallelism == "arms" else len(gpu_tokens)
    if parallelism == "arms" and args.seeds is not None and len(tokens(args.seeds, "Seeds")) != 1:
        raise ValueError("Arm parallelism runs exactly one seed; pass one --seeds value or use --parallelism seeds.")
    seeds = select_seeds(seed_count, args.base_seed, args.seeds)
    resolver = resolver or resolve_settings
    workers = []
    settings_by_seed = {}
    for seed, token in zip(seeds, gpu_tokens):
        output = output_root / f"seed_{seed}"
        if output.is_symlink():
            raise ValueError(f"Seed output must not be a symlink: {output}")
        settings_by_seed[str(seed)] = resolver(runtime, recipe, output, args.stage, seed, environment)
        validate_settings(settings_by_seed[str(seed)], vram_gb)
        workers.append({"seed": seed, "gpu_token": token, "output": str(output),
                        "log": str(log_root / f"seed_{seed}.log"),
                        "command": worker_command(runtime, recipe, output, args.stage, seed, vram_gb)})
    identity = {"version": 1, "seeds": seeds, "workers": len(workers),
                "one_visible_gpu_per_worker": True, "data_seed": 42, "arms": ARMS,
                "recipe_sha256": hashlib.sha256(recipe.read_bytes()).hexdigest(),
                "runtime_sha256": fingerprint, "settings_by_seed": settings_by_seed}
    plan = {"runtime": str(runtime), "recipe": str(recipe), "output_root": str(output_root),
            "log_root": str(log_root), "state_root": str(state_root),
            "stage": args.stage, "identity": identity, "workers": workers,
            "memory_profile": memory_profile,
            "parallelism": parallelism, "unused_gpu_tokens": [],
            "gpu_probe": devices, "gpu_validation": "skipped (dry run)" if args.dry_run else "passed"}
    if os.name == "nt":
        plan["io_policy"] = {"name": "windows_atomic_replace_retry_v1",
                             "helper_sha256": hashlib.sha256(Path(__file__).with_name("windows_io.py").read_bytes()).hexdigest()}
    if parallelism == "arms":
        seed = seeds[0]
        canonical = workers[0]
        prepared = output_root / f"seed_{seed}_prepare"
        if prepared.is_symlink():
            raise ValueError(f"Preparation output must not be a symlink: {prepared}")
        preparation = {**canonical, "output": str(prepared), "label": f"seed {seed} preparation",
                       "log": str(log_root / f"seed_{seed}_prepare.log"),
                       "command": worker_command(runtime, recipe, prepared, "prepare", seed, vram_gb)}
        arm_workers = []
        for arm, token in zip(ARMS, gpu_tokens[:3]):
            destination = output_root / f"seed_{seed}_{arm}"
            if destination.is_symlink():
                raise ValueError(f"Arm output must not be a symlink: {destination}")
            arm_workers.append({"seed": seed, "arm": arm, "gpu_token": token,
                "output": str(destination), "log": str(log_root / f"seed_{seed}_{arm}.log"),
                "label": f"seed {seed} {arm}",
                "command": worker_command(runtime, recipe, destination,
                            "pilot" if args.stage == "prepare" else args.stage, seed, vram_gb, arm),
                "scheduled": args.stage != "prepare"})
        # Helpers are outside runtime so existing sequential runs retain their
        # source identity. New parallel runs additionally freeze this orchestration.
        helper_names = ("launch_seeds.py", "parallel_state.py", "run_arm.py", "windows_job.py",
                        "run_profile.py", "windows_io.py")
        helper_hashes = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                         for name in helper_names}
        identity.update(version=2, parallelism="arms", workers=3,
                        orchestration_sha256=hashlib.sha256(json.dumps(helper_hashes, sort_keys=True).encode()).hexdigest())
        plan.update(workers=arm_workers, preparation_workers=[preparation],
                    finalization_workers=[{**canonical, "label": f"seed {seed} evaluation",
                        "log": str(log_root / f"seed_{seed}_finalize.log")}],
                    unused_gpu_tokens=gpu_tokens[3:])
    if vram_gb != 24:
        identity.update(version=3, memory_profile=memory_profile_identity(vram_gb))
    return plan


def check_manifest(plan):
    root = Path(plan.get("state_root", plan["output_root"]))
    path = root / "launch_manifest.json"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old.get("identity") != plan["identity"]:
            raise ValueError("Existing launch manifest differs in seeds, topology, VRAM profile, recipe, source, or resolved settings. "
                             "Use the original configuration to resume, or a new output root for a new experiment.")
        if STAGES[plan["stage"]] < STAGES.get(old.get("highest_stage"), 99):
            raise ValueError("Resume stages may advance prepare -> pilot -> full, but cannot move backwards.")
        return {**old, "highest_stage": plan["stage"], "last_started_at": utc_now(),
                **({"io_policy": plan["io_policy"]} if "io_policy" in plan else {})}
    if root.exists() and any(p.name not in {"launcher.lock", ".pipeline.lock"} for p in root.iterdir()):
        raise ValueError("Nonempty state root has no launch manifest; refusing to adopt or overwrite untracked runs.")
    outputs = Path(plan["output_root"])
    if outputs != root and outputs.exists() and any(outputs.iterdir()):
        raise ValueError("Nonempty output root has no launch manifest; refusing to adopt untracked runs.")
    return {"identity": plan["identity"], "highest_stage": plan["stage"], "created_at": utc_now(),
            **({"io_policy": plan["io_policy"]} if "io_policy" in plan else {}),
            "last_started_at": utc_now(), "initial_gpu_assignment":
            {(f"{w['seed']}/{w['arm']}" if "arm" in w else str(w["seed"])): w["gpu_token"]
             for w in plan["workers"]}}


def atomic_json(path, value):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def exclusive_lock(root):
    """OS lock is released automatically on a crash; retain its inode on disk."""
    root.mkdir(parents=True, exist_ok=True)
    stream = (root / "launcher.lock").open("a+b")
    try:
        if os.name == "posix":
            import fcntl
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                raise ValueError(f"Another launcher owns {root}") from error
        else:
            import msvcrt
            if stream.seek(0, os.SEEK_END) == 0:
                stream.write(b" "); stream.flush()
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise ValueError(f"Another launcher owns {root}") from error
        yield
    finally:
        stream.close()


def stop_workers(workers, grace=20.0):
    """Terminate owned Linux sessions or Windows jobs, including descendants."""
    def send(worker, force=False):
        process = worker["process"]
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL if force else signal.SIGTERM)
            elif worker.get("job") is not None:
                worker["job"].close()
            elif process.poll() is None:
                process.kill() if force else process.terminate()
        except ProcessLookupError:
            pass
    for worker in workers:
        send(worker)
    deadline = time.monotonic() + grace
    while any(w["process"].poll() is None for w in workers) and time.monotonic() < deadline:
        time.sleep(.1)
    # Kill groups even if a CLI leader exited; descendants may still be alive.
    for worker in workers:
        send(worker, True)
    for worker in workers:
        worker["process"].wait()


def run_workers(plan, environment, stop_event=None, poll_seconds=.2, grace=20.0):
    stop_event = stop_event or threading.Event()
    workers = []
    console_lock = threading.Lock()
    root = Path(plan.get("state_root", plan["output_root"]))
    status_path = root / "launch_status.json"
    started = utc_now()
    result = 0
    signal_received = []
    old_handlers = {}

    def emit(message):
        with console_lock:
            print(message, flush=True)

    def interrupted(number, _frame):
        signal_received.append(number)
        stop_event.set()

    def tee(worker):
        try:
            for line in worker["process"].stdout:
                worker["stream"].write(line)
                worker["stream"].flush()
                emit(f"[{worker.get('label', 'seed ' + str(worker['seed']))}] {line.rstrip()}")
        finally:
            worker["process"].stdout.close()

    def save_status(state):
        atomic_json(status_path, {"state": state, "started_at": started, "updated_at": utc_now(),
                    **({"phase": plan["phase"], "parallelism": "arms"} if "phase" in plan else {}),
                    "stage": plan["stage"], "workers": [{"seed": w["seed"], "gpu_token": w["gpu_token"],
                    **({"arm": w["arm"]} if "arm" in w else {}),
                    "pid": w["process"].pid, "returncode": w["process"].poll(),
                    "log": w["log"], "output": w["output"]} for w in workers]})

    try:
        if threading.current_thread() is threading.main_thread():
            for number in (signal.SIGINT, signal.SIGTERM):
                old_handlers[number] = signal.signal(number, interrupted)
        Path(plan["log_root"]).mkdir(parents=True, exist_ok=True)
        for spec in plan["workers"]:
            if stop_event.is_set():
                break
            env = dict(environment)
            env.update(CUDA_VISIBLE_DEVICES=spec["gpu_token"], PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
            stream = Path(spec["log"]).open("a", encoding="utf-8", buffering=1)
            stream.write(f"\n=== {utc_now()} seed={spec['seed']} GPU={spec['gpu_token']} stage={plan['stage']} ===\n")
            job = None
            try:
                options = dict(cwd=plan["runtime"], env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                               errors="replace", bufsize=1)
                if os.name == "nt":
                    from windows_job import start_worker
                    process, job = start_worker(spec["command"], **options)
                else:
                    process = subprocess.Popen(spec["command"], **options, start_new_session=True)
            except BaseException:
                stream.close()
                raise
            worker = {**spec, "process": process, "job": job, "stream": stream, "reported": False}
            workers.append(worker)
            emit(f"[{spec.get('label', 'seed ' + str(spec['seed']))}] GPU token {spec['gpu_token']} -> cuda:0; "
                 f"log: {spec['log']}; output: {spec['output']}")
            worker["thread"] = threading.Thread(target=tee, args=(worker,), daemon=True)
            worker["thread"].start()
        save_status("running")
        while True:
            if stop_event.is_set():
                result = 128 + (signal_received[0] if signal_received else signal.SIGTERM)
                emit("Stopping every seed process group after interruption.")
                break
            failed = None
            all_done = True
            for worker in workers:
                code = worker["process"].poll()
                all_done &= code is not None
                if code is not None and not worker["reported"]:
                    # Descendants can outlive a successfully exited CLI leader.
                    # Closing now also releases any inherited stdout pipe ends.
                    if worker.get("job") is not None:
                        worker["job"].close()
                    worker["reported"] = True
                    emit(f"[seed {worker['seed']}] exited with status {code}; log: {worker['log']}")
                if code not in (None, 0) and failed is None:
                    failed = code
            if failed is not None:
                result = failed if failed > 0 else 128 - failed
                emit("One seed failed; stopping every remaining seed process group.")
                break
            if all_done:
                break
            stop_event.wait(poll_seconds)
        if result:
            stop_workers(workers, grace)
        for worker in workers:
            worker["thread"].join(timeout=5)
        save_status(("phase_complete" if "phase" in plan else "workers_complete" if plan["stage"] == "full" else "complete")
                    if result == 0 else "interrupted" if stop_event.is_set() else "failed")
        return result
    except BaseException:
        stop_workers(workers, grace)
        save_status("failed")
        raise
    finally:
        for worker in workers:
            if worker.get("job") is not None:
                worker["job"].close()
            thread = worker.get("thread")
            if thread is not None and thread.ident is not None:
                thread.join(timeout=5)
            if thread is None or not thread.is_alive():
                worker["stream"].close()
        for number, previous in old_handlers.items():
            signal.signal(number, previous)


def recorded_arm_complete(worker, settings, stage):
    """Avoid mutating completed worker logs while resuming final evaluation.

    This is only a scheduling hint. merge_seed independently verifies every
    checkpoint, budget, preparation origin, and evaluation before publication.
    """
    folder = Path(worker["output"])
    status_path = folder / "status.json"
    completed_path = folder / "arms" / worker["arm"] / "completed.json"
    if not status_path.exists() or not completed_path.exists():
        return False
    status = json.loads(status_path.read_text(encoding="utf-8"))
    completed = json.loads(completed_path.read_text(encoding="utf-8"))
    target = settings["ppo"]["full_updates" if stage == "full" else "pilot_updates"]
    return (status.get("stage") == "complete" and status.get("run_stage") == "pilot"
            and status.get("arms") == [worker["arm"]] and status.get("updates") == target
            and not status.get("skipped_arms") and completed.get("update") == target)


def run_parallel_arms(plan, environment):
    """Prepare once, train independent arms, then evaluate with the original CLI."""
    from parallel_state import preparation_ready, clone_preparation, merge_seed
    seed = plan["identity"]["seeds"][0]
    settings = plan["identity"]["settings_by_seed"][str(seed)]
    preparation = plan["preparation_workers"][0]
    prepared = Path(preparation["output"])
    status_path = Path(plan["state_root"]) / "launch_status.json"
    phase = "preparation"
    try:
        if not preparation_ready(prepared, settings):
            result = run_workers({**plan, "workers": [preparation], "phase": phase}, environment)
            if result:
                return result
        if not preparation_ready(prepared, settings):
            raise ValueError("Preparation did not produce the complete shared training inputs.")
        if plan["stage"] == "prepare":
            atomic_json(status_path, {"state": "complete", "phase": "preparation", "stage": "prepare",
                        "parallelism": "arms", "updated_at": utc_now(), "workers": []})
            return 0
        phase = "clone_preparation"
        for worker in plan["workers"]:
            clone_preparation(prepared, Path(worker["output"]), worker["arm"], settings)
        phase = "ppo_arms"
        pending = [worker for worker in plan["workers"] if not recorded_arm_complete(worker, settings, plan["stage"])]
        if pending:
            result = run_workers({**plan, "workers": pending, "phase": phase}, environment)
            if result:
                return result
        phase = "merge_checkpoints"
        finalizer = plan["finalization_workers"][0]
        merge_seed(Path(plan["runtime"]), prepared,
                   {worker["arm"]: Path(worker["output"]) for worker in plan["workers"]},
                   Path(finalizer["output"]), settings, plan["stage"])
        # All three checkpoints are at target: the unchanged original runner
        # performs no additional PPO updates and owns the final test protocol.
        phase = "evaluation"
        result = run_workers({**plan, "workers": [finalizer], "phase": phase}, environment)
        if result:
            return result
        if plan["stage"] != "full":
            status = json.loads(status_path.read_text(encoding="utf-8"))
            atomic_json(status_path, {**status, "state": "complete", "updated_at": utc_now()})
        return 0
    except BaseException as error:
        previous = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
        atomic_json(status_path, {**previous, "state": "failed", "phase": phase,
                    "stage": plan["stage"], "parallelism": "arms", "error": str(error), "updated_at": utc_now()})
        raise


def verify_full_results(plan):
    """The general suite allows skipped arms; this complete-three-arm launcher does not."""
    for worker in plan.get("finalization_workers", plan["workers"]):
        folder = Path(worker["output"])
        try:
            marker = json.loads((folder / "final_protocol.json").read_text(encoding="utf-8"))
            summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
            target = plan["identity"]["settings_by_seed"][str(worker["seed"])]["ppo"]["full_updates"]
            finals = [r["arm"] for r in summary["metrics"] if r["cohort"] == "final"]
            complete = summary["training_completion"]
            valid = (marker["arms"] == ARMS and marker["updates"] == target
                     and summary["seed"] == worker["seed"] and summary["target_updates"] == target
                     and summary["arms"] == ARMS and not summary.get("skipped_arms")
                     and sorted(finals) == sorted(["base", *ARMS])
                     and all(complete[a]["update"] == target for a in ARMS))
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise ValueError(f"Seed {worker['seed']} lacks complete final result records: {error}") from error
        if not valid:
            raise ValueError(f"Seed {worker['seed']} is not a complete three-arm full-budget run; inspect its report.")


def aggregate_results(plan, environment):
    verify_full_results(plan)
    code = ("import json,sys; sys.path.insert(0,sys.argv[4]); "
            "from windows_io import replacement_retries; from gsm8k_experiment.suite import aggregate\n"
            "with replacement_retries():\n"
            "    aggregate(sys.argv[1], json.loads(sys.argv[2]), json.loads(sys.argv[3]))")
    env = dict(environment)
    env.update(CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-u", "-c", code, plan["output_root"],
                             json.dumps(plan["identity"]["seeds"]), json.dumps(ARMS), str(Path(__file__).resolve().parent)],
                            cwd=plan["runtime"], env=env, check=False)
    if result.returncode:
        raise ValueError(f"Original suite aggregation failed with status {result.returncode}.")
    state_root = Path(plan.get("state_root", plan["output_root"]))
    output_root = Path(plan["output_root"])
    for name in ("suite_summary.json", "suite_report.md"):
        source = output_root / name
        if not source.is_file():
            raise ValueError(f"Original aggregation did not produce {source}.")
        if state_root != output_root:
            fd, temporary = tempfile.mkstemp(prefix=name + ".", suffix=".tmp", dir=state_root)
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(source.read_bytes())
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, state_root / name)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
    print(f"Suite report: {state_root / 'suite_report.md'}", flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime", required=True, help="Restored original CLI runtime directory")
    p.add_argument("--recipe", help="Single-seed three-arm YAML file or preset; defaults to the selected VRAM profile")
    p.add_argument("--vram-gb", type=int, choices=(12, 24, 48), default=24,
                   help="Per-GPU memory profile (default: 24); 12 offloads inactive graders to CPU")
    p.add_argument("--output-root", required=True, help="Per-seed experiment output directories and local suite reports")
    p.add_argument("--log-root", required=True, help="Append-only per-seed logs (also streamed to console)")
    p.add_argument("--state-root", help="Persistent manifest, lock, and status directory (default: --output-root)")
    p.add_argument("--base-seed", type=int, default=42)
    p.add_argument("--seeds", help="Comma-separated explicit seeds, one per selected GPU")
    p.add_argument("--gpus", help="Comma-separated native CUDA visibility tokens; subset of any inherited allocation")
    p.add_argument("--parallelism", choices=("auto", "seeds", "arms"), default="auto",
                   help="Auto: with 3+ GPUs train one seed's arms on the first three; otherwise one seed per GPU. Existing runs retain their scheduling.")
    p.add_argument("--stage", choices=STAGES, default="full")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Resolve and print without CUDA probing or filesystem writes; specify GPU tokens")
    mode.add_argument("--check", action="store_true", help="Probe CUDA and validate configuration/resume identity; do not write or launch experiments")
    return p


def main(argv=None):
    from windows_io import replacement_retries
    with replacement_retries():
        return _main(argv)


def _main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        plan = make_plan(args, dict(os.environ))
        if args.dry_run or args.check:
            check_manifest(plan)
            print(json.dumps(plan, indent=2, sort_keys=True))
            return 0
        with exclusive_lock(Path(plan["state_root"])):
            manifest = check_manifest(plan)
            atomic_json(Path(plan["state_root"]) / "launch_manifest.json", manifest)
            Path(plan["output_root"]).mkdir(parents=True, exist_ok=True)
            result = (run_parallel_arms(plan, dict(os.environ)) if plan["parallelism"] == "arms"
                      else run_workers(plan, dict(os.environ)))
            if result == 0 and plan["stage"] == "full":
                try:
                    aggregate_results(plan, dict(os.environ))
                except (ValueError, OSError) as error:
                    status_path = Path(plan["state_root"]) / "launch_status.json"
                    status = json.loads(status_path.read_text(encoding="utf-8"))
                    atomic_json(status_path, {**status, "state": "aggregation_failed", "error": str(error)})
                    raise
                status_path = Path(plan["state_root"]) / "launch_status.json"
                status = json.loads(status_path.read_text(encoding="utf-8"))
                atomic_json(status_path, {**status, "state": "complete", "updated_at": utc_now(),
                                         "suite_report": str(Path(plan["state_root"]) / "suite_report.md")})
            return result
    except (ValueError, OSError, subprocess.TimeoutExpired) as error:
        print(f"launch_seeds: {error}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
