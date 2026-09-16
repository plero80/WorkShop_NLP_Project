# YAML experiments and CLI

Notebooks are optional interfaces. YAML describes a run, and the CLI invokes the
existing Python runner with a resolved JSON configuration. Both notebook and
terminal runs use the same shared PPO engine.

The first supported family is **GSM8K**, including all its reward arms and
multi-seed suites. Best-of-N, distillation, follow-up, and cluster exploration
keep their existing entry points; the YAML CLI does not wrap them yet.

## Setup

From the repository root, enter the project folder and prepare the runtime:

```text
cd workshop_project
python gsm8k.py setup
python -m pip install -r ../run/gsm8k/experiment_cli/requirements.txt
```

This installs the YAML dependency as well as the existing experiment requirements.
Run the commands below from `workshop_project/`, including on Runpod. The launcher
uses `../run/gsm8k` for execution and outputs, reusing existing checkpoints. It
automatically prepares the runtime for a first run. Setup is useful before installing
dependencies. Use the CUDA PyTorch environment described in the GSM8K run guide.

Inside an already restored `run/gsm8k/`, the original `python -m experiment_cli`
commands also remain supported. Both launchers resolve identical run settings.

## Everyday commands

```bash
# Inspect the complete configuration; no model imports or downloads.
python gsm8k.py show gsm8k
python gsm8k.py run gsm8k --dry-run

# Run 100 pilot attempts, then resume to 400 total attempts and final evaluation.
python gsm8k.py run gsm8k
python gsm8k.py run gsm8k --stage full

python gsm8k.py status gsm8k
python gsm8k.py export gsm8k

# Sequential seeds 42, 43, 44, each with isolated outputs.
python gsm8k.py run gsm8k-three-seeds
```

Commands run in the foreground and return the underlying runner's exit code.
For a Runpod terminal session that can disconnect:

```bash
export HF_HOME=/workspace/hf-cache
nohup python -u gsm8k.py run gsm8k > pilot.log 2>&1 < /dev/null &
tail -f pilot.log
```

Wait for successful pilot completion before starting full training. `status`
reads saved progress; it does not guarantee that a process is still alive.

## Write a recipe

### Runpod B200

Use the `gsm8k-b200` preset for one B200. Select the official Runpod PyTorch
2.8 / CUDA 12.8 template, or a newer Blackwell-compatible build. Blackwell support
and CUDA 12.8 wheels were introduced in [PyTorch 2.7](https://pytorch.org/blog/pytorch-2-7/);
Runpod provides a [PyTorch 2.8 / CUDA 12.8 environment](https://www.runpod.io/articles/guides/pytorch-2-8-cuda-12-8).

From `workshop_project/` on the pod, preserve the template's CUDA PyTorch:

```bash
python -m venv --system-site-packages .venv
source .venv/bin/activate
python gsm8k.py setup
python -m pip install -r ../run/gsm8k/experiment_cli/requirements.txt
export HF_HOME=/workspace/hf-cache

# Check the installed GPU build before downloading the experiment models.
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA is unavailable"
print(torch.cuda.get_device_name(0), torch.__version__, torch.version.cuda)
assert torch.cuda.is_bf16_supported(), "BF16 is unavailable"
x = torch.randn((128, 128), device="cuda", dtype=torch.bfloat16)
assert torch.isfinite(x @ x).all().item()
torch.cuda.synchronize()
print("CUDA/BF16 smoke check passed")
PY

python gsm8k.py run gsm8k-b200 --dry-run
python gsm8k.py run gsm8k-b200
# After the pilot completes:
python gsm8k.py run gsm8k-b200 --stage full
python gsm8k.py status gsm8k-b200
```

The preset makes the existing BF16, SDPA, and batch settings explicit, with outputs
in `../run/gsm8k/gsm8k_outputs/b200` from the project folder. It retains the same scientific settings and shared PPO
engine. This is a starting preset, not a measured B200 optimization or a GPU-tested
memory-fit guarantee. Use the pilot to assess memory use and throughput before
changing batch sizes; choose a new output directory if changing settings.

### Custom recipes

Presets live in `configs/experiments/`. Copy one to `my-run.yaml`,
edit it, then use `python gsm8k.py run my-run.yaml`:

```yaml
version: 1
experiment: gsm8k
stage: pilot
output: gsm8k_outputs/my-run
settings:
  seed: 42
  generation:
    batch_size: 16
  ppo:
    learning_rate: 1.0e-5
    pilot_updates: 100
    full_updates: 400
```

Unspecified settings inherit `configs/gsm8k/settings.json`. Mappings merge
recursively; lists replace entire lists. Unknown keys, duplicate keys, and wrong
types are rejected. Use `1.0e-5` for a floating-point YAML value.
The loader uses [PyYAML's safe loader](https://pyyaml.org/wiki/PyYAMLDocumentation).
Dry runs resolve settings and check their structure; the training runner retains
its full protocol and runtime validation.

Override individual settings without editing the recipe:

```bash
python gsm8k.py run my-run.yaml --output gsm8k_outputs/trial-2 --set ppo.learning_rate=2.0e-5 --set generation.batch_size=8
```

Setting paths are relative to `settings`, so use `ppo.learning_rate`, not
`settings.ppo.learning_rate`. Relative output paths always use the restored
runtime root (`../run/gsm8k`), independent of the working directory or notebook. A custom YAML
filename is relative to the invoking working directory. `export --destination`
also accepts a path relative to that working directory.

For a suite, add `seeds: [42, 43, 44]`; these override the individual training
seed. `settings.data_seed` controls the shared data partition. Keep a separate
output directory for each single-run or suite protocol.

Resolved JSON files are stored by content hash in the runtime's `.experiment_cli/configs/`.
Pilot and full use the same configuration identity. Changing scientific settings
changes the resolved configuration; the existing runner rejects incompatible
resume attempts in an old output directory. Stage and output are launch options,
not scientific settings. The CLI does not alter existing engine source files.

## Notebooks

GSM8K saves ungradable examples to `<output>/review/ungraded/` and continues.
Missing grades are excluded from learning and explicitly counted in reports.
To install this behavior in an existing runtime while preserving saved work, use
the [GSM8K upgrade instructions](gsm8k/README.md#continue-past-ungradable-examples).

The restored GSM8K notebook calls the same underlying CLI. A cell can run any recipe:

```python
import subprocess
import sys
subprocess.run([sys.executable, "-m", "experiment_cli", "run", "gsm8k"], check=True)
```

Use terminal commands for long jobs and notebooks for inspecting reports and
plots, or run the same commands from notebook cells. Neither changes the training
implementation.
