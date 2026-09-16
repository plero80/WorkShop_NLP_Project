# Reward-gap showcase

Open **[workshop_project/](workshop_project/README.md)** for the active project:
**Best-of-N, exploration, distillation, follow-up, and GSM8K**.

The new [GSM8K experiment](workshop_project/docs/gsm8k/README.md) uses the existing
shared PPO engine. Generated GSM8K outputs are kept locally.

`workshop_project/` separates retained code, notebooks, settings, inputs, and
results. Required parent studies are in `data/prerequisites/`.
The previous full snapshot remains in `original_project/`.

Use the [restore instructions](workshop_project/README.md#running-the-code) to
recreate the selected runtime layout without restoring removed experiment sources.

## Run GSM8K from a Git clone

Run commands from `workshop_project/`. For a fresh clone, prepare the runtime
and install dependencies in your CUDA PyTorch environment:

```bash
cd workshop_project
python gsm8k.py setup
python -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -r ../run/gsm8k/experiment_cli/requirements.txt
export HF_HOME=/workspace/hf-cache
python gsm8k.py run gsm8k-b200 --dry-run
python gsm8k.py run gsm8k-b200
```

After the pilot completes, use `python gsm8k.py run gsm8k-b200 --stage full`.
The launcher reuses `run/gsm8k`, including existing Runpod checkpoints and outputs.
It prepares that folder automatically on the first run; no test copies are needed.
See the [YAML and CLI guide](workshop_project/docs/EXPERIMENT_CLI.md) for B200
environment checks, custom settings, status, and exports.

Git includes source, notebooks, configurations, manifests, and original inputs.
The local `original_project/` backup, `workshop_project/results/`, and saved
prerequisite studies are excluded. Links to those local artifacts will not work
in a fresh clone. GSM8K downloads its own models and dataset; the retained
Best-of-N, distillation, and exploration workflows need their saved prerequisite
artifacts copied separately. Full restoration and full artifact verification
require those local artifacts; use `--code-only` for a Git clone.
