# Reward-gap showcase

Open **[workshop_project/](workshop_project/README.md)** for the active project:
**Best-of-N, exploration, distillation, follow-up, and GSM8K**.

The new [GSM8K experiment](workshop_project/docs/gsm8k/README.md) uses the existing
shared PPO engine. Generated GSM8K outputs are kept locally.

`workshop_project/` separates retained code, notebooks, settings, inputs, and
results. Required parent studies are in `data/prerequisites/`.
The previous full snapshot remains in `original_project/`.

GSM8K runs directly from the project as shown below. The older notebook workflows
use the [restore instructions](workshop_project/README.md#running-the-code).

## Run GSM8K from a Git clone

Run commands from `workshop_project/`. New runs import the code directly from
`code/core` and `code/experiments`; no second source tree is created.

```bash
cd workshop_project
python -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -r requirements-gsm8k.txt
export HF_HOME=/workspace/hf-cache
python gsm8k.py run gsm8k-b200 --dry-run
python gsm8k.py run gsm8k-b200
```

After the pilot completes, use `python gsm8k.py run gsm8k-b200 --stage full`.
New outputs are saved under `workshop_project/gsm8k_outputs/`.
If an existing run is found in `run/gsm8k`, the launcher resumes it with its original
code and output paths. Let any currently active Runpod run finish in place; do not
delete or replace its runtime files. No restart is required for this layout change.
See the [YAML and CLI guide](workshop_project/docs/EXPERIMENT_CLI.md) for B200
environment checks, custom settings, status, and exports.

Git includes source, notebooks, configurations, manifests, and original inputs.
The local `original_project/` backup, `workshop_project/results/`, and saved
prerequisite studies are excluded. Links to those local artifacts will not work
in a fresh clone. GSM8K downloads its own models and dataset; the retained
Best-of-N, distillation, and exploration workflows need their saved prerequisite
artifacts copied separately. Full restoration and full artifact verification
require those local artifacts; use `--code-only` for a Git clone.

## Submission

`run/` and `original_project/` are excluded from Git. Build an archive containing
one source tree with `python workshop_project/submission.py --destination submission-source.zip`.
After training finishes, the same tool can add completed results without adding
another code copy or model checkpoints. See the [submission guide](workshop_project/docs/SUBMISSION.md).
