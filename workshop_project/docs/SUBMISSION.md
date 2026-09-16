# Submission and an active Runpod experiment

## While training is running

Let the current Runpod process continue. Do not delete, move, or replace
`run/gsm8k`: it contains the code currently executing, checkpoints, cached grades,
and generated answers. You do not need to restart it for the project layout change.

The copied runtime is an execution artifact. Both `run/` and `original_project/`
are excluded from Git and from the submission builder.

## A single source tree

New GSM8K experiments execute the files in `workshop_project/code` directly.
Their outputs go to `workshop_project/gsm8k_outputs`. The project launcher does
not restore or copy source files. If it finds an existing run in the old runtime,
it routes commands to that run's original code and checkpoints.

From the repository root, create a source-only submission:

```bash
python workshop_project/submission.py --destination submission-source.zip
```

This includes tracked project code, configurations, notebooks, input files
(including the original reference adapters), documentation, and provenance.
There is one active source tree, no `run/`, no
backup snapshot, and no environments. `SUBMISSION_MANIFEST.json` records file
hashes and the Git commit at packaging time. The builder needs a Git checkout;
running GSM8K from an extracted submission does not require Git.

## Add the finished experiment

After the current Runpod run reports `stage: complete`, run from the repository root:

```bash
python workshop_project/submission.py \
  --destination submission-with-gsm8k-results.zip \
  --results run/gsm8k/gsm8k_outputs/b200
```

For a new run launched directly from the project, use
`--results workshop_project/gsm8k_outputs/b200` instead. Repeat `--results` to add
other completed runs with distinct folder names. Archives are never overwritten.

The builder places reports, metrics, saved evaluated answers, training logs,
review records, and initial memory arrays under `workshop_project/results/gsm8k/`.
It omits duplicated source, model checkpoints, adapters, generation caches, and
the SQLite reward cache. It refuses to package an active or unfinished run.
Keep a separate backup of the original output folder if you need to resume training.

Saved result manifests retain the source identity actually used for training.
Packaging results does not relabel them as a run of a later source revision.
The current source layout changes resource paths and exports; it retains the
same shared PPO and kNN implementations.

Saved outputs for the other showcase experiments are local artifacts excluded
from Git. Their source and notebooks are in the source archive; include their
chosen result artifacts separately if your submission requires those saved studies.
