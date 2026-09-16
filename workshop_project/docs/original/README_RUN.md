# Reward-gap follow-up: one notebook

Extract the **whole ZIP** into `/workspace/`, then open
`reward_gap_followup/RUN_FOLLOWUP.ipynb` with a fresh Python 3.12 kernel.
Choose **Run → Run All Cells**. The notebook installs/checks requirements, runs
CPU correctness tests, runs a real GPU preflight, and starts the study in a
separate process. You do not run the Python files individually.

The ZIP includes the nine saved final policy adapters, the original kNN memory,
the calibration, and fixed data cohorts. No older project folder is needed.
It does not include the large base/proxy/judge model weights. Use the RunPod
where those exact revisions are already cached. `EXTRA_HF_CACHE` accepts another
Hugging Face **hub cache directory**, if needed. Complete local snapshots are
used before authentication or network requests. There is no interactive login
cell. Missing files can be downloaded if `ALLOW_DOWNLOADS=True`; genuine server
rate limits can still prevent downloads. A token cannot override a rate limit.

## Default run

1. Re-evaluate all nine saved checkpoints, plus the initial policy, at **128
   and 256 generated tokens** on the original 2,000 prompts. Both caps are run
   in the same environment to isolate the answer-limit change.
2. Run an offline refresh diagnostic with affine proxy and length/EOS controls.
3. Run **two 100-update PPO rounds**, with three training seeds and matched
   prompts. Round one trains raw and static kNN policies. Round two forks the
   static checkpoint into static, iterative, and capped iterative conditions.
4. At the fork, teacher-score 1,024 new answers per seed and append their
   frozen-proxy vectors and **all signed gaps** to the memory. An additional
   512 validation answers per seed select only the capped ablation parameters.
5. After every new policy is finished and every memory is locked, evaluate the
   new policies and saved policies on 1,024 fresh held-out conversation groups.
6. Create HTML reports, detailed CSVs, three blinded human-review ZIPs and
   `important_outcomes_followup.zip`.

The main comparison is **iterative kNN versus static kNN**. The two conditions
start round two with the exact same policy, value head and optimizer state;
only the memory changes. The proxy/judge weights, calibration, threshold,
neighbor count, temperature, and original KL reference stay fixed.

The new two-round training starts from the base policy so both rounds use the
256-token protocol. The earlier 128-token checkpoints are reused for evaluation
and offline diagnostics; they are not silently treated as a matched new round
one. This distinction keeps the main comparison interpretable.

Defaults entail **1,800 PPO rollout/update iterations** (each includes multiple optimizer steps), counting shared round
one once, and about **76,352 teacher-scored evaluation/development answers**,
plus small preflight checks. PPO rollout rewards use the proxy and memory, not
the teacher. The full study can take longer than one night; completion time
must be measured on the pod. Do not assume an overnight guarantee.

## Notebook settings

- `RUN_NEW_PPO=True`: full study. Set `False` for checkpoint evaluation, offline
  diagnostics and the first review pack only. Later set it back to `True` and
  run the start cell to continue in the same experiment directory.
- `ALLOW_DOWNLOADS=True`: download missing pinned model files. Set `False` to
  require all files to exist locally.
- `EXTRA_HF_CACHE=None`: use standard cache locations. Set an absolute hub-cache
  directory if your models live elsewhere.
- `REPAIR_CUDA=False`: keep a working CUDA installation. Set `True` only when
  the environment checker says the build is incompatible.

Scientific settings live in `config.json`. Changing them, code, inputs, or
relevant package versions creates a new experiment identity and output folder.
Execution controls above do not invalidate compatible completed work.
Keep the environment unchanged when resuming an expensive run.

## Status, pause, resume

The start cell returns after launching the background process. This does **not**
mean the experiment is finished. Re-run the status cell to see the actual live
PID, current phase, progress and log tail. A notebook message alone is never
used as proof that a worker is running.

You can close the browser/laptop while the **RunPod remains running**. To pause,
set `PAUSE_NOW=True` in the last cell and run that cell. Wait for `stage=paused`
and no running process before stopping the pod. Resume by setting it back to
`False` and running the start cell. Evaluation resumes at complete batches;
PPO resumes at saved updates. An abrupt shutdown may lose work after the last
checkpoint. Preserve this entire project directory, including `outputs`, when
moving pods. The downloadable results ZIP omits optimizer checkpoints.

## Results and manual human review

The notebook links the latest report and result ZIP when they exist. Outputs
are under `outputs/study_<identity>/`. Final status is `complete`; the human
ratings remain a separate manual step.

Only distribute the files ending in **`_BLINDED.zip`** to reviewers:

| Pack | Default pairs | Purpose |
|---|---:|---|
| `checkpoint_review` | 240 | Saved raw versus positive/signed policies at both caps |
| `iterative_vs_static_review` | 60 | Primary memory-refresh comparison |
| `new_policy_vs_raw_review` | 180 | New corrected policies versus raw PPO |

Each pack contains an offline `REVIEW.html`. Extract it, open it in a browser,
enter a reviewer ID, and rate usefulness, correctness, inappropriate refusal,
completeness, and overall preference. Models and scores are hidden; A/B order
is balanced and randomized. Sample selection does not depend on model scores.
Reviewers should work independently and export JSON backups regularly. Export
preserves completed pairs; a partially filled current pair stays in browser
storage until finished. The form sends no network requests.

Do not give reviewers the full results ZIP: it contains the private mapping.
Use the notebook's optional import cell for their exported JSON files. Import
all intended current reviewer exports for that pack together; conflicting
versions of the same rating are rejected. Partial coverage is reported openly.
No automated score is substituted for an uncompleted human review.

## Scoring and interpretation

Reward scoring tokenizes the entire original conversation plus generated
answer with `truncation=False`. Inputs exceeding the 4,096-token guard stop the
run with an explicit error; nothing is silently cut off. The policy itself
still has its fixed 512-token prompt limit. Proxy and judge token counts and
EOS status are recorded for every evaluated answer.

The inherited memory was produced with the earlier scoring protocol. Its
weights, labels and calibration are deliberately retained, and preflight
checks compatible complete references. All new refresh and evaluation scores
use the whole-answer protocol. The study estimates the effect of refreshing
this inherited memory; it is not a clean claim that every old label was
originally scored without truncation.

A lower proxy–judge gap is **not enough** to claim better answers or proven
reward-hacking prevention. Check judge scores, response completion, length,
and blinded human usefulness/refusal ratings together. The proposed decreasing
gap percentages are hypothetical, not expected or measured results.

See `PROTOCOL.md` for the design and `RELATED_PAPERS.md` for the related work.
`VALIDATION.json` records local tests. The full study and GPU preflight have
not been executed in the CPU build environment.
