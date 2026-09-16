# Frozen-memory best-of-N experiment

This add-on tests whether kNN correction improves answer selection, without PPO training.
Use it in the ORIGINAL completed RunPod project, with its inputs, cached models and full checkpoints.
The outcome ZIPs alone do not contain the required model checkpoints.

## Quick start

Extract `reward_gap_best_of_n_addon.zip` into `/workspace/reward_gap_followup`.
This adds `best_of_n/`, `RUN_BEST_OF_N.ipynb`, and `RUN_BEST_OF_N_CONFIRMATION.ipynb`.
It does not replace the original project modules or notebooks.

Open `RUN_BEST_OF_N.ipynb` and run its cells. It runs in the foreground and shows the report when complete.
Alternatively, start a detached worker from a terminal using the SAME Python environment as the successful oracle run:

```bash
cd /workspace/reward_gap_followup
python best_of_n/launch_best_of_n.py start --project .
```

Check progress:

```bash
python best_of_n/launch_best_of_n.py status --project /workspace/reward_gap_followup
```

Pause safely:

```bash
python best_of_n/launch_best_of_n.py pause --project /workspace/reward_gap_followup
```

Rerun the `start` command to resume. Complete generated batches are reused even if scoring was interrupted.
Keep the pod on. Do not launch another training/evaluation worker on the same project GPU; the original GPU lock is honored.

For sequential notebook execution in the terminal:

```bash
jupyter nbconvert --to notebook --execute RUN_BEST_OF_N.ipynb --output RUN_BEST_OF_N.executed.ipynb --ExecutePreprocessor.timeout=-1
```

## Exact default protocol

- Generator checkpoints: initial and final proxy-PPO policies from the completed oracle study, training seed 42.
- Frozen correction: that oracle study's 4,096-row memory, k=31, cosine-softmax temperature=0.05, signed correction `z_proxy - predicted_gap`.
- 512 fresh development prompts plus 512 separately reserved confirmation prompts, grouped by normalized opening user message.
- All recognized earlier memory, training, monitoring, final and reserved study cohorts are excluded.
- Each phase generates 32 answers per prompt per generator: 32,768 answers, scored by both reward models.
- Answer limit: 256 tokens. Sampling matches original PPOActor: temperature=1, top-p=1, top-k=0, original answer `.strip()` behavior.
- One fixed generation-seed protocol, reproducible in the pinned runtime and unchanged batching. This is NOT three training seeds.
- All selectors share identical candidates. Nested pools use the first 1, 4, 8, 16 and 32 candidates in generation order.
- Proxy maximizes `z_proxy`; kNN maximizes `z_proxy - gap_hat`; judge maximizes `z_judge`.
- Lowest candidate index breaks exact score ties for all selectors. Duplicate answers are retained and their frequency reported.
- Full prompt and answer are scored; overlong inputs raise an error rather than being truncated.
- Judge scores are calculated after the kNN reward is fixed. They never update memory or enter the kNN selector.
- Primary endpoint: kNN minus proxy selected judge score at N=32 using the proxy-trained generator.
- Paired 95% bootstrap intervals resample prompts, not candidates. Secondary endpoints are exploratory.

Both generator policies are useful: initial tests correction under its original generation distribution;
proxy-PPO tests answers after optimization against the weaker reward. Neither generator is selected by best-of-N results.

## Confirmation

The default run only evaluates DEVELOPMENT. It reserves confirmation prompts without generating or scoring their answers.
When the protocol is fixed, run `RUN_BEST_OF_N_CONFIRMATION.ipynb`, or:

```bash
python best_of_n/launch_best_of_n.py start --project /workspace/reward_gap_followup --phase confirmation
```

If unchanged, this confirms the prespecified default protocol on the reserved cohort.
If you revise settings or code using development results, they create a new study identity and reserve NEW cohorts;
previous reserved cohorts remain excluded. Do not tune on confirmation or reuse it as a fresh test.
The software cannot establish that a human did not manually inspect a cohort; document your actual use.

## Output

The console prints the exact output path. `best_of_n_outputs/latest.json` records it.
Within the study folder:

- `important_outcomes_best_of_n_development.zip`: upload this for analysis after development finishes.
- `important_outcomes_best_of_n_confirmation.zip`: corresponding confirmation archive, after that phase finishes.
- `<phase>/reports/primary_result.json`: primary effect and paired interval.
- `<phase>/reports/selection_curves.png` and `.pdf`: selected judge scores and kNN improvement vs N.
- `<phase>/reports/summary.csv`: selection performance, high-gap rate, length, cap rate, pairwise ranking agreement, and MSE.
- `<phase>/reports/paired_comparisons.csv`: paired differences and confidence intervals.
- `<phase>/reports/headroom.json`: judge-selection ceiling and exploratory fraction recovered by kNN.
- `<phase>/reports/selected_text.csv`: selected full answers for inspection, with prompt and candidate IDs.
- `<phase>/runs/`: checksummed generated and scored candidate batches; originals are preserved for audit/resume.
- `<phase>/costs.json`: completed generation/scoring counts and measured runtimes, separately from historical memory labels.

## How to interpret

A positive primary difference with an interval above zero supports improved judge-based selection under this protocol.
If judge selection beats proxy but kNN does not, correction has not fixed the consequential rankings.
If all three are close, this candidate distribution provides little measured opportunity for correction.
Compare ordinary MSE with prompt-centered MSE: the latter subtracts each prompt's mean error before squaring,
so a constant offset that leaves rankings unchanged cannot account for its improvement.

Judge selection is an exact maximum ONLY within the sampled candidate pool under this judge.
It is not an upper bound on PPO and not independent evidence of human usefulness, factuality or appropriate refusal.
Better best-of-N selection does not by itself demonstrate better PPO training.
The fixed checkpoints and one pool per prompt limit inference about other training and sampling seeds.
No equal-runtime efficiency or reduced research-evaluation cost claim follows from these results:
all candidates are judge-scored for this diagnostic, even though kNN selection itself uses zero new judge scores.

## Troubleshooting and configuration

Edit `best_of_n/settings.json` BEFORE a run. Scientific settings, source code, checkpoint hashes, memory,
and runtime are recorded in the study identity. Changing them starts a new study rather than mixing results.

- Models/data are local-only by default. If a pinned dataset snapshot is missing, set `allow_downloads` to true
  or `extra_hf_cache` to the correct cache. Download permission/cache paths do not alter statistical identity.
- Insufficient fresh groups: the runner reports eligible and required counts; it never reuses or silently shrinks cohorts.
  Explicitly lower cohort counts in a new protocol, or choose `dataset_split: "test"` before examining its results.
- Multiple oracle suites: supply `--oracle /workspace/reward_gap_followup/next_studies_outputs/suite_NAME/oracle`
  to the launcher. In the notebook set `ORACLE` to that directory.
- Out of memory: lower `candidate_batch_size` before starting a new study. It changes the generation protocol/identity.
- Missing `.pt`: use the complete original RunPod project, not the code-only or outcomes-only archive.
- Changed source or parity mismatch: review the difference; do not disable the checks to force a run.
- Environment versions: reuse the exact torch/transformers/peft environment recorded by your successful oracle suite.
  The add-on does not install or upgrade your model environment.

## Validation

```bash
python -m unittest discover -s best_of_n -p 'test_*.py' -v
```

CPU tests cover perfect correction, offset-only correction, judge isolation, candidate pairing/ties,
corrupt resume shards, prior-cohort exclusion, paired bootstrap, full report generation and interrupted-scoring recovery.
The package was checked against your uploaded source and oracle output metadata.
Actual CUDA/model inference must still pass the built-in preflight on your RunPod; it was not executed during packaging.

Method references: https://arxiv.org/abs/2210.10760 (best-of-N and PPO reward overoptimization).
