# Distill the kNN-corrected reward into a proxy

This runnable experiment trains a separate small reward model to imitate the frozen proxy+kNN reward, then tests whether the distilled reward preserves its teacher's benefit during PPO. It includes offline fidelity, timing/storage measurements, matched PPO branches, fresh final evaluation and reward-adapter export.

## Start

Extract the ZIP inside your existing completed `reward_gap_followup` project. Keep `knn_distillation` beside `ppo_engine.py` and `RUN_KNN_MEMORY_DISTILLATION.ipynb`. Open that notebook in the original RunPod Python environment and run all cells.

Choose the source explicitly in its configuration cell:

```python
SOURCE_KIND = 'refresh2'    # use the completed M2 policy, normally update 300
# SOURCE_KIND = 'refresh34' # use the final adaptive refresh3/4 policy, normally M4/update 500
```

The original full PPO checkpoints, memories, manifests, `inputs` and review template must remain on the pod. A code-only archive or compact outcomes ZIP cannot restore optimizer/value/RNG state. Selecting refresh34 requires the matching refresh3/4 study to be complete; its final adaptive checkpoint and memory are verified against the locked endpoint.

Terminal alternative, using the defaults in `knn_distillation/settings.json`:

```bash
python -u -m knn_distillation.run --project .
```

To execute the notebook from a terminal and wait for the experiment to finish:

```bash
jupyter nbconvert --to notebook --execute RUN_KNN_MEMORY_DISTILLATION.ipynb \
  --output RUN_KNN_MEMORY_DISTILLATION.executed.ipynb \
  --ExecutePreprocessor.timeout=-1
```

The notebook writes `knn_distillation/notebook_settings.json`. Resume that configuration from a terminal with:

```bash
python -u -m knn_distillation.run --project . \
  --config knn_distillation/notebook_settings.json
```

Use `--followup`, `--refresh2` or `--refresh34` to specify source-study paths when needed. Otherwise the corresponding existing latest pointers are used. Run one GPU experiment at a time; the original project lock is honored.

## Teacher, student and units

The teacher is the composite frozen function

\[
T_M(x,y)=z_P(x,y)-\widehat g_M(x,y),
\]

with the source's original frozen proxy, one fixed memory M, k=31, temperature=.05, L2-normalized proxy vectors and all signed gaps. The large judge previously provided gap labels to M. It is not queried to generate the main student's fit/validation targets.

The student starts as an independent copy of the original proxy reward checkpoint. It learns LoRA parameters in q/k/v/o attention projections and the scalar reward head. All other student base parameters remain frozen. The teacher model instance and its memory never change. Adapter/head save-load uses the official [PEFT model API](https://huggingface.co/docs/peft/en/package_reference/peft_model) and [LoRA configuration API](https://huggingface.co/docs/peft/en/package_reference/lora).

The loss uses the original proxy calibration:

\[
\mathcal L=\frac1N\sum_i\left(\frac{r_S(x_i,y_i)-\mu_P}{\sigma_P}-T_M(x_i,y_i)\right)^2.
\]

The equivalent raw-scale target is `proxy_raw - proxy_std * predicted_gap`. During PPO/inference the student reward is `(student_raw - proxy_mean) / proxy_std`. There is no student-distribution renormalization and no second kNN subtraction. Both overestimation penalties and underestimation bonuses are represented in the teacher target.

This is score regression, not a pairwise training loss. Pairwise ranking is measured offline on the two generated answers per prompt. The implementation learns an approximation of the teacher function; it does not encode every memory row losslessly.

## Default data and compute budget

For each existing seed 42/43/44, split the source refresh2 `ppo3` training pool by conversation opening:

| Split | Prompts | New answers per prompt | Examples per seed | Purpose |
|---|---:|---:|---:|---|
| Student fit | 2,400 | 2 | 4,800 | Regression training |
| Student validation | 400 | 2 | 800 | Select an epoch by its teacher-target MSE |
| Student offline test | 400 | 2 | 800 | Held-out teacher/judge fidelity |

Each prompt receives one new answer from the base policy and one from the selected source policy. The same source policy generates all per-seed labels before any new PPO. The source training pool is checked against known memory prompt groups, so these queries do not retrieve their own stored prompt-answer entries as teacher labels.

The split groups are disjoint. New PPO uses only the 2,400 student-fit prompt groups, in deterministic repeated epochs with new policy answers; validation and offline groups are not used in that PPO. The source policy may already have seen these three split groups during its earlier PPO. Accordingly, the offline set is held out from student fitting and new PPO, not asserted to be unseen by the source policy.

Monitoring uses the separate fixed refresh2 audit prompts. Final evaluation reserves 1,024 previously unused HH test conversation groups, excluding recognized earlier follow-up, refresh2, refresh34, next-studies, stress and distillation reservations. Final labels and answers are generated only after all new PPO branches finish. The pool is never silently shrunk or filled with old test prompts. New pinned test files may be downloaded if absent and permitted.

Default student settings: three epochs, microbatch 4, accumulation 8 (effective batch 32), LoRA rank 8/alpha 16, learning rates 2e-5 for LoRA and scalar head, gradient clipping 1, dropout zero and non-reentrant gradient checkpointing. Each main student processes 14,400 training-example exposures across three epochs. Epoch selection uses only the separate validation MSE. Original base weights and normalization remain the same for all students.

Default PPO: 100 updates per branch, rollout batch inherited from the source (normally 32), three seeds. The three-arm default totals 900 new PPO updates and 28,800 generated PPO answers. The source policy/value/optimizer/Torch-CUDA-RNG state is restored identically into all reward arms, and prompts/sampling seeds are matched. Original PPO learning rates, KL coefficient, base reference and other scientific hyperparameters are retained. Answers remain capped at 256 tokens and rewards score the entire formatted prompt-answer input.

## PPO branches and optional control

| Branch | Frozen PPO reward | Teacher or memory access during PPO reward scoring |
|---|---|---|
| `proxy` | Original proxy z-score | One original proxy forward |
| `knn` | Frozen proxy+kNN teacher score | Original proxy forward and memory lookup |
| `student` | Distilled proxy z-score | Student only; zero original-proxy forwards, judge calls or kNN lookups |
| `judge_student` (optional) | A separate student distilled directly from large-judge z-scores | That student only |

Enable the optional baseline before starting:

```python
INCLUDE_DIRECT_JUDGE_STUDENT = True
```

It trains an independent student from the same initial proxy state, with the same inputs, training order, hyperparameters and number of epochs, but using direct judge targets. Each student's epoch is selected on its own declared target's validation MSE. This control adds large-judge scores for the common fit and validation examples and adds one PPO branch per seed. It is not a claim of matched large-judge-query budgets.

With the default False setting, 16,800 fit/validation examples across three seeds receive proxy+kNN pseudo-labels and zero new large-judge labels. The 2,400 offline examples across seeds are judge-scored for evaluation. Policy monitoring/final evaluation also call the judge. Label-generation and evaluation costs are reported separately; previous memory-construction judge costs are not included in the new-work totals. Do not interpret the entire research run as making zero judge calls.

The legacy PPO trainer requires proxy-gap diagnostic fields. For a student-only reward call these fields are internally placeholders, then cleared to `None` before any history is written. They are marked as unmeasured. This avoids running a second proxy forward just to fill diagnostics. Monitoring/final evaluation separately scores all rewards on the generated answers, so its true original-proxy and teacher-fidelity diagnostics remain available.

## Outputs and interpretation

Outputs are under `knn_distillation_outputs/study_<id>/`:

- `labels/seed_*/train|validation|offline/`: cached generated answers, frozen targets and exact completed model-call counts.
- `students/seed_*/student/`: training history, validation losses, selected adapter/head/tokenizer, and rolling full student optimizer/cursor checkpoints. Optional `judge_student` has a separate folder.
- `offline/seed_*/metrics.json`: MSE/MAE, Pearson/Spearman, within-prompt pairwise agreement, and teacher-gap-quartile fidelity. Correlations are null for constant arrays; teacher ties are excluded from ordering accuracy.
- `latency/seed_*/results.json`: synchronized repeated scoring times, throughput, incremental CUDA peaks, memory array/file bytes and student adapter bytes.
- `runs/seed_*/proxy|knn|student/`: full new PPO checkpoints, common-source fingerprints, matched prompt IDs and call counters.
- `reports/final_by_seed.csv`, `paired_deltas_by_seed.csv`, `conditional_intervals.json` and monitoring curves.
- `reports/post_ppo_student_teacher_fidelity.json`: student-teacher MSE/correlation on answers from every final policy branch, revealing approximation errors under PPO-induced distribution shift.
- `selected_students.json` and `endpoints.json`: selected reward bundles and final policy checkpoints.
- `important_outcomes_knn_distillation.zip`: compact results, code and blinded review. Optimizer tensors and adapter weights are excluded from this archive.
- `distilled_reward_adapters.zip`: selected adapters, scalar heads, tokenizers and standalone scoring code for every seed/student type.

The primary PPO comparison is **student minus knn at the final configured update**: normally update 400 from M2 or update 600 from M4. Proxy and optional direct-judge-student comparisons are also reported against kNN. The study asks whether teacher behavior can be preserved at a lower measured runtime cost. A near-zero quality difference alone is not proof of equivalence; interpret its uncertainty and do not infer a noninferiority guarantee without a prespecified margin/design.

Bootstrap intervals average seeds within each prompt and then resample prompts. They condition on those trained seeds and are not a general guarantee across runs. Judge reward is a surrogate quality measure. Use the supplied blinded usefulness/correctness/refusal review before making stronger behavioral claims. Share only its nested reviewer ZIP with raters; the complete outcomes contain the private orientation key.

Timing uses identical full inputs and matching scoring batches, two warmups and repeated calls. The student route excludes diagnostic passes through other models. All scorers remain resident during the benchmark, so reported incremental CUDA peaks represent activation overhead, not isolated deployment GPU footprint. Adapter-vs-memory bytes also omit shared base/tokenizer costs. Fit histories report completed optimizer time; validation/loading/IO and discarded/repeated work are not included in that sum. Measure how many later scoring calls amortize fitting before claiming lower total cost.

## Use a trained student without kNN

Extract `distilled_reward_adapters.zip`. Supply the original pinned proxy base snapshot specified in each `reward_config.json`. The loader checks its base configuration and, for production exports, base weight hashes. It loads one reward model with the trained adapter/head; it does not load a teacher memory or judge.

Prepare a JSONL file with `prompt` and `answer` fields, then run:

```bash
python -m knn_distillation.score_student \
  --snapshot /path/to/pinned/proxy/snapshot \
  --student students/seed_42/student \
  --input answers.jsonl --output student_scores.jsonl
```

Use `--device cpu` for CPU inference. Output contains raw student scores and normalized PPO rewards. The base checkpoint is intentionally not duplicated in the adapter archive.

## Resume and validation

From a separate terminal:

```bash
python -m knn_distillation.control status --project .
python -m knn_distillation.control pause --project .
```

Wait for `paused` before stopping the pod. Resume with the exact same configuration. Pseudo-label and evaluation shards are reused. Student optimizer, LoRA/head weights, epoch/cursor and RNG state are restored; final partial accumulation batches use the correct denominator. PPO resumes its own full saved states. A changed scientific configuration creates a separate study from the chosen original source; it does not extend a completed run in place.

The delivered tests include real locally generated tiny Qwen reward/policy models: gradient regression, untrained proxy parity, whole-input guards, exact adapter-weight equality after interrupted/resumed training, adapter save/load, actual PPO updates with all three rewards, frozen teacher/student checks, and a complete optional-four-arm run through offline metrics, paired reports, human-review packaging and export. Protocol checks cover split exclusion, signed units, zero large-judge calls for default fit/validation targets, zero teacher or memory calls for student PPO, pseudo-label resume, pairwise ties, M4 endpoint provenance and configuration validation.

Run tests from the original project environment:

```bash
HF_HUB_DISABLE_PROGRESS_BARS=1 python -m unittest discover -s knn_distillation/tests -v
```

See `VALIDATION_KNN_DISTILLATION.json` and `CPU_TEST_LOG_KNN_DISTILLATION.txt` for delivered results and exact test versions. Tiny models are generated locally; tests download no model weights. Full-size Skywork/Qwen GPU training has not been executed here. RunPod preflight checks actual source states, pinned runtime parity, frozen encoder parity, a real student gradient step, adapter save/load and a real student-only PPO update. This add-on does not replace packages or modify original scientific source files.

## Code map

| File | Responsibility |
|---|---|
| `run.py` | Source selection, preflight, complete experiment ordering, exports |
| `io.py`, `data.py` | Identity/checkpoint guards and grouped data separation |
| `labels.py`, `reward.py`, `maths.py` | Frozen teacher targets, cached examples, signed kNN and student-only reward routing |
| `student.py`, `fit.py` | Separate proxy copy, LoRA/head regression, normalization, epoch selection and resume |
| `ppo_training.py`, `policy_eval.py` | Matched PPO continuation and complete held-out reward comparisons |
| `offline.py`, `reports.py` | Fidelity/ranking metrics, scoring costs, final comparisons and review packs |
| `score_student.py` | Standalone no-kNN reward scoring |

The module uses the reviewed original PPO engine and preserves its integrity checks. kNN formula and reporting helpers are adapted from the earlier add-on under an independent namespace, so installing this package does not require overwriting refresh34 code.
