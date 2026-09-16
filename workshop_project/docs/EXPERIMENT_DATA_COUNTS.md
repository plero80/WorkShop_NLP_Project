# Experiment prompt counts, kNN memories, and training budgets

This reference covers all five retained experiment families: follow-up,
Best-of-N, kNN distillation, cluster exploration, and GSM8K. Counts were checked
against the project configurations, execution code, and available saved artifacts
on 2026-09-16. Historical counts describe the saved runs. GSM8K includes both
the configured budget and verified counts from the uploaded completed B200
archive; the live Runpod process was not accessed.

## How to read the numbers

- A **prompt** is one input question or conversation. Generating two answers to
  it produces two examples, while the prompt count stays one.
- A **memory row** is a prompt–answer example with an embedding and a reward-gap
  label. The memory size is the number of stored examples.
- **k** is the number of memory neighbors retrieved for each query. It is not
  the number of prompts or the number of clusters.
- A **PPO rollout** is a batch of newly generated answers used for an update.
  Reusing a prompt in a later rollout produces another answer, not another
  unique prompt. PPO epochs reuse the same rollout for optimization.
- An **arm** is a reward condition, such as proxy-only or kNN-corrected reward.
  Arms normally share prompt cohorts but produce their own answers.
- Counts below are **per seed and per arm/checkpoint**, unless a total is
  explicitly stated. Seeds do not necessarily create disjoint prompt sets.

## Quick comparison

| Experiment | Main prompt budget | kNN memory | Neighbors | Seeds / saved scope |
|---|---|---|---:|---|
| Follow-up PPO | 6,400 training prompts per seed; 1,024 refresh; 1,024 fresh final | 7,990 original rows; 9,014 after refresh | 31 | 42, 43, 44; completed |
| Best-of-N | 512 development prompts × 32 answers × 2 fixed generators; another 512 confirmation prompts reserved | 4,096 rows, reused unchanged | 31 | Generator checkpoints from seed 42; development completed |
| kNN distillation | 2,400 student-training prompts; 400 validation; 400 offline; 512 final in the completed run | 10,038 frozen rows per seed | 31 | 42, 43, 44 in the completed run |
| Cluster exploration | No new generated answers; 31,030 saved example rows in the saved second-refresh explorer | 7,990 original rows for clustering and neighbor reference | 31; **20 clusters** separately | Analysis seed 42; overlays include policy seeds 42–44 |
| GSM8K B200 | 5,000-question PPO pool; 3,200 used per arm in the completed full run | 1,024 verified rows from 512 questions × 2 answers in each matched memory | 32 | Completed seed 42; three-seed recipe is a separate option |

## Shared inputs for the HH experiments

Follow-up, Best-of-N, and distillation use the retained Anthropic HH-RLHF
conversation setup. Their policy is Qwen2.5-0.5B-Instruct, their frozen proxy
reward model is Skywork-Reward-V2-Qwen3-0.6B, and their judge is
Skywork-Reward-V2-Qwen3-4B. Exploration reads their saved answers and embeddings.

The original candidate bank has **3,200 distinct prompt texts and 12,785
prompt–answer rows**:

| Candidate-bank split | Distinct prompt texts | Saved answer rows |
|---|---:|---:|
| Training / original kNN memory | 2,000 | 7,990 |
| Model validation | 400 | 1,599 |
| Calibration | 200 | 799 |
| Selection | 200 | 797 |
| Test | 400 | 1,600 |
| Total | 3,200 | 12,785 |

These are saved counts, not an assumption of exactly four retained answers per
prompt. The original memory uses the **training rows only**. Its embeddings have
1,024 dimensions; retrieval uses cosine similarity with temperature 0.05.
Distinct full prompt texts and normalized first-user conversation groups are
different counting units; the latter are used in cohort-overlap checks.

Sources: [candidate bank](../data/inputs/candidate_bank.csv),
[original memory](../data/inputs/memory/detectors/gap_knn.npz),
[frozen protocol](original/PROTOCOL.md), [model definitions](../code/core/assets.py).

## 1. Follow-up: does refreshing memory improve PPO?

Saved study: `study_cfdbaf579047d418`.

The policy generates answers, the proxy and kNN memory provide its training
reward, and PPO updates the policy. Between two training rounds, new answers
are scored by both proxy and judge and appended to the memory.

| Stage | Prompts | Answers / use |
|---|---:|---|
| PPO round 1 | 3,200 per seed | 100 updates × 32 prompts; one new answer per prompt |
| PPO round 2 | Another 3,200 per seed | Another 100 updates × 32 prompts; one new answer per prompt |
| Round-one memory refresh | 1,024 | One answer from the round-one kNN policy per prompt and seed |
| Development fit | 1,024 | Two answers per prompt: one from each saved raw/signed seed-42 policy; 2,048 examples |
| Development validation | 512 | 1,024 old-policy examples for offline selection; also 512 newly generated examples per round-one kNN policy for its capped-arm selection |
| Offline test | 512 | Two saved-policy generators produce 1,024 evaluation examples |
| Fresh final | 1,024 | One answer per evaluated policy and seed |
| Legacy recheck | 2,000 | Re-evaluate fixed policies at both 128- and 256-token answer caps |

Each seed's PPO schedule contains **6,400 distinct prompt IDs**. All conditions
within that seed use the same schedule. Across the three saved schedules there
are **13,677 distinct prompt IDs**, so 3 × 6,400 is a count of training slots,
not the number of distinct prompts across seeds.

The four final conditions are:

| Condition | Round 1 | Round 2 | Memory rows in round 2 |
|---|---|---|---:|
| `raw` | Raw proxy reward | Continue raw parent | No kNN reward |
| `knn_signed` | Original kNN reward | Continue with the original memory | 7,990 |
| `iterative_knn` | Shares the `knn_signed` parent | Continue with refreshed memory | 9,014 |
| `iterative_capped` | Shares the `knn_signed` parent | Refreshed memory with validation-selected correction limits | 9,014 |

The refreshed memory is **7,990 + 1,024 = 9,014 examples per seed**. All refresh
answers are included, rather than only examples with a large gap. Retrieval
still uses **k = 31**, temperature **0.05**. The separate development-only
diagnostic memory adds 2,048 examples to the original memory; it is not the
memory used by the main iterative PPO arms.

Training uses two PPO epochs per rollout, minibatches of 8, microbatches of 4,
and a 256-token answer cap. Round-one kNN training is shared across three
descendant conditions, so it should not be counted as three separate runs.
The four main final arms produce **1,024 × 4 × 3 = 12,288 final answers**;
the saved study additionally evaluates parent and legacy checkpoints.
The retained 128-prompt `monitor.json` input is not a periodic monitoring
stage in this follow-up training loop.

Sources: [configuration](../configs/config.json),
[prompt cohorts](../data/inputs/data/), [training schedules](../data/inputs/training/),
[execution logic](../code/core/run_study.py),
[saved manifest](../results/followup/study_cfdbaf579047d418/manifest.json),
[seed-42 memory lock](../results/followup/study_cfdbaf579047d418/refresh/seed_42/locked_reward.json).

## 2. Best-of-N: which generated answer should we select?

Saved study: `study_40100cfcb78f1920`, development phase.

This experiment performs **no new PPO training**. Two fixed checkpoints generate
candidate answers: the initial policy and the proxy-trained policy at update
200, both from the seed-42 parent study. Proxy, kNN-corrected, and judge-based
selectors choose among the same candidate answers; judge selection is a
reference comparison.

| Quantity | Count |
|---|---:|
| Development prompts, shared by both generators | 512 |
| Reserved confirmation prompts, disjoint from development | 512 |
| Answers per prompt per generator | 32 |
| Fixed generators | 2 |
| Saved development answers per generator | 512 × 32 = 16,384 |
| Saved development answers across both generators | 32,768 |
| Candidate-pool sizes evaluated | 1, 4, 8, 16, 32 |
| Frozen kNN memory | 4,096 examples from 4,096 parent-study memory prompts |
| kNN neighbors / temperature | 31 / 0.05 |

The smaller pools are prefixes of the same 32 candidates. Evaluating five pool
sizes does **not** generate five independent candidate pools. Candidates are
generated in batches of 8, with a 256-token answer cap.

The 4,096-example memory is inherited from the teacher-comparison parent study;
Best-of-N adds **zero** new memory rows. It is a separate memory from the original
7,990-row datastore and the distillation memory. The primary comparison uses
the proxy-trained generator at N = 32 on the 512 development prompts.

The confirmation notebook uses the separately reserved 512 prompts. If run with
the same settings it would generate another 32,768 candidates. **No completed
confirmation result is present in the saved project.**

Sources: [settings](../configs/best_of_n/settings.json),
[candidate generation](../code/experiments/best_of_n/bon_run.py),
[selection logic](../code/experiments/best_of_n/bon_metrics.py),
[saved manifest and memory lock](../results/best_of_n/study_40100cfcb78f1920/manifest.json),
[development completion](../results/best_of_n/study_40100cfcb78f1920/development/complete.json).

## 3. kNN distillation: can a student reproduce the corrected reward?

Completed saved study: `study_c18793a5593485e3`.

First, a frozen proxy+kNN teacher labels generated answers. A student reward
model learns those labels. Then three reward arms continue the same saved PPO
parent: `proxy`, `knn`, and `student`. Student PPO obtains its reward directly
from the student; it does not query the kNN memory or large judge for training
rewards.

| Stage | Distinct prompts | Examples / use per seed |
|---|---:|---|
| Student fitting | 2,400 | 4,800 answers: one base-policy answer and one parent-policy answer per prompt |
| Student validation | 400 | 800 answers; select the student by validation MSE |
| Offline reward evaluation | 400 | 800 answers; evaluate reward fidelity and judge agreement |
| PPO continuation | Reuses the 2,400 fitting prompts | 100 updates × 32 prompts = 3,200 new rollout answers per arm |
| Policy monitoring | 512 | One answer per prompt, evaluated at the parent and after +50 / +100 updates |
| Final policy evaluation | **512 in the completed run** | One answer per prompt for the parent and each of the three trained arms |

The 2,400/400/400 split comes from the parent study's 3,200-prompt training pool.
Validation and offline prompts are withheld from the new student fit and new
PPO continuation, but the parent policy may already have trained on them.
The final 512 prompts are freshly reserved HH test conversations.

The student is trained for **3 epochs** on its 4,800 fitting examples, with a
batch size of 4 and gradient accumulation of 8 (32 examples per full optimizer
step). Validation/offline examples are not student-fitting data. The default
does not fit the optional direct-judge student.

Each seed reuses a frozen **10,038-row memory**:

```text
7,990 original examples + 1,024 first-refresh examples
                        + 1,024 second-refresh examples = 10,038
```

The memory comes from the completed second-refresh parent, uses **k = 31** and
temperature **0.05**, and receives no additional rows during distillation.
The parent PPO checkpoint is at update 300; continuation ends at update 400.
The 3,200 continuation slots cover the 2,400 fitting prompts once, then repeat
800 of them with newly generated answers. Each rollout uses two PPO epochs,
minibatches of 8, and a 256-token answer cap.

With three seeds and three continuation arms, this is **28,800 new PPO answers**.
Final evaluation of the parent plus three arms produces
**512 × 4 × 3 = 6,144 answers**. The same final prompt cohort is reused across
these policies and seeds.

There are three different configurations to distinguish:

| Configuration | Seeds | Final prompts |
|---|---|---:|
| Completed saved study | 42, 43, 44 | **512** |
| `settings.json` defaults | 42, 43, 44 | 1,024 |
| `notebook_settings.json` | 42 | 512 |

Use the completed study's 512-prompt count when describing its results. The two
earlier failed study directories do not represent additional completed trials.

Sources: [default settings](../configs/knn_distillation/settings.json),
[notebook settings](../configs/knn_distillation/notebook_settings.json),
[label generation](../code/experiments/knn_distillation/labels.py),
[PPO schedule](../code/experiments/knn_distillation/data.py),
[saved manifest](../results/distillation/study_c18793a5593485e3/manifest.json),
[saved cohort counts](../results/distillation/study_c18793a5593485e3/data/complete.json),
[parent memory lock](../data/prerequisites/memory_refresh/study_e7a7994106548cb9/memories/seed_42/locked_reward.json).

## 4. Exploration: what do the saved embeddings and gaps look like?

Saved analysis: `analysis_6ee2efa375535c11`, from the second-refresh explorer.

This is a CPU analysis of saved examples. It generates **zero new model
answers**, performs **zero PPO updates**, and adds **zero memory rows**.
It fits 20 clusters on the original 7,990 memory vectors, then projects saved
policy answers into that space. PCA gives a two-dimensional display.

| Included source | Prompt cohort | Saved answer rows displayed |
|---|---:|---:|
| Original memory; cluster-fit and kNN reference | 2,000 distinct full prompt texts | 7,990 |
| Second-refresh answers | 1,024 prompts, reused across 3 policy seeds | 3,072 |
| Second-refresh audit answers | 512 prompts, reused across 3 policy seeds | 1,536 |
| Parent/static/refreshed final answers | 2,048 prompts × 3 policies × 3 seeds | 18,432 |
| Total | 5,469 normalized first-user conversation groups in the saved table | **31,030 rows** |

The separate neighbor-geometry check uses **1,200 saved query answers covering
1,015 conversation groups**, with **31 neighbors** per query and temperature
**0.05**. It excludes the query's own conversation group from the reference
neighbors. The 20 clusters are descriptive groups; they are not the kNN
neighbors and are not used to build a new training reward.

Both exploration notebooks default to 20 clusters, 31 neighbors, and analysis
seed 42. `REWARD_GAP_CLUSTER_EXPLORER.ipynb` loads the earlier follow-up outputs;
its offline-test cohort supplies 512 prompts × 2 policies = 1,024 query answers
when those saved vectors are available. `EXPLORE_SECOND_REFRESH.ipynb` loads
the second-refresh outputs described in the table above. Loaded overlays depend
on which saved files are available; the 31,030-row total belongs specifically
to the saved second-refresh analysis.

Sources: [exploration notebooks](../notebooks/exploration/),
[source inventory](../results/exploration/analysis_6ee2efa375535c11/source_inventory.json),
[cluster settings](../results/exploration/analysis_6ee2efa375535c11/clustering_info.json),
[geometry counts](../results/exploration/analysis_6ee2efa375535c11/geometry_summary.json),
[saved member table](../results/exploration/analysis_6ee2efa375535c11/cluster_members.csv).

## 5. GSM8K B200 experiment

These numbers come from `settings.json` plus the `gsm8k-b200` YAML recipe.
The YAML changes processing batch sizes, not the dataset cohort counts.
The default is **one seed: 42**, with data-split seed 42.

### Verified completed run from the uploaded archive

The uploaded `gsm8k_outputs.zip` contains a **completed full run**, matching
these settings. The counts below were checked against saved splits, memory
arrays, training statistics, rollout rows, and evaluation answers:

| Item | Actual saved count |
|---|---:|
| Seeds completed | 1: seed 42 |
| Reward arms completed | All 4 default arms |
| Successful PPO updates | 400 per arm; 1,600 across arms |
| Skipped updates / excluded training answers | 0 / 0 |
| PPO questions used | 3,200 distinct per arm; the same schedule across arms |
| PPO answers | 6,400 per arm; 25,600 across arms |
| 4B- and 30B-labeled memories | 1,024 rows each, covering the same 512 questions |
| Calibration / memory / selection answers with valid preparation grades | 256 / 1,024 / 256; no preparation exclusions |
| Monitor evaluations / answer slots | 65 / 8,320, reusing the same 128 questions |
| Final policies / answer slots | 5 / 6,595, reusing the same 1,319 questions |
| Permanently ungraded scorer cases | 2, both in intermediate monitoring |
| Missing final-test grades | 0 |
| Unused questions in the reserved PPO pool | 1,800 |
| Training-source questions outside all reserved cohorts | 1,065 |

The two monitoring cases were at updates 150 and 350 of `knn_static`.
They entered the review queue without interrupting training. They did not
remove any final-test questions. No optional refresh or oracle arm was run.
See the [results analysis](gsm8k/B200_RESULTS_ANALYSIS.md) and
[verified counts](gsm8k/b200_seed42/verified_summary.json) for evidence and results.
The later [threshold analysis](gsm8k/B200_THRESHOLD_ANALYSIS.md) reuses the
256 calibration and 256 selection answers, then evaluates the fixed diagnostic
cutoff on the existing 6,595 final answers. It generates no additional answers
and leaves these experiment counts unchanged.
The integrated [validation step](gsm8k/VALIDATION.md) uses that same selection
cohort to freeze diagnostic cutoffs before final scoring in new runs. It adds
no training, calibration, memory or evaluation questions; selection metrics
remain tuning diagnostics rather than an additional independent test set.
The [reward bug audit](gsm8k/B200_REWARD_BUG_AUDIT.md) also reuses the saved
25,600 training answers and cached embeddings. It adds reward/correctness
AUROCs and checks 19,907 saved predictions without generating new answers or
changing these counts. On the 256 validation answers, 4B-corrected reward
AUROC for strict correctness is 0.8788, separate from high-gap AUROC 0.7307.
The following sections explain how these counts arise and what other profiles
would request.

### Question allocation

The six training-source cohorts are disjoint and reserve **6,408 questions**.
Final evaluation separately reserves **1,319 official test questions**.

| Cohort | Questions | Generated answers | Purpose |
|---|---:|---:|---|
| Calibration | 128 | 256: two per question | Fit proxy/judge normalization and the gap threshold |
| Initial memory | 512 | 1,024: two per question | Store proxy embeddings and proxy-minus-judge gap labels |
| Selection / memory diagnostics | 128 | 256: two per question | Evaluate/select kNN settings; the default grid has only one candidate |
| Monitor | 128 | 128 per evaluated checkpoint | Greedy progress evaluation; no training or memory insertion |
| Refresh reserve | 512 | None in the default four-arm experiment | Reserved for the optional `knn_refresh` arm |
| PPO pool | 5,000 | Two per selected question per update | Supply training rollouts; not all 5,000 are used at the default targets |
| Final test | 1,319 | 1,319 per final policy | Greedy final evaluation, opened only in the full stage |

Calibration + memory + selection generate **1,536 preparation answers** in
total. Only the memory cohort's **1,024 answer slots** are candidate kNN rows;
calibration and selection answers are not inserted into memory.

The policy is Qwen2.5-0.5B-Instruct; the proxy is Qwen2.5-1.5B-Instruct; the
standard judge is Qwen3-4B-Instruct-2507; the stronger memory-labeling teacher
is Qwen3-30B-A3B-Instruct-2507. These generative grading models differ from the
Skywork reward models used in the HH experiments.

### What each reward arm consumes

| Arm | PPO reward | kNN memory used for training |
|---|---|---|
| `proxy` | Standardized proxy grade | None |
| `judge` | Standardized 4B judge grade | None |
| `knn_static` | Proxy grade minus predicted proxy–4B-judge gap | Up to 1,024 fixed rows |
| `knn_static_30b` | Proxy grade minus predicted proxy–30B-teacher gap | The same example rows and embeddings, with 30B gap labels |

The two memories reuse the **same 512 questions and generated answers**.
The 30B teacher relabels the saved preparation answers; it does not generate
a second memory answer set. Both memories use **k = 32**, temperature **0.05**,
and the same valid memory subset when a required grade is missing. The standard
4B judge evaluates all arms; optional 30B final evaluation is disabled by default.

### PPO and evaluation counts

| Quantity, per arm and seed | Pilot | Full run, including the pilot |
|---|---:|---:|
| Scheduled rollout/update attempts | 100 | 400 |
| Questions per attempt | 8 | 8 |
| Answers per question | 2 | 2 |
| Answers per rollout | 16 | 16 |
| Distinct PPO questions reached | 800 | 3,200 |
| Generated PPO answers | 1,600 | 6,400 |
| Monitor checkpoints after training starts | 4: 25, 50, 75, 100 | 16: every 25 attempts through 400 |
| Monitor answers per arm | 512 | 2,048 |
| Official final-test answers per arm | 0 | 1,319 |

The schedule traverses the shuffled 5,000-question pool without replacement
before beginning a new cycle. Consequently, 400 × 8 uses 3,200 distinct
questions, leaving 1,800 pool questions unused in that seed's default full run.
All four arms use the same question schedule.

Continuing a completed pilot to full runs attempts **101–400**, so the extra
budget is **300 × 8 = 2,400 questions and 4,800 answers per arm**. It does not
add 400 attempts on top of the pilot.

Across all four default arms, a full seed generates **25,600 PPO answers**.
Final evaluation covers the base policy plus four trained policies:
**1,319 × 5 = 6,595 final answers**. Monitor evaluation also includes one
128-answer base-policy evaluation; the full-stage monitor total is therefore
**128 + 4 × 2,048 = 8,320 answers**. Resuming reuses completed cached evaluations.
These totals exclude preparation, optional refreshes, grading retries, and the
small discarded PPO preflight test.

PPO uses two epochs per rollout and minibatches of 8. The B200 recipe sets a
generation batch limit of 128, scoring batch limit of 64, and 30B scoring batch
limit of 4. Those are processing batch limits, not extra prompts. Training
answers have a 768-token maximum.

### Missing grades and optional variants

The counts above are generated-answer budgets, not guarantees that every answer
is usable for reward training. Ungradable answers are recorded for review and
excluded from the affected reward calculation. An entirely ungraded rollout
skips its optimizer update while advancing the attempt counter. Therefore,
**400 attempts can mean fewer than 400 successful PPO updates**, and the actual
memory can contain fewer than 1,024 rows. If grading cannot support an arm at
all, the run records and skips that arm rather than fabricating scores.

The optional `knn_refresh` arm uses the reserved refresh cohort: every 100
attempts it samples **64 questions × 2 answers = 128 candidate additions**.
Through attempt 400 this uses 256 distinct refresh questions and can add 512
rows, ending with at most **1,536 rows** when the initial memory is full and
all grades succeed. The addition at attempt 400 happens after the last PPO
update. The optional `oracle` arm uses numeric answer correctness as its reward
and requires no kNN memory. Neither arm is in the default B200 four-arm list.

`gsm8k-three-seeds.yaml` requests seeds **42, 43, 44**, keeps data-split seed
42, and runs the full stage. Each seed has its own generated answers, memories,
and policies; the reserved question cohorts are shared. Its default full PPO
budget is **76,800 answers across 3 seeds × 4 arms**, subject to skipped arms.
The separate `format_pilot.json` profile changes completion penalties, not
these prompt counts; it is not automatically selected by `gsm8k-b200`.

For an actual Runpod result, use that output folder's `config.json`,
`data/splits.json`, `prepared/grading_coverage.json`, preparation completion
records, and per-arm training records to report realized counts. Those saved
records take precedence if the run was launched with overrides or an older
configuration. For the uploaded seed-42 archive, the realized counts were
verified as shown above; this is an archive audit, not a live GPU inspection.

Sources: [base settings](../configs/gsm8k/settings.json),
[B200 recipe](../configs/experiments/gsm8k-b200.yaml),
[three-seed recipe](../configs/experiments/gsm8k-three-seeds.yaml),
[optional arms](../configs/gsm8k/with_optional_arms.json),
[data partition and schedule](../code/experiments/gsm8k_experiment/data.py),
[preparation/training/evaluation](../code/experiments/gsm8k_experiment/run.py),
[matched teacher memories](../code/experiments/gsm8k_experiment/teacher_memory.py).

## Scope of the saved evidence

The removed experiment implementations are not additional active experiments
in this guide. Their saved parent artifacts remain dependencies where stated:
the second-refresh study supplies distillation's memory and checkpoints and the
exploration overlays; the teacher-comparison study supplies Best-of-N's memory
and fixed generators.

Historical result and prerequisite links above refer to local artifacts, which
are excluded from Git. Their inspected counts are written explicitly here so
the document remains readable in a source-only submission. Sources and current
configurations are tracked. See [SAVED_RUNS.md](SAVED_RUNS.md) for saved completion
states and [SUBMISSION.md](SUBMISSION.md) for packaging finished results.
