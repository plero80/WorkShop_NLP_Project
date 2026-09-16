# Frozen protocol: iterative representation-memory correction

## Question and scope

Does adding teacher-labeled outputs from the optimized policy to a frozen
reward-model representation datastore improve the next round of PPO relative
to keeping the original datastore unchanged?

This is a justified experimental extension. Iterative feedback, kNN retrieval
and teacher-based evaluation have prior art. This package does not establish
that their combination is globally novel. Its contribution is the specified
proxy-gap application, controlled comparison, and observed outcomes, including
negative results.

## Representation and reward

For frozen proxy P and teacher J, store h = L2-normalized input to P's scalar
reward head at the scored final token. Store g = zP - zJ using the original
calibration. Neither P nor J is fine-tuned. Teacher feedback labels vectors;
the teacher does not generate the embedding.

Use the 31 nearest cosine neighbors, with weights proportional to
exp(-distance / 0.05). Predict the signed weighted mean gap, ghat.
Static and main iterative PPO both optimize zP - ghat. The original base policy
remains the KL reference, with coefficient 0.05. Policy LoRA rank 8, alpha 16,
q/k/v/o projections; policy learning rate 3e-6; value-head learning rate 1e-4.
Other PPO details are in `config.json` and `ppo_engine.py`.

Frozen quantities:

| Quantity | Value |
|---|---:|
| Proxy mean | -1.8856342163085937 |
| Proxy SD | 4.033386063962294 |
| Judge mean | 0.11191116714477539 |
| Judge SD | 8.365239331680028 |
| High-gap threshold theta | 0.8998121070053893 |
| Original memory rows | 7,990 |
| Embedding dimension | 1,024 |
| Neighbors | 31 |
| Temperature | 0.05 |

The actual proxy gap zP-zJ defines high-gap events throughout, regardless of
which reward PPO receives. Do not use the corrected reward minus judge to
claim the *original* high-gap rate declined.

## Main comparison

| Condition | Round one, updates 1–100 | Round two, updates 101–200 |
|---|---|---|
| Raw | Raw proxy | Continue raw parent; raw proxy |
| Static kNN | M0 signed correction | Continue static parent; same M0 |
| Iterative kNN | Shares static parent | Same parent/optimizer; M0 plus its 1,024 new teacher-labeled outputs |
| Iterative capped | Shares static parent | Same new memory; separate validation-selected cap/shrinkage |

Seeds 42,43,44. Rollout batch 32, two PPO epochs, minibatch 8, microbatch 4.
Every condition in a seed sees the same ordered 6,400 training prompts across
two rounds. The static and iterative conditions share the *same saved parent
file*, not just a nominal seed or similar initialization. Policy/value/Adam
state and the update counter are restored. The original base KL reference is
not reset to the round-one policy. Identical RNG seeds do not guarantee identical
outputs after policies diverge, but make the sampling protocol matched.

Memory refresh occurs once, between rounds. Each seed gets its own refreshed
memory from its own static round-one policy. All selected answers enter memory,
including low, negative and high gaps. There is no classifier threshold filter,
active query selection, or test-answer refresh. No fine-tuned teacher student
is trained here.

The capped ablation applies
`alpha*max(ghat,0) - min(alpha*max(-ghat,0), bonus_cap)`.
Validation selects alpha in {0.25,0.5,1}, bonus cap in {0,0.1,0.25,0.5}, and
optional old distance gating, plus a zero-correction candidate. It minimizes
validation prompt-weighted judge-score MSE. Its selection cannot change the
main iterative arm. Keeping the old proxy outside a distance gate does not
establish that unfamiliar outputs are safe.

## Cohorts and information separation

| Cohort | Prompts | Use |
|---|---:|---|
| Legacy evaluation | 2,000 | Previously examined prompts; paired 128/256 recheck |
| Development fit | 1,024 | Two old-policy outputs per prompt; offline-only memory/control fits |
| Development validation | 512 | Old-policy offline selection; new round-one capped-arm validation |
| Offline test | 512 | Evaluate the locked offline controls |
| Round-one refresh | 1,024 | New answers from each seed's shared round-one policy; append all |
| Fresh final | 1,024 | Final evaluation after every memory and new policy is frozen |

The five fresh cohorts have distinct first-user-turn groups, excluding groups
in previously used PPO schedules, previous final/monitor data, the historical
candidate bank and earlier prompt lists. Normalization uses Unicode NFKC,
casefolding and collapsed whitespace. Exact copies/case variants are excluded;
this is not a semantic-paraphrase deduplication guarantee. Hashes and an input
audit are included.

Fresh cohorts come from unused rows of the original HH train-source pool.
They are held out from this experiment's training/memory use, not the official
HH test split, and not guaranteed absent from model pretraining. No answer
scores informed cohort selection. All new training finishes before fresh-final
scoring begins. Results are a follow-up motivated by previously inspected data;
report that provenance.

Offline controls append 2,048 outputs of saved raw/signed seed-42 policies to a
separate diagnostic memory. They do not become the round-one refresh memory.
Compare raw, original signed, refreshed signed, capped refreshed, affine-proxy,
and proxy+log(length)+EOS residual predictors on the same offline-test answers.

## Answer length and scoring

Saved checkpoints and initial policy are evaluated at both caps using the same
current software stack. New PPO, refresh and fresh-final evaluation use 256.
Sampling is temperature 1, top-p 1, top-k 0. Evaluation RNG depends on cohort
and batch, not method/cap, providing matched streams without pretending answers
remain identical after EOS/length differences.

Policy prompts retain their fixed 512-token limit. Reward scorers receive the
full original conversation and decoded generated answer. Neither proxy nor
judge truncates its input; a context guard raises before inference if exceeded.
The scalar-head hook verifies that the stored state matches the scored token.
Logs record response length, EOS, both reward input lengths and truncation flags.

A 256-token output limit remains a limit. EOS/completeness still need examination.
The inherited datastore retains earlier 1,024-token scoring labels; preflight
compares only complete compatible references. The new refresh explicitly
adapts that inherited datastore under the full-answer scoring protocol.

## Endpoints and uncertainty

Primary contrast: iterative minus static after round two. Assess mean teacher
reward and actual high-gap rate together. There is no automatic success claim
or best-checkpoint selection. Improved gap with worse usefulness is not success.

Report high-gap count/rate, conditional tail mean, gap Q95/Q99, raw/corrected
judge-score MSE, detector AUROC/AP, mean gap, response length, EOS, and literal
refusal-template counts. Q99 at 1,024 prompts has sparse tail support; interpret
it descriptively. AUROC is undefined for a one-class evaluation set, and AP is
reported with its class prevalence. These are model-judge overestimation labels,
not human-confirmed hacking labels.

Paired prompt bootstrap intervals use 2,000 draws within each trained seed;
training-seed SD/range are separate. Three seeds give limited evidence about
training randomness. Refusal templates and EOS subgroups are descriptive,
not human judgments or causal adjustments. Round-two versus its round-one
parent is also reported on identical final prompts.

Human review is independently sampled, blinded A/B assessment. Default 480
unique prompt pairs across packs, with only 60 for the primary comparison;
that primary human estimate will be imprecise. Report completed coverage and
reviewer count. Repeated reviewers are averaged within prompt before bootstrap;
agreement on shared pairs is reported. Correctness/completeness summaries
exclude uncertain labels and show the effective rated-prompt count. Do not
present model scores as a substitute for missing human annotations.

Teacher-query accounting separates evaluation, development, refresh and
validation. This study does not compare active selection against equal-budget
random selection, nor isolate policy-adaptive data from simply adding more data.
A positive iterative/static contrast supports the benefit of this refresh as a
whole; a same-size off-policy refresh control would be needed for that narrower
mechanistic attribution.

## Reproducibility and failures

Input/source/runtime identities prevent mixing changed experiments. Atomic
shards and checkpoints support resume; completed-output hashes are verified.
A cooperative pause occurs at a completed evaluation batch or PPO update.
The result ZIP includes predictions, refresh memories, adapters, provenance,
reports, review materials and source; full Adam checkpoints stay on disk.

Tests use tiny CPU Qwen models and synthetic rewards where noted. They verify
implementation, not the scientific hypothesis. Real RTX GPU model parity,
adapter loading, whole-answer scoring and a discarded real PPO update run in
preflight before detached launch. There are no fabricated GPU results in the
package. Hardware throughput and final study outcomes must be measured.
