# Retro-E 4×H20 Joint ReFIT-PO + Forward FlowER Reward + DPO
**Codex execution handoff | 2026-11-18 | design and configuration, NOT a launched GPU job**

## Decision / unambiguous algorithm names

Four **additional, user-designated H20 GPUs** are allocated **conceptually** to a new independent Joint experiment. Confirm that the new four-device allocation is actually schedulable and not overlapping the eight H20 devices currently recorded for D3 Standard-D and D1. The separate 4×A100 is for non-ReFIT GRPO controls / offline DPO in nonoverlapping windows. Do not silently move the Standard-D job, reuse another active GPU, or interpret this docs file as an existing allocation.

**ReFIT-PO + Forward Reward is NOT yet DPO**. The term *Joint DPO* in this proposal means **all three terms** train one shared Student actor:

\[
\mathcal L_J=
\mathcal L_{\mathrm{ReFIT\!-\!PO}}(r_{\mathrm{route}}+r_{\mathrm{FlowER}})
+\lambda_D\mathcal L_{\mathrm{DPO}}(P_{\mathrm{FlowER}})
+\beta_{\mathrm{KL}}\mathcal L_{\mathrm{KL}}.
\]

Existing \`route_policy_objective.action_loss\` already includes the frozen-reference KL; do **not** add the same KL twice. In implementation \`L_PO\` includes its original KL, and \`L_J=L_PO+lambda_D*L_DPO\`. FlowER provides (i) **after-FINISH forward reaction evidence for the process reward**, and (ii) comparable evidence for a pairwise **DPO** preference. The same teacher result is cached and shared between the reward and DPO consumers; it is NOT computed twice. Record \`r_route\`, \`r_F\`, \`L_PO\`, \`L_DPO\` separately to support ablation.

RetroChimera is **not necessary** for the first 4×H20 Joint-F run (avoids an inaccessible proprietary-Pistachio checkpoint and a second inference cost). Keep its existing inverse-evidence interface as a later *Joint-F+C / Pro* condition, never silently add it to primary +F. Baseline Standard ReFIT-PO remains independent, with no extra forward teacher.

## 1. Sources that must be reused

- Source policy/reference: corrected FlowER-trained Qwen3.5-0.8B History-SFT checkpoint-89442, actor SHA specified in PR110/112.
- ReFIT exact-conditioning and M1 branch credit: PR110/112 \`route_refit_credit.py\`, \`route_refit_runtime.py\`; retain behavior logprobs from REAL on-policy samples, shared-prefix uniqueness and original G4.
- Reward: PR110 \`trusted_progress_first_rejection_cost_v2\`, \`route_flow_credit.py\`, \`route_process_reward.py\`. Retain B=1 only for connected reference-supported nonzero stock-closed route, bounded failure penalty, progress rollback on rejected rollout, resource censoring.
- Compute: PR112 \`route_work_broker.py\`, \`route_global_sampler.py\`, \`route_execution_cache.py\`, vLLM/HF backend, native token-logprob evidence and ref frozen-logprob cache.
- Step-budget: PR114 32 root-task draws per update, G=4 originals, M=1, initially 2 extra credit forks per original tree, 64 optimizer updates (16/32/64 reporting, optional extension 128); do NOT train all 156,723 D3 source rows.
- Teacher: original public FlowER forward checkpoint, https://github.com/FongMunHong/FlowER (pinned model+code SHA), supplied reactant inventory rather than only product/synthon. Public FlowER is an electron redistribution generator with forward BE intermediate trajectories, but these are predicted paths, not an independent experimentally verified mechanism.

## 2. Frozen Forward Replay Teacher at accepted FINISH

FlowER is called **once per distinct eligible committed reaction** after a verified accepted \`finish_trace\`, not once per FLOW. Build full precursor-side species including any required participant introduced by IMPORT; preserve atom maps and a separately declared stereochemical/isotope/charge normalization. If complete applicable reactants cannot be constructed or teacher fails: \`unavailable\` (missing score), not a negative chemistry label. Rejected/invalid FINISH is never a forward query. No reference product or GT precursor is injected into the actor observation or passed as a substitute reactant.

Input key for teacher cache:

    sha256(checkpoint + code + normalization + full_reactants_mapped +
           target_product + beam_or_sample_budget + stereo_contract)

Reward signal in first version **product-level forward recovery**, not a fabricated per-step mechanism label:

\[
s_F(e)=\frac{1}{M_F}\sum_{k=1}^{M_F}
{\bf1}[\mathrm{canon}(\hat P_k)=\mathrm{canon}(P_e)]\in[0,1]
\]

where the fixed \`M_F\` samples/beam hypotheses use one declared teacher protocol; if using ranked beam, report Hit@K and do not mislabel its rank statistics as calibrated likelihood. Two comparable candidate reactions must have the same teacher budget and usable reactants.

Optional second phase: align genuine forward BE states to reversed observed electron states with strict map correspondence; verify coverage first and never require exact mechanistic step ordering as a prerequisite for valid alternative chemistry.

### FlowER reward on complete and partial routes

Reward is additive **but strictly bounded per route**, and credited at the accepted FINISH. Initial proposal:

\[
q_F(t)=\lambda_F\min\left\{
1,\frac{1}{\max(L,1)}
\sum_{e\in\text{unique root-connected committed rxns through }t}
m_F(e)\,s_F(e)\right\},\ \lambda_F=0.1,
\]

with \`m_F=1\` if score available and \`0\` otherwise; unavailable coverage is **separately masked** for pair comparisons, not called failure. \`L\` is the frozen TRAIN reference step count already available to the hidden reward evaluator; teacher itself does NOT need the TRAIN route.

- Attach each \`q_F(t)-q_F(t-1)\` increment to the accepted FINISH; optionally move a conservative fraction of the **same** increment to uniquely attributable real FLOW receipts using existing \`route_flow_credit.py\`. No duplicate positive reward for repeated semantic FINISH or repeated reaction edges; one verified committed source reaction edge earns at most one teacher increment.
- If a later non-resource rejected action terminates that rollout, the teacher progress is **terminally rolled back**, exactly like source route progress; the failed rollout cannot retain a positive net teacher bonus. Resource-censored groups remain undefined, not negative.
- The total teacher component cannot exceed \`lambda_F\` even if there are many FINISH calls; do not increase reward by adding useless extra reactions. Also report teacher-supported-but-route-unclosed outcomes separately. This is proposed auxiliary chemical plausibility shaping, not a calibrated synthesis success probability.
- Compare at least two frozen control variants at equal budget: \`J_R\` (ReFIT-PO + Forward Reward only, no DPO) vs \`J\` (identical + pairwise DPO); main Standard \`R\` has neither.

First-version reward gating may be adjusted ONLY before the new frozen training contract and only using TRAIN/CLEAN VALID; do not tune to PaRoutes n5 TEST or RetroCast Reference. Keep original route/executor reward as the task-defining main component.

## 3. DPO from the SAME sampled FlowER evidence

Build preference pairs only from **two distinct native FLOW actions proposed under exactly the same sampled policy observation, route/occurrence/history, behavior version and frozen teacher protocol**. Use their real native assistant action tokens/EOS. Do not compare actions from different chemical states; no teacher-generated text is automatically a student target.

Candidate A and B may form a DPO pair if:
1. Both have usable, completed, independently observed continuations that reach accepted reaction FINISH, with teacher-valid full reactants for each; teacher scores were calculated with same fixed protocol.
2. Their FlowER forward product support differs by a predeclared margin, e.g. \`delta_s=0.25\` on [0,1] for first condition; their route support and Executor legality are also recorded.
3. If an explicitly failed FLOW is used as rejected, distinguish an **Executor-hard-negative** comparison from a **FlowER valid-vs-valid chemistry** comparison. Never infer chemical invalidity from a teacher miss alone.
4. Do not pair a score-unavailable action with score=0 as if observed, nor reuse duplicate semantic proposals as multiple distinct preference states.
5. For M1 incomplete/censored branches, skip DPO pair; the underlying original PO receipt behavior stays valid per its own return/coverage contract.

Exact native action likelihood \`log pi(a|s)\` sums generated assistant tokens only, including EOS. DPO uses one frozen reference actor and the standard logistic preference loss:

\[
\mathcal L_{\mathrm{DPO}}=-\mathbb E\log\sigma
[\beta_D\{(\log\pi_\theta(a^+|s)-\log\pi_{\rm ref}(a^+|s))
-(\log\pi_\theta(a^-|s)-\log\pi_{\rm ref}(a^-|s))\}],
\]

with proposed \`beta_D=0.1\`, \`lambda_D=0.1\`; require an ablation \`lambda_D=0\`. If no eligible pairs, only the existing ReFIT-PO objective contributes. Track usable preference pairs, distinct state count, valid-vs-valid fraction, confidence and old/new policy KL.

Joint learner reuses existing token logprobs in \`optimize_routes\` and **one optimizer update**, not two independent optimizer steps. For offline independent 4×A100 DPO baseline, use pinned Hugging Face TRL \`DPOTrainer\`; for online Joint, reuse DPO loss math and reference/actor forward from the existing PR112 learner, or migrate to verl if parity passes. Do not embed a second full TRL Trainer inside a live GRPO optimizer.

**Do not double-count teacher evidence unnoticed**: the primary full Joint purposely uses teacher in both forward reward and DPO; preserve \`J_R\` reward-only ablation, \`J_D\` DPO-only optional control (if cost permits), and teacher-free \`R\`.

## 4. Four-H20 resource and throughput protocol

**Resource contract: 4×H20 for this new Joint experiment** (additional pool designated by user; allocation must be confirmed from the actual scheduler). There is no 5th dedicated teacher GPU and no claim of free capacity. Use existing PR112 global inference queue and CUDA memory metrics. Two eligible deployment profiles:

**Profile A — preferred if benchmark proves it viable: 4 H20 student plus opportunistic teacher sidecar on one GPU**
- 4 student model replicas for generation/learner updates; freeze behavior actor throughout one full collection.
- Co-locate a frozen FlowER model on GPU3 **only if actual peak memory** of actor+reference+vLLM+FlowER+KV+optimizer fits within a declared safe limit. Reserve a memory fraction explicitly; don't use an unbounded vLLM GPU memory fraction.
- Teacher jobs run batched/asynchronous on distinct finished reaction keys, and yield priority to active student GPU requests. If too much contention, use separate **teacher microphase** after collection; no new actor policy version until matching teacher scores and complete on-policy group returns are finalized.
- Persistent cache across training rounds, keyed by teacher checkpoint and canonical full reaction input. Negative result is a real zero teacher score only if teacher actually ran with valid input and completed under the declared inference budget. Error is unavailable.

**Profile B — if co-location slows or OOMs: 3 H20 student / 1 H20 FlowER teacher**
- Use 3 student GPU workers for generation; a dedicated one serves frozen teacher reward in continuous batches. Training DDP topology and global loss denominator must be explicitly configured for 3 training ranks; don't pretend it is equivalent to the original 4-rank student reference and don't force a dummy fourth learner gradient.
- Compare both profiles using identical real request sets, teacher coverage, valid branch Q groups and complete-step cost. Choose once using TRAIN-only timing; freeze before official comparison.
- Alternative to profile B if incompatible with DDP: time-slice all four GPUs between student collection and teacher scoring, maintaining four training ranks.

**No per-FLOW synchronous teacher inference**. Precompute RetroChimera **only** for a later separately labelled Joint-F+C experiment; first Joint-F teacher is just forward FlowER. Cache hit rate, teacher queries/s, batch utilization, teacher latency, % accepted FINISH scored, teacher queue lag, student GPU decoding throughput, GPU-hours and total wall must be logged.

Do not infer an overall speedup from vLLM alone: PR112's own numerical admission showed HF step-vs-HF teacher-forced drift and initial H20 native throughput uncertainty. Keep original sampled behavior logprob provenance (old policy), KL and deterministic receipt contracts. Spot-check exact saved token scores and gradient/loss consistency across model versions.

## 5. Finite RL schedule and validation

All scientific algorithm comparisons use **D3 TRAIN source-pinned candidate pool** and the **same frozen root-task stream** (PR114; target uniqueness/strata and holdout audit). The source D3 156,723 rows are **available route references**, not an obligatory 1-epoch online RL run.

| Stage | Optimizer updates | Cumulative root tasks | Diagnostic |
|---|---:|---:|---|
| Baseline | 0 | 0 | frozen Stage-II evaluation, clean target-disjoint VALID |
| Early | 8 | 256 | pair coverage, forward recovery, token probabilities and throughput |
| Intermediate | 16 | 512 | policy-only G4 VALID 256, matched ReFIT search, pair-quality/coverage |
| Main | 32 | 1,024 | clean VALID 512 |
| Final main | 64 | 2,048 | clean VALID 512; select best checkpoint |
| Optional extension | 128 | 4,096 | only if validation and observed GPU-hours justify |

Per update: 32 root-task draws, G=4 independent root route rollouts, \`matched_continuations=1\`, at most 2 extra ReFIT credit forks per original tree, \`exploration_forks_per_tree=0\` for new 4H20 Joint initial condition. Upper bound \`32*4*(1+2)=384\` complete-or-terminated reports, but actual GPU model requests/tokens and teacher queries are variable.

Save immutable checkpoints every 8 updates. Early sample pool and same-seed VALID are fixed BEFORE tuning. Report resource/censor fraction, route StockClosed@4 and ReferenceSupported@4, forward teacher availability/recovery, DPO pair counts, nonzero Advantage groups, execution legality and RouteSolved per generated request/actual GPU-hour. Use identical inference-time ReFIT budget to compare Standalone R and Joint J; avoid comparing unequal budget tree sizes as pure policy improvement.

Clean VALID must be disjoint at canonical-target level from the selected RL training roots and DPO/teacher pseudo-label pool. Expose reaction/patent overlap and fixed Stage-II FlowER SFT training-test contamination separately. For large public testing use RetroCast Reference 1,084 and PaRoutes n5 10,000 only after checkpoint selection; do not tune Joint/DPO with public TEST data.

### Minimal reporting table

| Model | Train algorithm / teacher | TRAIN root exposures | Teacher GPU-hours | Policy-only VALID | ReFIT VALID at fixed budget |
|---|---|---:|---:|---:|---:|
| Stage-II | frozen SFT, no external teacher | 0 | 0 | measure | measure |
| R | ReFIT-PO, no extra forward teacher | 2,048 | 0 | measure | measure |
| J_R | ReFIT-PO + FlowER reward | 2,048 | measure | measure | measure |
| **J** | **ReFIT-PO + FlowER reward + FlowER-ranked DPO** | **2,048** | measure | measure | measure |
| J-distill | TRAIN-only ReFIT search-to-policy distillation | record separate TRAIN-only distill budget | as used | measure | measure |
| P (A100) | independent offline DPO from teacher-scored real action pairs | record separately | shared frozen teacher evidence | measure | measure |

Keep a full joint \`FlowER+RetroChimera\` variant only if it can actually be funded and its chemical-label coverage audited; otherwise do not crowd the first 4H20 experiment.

## 6. Implementation map (reuse source rather than rebuilding)

1. **FlowER adapter**: wrap the existing FlowER official model inference into one frozen batched accepted-FINISH scorer, producing a versioned \`ForwardEvidence\` with full input, pred product, mapping, teacher protocol and score. This is a lightweight adapter, not a new teacher network.
2. **Shared teacher cache**: persistent content-addressed cache/in-flight dedup; never cache transient errors as negative labels; no repeated GPU inference for the same reaction score used by both RL and DPO.
3. **Reward**: add bounded \`q_F\` increments at FINISH into existing process event rewards with optional conservative real FLOW redistribution and terminal failure rollback. Preserve original \`r_route\`, cost, route reward SHA and actor observation; version a separate Joint-F reward contract.
4. **Preference miner**: reuse exact-condition ReFIT branch grouping to construct valid FLOW pairs from observed originals, teacher outcomes and complete suffixes, including token IDs/EOS. No synthetic action tokenization or repeated shared-prefix gradient.
5. **Joint objective**: reuse PR112 \`external_route_rl.optimize_routes\` current actor/ref forward and LoRA optimizer. Add \`lambda_D*loss_dpo\` on eligible pairs in same backward step; DPO normalization by pair count, GRPO fixed-scale normalizer unchanged. Track loss/grad norms independently and combined.
6. **Framework reuse**: offline DPO comparator on 4×A100 uses pinned TRL DPOTrainer. PR112 global request scheduling and vLLM remain for the online chemistry sampler; future verl migration uses AgentLoop/RewardLoop but is NOT required to start Joint-F.
7. **Performance**: teacher microbatches + cache, actual GPU admission, balanced global request queue, no redundant accepted executor work, max2 credit forks, ref logprob cache, minibatch4+ length buckets, batched journal writes. No assumption of H20 speedup before full-step measurement.
8. **Run stages**: create independent NEW Joint-F task flag, contract SHA, explicit exactly 4 H20 and frozen D3 root manifest. Perform frozen initialization/candidate sampling parity and reward/DPO synthetic receipts tests. Only submit GPU run when FlowER checkpoint/input compatibility and device admission pass.
9. **Final REQUIRED**: freeze best Joint actor, collect TRAIN-only ReFIT search, run actual native weighted CE search-to-policy distillation with separate checkpoint; compare no search / small ReFIT / fixed full ReFIT and per-route compute.

### Acceptance checks

- \`lambda_F=0,lambda_D=0\` exactly reproduces existing ReFIT-PO reward and learner on identical receipts (within declared numeric tolerance).
- Teacher queries ONLY after accepted FINISH and on full precursor-side reactants; zero calls on invalid FLOW/premature FINISH; missing teacher inputs = unavailable, not a negative.
- Cached teacher result identical between PO reward and DPO pair scorer; overlapping routes share teacher execution, but distinct Policy proposal tokens/RTG/Advantages are never merged.
- ReFIT sibling branch credit, G4 fallback and KL from PR110 remain valid; no double KL, no duplicated shared-prefix policy loss or duplicate branch Q votes.
- A failed continuation rolls back provisional teacher progress and existing route progress; bad edge penalty unchanged; valid completed alternate route can receive teacher chemical support without reference exact-match.
- Teacher reward total <=0.1 and normalized by reference step count, not unbounded per FINISH.
- Pairwise DPO never compares mismatched state/prompt/occurrence/policy version or unavailable/asymmetric teacher scores; pair count and valid-vs-valid coverage reported.
- 4H20 memory/throughput profile verified on actual frozen actor and complete batch; 3+1 alternative not silently mixed with 4-way learner config; no extra model/teacher GPU beyond the four allocated.
- Same D3 root stream, frozen clean VALID, recorded teacher compute, fixed step cap, TEST never used for model selection.
- A separate TRAIN-only distilled checkpoint and evaluated policy exists before claiming the final complete method.

**Research objective:** show whether an independently frozen forward electron-redistribution model provides useful local chemical signal beyond exact Executor validity/route references, and whether that evidence improves Retro-E ReFIT-PO through bounded process reward + teacher-ranked DPO, at controlled 4×H20 GPU cost. No unsupported claim that FlowER's predicted mechanism is ground-truth chemistry.
