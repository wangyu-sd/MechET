# Retro-E: Step-Budgeted On-Policy RL, D3 Data Fairness and Distillation Reserve

**2026-11-18 | Codex research/implementation plan | DOCS ONLY | supersedes "finish all 156,723 D3 rows" as the proposed NEW scientific training schedule**

## Executive decision

Do NOT equate the official D3 route-training source pool (156,723 route rows) with a mandatory one-epoch RL budget.

Standard GRPO/verl/TRL uses a population of training prompts, draws online groups each optimization step, and trains to a declared **number of steps, generated rollouts and compute budget**; full one-pass coverage is not required. See https://verl.readthedocs.io/en/latest/algo/grpo.html and https://huggingface.co/docs/trl/main/grpo_trainer .

**New controlled default:** all models initialize from identical corrected FlowER Qwen3.5-0.8B History-89442 (unless a clearly labelled engineering-only continuation), draw TRAIN roots from source-pinned, audited D3, sample G=4 independent root routes per root, then perform fixed-step on-policy GRPO with a small ReFIT credit-branch budget. The primary planned horizon is **64 optimizer updates × 32 root-task draws = 2,048 root-task exposures**. Evaluate at frozen checkpoints; authorize **128 updates / 4,096 root-task exposures** only if clean validation and measured compute justify extension. A further 256-update condition (8,192 root exposures) is supplementary and never automatic.

156,723 remains the **available source pool denominator**, not a claim that all 156,723 are used for gradient updates.

**Mandatory final phase:** reserve GPU time for TRAIN-only ReFIT search-to-policy distillation. Do not spend the entire study budget on route RL.

No PR110/112 production training task is changed by this document. It is a next-run plan; changing active submitted jobs, their resume contract or their optimizers needs separate explicit user authorization.

## 1. Why the old full-epoch contract is unsuitable

Observed previous PR110 full-D3 Standard-D HF round (2026-11-17):
- 64 source route rows per update; 4 independent G4 rollouts each; up to 16 extra ReFIT suffixes **per root G4 tree**; 3,300 reports and 63,925 physical model decision requests for the audited 64-row batch.
- 384 valid vs 1,538 masked conditional sibling groups, including potentially zero-advantage valid groups.
- 92.29 min total wall for that round; projection 2,449×92.29min ~157 days on a 4-H20 job **at the old measured speed only**. No such projection can be made for PR112's still-unverified native full-round performance.
- The 156,723 rows are source **route occurrences**, NOT 156,723 different unique retrosynthesis product prompts; repeated root targets and distinct reference paths must be audited.
- Full validation of 8,250 routes at several full-epoch percentages is also computationally expensive; first nonzero evaluation near round613 is too late for a budget-limited study.

PR112 implements target work-stealing, global GPU requests, executor-cache, credit/exploration lanes and a vLLM candidate. Its 2026-11-18 resumed jobs started from full step19 (D3) and89 (D1), but native probability-admission comparisons previously disagreed by >0.25 in isolated cases. PR112's new run records numerical drift rather than gating all launches. **Do not assume runtime acceleration, performance, or sufficient behavior-probability consistency until measured complete-round real receipts are inspected.**

## 2. Source D3 pool vs actual training population

Dataset: RetroCast-published PaRoutes reaction-holdout-n1-n5 route TRAIN 156,723 source rows / VALID 8,250; D2 route-holdout is 160,670/8,456. Source D3 removes PaRoutes n1/n5 test reactions according to that released holdout definition, but that does not imply actor FlowER SFT pretraining is test-reaction-disjoint.

Create frozen manifests BEFORE optimization:
1. Unique canonical root product IDs and source-row multiplicity per root; source route ID, patent/source, route depth, convergent AND branch count, route/reference reaction signatures, source manifest SHA.
2. Frozen STARTING STOCK contract (current guarded project URSA, not upstream official training stock). Report reference route leaf-stock reachability against that stock; note when success cannot achieve terminal +1 under the chosen stock. A training-only stock-eligible filter may be an explicit new condition, **not** silently applied to tests or mislabelled as original full D3.
3. Exact TRAIN/VALID overlap. The current published D3 run audit observes **812 shared canonical targets, 5,643 shared reaction edges, 465 shared route signatures, 11,224 shared source patents** (among other overlap types). This is a project recorded comparison under its own signature rules, not a verified assertion that the two source files are chemistry-disjoint.
4. Exact route/reaction/product/source overlap of chosen RL training roots vs PaRoutes n1/n5 and RetroCast Reference/Market. The official D3 test reaction exclusion is a route-RL stage property; separately inspect whether the fixed FlowER Stage-II SFT training reaction corpus (243,753 replay-approved reactions) saw these chemistry instances. Do not claim end-to-end zero overlap without that audit.
5. Preserve a full list of eligible source rows; never infer missing ground-truth electron action traces from multistep endpoint references.

**Unique-root-aware selection**:
- Freeze a reproducible, hashed ordered candidate list drawn from D3 TRAIN, sampled without replacement for primary fair comparisons where possible.
- Select by canonical target first (to limit repeated molecules), then choose one labelled source-route occurrence under a prespecified deterministic mapping/rotation. If multiple route references are combined, it changes hidden reward semantics and must be a distinct explicitly tested condition. Never reinterpret distinct route rows as independent chemical targets.
- Stratify from TRAIN-only source metadata: route step-count/depth, linear vs convergent AND obligations, inventory reachability and source/patent grouping; define bins from D3 TRAIN quantiles or explicit meaningful cutoffs, then freeze quotas before seeing any TEST route outcomes. Retain a representative fraction of difficult and rare chemistry; do not cherry-pick only early success.
- Initial candidate draw pool target: approximately 8,192 **eligible distinct root products** if enough exist. Use first 2,048 exposures for the 64-step condition; extend to 4,096 and only optionally 8,192. If unique eligibility falls below the requested size, report actual counts rather than fabricate unique targets.
- All B/C/D matched RL algorithm conditions consume exactly the **same frozen target/root-route stream** and planned optimization boundaries. Each Student samples **its own** real on-policy actions; do not copy other models' sample receipts.

Adaptive hard-case/curriculum sampling can be explored **only as a separately labelled follow-up** using TRAIN-only feedback and explicit target sampling probabilities. It is not the main fixed-stream causal comparison; VAL/TEST outcomes must never influence prompt sampling.

## 3. Primary on-policy scheduler

The following is a **proposed, not yet implemented or validated new run contract**:

    algorithm_family: native_process_grpo_refit_po
    data_source: frozen_D3_reaction_holdout_training
    schedule_unit: optimizer_updates
    total_updates_initial: 64
    total_updates_extension: 128  # requires clean VALID + compute gate
    optional_upper_bound: 256     # separate extension, not automatic

    roots_per_optimizer_update: 32
    original_independent_rollouts_per_root: 4
    loss_batch_size_per_gpu: 4     # from PR112 if real gradient parity remains valid
    on_policy_actor_version: frozen_for_entire_collection_update
    optimizer_steps_per_collection: 1
    replay_old_RL_episodes: false
    reference: frozen_original_StageII

    refit:
      matched_continuations: 1
      max_nodes_per_tree: 4
      max_expansions: 2
      max_duplicates: 2
      max_forks_per_original_G4_tree: 2
      credit_forks_per_tree: 2
      exploration_forks_per_tree: 0

    reward:
      existing_PR110_process_reward: unchanged
      flow_credit_alpha: 0.5
      failure_cost_contract: unchanged
    policy_loss:
      existing_clip_KL_normalizer: unchanged

32 root tasks per update gives 128 independent original G4 routes, plus at most 256 extra suffixes, i.e. at most **384 trajectory reports** before censoring/duplicate skips; physical model decision count remains **variable** and must be reported. This caps ReFIT extra suffixes at 2 instead of the current 16 and keeps genuine G4 original episodes independent. It does **not** guarantee two distinct FLOW siblings or a nonzero advantage.

If smaller micro-collections are useful for GPU queue fill, collect 8 or 16 root groups at a time and **aggregate to the declared 32-group global update** before learner optimization. Preserve one actor version and fixed global GRPO denominator across those subcollections; do not silently convert each 8-root micro-collection into a separate optimizer step.

Do not force distinct FLOW via action masking and continue to call it the same original policy. If the lower branch budget severely reduces same-state group eligibility, the candidate node eligibility and unique-FLOW sampling rate must be reported and the **4-fork condition can be proposed as a separately frozen compute-matched alternative**, rather than silently restoring 16 forks.

For each update record:
- sampled root task IDs and unique product IDs;
- 4 independent original route results and all additional ReFIT credit continuations;
- physical model requests, generated token counts, valid/nonzero/zero/masked/all-failed sibling groups;
- reference-supported complete routes and stock-closed nonzero routes, separately;
- useful credit groups per 1,000 physical model requests and per GPU-hour;
- rejected FLOW / premature FINISH / repeats / censored budget;
- behavior sample probabilities and distribution discrepancies, actor version, reward-digest and checkpoint SHA;
- per-GPU productive generation and idle/wait, plus learner time.

If a physical work/time cap is needed, treat unfinished samples as **budget-censored/unknown** under an explicit new protocol rather than chemical failures, and do not cherry-pick only fast successful routes into GRPO. Excess censoring is a reason to adjust a future frozen condition, not silently delete target denominators.

## 4. Training stages and VALID cadence

| Stage | Steps from initialization | Max root-task exposures | Required evidence |
|---|---:|---:|---|
| Stage 0: Preparation | 0 | 0 | data/stock/overlap audits; exact actor+logprob gates; baseline VALID before updating |
| Stage 1: online RL signal | 16 | 512 | real step throughput, rejection distribution, valid nonzero branch advantage, first clean VALID |
| Stage 2: primary run | 64 | 2,048 | 16/32/64 checkpoints with paired clean VALID, cost/quality curves |
| Stage 3: conditional extension | 128 | 4,096 | only if declared VAL signal + measured cost justify |
| Stage 4: optional scaling | 256 | 8,192 | separately budgeted coverage experiment, not mandatory |
| Stage 5: **required distillation** | AFTER best RL checkpoint | TRAIN-only separate search corpus | frozen search teacher → actual native CE update → distilled checkpoint |

No need to finish source D3 156,723 rows. If the same target is revisited in an exploratory follow-up, count **root exposures, distinct roots, unique canonical targets, on-policy decisions, model tokens and GPU-hours separately**.

**Validation**: 
- Preserve the D3 release VALID and its full 8,250 rows for source-native reproducibility, but do **not** claim it is target-/reaction-disjoint from D3 TRAIN. It has documented overlap. Use it as **secondary published-split diagnostic** at a selected finalist checkpoint, not as evidence of a strict clean holdout.
- Build a NEW frozen **clean internal VALID**, for example 512 distinct canonical targets with zero exact overlap with the selected RL training root products; ensure all references/targets from the same canonical product are grouped and kept off training. Audit route/reaction/patent overlap and report them. When feasible, report a stricter reaction-disjoint secondary subset, labelled with its actual denominator; do not claim reaction-disjoint if there is no sufficient coverage.
- For expensive evaluations use an additional 256-root fixed diagnostic subset of the clean VALID (nested in 512), with predeclared sampling seeds, at steps 0/16/32; run all clean 512 at steps 0/32/64 and on finalist extensions. This VAL set is for checkpoint selection; no public TEST target is used as a training tuning set.
- Evaluate in original product-only interface, with frozen stock, valid G4 independent rollouts, same actual actor inference budget, no ReFIT leakage into the policy-only selection metric. Report stock closure and reference-supported closure with paired target-level intervals, censoring and request cost.
- External **RetroCast Reference 1,084**, **PaRoutes n5 10,000**, curated 460 and Market660 remain **TEST**, never used to decide optimizer steps/reward weights/branch budget or distillation stopping. The previously planned 100 Reference cost panel is TEST-only and cannot serve as hyperparameter tuning.
- Obtain a real frozen Stage-II baseline VALID **before RL**; avoid the current schedule's deferred baseline/first nonzero validation at round ~613.

**Stop/extend rule, predeclared before actual TRAIN**:
- Stage 1->2 continues if complete receipts and numerical probability provenance pass, with no pathological censoring/excess cost; it is a software/cost gate, NOT a performance headline.
- Stage 2->3 is optional and requires positive paired clean-VALID evidence on the primary route-success metric or predeclared scientifically meaningful improvement in validity/credit-efficiency without a concerning decrease in closure; the exact minimum improvement and confidence criterion must be registered using Stage-0 uncertainty before trial and must be reported. Do not select according to the 100 Reference or PaRoutes n5 TEST.
- Stop after 64 or 128 steps if no useful validation trend, persistent noninformative GRPO groups, major numerical drift or resource ceiling; retain the best valid checkpoint with no ad hoc repeated TEST checking.
- Freeze a **real measured per-step wall-time and GPU-hour cost gate** after the accelerated backend's first complete new rounds; don't assume a proposed 64/128-step condition takes a specific wall-time. If the budget is exhausted before the step target, stop as budget-limited and report the actual steps/exposures. Do not call it a completed 64-step condition.

## 5. Scientific fairness: the required four comparisons

**Factorial A — isolate algorithm, fix training data/initialization**:
- A: shared Stage-II SFT, no RL.
- B: Process GRPO (current same-D3 independent G4, no external teachers).
- C: failure-aware Process GRPO and reward-conserved FLOW credit, no ReFIT conditional replacement.
- D: ReFIT-PO with exact same Stage-II actor, source D3 selected TRAIN streams, reward/validation/optimizer protocol.
- Every condition uses real on-policy sampling; the exact same prompts/route row IDs are fed in predeclared order, but generated action sequences will differ naturally. No RetroChimera or extra forward FlowER.
- Two separate learning comparison axes: (i) equal source-root exposures/optimizer steps, and (ii) **equal physical model decision requests or GPU-hours**. D uses extra branch samples; giving it more compute and claiming this is pure algorithm improvement is not acceptable. For equal-physical-budget B/C, allocate equivalent real independent on-policy root groups rather than copy D's branches.
- Test all selected final checkpoints with **identical inference-time** ReFIT budget, source stocks, model-call counts and selection criterion; also report no-search / small-search to distinguish learned policy from search gain.

**Factorial B — isolate training-data variation**:
- D1 legacy curated TRAIN vs D2 route-holdout vs D3 reaction-holdout, all from exact same Stage-II actor with their own frozen source manifests and data cohort.
- Within each dataset vary B/C/D using that dataset's own matched conditions. Do NOT compare D3-D to D1-B and call the difference a ReFIT-PO algorithm gain.
- D3 reaction-holdout only protects PaRoutes n1/n5 TEST reaction signatures **at route-RL stage**; it does not prove the FlowER pretraining lacks exact TEST reaction/product/scaffold overlap. Include base-SFT exposure audit and source-specific performance breakdown.
- Same-stock training conditions must be explicitly named; current guarded URSA TRAIN stock is not the source PaRoutes evaluation inventory. PaRoutes n5 TEST uses its published stock. No test route reference leaves injected into RL stock to inflate closure.

**External baseline fairness**:
- RetroChimera (Pistachio or USPTO-FULL) + Syntheseus Retro*/MCTS, AiZynthFinder, other reproducible strong planners must use the exact TEST roots and source inventory; report checkpoints, training sources, native search budgets, time/hardware, original request/expansion metrics and reference-route quality.
- Pistachio RetroChimera training data are different: this is a **whole-system/open-enhanced comparator**, not a same-data same-backbone algorithm ablation.
- Do not insert RetroChimera as Standard-D training reward or distillation teacher; optional +F uses the separate original forward FlowER only after accepted FINISH and must be separately labelled.
- Training step count is not a fair resource proxy if actual decision count and tokens differ. Provide a Pareto chart of VAL/TEST solved rate vs physical model requests, vs wall/GPU-hours.

## 6. ReFIT search-to-policy distillation is the LAST mandatory phase

After best 64- or 128-step Standard-D actor selection:
- Freeze it and collect extra, higher-quality ReFIT search trajectories on **TRAIN-only targets**, under a separately capped search budget. Choose e.g. a preregistered 2,048–4,096 distinct training-root corpus from a frozen D3 pool; no need for all 156,723; ensure none of the clean VALID or public TEST roots are included.
- Keep executed edge ledger, failed proposals, exact input states, native assistant action tokens/EOS, branch suffix/route Q, evidence/trust tier and source actor SHA.
- Realize the already implemented native weighted search-selected CE objective, using valid search-supported actions as positive targets, explicit failure comparisons, and SFT/actor-state anchors where verified. No GRPO on deterministic PUCT-selected search proposals.
- Train an actual separate distilled checkpoint, evaluate vs its own pre-distill RL actor at **zero search**, equal small ReFIT, equal larger ReFIT. Distillation quality = amortized route success and cost, not lower CE alone.
- Reserve a material fraction of project compute for distillation and final benchmarking. An initial allocation could be **50% RL, 25% TRAIN-only search+distillation, 25% clean VALID/public benchmarks and external baseline work**; this is planning, not a claim that these costs are already known.
- If optional +F is pursued, train/evaluate/distill it separately; do not delay completion of teacher-free Standard-D.

## 7. Required Codex changes and verification

This is a **new contract and scheduler**, not a patch to the active long full-epoch job. Codex should:

1. Implement versioned step-budget training mode in scripts/prepare_external_route_rl.py and scripts/train_external_route_rl.py (actual private runtime mirrored under docs/audits). Do not use source epoch traversal to derive total rounds. Sample frozen root target stream from D3 with exact manifest, then schedule 32-root group updates. Correct resume/optimizer actor-version/seed/checkpoint/gradient normalization.
2. Implement TRAIN candidate inventory auditing, target multiplicity, route/source grouping, TRAIN vs clean internal VALID membership/route/reaction overlap, base-StageII pretrain-vs-test overlap where accessible, and source-stock reachability.
3. Freeze one clean internal VALID / nested diagnostic subset; record checkpoint-selection metric and stop/extension rule. Public TEST never used to choose RL budget.
4. Support new small ReFIT credit lane (max forks 2 / max nodes 4); preserve the same-state Q estimator and reward conservation; perform exact original G4 root trajectory completions without synthetic cache-generated rollouts. Track credit-group value/coverage vs actual work.
5. Create complete initial-actor VAL and all registered checkpoint VALID before claiming improvement. Exact per-step physical requests/GPU-hours and per-request sampled logprob provenance; investigate PR112's HF incremental / full-teacher logprob discrepancy instead of hiding it by dropping gate or changing reward.
6. Build matched same-D3 B/C/D conditions and fair cost-normalized comparisons, preserving baseline original Stage-II weights and source target order. Do not launch all three automatically; only prepare independent frozen configs until approval.
7. Preserve the terminal TRAIN-only self-search/distillation study as a required separately budgeted deliverable and later build an actual distilled checkpoint.
8. Do **not** mutate/restart current D3 step19 and D1 step89 PR112 jobs, or claim full-source passes are required for science. Require explicit user approval for any scheduler replacement, cancellation, new training submission or inference backend reconfiguration.

**Focused acceptance tests**:
- 32 root tasks×G4 grouped correctly; exactly one on-policy policy version per update; global N×G×1024 fixed normalization; same terminal reward/credit on identical receipts.
- 64/128 steps stop independent of 156,723 input-row count; deterministic target list/strata and resume hashes; optional 256 steps not silently entered.
- duplicate source rows do not silently overweight identical canonical roots; no route-source hidden reference in Student prompts.
- clean VALID contains no training target products; reported route/reaction/patent overlap honest; D3 release and SFT exposure scope accurately disclosed.
- zero-step stock starts separately scored, resource/budget censor not negative chemistry, and test inventory isn't constructed from evaluation answer leaf molecules.
- B/C/D share predeclared training root stream and Stage-II init; equality of step count distinguished from equality of model requests and GPU-hours.
- full G4 completed; ReFIT extra branch pruning never supplies copied Q/rewards to a fake new policy sample; cached chemical Executor result ≠ shared stochastic trajectory.
- checkpoints evaluated on clean VALID before TEST; best checkpoint locked to metadata; TEST results never feed back to RL weights/step choices/distillation targets.
- posterior-accepted improved checkpoint and distillation checkpoint are distinct with own SHA.
- no external RetroChimera or extra forward FlowER in teacher-free Standard condition.

## 8. Expected outputs (not yet measured)

- NEW model/optimizer/cost contract, step-count schedule + frozen selected source target manifest and train/val/test exposure reports.
- Baseline Stage-II VAL and B/C/D 16/32/64/optional128-step results with full trajectory/decision/credit metrics.
- Evidence for physical request/token/GPU-hour savings vs old full-epoch condition and policy improvement vs SFT; no fabricated speedup when only code changed.
- Final search-to-policy distilled native tool actor checkpoint, plus no-search/small-search/budget-matched full-route evaluations on original TEST sources.
- Auditable README with exact source hashes and reproducible commands, optional completed model registry and manuscript figure data.

**One-sentence goal:** demonstrate ReFIT-PO policy gains **per verified on-policy sample and per GPU-hour**, rather than require coverage of an arbitrary 156,723-row route corpus; then amortize high-budget ReFIT search into a standalone distilled Retro-E policy.
