# Retro-E: Progressive ReFIT-PO, Forward Verification and Final Search-to-Policy Distillation
**Research and Codex implementation plan | 2026-11-16 | proposal, not completed experiments**

## 0. Decisions to freeze

The original MechET implementation project is named Retro-E in the study; the existing GitHub repository name need not change. Preserve the paper's core: executable inverse electron-flow actions, Route Environment/Executor, Process GRPO, multi-step AND/OR obligations and partial-electronic-state ReFIT sampling.

**Exactly two research tracks:**
- **Standard / teacher-free:** FlowER-trained Qwen3.5-0.8B History-SFT checkpoint as the common **initial Student**, but no additional FlowER forward inference and **no RetroChimera** during route-RL, reward, inference or final search distillation. Use executable process evidence, route outcomes and ReFIT-PO.
- **Forward-verified enhancement (+F):** the same starting checkpoint, training data, initial rewards, rollout and optimizer settings; only add frozen original FlowER forward verification on completed reactions, asynchronously/offline, with auditable coverage and time/cost. **Still no RetroChimera.** Forward-FlowER must be removable without changing Standard.
- **Retro-E-Pro / open-enhanced:** a separately labelled production-style system, potentially with RetroChimera, larger/fused training data, broader search and (later) forward FlowER; it participates in benchmark tables under an open-enhanced track and cannot be used to establish the isolated ReFIT-PO algorithm effect.

**FINAL STAGE IS MANDATORY SEARCH-TO-POLICY DISTILLATION.** After policy RL (+ optional forward-verified variant), use ReFIT search outcomes, sibling decision evidence and verified route certificates to train a **standalone amortized Retro-E policy**. Search is the source of better action targets; the distilled Student should run without large tree search or external teacher at deployment. Distill both Standard and +F policies separately under their own evidence contracts. Do not confuse this with RetroChimera teacher OPD.

No new DPO/Critic/PRM/hierarchical RL; no retraining the 0.8B base from scratch. This is a phased proposal: no already-running job may be stopped or altered, and this doc alone does not authorize production job submissions.

## 1. Existing paper / code authority

Paper: https://github.com/wangyu-sd/MechET-paper, main manuscript paper_parts/02_theory_method.tex and paper_parts/03_experiments_results.tex.

Existing method and benchmark facts in the manuscript:
- Stage-II source-locked FlowER-trained Qwen3.5-0.8B History-SFT, checkpoint-89442; underlying initial supervised training used 243,753 replay-approved FlowER reactions, not an unspecified universal full corpus.
- Route-RL TRAIN 23,730 and VALID 2,986 external targets, per existing frozen RetroCast-derived data protocol.
- The paper's four route-evaluation target cohorts contain 100 expert-selected + 100 drug/clinical + 160 convergent + 100 PaRoutes-derived = 460 targets. Each cohort has its declared frozen stock; NOT a single globally interchangeable stock.
- Inference-time electronic-state tree search is separated from training-time policy updates.
- Route success requires all non-stock precursor occurrences to close, and executed reaction receipts to be traceable; reference support and independent chemical precedent are distinct outcome grades.

Code: PR #107 SFT-native Process GRPO, PR #108 first nonresource rejection (draft), PR #109 ReFIT-PO Codex implementation design (this stack). The docs/audits/.../source snapshots may not be directly deployable public production files. Codex must discover actual runtime and reconcile latest PRs.

The new phases below are **future experiment conditions**, not claims of existing scores or available production checkpoints.

## 2. Keep sampling, evaluation, optimization and distillation separate

| Dimension | Module | Job |
|---|---|---|
| Sampling | ReFIT | Partial electronic-state branching, continuation budget, state/edge/route evidence; it is not reward or optimizer |
| Hard validation | Executor | Exact accepted/rejected chemical transition, inventory/bonds/charges, rejection semantics |
| Outcome evaluation | Route evaluator | Reference-supported progress, noncyclic stock closure, multi-step AND/OR terminal result |
| Optional forward evaluation | frozen original FlowER | Forward reachability from complete precursor-side participants after successful FINISH; not an absolute lab feasibility oracle |
| Optimization | Process GRPO + ReFIT-PO | Action-level returns; matched-state FLOW sibling relative advantage with original G4 fallback |
| FINAL amortization | ReFIT search-to-policy distillation | Turn costly search decisions into one policy's native action distribution |

Do not introduce RetroChimera into Standard or +F. It is reserved for Pro/open-enhanced, with explicit teacher model version and data disclosures.

## 3. Progressive training/evaluation conditions

### 3.1 Common origin

Freeze exactly one baseline Stage-II checkpoint SHA, actor/tokenizer/tool schema/Executor/stock-material contract. All **independent RL condition comparisons** initialize from that exact checkpoint. Keep a frozen reference policy.

Stage 0 — SFT baseline: no route-RL. Report independent greedy/sampling performance and matched ReFIT search performance.

Stage 1 — existing Process GRPO: PR #107 G4 independent full route rollouts; action-level RTG; clipped Dr.GRPO fixed normalizer; reference KL. Existing official training jobs are immutable.

Stage 2 — teacher-free **failure-aware executable process credit**:
- Validator cost only on the actual nonresource failed action: bad IMPORT/FLOW, premature or repeated invalid FINISH, semantic duplicate rejection, cycles; use exactly the frozen PR #107/#109 bounded cost, do not double-penalize. A valid single-FLOW FINISH is allowed.
- The failed **rollout** ends and receives failed terminal outcome/partial progress rollback; parent chemical state is not failed, other ReFIT siblings are independent. Resource censoring is not chemistry error.
- For a successfully FINISHed reaction with a reference-supported progress increment m, optionally move a bounded fraction alpha*m from FINISH to contributory accepted FLOW decisions using verified BE/bond/charge net-change attribution and proper atom map correspondence. Remaining (1-alpha)*m stays at FINISH. Sum of reaction progress reward must remain **exactly m**; no reward for merely accepted FLOW, longer traces, redundant back-and-forth actions or unsupported teacher coverage.
- Attribution must be conservative: if the net-edit contribution is ambiguous or the path cancels changes, leave the credit at FINISH. The endpoint net edit does not uniquely determine physical microscopic mechanism. Do not invent causality from a graph difference. The simplest correct implementation with alpha=0 must remain an exact historical Process GRPO fallback.

Stage 3 — **ReFIT-PO same-state branch-relative credit**:
- Keep ReFIT as the sampling protocol. From the exact same policy observation/prompt, local history, route context, reaction occurrence and behavior version, sample distinct FLOW actions under a known behavior policy.
- Complete each sibling with matched continuation budget; branch Q is the suffix reward through full route stopping. Same-state A(v,a)=Q(v,a)-mean_sibling Q. An explicit failed action can be a negative-return sibling, while a resource-censored sibling causes conditional comparison masking (not negative Q).
- Replace (NOT sum with) the original G4 action advantage for the actual branching FLOW receipt; all other valid independent G4 actions preserve current action RTG fallback. Shared prefix receipts are recorded/trained once, not multiplied by number of leaves.
- Any deterministic PUCT-selected, deduplicated, reject-sampled or otherwise biased search proposal without a known actual behavior distribution must not be inserted as native on-policy GRPO. Such data may inform the final distillation, with its search provenance labelled separately.

Stage 4 (OPTIONAL) — **+F frozen FlowER forward verification**:
- Trigger only for a completed, Executor-accepted reaction, after structural precursors and **full precursor-side participating species** are available. Feed the original forward model with a frozen, verified atom/molecule normalization contract. Some retro structural precursor bags omit spectator/reagent context; query failure must be marked unavailable, not chemical false.
- Predict forward products with frozen checkpoint and fixed sample/beam budget; endpoint match uses frozen product-set mapping/stereochemical policy. If available, return actual BE intermediate transitions, and add transition consistency only after verified alignments.
- First implementation reward is bounded +lambda_F times estimated forward recovery support at FINISH, **not per-FLOW pseudo-labels**. A completed, valid query that did not recover the target gives zero support, not negative chemical truth; teacher unavailable is masked; all query coverage and cost reported.
- To keep comparable GRPO groups, teacher bonus coverage/timeout must be a fixed group-level protocol. An episode lacking teacher response must not be silently treated as failed.
- No external FlowER inference in Standard. Avoid claiming forward model independence from Student's FlowER-derived pretraining data. No new FlowER reward until the teacher interface and conditioning are checked.

Stage 5 (REQUIRED, LAST) — **ReFIT search-to-policy distillation**:
- Freeze a trained Stage 3 Standard actor (and a trained Stage 4 +F actor, if that condition exists), and use it to collect **training-only** ReFIT searches and their executed state/edge/route receipts. Neither frozen 2,986 VALID nor any 460/1084/10k benchmark TEST target is eligible for teacher-search label generation.
- Each distillation state s corresponds to the exact SFT-native observation/prompt/tool schema and current chemical inventory; record canonical semantic action candidates, original raw text token IDs, accepted/rejected receipt, branch suffix value, target/route outcome, policy/checkpoint and source provenance.
- Trust tiers: (i) complete stock-closed and reference-supported route; (ii) formally stock-closed but reference-unsupported, kept separate/require stronger evidence for positives; (iii) partially productive but not solved, only for calibrated local evidence/coverage; (iv) Executor rejected and censored, never positive action targets. Do not collapse them into one positive pool.
- Build a **search-improved action target distribution** among real observed candidate actions in the *same exact state*:
  \[
  q_{\mathcal T}(a\mid s)
  \propto m_{\mathrm{eligible}}(s,a)\exp\{Q_{\mathrm{search}}(s,a)/T_D\}.
  \]
  Q_search comes from matched-budget verified suffix outcomes; this q is a teacher **training target**, not a calibrated probability of chemistry or an unbiased on-policy behavior distribution. Don't simply normalize PUCT visit counts and claim they are policy probabilities.
- Train a student policy initialized **from the corresponding RL actor**, without introducing a new backbone, with weighted action-sequence cross entropy:
  \[
  \mathcal L_{\mathrm{distill}}
  =-\mathbb E_{s\sim D_{\mathrm{search}}}
  \sum_{a\in A_s}q_{\mathcal T}(a\mid s)
  \log\pi_\phi(a\mid o_s).
  \]
  Log pi is the teacher-forced log likelihood of the full generated *native tool-call token sequence*, using only generated assistant tokens; do not mask out the action EOS by accident, do not supervise tools/prompts, and do not conflate native action normalization across differing lengths. Use temperature, value clipping and finite source weights only if declared in training config.
- Retain a modest anchor mix of replay-verified original SFT actions and actor-collected states to reduce forgetting and search-state covariate shift; no hidden route answer enters actor input. Do not train on a successful route chosen using external TEST labels.
- Failed actions may enter as **excluded or low-weight choices** in the same-state target normalization, but should not be added as positive behavior-cloning targets. If all candidate evidence is invalid/censored or Q is non-discriminative, skip that state's distillation target rather than producing arbitrary labels.
- Distillation is **offline supervised amortization** of search into a policy; do not run GRPO on deterministically selected ReFIT branches. Keep distillation checkpoint and teacher-search generator checkpoint distinct, with lineage in manifest.
- End product: (a) Standard RL actor, (b) **Standard distilled standalone actor**, (c) optional +F RL actor, (d) **+F distilled actor**; each can run no-search, small-budget ReFIT and fixed large-budget ReFIT. Standard distilled model uses no RetroChimera and no forward FlowER even during distillation. The +F model may use FlowER only for training evidence, with teacher-free inference as a measured deployment condition.
- **Success criterion** is not lower distillation loss: report policy-only first-action validity, single-step precursor completion, complete route Solved@Budget, route costs, and recovery of search gains under the same inference budget. Compare distilled versus its own RL parent and original SFT, with the **same** independent test targets, stock and budget.

### 3.2 Minimal algorithm comparisons

| ID | Actor / training | Search | Forward FlowER | Primary question |
|---|---|---|---|---|
| A | Frozen Stage-II SFT | fixed independent sampling and matched ReFIT | No | baseline |
| B | Existing Process GRPO | same fixed configuration | No | route RL gain |
| C | Failure-aware / net-edit Process GRPO | same fixed configuration | No | local credit gain |
| D | ReFIT-PO teacher-free | same fixed configuration | No | same-state branch advantage gain |
| E | D + frozen FlowER forward reward | same fixed configuration | Yes **during training only** | extra forward evidence gain |
| D-distill | Final distillation of D via TRAIN ReFIT search | no-search + matched small/large search | No | amortization of search |
| E-distill | Final distillation of E via TRAIN ReFIT search | no-search + matched small/large search | Optional during training, no teacher at baseline inference | amortization of enhanced policy |
| Pro | open-enhanced larger system | declared search budget | Optional | operational score, separate data track |

To isolate a factor compare B/C/D on the **same route training dataset** and D/E on the **same dataset and budget**. E must not inherit external RetroChimera. Distillation is evaluated as an additional completed stage, not quietly inserted into D/E.

## 4. Progressive benchmarking and honest data versions

**Two orthogonal axes**: training dataset (D0/D1/D2/D3) and algorithm (A–E, distillation). Do not call cross-dataset differences algorithmic causal gains. All versions share the same initial FlowER-trained checkpoint; each training data version has its own manifest and independent isolated RL condition.

Dataset versions to produce as frozen assets:
- **D0**: SFT-only, no multi-step route training.
- **D1**: current RetroCast-derived route TRAIN 23,730, frozen VALID 2,986, existing route identity audits.
- **D2**: source-defined PaRoutes **route-holdout** post-training version if obtainable under the official data license/version; verify exact scope and overlap first.
- **D3**: source-defined PaRoutes **reaction-holdout** post-training version if obtainable; stricter than complete-route exclusion but still does **not** erase reactions seen by the previously frozen FlowER SFT checkpoint.
- **D4 Pro-only**: broader source/license-compliant multi-source training. Never use to claim D1–D3 algorithm fairness.

No RetroChimera in D0–D3 Standard versions. Record reaction, product, route, source patent, structural/scaffold overlap and frozen stock inventories. "Route-RL reaction-holdout" is not "pretraining-clean chemistry" when the base SFT model saw related patents.

**Evaluation ladder** (same saved checkpoint at each rung; do NOT retrain at each rung):

1. **100 targets**: frozen, stratified subset of the manuscript's 460 (e.g. 20 expert, 20 drug/clinical, 40 convergent, 20 PaRoutes-derived) for early cost/error reporting only. **Do not use this subset to choose model weights, tune reward weights, adjust beam/budget or stop training;** checkpoint selection uses original 2,986 route VALID, which remains separate. The 100 are part of the 460, not 100 extra independent benchmark targets.
2. **460 targets**: full manuscript primary four cohorts, each evaluated under its original frozen stock and target denominator.
3. **RetroCast Reference 1,084**: 600 linear + 400 convergent + 84 long-reference-route targets under its declared fixed stock(s); first compare D vs B and D-distill vs D; stratify by depth and AND branches.
4. **PaRoutes n5 10,000**: formal large population extension; optionally n1 10,000 as additional robustness. Check test IDs/reaction ancestry vs the source route-training versions, and source stock. Do not train on these evaluation examples.
5. **RetroCast Market 660**: practical inventory/operational comparison; optional Pro can participate as a separately labelled **open-enhanced** participant. Do not pool Market rate with Reference or the 460 to form an unweighted universal solved rate.

Do not mistake the 460 manuscript targets or the 100 subset for disjoint from PaRoutes / RetroCast unless verified by target/source manifest. Overlap analysis is a report field, not an excuse to silently delete hard targets and change denominators. If benchmark/test IDs overlap the training corpus, create a properly isolated training version or explicitly label the affected comparison ineligible for clean generalization.

Metric ledger:
- primary: reference-supported *complete* route closure (full cohort denominator) and stock-closed nonzero-step route Solved@Budget, explicitly separate;
- route: unique route count, depth, AND/convergent obligations, cycles, unsupported closures;
- electron level: rejected FLOW, premature FINISH, repeated FINISH, invalid IMPORT, net-change contribution coverage, unique executed successors;
- search: model decision requests, generated tokens, Executor calls, search expansions, wall-time, memory, forward-teacher GPU cost;
- learning: % effective nonzero-Advantage sibling groups, all-failure groups, censored groups, flow-action invalidity, search-distillation state coverage;
- chemical evidence: reference exact, analogue, unverified and conflicting literature support when available; computational closure is not physical feasibility.

Reporting:
- **Controlled Standard**: no external forward FlowER, no RetroChimera; compare A/B/C/D and D-distill.
- **Controlled +F**: exact same Student origin/route data except forward teacher at training; compare D/E and D-distill/E-distill.
- **Open/Pro**: all additional models, data, search and costs disclosed; not used for claims about standard method.

## 5. Implementation sequence for Codex

### Task 0: reconcile current branches
Read latest PRs #107/#108/#109 and paper Methods/Results, identify actual runtime source tree vs audits/ source snapshots. Preserve running jobs. Do not edit active frozen contract or model adapter in place.

### Task 1: teacher-free fine-grained reward
Implement bounded Executor failures, per-occurrence FINISH guards, rejection ledger and parent backtracking per PR #109; test correct edge/rollout/tree boundaries. Add a separate codepath that redistributes a strictly conserved fraction of already-verified reaction progress to attributable FLOW edges; implement conservative matching and zero-credit fallback.

### Task 2: ReFIT-PO sibling optimizer
Implement exact same-decision-state identity, independent same-policy sibling sampling, branch suffix return, group-relative advantage replacing existing G4 only at the eligible FLOW edge. Verify shared-prefix uniqueness, proposal probabilities, resource censoring, deterministic PUCT search exclusion from on-policy loss. Keep original G4 as fallback.

### Task 3: train and test **Standard**
Use the frozen Stage-II init and existing route dataset; run existing B vs C vs D at matched resources. 100-cost panel then all 460 then Reference then n5, using same selected checkpoint; do not turn the 100 into model tuning. Keep full original 2,986 VALID for model selection.

### Task 4: optional +F forward verifier
Implement offline/asynchronous original FlowER from full precursor-side participants after FINISH, auditable request-response cache, group-consistent missingness, bounded FINISH-level reward. Quantify teacher availability and extra GPU cost before interpreting E gain. Compare D vs E on equal route sampling/training and inference budgets.

### Task 5: MANDATORY final **search-to-policy distillation**
Run TRAIN-only ReFIT high-budget search with trained D (and E if present), write state/action/candidate Q/outcome teacher targets, train distilled student via weighted native action cross-entropy starting from that actor, preserve SFT replay anchors, evaluate standalone and matched-search models against the undisclosed test outcomes. There must be separate, reproducible completed distillation checkpoints and evaluation reports for Standard, plus +F if implemented.

### Task 6: reporting and paper
Deliver model registry/manifest for all released checkpoints, data split/license/provenance ledger, frozen evaluation cohort IDs and stock checksums, reproducible command/configs, matched-budget score tables and actual measured Figure 2–4 data. Avoid copying historical one-step evidence into current route-RL/Distillation results. Keep proposed cells explicitly unmeasured until genuine full runs.

## 6. Hard acceptance checks

- Same frozen Stage-II 0.8B checkpoint initializes all compared RL conditions.
- Standard contains **zero** external RetroChimera or forward FlowER calls, rewards, labels and dependencies.
- +F contains no RetroChimera and uses FlowER only on accepted FINISH with full reactants/reagent context and frozen caching; unavailable response is not a chemical-negative label.
- Failure cost attaches to the rejected action; failed rollout closes with appropriate terminal zero and rollback, while other ReFIT siblings and parent state remain intact.
- A legitimate one-FLOW FINISH is accepted; premature/no-net-change/repeated rejected FINISH are distinct.
- Reference progress redistribution to FLOW is bounded and **reward-conserving**, with no reward simply for executing a valid but unproductive action.
- ReFIT remains sampling/search, not the optimizer and not teacher reward.
- ReFIT-PO does not treat PUCT deterministic selection as on-policy or duplicate shared-prefix token loss.
- The 100 subset is nested in 460 and never used for model selection.
- D/E and D-distill/E-distill use the **same** data, budget and evaluation contracts within each causal comparison.
- Distillation is **really trained** as the last phase, from verified TRAIN-only ReFIT search evidence, and student performance is measured at zero and matched search budget.
- Search distillation must not rely on TEST source route answers, on unverified "success" fragments, or on teacher actions that were never executable in the student state.
- Real checkpoints, hashes, full source-split denominators and test results are required before claiming improvements.

**Research claim under test**: executable electron-flow decisions can be improved with verifiable local process credit and long-horizon route outcomes; same-state ReFIT branch comparison improves credit assignment; final search-to-policy distillation can amortize some of the search gain into a deployable electronic policy. Forward FlowER is a separable extra experiment, not the prerequisite for a publishable standard method.
