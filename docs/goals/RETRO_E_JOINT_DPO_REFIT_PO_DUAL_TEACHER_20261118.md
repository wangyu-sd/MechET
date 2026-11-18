# Retro-E Joint ReFIT-PO + Teacher-guided DPO
**Actionable Codex implementation plan | 2026-11-18 | design only**

## Decision

**YES: train ReFIT-PO and DPO in the same Student actor optimization loop.** Do not complete a full ReFIT-PO epoch first, nor train a separate DPO backbone. One on-policy branch sampler produces the actual electronic candidate actions. A frozen dual-teacher evaluation service enriches the resulting reaction/route evidence. A single optimizer step combines the route RL loss and same-state preference DPO loss. The final mandatory **TRAIN-only ReFIT search-to-policy distillation** remains after RL.

**Controlled comparators must still exist:** R (ReFIT-PO without external teachers), P (DPO alone under the same two frozen teachers), and J (joint ReFIT-PO+DPO under the same teachers); common Stage-II Qwen3.5-0.8B History-89442 initialization, D3 TRAIN source selection, clean VAL and fixed inference contracts. J is an **explicit teacher-enhanced** condition; do not label it the teacher-free Standard method, nor use J-vs-R to claim a pure ReFIT-PO advantage. Existing D1/D3 jobs need not finish full datasets and must not be silently modified.

## 1. Teacher responsibilities and what is actually fine-grained

| Source | Input available | Evidence output | When evaluated |
|---|---|---|---|
| Executor (always) | Current mapped BE/charge/bond state + native action | Hard accept/reject, exact next state, repeat/loop validity | Every generated action, cheap |
| RetroChimera frozen one-step inverse model | Product at the current reaction occurrence | Alternative reaction centers, edits/templates and precursor candidate support | Compute/cache once per distinct product or frozen initial context; compare real executor FINISH outcomes |
| Original FlowER frozen **forward** model | Complete precursor-side participating reactants, mapped composition | Forward product recovery/support and (only where authentic alignments exist) BE intermediate compatibility | **After an accepted FINISH**, batch/offline, cache by full reactants + model/contract |
| Hidden D3 TRAIN route references | Connected reaction graph and inventory obligations | Route full closure and trusted prefix progress | At completed rollout, no reference graph in actor observations |

**Important distinction:** Neither RetroChimera product-to-precursor candidate probabilities nor full-reaction FlowER forward outcomes automatically assign a correct label to an arbitrary partial reverse FLOW. Use actual observed intermediate states, atom mapping and executor receipts to attach teacher evidence to eligible chemical transitions. Otherwise attach the evidence at FINISH and transfer bounded credit to existing FLOW actions through the already implemented conservative \`route_flow_credit.py\` mechanism. No fake electron-arrow teacher labels.

### Teacher scorers

1. **Inverse evidence \`s_C\`:** teacher's support for the **observed complete committed precursor multiset** / aligned graph edit or center. Normalize/calibrate on TRAIN to a declared bounded [0,1] support statistic; missing rank/support is \`unavailable\` unless teacher actually evaluated this alternative with a defined candidate/scorer. Do not misinterpret absence from a top-K list as proof of chemical invalidity.
2. **Forward evidence \`s_F\`:** run original frozen FlowER on full precursor-side species; assess correct target product reconstruction under fixed atom-map/stereo conventions, optionally the reversed intermediate BE sequence only if the forward path can actually be mapped. When the input lacks required reactants or conditions mark unavailable. No direct inverse partial-state query.
3. Require separate \`teacher_model_sha\`, \`teacher_data_source\`, exact input hash, exact source-target context, observed topK/beam/sampling settings, coverage/error flags and per-query GPU time. Teachers have no gradients.

For a genuinely eligible committed reaction define a bounded optional **teacher bonus** once per unique committed reaction edge:

    e_rxn = w_C * mask_C * s_C + w_F * mask_F * s_F

Use a small, configured magnitude relative to full route closure +1 (proposal to test: w_C=w_F=0.05, no claim calibrated). Do not turn both missing scores into zero or rescale absent teacher weight to inflate covered ones. Keep the absolute component and availability masks separate. A teacher-supported reaction whose path is not stock-closed may still be useful in training; exact reference-match is not a precondition for a teacher-backed **local** preference label.

Redistribute an explicitly declared fraction alpha_teacher of this already counted FINISH bonus back to conservative, actual accepted contributor FLOW receipts **without changing its total amount**; no reward for FLOW mere acceptance, repeated edits or undo/re-do paths. Keep old verified reference-progress FLOW redistribution alpha=0.5 intact as a **separate** conserved source. Invalid actions still receive existing bounded executor cost. Teacher bonus should be disabled when confidence/mapping is unknown.

## 2. Same-state DPO pairs from REAL policy samples

At a ReFIT decision node v, retain only proposals sampled under the exact same policy observation, local accepted history, occurrence and behavior actor version. Pair different **semantic FLOW** actions \`a_i\` from the same parent. Use their real sampled native tool-call token IDs (including EOS). Choose/reject labels come from frozen evaluator on the actual continuations, e.g.:

    score(v, a_i) = route_suffix_return
                    + valid_masked_teacher_chemical_evidence

Only construct \`(a+,a-)\` when both actions have comparable observed suffixes, teacher coverage for any compared component is symmetric/explicit, and a predeclared score-margin is exceeded. Distinguish preference tiers:
- Strong positive-vs-negative: valid teacher-supported action with verified productive reaction vs Executor-rejected/invalid action.
- **Primary chemistry-rich pair**: two executable FLOW siblings with different verified reaction/route evidence; prioritize these instead of filling DPO solely with trivial invalid FINISH.
- Route-success differentiators: same-state sibling yielding supported stock-closed route vs unsuccessful completed route; teacher score breaks ties or assists partial progress.
- If either teacher is unavailable, use only the fully comparable sources or skip that teacher for the pair. Never make missing coverage a negative chemical label.
- No teacher-generated action strings as direct student targets unless independently compiler/executor checked against the exact student state; first release uses **only sampled student actions**.

### DPO objective and joint optimizer

For a frozen Stage-II (or separately declared fixed) reference \`pi_ref\`, full native action likelihood \`log pi(a|o)\` is the sum of generated assistant token log probabilities for that action:

\[
L_{\mathrm{DPO}}
=-\mathbb{E}_{(o,a^+,a^-)}
\log\sigma\left(\beta_{\mathrm{DPO}}
\big[
(\log\pi_\theta(a^+|o)-\log\pi_\mathrm{ref}(a^+|o))
-(\log\pi_\theta(a^-|o)-\log\pi_\mathrm{ref}(a^-|o))
\big]\right).
\]

\[
\boxed{L_{\mathrm{joint}}=L_{\mathrm{ReFIT-PO}}+
\lambda_{\mathrm{DPO}}L_{\mathrm{DPO}}.}
\]

Use **one shared actor, one frozen reference, one optimizer update per collection**. Both losses can be computed from the same already generated native actions, reference logprob cache, and current policy forward, avoiding duplicate model inference where possible. DPO is supervised on observed on-policy preference pairs; ReFIT-PO retains real behavior policy \`logp_old\`, action-level RTG, exact sibling conditioning and unique receipts. Do not pretend DPO loss uses importance ratios or feed deterministically selected PUCT data as on-policy GRPO.

Starting hyperparameter suggestions ONLY: beta_DPO=0.1; lambda_DPO=0.1; cap pairs per root, balanced between useful valid-vs-valid chemistry and invalid-vs-valid. If no valid DPO pair, RL continues using its original objective. Monitor combined loss, separate DPO/GRPO gradients, objective scales, policy KL and real route outcomes. Teacher bonus plus DPO reuses some evidence on purpose; ablations must identify its value and guard against teaching teacher biases.

## 3. Efficient concurrent pipeline

**No separate full DPO training run is required.** Use one same-policy collection version:

    D3 TRAIN root batch (frozen identities)
               |
         Student G4 + ReFIT
               |
         Real Executor receipts
             /   \
      ReFIT/route  Teacher evaluation queue
      suffix Q     RetroChimera product cache
                   FlowER accepted-FINISH batching
             \   /
      Verified combined reward ledger
       + matched-state DPO pair builder
               |
      joint loss (RL + lambda*DPO)
               |
            one update
               |
      next actor version

- RetroChimera cached per product/context/teacher version; no model request per individual FLOW.
- FlowER only for distinct eligible accepted FINISH precursor-side compositions. Use a **durable asynchronous queue** (or separate offline scoring microphase if no additional GPU is free), not a blocking forward teacher call per generated electron step. Exact reaction cache and in-flight dedup.
- Use PR112's global GPU request scheduler and no unnecessary extra stochastic suffixes; set a new **step-budgeted joint condition**: 32 unique-root tasks/update, G4 root rollouts, ReFIT at most 2 additional credit suffixes/tree initially, M=1, target 64 updates with 16/32/64 clean VAL; optionally 128 steps only if useful. This schedule is a new condition; benchmark it, don't silently alter existing running jobs.
- Real-time teacher availability does not change the original Executor outcome. A batch for which teachers could not complete uses the declared baseline route-only ReFIT loss and skips unavailable teacher-comparison DPO pairs; publish coverage and timeout denominators. Prefer batch-closed teacher results so a completed teacher result cannot arrive after the student version changed and be attached to an unrelated newer policy.
- With a strict maximum of 8 H20 already allocated across two existing 4-H20 jobs, do **not** claim an extra dedicated teacher GPU. Either share a time-sliced GPU between collection phases or reallocate after an explicitly authorized future job configuration. Precompute RetroChimera product records where possible to minimize live teacher memory; FlowER reward can be cached and batch-computed in a separate microphase.

## 4. Fair comparisons and datasets

**Primary combined-method comparison** on the same D3 release TRAIN root stream, frozen common initial actor:
- R: teacher-free ReFIT-PO baseline.
- P: DPO from both frozen teachers without ReFIT-PO branch-policy-gradient replacement, but with the same candidate data generation budget.
- J: ReFIT-PO + dual-teacher DPO **with teacher bonus optionally also in route RL reward**.
- J-minus-reward: teacher-guided DPO + route-only ReFIT-PO, no teacher bonus in RL, to isolate whether the teacher is needed twice.
- Optional +C/+F teacher ablation (RetroChimera only, FlowER only, both), only if budget allows.

Use same target pool, equivalent actor-step budgets and also equal actual physical model requests/GPU-hours. Report teacher GPU hours separately. Evaluate both no-search and fixed-budget ReFIT at the same source stocks.

Clean model-selection VALID: target-disjoint from selected D3 TRAIN roots, with honest route/reaction/patent overlap. Do not tune to RetroCast Reference 1,084, PaRoutes n5 10k or custom 460 TEST. Check original FlowER SFT pretraining vs TEST exposure separately; PetroChimera Pistachio pretraining is an **external-data** teacher and changes the data fairness class. Therefore J is a **Teacher-Enhanced / Open-Data track**, while R remains controlled Standard. Do not claim J-vs-R is equal-data or teacher-free.

## 5. Mandatory final search-to-policy distillation

After selecting the RL/J actor on clean VALID, run TRAIN-only high-quality ReFIT search, freeze real receipt/route evidence, and distill actual chosen/native tool-action targets into that actor using the preexisting weighted native CE code. Output an independently trained \`J-distill\` checkpoint (and R-distill for the controlled baseline if budgeted), then measure zero-search, low-budget search, equal-search-budget complete route performance. This is **distinct from DPO**: DPO is pairwise guidance during RL; final search distillation amortizes high-budget search into the deployed actor.

## 6. Implementation checklist and completion gate

1. Inspect PR112 current live/private training interfaces: \`src/mechet/route_refit_credit.py\`, \`route_flow_credit.py\`, actual \`external_route_rl.optimize_routes\`, ReFIT state IDs, output receipts and native actor/ref logprobs.
2. Add frozen \`TeacherEvidence\` schema + async cached \`RetroChimeraProductScorer\` + \`FlowERForwardReactionScorer\`, coverage stats and a deterministic trust/mapping layer. No new trainable teacher.
3. Add reward-conserving teacher FINISH/flow redistribution on eligible reactions; preserve original reward and failure semantics.
4. Add same-state \`PreferencePair\` builder and native sequence DPO loss using existing token-logprob forward. Pairs must have exact compatible conditions; failed/missing/reused receipts cannot be invented.
5. Add joint loss in the one existing optimizer, with separate finite/gradient/KL diagnostics. Keep true old behavior logprobs for ReFIT-PO.
6. Use 64-step step-budgeted D3 schedule, cached teacher microphases and PR112 global GPU schedule; compare R/P/J on the same frozen root stream and measured costs.
7. Preserve final TRAIN-only search-to-policy distillation requirement and clean TEST boundaries.
8. No active existing D3/D1 tasks should be rewritten by this **design-only** document. Exact new runtime change/submission should be an explicit separate execution action.

**Short definition:** FlowER verifies **forward chemical consistency after FINISH**, RetroChimera proposes **product-conditioned inverse reaction knowledge**, DPO teaches **comparative local native FLOW choices**, ReFIT-PO teaches **long-horizon route-conditioned native FLOW choices**, and the final stage distills high-budget ReFIT search into a single deployable Retro-E policy.
