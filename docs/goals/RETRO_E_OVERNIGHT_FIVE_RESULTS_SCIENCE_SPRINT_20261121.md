# SUPERSEDED — PR117 scientific goal has been revised (2026-11-21 CST)

**DO NOT execute this earlier five-result-first checklist as the primary instruction.** The user's new goal is convergence to ONE strong electron-flow-granularity retrosynthesis method with bounded Codex Autoresearch and an evidence-driven redesign of NMI Results. Read the NEW authoritative goal first:

[PR117 Autoresearch Method Convergence and NMI Results](PR117_AUTORESEARCH_METHOD_CONVERGENCE_AND_NMI_RESULTS_20261121.md)

The previous overnight checklist is preserved below for historical provenance and optional compatible evidence-task detail. Where the two documents differ, **the new Autoresearch goal takes precedence**, including model limits, method selection, finite loops, scientific falsification, and permission to revise Results. Do not assume a PR commit or GitHub comment launches Codex/GPU work.

---

# Retro-E Overnight Scientific Sprint — Five Results, One Electron Actor
**Author-approved goal: 2026-11-21. CODEX: EXECUTE, not another planning cycle.**

## 0. Goal and hard scope
Deliver by the end of this overnight Codex working session an **honest, machine-reproducible SMALL evidence dossier covering all five current Retro-E manuscript Results (Figures 2–6)**, with fresh experiments where hardware permits, authentic archived analyses otherwise, and clearly flagged missing experiments. Seek a scientific discovery, not a fabricated all-positive result. An overnight sprint **cannot establish** the full Reference 1,084 + PaRoutes n5 10,000 final leaderboard, full D3 training effects or an unavailable distillation checkpoint. Do not pretend it has.

**Paper source of truth:** read MechET-paper/main/paper_parts/03_experiments_results.tex and MechET-paper/main/EXPERIMENT_MATRIX.md. Current main paper has **five Results and six main figures**:
R1/Fig2 whole-route benchmark; R2/Fig3 same-state FLOW credit; R3/Fig4 electron-level versus reaction-level search; R4/Fig5 search-to-policy distillation; R5/Fig6 chemical disconnections/complete routes. Fig1 is the method diagram, not a new experiment. An uncommitted alternative four-Result/SEDS outline must NOT replace the existing paper contract in this sprint.

**Core method unchanged:** correct original Qwen3.5-0.8B Stage-II History-89442 actor, native import_fragments/apply_electron_flow/finish_trace as ONLY reaction generation actions, authentic Executor transitions, AND/OR pending precursor obligations, FINISH process reward, and real CRN-JEPA PR18 online reverse/verified-forward evidence wherever integrated. 0.8B is the ONLY actor to run tonight; absolute research ceiling 4B, NO 9B, NO second strategy LLM, NO new route value network, NO new electron-program architecture, NO new full-data SFT or RL training. Existing D3/PR116 weights and results unchanged. The desired novel small probe is **short route context + root-level distinct early electronic FLOW candidates + real cross-reaction backtracking** using the SAME actor.

**Absolute hardware isolation:** an 8-H20 training job is actively advancing. Never kill, restart, repurpose, preempt, reconfigure, or take GPUs/outputs of the existing D1/D3/PR116 runs. Verify actual Taiji instance/device occupancy and filesystem owner before any new work. Use archived CPU-only evidence first; use one genuinely FREE H20/A100 or otherwise preallocated GPU ONLY if verified. No paid cloud job and no assumption of spare cards. If none are free, complete offline R1/R2/R5, implement/test search changes without GPU, prepare exact resume/runnable commands and mark live comparisons BLOCKED. Do not wait uselessly for eight H20.

**Storage mandatory:** obey repository AGENTS.md, read PROJECT_MEMORY.md and source manifests before jobs. On the actual host source /aaa/fionafyang/buddy1/whaleywang/MechET/configs/runtime/shared_host_storage.env and verify every model/data/temp/test/pkg cache and output resolves to shared Ceph under /aaa/fionafyang/buddy1/. Never store new data in /tmp or /root/.cache. Use isolated Git worktree and a brand-new outputs/diagnostics/retroe_overnight_science_20261121 directory. Never mutate existing experiment receipts.

## 1. Read actual code and existing proof before modifying anything
- Check source SHA and contents of MechET main; PR110 feat/pr109-standard-refit-20261116; PR112 feat/pr111-accelerated-runtime-20261117; PR116 docs/joint-forward-replay-4h20-protocol-20261118; and newer stacked implementation branches. PR116 may contain only design, not the proposed new pipeline. Select a real runnable code revision; do not invent flags or imports.
- Inspect the actually available route_policy_episode.py, route_refit.py, route_refit_runtime.py, route_native_observation.py, route_execution_cache.py, route_distributed_collection.py, and the external-route trainer (train_external_route_rl.py in the contract-pinned training checkout), or the corresponding PR110 source mirrors. Verify exact CLI before execution.
- Read accessible immutable PR116 old 64-update/2048 TRAIN/512 VALID receipts, 2026-11-21 52-target cycle audit, PR110 D3/D1 receipts, D3 reference DB and stock manifest, TRAIN-only distillation receipts, final student checkpoint if any, CRN-JEPA actual integration (not merely PR18 specification), 1084 Reference manifest, and original paper experiment matrix. Inventory statuses: REAL, ARCHIVED, MISSING, INCOMPATIBLE.
- D3 reaction-holdout-n1-n5 has 156,723 TRAIN and 8,250 VALID **source route rows**, NOT distinct unique target roots. D3 source routes have no authoritative electron-arrow action labels. The old 512 VALID cohort is not automatically clean after expanded D3 training. Keep source stock, isotope/stereo identities and source overlap exact.

## 2. ONE minimal new inference algorithm, frozen 0.8B actor
Run a fixed-actor comparison under SAME verified stock, target IDs and **total actual Qwen generation-request budget including rejected/duplicate actions**:

A. **Original** PR116-compatible policy-only independent route inference, unchanged observation/tool contract.
B. **Route-context**: identical actor but add only root target, current product, exact current-branch ancestors, pending AND precursor count/identity and failed fully completed precursor outcomes to current observation; route histories must not become hidden gold hints. This is a changed-prompt inference intervention, NOT unchanged-policy on-policy RL.
C. **Route-context + root FLOW branching/backjump**: same policy. At the initial target reaction and optionally one hard descendant reaction, generate up to 3 actual distinct early FLOW proposals, execute each on a temporary authentic state, keep real alternate states and their original receipt identity. Expand only authentic alternatives to first FINISH; valid unfinished branches are UNKNOWN, not false successes. After a FINISH would regenerate an ancestor, do NOT repeat FINISH four times. Restore the causally responsible earlier electronic node/checkpoint and choose a genuinely different authentic FLOW; if no alternatives after a bounded number of attempts, stop and report. Never fabricate candidate precursor SMILES, policy logprobs, unseen continuations or proof of chemistry.
- Existing import/flow/finish chemistry semantics, Executor and stock closure unchanged. PR18 online scorer must stay correctly BEFORE selection/commit if available, and both arms have the identical scorer condition; missing forward replay must be marked unavailable, NEVER faked.
- Replay original bad case 4b6b7bd7e582; verify return from B -> A cycle to the A-root electron decision is possible without duplicating tokens or imaginary outputs.
- Test cases: root in-stock never a multi-step solution; one FINISH not all-stock closure; duplicate semantic FINISH is ONE candidate; different accepted FLOW successors remain distinct; no ancestor cycle commits; restored snapshot preserves pending AND obligations and actor receipt history; selector-conditioned likelihoods explicitly recorded and not misrepresented as native-policy GRPO.
- Do not train another planner, CRN-JEPA, general search-value model, 4B actor or full D3 checkpoint overnight.

## 3. Target cohorts frozen BEFORE inference
**Historical diagnostic 52:** two 26-target Nov21 audited cohorts, including 19 ancestor-loop endings (not new unseen TEST). If original rows unavailable, record precise blocker, do not reconstruct molecules from memory.
**Independent small Reference panel:** if free GPU, pre-register up to 48 distinct targets (16 linear/16 convergent/16 long) from Reference 1084, based on fixed seed and source-defined stocks; scale to 8+8+8 if time/resource constrained. Do not hand-pick successes. Report reference/flow pretraining overlap. This is exploratory, never the full 1084.
**CbyD true total-synthesis probe:** if accessible, use https://github.com/chemistry-by-design/cbyd; preserve commit/license, source citation, mapped reaction SMILES, conditions and official build_route_tree.py outputs. Select 8–12 diverse natural-product targets *before* evaluation using declared ring/stereo/route-depth rules. Raw steps are human-curated, but inferred tree links are computer constructed; distinguish that fact. No CbyD ground-truth electron arrows. A reported market-stock closure needs an independent declared stock; literature terminal leaves are NOT automatically buyable stock. Measure key bond-disconnection coverage, chemistry executor eligibility and independent literature alignment even when no complete stock-closed route exists. If no CPU/network access, report missing without placeholder claimed results.
**NO full PaRoutes n5-10k, NO Reference-1084 full, NO 460 complete sweep, NO new D3 epoch** in this sprint.

## 4. Required outputs: EACH of the five Results gets a real status
Emit new immutable manifest.json and results/R1.json ... results/R5.json plus per-target CSV/JSONL for any actually executed condition. Each R file has status OBSERVED / REANALYZED / PARTIAL / NOT_RUN / BLOCKED, exact cohort/denominator, dataset/checkpoint/stock hashes, actual model calls/tokens/GPU-hours, genuine evidence path, claims supported, claims not supported and blocker. Use NULL not invented numbers.

**R1 — whole-route quality / Fig2.**
Reproduce historical fixed 512-VALID: Stage-II 124/512 and old ReFIT64 128/512 stock-closed INCLUDING 13 already-stock targets; 22/512 -> 22/512 fully reference-verified, with exact original cohort and costs. Cite as ARCHIVED historical, not final D3/public test improvement. With free inference run A/B/C on fixed small Reference subset; log per target nonzero-step full AND-stock closure vs reference-verified, ancestor cycles, censored statuses, root-already-stock, requests/tokens/Executor/CRN/wall time. Discovery question: does the search-state repair change true complete-route outcomes at matched effort, or only reduce redundant loops?

**R2 — same-state FLOW local credit / Fig3.**
Reanalyse saved PR110/PR116 actual sampled TRAIN receipts: exact prompt/state/route identity, eligible vs masked groups, distinct meaningful accepted FLOW successors, nonzero local Q-spreads and fraction without any stock-closed continuation, FLOW credit versus G4 fallback disagreement, usable contrast per 1,000 real Qwen calls. One replayable real electronic branching example. Offline estimator reanalysis is NOT matched-training causal improvement. If no same-D3 trained baseline, report that evidence gap; do not start a full matched-training control.

**R3 — electron versus reaction-first search / Fig4.**
Same frozen actor, stocks and budgets. Compare (i) independent whole-route search, (ii) reaction-first selection among true FINISH-completed precursor sets generated by same actor, and (iii) C early partial-electron branching. Reuse authentic R1 inference receipts/route pool. If (ii) cannot be genuinely run, label its comparison missing, NEVER generate fake independent baseline by rearranging existing electronic search log. Count unique accepted partial FLOW successors, distinct precursor sets, duplicate FINISH decisions, first-solution calls, route-depth strata, different electronic paths collapsing to same precursors. Discovery question: what added chemical alternatives exist BEFORE the FINISH boundary under the SAME cost?

**R4 — search distillation / Fig5.**
Read existing parent/student checkpoint, verified TRAIN-only distillation records and outcome-only imitation control first. IF a real student exists: small held-out policy-only parent-vs-student run, same IDs and stocks; optionally matched search budget. IF not: report verified TRAIN-only record eligibility, prepare minimal reproducible student command; only execute tiny LoRA if spare GPU confirmed and no interference. CE decrease or parent token likelihood on saved data is NOT evidence of improved autonomous route closure. Mark R4 NOT_RUN/PARTIAL rather than fabricate final student figures.

**R5 — chemistry / Fig6.**
Reuse R1/R3 genuine complete routes, require exact executed electron proof replay and every AND leaf in stock for a 'complete synthetic route'. Pick 2-3 examples by predeclared fixed-pool rule IF they exist; if none, show partial route cases labelled partial and negative findings. Draw RDKit molecules and electron source/sinks only from real emitted chemistry states. Separate exact source precedent, close chemical analogue, unresolved proposal and conflicting precedent; TRAIN reactions/reference cannot be shown to actor. CbyD natural product probe is an additional target challenge, not assumed solved by an automatically assembled source route. Find whether alternative early FLOW changes skeletal key bonds, permits nonmonotone preparation, or enables convergent obligations.

## 5. Priority order (finish actual evidence, not cosmetic reports)
Phase 0: inspect refs/checkpoints, occupied GPUs, hashes, storage + output manifest. Use one isolated worktree; don't block on all eight H20.
Phase 1: CPU-only existing receipt reanalysis (R2 and historical R1, R5 candidate inventory).
Phase 2: implement minimal root FLOW branching + real across-reaction backtracking with tightly scoped unit tests and replay of known 4b6... case. Preserve all old numerical evidence.
Phase 3: IF actual free GPU, run bounded frozen 0.8B small A/B/C and genuine comparison controller for R1/R3, same paid compute accounting; shrink target cohort if needed, don't promise unrealistic timeline.
Phase 4: evaluate any EXISTING search-distilled student or prepare real reproducible command (R4).
Phase 5: CbyD limited chemistry probe + source-verified cases; compact REPORT.md and PR comment. Run science first; no paper LaTeX edits or elaborate plotting before evidence.

**Optional compute cutoff:** target at most ~3,000 new native Qwen action-generation requests for this isolated exploratory sprint unless a separately authorized free GPU permits more. No endless suffix generation. If valid new FLOW branches remain absent after initial bounded cohort, STOP wasting requests and report the exact failure mode. Do not extrapolate new search quality from throughput or replay-only cases.

## 6. Continuous self-review / no goal drift
At EACH phase boundary, when introducing a new configuration, and BEFORE every R1-R5 scientific claim, write/update goal_guard.json:
- phase, sourced repo ref SHA, selected checkpoint SHA, dataset/stock hashes, target IDs, real allocated GPU IDs and ownership, real requests, generated tokens, elapsed wall time, actual new evidence files;
- a check array: no_9b, no_second_llm, no_new_large_training, no_interference_with_existing_8H20, no_reference_leakage, no_fake_executor_replay, no_all_stock_shortcut, no_fabricated_behavior_probabilities, no_fabricated_R4_learning_gain, no_denominator_switch, data_on_Ceph_only;
- decision CONTINUE / CUT_OPTIONAL / STOP_AND_REPORT with reason.
Any failed hard guard -> STOP that operation, repair scope before continuing. Never spin in recurring no-op review loops. Prefer useful honest partial results over stalled ambitious experiments.

**Independent falsification checks:** deterministic exact executor replay for every complete synthesis claim; distinct actor-receipt and data-source hashes; paired target-matched budget; original vs new route history intervention distinguished; CbyD natural-product chemical precedent independently cross-checkable; stock provenance audited. Negative and null findings are publishable scientific diagnostics, not failures to hide.

## 7. Mandatory final artifact set
Create, under new output root: manifest.json; availability.json; goal_guard.json; results/R1.json through R5.json; authentic per-target result table if run; at least one stored real electronic fork or honest explanation of missing; natural_product_coverage.json if CbyD accessible; discovery_candidates.json; limitations.json; REPORT.md; exact executable reproduce.sh with real verified paths/CLI flags. No synthetic numbers in manuscript tables.
REPORT.md: one paragraph per R1–R5, cross-figure scientific finding (positive OR negative), what evidence falsifies it, and next decisive follow-up. Rank discoveries by actual scientific meaning, not optimism. Post a short PR comment with actual execution timestamps, checkpoint+source hashes, actual hardware ownership, completed target counts, R1–R5 status and only verified discoveries.

**This newly created GitHub PR/goal is a Codex handoff. It is NOT automatically running Codex or launching a GPU job. A Codex agent must be invoked in an authorized workspace to execute it.**
