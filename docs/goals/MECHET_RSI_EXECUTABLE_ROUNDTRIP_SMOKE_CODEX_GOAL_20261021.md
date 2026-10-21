# MechET-RSI — Executable Bidirectional Round-trip Policy Iteration
## Codex implementation goal + real GPU smoke, 2026-10-21

**Status:** EXECUTION GOAL. This PR deliberately contains a detailed, executable assignment and a proposed experiment manifest; it does **not** contain implemented RSI training, measured speedups, completed smoke results, new checkpoints, or claims of model improvement.

**Scientific aim:** Starting from the already trained, independent reverse and forward Qwen3-0.6B electron-flow agents, test whether forward reconstruction can create useful preferences for *model-generated retrosynthetic decisions*, improve the reverse agent itself, and sustain a second data-refresh improvement round. If successful, enable a short forward-agent update to close a bidirectional cycle. Preserve the existing MechET reaction representation and executable IMPORT/FLOW/FINISH semantics.

**Immediate delivery:** An actual, reproducible smoke, not another design-only PR. Codex should complete implementation, unit/integration tests, GPU candidate collection, forward scoring, genuine LoRA optimizer updates, and before/after paired validation. Continue through correctable engineering blockers; do not stop after a dry run and declare success.

### A. Source of truth and non-negotiable scope

- Code repository: wangyu-sd/MechET. Implement on **this PR branch**, based on main, rather than merging PR #83/#84 or their Graph–Qwen lineage.
- Current reverse policy: Qwen/Qwen3-0.6B base revision c1899de289a04d12100db370d81485cdf75e47ca; Stage-II compressed-history adapter at outputs/agent/natural_language_event_history_v2_qwen3_0_6b_seed17. Previously recorded adapter_model.safetensors SHA256: **40204c7a059d16976fc4b24b315c3bdde7bf61c705d24bbde515071757e9de75**. Authenticate on the actual machine, including tokenizer, adapter, SFT prefix, execution contract, and old manifest mismatch; do not fabricate a new lineage or use an unmatched adapter.
- Current forward-policy source: independent Qwen3-0.6B LoRA, training configuration configs/agent/forward_event_flowER_qwen3_0_6b_4h20_20261016.yaml and implementation src/mechet/forward_qwen_events.py, found on branch ops/pr83-route-greedy-20261011 at commit d77cf1584520dc2bb26dc409c12a59266eeb8a02. Expected Ceph output directory: outputs/agent/forward_event_flowER_qwen3_0_6b_4h20_20261016_seed17. **The complete forward model is not certified by this PR.** Inspect the live saved final weights or newest coherent checkpoint and record its actual SHA/step; Oct 17 snapshot last certified checkpoint-22000/59440 was a running-training progress observation, not final model quality.
- Frozen reverse/forward are **two independent LoRAs** on the same pinned base. Never overwrite either. No CRN-JEPA, Graph-Transformer/Graph-FLOW/GNN, alternative backbone, new value head, new generation language, or retraining of Stage-I/II.
- Port **only necessary files/functions** from the forward branch and the existing continuous-batching helper scripts/rolling_vllm_search.py into this main-based branch, with appropriate regression tests. Do not cherry-pick/merge its entire Graph-containing commit history.
- Existing main functions to reuse: scripts/run_natural_language_value_search.py (Node/Action/execute/policy_prompt), scripts/infer_dynamic_electron_trees.py (frozen policy generation and native execution records), src/mechet/endpoints.py (structural vs auxiliary), scripts/train_proof_dpo.py (cached frozen-reference log ratio), src/mechet/selected_policy_logps.py (action-token projection), src/mechet/group_policy_objective.py (optional comparison), src/mechet/assistant_masking.py (tool prefix and action masks).
- Leave all existing R1/R2/31k MCTS, manuscript, PR #89/#90 analyses, results, checkpoints and running tasks intact. Do not delete historical code, stop any GPU job, merge this PR into main, or modify MechET-paper.

### B. First deliverable: checkpoint and interface gate (real data)

Codex first inspects the ACTUAL cluster/filesystem and records:
1. R0 and F0 absolute paths, checkpoint/step, model revision, adapter weight SHA256, PEFT modules/rank, tokenizer SHA, runtime git commit, RDKit and vLLM versions; distinguish a final checkpoint from a partial saved checkpoint.
2. Read-only R0 prompt and generation parity with existing Stage-II: model sees original product, live executor current state/inventory and compact accepted history; actual Qwen SFT-aligned assistant tool prefix and correct <|im_end|> semantics. No GT action-type prompt split.
3. F0 independently exposes initialize_forward_state, forward_prompt, forward_tools and execute_forward_call. Forward policy sees **only the actual initial reactant mixture, current forward state/inventory and accepted forward history**, not the target product or a copied reverse trajectory.
4. A genuine baseline from frozen R0/F0: independent forward rollout from a FlowER TRAIN recorded reactant mixture, executable reverse rollout from a TRAIN product, plus a paired 32-ID validation set. Report terminal rates and structural agreement, not only teacher-forced local replay.

If the forward two-epoch final adapter exists, use it. If not, explicitly select the latest coherent saved F0 checkpoint with an honest provisional label and continue the smoke; never pretend a missing final adapter exists. Only block if no loadable genuine F0 checkpoint exists.

### C. Core algorithm: Cycle-Grounded Policy Iteration (CGPI)

This is an **iterative self-improvement training loop**, not an inference-time reranker. The required minimal recursive sequence is:

~~~text
freeze R0, F0
D0 = collect_train_only(R0) -> execute -> forward feedback(F0) -> state-conditional preferences
R1 = reverse_DPO(init=R0, fixed_reference=R0, pairs=D0)
evaluate R0 vs R1 using identical product-start protocol and frozen F0
D1 = fresh_collect_train_only(R1) -> execute -> forward feedback(F0) -> NEW preferences
R2 = reverse_DPO(init=R1, fixed_reference=R0, pairs=D1)
evaluate R0, R1, R2 on same frozen validation IDs and budgets
then (if comparable forward preferences exist) train F1 from F0 using frozen R2 feedback
~~~

D1 must be freshly generated **by R1**. Running another epoch over D0 and calling it RSI is not allowed. Keep R0 as immutable reference for both reverse DPO rounds, and publish the fresh-D1 versus static-D0 continuation control when the preference pool permits. Do not introduce separate RL value/critic training in the smoke.

**Reverse generation**: independent autoregressive Qwen generation of complete real IMPORT/FLOW/FINISH tool calls; executor is the sole owner of chemical state. Preserve every rejected attempt, state cycle, truncated action and finish decision with token cost. Generate 2 initial candidates per TRAIN target; if candidates are duplicate / uninformative, selectively grow to 4 and at most 8, accounting for all additional work. For smoke, never run full R1/R2 stock MCTS.

**Forward bridge**: define a reusable reverse-complete-reaction -> forward initial-mixture and forward-step-witness interface. Preserve mapped executor states, initial product, reaction actions and import provenance. Split full structural precursors and auxiliary species using existing endpoint tools, and keep the full-mixture variant as a separate well-defined experiment. Re-map only from the chosen forward initial mixture. Invert observed executed **state transitions**, not the natural-language electron arrow text: use the established forward BE-delta / inferred-move machinery, render forward tool calls in their real states, and verify each converted step with execute_forward_call. Reverse IMPORT-only changes are not forward electron steps; remove/project disconnected auxiliary context only when the operation is explicitly justified and recorded. If a witness cannot be replayed exactly, label bridge_failed/witness_unavailable and retain the original candidate for autonomous forward checking; never synthesize a fake successful witness.

**Forward feedback has two distinct kinds**:
- **cheap continuous score**: For an actually replayed forward witness, frozen F0 computes length-normalized teacher-forced assistant-token log likelihood of forward FLOW/FINISH tool calls, conditional on the authentic forward state/history. Cache by (F0_SHA, complete prompt/token sequence, bridge policy). This is a model path-support score, **not** a measured forward product probability or an autonomous round-trip success.
- **true round trip**: From the constructed forward reactant mixture, F0 independently generates and executes its own forward electron FLOW/FINISH until explicit finish or declared budget. Score whether the executor-derived product matches the original target under a fixed mapless/structural, component-aware comparator. The target is only on the scorer side. A successful trivial target-retaining mixture must not count as a synthesized transformation. Record forward failures and insufficient composition as separate outcomes.

Run the cheap path scorer for all replayable unique candidates. Run the true forward agent for a fixed, documented subset including high/low score candidates, at least 32--64 attempted candidates when available; also evaluate frozen validation outcomes. Quantify whether path likelihood orders true forward reconstruction before using its preferences. If the proxy fails this audit, use actual forward round-trip and executor outcomes for DPO; log this as a measured finding rather than discarding the run.

**Preference construction**:
- Choose and reject actions sampled at the **same exact model-visible reverse prompt**: identical original product, private executable state view, atom/bond inventory, accepted action history, tools and Qwen prefix. Two trajectory branches at their first genuine common-state divergence are valid; two unrelated states are not.
- Positive preference evidence: autonomous forward reconstruction of a nontrivial inverse candidate, or a replay-verified high-support forward witness after its score has demonstrated discriminative value. Executor-invalid or nonproductive transitions are strong negative controls.
- Store continuous scores and empirical margins, but never label a sample negative merely because it does not match FlowER's one recorded precursor. A completed valid alternative with unknown forward result remains separately accounted for. Avoid selecting duplicates of the same chemical successor as independent new discoveries.
- Store both complete original sampled tool calls (all generated tokens, including EOS), not hand-written abbreviated actions. GT reference paths may be used for independent diagnostics and the pre-existing SFT parent only, not silently inserted into model-generated pair collection.
- Credit is attached to the first distinct chemical action at a common state. Preserve the entire suffix outcome as metadata, but train the common-prompt chosen/rejected **tool-call tokens**, not the whole post-divergence transcript with unrelated prompts. Group IMPORT/FLOW/FINISH statistics separately.

**Policy objective**: Standard state-conditional DPO with frozen R0 reference:
~~~text
L_DPO = -log sigmoid(beta * (
    [log R_theta(a+|o) - log R_theta(a-|o)]
  - [log R0(a+|o) - log R0(a-|o)]))
~~~
Default beta=0.1, LoRA-only learning rate=2e-6, gradient accumulation=4, microbatch=1 (revisit only after profiling); up to 50 optimizer steps per cycle as smoke *upper bound*, with preference re-use clearly reported. No optimizer step is credited as RL if there are no authentic chosen/rejected pairs. Cache frozen R0 reference margins once per immutable preference dataset, versioned by prompt bytes and ref SHA. Never use mean-NLL as a replacement for full action log probability inside DPO. The reverse model checkpoint is immutable R0; R1 and R2 go to fresh directories.

**Forward update (bidirectional extension)**: After R2, use the same framework with roles swapped: start from frozen TRAIN reactant mixtures, F0 generates candidate forward tool trajectories; frozen R2 attempts reverse reconstruction of the structural reactant set and/or provides executable inverse witness support. Make same-forward-state F0 tool-call preferences and update a NEW F1 LoRA, comparing F0/F1 on independent source forward validation. Do not overwrite F0 or change the frozen F0 judge used in comparing R0/R1/R2. If adequate forward preferences cannot be built, report a reproducible blocked F1 stage and deliver the completed reverse RSI loop; do not claim full bidirectional parameter co-evolution in that case.

### D. Fixed smoke budget and sampling contract

| Item | Mandatory default |
|---|---|
| Corpus | FlowER strict-executable; TRAIN only for any collected pairs or updates; VALID for frozen evaluation; never TEST |
| Train targets | 128 distinct stable IDs selected reproducibly; same IDs R0 and R1 collection rounds; allow expanding to 256 only if needed for informative comparisons |
| Validation | 32 disjoint stable FlowER valid IDs, stratify by electron-event depth and mandatory IMPORT count; freeze before all updates |
| Reverse samples | Adaptive K=2 -> 4 -> 8 max per target, temp=1, top_p=1, no top-k/truncation sampling bias |
| Reverse tool/length budget | up to 24 tool attempts/candidate; 384 model-generated tokens/attempt; every rejected proposal counted |
| True forward samples | K=1 for 32--64 selected unique train candidates when available and all feasible frozen validation comparator candidates; record all attempted failures |
| Forward tool/length budget | up to 24 tool attempts/candidate; 384 generated tokens/attempt; explicit finish required |
| DPO updates | max 50 steps per reverse iteration; at most 2 iterations; no full Stage-I/II retrain; F1 one bounded iteration only if viable |
| Hardware | Prefer existing allocated 1--2 A100/H20 GPUs and the checked runtime; submit fresh job only on authorized resources using established Taiji launcher conventions |
| Seeds | 17 for selection/sampling; derive per-(reaction,state,round,candidate) seeds so asynchronous execution order does not change chemistry distributions |
| Checkpoints | R0/F0 frozen; new immutable R1, R2, optionally F1; save actual SHA before/after every phase |

For the first sample, K=2 does not mean beam search or two tokens; it is two independent complete agent trajectories. Match before/after evaluation by the original reaction IDs, sampling budget and policy observation contract. Keep genuine skipped/failed runs in the denominator. The 24-decision smoke limit is an explicitly new smoke budget, **not** an update to the deployed 40-decision/32-import frozen benchmark.

### E. Implementation tasks; complete them in order

1. **Runtime and provenance:** Add scripts/run_mechet_rsi_smoke.py (or equivalent orchestration), frozen settings, output manifest, exact Stage-II/forward adapter checks; correct only source-specific metadata after read-only audit. Do not change already trained weights.
2. **Forward inference:** Selectively port src/mechet/forward_qwen_events.py and its necessary dependencies from the specified branch. Add scripts/run_forward_qwen_rollout.py using the genuine F0 tool template, proper SFT-aligned prefix, termination, batched vLLM sampling and authoritative forward executor.
3. **Bridge + reward:** Add src/mechet/roundtrip_bridge.py and src/mechet/roundtrip_reward.py, with explicit verified witness coverage, full-versus-structural mixture handling, target-matching and failure classifications.
4. **Cheap forward scoring:** Add scripts/score_forward_witnesses.py. Reuse selected-token scoring with explicit token masks, batch immutable prompts, cache by F0 SHA. Test against full forward K=1 on held-out samples.
5. **Collect preference data:** Add scripts/collect_roundtrip_preferences.py with verified common-prompt pair formation, positive/negative/unknown accounting, candidate provenance, full raw logs, independent R0/R1 data refresh and chemistry successor dedup.
6. **Short DPO:** Add scripts/train_roundtrip_dpo.py based on frozen reference-margin computation in scripts/train_proof_dpo.py. Train chosen/rejected tool-call assistant tokens only; reference LoRA remains frozen; save optimizer stats, grad norm, per-tool learning diagnostics, actual weight delta and checkpoint SHA. Separate source-rank selection and reference-probability caching.
7. **Rolling generation:** Selectively port or adapt scripts/rolling_vllm_search.py from ops/pr83-route-greedy-20261011. One GPU-owner drives continuous vLLM add_request/step; CPU RDKit and durable recording run independently, and completed branches refill GPU requests without a global future.result() wave barrier. Guard per-request ordering, cancellation, seed, memory and every full EOS/tool callback.
8. **Real smoke:** Complete R0/F0 baseline, R0->R1, **fresh** R1->R2, paired validation R0/R1/R2 under frozen F0, optional F1 update, and report including failures. Do not declare the goal completed after static tests or only a collector dry run.

All new runtime behavior must remain compatible with current executor and frozen Qwen prompt protocols. Implement minimal targeted test and code changes. Avoid broad refactors or touching full-tree MCTS search logic until the rolling engine has a proved A/B speed result.

### F. Efficient inference and optimizer requirements

- Avoid putting full route MCTS (existing ~8.53e8 generated R1 tokens in the Oct 18 frozen work snapshot) into the training feedback loop; train on bounded complete single-reaction trajectories.
- Apply chemical state/action dedup **before** expensive forward scoring or repeat rollouts; retain distinct private-state/history contexts when they alter future legal choices.
- GPU continuous-batching queue should overlap generation with RDKit and result serialization. Measure before/after wall time, GPU generation throughput, CPU executor time, stalled batch time, unique valid successors, and duplicates. Do not claim an unmeasured speedup.
- Reuse a pinned Qwen3-0.6B base and swap LoRA adapters when practical; 1 GPU sequential or 2 GPUs R/F producer--consumer are both acceptable. Avoid repeatedly reloading base weights, and preserve each adapter SHA.
- Frozen F0 witness scoring should be batched in teacher-forcing mode rather than forcing all candidates through a full forward rollout. Use source-conditioned path lengths and true-forward calibration, not unvalidated absolute likelihood thresholds.
- During DPO use cached reference margins and assistant-token-selected logits, BF16/LoRA, grad accumulation and prompt prefix caching when safe; measure max memory and tokens/sec. The existing selected_policy_logps currently requires microbatch=1 and **reduces logits memory, not transformer attention FLOPs**.

### G. Required evaluations and honest evidence

Freeze validation IDs, base models, R0/F0 SHA, decoder, action budget and RDKit runtime before updates. Score four models where available: (R0,F0), (R1,F0), (R2,F0), optionally (R2,F1), plus direct F0/F1 forward validation. R0/R1/R2 must have the same F0 evaluator for attribution. A second-round **static D0 reuse** control should receive the same number of further optimizer steps as dynamic D1 if viable, to isolate RSI data refresh from additional fitting.

Mandatory metrics:
- checkpoint authentication, assistant prefix / EOS / tool token alignment and true parameter changes;
- raw/unique trajectories and completed nontrivial reactions per target; reverse IMPORT/FLOW/FINISH and forward FLOW/FINISH failure breakdown;
- forward witness conversion coverage; valid witness path score distribution; autonomous Forward@1, genuine RoundTrip@1 and @K with exact denominator;
- **informative preference groups**, exact common-prompt action pairs, high/low score separation, proxy-to-autonomous success association; unknown and invalid candidates reported separately;
- DPO total optimizer steps vs unique pairs, loss, correct reference/policy margins, gradient norms, candidate preference accuracy, and per-tool update participation;
- **R0/R1/R2 independent product-start forward-verified discovery at matched generated-token budget**; frozen-validation reference structural precursor agreement as diagnostic, distinct structural disconnections, executable/terminal rates;
- GPU-generated tokens, generated tokens per distinct forward-verified reaction, real runtime, CPU execution share, cache/dedup hits, peak VRAM; cost for all attempted candidates, not success-only.

Output in outputs/agent/mechet_rsi_roundtrip_smoke_20261021/ (new directory; do not overwrite prior experiments):
- provenance.json, smoke_config_resolved.yaml, selected_train_ids.json, selected_valid_ids.json;
- r0_rollouts.jsonl, r1_rollouts.jsonl, forward_rollouts.jsonl, forward_witness_scores.jsonl, bridge_audit.jsonl;
- d0_preferences.jsonl, d1_preferences.jsonl, r1_training_steps.jsonl, r2_training_steps.jsonl;
- models/R1, models/R2 and optional models/F1 with SHA + source lineage;
- evaluation/paired_validation.json, evaluation/speed_comparison.json and final README.md including failures and the exact commands used.

Files can be sharded/streamed at runtime, but final manifest must bind every shard count and SHA. Expose immutable checkpoint and full path; never report a planned number as measured.

### H. Tests / acceptance gates

- State bridge test: meaningful FLOW reversal, IMPORT/projection cases, copied auxiliary, explicit H, repeated identical molecules, atom-map re-assignment, stereo and terminal status. No model target leakage into Forward prompt.
- Prompt parity test: R0 Stage-II exact unified current-state/compact-history prefix; F0 genuine independent SFT system/schema; real token EOS boundary; sampled selected-token logps parity with teacher-forced scoring.
- Preference test: same-prompt chosen/rejected exactness; no unrelated-state pairs; duplicate successors do not multiply independent chemical support; no path-likelihood-only label presented as autonomous success; missing forward completion is not rebranded success.
- Loss test: known preferred token raises DPO log-prob margin; frozen R0 stays unchanged; no update when no meaningful pairs; positive/negative/tool gradients logged; independent R1/R2 actual SHA differs when an update occurred.
- RSI test: D1 records actor_sha=R1, not R0, contains freshly generated raw actions; R2 starts from R1; static-D0 continuation is separately named.
- Runtime test: continuous GPU owner / CPU executor safe shutdown, prompt request IDs, deterministic seeds, missed-result recovery, bounded in-flight queue; a blocked CPU event does not stop unrelated ready GPU requests.
- Evaluation test: independent train/validation ID separation; no TEST in optimization; all skipped, rejected, censored and failed records retained; matched decoder and budgets.

**Pass criteria for a valid engineering smoke:** actual R0+F0 model inference, executor replay, forward scoring and at least one real verified DPO update using true same-state pairs, a fresh R1-generated D1 dataset, a second R2 update when informative pairs exist, saved usable checkpoints, paired frozen-validation tables and measured time/tokens. Never label static/no-update runs as successful RSI. If positive pairs are scarce, extend sampling to the declared max and produce a complete root-cause report (with artifacts) instead of fabricating an improvement. Scientific effect is judged by validation: training loss alone is insufficient.

### I. Execution and reporting discipline

- Codex is authorized by the user's follow-up to implement **on this PR branch** and run the bounded smoke on the user's established environment, respecting live job resources. This goal document itself does not run jobs.
- Continue through data, build, dependency, runtime, sampling and training mistakes until real results or a reproducible hard infrastructure/data blocker are reached. Fix root causes and re-run affected phases rather than adding placeholders.
- Keep updates and all real metrics in this PR, preferably append a dated results note under docs/results/mechet_rsi_smoke_20261021.md with model SHA, exact train IDs, held-out IDs, real work and failure classification. No change to paper files or claims.
- Never silently change evaluation protocol between R0/R1/R2, cherry-pick Graph code, count synthetic forward witnesses as real model success, insert reference precursor paths as autonomous samples, or stop unrelated tasks.
- When complete, publish an explicit decision: (a) round-trip preference helps the frozen Forward-validated reverse policy; (b) only one-step preference learns but recursive refresh fails; (c) forward scoring is too weak / bridge too sparse; or (d) infrastructure/data blocker. Report negative results faithfully, with specific actionable bottlenecks.

**Completion question:** On the same FlowER product-start validation cases and comparable generated-token budget, does an inverse electron-flow policy trained with its own forward-checked, refreshed preferences discover more distinct, executable and forward-reconstructable retrosynthetic reactions than its frozen Stage-II parent? If yes, does a second genuinely fresh preference round help beyond simply reusing the old pairs?
