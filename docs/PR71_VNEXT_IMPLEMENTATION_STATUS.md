# PR #71 vNext implementation and gates

This branch extends the Stage-II protocol-v2 implementation in PR #70. It is
an **opt-in research track**; it does not replace the frozen paper protocol,
dataset denominators, or chemistry executor. The 31k trace-view denominator is
10,152/1,319/1,253 reactions, not its full endpoint denominator of
24,959/3,120/3,120 reactions. Unqualified FlowER full remains
257,171/2,890/28,971 reactions.

| Roadmap component | Implementation | Promotion gate |
| --- | --- | --- |
| Complexity-stratified audit | `scripts/audit_vnext_complexity.py` | Source denominator, coverage, changed bonds, reacting atoms, ring, flows, imports, length |
| vLLM eager/graph and structured action benchmark | `scripts/benchmark_vnext_structured_vllm.py`; per-state schema in `src/mechet/vnext_structured_actions.py`; matched-state SGLang request harness in `scripts/benchmark_vnext_sglang.py` | Run same oracle states; compare parse/handle validity, output equivalence, latency and chemistry |
| Source/sink pointer and optional joint LM objective | `scripts/train_electron_pointer.py` (`--joint-policy`); runtime scoring in `src/mechet/electron_pointer_runtime.py` | Pair Recall@K, all-pair Recall@K and flow strata; then matched product-start strict event/successor and endpoint |
| Sibling chemical-state credit | `src/mechet/vnext_tree_credit.py` and opt-in `--vnext-credit` in `scripts/natural_language_anchor_branch_stage.py` | Same-state candidate budget, unique successor pooling, all-negative skip, matched GRPO/GSPO/tree |
| Endpoint-reachability critic data | `scripts/build_vnext_reachability_value.py`; `reachability_pn` search mode | Train-only on-policy labels; off-reference is not automatically negative; held-out ranking and calibration |
| Search-to-policy teacher | `scripts/build_vnext_search_distill.py` | Verified exact full endpoint, train-only source, Stage-II history parity, greedy distilled policy |
| Assistant-only packing microbenchmark | `src/mechet/segment_packing.py`, `scripts/benchmark_vnext_packing.py` | Block-diagonal mask/position reset parity, then matched GPU backward throughput and peak memory; paper trainer unchanged |

The completed pointer pilot used a **frozen** Stage-II Qwen adapter and an
independently trained pointer head. The optional joint LM-plus-pointer mode is
implemented but has not been trained or promoted. Source/sink marginal Recall alone is not a
strict-event result; use the paired and executed-successor gates above.

The vLLM/SGLang benchmarks use oracle-provided current states and are not a
product-start endpoint evaluation. The structured JSON envelope is an
experimental output mode; it still requires GPU/runtime verification with
vLLM 0.8.5 XGrammar before any speed or validity claim. GPU speedups are not
reported until measured. The packing code has a small-Qwen3 exact-logit test,
but its 8B training speed/memory benchmark has not run; the current paper
trainer still rejects `packing=true`. The search runtime's vNext matched mode requires
the frozen privately mapped compiler source for deterministic stereo replay;
the model prompt remains unmapped/product-only.

P2 GNN, graph-token adapter, speculative draft model and asynchronous RL are
explicitly gated by the PR #71 roadmap; they are not part of this branch and
must not be reported as completed or necessary for current experiments.

The first pointer artifact is under
`outputs/agent/pr71_p0_pointer_31k_stage2_8a100_20261001/`. Its original
validation JSON contains only marginal localization metrics; the extended
evaluation under `outputs/agent/pr71_pointer_pair_valid_8a100_20261001/`
computes paired and multi-flow metrics without retraining: 2,543 valid events,
pair Recall@1/4/8 = 56.5%/95.4%/98.1%, all-gold-pair Recall@8 = 86.7%.
This validation trace view contains 328 one-flow and 2,215 two-flow events,
and **zero** 3+-flow events; it cannot establish a complex 3+-flow claim.
An opt-in `CoupledPointerHead` directly models source-conditioned sink scores
and has been evaluated against the independent-head pilot. Its 8×A100 task
`meteor_mechet_pr71_conditional_pointer_8a100_qy_20261001_01` completed
successfully on the same 2,543 events: pair Recall@1/4/8 =
91.8%/98.9%/99.8%, all-gold-pair Recall@8 = 92.4%. The independent-head
values were 56.5%/95.4%/98.1% and 86.7%. These are teacher-state
localization metrics only; neither model has yet established a product-only
strict-event, successor or terminal-endpoint gain.
