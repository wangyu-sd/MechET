# Frozen Qwen R1/R2 full-route execution plan

> **For agentic workers:** Use superpowers:executing-plans inline. The user
> has directed execution of the cited paper protocol; this records its
> implementation, not a new model or self-improvement design.

**Goal:** Run the frozen FlowER Stage-II Qwen3-0.6B on all460 declared route
targets with greedy and adaptive reverse electron search; analyze the same
160 RetroCast outputs for R2, keeping all original denominators.

**Architecture:** Reuse the already submitted vLLM electronic controller.
Each terminal reaction contributes an AND hyperedge to its structural
precursors; nonstock obligations recursively use the same frozen actor.
Stock closure is a least fixed point, not a FINISH token or one solved branch.

**Tech Stack:** Python3.11, RDKit2026.03.4, vLLM0.8.5/torch2.6 CUDA11.8.

**Spec:** User-selected paper file
`editorial/R1_R2_COMPLETE_ROUTE_SCENARIOS_AND_COMPOSITION_PROTOCOL_20261013.md`
at MechET-paper revision `ee96f346ccfc2c40038dd1c4f9b6e7f955915659`, plus
`docs/DYNAMIC_ELECTRON_TREE_PROTOCOL.md` and the user's no-chemical-cap ruling.

## Global constraints

- Frozen authenticated Stage-II SHA40204c7a...7e9de75; no RSI/RL or training.
- URSA Expert100, Drugs/Clinicals100, publication mkt-cnv160 and random-n5-100.
- Separate exact stock per cohort; original IDs and all failures retained.
- No GT in policy/search, no fixed depth/beam/import-count maximum.
- Native executor states/history contexts remain independent; graph chemistry
  merges only for display/outer equivalent precursor obligations.
- All raw prompts/completions/probabilities and all accepted graphs saved.
- Tensor/memory guards only; task work windows renew dynamically rather than
  impose token/time hard stops (user October14 ruling); no unrequested smoke/pilot.
- Only canonicalization preserves isotope/stereo/charge; no neutralization,
  tautomer fallback or automatic split of stock salt entries.
- Only A/B frozen policy conditions; C/D self-improvement deferred explicitly.
- Source bond mapping is not an electron program; missing annotations Unknown.
- meteor task prefix, proven Tencent/Ceph donor, default live POD stdout.

## Review focus

1. One stock branch plus one unresolved branch must not solve an AND route.
2. Chemical cycles cannot bootstrap themselves into a solved route.
3. Zero-step in-stock targets stay separate from nontrivial model solutions.
4. Disconnected imported environment must not become a structural precursor.
5. Missing map/program/route data must not become zero complexity or C2.

### Task 1: freeze source assets and reference complexity

**Files:** `src/mechet/frozen_route_assets.py`,
`scripts/prepare_frozen_route_benchmarks.py`,
`configs/eval/frozen_route_benchmarks_20261013.json`,
`tests/test_frozen_route_assets.py`.
**Interfaces:** stock normalization text files and cohort JSONL rows with
`id,source_id,cohort,product_unmapped,reference_routes`; manifest records
actual source/stock/derived SHA256 and all460 immutable target IDs.

- [ ] Write RED tests: reference depth/AND branching, strict mapless stock
  normalization, missing-program Unknown and corrupt source rejection.
- [ ] Implement `extract_reference_route(route, target_id, route_index)` and
  `freeze_assets(config_path, output, workers)`; stock work uses a CPU pool.
- [ ] Run focused tests, expect all pass; commit ONLY task files.
- [ ] Verify every source SHA and the expected100/100/160/100 populations,
  source reference-depth strata, target overlap and source stock closure.

### Task 2: recursive electron forest with AND closure

**Files:** `src/mechet/reverse_route_network.py`,
`tests/test_reverse_route_network.py`.
**Interfaces:** `RouteNetwork(case_id, product, stock, mode)` supplies
`contexts,next_context(),incorporate(),work,to_dict(),report()` to the
batched runner. Each obligation owns an ElectronTree and its native history.

- [ ] Write RED tests for all five review input classes, native terminal
  structural extraction, greedy recursive continuation and no depth cap.
- [ ] Implement least-fixed-point stock closure, cyclic-route protection,
  equivalent-obligation merge and gold-independent likelihood ordering.
- [ ] Save every terminal program/auxiliary state; nonproductive/cyclic
  reactions are visible but cannot count as a nontrivial stock solution.
- [ ] Run focused tests, expect all pass; commit task files.

### Task 3: full-cohort batched inference and observed R2 analysis

**Files:** modify `scripts/infer_dynamic_electron_trees.py`; create
`scripts/analyze_frozen_route_search.py`,
`scripts/run_taiji_frozen_route_search_a100.sh`, one meteor Taiji config;
test `tests/test_frozen_route_analysis.py`.
**Interfaces:** same raw journals plus complete per-target route graphs;
immutable R1 outputs feed R2 complexity/strata/paired bootstrap2000 and figures.

- [ ] Write RED tests for full-ID aggregation, paired CIs, zero-step treatment
  and reference-route recovery without label-dependent branch ranking.
- [ ] Reuse the proven cross-product vLLM scheduler for all460 targets.
- [ ] Separate actual compute use from nominal budget; never assert an
  equal-work gain for unequal-work greedy and tree outputs.
- [ ] Keep unresolved mechanism labels Unknown until exact frozen corpus
  match; H2 is historical and NOT rerun with this full-data actor.
- [ ] Run entire pytest suite and shell syntax checks; expect zero failures.
- [ ] Review whole change; repair Important findings with RED-to-GREEN tests.
- [ ] Commit immutable runtime; submit one8A100 task with proven private
  Ceph init and verify actual queued/running state, never infer it.
- [ ] Once model outputs exist, verify all460 IDs, produce R2 artifacts and
  actual PDF panels/provenance; no guessed numerical results.
