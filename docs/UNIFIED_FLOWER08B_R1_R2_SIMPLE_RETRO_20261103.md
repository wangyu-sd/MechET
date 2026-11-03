# MechET unified reverse-electron tree: FlowER 0.8B R1/R2 + simple retro

## What is the task?

- **Model:** Qwen/Qwen3.5-0.8B, *FlowER* Stage-II compressed-history actor, frozen.
- **Source code:** PR #97 merged on \`main\` (merge SHA \`8b29c0718ff9b0648474e9d4729d15f1e8d3d6f0\`).
- **Weights:** \`/aaa/fionafyang/buddy1/whaleywang/MechET/outputs/agent/pr96_flower_qwen35_08b_mainonly_20261023_06/models/history/checkpoint-89442\` (SHA-256 \`2c98a9bff3afeaffbf2ee2733885cccc68e936f9310fd9f46ba4ee2211e5f453\`).
- **NO** mech31k actor, older 0.6B actor, PR97 FlowER EARHO-unverified weights, model retraining or hidden reference precursors.
- **Control unit:** executor-accepted coupled electron-flow event; \`IMPORT\` is explicit, budgeted open-system participant preparation. \`FINISH\` is a *boundary marker*, not a PUCT child or retrosynthetic decision. It still needs the frozen actor to produce a valid tool-call and consumes tokens and a model call.
- **State:** all currently registered non-stock precursor obligations plus each molecule's live executor-owned electronic states and policy histories. \`reverse_route_network\` is a bookkeeping AND/OR hypergraph, **not** an independent reaction-proposal planner. \`RouteNetwork.next_context\` is never used by the event-PUCT controller.
- **Search:** use a single logical PUCT-style frontier over electronic states from all obligations, progressively widening *distinct executed chemical successors*; route certificate cost and success require complete AND-stock closure. Positive route credit goes to the actual precursor-producing electronic histories, not to arbitrary failed or unvisited branches.

This initial implementation is **PUCT-style frontier search with sparse complete-route backup**; it has **no separately trained critic, no direct endpoint decoder and no RSI**. It is not proof of an advantage over full-reaction MCTS. Branching and route-credit efficiency require matched GPU evaluations.

## Prepare source-locked R1/R2 assets (once, on shared storage)

Run from the repository root:

\`\`\`bash
export ROOT=/aaa/fionafyang/buddy1/whaleywang/MechET
export DATA=/aaa/fionafyang/buddy1/whaleywang/datasets
export OUT=$DATA/mechet_unified_flower08b_20261103
cd "$ROOT"
git fetch origin feat/flower08b-unified-electron-tree-20261103
git switch feat/flower08b-unified-electron-tree-20261103
python -m pip install -e . pytest psutil
python -m pytest -q tests/test_unified_electron_search.py tests/test_unified_electron_protocol.py
python scripts/prepare_frozen_route_benchmarks.py \
  --config configs/eval/frozen_route_benchmarks_20261013.json \
  --output "$OUT/assets" --workers 8
\`\`\`

The prep script verifies hashes of the original source files and freezes 460 targets: URSA Expert 100, URSA Drugs/Clinicals 100, RetroCast mkt-cnv160 and PaRoutes-derived n5 100. **Preflight:** the original files referenced by \`configs/eval/frozen_route_benchmarks_20261013.json\` must exist under its \`raw_root\`; the raw files are not in Git. To use a different root, copy the JSON config, edit \`raw_root\` only, and use that copy while retaining the source SHA checks. Do not silently replace stocks or evaluate another 160/100 target list.

## Route search: 16-target functional evaluation, then 460 frozen targets

\`\`\`bash
for METHOD in greedy heuristic event_puct; do
  python scripts/eval_unified_electron_search.py --suite routes \
    --assets "$OUT/assets" --output "$OUT/eval16" \
    --method "$METHOD" --pilot16 --device cuda:0 \
    --max-calls 128 --max-tokens 12000 --max-new-tokens 512 --seed 17
done
\`\`\`

The 16-target suite checks executability, bookkeeping, memory and costs; **not** a publishable performance statement. All methods use the same **FlowER** 0.8B actor, target IDs, frozen stocks, and limits. The legacy \`heuristic\` mode is the earlier electron-tree priority; \`greedy\` is a deterministic *single accepted-path* baseline, NOT independent-K sampling. \`event_puct\` is the new unified electronic frontier, with FINISH as a marker. Full formal R1 runs omit \`--pilot16\`, using the exact same full 460 IDs with a fixed computation budget.

\`\`\`bash
python scripts/eval_unified_electron_search.py --suite routes \
  --assets "$OUT/assets" --output "$OUT/eval460" \
  --method event_puct --device cuda:0 \
  --max-calls 128 --max-tokens 12000 --max-new-tokens 512 --seed 17
\`\`\`

Each target writes \`report.json\`, raw \`actor_calls.jsonl\`, \`calls.jsonl\`, and durable \`journal\` records. On interruption, pass \`--resume\`. If a journal was committed before the independent MCTS snapshot was atomically updated, resume **fails closed** with \`MCTS_STATS_JOURNAL_MISMATCH_RECONCILIATION_REQUIRED\`; do not discard provenance or resume with a different actor/stock/budget.

### R1

Compute nonzero AND-stock \`Solved@Budget\`, first discovery cost, distinct routes, depth, generated tokens, executor calls, unique successor states, and actual elapsed time. Stock closure is **not** independently verified synthetic feasibility. The old 171/460 historical report coverage and 170 solved archived targets are **not** used as a new comparable score.

### R2

R2 is analyzed on the **same R1 target outputs**. \`r2_same_r1_outputs\` contains RetroCast 160 reference depths, solved-by-depth, and composition-status counts. The old electronic \`C0/C1/C2\` labels may be reported **only for independently authenticated electronic programs**. RetroCast reference mapped reactions are *not* automatically an electronic-mechanism label; \`Unknown\` is preserved otherwise. Do not rerun only the successful targets or invent source-verified chemistry corroboration.

## Simple retro: one-step precursor recovery with exactly this electronic search

Prepare a source-frozen JSONL file with unique \`id\` and \`target_smiles\` (or \`product_unmapped\`), optionally \`gold_structural_precursors\`. Example:

\`\`\`json
{"id":"valid-example-01","target_smiles":"CCO","gold_structural_precursors":"CCBr.O"}
\`\`\`

The JSON line above is **schema illustration only**, not a claimed experimental reaction. For formal FlowER evaluation use its approved, frozen main-product-only VALID/TEST source, retain all IDs and their original coverage/quarantine status, and do not confuse the strict-executable test denominator 28,967 with the unfiltered 28,971. The reference precursor is loaded for scoring *after* the model has produced candidates, not for prompts, continuation decisions or candidate selection.

\`\`\`bash
python scripts/eval_unified_electron_search.py --suite simple_retro \
  --one-step-jsonl /absolute/source_verified_flower_main_product_test.jsonl \
  --split TEST --output "$OUT/simple_retro" --device cuda:0 \
  --max-calls 128 --max-tokens 12000 --max-new-tokens 512 --seed 17
\`\`\`

Reports contain candidate structural precursor sets and Top-1/5/10 structural Hits. **Ranking is normalized action log-probability**, not PR96's independently defined NLL-selected ranking. Do not label the result official PR96 Top-k or claim superiority based on incomparable ranking protocols. A complete reference source and explicit missing-label coverage are required for publication.

## Evidence and next experimental gates

- Code + CPU checks are different from **actual frozen-Qwen GPU inference**; no remote GPU or complete route evaluation was performed by this PR authoring session.
- For publication, add matched independent-K electron trajectories and reaction-level MCTS/Retro* using the same actor, unique-precursor budget, inventory and generated-token cost; record a full decision trace showing route feedback changes a previous electronic branch.
- In R1/R2/one-step comparisons, do not leak reference routes or annotated electronic events into the actor or PUCT score. Report unfinished/censored cases separately from proven failures.
- A valid pathway under valence/electron bookkeeping and stock closure **does not establish reaction yield, selectivity, catalyst or lab feasibility**.
