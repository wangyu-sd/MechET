# PR109 external route baselines: prepared adapters, not executed benchmarks

The external comparison remains **prepared, not executed**. The current implementation priority is Standard/ReFIT-PO and its mandatory final TRAIN-only distillation. This change adds structural output normalization and checks; it installs no planner/model dependencies, downloads no weights, runs no neural inference or search, and submits no jobs. RetroChimera remains an external comparator and must not enter Standard training, rewards or distillation.

## What is implemented

`src/mechet/route_baseline_export.py` supplies four entry points:

- `normalize_route_tree(raw)`: strict explicit AND-tree normalization for PaRoutes/Syntheseus/AiZynthFinder bipartite molecule/reaction trees, DMS molecule children, and RetroCast schema2 `route`/`target`/`product_of` trees. Multiple alternative reactions must be separate candidates. Malformed chemistry, empty explicit reactions, conflicting shapes and molecular ancestor cycles raise errors; nothing is pruned silently.
- `normalize_syntheseus_route(graph, route_nodes, target_smiles)`: calls the verified public `to_synthesis_graph(route_nodes)` bridge supported by both Syntheseus AndOrGraph and MolSetGraph. It expands the returned reaction graph's iterable reactant Bags into independent occurrences, rejects ambiguous/disconnected reactions and requires the declared target identity. It does not extract routes or run search.
- `route_stock_status(route, stock)`: validates normalized occurrence integrity and computes membership using declared route leaves. It distinguishes `target_in_stock`, `stock_closed`, `nonzero_step` and `solved`; a purchasable zero-reaction target is stock-closed but unsolved. Producer `in_stock` flags are ignored. The adapter does not trim extra chemistry at an internal stock molecule.
- `align_predictions(targets, predictions, stock)`: accepts an ordered frozen sequence of `{id, smiles}` targets and ID-keyed ordered candidate lists. Unknown output IDs, duplicate target IDs and candidate target mismatches are contract errors. Missing output and empty lists retain unsolved target rows; invalid individual routes retain their original candidate ranks as failures. Equal target identities under different source IDs remain distinct targets.

The normalized schema is `mechet_route_baseline_occurrences_v1`:

```json
{
  "schema_version": "mechet_route_baseline_occurrences_v1",
  "root": "m0",
  "target": "CC",
  "molecules": [
    {"id": "m0", "smiles": "CC"},
    {"id": "m1", "smiles": "C"},
    {"id": "m2", "smiles": "C"}
  ],
  "reactions": [{"id": "r0", "product": "m0", "reactants": ["m1", "m2"]}],
  "leaf_ids": ["m1", "m2"],
  "electronic_replay_verified": false
}
```

Repeated equal molecules have different occurrence IDs. Identical intermediates on different branches can retain different downstream reaction choices; they are not collapsed into one product-to-precursor map. Stock is membership-only, with no claim of finite commercial quantities. Atom maps and equivalent explicit hydrogens are normalized using the existing route identity convention; charge, isotopes, stereochemistry and disconnected material composition remain significant. There is no neutralization, salt stripping, tautomer aliasing, product correction or chemical-feasibility inference.

The adapter uses lazy RDKit chemistry only and can be loaded as a standalone Python module without Torch, Transformers, RetroChimera, Syntheseus or AiZynthFinder. Normal package import may still traverse the repository's existing `mechet.__init__`, which eagerly imports unrelated Torch helpers; this patch does not modify that shared initialization behavior. A child-process test loads the adapter by file and blocks all neural/planner imports while converting and scoring an explicit route.

## Verification and limits

TDD evidence in `/aaa/fionafyang/buddy1/whaleywang/MechET/outputs/diagnostics/pr109_baseline_readiness_20261116/`:

- Initial22 tests failed because the adapter was missing, then passed.
- Three additional normalized-table integrity tests failed before validation was added: missing leaf obligations, orphan molecules and reused occurrence IDs. These now reject forged closure claims.
- Standalone chemistry-only loading initially failed due to the package-relative identity import; the adapter now implements the same limited RDKit identity convention without importing the package and passes with neural imports blocked.
- Final focused suite: `tests/test_route_baseline_export.py` and adjacent `tests/test_external_route_data.py`, **38 passed**. These are software checks, not chemical benchmark results or a full-project green claim.
- Fake AndOr/MolSet graph fixtures cover selected-route bridging, repeated Bag occurrences, wrong targets and disconnected reaction nodes. A separate no-install check also constructed both **actual pinned upstream graph classes** from Syntheseus source commit `13fbaab2096a7cbd36ec7f97cba8d534020c4fc0`, bridged duplicate-reactant routes and retained four equal leaf occurrences/three reaction occurrences. Torch import was blocked. `actual_source_bridge_verification.json` records this structural API check; no search algorithm or model was run.

Reproduce the focused software checks from the private runtime worktree:

```bash
source /aaa/fionafyang/buddy1/whaleywang/MechET/configs/runtime/shared_host_storage.env
python -m pytest tests/test_route_baseline_export.py tests/test_external_route_data.py -q \
  --basetemp /aaa/fionafyang/buddy1/whaleywang/MechET/outputs/diagnostics/pr109_baseline_readiness_20261116/pytest_adapter_final
```

This is a structural adapter, not an electron-Executor certificate, a reaction feasibility model, a search implementation or a RetroCast tiered-score replacement. It does not open pickle files, extract native planner routes, unwrap AiZynthFinder DataFrame output, or materialize the RetroCast canonical scoring artifact. Those producer envelope integrations must retain raw files/metadata and frozen target sidecars separately. Its strict declared-leaf closure must be explicitly reconciled with any evaluator that permits stock-based internal-route truncation.

## Existing registry and execution drafts

The separate readiness audit is indexed at:

`/aaa/fionafyang/buddy1/whaleywang/MechET/outputs/diagnostics/pr109_baseline_readiness_20261116/`

- `REPORT.md`: detailed source/runtime/stock/exposure/budget audit.
- `baseline_registry.json`, `source_registry.json`: pinned official sources and model identifiers, missing weight hashes and readiness states.
- `make_draft_configs.py`, `draft_configs/`, `run_drafts.sh`:16 reviewable **draft** full-cohort configs for RetroChimera Pistachio+Syntheseus Retro*/MCTS and AiZynthFinder MCTS/Retro*. They are not frozen execution authorization and have never been run for inference.
- `hosted_metadata_registry.json`, `hosted_auxiliary_registry.json`, `published_runtime_metadata.json`: checksummed contextual source evidence. Raw producer `solved_count` is not the final source-stock solve metric; incomplete model/budget/hardware provenance prevents calling these matched new results.
- `verification.json`, `audit_manifest.json`, `adapter_final.log`, `actual_source_bridge_verification.json`: audit/static/software evidence.

Remaining execution blockers include authenticated weight/config hashes, separate compatible environments, exact full-cohort ID sidecars, source-stock SMILES/InChIKey/material parity, producer envelopes, current RetroCast artifact integration and frozen matched budgets. The repository pins Syntheseus<0.8 for its old planning extra whereas current RetroChimera needs>=0.8; never install into a live Student environment. HostPython3.13 is also outside AiZynthFinder4.4.1's supported range. Optional DMS remains separate. Original forward FlowER **+F** is a separate accepted-FINISH verifier condition with its own readiness checks; RetroChimera is not +F.

Reference full denominators remain600/400/84=1,084 with n5-stock/n5-stock/n1-n5-stock respectively, followed by PaRoutes n5 10,000. No adapter test is a small-target inference evaluation. Any future external GPU lane must preserve existing jobs and stay within the user's aggregate8H20 ceiling; the audit itself allocates zero GPUs.
