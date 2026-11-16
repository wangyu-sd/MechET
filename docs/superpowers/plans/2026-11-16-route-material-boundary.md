# Route material boundary repair

> **For agentic workers:** Use superpowers:executing-plans for this authorized repair.

**Goal:** Stop asking the mechanism policy to synthesize an active thiolate when its explicit protonated material is already in stock, without changing executor chemistry.

**Architecture:** Add an opt-in, named material contract beside the unchanged target-heavy-atom endpoint contract. Resolve only a single terminal, nonradical O-/S- attached by a single bond to carbon; add one ordinary proton only when the exact resulting material is in the fixed stock. Preserve the full mapped executor state and independently reconstruct this projection in reference scoring and route certificates.

**Tech stack:** Python, RDKit 2026.03.4, pytest; saved TRAIN action replay, no new model generation.

**Spec:** This document, including the scope and invariants below.

## Constraints and review focus

- Exact matching is the default; old snapshots and the live run must remain unchanged.
- No hidden reference informs collection, endpoint mapping, or stock stopping.
- No changes to heavy-atom connectivity, bond orders, stereochemistry, isotopes, radicals, permanent ions or multicharged species.
- Preserve multiplicity, whole-mixture stock priority, explicit executor hydrogens, and ancestry cycle checks after projection.
- Recompute material evidence from full mapped final/initial states; reject forged evidence and mismatched scoring contracts.
- This is a declared protonation/workup availability convention, not proof of a missing mechanistic step or general chemical validity.

## Task 1: Material projection and all route consumers

Files: new `src/mechet/route_materials.py`, `tests/test_route_materials.py`; modify `route_policy_episode.py`, `external_route_rl.py`, `prepare_external_route_rl.py`, `train_external_route_rl.py`, `eval_pr98_route_policy.py`.

- [x] Add regression tests for actual thiolate endpoint, exact-stock priority, negative controls, no-op/cycle guards, occurrence multiplicity, independent reward/certificate checks, restore integrity and unchanged legacy behavior.
- [x] Run tests RED before implementation.
- [x] Implement `material_contract(cfg)`, `project_material_endpoints(public_structural, stock, contract)` and `validate_material_edge(edge, stock, contract)` in the new module; wire identical semantics through commit, scoring, contract validation, full VALID and certificate replay.
- [x] Run targeted tests GREEN, then the project pytest command; report unrelated collection/environment failures explicitly.
- [x] Replay saved real TRAIN actions through the new adapter, showing original mapped state unchanged and corrected material/reference handoff. Report deterministic software verification separately from model performance.
- [x] Obtain independent code review, address material findings, commit only owned files, update PR 107 with implementation and evidence.

## Execution ledger

Pre-flight: commit projection produces the exact children/evidence consumed by hidden reward and certificate validation. All three use one pure resolver; reward additionally reconstructs the mechanism endpoint from mapped states. Existing code is in linked worktree `MechET-earho-publication-20261026`; unrelated dirty MCTS files are excluded.

Final verification: 125 affected CPU tests PASS; 48 pinned Python3.11 tests PASS; four saved TRAIN boundary witnesses replayed PASS. Two independent review findings were reproduced RED (three tests) and fixed: per-map element/isotope identity, and full opt-in restore replay. Broad project run completed with 1615 PASS, 22 failures and 3 errors; three compatibility failures fixed, one incomplete mock updated, all four now included in the final green affected suite. Remaining 18 failures/3 errors are listed by full test ID in verification.json; no all-project-green claim. Integration patch dry-run PASS, six source copies byte-identical. Only this repair is staged; unrelated dirty MCTS work stays untouched.
