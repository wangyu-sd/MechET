# LGRV-v1 literature evidence implementation plan

> **For agentic workers:** use superpowers:executing-plans; execute this approved PR89 design inline. Do not launch training or model inference.

**Goal:** reproducible literature-source verification of frozen model-generated complete reactions, all-460 route outcomes, all-3120 single-reaction outcomes, same-ID Greedy comparisons, and chemistry-led publication figures.

**Authority:** PR89 `17e2deb99919f5d9460c488a3d6e2ccb1ce015ee`; `docs/goals/MECHET_LITERATURE_GROUNDED_REACTION_ROUTE_PROTOCOL_20261018.md` supersedes earlier ranking requests. This file records implementation sequencing, not experimental results.

**Workspace:** existing linked `MechET-pr83-greedy`. Preserve all user changes and previous PR87/88 artifacts. Deliver commits descending from PR89 through a separate Git index; do not change the operational worktree HEAD/index or merge paper/main.

**Global constraints:** no GT retrieval queries; preserve stereo, isotope, charge and component multiplicity; structural/auxiliary roles stay explicit; only actual performed primary-source examples with reconciled structures earn A/B; no match is C, not D. No per-FLOW experimental claims, silent neutralization, new GPU jobs, ranking main metrics, or new target filters. Freeze all reaction identities before retrieval. Original denominators and pending status stay visible.

## Task 1: reaction identity and immutable prediction extraction

Interfaces: raw saved reports, PR87 typed tables, PR88 snapshot/CSV and stock manifest -> canonical complete reactions, occurrences, routes and source hashes.

- [ ] RED: tests for component multiplicity/maps/stereo/charge, ambiguous mapping, structural versus auxiliary roles, route edge extraction, checksum mismatch and snapshot-only selection.
- [ ] GREEN: implement `literature_reaction_keys.py` and extraction adapter; no use of reference endpoint fields.
- [ ] Run full frozen input extraction (200 R1 reports, two complete 460 Greedies, 3120 MCTS and two 3120 Greedies), retain original populations.
- [ ] Verify focused then full repository tests; commit with source inventory and ledger evidence.

## Task 2: lawful structured reaction corpus and retrieval

Interfaces: official pinned ORD/Lowe source payloads -> reproducible exact/family index and corpus manifest -> bounded distinct-source hits per reaction.

- [ ] RED: tests for exact versus wrong-substrate retrieval, mapped centre/radius compatibility, source/patent-family de-duplication, missing roles and deterministic retrieval.
- [ ] GREEN: implement SQLite index, ORD/USPTO adapters and official-source download/manifest tool. Record version, SHA, license, source overlap uncertainty and conversion failures.
- [ ] Download lawful official public corpora, build full available index, freeze query/radius policy before prediction evidence retrieval.
- [ ] Run bulk retrieval over every frozen unique reaction, preserve no-hit cases and raw source locators; verify tests and commit.

## Task 3: primary evidence and symbolic mechanism assessment

Interfaces: indexed structural candidates + original document/locator/structures -> independent tier/retrieval/verification states; internal FLOW evidence stays separate.

- [ ] RED: tests for claims-only, uninspected exact hit, absent stereo, supported analogue, actual-source checksum/locator, direct contradiction and no-hit.
- [ ] GREEN: implement evidence records, primary-source cache and typed mechanism assessment.
- [ ] Retrieve and check original performed examples for accessible strongest candidates under identical cap=5; include source structures/conditions and explicit substrate differences. Preserve unresolved sources without claiming verified A.
- [ ] Verify tests and actual positive-case evidence, commit.

## Task 4: route, composition, paired baseline and efficiency metrics

Interfaces: all reaction occurrences/routes + evidence -> all-edge AND-route metrics, R2 program exposure joins, paired same-target differences and honest work units.

- [ ] RED: 2/3/5-edge bottlenecks, zero-step exclusion, one unpurchased leaf, pending versus failure, target-paired 2x2/2000-bootstrap, no occurrence inflation, unseen program scope tests.
- [ ] GREEN: implement literature route metrics and paired/disconnection comparison modules plus `verify_nmi_literature_reactions.py` CLI.
- [ ] Run complete frozen population; retain 460/3120/160 denominators, source coverage and actual token/executor work. Independent-K output absent -> prepare manifest/runner only; no GPU launch.
- [ ] Verify tests, hashes, source contracts and metrics, commit.

## Task 5: figures and reproducible delivery

Interfaces: versioned full analysis -> SVG/PDF/PNG chemistry figures, captions, case registry and manifests.

- [ ] Implement tested render pipeline; use true RDKit structures/actual FLOW arrows, source-checked reaction-level callouts, paired statistics and explicit route evidence bottlenecks.
- [ ] Export Fig2/3 and genuinely supported Fig4/appendix; preserve PR87's 23 figure sets. 183-mm editable SVG/PDF, <=170mm height, fonts embedded, PNG300dpi.
- [ ] Produce all protocol artifacts, documentation and full raw-result pointers/hashes; inspect rendered artifacts.
- [ ] Full suite and requirement-by-requirement completion audit; one fresh whole-branch review, fix important findings with regression tests.
- [ ] Push tested implementation/results to PR89 without merging paper/main; goal remains active until full requirements are verified.

## Review focus

Mapped atom identity versus graph isomorphism; multiplicity/stereo preserved; source roles not guessed from small size; route AND joins do not accept one successful leaf; machine matches never promoted to performed A/B; missing retrieval not negative chemistry; snapshot drift/checksum failure; corpus/patent-family duplication; unequal work descriptive only; unobserved targets retained; cached/shared model cost not added as fictitious GPU time; primary-source text not fabricated/reprinted wholesale; actual saved programs versus inferred arrows; novel programs distinct from novel reactions and strict H2.
