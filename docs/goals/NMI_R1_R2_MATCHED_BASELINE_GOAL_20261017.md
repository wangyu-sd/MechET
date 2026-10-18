# Codex Goal · MechET R1/R2: literature-grounded chemistry, matched baselines and comparative NMI figures

Updated 2026-10-18. **This document now follows the literature-first scientific objective**, not reference-endpoint ranking. Mandatory source-first execution protocol: [LGRV-v1: Literature-Grounded Reaction and Route Verification](MECHET_LITERATURE_GROUNDED_REACTION_ROUTE_PROTOCOL_20261018.md). Historical endpoint Top-K is optional legacy support and must not dominate R1/R2. This is an executable implementation goal, not a paper-story redesign.
Current data/code authority: [MechET PR #87](https://github.com/wangyu-sd/MechET/pull/87), commit `2af03e8cca4aca46e04d7619aeda4b987ee32c34`; paper authority [MechET-paper PR #29](https://github.com/wangyu-sd/MechET-paper/pull/29), commit `ffc2890e02fb265b6292cca6fe63646d0d612972`. This task is stacked directly **on PR #87**, which itself is stacked on PR #84. Do not overwrite either PR or merge to `main` on your own.

## Goal: turn existing chemical-state tree findings into comparative scientific conclusions

The user's priority is **mine and present advantages through independently documented chemistry, not exact-match reference endpoints**. Do not produce a reviewer-style checklist as the primary deliverable, and do not reduce R1 to Top-K or R2 to a signature inventory. Find precisely *which* solved targets, distinct chemical disconnections, compositional programs and actual electron-flow forks arise beyond the frozen baselines, while measuring the search budget and illustrating the mechanism with authentic molecules.

**Deliver:** reproducible matched-target baseline comparison, scientifically meaningful chemical-discovery metrics, figures for the reserved **R1 Figure 2 and R2 Figure 3**, a compelling **model-owned MCTS-only route/disconnection case** with RDKit molecule structures and electronically executed fork, Extended Data and a compact measured-result report that Codex can feed to paper PR #29 later. **Never place made-up points/structures/values on a publication figure.**

## 1. Actual input snapshot and immutable already observed evidence

Read `AGENTS.md`, `PROJECT_MEMORY.md`, [PR #87 result index](../results/nmi_tree_evidence_20261016/README.md) (from `docs/goals/` this relative link is optional; canonical actual repo location `docs/results/nmi_tree_evidence_20261016/README.md`), `docs/analysis/NMI_TREE_EVIDENCE_20261016.md`, `docs/results/nmi_tree_evidence_20261016/{metrics.json,analysis_availability.json,enrichment_manifest.json,source_manifest.json.gz,work_manifest.json}`, existing parsers in `src/mechet/analysis/`, the original route manifests, and the active R1/R2 paper protocol. Preserve already completed 23 export sets, their hashes, and raw data.

**Recorded runs to compare, with these exact names (not fabricated aliases):**

| Condition | Source run id | Target population | Recorded result |
|---|---|---|---|
| Frozen Stage-II legacy greedy A | `r1_r2_frozen_06b_dynamic_routes_20261013:greedy` | 460/460 | PaRoutes 3, RetroCast 19, URSA Drugs 15, URSA Expert 1 nonzero stock AND closures, plus one already-stock case |
| Frozen Stage-II legacy greedy B | `r1_r2_frozen_06b_shared_routes_20261014_05:greedy` | 460/460 | PaRoutes 3, RetroCast 17, URSA Drugs 15, URSA Expert 1 nonzero closures, plus one already-stock |
| Saved electronic-state MCTS | `r1_r2_frozen_06b_shared_mcts_20261014_07` | 171/460 completed exported reports | 199 nonzero stock-closed routes in the PR #88 200-report update (historical PR #87 has 170); 1 already-stock; 14,040 FLOW-fork contexts; 71 cases with route-count lower bound >1; 7,082 terminal contexts |
| Saved single-reaction MCTS | `mech_uspto31k_full_mcts_ray_20261016_03` | 1,711/3,120 exported | old `rxn_prod_min` **proxy**: top1 32, top5 72, pool 79; not official principal-product accuracy |
| Two historical single-reaction greedy cohorts | October 13 `mech_uspto31k_full_dynamic_electron_tree_20261013`, October 14 `mech_uspto31k_full_shared_electron_tree_20261014_05` | 3,120 each | archived proxy-reference hits 139 / 144, **not matched-budget MCTS outcomes** |

MCTS per-target observed elapsed median/P90 **10.04 / 32.73 h** and sampled token median/P90 **392,892 / 1,449,980** (171 exported target reports). Those are recorded, not additive GPU-time totals. Native MCTS context/edge counts and source-scoped shared caches must not be presented as independent global observations.

Use source assets under `outputs/paper/nmi_tree_evidence/observed_20261016_v1` and manifest-declared original files. Do not assume these data paths are available on a new Codex cloud host; resolve them in the compute environment. Do not count storage replicas, shared WALs, cached policy branches or resumed runs twice. Every output row carries frozen target/run/stock/checkpoint/evaluator/source SHA and `source_record_pointer`.

## 2. R1: baseline comparisons that reveal actual synthesis utility

### R1-A: immediate **paired target** analysis from existing saved runs — mandatory, no new GPU sampling

For the same latest 200 MCTS target IDs (or the frozen 171-ID PR #87 historical subset for exact reproduction) and **each** 460-target greedy snapshot:
- Join by `(dataset/cohort, target_id, product, frozen stock SHA, checkpoint/observation/executor contract)`. Verify route product, same stock, nonzero reaction edges and full AND closure. If a field is not comparable, show the mismatch; do not silently combine runs.
- Compute a target-paired 2×2 table: **both solved / only MCTS solved / only Greedy solved / neither**, separately for PaRoutes 100, RetroCast 160, URSA Drugs 100, URSA Expert 100 and the 171 observed-ID subset. Preserve source IDs of all rescue and reverse-loss cases.
- Show 460-population **observed MCTS lower bound** alongside full-run greedy, and a separate **paired 171-ID descriptive comparison**. If budgets are unequal, title it "observed-run recovery" / "target-paired descriptive contrast", NOT "equal-work algorithm advantage".
- Quantify a **new disconnection** by canonical unique structural precursor *set* together with a stable mapped product reaction-centre/bond-change key. Separate (i) new electron-token ordering only, (ii) new executed successor, (iii) changed precursor set, (iv) changed chemical disconnection and (v) new complete stock-connected route. The same molecule under changed temporary atom IDs must not count as new chemistry.
- For each MCTS-only stocked-solved target, characterize the earliest **actual** recorded same-parent FLOW/IMPORT branching whose alternatives lead to different *chemical successors* and, where traceable, different terminal precursors and full routes. Count `rescue_target_with_verified_fork` as a distinct evidence set, not automatically 170. Report how many winners have a genuine *different* route graph, rather than merely alternative textual/program encodings.
- For the 71 cases where multiple routes are recorded, compare **distinct full-route reaction-edge graphs**, distinct intermediate molecular species, first-disconnection difference, reaction-centre difference and stock-leaf identity; report a lower bound, not exhaustive reaction possibility.

Primary R1 metrics:
`StockSolved@B` (all leaves in frozen stock, at least one model reaction);
`MCTS-only Rescue@B` and `Greedy-only Loss@B` on identical target IDs;
`NetAdditionalSolved@B`;
`NovelDisconnection@B` **per target** and per actual work unit;
`DistinctCompleteRoutes@B` under structural route-key deduplication;
`Cost-to-First-Solved` only when a chronological witness/counter exists;
`UsefulDisconnections / 1e6 actually generated tokens` and per 1,000 executor attempts;
`UniqueChemicalSuccessors / 1,000 attempted actions`.
Report exact numerators, denominators, label-defined stock closure and paired target bootstrap 95% intervals where comparability holds. PaRoutes/RetroCast overlap must be audited and their totals not called independent disjoint molecules.

### R1-B: budget-controlled evidence — reuse first, new work only when required

**First** inspect native MCTS visited-node/event order, temporal route witnesses, cache charging, and existing greedy/independent sampling traces. A target summary with only final aggregate cost cannot be transformed into a fictional `Solved@budget` anytime curve.

Compare at the same frozen Stage-II Qwen3-0.6B adapter (actual weight SHA `40204c7a059d16976fc4b24b315c3bdde515071757e9de75`):
- **A Greedy K1**: existing frozen full-route greedy runs, with proper model ancestry and input matching.
- **B Independent complete-electron-trajectory sampling** (true independently seeded paths; *not* alternative visits from a shared MCTS). If existing independent-K outputs do not exist, implement and execute the **needed full declared matched target cohort**, using the same Qwen observation/tool/executor contract, same target/stock and predeclared total inference budgets. Batch inference and cache responsibly; count *all* attempts.
- **C Conventional complete-reaction candidate planning** (existing Syntheseus/Retro* or best-first adapter, when runnable): same actor-generated reaction edges/complete candidate work, same outer AND-route condition and gold-independent rank; isolate *search between complete reactions* from *search within electron-flow transformations*. Do not mislabel a missing planner as a completed baseline.
- **D MechET executed electron-flow MCTS**: consume existing actual MCTS records; if data contain chronological work logs, compute honest budget curves. Do not rerun 10-hour/target tree expansions just to change plotting, and do not assume cached/shared policy calls are additive GPU work.

**Budget contract:** publish actual `sampled_tokens`, `prompt_tokens`, `policy_request_batches`, `executor_calls`, node expansions, executor/cache hits, and measured wall time where available; different non-additive GPU-cache stats must remain explicitly distinct. Freeze one budget grid on validation/config **before evaluating held-out outcomes**. If matched budgets require new independent sampling, use the exact existing latest 200 MCTS target IDs for the immediate paired contrast and all **460** R1 targets as the final official route cohort; save any incomplete runs with an explicit progress manifest. Do not use arbitrarily small post-selected success subsets as the headline. No data-dependent success target/seed selection.

Search efficiency needs a meaningful Pareto chart: **stock-AND successful targets vs actual model work**, and **number of distinct useful disconnections vs actual work**. Pure speed optimization is a supporting plot, not a new method headline.

## 3. R2: demonstrate chemical construction beyond program-count novelty

Preserve both original R2 tasks; do not collapse R2 into morphology:
- **Full-route R2**: exactly the same 160 RetroCast `mkt-cnv-160` target IDs, frozen stock and saved trajectories as R1. Reference depth 2/3/4/5 each n=40. C0/C1/C2/Unknown only when every required reference electron program has independent source verification. The current 160 reference-route composition statuses are **Unknown**; don't infer C1 from generated program novelty.
- **Strict reaction-level H2**: original split `223,863 / 6,200 / 27,104` and Qwen3-8B H2-only trained model lineage, with Direct, Open-Flow, Closed-Loop and Closed-Edit controls if actual checkpoints and full output files exist. Full-corpus Stage-II Qwen0.6B is **not** an H2 model. Historical Direct 14.87% first / 28.15% Pass@10 remain independent, not comparable to the 0.6B full-corpus MCTS.

**Immediate comparative analysis on the R1 saved 171 MCTS targets and each archived greedy baseline:**
- Compare **unseen ordered complete electronic-program signatures under the correct model exposure** at the **per completed reaction / per canonical distinct terminal precursor / per target** level, not just 3,672/7,082 terminal *contexts* (context multiplicity matters).
- Evaluate `UnseenProgram→NewPrecursor→NewDisconnection→StockSolved` as a cascade **with source-linkable reaction IDs**. Report fractions/counts on the whole observed target population, including zero-novelty targets. Relative training primitive support is displayed per component move; `BE_DELTA` and unusual import identity remain separate from the ordinary primitive signature abstraction.
- Chemistry stratification: mapped changed-bond and reacting-atom counts, source-sink multi-move coupling, ring closure/opening/reorganization and reference route depths. Use model-observed complexity and reference-defined complexity as separate variables.
- For a genuine paired advantage claim, include matched independent K sampling under the same representation and equivalent work. A direct SMILES model has no electronic-program signature; compare it on endpoint/disconnection quality, not native electron signature counts.
- If authenticated H2 model outputs are available, report exact same 27,104 IDs, both First/Pass@K semantics and paired CI; if absent, complete all current-tree chemistry comparisons and put H2 as a **separately named pending panel**, not a fake quantitative result.

Recorded anchors from PR #87: **59,289** distinct accepted FLOW transitions with chemical deltas in the R1 MCTS; **7,082** terminal contexts, of which **3,672** carry unseen whole ordinary-program signatures, **3,392** seen, **18** unavailable/nonordinary. Strict FlowER historical signature census **604/6,553** differs from current actor's exposed ordinary local/whole **603/6,491** plus **1,377** nonordinary signatures. Preserve both bases, labelled distinctly.


## 4. LGRV-v1: independent literature verification of EVERY generated synthetic reaction step

**This replaces the former mandatory reference-endpoint/ranking analysis.** Follow [the complete protocol](MECHET_LITERATURE_GROUNDED_REACTION_ROUTE_PROTOCOL_20261018.md), sections 0–4. A step is a full forward-oriented structural precursor set → product from an executor-finalized reaction; an individual IMPORT/FLOW/FINISH is a separate symbolic decision and receives *mechanistic consistency* descriptors, not an unjustified experimentally validated elementary-step label.

**Immediate source population** (PR #88, latest file snapshot 17 Oct 23:09 +0800): R1 **200/460** saved target reports, **199** nonzero frozen-stock AND closures, one already-stock, four original cohorts. Single-reaction MCTS **3,120/3,120** complete target results under 10 total full trajectory attempts each, of which **3,022** produced at least one terminal structural precursor. Older PR #87 **171** R1 / **1,711** 31k populations remain clearly labeled historical snapshots, not silently pooled. Source files are from PR #88's snapshot and PR #87's manifests, not new inference.

Build:
1. All structurally unique completed precursor→product reaction records from actual MCTS routes and single-reaction terminal candidates, separating structural precursors from imported agents/solvents and attaching true action/source pointers.
2. Local structural reaction lookup in **ORD** and the Lowe US-patent reaction extract, followed by original patent preparation/academic experimental passage inspection. DOI, patent ID, section/example, experimentally reported reactants/product, conditions, stereo, mapping and source/training overlap are required evidence fields. A title/keyword/patent *claim* without actual performed experiment is not Tier A.
3. Exact reaction Tier A / experimentally analogous reaction Tier B / unresolved Tier C / specifically documented conflict Tier D, **and independent machine/review status**. A missing hit is **C, not D**. Preserve source URLs/patent family dedup and all misses. Automated structure match by itself is 'machine candidate', never silently 'source verified'.
4. Side-by-side comparison to *the same* Greedy reactions on the same target IDs, using **exactly the same evidence corpus, normalisation, retrieval budgets and grading**. Don't use dataset GT reference precursors in retrieval or ranking. Report observed-case counts, deduplicated reaction counts, per-target source-supported discovery, primary-source-checked coverage and distinct chemical disconnections; no NLL/Top-K headline.
5. Mechanistic arrow compatibility is separately traced to actual source–sink symbolic operations, reaction class and, where available, an explicit literature mechanism. Documented **net reaction** does not experimentally validate the exact model-generated arrow ordering.

**Current real precedent anchor:** the Cabozantinib free-base acid-chloride/aniline structural coupling (one reaction edge) is supported by US20160022662A1; this does **not** support the entire route or all internal arrows. Use the same standards on baselines.


## 5. Route-level chemistry evidence and main R1/R2 metrics

**The entire route, not reference matching, is the outcome.** For each generated route, verify frozen-stock AND closure; for each *essential complete reaction edge*, attach A/B/C/D and `verification_status`. Use a route graph keyed by actual products, canonical structural precursors and purchaseable leaves, not a list of accepted electron arrows.

Mandatory R1 statistics:
- `StockSolved@B`: at least one nonzero complete stock-connected route over **all 460 target IDs**; PR #88 observed lower bound 199/460.
- `AnyExactStepPrecedent@B`: complete stocked route contains at least one source-confirmed exact reaction step, explicitly distinct from all-steps supported.
- `AllStepExactPrecedentSolved@B`: complete stocked route where every essential edge has Tier A exact experimental documentation.
- `PrecedentCoveredSolved@B`: complete stocked route where every essential edge has Tier A or source-checked strong analogue Tier B; named "precedent-covered", **not wet-lab validated whole route**.
- `RouteStepEvidenceCoverage`: fraction of distinct essential route edges with A/B experimental precedent; show distribution and unretrieved source gap, not invented failures.
- `LiteratureSupportedDisconnection` and `MCTSOnlySupportedDisconnection` vs Greedy, plus target-paired `PrecedentCoveredSolved` both/only MCTS/only Greedy/neither and normalized supported chemistry per actual generation tokens/engine work.
- Differentiate `indexed_match`, `source_passage_checked`, `verified` and `unresolved`; plot exact numerator/denominator for each rather than a fabricated validation yield.

Mandatory R2 metrics: `unseen whole program→chemically distinct reaction→independently precedent-supported transformation→stock-connected route`, after deduplication into reaction / route / target units and comparison to same-ID Greedy. Preserve historical 604/6,553 vs actual-actor 603/6,491 exposure, original RetroCast 160 target identities, H2 27,104 as separately authenticated lineage. Current C0/C1/C2 reference route annotations remain Unknown; generated program novelty does not create source reference C1.

**Single reaction 31k**: source-documented reaction coverage and `AnySourceSupportedReaction@10Attempts`, `DistinctSupportedDisconnections/target`, supported unique reaction counts vs Greedy; evaluate all *actual terminal candidates*, with reference endpoint accuracy moved to an explicitly optional legacy appendix. A complete reaction supported by external literature counts as chemical evidence **even when it differs from the database GT**.


## 6. Code and artifacts Codex MUST implement

Extend PR #87's genuine source-provenance analysis code, **do not** delete the old observed figures or core PR #89 paired-target work. Implement the detailed LGRV-v1 modules and source corpus adapters:
- `src/mechet/analysis/literature_reaction_keys.py`: forward oriented complete reaction keys, stereo/roles, mapping and change-centre.
- `src/mechet/analysis/literature_index.py`: source-versioned bulk reaction-structure index and patent-family dedup, exact/analogue retrieval.
- `src/mechet/analysis/literature_evidence.py`: primary source example/paragraph and DOI/patent, explicit verified/retrieved status, A/B/C/D, model/executor hashes and query/retrieval budget.
- `src/mechet/analysis/literature_route_metrics.py`: every reaction edge to source evidence, full AND-route coverage, target-level paired Greedy/MCTS comparisons, supported disconnection and R2 composition cascade.
- `src/mechet/analysis/mechanism_consistency.py` (only when actual source allows): separate net reaction precedent from individual symbolic FLOW compatibility; no invented physical intermediates.
- `src/mechet/analysis/paired_baselines.py` and `disconnection_comparison.py` (or equivalently tested implementations): already-source-bound structural chemistry and 460 target cohort comparison.
- `scripts/verify_nmi_literature_reactions.py` and `scripts/render_nmi_literature_figures.py`: actual records, source lookup/role matching, source-checked evidence, citations and chemistry-led vector figures; deterministic and restartable.
- `tests/test_nmi_literature_protocol.py`: different reactants/same product, valid analogue vs weak centre collision, no paper -> C, patent claim only -> C, route step missing evidence, per-target source and stock join, model/GR-ready use, 200/460 incomplete snapshot vs all target denominator, latest 3120/3120, no GT leakage.

Produce versioned `outputs/paper/nmi_literature_evidence/<slug>/` with frozen predictions/corpora, reaction steps, raw source pointers, retrieval hits, source passages, evidence table, route evidence, mechanism tags, MCTS-vs-Greedy comparisons and work, exact counts/coverage, chemistry evidence figures, English Nature captions, source/ChemDraw-ready atom-level RDKit SVG and full manifest. Use low-saturation [MechET figure palette](../../configs/figures/mechet_nmi_tree_style_v1.yaml), 183mm PDF/SVG and 300dpi PNG. No fake source evidence, invented experimental yield or fabricated MCTS task completion.

The requested **main figures** are `fig2_r1_literature_supported_routes` and `fig3_r2_supported_composition`; R3 may reuse the evaluator for `fig4_r3_structural_precedent_cases` but do not alter the five Results or paper PR #29 without a reviewed patch.

## 7. Figure and paper editorial contract

Use [companion figure storyboard](NMI_R1_R2_MATCHED_FIGURES_20261017.md).
- Figure 2 should make *the comparative advantage* prominent: paired target recovery (MCTS-only / greedy-only) + budget–utility evidence where measured; a **real rescued full route and key electron-flow fork** in large RDKit chemistry space; compact recovery/disconnection/source info. Original URSA/RetroCast/PaRoutes contexts stay.
- Figure 3 should show *what chemical property changed*: unseen program vs previously seen program, downstream unique endpoint/disconnection and stock closure, plus reaction centre/ring/flow complexity and a real complete source-linked program; H2 true generalization panel if independently authenticated. Preserve existing 604/6,553 signature source inset.
- Extended Data: candidate rank gap/official endpoint, route-overlap diagrams, new disconnection count per work, paired 2×2 table, budget/cost Pareto, structural Tier A/B case evidence, no need for more decorative tree galleries.
- No new seventh main item: keep final NMI six-display-item decision for paper PR #29, and don't silently delete Table 1 or rewrite R3–R5.
- Use low-saturation slate/steel-blue/sea-teal/ochre, white ground, precise black RDKit bonds and heteroatom defaults, heavy enough chemical bond/atom rendering, crisp true curved arrows, 183 mm page width, 300-dpi raster preview, embedded vector fonts. Prefer asymmetric panel sizes and chemistry insight rather than large empty trees or default slide-card compositions.


## 8. Execution priorities and completion criterion

**E1 — current saved-data source freeze (no new GPU work):** load PR #88's R1 200/460 and all 3,120 mech31k searches, PR #87's source-indexed tree parser/true action edges, and the two archived Greedy 460-route runs. Export every observed *complete reaction* with a structural precursor/product key, agent roles, verified frozen-stock route ID and source pointer.

**E2 — primary literature retrieval and experimental-evidence typing:** download/index **lawfully accessible** ORD and patent reaction files (or use existing indexed corpora), run source-preserving bulk structural lookup on all unique generated reactions, retrieve original relevant patent examples/paper procedures, and classify source-positive exact/analogues with explicit verification status. Do not wait for a new human annotation team before producing the machine-index + checked-source evidence figures; distinguish machine-only from source-passage-checked outputs.

**E3 — direct chemical advantage vs baselines:** compare matched IDs of greedy, MCTS and available independent-K complete routes under the same literature protocol, count `AnyExactStepPrecedent`, `AllStepExactPrecedentSolved`, `PrecedentCoveredSolved`, new A/B supported disconnections and program→chemistry→route R2 cascade. Label unequal work as descriptive; new independent route sampling requires existing approved resources, no redundant 10h/target MCTS reruns.

**E4 — NMI figure delivery:** real Figure 2/3 with source-backed chemical reactions and at least one readable positive fork/route example, matched route evidence/cost panels and Extended Data coverage curves. Deliver tested scripts, source-verified example links, actual target counts, manifest, captions, file hashes and explicit unsearched/unavailable fractions. Do not output invented literature-supported counts.

**Main target accuracy or ranking Top-K is no longer the assigned headline task.** Do not spend GPU time on NLL re-ranking, train a new actor or reintroduce recursive self-improvement. Existing unrelated historical single-step benchmark tables may remain in manuscript support until a separate editorial decision.


**Definition of Done:** implemented and source-traceable literature retrieval, A/B/C/D evidence for *complete generated reaction steps*, genuine per-route evidence coverage and stock closure, same-ID Greedy/MCTS chemical-source comparisons, R2 novelty-to-supported-chemistry cascade, NMI chemistry-led vector Figure 2/3, working code/tests and explicit source-corpus coverage; no reference endpoint Top-K as the primary metric.