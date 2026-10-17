# Codex Goal · MechET R1/R2: matched baselines, discovery attribution and comparative NMI figures

Date: 2026-10-17. **Executable implementation goal**, not another paper-story redesign.
Current data/code authority: [MechET PR #87](https://github.com/wangyu-sd/MechET/pull/87), commit `2af03e8cca4aca46e04d7619aeda4b987ee32c34`; paper authority [MechET-paper PR #29](https://github.com/wangyu-sd/MechET-paper/pull/29), commit `ffc2890e02fb265b6292cca6fe63646d0d612972`. This task is stacked directly **on PR #87**, which itself is stacked on PR #84. Do not overwrite either PR or merge to `main` on your own.

## Goal: turn existing chemical-state tree findings into comparative scientific conclusions

The user's priority is **mine and present the advantages supported by the existing MechET results**. Do not produce a reviewer-style checklist as the primary deliverable, and do not reduce R1 to Top-K or R2 to a signature inventory. Find precisely *which* solved targets, distinct chemical disconnections, compositional programs and actual electron-flow forks arise beyond the frozen baselines, while measuring the search budget and illustrating the mechanism with authentic molecules.

**Deliver:** reproducible matched-target baseline comparison, scientifically meaningful chemical-discovery metrics, figures for the reserved **R1 Figure 2 and R2 Figure 3**, a compelling **model-owned MCTS-only route/disconnection case** with RDKit molecule structures and electronically executed fork, Extended Data and a compact measured-result report that Codex can feed to paper PR #29 later. **Never place made-up points/structures/values on a publication figure.**

## 1. Actual input snapshot and immutable already observed evidence

Read `AGENTS.md`, `PROJECT_MEMORY.md`, [PR #87 result index](../results/nmi_tree_evidence_20261016/README.md) (from `docs/goals/` this relative link is optional; canonical actual repo location `docs/results/nmi_tree_evidence_20261016/README.md`), `docs/analysis/NMI_TREE_EVIDENCE_20261016.md`, `docs/results/nmi_tree_evidence_20261016/{metrics.json,analysis_availability.json,enrichment_manifest.json,source_manifest.json.gz,work_manifest.json}`, existing parsers in `src/mechet/analysis/`, the original route manifests, and the active R1/R2 paper protocol. Preserve already completed 23 export sets, their hashes, and raw data.

**Recorded runs to compare, with these exact names (not fabricated aliases):**

| Condition | Source run id | Target population | Recorded result |
|---|---|---|---|
| Frozen Stage-II legacy greedy A | `r1_r2_frozen_06b_dynamic_routes_20261013:greedy` | 460/460 | PaRoutes 3, RetroCast 19, URSA Drugs 15, URSA Expert 1 nonzero stock AND closures, plus one already-stock case |
| Frozen Stage-II legacy greedy B | `r1_r2_frozen_06b_shared_routes_20261014_05:greedy` | 460/460 | PaRoutes 3, RetroCast 17, URSA Drugs 15, URSA Expert 1 nonzero closures, plus one already-stock |
| Saved electronic-state MCTS | `r1_r2_frozen_06b_shared_mcts_20261014_07` | 171/460 completed exported reports | 170 nonzero stock-closed routes; 1 already-stock; 14,040 FLOW-fork contexts; 71 cases with route-count lower bound >1; 7,082 terminal contexts |
| Saved single-reaction MCTS | `mech_uspto31k_full_mcts_ray_20261016_03` | 1,711/3,120 exported | old `rxn_prod_min` **proxy**: top1 32, top5 72, pool 79; not official principal-product accuracy |
| Two historical single-reaction greedy cohorts | October 13 `mech_uspto31k_full_dynamic_electron_tree_20261013`, October 14 `mech_uspto31k_full_shared_electron_tree_20261014_05` | 3,120 each | archived proxy-reference hits 139 / 144, **not matched-budget MCTS outcomes** |

MCTS per-target observed elapsed median/P90 **10.04 / 32.73 h** and sampled token median/P90 **392,892 / 1,449,980** (171 exported target reports). Those are recorded, not additive GPU-time totals. Native MCTS context/edge counts and source-scoped shared caches must not be presented as independent global observations.

Use source assets under `outputs/paper/nmi_tree_evidence/observed_20261016_v1` and manifest-declared original files. Do not assume these data paths are available on a new Codex cloud host; resolve them in the compute environment. Do not count storage replicas, shared WALs, cached policy branches or resumed runs twice. Every output row carries frozen target/run/stock/checkpoint/evaluator/source SHA and `source_record_pointer`.

## 2. R1: baseline comparisons that reveal actual synthesis utility

### R1-A: immediate **paired target** analysis from existing saved runs — mandatory, no new GPU sampling

For the same exact MCTS 171 target IDs and **each** 460-target greedy snapshot:
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

**Budget contract:** publish actual `sampled_tokens`, `prompt_tokens`, `policy_request_batches`, `executor_calls`, node expansions, executor/cache hits, and measured wall time where available; different non-additive GPU-cache stats must remain explicitly distinct. Freeze one budget grid on validation/config **before evaluating held-out outcomes**. If matched budgets require new independent sampling, use the exact existing 171 MCTS target IDs for the immediate paired contrast and all **460** R1 targets as the final official route cohort; save any incomplete runs with an explicit progress manifest. Do not use arbitrarily small post-selected success subsets as the headline. No data-dependent success target/seed selection.

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

## 4. Candidate selection/ranking: re-evaluate without rebuilding the MCTS pool

On the **same saved 1,711 MCTS product IDs** in mech-USPTO-31k and the same-ID subset of the two **3,120-ID** greedy archives:
- Re-score original actual structural precursor sets against the **corrected main-product structural endpoint contract**, using the official endpoint evaluator/canonical stereo+component normalization, not only old `rxn_prod_min`.
- Report `PoolRecall@B`, generation-order First/Pass@K when genuinely independent complete episodes exist, `NLL RankedTop1/5/10`, MRR and `RankingGap=PoolRecall-Top1` using the same canonical-unique candidate pool and actual recorded budget. A pool hit is not independent Pass@K.
- Diagnose ranking quality **conditional on any recorded reference hit** and across the full attempted target denominator. The historical proxy Top1=32/1711, Top5=72, pool=79 remain historical, not the new main endpoint results.
- Compare at least a fixed **mean-NLL selector**, a source-independent **lightweight executor/chemical-score ranker** or documented frozen heuristic, and an optional train/validation-fitted ranker with the **identical saved candidate pool**. Do not use held-out GT in inference scoring, sorting, candidate generation or score-weight selection. Distinguish score reranking from search-generation gains.
- If a correct-endpoint evaluator needs additional source fields, recover them from manifests/raw reference records; if absent, report field dependency and do not convert proxy hit percentages into official results.

This candidate-ranking analysis is a **supporting evidence panel** for R1 and R4, not authorization to rewrite R4 or train a new large actor.

## 5. Independent chemical evidence: matched baselines, no extra annotation program

PR #87 already established **one source-supported Cabozantinib free-base acid-chloride/aniline structural coupling** via `US20160022662A1`. Keep this with precise scope; training/source overlap and whole-route feasibility remain unknown.

For **the same selected rescued/disconnection cases**, pull available independent source-matched complete reaction data and compare MechET predictions to baseline **at identical product IDs and retrieval rules**:
- Tier A exact independently sourced precursor–product transformation;
- Tier B experimental close analogue;
- Tier C unresolved, no independent source;
- Tier D actual documented chemical contradiction, not a missing search hit.
- Source level DOI/patent/reaction ID + structural match + stereochemistry + dated source/split overlap. Candidate-specific evidence, not attached to symbolic intermediate FLOW states.
- Start with *all frozen eligible* fully structural reaction pairs for automated exact reaction retrieval where available; show coverage of retrieval vs validated source support, and selection rule for any representative case gallery. Do not create a new staffed-human-labeling dependency just to produce the baseline figures. When no reviewer/source verifies a pair, leave it at `retrieved_unreviewed` or C, but **still deliver** the population-level intrinsic and structural comparative metrics.

Chemistry case visualization should pair: same product and stock; actual baseline candidate route (or an explicit no-witness budget outcome); actual MechET novel FLOW fork; different endpoint precursor structure; complete stock-leaf witness; and independently supported chemistry when available. Use Cabozantinib, Nivasorexant and RetroCast `n5-00853` only if comparison/source data support their intended claim. Do not label unrelated chemically different routes as equivalent in literature.

## 6. Code and artifacts Codex MUST implement

Extend PR #87's existing `src/mechet/analysis/` layer. Suggested new code:
- `src/mechet/analysis/paired_baselines.py`: immutable target/stock/model/evaluator joins; per-target 2×2 route outcomes; exact numerator/denominator+bootstrap; partial-vs-full separation.
- `src/mechet/analysis/disconnection_comparison.py`: chemical-vs-program dedup, fork→terminal→route lineage, route graph keys and new-disconnection attribution.
- `src/mechet/analysis/comparative_ranking.py`: same-pool official endpoint rescoring, frozen rankers, score provenance and nonterminal handling.
- `src/mechet/analysis/comparative_composition.py`: current-exposure-scoped program novelty, typed nonordinary controls and useful-chemistry cascades.
- `scripts/analyze_nmi_r1_r2_comparisons.py`: CLI taking PR #87 `--analysis-root`, `--source-root`, `--assets-manifest`, optional added baseline prediction dirs, immutable `--comparison-config` and a NEW output root; resumable and deterministic.
- `scripts/render_nmi_r1_r2_comparisons.py`: cameras-ready editable PDF/SVG, PNG previews, genuine RDKit molecules/recorded electron arrows, actual closed-AND-route example and exact source provenance.
- `tests/test_nmi_r1_r2_comparisons.py`: paired ID exactness, all-leaf route witness and already-stock exclusion, outcome 2×2, rank first/TopK/pool same candidates, true independent sampling semantics, stereo/canonical precursors, route/disconnection dedup, repeated transient contexts, cash/share work, unknown H2 split and program exposure, missing chronological budget, DOI evidence case.

Reuse `configs/figures/mechet_nmi_tree_style_v1.yaml`; benchmark/run settings are predeclared in `configs/analysis/nmi_r1_r2_matched_baselines_20261017.yaml`. If separate matched full-cohort inference is needed, add a manifest-bound pipeline that **uses existing frozen actor/runtime**, not a newly designed model or training reward. Do not modify existing PR #87 raw outputs or figure hashes. All large source artifacts remain on the compute volume; commit small figures, charts, run/readme manifests and tested code, not multi-GB search caches.

Expected new versioned output:
`outputs/paper/nmi_r1_r2_comparisons/<run_slug>/`
with `input_join_audit.json`, `paired_route_outcomes.csv`, `baseline_method_manifest.json`, `disconnection_attribution.jsonl.gz`, `composition_cascade.json`, `ranking_matched.json`, `budget_work_comparison.json`, `case_registry.csv`, `summary_measured.md`, `figures/fig2_r1_comparative.{pdf,svg,png}`, `figures/fig3_r2_comparative.{pdf,svg,png}`, Extended Data as needed and `figure_manifest.json`.

## 7. Figure and paper editorial contract

Use [companion figure storyboard](NMI_R1_R2_MATCHED_FIGURES_20261017.md).
- Figure 2 should make *the comparative advantage* prominent: paired target recovery (MCTS-only / greedy-only) + budget–utility evidence where measured; a **real rescued full route and key electron-flow fork** in large RDKit chemistry space; compact recovery/disconnection/source info. Original URSA/RetroCast/PaRoutes contexts stay.
- Figure 3 should show *what chemical property changed*: unseen program vs previously seen program, downstream unique endpoint/disconnection and stock closure, plus reaction centre/ring/flow complexity and a real complete source-linked program; H2 true generalization panel if independently authenticated. Preserve existing 604/6,553 signature source inset.
- Extended Data: candidate rank gap/official endpoint, route-overlap diagrams, new disconnection count per work, paired 2×2 table, budget/cost Pareto, structural Tier A/B case evidence, no need for more decorative tree galleries.
- No new seventh main item: keep final NMI six-display-item decision for paper PR #29, and don't silently delete Table 1 or rewrite R3–R5.
- Use low-saturation slate/steel-blue/sea-teal/ochre, white ground, precise black RDKit bonds and heteroatom defaults, heavy enough chemical bond/atom rendering, crisp true curved arrows, 183 mm page width, 300-dpi raster preview, embedded vector fonts. Prefer asymmetric panel sizes and chemistry insight rather than large empty trees or default slide-card compositions.

## 8. Execution priorities and completion criterion

**E1: Deliver NOW from already stored outputs:** paired 171-ID MCTS/greedy outcomes and R1 full-460 closure lower-bound context; MCTS-only actual target case registry; qualitative and quantitative new disconnection/fork/route relationship; R2 original on same 160 RetroCast IDs; saved MCTS 1,711-ID rank diagnostics. Publish real comparative figure drafts + manifests with no new GPU allocation required.

**E2: Fix comparability where existing data permit:** corrected full principal-product 31k rescoring (if source reference fields exist); budget-prefix/witness curves only if chronological data are real; exact measured cost/target-matched status; source-verified reaction/patent match on selected cases.

**E3: Complete required baselines without changing the scientific task:** implement + execute full matched target frozen-actor independent K sampling/standard full-reaction candidate planner where records are missing, using existing approved runtime and resources. Count every request/token/execution, freeze budgets before outcome analysis. **Do not submit unbounded expensive new MCTS**, restart existing tasks, switch model family, or create unauthorized compute allocations. If necessary original machine/GPU assets are unavailable, finish E1/E2 and produce exact queued baseline commands/target manifests and missing-dependency list; do not claim E3 done.

**E4: Final R1/R2 article-ready figures:** generate from measured same-protocol rows; include paired CIs, source case registry, full caption, color/style review and figure source hashes. Preserve the original sixteen research packages and append measured results for eventual paper PR #29 integration. Do not automatically merge the paper branch.

Do **not** stop after code/test scaffolding or another internal report: run on the complete **available** records, render *actual* differences, trace the genuine interesting chemistry and carry necessary matched baselines through available resources. Keep the substantive user-facing writeup centered on advantages actually observed. Software QA and provenance should support the scientific story, not replace it.

**Definition of Done:** a new implementation PR with measured paired wins/losses and useful-disconnection figures, honest matched-budget evidence when feasible, scripts/tests, publication-quality real molecule/route visualizations, exact IDs, meaningful baseline comparison tables, and clear remaining formal benchmark dependencies; never fabricate missing H2 C1 labels or force unverified legacy baseline scores into the main result.