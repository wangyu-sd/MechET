# MechET NMI — Literature-first R1/R2 comparative figure storyboards
Updated 2026-10-18. Mandatory precedent protocol: [LGRV-v1](MECHET_LITERATURE_GROUNDED_REACTION_ROUTE_PROTOCOL_20261018.md). Reference endpoint Top-K and NLL rank are **not main chemistry figures**. Applies to the [baseline execution goal](NMI_R1_R2_MATCHED_BASELINE_GOAL_20261017.md), continuing [PR #87 real-figure source](https://github.com/wangyu-sd/MechET/pull/87). The earlier comprehensive tree atlas and 7/13/5-action galleries remain valuable as **Extended Data**, not repeated full-size in the main comparative figures.

## Editorial principle: figure panels must establish a scientific advantage

A rich tree alone says "search ran." A publication main panel should instead say **what independently documented chemical transformations, different precursor options and precedent-covered complete routes the search constructed relative to identical-target baselines**.

Use actual paired outcomes, exact route chemistry, actual source–sink actions, source-backed structural transformations, and honest work accounting. Depict an advantage when the records support it. Present unavailable/not-yet-matched comparisons separately and compactly, not as a large negative central callout.

Style source of truth: `configs/figures/mechet_nmi_tree_style_v1.yaml`. PDF/SVG vector at 183 mm width, white background, identical muted blue–grey and teal mapping across Figure 2/3 and supplementary. Real molecular graphs via RDKit MolDraw2DSVG with visible heavy bonds, heteroatom colors, stereochemistry and selected atom maps. Avoid narrow uniform tiles, long decorative headers, icons, gradients, full-tree spaghetti over a molecule or giant text-only caveat boxes. One primary finding dominates per figure. Captions carry detailed scope and provenance; panels carry n, axis units, methods and short chemical callouts.

## Figure 2 — Discovery of stock-connected synthesis routes through electron-flow-state search (R1)

**Core message:** *MechET constructs stock-connected synthesis routes whose individual precursor→product reaction steps can be matched to experimentally documented exact/analogous precedents, including new disconnections introduced by actual electronic-state forks.* Show the actor/method and actual source IDs, not a textbook synthetic scheme.

At two-column width favor a quantitative + chemistry asymmetric layout:

```text
 a · DOMINANT COMPARATIVE RESULT: chemistry-backed synthesis
 ┌─────────────────────────────────────────────────────────────────┐
 │  URSA Expert | URSA Drugs | RetroCast | PaRoutes                 │
 │  StockSolved / AnyExactStep / AllStepExact / AllStepA+B  │
 │  matched Greedy vs MCTS, correct n / source-checked rates  │
 └─────────────────────────────────────────────────────────────────┘
 b · EVIDENCE vs WORK, if source-traceable,      c · ELECTRON FORK
 ┌─────────────────────────────────┐ ┌───────────────────────────┐
 │  precedent-covered route      │ │ compact true state tree  │
 │  Qwen tokens, executor calls    │ │ highlighted branch     │
 │  and wall-time                 │ │ before vs after FLOW   │
 └─────────────────────────────────┘ └───────────────────────────┘
 d · ACTUAL RDKit CHEMISTRY: PRODUCT → REACTION STEPS → SOURCES
 ┌─────────────────────────────────────────────────────────────────┐
 │  actual source-target chemistry, selected key electronic fork, │
 │  different reaction centre / precursor sets / AND-stock route  │
 │  A/B/C source tier at each step + complete AND-stock closure   │
 └─────────────────────────────────────────────────────────────────┘
```

In the real paper, panel (d) may take **nearly half the height** if the molecule is large; move extra curves to Extended Data. Layout is a guide, not forced equally sized grid.

**Panel a**: The principal quantitative axis is **literature evidence along stock-connected reaction routes**, not similarity to a database reference. Show `StockSolved@B`, `AnyExactStepPrecedent@B`, `AllStepExactPrecedentSolved@B`, `PrecedentCoveredSolved@B` and per-route `RouteStepEvidenceCoverage`. Distinguish source-retrieved/structure-checked from source-passage-checked evidence. Use two clearly named evidence regimes when needed:
1. **Existing-run descriptive**: paired latest 200 exported MCTS target IDs (the 171-ID PR #87 snapshot is historical) versus the frozen 460 greedy outputs restricted to those *same* IDs; keep unequal budgets visible. The latest 199/460 nonzero stock-connected MCTS rate is a lower bound; avoid depicting unfinished targets as confidently failed searches.
2. **Matched-work**: once per-budget control traces exist, compare full R1 460 targets (including all unfinished/missing as outcomes under the declared budget) with per-target count and paired CI.

Do not conflate MCTS-only success with a performance increase caused specifically by chemical-state branching until independent-K and complete-reaction-search controls are available.

**Panel b**: x-axis actual counted work (generated tokens, actual executor calls, or wall time; separate clearly labeled subpanels), y-axis source-supported new disconnections or fully stock-connected `PrecedentCoveredSolved@B` and `StockSolved@B` with every unsolved target in frozen cohort. Plot an anytime curve ONLY from time/cost-indexed actual witness records. If only final total cost per target is available, replace with measured budget/discovery scatter or cost ECDF in Extended Data, not a synthetic curve. Show P50/P90 and cache charges explicitly, avoiding double-counted GPU resource claims.

**Panel c–d**: choose first compelling **source-traceable** `MCTS-only` case from the paired candidate registry. The key is causal readability:
- product drawn once;
- actual normalized parent state `S_t` (symbolic, not experimentally observed);
- two **recorded** FLOW/IMPORT alternatives, source–sink arrows from exact state addresses, unique *canonical molecular successors*;
- the resulting distinct structural precursor choices and at least one actual full-route all-stock witness;
- baseline's original complete reaction/route evidence or source-bound no-witness result;
- highlight one structurally specific changed bond or disconnection and include the model checkpoint + work comparator in caption.
If Cabozantinib/Nivasorexant/RetroCast n5-00853 are not proven **MCTS-only rescue under the joined same IDs**, keep them as valid standalone case Extended Data instead of calling them the main rescue. The verified Cabozantinib acid-chloride/aniline structural pair supports **one reaction edge only**, not the whole route or symbolic proton-transfer intermediates.

Suggested figure caption opening (reword after results):
"Figure 2 | Literature-grounded reaction evidence along executed stock-connected retrosynthetic routes. (a) Target-paired stock closure and exact/analogue experimental-reaction evidence from the fixed Stage-II policy under single-path and electronic-state search; the denominator and recorded work budget are provided for each scenario. (b) Candidate/route recovery as a function of counted inference work when time-indexed route witnesses are available. (c–d) Actual model-generated alternative inverse electronic operations from the same chemical state lead to distinct precursors and stock-connected routes."

## Figure 3 — From new electronic compositions to chemically distinct reactions (R2)

**Core message:** *Search recombines local electronic operations into novel program sequences; the chemically meaningful fraction produces distinct whole reactions with source-supported experimental precedents and usable stock-connected routes.*

Suggested asymmetric design:

```text
 a · DOMINANT COMPOSITION-TO-UTILITY CASCADE
 ┌───────────────────────────────────────────────────────────────┐
 │ real per-target/unique-reaction units, paired MCTS vs greedy  │
 │ all generated → unseen whole program → changed precursor     │
 │ → new disconnection → A/B documented reaction → stock route                      │
 │ exact DOI/patent sources; unknown preserved                        │
 └───────────────────────────────────────────────────────────────┘
 b · COMPLEXITY-CONDITIONED EFFECT           c · CORPUS REUSE
 ┌────────────────────────────────────────┐ ┌──────────────────┐
 │ reacting atoms, bond/ring operations;  │ │ historic local   │
 │ depth 2/3/4/5 exact n / baseline delta │ │ 604 / whole 6553 │
 └────────────────────────────────────────┘ └──────────────────┘
 d · MODEL-OWNED REAL MOLECULAR ELECTRON PROGRAM (large)
 ┌───────────────────────────────────────────────────────────────┐
 │ P → executed states → structural precursors                   │
 │ localized electron arrows and bond changes, original source  │
 │ source-history program/primitive counts, endpoint/route link │
 └───────────────────────────────────────────────────────────────┘
```

Important evidence grouping:
- **Ordinary generated program novelty under the full-data actor's verified exposure** is a measured program-space exploration statistic; compare to Greedy and independent same-actor sampling on identical targets and work. Do not label it `C1` or generalization on the H2 split.
- **Full-route C0/C1/C2/Unknown** annotations of the 160 RetroCast references may remain entirely Unknown; still compare the empirical observed *generated* composition novelty and actual chemistry complexity with careful caption.
- **H2 Qwen3-8B Direct/Open-Flow/Closed-Loop/Closed-Edit** remains the primary proof of strict held-out-composition *prediction* only when verified trained model outputs exist on all 27,104 test reactions. It may take the dominant (a) location after data arrive; until then use the actually measured cascade. The historical 604/6,553 must be labelled corpus counts, not trained-model performance.
- A direct SMILES baseline has no electronic program: compare target-level predicted *chemical disconnections and routes*, do not force an electron signature column.
- No arrow is invented from source bond-diff heuristics. All electronic arrows must be the model's committed executor actions.

Caption opening:
"Figure 3 | Electronic-program recombination yields distinct chemical transformations with identifiable experimental precedents. (a) The fraction of completed model-generated electronic programs outside the authenticated actor's observed whole-program combinations and their progression to distinct structural precursor choices, exact/analogue experimental reaction evidence and stock-closed routes. (b) Effects stratified by independently available transformation complexity. (c) Historical strict-training local versus ordered full-program signature coverage. (d) Actual complete executed reverse-electron-flow construction."

## Extended Data (comparison-focused; preserve PR #87 assets)

**ED1 | Matched route contingency and observed work.** Four cohorts, two archived greedy versions, exact matched IDs, outcome 2×2 mosaic, per-target tokens/work and missingness; if a valid independent-K baseline has run, add measured matched-budget paired improvement.

**ED2 | Disconnection novelty versus search work.** Candidate pool size, distinct generated products, changed bond/reaction-centre classes, novel disconnections per 1e6 tokens, executor calls and unique chemical successors. Include alternate-branch reachability and asymmetry of success/failure. MCTS context multiplicity is not independent target sample size.

**ED3 | Single-reaction source-supported chemistry.** Using the completed 3,120 MCTS target outputs (up to 10 full trajectory attempts each), compare distinct terminal reactions with the same-ID Greedy archives for `AnySourceSupportedReaction@10Attempts`, Tier A/B per target and confirmed different disconnections under the same retrieval corpus and verification workload. Historical `rxn_prod_min` or NLL rank is optional separately labeled old-source metadata, not a main quality evaluation.

**ED4 | Documented chemical precedent and retrieval coverage.** Show exact vs experimentally analogous source-supported complete reactions, source DOI/patent and example locator, patent-family dedup, and machine-structure matched vs source-experimental passage checked. Show both absolute A/B yields over frozen prediction corpus and retrieval resolution coverage; no paper found remains Tier C. A complete documented reaction does NOT validate each internal FLOW arrow.

**ED5 | Positive case gallery.** Show a MCTS-only stock-AND rescue when truly evidenced; an actually new disconnection caused by a branch; and one credible chemical source-aligned alternative. Full trees + all action programs stay as linkage in PR #87 Extended Data.

## Figure release criteria

1. No placeholder molecule or simulated performance bar/curve; exact method, cohort, n, source line and runtime/model sha for each result.
2. Figure 2(a) uses paired targets and correct budget qualification, not the *misleading* comparison of 170 MCTS partial with 36/38 full greedy as if head-to-head equal work.
3. Figure 3(a) uses independent units: program contexts, distinct complete reactions, target-level stock routes; the cascade must make changing denominators visible.
4. Front-end of the paper tells a **compelling benefit**: additional plausible disconnections and usable routes emerging from actual electronic-state forks. Qualification lives in caption and methods, not as an oversized headline.
5. 183 mm wide PDF/SVG, embedded sans-serif fonts, source-stable IDs, heavy enough molecule bonds, correctly placed actual source–sink arrows, no schematic chemistry invented. Zoomed target and all precursor atoms legible when printed.
6. Name new files `fig2_r1_comparative` and `fig3_r2_comparative`; do not overwrite `fig2_r1_observed_tree_analysis` and `fig3_r2_observed_chemistry` or mutate PR #87 manifests. The current paper PR #29 six figures + Table 1 formatting decision is independent.

The outcome sought is **actual comparative scientific evidence**, not another visual audit or generic figure redesign.