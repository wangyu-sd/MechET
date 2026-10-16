# Visual art direction · MechET R1/R2 NMI search-tree figures

This document is the editorial blueprint for [the Codex figure goal](NMI_R1_R2_TREE_EVIDENCE_GOAL_20261016.md). It is **not a request for illustrator mockups** or a license to generate chemical structures not present in the actual output. It is a technical plan for fully reproducible, NMI-main-text-quality figures that integrate quantitative evidence, molecular structures, a dot–edge search tree and interpretable chemistry.

References: [Nature research figure guide](https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/), [figure dimensions and panel layout](https://research-figure-guide.nature.com/figures/building-and-exporting-figure-panels/), [NMI Article six display items](https://www.nature.com/natmachintell/content). The target PDF is **183 mm** wide (double column), normally below **170 mm** high, with editable vector art. Use Arial/Helvetica or equivalent sans-serif; body labels **5–7 pt at final size**, bold **8 pt lowercase panel letters**, thin yet legible lines, 300-dpi PNG preview, editable PDF/SVG, embed TrueType Type 42. The manuscript remains six main Figures plus a currently separate Table 1 in paper PR #29, so the eventual journal six-display-item limit needs editorial reconciliation without changing these source-data results.

## Core art direction

Visual signature: **scientific editorial, not AI-produced diagram aesthetic**. Bright white ground, slate text, desaturated steel blue for electronic search, muted sea-teal for accepted endpoint/path, warm ochre for external IMPORT/conditional contributions, restrained clay/rust for rejected/unsupported symbolic attempts, pale grey for context. Use contrast and geometry, not saturated rainbow or drop shadow. Define all hexadecimal colors in the shared YAML; never add per-panel ad-hoc colors. Use the same colors across Figures 1–6 where labels mean the same thing. The paper must be readable in grayscale and color-blind modes: shapes, marker outlines, line styles and explicit labels are essential.

All charts: axis lines/ticks, short English units, direct data labels when they clarify without collisions, n and uncertainty when applicable, concise plain-language panel subtitle, minimal grid (usually none). Avoid equally sized three-column dashboard cards, oversized titles, gratuitous iconography, pictograms, giant arrows, pie/donut charts and outlines around every panel. A strong dominant evidence panel plus asymmetrical supporting panels is preferred.

All chemistry: RDKit MolDraw2DSVG / a vector-capable molecule renderer (not a generated image model); usual element colors for heteroatoms and charcoal/black carbon bonds; bond line widths scaled consistently, stereobonds preserved, no squeezed structures. Use atom mapping only for the magnified electronic region, not across entire large precursors. Draw model-recorded symbolic electron source→sink arrows in an understated contrasting stroke with *correct current-state mapped atom/bond references*. Do not manufacture mechanisms from endpoint bond differences. All molecule positions and original source IDs are recoverable from the panel manifest.

## Figure 2 (R1) — full-route utility and evidence inside the search

**Scientific claim to test:** *Executable reverse-electron-flow search can construct useful disconnections and complete synthesis routes within a finite search-work budget.* Do NOT redefine this as generic one-step Top-K.

Suggested asymmetric composition at 183 mm width:

```text
       FIGURE 2 · R1 (layout at final double-column width)
 ┌────────────────────────────────────────────────────────────┐
 │ a  Dominant budget–utility evidence (full width; ~36% h)   │
 │    URSA Expert | Drugs/Clinicals | RetroCast | PaRoutes     │
 │    all-target stock-AND Solved@B vs work, paired methods   │
 ├───────────────────────────────────┬────────────────────────┤
 │ b  Real electronic search tree    │ c  Tree morphology /    │
 │    dot-edge macro (~62% width)    │    efficiency (~34%)    │
 │    big focal highlighted path     │    unique successors,  │
 │    zoom anchor to key fork        │    terminal/pool/rank  │
 ├───────────────────────────────────┴────────────────────────┤
 │ d  Authentic molecular zoom: state → two recorded moves    │
 │    → distinct successor states → distinct terminal choices │
 │    source-map atoms, molecular structures, electronic arrow │
 └────────────────────────────────────────────────────────────┘
```

Actual panel count/layout may be adapted for molecule legibility; **a and d deserve room**, not four identical tiles. Figure 2(a) can be small multiples sharing x-axis if actual benchmark conditions are available, or a single aggregate across correctly specified cohorts. Labels must show method, frozen model, exact cohort n, and budget definition. Do **not** plot a complete 460-identity route curve when actual runs are only a small search-tree subset. For early data, output a separate truthful `fig2_r1_observed_tree_analysis` with (a) tree-size/depth/terminal distributions on observed n, (b) macro tree, (c) candidate and cost funnel and (d) chemical zoom. Preserve the formal a-panel for final real complete-route results.

**Panel b: tree grammar**
- A filled dot represents a **recorded committed chemical search state**, not an experiment-isolated intermediate. Depth from the root is action-search depth, **not reaction-route depth**.
- A directed line represents a model-proposed action; FLOW, IMPORT and FINISH use a restrained action-type encoding (prefer line dash/weight or small mark to avoid confounding status colors).
- Rejected proposals are **short open-ended stubs/crosses from the parent**, not fake child molecule states.
- State/terminal/reached-stock statuses encoded with combination of hollow/solid markers, grayscale/blue-teal and labels; true route-wide stock closure requires the separate outer AND tree.
- Node radius from `visit_count` **only if recorded MCTS visits exist**; otherwise use a constant size or recorded expansion count and label it exactly. Never invent pseudo-visits.
- Tree layout top-down or left-right deterministic; preserve hierarchy. For dense trees, visualize **all recorded nodes lightly** with a strong central path and contextual subtree opacity. If subtrees are collapsed for readability, include their actual count and full-tree totals. An inset makes a chosen node branch understandable.
- Transpositions/shared successors are **graph** edges; use clear subtle curved arcs or a separately flagged aggregate view. Do not draw a false simple tree to hide merges.

**Panel d: chemical microscope**
- Parent state shown as real RDKit molecular mixture (if many disconnected fragments, show the exact changing component at visual focus and reference the remaining inventory);
- two actual actions `a` and `a'` with exact electron-source/sink information, atom-map correspondence and executor before/after;
- final accepted states / terminal precursors, if those trajectories are actually observed, with the meaningful changed bond(s) accentuated;
- a small evidence ribbon: action accepted? unique successor? terminal route? matched precursor? chemical precedent Tier? **Every attribute has an actual record source**.
- Do not label `baseline rescue` without same target and work-matched baseline data.

**R1 metric insets:** `Solved@B` on full routes if known; `PoolRecall@B` vs frozen RankedTop1@B where reference and coherent pool exist; useful unique precursors per token; source-derived time cost; duplicate successor share. The narrative is "quality per work" rather than "largest possible tree".

## Figure 3 (R2) — compositional reuse and complex inverse chemistry

**Scientific claim to test:** *Reusable local electron-flow operations can support unfamiliar full transformations; complexity defines when the model succeeds or fails.* Maintain both the original reaction-level H2 and same-R1 full-route RetroCast context.

Suggested asymmetry:

```text
       FIGURE 3 · R2
 ┌───────────────────────────────────┬────────────────────────┐
 │ a  Dominant predictive chemistry  │ b  Small audited        │
 │    H2 full 27,104 / verified C1   │    primitive vs full    │
 │    bins + complexity, measured n  │    program accumulation │
 ├───────────────────────────────────┴────────────────────────┤
 │ c  Large genuine electron-flow chemistry trajectory       │
 │    product → S1 → S2 → ... → terminal precursors            │
 │    bond-change highlights, import species and local        │
 │    primitive source counts under the stated split          │
 └────────────────────────────────────────────────────────────┘
```

**Panel a priorities:** If authenticated H2-trained Direct/Open-Flow/Closed-Loop/Closed-Edit outputs exist, show first-candidate or Pass@K with the EXACT same 27,104 test IDs, true training exposure, correct base scale and paired uncertainty. Include primitive familiarity and real reaction-centre/ring/coupled-move complexity strata when row-level data exist. Separately, if the 160 RetroCast source-reference routes have full verified programs, plot C0/C1 (C2/Unknown separately), depth 2/3/4/5, stock-connected Solved@B under the same R1 predictions. These are **not** interchangeable populations; label them distinctly rather than combining in one apparent cohort.

**Panel b:** actual strict training signature accumulation against fractions 1/10/50/100%, with only observed source data. Terminal text annotations "604 primitive signatures" and "6,553 complete ordered signatures" refer to the specific train corpus and signature implementation. Do not extrapolate every point or claim 6,553 experimentally different mechanisms.

**Panel c:** choose an actual executed full program with reliable mapped reaction, trace and exposure ledger. For a C1 "unseen combination" caption, require a genuinely H2-exposure-controlled model. If a full-data model generated the example, call it an **executed complex program**, NOT strict C1 generalization. Clear labeled arrows (symbolic electron movement), only the modified atom neighborhood enlarged, one genuine terminal pair, optionally a modest reference disconnection. Link every illustrated move to `attempt_id` and saved state.

## Extended Data / Supplementary visual suite

**ED1 / Figure S: Whole-tree atlas** — one generous full-page large dot–edge tree/graph, no molecule thumbnails on every node; zoom-lens panels on selected forks and a mini legend "one state / one attempted electronic action / finish". Must preserve overall tree size and actual N. Export separate paper-PDF and optional local interactive HTML (click a node to see molecular structure, action, score and evidence; no internet required).

**ED2: Search morphology** — number of recorded nodes, accepted actions, branching/depth, terminal concentration and repeated successor distribution across all observed target trees. Prefer small-multiple empirical CDFs/ridge/dot distributions over five unrelated histograms.

**ED3: Ranking and search efficiency** — frozen candidate PoolRecall, NLL selected Top-1 and RankingGap at matched B, observed cost decomposition with missing-time categories marked as missing, work-to-first terminal/exact if logged. Graph versus graph-hybrid and pure Qwen score scales are never pooled into a made-up unified likelihood.

**ED4: Chemical evidence funnel** — symbolic executor acceptance → distinct terminal reactions → exact reference versus novel disconnections → independent exact/analogue precedent tiers; label candidate-, episode- and target-level denominators separately. Ambiguous cases are unresolved, not chemically invalid.

**ED5: Chemistry case gallery** — ideally a true search-based alternative, an independently source-supported non-reference full reaction, and a real high-score yet rejected/unproductive branch. Only qualified real data: no synthetic plausible structures or invented paper DOI.

## Specific visualization decisions Codex should enforce

- Stable deterministic layout seed and stable node IDs; tree position should not change across regeneration solely because parsing order changed.
- Avoid using *edge thickness as both score and frequency*. Reserve tree branch highlight for one selected path; other model statistics go to inset and caption.
- Empty/missing values are neutral grey or "N/A", not red "failed".
- Do not annotate 200 individual nodes with tiny text. One root label, one actual fork zoom, one terminal and one legend is preferable.
- Callouts should identify **why** a chosen electron-flow branch matters (new bond cleavage, different precursor set, valid source evidence) instead of restating a score without chemical meaning.
- Highlight reacting bonds in actual chemical drawings; maintain proportional bond lengths and consistent source heteroatom colors.
- Direct labels beat giant legends. Use caption to define `accepted`, `reference matched`, `supported analogue`, `complete stock-connected`; these are different outcomes.
- If a figure currently lacks authenticated route data, do not create a fancy template filled with manufactured results. Render genuine observed summary data into a truthful interim figure and log the missing final panel.

## Captions: journal-ready grammar (fill only from measured artifacts)

**Figure 2 —** *Executable electron-flow search and synthesis-route discovery.* (a) Stock-connected synthesis-route recovery as a function of total inference work, with all declared targets included. (b) An actual model-generated reverse electron-flow search tree highlighting the selected branch and relevant terminal states. (c) Distinct executable successors and candidate prioritization against recorded budget. (d) Enlargement of a model-owned chemical-state fork with source–sink electronic operations and terminal precursor differences.

**Figure 3 —** *Compositional reuse in complex reverse electronic transformations.* (a) Endpoint recovery under a genuinely composition-disjoint test or route-level complexity strata, with the correct population and training ancestry. (b) The number of distinct local and complete ordered electronic signatures across training-set fractions. (c) An authenticated model-generated electron-flow trajectory, with exact local operation types, altered bonds, state sequence and precursor projection.

Caption prose is a **template**, not evidence that each panel has already been measured. Once rendered, rewrite to name exact dataset, n, bootstrap unit, uncertainty, time/search budget, node/edge meanings, the source identity of the chemistry and what the cited literature does or does not confirm. Do not put placeholder `[TARGET]`, "expected 80%", or source-free reaction arrows in the exported publication PDF.
