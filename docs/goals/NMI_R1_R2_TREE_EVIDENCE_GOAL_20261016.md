# Codex Goal · NMI-ready R1/R2 search-tree evidence and chemistry figures
**Date:** 2026-10-16  
**Scope:** `MechET` code repository; consume **already generated** inverse electron-flow search-tree artifacts.  
**Paper authority:** [MechET-paper PR #29](https://github.com/wangyu-sd/MechET-paper/pull/29), especially [EXPERIMENT_MATRIX.md](https://github.com/wangyu-sd/MechET-paper/blob/paper/pr83-graph-qwen-executable-endpoint-rl-20261012/EXPERIMENT_MATRIX.md) and [R1/R2 full-route protocol](https://github.com/wangyu-sd/MechET-paper/blob/paper/pr83-graph-qwen-executable-endpoint-rl-20261012/editorial/R1_R2_COMPLETE_ROUTE_SCENARIOS_AND_COMPOSITION_PROTOCOL_20261013.md).  
**Visual authority:** [NMI layouts](NMI_R1_R2_FIGURE_DESIGN_20261016.md) and [style tokens](../../configs/figures/mechet_nmi_tree_style_v1.yaml).

## The user's actual deliverable

Build an **evidence-backed, editorially polished visualization and analysis pipeline**, ready to generate NMI main-paper figures from the **current electronic-search-tree outputs**. Not a slide deck or debugging dashboard. Every figure must be aesthetically restrained, precise, information-dense, and built from real chemistry: white background, low-saturation consistent colors, good scientific hierarchy, RDKit-rendered molecules and electron-flow arrows grounded in recorded operations, crisp vector export. Show both the macroscopic dot–edge search-tree structure and a microscope-like explanation of what an individual node, branch and terminal means.

This is an **analysis + visualization Codex goal**. It is NOT a request to re-train Qwen, re-run expensive MCTS, launch Taiji/A100/H20 jobs, invent new simulation data, relabel Graph–Qwen results, reinstate Recursive Self-Improvement (RSI), or redesign the paper's five-Result scientific storyline. The user wants *camera-ready* scientific presentation, not an onerous independent validation subproject. Produce substantial figures, not only QA reports or a plan.

**Research hypotheses that the figures test rather than assume:**
1. An executed inverse electron-flow state tree explores chemically distinguishable successors, not just duplicate strings or invalid steps.
2. Search produces useful terminal precursor choices and, where route data are available, stock-connected solutions under measured compute budgets.
3. Chemically meaningful complete transformations can be composed from local electronic operations; the evidence must respect the source program signatures and training-exposure boundaries.
4. A valid executor step is a symbolic check, not laboratory feasibility; independent reaction precedent applies to **completed precursor→product reactions**, not arbitrarily to each partial FLOW state.

## 0. Work on existing assets first; do not quietly invent missing tree topology

Read repo `AGENTS.md`, `PROJECT_MEMORY.md`, and the paper PR #29 documents above. Inspect actual search/tree/rollout/route artifacts on the accessible workspace, including manifests and saved outputs. Expected **possible** locations/lineages, NOT assertions that complete tree logs exist:

- `outputs/` and saved remote mirrored results (only paths truly present).
- `scripts/run_natural_language_value_search.py`: `search_unlabeled()`, `attempts`, `top_actions`, `terminal`, `Node`, `policy_score`.
- `scripts/eval_reliable_independent_episodes.py`: full independent trajectory `episodes`, frozen candidate mean-NLL ranking, generation Pass@K, process reliable candidates and explicit missing IDs.
- `src/mechet/proof_routes.py` and `scripts/run_syntheseus_online.py`: stock-connected *outer route* evidence, separate from the within-reaction electronic state tree.
- The user's potentially more recent uncommitted/MCTS search output folders. Detect them by recorded schema and provenance, **not** by assigning a new made-up format to every file.

Start with a real `input_inventory.json` of discovered paths and types, source hashes, target IDs, run/checkpoint IDs, complete versus partial tree lineage, available timing fields, stock/source manifests, action kinds and available model scores. If existing data expose only ranked endpoints / flattened attempts without parent IDs, do **not** fabricate a full branching tree. You may reconstruct an accepted action *chain* when exact predecessor states, actions and orders permit it, but mark it `reconstructed_path` versus `recorded_full_tree`. Create a clear missing-field status and still finish all feasible analyses/figures using real data. The final goal should ingest all **available** trees, not hand-pick the easiest cases or run a tiny experimental pilot.

## 1. Build an immutable normalized analysis layer

Implement an adapter-based loader so different real output schemas map to a shared `tree_id/target_id/run_id` registry; do not change existing training, search, prompt, executor or evaluator logic.

Recommended contract; adapt field names to actual records and record transformations:

- **Tree:** target product and benchmark ID, algorithm/runtime and source SHA, full/partial status, checkpoint and executor fingerprint, search work budget, stock ID, independent seeds, `has_gold_endpoint`, `has_complete_route`, `has_node_lineage`.
- **Node:** `node_id`, `tree_id`, recorded parent reference(s) if present, search depth, exact mapped chemical state, canonical unmapped stereochemical projection, imported fragment multiset, current map/address inventory, compressed model-visible history digest, terminal reason, available visits/value/policy fields, cost-so-far and presence mask for each optional field.
- **Attempted action edge:** `attempt_id`, `parent_node_id`, `child_node_id|null`, original action family `IMPORT/FLOW/FINISH`, full tool payload, FLOW source/sink/move list and coupled-move grouping, emitted tokens/logprobs, execution accepted/rejected and reason, before/after state when recorded, CPU/executor time where available. **Rejected actions do not create fictional successor states.** A multi-electron FLOW call is one transition edge, possibly with several component moves.
- **Terminal reaction:** true model-generated complete precursor set, full mixture versus structural projection, model ranking and source-reference exactness (only when reference is available), unique canonical reaction/disconnection key, independently replayed execution status where source allows, chemical support tiers with provenance.
- **Full route (if present):** product→precursor reaction edges, AND obligations, frozen buyables stock and every leaf membership; `stock_solved` is defined only when *all* leaves are in stock, not when one subbranch reaches stock.

Never merge chemically important states solely on stripped SMILES: stereochemistry, imports, maps, context/history, legality handles and remaining budgets can matter. For *display* an equivalent-molecule projection may aggregate branches, but retain original multiplicity and annotate what was compressed. All persisted analyses should retain primary row IDs, not just anonymous plotted coordinates.

Expected files under `outputs/paper/nmi_tree_evidence/<run_slug>/`:

`input_inventory.json`, `source_manifest.json`, `trees.parquet`, `nodes.parquet`, `attempted_edges.parquet`, `terminals.parquet`, `routes.parquet` (if data exist), `metrics.json`, `case_registry.csv`, `figure_manifest.json`, `figure_qa.json`. Use JSONL/CSV if Parquet dependency absent, but preserve typed semantics and schema manifest. Large raw predictions and expensive binary artifacts must not be committed to Git.

## 2. Preserve original R1 end-to-end metrics; add tree-level chemical quality, not replace R1

**R1 full planned context:**
- FlowER full endpoint **28,971** and mech-USPTO-31K endpoint **3,120** complete test IDs. Strict FlowER **28,967** is a separate electronic-program diagnostic view.
- **URSA Expert 100 + Drugs/Clinicals 100**, **RetroCast mkt-cnv-160 160** (40 each at reference depth 2/3/4/5, fixed buyables stock), **PaRoutes 100** fixed route-recovery targets, plus optional authenticated 30–50 chemist-directed constrained cases.
- RetroCast derives from PaRoutes: don't claim the cohorts are independent without an actual target-overlap audit. Proposed task sizes do not imply these files are already locally available.

Keep original R1 output contract: full-route `Solved@B`, published/reference-route recovery where ground truth exists, stock-connected route diversity, all-target denominators, work-to-first-solution and all-leaf stock closure. Record each real algorithm's wall time, Qwen tokens/forwards and executor calls, if present. Compare fixed Stage-II greedy/independent full K-trajectory sampling/state-tree search only when runs with **same IDs, stock, model and defined work budgets** exist. No forced comparison to absent baselines.

For current **already-run inference trees**, add:
- Per-target morphology: recorded nodes, attempts, accepted FLOW edges, unique successor states, branching factor at actually expanded nodes, tree depth, terminal count, no-op/cycle/rejection rate, transposition/duplicate-successor rate, visit distribution (only if MCTS visits actually logged), path length and terminal class mix.
- Candidate generation vs selection: first-candidate exact, generation-order Pass@K (if K genuinely independent full episodes), `PoolRecall@B` = any canonical-unique **recorded** precursor hit in actually searched terminal pool, ranked Top-1/5/10 under a documented reference-independent ranking rule, `RankingGap@B = PoolRecall@B - RankedTop1@B` under the **same pool and budget**, MRR, unique executable candidate count. Never call a pool recall ceiling a deployable oracle ranker; if reference unavailable mark exact/rank gap unavailable.
- Actual search efficiency: terminal precursors per 100 model calls, unique accepted successor / attempted action, chemically interesting accepted branches per 1,000 generated tokens, first terminal/first endpoint hit observed cost; wall-time median/P90, CPU executor vs policy generation vs value/ranking vs scheduling/serialization only if measured. No arithmetic decomposition from missing timing fields.
- If complete matched route logs exist: discovered **new disconnection** defined by structural precursor set and mapped reaction centre rather than differently ordered electron tokens; stock-constrained rescue versus baseline, opposite-direction losses and net difference over original target cohort. Reference-different is NOT chemically invalid.

Use paired target bootstrap (default 2,000 samples) where matched outputs permit; report n **for each real output universe**, missing targets as failed only for a genuinely declared complete benchmark run, and provenance/version/seed alongside each metric. For incomplete artifact subsets, say `observed trees n=...` instead of silently promoting to n=28,971, 160, etc.

## 3. Preserve original R2 compositional and complexity metrics

**Do not rename R2 into generic tree visualization**. Original Results:
- strict FlowER train 257,167 reactions, measured **604 local electron-operation signatures / 6,553 ordered complete signatures**; known limitations: local signature abstraction and complete-signature hashing ignore fragment-import identity and can depend on within-event ordering.
- independent H2 composition-disjoint split **223,863 train / 6,200 valid / 27,104 test**. Official H2 predictive claims require *H2-trained* model source exposure; a model trained on the full FlowER pool is not eligible for strict H2-unseen-composition claims.
- On full-route R2, reuse the **same 160 RetroCast IDs and immutable R1 tree/route outputs** with independent reference-route classification `C0` (seen complete programs), `C1` (unseen whole combination; all primitives seen ≥5 times), `C2` (genuinely unseen primitive), `Unknown` (reference mechanism insufficient). Do not infer an electron program from plain atom-map bond changes. All 160 count in overall Solved@B.

Current-tree analysis should compute observed (where the data permit): source/sink move types, moved-electron count, linked/uncoupled FLOW moves, accepted program length, import count, mapped reacting-atom count, changed bonds, ring opening/closure, terminal disconnection diversity, primitive support frequency, **ordered complete program novelty under a specifically named training exposure**. Provide a `not_establishable` flag for missing source reference, unsupported executor chemistry, unknown training ancestry or absent whole programs. Novel generated program hash alone ≠ literature-supported reaction.

Main R2 outcomes must keep **H2 complete full-test endpoint comparison** (Direct/Open-Flow/Closed-Loop/Closed-Edit only if corresponding authenticated H2 models/results exist), verified RetroCast C0/C1 complexity-conditioned `Solved@B` (with n), corpus signature accumulation, and a model-owned chemistry-accurate reaction example. Missing H2/route observations remain `PENDING`, not estimated from current tree plots.

## 4. Chemical validity: three separate evidence levels, usable now

**L1 formal execution** (automatic): per proposed action accepted/rejected, no-op/cycle, chemically valid symbol/atom addresses, commitment and terminal replay; executor acceptance is never labeled experiment-valid.

**L2 chemical plausibility/compatibility** (automatic descriptive features): local bond changes, valence/charge/aromaticity, reaction centre, ring and fragment changes, identifiable named transformation/motif and support among frozen training/reference reactions, optional independent forward/round-trip model agreement if saved or available offline. Prefer **interpretable feature profiles, distributions and stratification** to an arbitrary single `chemical_plausibility = 0.83` score. Rare reaction centres and unusual ring edits are not automatically impossible; report them as rarity/context descriptors, not false negative chemistry labels.

**L3 independent reaction precedent** (candidate-level source retrieval; may initially be partial): for **complete product + model-generated precursor sets** only, search available paper/patent reaction sources with exact structural and stereo provenance, then related analogues. Keep fields `source_type, DOI/patent_id, reaction_id, matched_atoms, product_match, precursor_match, stereo_match, source_overlap, retrieval_query, evidence_status`. Classify:
- Tier A: independently recorded **same precursor→product** chemistry with appropriate substrate/structure and mapping identity;
- Tier B: relevant experimentally recorded **analogue** chemistry, explicitly not identical substrate validation;
- Tier C: unresolved/no independently confirmed source (absence of retrieval ≠ impossible chemistry);
- Tier D: independently documented *conflicting* chemistry (requires actual contradiction evidence, **never** assign D merely because no paper was found).
- `retrieved_unreviewed` must not be silently called verified Tier A/B. When exact source structures support automatic matching, retain original source and duplicate/training overlap flags. Review a fixed representative, chemically diverse pool if feasible rather than block all plots on new annotation staffing.

**CRITICAL:** internal partial electronic-state nodes may carry source-derived `local motif support` / `operation familiarity`, but **cannot each receive a full-reaction literature Tier A**. A complete reaction may have literature precedent without proving that the model's exact symbolic intermediate steps are the physically observed mechanism. This distinction must be obvious in figure captions.

Chemistry-evidence funnel should distinguish: attempted actions → accepted distinct successor states → terminal complete reactions → reference-matched / reference-different → independently source-supported whole reactions. Use denominator labels and avoid treating nested per-action and per-target rates as the same probability.

## 5. Figures: final journal composition, not a debugging plot set

Implement exact editorial storyboards in [NMI_R1_R2_FIGURE_DESIGN_20261016.md](NMI_R1_R2_FIGURE_DESIGN_20261016.md). Figure 2 and Figure 3 are **reserved R1/R2 main-paper slots** in paper PR #29; the *existing* Figure 1–6 story and R3/R4/R5 remain unchanged. Original 16 work packages are retained; do NOT add a seventh main display item. NMI currently specifies **up to six figures/tables combined** for Articles: https://www.nature.com/natmachintell/content . The existing six-figure-plus-Table-1 layout must eventually resolve one extra display item (for example move full numeric Table 1 to Supplementary), but **do not silently delete existing Table 1** as part of this figures goal.

Requested visual outputs:
1. **Macro tree atlas**, a large static point–edge plot of an actual searched chemical-state tree/forest, visually legible at journal two-column size. Show whole-tree scale and true exploration shape, hierarchy by actual depth, bounded node-size/density encoding, branch type, terminal location, path selection; illustrate which elements were collapsed, if any.
2. **Chemical-node microscope**: visually magnify one real parent chemical state and two **actual** branching actions; use RDKit vector molecular structures, atom/bond highlights, actual IMPORT/FLOW arrows and terminal precursor structures. Mark one successful or supported path and one rejected, alternative or censored branch only if such actual records exist. Provide a small yet unmistakable `node = symbolic molecular state; edge = recorded attempted action; terminal = executed precursor` legend. No generic textbook reaction masquerading as MechET output.
3. **Real quantitative R1 Figure 2** with dominant matched-`Solved@B` / cost evidence when present, plus strong actual tree/branch example and a compact efficiency/coverage panel. If only current incomplete tree evidence exists, generate a separate `figure2_r1_observed_tree_analysis.pdf` grounded in observed n, and do **not** place a fabricated complete-benchmark curve into `figure2_r1_final.pdf`.
4. **Real R2 Figure 3** retaining H2, exact 604/6,553 corpus evidence, verified composition/complexity strata where available and actual electron-flow chemistry pathway. Again, missing authenticated model cohorts must be explicitly available/unavailable in an evidence ledger; do not silently infer novelty from full-data trained models.
5. **Supplementary / Extended Data**: full tree atlas (larger than the main inset), morphology and density distributions, quality funnel, executor error taxonomy, score/RankGap, runtime decomposition if timed, and at least one chemically revealing negative/ambiguous case. An optional local interactive HTML/SVG tree inspector may offer click-to-open node chemistry, but is NOT a replacement for journal-ready static PDFs.

**Real-case selection**: freeze candidate pool first. Prefer a `rescue` if a genuine matched baseline failure and valid successful MechET route are both recorded; otherwise label the case `alternative-branch exploration`, NOT `baseline rescue`. Other eligible cases: executable non-reference terminal with independently supported Tier A/B evidence, C1 program chemistry **only** under verified training exposure, and a high-likelihood but chemically unproductive/rejected branch illustrating decision consequences. Record the inclusion/selection rule and source IDs in `case_registry.csv`. Do not choose unrelated public molecule images.

## 6. Required implementation files and interface

Codex should implement modular, tested code, not just create documents:

- `src/mechet/analysis/tree_records.py` or equivalent: schema adapters, strict provenance ledger, tree/path distinction, canonical chemical states, before/after trace capture.
- `src/mechet/analysis/tree_metrics.py`: R1 candidate/route/budget/efficiency metrics and R2 composition/complexity analysis, explicit `available/not_available` statuses, no denominator leaks.
- `src/mechet/analysis/chemical_evidence.py`: executor quality, local motif and source-record evidence schema, no model-generated Tier A.
- `scripts/analyze_nmi_tree_evidence.py`: CLI to ingest **existing** `--input-roots` / optional manifests, normalize, compute metrics and select observed cases. Support resumed deterministic processing and a final exact input coverage report.
- `scripts/render_nmi_tree_figures.py`: builds publication vector PDF/SVG, 300-dpi preview PNG, layouts from the authored style YAML. RDKit MolDraw2DSVG for real structures; Matplotlib for quantitative panels; deterministic stable NetworkX/Graphviz layouts where available; record all plotted input IDs.
- `tests/test_nmi_tree_evidence.py` and figure smoke-free deterministic unit fixture checks: per-edge accepted vs rejected, coupled FLOW, duplicated stereochemical states, history-dependent state equivalence, structural vs full-mixture endpoint, incomplete tree logs, all-leaf stock semantics, ranking on same candidate pool, R2 signature exposure, file manifests.
- `outputs/paper/nmi_tree_evidence/<run_slug>/figures/`: `fig2_r1_observed_tree_analysis.pdf` and `.svg`, `fig3_r2_observed_chemistry.pdf` and `.svg`, `tree_atlas.pdf`, `node_microscope.pdf`, `extended_data_*.pdf`, `case_*.pdf`, matching preview PNGs, plus manifest/caption draft files.

Final figure filenames suitable for `MechET-paper` PR #29 can be exported once their **authentic** final supporting cohorts exist. Do not independently push into paper `main` or override its current placeholders without an explicitly approved paper-PR integration. Do not add arbitrary new task generation or Qwen training.

Suggested **new CLI interface** (implement rather than assume already present):
```bash
python scripts/analyze_nmi_tree_evidence.py \
  --input-roots /path/to/existing/search_outputs \
  --output-root outputs/paper/nmi_tree_evidence/current \
  --style configs/figures/mechet_nmi_tree_style_v1.yaml

python scripts/render_nmi_tree_figures.py \
  --analysis-root outputs/paper/nmi_tree_evidence/current \
  --style configs/figures/mechet_nmi_tree_style_v1.yaml
```
Discovery should accept multiple existing run roots and never hardcode user-specific absolute paths. If inference files are absent in the Codex environment, finish the script/config/tests and provide an exact missing-path manifest. **Do not report completed science plots or infer fake node counts.**

## 7. Deliverable and quality definition ("done")

- [ ] Inventory ALL accessible real inference tree artifacts, show input ID/source/checkpoint/field coverage and exact source hashes; preserve originals.
- [ ] Compute reproducible R1/R2-compatible observed metrics (including morphology, ranking/Pool Recall, executor/literature evidence and cost where present), exact Ns/statuses, target-paired intervals if comparable.
- [ ] Produce **beautiful, non-placeholder, chemistry-grounded** multi-panel figures with consistent low-saturation style; a dot–edge whole-tree atlas AND an actual RDKit chemical-node magnification are essential.
- [ ] Produce Figure 2 / Figure 3 `observed` versions without overwriting unmeasured formal benchmarks. Preserve original R1 full-route and R2 H2/RetroCast metrics in the editorial plan.
- [ ] Generate `figure_manifest.json` linking every displayed panel and showcased molecule to real node/edge/terminal/route IDs, source data, transformation and run hashes; include concise Nature-style figure legends and a brief result narrative **only for measured findings**.
- [ ] Verify vector outputs, font embedding and size; inspect previews at **183 mm publication width** for clipped bonds, text overlap, colormaps, legend/order and visual hierarchy. Fix real presentation issues, not just matplotlib runtime errors.
- [ ] Document deterministic regeneration CLI, dependencies and exact output files; report explicitly unavailable analyses without inventing values. Commit code/specs, not multi-gigabyte predictions.

**Visual standard:** restrained, asymmetric, Nature/NMI-style figures. Use a consistent palette and logical visual hierarchy, no default pastel cards, gradients, decorative arrows, random algorithm icons or noisy legends. Scientific legibility over density where they conflict; data should carry the visual.

**No RSI; no new GPU search or training; no result fabrication; no formulaic compliance-only response.** The intended outcome is scientific figures worth placing in the NMI Results, plus reusable analysis code for future full R1/R2 runs.
