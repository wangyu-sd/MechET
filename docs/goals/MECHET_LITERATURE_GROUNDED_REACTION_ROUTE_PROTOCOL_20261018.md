# MechET NMI · Literature-Grounded Reaction and Route Verification Protocol (LGRV-v1)
**Active scientific protocol · 18 October 2026**  
**Authority:** This protocol supersedes **the old "reference endpoint / NLL Top-K ranking as a main evaluation goal" sections** of [Codex R1/R2 Goal](NMI_R1_R2_MATCHED_BASELINE_GOAL_20261017.md), [figures](NMI_R1_R2_MATCHED_FIGURES_20261017.md), [direct prompt](NMI_R1_R2_MATCHED_BASELINES_CODEX_PROMPT_20261017.md) and the comparative YAML. It does **not** cancel R1/R2, erase their original discovery/stock-close metrics, remove the historical FlowER/H2 experiments, reinstate RSI, or alter R3/R4/R5.

## 0. Central scientific question and precise meaning of a "step"

**Research question:** *Do the complete synthetic reaction transformations constructed by MechET's inverse electron-flow search have identifiable, independent experimental literature/patent precedent, and can that evidence be assembled into chemically interpretable, stock-connected multistep routes? How does this compare to frozen-actor Greedy and other available same-budget planners?*

We must distinguish TWO ontologies:
1. **Synthetic reaction step**: complete actual model-executed precursor set `R_1 + ... + R_k → P` corresponding to one finalized product expansion/route edge. **This** is the unit of independent reaction literature search, structural precedent and whole-route evidence.
2. **Internal symbolic electron operation**: model-owned IMPORT/FLOW/FINISH at an executor state `S_t → S_{t+1}`. Its graph/rule legality is mechanical and its *mechanistic consistency* may be supported by reaction-type chemistry, but ordinary reaction papers/patents **do not experimentally verify every proposed symbolic arrow or intermediate**. Never assign exact experimental reaction evidence to each internal FLOW step.

The old question "Did we predict the single recorded reference precursor?" is only a **separately labeled legacy recovery benchmark**, not a test of correctness of a model-generated alternative. The current source-reported `rxn_prod_min` 300/3,120 endpoint discovery (PR #88) is historical search-process metadata and must not be the headline chemistry metric.

**Scientific display thesis:** `executed electronic construction → structurally distinct disconnections → paper/patent-grounded complete reactions → credible stock-connected synthesis choices`. Actual sourced evidence, not aesthetic novelty or a black-box "plausibility score", drives the main figures.

## 1. Source inventory and immutable eligible outputs (no new GPU runs)

- Latest source progress [PR #88](https://github.com/wangyu-sd/MechET/pull/88), `docs/results/EXPERIMENT_PROGRESS_20261017.md`: **200/460** saved R1 route target reports; **199** nonzero complete frozen stock-AND closures, **one** already-stock target. Four cohorts: URSA Expert 100, URSA Drugs 100, RetroCast `mkt-cnv-160` 160, PaRoutes 100. Pending 260 are not completed reactions. Two older full-460 Greedy runs are described and linked by [PR #87](https://github.com/wangyu-sd/MechET/pull/87).
- Latest single-reaction source [PR #88](https://github.com/wangyu-sd/MechET/pull/88): all **3,120/3,120** target IDs have a completed **10 TOTAL trajectory-attempt budget per target**, and **3,022** targets contain at least one model-generated terminal precursor. Analyze the *actual frozen structural precursor sets* for chemical evidence independent of reference hits. Do not regress to the PR #87 **1,711-ID** historical partial snapshot when full source records exist.
- Historical PR #87 chemistry has precisely **one** sourced Cabozantinib free-base acid-chloride/aniline **complete transformation** matching [US20160022662A1](https://patents.google.com/patent/US20160022662A1/en). The patent does not establish all symbolic proton transfers, actor's auxiliary mixture or whole route. This is a seed **example**, not a population validation rate.
- Existing R2 MCTS observed contexts contain **3,672 training-unseen ordinary whole program signatures** among 7,082 terminal contexts in the earlier 171-target export, versus **3,392 seen** and **18 unavailable/nonordinary**. Recompute **unique complete chemical reactions** and target-level counts with the latest 200-target export, and retain correct Stage-II actor training-exposure SHA. Never call full-data actor composition novelty the strict H2 generalization result.
- Preserve exact source run IDs, old/new snapshot versions, model/checkpoint SHA, frozen stock, executor version, original timing and attempted actions, authors' figure source manifests and a target/route/reaction canonical ID. Verify staged branches PR #87/#88/#89 are not silently merged as though code and data were from one commit.

Input/outputs are **read-only**. Evaluate *all fixed eligible available route reaction edges* and *all unique eligible single-step terminal precursor sets*. Index the candidate pool before retrieving any paper; keep every missing/rejected/unmatched case. Deduplicate evidence retrieval by chemical reaction key but retain per-route/per-target occurrences for denominators.

## 2. Per-step chemical standardization and two-level verification

For each completed model-generated reaction edge record:
- `target_id, route_id, reaction_edge_id, actor_sha, executor_sha, stock_sha, source_report_sha, reaction_product, structural_precursors, auxiliary_imports, electron_program_source_pointer, terminal_record_id`.
- Convert the prediction to **forward orientation** `structural_precursors >> product`; use RDKit to canonicalize isotopes, charge, salts, tautomer/protecting group and isomeric stereo with a *documented, conservative policy*. Preserve raw original SMILES. Do not silently neutralize/desalt/strip stereo or change reaction substrate membership. Agents/catalysts/solvent/proton sources have **roles**, not automatic additional structural precursors.
- Create `reaction_key` (stereospecific multi-precursor set + product), `reaction_family_key` (mapped reaction-centre bond-edit signature and role/type when defensible), and `route_key` (whole AND-route reaction graph, not mere text/path). If mapping unavailable or inconsistent, record `mapping_unknown` and use source-supported atom matching *without manufacturing arrows*.
- **L1 formal correctness**: original accepted/rejected executor operations, mapped-handle consistency, valence/charge and observed endpoint. L1 does not confer A/B chemical precedent.
- **L2 chemical plausibility context**: reaction-centre atoms/bonds, leaving group/nucleophile/electrophile patterns when independently derivable, ring reorganization, coupling/heterocycle/protection class, ancillary reagent compatibility, steric/synthetic handles and forward/round-trip agreement *if genuinely available*. Use *descriptive features*, not an arbitrary "chemist probability" assigned to exotic chemistry.
- **L3 external paper/patent precedent**: independently recorded reaction *structures* and experimental context; quantify documented support separately from symbol validity and reference equality.
- **Mechanistic assessment**: optional second typed report `mechanistic_consistency = formal_only | conventional_mechanism_compatible | source_explicit_mechanism_compatible | unresolved | documented_conflict`. A typical experimental patent confirming an amide bond does **not** establish the model's exact arrow order. Only an explicit source reaction-mechanism figure/path with reconciled mapping can provide source-explicit support; never claim physical elementary intermediates from executor states.

## 3. Retrieval pipeline: source-anchored structure search, then primary evidence extraction

**Three complementary retrieval sources; freeze versions and licensing first:**

**(i) Open Reaction Database (ORD)** — read documented `Reaction.inputs`, `outcomes`, `conditions`, `provenance`, identifiers and actual reaction ID, optionally `DOI` and lab measurements. It stores structured reactant/agent/condition/procedure information; confirm provenance for each row, including whether it itself originated in USPTO text. Sources: https://docs.open-reaction-database.org/en/stable/schema.html and https://github.com/Open-Reaction-Database/ord-data

**(ii) Open US patent reaction extracts** — chemical reactions from US patents (Lowe/NextMove public data), including reaction SMILES/CML and source patent identifiers/paragraphs where present: https://figshare.com/articles/dataset/Chemical_reactions_from_US_patents_1976-Sep2016_/5104873 . Patent reaction records require de-duplicating **patent family, application vs grant and repeated paragraphs**. An extracted reaction `SMILES` is a retrieval candidate, not by itself evidence of an experimentally **performed** preparation.

**(iii) Primary patent / paper full text and supporting information** — after structural retrieval, inspect the original patent **specific preparation/example** or academic experimental section/SI. Save `patent_id/DOI, source_url, section_or_example/paragraph locator, local quoted evidence <= permitted limits, reported reactants/product, reagent/catalyst, conditions, yield or actual product characterization if available`. For source discovery use Google Patents and DOI/Crossref/OpenAlex, but **bibliographic keyword match or text similarity alone never qualifies** as structure-verified chemical evidence.

**Retrieval stages:**
1. **Exact structure-key lookup** using predicted `R_1...R_k >> P` and atom/stereo/roles, from frozen structural reaction indices. Search over unique reaction keys in bulk; cache and record source/dataset hash.
2. **Analog search** for the correct *bond edits + local environment*, with predeclared reaction-center radius (e.g. 1/2 bonds) and peripheral structure similarity, actual leaving-group/reaction-family compatibility and experimental source. Atom/bond-edit match is a *candidate*, not proof that all substrates behave identically. Freeze similarity rules using independent source-positive and hard-negative examples **before** analyzing MechET test results. Retain exact value and radius rather than a secretly adjusted similarity threshold.
3. **Primary-source verification:** resolve 1–5 best structurally admissible exact/analog matches per distinct generated reaction and confirm the original cited specific step; preserve source excerpt locator and reported compounds. A generative LLM may help extract terms from the passage but cannot supply DOI, reaction structures, experimental completion or a chemistry grade without actual source payload and independent structure checks.
4. **Document source overlap and retrieval coverage**: check whether the patent reaction likely occurs in the MechET/FlowER source training corpus, exact reaction source duplication, patent family and post-publication dates. This affects **novelty independence**, not the literal existence of a precedent. A source overlap unknown remains `unknown`, not asserted disjoint.
5. If no index is accessible in the worker environment, implement downloader/adapter and document legitimate official URLs, then still evaluate all accessible sources; do not pretend public web keyword search covered the entire reaction literature.

**Do not download subscription-only Reaxys/SciFinder data without rights.** Licensed provider results can be additional evidence only when legally available; public-source baseline should be reproducible without subscriptions.

## 4. Reaction-level grading with explicit evidence states

Grade each *complete, structurally deduplicated* model reaction on one scale, alongside an independent `verification_status`:

| Tier | Required positive evidence | Scientific interpretation |
|---|---|---|
| **A — exact experimental reaction precedent** | Original paper/patent performed example identifies the same essential *structural substrates and product* and compatible stereo, reaction center and source locator. Agents/conditions can be reported separately; incomplete reagent inventory does not get silently repaired. | This **transformation** is experimentally documented. Not proof the actor's exact internal electron arrows, conditions or adjacent route steps are correct. |
| **B — strong experimental analogue precedent** | Performed related reaction in a primary source, chemically compatible reaction center, relevant neighboring atom environments and functional group/condition class, with actual substrate difference recorded. | A close analogous transformation is known; predicted specific substrates remain an inference, not directly validated. |
| **C — unresolved** | Missing accessible primary source; only computational agreement; patent claim with no carried-out example; text-only match; weak analog/ambiguous atom roles; no match after bounded retrieval. | **Insufficient supporting evidence**, not impossible chemistry. |
| **D — documented chemical contradiction** | Explicit external chemistry evidence showing a concrete incompatible identity/reaction claim; source and contradiction class must be given. | A particular claim is demonstrably conflicting. No paper found does NOT qualify. |

Additionally `retrieval_status = indexed_candidate | source_retrieved | structural_checked | experimental_example_checked | unresolved` and `verification_status = machine_structural_match | source_passage_checked | source_human_reviewed | unresolved`. **Never report a retrieved, unverified source as `verified Tier A`**. In machine-only unreviewed cohorts call the measure `automated source-supported candidate` and keep stricter manually inspected showcase results separate. A/B source evidence may be logged without employing a staffed annotator, but require direct primary-source text and structural match for `source_passage_checked`.

For each reaction save *all* matched source candidates, selected strongest evidence, alternative conflicting records and search queries. Do not hide unsuccessful retrievals.

**Example current evidence:** Cabozantinib free-base structural **amide coupling** matches an explicit patent preparation `US20160022662A1`; source reported `K2CO3, aqueous THF`, whereas some actor auxiliary steps differ. This is the *coupling transformation only*. It is a strong seed for a high-quality chemically grounded NMI case, not evidence that the whole route has independently confirmed chemistry.

## 5. Route-level evidence: the primary R1 outcome rather than reference recovery

Let a frozen generated complete route be `r = (E_r, L_r)`: its set of actual precursor→product chemical reaction edges `E_r` and frozen-stock leaves `L_r`. Verify **AND** stock closure first, then attach step evidence to *every unique reaction edge*. All target-level denominators remain fixed at **460** for main R1, with separate 200 observed-report population/status; zero-step already-stock target excluded from newly generated route success.

Required metrics, all computed on **the whole original target population** with exact `n` and missing states:

- **StockSolved@B** — at least one model-generated nonzero route reaches all frozen-stock leaves (already tracked as 199/460 lower bound in 17 Oct snapshot).
- **AnyExactStepPrecedent@B** — target has ≥1 stock-closed generated route containing ≥1 **Tier A** reaction edge. Shows anchored known chemistry; **not** all-step route verification.
- **AllStepExactPrecedentSolved@B** — target has ≥1 stock-closed route where **every essential complete reaction edge is confirmed Tier A**.
- **PrecedentCoveredSolved@B** — target has ≥1 stock-closed route where every essential edge is Tier A or source-checked Tier B (explicitly "precedent-covered / analogue-supported", NOT laboratory-proven executable route).
- **RouteStepEvidenceCoverage** — for each frozen complete route, `# Tier A/B evidenced essential edges / # all essential distinct edges`; report population distribution, mean/median and fully covered fraction, plus retrieval-only/unknown portions. Distinguish edge occurrences, deduplicated edges, routes and unique targets.
- **LiteratureSupportedDisconnection@B** — count distinct product→canonical precursor **chemical disconnections** with independent A/B evidence. Main useful discovery for non-reference reactions; record shared supported reactions versus baseline, MCTS-only distinct supported reaction edges and full-route target wins.
- **SupportedDiscoveryEfficiency** — targets with `PrecedentCoveredSolved` and distinct A/B supported disconnections per million *actual generation tokens* / per executor work / actual recorded elapsed time, respecting shared cache and mixed cost units.
- **Full-route evidence bottleneck** — for each route show missing C (unknown) steps explicitly; route evidence is *not* an average of a few well-supported edges. A route's total source strength is limited by its least supported essential reaction.
- Optional: explicit source-reported yields and reaction conditions distribution, WITHOUT inventing yield for a predicted unperformed specific reaction.

**Avoid attrition:** show `460 target universe → 200 saved reports → 199 nonzero stock AND closures → n with any exact/analogue supported reaction → n with all essential steps supported` in figures, once actually counted. No fabricated A/B n values. Report *both* full-460 lower-bound rate and rate conditioned on fully inspected routes, with different labels. Do not silently mark uninspected routes invalid.

## 6. R2: test whether electronic compositional novelty produces documented chemistry

On the **same** R1 route/tree outputs and actor exposure:
- Map model-generated `unseen ordered whole electronic program` (per frozen true training exposure, not a database reference split) to the canonical **complete reaction**, changed bond/reaction center, distinct precursor set, whether it occurs on any all-stock route and actual **Tier A/B experimental precedent**.
- Compute `UnseenProgramToSupportedReaction` by target and unique reaction; `NovelProgramSupportedDisconnection`; `NovelProgramPrecedentCoveredRoute` (if every route edge evidence exists); and `SupportedNovelChemistry per work`. Report seen-vs-unseen stratified source coverage and chemical complexity (ring change, reacting atom/bond count, coupled FLOW).
- **Baseline Greedy**: same target, stock, exact evidence corpus/source rules and cap per distinct reaction; show generated-program novelty AND actual sourced distinct reaction transformations. For Direct precursor-only methods, compare `source-supported chemical reaction/route` only, not inaccessible electron primitive signatures.
- The 160 RetroCast *reference* C0/C1/C2 composition labels are **Unknown** in current evidence. **A generated novel program with Tier A exact chemistry does not make that benchmark reference C1.** Historical strict H2 (223,863/6,200/27,104) is separate; keep source Qwen3-8B lineage intact if/when available.
- Support level must derive from **completed reaction** evidence, not a patent claiming a particular unobservable symbolic microstate.

## 7. Baseline fairness: evaluate chemical support rather than reference Top-K

Compare frozen Qwen3-0.6B Stage-II Greedy A/B, independent full electronic trajectories if available, and MCTS on **the exact same target IDs**. Preserve 460 full set, latest 200 saved MCTS snapshot, route budgets and historical PR #87's 171 snapshot separately. If an independent-K / complete-reaction candidate planner is absent, use existing data first and leave clearly identified missing source output; do not rerun unbounded MCTS or launch unauthorized GPU jobs.

**Run the identical literature retrieval and verification protocol** on all baseline and MechET complete reaction steps:
- Same *reaction structure* normalization, frozen source corpora, patent-family/source de-duplication, retrieval/query budget per unique chemical reaction and per target, Tier assignment, source validation. MCTS cannot claim an advantage simply because it produced 100× more candidates and got 100× more literature hits. Report **per generated target** and **per work unit** efficiency and fixed candidate-budget results when feasible.
- Create target paired 2×2 contingency for `StockSolved` and separate 2×2 for `PrecedentCoveredSolved`. Also `MCTS-only A/B-supported disconnection` vs Greedy and the opposite direction, without requiring the source chemistry to be an exact reference.
- Use 2000 target-paired bootstrap replicates and 95% CIs for genuinely comparable source/budget conditions. When archived work budgets differ, call comparisons `recorded-run descriptive` and reserve matched-compute causal claims for actual matched data.
- **Single-step 31k**: evaluate source evidence on *all available canonical-unique terminal complete reactions* from the completed 3,120 target/10-attempt output, then report `AnySourceSupportedReaction@10 attempts`, `ExactSourceSupported@10`, `DistinctSupportedDisconnections/target`, source-reviewed fraction, and confirmed Tier A/B versus Greedy under same evidence rules. **No NLL Top-K or reference-endpoint matching in main figures.**
- A target-specific result can have source support despite not matching the archived GT reactants. Count it positively under LGRV even if old benchmark calls it wrong.

## 8. R3 and the five-Results paper: preserve work scope

R1: stock-connected practical routes **and chemical-evidence coverage** on URSA/RetroCast/PaRoutes, with matched Greedy/independent-sampling comparisons and one detailed source-linked route; main Figure 2.

R2: seen/unseen electronic program composition → actual chemically distinct reaction → **primary-source-backed precedent** and route utility; same R1 route population and independently authenticated historical H2 diagnostics; main Figure 3.

R3: the existing dedicated **Chemical Diversity Beyond Reference Precursors** section performs the *deep population-level* literature evidence strata A/B/C/D with structural case gallery and independent source metadata; avoid duplicating the exact same big funnel three times. Source evaluation pipeline is shared by R1/R2/R3 but denominators differ transparently.

R4: retain the separate once-endpoint-aligned training/actual error recovery experiment if evidence exists, but old numeric reference endpoints do not replace the new **R1/R2** chemical-evidence task.

R5: reaction-record auditing and repair remains a separate downstream application, not this literature evaluator.

Keep all 16 historical paper work packages as supplementary/historical when relevant. This protocol is a **new priority for the currently executed R1/R2 experiments**, not permission to delete paper content.

## 9. Required Codex implementation and deliverables

Implement tested modular analysis against actual PR #87's typed source tables, PR #88's latest progress and original raw source snapshots. Suggested files:

- `src/mechet/analysis/literature_reaction_keys.py`: canonical structural vs auxiliary components, exact reaction structural key, role/stereo handling, reaction-centre/mapped compatibility.
- `src/mechet/analysis/literature_index.py`: offline ORD/USPTO structured source indexing (CSV/SMILES/protobuf adapters where real source exists), patent family/source dedup and versioning, deterministic exact/analogue retrieval with source IDs and flags.
- `src/mechet/analysis/literature_evidence.py`: original-paper/patent source citation record, structural-match checker, four tiers + verification status, cache, retrieval cost, source-overlap.
- `src/mechet/analysis/literature_route_metrics.py`: step-to-complete-route AND tree join, literature evidence aggregation, original target denominators, baseline paired target analyses and unique novel disconnections.
- `src/mechet/analysis/mechanism_consistency.py`: optional independent comparison of recorded symbolic arrow type against explicitly supported mechanistic classes; keep source-explicit mechanism separate from chemistry net transformation.
- `scripts/verify_nmi_literature_reactions.py`: CLI:
  `--tree-analysis-root <PR87 analysis> --progress-snapshot <PR88 JSON> --source-reaction-index <ORD/USPTO> --route-assets-manifest <frozen stock> --output-root <new versioned output> --retrieval-config <frozen YAML>`.
- `scripts/render_nmi_literature_figures.py`: actual RDKit reaction structures, patent example/DOI source callouts, route-by-step support schematic, same-target MCTS vs Greedy comparative stats, English NMI captions, editable 183-mm SVG/PDF and PNG 300dpi.
- `tests/test_nmi_literature_protocol.py`: exact-vs-analogue vs same product with wrong reactants; absent stereo; salts, missing reactant agents, mapped reaction-center identity, duplicate patents, claimed-only vs actually performed examples, retrieval hit without primary passage; no hit -> C not D; route 2/3/5 all-edge support; route zero-step exclusion; 200/460 incomplete vs full 460, same-ID comparison and no reference leakage; R2 seen/unseen exposure/scope.

Versioned output `outputs/paper/nmi_literature_evidence/<run_slug>/`:
- `source_corpus_manifest.json`, `prediction_freeze_manifest.json`, `generated_reaction_steps.jsonl.gz`, `literature_hits.jsonl.gz`, `source_passages.jsonl.gz` (obey copyright: metadata+short factual source locator; don't republish full text without rights), `reaction_evidence.jsonl.gz`, `route_evidence.jsonl.gz`, `baseline_comparison.csv`, `mechanism_consistency.jsonl.gz`, `evidence_coverage.json`, `metrics.json`, `case_registry.csv`, `captions.md`, `figure_manifest.json`, plus honest source-specific missingness.
- `figures/fig2_r1_literature_supported_routes.{pdf,svg,png}`, `figures/fig3_r2_supported_composition.{pdf,svg,png}`, `figures/fig4_r3_structural_precedent_cases.{pdf,svg,png}` only when generated evidence actually supports the necessary panels. Do **not overwrite PR #87** observed figure binaries or the active paper main without review.
- Full experiment source records remain immutable; no model training or new costly MCTS, no injected GT references, no fabricated DOI/patents or fake positive reaction passages. Build corpus indices and run on **all** generated steps that have available source files; keep lack of source/literature coverage explicit without blocking substantive case and quantitative figure delivery.
- Case gallery selection requires a concrete actual MCTS-only disconnection/route *and* source-checked A or B evidence when available, or use separate cases and caption precisely. Display exact graph changes with RDKit, the actual source–sink arrow **only for symbolic executed FLOW actions**, and cited source for the complete reaction. If source only supports one route edge (Cabozantinib), highlight just that edge.

**Completion criterion:** independently reproducible literature-source retrieval over frozen genuine model-generated reaction transformations; per-edge source Tier with document and structural proof; **per-460-target** literature-covered stock-route metrics, 3,120-target reaction-candidate source support (not GT match) and paired same-ID Greedy comparisons; positive chemistry-led NMI Figures 2/3 + detailed R3 case appendix; all outputs traced to saved model source hashes. Software acceptance tests and source QA support presentation but do not replace results.

## Source bibliography for retrieval protocol

- ORD schema and recorded reaction procedure/outcome/source fields: https://docs.open-reaction-database.org/en/stable/schema.html
- ORD data access: https://docs.open-reaction-database.org/en/stable/overview.html
- Public US patent extracted reactions (Lowe): https://figshare.com/articles/dataset/Chemical_reactions_from_US_patents_1976-Sep2016_/5104873
- Existing pilot patent: https://patents.google.com/patent/US20160022662A1/en

This document is an **execution protocol, not a new empirical result**. Every reported chemical precedent requires its own actual retrieved primary source and structural match.
