# Current full-population figures and extended analyses, v5

All five figures were generated from the complete `source_checked_20261018_v5`
output, not a selected evaluation subset. All18,210 reaction queries and six
frozen conditions remain. R1 retains460 targets,200 saved MCTS reports and260
pending reports. The31k source remains the frozen `rxn_prod_min` proxy diagnostic.

| Figure | Content |
|---|---|
| [Figure2](figures/fig2_r1_literature_supported_routes.pdf) | Original cohort stock-AND closures;5 MCTS targets with any exact step;1 every-essential-step-covered target; actual generated net transformations. |
| [Figure3](figures/fig3_r2_supported_composition.pdf) | Program occurrences/signatures/reactions as distinct units;4 supported distinct reactions reached by training-unseen programs; actual saved inverse FLOW arrows. |
| [Paired outcomes](figures/extended_data_paired_route.pdf) | Same-target2x2, full460 recorded-positive lower bounds and separate200 common-observed subset;2000 descriptive bootstrap draws. |
| [Structural complexity](figures/extended_data_source_retrieval_coverage.pdf) | All18,210 reactions stratified by net changed bonds, product heavy atoms and ring-count change; source candidates and checked A/B remain distinct. |
| [Actual work](figures/extended_data_work_attribution.pdf) | Reaudited eight source-wave prefixes, including pending-target work;852,951,580 new output tokens, cached returns excluded; no invented GPU-active time. |

Every PDF has an editable SVG and300dpi PNG counterpart. All five final PNGs
were opened and inspected: no missing external images, clipped labels or
overlapping table headings were seen; molecular structures and actual saved
FLOW arrows are visible. All15 export hashes were rechecked. Machine QA,
embedded fonts,183mm width and <=170mm heights passed.

The [complete panel data](extended_panel_source_data.json) include all paired
tables, confidence intervals, complexity counts and work ratios. The
[actual work reaudit](actual_work_reaudit.json) exactly equals the earlier
receipt after re-reading all original prefixes, actor/freeze and stock
witnesses. It is not a solved-target-only cost estimate. The
[figure manifest](figure_manifest.json) binds all inputs, actual producer files,
exports and extended data hashes.

The five [additional source-checked chemical cases](../additional_precedent_gallery_20261018/README.md)
are separate net-reaction plates. Net patent precedent does not validate the
actor's auxiliary mixture, elementary arrows or an entire physical route.
Unresolved evidence is not chemically invalid. Source/training overlap remains
unknown. Paired differences and work-normalized counts are descriptive under
unequal work and incomplete source review, not a causal search advantage or
population chemical-accuracy estimate. Reference C0/C1/C2 labels remain Unknown.

Reproduce using `scripts/render_nmi_literature_population_figures.py` with the
complete v5 analysis, original frozen pool/route assets/actor exposure,
complete native program join, Cabozantinib case, all four primary registries,
the frozen retrieval/style configs, `--retrieval-context-workers 8`,
`--extended-panels` and `--actual-work-report` pointing to
`../R1_ACTUAL_GENERATION_WORK_20261018.json`. Use a new output directory.
The renderer rederives every native context and re-executes the independent
controls before source/evidence/route/composition/work verification and export.
