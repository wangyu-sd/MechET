# Full frozen-population evidence figures

Current-evidence drafts, not completed chemical population validation.
Sources: all18,210 queries and all six frozen conditions, complete three-registry
analysis `source_checked_20261018_v3`; its manifest SHA is
`04ae5908598dae09e32b7e7d0cd3fe455f812b38f6022132391ac48343c2cc07`.

- [Figure2 PDF](figures/fig2_r1_literature_supported_routes.pdf): all460 original
  route targets, four cohorts,260 pending MCTS reports, stock closure and
  checked complete-reaction evidence. The current checked-source coverage is
  four distinct reactions, not an accuracy estimate.
- [Figure3 PDF](figures/fig3_r2_supported_composition.pdf): occurrences,
  signatures, reactions and targets remain separate units. The lower panels
  use exact saved inverse FLOW actions, not arrows inferred from net edits.
- Both have matching editable SVG and300dpi PNG files in `figures/`.

Both final PNGs were opened and inspected: structures and arrows are visible,
and no missing external images, overlapping table headers or clipped labels
were seen. PDF fonts are embedded;183mm width, heights below170mm. This visual
inspection is not independent expert chemical validation.

The [actual selected state fork](../cabozantinib_actual_fork_20261018/README.md)
and [four-source reaction gallery](../reaction_precedent_gallery_20261018/README.md)
are separate detailed plates. The full15-action case remains in the earlier
Cabozantinib plates. Source support concerns complete reactions, not physical
intermediates. Broader primary review, analogue calibration, missing matched
controls, complexity/work/paired extended panels and whole-goal audit remain.

Reproduce with `scripts/render_nmi_literature_population_figures.py`, supplying
the full completed analysis, frozen prediction pool, original route assets,
Cabozantinib case spec, exact actor exposure, all three primary registries,
the frozen retrieval YAML, complete program join and the PR86 style YAML.
Use a new output root; existing artifacts are not overwritten.
