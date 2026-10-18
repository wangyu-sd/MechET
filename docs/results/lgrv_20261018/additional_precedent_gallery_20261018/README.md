# Five additional actual generated reactions with primary precedent

- [Plate1](figures/extended_data_additional_precedents_1.pdf): para-hydroxybenzoyl
  chloride,3,5-difluorobenzoyl chloride and o-cresyl acetate.
- [Plate2](figures/extended_data_additional_precedents_2.pdf): pyridone–thiazole
  C–N coupling and meta-cyanopropylbenzoyl chloride.

These are all five additional exact transformations in
`configs/analysis/lgrv_primary_sources_20261018_v4.json`, not simulated model
outputs. They occur in the same frozen18,210-query population; exact identities,
cached primary passages and named source structures were checked. Source
conditions and source-specific yield limitations are preserved in the
[case registry](case_registry.json). See the
[review table](../PRIMARY_REVIEW_ADDITIONS_20261018.md) for primary URLs and
preparation locators. This source-positive gallery is not a population rate or
an assertion that all five are MCTS-only discoveries.

Both pages have matching editable SVG and300dpi PNG. The final PNGs were opened
and visually inspected; structures, source labels and conditions are visible
without clipping or external raster dependencies. Both PDFs use embedded fonts,
183mm width and <=170mm height; machine QA passed. Full-source and export hashes
and the exact selected reaction-key order are in `figure_manifest.json`.

Use `mechet.analysis.literature_figures.render_net_precedent_gallery` with the
complete v5 enriched query view, all five registry records, frozen style and
the manifest's reaction-key order: first three cases on Plate1, last two on
Plate2. It recomputes exact query identities and source grades before drawing.
Export using `scripts.render_nmi_tree_figures.export` and run
`mechet.analysis.figure_qa.inspect_figure`; the old four-case gallery remains
unchanged. These net-reaction plates intentionally do not manufacture electron
arrows from a net structural difference. Actual saved inverse FLOW arrows
remain in Figure3, the fork plate and the complete program plates.
