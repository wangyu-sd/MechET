# Stereo-aware full-population Figure2/3, v4

These figures use the completed `source_checked_20261018_v4` analysis, all18,210
reactions and all six frozen conditions. The original460 route target universe
and200 observed/260 pending MCTS reports are preserved. The31k condition remains
the frozen `rxn_prod_min` proxy diagnostic, not a corrected endpoint benchmark.

- [Figure2](figures/fig2_r1_literature_supported_routes.pdf): original cohort
  stock-AND lower bounds, complete-route source coverage and actual net reactions.
- [Figure3](figures/fig3_r2_supported_composition.pdf): authenticated program
  composition, supported whole reactions and actual saved inverse FLOW arrows.

Both have editable SVG and300dpi PNG counterparts. The final PNGs were opened
and inspected; molecular structures, arrows and labels are visible without
external image dependencies or clipped table headings. Both PDFs have embedded
fonts,183mm width, and heights167.22/160.16mm, respectively. Machine QA passed.
These checks do not validate physical elementary mechanisms.

The renderer re-executed all independent structure controls and rederived all
native retrieval contexts, rather than accepting a stored pass flag. Every
source/input/export hash and exact producer hash is in `figure_manifest.json`.
Its source/runtime code corresponds to Git tree
`02c9d184f84605730d93daa6516f222b3175c79d` before the optional extended-panel
addition. Older v3 outputs are not overwritten. The subsequently exported v5
figures add reviewed primary preparations and extended analyses separately.
