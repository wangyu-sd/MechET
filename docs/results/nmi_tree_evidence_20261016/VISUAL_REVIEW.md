# Publication-width visual inspection

All 23 final PDF export sets were rasterized with PDFium (`pypdfium2 4.30.0`)
at 300 dpi and their **PDF-derived**, not merely SVG-derived, previews were
opened by the implementing agent. The review used the declared 183-mm page
width; all pages are below the 170-mm height limit. Hashes identify the reviewed
PDFs in `figure_manifest.json`, `chemistry_figure_manifest.json` and
`work_manifest.json`; automated geometry/font/vector checks are separately
recorded in those manifests and `figure_qa.json`.

Reviewed sets:

- `fig2_r1_observed_tree_analysis`, `fig3_r2_observed_chemistry`.
- `case_1_atlas`, `case_2_atlas`, `case_3_atlas`, `tree_atlas` (case-1 alias).
- `case_1_microscope`, `case_2_microscope`, `case_3_microscope`,
  `node_microscope` (case-1 alias).
- `case_1_action_program`, `case_2_action_program`,
  `case_2_action_program_page_2`, `case_3_action_program`.
- `extended_data_morphology`, `extended_data_executor_errors`,
  `extended_data_execution_funnel`, `extended_data_endpoint_ranking`,
  `extended_data_chemical_ambiguity`.
- `extended_data_local_chemistry`, `extended_data_program_support`,
  `extended_data_mcts_visits`, `extended_data_observed_work`.

Checks and dispositions:

1. Text, legends, labels and boundary bonds were inspected for clipping and
   overlap. No blocking collision was observed. Dense native atlases preserve
   all nodes and reserve molecular detail for the linked microscope.
2. RDKit heteroatom colors, heavier bonds, low-saturation overlays and matching
   font hierarchy are consistent. Small imported species are intentionally
   enlarged in their own cells; this is not a common-scale geometry comparison.
3. Flow arrows refer to the **current-state** compiled private atom/bond handles.
   New-bond sinks anchor on the other atom, not a nonexistent bond midpoint.
   Multi-move calls retain their actual event grouping. The explicit terminal
   panel is a precursor set, not a claimed physical intermediate.
4. The Nivasorexant 13-action program is paginated 8+5, without dropping imports
   or ending the displayed sequence before FINISH. Displayed steps are actual
   source output, including chemically unresolved symbolic states.
5. The visits plot was corrected to a nonnegative count axis and the regenerated
   PDF was reopened. The cost plot was reopened after its final regeneration;
   token/time units and non-additive accounting notes remain legible.
6. Automated font embedding, vector-only structure, SVG ID/reference integrity
   and preview dimensions passed. SVG deterministic identity is tested; PDF
   creation metadata need not be byte-identical on regeneration.

This is an agent visual/provenance review, not independent human chemistry
annotation, color-vision certification, experimental mechanism validation or a
guarantee that every formal paper benchmark is finished. The source-supported
Cabozantinib pair is explicitly narrower than the complete generated route.
