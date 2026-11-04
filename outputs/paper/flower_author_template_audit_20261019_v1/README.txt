FLOWER AUTHOR-TEMPLATE TRANSITION AUDIT — 2026-10-19

Source: jfjoung/Mechanistic_dataset, author snapshot
8bd5b38e9602c4d45485bf50076d03c292dcc486. Both Reaction_templates.py
and Reaction_templates_20250216.py are tested separately; neither is assumed
to be the exact generating revision of the local frozen FlowER archive.
Acid/base extensions come ONLY from the same author's AcidBase_lookup.py,
with the author's role/pKa threshold logic. This is not the earlier handmade
qualitative rule library and is not an independent chemical-truth database.

Population:
- all 71,868 exact run/private-state/argument saved model events in the frozen
  complete-program analysis pool (95,468 per-tree rows; 124,808 occurrences
  across 37,922 saved programs). These are NOT all rejected/incomplete paths.
- all 117,365 elementary-event targets in the 28,967 strict compiled FlowER
  test traces. Official reaction test denominator remains 28,971; four named
  upstream non-atom-conserving endpoint rows do not have strict traces.
  GT uses the stored authoritative trace-plan states, not new model outputs.

Matching:
- reverse the entire event's state orientation to the author's FORWARD rule;
- use one shared private-atom correspondence across BOTH complete SMARTS;
- require all normalized changed bonds, charges and H counts to be explained;
- evaluate recursive atom SMARTS in the actual molecular context;
- preserve spectators, charge and isotope; ordinary mapped attached H becomes
  an H count. No neutralization. No matching by gold reaction-class selection.
- source SMARTS have no stereochemical constraints; stereo feasibility is not
  inferred. A full mechanistic stage sequence is NOT validated here.

Limits explicitly retained in denominators:
- H2/free-H/hydride transitions can change the atom set after H projection;
  these require another matching representation and are not chemical errors.
- explicit-H author rules, author rules adding/deleting or repeating mapped
  atoms, and radical-changing events are not certified by this matcher.
- arbitrary bond query/aromatic-Kekule representation and template ambiguity
  can cause nonmatches; nonmatch is NOT a chemical-negative label.
- full reaction-level reagent/exclusion screening, solvent/temperature,
  kinetics, unique curved-arrow pairing and competition are NOT certified.

Results directory: results_v2 (progress.json while running; summary.json only
after all 189,233 events finish). event_inputs.jsonl.gz keeps original inputs,
successors, electron moves and native provenance. event_matches.jsonl.gz keeps
template IDs and actual private-atom bindings. compiled_author_templates.jsonl.gz
maps each ID back to raw SMARTS, version, class, mechanism, stage and pKa source.
No training, GPU sampling, executor alteration, or Taiji job is part of this audit.

Verification:
Standalone matcher/reconciliation unit tests are under this directory.
The main repository's bare pytest run on 2026-10-19 stopped during collection:
tests/test_mech_uspto31k_causal_import.py imports missing mechet.a7_rescue;
tests/test_nl_history_beam.py imports missing mechet.in_place_grounded_flow.
These existing cross-worktree imports are not changed by this offline audit.

Post-run verification: results_v2/verification.json and unit_tests.log.
After the full match, the catalog loader received an I/O-only file-handle
cleanup and a ResourceWarning regression assertion. No matching logic changed.
The exact original matcher source and SHA remain recoverable from
results_v2/runtime_matcher_source.json; historical hashes are not overwritten.
