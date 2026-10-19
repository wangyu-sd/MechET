# Source-family correction: whole-population audit

The fresh independent review found publication-level fallback where the
protocol requires known patent families to share a retrieval slot. The fix
authenticates137 original cached patent pages, covering103 recorded families,
without changing the source index, chemical identity, radius1, cap5, actor,
model predictions, stock, target population or authored A/B judgments.

The completed full reanalysis has18,210 reactions/six conditions. Exactly9
candidate lists change;210 retained hits become198. Every retained patent
has resolved original-page family metadata, and no query repeats a known
family among its retained slots. Publication/representation IDs remain in the
selected candidate's provenance. DOI and ORD identities remain explicitly
distinct source types; this does not assert universal bibliographic independence.

All reaction grades and selected primary sources are unchanged:47 A,26 B,
18,137 C. Full route tables, R2 composition, same-target paired comparisons
and efficiency numerators are identical. The overlay is bibliographic evidence,
not chemical support; unresolved reactions remain unresolved.

- [Full nine-list before/after record and input hashes](whole_population_audit.json).
- [Runnable full-data comparison](audit.py).
- [Independent review and root dispositions](../FINAL_REVIEW_V17_20261019.md).
- [Current full reproduction command](../REPRODUCE_CURRENT_V18_20261019.md).

Current analysis manifest:
`f35ab780a02025555bb92a9a7e8b08f8089e62338d2dae8121ef654cf53b84e3`.
The previous v17 manifest and all its tables/figures are preserved, not relabelled.
Nine source-authentication/cap regressions failed before the fix and passed
afterward; focused83 tests and full1,296 tests passed (14 skipped, one existing
PEFT warning). The unchanged predeclared12 source-structure controls were
regenerated in a separate v4 output to bind the new reader implementation;
12/12 passed with no frozen-query overlap. No model inference or GPU job ran.
