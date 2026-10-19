# Complete unresolved-query review state, v12

All120 unresolved indexed-candidate reactions are retained, not sampled:

| Work state | Queries |
|---|---:|
| Original preparation pointer unresolved |33|
| Primary documents cached; preparation review pending |82|
| Performed candidate checked; substrate applicability unresolved |3|
| Performed candidate checked; stereochemical identity unresolved |1|
| One source structure mismatches; other candidates pending |1|

The queue references79 distinct cached original documents. Availability and
checksum/normalization checks do not establish chemical applicability. Earlier
queue versions remain unchanged. No unresolved entry is called chemically
invalid, and this table does not include all18,045 index no-hit queries;
the complete population coverage is retained separately.

The [manifest](review_states_manifest.json) binds original full analysis,
all review/source inputs, the output bytes and producer SHA. The byte-identical
[recipe](audit_recipe.py) physically rebinds the three scope reviews, one
source-isomer review and the source stereo-identity review before joining all
remaining queries by `(query_reaction_key,candidate_source_id)`.
