# Complete v13 unresolved-query review inventory

All119 unresolved indexed-candidate reactions remain:

| Work state | Queries |
|---|---:|
| Original preparation pointer unresolved |33|
| Primary document cached; preparation review pending |80|
| Performed candidate checked; applicability unresolved |4|
| Performed candidate checked; stereochemical identity unresolved |2|

All77 referenced original documents are cached and byte/normalization checked;
that is not77 verified reactions. The former partially reviewed isomer query
is no longer in this queue because its actual matching **A23** preparation
was found. The earlier nonmatching A26 record remains unchanged historically.
The macrocycle and335f analogue now have specific checked-but-unresolved
states rather than being counted as unread documents.

The [manifest](review_states_manifest.json) binds the full analysis and all
source reviews. The byte-identical [recipe](audit_recipe.py) physically
rebinds the new macrocycle page/image and335f scope review as well as the
earlier records. Joins use query/candidate compound identity. Neither source
availability nor unresolved status is interpreted as chemical correctness.
