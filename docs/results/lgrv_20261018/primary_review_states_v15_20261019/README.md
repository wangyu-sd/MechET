# Complete v15 unresolved-query work states

All110 unresolved indexed-candidate queries remain,not a selected sample:

| Work state | Queries |
|---|---:|
| Original preparation pointer unresolved |33|
| Primary cached;preparation review pending |52|
| Performed candidate checked;applicability unresolved |23|
| Performed candidate checked;stereochemical identity unresolved |2|

All77 referenced primary documents are available andbyte/normalization
checked;that is not77 verified query reactions. The two new batches remove
three now-supported queries andmove seven from unread to a documented
applicability limit. No unresolved query is labelled a chemical failure.

The [manifest](review_states_manifest.json) binds the complete v15 analysis,
all source reviews andoriginal output hashes. The [recipe](audit_recipe.py)
physically recompiles all three grouped-review bundles against their exact
frozenv13/v14 inputs andtheir original capped candidates. Earlier source
scope/isomer/stereo/image records andcached documents are rechecked too.
The query-hit map is separate from review objects;cross-bundle duplicate
source/query pairs are rejected.

Full-data regression first failed on the seven checkedC queries still marked
unread in the old inventory. After regeneration,the full110 records equal
the old113 minus exactly three newly supported queries,with only those seven
review-state fields changed. Every other field andunreviewed row remains
unchanged. This verifies work-state accounting,not new chemistry.
