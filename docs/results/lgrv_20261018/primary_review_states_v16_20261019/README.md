# Complete v16 unresolved-query work states

All 108 unresolved queries with indexed candidates are retained:

| Work state | Queries |
|---|---:|
| Original preparation pointer unresolved | 33 |
| Primary cached; preparation review pending | 40 |
| Performed candidate checked; substrate applicability unresolved | 28 |
| Candidate checked; indexed preparation/formulation evidence unresolved | 5 |
| Performed candidate checked; stereochemical identity unresolved | 2 |

All 77 referenced primary documents are cached and byte/normalization checked.
That is document availability, not 77 validated generated reactions. The five
source-side preparation limits are distinct from demonstrated chemical failure:
a matching product or performed related reaction need not establish the indexed
precursor identity or the proposed generated substrates.

The [recipe](audit_recipe.py) reauthenticates every earlier source review, all
six grouped-review bundles, both new preparation-limit audits, independently
transcribed candidate structures, and original capped query/source membership.
The [manifest](review_states_manifest.json) binds those inputs and the full
v16 analysis. Scientific applicability judgements remain authored inputs; the
recipe verifies their provenance and accounting, not chemical truth by itself.

The actual full-data regression first failed against the old v15 inventory.
After regeneration, v16 equals all 110 old records minus exactly two newly
supported queries, with only the ten newly checked queries' review-state fields
updated. Every other field and all remaining unreviewed rows are unchanged.

Review states do not change evidence tiers: every record here remains Tier C
and explicitly does not declare the generated reaction experimentally wrong.
The 40 cached-but-unread preparations and 33 unresolved original pointers remain
visible work, not omitted samples. No model/GPU job ran.
