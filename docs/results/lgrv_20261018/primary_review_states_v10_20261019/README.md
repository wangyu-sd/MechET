# Review progress is not the chemical evidence tier

This full unresolved-indexed-query inventory separates the following states:

| Actual review state | Unresolved indexed queries |
|---|---:|
| Original-preparation pointer unresolved |33|
| Cached primary documents; preparation review pending |92|
| Performed candidate checked; substrate applicability unresolved |3|
| One source structure mismatch checked; other candidates pending |1|
| Total |129|

All90 distinct referenced primary documents are available and have raw/text
checksum and normalization verification. None of these129 queries is promoted
to a supported reaction by this inventory. Its whole-output SHA is
`c8d43a063b7883bc220232f0743d8f1138813d39543c0397bdf095a839c6305e`.

- A performed candidate was read, but applicability to the generated substrate
  is unresolved. These are the three previously bound analogue scope cases.
- One inspected exact-candidate preparation has a different positional isomer;
  other candidates for that query have not been ruled out.
- Original patent documents are cached and checksum/normalization checked,
  but the particular preparation still needs to be read and reconciled.
- The archived candidate and exact-chemistry traceback do not yet identify
  an original specific preparation. A dataset/method DOI alone is not one.
- A primary pointer exists but its document is unavailable, if observed.

These states describe **work and source availability**, not new TierA/B/C/D
assignments. In particular, unresolved is not invalid, cached is not reviewed,
and source-read is not substrate-compatible. Known reviewed cases are bound
to actual raw primary bytes, not inferred from document acquisition status.

`primary_review_states.jsonl.gz` preserves every unresolved indexed query,
every original capped candidate and its actual recovered primary pointers.
Alternate patent publications remain bibliography for the same candidate,
not additional independent chemical candidates. Traceback joins use both
query identity and candidate source identity because one candidate can serve
several generated reactions. All referenced available raw/text receipts and
normalization are verified, with input/output/producer hashes in the manifest.

The [method citation access observation](method_citation_access.json) keeps
bibliographic metadata separate from full text. Accessible
[publisher metadata](https://pubs.rsc.org/en/content/articlelanding/2019/sc/c8sc04228d)
and the [MIT author-repository record](https://dspace.mit.edu/entities/publication/ada9c451-480c-4c61-866c-586f43489a41)
identify the graph-convolutional reactivity paper. Its full text/SI was not
successfully read in this follow-up. Access/challenge/rate-limit failures were
not bypassed. No assertion is made that the paper contains no experiments.

The byte-identical recipe can be run from the repository root with the original
frozen analysis and archived local raw caches accessible:

```bash
PYTHONPATH=.:src python docs/results/lgrv_20261018/primary_review_states_v10_20261019/audit_recipe.py \
  --analysis-root /path/to/source_checked_20261019_v10 \
  --output-root /new/path/to/primary_review_states
```

No model/GPU inference, target filtering, source-cap change, independent
calibration claim or chemical-performance gain follows from this inventory.
