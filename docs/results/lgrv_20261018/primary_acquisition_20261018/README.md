# Primary-document acquisition, complete frozen candidate pool

The lossless [acquisition manifest](acquisition_manifest.json.gz) contains all
210 indexed source candidate records from `source_checked_20261018_v4`.
Seventy-one patent references deduplicate to **66 public lookup URLs**; all66
were downloaded successfully. The remaining139 non-patent records are retained
in `deferred_records`, not silently removed or counted as checked papers.

Original ORD patent IDs and URLs remain unchanged. An additional lookup field
removes leading numeric zeros from US identifiers, for example
`US08501746B2` → `US8501746B2`; no publication kind or patent family is inferred.
The older v3 acquisition tried63 original URLs and recorded63 HTTP404 failures.
That failure receipt remains outside Git in the original output directory.
This is bibliographic address repair, not circumvention of a403 restriction.

Original completed manifest SHA256:
`8c90030bd1792459f1626f330c86acc2bbf082574e717db34d98b6c5a4cd2715`.
All66 original HTML and normalized-text cache hashes were independently checked.
The manifest includes every original query/source reference and the raw/text
receipt paths and hashes. Full patent documents remain in the external cache:

`/aaa/fionafyang/buddy1/whaleywang/MechET/data/raw/literature/primary_documents_bulk_20261018_v2/`

Download success is not experimental verification: every acquisition receipt
keeps `source_passage_checked=false`. Specific checked preparations have a
separate, structure-reconciled primary registry. No full text is redistributed.

Reproduce with `python -m mechet.analysis.literature_primary_acquisition`, using
`--literature-hits <v4/literature_hits.jsonl.gz>`, the above cache root,
`--output-root <new directory>` and `--workers 4`. Existing output roots are
refused. HTTP failures are recorded once; they are not retried or upgraded to
chemical contradictions.
