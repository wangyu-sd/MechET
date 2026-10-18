# Further original-preparation review

## Exact source recovered: A23, not A26

The earlier positional-isomer review inspected A26 in US7790756B2 and correctly
found it did not match query995f7608…. It did **not** establish that the query
had no matching preparation elsewhere in the patent.

Direct inspection now finds **A23** in both US8586565B2 and US7790756B2:
2,4-dichloropyridine plus3-fluoro4-aminophenol gives the required
4-(2-chloropyridin4-yloxy)-2-fluorophenylamine,11g86%,NMR/ESI239.2(M+H).
The named source structures independently reconcile to the frozen query.
A23 explicitly follows A2:KOtBu/anhydrousDMA,argon,100 C overnight. A2's
different picolinamide identity/quantities/77%yield are not transferred.
Only one preparation supports one reaction; the two publications do not
double its count. The earlier A26 mismatch artifact remains historically
unchanged and is not treated as evidence against the actual A23 reaction.

[Specification](../../../../configs/analysis/lgrv_checked_preparations_20261019_v7.json),
[bound registry](../../../../configs/analysis/lgrv_primary_sources_20261019_v7.json)
and [fresh validation](review_validation.json) bind original raw/text,
normalization, unique passage and source/query graphs. Validation uses the
unchanged [three-source verifier recipe](../primary_review_v12_20261019/verify_preparations.py)
with the v7 specification/registry; its new receipt contains this one record.

## A performed analogue is not automatically compatible

US8618107B2 Example335f is independently reconciled to an original candidate
for query6569547b…. It reports100mg51% isolated product andLCMS680(M+H).
Its acetate-protected boronate/fused diamine/phenyl scaffold differs from the
query's free alcohol/pyridyl boronate/other fused diazinone and oxetanyl
piperazine. The cited method148h is not resolved by the actual source's
148a/148b preparation numbering. The real148b Suzuki procedure is recorded
as a related passage, **not silently assigned** to335f.

[Bound scope review](source_scope_review.json) therefore retainsC with the
actual preparation, names, quantities and unresolved applicability noted.
This is not a claim the query reaction is impossible. The separate
[macrocycle image review](../primary_macrocycle_stereo_review_20261019/README.md)
similarly records checked-but-unresolved stereo rather than an unread source.
Neither case is used as an independent retrieval-policy calibration example.
