# Complete stereo-aware retrieval output, v4

All18,210 original unique reaction queries and all six frozen conditions were
processed using `full_stereo_local_v2`, the complete93-file original index,
the authenticated native-mapping view and the unchanged three primary registries.
No model inference, target filtering or dataset regeneration was performed.

The completed analysis manifest SHA256 is
`c34cbf24ccc9b7239937704856bbf245a173f3673b9dc3da5b753249c3306653`.
All21 original output hashes were verified. The
[distribution manifest](stored_artifacts_manifest.json) binds17 complete
stored artifacts to the original bytes. Only `metrics.json` is additionally
gzipped; decompression recovers the entire original JSON, not selected rows.
Large omitted native bindings retain their original Ceph paths and hashes.

Result:165 distinct reactions have indexed candidates,210 source records
(92 exact,118 local-analogue candidates). Four reactions have checked exact
experimental precedent and18,206 remain unresolved. Candidate coverage is not
chemical accuracy. All18,210 context records are included:17,755 available,
354 no reaction centre and101 conflicting native families. No alias-budget
overflow occurred. Opposite full stereochemistry is not merged.

The passed12-control receipt and the earlier failed control receipts remain
separate from these model-generated population results. Source overlap with
training remains unknown. The original v3 tables and figures remain preserved.
The subsequent v5 result adds five individually reviewed primary preparations,
not new model predictions or a new structural retrieval policy.
