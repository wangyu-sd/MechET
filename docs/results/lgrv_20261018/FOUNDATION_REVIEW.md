# LGRV foundation review and correction

Read-only independent review covered candidate`c7e4bff` against PR89 head
`17e2deb`. It judged this an honestly labeled, unfinished foundation; no
critical issue was reported. This is not the final scientific-completion audit.

The five important findings were reproduced and corrected:

1. ORD reaction`ord-bbd8fc960495476d8ef6e1a1b41ce03d`, row4419 of
   `ord_dataset-0c0a93e5eb286a8fa697b1fdd8231f85`, triggers RDKit's partial
   stereochemical-fragment serialization error. Exact complete-reaction
   identity succeeds. Keep that exact record and stereo; mark only its local
   family unavailable with an explicit diagnostic. Worker failures are now
   persisted immediately, without hiding behind process-pool shutdown.
2. Require the original-byte cache receipt, original SHA, normalized-text SHA,
   source URL and actual normalization relationship before source-checked A/B.
   A passage hash or self-declared normalized text alone is insufficient.
3. Completed index reruns validate identity and return without mutating the
   completed manifest. Interrupted merges use a transactional per-shard ledger
   and publish the final database atomically. Implementation/normalization
   identity is explicit. Adoption of pre-v1.1 shards requires an explicit flag;
   source, codec, SQLite and conversion-failure-ledger hashes are checked.
4. ORD reactant-role and mapped-identifier representations from the same
   underlying reaction consume one source-family slot. Alternative
   representation IDs remain in the retrieval record. The corrected reader
   also works on intact v1 source records, avoiding a mandatory corpus rebuild.
5. Route membership checks use only their own report's generated reaction
   keys, never the cross-target/method retrieval deduplication pool. Independent
   post-build inspection checked all474 frozen routes in273 source reports:
   every route edge was present in its own report. See
   [the bound audit](report_local_route_audit.json); v2 predictions do not need
   regeneration for this guard correction.

The reviewer labeled failure-ledger checksum validation minor. It was treated
as part of important item3: losing source-conversion diagnostics would make
corpus missingness unauditable. The fix is included; no minor is deferred.

Items deliberately not claimed complete by this review remain active:
population primary-document verification, full R2 exposure mapping, analogue
policy calibration, work-efficiency comparisons, final figures and independent
source/training separation. The chemically substantive seed interpretation
was not independently reviewed by the software reviewer. It remains explicitly
assistant-source-checked, not human-reviewed or a physically observed arrow path.

Operational note: initial v1 full-corpus workers continue retaining successful
shards while their old driver waits after the known failed future. A separate
repair completed the **entire failed source file** with the corrected code,
without changing original source data, other active workers or any model job.
Do not infer a complete combined index until its final manifest is verified.
