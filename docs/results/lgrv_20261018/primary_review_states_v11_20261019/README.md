# Complete v11 primary-review work states

This inventory retains every unresolved indexed query from the complete
18,210-query/six-condition v11 analysis. It separates source availability,
original preparation inspection and compatibility with generated substrates.
Cached documents are not automatically verified reactions; reviewed sources
do not automatically support a different substrate or unspecified stereo.

| Actual state | Unresolved indexed queries |
|---|---:|
| Original-preparation pointer unresolved |33|
| Cached originals; preparation review pending |85|
| Performed candidate checked; substrate applicability unresolved |3|
| Performed candidate checked; stereo identity unresolved |1|
| One source-isomer mismatch checked; other candidates pending |1|
| Total |123|

All82 distinct primary documents referenced by this remaining queue are
available and physically checksum/normalization checked. The whole queue
SHA is `34ee4f57e1cdb3646aa487499125349ae42188be6d3ff815bbcd0f820fba815a`.

The three previously checked analogue scope cases and positional-isomer case
remain explicitly unresolved. The [stereo-qualified preparation review](../primary_stereo_review_20261019/README.md)
adds one checked-but-unresolved query whose index key omits the original
relative stereochemical constraint. All three underlying original patent
paragraphs and raw/text receipts are reauthenticated by the tracker, without
changing its TierC or calling it chemically invalid.

`primary_review_states.jsonl.gz` contains the whole queue and all original
capped candidate identities/primary pointers. Query and source identity are
joined together; repeated bibliographic publications do not create additional
independent chemical candidates. The manifest binds all inputs, reviewed-source
records, producer and output hashes. Original cached documents remain outside Git.

The method-paper access observations remain [separately recorded](../primary_review_states_v10_20261019/method_citation_access.json).
No access failure was bypassed; a method DOI alone does not identify an original
performed preparation. This is review progress, not an independent policy
calibration, new model inference or a population chemical-accuracy estimate.
