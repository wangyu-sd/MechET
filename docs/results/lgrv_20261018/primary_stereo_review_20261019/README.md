# Exact index identity versus a stereo-qualified preparation

Frozen query `7e087bef…bc1a9aa3` has exact extracted-index candidates, but the
actual Example11 preparations in [US5098930](https://patents.google.com/patent/US5098930/en),
[US5403859](https://patents.google.com/patent/US5403859/en) and
[US5234950](https://patents.google.com/patent/US5234950/en) explicitly specify
the(3RS,4SR) alcohol and dimethoxy product. They report methyl iodide
O-methylation,90% yield and mp131-133C. The frozen query/index key does not
encode that relative stereochemical constraint.

An independent ring traversal confirms the same **constitution**, not a fully
stereo-resolved source identity. The record retains the original source's
relative descriptor rather than inventing a single absolute enantiomer or
removing source stereo to force an exact match. Multiple patent publications
reporting the same preparation are not three independent chemical successes.

This query stays **TierC, unresolved**, not TierD or chemically impossible.
The source supports a particular stereochemical preparation, while the query
does not establish that specificity. The performed paragraphs were directly
read; the referenced general method is not represented as a separately
completed conditions audit. No claim about each symbolic electron arrow follows.

`source_stereo_review.json` binds each inspected paragraph to actual raw/text
cache bytes, normalization, receipt hashes, unique short witnesses and the
immutable query view. The companion spec and binding recipe are included.
It is a post-hoc identity-limit inspection, not independent retrieval-policy
calibration, a new positive/negative override, or a model experiment.
