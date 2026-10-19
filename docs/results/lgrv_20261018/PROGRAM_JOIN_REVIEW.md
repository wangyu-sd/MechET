# Incremental program-join review

Read-only review compared tree`0520b64ba57d15f1f359fa92a28acd6bfd1235c9`
with published PR89 commit`7a0fdf86467838337a6e2cbfac7aa994b3c624a5`.
This review covers the new program/chemistry join increment, not completion of
the entire literature protocol. No model experiment was performed.

## Findings and corrections

| Finding | Correction / regression evidence |
|---|---|
| Cache annotations could change while source tree hashes/counts stayed constant | Complete artifact hashes checked; partial adoption requires externally pinned manifest and links hashes; compiler/endpoint producer hashes unchanged; actual native action payload digest checked. Modified classification fixture now rejects reuse. |
| A program could start at a different molecule from its attributed reaction | Map-preserving root and terminal binding plus available public-result continuity checks; initial/final/continuity mismatch fixtures become unavailable with no complete signature. |
| The helper could combine two methods/reports for one target | Explicit one-run/one-report ownership guard; cross-method/report fixtures rejected. |
| Standalone CSV omitted pending/common-observed safeguards | Export lower-bound label, pending counts and common-observed2x2/difference fields. Original full denominator retained. |
| Top-level source-inspected coverage excluded checked contradictions | Separate inspected versus positively supported counts, with checked-D CLI integration fixture. Contradiction records are looked up by their actual contradicted query key. |
| A partial program join vanished from remaining work | Keep R2 outstanding when unmatched occurrences remain; integration fixture verifies it. |
| Interrupted-cache recovery read beyond its declared committed prefix | Bound legacy cache reading to the completed report-prefix count; require independent byte hashes. Flush outputs before publishing a new atomic progress checkpoint. Arbitrary damaged prefixes are not silently repaired. |

All three Important findings were reproduced with failing tests and fixed in
one pass. CSV missingness and checked-D coverage were regraded from Minor to
Important because misleading standalone scientific outputs matter. The two
remaining review items were addressed in the same cache/CLI regressions.

## Scope rulings

- Broader primary retrieval, analogue calibration, remaining program joins,
  actual run-ledger cost attribution and figures stay required and unfinished.
- The reviewer did not independently transcribe the patent structures. The
  main implementation review checked the named compounds and actual performed
  text; source checks are assistant passage checks, not human expert review.
- The in-situ acid-chloride limitation is explicit. Non-isolation alone is not
  a protocol reason to relabel a documented performed preparation as a mere
  patent claim. Neither isolated analytical validation nor physical-arrow
  validation is asserted.
- Numerical source/census claims and test totals were verified by the main
  executor, not independently reproduced by the read-only reviewer.

The code runs from a committed immutable runtime for subsequent background
analysis. A preflight-only v2 CPU process was stopped after source-code updates
during a Ceph wait could otherwise have mislabeled the loaded producer version.
No GPU/Taiji experiment or original prediction was stopped or overwritten.
