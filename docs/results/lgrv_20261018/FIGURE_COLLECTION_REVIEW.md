# Incremental figure and source-collection review

Baseline: PR89 `ee384deacec8fa33c19936b3e89bad458dfea4fe`.
Read-only independent review of candidate tree
`302bbb321345e7758994ddb278fd5b5f0608490e`. The final export includes the
subsequent LF-only CSV correction and reviewed fixes, not that initial tree.

| Finding | Verified regression and correction |
|---|---|
| Assembly output could overwrite a donor manifest | Direct and directory-symlink collisions reproduced before the fix; source/donor directory overlap rejected before any write. Added and reproduced a hardlinked-building-DB case; inode aliases to raw sources, donor receipts, indices and failure ledgers are also rejected. |
| Final SQLite rename followed by interrupted manifest finalization could not resume | Reproduced the exact complete-database/incomplete-manifest state; recovery now checks source receipts, original row counts, SQLite integrity, complete merged-shard inventory and output record count, then completes the manifest without rewriting the database. |
| Completed-output reuse omitted codec identity | Changed-codec retry reproduced; reuse now rejects codec mismatch before returning previous success. |
| Generic C/D case wording could imply positive precedent | Regraded from Minor to Important because evidence wording affects scientific interpretation. Unresolved-case and actual cached contradiction-source fixtures reproduced the overclaim; captions and figure footer now distinguish A/B support, C unresolved and D documented contradiction. The actual A/A case drawing is unchanged. |

All findings received one regression-first fix pass. No deferred minors.
The reviewer independently ran the initial 16 new tests; the main executor
ran the post-fix23-test figure/collection suite. Fresh final whole repository
run: **1,102 passed /13 skipped /one existing PEFT warning**,117.07s.

## Scope rulings

- Whole-corpus completion, remaining program joins and population Figures2/3
  were declined as unfinished: they remain required, not waived. Cost of
  treating them otherwise would be a false completed scientific result.
- Independent chemical transcription verification was declined: the primary
  source checks remain assistant-source-checked, not independent human review.
  Visual inspection is not chemical verification. Physical arrow validation is
  not claimed; source overlap remains unknown.
- The review did not judge final v2/final asset hashes: all final exports pass
  machine QA, and all final PNG hashes equal the actually viewed originals.
- The review did not independently rerun the whole suite or live processes:
  these are separately checked by the main executor. No model experiment or
  GPU task was launched, stopped or altered by this increment.
