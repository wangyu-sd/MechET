# Population figure provenance review and fixes

This is an incremental implementation review, not the final whole-goal audit.
Read-only independent review compared `d220809` with tree `1b9b7dd8` before
the first production population export. No critical finding was reported.

| Finding | Effect and correction | Regression evidence |
|---|---|---|
| Internally consistent route counts could lack frozen witnesses | Recompute every R1 target's outcomes/status from frozen routes, evidence and original population | Invented all-step support, baseline stock route and completed-report flags each failed RED, then rejected GREEN |
| Case/population could use different training exposure | Bind program-join exposure SHA to the supplied case exposure | Different exposure SHA rejected despite consistent actor/freeze/analysis metadata |
| 93 repeated receipts could impersonate full corpus | Authenticate original catalogue hash and distinct membership/source hashes, plus completed index hash | Duplicate receipts, modified catalogue and changed index each rejected |
| Evidence outside the case lacked completed-output binding | Write final analysis manifest; verify all table hashes; re-assess every query against actual pinned primary registries; recompute inspected/supported coverage | Edited table, invented noncase source support, inflated checked numerator and missing manifest each rejected |
| Captions hard-coded positive case/census claims | Derive case title, tier, source IDs/locators/scope and exposure counts from authenticated inputs | Genuine unresolved fixture exports without Cabozantinib/patent/census inventions |

The misleading table label “Unseen reactions” was regraded Important because
it could be read as chemical novelty; it now says “Reactions with unseen
programs.” Reaction and program units remain distinct and potentially overlap.
No minor was deferred from this increment.

The boundary reproductions initially produced 11 expected failures; two
additional tests reproduced the label and byte-binding defects. The full
case/population suite then passed 30 tests. The existing case loader was
extracted without changing source checks or the saved figure artifacts.

The reviewer did not judge actual population values, rendered visual quality,
full-goal completion, independent chemical transcription, broader primary
checking, analogue calibration, work attribution or absent matched controls.
These remain requirements; they were not waived by this review.

## Execution and throughput corrections

The initial full verification stopped before query execution because the
checked-in actor hash was accidentally truncated to 48 characters. An actual
config-versus-PR88-snapshot regression reproduced the mismatch, then passed
after restoring the 64-character identity. Actual weights were not changed.

Ordinary read-only SQLite repeatedly waited on Ceph metadata locking. A
30-query **software I/O profile** measured 7.052 s ordinarily and 0.000379 s
with immutable reads; this is not a model benchmark or a promised end-to-end
speedup. The [SQLite URI contract](https://www.sqlite.org/uri.html) requires
an unchanging file. A separate `FrozenReactionIndex` therefore authenticates
the completed index SHA before opening it, rejects writes and checks file
identity again at close. It is never used by active builders. Original index
writer/normalization producer files were left unchanged. Wrong SHA, literal
URI characters and mid-analysis mutation have regression coverage.

Actual generation-work auditing independently binds each of the eight R1
wave prefixes to PR88's exact token counters and the frozen stock-route IDs.
Its ten tests cover ambiguous/missing boundaries, counter regression,
duplicate/phantom credit, prefix hashes and separate cached-token accounting.
This includes all unfinished-target work, not only the 200 saved reports.

The new-scope read-only review independently ran15 reader/work/gallery tests
and checked both added patent examples. It found one Important recovery
defect: a later cache-only wave could solve another target without generating
tokens, then incorrectly invalidate an earlier uniquely identifiable prefix.
The regression `test_work_prefix_unchanged_by_future_cache_only_solution`
failed before the fix. Boundary selection now matches both generated tokens
and the cumulative solved-ID set. Truly ambiguous compatible plateaus still
fail. All10 work tests passed afterwards. Re-auditing all eight actual frozen
prefixes produced byte-equivalent result objects and unchanged measured totals.

The reviewer found no additional Minor issues in that scope. It declined
whole-goal completion, analogue calibration, missing controls, wider primary
coverage, physical-arrow validity and final whole-suite/population replication.
Those requirements remain open; this is not a final whole-branch review.

## Software verification scope

Prior CPU-only repository gates:1,133 passed/14 skipped before the new gallery;
then1,134 passed/14 skipped,197.52s, before the work-prefix review fix. Focused
literature/case/population/work tests previously passed96 tests; gallery/
population/work passed40 tests. Final post-fix CPU-only whole repository gate:
**1,135 passed, 14 skipped**, one existing PEFT warning,155.93s. These are
software tests, not scientific evaluation results.

An earlier GPU-visible run had two local CUDA OOM failures:
`test_paired_export_reloads_real_peft_adapter_and_rejects_corruption` and
`test_gpu_expand_real_model_widths_and_irregular_batch_tails`. Only 17 MiB was
free on a local T4 occupied by unrelated work. No unrelated process was
stopped; a fresh GPU verification pass is not claimed.
