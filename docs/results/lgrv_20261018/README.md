# PR89 literature evidence execution

Status: **implementation and full-population processing in progress**. This is
not a completed chemical-validation result. No GPU job, model inference,
training, ranking evaluation, MCTS restart or paper merge was performed.

## Latest verified progress (18 October, supersedes older processing notes)

**Independent-control runtime:** the rolling batch scheduler, lazy pinned-vLLM
transport, durable complete input/output journals and non-ranking episode-prefix
aggregation are now implemented and covered by26 focused CPU software tests.
See the [prepared control runtime](independent_control_preparation_20261018/README.md).
Formal manifest-authenticated CLI/cross-shard collection and actual predictions
remain missing. No model, training or GPU task was launched; these tests are
not evidence of a model-performance gain.
Final whole-repository CPU verification: **1,240 passed/14 skipped**, one
existing PEFT warning,142.96s.

**Current v7 source review:** seven more performed preparations bring exact
source-supported complete reactions **15→22** under identical18,210-query
retrieval. All21 sealed output hashes were checked. MCTS now has8 targets with
an exact step on a stock-closed route; all-step coverage remains1 target.
The31k proxy diagnostic has7/5/4 source-supported targets for MCTS/Greedy I/II.
The same registry is applied to every condition, with no new model inference.
See [complete v7 tables](analysis_tables_primary_v7_20261018/README.md),
[all seven new RDKit cases](additional_selectivity_gallery_20261018_v7/README.md)
and a [checked positional-isomer source discrepancy](primary_structure_review_20261018/README.md).
That discrepancy remains unresolved, not a claim of impossible model chemistry.
Latest complete CPU suite: **1,229 passed/14 skipped**, one existing PEFT warning,
164.11s. Older v5/v6 outputs below are retained as versioned evidence.
The [current v7 Figure2/3 and three extended panels](population_figures_primary_v7_20261018/README.md)
are regenerated, hash/geometry checked and visually inspected; they retain
actual inverse FLOW arrows, full populations and recorded generation costs.

**Coverage follow-up / v6:** full source review now confirms **15** exact
performed reactions after six additional primary preparations; the same
18,210 queries,165 candidate-hit reactions and210 hits are unchanged.
[Complete v6 tables and all15 source cases](analysis_tables_primary_v6_20261018/README.md)
retain every condition, not only MCTS. R1 MCTS now has7 targets with an exact
step on a stock-closed route and1 target with an all-step source-covered route. Broader source
review also found that139 non-patent references cite one data-origin method
paper; exact-candidate traceback recovers106 patent-linked records and104
public patent URLs. All104 were acquired/verified,71 new to the prior cache.
[Full missingness audit and raw traceback](coverage_traceback_20261018/README.md).
This is improved evidence coverage, not a chemical accuracy estimate.

The [independent-route controller and full460/fixed200 manifest](independent_control_preparation_20261018/README.md)
are prepared and tested, but actor transport/CLI and actual control predictions
remain missing. No new GPU/model job was launched. Current whole CPU gate:
**1,222 passed/14 skipped**, one existing PEFT warning,194.15s. Existing v5
figures below retain their original v5 counts; they are not v6 exports.

**Latest v5 update:** the full stereo-aware retrieval is complete over all18,210
queries and six frozen conditions:165 candidate-hit reactions/210 source
records. Sixty-six public patent pages have now been cached after repairing
numeric leading-zero lookup addresses without changing source identities.
Five further performed examples were individually checked and structurally
reconciled, increasing checked exact net reactions **4→9**. See the
[five-source review](PRIMARY_REVIEW_ADDITIONS_20261018.md),
[complete v5 evidence/paired tables](analysis_tables_primary_v5_20261018/README.md),
[full v4 retrieval audit](analysis_tables_stereo_v4_20261018/README.md) and
[all-candidate acquisition receipts](primary_acquisition_20261018/README.md).
R1 MCTS has5 targets with any exact step on a stock-closed route, versus2 in
v3/v4; every-essential-step support remains1. None of these is a population
chemical-accuracy or matched-compute gain claim. Older v3/v4 data below remain
explicit historical stages, not current counts. No model jobs were added.

The [current v5 Figure2/3 and three extended panels](population_figures_primary_v5_20261018/README.md)
are exported and visually inspected, with complete paired, structural-complexity
and actual-work data. The [additional five-source plates](additional_precedent_gallery_20261018/README.md)
add two actual chemical-reaction pages. The earlier
[v4 Figure2/3](population_figures_stereo_v4_20261018/README.md) remain preserved;
do not relabel them with v5 source counts. Latest complete CPU software suite:
**1,202 passed/14 skipped**, one existing PEFT warning,213.78s. The32 focused
extended/population tests also passed. Software tests are not model results.

- **Original corpus complete:**93/93 official files,2,428,291 source rows and
  4,338,961 indexed structure representations. The latter are not unique
  chemical reactions. [Completion/hash receipt](CORPUS_COMPLETE_20261018.json).
- **Saved program join complete:**10,480 reports,37,922 program records and
  zero unmatched frozen reaction occurrences. All available native state
  bindings passed. The27 unavailable signatures contain no FLOW operation,
  not compiler/executor failures. [Full join receipt](PROGRAM_JOIN_COMPLETE_20261018.json).
- **Full structural retrieval complete:**all18,210 frozen unique reaction
  queries across six conditions,71 with indexed candidates/114 source records.
  Three-registry analysis `source_checked_20261018_v3` is complete and sealed
  with14 output hashes. Four reactions have checked exact experimental
  precedent;18,206 remain unresolved. This is source-review coverage, not a
  chemical correctness rate. [Complete evidence and paired tables](analysis_tables_20261018/README.md).
- **Broader source checks:**two additional performed patent examples have now
  been cached and structurally reconciled: phenoxyaniline acetylation
  ([US8188116B2](https://patents.google.com/patent/US8188116B2/en),Example11)
  and quinoline-phenol benzylation
  ([US7576215B2](https://patents.google.com/patent/US7576215B2/en),Example489).
  All six conditions have been reaggregated with the added registry in the
  separate v3 output. No source/query was replaced. [Population summary](POPULATION_RESULTS_20261018.json).
- **Actual generation work:**all eight exact PR88 wave prefixes now bind the
  frozen stock-route witnesses.852,951,580 newly generated output tokens and
  184,560,843 separate cache-return tokens;8,035,879 unique scheduled model
  calls. This includes unfinished-target work, not only200 saved reports.
  [Measured work and source-prefix hashes](R1_ACTUAL_GENERATION_WORK_20261018.json).
- **Software gate:**1,135 passed/14 skipped in the post-fix CPU-only whole
  suite,155.93s, with one existing PEFT warning. Its10 targeted work tests
  also passed. [Review/fixes](POPULATION_FIGURE_REVIEW.md).
  GPU checks were not rerun because unrelated work occupied the GPU.

The final [population Figure2/3](population_figures_20261018/README.md),
[four-reaction Figure4 gallery](reaction_precedent_gallery_20261018/README.md)
and [actual accepted electron-flow fork](cabozantinib_actual_fork_20261018/README.md)
are exported and visually inspected as SVG/PDF/PNG. They show the current
evidence, not completed population chemical validation. Broader primary
checking, independently calibrated analogue coverage, missing matched controls,
complexity/paired extended panels and the final whole-goal audit remain
required. The PR89 goal is **not complete**.

## Native-map retrieval coverage follow-up

The original query freeze marked17,128 reactions `mapping_unknown`, although
the completed native-program join preserves their actual atom correspondences.
`mechet.analysis.literature_mapping` now authenticates both complete input
manifests, recomputes every saved family's radius-1 key from the actual mapped
product/structural precursors, and requires agreement across all native programs
for each exact chemical identity. No reference mapping or new mapper is used.

Full execution over18,210 queries/37,922 programs resolved17,048 formerly
unknown queries. The separate derived view has18,109 available mappings and101
conflicting families retained as unknown. It preserves all original reaction
keys, raw structures, source pointers and environmental components. Archived
single-reaction `auxiliary_components` can be a recorded full-mixture remainder
even when `auxiliary_raw` is empty; this format is preserved, not silently
reconstructed.13 mapping tests and the fresh whole CPU suite passed:
**1,148 passed/14 skipped**, one existing PEFT warning,157.09s.

Output: `$DATA_ROOT/outputs/paper/nmi_literature_evidence/native_mapping_full_20261018_v1`.
The [manifest](native_mapping_full_20261018/mapping_enrichment_manifest.json)
and [complete18,210-query mapping view](native_mapping_full_20261018/unique_reaction_queries_enriched.jsonl.gz)
are byte-identical Git copies, not a sampled showcase or selected success set.
This is a prepared retrieval view, **not** a rerun of the evidence analysis;
the four checked source reactions and the published v3 tables are unchanged.
The evidence CLI now accepts `--native-mapping-view`, and the population
renderer consumes its authenticated copied table rather than silently
reverting to the old queries. Both rederive every row from the pinned native
programs; self-consistently resealed invented mappings are rejected. The real
complete18,210-query consumer audit passed. Existing v3 outputs are unchanged.
Independent analogue-policy validation and full-population re-retrieval remain
separate requirements. The initial control failure below is preserved; the
versioned repair and subsequent full-population run are described next.

```bash
PYTHONPATH=src python -m mechet.analysis.literature_mapping \
  --prediction-freeze-root "$FREEZE_ROOT" \
  --program-join-root "$DATA_ROOT/outputs/paper/nmi_literature_evidence/program_join_pr88_20261018_v3" \
  --output-root "$DATA_ROOT/outputs/paper/nmi_literature_evidence/new_native_mapping_view"
```

### Independent structure-policy controls: one real representation defect

The frozen [control specification](../../../configs/analysis/lgrv_independent_retrieval_controls_20261018.json)
uses all three actual reactions from [Organic Syntheses95,80–96](https://www.orgsyn.org/demo.aspx?prep=v95p0080),
procedures A–C, each with four declared structure-policy variants. Every source
structure was reconciled to its original ORD row. None of the three source
reaction keys occurs in the frozen18,210 MechET query pool. Training-source
overlap is still unknown. These controls are not model evaluations or proof
of experimental performance on hypothetical substituted molecules.

Actual result: **11/12 passed**, with all12 machine-only evidence grades C.
The [complete control inputs/outputs and manifest](independent_policy_controls_20261018/policy_control_results.json)
and the three-source SQLite control index are copied byte-for-byte into this
delivery. This is an independent source-anchored policy-control index, never a
replacement for the93-file production source corpus.
The remote-fluoro analogue of procedure C was unexpectedly unresolved. Its
full product stereocentre remains R on both sides of the comparison, but
RDKit's truncated radius-1 fragment emits `C[C@@H:1](C)C` versus
`C[C@H:1](C)C`. Both fragments have three indistinguishable methyl groups and
lose that invalid local stereotag when parsed. Thus full-molecule traversal
parity has leaked into a nominally local retrieval key. The radius and source
cap were not adjusted to hide this failure; the failed result is retained.

The primary procedure was actually read using the web reader. Direct HTML
download returned HTTP403, so no raw HTML cache/hash or automatic
`source_passage_checked` claim is fabricated. The original lawful ORD bytes
remain available. Procedure B reports98%ee, whereas the ORD row says99%; that
disagreement is not transferred into any MechET yield or evidence claim.

That failure required stable local serialization **without erasing genuine
full stereochemistry**, while preserving original index bytes/provenance.
The expected control label, radius and source cap were not retuned.

### Versioned repair and full-population continuation

`literature_stereo_family.py` supplies an explicit `full_stereo_local_v2`
reader for the unchanged production index. Legacy stereo spellings are lookup
aliases only when their parsed local graphs are canonically equivalent.
Candidate analogues must then match a separately regenerated explicit-H local
graph carrying full-graph CIP labels; opposite full enantiomers and E/Z remain
different. Cut aromatic boundaries are treated as graph fragments rather than
erroneously requiring a complete kekulizable ring. No exact identity,
isotope/charge/component policy, model output or old source-index row changes.

A first repair fixed procedure C but exposed the aromatic-boundary case B:
its11/12 result is retained in
[the failed boundary attempt](independent_policy_controls_stereo_v2_20261018/failed_aromatic_boundary/policy_control_results.json).
The final repair passed **all12 unchanged independent controls**, with zero
source-key overlap and no automatic evidence-grade promotion. See
[complete passed inputs/outputs](independent_policy_controls_stereo_v2_20261018/passed/policy_control_results.json).
These limited structure-policy controls do not establish universal analogue
precision or chemical performance on the hypothetical query variants.

The reader has a declared256-alias implementation-work ceiling (below SQLite's
portable binding limit). Exceeding it retains the full reaction and exact
search but explicitly marks the analogue context unresolved. All native
programs for a reaction must agree on the corrected context; a representative
is chosen by saved program ID, never by outcome. The source cap remains five.
Both analysis and publication consumers rederive the complete context table
and re-execute the independent controls; an editable pass flag is insufficient.

The full CPU continuation `source_checked_20261018_v4` completed from
immutable runtime tree `64ec0fbc5af599de43dd401727aba2e79d592279`, over the same
18,210 queries, six conditions,93-file index and original primary registries.
Native-context work uses eight CPU workers; no new model/GPU inference runs.
Its21 sealed outputs were verified:165 reactions with indexed candidates,
210 source records,4 checked exact reactions. The completed v5 reaggregation
adds five separately checked preparations, for9 exact reactions, without
changing retrieval, actor outputs or target populations. Existing v3/v4
tables and figures remain preserved and must not be relabelled as v5 results.

Final software gate for this increment: **1,187 passed/14 skipped**, one
existing PEFT warning,144.51s;23 population tests and86 focused core tests also
passed. These are software tests; the independently executed12-control source
assay and the full18,210-query literature analysis have separate receipts.

Software verification after integration: **1,165 passed/14 skipped**, one
existing PEFT warning,136.74s. The100-test focused mapping/population/protocol
suite and7 policy-runner tests also passed. Software checks do not turn the
observed11/12 independent control result into a passed calibration gate.

## Completed

- Read the updated PR89 LGRV-v1 specification. Its complete-reaction evidence
  protocol supersedes the obsolete ranking-oriented review comment.
- Implement conservative reaction identity, explicit auxiliary roles, local
  mapped-edit retrieval keys, SQLite source retrieval, independent evidence
  states, AND-route evidence aggregation and paired-target statistics.
- Implement immutable prediction extraction, official-source downloader,
  streaming ORD protobuf/Parquet adapter, parallel source index builder and
  evidence-analysis CLI. The source match alone never becomes verified A/B.
- Download **all93 official production ORD Parquet files**, **1,256,763,947
  bytes**, cross-checking every mirror LFS content SHA against its official Git
  pointer. The official mirror's five extra test fixtures are explicitly
  excluded; this is not a reaction-level filter. File hashes and URLs are in
  [the download manifest](source_corpus_download_manifest.json).
- File-footer inventory: **2,428,291 source reaction rows**, including
  **1,771,032 USPTO/Lowe rows across41 yearly patent datasets**. These are
  source records, not unique chemical reactions or verified model predictions.
- Re-fetch and check the existing Cabozantinib free-base acid-chloride/aniline
  coupling in [US20160022662A1](https://patents.google.com/patent/US20160022662A1/en).
  The primary registry binds the original cached bytes, normalized document,
  actual performed preparation, substrate/product structures and a14-word
  passage. This remains **one source-checked complete reaction**, not evidence
  for all symbolic FLOW actions or the full route. Source conditions differ
  from the actor's auxiliary mixture; the adjacent acid-chloride preparation
  was later independently checked against WO2018104954A1, as documented below.
- Complete the full frozen prediction extraction: **26,330 reaction occurrence
  records**, **18,210 distinct complete reaction queries**, **474 saved AND-stock
  routes**, and **10,740 target/run rows** across six conditions. Counts are not
  literature support or chemistry correctness. Exact per-condition denominators
  and full artifact hashes are in [the freeze summary](prediction_freeze_summary.json).
- Software checks: **52 LGRV tests passed**; fresh full repository run **1,056
  passed /13 skipped**, with one pre-existing PEFT vocabulary-config warning.

## Processing

- The completed prediction freeze uses precisely PR88's200 saved R1 reports and original
  **460-target** universe; both complete460-target Greedies; completed3120
  MCTS targets and both complete3120-target Greedies. It never copies reference
  precursors into retrieval queries. No additional report from the growing R1
  run is silently included.
- Source indexing streamed the entire downloaded corpus with original and
  corrected repair workers; all93 completed shards are now assembled.
  Binary source data and generated SQLite shards remain outside Git. Source
  conversion failures are retained in per-file failure ledgers; they are not
  classified as chemical negatives.
- Independent review reproduced a source-file partial-stereo exception and
  four correctness/recovery issues. All were corrected with regressions; the
  entire failed source file completed conversion with the corrected worker
  while other original successful shard work is preserved. See
  [review and fixes](FOUNDATION_REVIEW.md). The combined source index is now
  complete; the old driver's16-file manifest is historical, not the count of
  completed on-disk source shard receipts.
- [The entire-file repair](source_file_repair.json) read70,725 source rows and
  retained129,079 structured representations, including the formerly crashing
  exact reaction. Sixteen invalid source representations are logged, not
  treated as chemical negatives. This is source conversion, not a model test.
- Prediction freeze v1 stopped on a serializer compatibility error: aggregate
  31k rows mix absolute resumed-tree pointers and relative new-tree pointers.
  The resolver now handles both and rejects ambiguous ownership, with a
  regression test. v1 is an **incomplete diagnostic**, preserved read-only;
  v2 is the completed replacement. Cached immutable shard contracts avoid repeated Ceph
  manifest reads. No model output or original source report was changed.
- All-route post-build audit checked474 routes in273 original reports: every
  edge belongs to its own target/method report. The rechecked
  [existing seed binding](seed_source_binding.json) occurs once in this frozen
  MCTS pool and not in either full archived Greedy pool. This is a descriptive
  source-backed disconnection example, not a matched-work population advantage
  or a newly discovered literature source.

## Not yet delivered

Broader primary-document checks beyond the four checked reactions, validated
analogue retrieval settings, all-condition measured-work comparisons, missing
control preparation, extended panels and the final whole-goal audit remain
unfinished. Current Figure2/3, the four-reaction Figure4 gallery and the
supported two-edge case are exported. The18,206 unresolved reactions are not
chemical negatives. Native-program mapping may provide additional defensible
analogue keys; current `mapping_unknown` counts are not complete analogue
coverage.
The goal and original scope remain active; unit tests do not replace results.

## Follow-up: actual program joins and a complete source-backed case

`scripts/annotate_nmi_literature_programs.py` now joins saved terminal contexts
to the frozen complete-reaction pool, authenticates the actor exposure, records
native source hashes/pointers and graph-complexity features, and preserves the
original target populations. Both real serializers (nested route network and
flat single-reaction tree) have regression coverage. Explicit compatible-cache
adoption rechecks source bytes and only adopts a completed report prefix; it
does not replay GPU inference or overwrite the failed version.

All fixed200 R1 reports have been associated: **7,996/7,996 reaction occurrence
records**, **11,480 terminal programs**, including **6,288 training-unseen
ordinary program occurrences /3,890 distinct unseen v1 signatures**. This
is not a chemical correctness rate. The combined six-condition annotation
and post-review action-state binding audit are now complete. These are
archival v1 linkage/signature counts. See [the exact R1 snapshot](R1_PROGRAM_JOIN_SNAPSHOT.json)
for exposure scope, overlap limitations and immutable source-prefix hashes.

One **two-reaction Cabozantinib free-base route** now has checked exact net
precedent on both essential edges. The new source checks the acid-chloride
formation performed **in situ**, not an isolated characterized intermediate.
Its generated five-FLOW program is training-unseen under the fixed v1
abstraction, although every primitive occurs in at least4,677 training
reactions. See [the case and limitations](CABOZANTINIB_CASE.md). This is one
traceable positive case, not full-corpus coverage or a matched-compute gain.

The evidence CLI now accepts a saved program-join root and multiple independent
primary registries. It joins R2 to actual source-supported chemical reactions,
compares both31k Greedies at the same target IDs with2,000 paired bootstrap
draws, and separates distinct reaction identities from target–reaction pairs.
Paired tables retain the full original denominator and separately show the
common completed-report intersection; pending reports are never called failures.
Source-inspected coverage includes documented contradictions, while positive
source support remains a different field.
Work summaries expose cache/restart and missing-cost fields; summed target
elapsed seconds are **not** represented as global GPU time.

The final reviewed feature set has **75 focused tests passed**; the fresh
whole suite has **1,079 passed /13 skipped**, with one existing PEFT warning.
No new model experiment was performed.

Latest figure/collection increment: **1,102 passed /13 skipped** in the fresh
whole repository run (117.07s), with the same existing PEFT warning. All four
review issues were reproduced and fixed; the23 figure/collection tests pass.
The earlier counts above identify prior implementation gates, not new model
results or the final scientific evaluation.

## Source-bound chemical figures and full-catalogue assembly

[The actual Cabozantinib figure set](cabozantinib_figures_20261018/README.md)
contains five SVG/PDF/PNG plates: both complete source-checked reactions and
all15 saved inverse actions, including18 real electron-pair moves. Molecular
drawings use RDKit's element colours; arrows come from exact saved current-state
addresses. Every figure is183mm wide, at most164.05mm high, with embedded PDF
fonts, editable SVG text and approximately300dpi PNG previews. Final PNG hashes
equal all five actually viewed originals. The source checks do not establish
the physical microscopic mechanism or a matched-work population advantage.

The new `scripts/assemble_nmi_literature_index.py` combines completed immutable
source shards across the original root and repair roots without overwriting
old live partial databases/journals. It requires every file from the **original
93-file catalogue**, verifies raw/index/failure-ledger hashes and original
Parquet row counts, preserves legacy producer provenance through explicit
opt-in, and refuses partial-corpus substitution. Finalization is resumable;
donor/source path and inode aliases are rejected. [Review](FIGURE_COLLECTION_REVIEW.md).

For reproducibility, the completed93-source assembly used:

```bash
python scripts/assemble_nmi_literature_index.py \
  --source-manifest "$ORD_ROOT/source_corpus_manifest.json" \
  --donor-root "$DATA_ROOT/data/derived/literature/ord_lgrv_46980f8_v1" \
  --donor-root "$DATA_ROOT/data/derived/literature/ord_lgrv_46980f8_repair_20261018" \
  --donor-root "$DATA_ROOT/data/derived/literature/ord_lgrv_remaining_repair_20261018_v1" \
  --codec-sha256 f33ac383b4c0517355188a6e2b4d77bc679dc9ea553b16d467043e478f4f16dd \
  --allow-compatible-v1-shards --workers 8 \
  --output-root "$DATA_ROOT/data/derived/literature/ord_lgrv_full_20261018_v1"
```

Use that completed full index, not a23-file repair-batch index, for the final
evidence CLI. Source conversion and the full program join are now complete;
scientific evidence coverage and publication delivery remain separate gates.

## Exact source versions

| Source | Frozen version |
|---|---|
| ORD official Git | `704870642e5c93a141e6666436bc14a9b5f880ed` |
| ORD official HF read mirror | `46980f8cb68aa43ed9707edec319d4eac5a2bfed` |
| ORD schema generated protobuf | `51b3a4968835f0090268cab9c0be034b48ac62d4` |
| `reaction_pb2.py` SHA256 | `f33ac383b4c0517355188a6e2b4d77bc679dc9ea553b16d467043e478f4f16dd` |
| Data/code licenses | ORD data CC-BY-SA-4.0; ORD schema code Apache-2.0 |
| Source/training independence | Unknown unless separately demonstrated |

The official ORD repository documents the lawful HF read mirror and source
format: [ORD data](https://github.com/open-reaction-database/ord-data),
[schema](https://docs.open-reaction-database.org/en/stable/schema.html).
No gated or subscription corpus was accessed.

## Reproduce

Run from the PR89 checkout with RDKit, PyArrow, protobuf, PyYAML and pytest.
Set `DATA_ROOT` to the shared operational repository containing the frozen
source artifacts. Local analysis variables below do not replace system paths.

```bash
DATA_ROOT=/aaa/fionafyang/buddy1/whaleywang/MechET
ORD_ROOT=$DATA_ROOT/data/raw/literature/ord_46980f8cb68aa43ed9707edec319d4eac5a2bfed
INDEX_ROOT=$DATA_ROOT/data/derived/literature/ord_lgrv_full_20261018_v1
FREEZE_ROOT=$DATA_ROOT/outputs/paper/nmi_literature_evidence/frozen_pr88_20261018_v2

python scripts/download_nmi_literature_sources.py \
  --output-root "$ORD_ROOT" \
  --revision 46980f8cb68aa43ed9707edec319d4eac5a2bfed \
  --github-revision 704870642e5c93a141e6666436bc14a9b5f880ed \
  --github-client gh --workers 8

# Download the official generated codec into an analysis-only directory.
# Verify its SHA against the table above before executing the index builder.
curl --fail --location --output /tmp/mechet_lgrv_reaction_pb2.py \
  https://raw.githubusercontent.com/open-reaction-database/ord-schema/51b3a4968835f0090268cab9c0be034b48ac62d4/ord_schema/proto/reaction_pb2.py

python scripts/build_nmi_literature_index.py \
  --source-manifest "$ORD_ROOT/source_corpus_manifest.json" \
  --codec-file /tmp/mechet_lgrv_reaction_pb2.py \
  --codec-sha256 f33ac383b4c0517355188a6e2b4d77bc679dc9ea553b16d467043e478f4f16dd \
  --codec-source-revision 51b3a4968835f0090268cab9c0be034b48ac62d4 \
  --output-root "$DATA_ROOT/data/derived/literature/ord_lgrv_fresh_rebuild" --workers 16

# To resume the existing pre-v1.1 successful shards after the old workers exit,
# add --allow-compatible-v1-shards. Source/codec/database/failure-ledger hashes
# remain required; implementation migration is never silent.

python scripts/freeze_nmi_literature_predictions.py \
  --repository-data-root "$DATA_ROOT" \
  --tree-analysis-root "$DATA_ROOT/outputs/paper/nmi_tree_evidence/observed_20261016_v1" \
  --progress-snapshot docs/results/progress_20261017/snapshot.json \
  --route-assets-manifest "$DATA_ROOT/data/route_benchmarks/frozen_r1_r2_20261013/manifest.json" \
  --output-root "$FREEZE_ROOT" --workers 16

python scripts/annotate_nmi_literature_programs.py \
  --prediction-freeze-root "$FREEZE_ROOT" \
  --actor-exposure-manifest "$DATA_ROOT/outputs/analysis/r1_r2_actual_sft_mechanism_index_20261014/manifest.json" \
  --output-root "$DATA_ROOT/outputs/paper/nmi_literature_evidence/program_join_pr88_20261018_v3" \
  --workers 4

# The actual v3 continuation uses an immutable committed runtime plus:
# --compatible-program-cache .../program_join_pr88_20261018_v1
# --cache-manifest-sha256 a23e58337c56c41d5eb58dfe7f2019a715f03b0f95f4d0d90f501820c730eef3
# --cache-links-sha256 d0252d47fc1134a7892e92ccfcb57117a2d3d1bae67a35359d517202818a1259
# It revalidates action payloads and endpoint/continuity binding before reuse.
# v1's source-layout failure remains preserved. The v2 preflight was stopped
# before output when source code changed during its Ceph wait; no GPU stopped.

# Only after BOTH index and prediction-freeze manifests say complete=true.
# Use a new output root; prior analysis is never overwritten.
python scripts/verify_nmi_literature_reactions.py \
  --tree-analysis-root "$DATA_ROOT/outputs/paper/nmi_tree_evidence/observed_20261016_v1" \
  --progress-snapshot docs/results/progress_20261017/snapshot.json \
  --source-reaction-index "$INDEX_ROOT/reactions.sqlite" \
  --route-assets-manifest "$DATA_ROOT/data/route_benchmarks/frozen_r1_r2_20261013/manifest.json" \
  --prediction-freeze-root "$FREEZE_ROOT" \
  --retrieval-config configs/analysis/nmi_r1_r2_matched_baselines_20261017.yaml \
  --primary-source-registry configs/analysis/lgrv_primary_source_seed_20261018.json \
  --primary-source-registry configs/analysis/lgrv_primary_sources_20261018_v2.json \
  --primary-source-registry configs/analysis/lgrv_primary_sources_20261018_v3.json \
  --program-join-root "$DATA_ROOT/outputs/paper/nmi_literature_evidence/program_join_pr88_20261018_v3" \
  --output-root "$DATA_ROOT/outputs/paper/nmi_literature_evidence/new_full_analysis"

# Actual generation-wave cost at the same immutable PR88 snapshot; no GPU run.
PYTHONPATH=src python -m mechet.analysis.literature_work \
  --progress-snapshot docs/results/progress_20261017/snapshot.json \
  --prediction-freeze-root "$FREEZE_ROOT" \
  --output "$DATA_ROOT/outputs/paper/nmi_literature_evidence/new_snapshot_work.json"
```

The primary seed registry references the captured document path on Ceph. On a
different machine, call `snapshot_primary_document` on its public URL, verify
both hashes in the registry, then explicitly relocate `document_path`. A
changed live page is a new snapshot, not an automatic checksum exception.

The two PR88 input files under `docs/results/progress_20261017/` are copied
byte-for-byte from PR88 commit`22e5a9798376787b6d4cf44eadbdb7bc6901ac17`,
not silently represented as PR87's older171-target snapshot. The source PRs
remain separate. No software fixture is described as an experimental smoke result.
