# PR89 literature evidence execution

Status: **implementation and full-population processing in progress**. This is
not a completed chemical-validation result. No GPU job, model inference,
training, ranking evaluation, MCTS restart or paper merge was performed.

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
  is not silently promoted to supported chemistry.
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
- Source indexing streams the entire downloaded corpus using16 CPU processes.
  Binary source data and generated SQLite shards remain outside Git. Source
  conversion failures are retained in per-file failure ledgers; they are not
  classified as chemical negatives.
- Independent review reproduced a source-file partial-stereo exception and
  four correctness/recovery issues. All were corrected with regressions; the
  entire failed source file completed conversion with the corrected worker
  while other original successful shard work is preserved. See
  [review and fixes](FOUNDATION_REVIEW.md). The combined source index is **not
  yet complete**; the old driver's16-file manifest is not the count of completed
  on-disk source shard receipts.
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

Full source retrieval, new primary-document checks beyond the seed, complete
R2 program-exposure joins, validated analogue retrieval settings, all-work
efficiency comparisons, chemical case selection and the final Figure2/3/4
exports remain unfinished. No population Tier-A/B rate is claimed here.
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
and post-review action-state binding audit remain in progress. These are
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
INDEX_ROOT=$DATA_ROOT/data/derived/literature/ord_lgrv_46980f8_v1
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
  --output-root "$INDEX_ROOT" --workers 16

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
  --program-join-root "$DATA_ROOT/outputs/paper/nmi_literature_evidence/program_join_pr88_20261018_v3" \
  --output-root "$DATA_ROOT/outputs/paper/nmi_literature_evidence/source_checked_20261018_v1"
```

The primary seed registry references the captured document path on Ceph. On a
different machine, call `snapshot_primary_document` on its public URL, verify
both hashes in the registry, then explicitly relocate `document_path`. A
changed live page is a new snapshot, not an automatic checksum exception.

The two PR88 input files under `docs/results/progress_20261017/` are copied
byte-for-byte from PR88 commit`22e5a9798376787b6d4cf44eadbdb7bc6901ac17`,
not silently represented as PR87's older171-target snapshot. The source PRs
remain separate. No software fixture is described as an experimental smoke result.
