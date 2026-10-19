# Current full-population reproduction (v18 bibliographic correction)

This is the current complete invocation, not the historical three-registry
command in the main README. It uses all 22 unchanged primary registries, the
authenticated native mapping and full-stereo independent policy controls, and
the original-source patent-family overlay before cap-five selection.

Run from the repository root in the recorded compatible CPU/RDKit environment.
Set `DATA_ROOT` to the existing frozen data checkout; it is not a new dataset.
The analysis output directory must not exist. Original data and earlier results
must never be overwritten. The index may be a byte-identical local transport;
its manifest and SHA must match the original frozen index.

```bash
DATA_ROOT=/aaa/fionafyang/buddy1/whaleywang/MechET
LGRV_ROOT="$DATA_ROOT/outputs/paper/nmi_literature_evidence"
INDEX_ROOT="$DATA_ROOT/data/derived/literature/ord_lgrv_full_20261018_v1"

PYTHONPATH=src python scripts/verify_nmi_literature_reactions.py \
  --tree-analysis-root "$DATA_ROOT/outputs/paper/nmi_tree_evidence/observed_20261016_v1" \
  --progress-snapshot docs/results/progress_20261017/snapshot.json \
  --source-reaction-index "$INDEX_ROOT/reactions.sqlite" \
  --route-assets-manifest "$DATA_ROOT/data/route_benchmarks/frozen_r1_r2_20261013/manifest.json" \
  --prediction-freeze-root "$LGRV_ROOT/frozen_pr88_20261018_v2" \
  --program-join-root "$LGRV_ROOT/program_join_pr88_20261018_v3" \
  --native-mapping-view "$LGRV_ROOT/native_mapping_full_20261018_v1" \
  --family-policy full_stereo_local_v2 \
  --policy-control-root "$LGRV_ROOT/independent_policy_controls_20261019_v4" \
  --policy-control-spec configs/analysis/lgrv_independent_retrieval_controls_20261018.json \
  --retrieval-context-workers 8 \
  --patent-family-registry configs/analysis/lgrv_patent_family_sources_20261019_v1.json \
  --primary-source-registry configs/analysis/lgrv_primary_source_seed_20261018.json \
  --primary-source-registry configs/analysis/lgrv_primary_sources_20261018_v2.json \
  --primary-source-registry configs/analysis/lgrv_primary_sources_20261018_v3.json \
  --primary-source-registry configs/analysis/lgrv_primary_sources_20261018_v4.json \
  --primary-source-registry configs/analysis/lgrv_primary_sources_20261018_v5.json \
  --primary-source-registry configs/analysis/lgrv_primary_sources_20261018_v6.json \
  --primary-source-registry configs/analysis/lgrv_primary_sources_20261019_v1.json \
  --primary-source-registry configs/analysis/lgrv_primary_sources_20261019_v2.json \
  --primary-source-registry configs/analysis/lgrv_primary_sources_20261019_v3.json \
  --primary-source-registry configs/analysis/lgrv_primary_sources_20261019_v4.json \
  --primary-source-registry configs/analysis/lgrv_primary_sources_20261019_v5.json \
  --primary-source-registry configs/analysis/lgrv_primary_sources_20261019_v6.json \
  --primary-source-registry configs/analysis/lgrv_primary_sources_20261019_v7.json \
  --primary-source-registry configs/analysis/lgrv_primary_sources_20261019_v8.json \
  --primary-source-registry configs/analysis/lgrv_primary_sources_20261019_v9.json \
  --primary-source-registry configs/analysis/lgrv_primary_sources_20261019_v10.json \
  --primary-source-registry configs/analysis/lgrv_primary_sources_20261019_v11.json \
  --primary-source-registry configs/analysis/lgrv_primary_sources_20261019_v12.json \
  --primary-source-registry configs/analysis/lgrv_primary_sources_20261019_v13.json \
  --primary-source-registry configs/analysis/lgrv_primary_sources_20261019_v14.json \
  --primary-source-registry configs/analysis/lgrv_primary_sources_20261019_v15.json \
  --primary-source-registry configs/analysis/lgrv_primary_sources_20261019_v16.json \
  --retrieval-config configs/analysis/nmi_r1_r2_matched_baselines_20261017.yaml \
  --output-root "$LGRV_ROOT/new_full_family_deduplicated_analysis"
```

The family source registry pins 137 existing original-page receipts, each
rechecked against raw HTML and normalized text. Full source documents remain
outside Git. It reads only each page's own Family section; publication kinds
and chemical grades are not inferred. Sources lacking family metadata remain
explicitly unresolved at family level, not independent families by assertion.
Moving a cache requires an explicit new path-bound registry, not suppressing
checksum validation. Public source URLs, document hashes and metadata locators
are in the distributed overlay artifact.

The scope remains all 18,210 frozen complete reaction queries and six recorded
conditions, all 460 route target IDs and all 3,120 single-reaction target IDs.
This CPU analysis does not launch training, model sampling or new MCTS.
