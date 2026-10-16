# PR #86: existing electron-flow tree evidence

This analysis consumes saved outputs only. It does not submit a training job,
sample another trajectory, change the live executor/search, or modify the paper.
The specification is code PR #86 at `e8b2595`; the paper design is PR #29 at
`ffc2890`. Formal R1/R2 comparisons remain distinct from these observed figures.

## What the saved primary results establish

| Source snapshot | Recorded observations | Interpretation |
|---|---:|---|
| R1/R2 shared MCTS, October 14 `_07` | 171 completed target reports of 460 | 170 nonzero stock-connected routes, one already-stock target; unfinished searches remain pending |
| Same run, per-target contexts | 215,114 contexts; 834,112 attempted actions; 210,676 distinct context transitions | Shared states can occur in several target snapshots; these are not globally unique molecules or independent experiments |
| Same run, electronic alternatives | 14,040 FLOW-fork contexts; 170 targets have a FLOW fork | Actual distinct accepted electronic successors, not merely repeated proposals |
| Same run, route alternatives | 71 cases have route-count lower bound above one | Not an exhaustive count of chemically different syntheses |
| mech31k MCTS, October 16 `_03` | 1,711 saved target reports of 3,120 | Historical `rxn_prod_min` proxy diagnostic, not the corrected full principal-product benchmark |
| Same 31k pools, canonical-unique NLL ranking | Top-1: 32; Top-5: 72; pool: 79 | 47 same-pool ranking losses; MCTS pool recall is not independent Pass@K |

The stock check reconstructs an AND-route witness and rechecks every leaf
against the hashed frozen stock. This establishes catalog closure, not chemical
feasibility. R1/R2 observes 20/100 PaRoutes, 91/160 RetroCast, 41/100 URSA Drugs,
and 18/100 URSA Expert nonzero closures. These are lower bounds on the original
460 targets, not success rates conditioned on finished easy searches. No
equal-compute MCTS-versus-greedy gain is claimed.

The 31k ranking example is `mech-uspto31k-full:test:10731`: the recorded matching
precursor set ranks seventh. The first-ranked alternative is structurally
different and has an unusual sulfur-containing participant. We show that actual
decision rather than label it experimentally impossible without evidence. Better
ranking alone cannot repair the many pools that contain no recorded match.

## Figures and chemical meaning

`fig2_r1_observed_tree_analysis` combines frozen-cohort closure lower bounds,
one complete electronic-state forest, morphology and a genuine shared-state
FLOW fork. `fig3_r2_observed_chemistry` combines the same RetroCast reference
depths, actual historical signature accumulation and a complete generated
inverse program. Neither filename is a final complete-benchmark claim.

The three fixed cases are Cabozantinib, Nivasorexant and RetroCast `n5-00853`.
Each has an atlas, a before/after chemical microscope and a complete action-order
gallery. IMPORT species are drawn with their multiplicities and purposes;
FLOW cells show recorded pre-action mixtures and source–sink electron arrows;
FINISH shows executed structural precursors. A coupled multi-move FLOW remains
one transition. Rejected actions are stubs, not invented molecular nodes. A bold
atlas path is a shortest recorded root-to-terminal chain, not necessarily the
selected all-leaf synthesis route.

The Cabozantinib benchmark label mentions S-malate, but the actual target is
the free base. Its generated acid-chloride/aniline structural coupling has a
matching preparative description in [US20160022662A1](https://patents.google.com/patent/US20160022662A1/en).
The source uses carbonate in aqueous THF. This does not corroborate the model's
auxiliary mixture or exact proton transfers. The cited preceding chlorination
uses oxalyl chloride, not the model's thionyl chloride. The registry therefore
supports one structural pair, not every route step, condition or physical
intermediate. Training-source overlap remains unknown.

Extended Data cover full trees, empirical morphology, recorded error taxonomy,
execution counts at distinct units, same-pool ranking, and a chemical-ambiguity
case. Further source-hashed tables retain local atom/bond/charge/component
changes, actual MCTS visits and exposure-scoped program signatures. A novel
generated signature is not automatically route C1 or literature support.

## Source and exposure boundaries

Official FlowER remains 257,171/2,890/28,971 reactions. The strict program view
is 257,167/2,890/28,967. The current actor's actual exposure gives 603 local and
6,491 ordinary whole-program signatures, plus 1,377 nonordinary signatures.
Do not replace the historical 604/6,553 census with that distinct basis.

The H2 split remains 223,863/6,200/27,104, requiring an authenticated H2-trained
actor. The present full-data 0.6B model does not qualify. All 160 current
RetroCast reference composition labels remain Unknown; reference depth is a
separate descriptive variable. Existing paper work packages are retained.
Six main figures plus Table 1 still require an editorial display-limit decision.

The collection includes 12 native run roots, two full 31k greedy conditions
stored in an older top-level-context schema, and the historical parent-linked
`qwen_electron_tree_val56_20261013/tree.json`. That single-molecule demonstration
is archived separately, not promoted to a full validation benchmark. Shared
WAL/cache files are backing stores, not additional independent target trees.

## Regeneration

Use an analysis environment with Python, RDKit, Matplotlib, CairoSVG, PyYAML,
Pillow and pypdf. The measured rendering environment was Python 3.13 with RDKit
2026.3.4, Matplotlib 3.10.6, CairoSVG 2.9.0, PyYAML 6.0.3, Pillow 12.0.0 and
pypdf 6.8.0. The full repository tests also run in the existing pinned meteor
Python 3.11 environment. No model weights are loaded by these commands.

Set `task_repo` to the shared data checkout and `task_output` to a new analysis
directory. Run scripts from the code checkout with `PYTHONPATH=.:src`.

```bash
python scripts/analyze_nmi_tree_evidence.py \
  --input-roots "$task_repo/outputs/eval/r1_r2_frozen_06b_dynamic_routes_20261013" \
    "$task_repo/outputs/eval/r1_r2_frozen_06b_shared_routes_20261014_05" \
    "$task_repo/outputs/eval/r1_r2_frozen_06b_shared_routes_20261014_06" \
    "$task_repo/outputs/eval/r1_r2_frozen_06b_shared_mcts_20261014_07" \
    "$task_repo/outputs/eval/mech_uspto31k_full_dynamic_electron_tree_20261013" \
    "$task_repo/outputs/eval/mech_uspto31k_full_shared_electron_tree_20261014_05" \
    "$task_repo/outputs/eval/mech_uspto31k_full_mcts_rolling_20261015_01" \
    "$task_repo/outputs/eval/mech_uspto31k_mcts_ray_migration_20261015_01" \
    "$task_repo/outputs/eval/mech_uspto31k_mcts_ray_migration_20261015_02" \
    "$task_repo/outputs/eval/mech_uspto31k_mcts_ray_migration_20261015_03" \
    "$task_repo/outputs/eval/mech_uspto31k_full_mcts_ray_20261015_02" \
    "$task_repo/outputs/eval/mech_uspto31k_full_mcts_ray_20261016_03" \
  --output-root "$task_output" --workers 8 \
  --route-assets-manifest "$task_repo/data/route_benchmarks/frozen_r1_r2_20261013/manifest.json" \
  --style configs/figures/mechet_nmi_tree_style_v1.yaml

python scripts/analyze_nmi_tree_evidence.py \
  --input-roots "$task_repo/outputs/eval/mech_uspto31k_full_dynamic_electron_tree_20261013" \
    "$task_repo/outputs/eval/mech_uspto31k_full_shared_electron_tree_20261014_05" \
  --tree-schema single_reaction --workers 8 \
  --output-root "$task_output/single_reaction_archives"

python scripts/enrich_nmi_tree_evidence.py --analysis-root "$task_output" \
  --exposure-manifest "$task_repo/outputs/analysis/r1_r2_actual_sft_mechanism_index_20261014/manifest.json" \
  --workers 8

python scripts/render_nmi_tree_figures.py --analysis-root "$task_output" \
  --style configs/figures/mechet_nmi_tree_style_v1.yaml \
  --run-id r1_r2_frozen_06b_shared_mcts_20261014_07 \
  --case-id 'ursa_drugs100:Cabozantinib S-malate' \
  --case-id 'ursa_drugs100:Nivasorexant' --case-id retrocast_mkt_cnv160:n5-00853 \
  --route-assets-manifest "$task_repo/data/route_benchmarks/frozen_r1_r2_20261013/manifest.json" \
  --historical-report "$task_repo/outputs/issue79/nmi_operator_basis_train_20261002.json" \
  --endpoint-run mech_uspto31k_full_mcts_ray_20261016_03

python scripts/render_nmi_chemistry_extended.py --analysis-root "$task_output" \
  --style configs/figures/mechet_nmi_tree_style_v1.yaml \
  --run-id r1_r2_frozen_06b_shared_mcts_20261014_07

python scripts/finalize_nmi_tree_evidence.py --analysis-root "$task_output" \
  --single-analysis-root "$task_output/single_reaction_archives" \
  --flat-tree "$task_repo/outputs/eval/qwen_electron_tree_val56_20261013/tree.json" \
  --evidence-registry configs/figures/mechet_nmi_chemistry_sources_20261016.json

python -m pytest tests/test_nmi_tree_evidence.py -q
```

`--resume` preserves existing analysis caches; without it an existing output
directory is rejected. Producer code, source files, stocks and exposure files
are hashed. Identical SVG renders and gzip tables are deterministic. PDF
creation metadata may vary; use semantic/provenance checks, not an unsupported
claim of byte-identical PDF regeneration. Input discovery freezes a per-file
snapshot, not a simultaneous final state of live remote jobs.

## Output contract and checks

Large typed tables use deterministic UTF-8 JSONL.gz rather than committing
multi-GB predictions. `trees`, `nodes`, `attempted_edges`, `terminals` and
`routes` have explicit primary IDs. Null fields mean unavailable, never zero.
Schema descriptions are in the loader docstrings and tests; each source
manifest records table hashes and row counts. Single-reaction and flat adapters
have separate output manifests; `artifact_coverage.json` reconciles them without
counting storage aliases as new observations. Run cohorts are never pooled.

`edge_chemistry`, `programs` and `planning_states` have separate counting units.
Each displayed molecule/edge/terminal can be traced through `figure_manifest`
and the original source hash. `case_registry.csv` freezes the actual figure
selection. `captions.md` describes the measured panels. `figure_qa.json` checks
183-mm page width, height below 170 mm, vector paths, embedded PDF fonts,
SVG identifier/reference integrity and 300-dpi previews. It explicitly does
not substitute for visual or chemistry review.

The checked source-data directory is
`outputs/paper/nmi_tree_evidence/observed_20261016_v1` in the shared data
checkout. Only scripts, bounded-size vector figures/previews and provenance
summaries belong in Git. Raw outputs, data, checkpoints, caches and credentials
remain outside the commit.
