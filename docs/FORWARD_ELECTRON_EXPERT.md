# Compact Forward Electron-Flow Expert

## Purpose

MechET keeps the large inverse model and the forward verifier deliberately
separate:

```text
mapped product
  -> inverse LLM actor: executable retrosynthetic electron-flow CoT
  -> executor-derived precursor candidates
  -> compact forward expert: precursor-to-target and competitor scoring
```

The forward expert is **not** a second general-purpose chat model. It is a small
PyTorch graph model with source and sink pointer heads. Its job is narrow:

1. rank the next forward electron source and sink;
2. score whether proposed precursors recover the target product;
3. compare the target with explicitly enumerated competitor products;
4. provide a soft process/edge score for RL and multistep search.

The deterministic proof executor remains the hard gate. A learned forward score
never rescues an atom-map, valence, charge or electron-conservation failure.

## Supported chemistry

Version 1 is intentionally restricted to mapped, closed-shell, two-electron
polar chemistry. Electron containers are:

- `LP(atom)` — a lone-pair source;
- `BOND(atom_i, atom_j)` — a bond source or bond sink;
- `ATOM(atom)` — the atom receiving the electron pair when a bond cleaves.

A chemically elementary event may contain multiple coupled arrows. They are
applied atomically, so an SN2 step can form the nucleophile bond and break the
leaving-group bond without requiring an invalid pentavalent-carbon intermediate.
Radicals, spin states, transition-metal orbitals, coordination changes and
photochemical one-electron steps are outside the v1 scope and should be reported
as `UNKNOWN` by downstream systems.

## Independence from the inverse actor

The intended paper configuration is:

| Component | Recommended architecture | Training direction |
|---|---|---|
| Inverse actor | Qwen-family LLM with electron-flow CoT | product to proof/precursors |
| Formal executor | deterministic RDKit program | no training |
| Forward expert | compact graph message-passing model | precursors to target/competitors |

Use patent-disjoint, temporal and reaction-family holdouts. Freeze the forward
expert during actor RL. For high-stakes experiments, use cross-fitting so that a
reaction is scored by a forward fold that did not train on that reaction.

## Canonical data schema

`forward_expert_data.py standardize` writes one JSON object per reaction:

```json
{
  "id": "reaction-id",
  "source": "mech_uspto_31k",
  "reaction_smiles": "mapped_reactants>mapped_reagents>mapped_products",
  "reactants": "mapped reactants",
  "reagents": "optional agents",
  "products": "mapped products",
  "mechanism_class": "optional class",
  "conditions": {},
  "competitor_products": [],
  "steps": [
    {
      "step_index": 0,
      "state_smiles": "mapped current state",
      "target_product": "mapped next/final state",
      "moves": [
        {
          "source": {"kind": "LP", "atoms": [12]},
          "sink": {"kind": "BOND", "atoms": [4, 12]},
          "electrons": 2
        }
      ]
    }
  ],
  "split": "train"
}
```

The normalizer never invents arrows. Rows without unambiguous source-sink labels
can train the reaction-compatibility head but do not train the move-pointer
heads. Invalid or unmapped rows are written to a quarantine JSONL with a reason.

## Data acquisition and licenses

The registered sources live in `configs/forward/data_sources.yaml`.

```bash
# Inspect the action without network access.
python scripts/forward_expert_data.py download \
  --dataset mech_uspto_31k --dry-run

# Download a public snapshot and write file-level SHA-256 provenance.
python scripts/forward_expert_data.py download \
  --dataset mech_uspto_31k \
  --revision <frozen-revision> \
  --output data/raw
```

PMechDB-derived material is marked restricted in the downloader. It is not
fetched unless the caller has reviewed the upstream terms and passes
`--accept-restricted-license`. Do not commit downloaded datasets or third-party
checkpoints to this repository.

The downloader uses `huggingface_hub.snapshot_download`. The standardizer uses
Hugging Face `datasets` for Arrow/Parquet when installed, while JSON/JSONL/CSV
remain usable with the Python standard library.

## Standardization and step construction

```bash
python scripts/forward_expert_data.py inspect \
  --input data/raw/mech_uspto_31k

python scripts/forward_expert_data.py standardize \
  --input data/raw/mech_uspto_31k \
  --output data/forward_expert/reactions.jsonl \
  --source mech_uspto_31k

# ORD/outcome-only records: decode first, then map explicitly.
python scripts/forward_expert_data.py standardize \
  --input data/raw/ord_data \
  --output data/forward_expert/ord_unmapped.jsonl \
  --source ord_data --allow-unmapped

python scripts/forward_expert_data.py map \
  --input data/forward_expert/ord_unmapped.jsonl \
  --output data/forward_expert/ord_mapped.jsonl

python scripts/forward_expert_data.py build \
  --input data/forward_expert/reactions.jsonl \
  --output-dir data/forward_expert/steps
```

The source adapter decodes FlowER/mech-USPTO mechanistic SMILES such as
`state|(source,sink);...` and PMechDB arrow codes such as `10=20;20,21=21`.
ORD Protobuf records are decoded with the official `ord-schema`. RXNMapper is an
optional explicit stage for records that do not contain atom maps.

Every standardized run writes a report and quarantine file. Freeze their hashes
alongside the model configuration before producing paper results.

## Optional baseline pre-download

The compact graph model does not require a language-model checkpoint. Optional
ablation backbones can be cached reproducibly:

```bash
python scripts/forward_expert_data.py predownload \
  --model chemberta \
  --model molformer \
  --revision <frozen-revision> \
  --output models/baselines
```

`qwen_small` is also registered for a small sequence-model ablation. The default
method remains the graph-pointer expert because it is architecturally independent
from the inverse Qwen actor and can score all source-sink competitors cheaply.

## Model

The default model uses:

1. atom and bond categorical embeddings;
2. compact message-passing blocks;
3. an electron-container encoder;
4. a source pointer head;
5. a source-conditioned sink pointer head;
6. a reactant-product compatibility head;
7. a hashed condition channel for sparse solvent/reagent metadata.

No PyTorch Geometric dependency is required. This avoids CUDA-extension version
coupling while retaining a standard PyTorch implementation. The module can be
replaced later by PyG/DGL without changing the JSON schema or command-line API.

## Training objectives

For a positive forward path and a competitor product, training minimizes:

```text
L = L_source + L_sink + lambda_r L_reaction + lambda_m L_margin
```

- `L_source`: next electron-source cross entropy;
- `L_sink`: source-conditioned sink cross entropy;
- `L_reaction`: positive/negative precursor-product binary loss;
- `L_margin`: target score must exceed the competitor score.

Training:

```bash
python scripts/train_forward_expert.py \
  --config configs/forward/forward_expert_small.yaml
```

A CPU smoke configuration is included:

```bash
python scripts/train_forward_expert.py \
  --config configs/forward/forward_expert_tiny.yaml \
  --device cpu
```

The script saves `best/` and `last/` checkpoints, configuration metadata and a
JSONL training log. The tiny configuration is for CI/smoke testing only.

## Inference and scoring

```bash
python scripts/run_forward_expert.py infer \
  --checkpoint outputs/forward_expert/small/best \
  --input data/forward_expert/steps/test.jsonl \
  --output outputs/forward_expert/test_predictions.jsonl \
  --auto-competitors 8
```

Explicit competitors from curated data are preferred. `--auto-competitors`
adds formally reachable alternative states as bounded hard negatives; it cannot
claim exhaustive experimental side-product coverage.

Each output contains:

- formal execution of the labeled coupled arrows;
- gold-move ranks under the pointer model;
- target score and target rank;
- best competitor score;
- selectivity margin;
- uncertainty and structured verdict.

The existing plausibility interface can load the expert as an oracle:

```bash
export MECHET_FORWARD_EXPERT_PATH=outputs/forward_expert/small/best
export MECHET_FORWARD_EXPERT_DEVICE=cpu
```

```python
from mechet.plausibility import load_oracle
oracle = load_oracle("mechet.forward_oracle:score_payload")
evidence = oracle({
    "precursors": "...",
    "target": "...",
    "competitor_products": ["..."],
})
```

## Forward electron-flow generation

The generator performs beam search over model-ranked electron moves. Formal
execution is used as a hard filter; learned product scores are optional soft
reranking evidence. Single arrows and locally coupled two-arrow events are
considered.

```bash
python scripts/run_forward_expert.py generate \
  --checkpoint outputs/forward_expert/small/best \
  --input data/forward_expert/steps/test.jsonl \
  --output outputs/forward_expert/generated_paths.jsonl \
  --beam-size 16 \
  --branch-limit 24 \
  --proposal-pool 48 \
  --max-steps 6 \
  --stop-when-solved
```

This is a bounded research generator, not a transition-state simulator. A path
that is executable is formally consistent, not necessarily fast or experimentally
successful.

## Evaluation

```bash
python scripts/run_forward_expert.py eval \
  --predictions outputs/forward_expert/test_predictions.jsonl \
  --output outputs/forward_expert/test_metrics.json

python scripts/run_forward_expert.py eval-generation \
  --predictions outputs/forward_expert/generated_paths.jsonl \
  --output outputs/forward_expert/generation_metrics.json
```

The evaluator reports:

- formal pass rate, false acceptance and false rejection when labels exist;
- next-move Top-1 and reciprocal rank;
- target-product Top-1;
- target-versus-competitor selectivity support;
- Brier score and expected calibration error.

Paper evaluation must additionally stratify by mechanism family, complexity,
reaction-centre multiplicity, patent/time split and family holdout. Thresholds
for `SELECTIVITY_AMBIGUOUS` must be calibrated on validation data per family;
they must not be chosen on the test set.

## Use in inverse RL and multistep planning

The forward expert supplies soft evidence:

```text
formal executor: hard reject
forward target recovery: terminal/process reward
selectivity margin: soft reward and reranking
uncertainty: cost/abstention signal
```

`forward_edge_cost` converts this evidence into a route-search edge cost.
`score_inverse_proof_forward` exposes the same independent forward signal as an
RL reward after deterministic proof execution. Complete hypothesis files can be
reranked with:

```bash
python scripts/rerank_proof_hypotheses_forward.py \
  --predictions outputs/proof/hypotheses.jsonl \
  --checkpoint outputs/forward_expert/small/best \
  --output outputs/proof/hypotheses_forward_ranked.jsonl
```

Do not hard-prune a route solely because a learned score is low. Learned verifiers are
fallible; formal violations may be pruned, while forward/selectivity evidence
should initially be used for ranking and calibrated abstention.

## Reproducibility contract

For every reported checkpoint retain:

- exact dataset and model revisions;
- download manifests and SHA-256 hashes;
- standardization report and quarantine file;
- split-generation policy;
- YAML training config;
- checkpoint metadata;
- inference parameters and competitor-generation policy;
- calibration set and family-specific thresholds.
