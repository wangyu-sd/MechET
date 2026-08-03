<div align="center">

# MechET

**Causal and compositional electron-flow reasoning for retrosynthesis**

[![Proof tests](https://github.com/wangyu-sd/MechET/actions/workflows/proof-tests.yml/badge.svg)](https://github.com/wangyu-sd/MechET/actions/workflows/proof-tests.yml)
[![Runtime tests](https://github.com/wangyu-sd/MechET/actions/workflows/agent-framework-tests.yml/badge.svg)](https://github.com/wangyu-sd/MechET/actions/workflows/agent-framework-tests.yml)
[![Evidence tests](https://github.com/wangyu-sd/MechET/actions/workflows/knowledge-ablation-tests.yml/badge.svg)](https://github.com/wangyu-sd/MechET/actions/workflows/knowledge-ablation-tests.yml)

[Scientific question](#scientific-question) · [Method](#main-method) · [Data](#data-flow) · [Training](#training) · [Inference](#inference) · [Evaluation](#evaluation) · [Status](#current-status)

</div>

---

## One-sentence contribution

MechET formulates retrosynthesis as causal program induction over explicit source-to-sink electron-flow actions: an environment-owned trace is the only computational source of the executable proof and precursor, enabling controlled tests of causal faithfulness and primitive-seen/composition-unseen generalization.

## Scientific question

> Can mechanistic reasoning in retrosynthesis be made causal and compositional, rather than merely plausible in language?

The study tests three falsifiable hypotheses:

1. **H1 — causal faithfulness:** the trace, rather than an independent answer channel, determines the endpoint and responds to interventions on tool observations.
2. **H2 — compositional basis:** source-to-sink execution primitives can be recombined into complete move compositions absent from training.
3. **H3 — evidence separation:** formal executability and empirical mechanistic support are distinct evidence layers.

The authoritative definitions and claim boundaries are in [`docs/SCIENTIFIC_THESIS.md`](docs/SCIENTIFIC_THESIS.md).

## Main method

```text
atom-mapped product
  -> explicit source-to-sink tool calls
  -> environment-owned molecular-state transitions
  -> committed move trace
  -> finish_trace
  -> replay declared moves
  -> deterministic MECH_PROOF v1 compilation
  -> executor-derived full precursor state
  -> atom-contributing structural precursor for evaluation
```

The main TRL-facing environment exposes only the declared tools:

```text
inspect_state
import_fragment
apply_electron_move
apply_coupled_electron_moves
finish_trace
abstain
```

Evidence conditions additionally expose `retrieve_textbook_guidance` and/or `retrieve_primitives`. Internal helpers such as `state_dict` are private, and the main method does not expose `submit_proof`.

### Endpoint views

Every executable prediction distinguishes:

- `full_precursor_state`: complete executor-derived state, including imported or auxiliary fragments;
- `structural_precursor`: fragments containing atoms originating from the target, used as the primary endpoint metric;
- `auxiliary_fragments`: remaining salts, leaving groups, or other mapped auxiliary species.

Primary structural accuracy ignores atom-map labels. Mapped exact match is reported separately.

### Evidence boundary

Textbook passages, structured mechanistic knowledge anchors, and learned forward scores are soft evidence. They cannot:

- define formal validity;
- directly return or reward a precursor;
- override executor failure;
- establish kinetics, yield, condition compatibility, or experimental success.

## Terminology

- **Electron-flow execution primitive:** a local source-to-sink action such as `LP -> BOND`, `BOND -> ATOM`, or `BOND -> BOND`. These define H2.
- **Mechanistic knowledge anchor:** a provenance-aware retrieval record with role bindings, candidate moves, warnings, and competitors. Anchor IDs do not define H2.
- **MECH_PROOF v1:** deterministic bond/lone-pair/charge program compiled from the move-bound trace in the main method; independently generated proof is a baseline.

## Install

```bash
git clone https://github.com/wangyu-sd/MechET.git
cd MechET

pip install -e ".[dev]"
pip install -e ".[agent]"       # TRL, Transformers, PEFT, datasets JSON tool columns
pip install -e ".[knowledge]"   # textbook corpus and knowledge anchors
pip install -e ".[forward]"     # optional independent forward evidence
pip install -e ".[planning]"    # optional downstream planning
```

The agent extra is pinned to released compatible ranges, including `trl>=1.8,<2`, `transformers>=5.2,<6`, and `datasets>=4.7,<6`.

## Data flow

### 1. Build executable proof rows

```bash
python scripts/build_mechet_sft.py \
  --flower-root /path/to/flower_new_dataset \
  --out-dir data/mechet_sft \
  --splits train valid test

python scripts/build_mechet_proof_sft.py \
  --input-dir data/mechet_sft \
  --output-dir data/mechet_proof_sft \
  --splits train valid test
```

Each accepted row stores the full, structural, and auxiliary endpoint views.

### 2. Freeze benchmark overlap and clean training data

```bash
python scripts/audit_reaction_overlap.py \
  --train data/mechet_proof_sft/train.jsonl \
  --benchmark data/benchmarks/uspto50k/test.csv \
  --benchmark-format reaction_table \
  --reaction-field reaction_smiles \
  --out-dir outputs/data_audit/flower_vs_uspto50k_test

python scripts/build_decontaminated_dataset.py \
  --train data/mechet_proof_sft/train.jsonl \
  --benchmark data/benchmarks/uspto50k/test.csv \
  --output data/mechet_proof_clean/train.jsonl \
  --manifest data/mechet_proof_clean/manifest.json \
  --policy exact_structural product
```

### 3. Build provenance-aware textbook evidence

```bash
python scripts/download_mechanistic_sources.py \
  --registry knowledge/source_registry.yaml \
  download \
  --source iupac_goldbook_terms \
  --source rxno \
  --source wikibooks_organic_chemistry \
  --output knowledge/raw

python scripts/build_textbook_corpus.py \
  --download-root knowledge/raw \
  --output knowledge/corpus/passages.jsonl

python scripts/index_textbook_corpus.py \
  --corpus knowledge/corpus/passages.jsonl \
  --output knowledge/corpus/bm25_index.json
```

### 4. Convert proofs into replay-verified tool trajectories

Main inference-faithful textbook condition:

```bash
python scripts/build_textbook_tool_sft.py \
  --input data/mechet_proof_clean/train.jsonl \
  --corpus knowledge/corpus/passages.jsonl \
  --output data/textbook_tool_sft/train.jsonl \
  --query-mode state
```

Combined textbook-plus-anchor condition:

```bash
python scripts/build_textbook_tool_sft.py \
  --input data/mechet_proof_clean/train.jsonl \
  --corpus knowledge/corpus/passages.jsonl \
  --output data/textbook_tool_sft/train_text_and_anchors.jsonl \
  --enable-structured-primitives \
  --query-mode state
```

`--query-mode state` uses only inference-available molecular-state terms. `--query-mode label_oracle` is an explicitly named upper bound and is not headline-eligible.

The conversion report includes root imports, family coverage, source-to-sink move counts, stable quarantine reasons, and the fraction of proof rows that replay exactly.

### 5. Derive all six matched evidence conditions

```bash
python scripts/build_knowledge_ablation_suite.py \
  --config configs/experiments/textbook_ablation.yaml
```

This generates:

```text
trace_no_knowledge
trace_length_matched_irrelevant
trace_textbook_rag
trace_structured_anchors
trace_text_plus_anchors
direct_textbook_rag
```

Validate IDs, endpoints, tool schemas, leakage, evidence characters, tokenizer rendering, assistant masks, truncation, and supervised-token budgets:

```bash
python scripts/validate_experiment_contract.py \
  --model-name Qwen/Qwen3-0.6B \
  --condition trace_no_knowledge=data/knowledge_ablation/v2/trace_no_knowledge.jsonl \
  --condition trace_length_matched_irrelevant=data/knowledge_ablation/v2/trace_length_matched_irrelevant.jsonl \
  --condition trace_textbook_rag=data/knowledge_ablation/v2/trace_textbook_rag.jsonl \
  --condition trace_structured_anchors=data/knowledge_ablation/v2/trace_structured_anchors.jsonl \
  --condition trace_text_plus_anchors=data/knowledge_ablation/v2/trace_text_plus_anchors.jsonl \
  --condition direct_textbook_rag=data/knowledge_ablation/v2/direct_textbook_rag.jsonl \
  --output outputs/contracts/evidence_conditions.json
```

Direct and tool syntaxes are not falsely assumed to have equal raw length. Final comparisons report real tokenizer input and assistant-mask tokens and use disclosed supervised-token-normalized compute.

## Training

### Tool-SFT first

Run a 32–128-example overfit test before any on-policy experiment:

```bash
python scripts/train_tool_sft.py \
  --config configs/knowledge/tool_sft_trace_no_knowledge.yaml \
  --limit 32 --max-steps 100
```

Matched SFT configs:

```text
configs/knowledge/tool_sft_trace_no_knowledge.yaml
configs/knowledge/tool_sft_irrelevant.yaml
configs/knowledge/tool_sft_textbook.yaml
configs/knowledge/tool_sft_anchors.yaml
configs/knowledge/tool_sft_combined.yaml
configs/knowledge/tool_sft_direct_textbook.yaml
```

Each real run validates the tokenizer assistant mask and writes:

```text
data_contract.json
adapter_manifest.json
adapter SHA-256
base model and condition
executor/environment revisions
```

### Optional on-policy training

Trace-only main condition:

```bash
python scripts/train_inverse_agent_trace.py \
  --config configs/agent/inverse_trace_grpo.yaml
```

Evidence conditions:

```bash
python scripts/train_inverse_agent_knowledge.py \
  --config configs/knowledge/inverse_textbook_trace_grpo.yaml

python scripts/train_inverse_agent_knowledge.py \
  --config configs/knowledge/inverse_anchor_trace_grpo.yaml

python scripts/train_inverse_agent_knowledge.py \
  --config configs/knowledge/inverse_combined_trace_grpo.yaml
```

Every required GRPO configuration loads a trainable Tool-SFT PEFT adapter and verifies its manifest, hash, base model, data contract, and executor/environment revisions. From-base RL must be a separately named ablation.

Legacy loose-trace/complete-proof baseline:

```bash
python scripts/train_inverse_agent_trl.py \
  --config configs/agent/inverse_trl_grpo.yaml
```

## H2 composition split

Build primitive-seen/composition-unseen splits from replay-verified Tool-SFT rows, not complete-proof net deltas:

```bash
python scripts/build_mechcomp_ood.py \
  --input data/knowledge_ablation/v2/trace_no_knowledge.jsonl \
  --output-dir data/ood/mechcomp_source_sink \
  --test-fraction 0.10 \
  --valid-fraction 0.10 \
  --min-train-primitive-count 5 \
  --seed 42
```

The manifest must report:

```text
primitive_basis = source_to_sink_execution_moves_v1
zero train/test composition overlap
all test primitives represented in train
non-empty held-out split
achieved split fractions
```

## Inference

`scripts/infer_mechet.py` is the canonical inference entrypoint. It produces `artifact_type=prediction` JSONL files containing complete transcripts, candidate rollouts, environment state, model/adapter hashes, generation settings, and intervention metadata.

### Trace-owned prediction

```bash
python scripts/infer_mechet.py \
  --config configs/agent/inverse_trace_grpo.yaml \
  --data data/ood/mechcomp_source_sink/test.jsonl \
  --output outputs/predictions/trace.jsonl \
  --mode trace \
  --condition-name trace_no_knowledge \
  --samples-per-target 4
```

Evidence modes use `textbook`, `irrelevant`, `anchors`, or `combined` and replay each row's frozen evidence result. Direct open-book prediction uses `--mode direct`. Legacy complete-proof inference uses `--mode legacy`.

### H1 interventions

Run the same model, adapter, generation settings, and frozen examples:

```bash
python scripts/infer_mechet.py ... \
  --mode trace --intervention none \
  --output outputs/h1/normal.jsonl

python scripts/infer_mechet.py ... \
  --mode trace --intervention remove_tool_observations \
  --output outputs/h1/remove.jsonl

python scripts/infer_mechet.py ... \
  --mode trace --intervention stale_tool_observations \
  --output outputs/h1/stale.jsonl

python scripts/infer_mechet.py ... \
  --mode trace --intervention shuffle_tool_observations \
  --intervention-source outputs/h1/normal.jsonl \
  --output outputs/h1/shuffle.jsonl
```

Additional controls are `disable_inspect_state` and `disable_intermediate_execution`.

## Evaluation

Prediction evaluation always uses a frozen reference universe. Missing predictions remain in the denominator; duplicate or extra IDs and supervision rows masquerading as predictions are hard errors. Trace metrics are recomputed by move replay and proof execution rather than trusted from stored booleans.

### H1 causal faithfulness

```bash
python scripts/evaluate_faithfulness.py \
  --reference data/ood/mechcomp_source_sink/test.jsonl \
  --normal outputs/h1/normal.jsonl \
  --intervention remove_tool_observations=outputs/h1/remove.jsonl \
  --intervention stale_tool_observations=outputs/h1/stale.jsonl \
  --intervention shuffle_tool_observations=outputs/h1/shuffle.jsonl \
  --output outputs/h1/summary.json
```

The evaluator verifies the same base model, adapter, model revision, temperature, top-p, token budget, iteration budget, and K across normal and intervention artifacts.

### H3 matched evidence

Build optional evidence-content interventions:

```bash
python scripts/build_evidence_interventions.py \
  --input data/knowledge_ablation/v2/trace_text_plus_anchors.jsonl \
  --output-dir data/evidence_interventions/v2 \
  --intervention passage_shuffle \
  --intervention same_topic_wrong \
  --intervention remove_warnings \
  --intervention remove_competing_pathways
```

After generating all six prediction files:

```bash
python scripts/evaluate_knowledge_ablation.py \
  --reference data/knowledge_ablation/v2/trace_textbook_rag.jsonl \
  --condition trace_no_knowledge=outputs/h3/trace_no_knowledge.jsonl \
  --condition trace_length_matched_irrelevant=outputs/h3/irrelevant.jsonl \
  --condition trace_textbook_rag=outputs/h3/textbook.jsonl \
  --condition trace_structured_anchors=outputs/h3/anchors.jsonl \
  --condition trace_text_plus_anchors=outputs/h3/combined.jsonl \
  --condition direct_textbook_rag=outputs/h3/direct.jsonl \
  --output outputs/h3/summary.json
```

The evaluator checks matched generation contracts while reporting condition-specific adapter lineage.

### Metrics

Implemented prediction-level metrics include:

```text
structural precursor Top-1/5/10
mapped structural precursor Top-1/5/10
ExecutePass@1/5/10
TraceBoundPass@1/5/10
coverage and selective risk
abstention rate
tool-failure recovery
retrieval Recall@K / Precision@K when frozen gold passage IDs exist
retrieval latency
missing-prediction and re-execution error rates
```

Reaction-center and synthon metrics remain explicitly unavailable until frozen reference labels are supplied; they are not fabricated from endpoints.

## Current status

| Component | Status |
|---|---|
| Deterministic proof executor | available |
| Trace-owned move replay and proof compilation | available |
| Explicit TRL tool facades | available |
| Root-import-preserving proof-to-trace conversion | available |
| Inference-faithful textbook query | available |
| Six matched Tool-SFT conditions | available |
| Strict tokenizer/mask and adapter-lineage contracts | available |
| Canonical multi-turn inference runner | available |
| H1 intervention runner and evaluator | available |
| Source-to-sink MechComp-OOD | available |
| H3 frozen evidence controls and evaluator | available |
| Paper-scale checkpoints and frozen results | not released |
| Experimental feasibility or kinetic validation | external evidence required |

The infrastructure now supports the complete experimental data flow, but no H1, H2, or H3 conclusion is claimed before full-data conversion coverage, real small-set overfit, frozen pilots, and multi-seed paper-scale experiments are completed.

## Documentation

- [`docs/SCIENTIFIC_THESIS.md`](docs/SCIENTIFIC_THESIS.md) — authoritative scientific claims and boundaries
- [`docs/EXECUTION_PLAN.md`](docs/EXECUTION_PLAN.md) — ordered operational gates
- [`docs/TRACE_FAITHFULNESS.md`](docs/TRACE_FAITHFULNESS.md) — H1 and trace contract
- [`docs/TOOL_SFT.md`](docs/TOOL_SFT.md) — replay-verified supervision and runtime schema
- [`docs/KNOWLEDGE_ABLATIONS.md`](docs/KNOWLEDGE_ABLATIONS.md) — H3 conditions and controls
- [`docs/PROOF_EQUIVALENCE.md`](docs/PROOF_EQUIVALENCE.md) — H2 move-composition split and equivalence
- [`docs/FORWARD_ELECTRON_EXPERT.md`](docs/FORWARD_ELECTRON_EXPERT.md) — optional independent soft evidence
- [`docs/README.md`](docs/README.md) — documentation map
