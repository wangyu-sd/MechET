# Execution plan

> **Role:** operational source of truth for the MechET study  
> **Principle:** no phase begins until the preceding scientific gate passes  
> **Default scope:** mapped, closed-shell, two-electron polar organic chemistry

## Phase map

| Phase | Objective | Primary artifact | Gate |
|---|---|---|---|
| **0** | Freeze code, data, model, and reporting contracts | Reproducibility manifest | All revisions and seeds explicit |
| **1** | Build executable proof data and remove benchmark overlap | Clean proof dataset | Execution, endpoint, and leakage audits pass |
| **2** | Measure proof-to-trace coverage and build evidence assets | Replay-verified source conditions | Coverage supports the declared chemistry scope |
| **3** | Derive six matched evidence conditions | Frozen suite manifest | IDs, endpoints, schemas, tokens, and budgets align |
| **4** | Establish real Tool-SFT learnability | Adapter manifest and pilot report | Tool syntax and held-out completion improve |
| **5** | Test H1 causal faithfulness | Normal/intervention prediction artifacts | Strict trace integrity and paired sensitivity |
| **6** | Test H2 compositional generalization | Frozen composition-OOD split | Known primitives, unseen complete compositions |
| **7** | Test H3 evidence separation | Six-condition evaluation | Evidence exceeds trace-only and matched context controls |
| **8** | Scale or extend | Scale/forward/planning results | H1–H3 pilots already passed |

---

## Phase 0 — Freeze contracts

### Objective

Make every later artifact attributable to one immutable experimental state.

### Freeze

```text
repository commit
source revisions, licenses, and content hashes
benchmark and dataset SHA-256 files
executor revision = MECH_PROOF_v1_move_bound
environment revision
base-model and tokenizer commit revisions
adapter lineage
random seeds and seed policy
headline tool-call budget = 16
candidate generation and selector semantics
```

### Required artifact

A machine-readable run manifest containing code, data, model, tokenizer, adapter, environment, executor, and seed revisions.

### Gate

- all CI workflows pass;
- `README.md` and `SCIENTIFIC_THESIS.md` agree;
- no reported model or tokenizer is referenced only by a mutable name.

### Stop

Do not train when a revision, seed, source license, or benchmark hash is unresolved.

---

## Phase 1 — Build executable proof data

### Objective

Construct a formally executable proof dataset before introducing tool learning.

### Commands

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

Every accepted row must execute and store:

```text
full_precursor_state
structural_precursor
auxiliary_fragments
```

Audit benchmark overlap and freeze a clean training set:

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

### Required artifacts

```text
source and benchmark manifests
accepted/quarantined row counts
proof execution diagnostics
overlap matrices
clean dataset manifest
endpoint-view distributions
```

### Gate

All retained rows execute; endpoint views are populated; benchmark overlap is disclosed and removed according to the frozen policy.

### Stop

Do not continue if execution failures are silently dropped or if benchmark overlap remains unresolved.

---

## Phase 2 — Build evidence assets and measure proof-to-trace coverage

### Objective

Establish the actual chemistry coverage of the trace-owned representation before training a model.

### Evidence assets

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

### Replay-verified source conditions

```bash
python scripts/build_textbook_tool_sft.py \
  --input data/mechet_proof_clean/train.jsonl \
  --corpus knowledge/corpus/passages.jsonl \
  --output data/textbook_tool_sft/train.jsonl \
  --query-mode state

python scripts/build_textbook_tool_sft.py \
  --input data/mechet_proof_clean/train.jsonl \
  --corpus knowledge/corpus/passages.jsonl \
  --output data/textbook_tool_sft/train_text_and_anchors.jsonl \
  --enable-structured-primitives \
  --query-mode state
```

`--query-mode state` is the inference-faithful headline condition. `label_oracle` is an explicitly named upper bound and must not enter headline results.

### Required coverage report

```text
root imports preserved
proof rows read, written, and quarantined
stable quarantine reason codes
TOOL_BUDGET_EXCEEDED counts
conversion rate by mechanism family
trace steps, source-to-sink moves, and imports
endpoint replay rate
retained versus rejected structural complexity
```

### Gate

The retained family and complexity distribution supports the declared scope. Every accepted row fits the frozen 16-call headline budget.

### Stop

Narrow the scientific scope or extend the converter if accepted data collapse to a small, systematically simpler subset.

---

## Phase 3 — Derive and validate six matched conditions

### Objective

Create H3 conditions that differ only in the declared evidence intervention.

### Build

```bash
python scripts/build_knowledge_ablation_suite.py \
  --config configs/experiments/textbook_ablation.yaml
```

### Validate

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

### Required artifacts

```text
same stable-ID universe
same targets and endpoint references
no gold reaction-label retrieval query
valid tool-call/result pairing
real tokenizer rendering
non-empty assistant masks
zero headline truncation
frozen evidence character budgets
input-token and supervised-token distributions
normalization multipliers when used
```

### Gate

The validator passes without assuming raw direct and tool syntax lengths are equal. Compute differences are disclosed using real tokenizer and assistant-mask tokens.

### Stop

Do not train conditions whose stable IDs, endpoint references, tool budgets, or tokenizer contracts differ silently.

---

## Phase 4 — Establish Tool-SFT learnability

### Objective

Demonstrate that the interaction contract is learnable before paper-scale on-policy training.

### Pilot

```bash
python scripts/train_tool_sft.py \
  --config configs/knowledge/tool_sft_trace_no_knowledge.yaml \
  --limit 32 \
  --max-steps 100
```

After the pilot succeeds, run the matched configurations:

```text
tool_sft_trace_no_knowledge.yaml
tool_sft_irrelevant.yaml
tool_sft_textbook.yaml
tool_sft_anchors.yaml
tool_sft_combined.yaml
tool_sft_direct_textbook.yaml
```

### Required artifacts

```text
data_contract.json
adapter_manifest.json
adapter SHA-256
base-model and tokenizer commit revisions
seed and data seed
assistant-mask and token audit
loss curve
valid tool-call rate
finish_trace rate
held-out execution rate
```

### Gate

Loss falls and held-out valid tool use, `finish_trace`, and execution improve relative to initialization.

### Stop

A schema-only dry run is not evidence of learnability. Do not start GRPO from an untrained or revision-ambiguous tool policy.

---

## Phase 5 — Test H1 causal faithfulness

### Objective

Determine whether model behavior causally depends on information returned by the environment.

### Optional on-policy refinement

```bash
python scripts/train_inverse_agent_trace.py \
  --config configs/agent/inverse_trace_grpo.yaml
```

### Normal prediction artifact

```bash
python scripts/infer_mechet.py \
  --config configs/agent/inverse_trace_grpo.yaml \
  --data data/benchmarks/h1/test.jsonl \
  --output outputs/h1/normal.jsonl \
  --mode trace \
  --condition-name trace_no_knowledge \
  --intervention none \
  --samples-per-target 4 \
  --seed 17 \
  --resume
```

Repeat with the identical model, adapter, model/tokenizer revisions, global seed, seed policy, candidate count, temperature, top-p, token limit, iteration limit, and frozen IDs:

```text
remove_tool_observations
stale_tool_observations
shuffle_tool_observations
disable_inspect_state
disable_intermediate_execution
```

For shuffle, pass the normal artifact through `--intervention-source`.

### Evaluate

```bash
python scripts/evaluate_faithfulness.py \
  --reference data/benchmarks/h1/test.jsonl \
  --normal outputs/h1/normal.jsonl \
  --intervention remove_tool_observations=outputs/h1/remove.jsonl \
  --intervention stale_tool_observations=outputs/h1/stale.jsonl \
  --intervention shuffle_tool_observations=outputs/h1/shuffle.jsonl \
  --output outputs/h1/summary.json
```

### Gate

```text
all frozen IDs evaluated
missing predictions retained as failures
normal completed predictions explicitly use finish_trace
trace, moves, proof, and endpoint recompute without error
runtime metadata complete and identical across interventions
intervention construction audited
paired effects reported with uncertainty
```

### Stop

If observation interventions have no material paired effect, do not claim tool-grounded causal reasoning.

---

## Phase 6 — Test H2 compositional generalization

### Objective

Hold out complete source-to-sink move compositions while retaining all constituent primitives in training.

### Build split

```bash
python scripts/build_mechcomp_ood.py \
  --input data/knowledge_ablation/v2/trace_no_knowledge.jsonl \
  --output-dir data/ood/mechcomp_source_sink \
  --test-fraction 0.10 \
  --valid-fraction 0.10 \
  --min-train-primitive-count 5 \
  --seed 42
```

### Required manifest

```text
primitive_basis = source_to_sink_execution_moves_v1
non-empty validation and test sets
zero train/test complete-composition overlap
all test primitives seen in train
requested and achieved split fractions
exact product and reaction overlap audit
scaffold and reaction-center overlap strata
```

### Evaluation

Train and evaluate direct, CoT, net-edit, complete-proof, and trace-owned representations on the same frozen examples and budgets.

### Gate

Primitive coverage, composition novelty, and structural novelty are separately reported. No held-out primitive is mislabeled as composition OOD.

### Stop

Do not claim H2 when the test set is empty, contains unseen primitives, or is dominated by unreported structural near-duplicates.

---

## Phase 7 — Test H3 evidence separation

### Objective

Determine whether mechanistic evidence adds information beyond trace ownership and context length.

### Build evidence-content interventions

```bash
python scripts/build_evidence_interventions.py \
  --input data/knowledge_ablation/v2/trace_text_plus_anchors.jsonl \
  --output-dir data/evidence_interventions/v2 \
  --intervention passage_shuffle \
  --intervention same_topic_wrong \
  --intervention remove_warnings \
  --intervention remove_competing_pathways
```

Generate prediction artifacts using modes:

```text
trace
irrelevant
textbook
anchors
combined
direct
```

Evidence modes replay row-specific frozen evidence so direct and trace comparisons receive the same bounded content.

### Evaluate

```bash
python scripts/evaluate_knowledge_ablation.py \
  --reference data/knowledge_ablation/v2/trace_textbook_rag.jsonl \
  --condition trace_no_knowledge=outputs/h3/trace.jsonl \
  --condition trace_length_matched_irrelevant=outputs/h3/irrelevant.jsonl \
  --condition trace_textbook_rag=outputs/h3/textbook.jsonl \
  --condition trace_structured_anchors=outputs/h3/anchors.jsonl \
  --condition trace_text_plus_anchors=outputs/h3/combined.jsonl \
  --condition direct_textbook_rag=outputs/h3/direct.jsonl \
  --output outputs/h3/summary.json
```

### Gate

```text
all frozen IDs evaluated; missing predictions count as failures
no supervision rows accepted as predictions
trace outputs require explicit successful finish_trace
same base/model revision and generation contract across conditions
condition-specific adapter hashes and token-normalized compute reported
textbook > trace-only and textbook > irrelevant for a text-evidence claim
combined > each individual evidence condition for a combined claim
```

### Stop

A gain explained by irrelevant context, label leakage, post-test evidence editing, missing predictions, or runtime mismatch does not support H3.

---

## Phase 8 — Scale, forward evidence, and planning

### Objective

Extend a validated core rather than use scale or downstream search to substitute for missing scientific evidence.

Only after H1–H3 pilots pass:

```text
0.6B / 1.7B / 8B scale study
formal-process RL
calibrated forward closure with explicit competitors
K = {1, 4, 16, 64}
multistep planning under frozen candidate pools
```

The forward expert remains frozen soft evidence. Planning is a downstream extension and cannot rescue failed H1 or H2 results.

---

## Prediction metrics

Candidate generations are unranked unless a frozen selector score is stored. Report:

```text
StructuralEndpointPass@1/5/10
MappedEndpointPass@1/5/10
ExecutePass@1/5/10
TraceBoundPass@1/5/10
coverage and selective risk
abstention rate
tool-failure recovery
retrieval Recall@K / Precision@K with frozen labels
retrieval latency
missing-prediction and re-execution error rates
```

Reaction-center and synthon metrics remain unavailable until frozen labels exist.

## Global stopping rules

Stop or narrow a claim when:

- proof-to-trace coverage is systematically narrow;
- a headline retrieval query uses gold reaction labels;
- root imports or declared moves do not replay;
- examples exceed the frozen tool budget;
- tokenizer masks are empty or headline examples truncate;
- adapter manifests, hashes, or model revisions do not match;
- missing predictions are removed from the denominator;
- a trace condition receives credit without explicit `finish_trace`;
- runtime metadata are incomplete or differ across a claimed comparison;
- H1 is insensitive to tool observations;
- H2 contains unseen primitives rather than unseen compositions;
- irrelevant text explains an evidence gain;
- a learned score overrides deterministic execution.

## Integrity utilities added after the pilot contract

Before a final H1/H2/H3 result package:

```bash
python scripts/check_documentation_integrity.py --output outputs/documentation_integrity.json
python scripts/check_source_health.py --output outputs/source_health.json
python scripts/aggregate_evaluation_seeds.py \
  --evaluation outputs/seed1/evaluation.json \
  --evaluation outputs/seed2/evaluation.json \
  --evaluation outputs/seed3/evaluation.json \
  --output outputs/multi_seed_summary.json
```

The H2 split manifest must include the structural-overlap audit and stratified composition-OOD counts. H1/H3 result artifacts must include paired uncertainty and corrected tests; the multi-seed aggregate must satisfy the declared seed-count, interval and direction gates.
