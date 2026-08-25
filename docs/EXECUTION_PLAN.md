# Execution plan

> **Role:** operational plan subordinate to the paper experiment matrix
> **Principle:** no phase begins until the preceding scientific gate passes  
> **Default scope:** mapped, closed-shell, two-electron polar organic chemistry

## Paper-first phase map

| Phase | Objective | Primary artifact | Gate |
|---|---|---|---|
| **0** | Freeze code, data, model, compute and reporting contracts | Reproducibility manifest | All revisions, hashes, budgets and seeds explicit |
| **R1** | Two-corpus overall inverse performance | Matched endpoint/execution table | FlowER and mech-USPTO results use frozen denominators |
| **R2** | Reasoning provenance and endpoint consistency | CEIR/TraceBound audit | Reasoning parsers and execution checks are frozen |
| **R3** | Counterfactual state adaptation | Paired accurate-vs-stale intervention | Positive SAE under the same actual environment state |
| **R4** | MechComp-OOD C1/C2/C3 | Fresh split-specific checkpoints | C2 has seen primitives/bigrams and unseen full programs |
| **R5** | Data efficiency | 5/10/25/50/100% learning curves | Fixed subset seeds and matched token/step reporting |
| **R6** | Structural OOD and transfer | OOD retention and bidirectional transfer | Exact-clean lineage and map controls pass |
| **R7** | Theory-linked analysis and recovery | Prespecified regression and recovery curves | Covariates and intervention windows are frozen |

Mandatory internal conditions are A0 Direct, A1 Free-CoT, A2 State-CoT, A3
NetEdit, A4 OpenFlow, A5 Loose trace + answer, A6 MechSMILES-format and A7
MechET. Required ablations are B1--B5 as defined in
[`PAPER_EXPERIMENT_PROTOCOL.md`](PAPER_EXPERIMENT_PROTOCOL.md).

The older numbered implementation phases below remain useful for building data
and runtime artifacts, but they do not override R1--R7 priority. In particular,
the H3/textbook phase is outside the current ICLR protocol.

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

A configuration may request a human-readable Hugging Face revision such as `main`, but a real Tool-SFT run must resolve it to the actual full 40-hex commit SHA. `adapter_manifest.json` stores both the requested revision and immutable resolved revision. Required GRPO runs inherit the immutable adapter revision; a mutable adapter revision is a hard error.

### Gate

- all CI workflows pass;
- `README.md` and `SCIENTIFIC_THESIS.md` agree;
- no reported remote model or tokenizer remains referenced only by a mutable name.

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
full_precursor_state / structural_precursor / auxiliary_fragments
```

### Gate

All retained rows execute; endpoint views are populated; benchmark overlap is disclosed and removed according to the frozen policy.

### Stop

Do not continue if execution failures are silently dropped or benchmark overlap remains unresolved.

---

## Phase 2 — Build replay-verified train/valid/test evidence sources

### Objective

Establish chemistry coverage and create evidence-bearing source rows without mixing training and final evaluation examples.

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

Build both textbook-only and textbook+anchor source rows for every frozen proof split:

```bash
for split in train valid test; do
  python scripts/build_textbook_tool_sft.py \
    --input data/mechet_proof_clean/${split}.jsonl \
    --corpus knowledge/corpus/passages.jsonl \
    --output data/textbook_tool_sft/${split}.jsonl \
    --query-mode state

  python scripts/build_textbook_tool_sft.py \
    --input data/mechet_proof_clean/${split}.jsonl \
    --corpus knowledge/corpus/passages.jsonl \
    --output data/textbook_tool_sft/${split}_text_and_anchors.jsonl \
    --enable-structured-primitives \
    --query-mode state
done
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

Narrow the scientific scope or extend the converter if accepted data collapse to a systematically simpler subset.

---

## Phase 3 — Derive and validate split-isolated H3 conditions

### Objective

Create the six H3 conditions independently inside train, valid, and test so final evaluation never reuses train-derived evidence rows.

### Build

```bash
python scripts/build_knowledge_ablation_suite.py \
  --config configs/experiments/textbook_ablation.yaml
```

This writes:

```text
data/knowledge_ablation/v2/train/*.jsonl
data/knowledge_ablation/v2/valid/*.jsonl
data/knowledge_ablation/v2/test/*.jsonl
data/knowledge_ablation/v2/{train,valid,test}/manifest.json
data/knowledge_ablation/v2/manifest.json
```

The top manifest hard-checks zero stable-ID overlap between train, valid, and test.

### Validate the training contract

```bash
python scripts/validate_experiment_contract.py \
  --model-name Qwen/Qwen3-0.6B \
  --condition trace_no_knowledge=data/knowledge_ablation/v2/train/trace_no_knowledge.jsonl \
  --condition trace_length_matched_irrelevant=data/knowledge_ablation/v2/train/trace_length_matched_irrelevant.jsonl \
  --condition trace_textbook_rag=data/knowledge_ablation/v2/train/trace_textbook_rag.jsonl \
  --condition trace_structured_anchors=data/knowledge_ablation/v2/train/trace_structured_anchors.jsonl \
  --condition trace_text_plus_anchors=data/knowledge_ablation/v2/train/trace_text_plus_anchors.jsonl \
  --condition direct_textbook_rag=data/knowledge_ablation/v2/train/direct_textbook_rag.jsonl \
  --output outputs/contracts/evidence_conditions_train.json
```

Repeat the same validation for `valid/` and `test/` before model selection or final evaluation.

### Required artifacts

```text
same stable-ID universe within each split
zero stable-ID overlap across train/valid/test
same targets and endpoint references within matched conditions
no gold reaction-label retrieval query
valid tool-call/result pairing
real tokenizer rendering
non-empty assistant masks
zero headline truncation
frozen evidence character budgets
input-token and supervised-token distributions
```

### Gate

All three split manifests pass. Do not use a train-derived suite as H3 reference data.

### Stop

Do not train or evaluate conditions whose stable IDs, endpoint references, tool budgets, tokenizer contracts, or split identities differ silently.

---

## Phase 4 — Establish Tool-SFT learnability

### Objective

Demonstrate that the interaction contract is learnable before paper-scale on-policy training.

### Qwen3 tokenizer contract

The six matched Tool-SFT configurations use:

```text
max_length = 12288
assistant_mask_method = final_chatml_token_scan_v1
packing = false
assistant_only_loss = true
```

Qwen3 does not expose an automatic assistant-token mask through its shipped chat template. MechET renders each complete tool-bearing conversation once, tokenizes once, scans final ChatML assistant spans, and refuses any headline truncation.

### Pilot

```bash
python scripts/train_tool_sft.py \
  --config configs/knowledge/tool_sft_trace_no_knowledge.yaml \
  --limit 32 \
  --max-steps 100
```

After the pilot succeeds, train the matched configurations:

```text
tool_sft_trace_no_knowledge.yaml
tool_sft_irrelevant.yaml
tool_sft_textbook.yaml
tool_sft_anchors.yaml
tool_sft_combined.yaml
tool_sft_direct_textbook.yaml
```

Each configuration reads only `data/knowledge_ablation/v2/train/`.

### Required artifacts

```text
data_contract.json
adapter_manifest.json
adapter SHA-256
requested model revision
immutable resolved base-model and tokenizer commit revisions
seed and data seed
assistant-mask method and P50/P95/P99/max token audit
zero truncation
loss curve
valid tool-call rate
finish_trace rate
held-out execution rate
```

For Transformers 5.x, length grouping is applied through `train_sampling_strategy=group_by_length` when available; older compatible APIs may use `group_by_length`. The actual applied field is written to the data contract.

### Gate

Loss falls and held-out valid tool use, `finish_trace`, and execution improve relative to initialization.

### Stop

A schema-only dry run is not evidence of learnability. Do not start GRPO from an untrained, mutable-revision, or train/test-contaminated tool policy.

---

## Phase 5 — Test H1 causal faithfulness

### Objective

Determine whether model behavior causally depends on information returned by the environment.

### Freeze the H1 benchmark

Build H1 from the held-out trace condition, not from training rows:

```bash
python scripts/build_h1_benchmark.py \
  --input data/knowledge_ablation/v2/test/trace_no_knowledge.jsonl \
  --train-reference data/knowledge_ablation/v2/train/trace_no_knowledge.jsonl \
  --output-dir data/benchmarks/h1 \
  --max-tool-calls 16
```

The builder requires explicit successful `finish_trace`, executor replay, trace ownership, reference endpoints, the frozen tool budget, and zero stable-ID overlap with the supplied training reference.

### Optional on-policy refinement

Portable smoke/default:

```bash
python scripts/train_inverse_agent_trace.py \
  --config configs/agent/inverse_trace_grpo.yaml
```

Paper-scale throughput profile:

```bash
python scripts/train_inverse_agent_trace.py \
  --config configs/agent/inverse_trace_grpo_vllm.yaml
```

vLLM is an execution backend, not a scientific condition.

### Run the matched H1 suite

```bash
python scripts/run_h1_suite.py \
  --config configs/agent/inverse_trace_grpo.yaml \
  --data data/benchmarks/h1/test.jsonl \
  --out-dir outputs/h1 \
  --samples-per-target 4 \
  --seed 17 \
  --resume
```

The runner executes the normal condition plus:

```text
remove_tool_observations
stale_tool_observations
shuffle_tool_observations
disable_inspect_state
disable_intermediate_execution
```

It calls `scripts/infer_mechet.py` under the same model, adapter, immutable revision, seed policy, candidate count, temperature, top-p, token budget, and iteration budget, then calls `scripts/evaluate_faithfulness.py`.

### Gate

```text
all frozen IDs evaluated
missing predictions retained as failures
normal completed predictions explicitly use `finish_trace`
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

### Build the split before training the H2 model

```bash
python scripts/build_mechcomp_ood.py \
  --input data/knowledge_ablation/v2/train/trace_no_knowledge.jsonl \
  --output-dir data/ood/mechcomp_source_sink \
  --test-fraction 0.10 \
  --valid-fraction 0.10 \
  --min-train-primitive-count 5 \
  --seed 42
```

The reaction-center audit is step-state-aware: each local center is computed from `step.state_before + step.imports`. Atoms legitimately introduced on the precursor side are therefore not quarantined merely because they are absent from the product target.

### Train only on the H2 train split

```bash
python scripts/train_tool_sft.py \
  --config configs/agent/tool_sft_mechcomp_trace.yaml
```

Do **not** reuse an adapter trained on the full pre-split `trace_no_knowledge` data. `run_h2_suite.py` checks that `adapter_manifest.json::train_file_sha256` exactly matches `data/ood/mechcomp_source_sink/train.jsonl`.

### Headline trace-owned inference

```bash
python scripts/run_h2_suite.py \
  --split-dir data/ood/mechcomp_source_sink \
  --adapter outputs/h2/tool_sft_trace_qwen3_0_6b \
  --out-dir outputs/h2 \
  --samples-per-target 4 \
  --seed 17
```

The headline MechET condition uses `scripts/infer_mechet.py --mode trace`. `scripts/infer_mechet_proof.py` is only the independent complete-proof baseline; it is not the trace-owned H2 path. Direct, CoT, net-edit, complete-proof, and trace-owned baselines must each be trained on the same frozen H2/train IDs.

### Required manifest

```text
primitive_basis = source_to_sink_execution_moves_v1
non-empty validation and test sets
zero train/test complete-composition overlap
all test primitives seen in train
requested and achieved split fractions
exact product and reaction overlap audit
scaffold and step-state reaction-center overlap strata
near-duplicate audit
```

### Gate

Primitive coverage, composition novelty, and structural novelty are separately reported, including scaffold-seen/unseen, reaction-center-seen/unseen, and family-seen/unseen strata.

### Stop

Do not claim H2 when the test set is empty, contains unseen primitives, reuses a full-data-trained checkpoint, or is dominated by unreported structural near-duplicates.

---

## Legacy Phase 7 — Test H3 evidence separation (outside current ICLR)

### Objective

Determine whether mechanistic evidence adds information beyond trace ownership and context length on a frozen held-out test split.

### Six-condition held-out inference

```bash
python scripts/run_h3_suite.py \
  --suite-root data/knowledge_ablation/v2/test \
  --out-dir outputs/h3 \
  --samples-per-target 4 \
  --seed 17
```

The runner verifies that every condition-specific adapter was trained on the corresponding file under `data/knowledge_ablation/v2/train/`, runs all six `scripts/infer_mechet.py` modes, and calls `scripts/evaluate_knowledge_ablation.py` with `test/trace_textbook_rag.jsonl` as the frozen reference.

### Build evidence-content interventions

```bash
python scripts/build_evidence_interventions.py \
  --input data/knowledge_ablation/v2/test/trace_text_plus_anchors.jsonl \
  --output-dir data/evidence_interventions/v2 \
  --intervention passage_shuffle \
  --intervention same_topic_wrong \
  --intervention remove_warnings \
  --intervention remove_competing_pathways
```

`same_topic_wrong` may legitimately lack a donor for some examples. Those examples are quarantined **for that intervention only**. The builder writes:

```text
same_topic_wrong.jsonl
same_topic_wrong.reference.jsonl
same_topic_wrong.eligible_ids.json
same_topic_wrong.quarantine.jsonl   # only when needed
```

Any paired intervention analysis must use the generated `*.reference.jsonl`; it must not compare the reduced intervention set against the full baseline ID universe.

### Gate

```text
all frozen test IDs evaluated for the six headline conditions
missing predictions retained as failures
no supervision rows accepted as predictions
trace outputs require explicit successful finish_trace
same base/model revision and generation contract across compared conditions
condition-specific adapter hashes and token-normalized compute reported
textbook > trace-only and textbook > irrelevant for a text-evidence claim
combined > each individual evidence condition for a combined claim
subset evidence interventions use their paired eligible-ID reference
```

### Stop

A gain explained by irrelevant context, label leakage, train-derived evaluation, post-test evidence editing, missing predictions, unmatched eligible IDs, or runtime mismatch does not support H3.

---

## Phase 8 — Scale, forward evidence, and planning

Only after matched SFT, R3 and R4-C2 are frozen:

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
- adapter manifests, hashes, or immutable model revisions do not match;
- missing predictions are removed from the denominator;
- a trace condition receives credit without explicit `finish_trace`;
- runtime metadata are incomplete or differ across a claimed comparison;
- H1 is insensitive to tool observations;
- H2 contains unseen primitives rather than unseen compositions;
- an H2 checkpoint saw held-out composition examples during training;
- a future H3 evaluation uses train-derived rows;
- a subset evidence intervention is compared on unmatched IDs;
- irrelevant text explains an evidence gain;
- a learned score overrides deterministic execution.

## Integrity utilities

Before a final R1--R7 result package:

```bash
python scripts/check_documentation_integrity.py --output outputs/documentation_integrity.json
python scripts/check_source_health.py --output outputs/source_health.json
python scripts/aggregate_evaluation_seeds.py \
  --evaluation outputs/seed1/evaluation.json \
  --evaluation outputs/seed2/evaluation.json \
  --evaluation outputs/seed3/evaluation.json \
  --output outputs/multi_seed_summary.json
```

The R4 split manifest must include the structural-overlap audit and stratified
composition-OOD counts. R3 artifacts must include paired uncertainty and
corrected tests; every headline aggregate must satisfy the declared seed-count,
interval, and direction gates.
