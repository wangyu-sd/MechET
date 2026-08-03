# MechET execution plan

This is the operational run order for the causal and compositional MechET study. It intentionally postpones RL, forward reranking and planning until the main scientific hypotheses pass smaller falsification gates.

## Phase 0 — freeze the scientific contract

Before data processing or training:

1. record the repository commit, executor revision and environment revision;
2. freeze dataset revisions, licenses and SHA-256 manifests;
3. freeze benchmark files before model training;
4. define the primary scope as mapped, closed-shell, two-electron polar chemistry;
5. define execution primitives separately from mechanistic knowledge anchors;
6. declare the main method as `TraceOwnedAgentEnv` plus `finish_trace`;
7. retain independent complete-proof generation only as a baseline.

Gate: documentation-contract tests and all core CI workflows pass.

## Phase 1 — data feasibility and conversion coverage

### 1.1 Build executable proof rows

```bash
python scripts/build_mechet_proof_sft.py \
  --input-dir data/mechet_sft \
  --output-dir data/mechet_proof_sft \
  --splits train valid test
```

Every accepted proof must execute and reconstruct the stored structural endpoint.

### 1.2 Audit train–benchmark overlap

```bash
python scripts/audit_reaction_overlap.py \
  --train data/mechet_proof_sft/train.jsonl \
  --benchmark data/benchmarks/uspto50k/test.csv \
  --benchmark-format reaction_table \
  --reaction-field reaction_smiles \
  --out-dir outputs/data_audit/flower_vs_uspto50k_test
```

### 1.3 Build the clean proof dataset

```bash
python scripts/build_decontaminated_dataset.py \
  --train data/mechet_proof_sft/train.jsonl \
  --benchmark data/benchmarks/uspto50k/test.csv \
  --output data/mechet_proof_clean/train.jsonl \
  --manifest data/mechet_proof_clean/manifest.json \
  --policy exact_structural product
```

### 1.4 Build the provenance-aware textbook corpus

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

### 1.5 Build replay-verified Tool-SFT rows

Textbook-only:

```bash
python scripts/build_textbook_tool_sft.py \
  --input data/mechet_proof_clean/train.jsonl \
  --corpus knowledge/corpus/passages.jsonl \
  --output data/textbook_tool_sft/train.jsonl \
  --quarantine data/textbook_tool_sft/quarantine.jsonl
```

Textbook plus structured knowledge anchors:

```bash
python scripts/build_textbook_tool_sft.py \
  --input data/mechet_proof_clean/train.jsonl \
  --corpus knowledge/corpus/passages.jsonl \
  --output data/textbook_tool_sft/train_text_and_anchors.jsonl \
  --enable-structured-primitives
```

Required first report:

```text
proof rows read
executable proof rows
trace-convertible rows
conversion rate
quarantine reason codes
coverage by reaction family
coverage by proof length and topology
imports and moves per trace
endpoint replay rate
```

Gate: the retained dataset has sufficient family and complexity coverage for the intended paper scope. If not, narrow the scope or extend the converter before training.

## Phase 2 — construct matched scientific conditions

All evidence conditions are derived from the same stable-ID intersection.

```bash
python scripts/build_knowledge_ablation_suite.py \
  --config configs/experiments/textbook_ablation.yaml
```

The suite generates:

```text
trace_no_knowledge
trace_length_matched_irrelevant
trace_textbook_rag
trace_structured_anchors
trace_text_plus_anchors
direct_textbook_rag
```

Validate deterministic budgets and alignment:

```bash
python scripts/validate_experiment_contract.py \
  --condition trace_none=data/knowledge_ablation/v2/trace_no_knowledge.jsonl \
  --condition irrelevant=data/knowledge_ablation/v2/trace_length_matched_irrelevant.jsonl \
  --condition textbook=data/knowledge_ablation/v2/trace_textbook_rag.jsonl \
  --condition anchors=data/knowledge_ablation/v2/trace_structured_anchors.jsonl \
  --condition combined=data/knowledge_ablation/v2/trace_text_plus_anchors.jsonl \
  --condition direct=data/knowledge_ablation/v2/direct_textbook_rag.jsonl \
  --output outputs/contracts/evidence_conditions.json
```

Gate: identical IDs, targets and structural endpoints. Tokenizer-specific input and supervised token counts must be frozen before headline training.

## Phase 3 — real Tool-SFT smoke tests

Do not start paper-scale RL from an untrained tool policy.

### 3.1 Overfit a tiny set

Use 32–128 examples with Qwen3-0.6B. Confirm:

```text
non-empty assistant supervision mask
loss decreases
valid tool-call syntax increases
finish_trace call rate increases
trace_bound rate approaches one
endpoint exact approaches the small-set overfit ceiling
```

### 3.2 Train the matched Tool-SFT pilot

```bash
python scripts/train_tool_sft.py \
  --config configs/knowledge/tool_sft_textbook.yaml
```

Repeat with matched configs for no-knowledge, irrelevant text, anchors and combined evidence.

Gate: credible executable-learning signal on a frozen validation set. If Tool-SFT cannot learn the interaction contract, do not proceed to GRPO.

## Phase 4 — test H1: causal faithfulness

Train or evaluate the following matched conditions:

```text
outcome-only direct generation
free-form CoT plus answer
state-CoT plus answer
net-edit generation
independent complete MECH_PROOF generation
legacy loose tool trace plus submitted proof
trace-owned Tool-CoT with finish_trace
```

Required interventions:

```text
remove tool observations
shuffle tool observations
replace observations with stale states
disable inspect_state
disable intermediate move execution
allow independent proof submission only in the baseline
```

Primary metrics:

```text
structural precursor Top-k
ExecutePass
trace–proof agreement
trace–endpoint agreement
answer–reasoning disagreement
tool-failure recovery
intervention effect size
```

Gate: the trace-owned model must be causally sensitive to environment feedback and unable to preserve endpoint credit through an incompatible trace.

## Phase 5 — test H2: compositional generalization

Build primitive-seen/composition-unseen splits using electron-flow execution primitives, not knowledge-anchor IDs.

Compare:

```text
direct answer
free-form CoT
net edit
complete proof
trace-owned Tool-CoT
trace-owned Tool-CoT plus evidence
```

Report separately by:

```text
composition frequency
proof length
number of changed atoms and bonds
ring topology
chain/tree/DAG topology
reaction family
product scaffold
```

Gate: every held-out composition uses execution primitives represented in training above the declared minimum count.

## Phase 6 — test H3: empirical evidence separation

### 6.1 Textbook and anchor evidence

Evaluate the six matched evidence conditions and a frozen gold-passage upper bound when labels exist.

Claim gate:

```text
textbook > trace-only
and
textbook > length-matched irrelevant text
```

Causal evidence interventions:

```text
passage shuffle
same-topic wrong passage
remove warnings
remove competing-pathway text
```

### 6.2 Independent forward evidence

First validate the forward expert independently:

```text
source/sink accuracy
move MRR
target rank
competitor margin
Brier score
ECE
risk–coverage
```

Then compare ranking and reward conditions only after calibration:

```text
inverse score only
ordinary forward compatibility
forward process evidence
process plus explicit competitors
process plus competitors plus uncertainty
```

Formal execution remains a hard gate.

## Phase 7 — scale and optimization

After H1–H3 pilots pass:

```text
0.6B trace-owned actor
1.7B trace-owned actor
8B trace-owned actor
8B direct-answer reference
8B direct-answer plus identical bounded textbook evidence
```

Report accuracy, reliability, GPU hours, peak memory, generated tokens, tool calls, latency and verified endpoints per compute budget.

Only then run:

```text
Tool-SFT plus formal process RL
Tool-SFT plus endpoint RL
Tool-SFT plus calibrated forward evidence
```

Every RL checkpoint must record the Tool-SFT adapter hash, data manifest, environment revision and executor revision.

## Phase 8 — test-time hypotheses and planning extensions

Run K in `{1, 4, 16, 64}` for complete-proof and trace-owned inference when supported.

Report:

```text
ExecutePass@K
EndpointPass@K
unique executable proof classes
unique mechanism compositions
unique endpoints
latency and model/tool calls
```

Planning is an extension, not a prerequisite for the core scientific claim. Use frozen offline candidate pools before online actor planning.

## Required result order

The main paper result sequence is:

1. **Causal reasoning** — the trace determines the endpoint and responds to interventions;
2. **Compositional reasoning** — execution primitives generalize to unseen compositions;
3. **Evidence separation** — soft evidence improves supported choices without replacing execution;
4. **Scale and downstream reliability** — small-model efficiency and planning, if supported.

## Stop conditions

Stop or narrow a claim when:

- trace conversion covers only a narrow unreported subset;
- matched conditions differ in IDs or endpoints;
- tool observations can be removed without material effect;
- composition-OOD includes unseen primitives;
- textbook gains are explained by irrelevant text;
- a learned score overrides formal execution;
- RL starts from an untrained tool policy;
- forward evidence improves mined examples but degrades the frozen audit set.
