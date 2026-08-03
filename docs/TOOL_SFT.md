# Replay-verified Tool-SFT

## Purpose

Tool-SFT teaches the interaction contract required by the causal main method. It is not merely a formatting stage and it must precede paper-scale on-policy training.

```text
executable MECH_PROOF v1
  + frozen evidence assets
  -> conservative proof-to-action conversion
  -> explicit imports and source-to-sink actions
  -> replay through TraceOwnedAgentEnv
  -> finish_trace
  -> environment-compiled proof
  -> executor-derived endpoint
  -> accepted Tool-SFT row
```

Only trajectories that replay through the same environment used at inference are retained.

## Conversion scope

`proof_to_trace_plan` currently accepts linear proof paths when two-electron source/sink pairings are uniquely recoverable:

```text
LP loss + bond increase                 -> LP to BOND
bond decrease + LP gain                 -> BOND to ATOM
one unique local bond decrease/increase -> BOND to BOND
```

The builder does not invent ambiguous arrows.

Stable quarantine families include:

```text
NONLINEAR_PROOF_UNSUPPORTED
CYCLIC_PROOF_UNSUPPORTED
AMBIGUOUS_ELECTRON_PAIRING
UNPAIRED_LONE_PAIR_DELTA
ODD_LONE_PAIR_DELTA
EDGE_HAS_NO_INFERABLE_MOVES
IMPORT_REPLAY_FAILED
MOVE_REPLAY_FAILED
MOVE_REPLAY_STATE_MISMATCH
TRACE_TERMINAL_REPLAY_FAILED
```

The paper scope must follow measured conversion coverage. A narrow retained subset must not be described as broad organic chemistry.

## Build source conditions

### Textbook-only trace rows

```bash
python scripts/build_textbook_tool_sft.py \
  --input data/mechet_proof_clean/train.jsonl \
  --corpus knowledge/corpus/passages.jsonl \
  --output data/textbook_tool_sft/train.jsonl \
  --quarantine data/textbook_tool_sft/quarantine.jsonl
```

### Textbook plus structured mechanistic knowledge anchors

```bash
python scripts/build_textbook_tool_sft.py \
  --input data/mechet_proof_clean/train.jsonl \
  --corpus knowledge/corpus/passages.jsonl \
  --output data/textbook_tool_sft/train_text_and_anchors.jsonl \
  --enable-structured-primitives
```

These are the only source files required by the matched evidence suite. No-knowledge, irrelevant text, anchors-only and direct open-book rows are derived automatically.

## Build all matched conditions

```bash
python scripts/build_knowledge_ablation_suite.py \
  --config configs/experiments/textbook_ablation.yaml
```

The suite derives:

```text
trace_no_knowledge                  from textbook trace rows
trace_length_matched_irrelevant     from textbook trace rows
trace_textbook_rag                  source condition
trace_structured_anchors            from combined rows by removing textbook retrieval
trace_text_plus_anchors             source condition
direct_textbook_rag                 from textbook rows using the same bounded evidence card
```

All conditions preserve stable IDs, targets and structural endpoints.

## Accepted row contract

Each trace-owned row stores:

```text
chat messages with tool calls and results
target and expected structural precursor
original proof hash
conservative trace plan
compiled trace proof
trace digest
retrieval query and bounded evidence hash
knowledge condition
executor_replayed = true
endpoint_source = environment_owned_trace
```

The direct open-book baseline stores the same target, endpoint and bounded textbook evidence but no chemistry tools or trace-derived endpoint.

## Required conversion report

Before model training report:

```text
rows read
rows with parseable proofs
rows with executable proofs
rows converted and replayed
quarantined rows
quarantine counts by stable reason code
conversion rate by reaction family
conversion rate by proof length
conversion rate by chain/tree/DAG topology
changed-atom and changed-bond complexity
imports and moves per trace
retrieval passage coverage
context length
endpoint replay rate
```

The accepted and rejected distributions must both be released.

## Matched training contract

Hold constant where applicable:

```text
base model and tokenizer revision
stable training IDs
LoRA rank and target modules
optimizer and schedule
number of updates
effective batch size
random seeds
maximum input and completion length
tool budget for trace-owned conditions
executor and environment revision
```

Report both character budgets and tokenizer-specific input/supervised token counts.

```bash
python scripts/validate_experiment_contract.py \
  --condition none=data/knowledge_ablation/v2/trace_no_knowledge.jsonl \
  --condition irrelevant=data/knowledge_ablation/v2/trace_length_matched_irrelevant.jsonl \
  --condition textbook=data/knowledge_ablation/v2/trace_textbook_rag.jsonl \
  --condition anchors=data/knowledge_ablation/v2/trace_structured_anchors.jsonl \
  --condition combined=data/knowledge_ablation/v2/trace_text_plus_anchors.jsonl \
  --condition direct=data/knowledge_ablation/v2/direct_textbook_rag.jsonl \
  --output outputs/contracts/evidence_conditions.json
```

## Real training smoke test

A dry-run verifies schemas but not learnability. Before paper-scale training, overfit 32–128 examples and confirm:

```text
assistant-only supervision mask is non-empty
loss decreases
valid tool-call syntax increases
finish_trace call rate increases
trace-bound execution increases
endpoint exact approaches the small-set ceiling
```

A tokenizer/chat template that cannot preserve assistant/tool boundaries is not acceptable.

## Training

```bash
python scripts/train_tool_sft.py \
  --config configs/knowledge/tool_sft_textbook.yaml \
  --dry-run

python scripts/train_tool_sft.py \
  --config configs/knowledge/tool_sft_textbook.yaml
```

Equivalent configs are required for the matched trace conditions.

## Tool-SFT to RL lineage

On-policy training starts only after a frozen Tool-SFT checkpoint shows credible executable learning.

Every RL config and checkpoint must record:

```text
initial_adapter_path
initial_adapter_sha256
Tool-SFT data-manifest hash
base-model/tokenizer revision
environment revision
executor revision
reward config and seed
```

The primary condition must not silently restart from the base model.

## Scientific interpretation

Tool-SFT success alone does not prove causal reasoning or compositional generalization. It establishes that the model can learn the trace-owned interaction contract. H1 requires causal interventions; H2 requires execution-primitive composition holdouts; H3 requires matched evidence controls.
