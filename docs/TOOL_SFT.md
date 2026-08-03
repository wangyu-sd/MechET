# Replay-verified textbook Tool-SFT

## Goal

The first small-model training stage should not begin with unconstrained online RL. MechET converts existing executable proofs into tool-interleaved trajectories and retains only traces that replay through the same environment used at inference.

```text
executable MECH_PROOF v1
+ frozen textbook corpus
        ↓
conservative source-sink pairing
        ↓
textbook retrieval tool call
        ↓
import and electron-flow tool calls
        ↓
finish_trace
        ↓
executor replay and endpoint check
        ↓
Tool-SFT JSONL
```

## Conservative conversion

`proof_to_trace_plan` currently supports linear proof paths. It pairs proof deltas only when the two-electron source and sink are uniquely recoverable:

```text
LP loss + bond increase        → LP to BOND
bond decrease + LP gain        → BOND to ATOM
one local bond decrease/increase pair → BOND to BOND
```

Ambiguous pairings, unpaired lone-pair changes, nonlinear branches and failed replays are quarantined. The builder does not invent arrow labels.

## Build

```bash
python scripts/build_textbook_tool_sft.py \
  --input data/mechet_proof_clean/train.jsonl \
  --corpus knowledge/corpus/passages.jsonl \
  --output data/textbook_tool_sft/train.jsonl \
  --quarantine data/textbook_tool_sft/quarantine.jsonl
```

Optional structured-anchor condition:

```bash
python scripts/build_textbook_tool_sft.py \
  --input data/mechet_proof_clean/train.jsonl \
  --corpus knowledge/corpus/passages.jsonl \
  --output data/textbook_tool_sft/train_text_and_anchors.jsonl \
  --enable-structured-primitives
```

## Row contract

Each accepted row stores:

```text
chat messages with tool calls and tool results
target and executor-derived precursor
original proof hash
conservative trace plan
compiled trace proof
trace digest
retrieval query
passage IDs and context hash
structured-anchor condition
endpoint_source = environment_owned_trace
executor_replayed = true
```

## Required reports

```text
rows read/written/quarantined
quarantine reasons
proof topology coverage
mechanism-family coverage
number of imported fragments
move counts and trace length
retrieval passage coverage
context length
endpoint replay rate
```

The first training comparison is Tool-SFT without textbook evidence versus Tool-SFT with retrieved evidence under matched row IDs and assistant-token budgets.
