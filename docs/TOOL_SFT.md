# Replay-verified Tool-SFT

## Purpose

Tool-SFT teaches the trace-owned interaction contract before any paper-scale on-policy training.

```text
executable MECH_PROOF v1
  -> conservative source-to-sink conversion
  -> root and edge imports
  -> environment replay
  -> finish_trace
  -> compiled proof and executor-derived endpoint
```

Only rows that replay through the inference environment are accepted. Ambiguous electron pairing, unsupported proof topology, failed imports, move mismatch and terminal replay failure are quarantined with stable reason codes.

## Retrieval query contract

Headline textbook rows use only inference-available molecular information:

```bash
python scripts/build_textbook_tool_sft.py \
  --input data/mechet_proof_clean/train.jsonl \
  --corpus knowledge/corpus/passages.jsonl \
  --output data/textbook_tool_sft/train.jsonl \
  --query-mode state
```

`--query-mode state` derives terms from the target or current state. `--query-mode label_oracle` is an explicitly named oracle upper bound and cannot support the headline evidence claim.

## Conversational schema

Every row contains top-level `messages` and `tools` fields. Assistant tool calls use JSON-object arguments, not serialized JSON strings. Every call has exactly one matched tool result, and tool results cannot appear without an assistant call. The initial environment observation is part of the user turn rather than an orphan tool message.

Trace-owned rows include exactly one successful `finish_trace` and record:

```text
target_smiles
full_precursor_state
structural_precursor
auxiliary_fragments
initial_imports and transition moves
source-to-sink primitive signatures
compiled proof
trace_digest and move_sequence_digest
executor_replayed = true
endpoint_source = environment_owned_trace
```

## Six matched SFT conditions

```text
configs/knowledge/tool_sft_trace_no_knowledge.yaml
configs/knowledge/tool_sft_irrelevant.yaml
configs/knowledge/tool_sft_textbook.yaml
configs/knowledge/tool_sft_anchors.yaml
configs/knowledge/tool_sft_combined.yaml
configs/knowledge/tool_sft_direct_textbook.yaml
```

The no-knowledge, irrelevant-text, anchors-only and direct open-book rows are derived from the same frozen stable-ID intersection as the two source conditions.

## Tokenizer and supervision audit

A real run renders each conversation through the frozen tool-aware chat template and verifies:

```text
valid messages and tools
JSON-object arguments
paired calls and results
non-empty assistant masks
non-zero supervised tokens
zero truncation for headline rows
input-token and supervised-token distributions
```

A dry-run validates schemas but does not establish learnability. Before scale, overfit 32–128 examples and verify falling loss and improving valid tool use.

## Adapter lineage

Every run writes `data_contract.json` and `adapter_manifest.json`. The latter records a non-self-referential adapter SHA-256, base model, condition, data contract, environment revision and executor revision. Required GRPO configurations validate these fields before loading the Tool-SFT adapter as trainable. From-base RL is a separately named ablation.

## Interpretation

Tool-SFT success establishes that the interaction contract can be learned. H1 still requires causal interventions, H2 requires source-to-sink composition holdouts, and H3 requires matched frozen evidence predictions.