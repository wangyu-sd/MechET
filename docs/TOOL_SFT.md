# Replay-verified Tool-SFT

> **Purpose:** teach the trace-owned interaction contract before on-policy optimization  
> **Acceptance rule:** a row enters training only if the inference environment can replay it exactly

## Why Tool-SFT is necessary

A tool-aware chat schema is not, by itself, a learned policy. Before testing causal use, compositional generalization, or evidence benefit, the model must first learn to:

- inspect mapped states;
- import missing fragments;
- emit valid source-to-sink actions;
- use environment observations;
- terminate with `finish_trace`;
- avoid independent proof or answer channels.

Tool-SFT is therefore a **learnability gate**, not a positive H1/H2/H3 result.

## Data lineage

```text
executable MECH_PROOF v1
  -> conservative proof-to-trace conversion
  -> root and edge imports
  -> source-to-sink move pairing
  -> environment replay
  -> finish_trace
  -> compiled proof and executor-derived endpoint
  -> Tool-SFT conversation
```

Only replay-verified rows are accepted. Unsupported topology, ambiguous electron pairing, failed imports, move/state mismatch, terminal replay failure, and tool-budget overflow are quarantined with stable reason codes.

## Retrieval query contract

Headline textbook rows use only inference-available molecular information:

```bash
python scripts/build_textbook_tool_sft.py \
  --input data/mechet_proof_clean/train.jsonl \
  --corpus knowledge/corpus/passages.jsonl \
  --output data/textbook_tool_sft/train.jsonl \
  --query-mode state
```

`--query-mode state` derives terms from the target or current molecular state. `label_oracle` is an explicitly named upper bound and cannot support a headline evidence claim.

## Acceptance contract

A trace-owned row is accepted only when:

| Requirement | Evidence in the row |
|---|---|
| Stable example identity | Non-empty unique `id` |
| Executable conversation | Top-level `messages` and canonical `tools` schema |
| Valid tool syntax | JSON-object arguments, not serialized JSON strings |
| Complete interaction | Exactly one result for every tool call and no orphan result |
| Trace-owned endpoint | Exactly one successful `finish_trace` |
| Replay integrity | `executor_replayed = true`, trace and move-sequence digests |
| Endpoint integrity | `full_precursor_state`, `structural_precursor`, `auxiliary_fragments` |
| Budget compatibility | Required calls do not exceed the frozen 16-call headline budget |
| Query integrity | `gold_label_query_used = false` for headline conditions |

The initial environment observation is embedded in the user turn rather than represented as an orphan tool result.

## Conversation schema

A canonical trace row has the form:

```text
system: causal and evidence-boundary instructions
user: target + initial environment observation
assistant: tool call
 tool: JSON result
...
assistant: finish_trace
 tool: raw environment-owned terminal result
assistant: brief completion acknowledgement
```

Every assistant tool call contains:

```json
{
  "id": "call_000",
  "type": "function",
  "function": {
    "name": "inspect_state",
    "arguments": {}
  }
}
```

Arguments remain JSON objects throughout data loading, tokenization, training, inference, and audit.

## Six matched conditions

```text
configs/knowledge/tool_sft_trace_no_knowledge.yaml
configs/knowledge/tool_sft_irrelevant.yaml
configs/knowledge/tool_sft_textbook.yaml
configs/knowledge/tool_sft_anchors.yaml
configs/knowledge/tool_sft_combined.yaml
configs/knowledge/tool_sft_direct_textbook.yaml
```

The no-knowledge, irrelevant-text, anchors-only, and direct open-book rows are derived from the same frozen stable-ID intersection as the two replay-verified source conditions.

## Tokenizer and supervision audit

A real run renders every conversation through the frozen tool-aware chat template and verifies:

```text
valid messages and tools
JSON-object arguments
paired calls and results
non-empty assistant masks
non-zero supervised tokens
zero truncation for headline rows
input-token and supervised-token distributions
```

Raw direct and tool syntax lengths are not assumed equal. Comparisons disclose real tokenizer input and assistant-mask tokens, and use supervised-token-normalized compute when required.

## Learnability pilot

Start with a real 32–128-example overfit:

```bash
python scripts/train_tool_sft.py \
  --config configs/knowledge/tool_sft_trace_no_knowledge.yaml \
  --limit 32 \
  --max-steps 100
```

A credible pilot reports:

```text
training and validation loss
valid JSON/tool-call rate
unknown-tool and argument-error rates
finish_trace rate
formal execution rate
trace-bound rate
held-out endpoint performance
```

A schema-only dry run does not establish learnability.

## Training reproducibility

Tool-SFT resolves and records the immutable model/tokenizer commit. Training configuration includes:

```text
seed and data_seed
BF16 or FP16 policy
TF32 setting
gradient checkpointing
length grouping and dataloader workers
maximum sequence length
optimizer update count
checkpoint retention policy
```

Headline runs must preserve the frozen revision and seed policy across matched conditions.

## Adapter lineage

Every real run writes:

### `data_contract.json`

```text
training data path and SHA-256
condition and scientific hypothesis
row/tool-call statistics
assistant-mask and token audit
maximum tool calls per row
model and tokenizer revisions
seed and data seed
environment and executor revisions
```

### `adapter_manifest.json`

```text
artifact_type = trainable_peft_adapter
adapter path and non-self-referential SHA-256
base model and frozen base-model revision
tokenizer revision
condition name
data contract path and training-data hash
environment and executor revisions
seed and data seed
```

Required GRPO configurations validate these fields before loading the Tool-SFT adapter as trainable PEFT state. A revision or hash mismatch is a hard error. From-base RL is a separately named ablation.

## Transition to on-policy training

On-policy training begins only when the Tool-SFT pilot shows:

1. valid tool syntax substantially above initialization;
2. non-trivial successful `finish_trace` behavior;
3. improved held-out execution;
4. stable checkpoint reload under the frozen revision;
5. no train–runtime tool-budget mismatch.

## Interpretation

Tool-SFT success establishes that the interaction contract can be learned. H1 still requires causal interventions, H2 requires source-to-sink composition holdouts, and H3 requires matched frozen evidence predictions.
