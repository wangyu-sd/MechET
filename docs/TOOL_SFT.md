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

Build the corresponding `valid.jsonl`, `test.jsonl`, `valid_text_and_anchors.jsonl`, and `test_text_and_anchors.jsonl` from the frozen proof validation/test splits before constructing H3. `--query-mode state` derives terms from the target or current molecular state. `label_oracle` is an explicitly named upper bound and cannot support a headline evidence claim.

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

Every assistant tool call contains JSON-object arguments. Arguments remain JSON objects throughout data loading, tokenization, training, inference, and audit.

## Six matched conditions

```text
configs/knowledge/tool_sft_trace_no_knowledge.yaml
configs/knowledge/tool_sft_irrelevant.yaml
configs/knowledge/tool_sft_textbook.yaml
configs/knowledge/tool_sft_anchors.yaml
configs/knowledge/tool_sft_combined.yaml
configs/knowledge/tool_sft_direct_textbook.yaml
```

All six train on `data/knowledge_ablation/v2/train/`. Validation and final H3 evaluation use the separately derived `valid/` and `test/` stable-ID universes. A train-derived evidence suite is never a final H3 benchmark.

## Qwen3 assistant-only supervision

Qwen3's shipped chat template does not provide the `{% generation %}` block required by Transformers to return an automatic assistant-token mask. MechET therefore does **not** call `apply_chat_template(..., return_assistant_tokens_mask=True)`.

The frozen masking contract is:

```text
render the complete tool-bearing conversation exactly once
  -> tokenize the complete rendered ChatML exactly once
  -> scan final token IDs for <|im_start|>assistant ... <|im_end|>
  -> supervise every assistant span and mask all non-assistant spans with -100
```

The implementation is `final_chatml_token_scan_v1` in `src/mechet/assistant_masking.py`. It requires the observed assistant-span count to equal the number of assistant messages. It must retain the tool schema; failure to render `tools=...` is a hard error rather than a silent prompt change.

Regression tests include multi-turn tool conversations with assistant tool calls, tool results, and a final assistant turn.

## Tokenizer and zero-truncation audit

The six matched Qwen3 configurations use one frozen headline budget:

```text
max_length = 12288
```

This value is a contract, not permission to truncate. `train_tool_sft.py` tokenizes the full row first and records:

```text
P50 / P95 / P99 / maximum input tokens
supervised assistant tokens
assistant-turn count and mask method
over-budget stable IDs
```

If any headline row exceeds 12,288 tokens, the run fails. The code does not silently slice the conversation. If the budget changes, all matched conditions must be audited and changed together.

Raw direct and tool syntax lengths are not assumed equal. Comparisons disclose real tokenizer input and assistant-mask tokens and use supervised-token-normalized compute when required.

## Learnability pilot

Start with a real 32–128-example overfit:

```bash
python scripts/train_tool_sft.py \
  --config configs/knowledge/tool_sft_trace_no_knowledge.yaml \
  --limit 32 \
  --max-steps 100
```

A credible pilot reports training/validation loss, valid tool-call rate, `finish_trace` rate, formal execution, trace-bound behavior, and held-out endpoint performance. A schema-only dry run does not establish learnability.

## Training reproducibility

### Requested versus resolved model revision

A configuration may request a human-readable Hugging Face revision such as:

```yaml
model_revision: main
```

That value is **not** a frozen scientific revision. After loading the tokenizer, `train_tool_sft.py` resolves the snapshot to the actual full 40-hex commit SHA and writes both:

```text
requested_model_revision = main
base_model_revision = <immutable 40-hex commit>
tokenizer_revision = <immutable 40-hex commit>
```

The base model is then loaded at the resolved immutable commit. Required GRPO adapters must carry that immutable revision in `adapter_manifest.json`; a mutable adapter revision is a hard lineage error.

Training configuration also records seed and data seed, precision policy, TF32, gradient checkpointing, maximum sequence length, optimizer updates, and checkpoint retention.

### Transformers 4/5 length grouping

Length grouping is treated as an efficiency setting rather than a scientific variable. The launcher inspects `TrainingArguments`:

```text
Transformers 5.x: train_sampling_strategy = group_by_length
older compatible API: group_by_length = true
unsupported API: no grouping argument is injected
```

`data_contract.json` records the argument actually applied; it never reports grouping as enabled when the installed Transformers API ignored it.

`packing=true` is rejected by the current pretokenized Trainer path. `assistant_only_loss=true` is mandatory because the explicit `labels=-100` mask defines the supervised objective.

## Adapter lineage

Every real run writes `data_contract.json` and `adapter_manifest.json` with the training-data SHA-256, token/mask audit, immutable base-model/tokenizer revisions, environment/executor revisions, seed policy, and non-self-referential adapter SHA-256.

Required GRPO configurations validate these fields before loading the Tool-SFT adapter as trainable PEFT state. A revision or hash mismatch is a hard error. From-base RL is a separately named ablation and requires an explicitly immutable model revision.

## GRPO runtime profiles

The portable default remains:

```text
configs/agent/inverse_trace_grpo.yaml      use_vllm = false
```

Paper-scale throughput can use:

```text
configs/agent/inverse_trace_grpo_vllm.yaml use_vllm = true
```

vLLM is a serving/throughput backend, not part of the scientific method. The scientific comparison must keep model, adapter, rollout budget, seed policy, and generation contract fixed regardless of serving backend.

## Transition to on-policy training

On-policy training begins only when the Tool-SFT pilot shows:

1. valid tool syntax substantially above initialization;
2. non-trivial successful `finish_trace` behavior;
3. improved held-out execution;
4. stable checkpoint reload under the frozen revision;
5. no train–runtime tool-budget mismatch.

## Interpretation

Tool-SFT success establishes that the interaction contract can be learned. H1 still requires causal interventions, H2 requires source-to-sink composition holdouts, and H3 requires matched frozen held-out evidence predictions.
