# Replay-verified Tool-SFT

> **Purpose:** teach the trace-owned interaction contract before on-policy optimization  
> **Acceptance rule:** a row enters training only if the inference environment can replay it exactly

## Why Tool-SFT is necessary

A tool-aware chat schema is not, by itself, a learned policy. Before testing causal use, compositional generalization, or evidence benefit, the model must learn to inspect mapped states, import missing fragments, emit valid source-to-sink actions, use environment observations, terminate with `finish_trace`, and avoid independent proof or answer channels.

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

The mech-USPTO-31k inverse path starts from globally stitched forward traces
rather than MECH_PROOF conversion. Its separate `trace_no_knowledge` v2
contract, stereochemistry normalization boundary, 11,429-row coverage report,
and complete reproduction commands are documented in
[`MECH_USPTO_31K_INVERSE_TOOL_SFT.md`](MECH_USPTO_31K_INVERSE_TOOL_SFT.md).

## Retrieval query contract

Headline textbook rows use only inference-available molecular information:

```bash
python scripts/build_textbook_tool_sft.py \
  --input data/mechet_proof_clean/train.jsonl \
  --corpus knowledge/corpus/passages.jsonl \
  --output data/textbook_tool_sft/train.jsonl \
  --query-mode state
```

Build corresponding validation and test source rows before H3. `label_oracle` is an explicitly named upper bound and cannot support a headline evidence claim.

## Acceptance contract

A trace-owned row is accepted only when it has stable identity, valid top-level `messages` and canonical `tools`, JSON-object arguments, exactly one result per tool call, exactly one successful `finish_trace`, executor replay metadata, endpoint views, and a tool budget compatible with the frozen runtime.

The initial environment observation is embedded in the user turn rather than represented as an orphan tool result.

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

## Qwen3 assistant masks

Qwen3's shipped chat template does not provide the `{% generation %}` block required by Transformers to return automatic assistant masks. MechET therefore does **not** call `apply_chat_template(..., return_assistant_tokens_mask=True)`.

The frozen masking contract is:

```text
render the complete tool-bearing conversation exactly once
  -> tokenize the complete rendered ChatML exactly once
  -> scan final token IDs for <|im_start|>assistant ... <|im_end|>
  -> supervise every assistant span and mask all non-assistant spans with -100
```

The implementation is `final_chatml_token_scan_v1` in `src/mechet/assistant_masking.py`. It requires the observed assistant-span count to equal the number of assistant messages and retains the tool schema. Failure to render `tools=...` is a hard error rather than a silent prompt change.

Regression tests include multi-turn tool conversations with assistant tool calls, tool results, and a final assistant turn.

## Tokenizer and zero truncation audit

All six matched Qwen3 configurations use:

```text
max_length = 12288
```

This is a contract, not permission to truncate. `train_tool_sft.py` tokenizes the full row first and records P50/P95/P99/maximum input tokens, supervised assistant tokens, assistant-turn count, mask method, and over-budget stable IDs.

If any headline row exceeds 12,288 tokens, the run fails. The code does not silently slice the conversation. If the budget changes, all matched conditions must be audited and changed together.

Raw direct and tool syntax lengths are not assumed equal. Comparisons disclose real tokenizer input and assistant-mask tokens and use supervised-token-normalized compute when required.

## Learnability pilot

```bash
python scripts/train_tool_sft.py \
  --config configs/knowledge/tool_sft_trace_no_knowledge.yaml \
  --limit 32 \
  --max-steps 100
```

A credible pilot reports training/validation loss, valid tool-call rate, `finish_trace` rate, formal execution, trace-bound behavior, and held-out endpoint performance. A schema-only dry run does not establish learnability.

## Mixed inverse pretraining run

The shared Qwen3-8B run combines the two datasets that implement the same
product-to-executable-inverse-trace objective:

- FlowER mechanism proofs converted to trace-owned Tool-SFT;
- stitched mech-USPTO-31k inverse traces.

Knowledge retrieval is stripped from the FlowER rows, so every example is in
the `trace_no_knowledge` condition and the textbook corpus is not a training
input. The natural-frequency mixture is structurally decontaminated against
the union of valid and test before training. `training_manifest.json` binds the
split hashes, source counts, overlap report, tokenizer audits, model revision,
and the zero-corpus contract. The A100-40GB run uses a 20,480-token frozen
budget. Four 20,481--31,605-token FlowER training traces are explicitly
quarantined because Qwen3's full-vocabulary FP32 loss materialization does not
fit on A100-40GB at those lengths. Validation and test remain complete, and no
accepted trace is silently truncated.

Training uses Liger 0.6.2 fused linear cross entropy for Qwen3. This computes
the language-model head and assistant-only cross entropy in chunks instead of
materializing a `[sequence, 151669]` FP32 logits tensor. Only that fused loss is
enabled; Liger RoPE, RMSNorm, and SwiGLU replacements are disabled so the model
definition otherwise remains unchanged. The launcher verifies the vendored
wheel SHA-256 before installing it into a per-job temporary directory.

```bash
python scripts/build_mixed_inverse_tool_sft.py
python scripts/finalize_mixed_inverse_tool_sft.py

torchrun --standalone --nproc_per_node=8 scripts/train_tool_sft.py \
  --config configs/agent/tool_sft_mixed_inverse_qwen3_8b.yaml
```

The checked-in Taiji launcher requests one host with eight H20 GPUs. Its
resource fallback order is H20, A100, then V100; only one task should be active
at a time.

## Training reproducibility

### Requested versus resolved model revision

A configuration may request:

```yaml
model_revision: main
```

That value is not a frozen scientific revision. After loading the tokenizer, `train_tool_sft.py` resolves the snapshot and records:

```text
requested_model_revision = main
base_model_revision = <immutable 40-hex commit>
tokenizer_revision = <immutable 40-hex commit>
```

The model is then loaded at the immutable commit. Required GRPO adapters must carry that immutable base-model revision in `adapter_manifest.json`; a mutable adapter revision is a hard lineage error.

Every run also records seed and data seed, precision policy, TF32, gradient checkpointing, maximum sequence length, optimizer updates, and checkpoint retention.

### Transformers 4/5 length grouping

Length grouping is an efficiency setting, not a scientific variable. The launcher inspects `TrainingArguments` and uses `train_sampling_strategy=group_by_length` for the Transformers 5 API, falling back to the older compatible `group_by_length` field when present. The applied field is written to `data_contract.json`.

`packing=true` is rejected by the current pretokenized Trainer path. `assistant_only_loss=true` is mandatory because explicit `labels=-100` masking defines the supervised objective.

## Adapter lineage

Every real run writes `data_contract.json` and `adapter_manifest.json` with the training-data SHA-256, token/mask audit, immutable base-model/tokenizer revisions, environment/executor revisions, seed policy, and non-self-referential SHA-256 for the adapter directory.

Required GRPO configurations validate these fields before loading the Tool-SFT adapter as trainable PEFT state. A revision or hash mismatch is a hard error. From-base RL is a separately named ablation and requires an explicitly immutable model revision.

## GRPO runtime profiles

Portable default:

```text
configs/agent/inverse_trace_grpo.yaml       use_vllm = false
```

Paper-scale throughput:

```text
configs/agent/inverse_trace_grpo_vllm.yaml  use_vllm = true
```

vLLM is a serving/throughput backend, not part of the scientific method. The scientific comparison must keep model, adapter, rollout budget, seed policy, and generation contract fixed regardless of serving backend.

## Interpretation

Tool-SFT success establishes that the interaction contract can be learned. H1 still requires causal interventions, H2 requires source-to-sink composition holdouts, and H3 requires matched frozen held-out evidence predictions.
