# Compact full-state A7 protocol

`compact_full_state_v1` is the main A7 candidate defined by Issue #37. It is a
lossless model-facing serialization change, not a new chemistry task or a new
executor.

## Model-visible contract

- The mapped product appears once in the user `TARGET` line.
- Every nonterminal tool result contains exactly one authoritative
  `current_state_smiles`, the call outcome/code and remaining call budget.
- `inspect_state` may additionally return the legal source/sink inventory.
- `import_fragment` may additionally return `pending_import_count`.
- Accepted move results may additionally return `trace_bound`.
- `finish_trace` returns one executed `derived_precursor` and the minimal
  execution/reward status needed by the agent loop.

The following remain internal artifact metadata and never enter a
compact-full-state tool result: compiled proof text, trace and move-sequence
digests, duplicated `state_before`/`state_after`, full/structural endpoint
copies, imported-fragment copies and pending-import lists.

## Frozen chemistry and data universe

The source is `data/mechet_proof_sft_flower_full_v4`: the strict executable
FlowER universe of 257,167 train, 2,890 valid and 28,967 test rows. The only
excluded official rows are the named upstream non-atom-conserving endpoints
(4 train and 4 test). No overlap, token-length or mechanism-family filtering is
allowed.

The action sequence, atom maps, compiler, executor-owned endpoint and stable
IDs must be identical to the matching legacy full-state row. Complete internal
traces remain available for replay and audit.

## Mandatory pre-training gate

Run a frozen 1k–5k slice through
`scripts/audit_compact_full_state_gate.py`. Training is forbidden unless the
gate reports:

1. identical IDs and byte-identical assistant tool calls;
2. identical executor state after every nonterminal call;
3. identical final precursor and execution outcome;
4. zero drops and zero model-visible audit-field leaks;
5. compact Qwen3-8B chat-template tokens at no more than 60% of legacy
   full-state.

The frozen 2,048-row gate on 2026-08-26 passed all equivalence checks at a
50.38% token ratio.

## Training decision rule

Train Qwen3-8B with the frozen A7 LoRA/optimizer settings for one resumable
epoch first. Run K=1 on frozen validation immediately after the epoch. Only if
it recovers clearly from action-delta and approaches the full-state reference
should it proceed to full-test K=10 + NLL ranking. Continue to epoch 2 only when
validation shows clear headroom. Do not launch representation ablations as part
of this run.
