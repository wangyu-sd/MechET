# Trace-owned electron-flow faithfulness

This document is authoritative for the MechET main inference contract.

## Scientific hypothesis

A mechanistic rationale is faithful only when it is causally upstream of the predicted endpoint. The main MechET path therefore makes the environment-owned electron-flow trace the sole computational source of the proof and precursor.

```text
explicit tool actions
  -> committed environment transitions
  -> authoritative source-to-sink move trace
  -> replay declared moves
  -> deterministic trace compiler
  -> MECH_PROOF v1
  -> executor-derived full precursor state
  -> atom-contributing structural precursor
```

The model cannot receive endpoint credit from a proof or answer generated independently of its committed actions.

## Public runtime surface

The model-facing main environment is `TraceOwnedTRLEnvironment`, an explicit TRL facade around `TraceOwnedAgentEnv`. The facade exposes only:

```text
inspect_state
import_fragment
apply_electron_move
apply_coupled_electron_moves
finish_trace
abstain
```

`reset` and `get_reward` are framework methods rather than model tools. Internal helpers such as `state_dict` and `_snapshot` remain private. `submit_proof` is not present on the main tool surface; the underlying trace environment also rejects it with `FREE_FORM_PROOF_DISABLED`.

Evidence variants add only the declared soft-evidence tools:

```text
retrieve_textbook_guidance
retrieve_primitives
```

The legacy complete-proof facade is a named baseline and is the only path that exposes `submit_proof`.

## Tool-budget integrity

Every attempted model action must be observable and budgeted. Empty move lists, malformed arguments, unavailable tools, disabled tools and runtime failures consume a tool call and increase failure accounting. A model cannot extend an episode through repeated invalid calls.

## Imports

Atoms absent from the product must enter through `import_fragment`.

- imported atoms require unique positive maps;
- imported maps cannot collide with the current state;
- root-level proof imports are preserved as `initial_imports` during proof-to-trace conversion;
- edge-level imports are attached to the next successful transition;
- the compiled proof records the same imports;
- `finish_trace` rejects uncommitted imports.

This guarantees that proof execution and trace replay begin from the same molecular state.

## Committed transitions

A transition becomes authoritative only after the deterministic environment successfully applies the proposed electron-flow action or coupled action set.

For each transition the trace records:

```text
state_before
state_after
imports
source-to-sink moves
trace step index
```

Failed actions remain in the rollout log but do not become proof edges.

## Deterministic compilation and move replay

Before deriving BOND, LP and CHARGE deltas, the compiler replays the declared source-to-sink moves against `state_before + imports` and requires exact mapped agreement with `state_after`. This prevents a chemically correct state transition from being paired with an unrelated stated arrow sequence.

The compiler then:

1. emits `MECH_PROOF v1`;
2. executes the compiled proof;
3. checks every compiled intermediate against the recorded mapped state;
4. derives full, structural and auxiliary endpoint views;
5. returns stable trace and move-sequence digests.

The terminal result contains:

```text
trace_bound = true
trace_digest
move_sequence_digest
declared_moves_replayed = true
compiled_proof
n_trace_transitions
endpoint_source = environment_owned_trace
full_precursor_state
structural_precursor
auxiliary_fragments
```

Evaluation reconstructs the trace and re-executes the proof rather than trusting stored booleans.

## Main path versus baselines

### Main path

```text
trace-owned tool reasoning
finish_trace
environment-compiled proof
executor-derived endpoint
```

### Required baselines

```text
outcome-only direct generation
answer-bearing free-form or state CoT
independent complete-proof generation
legacy loose tool trace plus submitted proof
```

The baselines quantify the reasoning–endpoint bypass and must not be described as equivalent implementations of the main method.

## Training lineage

Replay-verified Tool-SFT is the preferred initialization. Paper-scale on-policy training must not begin from an untrained tool policy.

```bash
python scripts/train_tool_sft.py \
  --config configs/knowledge/tool_sft_trace_no_knowledge.yaml

python scripts/train_inverse_agent_trace.py \
  --config configs/agent/inverse_trace_grpo.yaml
```

Every required RL checkpoint records and validates the Tool-SFT adapter SHA-256, base model, condition, data contract, environment revision and executor revision.

## H1 causal interventions

The canonical inference runner supports:

```text
remove_tool_observations
stale_tool_observations
shuffle_tool_observations
disable_inspect_state
disable_intermediate_execution
```

Normal and intervention artifacts must use the same model, adapter, model revision, temperature, top-p, maximum new tokens, maximum iterations, candidate count and frozen reference IDs.

```bash
python scripts/evaluate_faithfulness.py \
  --reference data/benchmark/test.jsonl \
  --normal outputs/h1/normal.jsonl \
  --intervention remove_tool_observations=outputs/h1/remove.jsonl \
  --intervention stale_tool_observations=outputs/h1/stale.jsonl \
  --intervention shuffle_tool_observations=outputs/h1/shuffle.jsonl \
  --output outputs/h1/summary.json
```

The claim is unsupported if the normal path is not completely trace-bound, artifacts are incomplete, trace/proof recomputation fails, or corrupting observations has no measurable paired effect.

## Scope

Trace ownership establishes a causal computational contract. It does not establish that an inferred mechanism is unique, kinetically favored, high yielding or experimentally realized.