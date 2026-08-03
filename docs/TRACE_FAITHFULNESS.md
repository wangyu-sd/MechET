# Trace-owned electron-flow faithfulness

This document is authoritative for the MechET main inference contract.

## Scientific hypothesis

A mechanistic rationale is faithful only when it is causally upstream of the predicted endpoint. The main MechET path therefore makes the environment-owned electron-flow trace the sole computational source of the proof and precursor.

```text
explicit tool actions
  -> committed environment transitions
  -> authoritative trace
  -> deterministic trace compiler
  -> MECH_PROOF v1
  -> executor-derived structural precursor
```

The model cannot receive endpoint credit from a proof or answer generated independently of its committed actions.

## Main environment

`TraceOwnedAgentEnv` exposes:

```text
inspect_state
import_fragment
apply_electron_move
apply_coupled_electron_moves
finish_trace
abstain
get_reward
```

`submit_proof` is deliberately disabled and returns `FREE_FORM_PROOF_DISABLED`.

`KnowledgeAugmentedAgentEnv` inherits the same causal contract and adds only soft-evidence tools:

```text
retrieve_textbook_guidance
retrieve_primitives
```

Evidence tools do not return the precursor, do not receive direct reward and do not override execution.

## Imports

Atoms absent from the product must enter through `import_fragment`.

- imported atoms require unique positive maps;
- imported maps cannot collide with the current state;
- an import is attached to the next successful transition;
- the compiled proof records the import;
- `finish_trace` rejects uncommitted imports.

## Committed transitions

A transition becomes authoritative only after the deterministic environment successfully applies the proposed electron-flow action or coupled action set.

For each transition the trace records:

```text
state before
state after
imports
source-to-sink moves
trace step index
```

Failed actions remain in the rollout log but do not become proof edges.

## Deterministic compilation

The trace compiler derives:

```text
BOND changes
lone-pair changes
formal-charge transitions
imports and dependencies
```

It then:

1. emits `MECH_PROOF v1`;
2. executes the compiled proof;
3. checks each compiled intermediate against the recorded mapped state;
4. derives the structural precursor;
5. returns a stable trace digest.

The terminal result contains:

```text
trace_bound = true
trace_digest
compiled_proof
n_trace_transitions
endpoint_source = environment_owned_trace
```

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
independent complete-proof generation
legacy loose tool trace plus submitted proof
answer-bearing free-form/state CoT
```

The baselines are retained to quantify the reasoning–endpoint bypass. They must not be described as equivalent implementations of the main method.

## Training

Replay-verified Tool-SFT is the preferred initialization. Paper-scale on-policy training should not begin from an untrained tool policy.

Tool-SFT:

```bash
python scripts/train_tool_sft.py \
  --config configs/knowledge/tool_sft_textbook.yaml
```

Trace-owned dry-run:

```bash
python scripts/train_inverse_agent_trace.py \
  --config configs/agent/inverse_trace_grpo.yaml \
  --dry-run --limit 8
```

Trace-owned training:

```bash
python scripts/train_inverse_agent_trace.py \
  --config configs/agent/inverse_trace_grpo.yaml
```

Knowledge condition:

```bash
python scripts/train_inverse_agent_knowledge.py \
  --config configs/knowledge/inverse_textbook_trace_grpo.yaml
```

Legacy loose-trace/complete-proof baseline:

```bash
python scripts/train_inverse_agent_trl.py \
  --config configs/agent/inverse_trl_grpo.yaml
```

Every RL checkpoint must record the Tool-SFT adapter hash, data-manifest hash, environment revision and executor revision.

## Evaluation

### Consistency metrics

```text
trace–proof agreement
trace–endpoint agreement
compiled-state replay agreement
execution rate
endpoint accuracy
unnecessary-action count
failed-action distribution
tool-failure recovery
abstention and risk–coverage
```

### Causal interventions

Required interventions:

```text
remove tool observations
shuffle tool observations
replace observations with stale states
disable inspect_state
disable intermediate execution
remove failure certificates
allow independent proof submission only in a baseline
```

Report the effect on endpoint accuracy, execution, trace consistency, action choice and abstention.

### Claim gate

The tool-grounded causal reasoning claim is unsupported when:

- the model is insensitive to removing or corrupting tool observations;
- the reported endpoint can be produced through an independent answer/proof channel;
- trace and compiled proof disagree;
- a knowledge or learned score overrides formal failure;
- successful results depend on silently repairing semantically meaningful bond or charge actions.

## Scope

Trace ownership establishes a causal computational contract. It does not establish that the inferred mechanism is unique, kinetically favored, high yielding or experimentally realized.
