# Trace-owned electron-flow faithfulness

## Purpose

The main MechET agent path must not receive endpoint credit from a proof that is independent of its tool trajectory. `TraceOwnedAgentEnv` therefore makes the environment-owned electron-flow trace the sole source of the final proof and precursor.

```text
inspect/import/execute tools
        ↓
committed environment trace
        ↓
deterministic trace compiler
        ↓
MECH_PROOF v1
        ↓
executor-derived precursor
```

The legacy `MechETAgentEnv.submit_proof` path remains available only as a complete-proof baseline.

## Tools

```text
inspect_state
import_fragment
apply_electron_move
apply_coupled_electron_moves
finish_trace
abstain
get_reward
```

`submit_proof` is deliberately disabled in `TraceOwnedAgentEnv`.

## Imports

Atoms absent from the product must be introduced with `import_fragment`. Imported fragments require unique positive atom maps and cannot collide with maps already present in the state. An import is attached to the next committed transition and appears as an `IMPORT` in the compiled proof.

A trace cannot finish with uncommitted imports.

## Compilation

For every committed state transition, the compiler derives:

```text
BOND changes
LP changes
formal-charge transitions
```

from the pre- and post-transition mapped states. It then executes the generated proof and checks every compiled intermediate against the recorded trace with atom maps preserved.

The terminal result contains:

```text
trace_bound = true
trace_digest
compiled_proof
n_trace_transitions
endpoint_source = environment_owned_trace
```

## Training

Dry-run:

```bash
python scripts/train_inverse_agent_trace.py \
  --config configs/agent/inverse_trace_grpo.yaml \
  --dry-run --limit 8
```

Training:

```bash
python scripts/train_inverse_agent_trace.py \
  --config configs/agent/inverse_trace_grpo.yaml
```

The matched complete-proof baseline remains:

```bash
python scripts/train_inverse_agent_trl.py \
  --config configs/agent/inverse_trl_grpo.yaml
```

## Evaluation

Required comparisons:

```text
complete proof generation
legacy loose tool trace + submitted proof
trace-owned tool reasoning
```

Required metrics include trace–proof agreement, trace–endpoint agreement, endpoint accuracy, execution rate, unnecessary-action count, tool-failure recovery, and causal interventions on tool observations.
