# Trace-owned faithfulness contract

> **Authority:** main inference path and H1 causal intervention semantics  
> **Core principle:** no independent proof or answer channel may receive endpoint credit

## Scientific statement

A mechanistic rationale is faithful only when it is causally upstream of the predicted endpoint. MechET therefore makes the environment-owned trace the sole computational source of the proof and precursor.

```text
model tool actions
  -> committed environment transitions
  -> authoritative source-to-sink move trace
  -> finish_trace
  -> replay declared moves
  -> deterministic trace compiler
  -> MECH_PROOF v1
  -> executor-derived endpoint views
```

Trace/proof consistency by construction is necessary but not sufficient for H1. The model must also respond to interventions on information returned by the environment.

## Runtime invariants

| Invariant | Enforcement | Violation |
|---|---|---|
| **Single endpoint path** | `finish_trace` is the only successful endpoint-producing terminal method | Free-form proof or precursor is credited |
| **Explicit completion** | The model must call `finish_trace`; the evaluator never completes a trace | Max-iteration or partial trace is scored as a prediction |
| **Move–state binding** | Declared source-to-sink moves replay to the recorded mapped state | Correct state paired with unrelated arrow claims |
| **Proof–trace binding** | The proof is compiled from the committed trace | Independently authored proof replaces the trace |
| **Import conservation** | Root and edge imports survive conversion, replay, and compilation | Proof and trace begin from different molecular states |
| **Budget integrity** | Valid, invalid, unavailable, and disabled actions all consume budget | Malformed calls extend an episode for free |
| **Evaluation independence** | Trace and proof are recomputed from artifacts | Stored success booleans are trusted without replay |

## Public runtime surface

The model-facing main implementation is `TraceOwnedTRLEnvironment`, an explicit TRL facade around `TraceOwnedAgentEnv`.

It exposes only:

```text
inspect_state
import_fragment
apply_electron_move
apply_coupled_electron_moves
finish_trace
abstain
```

`reset` and `get_reward` are framework methods. Internal helpers such as `state_dict` and `_snapshot` remain private. `submit_proof` is absent from the main tool surface, and the underlying trace environment rejects it with `FREE_FORM_PROOF_DISABLED`.

Evidence variants add only the declared soft-evidence tools:

```text
retrieve_textbook_guidance
retrieve_primitives
```

The legacy complete-proof facade is a named baseline and is the only path that exposes `submit_proof`.

## Episode lifecycle

### 1. Reset

The environment initializes an atom-mapped target, the frozen tool budget, the current molecular state, and the trace-owned endpoint contract.

### 2. Inspect and import

`inspect_state` exposes legal electron sources and sinks. Atoms absent from the product enter only through `import_fragment`.

Import invariants:

- imported atoms require unique positive maps;
- imported maps cannot collide with the current state;
- root-level proof imports are preserved as `initial_imports`;
- edge-level imports are attached to the next committed transition;
- `finish_trace` rejects uncommitted imports.

### 3. Commit transitions

A transition becomes authoritative only after the deterministic environment successfully applies the proposed electron-flow action or coupled action set.

Each committed transition records:

```text
state_before
state_after
imports
source-to-sink moves
trace step index
```

Failed actions remain visible in the rollout log but do not become proof edges.

### 4. Finish and compile

Before deriving BOND, LP, and CHARGE deltas, the compiler replays declared moves against `state_before + imports` and requires exact mapped agreement with `state_after`.

The compiler then:

1. emits `MECH_PROOF v1`;
2. executes the compiled proof;
3. checks every compiled intermediate against the recorded state;
4. derives full, structural, and auxiliary endpoint views;
5. returns stable trace and move-sequence digests.

The raw terminal result contains:

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

### 5. Evaluate

A trace prediction is valid only when the artifact shows one actual `finish_trace` call and the raw terminal state recomputes exactly. Observation interventions may redact the model-visible terminal message, but evaluation uses the environment-owned raw terminal state and verifies that the terminal call occurred.

The evaluator rejects:

```text
unfinished trace
abstention as a positive prediction
missing or repeated finish_trace result
trace-to-direct fallback
trace, move, proof, or endpoint digest mismatch
re-execution failure
```

## Tool-budget integrity

Every attempted action is observable and budgeted. This includes:

```text
empty move list
malformed JSON or arguments
unknown or unavailable tool
disabled intervention tool
runtime execution failure
```

The main trace, textbook, anchor, and combined conditions use the same frozen 16-call headline budget. Training rows that require more calls are quarantined rather than executed under a larger hidden budget.

## Main path and baselines

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
answer-bearing free-form CoT
answer-bearing state CoT
independent complete-proof generation
legacy loose tool trace plus submitted proof
```

These baselines quantify the reasoning–endpoint bypass and must not be described as equivalent implementations of the main method.

## Training lineage

Replay-verified Tool-SFT is the preferred initialization. Paper-scale on-policy training must not begin from an untrained tool policy.

```bash
python scripts/train_tool_sft.py \
  --config configs/knowledge/tool_sft_trace_no_knowledge.yaml

python scripts/train_inverse_agent_trace.py \
  --config configs/agent/inverse_trace_grpo.yaml
```

Every required checkpoint validates:

```text
adapter SHA-256
base model and frozen model revision
tokenizer revision
condition and data contract
environment and executor revisions
seed and data seed
```

## H1 interventions

| Intervention | Information changed | Required control |
|---|---|---|
| `remove_tool_observations` | Chemistry-bearing observation content is redacted | JSON structure, control fields, and serialized length preserved |
| `stale_tool_observations` | Previous result from the same tool type is replayed | Tool identity preserved |
| `shuffle_tool_observations` | Observation comes from a different target under the same tool type | Self donors forbidden; donor manifest audited |
| `disable_inspect_state` | State inspection unavailable | Call still consumes budget |
| `disable_intermediate_execution` | State-changing execution unavailable | Call still consumes budget |

Normal and intervention artifacts must use the same model, adapter, model/tokenizer revisions, seed policy, temperature, top-p, maximum new tokens, maximum iterations, candidate count, and frozen IDs.

```bash
python scripts/evaluate_faithfulness.py \
  --reference data/benchmark/test.jsonl \
  --normal outputs/h1/normal.jsonl \
  --intervention remove_tool_observations=outputs/h1/remove.jsonl \
  --intervention stale_tool_observations=outputs/h1/stale.jsonl \
  --intervention shuffle_tool_observations=outputs/h1/shuffle.jsonl \
  --output outputs/h1/summary.json
```

## Claim gate

H1 is unsupported when any of the following hold:

- the normal path is not completely trace-bound among credited predictions;
- prediction artifacts or runtime metadata are incomplete;
- trace/proof recomputation fails;
- intervention construction is invalid;
- corrupting observations produces no material paired effect.

## Scope

Trace ownership establishes a causal computational contract. It does not establish that an inferred mechanism is unique, kinetically favored, high yielding, or experimentally realized.
