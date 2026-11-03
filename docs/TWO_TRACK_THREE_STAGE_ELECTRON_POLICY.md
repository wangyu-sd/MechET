# Two-track, three-stage electron policy

## Objective

Both model families solve the same product-only retrosynthetic control problem.
Given the fixed product `P`, the executor-owned current molecular state `s_t`,
and (after stage 1) a compact accepted-action history `h_t`, predict exactly one
next inverse electron event `a_t`:

```text
pi(a_t | P, s_t)                 stage 1
pi(a_t | P, s_t, h_t)            stages 2 and 3
```

The executor, not the policy, applies the event and constructs `s_(t+1)`.  The
policy never receives the expected precursor, the reference remaining horizon,
or a list containing the gold action.

## Two tracks

| Track | Observation | Decoder | Shared semantic output |
|---|---|---|---|
| LLM | unmapped/temporarily annotated SMILES plus compact ledger | canonical JSON tool action | one import, electron-flow, BE-delta, or finish event |
| Graph | atom/bond tensors plus a learned ledger encoder | family heads, graph-node pointers, open fragment graph program | the same canonical event |

Private atom maps are alignment handles inside the compiler/executor.  They are
not node features and are not rendered to the LLM.  LLM atom names (`A01`,
`A02`, ...) are regenerated from the current state at every decision.

## Three stages

### Stage 1 — state behavior cloning (`state_bc`)

Input: fixed target product and exact current executor state.  The compact
history is absent.  Target: the next reference event.  This stage teaches the
action grammar, atom/bond grounding, import construction and ordinary chemical
priors before any sparse endpoint reward is introduced.

### Stage 2 — trajectory behavior cloning (`trajectory_bc`)

Initialize from stage 1. Add the ordered ledger of accepted and recently rejected
events. The ledger contains action families, container kinds, import roles,
small counts, map-free local chemical-site fingerprints and executor error codes;
it does not repeat molecular states or expose atom maps, gold suffixes, or the
expected precursor.  The exact current state remains authoritative.  A residual
history gate is initialized at zero so stage-2 initialization is identical to
the state-only policy.

### Stage 3 — direct online RL (`online_rl`)

Initialize from stage 2 and sample directly from the policy distribution:

```text
family -> container kind -> graph node pointer(s) -> event commit
family -> open fragment graph program -> fragment commit
```

There is no enumerated route/event candidate list and no retrieval-ranked
fragment bank. Factorized pointers use reference-independent hard masks for
structural and electron-container legality. The executor accepts or rejects the
generated event and returns the real successor state. Rejected actions and
public error codes enter the next compact observation so the policy can revise
instead of repeating. RL stores the sampled action/log-probability, public
executor result, successor state and terminal reward.

Recommended reward components are:

```text
R = endpoint_exact
  + executable_event_credit
  + successor_progress
  - invalid_event_penalty
  - repeated_state_penalty
  - length_cost
```

Only endpoint exactness is a success condition.  Shaping terms accelerate
credit assignment but cannot turn an incorrect endpoint into a success.

## Canonical action families

- `IMPORT_ENV`: generate an environment fragment graph program;
- `IMPORT_REACTIVE`: generate a reactive fragment graph program and active site;
- `FLOW`: generate one atomic, possibly multi-arrow electron event;
- `BE_DELTA`: generate one sparse bond/charge edit event;
- `FINISH`: ask the executor to validate and return the precursor endpoint.

Both tracks compile to this schema.  Surface syntax is track-specific, while
execution and evaluation are shared.

## Data contract

The canonical source remains the frozen strict executable FlowER universe:

| split | reactions |
|---|---:|
| train | 257,167 |
| valid | 2,890 |
| test | 28,967 |

This is not the unqualified FlowER-full reaction split of
257,171 / 2,890 / 28,971.  The four train and four test upstream-corrupt rows
have no executable mechanism and remain outside this two-track policy dataset.

Each compiled decision stores `target`, `current`, `kind`, the canonical action
payload, and the compressed history *before* that action. Schema v4 schedules
reactive imports at first electron use and environment-only imports immediately
before the terminal endpoint; it also stores reaction/family balancing metadata.
The same build supports both BC stages: stage 1 ignores `history`; stage 2 consumes
it. JSONL is the audit artifact. Production trainers access it through a
standard `torch.utils.data.Dataset/DataLoader`; packed tensor shards can replace
the Dataset backend without changing the model protocol.

## Stage gates

1. Stage 1: next-family accuracy, pointer accuracy, import-program validity and
   teacher-forced executable-event accuracy.
2. Stage 2: the same metrics plus suffix accuracy by horizon; it must not regress
   one-step performance and should reduce long-horizon error accumulation.
3. Stage 3: strict executable-event rate, repeated-state rate, full endpoint
   exact match, mean steps and wall-clock cost.  Report product-only K=1 first;
   K>1 is a sampling metric, not a substitute for a strong single policy.

## Implementation

- shared protocol/history: `src/mechet/electron_policy_protocol.py`;
- graph policy and history encoder: `src/mechet/graph_electron_policy.py`;
- schema-v4 compiler: `scripts/build_graph_electron_full.py`;
- stage-1/stage-2 DDP trainer: `scripts/train_graph_electron_full_batched.py`;
- product-only closed-loop evaluator: `scripts/evaluate_graph_electron_policy.py`;
- LLM serialization: `llm_training_record()` in the shared protocol module.
