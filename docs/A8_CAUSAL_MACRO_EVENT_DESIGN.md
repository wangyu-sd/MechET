# A8: causal electron-flow macro-events

> **Status: design proposal.** This document supersedes `A8_ETPG_DESIGN.md` as
> the primary A8 candidate. The partial-order ETPG design remains a useful
> analysis/ablation, but it is not the preferred main algorithm after reviewing
> the existing A4, proof-equivalence and GFR infrastructure. This proposal does
> not authorize training, change the frozen FlowER universe, or change the active
> paper condition.

## Why A8 needs a different abstraction

The current scientific bottleneck is the tension between A0 and A7:

- **A0** makes a short global endpoint decision and avoids long-horizon rollout
  divergence, but electron transfer is implicit and is not the causal route to
  the precursor.
- **A7** makes electron flow explicit, executable and endpoint-owning, but the
  prediction is interactive: an accepted early error changes the authoritative
  state seen by every later decision.

A8 should retain electron transfer as the chemical coordinate while reducing
how often the model must make a new state-dependent decision.

A key correction is that current A7 is **already elementary-event level**, not
single-arrow level. The frozen inverse-data builder emits one
`apply_coupled_electron_moves` call for each `trace_plan` step, so all moves in
one source elementary step are already executed atomically. Therefore defining
an A8 "chemical event" as one elementary step would reproduce A7 rather than
create a new temporal abstraction.

The measured full-universe diagnostic strengthens this point. In the current
strict FlowER condition, the mean trace contains approximately:

```text
4.05 mechanism steps
8.16 electron moves
5.22 fragment imports
```

The historical 3,080-target trace view averaged only 1.93 mechanism steps and
3.86 electron moves. These values are diagnostic rather than causal proof, but
they show that a useful A8 must compress **multiple elementary steps and/or the
fragment-call horizon**, not merely bundle arrows that A7 already couples.

## Main proposal

A8 predicts a sequence of **causal electron-flow macro-events**. A macro-event
is a maximal contiguous episode of already valid elementary electron-flow steps
that belongs to one local causal transformation episode. The executor dry-runs
the complete macro-event transactionally and exposes the next authoritative
state only at the macro boundary.

The high-level contract is:

```text
mapped product
  -> predict complete auxiliary-fragment inventory once
  -> canonicalize/map inventory without reference lookup
  -> predict next causal electron-flow macro-event
  -> executor dry-run
       -> all elementary steps valid: commit one new authoritative state
       -> failure: rollback and return a bounded failure certificate
  -> predict next macro-event
  -> finish_trace
  -> executor-owned precursor
```

The credited endpoint remains causal:

```text
precursor = execute(committed macro-events)
```

There is no independent answer channel.

## Level 0: existing elementary events

Let the frozen expert inverse trace be

```text
E_1, E_2, ..., E_T
```

where each `E_t` is one existing `trace_plan` step after inversion. `E_t`
already contains one or more coupled source-to-sink electron moves:

```text
E_t = {a_t1, ..., a_tm}
```

and A7 executes it atomically through `apply_coupled_electron_moves`.

For each elementary event define:

- `R_t`: affected atom-map set from all source/sink containers;
- `Q_t`: electron-container preconditions read by the event;
- `D_t`: bond, formal-charge and lone-pair/container deltas produced by the
  event;
- `S_t -> S_{t+1}`: the deterministic executor transition.

A8 does not alter these elementary semantics.

## Level 1: causal continuation between elementary events

Two **adjacent** expert elementary events `E_t` and `E_{t+1}` belong to the same
candidate macro-episode when the second is a causal continuation of the first,
not merely the next text item.

The first implementation should use executor-grounded tests, in priority order.

### 1. Producer-consumer dependency

Continue the episode when the later event consumes electronic structure created
or modified by the earlier event. Operationally, this includes cases where:

- a source/sink bond in `E_{t+1}` was created or changed by `E_t`;
- a formal charge/lone-pair state required by `E_{t+1}` was changed by `E_t`;
- a source/sink container addressed by `E_{t+1}` does not exist with the same
  semantics before `E_t` but does after `E_t`.

This is the strongest causal signal.

### 2. Active-center continuity

If no exact producer-consumer edge is found, continue only when the two events
remain in the same local reaction episode. Use the authoritative boundary state
and require their affected atom sets to overlap or lie in the same small local
reaction-center component. The radius must be frozen from training/validation
only and reported; it must not depend on reaction-family labels.

### 3. Executor intervention check

For ambiguous adjacent pairs, replay the prefix and test whether `E_{t+1}` can
be moved before `E_t` while preserving formal execution and the canonical
post-pair state.

```text
S --E_t--> S_t --E_{t+1}--> S_ab
S --E_{t+1}--> S_b --E_t--> S_ba
```

If the swapped order is invalid or reaches a different canonical state, this is
evidence of a state dependency. If both orders are legal and equivalent, order
alone is not evidence that the events form one causal episode.

The existing proof-equivalence machinery can supply commuting-event utilities,
but A8 uses them for **temporal abstraction**, not merely evaluation.

## Level 2: macro-event definition

A causal macro-event is a maximal contiguous segment

```text
M_k = (E_i, E_{i+1}, ..., E_j)
```

such that each internal boundary is supported by causal continuation and the
union remains one local reaction episode.

Important boundaries:

- a transition to a disjoint reaction center starts a new macro-event;
- two independent elementary events are not merged merely to shorten the
  sequence;
- source elementary-event atomicity is never broken;
- no reaction-family or named-mechanism label is used to determine boundaries;
- no endpoint/reference lookup is used to choose a boundary.

This makes A8 a hierarchy:

```text
source/sink arrows
    -> elementary executable event (already A7)
    -> causal macro-event (new A8 unit)
    -> complete reaction
```

The scientific variable is therefore **temporal abstraction above elementary
mechanism steps**.

## Fragment horizon: compress a separate source of divergence

The full-universe diagnostic averages more fragment-import calls than mechanism
steps. A8 should therefore separate fragment chemistry from repeated import
interaction.

Recommended main interface:

```text
FRAGMENT_SET_V1
F1 = <complete explicit unmapped structure>
F2 = <complete explicit unmapped structure>
...
END_FRAGMENT_SET
```

The model predicts the complete chemical graph of every auxiliary fragment in
one initial decision. The environment may canonicalize each graph and assign
fresh persistent addresses, then commits the complete inventory in one
transaction.

The environment must not:

- search the reference precursor;
- complete a partially specified chemical graph;
- choose among candidate fragments using gold endpoint information.

This factor must be isolated experimentally. Required controls include:

```text
A7                  existing per-fragment import interface
A7 + batch-fragment same elementary-step trajectory, one fragment inventory call
A8-macro            causal macro-events with the old fragment interface
A8-full             causal macro-events + batch fragment inventory
```

This prevents a fragment-interface improvement from being misattributed to the
macro-event abstraction.

## Transactional macro execution

For a predicted macro-event

```text
M = (E_1, ..., E_m)
```

execution begins from authoritative `S_t` but uses a private sandbox copy:

```text
S_t -> E_1 -> S'_1 -> ... -> E_m -> S'_m
```

If every elementary event succeeds and the resulting state passes the existing
formal checks:

```text
commit S_{t+1} = S'_m
```

Otherwise:

```text
rollback to S_t
return first failing elementary event + stable failure code
```

A rejected macro must never partially mutate the authoritative state. This is
important because the point of temporal abstraction is to reduce accepted
off-distribution intermediate states, not to hide them inside a longer tool
call.

The model may receive one bounded retry for the current macro in the first
pilot. More general GFR-style repair remains a separate later condition because
bounded repair already exists elsewhere in MechET and is not the core A8
novelty.

## Minimal model-facing format

Do not introduce a dependency DAG or a new free-form rationale into the primary
A8 output. The model predicts one macro-event at a time:

```text
A8_MACRO_EVENT_V1
STEP
  LP(7) > BOND(3,8)
  BOND(3,7) > ATOM(7)
END_STEP
STEP
  BOND(8,9) > ATOM(9)
  LP(12) > BOND(8,12)
END_STEP
COMMIT_EVENT
```

The `STEP` boundary preserves existing elementary-event semantics. The
`COMMIT_EVENT` boundary is the new A8 abstraction.

At inference the observation is:

```text
(target product once,
 current authoritative electronic/molecular state,
 committed macro-event ledger,
 last macro outcome,
 remaining budget)
    -> next A8_MACRO_EVENT_V1 or finish_trace
```

`state_trace_v1` is therefore useful at **macro boundaries**, not after every
expert elementary step.

## Gold conversion

The conversion is deterministic and does not require new reaction annotations.

For each frozen replay-verified row:

1. keep the existing inverted `trace_plan` elementary steps unchanged;
2. compute `R_t`, container preconditions and state deltas from executor replay;
3. test adjacent causal continuation using producer-consumer dependency first;
4. use local reaction-center continuity only for unresolved cases;
5. use swap/intervention tests for ambiguous boundaries;
6. take maximal contiguous causal closures as macro-events;
7. retain the original A7 trace, macro segmentation, every authoritative state
   and the final endpoint for audit.

Every converted row must replay to the exact same endpoint as the source A7
row. Segmentation cannot drop, reorder or alter elementary moves.

## Optional data-driven macro vocabulary

Do not make this a requirement for the first A8 pilot. If the structural
segmentation is useful, a second-stage compression can canonicalize each
macro-event into a map-invariant local signature and measure recurrence across
training reactions.

If recurrent signatures exist, they can later become executable **electron-flow
options**:

```text
option ID + atom/fragment bindings -> deterministic expansion -> primitive steps
```

This would reduce generation length as well as interaction horizon. It should be
introduced only after demonstrating that the macro segmentation itself is
stable, recurrent and endpoint-preserving.

## Why ETPG is secondary rather than primary

PR #47 proposed a global partial-order electron-transfer graph. That remains a
useful analysis because it removes arbitrary total-order distinctions. It is not
preferred as the main A8 algorithm for three reasons:

1. A4 already defines one-shot complete electron-flow prediction;
2. partial-order proof equivalence and commuting-event handling already exist;
3. bounded proof repair already exists in GFR infrastructure.

A global ETPG therefore risks looking like a recombination of existing pieces.
The new A8 variable here is narrower and easier to identify:

> **change the temporal unit of closed-loop prediction from an elementary
> mechanism step to an executor-grounded causal reaction episode.**

ETPG can remain as an optional graph-output control or as machinery for deriving
macro boundaries.

## First gate: no-training segmentation audit

Do not train A8 before showing that the abstraction actually reduces the
problem.

Run the complete frozen training/validation traces and report:

- elementary steps per reaction;
- causal macro-events per reaction;
- compression ratio `T_elementary / T_macro`;
- macro size distribution;
- fraction of single-step macros;
- fraction of reactions collapsed to one macro;
- reaction-center diameter of each macro;
- recurrence of map-invariant macro signatures;
- fragment-import calls before and after batch inventory;
- total model/environment decisions before and after both changes;
- exact replay/endpoint-equivalence rate.

### Go/no-go conditions

Do not promote macro-events to A8 if either holds:

- the median/mean macro compression is negligible, for example most macros are
  single elementary steps; or
- useful compression is obtained only by collapsing nearly every reaction into
  one global program, making the condition effectively A4.

The desired regime is nontrivial intermediate abstraction: fewer decisions than
A7 while retaining multiple state-feedback boundaries on reactions that contain
causally distinct episodes.

## Second gate: matched pilot

Only after the segmentation audit passes, train a 5k--10k matched pilot on the
same frozen base model and reaction IDs.

Minimum comparison:

| Condition | Prediction unit | Feedback frequency | Fragment interface |
|---|---|---|---|
| A0 | precursor endpoint | none | endpoint text |
| A4 | complete electron-flow program | execute at end | matched frozen interface |
| A7 | one elementary mechanism step | every elementary step | current imports |
| A7-batchfrag | one elementary mechanism step | every elementary step | one inventory transaction |
| A8-macro | one causal macro-event | every macro boundary | current imports |
| A8-full | one causal macro-event | every macro boundary | one inventory transaction |

Do not add RLVR, a macro vocabulary, ETPG prediction or general repair until the
matched SFT pilot establishes a nontrivial structural endpoint result.

## Required metrics

Always pair formal execution with endpoint correctness:

- structural Pass/Success@1 and @K;
- ExecutePass@1/@K;
- endpoint correctness conditional on execution;
- macro acceptance/rejection rate;
- first-failing elementary-step distribution inside rejected macros;
- rollback/retry rate;
- expert elementary-step count;
- predicted macro-event count;
- model/environment decision count;
- fragment inventory accuracy;
- context and generated tokens;
- wall-clock rollout latency;
- performance stratified by expert mechanism-step count and macro count.

A8 is not successful if it only reduces calls or raises executability while
structural endpoint accuracy remains trivial.

## Strongest paper interpretation if validated

The intended result is not "longer chunks are better." The chemical claim is:

> Mechanistic reasoning has a hierarchy. Individual curved-arrow moves compose
> into elementary electron-flow events, and elementary events compose into
> local causal reaction episodes. Closed-loop prediction at the elementary-step
> level can overexpose a model to its own state errors, while globally predicting
> an entire reaction removes useful state feedback. An executor-grounded
> macro-event abstraction preserves electron-transfer causality at a coarser,
> chemically local temporal scale.

This would place A8 between A7 and A4 for a principled chemical reason rather
than an arbitrary fixed chunk size.

## Non-goals and claim boundaries

A8 does not claim that:

- the derived macro-event is a unique physical elementary mechanism;
- formal execution establishes kinetics, selectivity, yield or laboratory
  feasibility;
- a named reaction class defines event boundaries;
- the executor may infer missing fragment chemistry;
- macro segmentation itself proves improved learning;
- historical partial-order or repair infrastructure is newly invented here.

A8 should enter the paper only after the no-training compression audit and
matched structural endpoint pilot pass.

## Implementation checklist

- [ ] Freeze `A8_MACRO_EVENT_V1` and `FRAGMENT_SET_V1` schemas.
- [ ] Implement elementary-step footprint/container dependency extraction.
- [ ] Reuse executor replay for boundary intervention tests.
- [ ] Build deterministic macro segmentation and exact replay audit.
- [ ] Measure full train/valid compression and macro recurrence.
- [ ] Implement transactional macro dry-run / commit / rollback.
- [ ] Add one batch-fragment inventory transaction without reference lookup.
- [ ] Build matched A7-batchfrag, A8-macro and A8-full pilot rows.
- [ ] Run the 5k--10k SFT pilot only if the segmentation audit passes.
- [ ] Keep ETPG, learned options and RLVR as later ablations/extensions.
