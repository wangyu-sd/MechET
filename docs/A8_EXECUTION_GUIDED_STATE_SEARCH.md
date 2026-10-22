# A8: execution-guided search in electron-transfer state space

> **Status: design proposal.** This document supersedes
> `A8_CAUSAL_MACRO_EVENT_DESIGN.md` as the preferred A8 direction. The macro-event
> and partial-order ETPG ideas remain optional horizon-reduction analyses, not the
> primary algorithm. This proposal does not change the frozen FlowER universe,
> benchmark denominators, active paper condition, or submit training jobs.

## Scientific bottleneck

The current MechET results expose a concrete conflict between endpoint prediction
and executable mechanistic prediction:

- **A0** makes a short global endpoint prediction. It can achieve strong structural
  accuracy, but the model is free to reach the precursor without representing the
  electron redistribution that causally produces it.
- **A7** makes inverse electron flow explicit and executable. The precursor is
  derived only from committed environment transitions, but the policy commits to
  one state trajectory. A formally executable early mistake changes the
  authoritative state and therefore changes every later decision.

The problem is therefore not simply that A7 lacks state visibility or uses an
unfortunate serialization. The deeper issue is **single-path commitment in a
long-horizon executable decision process**.

A deterministic executor can answer:

> Is this proposed electron-flow transition formally executable from the current
> state?

It cannot answer:

> Does this executable transition keep the search on a path that can reach the
> correct precursor?

A8 addresses this missing layer.

## Core formulation: retrosynthesis as planning over executable electronic states

For a fixed product `X`, define an executor-induced state space

```text
M_X = (S, A, F, T)
```

where:

- `S` is the set of authoritative electronic/molecular states reachable from the
  mapped product and explicitly proposed auxiliary fragments;
- `A(S)` is the set of model-proposed inverse electron-flow events available at
  state `S`;
- `F(S, a)` is the deterministic executor transition, defined only when `a` is
  formally valid;
- `T` is the terminal condition reached through `finish_trace`, whose precursor
  is executor-derived.

Current A7 learns a policy over one trajectory:

```text
pi_theta(a_t | X, S_t, h_t)
S_{t+1} = F(S_t, a_t)
```

and commits immediately to the selected successor.

A8 instead treats the executor as a **symbolic chemical transition model** and
searches a graph of alternative executable states:

```text
product X
  -> propose several electron-flow actions
  -> executor expands several valid successor states
  -> merge equivalent states
  -> score goal reachability
  -> retain a bounded frontier
  -> expand / backtrack until finish_trace
  -> executor-owned precursor
```

The central algorithmic change is therefore:

> **from single-trajectory electron-flow generation to verified planning in
> electron-transfer state space.**

## Why this is distinct from existing MechET components

The proposal must be interpreted relative to existing repository conditions.

| Existing component | What it already provides | What A8 adds |
|---|---|---|
| A0 Direct | strong global endpoint prior | no mechanistic state-space search |
| A4 OpenFlow | one-shot complete electron-flow program executed at the end | no intermediate branching or state-value guidance |
| A7 MechET | closed-loop executor feedback and endpoint ownership | commits to one online trajectory |
| PR #3 proof equivalence | recognizes commuting/equivalent proof serializations | does not search alternative electronic states |
| PR #6 GFR | verifier certificates and bounded proof repair | repairs generated proof objects rather than maintaining a goal-directed electronic-state frontier |
| `state_trace_v1` | authoritative state/history presentation | observation protocol, not a search/control algorithm |

The irreducible A8 variable is **persistent alternative-state search under the
same deterministic electron-flow executor**, optionally guided by a learned
state/action reachability value.

## A8-Search: execution-guided state beam

The first A8 condition should require no new training.

At search depth `t`, maintain a frontier

```text
B_t = {n_1, ..., n_B}
```

where each node stores:

```text
state_id
state_hash
authoritative_state
fragment_inventory
committed_electron_flow
cumulative_policy_logprob
depth
terminal_status
```

For every nonterminal frontier node `n`:

1. render the exact model observation from the node's authoritative state;
2. sample or beam-decode `K` candidate next electron-flow events from the frozen
   A7 policy;
3. execute every candidate independently in a sandbox copy of the node state;
4. discard formal failures but retain their failure codes for diagnostics;
5. create one child node for every accepted transition;
6. canonicalize and deduplicate children with equivalent authoritative state;
7. score the surviving children;
8. retain the top `B` frontier nodes under a frozen, gold-independent rule.

A dead branch does not terminate the entire prediction while another viable
frontier state remains. This converts A7's irreversible early commitment into a
bounded search problem.

### Initial scoring without a learned value

The no-training pilot should first use only information already available at test
time:

```text
score(n) = normalized cumulative policy log-probability
           - alpha * repeated-state penalty
           - beta  * failed-transition burden
           - gamma * depth penalty
```

Formal execution is a hard gate, not a positive chemistry score.

Do not use the reference precursor, gold trace, forward-reference match, or any
post-hoc endpoint metric for candidate selection.

### State merging

Different action histories may reach the same authoritative electronic state.
A8 should exploit this rather than spending beam width on duplicate histories.

If canonical state normalization yields

```text
hash(S_i) == hash(S_j)
```

and the fragment inventory / required execution metadata are semantically
identical, merge the nodes and retain the better gold-independent path score.

This is a state-space operation, not proof-text deduplication.

## Backtracking and frontier semantics

A8 does not need unrestricted MCTS in the first implementation. A bounded
best-first/state-beam search is sufficient to test the hypothesis.

Recommended first grid:

```text
beam width B in {1, 2, 4, 8}
candidates per expansion K in {2, 4}
```

`B=1` should reproduce the matched single-path A7 search contract as closely as
possible. Larger `B` changes only the commitment policy, not the electron-flow
coordinate or executor semantics.

When every child of the current best node fails, search returns to the best
remaining unexpanded frontier node. The implementation must log explicit
backtracks rather than silently restarting a rollout.

## Root fragment handling

Current traces contain substantial fragment-import burden, but fragment
simplification must not be conflated with state search.

The first A8-Search pilot should preserve the current A7 fragment interface.
Only after the search effect is measured should a separately named condition
batch the complete model-predicted fragment inventory.

If batch fragment proposal is later tested:

- the model must specify the full chemical graph of every fragment;
- the environment may canonicalize and assign persistent addresses;
- the environment must not search or complete chemistry from the reference
  precursor;
- the old and new fragment interfaces must be compared on the same search
  algorithm.

## Why execution alone is insufficient

An executable state can still lie on a trajectory that reaches the wrong
precursor. Therefore a stronger A8 requires a second quantity beyond the hard
executor:

```text
V_psi(X, S) ~= probability / score that state S can reach the reference endpoint
               under the frozen continuation protocol
```

or

```text
Q_psi(X, S, a) ~= reachability value of taking executable action a from S.
```

The executor answers **can this transition occur formally?**; the learned value
answers **is this transition promising for the final retrosynthetic objective?**

The value is a search heuristic, not a replacement executor and not a direct
endpoint channel.

## Training a goal-reachability value

Do not train the value before establishing that state branching itself helps.
If A8-Search clears the no-training gate, build value supervision only from the
training/validation universe.

### Positive states

Replay-verified expert states are positive examples for reaching their known
training reference under the expert continuation.

For an expert trajectory

```text
S_0 -> S_1 -> ... -> S_T
```

store distance-to-terminal and remaining expert-step count for auxiliary
supervision.

### Off-policy / executable negative states

Generate executable alternatives from expert and model states using the same
policy and executor. Label them with **observed bounded continuation return**,
not with a claim that they are chemically impossible.

For example:

```text
return = 1 if a frozen continuation/search reaches the training reference
return = 0 otherwise within the declared budget
```

A zero label means failure to reach the benchmark reference under that bounded
protocol; it must not be described as proof that the state is chemically wrong,
because alternative valid precursors may exist outside a single-reference
benchmark.

### Pairwise successor ranking

A safer initial objective is pairwise ranking from the same parent state:

```text
score(expert successor) > score(executable alternative successor)
```

only when the alternative fails to recover the training reference under a frozen
continuation budget.

This directly trains local branch selection while reducing dependence on global
calibration.

### Optional temporal-difference view

Once rollout returns are stable, the transition contribution can be analyzed as

```text
delta_t = V(X, S_{t+1}) - V(X, S_t)
```

so the model can distinguish formally legal actions that improve estimated goal
reachability from actions that move the state away from successful training
trajectories.

This is an optional learning extension, not required for the first A8 claim.

## A8-Value: value-guided verified search

After the value pilot passes, rank accepted children by a frozen combination such
as

```text
score(n) = lambda_pi * policy_score(n)
           + lambda_v * V_psi(X, S_n)
           - lambda_d * depth(n)
           - lambda_f * failure_burden(n)
```

All coefficients must be frozen on validation only.

The final precursor remains

```text
Y = finish_trace(committed executor path)
```

and no direct `Y` prediction may enter search ranking.

## Mechanistic options are secondary horizon reduction

The causal macro-event idea from the earlier PR remains useful, but it should be
separated from the primary A8 claim.

If recurring executor-verified subtrajectories are discovered, they can become
parameterized **electron-flow options**:

```text
option = (electronic precondition,
          executable electron-flow subprogram,
          termination condition)
```

A high-level policy may propose either an option or the existing elementary A7
event. Options can reduce search depth, while state search protects against wrong
high-level commitments.

This gives a natural hierarchy:

```text
primitive/coupled electron-flow event
  -> optional reusable mechanistic option
  -> executor transition
  -> state-space search
```

Options are not required for the first A8 experiment because the main scientific
question can be tested with the existing A7 action vocabulary.

## Proposed experiment ladder

The minimum ladder should isolate one variable at a time.

| Condition | Electron-flow representation | Commitment/search | Learned reachability |
|---|---|---|---|
| A0 | none / endpoint | one-shot endpoint | no |
| A4 | complete ET program | one-shot program | no |
| A7 | existing elementary ET events | single committed trajectory | no |
| A8-Search-B2 | same as A7 | verified state beam `B=2` | no |
| A8-Search-B4 | same as A7 | verified state beam `B=4` | no |
| A8-Search-B8 | same as A7 | verified state beam `B=8` | no |
| A8-Value | same as A7 | verified state beam | yes |
| A8-Option | same executor + optional macro option | verified state beam | optional |

This ladder directly answers:

1. **A0 vs A7:** what is lost when chemistry becomes an explicit closed-loop
   trajectory?
2. **A7 vs A8-Search:** how much of the loss is caused by irreversible
   single-path commitment?
3. **A8-Search vs A8-Value:** does goal-directed reachability improve selection
   beyond model likelihood plus formal execution?
4. **A8-Value vs A8-Option:** does temporal abstraction provide additional
   efficiency/accuracy after search already controls divergence?

## No-training gate

Run A8-Search with the existing frozen A7 checkpoint before changing training.
Report on the frozen validation set first:

- Structural Pass/Success@1 and @K;
- ExecutePass@1/@K;
- endpoint correctness conditional on execution;
- success versus beam width;
- unique authoritative states expanded;
- state-merge rate;
- dead-branch rate;
- backtrack count;
- first accepted off-expert transition depth where gold traces exist;
- model calls / generated tokens / executor calls;
- wall time and peak memory;
- performance stratified by expert mechanism-step count and fragment-import
  count.

A8-Search is informative even if it is not immediately the final method:

- if modest beam width recovers structural accuracy, single-path commitment is a
  major bottleneck;
- if execution rises but structural accuracy does not, the missing component is
  goal-directed scoring rather than more formal validation;
- if wider search does not help, the proposal/representation itself is likely the
  dominant bottleneck and value training should not be used to hide it.

## Value-learning gate

Only add `V/Q` after the state-beam screen is complete. Require:

- held-out validation ranking accuracy above model-likelihood ranking;
- calibrated or at least monotonic success-versus-value bins;
- improvement in final structural endpoint accuracy under the same search
  budget;
- no test-time reference precursor or forward-reference lookup;
- explicit reporting of alternative-endpoint label ambiguity.

## Paper-level innovation if validated

The intended paper claim becomes a problem formulation rather than a prompt
engineering claim:

> **Retrosynthesis can be formulated as planning in an executable
> electron-transfer state space.** Direct endpoint models collapse chemistry into
> one answer, while single-path mechanistic agents expose chemistry but accumulate
> irreversible state errors. MechET uses a deterministic electron-flow executor
> as the transition model, preserves multiple executable electronic states during
> search, and optionally learns goal reachability to choose which chemically valid
> branches should be explored.

The conceptual hierarchy is:

```text
Electron transfer       = chemical coordinate
Executor                = deterministic symbolic transition model
Authoritative state     = world state
Policy                  = proposal prior
Reachability value      = goal-directed heuristic
Verified state search   = planning algorithm
Executor-owned endpoint = causal answer path
```

This is stronger than claiming that a longer or shorter trace format is better.
The key scientific tension becomes:

```text
endpoint accuracy without explicit chemistry
        versus
explicit chemistry under long-horizon compounding error
```

and the proposed resolution is:

```text
explicit electron-flow chemistry
+ deterministic execution
+ alternative-state search
+ goal-directed reachability
```

## Relation to the earlier A8 designs

### Partial-order ETPG

Retain as an analysis/ablation for trajectory-equivalence and program structure.
Do not make it the primary method because one-shot executable programs,
partial-order proof equivalence and repair infrastructure already exist in other
MechET conditions.

### Causal macro-events

Retain as a possible option-discovery / horizon-reduction extension. Do not make
macro segmentation a prerequisite for A8 because current A7 already operates at
coupled elementary-event granularity and state search can be tested immediately
with the existing action vocabulary.

## Claim boundaries

A8 must not claim that:

- formal executability establishes experimental feasibility, kinetics, yield or
  selectivity;
- the learned value is a physical energy or mechanistic truth score;
- failure to reach the single benchmark reference proves an alternative state is
  chemically invalid;
- state beam search itself is a new generic search algorithm;
- a deterministic inverse electron-flow path is the unique physical forward
  mechanism.

The paper novelty lies in the **executor-grounded electron-transfer planning
formulation and its experimentally isolated resolution of the A0/A7 trade-off**,
not in renaming beam search or value learning.

## Implementation checklist

### P0: no new training

- [ ] Freeze an A8 prediction artifact schema with explicit state nodes, parent
      links, state hashes, path scores, failures and terminal outcomes.
- [ ] Implement copy-on-expand executor states without mutating sibling branches.
- [ ] Reuse the exact A7 observation/action contract for each node expansion.
- [ ] Deduplicate equivalent authoritative states.
- [ ] Implement bounded state beam / best-first frontier with `B=1,2,4,8`.
- [ ] Verify that `B=1` reproduces the matched A7 single-path contract.
- [ ] Run frozen validation and report structural accuracy, execution, search
      cost, state merges and backtracks.

### P1: learned reachability

- [ ] Build replay-verified expert state examples from train only.
- [ ] Generate executable off-policy alternatives and bounded continuation
      returns without test leakage.
- [ ] Train a small value/ranking head or separate lightweight scorer.
- [ ] Freeze ranking coefficients and search budget on validation.
- [ ] Compare likelihood-only search against value-guided search.

### P2: optional temporal abstraction

- [ ] Mine recurrent executable subtrajectories only if search remains too deep
      or expensive.
- [ ] Compile verified recurring subprograms as parameterized electron-flow
      options with elementary-event fallback.
- [ ] Evaluate options under the same state-search and endpoint contract.

### Stop conditions

Do not promote A8 to the active paper method if:

- wider executable search does not recover any meaningful structural endpoint
  accuracy over A7;
- learned value improves only execution but not endpoint correctness;
- improvements require gold/reference information at test time; or
- search cost grows without a stable accuracy gain.
