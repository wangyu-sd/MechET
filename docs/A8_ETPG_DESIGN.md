# A8: partial-order electron-transfer program with executable repair

> **Status: design proposal.** A8 addresses a different failure mode from the
> A7 `state_trace_v1` redesign. A7 asks how to expose authoritative state and
> history during a sequential electron-flow rollout. A8 asks whether the long
> autoregressive electron-flow trajectory should be the primary prediction
> object at all. This document does not authorize training or change the frozen
> FlowER universe, benchmark denominator, paper condition, or scientific claim.

## Core challenge

The current results expose a tension between A0 and A7:

- **A0 endpoint prediction** is a short global decision and therefore avoids
  long-horizon trajectory error, but the chemical transformation remains
  implicit and need not be mechanistically faithful.
- **A7 executable electron flow** makes chemistry explicit and auditable, but
  factorizes a reaction into a sequence of autoregressive decisions. Errors in
  early electron-transfer steps change the subsequent molecular state and can
  compound into trajectory divergence.

The target A8 question is therefore:

> Can retrosynthesis retain electron transfer as the chemical coordinate while
> avoiding the instability of a fully autoregressive mechanistic trajectory?

A8 proposes to predict a **global partially ordered electron-transfer program**
first, execute it under the chemistry environment, and invoke a bounded local
repair policy only when execution reveals a conflict.

## A7 versus A8 factorization

A7 models an ordered trajectory

```text
p(tau | X) = product_t p(a_t | X, S_t, h_t),
S_{t+1} = F(S_t, a_t).
```

This is chemically explicit but long-horizon: one accepted wrong move changes
all later conditioning states. It also forces the model to learn a particular
linearization even when two electron-transfer operations are independent and
commute under the executor.

A8 instead predicts

```text
P = (F_frag, V_ET, C, E_dep)
```

where:

- `F_frag` is the set of explicitly proposed auxiliary fragments;
- `V_ET` is the set of electron-transfer nodes;
- `C` is the set of atomic coupled-move groups that must execute together; and
- `E_dep` is a directed acyclic dependency relation over nodes or coupled
  groups.

The program represents a **partial order**, not one privileged text ordering.
The executor chooses a deterministic legal topological schedule. If execution
fails, a bounded repair model may patch only the unresolved local program.

## Chemical primitive

A single electron-transfer node keeps the existing MechET chemical semantics:

```text
ET e1 = <SOURCE_KIND>(<source atoms>) > <SINK_KIND>(<sink atoms>)
```

The current closed-shell two-electron scope uses the runtime-supported
source/sink container vocabulary. A canonical example is:

```text
ET e1 = LP(7) > BOND(3,8)
ET e2 = BOND(3,7) > ATOM(7)
```

Semantically each node is

```text
a_i = (c_i^src, c_i^dst, 2).
```

A8 remains an **electron-transfer** representation, not a generic bond-edit
language. Bond and charge changes are executor consequences of source-to-sink
electron redistribution.

## Coupled moves

Some curved-arrow operations must be executed atomically because executing only
one member would create an unsupported intermediate. These are not represented
as an artificial sequence.

Example:

```text
ET e1 = BOND(7,8) > ATOM(8)
ET e2 = LP(9) > BOND(7,9)
COUPLE = {e1,e2}
```

The executor treats the coupled group as one atomic transition:

```text
S_{t+1} = F(S_t, {e1,e2}).
```

Dependencies are defined between executable units after coupled groups are
contracted.

## Dependency relation

Dependencies express chemical/state prerequisites, not arbitrary text order.

Example:

```text
ET e1 = LP(7) > BOND(3,8)
ET e2 = BOND(3,7) > ATOM(7)
ET e3 = BOND(8,9) > ATOM(9)
DEP = e1>e2, e1>e3
```

This states

```text
e1 precedes e2
e1 precedes e3
e2 and e3 are unordered relative to one another
```

If both legal topological orders execute to the same authoritative state, the
program should not force the model to distinguish them.

## Canonical A8 DSL

The first implementation should use a compact, versioned DSL rather than a
verbose tool transcript.

```text
A8_ETPG_V1
FRAG F1 = <explicit unmapped molecular structure>

ET e1 = LP(7) > BOND(3,8)
ET e2 = BOND(3,7) > ATOM(7)
ET e3 = BOND(8,9) > ATOM(9)

COUPLE = {}
DEP = e1>e2, e1>e3
END
```

The parser must reject unknown fields, duplicate IDs, cyclic dependencies,
unsupported source/sink kinds, malformed atom references, and coupled groups
with overlapping inconsistent semantics.

### Fragment boundary

The model remains responsible for the **chemical graph** of every proposed
fragment. The environment may canonicalize that graph and assign fresh
persistent addresses, but it must not complete a partially specified chemical
structure from the reference precursor or search the reference answer.

Recommended first contract:

```text
FRAG F1 = <complete unmapped SMILES or equivalent explicit graph>
```

Environment response:

```text
F1 -> canonical structure + fresh persistent addresses
```

This separates chemically meaningful fragment prediction from nuisance atom-map
serialization without turning the executor into a precursor generator.

## Deterministic execution

Given a parsed program `P`, initialize `S0 = X`. At each executor step:

1. contract each coupled group into one executable unit;
2. identify unresolved units whose predecessors in `E_dep` are all complete;
3. sort enabled units by one frozen canonical scheduling rule;
4. execute the first unit with the existing chemistry operator;
5. commit the resulting authoritative state; and
6. continue until all units are complete or a conflict occurs.

The default scheduler must be deterministic and gold-independent. A later
robustness audit may sample alternative valid topological orders to test whether
program semantics are invariant to linearization.

## Bounded executable repair

A8 does not return to unconstrained long-horizon generation after an execution
failure. It exposes the current authoritative electronic/molecular state, the
remaining program, and the failure code to a local repair policy. The allowed
patch language is deliberately small:

```text
REPLACE <et_id> = <new electron-transfer node>
ADD_DEP <u>><v>
DEL_DEP <u>><v>
DELETE <et_id>
REPLACE_FRAG <frag_id> = <complete explicit structure>
ABSTAIN
```

The repair model may not emit an independent precursor endpoint. The final
precursor remains the executor-owned result of the committed electron-transfer
program.

Use a small frozen cap such as `R_max = 2` or `3` repair rounds. Exhausting the
cap is a failed prediction and remains in the denominator.

## Role of `state_trace_v1`

`state_trace_v1` is still useful, but A8 changes its role. Rather than being the
primary long-horizon generation interface, it is the preferred candidate for
**local repair/replanning observations**:

```text
(current authoritative state,
 remaining ETPG,
 committed electron-transfer ledger,
 last failure,
 blocked failed signatures)
    -> one bounded patch
```

This preserves exact state visibility where it is needed without repeatedly
asking the model to regenerate a full sequential trace.

For the paper interpretation, the state should be treated as an authoritative
**electronic state**, conceptually `S_t = (G_t, E_t)`. The default observation
may expose graph/formal-electron information required to interpret the current
state, while full legal source/sink enumeration remains a separately named
condition so that state visibility is not conflated with an action oracle.

## Gold conversion from existing replay traces

No new reaction labels are required. Existing replay-verified A7 trajectories
can be converted into A8 targets.

Starting from

```text
tau* = (a1, ..., aT),
```

construct the electron-transfer units and infer the weakest dependency graph
that preserves executable semantics.

### Pairwise commutativity / swap test

For candidate adjacent executable units `u` and `v` from state `S`, compare:

```text
S --u--> S_u --v--> S_uv
S --v--> S_v --u--> S_vu
```

If both paths are legal and canonical state normalization gives
`S_uv == S_vu`, the pair is locally order-independent and no precedence edge is
required solely because of its order in the source trace.

If only the source order is executable, retain `u > v`. Coupled operations that
must occur atomically are detected before this test. After collecting required
precedence relations, apply transitive reduction to produce the canonical DAG.

The conversion artifact must preserve:

- stable reaction ID and split;
- original authoritative replay states and endpoint;
- original ordered expert trace for audit;
- derived A8 nodes, coupled groups, and dependency DAG; and
- proof that at least the source topological order replays to the identical
  endpoint.

## Training plan

### Stage A: global ETPG SFT

Train the same frozen base revision on

```text
mapped product -> A8_ETPG_V1
```

without an independent endpoint channel.

For the first pilot, keep the decoder simple and use canonical DSL
serialization. To reduce arbitrary ordering bias, sample multiple valid
serialization orders of independent nodes while parsing all outputs back to the
same program graph for evaluation.

### Stage B: bounded repair SFT

Build repair examples from executor failures. Separate two regimes:

1. **rejected-action conflicts:** malformed/illegal/stale proposals for which the
   authoritative state does not change;
2. **accepted off-gold states:** formally executable but incorrect decisions
   that move the rollout away from the expert path.

The second regime is scientifically important because it represents real
closed-loop compounding error. Report its frequency, recovery rate, cycle rate,
abstention rate, and endpoint outcome separately.

### Stage C: optional RLVR

Only after SFT clears the pilot gate, optionally optimize a bounded reward over
program parse validity, DAG validity, executable scheduling, non-cycle
transitions, final endpoint correctness, and limited repair use. Endpoint
correctness must dominate process rewards so an unnecessarily long or merely
executable wrong program cannot obtain a high score.

## Minimal A8 pilot matrix

Do not start with a large architecture sweep. The minimum scientific comparison
is:

| Condition | Chemical coordinate | Factorization | Execution |
|---|---|---|---|
| A0 | endpoint | one-shot endpoint | no mechanistic execution |
| A7 | electron transfer | sequential autoregressive trajectory | closed-loop |
| A8-set | electron-transfer nodes | global unordered/canonical set | executor scheduling |
| A8-graph | electron-transfer nodes + dependencies | global partial order | executor scheduling |
| A8-full | partial-order electron-transfer graph | global plan + bounded local repair | executor + repair |

This matrix answers four different questions:

- A0 versus A7: chemistry visibility versus trajectory instability;
- A7 versus A8-set: does removing long-horizon autoregressive factorization help?;
- A8-set versus A8-graph: do chemical dependencies matter beyond a bag of moves?;
- A8-graph versus A8-full: does bounded local replanning recover execution
  conflicts without reintroducing trajectory divergence?

Fragment-ID/canonicalization changes must be isolated from the main state/program
factorization comparison. At minimum compare the same A8 program format with the
old and new fragment interfaces on a matched pilot subset.

## Required pilot metrics

Always pair mechanistic reliability with endpoint correctness. Report:

- DSL parse validity;
- DAG validity / cycle rate;
- fragment validity;
- program execution@1/@K;
- structural Success/Pass@1/@K;
- full-precursor recovery where defined;
- endpoint correctness conditional on execution;
- number of electron-transfer units;
- dependency depth and width;
- coupled-group frequency;
- number of repair rounds;
- repair success rate;
- accepted off-gold recovery rate;
- final failure taxonomy; and
- token length / inference cost versus A7.

A8 is not successful if it only raises execution while structural endpoint
accuracy remains trivial.

## Adoption gates

### Gate 1: target-conversion equivalence

On a frozen 2,048-row slice require:

- identical reaction IDs and zero dropped rows;
- source A7 trace still replays identically;
- derived A8 program has at least one executable topological order;
- deterministic canonical graph/DSL rendering;
- execution of the canonical schedule reaches the identical endpoint;
- no reference endpoint, expected precursor, proof, or endpoint digest is
  model-visible; and
- report distribution of node count, dependency edges, width/depth, coupled
  groups, and serialization lengths.

### Gate 2: small global-program pilot

Train on approximately 5,000--10,000 reactions and run frozen validation. The
first decision is whether A8-set/A8-graph restores nontrivial structural endpoint
accuracy relative to A7 while retaining meaningful execution. Do not add repair
or RLVR until the global ETPG prediction itself is shown to work.

### Gate 3: repair pilot

Add bounded repair only after Gate 2 passes. Require improvement in structural
endpoint accuracy or successful executable coverage, not merely fewer parser
errors. Keep `R_max`, decoding budget, and selector gold-independent and frozen.

### Gate 4: full run

Only after the above gates pass may A8 become an approved paper condition and
run on the frozen strict executable FlowER denominator. Missing, failed, timed
out, and OOM examples remain failures in the denominator.

## Paper-level interpretation if validated

The intended claim is not merely that A8 uses a shorter prompt. The scientific
hypothesis is:

> Endpoint models avoid trajectory error but leave chemistry implicit; fully
> autoregressive mechanistic models expose electron flow but can accumulate
> state-dependent errors. A partially ordered executable electron-transfer
> program retains chemical structure while quotienting out unnecessary
> trajectory linearization, and bounded repair localizes the remaining
> closed-loop correction problem.

This is a stronger claim than history compression and should only enter the
paper after the A8 pilot establishes endpoint-level evidence.

## Non-goals

A8 does not:

- use A0 endpoint predictions as a hidden test-time answer bypass;
- allow the environment to infer missing fragment chemistry from the reference;
- claim that one formal inverse electron-flow program is the unique physical
  forward mechanism;
- replace full benchmark denominators with executor-selected subsets;
- treat formal executability as chemical correctness; or
- authorize a full training run before the equivalence and pilot gates pass.

## Implementation checklist

- [ ] Freeze `A8_ETPG_V1` grammar and parser.
- [ ] Implement coupled-group validation and DAG validation.
- [ ] Implement deterministic canonical topological scheduler.
- [ ] Build A7-to-A8 target converter with swap/commutativity audit.
- [ ] Freeze the 2,048-row equivalence artifact.
- [ ] Implement graph-level evaluation independent of DSL serialization order.
- [ ] Train A8-set and A8-graph small pilots before repair.
- [ ] Reuse/adapt `state_trace_v1` for bounded repair observations.
- [ ] Isolate fragment-interface changes in a matched pilot ablation.
- [ ] Add accepted-off-gold recovery diagnostics.
- [ ] Approve or reject A8 before any full run or paper-protocol update.
