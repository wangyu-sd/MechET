# Execution-Anchored Receding-Horizon Optimization

## Method in one sentence

MechET learns executable retrosynthesis in three stages: first learn a correct
local inverse transition, then learn to condition that transition on a compact
trajectory memory, and finally correct policy-induced long-horizon drift with
executor-anchored, adaptive-horizon post-training.

We refer to the third stage as **Execution-Anchored Receding-Horizon
Optimization (EARHO)**.  EARHO is not generic hard-example mining.  Its unit of
learning is the first recoverable decision frontier of a model-generated
trajectory, and its anchors are authoritative executor transitions.

## Problem formulation

Let `P` be the product, `s_t` the authoritative molecular state owned by the
executor, `a_t` an inverse electron-flow action, and

\[
s_{t+1}=\mathcal T(s_t,a_t)
\]

the deterministic execution operator.  The endpoint is read from the terminal
executed state rather than generated through a separate answer channel:

\[
\widehat R=\mathcal D(s_T).
\]

Direct full-trajectory learning must simultaneously solve local chemistry,
memory, long-horizon exploration and delayed credit assignment.  An early
mistake also changes every later state, so teacher-forced accuracy alone need
not imply closed-loop endpoint recovery.  The staged method separates these
sources of difficulty without changing the product-only inference contract.

## Three-stage learning curriculum

```text
                 standard replayed trajectories
                              |
                              v
Product + current state --> State-SFT ----------------------+
                              | local executable policy pi_0 |
                              v                             |
Product + current state + compact accepted history          |
                 --> Trajectory-SFT --> policy pi_1         |
                                          |                 |
                                          v                 |
                                  closed-loop rollouts       |
                                          |                 |
                        executor-valid successor states      |
                                          |                 |
                       first recoverable decision frontier <-+
                                          |
                 bounded, adaptive receding-horizon updates
                                          |
                                          v
                         EARHO policy pi_2 + executor
                                          |
                                          v
                              executed precursor endpoint
```

### Stage I: State-SFT

State-SFT learns the local inverse dynamics

\[
\pi_0(a_t\mid P,s_t).
\]

Each supervised decision starts from an executor-replayed authoritative state.
The target is one import, one retrosynthetic electron-flow event, or a finish
decision.  The executor result is paired with that action.  This stage isolates
local chemical competence: tool selection, atom/bond localization, electron
direction, argument compilation and formal executability.

The current implementation is the natural-language electron-event SFT over the
complete strict-executable FlowER universe of
257,167 / 2,890 / 28,967 reactions.  These counts must not be called the
unqualified FlowER full split.

### Stage II: compressed-history Trajectory-SFT

Trajectory-SFT learns

\[
\pi_1(a_t\mid P,s_t,h_t),
\]

where `h_t` is an executor-reconstructible summary of accepted past actions.
The current molecular state carries the chemistry; the compact history carries
control-flow information that is not reliably recoverable from that state:

- ordered accepted action types;
- committed import-batch and fragment counts;
- committed electron-event count;
- the immediately preceding action and result code.

This is trajectory supervision through prefix-conditioned decision windows,
not one enormous serialized transcript.  It avoids quadratic repetition of
all preceding molecular states.  Failed actor trajectories, expected
precursors, reference successors and gold remaining-step counts are absent.
The supervised tool call and authoritative tool result are byte-identical to
Stage I; only the causal prefix available at inference is added.

### Stage III: EARHO

EARHO adapts the policy to states induced by its own closed-loop behavior.  It
does not resample every reaction from the product to the endpoint on every
update.  Instead it concentrates compute at the current competence frontier:

1. roll out `pi_1` with the real executor;
2. identify the earliest transition at which the rollout leaves a productive,
   recoverable path;
3. reset to an executor-replayed state in a small band around that frontier;
4. sample alternative first actions at the shared state;
5. execute every proposal, reject invalid/no-op/cyclic transitions, and pool
   surface-different actions that reach the same chemical successor;
6. continue a bounded set of successor states for an adaptive horizon `H`;
7. update only from executor-verified local or terminal evidence;
8. advance the frontier when the current horizon is solved reliably, while
   retaining a fixed fraction of full product-only episodes.

The local horizon is therefore determined by observed competence rather than a
single manually fixed suffix length.

## Credit assignment

For a frontier at `t*`, EARHO optimizes a bounded segment rather than assigning
one sparse endpoint reward indiscriminately to the whole trajectory:

\[
J_{\mathrm{EARHO}}(\pi)=
\mathbb E_{\pi,\mathcal T}
\left[
R_{\mathrm{endpoint}}
+\lambda_s R_{\mathrm{successor}}
+\lambda_e R_{\mathrm{execute}}
-\lambda_c R_{\mathrm{cycle}}
-\lambda_n R_{\mathrm{no\mbox{-}op}}
\right]_{t^*:t^*+H}.
\]

The benchmark-positive event remains exact endpoint recovery.  During training,
a proposed first action may also receive local positive credit when its
executed successor matches the reference successor under map-invariant chemical
equivalence.  Generated executable non-reference successors become hard
negatives for a successor-value model.

Crucially, if every sampled successor in a group is wrong, the group receives
no positive policy advantage.  Relative normalization must not promote the
least-wrong action from an all-negative group.  A verified reference transition
supplies supervised replay for that state instead.

## Training-private anchors and inference contract

Reference successors and endpoints are privileged training signals only.  They
may score or construct replay examples, but they are never rendered in the
actor prompt and never determine inference-time stopping.

At inference the model receives only:

- the product;
- the executor's current public state representation;
- the compact history reconstructed from accepted actions;
- public execution success/failure feedback.

The learned successor-value model may rank candidate successor states using
only the product, current state, candidate successor and terminal flag.  Exact
endpoint scoring remains an evaluator, not an inference oracle.

## Why this is not ordinary hard-case RL

| Generic hard-case RL | EARHO |
|---|---|
| selects examples by loss or failure | locates a decision frontier in an executed trajectory |
| often retains one sparse terminal reward | assigns credit to authoritative local successors and endpoints |
| resamples complete episodes | resets to verified states and uses an adaptive bounded horizon |
| compares surface action strings | pools actions by executed chemical successor |
| may reinforce the least-bad failed sample | gives zero positive advantage to all-negative groups |
| treats replay as a buffer heuristic | uses verified transition replay to restore missing local support |

The algorithmic object is therefore a receding-horizon policy-improvement
operator over executable chemical states, not a hard-example sampler.

## Efficiency mechanisms

EARHO reduces post-training cost through:

- executor resets near the competence frontier;
- adaptive horizons rather than full-length rollouts for every update;
- bounded branching and successor-state deduplication;
- caching by canonical executor state;
- updating the first consequential action rather than copying its advantage to
  every later token;
- retaining full episodes only as a controlled rehearsal and evaluation stream.

Efficiency must be reported as executed transitions, generated tokens,
wall-clock time and endpoint successes—not only optimizer steps.

## Evidence required for the long-horizon claim

The method is designed to reduce long-horizon drift; that conclusion requires
the following matched comparison:

| Condition | Purpose |
|---|---|
| State-SFT | local inverse-dynamics competence |
| State-SFT + compressed-history Trajectory-SFT | value of causal trajectory memory |
| State-SFT + Trajectory-SFT + EARHO | value of execution-anchored policy improvement |

All conditions must use the same held-out reaction IDs and endpoint evaluator.
Report:

- product-only strict endpoint accuracy and Pass@K;
- formal execution and productive-execution rates;
- first-error depth and success as a function of reference trajectory length;
- invalid, no-op, repeated-import and cycle rates;
- map-invariant successor accuracy at matched decision states;
- generated tokens, executed transitions and wall-clock cost.

The key long-horizon result is a flatter degradation curve with trajectory
length and a later first-error frontier, not merely a lower teacher-forced
loss.  Until these measurements are complete, the appropriate claim is that
EARHO *targets* long-horizon error accumulation rather than that it has already
solved it.

## Repository mapping

| Stage | Protocol / implementation |
|---|---|
| State-SFT | `docs/EXECUTABLE_INVERSE_DYNAMICS.md`, `scripts/build_natural_language_event_sft.py` |
| Trajectory-SFT | `docs/COMPRESSED_HISTORY_TRANSITION_SFT.md`, `scripts/build_natural_language_history_sft.py` |
| EARHO collection and updates | `docs/NATURAL_LANGUAGE_ANCHOR_BRANCH_RL.md`, `scripts/run_natural_language_anchor_branch_rl.py` |
| frontier sampling and promotion | `src/mechet/anchor_branch_rl.py` |
| successor equivalence and value | `src/mechet/successor_value.py`, `src/mechet/constrained_event_search.py` |

The existing code uses the historical names `anchor_branch` and
`successor_horizon`; these are implementation components of EARHO.  Renaming
the scientific method does not alter frozen artifact identifiers or prior run
lineage.
