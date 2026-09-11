# RLVR for endpoint-valid inverse electron-flow reasoning

> **Status: design discussion only.** This document does not authorize data
> construction, optimization, evaluation submission, or a Taiji task. A parent
> checkpoint and reward contract must be selected explicitly before any run.

## 1. Why reconsider RLVR

The current executor establishes a causal endpoint path:

```text
product -> committed inverse electron-flow trace -> finish_trace
        -> compiled proof -> executor-derived precursor
```

This guarantees that a credited precursor is derived from the committed trace.
It does not guarantee that the precursor is a useful laboratory synthesis, that
the policy stops at the best state, or that a chemically valid alternative
precursor matches the single dataset reference.

Recent two-case diagnostic rollouts exposed both sides of this distinction:

- a non-reference HOBt active-ester route had a plausible forward acylation
  core but accumulated redundant event pairs and spectator fragments;
- another rollout reached a potentially useful aryl-trifluoroacetate precursor
  motif, failed to stop, and drifted into oxygen formal charges of -3 and -4.

The RLVR question is therefore not merely how to increase `ExecutePass`. It is
how to optimize a complete inverse trajectory toward a terminal state that is
formally certified, forward-compatible with the target, chemically supported,
and concise.

## 2. Objects and validity levels

For target product `P`, policy trajectory `tau`, deterministic executor `E`,
and executor-derived precursor state `R_tau = E(P, tau)`, keep four notions
separate:

1. **Trace validity**: every committed event executes and `finish_trace`
   recompiles/replays the same trace and endpoint.
2. **State validity**: all intermediate atom/charge/valence configurations lie
   inside the declared chemistry support.
3. **Forward compatibility**: the reactive precursor core can plausibly produce
   `P` in the forward direction.
4. **Reference agreement**: `R_tau` matches the frozen dataset precursor under
   the declared structural normalization.

Reference agreement is measurable but incomplete because retrosynthesis may
have several valid answers. Formal reverse replay is not an independent
chemical reward: swapping a reversible program's source and sink containers
largely restates the executor algebra.

## 3. Original MechET plan that must remain intact

- The model never receives an independent answer channel in the main method.
- The endpoint is emitted only by `finish_trace` from committed actions.
- Failed events do not enter the proof or mutate the authoritative state.
- Gold precursors and gold remaining-step counts are never placed in public
  rollout feedback.
- The evaluator never silently completes a partial trajectory.
- A learned forward model is soft evidence or a selector; it cannot override a
  formal execution failure or author a replacement endpoint.

RLVR should improve the policy over this interface, not replace the trace with
post-hoc answer generation.

## 4. Candidate constrained objective

Use hard feasibility before scalar optimization:

```text
C_exec(tau) = every attempted action consumes budget; the trace parses and
              executes; exactly one finish_trace is attempted and succeeds;
              no action follows it; replay reproduces the same endpoint exactly
C_state(tau) = every intermediate passes the frozen chemistry-support contract
```

Only trajectories satisfying both constraints may receive positive terminal
reward. For a feasible trajectory, the candidate objective is:

```text
J(tau) = w_ref * r_reference
       + w_fwd * r_forward
       + w_stop * r_terminal
       - w_loop * c_core_cycle
       - w_tail * c_noncausal_tail
       - w_aux * c_unnecessary_auxiliary
```

Definitions:

- `r_reference`: strict structural endpoint agreement, used as a train-only
  scalar and never exposed as textual feedback;
- `r_forward`: calibrated score from an independent or cross-fitted
  precursor-to-product model after separating reactive core and auxiliaries;
- `r_terminal`: explicit successful `finish_trace`, with no automatic endpoint
  completion by the evaluator;
- `c_core_cycle`: recurrence of the product-descended reactive-core state even
  if new spectator fragments were appended;
- `c_noncausal_tail`: accepted continuation after the best *hypothetical
  terminal prefix* identified by the frozen gold-independent terminal scorer.
  It is a diagnostic/training penalty over nonterminal continuations, not an
  impossible suffix after an actual successful `finish_trace`;
- `c_unnecessary_auxiliary`: added components that do not participate in the
  net reactive-core transformation and are not required by the frozen
  necessity contract. The contract must explicitly protect catalysts,
  counterions, proton shuttles, and solvents that are necessary for a supported
  event even when they do not contribute atoms to the product.

The weights are intentionally not fixed in this proposal. Reward separability
must be measured before policy optimization.

## 5. Chemistry-support contract

Do not define chemical validity as “RDKit parsed the SMILES.” Mine the frozen
gold intermediate states for supported tuples such as:

```text
(element, formal charge, valence, aromaticity, radical count)
```

An RL rollout state outside the declared support is a hard failure unless it
belongs to an explicitly reviewed extension set. This gives a reproducible,
data-backed reason to reject states such as O(-3) and O(-4) without claiming a
universal hand-written chemistry oracle.

Counterions, solvents, catalysts, and other auxiliary fragments must be tracked
separately from the product-descended reactive core. Otherwise appending an
unused fragment changes the full-state string and defeats ordinary cycle
detection.

## 6. Termination without endpoint leakage

At each accepted state, include `finish_trace` as a real candidate action and
retain its policy log-probability. A terminal candidate may be ranked using the
same gold-independent forward/core score as any other candidate, but it receives
endpoint reward during training only after explicit model completion.

Do not let the evaluator select an arbitrary earlier state and call it the
prediction. A best-prefix analysis may diagnose overshoot, but paper credit
still requires a model-issued `finish_trace`. If termination remains the main
error after RLVR, a separately declared stop/value head can be considered; it
must not read the reference endpoint.

## 7. Feedback and optimization boundary

Public feedback may contain only executor-observable information:

- parse/schema error;
- rejected event and stable error family;
- authoritative current state after rollback;
- chemistry-support violation category;
- remaining public action budget.

It must not contain the gold precursor, gold reaction centre, gold next event,
gold distance-to-end, or a statement that an alternative endpoint matches the
reference.

The first optimizer should be a short, frozen-ID, group-relative trajectory
pilot initialized from one chosen SFT checkpoint. Do not change representation,
parent checkpoint, executor, interaction budget, and RL algorithm in the same
comparison.

## 8. Decisions required before implementation

1. **Parent checkpoint**: choose one reproducible SFT lineage. The completed
   Python electron-program checkpoint and the closed-PR in-place-grounded
   checkpoint are not interchangeable conditions.
2. **Reactive-core projection**: freeze how atom-contributing/reactive fragments
   are separated from auxiliaries.
3. **Forward validator**: choose its training corpus, cross-fitting boundary,
   calibration split, and abstention threshold.
4. **Alternative endpoints**: decide whether forward-validated alternatives
   receive reward, analysis-only credit, or human-audit status.
5. **Termination**: decide between policy-only `finish_trace` scoring and a
   separately declared stop/value head.
6. **Optimization budget**: freeze rollout IDs, candidates per target, maximum
   turns/tokens, optimizer steps, and held-out monitor before training.

## 9. No-training validation gate

Before launching RLVR, run a model-free/reward-only audit on a frozen
development set containing:

- gold executable trajectories;
- syntactically damaged trajectories;
- executable trajectories with corrupted endpoints;
- core cycles hidden by appended spectators;
- early-good/later-drifted prefixes;
- curated plausible alternative precursor routes;
- chemically unsupported high-charge states.

Proceed only if the proposed hard gates and scalar reward order these groups in
the intended direction without using test labels. Then run one small RLVR pilot;
full-data post-training remains a separate authorization decision.

## 10. Required reporting

Report the following independently rather than collapsing them into one
“validity” number:

- `TraceBoundPass` and explicit-finish rate;
- intermediate-state support violation rate;
- reactive-core cycle and post-best-prefix drift rates;
- strict reference Structural EndpointPass@K;
- independently forward-validated endpoint rate;
- fraction of non-reference endpoints sent to alternative-route audit;
- coverage/abstention and generated-token cost.

This preserves the main scientific claim—an endpoint causally produced by
electron-flow execution—while making clear which additional evidence supports
reaction-level usefulness.
