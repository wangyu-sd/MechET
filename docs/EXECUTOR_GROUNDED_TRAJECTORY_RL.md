# Executor-Grounded Trajectory RL for MechET

## Purpose

This document replaces the earlier assumption that another atom-address or one-shot
serialization is likely to recover A7.  The current evidence points to a different
problem: MechET is being trained largely as offline sequence imitation, while the
paper claim and desired inference procedure are an **interactive executable
trajectory**.

The proposed main learning algorithm is **Executor-Grounded Trajectory RL (EGTR)**.
The name is provisional.  Its novelty is not GRPO itself.  It is the combination of:

1. **electron-flow events as the decision unit** rather than individual arrows or a
   complete one-shot program;
2. **the deterministic chemistry executor as the training environment**, with the
   same state transition contract at train and test time;
3. **reverse-horizon curriculum** over expert trajectory states so endpoint reward is
   reachable before the policy is asked to solve the full horizon;
4. **verifier-traced event credit** derived from executed molecular transitions,
   without a learned process-reward model;
5. **endpoint-dominant RL**, so formal executability is a constraint and diagnostic,
   not the optimization target;
6. **compact Markov context**: original product, current authoritative molecular
   state, last committed event / failure feedback, and a compact committed ledger,
   rather than a raw growing transcript.

The intended paper-level statement is:

> MechET learns executable retrosynthesis by reinforcement learning directly in
> electron-transfer state space.  A deterministic executor supplies the state
> transition and verifier structure; a reverse-horizon curriculum makes sparse
> endpoint reward learnable; and verifier-traced event credit assigns long-horizon
> outcome credit to the electron-flow decisions that caused it.

This keeps the precursor causally downstream of the committed executable trajectory.
No independent precursor answer is admitted at inference.

---

## 1. Evidence forcing this change

The corrected first-use checkpoint is
`first_use_state_qwen3_8b_v100_seed17_20260909/checkpoint-4019`, trained for one
full epoch on 257,167 executable FlowER training reactions.

Its interrupted 225/256 validation diagnostic is sufficient to reject several
previous hypotheses as the primary rescue path:

- syntax valid: 225/225;
- `finish()` with EOS: 225/225;
- no sample has >20 imports; maximum imports is 11;
- formal execution: 19/225 = 8.44%;
- endpoint exact: 0/225;
- 123 failures occur at `locate`;
- dominant errors are 76 `MISSING_SITE`, 47 non-equivalent SMARTS ambiguities,
  69 `STATE_ASSERTION_MISMATCH`, and 14 `INVALID_SMILES`.

Therefore first-use fragment scheduling fixed the previous import/termination
pathology, but a one-shot program still commits all grounding and downstream
state assertions before receiving execution feedback.

PR #50 independently removes Python syntax and hidden map parsing by exposing a
native transactional tool loop.  An unadapted base Qwen still fails there, which
shows that a better harness alone is not enough.  The missing experiment is to
**post-train the chemistry policy in that exact closed-loop environment**.

The horizon is also materially longer than the historical successful subset:
approximately 4.05 mechanism steps / 8.16 electron moves now versus 1.93 / 3.86
historically.  Long-horizon compounding and local grounding failure therefore have
to be addressed together.

---

## 2. MDP and action unit

For product `X`, define the deterministic executor MDP

\[
\mathcal M_X=(\mathcal S,\mathcal E,\mathcal F,R,H),
\]

where:

- `S_t` is the executor-owned current molecular state;
- `e_t` is one **electron-flow event**;
- `F(S_t,e_t)=S_{t+1}` is deterministic executor transition;
- `R` is training-only verifier / endpoint reward;
- `H` is the remaining number of mechanism events.

An electron-flow event is the set of coupled two-electron moves belonging to one
source mechanism step.  Existing source step boundaries are used; no learned
segmenter or reaction-class oracle is introduced.

This changes the average decision horizon from roughly

\[
8.16\;\text{individual moves} \rightarrow 4.05\;\text{mechanism events},
\]

while preserving every electron move and exact executor endpoint.

The policy is

\[
e_t \sim \pi_\theta(\cdot\mid X,S_t,m_t),
\]

where `m_t` is a compact execution memory, not a raw transcript.

---

## 3. Remove free-form one-shot `locate` from the main RL action

The 225-case result makes free-form SMARTS `locate` the dominant observed
failure.  EGTR therefore does not ask the model to author an entire SMARTS query,
all state assertions, and the remaining electron program before any feedback.

At each state the executor exposes the inference-available legal local containers /
site candidates.  The policy selects or scores a fully grounded event under the
current state.  No candidate can depend on the reference precursor.

The implementation may retain persistent position handles internally, but the
learning objective is over executable candidate events, not over the exact textual
spelling of arbitrary numeric identifiers.  A candidate's model score is computed
from its canonical event serialization conditioned on the current state; selection
is made from the legal candidate set.  Thus invalid addresses and chemically
impossible sources are rejected before they can become committed state changes.

This is a constrained policy, not a chemistry oracle: the executor supplies
**legality**, while the model must still decide which legal event leads toward the
correct precursor.

---

## 4. Stage 0: use the corrected SFT as the cold start

Do not train another full SFT model first.

Initialize EGTR from checkpoint 4019.  Its low token loss, reliable syntax and
termination show that it already carries useful formatting / chemistry adaptation,
even though its one-shot grounding is poor.

Before expensive RL, run only a minimal transition-reward preflight on 128--256
held-out training/validation event states with 4--8 sampled candidate events per
state.  This is not a new architecture-selection smoke.  It only checks that:

- the gold event is executable from the state;
- sampled groups have non-zero reward variance;
- successor-state / graph-delta reward code is correct;
- no test data or endpoint information is exposed to the actor.

If group variance is zero, fix sampling/reward immediately rather than starting a
large GRPO job.

---

## 5. Stage 1: Transition RLVR

The first RL stage turns each expert mechanism state into a short one-event RL
problem.  This directly fixes the local policy and avoids the sparse whole-episode
reward problem.

For expert transition

\[
S_t^* \xrightarrow{e_t^*} S_{t+1}^*,
\]

sample a group of candidate events from the current policy, execute every candidate,
and score the **executed successor**, not its text.

A practical training reward is

\[
r_t^{\rm trans}
= w_{\rm exact}\,\mathbf 1[\hat S_{t+1}=S_{t+1}^*]
+ w_{\Delta}\,F_1(\Delta\hat G_t,\Delta G_t^*)
- w_{\rm invalid}\,\mathbf 1[\text{executor rejects}],
\]

with exact successor equality dominant.  `Delta G` includes bond existence/order,
charge changes and required component participation under map-invariant structural
comparison where appropriate.

Formally executable but wrong successor states receive no success bonus.  This is
important because current full A7 has much higher Execute than endpoint accuracy;
rewarding execution strongly would optimize the wrong target.

Use GRPO / group-relative optimization over these short transitions.  Because one
training sample is one executable decision, the group advantage is already aligned
with the event and does not suffer long-trajectory broadcast credit.

Initial compute gate:

- 20k--50k expert event states, stratified by remaining horizon and failure type;
- oversample deeper states and transformations with historically low local accuracy;
- evaluate exact successor accuracy and closed-loop short-horizon endpoint recovery;
- continue only if the local event policy improves materially.

---

## 6. Stage 2: Reverse-Horizon Trajectory RL

Pure endpoint-only GRPO from full products is inappropriate at the current starting
point because endpoint success is effectively zero.  Group-relative advantages also
collapse when every rollout in a group receives the same reward.

Use the expert executable trajectory itself to define a **reverse-horizon
curriculum**.  For a gold trajectory

\[
S_0^*\to S_1^*\to\cdots\to S_T^*=R^*,
\]

train first from states near the endpoint, then progressively move the start state
toward the product:

- curriculum level H=1: start at `S_{T-1}*`;
- H=2: start at `S_{T-2}*`;
- H=3--4;
- finally H=T: start at the actual product `S_0*`.

Advance a reaction / bucket to the next horizon when the current policy reaches a
frozen success band (for example >=50% group success, with a lower threshold used
only if the reward-variance audit shows learning has saturated).  The final stage
uses the genuine product-only start distribution.

Intermediate gold states are **training start states**, not model-visible future
answers.  The same kind of authoritative state exists at inference after executor
transitions, so this is curriculum learning rather than endpoint leakage.

This curriculum converts one almost-impossible sparse-reward problem into a sequence
of increasingly long executable problems whose terminal reward is reachable.

---

## 7. Stage 3: On-policy closed-loop trajectory optimization

Once the policy can solve meaningful suffixes, run complete product-start episodes
in the exact native executor loop used at inference:

```text
product X
  -> compact current state S_t
  -> policy selects electron-flow event e_t
  -> executor executes F(S_t,e_t)
  -> typed success/failure + new authoritative state
  -> policy can revise / retry
  -> ...
  -> finish
  -> executor-owned precursor
```

A failed event does not silently poison the remaining generated suffix.  It produces
feedback before the next policy decision.  Permit a small fixed retry / undo budget
so the policy can learn self-correction from `MISSING_SITE`, invalid-source and other
executor errors.  Undone or rejected attempts are not part of the final committed
trajectory.

Train and inference must use the same interaction contract.

---

## 8. Endpoint-dominant verifier-traced credit

Do not reuse the current reward weights unchanged.  Current infrastructure gives
substantial positive credit to formal execution, while the evidence shows that many
formally executable trajectories end at the wrong precursor.

The global training objective is endpoint recovery:

\[
R_{\rm end}=\mathbf 1[\hat R=R^*].
\]

Formal execution is primarily a constraint.  Invalid/repeated operations receive a
small negative reward; formal success alone receives zero or only a very small
bonus.

For long trajectories, do not broadcast the same terminal advantage uniformly over
every event.  The executor already contains a richer verifier than a scalar reward:
it records which event changed which bonds/charges/components and the exact
successor state.

During training, use the gold trajectory privately to trace credit:

1. identify events whose executed successor matches the expert successor;
2. identify the first causal divergence from the expert-state manifold;
3. preserve positive credit for verified correct-prefix events even if a later event
   causes endpoint failure;
4. concentrate negative credit on unsupported / divergent events rather than all
   earlier correct events;
5. recognize recovery if an off-expert trajectory returns to a reference-compatible
   suffix state.

A simple advantage decomposition is

\[
A_{i,t}
= A_i^{\rm endpoint}\,w_{i,t}
+ \lambda\,A_{i,t}^{\rm transition},
\]

where the terminal verified endpoint determines the trajectory-level direction and
`w_{i,t}` redistributes its magnitude using executor-traced causal evidence.
`A_trans` is bounded and cannot overpower a wrong endpoint.

No learned process reward model is required for the first implementation.

---

## 9. Compact Markov context for long-horizon inference

The model should not reread the entire raw tool transcript at every event.

The executor is the source of truth, so the policy context contains only:

- original product `X` once;
- current authoritative molecular state `S_t`;
- current legal / grounded event candidates;
- last accepted event and most recent typed failure if any;
- a compact committed-event ledger or cumulative edit summary;
- remaining event / retry budget.

Old raw tool outputs are dropped after their information has been incorporated into
the current state and ledger.  This preserves the Markov information required for
chemical decisions without quadratic context growth.

---

## 10. Fast execution plan

The goal is a usable main result, not a large methodology sweep.

### Gate A — transition reward preflight

Use checkpoint 4019 and <=256 event states.  Require correct reward ordering and a
substantial fraction of sampled groups with non-zero reward variance.

### Gate B — Stage-1 transition RL

Run 20k--50k event states.  Evaluate held-out event exact-successor accuracy and a
small closed-loop validation subset.  If local execution remains near the current
failure regime, stop and inspect only the remaining grounding failure class.

### Gate C — reverse-horizon trajectory RL

Run a 10k--20k reaction curriculum pilot from H=1 to full product start.  Evaluate
product-start K=1 on the frozen validation subset.  Promotion requires endpoint
accuracy, not merely Execute, to rise clearly.

### Gate D — full validation

Evaluate all 2,890 FlowER validation reactions.  If endpoint recovery continues to
improve and no reward-hacking signature appears, extend RL on the full train split.

### Gate E — headline result

Freeze the policy and evaluate structural Success@1/3/5/10 on all eligible 28,967
FlowER test reactions, plus the separate complete 28,971 full-endpoint accounting.
Only after FlowER is resolved should the identical algorithm be ported to
mech-USPTO.

Do not run another independent representation branch in parallel.

---

## 11. Required ablations after the main result exists

Only after a competitive main checkpoint is obtained:

1. SFT only vs EGTR;
2. move-level vs event-level action unit;
3. endpoint-only GRPO vs reverse-horizon curriculum;
4. trajectory-level broadcast credit vs verifier-traced event credit;
5. no-feedback / stale-feedback closed-loop controls already required by the paper.

These are paper ablations, not blockers for obtaining the first main result.

---

## 12. Claim boundary

EGTR does not claim that executor acceptance establishes chemical feasibility,
kinetics, yield or selectivity.  It optimizes a formally executable inverse
transformation program against benchmark precursor recovery.

The final endpoint is still obtained only by executing the committed electron-flow
trajectory.  Training-time expert states and endpoint labels are never exposed to
the actor at test time.