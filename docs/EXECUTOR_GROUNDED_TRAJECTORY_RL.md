# Executor-Grounded Event RL for MechET

## Decision

The next MechET main model should not learn atom-map numbers or free-form SMARTS as
its chemical addressing system.  It should reason over **elementary electron-flow
events** grounded by the executor and should be post-trained in the same closed-loop
environment used at inference.

The method is intentionally one model and one training job.  Reverse-horizon
curriculum, gold event imitation and endpoint-oriented RL are mixed inside that run;
they are not separate checkpoints that must be trained serially.

The immediate prerequisite is one cheap **pure-Qwen3-8B zero-shot smoke** on frozen
validation states.  No old adapter is used for that smoke.

---

## 1. Evidence that determines the design

The corrected first-use checkpoint
`first_use_state_qwen3_8b_v100_seed17_20260909/checkpoint-4019` was trained for one
full epoch on 257,167 executable FlowER rows.  In the interrupted 225/256 validation
diagnostic:

- syntax valid: 225/225;
- `finish()` with EOS: 225/225;
- formal execution: 19/225 = 8.44%;
- endpoint exact: 0/225;
- 123/225 fail at `locate`;
- 76 are `MISSING_SITE`;
- 47 are non-equivalent SMARTS ambiguities;
- 69 show `STATE_ASSERTION_MISMATCH`;
- only 5 fail at fragment import.

Thus first-use scheduling solved the earlier import/termination pathology, but the
model still fails mainly when a generated text address must bind to a concrete
chemical site.  PR #50 also showed that replacing Python parsing with native tools
is not sufficient by itself.

At the same time the current full executable universe is longer than the historical
successful subset: approximately 4.05 mechanism steps / 8.16 electron moves versus
1.93 / 3.86 historically.  The method must therefore fix both **grounding** and
**long-horizon compounding**.

---

## 2. One action is one elementary electron-flow event

The correct action granularity already exists in the trace data.  Every
`trace_plan.steps[t]` contains a set of coupled electron moves and is submitted by
`apply_coupled_electron_moves` as one executor transition.

Define

\[
e_t = \{m_{t,1},\ldots,m_{t,k}\},
\qquad
S_{t+1}=\mathcal F(S_t,e_t).
\]

Each move remains an explicit directed two-electron transfer:

\[
\text{source container}\rightarrow\text{sink container}.
\]

Examples include

- lone pair on a nucleophile -> electrophilic atom, coupled with leaving-group bond
  -> leaving-group atom;
- sigma bond -> new bond, coupled with leaving-group bond -> leaving-group atom;
- pi bond -> atom together with the complementary bond-forming move.

The event is transactional: all moves are validated against the same pre-event
state and committed together, or the event is rejected and the state is unchanged.
This preserves explicit electron flow while reducing the autoregressive state horizon
from roughly 8.16 individual moves to roughly 4.05 mechanism events.

This is not a reaction-template label.  The event still records the actual source,
sink and electron motion.

---

## 3. Grounding: maps stay private, SMARTS generation disappears

The executor already knows the concrete mapped molecular graph and can enumerate
electron containers.  Those private atom maps remain the execution identity but are
not the model's prediction target.

For the main model, the executor exposes ephemeral grounded handles plus local
chemical descriptions.  Conceptually:

```text
SOURCE A
  type: lone pair
  atom: O
  charge: -1
  local environment: O-C(=O)

SOURCE B
  type: bond
  bond: C=O
  local environment: N-C(=O)-O
```

and analogous sink handles.  Handles are deterministically shuffled per state so a
model cannot attach chemistry to a stable ID.

A model event can then be represented compactly as

```text
EVENT
  SOURCE A -> SINK Q
  SOURCE B -> SINK P
COMMIT
```

while the executor privately resolves `A/Q/B/P` to mapped atoms and executes the
coupled event.  No SMARTS is generated and no atom-map integer must be copied by the
model.

For symmetry, actions producing the same authoritative successor state are the same
execution-equivalence class.  Evaluation must not penalize two site labels that
produce an identical chemical successor.

---

## 4. Pure-Qwen grounded-event smoke — run this first

This is the only pre-training smoke now required.

### Model

Use the frozen **Qwen3-8B base checkpoint only**:

- no LoRA / PEFT adapter;
- no first-use checkpoint;
- no legacy compact-full-state checkpoint;
- thinking disabled for the multiple-choice score;
- no free-form generation.

### Data

Use the already frozen seed-17 validation selection.  At each gold mechanism event:

1. materialize the authoritative pre-event state, including first-use imports for the
   diagnostic;
2. keep the full gold coupled electron-flow event;
3. construct hard alternatives by replacing one source or sink with an
   executor-grounded container of the same type;
4. execute every alternative and keep only formally executable successors;
5. deduplicate by successor molecular state;
6. shuffle labels `A`--`H` deterministically.

The candidate generator used in this smoke is intentionally **gold-derived** so the
experiment is a capability diagnostic, not a deployable inference algorithm.  It
must never be reported as endpoint accuracy.

The Qwen prompt contains the unmapped target, the unmapped current state, the
committed electron-flow history, and self-contained chemical descriptions of each
candidate event.  It never contains the private atom maps used by execution.

The evaluator scores candidate labels by conditional log probability instead of
asking Qwen to generate a mechanism.

Report:

\[
\mathrm{Top1},\quad
\mathrm{Recall@2},\quad
\mathrm{Recall@4},\quad
\mathrm{mean\ gold\ rank}
\]

by event depth and candidate count, together with the random-choice baseline.

### Interpretation

This smoke answers only one question:

> after syntax and site-address generation are removed, does an unadapted Qwen3-8B
> already contain useful chemical preference over explicit electron-flow events?

If Top1 is materially above the randomized baseline or Recall@4 is strong, the
failure of prior A7 variants is largely a grounding/trajectory interface problem.
If Qwen is near random, the unified training run simply starts with a larger event
imitation weight; there is no need to open another representation branch.

Implementation:

- `src/mechet/grounded_event_smoke.py`
- `scripts/build_qwen_grounded_event_smoke.py`
- `scripts/eval_qwen_grounded_event_smoke.py`
- `scripts/run_taiji_qwen_grounded_event_smoke.sh`

---

## 5. Main training: one joint run, not a four-stage pipeline

After the smoke, launch one Qwen3-8B training job in the grounded event environment.
Do not first train another SMARTS-SFT model.

The run mixes two losses with the same model and optimizer:

\[
\mathcal L
=
\mathcal L_{\rm RL}
+\lambda(s)\,\mathcal L_{\rm event\ imitation}.
\]

`L_event_imitation` teaches the explicit source->sink event from the gold trace.
`L_RL` trains complete executor interaction and endpoint recovery from states visited
by the current policy.  The imitation coefficient decays during the same run, for
example from a large early value to a small stabilizing value; this is a schedule,
not a separate checkpoint.

The reason for retaining a bounded imitation term is empirical: the existing SFT
can learn syntax while failing execution, but pure endpoint RL begins with very
sparse success.  Joint training gives the model a local electron-flow signal while
forcing the same weights to optimize closed-loop outcomes.

---

## 6. Mixed-horizon sampling inside the same training job

For a gold trajectory

\[
S_0^*\to S_1^*\to\cdots\to S_T^*=R^*,
\]

each RL episode samples a training start state `S_k*` from a curriculum distribution.
Early in training, more episodes begin near the endpoint; as success improves, mass
moves toward `k=0`, the real product start.

A simple initial mixture is:

- 30% full product start (`k=0`);
- 35% random middle state;
- 35% one- or two-event suffix state.

The proportions change continuously within the same run.  No optimizer reset and no
new model are introduced.

This makes endpoint reward reachable immediately while continually training the full
long-horizon task.  The final evaluation always starts from the actual product.

---

## 7. Reward: endpoint dominates; executor supplies dense bounded shaping

Do not optimize formal execution as the main reward.  Existing results already show
that executable wrong trajectories are common.

Use an endpoint-dominant bounded return:

\[
R
= w_{end}R_{end}
+ \sum_t r_t^{progress}
+ w_{exec}R_{exec}
- w_{bad}R_{bad}.
\]

The endpoint term has the largest possible magnitude:

\[
R_{end}=\mathbf 1[\hat R=R^*].
\]

For training only, define a potential from the mapped benchmark precursor, e.g. a
component-aware bond/charge edit distance `d(S,R*)`.  Use potential-difference
shaping

\[
r_t^{progress}=\operatorname{clip}
\left(d(S_t,R^*)-d(S_{t+1},R^*),-c,c\right).
\]

The entire progress contribution is capped so it can never outweigh a wrong final
endpoint.  Formal event execution receives only a small bonus; rejected events,
state cycles and wasted retries receive penalties.

The gold precursor and distance are training-only reward information.  They are not
placed in the model observation and do not exist at inference.

The repository already contains GRPO infrastructure; the new run should extend that
code rather than implement another RL framework.

---

## 8. Train/inference interaction contract

Training and test use the same causal loop:

```text
original product
  -> executor current state
  -> grounded source/sink handles
  -> LLM commits one coupled electron-flow event
  -> executor validates and executes transactionally
  -> updated authoritative state or typed rejection
  -> next event
  -> ...
  -> finish_trace
  -> executor-owned precursor
```

The model may retry after a rejected event, but rejected or undone proposals do not
enter the committed electron-flow trajectory.

The context contains only the information needed for the next decision:

- original product once;
- current authoritative state;
- grounded local handles;
- compact committed-event ledger;
- most recent typed failure;
- remaining event/retry budget.

Do not keep a growing raw transcript if the same information is already represented
by the current state and ledger.

---

## 9. Fragment handling for the first main run

Do not open a second fragment-retrieval project before the event policy is tested.
The 225-case diagnostic had only five fragment-import failures, far fewer than the
123 locate failures.

For the first main run, retain the corrected first-use fragment channel and change
only the site/electron-event decision.  If the grounded-event model later exposes a
clear fragment bottleneck, fragment selection can be replaced by a train-only
retrieval/ranking head as a separate ablation.

---

## 10. Fast promotion rule

The goal is one reliable main checkpoint.

1. Run the pure-Qwen grounded-event smoke.
2. Freeze the event interface and candidate/handle contract.
3. Launch one joint imitation + mixed-horizon GRPO run.
4. During training, evaluate a small frozen validation subset from the real product
   start; promote only when **endpoint** accuracy rises, not merely Execute.
5. Evaluate all 2,890 FlowER validation reactions at K=1.
6. If validation remains positive, freeze the checkpoint and run K=10 on the strict
   28,967 executable test denominator plus the separate 28,971 endpoint accounting.
7. Only then port the same algorithm to mech-USPTO.

Do not start another numbering system, SMARTS variant, harness rewrite, MCTS/value
model or multistep-planning branch before this run is resolved.

---

## 11. Paper-level novelty

The scientific contribution is not "we used GRPO".  It is the integrated
computational contract:

1. **electron-flow event actions** preserve explicit curved-arrow information while
   reducing effective decision horizon;
2. **executor-grounded addressing** separates chemical reasoning from fragile text
   localization;
3. **closed-loop trajectory post-training** exposes the model to its own state
   distribution;
4. **mixed-horizon endpoint RL** makes long-horizon reward learnable in one run;
5. **executor-derived bounded process shaping** improves credit without allowing
   formal execution to replace endpoint correctness.

The endpoint remains causally downstream of the committed electron-flow trajectory:

\[
e_t\sim\pi_\theta(\cdot\mid X,S_t,h_t),
\qquad
S_{t+1}=\mathcal F(S_t,e_t),
\qquad
Y=g_X(e_{0:T}).
\]

That is the method to test next.
