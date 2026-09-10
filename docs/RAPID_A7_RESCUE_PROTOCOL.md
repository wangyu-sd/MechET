# Rapid A7 rescue protocol

## Goal

Recover a **credible MechET main result quickly** without opening another representation/runtime branch.

The current evidence already rules out several explanations:

- changing numerical atom identifiers does not fix the model;
- changing to executor-owned `pN` positions does not fix the model;
- replacing Python/text parsing with native Qwen tool calls does not fix the model;
- compact state serialization reduces token cost but does not by itself restore endpoint accuracy;
- token-level SFT loss can become very small while formal execution and endpoint accuracy remain poor.

The remaining problem is best treated as a coupled failure of:

1. **local decision quality**: the policy often chooses a chemically wrong source/sink or fragment;
2. **long-horizon compounding**: a wrong accepted step changes the authoritative state and all later decisions.

We therefore stop iterating on address syntax, SMILES marking, harness design, MCTS, value models, or multistep planning until this protocol resolves those two quantities.

---

## 1. Working model of the failure

For an executable trajectory with `T` model decisions,

\[
P(\text{whole trajectory correct})
\approx
\prod_{t=1}^{T} p_t,
\]

where `p_t` is the probability that the policy makes a correct local decision at state `S_t`.

The historical successful FlowER subset averaged approximately:

- 1.93 mechanism steps;
- 3.86 electron moves;
- structural Pass@1 = 75.0%.

The current full executable FlowER universe averages approximately:

- 4.05 mechanism steps;
- 8.16 electron moves.

Thus the current task roughly doubles the decision horizon. However, the observed full-universe A7 accuracy is far lower than can be explained by doubling horizon alone. This means the current local policy is also substantially weaker.

The immediate objective is therefore not to invent a new model class. It is to measure and then improve the two factors directly:

\[
\boxed{\text{local action quality } p}
\qquad\text{and}\qquad
\boxed{\text{effective horizon } T}.
\]

---

## 2. Smoke 1 — local-policy competence on gold states

### Question

Does the current checkpoint know the correct next electron-flow decision when exposure bias is removed?

### Data

Use **256 validation reactions**, selected deterministically and stratified by expert mechanism length:

- short: 1--2 mechanism steps;
- medium: 3--4 steps;
- long: >=5 steps.

Do not use test reactions.

### Frozen execution snapshot (2026-09-10)

The model-free gate has started on the compact-full-state validation lineage.
The seed-17 selection contains 86 short, 85 medium and 85 long reactions: 256
reactions, 1,019 mechanism events and 2,078 electron moves. The selected-ID
manifest SHA-256 is
`599bcc59d819f53ca3d863cffb0ecbf139291acfe1b985b97910c1167527df63`;
the complete selected-row file SHA-256 is
`d16405259dd60bc6ef6f73fd5a3551827ef2b2ec672380887a09b0e6957d9f8a`.

All 256 supervised conversations align with their authoritative trace events.
The historical model-visible `inspect_state` inventory covers 959/1,019 gold
events. A rescue-only executor-aligned inventory covers all 1,019, exposing a
previously hidden observation mismatch rather than silently changing old
checkpoint inputs. Exact executor replay plus authoritative-successor equality
passes for 1,013/1,019 events (99.41%). The six retained failures are three
BE-DELTA events with missing explicit-H maps and three aromatic successor
mismatches. They remain in every denominator and count as misses.

The preregistered model-free promotion threshold is 99.0% event coverage, so
the K=1 F-oracle model diagnostic proceeds. This tolerance prevents six known
representation/runtime edge cases (0.59%) from blocking diagnosis of the much
larger policy failure; it does not relabel those events as correct or remove
them from later metrics.

The evaluator treats NF4 as an optional acceleration: it uses bitsandbytes
when available and otherwise loads Qwen3-8B in native BF16 on A100. This keeps
the frozen Taiji `meteor` environment runnable without installing packages at
job startup or changing the evaluated adapter.

### Observed checkpoint-lineage correction and first-use diagnostic

The completed 1,019-event gold-prefix run used the legacy 2026-08-26
`tool_sft_flower_compact_full_state_qwen3_8b_a100_20260826` adapter. That
checkpoint was trained before first-use fragment scheduling and must not be
used to judge the later interleaved-import representation. Its unconstrained
greedy result was 455/1,019 exact events and 688/1,019 formally executable
events. At depth one, 241/256 generations selected another `import_fragment`;
this is a legacy stage-routing result, not evidence that the corrected
first-use scheduler failed. This run also did not compute the required
candidate Rank@1/2/4/8 and is therefore not the final Smoke-1 decision metric.

A separate diagnostic used the corrected final first-use checkpoint at step
4,019 (one epoch), trained on 257,167 rows with imports scheduled immediately
before first electron use. It used the exact same seed-17 validation IDs as
this protocol and product-only input. The user stopped the run at 225/256, so
the retained artifact is explicitly partial and is not a benchmark result.
The frozen partial counts were:

- syntax valid: 225/225;
- `finish()` present with EOS: 225/225;
- more than 20 imports: 0/225; maximum imports: 11;
- formal execution: 19/225 (8.44%);
- full and neutralized endpoint exact: 0/225;
- failure stage: 123 `locate`, 78 program/proof, 5 fragment import, 19
  formally executable but endpoint-wrong;
- dominant errors: 76 `MISSING_SITE`, 47 non-equivalent SMARTS ambiguities,
  69 `STATE_ASSERTION_MISMATCH`, and 14 `INVALID_SMILES`.

This separates two conclusions. First-use scheduling and termination are
working at generation time: the previous runaway/import-stage behavior is not
present. The remaining bottleneck is chemical grounding. The model authors a
complete SMARTS query in a single uninterrupted program; a too-specific query
misses the site, a broad query binds non-equivalent sites, and neither error is
returned to the model before the rest of the program is generated. Predicted
`state(...)` statements are assertions evaluated only after generation, not
environment feedback, so one wrong binding or electron move invalidates the
remaining suffix.

The selected reference set itself has two retained runtime-sensitive state
assertion failures (`flower_mech_proof_val_2377` and
`flower_mech_proof_val_713`). They are not filtered and cap the local reference
replay at 254/256 until symmetry/binding portability is repaired. This 0.78%
reference issue is recorded separately and cannot explain the observed model
failure rate.

The next intervention must therefore isolate `locate`: compare the current
free-form SMARTS authoring contract with executor-enumerated local site
candidates while keeping the same product inputs, electron-event targets,
checkpoint family and denominators. Do not attribute a gain from candidate
enumeration to horizon rescue, and do not launch a full retraining run before
the fixed validation smoke shows that local site coverage improves.

### Evaluation unit

Evaluate every expert decision state `(X, S_t, a_t*)` from those 256 reactions.

For each state:

1. reconstruct the exact authoritative gold state `S_t`;
2. enumerate all executor-legal local electron actions available at that state;
3. verify that the gold next action is included in the legal set;
4. score each legal action using the **current checkpoint** without changing model weights;
5. report the rank of the gold action.

The executor/data-contract invariant is evaluated under **F-oracle**:

\[
\text{GoldLegalCoverage}_{F\text{-oracle}} \approx 100\%.
\]

If this fails, stop: the executor/data contract is still wrong and no new
training should be launched. Under **F-current**, report gold-action coverage
separately as the fragment-selection diagnostic; do not treat missing
expert-required fragments there as an executor/data-contract failure.

### Required metrics

Report:

- gold-action Rank@1 / Rank@2 / Rank@4 / Rank@8;
- source-site accuracy;
- sink-site accuracy;
- source-type accuracy;
- sink-type accuracy;
- exact coupled-move/event accuracy where applicable;
- metrics by trajectory depth `t`;
- metrics by short / medium / long reaction strata.

### Fragment isolation

Run the same local-action audit under two conditions:

**F-oracle**: make the expert-required participating fragment available to the executor before scoring the electron action.

**F-current**: use the current fragment-selection/import behavior.

This isolates:

\[
\Delta_{frag}
=
\mathrm{Rank@K}_{F\text{-oracle}}
-
\mathrm{Rank@K}_{F\text{-current}}.
\]

Do not interpret F-oracle as a model result. It is only a diagnostic.

### Decision rule

The useful quantity for greedy long-horizon survival is **Rank@1**. With an
average mechanism-level horizon near four, a per-event top-1 probability around
0.84 is needed for an idealized 50% whole-trajectory survival rate:

\[
0.5^{1/4.05}\approx0.84.
\]

Therefore:

- **Rank@1 >= 85% on F-oracle**: the greedy local policy is sufficiently strong;
  prioritize horizon rescue.
- **Rank@1 < 85% but Rank@4 >= 85% on F-oracle**: candidate coverage is
  strong but ordering is weak. A `B=4` beam is a useful diagnostic, but this is
  not evidence that the greedy policy is locally strong; structured ranking
  remains the training route unless beam improves the closed-loop endpoint.
- **Rank@4 < 85% on F-oracle**: local candidate scoring itself is inadequate;
  train a structured event/action policy before doing search.
- **F-oracle high but F-current drops >10 percentage points at the same K**:
  fragment prediction is a major bottleneck and must be separated from
  electron-action prediction.

This one smoke decides the next experiment. Do not launch several alternatives in parallel.

---

## 3. Smoke 2 — quantify the long-horizon penalty directly

### Question

How much accuracy is lost because the policy leaves the expert state distribution?

Use the same 256 validation reactions and the same checkpoint.

### A. Gold-state one-step reset

At every expert state `S_t`, reset the environment to that exact state and ask the model for only the next decision.

This produces a local competence curve:

\[
p_t=P(\hat a_t=a_t^*\mid S_t=S_t^*).
\]

Plot/report `p_t` versus depth.

If `p_t` is already poor at `t=1`, this is not primarily a long-horizon problem.

If `p_t` is strong at every gold state but closed-loop success collapses, exposure bias is the dominant problem.

### B. Gold-prefix / free-suffix intervention

For each trajectory, let `T_event` be its number of expert mechanism events.
Force the first `k_event` expert events through the executor, then let the model
continue freely. `T_move`, the number of individual electron moves, is recorded
separately and is never used to choose an event prefix.

Use:

- `k_event=0`;
- `k_event=1`;
- `k_event=2` when `T_event >= 3`;
- `k_event=floor(T_event/2)`;
- `k_event=T_event-1`.

Deduplicate repeated values for short trajectories.

Report final structural endpoint success after each intervention.

Interpretation:

- a large jump from `k=0` to `k=1` means early irreversible commitment dominates;
- gradual improvement with larger `k` means errors are distributed across the whole trajectory;
- high `k_event=T_event-1` but poor free rollout means the final local decision
  is learnable but compounding is severe;
- poor `k_event=T_event-1` means local action prediction remains weak even
  without long history.

This is the cleanest causal smoke for the long-horizon hypothesis and requires no new training.

---

## 4. Smoke 3 — reduce the effective horizon without changing the chemistry

The current data distinguish **mechanism steps** from individual **electron moves**. The current full universe averages ~4.05 mechanism steps but ~8.16 electron moves.

Before inventing macro-actions, test the smallest possible horizon reduction:

> one model decision = one source mechanism event containing all electron moves that must be committed together.

Use the existing source mechanism-step boundaries. Do not infer new reaction classes and do not introduce a learned segmenter.

For each trajectory:

1. group electron moves belonging to the same source mechanism step;
2. execute the grouped event transactionally with the existing coupled-move executor path;
3. require exact equality of intermediate authoritative state after each grouped event;
4. require exact final precursor equality;
5. record the new decision count.

Adoption gate:

- replay/endpoint equivalence = 100%;
- no new data filtering;
- mean decision horizon reduced by at least 1.5x.

Expected target from the current statistics is roughly:

\[
8.16\;\text{moves}
\rightarrow
4.05\;\text{events}.
\]

If this gate passes, use mechanism events as the next A7 action unit. This preserves electron flow while removing unnecessary autoregressive depth.

---

## 5. Rescue route A — if local Rank@4 is already high

If Smoke 1 gives F-oracle Rank@1 >=85%, or Rank@4 >=85% with a material
Rank@1--Rank@4 gap, do **not** retrain immediately. In the second case the beam
is explicitly an ordering diagnostic, not evidence of a strong greedy policy.

Run a minimal executor-guided state beam on validation:

\[
B\in\{1,2,4\}.
\]

At each state:

1. obtain the model's top local candidate actions/events;
2. execute each candidate independently;
3. drop formal failures;
4. deduplicate identical successor molecular states;
5. retain the top `B` successor states by frozen model score only;
6. finish only through the executor-owned endpoint.

No value model, no MCTS, no learned reward, no endpoint oracle.

Required validation metrics:

- structural Success@1 and Success@4;
- Execute@1 and Execute@4;
- unique states expanded;
- average executor calls;
- improvement as a function of expert trajectory length.

### Beam adoption gate

Proceed to full validation only if `B=4` gives a **clear structural endpoint gain**, not merely higher execution.

A8-style search is rejected if it only turns wrong trajectories into executable wrong trajectories.

---

## 6. Rescue route B — if local Rank@4 is low

If F-oracle Rank@4 <85%, the next training run should change the **learning target**, not the identifier system.

### Structured action/event objective

Do not make the LLM learn a long JSON string as the primary objective.

For each gold state, construct:

\[
\mathcal A_{legal}(S_t)
\]

and train the model to rank the expert action/event above legal negatives:

\[
\mathcal L_{rank}
=
-\log
\frac{\exp s_\theta(S_t,a_t^*)}
{\sum_{a\in\mathcal A_{legal}(S_t)}\exp s_\theta(S_t,a)}.
\]

Use hard negatives preferentially:

- same source, wrong sink;
- same sink, wrong source;
- same reaction-center neighborhood, wrong electron direction;
- formally executable actions that lead to a different successor state.

Keep the executable trajectory and endpoint contract unchanged.

### Turn-balanced supervision

If token SFT is retained as an auxiliary loss, normalize per decision/event so that fixed syntax and long SMILES do not dominate the gradient:

\[
\mathcal L_{turn}
=
\frac1T
\sum_t
\frac1{|a_t|}
\sum_j
-\log p(a_{t,j}).
\]

The structured decision loss is primary; token SFT is auxiliary.

### Pilot size

Do not train the full corpus first.

Use a deterministic **10k--20k train reaction pilot** and evaluate on the same 256-smoke set plus the full 2,890 FlowER validation set if the smoke improves.

Pilot success criterion:

- large increase in gold-state Rank@1/4;
- no loss of executor legality;
- closed-loop structural endpoint accuracy improves materially over the current A7 checkpoint.

Only then run a full-corpus epoch.

---

## 7. Fragment handling: only separate it if the smoke proves it matters

Do not redesign fragments pre-emptively.

If F-oracle versus F-current shows a >10-point Rank@4 gap, split the decision into:

\[
\text{fragment choice}
\rightarrow
\text{electron event choice}.
\]

First measure train-vocabulary coverage of required validation fragments.

If coverage is high, use train-only fragment retrieval/ranking with a free-generation fallback. If coverage is low, keep generative fragment prediction.

The objective is to stop forcing the electron-action policy to simultaneously solve open-vocabulary fragment generation and local electron-flow selection when the data show that separation is beneficial.

---

## 8. Recovery training is P1, not the first rescue

Only after gold-state local competence is strong should we address off-expert states.

If:

- gold-state Rank@4 is high;
- event-level horizon is reduced;
- small beam helps but a large free-rollout gap remains;

then collect executable student states from the current policy and perform a small DAgger-style recovery continuation.

Do not use PR #49-style resets from only expert states as evidence for off-policy recovery. The recovery set must contain states actually visited by the model.

A correction is added only when a bounded executor search can find a valid continuation back to the reference endpoint or a reference-compatible successor. Do not fabricate recovery labels for irrecoverable states.

No RL is required for the first recovery experiment.

---

## 9. Fast decision tree

```text
Current checkpoint
    |
    v
Smoke 1: gold-state legal-action/event ranking
    |
    +-- GoldLegalCoverage_F-oracle < ~100%
    |       -> data/executor bug; stop training
    |
    +-- F-oracle Rank@4 < 85%
    |       -> local policy is weak
    |       -> event-level structured ranking + turn-balanced pilot
    |
    +-- F-oracle Rank@4 >= 85%
            |
            +-- F-current drops >10 pp
            |       -> fix/separate fragment selection first
            |
            +-- Rank@1 < 85%
            |       -> B=4 ordering diagnostic
            |       -> structured ranking unless endpoints improve
            |
            +-- Rank@1 >= 85%
                    -> Smoke 2 horizon intervention
                    -> event-level horizon compression
                    -> B=2/4 executor-guided beam
```

Then:

```text
10k--20k pilot
    -> 256 smoke improves
    -> full 2,890 validation improves
    -> one full FlowER training epoch
    -> strict A7 headline evaluation on all 28,967 eligible test rows
    -> separate 28,971 full-endpoint coverage report, counting the four
       non-eligible upstream-corrupt rows as missing/failures
    -> only then port the identical protocol to mech-USPTO
```

Do not run FlowER and mech-USPTO rescue branches in parallel before the FlowER decision is resolved.

---

## 10. Full-run promotion gate

A new A7 condition is allowed to consume full training compute only after it satisfies all of:

1. gold-state legal coverage essentially complete;
2. gold-state event Rank@4 near or above 85%;
3. closed-loop validation endpoint accuracy clearly above the current A7 failure regime;
4. execution improvement accompanied by structural endpoint improvement;
5. no new use of gold precursor information at inference;
6. endpoint remains executor-owned;
7. no change to the full FlowER headline denominator.

For the first full run:

- run **one epoch**;
- evaluate the full 2,890 validation set at K=1 first;
- continue training only if validation is still improving;
- run K=10/full-test only after the K=1 validation gate passes.

The objective is not to finish every ablation. It is to obtain one reliable executable-transformation model that is competitive enough to support the paper, then freeze it and build the paper analyses around that checkpoint.

---

## 11. What we are explicitly not doing now

Until the above decision tree finishes, do not spend main-line compute on:

- another atom numbering/address system;
- marked reaction-center SMILES;
- another Python/JSON/native-tool harness rewrite;
- A8 value learning;
- MCTS;
- world-model training;
- multistep synthesis planning;
- new reaction representations unrelated to the local-policy/horizon diagnosis;
- three-epoch full-corpus runs without a passing smoke gate.

Those directions may be useful later, but they do not answer the current paper-critical question quickly enough.

---

## 12. Paper-preserving interpretation

The rescue does not change the central MechET contract.

The model still predicts an executable electron-flow transformation in closed loop:

\[
u_t\sim\pi_\theta(\cdot\mid X,S_t,h_t),
\qquad
S_{t+1}=\mathcal F(S_t,u_t),
\]

and the precursor remains only the executor-compiled result of the committed trajectory:

\[
Y=g_X(\tau).
\]

The proposed changes only make that policy learnable and reduce unnecessary decision depth:

- **event-level actions** reduce effective horizon;
- **structured ranking** improves local chemical decision quality;
- **small state beam** mitigates irreversible early commitment;
- **recovery continuation** is used only if the first three prove insufficient.

This keeps the paper centered on executable transformation space rather than turning the project into a generic agent-engineering study.
