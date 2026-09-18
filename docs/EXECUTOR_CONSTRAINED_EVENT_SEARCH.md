# Executor-constrained event search for inverse electron flow

## Status

This document defines the next **small-scale, no-retraining** MechET experiment after PR #54.

It does not introduce a new molecular representation, a second learned verifier, a forward critic, a value head, or a new reaction-template ontology. It also does not authorize RLVR yet.

The immediate question is narrower:

> Given the current inverse electron-flow policy and deterministic executor, can search recover better trajectories by expanding one elementary event at a time, rejecting impossible branches immediately, and reallocating generation budget to surviving states?

The method tested here is **executor-constrained event search**.

---

## 1. Motivation from existing results

Current MechET evidence supports three facts that jointly motivate search.

1. **Sampling breadth helps.** Historical and current K-sample experiments show a substantial gap between single-sample and multi-sample execution/endpoint coverage, so useful alternatives exist in the policy tail.
2. **Many failures are detectable early.** Prior diagnostics contain large numbers of localization, state-assertion, invalid-SMILES, executor, and cycle failures that should not consume the rest of a long rollout budget.
3. **Formal execution is not enough.** Some trajectories remain formally executable while reaching wrong endpoints, so executor success must be a hard feasibility filter rather than the main ranking score.

Therefore the search design is deliberately asymmetric:

```text
Qwen proposes chemistry
    -> executor removes impossible branches
    -> duplicate/cyclic states are removed
    -> Qwen likelihood ranks the remaining branches
```

The executor does **not** award a branch for merely being executable.

---

## 2. Search unit

The search unit is one complete elementary electron-flow event, not one token and not a complete trajectory.

For a partial trajectory node

\[
N_t=(S_t,\tau_t,\ell_t),
\]

where `S_t` is the authoritative executor state, `tau_t` the committed event history, and `ell_t` the accumulated model log-probability, the policy proposes complete next events

\[
e_{t,1},\ldots,e_{t,M}.
\]

Each event is executed immediately:

\[
S_{t+1,j}=\mathcal F(S_t,e_{t,j}).
\]

A child exists only if the executor accepts the event and all frozen hard-search constraints pass.

This preserves the existing causal requirement that every precursor state is produced by committed electron-flow execution.

---

## 3. Hard pruning contract

The following conditions remove a branch **before likelihood ranking**:

1. malformed or empty event;
2. executor rejection;
3. invalid atom/site reference detected by the active interface;
4. electron/bond/charge/valence failure already enforced by `verify_electron_step`;
5. invalid sanitized successor;
6. no-op successor;
7. return to an ancestor molecular state;
8. failure of an explicitly frozen chemistry-support validator, if enabled;
9. exact duplicate authoritative successor reached with the same ancestor-state
   set after another higher-likelihood branch has already reached it.

The current small-case implementation includes executor rejection, no-op rejection, ancestor-state cycle detection, optional support validation, and exact canonical successor deduplication.

### What is deliberately not a hard filter yet

Do not reject a branch solely because it is uncommon, absent from one training split, or does not match the reference mechanism. Such rules risk converting the search into a handcrafted reaction-template system.

Do not use the gold precursor, gold reaction center, gold next event, or gold remaining-event count during inference.

---

## 4. Ranking surviving states

Among branches that pass hard constraints, rank only by the policy's own generated-event likelihood.

For a partial trajectory with generated-event token log-probability sum `L` and generated-event token count `n`, the first implementation uses

\[
\mathrm{score}(N)=L/n.
\]

This matches the repository's existing preference for length-normalized sequence log-probabilities and prevents executor heuristics from silently becoming a chemistry value function.

The executor answers only:

> Is this transition admissible under the frozen formal contract?

The policy likelihood answers:

> Among admissible continuations, which branch does the model consider more likely?

No `formal_success` bonus, successful-step bonus, tool-use bonus, or reference-derived score participates in inference ranking.

---

## 5. Successor deduplication

Search budget must be allocated to distinct chemical states rather than distinct strings.

The first implementation deduplicates using deterministic canonical mapped SMILES. This safely removes differences such as component or traversal order while preserving atom identity, charge, stereochemistry and bond structure.

If two proposed events produce the same canonical authoritative successor **and
the same ancestor-state set**, retain only the branch with the higher cumulative
normalized model log-probability. Paths that converge on the same state through
different ancestor sets are not duplicates: a later transition can be a cycle
for one history but valid for the other. Keeping those histories distinct makes
cycle pruning correct; ordinary beam ranking may still remove one later as an
explicit capacity trade-off.

A later full search implementation may add a separately audited graph/symmetry equivalence key. That extension must not be mixed into this initial case test.

---

## 6. Cycle detection

Every search node retains canonical keys for all ancestor executor states.

If a new event returns to any ancestor state,

\[
K(S_{t+1})\in\{K(S_0),\ldots,K(S_t)\},
\]

that branch is immediately pruned.

The small-case test explicitly checks a valid forward event followed by a valid inverse event that returns to the original state. The second event is formally executable but must still be removed as a search cycle.

Reactive-core-only cycle detection, which ignores newly appended spectators, remains an important follow-up from PR #54 but is not silently added to the first smoke.

---

## 7. Terminal handling

The production integration must retain the existing `finish_trace` contract.

`finish_trace` is a real model action. A terminal result may enter the terminal pool only when the model explicitly calls it and the environment successfully compiles/replays the committed trace.

The search controller must never choose an arbitrary intermediate state and reinterpret it as a completed precursor.

This initial primitive smoke does not reimplement `finish_trace`; it tests nonterminal event pruning only. Production integration must call the existing trace-owned environment terminal operation.

---

## 8. Integration into current inference code

The repository already has the main pieces needed for full search:

- `_TraceRollout` stores a live environment and conversation;
- `_run_trace_candidates_batched` advances multiple active trajectories one event at a time;
- vLLM and Transformers backends can batch active requests;
- every candidate already owns an independent RNG stream;
- the trace-owned environment executes events and records failures.

The full implementation should modify this architecture rather than create a second inference stack.

### Required additions

#### 8.1 Search-node fork/restore

Add an explicit environment-supported fork mechanism for search states. Do not rely on unsafe generic `deepcopy` of model/cache objects.

A fork needs only the deterministic episode state:

```text
current molecular state
committed trace
visited-state keys
successful/failed-step counters
remaining tool budget
fragment/import ledger required by the active environment
```

Frozen models and shared resources remain process-level objects.

#### 8.2 Per-event model likelihood

Generation backends must return the generated event token log-probability sum and token count in addition to decoded text.

This value must be computed from the actual sampled tokens before tool parsing. Search ranking must never use environment reward as a proxy for policy likelihood.

#### 8.3 Layer scheduler

At every depth:

1. batch all active nodes;
2. sample `M_t` complete next actions per node;
3. parse and execute each action immediately;
4. hard-prune failures;
5. move explicit `finish_trace` successes to a terminal pool;
6. deduplicate surviving successor states;
7. rank survivors by normalized model likelihood;
8. retain at most beam width `B`;
9. stop when there are no active states, the event-depth budget is reached, or a separately frozen terminal policy is satisfied.

---

## 9. Do not guess beam width

The production beam width and branching factor must be selected from data, not chosen by intuition.

Before a full search run, use existing multi-sample rollout artifacts to perform a **prefix-retention audit**:

1. identify trajectories that eventually reach the strict structural endpoint;
2. replay them event by event;
3. at each depth collect all executor-valid sibling proposals available under the matched sampling budget;
4. rank valid prefixes by normalized model likelihood;
5. measure whether the successful prefix would survive `B = 1, 2, 4, 8`;
6. select the smallest `B` retaining at least 90% of already-observed successful prefixes, subject to cost constraints.

If successful prefixes routinely rank outside `B=8`, search is not the correct primary fix; endpoint-oriented RLVR should be used to move probability mass before increasing search width.

---

## 10. Adaptive branching factor

Once real per-depth valid-child rates are measured, choose branching factor from the probability of obtaining at least one valid child.

If `p_t` is the empirical probability that one sampled next event survives hard constraints at depth `t`, choose the smallest

\[
M_t=\left\lceil\frac{\log(1-q)}{\log(1-p_t)}\right\rceil
\]

for target survival probability `q`, initially `q=0.95`, with a frozen maximum such as 10.

This naturally gives more exploration where the policy is fragile and less where almost every event already survives.

The formula is a scheduling rule, not a scientific claim; it must be instantiated only after the valid-child audit.

---

## 11. Small deterministic case implemented in this PR

The initial mapped state is

```text
[O-:1].[CH3:2][Br:3]
```

The smoke presents four first-layer candidates:

1. an intentionally invalid highest-likelihood bond-cleavage event involving a nonexistent bond;
2. a coupled substitution-like event;
3. the same coupled event with reversed move serialization, producing the same successor;
4. a formally executable cleavage-only branch with lower policy likelihood.

Expected behavior:

- the invalid highest-scoring candidate is rejected **before ranking**;
- the two equivalent coupled-event serializations consume one successor slot;
- beam width 1 retains the higher-likelihood unique valid successor;
- the lower-likelihood but executable cleavage-only branch is recorded as beam-pruned, not chemically declared impossible.

A second layer applies the exact inverse coupled event to the surviving state. The executor accepts the chemistry, but the search layer recognizes that it returns to the root molecular state and prunes it as `STATE_CYCLE`.

This smoke therefore tests the exact separation required by the full method:

```text
formal validity != search value
```

and

```text
executor hard gate -> state dedup/cycle gate -> model likelihood ranking
```

Files:

- `src/mechet/constrained_event_search.py`
- `scripts/smoke_constrained_event_search.py`
- `tests/test_constrained_event_search.py`

---

## 12. Promotion gate after the deterministic smoke

Do not start RLVR from this PR.

After CI passes, the next no-training test should use one existing current checkpoint on a frozen 32--64 validation-case subset and compare, under a matched generation-token budget:

1. independent stochastic rollouts;
2. executor-constrained event search.

Required measurements:

- strict structural EndpointPass@1 and oracle EndpointPass@budget;
- explicit-finish rate;
- executor-invalid proposal rate by depth;
- fraction of proposals pruned before another model call;
- duplicate-successor rate;
- cycle rate;
- surviving-state diversity by depth;
- successful-prefix rank by depth;
- model tokens generated per target;
- wall-clock latency per target.

### Search promotion criterion

Promote constrained search to the main inference path only if, under matched or lower generated-token budget, it improves endpoint recovery or preserves endpoint recovery while materially reducing wasted rollout tokens.

If it improves ExecutePass but not endpoint recovery, do not promote it as a solution to the scientific problem.

### Implemented frozen-checkpoint integration

The integration uses the completed one-epoch in-place-grounded Qwen3-8B
adapter at checkpoint 8,037. This is an existing frozen checkpoint; the search
experiment performs no optimization. Its action surface packages optional
first-use imports and one coupled FLOW program into a single transaction, so
the operational branch is the elementary event defined in Section 2.

The runtime includes:

- an explicit JSON-serializable episode fork containing private executor state,
  committed transitions, transcript, import map counter, and ancestor states;
- actual generated-token log-probability extraction from sampled Qwen tokens;
- transactional rollback for rejected imports or FLOW programs;
- an explicit `finish_trace` terminal pool backed by full trace replay and
  proof compilation;
- deterministic 64-case validation selection and eight inference shards;
- an independent-rollout comparator with executor-visible repair feedback;
- a shared ceiling of 48 responses and 384 tokens per response per target for
  both methods.

The model prompt contains only the unmapped product and prior public executor
observations. Mapped target states, reference endpoints, and reference prefix
states are joined by stable ID only inside the executor/evaluator; they are not
passed to generation, pruning, or likelihood ranking.

A one-reaction, two-response local GPU integration smoke exercised both live
paths and produced finite token likelihoods. It is an interface check, not an
endpoint result. The 64-case run is the first decision-bearing gate.

### Completed 64-case gate

The gate completed successfully on 2026-09-11. It used Taiji instance
`8b1d818da08afca801a08fc1f79b0984` on one Qingyuan host with eight A100 GPUs.
The task ran for 3,626 seconds and produced all 64 expected records.

| Method | EndpointPass@1 | Oracle EndpointPass | Explicit finish | Mean responses | Mean tokens | Mean latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Four independent interactive rollouts | 0/64 | 0/64 | 52/64 (81.25%) | 42.41 | 4,625.67 | 266.30 s |
| Executor-constrained beam search | 0/64 | 0/64 | 2/64 (3.125%) | 18.38 | 2,236.58 | 110.68 s |

The independent condition produced 91 formally executable terminal traces, but
none reached the reference endpoint. Search produced three terminal traces on
two targets; 62 of 64 frontiers became empty before an explicit terminal. The
lower search cost is therefore early branch death rather than more efficient
endpoint recovery.

Three concrete cases make the failure mode explicit:

- `flower_mech_proof_val_1187`: independent rollouts reached two formally
  executable but wrong endpoints after 43 responses. Search retained one state
  at each of its first two depths, then lost the frontier after two duplicate
  successors and two rejected events.
- `flower_mech_proof_val_2009`: both first-layer search proposals failed, one
  because the imported SMILES was invalid and one because the event was
  rejected. Search stopped after two responses. Independent repair-and-retry
  continued to two formally executable terminals, although both endpoints were
  wrong.
- `flower_mech_proof_val_972`: search preserved two states for four depths and
  reached two formally executable terminals, but both copies represented the
  same wrong endpoint. This is direct evidence that formal execution alone is
  not an endpoint score.

The versioned metrics, Taiji identifiers, hashes, aggregate failure counts, and
readable product/reference/prediction SMILES for these cases are stored in
`docs/results/grounded_constrained_search_valid64_20260911.json`. Full
per-candidate records remain in the Ceph output directory recorded there.

This gate does **not** promote the current search. Hard rejection removes a
branch without giving the policy a chance to repair it, whereas the independent
interactive comparator can consume executor feedback and retry from the same
state. The next no-training test should add same-state repair/reproposal (or
executor-supplied admissible alternatives) under the same response/token
ceiling before any endpoint-oriented RLVR run.

---

## 13. Relationship to PR #54 / future RLVR

PR #54 established that formal validity and endpoint quality must remain separate.

This search pilot operationalizes that principle at inference time:

- executor validity is a hard gate;
- policy likelihood ranks feasible branches;
- endpoint correctness remains an external evaluation metric.

If the search audit shows that correct prefixes exist but are systematically low-ranked, the next training step is endpoint-oriented RLVR under the **same hard-gating environment**. Early-invalid trajectories can terminate immediately during RL rollout, while endpoint reward trains the policy to move probability mass toward branches that search currently has to rescue.

The intended progression is therefore:

```text
small deterministic search smoke
    -> frozen-checkpoint 32--64 case search audit
    -> freeze search contract
    -> endpoint-oriented RLVR if prefix ranks remain poor
    -> combine improved policy with the same constrained search
```

No further representation redesign is required for this experiment.
