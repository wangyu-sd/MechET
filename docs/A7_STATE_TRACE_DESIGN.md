# A7 state-visible trajectory redesign

> **Status: design proposal.** This document records the current A7 failure
> analysis and a bounded replacement contract, `state_trace_v1`. It does not
> change the frozen dataset universe, paper claims, or active experiment by
> itself. Adoption requires the equivalence and validation gates below and an
> explicit update to the paper protocol and `ACTIVE_ICLR_STATUS.md`.

## Decision to make

A7 should remain product-only executable retrosynthesis, but the model should
not have to reconstruct the chemical state from a growing, noisy chat log.
Each decision should expose the current executor state and the complete
committed trajectory in a canonical representation. Earlier full states remain
available for exact inspection and audit without being copied into every
prompt.

This preserves the scientific question: can a model synthesize an inverse
electron-flow program whose accepted steps are executed by the environment and
whose endpoint is a precursor prediction? It changes only how the executor's
history is presented to the policy.

## Evidence motivating the redesign

The current independent A7 checkpoint was trained on the strict executable
FlowER universe (257,167 train / 2,890 valid / 28,967 test) with the
`compact_full_state_v1` observation format. That format already returns the
complete mapped `current_state_smiles` after every accepted nonterminal action.
However, inference still accumulates the raw assistant/tool transcript, so old
actions, errors, and serialized states repeatedly consume context.

The following values are an operational snapshot from 2026-09-02, not frozen
paper results:

| Artifact | Coverage at snapshot | Diagnostic observation |
|---|---:|---:|
| independent A7, K=10, vLLM/A100 | 15,815 / 28,967 generated | 7,692 rows had at least one formally executable candidate (48.64% conditional on generation; Execute-Pass@10 = 26.56% over all 28,967 IDs, with missing rows counted as failures) |
| independent A7, K=1, V100 | 19,620 / 28,967 generated | 1,616 formally executable rows (8.24% conditional on generation; Execute-Pass@1 = 5.58% over all 28,967 IDs, with missing rows counted as failures) |
| A0 to A7 warm-start, K=1 | 28,967 / 28,967 | 247 formally executable rows (0.85%); 4 structural exact rows (0.014%) |
| completed A0 direct baseline, K=10 | 28,971 / 28,971 | generation Pass@1 52.75%, Pass@10 74.17%, NLL-ranked structural Top-1 61.27%, neutralized Top-1 61.99% |

Partial Execute-Pass is not retrosynthetic accuracy: an executable program can
still produce the wrong precursors. The independent A7 result must not be
reported as final until structural evaluation completes on the frozen
denominator. A 17-example structural spot audit was non-random and too small to
estimate accuracy.

The completed warm-start artifact exposes the dominant control failures:

| Terminal reason | Count | Share of 28,967 |
|---|---:|---:|
| maximum iterations | 21,118 | 72.9% |
| generation or JSON error | 3,679 | 12.7% |
| no valid tool call | 3,666 | 12.7% |
| abstain | 246 | 0.8% |
| terminal tool call | 258 | 0.9% |

Its traces repeatedly contain `CHEMICAL_STATE_INVALID`, `STATE_CYCLE`, stale or
uncommitted fragment operations, and tool JSON truncated by the 512-token
generation limit. The warm-start result is auxiliary evidence, not the main A7
condition: it continued the same rank-16 A0 LoRA and therefore also measures
negative transfer from an endpoint-output policy to a tool policy.

## Why the current interface is hard to learn

The product-only policy is currently asked to solve several coupled problems in
one autoregressive stream:

1. infer missing precursor fragments and their exact atom mapping;
2. choose valid electron-flow actions against the current graph;
3. recover after a rejected action without entering a state cycle;
4. decide when the trace is complete; and
5. serialize long, exact tool arguments within a fixed generation budget.

Teacher-forced SFT shows only expert histories, while rollout conditions the
model on its own earlier calls and executor errors. Repeating complete mapped
states in an ever-growing transcript increases attention cost and makes the
closed-loop distribution shift worse. Token loss, JSON validity, and formal
execution alone do not guarantee the correct chemical endpoint.

## Existing and proposed contracts

| Contract | State seen by the model | History seen by the model | Role |
|---|---|---|---|
| `action_delta_v1` | no full intermediate state | accepted action result and legal-action feedback | original paper A7; fallback/ablation under current status |
| `compact_full_state_v1` | current full mapped state after accepted actions | accumulated raw assistant/tool transcript | current A7 candidate |
| `state_trace_v1` | current full mapped state | canonical committed ledger plus on-demand access to any earlier full state | proposed replacement |

### Relation to experiments already run

`state_trace_v1` is not a wholly new scientific condition. Its central premise,
conditioning the next electron-flow decision on an executor-produced full
state, has already appeared in two A7 generations:

- the legacy full-state Tool-SFT exposed complete intermediate states in the
  growing tool transcript;
- `compact_full_state_v1` retained one authoritative current-state SMILES while
  removing duplicated proof and audit fields; and
- both existing formats trained the same expert next-action targets at every
  assistant turn.

Consequently, materializing one row per decision in Stage 1 does not by itself
create a new supervision signal. The new variable is the policy context:
bounded canonical re-rendering instead of an append-only raw transcript.

### Historical small-subset result and the long-horizon hypothesis

The earlier full-state checkpoint is an important positive control. It used the
same product-only, executor-mediated next-action formulation and exposed the
complete current state through tool results, but it was trained and evaluated
on the legacy MECH_PROOF-compilable subset rather than the frozen full FlowER
universe.

| Condition | Train / test reactions | Training | Mean mechanism steps | Mean electron moves | Mean fragment imports | Structural Pass@1 / Pass@10 | Execute-Pass@1 / Pass@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| legacy full-state Tool-SFT | 27,640 / 3,080 | 3 epochs | 1.93 | 3.86 | 4.18 | 75.00% / 92.11% | 94.03% / 99.74% |
| current `compact_full_state_v1` | 257,167 / 28,967 | 1 epoch | 4.05 | 8.16 | 5.22 | pending complete structural evaluation | partial operational snapshot above |

The legacy test contains only 10.63% as many targets as the strict current
test and is an incomplete, compiler-selected trace view. Its accuracy is
therefore evidence that the formulation can work on shorter executable
trajectories, not a headline estimate for full FlowER.

The central diagnosis is **closed-loop long-horizon error accumulation**. If a
trace of length $T$ requires actions $a_1,\ldots,a_T$, then its success
probability can be decomposed as

\[
P(\mathrm{success}\mid T)
= \prod_{t=1}^{T} P(a_t\ \mathrm{correct}\mid x,s_t,h_t).
\]

An accepted wrong action also changes the next executor state, so later
decisions are made off the expert trajectory. The current full test has 2.10
times as many mechanism steps and 2.11 times as many electron moves per target
as the legacy subset. That doubling gives the accumulation hypothesis direct
empirical support and is consistent with the large gap between teacher-forced
loss and rollout success.

This comparison does not identify trajectory length as the sole cause. The
legacy subset also differs in compiler selection, per-reaction repetition
(three epochs versus one), observation serialization, and reaction scope. In
aggregate token presentations, however, the current one-epoch run processed
3.23 times as many input tokens as all three legacy epochs, so the epoch count
alone is not a sufficient explanation.

The decisive no-retraining audit is to (1) evaluate the current checkpoint on
the exact 3,080 legacy IDs and (2) report full-test success stratified by expert
step and move count. Recovery on the legacy IDs together with monotonically
falling success as trajectory length grows would strongly support the
long-horizon diagnosis. The machine-readable measurements and claim limits are
frozen in
[`results/a7_historical_subset_long_horizon_audit_20260902.json`](results/a7_historical_subset_long_horizon_audit_20260902.json).

The parts not previously trained as one matched condition are stable
state-addressed history, stale-state rejection, repeated-failure blocking, the
fragment-ID interface, and recovery supervision from executor failures and
accepted off-expert states. The historical full-state and
`compact_full_state_v1` checkpoints therefore remain mandatory baselines for
any `state_trace_v1` claim. This redesign must be presented as a state/history
protocol repair unless those new components produce a separately measured
closed-loop effect.

`state_trace_v1` does not remove history. It separates three representations:

- **authoritative state history:** the environment and prediction artifact keep
  every full state `s0 ... st`;
- **default policy observation:** the current full state plus a compact,
  cumulative ledger of every accepted action and state delta;
- **on-demand history:** the policy may request any previous authoritative full
  state by stable state ID.

Each decision is rendered from environment state rather than formed by
appending an unbounded raw transcript. The target product appears once in the
canonical observation. No compiled proof, expected precursor, endpoint digest,
or reference answer is model-visible.

## Proposed observation schema

The exact JSON schema should be versioned, but the semantic minimum is:

```json
{
  "observation_version": "state_trace_v1",
  "target_product": "<mapped product SMILES>",
  "current_state": {
    "state_id": "s3",
    "state_hash": "<sha256>",
    "mapped_smiles": "<authoritative executor state>"
  },
  "committed_trace": [
    {
      "step": 1,
      "from_state": "s0",
      "to_state": "s1",
      "action": "import_fragment",
      "arguments": {"fragment_id": "F1"},
      "affected_atom_maps": [7, 12],
      "bond_charge_delta": ["<canonical delta>"],
      "local_before": "<reaction-centre view>",
      "local_after": "<reaction-centre view>"
    }
  ],
  "last_feedback": {"status": "accepted", "code": "OK"},
  "blocked_action_signatures": ["<canonical failed-call hash>"],
  "pending_fragments": ["F2"],
  "legal_action_summary": ["apply_electron_moves", "finish_trace"]
}
```

All mutating calls bind to `state_id`; a call against an older state is rejected
as stale. Failed calls record a canonical signature so identical retries can be
blocked without changing the authoritative state.

The history interface is explicit:

```json
inspect_state({"state_id": "s1", "view": "full"})
```

The environment returns the exact stored state and its hash. The same artifact
must also retain all full historical states for deterministic replay,
visualization, and post-hoc analysis even when the model never calls
`inspect_state`.

## Fragment proposal and execution

Long open-vocabulary mapped fragment strings are a major serialization failure
surface. A compatible two-phase interface is proposed:

```text
propose_fragments(state_id="s1", unmapped or partially specified candidates)
  -> environment canonicalizes, validates, maps, and returns F1 ... Fn
import_fragment(state_id="s1", fragment_id="F2")
  -> executor commits the chosen canonical fragment; failures echo the bound
     state ID in canonical feedback
```

The environment must not search the reference answer when canonicalizing or
mapping a proposed fragment. Fragment IDs are scoped to one rollout and their
resolved structures are retained in the artifact. Integration of A0 proposals
is optional and must be evaluated as a named condition rather than silently
changing A7.

## Training plan

### Stage 1: next-action Tool-SFT

Convert each replay-verified expert trajectory into decision examples:

```text
(target product, current state, committed ledger, feedback) -> next tool call
```

This creates several decision windows per reaction but no new reactions. Stable
reaction IDs, split membership, authoritative states, actions, and endpoints
must remain unchanged. Comparisons should match optimizer updates or supervised
action tokens, not epochs alone. The bounded windows are also expected to lower
attention cost relative to repeatedly encoding the full transcript.

### Stage 2: failure-recovery SFT

Construct controlled rejected-action observations, including invalid atom
references, repeated/cyclic calls, malformed arguments, and pending fragment
errors. When the executor state is unchanged, supervise the expert next action
from the error observation. When a synthetic deviation changes state, supervise
an explicit inspect/replan/abstain policy unless a replay-safe recovery operator
has first been approved. Report synthetic recovery examples separately from the
reaction count.

### Stage 3: optional RLVR

Only after the SFT validation gate passes, optimize a capped reward over JSON
validity, legal committed non-cycle transitions, successful `finish_trace`,
trace/proof replay, and structural endpoint correctness. Invalid, repeated, and
cyclic calls receive penalties. Endpoint correctness must dominate the capped
process reward so a long executable but chemically wrong trace cannot win by
reward accumulation. Gold endpoints may supervise training rewards but never
test-time candidate selection.

## Required gates before a full run

### Representation-equivalence gates

Build a frozen 2,048-row slice. Keep two gates separate so the proposed
fragment interface is not rejected merely because its syntax is intentionally
different.

For the representation-only `state_trace_v1` condition, require:

- identical stable IDs and zero dropped or quarantined rows;
- byte-identical expert assistant actions;
- identical authoritative state sequence and executed endpoint;
- no model-visible answer, expected precursor, proof, or digest leakage;
- deterministic rendering and state hashes; and
- token-length and estimated attention-cost comparison against
  `compact_full_state_v1`.

For the separately named fragment-interface condition, replace byte identity
with canonical semantic-action equivalence. Require identical proposed
fragment graphs, imported mapped structures, authoritative states, executed
endpoints and leakage audit, while recording the deliberate syntactic change
from free-form `import_fragment` arguments to state-bound fragment IDs.

### Closed-loop pilot gate

Run a real 5,000--10,000-reaction training pilot followed by frozen validation.
Report JSON-valid calls, committed-action rate, invalid-action rate,
post-failure recovery, cycle rate, `finish_trace` rate, Execute@1/@K,
TraceBound, structural Pass@1/@K, and structural accuracy conditional on formal
execution. Do not launch the full train/test run unless validation improves over
the independent compact-full-state A7 and the structural endpoint metric is
nontrivial under a frozen selector.

The final test denominator remains 28,967 for strict executable A7. Missing,
failed, and OOM predictions stay in the denominator. Candidate ranking must be
gold-independent and recorded in prediction metadata.

## Faithfulness and hallucination reporting

A **mechanistic hallucination** is a claimed electron-flow transition,
intermediate, or proof step that cannot be deterministically replayed from the
previous authoritative state. The executor makes unsupported transitions
rejectable and accepted transitions auditable; it does not guarantee that a
formally executable precursor is the chemically correct reference.

Report these cases separately:

- rejected invalid proposal: invalid-action rate;
- accepted executable trace with wrong precursor: chemical prediction error;
- alternative executable route not represented by a single ground truth:
  unresolved alternative, not automatically a hallucination; and
- abstention: coverage failure.

Faithfulness must always be paired with coverage and endpoint accuracy so that
abstaining on every example cannot appear to eliminate hallucination.

## Scope and non-goals

This proposal:

- does not change FlowER train/valid/test membership or denominators;
- does not make A0-to-A7 warm-start the main method;
- does not add textbook retrieval, an independent proof channel, or a direct
  answer bypass;
- does not authorize a new full training job before the gates pass; and
- does not establish a new headline result or scientific claim.

## Implementation checklist

- [ ] Freeze the `state_trace_v1` observation and prediction-artifact schemas.
- [ ] Add state IDs/hashes, canonical ledgers, stale-state rejection, and
      repeated-failure blocking to the runtime.
- [ ] Extend the existing current-state `inspect_state` tool to accept historical
      state IDs and retain all authoritative states in artifacts.
- [ ] Prototype the fragment proposal/ID interface without reference lookup.
- [ ] Build and audit the 2,048-row equivalence slice.
- [ ] Measure real tokenizer lengths and attention cost.
- [ ] Train the closed-loop pilot and evaluate frozen validation.
- [ ] Approve or reject adoption; only then update the active status and paper
      protocol and schedule a full run.
