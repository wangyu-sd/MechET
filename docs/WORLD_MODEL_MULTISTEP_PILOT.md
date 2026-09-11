# MechET world-model / multistep pilot

This pilot keeps the current trained MechET policy, deterministic executor, proof compiler, and one-step benchmark unchanged. It adds a narrow evaluation layer for two questions:

1. Can the frozen MechET policy be reused as an **experience generator** for action-conditioned lookahead?
2. Can the same frozen one-step policy be sampled recursively to form **multistep retrosynthetic trees** without retraining?

## Design boundary

MechET already owns an exact formal transition operator for supported chemistry:

```text
S_t --electron-flow action--> executor --> S_{t+1}
```

The pilot therefore does **not** learn `S_{t+1}` from text. A future learned world component should estimate downstream value or dead-end risk, while the existing executor remains the transition authority.

We use the terms:

- **actor**: the frozen trained MechET policy proposing executable inverse electron-flow programs;
- **transition model**: the existing deterministic executor/compiler;
- **world value**: a future learned or empirical estimate of downstream success from a state/action;
- **macro transition**: one completed executable one-step retrosynthetic prediction, producing one or more precursor molecules;
- **multistep rollout**: recursive application of macro transitions to unresolved precursor molecules.

## Pilot A — frozen-policy experience generation

For each target state, sample `K` independent MechET candidates with the existing inference runner. Retain every candidate, including invalid and unfinished trajectories. For every completed candidate, record:

```json
{
  "target_smiles": "...",
  "candidate_index": 0,
  "trace": ["..."],
  "formal_execute": true,
  "structural_precursor": "...",
  "assistant_nll": 12.34,
  "tool_calls": 7,
  "failed_steps": 0
}
```

This produces an offline experience table from the already-trained policy. A later value model can derive labels such as terminal exact match, forward round-trip consistency, dead-end status, recovery after failed actions, and remaining depth.

No new scientific claim is attached to this artifact by itself.

## Pilot B — recursive multistep sampling

A one-step MechET prediction is promoted to a macro transition:

```text
molecule M
  -> sampled executable electron-flow program tau
  -> precursor set {P_1, ..., P_n}
```

A multistep state is a set of unresolved molecules. At each search expansion:

1. choose one unresolved molecule;
2. invoke the frozen MechET checkpoint to sample up to `K` one-step candidates;
3. discard candidates that fail the frozen trace/execution contract;
4. split the resulting structural precursor into fragments;
5. mark purchasable/terminal fragments as solved when a stock oracle is configured;
6. insert unresolved fragments back into the search state;
7. continue until solved, maximum depth, or expansion budget.

The first implementation should use **best-first or beam search**, not MCTS. A simple candidate score is sufficient for the pilot:

```text
score = - mean_assistant_nll
        - depth_penalty * route_depth
        - failure_penalty * failed_steps
```

If a frozen forward round-trip score is available, it may be added as a separately named term. Gold precursor information must never enter search-time scoring.

## Required outputs

Each route artifact should preserve the full provenance of every edge:

```json
{
  "target": "...",
  "solved": false,
  "route_depth": 3,
  "nodes": ["..."],
  "edges": [
    {
      "product": "...",
      "precursors": ["..."],
      "program": ["..."],
      "formal_execute": true,
      "candidate_score": -1.23
    }
  ]
}
```

The route-level endpoint must remain a composition of executor-derived one-step endpoints. Do not introduce an independent route answer channel.

## Fast evaluation

The pilot is diagnostic. Start with a small fixed set of literature molecules or known multistep routes whose intermediates are absent from the training set when possible.

Report:

- route solved rate under a fixed expansion budget;
- percentage of route edges with formal execution;
- search expansions / wall time;
- route depth;
- recovery of known literature disconnections as a diagnostic only;
- branching factor and duplicate-state rate;
- cumulative one-step confidence / NLL.

Do not claim experimental feasibility from formal execution.

## Follow-up world-value experiment

If recursive sampling produces enough valid alternative branches, build an offline dataset of `(S_t, a_t, return)` tuples from the same rollouts. Train a small value model `Q_phi(S_t, a_t)` to predict a gold-independent downstream score. Compare:

```text
frozen MechET sampling
vs. frozen MechET + one-step value reranking
vs. frozen MechET + depth-d lookahead
```

The executor remains the exact transition authority in all three conditions. This isolates the value of anticipatory planning from the already-established executable representation.

## Scope

This pilot is intentionally outside the current headline one-step ICLR matrix until the frozen A7 one-step result is complete. It should not block the current one-step paper run.
