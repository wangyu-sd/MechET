# Scientific thesis

> **Authority:** implementation-level terminology and claim boundaries. The
> ICLR experiment matrix in `wangyu-sd/MechET-paper` is authoritative when
> experiment names, priorities, or scope differ.
> **Scope:** mapped, closed-shell, two-electron polar organic chemistry  
> **Interpretation:** computational mechanism induction, not experimental mechanism determination

This document defines the runtime interpretation of MechET. Paper experiments
must follow [`PAPER_EXPERIMENT_PROTOCOL.md`](PAPER_EXPERIMENT_PROTOCOL.md),
which mirrors the paper repository at a frozen revision.

## Thesis

> **Retrosynthesis can be formulated as causal program induction over local electron-flow execution primitives.**

The central question is:

> Can mechanistic reasoning in retrosynthesis be made causal and compositional, rather than merely plausible in language?

A generated rationale is not necessarily the computation that produced an answer. Outcome-only and answer-bearing chain-of-thought systems may emit a correct precursor together with an incomplete, inconsistent, or causally unused mechanism. MechET addresses this reasoning–answer bypass by making the environment-owned trace the only admissible computational path to the endpoint.

## Computational contract

```text
atom-mapped product
  -> model-issued source-to-sink actions
  -> deterministic environment transitions
  -> committed move trace
  -> finish_trace
  -> replay declared moves
  -> deterministic MECH_PROOF v1 compilation
  -> proof execution
  -> full_precursor_state
  -> structural_precursor + auxiliary_fragments
```

The model does not independently submit a proof or precursor in the main method. It commits actions to the environment and terminates with `finish_trace`. The environment compiles the committed trace into the only admissible proof and derives the endpoint by execution.

### Required properties

| Property | Operational meaning | Failure condition |
|---|---|---|
| **Causality** | The endpoint is derived only from committed environment transitions | An independent answer channel receives endpoint credit |
| **Executability** | Every accepted transition and final proof are deterministically checked | A claimed state or proof cannot be replayed |
| **Compositionality** | Local source-to-sink primitives form complete mechanisms absent from training | Held-out examples require unseen primitives or are explained only by near-duplicate templates |
| **Evidence separation** | External evidence guides but never defines formal validity | Retrieval or a learned score overrides executor failure |

## Falsifiable hypotheses

### H1 — Causal faithfulness

**Claim.** The trace-owned program is a causal computational path to the precursor rather than a post-hoc explanation.

**Required evidence.**

- every credited trace prediction explicitly completes `finish_trace`;
- declared moves replay to recorded states;
- trace, move-sequence, compiled-proof, and endpoint artifacts agree;
- normal and intervention runs use the same model, adapter, revision, seed policy, candidate count, and generation budget;
- corrupting chemically relevant tool observations produces a measurable paired effect.

**Falsifier.** If removed, stale, or shuffled observations do not materially alter behavior under a valid matched intervention, the claim that the model uses tool-grounded reasoning is unsupported.

**Non-equivalence.** Trace/proof consistency by construction is necessary but insufficient. H1 additionally requires sensitivity to information transmitted through the environment.

### H2 — Compositional basis

**Claim.** Familiar local electron-flow execution primitives can be recombined into complete move compositions absent from training.

**Required evidence.**

- the primitive basis is `source_to_sink_execution_moves_v1`;
- each test primitive appears in training at a declared minimum frequency;
- complete train/test composition overlap is zero;
- the held-out set is non-empty and fixed before final model evaluation;
- scaffold, family, reaction-center, and near-duplicate overlap are reported separately;
- performance is stratified by composition frequency, step count, move count, topology, and structural novelty.

**Falsifier.** A split containing unseen test primitives does not test composition of known units. A split dominated by product or template near-duplicates cannot support a broad compositional-generalization claim.

### H3 — Separation of formal and empirical evidence (outside current ICLR)

**Status.** This is a separate future-study claim. Textbook/RAG experiments do
not contribute to the current ICLR R1--R7 evidence chain.

**Claim.** External mechanistic evidence improves program induction beyond trace ownership and additional context alone, while remaining subordinate to deterministic execution.

**Required evidence.**

- frozen trace-only, irrelevant-text, textbook, anchor, combined, and direct open-book conditions;
- the same bounded evidence content where a direct/trace comparison is claimed;
- inference-available retrieval queries in headline conditions;
- zero direct evidence reward;
- identical base-model revision and generation contract across compared prediction artifacts;
- evidence-content interventions such as passage shuffle, same-topic wrong passage, warning removal, and competitor removal.

**Falsifier.** A gain explained by context presence, label leakage, retrieval drift, missing predictions, or runtime mismatch does not support H3.

## Formal assumptions

The main method assumes:

1. atom-mapped molecular states with unique positive map identifiers;
2. closed-shell, two-electron source-to-sink actions supported by the executor;
3. deterministic transition verification and proof execution;
4. a declared endpoint decomposition into `full_precursor_state`, `structural_precursor`, and `auxiliary_fragments`;
5. a frozen reference universe for evaluation.

These assumptions define the computational scope. They are not claims about universal chemical coverage.

## Terminology

### Electron-flow execution primitive

A local executable source-to-sink action, for example:

```text
LP -> BOND
BOND -> ATOM
BOND -> BOND
```

Execution primitives define the model-facing action vocabulary, deterministic transitions, and H2 composition signatures.

### Mechanistic knowledge anchor

A provenance-aware structured record containing molecular-role patterns, candidate moves, preconditions, warnings, competitors, and references. Knowledge anchors support retrieval and analysis; their IDs do not define the H2 primitive basis.

### Trace-owned main method

A `TraceOwnedAgentEnv`-derived environment in which `finish_trace` is the only successful endpoint-producing terminal method and free-form proof submission is disabled.

### Legacy complete-proof baseline

A model that independently generates a complete `MECH_PROOF v1` program, or uses a loose tool trace followed by a submitted proof. It is a required baseline for measuring the reasoning–answer bypass, not an alternative description of the main method.

### Soft evidence

Natural-language retrieval, structured knowledge-anchor matches, and learned forward scores. Soft evidence may guide, rank, or calibrate an executable proposal, but cannot establish formal validity or experimental truth.

## Claim ladder

Claims must be earned in order:

| Level | Claim | Minimum evidence |
|---|---|---|
| **L0 — infrastructure** | The trace/proof/evaluation contracts execute as specified | CI and scripted replay |
| **L1 — learnability** | A model can learn valid tool interaction | Real tokenizer audit and small-set Tool-SFT overfit |
| **L2 — causal use** | Tool observations are causally used | H1 paired interventions |
| **L3 — compositional generalization** | Known primitives form unseen compositions | H2 frozen composition-OOD benchmark |
| **L4 — robust utility** | Gains persist under data scaling, OOD and cross-corpus transfer | Paper R5/R6 matched experiments |
| **L5 — downstream utility** | Verified programs improve search or planning | Future frozen candidate pools and matched planning budgets |

A downstream result cannot retroactively establish an earlier level.

## Permitted headline claims

A paper may claim only what frozen experiments demonstrate, including:

1. trace ownership reduces or removes the reasoning–endpoint bypass;
2. execution primitives support primitive-seen/composition-unseen generalization;
3. executable local operators improve data efficiency and robust transfer;
4. independent empirical evidence improves calibration only when separately tested;
5. verified route edges improve planning reliability only in a future matched study.

## Prohibited claims

The current software alone does not establish:

- a unique physical mechanism from product alone;
- low activation barriers, favorable kinetics, high yield, or laboratory success;
- universal condition compatibility;
- correctness outside the declared closed-shell two-electron scope;
- chemical truth from a citation, retrieval match, or learned score;
- reaction discovery without external validation.

## Core versus extensions

The scientific core is:

```text
causal trace -> executable program -> compositional test
```

Tool-SFT, retrieval, structured anchors, forward evidence, RL, hypothesis sets, and multistep planning are experiments or extensions used to test or apply the core. They must not be presented as parallel headline innovations.

## Documentation authority

When documents disagree, resolve them in this order:

1. the frozen `MechET-paper/EXPERIMENT_MATRIX.md` revision recorded in `PAPER_EXPERIMENT_PROTOCOL.md`;
2. `PAPER_EXPERIMENT_PROTOCOL.md` — local implementation mapping;
3. `SCIENTIFIC_THESIS.md` and `TRACE_FAITHFULNESS.md` — runtime meaning and contracts;
4. `EXECUTION_PLAN.md` — operational order, artifacts, and stopping gates.

Lower-authority documents must be updated rather than used to create a parallel source of truth.
