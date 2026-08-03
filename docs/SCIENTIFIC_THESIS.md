# MechET scientific thesis

This document is the single source of truth for the scientific question, main method, terminology, permitted claims and falsification criteria. Public documentation, experiment configs and paper text must remain consistent with it.

## Central scientific question

> Can mechanistic reasoning in retrosynthesis be made causal and compositional, rather than merely plausible in language?

A generated rationale is not necessarily the computation that produced an answer. Outcome-only and answer-bearing chain-of-thought models can emit a correct precursor alongside a mechanism, state sequence or explanation that is incomplete, inconsistent or causally unused.

MechET therefore asks whether the intermediate chemical representation can satisfy three properties:

1. **causality** — the intermediate actions are the sole computational source of the endpoint;
2. **executability** — every claimed state transition is checked by a deterministic chemical environment;
3. **compositionality** — local electron-flow execution primitives can be recombined into mechanism compositions not observed during training.

## Main hypothesis

Retrosynthesis can be formulated as causal program induction over local electron-flow execution primitives.

```text
atom-mapped product
  -> explicit source-to-sink electron-flow actions
  -> environment-owned molecular-state transitions
  -> committed trace
  -> deterministic trace-to-proof compilation
  -> executor-derived structural precursors
```

The model does not submit an independent proof or answer in the main method. It completes an environment-owned trace with `finish_trace`. The environment deterministically compiles that trace into the only admissible `MECH_PROOF v1` program and derives the precursor by execution.

## Three falsifiable hypotheses

### H1 — Causal faithfulness

The trace-owned program should eliminate the reasoning–answer bypass.

Predictions:

- trace, compiled proof and endpoint agree by construction;
- removing, shuffling or replacing tool observations should materially change model behavior;
- a model cannot retain endpoint credit after committing an incompatible trace;
- the legacy loose-trace/independent-proof path should show more reasoning–endpoint disagreement.

A negligible intervention effect would falsify the claim that the model uses the tool-grounded reasoning process.

### H2 — Compositional basis

Local electron-flow execution primitives should support unseen mechanism compositions.

Predictions:

- each held-out composition is made only from execution primitives represented in training;
- trace-owned proof models should degrade less than direct, free-form CoT and net-edit baselines as composition novelty increases;
- success and failure should be explainable by primitive coverage, proof length and topology rather than by reaction-name memorization alone.

No claim of compositional generalization is permitted if the held-out primitive vocabulary is not covered in training.

### H3 — Separation of formal and empirical evidence

Formal executability and empirical chemical support are different evidence layers.

- the deterministic executor answers whether the proposed electron-flow program is formally executable;
- textbook passages and structured mechanistic knowledge anchors provide external but non-authoritative evidence;
- an independent forward expert may provide learned process, target-recovery, competitor and uncertainty evidence;
- neither retrieval nor a learned score may override a formal execution failure.

A knowledge or forward-evidence claim requires matched controls and causal interventions. These layers are not part of the causal endpoint path.

## Terminology

### Electron-flow execution primitive

A local executable source-to-sink action such as:

```text
LP -> BOND
BOND -> ATOM
BOND -> BOND
```

Execution primitives define the action vocabulary, deterministic transitions and composition-OOD signatures.

### Mechanistic knowledge anchor

A provenance-aware structured record containing molecular-role patterns, candidate moves, preconditions, warnings, competitors and references. Knowledge anchors guide retrieval; they are not the primitive basis used to define causal execution or compositional splits.

### Trace-owned main method

`TraceOwnedAgentEnv` or a subclass in which `submit_proof` is disabled and `finish_trace` is the only successful terminal method.

### Legacy complete-proof baseline

A model that independently generates a complete `MECH_PROOF v1` program or uses a loose tool trace followed by a submitted proof. It is required as a baseline, not described as the main method.

### Soft evidence

Natural-language retrieval, structured knowledge-anchor matches and learned forward scores. Soft evidence may guide or rank an executable proposal but cannot establish formal validity or experimental truth.

## Permitted headline claims

A paper may claim only what the corresponding frozen experiments demonstrate:

1. trace-owned electron-flow programs remove or reduce the reasoning–endpoint bypass;
2. execution primitives support primitive-seen/composition-unseen generalization;
3. external mechanistic evidence improves program induction beyond extra context alone;
4. independent empirical evidence improves calibration or competitor ranking after formal execution;
5. verified route edges improve planning reliability under matched budgets.

## Prohibited claims

Current software alone does not establish:

- the unique physical mechanism from a product;
- low activation barriers, favorable kinetics, high yield or experimental success;
- universal condition compatibility;
- correctness outside the declared closed-shell two-electron scope;
- chemical truth from a citation, retrieval match or learned score;
- reaction discovery without external validation.

## Main method versus extensions

The scientific core is:

```text
causal trace -> executable program -> compositional test
```

Textbook evidence, structured knowledge anchors, forward falsification, RL, hypothesis sets and multistep planning are experiments or extensions used to test the core hypotheses. They must not be presented as parallel headline innovations.

## Documentation rule

When documents disagree, this file and `TRACE_FAITHFULNESS.md` define the main method. The authoritative execution order is `EXECUTION_PLAN.md`; the complete experiment definitions are in `PROOF_CENTRIC_EXPERIMENT_PLAN.md`.
