# Executor-constrained graph electron policy

Status: experimental architecture/pilot.  This does not replace the current
LLM main condition until a frozen validation comparison succeeds.

## Objective

Represent inverse mechanism inference as an executor-defined MDP rather than a
text-generation problem.  At decision time the observation is the authoritative
current molecular graph `G_t` and the fixed product graph `G_0`.  The private
atom maps only join policy choices back to the executor; map integers are never
model features.

The transition is deterministic:

`G_(t+1) = Executor(G_t, a_t)`.

The terminal output is the complete precursor mixture returned by the existing
strict endpoint evaluator.  Formal execution is a feasibility constraint, not
evidence that the chemistry is correct.

## Factorized action space

The policy has five action families.

1. `FLOW`: select a state-derived electron source, then a source-conditioned
   sink, repeat for the arrows in one coupled event, and atomically `COMMIT`.
   `MoveInventory` provides the reference-independent legal mask.
2. `BE_DELTA`: select a sparse unordered atom pair and `-1/+1` bond-electron
   change, or an atom and `-1/+1` charge change, repeat, then atomically commit.
   This is `O(n^2)` per sparse edit instead of enumerating complete edit sets.
3. `IMPORT_ENV`: retrieve a canonical unmapped solvent, salt, catalyst or
   other endpoint-context molecule from a graph-encoded catalog.  This is a
   closed action because these molecules do not participate in the reference
   electron moves.
4. `IMPORT_REACTIVE`: generate an open-vocabulary, map-free molecular graph as
   an `ADD_ATOM`/`ADD_BOND`/`COMMIT` program and identify an active atom.  The
   executor sanitizes the graph, assigns fresh private maps, and requires the
   immediately following electron event to touch an active imported atom.
5. `FINISH`: ask the executor to return the current precursor mixture.

The ordinary FLOW inventory covered 2,008/2,008 ordinary/radical reference
moves in the existing first-1,000-event validation audit.  The three events
outside that inventory are represented by the separate `BE_DELTA` head.

The frozen-train import audit contains 1,343,982 import occurrences: 600,403
electron-participating occurrences (17,856 unique canonical fragments) and
743,579 environment occurrences (6,964 unique).  On strict test, a top-5,000
train catalog covers 95.996% of reactive occurrences but only 91.135% of
complete reactions; therefore a single closed catalog is not the main
reactive-import action.  The environment catalog remains appropriate because
its corresponding occurrence/reaction coverage is 98.998%/97.269%.

## Model

- shared six-layer edge-aware message-passing encoder for current state,
  product and candidate fragments;
- graph context from current/product pooled embeddings and their difference;
- hierarchical family, source, conditional-sink and continue/commit heads;
- sparse BE pair/atom/delta heads;
- graph-to-graph environment-fragment retrieval head;
- role-conditioned reactive-fragment graph decoder with atom, bond, pointer,
  ring-closure and active-site heads;
- state value `V(G)` and action value `Q(G,a)` heads.

No SMILES string, JSON, Python source or atom-map integer is generated.  The
reactive-fragment decoder emits typed graph operations; SMILES is only an
executor serialization after sanitization.  This eliminates text syntax
failures and makes the policy equivariant to atom traversal and map
renumbering.  The executor remains responsible for map allocation,
sanitization, electron accounting, reactive-import use, cycle rejection and
endpoint construction.

## Training

### Stage 1: expert behavior cloning

Train all action heads on the frozen strict executable FlowER universe:
257,167 train, 2,890 valid and 28,967 test reactions.  This is the strict
mechanism universe, not unqualified FlowER full (257,171/2,890/28,971).

Environment import retrieval uses in-batch and frequency-matched negative
fragments.  Reactive imports use teacher-forced graph-program likelihood before
IQL.  Report environment retrieval, reactive graph exact match, active-site
accuracy, executor acceptance and complete-fragment endpoint coverage
separately.

### Stage 2: executor-labelled graph IQL

Pure PPO/GRPO from scratch is not the default.  The terminal reward is sparse,
and a small graph policy can cheaply enumerate many legal-but-wrong actions.
Build an offline replay buffer with:

- expert transitions;
- legal source/sink and sparse-BE counterfactuals executed from expert states;
- short policy rollouts with invalid, cyclic, repeated and endpoint outcomes.

Use Implicit Q-Learning: expectile regression for `V`, a TD loss for `Q`, and
advantage-weighted behavior cloning for the actor.  This avoids bootstrapping
on unobserved actions while still preferring executor-verified improvements.

Recommended reward contract:

- `+1.0` strict full-precursor endpoint exact match;
- `+0.2` executable non-no-op transition;
- potential-based train-only shaping
  `gamma * Phi(G_(t+1)) - Phi(G_t)` from graph distance to the reference endpoint;
- `-0.2` executor rejection, cycle or exact repeated event;
- `-0.01` per committed event and a hard trajectory horizon.

Potential shaping is excluded from validation/test and does not require the
reference endpoint at inference.

### Stage 3: online replay refinement

Mix expert and fresh executor transitions (initially 50/50) and continue the
same IQL/AWR update.  Only after this is stable should a trajectory-balance or
GFlowNet objective be added for diverse valid precursors.  A one-reference
endpoint dataset does not by itself supply the multi-modal reward needed to
justify GFlowNet as the first implementation.

## Evaluation gate

Before a full run, require:

1. reference-independent action coverage, including BE events;
2. map-renumbering and atom-order invariance;
3. one-step family/source/sink/import/commit accuracy;
4. executor acceptance, cycle rate and mean committed events;
5. strict and neutralized full-precursor endpoint Top-1/Top-K;
6. wall time and executor calls per reaction against the Qwen policy.

An overfit or training-loss result is only an implementation smoke, never a
retrosynthesis result.

## Related method boundary

FlowER shows that BE-matrix electron redistribution gives mass/electron
conserving reaction prediction, while Graph2Edits shows that graph-edit
autoregression is effective for retrosynthesis.  This experiment differs by
learning a product-only *inverse*, executor-interactive policy with explicit
fragment import and terminal endpoint actions.  IQL supplies the conservative
offline-to-online RL update; it is not claimed as a new RL algorithm.

Primary references:

- https://www.nature.com/articles/s41586-025-09426-9
- https://www.nature.com/articles/s41467-023-38851-5
- https://arxiv.org/abs/2110.06169
