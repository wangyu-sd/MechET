# MechET causal and compositional experiment plan

This is the authoritative scientific experiment contract. The operational command order is in [`EXECUTION_PLAN.md`](EXECUTION_PLAN.md); the scientific definitions and permitted claims are in [`SCIENTIFIC_THESIS.md`](SCIENTIFIC_THESIS.md).

## 1. Scientific question

> Can mechanistic reasoning in retrosynthesis be made causal and compositional, rather than merely plausible in language?

The main object of study is an environment-owned electron-flow program. The actor proposes explicit source-to-sink actions; successful actions create authoritative molecular-state transitions; `finish_trace` deterministically compiles the committed trace into the only admissible `MECH_PROOF v1`; the executor derives the structural precursor.

```text
product
  -> electron-flow actions
  -> environment-owned trace
  -> deterministic proof compilation
  -> executor-derived precursor
```

The model has no independent answer or proof channel in the main method.

## 2. Hypotheses

### H1 — causal faithfulness

A trace-owned electron-flow program should prevent a correct endpoint from being generated independently of the stated reasoning.

Required evidence:

- trace–proof and trace–endpoint consistency;
- answer–reasoning disagreement in answer-bearing baselines;
- causal interventions on tool observations and molecular states;
- controlled corruptions and false-acceptance/false-rejection analysis;
- recovery or abstention after failed actions.

### H2 — compositional basis

Local electron-flow execution primitives should support primitive-seen/composition-unseen generalization.

Required evidence:

- execution-primitive coverage in training;
- complete mechanism compositions disjoint across train and test;
- matched direct, CoT, edit, proof and trace-owned baselines;
- performance by composition novelty, proof length, topology, family and scaffold;
- invariance to atom-map labels and valid serialization changes.

### H3 — formal and empirical evidence separation

Formal executability and empirical chemical support should be evaluated as distinct evidence layers.

Required evidence:

- the executor remains the hard validity gate;
- textbook and structured-anchor gains exceed no-knowledge and length-matched irrelevant-text controls;
- knowledge interventions change behavior in the expected direction;
- forward evidence is independently calibrated before use in ranking or reward;
- no evidence source overrides a formal failure.

## 3. Method conditions

### 3.1 Main method

`TraceOwnedAgentEnv` or `KnowledgeAugmentedAgentEnv`:

```text
inspect_state
retrieve_textbook_guidance      optional soft evidence
retrieve_primitives             optional structured knowledge anchors
import_fragment
apply_electron_move
apply_coupled_electron_moves
finish_trace
abstain
```

`submit_proof` is disabled. The environment compiles the trace and derives the endpoint.

### 3.2 Required baselines

```text
Outcome-only precursor generation
Free-form CoT plus answer
State-CoT plus answer
Reaction-center/synthon prediction
Net-edit generation
Independent complete MECH_PROOF generation
Legacy loose tool trace plus submitted proof
Trace-owned Tool-CoT
```

The complete-proof and loose-trace paths are baselines, not alternate definitions of the main method.

### 3.3 Evidence conditions

```text
trace_no_knowledge
trace_length_matched_irrelevant
trace_textbook_rag
trace_structured_anchors
trace_text_plus_anchors
direct_textbook_rag
gold_passage_upper_bound         when labels exist
```

Natural-language passages, structured mechanistic knowledge anchors and learned forward scores are soft evidence.

## 4. Terminology contract

### Electron-flow execution primitives

Local executable actions such as `LP -> BOND`, `BOND -> ATOM` and `BOND -> BOND`. They define the causal action space and the compositional split.

### Mechanistic knowledge anchors

Curated structured records containing patterns, role bindings, candidate moves, warnings, competitors and provenance. Anchor IDs must not be used as a substitute for execution-primitive composition in H2.

### Structural endpoint

The multiset of atom-contributing precursor fragments. Solvents, catalysts, salts and spectators are reported separately.

## 5. Data contract

### 5.1 Scope

Primary results use mapped, closed-shell, two-electron polar organic chemistry. Radicals, photochemistry, electrochemistry, transition-metal orbital changes and spin-changing reactions remain out of scope unless separately represented and verified.

### 5.2 Required sources

```text
FlowER-derived mechanistic trajectories and compiled proofs
USPTO-50K for standard comparability
USPTO-MIT/USPTO-FULL for overlap audit and secondary studies
mech-USPTO-31K or equivalent source/sink supervision
PMechDB-derived rows only under accepted upstream access terms
ORD only as optional condition/outcome evidence after quality control
PaRoutes or an equivalently frozen planning benchmark for extensions
```

### 5.3 Non-negotiable rules

1. Freeze benchmark hashes before training.
2. Pin source revisions and licenses.
3. Remove overlap from training, never from the test set after evaluation.
4. Build all headline baselines from the same stable-ID intersection.
5. Preserve structural endpoints separately from environmental components.
6. Never invent source/sink labels for ambiguous rows.
7. Never treat an alternative executable endpoint as a negative without independent evidence.
8. Keep actor, executor and forward-expert lineage explicit.
9. Report every quarantine reason and conversion filter.
10. Freeze the execution-primitive vocabulary before constructing MechComp-OOD.

## 6. Phase A — data feasibility

### A1. Proof execution coverage

Compile proof rows and report:

```text
rows read
parseable proofs
executable proofs
endpoint-reconstructing proofs
coverage by family, proof length, changed bonds and topology
```

### A2. Proof-to-trace conversion coverage

The current conservative converter accepts only uniquely recoverable two-electron actions. Report stable quarantine codes:

```text
NONLINEAR_PROOF_UNSUPPORTED
AMBIGUOUS_ELECTRON_PAIRING
UNPAIRED_LONE_PAIR_DELTA
ODD_LONE_PAIR_DELTA
EDGE_HAS_NO_INFERABLE_MOVES
IMPORT_REPLAY_FAILED
MOVE_REPLAY_FAILED
MOVE_REPLAY_STATE_MISMATCH
TRACE_TERMINAL_REPLAY_FAILED
```

Required output:

```text
overall conversion rate
conversion rate by family
conversion rate by proof length
conversion rate by topology
accepted and rejected complexity distributions
imports, moves and trace length
endpoint replay rate
```

Stopping rule: do not describe the retained subset as broad organic chemistry if the converter supports only a narrow family set.

### A3. Leakage and split integrity

Audit exact reaction, structural reaction, product, scaffold, reaction center, execution-primitive composition, patent family and temporal overlap where metadata permits.

Build at least:

```text
exact-clean
scaffold-clean
center-clean
```

## 7. Phase B — supervised learning feasibility

### B1. Matched task variants

Use identical stable IDs and structural endpoints for:

```text
outcome-only
free-form CoT
state-CoT
reaction-center/synthon
net-edit
complete proof
trace-owned Tool-CoT
```

### B2. Tool-SFT construction

Every accepted Tool-SFT row must:

- contain explicit tool calls and results;
- replay through the same trace-owned environment used at inference;
- end with `finish_trace`;
- store the trace digest and compiled proof;
- have `endpoint_source=environment_owned_trace`;
- reproduce the expected structural precursor.

### B3. Real training smoke test

Before paper-scale training, overfit 32–128 rows and verify:

```text
assistant supervision mask is non-empty
loss decreases
valid tool-call rate increases
finish_trace call rate increases
trace-bound execution increases
endpoint exact approaches the small-set ceiling
```

Dry-run validation alone is not sufficient.

### B4. SFT checkpoint lineage

Every trace-owned RL run must record:

```text
base-model revision
Tool-SFT adapter path and hash
Tool-SFT data-manifest hash
environment revision
executor revision
training config and seed
```

Pure RL from an untrained tool policy is not the primary method.

## 8. Phase C — H1 causal faithfulness

### C1. Endpoint and process comparison

Compare all required baselines under matched model/data/optimization budgets.

Metrics:

```text
structural precursor Top-1/5/10
reaction-center accuracy
synthon exact match
FormatPass
ExecutePass
trace–proof agreement
trace–endpoint agreement
answer–reasoning disagreement
unnecessary-action count
tool-failure recovery
abstention coverage and selective risk
```

### C2. Causal interventions

Run:

```text
remove tool observations
shuffle tool observations
replace observations with stale states
disable inspect_state
disable intermediate move execution
remove failure certificates
permit independent proof submission only in a baseline
```

Report absolute and relative changes in endpoint, execution and consistency metrics.

Claim gate: tool-grounded reasoning is unsupported if the trace-owned model is insensitive to observation removal or corruption.

### C3. Formal falsification benchmark

Controlled corruptions:

```text
parse
atom map
bond precondition
lone-pair accounting
charge transition
import
reachability
dependency
source empty
sink capacity
missing coupled arrow
```

Metrics:

```text
false acceptance rate
false rejection rate
failure-code accuracy
first-failing-action localization
repair success
new-error introduction
over-edit rate
```

## 9. Phase D — H2 compositional generalization

### D1. MechComp-OOD construction

Hold out complete execution-primitive compositions while requiring every constituent primitive to appear in training above a declared minimum frequency.

No knowledge-anchor IDs are used to define the split.

### D2. Comparisons

```text
direct answer
free-form CoT
state-CoT
net edit
complete proof
trace-owned Tool-CoT
trace-owned Tool-CoT plus evidence
```

### D3. Metrics and strata

```text
endpoint and execution metrics
partial-order proof equivalence
execution-primitive precision/recall
composition exact match
performance versus composition frequency
proof length and changed-atom complexity
ring-forming/ring-changing strata
stereochemical-change strata
chain/tree/DAG topology
family and scaffold
```

### D4. Representation invariance

Test synchronized:

```text
atom-map permutation
state-ID renaming
edge serialization
commuting independent events
component ordering
valid equivalent proof variants
```

Separate semantic invariance from exact-string equality.

## 10. Phase E — H3 evidence separation

### E1. Matched evidence suite

Build all six evidence conditions from two source datasets: textbook-only trace rows and textbook-plus-anchor trace rows. No manually prepared anchors-only or direct-open-book files are permitted.

Validate:

```text
same stable IDs
same targets and structural endpoints
same model and tokenizer revision
same optimizer, LoRA and updates
reported input and supervised token budgets
reported context and tool budgets
same seeds
```

### E2. Knowledge metrics

```text
retrieval Recall@K and Precision@K
passage and gold-passage rank
citation correctness
context characters and tokenizer-specific tokens
retrieval latency
knowledge-call rate
anchor-call rate
knowledge direct-reward violations
```

### E3. Knowledge interventions

```text
length-matched irrelevant text
passage shuffle
same-topic wrong passage
remove warnings
remove competing-pathway text
```

Claim gate:

```text
textbook RAG > trace no knowledge
and
textbook RAG > length-matched irrelevant text
```

### E4. Forward evidence

Validate the forward expert independently before actor integration.

Required comparisons:

```text
ordinary product compatibility
source/sink process model
process plus compatibility
process plus conditions
random negatives versus explicit competitors
```

Metrics:

```text
source/sink Top-k and move MRR
target rank and recovery at k
competitor margin
Brier score and ECE
risk–coverage
uncertainty–error correlation
family-wise false acceptance/rejection
```

Formal invalidity remains a hard prune. Forward evidence is a secondary result unless it materially improves calibration or explicit competitor ranking.

## 11. Phase F — scale and optimization

Compare approximately:

```text
0.6B trace-owned actor
1.7B trace-owned actor
8B trace-owned actor
8B direct-answer reference
8B direct-answer plus identical bounded textbook evidence
```

Report:

```text
accuracy and reliability
GPU hours and peak memory
input and generated tokens
tool calls and executor calls
latency
verified endpoints per compute budget
```

Only after SFT and H1 pilots pass, compare:

```text
Tool-SFT
Tool-SFT plus formal process RL
Tool-SFT plus endpoint RL
Tool-SFT plus calibrated forward evidence
```

Reward logging must separate formal, endpoint, forward, selectivity, failure and length terms. Soft rewards cannot offset formal failure.

## 12. Phase G — hypotheses, repair and planning extensions

### G1. Test-time hypotheses

For K in `{1, 4, 16, 64}` report:

```text
ExecutePass@K
EndpointPass@K
unique executable proof classes
unique execution-primitive compositions
unique structural endpoints
latency and model/tool calls
```

### G2. Repair

Compare:

```text
resample only
deterministic semantics-preserving repair
separate Repair Actor
same-agent revision after tool feedback
```

### G3. Planning

Use frozen offline candidate pools first. Compare inverse score, formal hard gating and calibrated empirical evidence under matched search budgets.

Metrics:

```text
solved rate
fully verified route rate
formal-invalid expansion rate
route length and diversity
nodes expanded
reaction-model calls
wall-clock time
```

Planning is an extension and cannot rescue failed H1 or H2 claims.

## 13. Required paper result package

### Result 1 — causal reasoning

- matched endpoint/process table;
- trace/proof/endpoint consistency;
- intervention curves;
- formal corruption benchmark;
- examples of answer-bearing bypass and trace-owned prevention.

### Result 2 — compositional reasoning

- MechComp-OOD construction audit;
- IID versus composition-OOD comparison;
- performance versus composition novelty and complexity;
- invariance controls;
- coverage-limited failures.

### Result 3 — evidence separation

- six-condition matched evidence table;
- knowledge intervention effects;
- independently calibrated forward evidence;
- examples of multiple executable paths with different empirical support.

### Optional Result 4 — efficiency and planning

- model-scale/compute-normalized comparison;
- hypothesis-set curves;
- fully verified planning results.

## 14. Reproducibility contract

Every reported checkpoint and result must include:

```text
repository commit
base-model/tokenizer revision
adapter hash and lineage
data and benchmark hashes
source licenses and revisions
executor and environment revision
condition manifest
optimizer and update count
seed
GPU hours and hardware
inference, tool and search budgets
calibration thresholds
raw predictions and evaluation config
```

## 15. Global stopping rules

Do not make the corresponding claim when:

- conversion coverage is narrow or unreported;
- matched IDs or endpoints differ;
- tool-observation interventions have negligible effect;
- composition-OOD contains unseen execution primitives;
- evidence gains are explained by irrelevant context;
- knowledge receives direct reward;
- a learned score overrides formal execution;
- forward calibration degrades on the frozen audit set;
- test assets are modified after observing final failures;
- paper-scale RL begins before Tool-SFT demonstrates executable learning.
