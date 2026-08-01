# MechET authoritative ICLR collaboration and experiment plan

This document is the authoritative scientific and execution contract for the MechET ICLR submission. It is written for collaborators: every claim is tied to a falsifiable experiment, every experiment is tied to a frozen data and model contract, and every result has an interpretation and stopping rule.

The current paper thesis is:

> Retrosynthesis can be learned as backward composition of reusable electron-flow primitives. A small tool-using inverse actor proposes and executes those primitives; a deterministic executor prevents formal hallucination; an independent compact forward expert attempts to falsify the proposal through forward process, target-recovery, and selectivity evidence. The same verified single-step objects support reliable multistep planning.

The principal method is:

```text
atom-mapped product
  -> small inverse actor reasons through electron-flow tools
  -> deterministic executor checks each state transition and derives precursors
  -> compact independent forward expert scores forward process and target recovery
  -> target is compared with explicit competing products/pathways
  -> proposal is retained, revised, abstained, or expanded in route search
```

The compatibility path remains:

```text
product
  -> sample K complete MECH_PROOF v1 programs
  -> execute and falsify every program
  -> Generate–Falsify–Repair or resample
  -> deduplicate executable programs by partial-order equivalence
  -> group structural endpoints and rerank with forward evidence
```

The deterministic executor is never trained. Learned forward and selectivity scores are soft evidence and can never override a formal execution failure.

---

# 0. Executive decision summary

## 0.1 Central scientific gap

Current retrosynthesis systems generally choose one of two representations:

1. high-level templates or named transformations, which are reliable but isolate reaction knowledge into discrete classes;
2. direct graph/string generation, which is flexible but can hallucinate reactions and gives weak evidence that the generated rationale caused the endpoint.

MechET tests an intermediate representation: reusable electron-flow primitives that can be composed, executed, falsified, and reused as route-search actions.

## 0.2 What is new

The submission does not claim that electron arrows, chain-of-thought, forward prediction, RL, or multistep planning are individually new. The novelty under test is the joint design in which electron-flow primitives are:

- the inverse actor's reasoning units;
- explicit stateful tool actions;
- deterministic formal-verification units;
- process-reward units for RL;
- compositional-generalization units;
- single-step edges in multistep planning;
- forward-process units evaluated by an independent compact expert.

## 0.3 Core ICLR claims

| Claim | Required evidence |
|---|---|
| C1. Electron-flow primitives are reusable | primitive accuracy, primitive coverage, composition-held-out performance, family and scaffold OOD |
| C2. Executable CoT is faithful | no independent answer channel, answer–reasoning disagreement baselines, causal tool-use ablations, controlled corruption tests |
| C3. Tool grounding reduces hallucination | execution rate, false acceptance rate, failure localization, recovery after failed tools, abstention quality |
| C4. Forward falsification improves reliability | forward target rank, process score, selectivity margin, calibration, risk–coverage, adversarial audit |
| C5. Small models can benefit disproportionately from tools | matched 0.6B/~2B/8B scale study, compute-normalized performance, process efficiency |
| C6. Verified edges improve planning | route solved rate, fully verified route rate, invalid edge rate, route cost, nodes expanded, route diversity |

## 0.4 Claims that are not permitted from current software alone

The paper must not claim:

- proven low activation barriers;
- favorable kinetics, yield, or laboratory success;
- universal condition compatibility;
- unique physical reaction mechanism from product alone;
- correct radical, photochemical, spin-state, organometallic, or coordination chemistry outside the stated scope;
- experimental discovery solely from formal execution or a learned score.

## 0.5 Current scope

The primary paper scope is mapped, closed-shell, two-electron polar organic chemistry. Radical, metal-mediated, photochemical, electrochemical, and spin-changing reactions may be reported as out-of-scope strata but cannot be mixed into the primary headline metrics unless an appropriate representation and verifier are implemented.

---

# 1. Method specification

## 1.1 Electron-flow state

A state is an atom-mapped molecular graph with:

```text
atoms and atom maps
formal charges
bond orders
implicit/explicit hydrogens
non-bonding electron accounting
component membership
optional reagents/conditions metadata
```

The executor is the source of truth for the state. The language model never maintains an authoritative hidden molecular state.

## 1.2 Electron containers and moves

The explicit source/sink environment initially supports:

```text
source: LP(atom) or BOND(atom_i, atom_j)
sink:   ATOM(atom) / LP_SLOT(atom) / BOND_SLOT(atom_i, atom_j)
electrons: 2
```

Coupled arrows belonging to one elementary event are applied atomically. This avoids forcing an invalid serialized intermediate, such as a pentavalent carbon during an SN2 event.

The current `MECH_PROOF v1` representation records the compiled bond/lone-pair/charge changes. Explicit source-to-sink tool calls are the agent-facing interface; `MECH_PROOF v1` remains the deterministic executable and compatibility format.

## 1.3 Inverse actor

The inverse actor receives a mapped product and may:

1. inspect the electron-container inventory;
2. select a candidate reaction center;
3. test one or more reverse electron moves;
4. read the resulting state or failure certificate;
5. continue, backtrack, or abstain;
6. submit one complete executable proof.

The primary actor is a small tool-using causal language model. Complete-proof generation with a larger Qwen backbone remains a matched baseline.

## 1.4 Deterministic executor

The executor checks:

- proof parsing and schema;
- atom-map existence and uniqueness;
- imports and molecular components;
- bond preconditions and bond-order transitions;
- formal-charge preconditions and transitions;
- derived lone-pair changes;
- exact electron conservation per edge;
- RDKit sanitization;
- reachability and precursor-state derivation;
- chain/tree/DAG join consistency;
- endpoint derivation without an independent answer channel.

A formal failure is a hard rejection.

## 1.5 Compact forward electron-flow expert

The forward expert is architecturally independent from the inverse actor. The default model is a compact graph message-passing network with:

- atom and bond embeddings;
- electron-container embeddings;
- source pointer head;
- source-conditioned sink pointer head;
- precursor-product compatibility head;
- condition embedding;
- target-versus-competitor contrastive score;
- uncertainty output.

It provides three different evidence types:

```text
step evidence:      p(source | state), p(sink | source, state)
closure evidence:   score(precursors, target | conditions)
selectivity evidence: score(target) - max score(competitor)
```

These must be reported separately. A single scalar `plausibility` score is not sufficient for paper analysis.

## 1.6 Framework-neutral environment

`MechETAgentEnv` owns the chemistry state, tool methods, tool budget, visited-state detection, proof submission, abstention, process reward, terminal reward, and rollout trace. Training frameworks only supply rollout and optimization infrastructure.

Reference implementation:

- TRL agentic GRPO for the first small-model experiments;
- verl as the preferred distributed multi-turn scale backend after reward validation;
- Syntheseus for matched-budget multistep planning;
- AiZynthFinder as an external template-planning baseline;
- Agent Lightning or Prime Verifiers only as optional tracing/packaging adapters.

## 1.7 Alternating actor–verifier learning

The inverse actor and forward expert are not updated simultaneously as a GAN.

Round \(r\):

```text
1. freeze forward expert F_r
2. train inverse actor A_r with formal process reward and soft forward terminal reward
3. freeze actor A_r
4. generate actor–verifier disagreements
5. place disagreements in an unreviewed audit queue
6. independently verify a subset
7. train/calibrate F_{r+1} only on verified labels
8. repeat for a small fixed number of rounds
```

An endpoint differing from one recorded patent route is not automatically a negative. Accepted negative label sources are:

```text
expert review
experiment
known competing product/pathway
independent calibrated ensemble
```

## 1.8 Multistep planning

A route edge contains:

```text
product
executor-derived structural precursors
proof
formal status
inverse actor score
forward target score
selectivity margin
uncertainty
conditions/evidence metadata when available
```

Formal invalidity is a hard prune. Learned evidence contributes a soft edge cost. Route search terminates only when all leaves satisfy the frozen building-block inventory.

---

# 2. Frozen data contract

## 2.1 Data sources

### Inverse/proof supervision

- FlowER-derived state trajectories and bond–electron semantics;
- compiled `MECH_PROOF v1` programs;
- USPTO-50K for standard one-step comparison;
- USPTO-MIT/USPTO-FULL for broader audit and secondary benchmarks.

### Forward/process supervision

- mech-USPTO-31K or equivalent source/sink mechanistic data;
- FlowER mechanistic states where source/sink pairing can be recovered without inventing labels;
- PMechDB-derived elementary polar steps only under explicit upstream license acceptance;
- outcome-only reaction data for precursor-product compatibility;
- ORD for optional condition/outcome/provenance evidence after mapping and quality filtering.

### Multistep planning

- PaRoutes or an equivalently frozen target and stock benchmark;
- a fixed building-block inventory;
- an offline one-step hypothesis pool for the first matched planner comparison.

## 2.2 Canonical directory layout

```text
data/
  raw/
    <source>/<revision>/manifest.json
  mechet_sft/
  mechet_proof_sft/
  mechet_proof_clean/
    train.jsonl
    valid.jsonl
    test.jsonl
    quarantine.jsonl
    manifest.json
  proof_curriculum/
    equivalence_train.jsonl
    corruptions.jsonl
    preferences.jsonl
    repairs.jsonl
  forward_expert/
    reactions.jsonl
    quarantine.jsonl
    steps/{train,valid,test}.jsonl
    audit_candidates.jsonl
    verified_negatives.jsonl
  ood/
    mechcomp/
    family/
    scaffold/
    temporal/
  benchmarks/
    uspto50k/
    external_mechanisms/
    planning/
  manifests/
    benchmark_hashes.json
    model_lineage.json
```

## 2.3 Required row fields

### Inverse proof row

```text
stable id
source dataset and split
mapped target product
MECH_PROOF v1 assistant output
executor-derived full precursor
atom-contributing structural precursor
proof topology
partial-order equivalence digest
primitive signatures
mechanism composition digest
reaction-center keys
patent/family/date metadata when recoverable
```

### Forward row

```text
stable id
mapped reactants/current state
mapped product/next state
ordered or coupled source-sink moves when supported
reaction/mechanism family when available
conditions and provenance
competitor products/pathways when available
label and label provenance
split and source revision
```

### Planning row

```text
target
frozen stock identifier
search budget
one-step candidate-pool revision
planner and planner config
solution graph/route artifacts
```

## 2.4 Non-negotiable data rules

1. Freeze benchmark files and SHA-256 hashes before training.
2. Pin external dataset revisions and preserve manifests.
3. Never filter a test set after seeing model results.
4. Remove overlap from training and write all removals to quarantine.
5. Build matched baselines from the same stable-ID intersection.
6. Keep structural precursors separate from solvents, catalysts, salts, and spectators.
7. Never invent source/sink arrows for ambiguous rows.
8. Outcome-only rows may train product compatibility but not arrow heads.
9. Never treat a different executable endpoint as a negative without independent evidence.
10. Preserve all licensing and access restrictions.
11. Keep inverse actor and forward expert lineage independent.
12. Use cross-fitting or held-out verifier folds for actor reward where feasible.

---

# 3. Pipeline A — source data, audit, and proof curriculum

## A0. Freeze source revisions and licenses

**Owner:** data lead

For every external source, record:

```text
repository/source URL
revision or release
license/access terms
download command
download timestamp
SHA-256 for every file
expected schema
```

Use:

```bash
python scripts/forward_expert_data.py download \
  --dataset mech_uspto_31k \
  --revision <frozen-revision> \
  --output data/raw
```

Restricted sources must require explicit acknowledgement. Dry-run output must be reviewed before download.

**Gate:** no training starts until the source registry is frozen.

## A1. Standardize forward reaction data

```bash
python scripts/forward_expert_data.py standardize \
  --input data/raw/mech_uspto_31k \
  --output data/forward_expert/reactions.jsonl \
  --source mech_uspto_31k \
  --quarantine data/forward_expert/quarantine.jsonl

python scripts/forward_expert_data.py build \
  --input data/forward_expert/reactions.jsonl \
  --output-dir data/forward_expert/steps
```

**Required report:** rows read/written/quarantined, rows with/without arrow labels, split counts, family counts, map failures, duplicate rate, condition coverage.

**Gate:** no source/sink label is accepted unless it is present in the source or deterministically recoverable under an audited rule.

## A2. Build state-annotated inverse rows

```bash
python scripts/build_mechet_sft.py \
  --flower-root /path/to/flower_new_dataset \
  --out-dir data/mechet_sft \
  --splits train valid test
```

**Gate:** stable source IDs, mapped products, states, edges, and original initial species parse; all failures are counted.

## A3. Compile executable proofs

```bash
python scripts/build_mechet_proof_sft.py \
  --input-dir data/mechet_sft \
  --output-dir data/mechet_proof_sft \
  --splits train valid test
```

**Gate:** every accepted proof executes, reconstructs the source endpoint, and contains no independent `<answer>` channel.

## A4. Freeze benchmark lineage and audit overlap

```bash
python scripts/audit_reaction_overlap.py \
  --train data/mechet_proof_sft/train.jsonl \
  --benchmark data/benchmarks/uspto50k/test.csv \
  --benchmark-format reaction_table \
  --reaction-field reaction_smiles \
  --out-dir outputs/data_audit/flower_vs_uspto50k_test
```

Required overlap keys:

```text
exact full reaction
exact structural reaction
product
Bemis–Murcko scaffold
reaction center
proof primitive composition
patent/family/date where available
```

**Gate:** standard USPTO results are described as comparability results, not fully external validation.

## A5. Build decontaminated conditions

```bash
python scripts/build_decontaminated_dataset.py \
  --train data/mechet_proof_sft/train.jsonl \
  --benchmark data/benchmarks/uspto50k/test.csv \
  --output data/mechet_proof_clean/train.jsonl \
  --manifest data/mechet_proof_clean/manifest.json \
  --policy exact_structural product
```

Required conditions:

```text
exact-clean
scaffold-clean
center-clean
```

Report original, retained, removed, and quarantined rows per policy.

## A6. Build matched task variants

```bash
python scripts/build_iclr_task_variants.py \
  --proof-input data/mechet_proof_clean/train.jsonl \
  --state-input data/mechet_state_clean/train.jsonl \
  --output-dir data/iclr_tasks

python scripts/validate_iclr_data_contract.py \
  --task outcome=data/iclr_tasks/outcome_only.jsonl \
  --task state=data/iclr_tasks/state_cot.jsonl \
  --task edit=data/iclr_tasks/net_edit.jsonl \
  --task proof=data/iclr_tasks/proof.jsonl \
  --out outputs/iclr/data_contract.json
```

Required matched tasks:

```text
Outcome-only
Free-form CoT
State-CoT
Reaction-center/synthon
Net-edit
Complete proof
Tool-interleaved electron-flow CoT
```

**Gate:** identical stable IDs and structural endpoints.

## A7. Build verified equivalence augmentation

```bash
python scripts/build_proof_equivalence_data.py \
  --input data/mechet_proof_clean/train.jsonl \
  --output data/proof_curriculum/equivalence_train.jsonl \
  --variants-per-row 4
```

Variants include synchronized atom-map permutation, state renaming, edge serialization changes, and commuting independent events. Every variant must execute and remain partial-order equivalent.

## A8. Build corruption and falsification curriculum

```bash
python scripts/build_proof_corruption_data.py \
  --input data/mechet_proof_clean/train.jsonl \
  --output data/proof_curriculum/corruptions.jsonl \
  --include-valid-controls
```

Corruption families include parse, atom-map, bond, lone-pair, charge, import, reachability, precursor, dependency, nonlocal-move, source-empty, sink-valence, and missing-coupled-arrow errors where supported.

Store the intended label, observed executor result, stable failure code, and first failing edge/action.

## A9. Build preference and repair baselines

Safe preference:

```text
formally executable proof > formally invalid proof
```

```bash
python scripts/build_proof_preferences.py \
  --corruptions data/proof_curriculum/corruptions.jsonl \
  --output data/proof_curriculum/preferences.jsonl

python scripts/build_proof_repair_data.py \
  --corruptions data/proof_curriculum/corruptions.jsonl \
  --output data/proof_curriculum/repairs.jsonl
```

Do not rank two executable endpoints automatically.

## A10. Build compositional and domain-shift splits

Required test conditions:

```text
IID random/patent split
primitive-seen, composition-unseen MechComp-OOD
reaction-family holdout
product-scaffold holdout
temporal holdout where dates exist
reaction-center complexity strata
ring-forming/ring-changing strata
stereochemical-change strata
proof-length/topology strata
```

**Gate:** every held-out composition uses primitives covered in training above a minimum frequency.

## A11. Build selectivity and hard-negative benchmark

Construct paired or grouped examples sharing the same reactants/current state:

- correct versus alternative reaction center;
- regioisomers;
- stereoisomers;
- SN2 versus E2 or other plausible family competition;
- alternative carbonyls, leaving groups, or nucleophiles;
- over-reaction, dimerization, and side-product candidates where supported;
- correct reaction under incompatible or missing conditions;
- formally valid but mechanistically incomplete arrow sequences.

Each negative must record its generation rule and evidence status.

## A12. Freeze planning benchmark

Freeze:

```text
target list
building-block inventory
candidate-pool revision
search algorithms
maximum reaction-model calls
maximum iterations
wall-clock budget
maximum routes
random seeds
```

The first planner comparison uses an offline candidate pool so all algorithms receive identical one-step proposals.

---

# 4. Pipeline B — matched baselines and proof models

## B0. Common training contract

For matched language-model baselines, hold constant where applicable:

```text
tokenizer
base-model family
LoRA rank and target modules
assistant-token budget
optimizer and schedule
number of updates
seeds
maximum context/completion length
training examples
```

Report three seeds unless the run is explicitly a pilot. Every checkpoint must retain the base-model revision, adapter hash, data hash, config, environment revision, and total GPU hours.

## B1. Direct and rationale baselines

Train:

1. Outcome-only precursor generation;
2. Free-form CoT plus answer;
3. State-CoT plus answer;
4. Reaction-center/synthon prediction;
5. Net-edit generation;
6. Complete `MECH_PROOF v1` generation.

The answer-bearing baselines are required to measure answer–reasoning disagreement.

## B2. Supervised objectives

### Assistant-token SFT

```text
L_SFT = - sum_t m_t log p_theta(y_t | x, y_<t)
```

where `m_t=1` only for supervised assistant tokens.

### Preference baseline

```text
L_DPO = - log sigmoid(beta * ((log pi(chosen)-log pi(rejected))
                               - reference_margin))
```

Preferences are limited to executor-grounded formal validity unless independently supported chemical labels exist.

### On-policy reward learning

```text
A_i = r_i - mean_{j != i}(r_j)
L_RLVR = - mean_i A_i * mean_t log p_theta(y_i,t | x, y_i,<t)
```

Complete-proof legacy training is accurately described as RLOO/group-relative REINFORCE. The agentic TRL reference path uses its configured GRPO implementation and must report its exact version and arguments.

### Repair baseline

```text
L_repair = - sum_t w_t log p_theta(y*_t | x, y*_<t)
```

with higher weight on corrected spans. This is a baseline; the main agent can instead react to tool feedback with the same policy.

## B3. Proof-SFT and equivalence augmentation

```bash
python scripts/train_mechet_sft.py \
  --config configs/proof/proof_actor_sft.yaml
```

Ablate:

```text
one serialization
verified equivalent serializations
map-only augmentation
edge-order-only augmentation
all verified augmentations
```

## B4. Verifier-DPO baseline

```bash
python scripts/train_proof_dpo.py \
  --config configs/proof/proof_dpo.yaml
```

Measure whether DPO reduces parse/formal failures before on-policy training without reducing endpoint diversity.

## B5. Complete-proof RLVR baseline

Retain accuracy and executable-class diversity modes. Invalid strings never receive diversity reward.

Required logging:

```text
reward components
execute rate
endpoint rate
effective-group rate
unique executable classes/group
proof length
token length
KL or policy drift statistics
```

## B6. Compact forward expert

```bash
python scripts/train_forward_expert.py \
  --config configs/forward/forward_expert_small.yaml
```

Train heads for:

```text
source
sink conditioned on source
reaction compatibility
target-versus-competitor ranking
```

Required ablations:

- outcome-only reaction scorer;
- process heads only;
- reaction compatibility only;
- process + reaction compatibility;
- condition channel removed;
- random negatives versus explicit competitors;
- 2D graph model versus optional pretrained encoder baseline.

The forward expert is frozen and calibrated before actor RL.

## B7. Small inverse tool-using actor

Reference command:

```bash
python scripts/train_inverse_agent_trl.py \
  --config configs/agent/inverse_trl_grpo.yaml \
  --dry-run --limit 8

python scripts/train_inverse_agent_trl.py \
  --config configs/agent/inverse_trl_grpo.yaml
```

Required actor scales:

```text
small: approximately 0.6B
medium: approximately 1–2B
reference: approximately 8B
```

All scale comparisons use the same tool environment and reward contract. The tool-use actor should be initialized from a matched supervised model where possible; pure RL from an untrained tool policy is not the primary condition.

## B8. Agent reward decomposition

Hard process terms:

```text
valid tool schema
source exists
sink exists and has legal capacity
move executes
state sanitizes
electron accounting is conserved
proof executes
```

Soft terms:

```text
forward process score
target recovery
selectivity margin
uncertainty
precedent/condition evidence where available
```

Reference reward:

```text
R = R_formal
  + lambda_step R_verified_steps
  + lambda_endpoint R_endpoint
  + lambda_forward R_forward
  + lambda_selectivity R_selectivity
  - lambda_fail N_failed_tools
  - lambda_length N_unnecessary_steps
  - lambda_uncertainty U
```

Formal failure cannot be offset by soft scores.

## B9. Alternating actor–verifier training

Required conditions:

```text
no alternating update
one audited alternating round
two audited alternating rounds
```

Workflow:

```bash
python scripts/mine_bidirectional_hard_negatives.py \
  --predictions outputs/proof/hypotheses_forward_ranked.jsonl \
  --checkpoint outputs/forward_expert/small/best \
  --output data/forward_expert/audit_candidates.jsonl
```

Audit outputs begin with:

```text
label = null
training_eligible = false
audit_status = unreviewed
```

Only independently verified negatives may enter:

```bash
python scripts/train_forward_expert_hard_negative.py \
  --checkpoint outputs/forward_expert/small/best \
  --positive-data data/forward_expert/steps/train.jsonl \
  --negative-data data/forward_expert/verified_negatives.jsonl \
  --output outputs/forward_expert/adversarial_round1
```

After each round, recalibrate the forward expert on a frozen validation set and rerun its false-acceptance/false-rejection audit.

## B10. Distributed scale migration

TRL is the reference path. Migrate the same `MechETAgentEnv` to verl only after:

- reward decomposition is stable;
- tool traces are reproducible;
- reward hacking audit passes;
- synchronous rollout latency dominates training;
- the small-model pilot shows a credible learning signal.

Do not maintain divergent chemistry semantics in TRL and verl.

---

# 5. Pipeline C — inference modes

## C0. Outcome-only inference

Generate Top-K precursor strings directly. Record validity, structural endpoint accuracy, and diversity. This is the minimum literature baseline.

## C1. Free-form and State-CoT inference

Generate rationale/states plus an answer. Independently compare the answer with the state-derived or rationale-implied endpoint to measure answer–reasoning disagreement.

## C2. Complete-proof inference

Generate one `MECH_PROOF v1`, execute it, and derive the endpoint. No answer channel is permitted.

## C3. Tool-interleaved electron-flow inference

The actor alternates between reasoning and environment tools. Record:

```text
all tool calls
all tool results
state sequence
failed actions
successful actions
proof submission
abstention
terminal reward decomposition
```

## C4. K-hypothesis inference

```bash
python scripts/infer_proof_hypotheses.py \
  --data data/mechet_proof_clean/test.jsonl \
  --adapter outputs/proof/actor/adapter \
  --samples-per-target 64 \
  --out outputs/proof/hypotheses.jsonl
```

Report K in `{1, 4, 16, 64}` where compute permits. Execute every candidate before deduplication.

## C5. Generate–Falsify–Repair

Run bounded repair or agent revision after a structured failure. Compare:

```text
resample only
separate Repair Actor
same-agent tool revision
deterministic LP-only repair where applicable
```

## C6. Forward reranking

```bash
python scripts/rerank_proof_hypotheses_forward.py \
  --predictions outputs/proof/hypotheses.jsonl \
  --checkpoint outputs/forward_expert/small/best \
  --output outputs/proof/hypotheses_forward_ranked.jsonl
```

Ranking must preserve the hierarchy:

```text
formal execution
then forward target recovery/selectivity/uncertainty
then inverse actor likelihood
then diversity or novelty
```

## C7. Verifier-guided test-time search

At selected intermediate states, branch over multiple candidate moves. Hard-prune only deterministic failures. Use learned forward evidence for soft ranking.

Report the exact node, tool-call, and forward-model-call budget.

## C8. Abstention

A prediction may terminate with `ABSTAIN` when no sufficiently supported path is found. Report coverage, selective risk, and abstention reason distribution.

## C9. Offline multistep planning

Use frozen candidate pools for the primary planner comparison:

```bash
python scripts/run_syntheseus_search.py \
  --candidate-pool outputs/proof/hypotheses_forward_ranked.jsonl \
  --targets data/benchmarks/planning/targets.smi \
  --inventory data/benchmarks/planning/inventory.smi \
  --output-dir outputs/planning/syntheseus_retrostar \
  --algorithm retro_star
```

Compare native best-first, Syntheseus Retro*, breadth-first, and an external template-planning baseline under matched budgets.

## C10. Online multistep planning

Only after offline benchmarking is frozen, replace the candidate pool with an online actor provider. Report online generation latency and stochasticity separately.

## C11. Reaction-network expansion

Store multiple formally executable and forward-supported hypotheses as a reaction hypergraph. This is a secondary result unless the network analysis is sufficiently strong to support a separate scientific claim.

---

# 6. Pipeline D — validation experiments

## D0. Data integrity and leakage

Required outputs:

- frozen source/benchmark hashes;
- source revisions and licenses;
- overlap matrix;
- quarantine counts and reasons;
- split sizes and family distributions;
- patent/temporal coverage limitations;
- actor/verifier cross-fitting policy.

A failure here blocks all headline comparisons.

## D1. Matched single-step endpoint performance

Compare:

```text
Outcome-only
Free-form CoT
State-CoT
Reaction-center/synthon
Net-edit
Proof-SFT
Proof-DPO
Complete-proof RLVR
Tool-CoT SFT
Tool-CoT RL
Tool-CoT RL + forward reward
Tool-CoT RL + audited alternating training
```

Metrics:

```text
structural precursor Top-1/5/10
reaction-center accuracy
synthon exact match
full precursor exact match
validity
stereochemical correctness where applicable
```

## D2. Electron-flow process accuracy

For examples with supported labels:

```text
source Top-1/Top-k
sink Top-1/Top-k conditioned on source
complete move Top-1
move MRR
step-sequence exact match
partial-order mechanism equivalence
primitive precision/recall
mechanism completeness
```

Do not score unlabeled arrows as wrong.

## D3. Faithfulness and causal dependence

Measure:

- answer–reasoning disagreement for answer-bearing baselines;
- endpoint impossibility of bypass for proof models;
- performance after shuffling or removing tool observations;
- performance when the actor cannot call `inspect_state`;
- performance when tool results are replaced with stale/incorrect states;
- proof-to-endpoint causal intervention tests.

A model that ignores tool feedback should not support the tool-grounded reasoning claim.

## D4. Formal falsification benchmark

Use controlled corruptions and valid controls.

Metrics:

```text
false acceptance rate
false rejection rate
failure-code accuracy
first-failing-edge/action localization
valid-control retention
repair success
new-error introduction
over-edit rate
```

Report by corruption family and proof topology.

## D5. Representation invariance

Test synchronized:

- atom-map permutation;
- state-ID renaming;
- edge serialization;
- commuting independent events;
- random-SMILES traversal where available;
- component ordering;
- equivalent proof variants.

Separate semantic robustness from exact-string equality.

## D6. Compositional generalization

Primary OOD test:

```text
training covers each primitive
validation/test hold out the full primitive composition
```

Report by:

- composition frequency;
- proof length;
- number of changed atoms/bonds;
- ring topology;
- chain/tree/DAG proof structure;
- family and scaffold.

The headline comparison is direct/CoT/edit/proof/tool-CoT under identical held-out compositions.

## D7. Family, scaffold, and temporal OOD

Report separate results for:

```text
reaction-family holdout
product-scaffold holdout
temporal holdout
patent-family holdout when available
```

Do not merge these into one OOD number.

## D8. Forward expert process and closure evaluation

Compare:

```text
ordinary forward product compatibility
forward source/sink process model
process + compatibility
process + compatibility + conditions
```

Metrics:

```text
move accuracy and MRR
target rank
target recovery at k
Brier score
expected calibration error
risk–coverage
uncertainty-error correlation
family-wise false acceptance/false rejection
```

## D9. Selectivity evaluation

Selectivity must be comparative. For each example define a competitor set and report:

```text
target rank
pairwise selectivity accuracy
target-minus-best-competitor margin
regioselectivity accuracy
stereochemical outcome accuracy
SN2/E2 or other family competition accuracy
per-family calibration
```

A target score without competitors is not reported as selectivity.

## D10. Process reward versus endpoint reward

Compare actor RL with:

```text
endpoint exact reward only
formal execution reward only
ordinary forward round-trip reward
verified step-level process rewards
step-level + forward closure
step-level + forward closure + selectivity
```

Report learning curves, reward variance, effective groups, reward hacking, trajectory length, failure distribution, and OOD performance.

## D11. Small-model scale study

Compare approximately:

```text
0.6B tool actor
1–2B tool actor
8B tool actor
8B direct-answer reference
```

Report:

```text
accuracy and reliability
GPU hours
peak memory
tokens generated
tool calls
latency
verified endpoints per compute budget
```

The desired result is not simply that a small model is cheaper; it must retain or improve OOD/reliability relative to a larger direct generator.

## D12. Alternating actor–verifier learning

For each audited round report:

```text
number of disagreements mined
number reviewed
verified-negative rate
label-source distribution
forward-expert calibration before/after
actor performance before/after
new reward-hacking modes
```

Stop alternating updates if the forward expert improves on mined negatives but degrades on the frozen audit set.

## D13. Test-time compute and hypothesis sets

For K in `{1,4,16,64}` report:

- `ExecutePass@K`;
- `EndpointPass@K`;
- executable proof classes@K;
- mechanism compositions@K;
- endpoints@K;
- forward-supported endpoints@K;
- latency and model/tool calls.

Compare repeated independent sampling with state-level branching.

## D14. Multistep planning

Under matched search budgets report:

```text
solved rate
fully verified route rate
formal-invalid expansion rate
forward-supported edge rate
route length
route cost
nodes expanded
reaction-model calls
wall-clock time
route diversity
building-block cost/complexity where available
```

Required conditions:

```text
inverse score only
inverse + formal hard gate
inverse + ordinary forward score
inverse + forward process/selectivity score
inverse + forward expert + uncertainty
```

## D15. Reaction-network analysis

Secondary metrics:

```text
verified branch count
unique endpoints
network depth
cycle count
frontier quality
fraction with external evidence
```

Do not claim reaction discovery without external validation.

## D16. Efficiency and systems analysis

Record:

- training GPU hours and energy proxy where available;
- actor and forward-expert parameter counts;
- rollout throughput;
- executor latency;
- forward-expert latency;
- planner overhead;
- storage for traces/candidate pools;
- failure and retry costs.

## D17. Qualitative and expert analysis

Required case groups:

1. direct answer correct but rationale inconsistent;
2. direct answer wrong, tool actor repaired after failure;
3. multiple executable endpoints with different forward support;
4. selectivity ambiguity and abstention;
5. composition-OOD success;
6. composition-OOD failure;
7. multistep route improved by forward edge ranking;
8. reward-hacking or verifier-error example.

Every displayed case must include the target, proof/moves, executor result, derived precursor, forward evidence, competitors, and provenance.

---

# 7. Ablation matrix

## 7.1 Representation ablations

```text
outcome only
free-form CoT
state-CoT
reaction center/synthon
net edit
complete proof
explicit source/sink tool CoT
```

## 7.2 Verification ablations

```text
no verifier
RDKit validity only
full deterministic executor
ordinary forward product scorer
forward process expert
forward process + selectivity
forward process + selectivity + calibrated abstention
```

## 7.3 Training ablations

```text
SFT only
SFT + DPO
SFT + complete-proof RLVR
Tool-SFT
Tool-SFT + formal process RL
Tool-SFT + formal + forward reward
Tool-SFT + audited alternating training
```

## 7.4 Tool ablations

```text
no inspect_state
no intermediate move execution
no coupled moves
no failure certificate
no abstention
fixed tool script versus autonomous tool choice
```

## 7.5 Forward-expert ablations

```text
model size
architecture
condition channel
random negatives
explicit competitors
process labels
hard-negative audit rounds
```

## 7.6 Search ablations

```text
sampling versus state-level branching
native best-first versus Retro* versus breadth-first
formal hard gate on/off
forward score on/off
uncertainty on/off
candidate-pool size
```

---

# 8. Metric definitions

## 8.1 Structural endpoint

The primary endpoint is the multiset of atom-contributing structural precursor fragments. Solvents, catalysts, salts, and spectators are reported separately.

## 8.2 Execution metrics

```text
FormatPass = parseable proof / generated proofs
ExecutePass = executable proof / generated proofs
ExecutePass@K = targets with at least one executable proof among K / targets
EndpointPass@K = targets with at least one endpoint-compatible proof among K / targets
```

## 8.3 False acceptance and rejection

```text
false acceptance rate = corrupted/invalid examples accepted / invalid examples
false rejection rate = valid controls rejected / valid controls
```

Report both. A conservative verifier with high false rejection is not automatically preferable.

## 8.4 Selectivity

```text
M_selectivity = score(target) - max score(competitor)
```

Report the competitor-generation policy, number of competitors, family, and threshold calibration.

## 8.5 Fully verified route

A route is fully verified only when:

- every edge has an executable proof;
- product/precursor connectivity is consistent;
- every non-stock leaf is expanded;
- every terminal leaf is in the frozen inventory;
- any claimed forward-supported-route metric also passes the stated forward threshold.

```text
fully verified route rate = targets with >=1 fully verified route / targets
```

## 8.6 Calibration and selective risk

Report Brier score, expected calibration error, coverage, error at fixed coverage, and area under the risk–coverage curve. Thresholds are selected on validation and frozen before test evaluation.

---

# 9. Required paper result package

No submission draft is considered complete without the following package.

## 9.1 Main tables

### Table 1 — Data, leakage, and benchmark contract

```text
source revisions
raw/retained/quarantined rows
overlap by key
split sizes
primitive/family coverage
condition/selectivity coverage
```

### Table 2 — Matched one-step retrosynthesis

Rows: all baselines and MechET variants.

Columns:

```text
Top-1/5/10 structural precursor
reaction-center
synthon
format/execute rate
forward closure
selectivity support
calibration/abstention
```

### Table 3 — Process and falsification

```text
source/sink/move accuracy
proof equivalence
false acceptance rate
false rejection rate
failure localization
repair/revision success
```

### Table 4 — OOD and small-model scaling

```text
IID
MechComp-OOD
family OOD
scaffold OOD
temporal OOD
parameter count
GPU hours
latency
```

### Table 5 — Multistep planning

```text
solved rate
fully verified route rate
invalid edges
route length/diversity
nodes/model calls/time
```

## 9.2 Main figures

### Figure 1 — Method

Product -> tool-using inverse electron flow -> deterministic execution -> precursor -> compact forward falsification -> verified single-step/multistep edge.

### Figure 2 — Why tool-grounded CoT matters

Matched direct/free-CoT/state-CoT/net-edit/proof/tool-CoT comparison, plus answer–reasoning disagreement and failure survival.

### Figure 3 — Compositional generalization

Performance versus primitive-composition novelty, proof length, and reaction-center complexity.

### Figure 4 — Forward falsification and selectivity

Target/competitor distributions, calibration, risk–coverage, and ordinary forward scorer versus process expert.

### Figure 5 — Small models and planning

Accuracy/reliability versus model size and compute; multistep route success under matched search budgets.

## 9.3 Supplementary tables

- full hyperparameters and seeds;
- per-family metrics;
- per-corruption metrics;
- per-OOD-split metrics;
- latency and hardware;
- data-field coverage;
- all ablations;
- alternating-round audit counts;
- qualitative case index.

## 9.4 Required qualitative cases

At least two cases from each D17 category, selected under a frozen policy rather than only cherry-picked successes.

---

# 10. Result interpretation and stopping rules

## 10.1 Data stopping rules

Stop and repair the data pipeline if:

- stable-ID intersections differ between matched tasks;
- source revisions or hashes are missing;
- arrow labels are invented or ambiguously decoded;
- benchmark overlap is discovered after model selection;
- train/test patent or temporal lineage cannot be described accurately.

## 10.2 Executor stopping rules

Stop model scaling if:

- valid-control false rejection is unexplained;
- corrupted examples pass due to a verifier bug;
- endpoint derivation can bypass the proof;
- equivalent proofs are scored as different solely because of serialization.

## 10.3 Forward-expert stopping rules

Do not use the forward expert as an RL reward or search score until:

- target-versus-competitor performance exceeds a frozen baseline;
- calibration is measured;
- family-wise false acceptance and false rejection are reported;
- the audit set includes hard actor-generated failures;
- uncertainty correlates with error better than chance.

Stop alternating updates if frozen-audit performance degrades.

## 10.4 Actor-RL stopping rules

Do not scale RL if:

- more than half of groups have zero effective advantage;
- the actor learns tool formatting but not chemical state use;
- proof length grows without endpoint/process improvement;
- formal validity rises while selectivity or OOD performance collapses;
- reward hacking dominates the top-scoring trajectories;
- actor and forward checkpoint lineage is not frozen.

## 10.5 Claim-level stopping rules

### C1 primitive compositionality fails if

MechET does not outperform matched net-edit/proof baselines on composition-held-out splits after controlling for model size and training tokens.

### C2 faithfulness fails if

The actor can obtain endpoint credit without executing a causally relevant proof or if tool observations can be shuffled without affecting performance.

### C3 forward falsification fails if

The compact forward expert does not improve risk–coverage, selection among executable alternatives, or planning-edge reliability over ordinary forward scoring.

### C4 small-model claim fails if

The small tool actor is only cheaper but materially worse than the larger direct baseline on both reliability and OOD generalization.

### C5 planning claim fails if

Forward/formal edge evidence does not improve fully verified route rate under matched candidate and search budgets.

Negative results remain publishable analyses but require a narrower claim.

---

# 11. Reproducibility and artifact contract

Every reported run must include:

```text
git commit
config file and digest
source/benchmark manifests
training-data hash
base-model and tokenizer revision
adapter/checkpoint hashes
executor version
forward-expert checkpoint
random seeds
hardware and software environment
training tokens/steps/GPU hours
inference sampling parameters
tool-call budget
search budget
raw predictions
raw tool traces
raw evaluation outputs
```

## 11.1 Checkpoint lineage

Inverse and forward checkpoints must have separate lineage files. Actor RL records the exact frozen forward checkpoint. Forward hard-negative updates record the exact actor checkpoint used for mining and the independent label source.

## 11.2 Evaluation freezing

Evaluation scripts, thresholds, competitor-generation policy, and benchmark hashes are frozen before the final test run. Any change requires a new result version.

## 11.3 Result directories

```text
outputs/
  manifests/
  data_audit/
  inverse/<method>/<seed>/
  forward/<method>/<seed>/
  hypotheses/<checkpoint>/<budget>/
  falsification/
  ood/
  planning/<planner>/<condition>/<seed>/
  figures/
  paper_tables/
```

## 11.4 CI contract

At minimum, CI must continue to cover:

- proof parser/executor/verifier;
- equivalence and diagnostics;
- forward electron-step execution and compact model smoke tests;
- framework-neutral environment and reward logic;
- TRL dry-run without downloading a model;
- planning adapter and CLI schema;
- documentation contract.

---

# 12. Collaboration work packages

## WP1 — Data and leakage

**Deliverables**

- frozen source/benchmark registry;
- standardized inverse/forward data;
- quarantine reports;
- matched task IDs;
- IID/OOD splits;
- competitor/selectivity benchmark.

**Dependencies:** none. This is the first blocking package.

## WP2 — Formal executor and process benchmark

**Deliverables**

- expanded source/sink error codes;
- corruption suite;
- false acceptance/rejection evaluation;
- invariance tests;
- process metrics.

**Dependencies:** WP1 schemas.

## WP3 — Compact forward expert

**Deliverables**

- trained/calibrated checkpoints;
- process, closure, and selectivity metrics;
- uncertainty/risk–coverage;
- hard actor-generated audit set.

**Dependencies:** WP1 forward data, WP2 formal labels.

## WP4 — Inverse actors and RL

**Deliverables**

- matched direct/CoT/edit/proof baselines;
- small/medium/reference tool actors;
- reward decomposition;
- test-time-compute experiments;
- tool-use causal ablations.

**Dependencies:** WP1, WP2, frozen WP3 checkpoint for RL.

## WP5 — OOD and scientific analysis

**Deliverables**

- MechComp-OOD;
- family/scaffold/temporal results;
- primitive reuse analysis;
- failure taxonomy;
- qualitative cases.

**Dependencies:** WP1–WP4 predictions.

## WP6 — Multistep planning

**Deliverables**

- frozen candidate pool and stock;
- native and Syntheseus results;
- external template baseline;
- fully verified route analysis;
- route cases.

**Dependencies:** stable single-step checkpoints and WP3 scores.

## WP7 — Paper and artifact release

**Deliverables**

- frozen result tables;
- figures;
- appendix;
- model/data cards;
- reproducibility scripts;
- collaboration decision log.

---

# 13. Execution schedule

## Phase 0 — contract freeze

- assign work-package owners;
- freeze target claims and primary metrics;
- freeze data revisions and licenses;
- confirm compute budget;
- confirm supported chemistry scope.

## Phase 1 — data and smoke validation

- build matched data;
- run leakage audit;
- train tiny/small forward smoke model;
- run tool-environment dry-run;
- run all CI and overfit tests.

## Phase 2 — supervised baselines

- train direct, CoT, edit, proof, and tool-SFT conditions;
- train/calibrate forward expert;
- produce the first frozen IID/OOD table.

## Phase 3 — RL and falsification

- run formal process RL;
- add forward closure/selectivity reward;
- mine and audit disagreements;
- complete one alternating round;
- run reward-hacking review.

## Phase 4 — scale and test-time compute

- small/medium/reference actor grid;
- K-hypothesis and state-level branching;
- efficiency and reliability analysis.

## Phase 5 — multistep planning

- freeze offline pools;
- run matched planners;
- add online actor search only as a separate condition;
- curate route cases.

## Phase 6 — paper freeze

- run final test once;
- freeze tables and figures;
- complete limitations and negative-result interpretation;
- archive raw predictions, traces, and manifests.

---

# 14. Immediate next actions

1. Assign an owner and backup to WP1–WP7.
2. Freeze the exact model scale grid and compute budget.
3. Build the matched task intersection and report row counts before any new training.
4. Train and calibrate the compact forward expert on the clean forward split.
5. Build multi-turn tool-SFT traces from executable proofs.
6. Run the 0.6B inverse-actor TRL dry-run and a small overfit experiment.
7. Freeze the falsification, selectivity, and MechComp-OOD benchmarks.
8. Produce a first matched table with no RL to determine whether the representation itself carries signal.
9. Only then start on-policy training and alternating actor–verifier rounds.
10. Keep multistep planning offline and matched until the one-step scientific story is stable.
