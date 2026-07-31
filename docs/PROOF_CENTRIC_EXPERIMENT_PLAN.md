# MechET-GFR authoritative experiment plan

This document is the authoritative experimental contract for MechET. It specifies what data must be built, which models are trained, which losses and algorithms are used, how inference and validation are performed, and which results are required before a scientific claim may be made.

The current method is:

```text
mapped product
  -> autoregressive Proof Actor samples K MECH_PROOF programs
  -> deterministic executor executes or falsifies each program
  -> invalid programs receive failure certificates
  -> Repair Actor edits locally or the Actor resamples
  -> executable programs are deduplicated by partial-order equivalence
  -> surviving proof classes produce precursor endpoints, routes, or network edges
```

The formal executor is deterministic and never trained. A learned model cannot override an execution failure.

---

# 1. Scientific questions and evidence requirements

## 1.1 ICLR-stage question

Does an executable proof representation change retrosynthesis from answer generation into falsifiable program search, and does that change improve faithfulness, robustness, hypothesis coverage, and compositional generalization?

A paper-scale answer requires evidence for all five claims:

| Claim | Required evidence |
|---|---|
| Faithfulness by construction | no independent answer channel; endpoint always executor-derived; answer-bearing baselines measured for answer–reasoning disagreement |
| Formal falsifiability | controlled-corruption FAR/FRR, failure-code accuracy, failing-edge localization, and valid-control retention |
| Representation invariance | atom-map, state-name, edge-serialization, commuting-order, and synchronized random-SMILES controls |
| Set-valued mechanism search | ExecutePass@K, EndpointPass@K, executable proof classes@K, compositions@K, endpoints@K, and survival curves |
| Compositional generalization | MechComp-OOD with seen primitives and unseen full compositions, stratified by proof length and topology |

Top-1 precursor accuracy is required for comparability but cannot, by itself, support the central claim.

## 1.2 NMI-stage question

Can formally executable proof hypotheses be used as a high-recall proposal space for external chemical evidence, multistep planning, reaction-network exploration, and candidate mechanism discovery?

The NMI claim additionally requires:

- a benchmark independent of the FlowER/USPTO training lineage;
- precedent, condition, energy, kinetic, expert, or experimental evidence with provenance;
- proof-carrying multistep routes under matched search budgets;
- prospective or post-cutoff cases where MechET proposes alternatives not copied from training;
- clear separation between formal execution and chemical plausibility.

## 1.3 Claims that are not currently permitted

The current software does not by itself establish:

- low activation barriers;
- favorable kinetics or yields;
- condition or catalyst compatibility;
- unique curved-arrow source-to-sink pairing;
- spin-state or organometallic orbital correctness;
- newly discovered reactions or catalytic cycles.

These claims require external evidence.

---

# 2. Frozen data contract

## 2.1 Canonical data layout

```text
data/
  mechet_sft/
    train.jsonl
    valid.jsonl
    test.jsonl
  mechet_proof_sft/
    train.jsonl
    valid.jsonl
    test.jsonl
  mechet_proof_clean/
    train.jsonl
    valid.jsonl
    test.jsonl
    manifest.json
    quarantine.jsonl
  proof_curriculum/
    equivalence_train.jsonl
    corruptions.jsonl
    preferences.jsonl
    repairs.jsonl
  mechet_proof_mechcomp/
    train.jsonl
    valid.jsonl
    test.jsonl
  benchmarks/
    uspto50k/
    uspto_mit/
    uspto_full/
    external_mechanisms/
  routes/
    targets.jsonl
    building_blocks.smi
```

## 2.2 Required proof-row fields

Every official proof row must retain:

```text
stable reaction id
source dataset and split
mapped product
MECH_PROOF v1 output
executor-derived full precursor
atom-contributing structural precursor
proof topology: chain/tree/DAG
partial-order equivalence digest
mechanism composition digest
reaction-center and leakage keys
patent id/family/date when recoverable
```

Rows without stable IDs cannot enter matched-baseline experiments. Rows without atom maps cannot support structural/environment separation and must be flagged.

## 2.3 Non-negotiable data rules

1. Freeze benchmark files and SHA-256 hashes before training.
2. Remove overlap from training; never filter test after model selection.
3. Write every removed training row to a quarantine file with reasons.
4. Build every matched baseline from the same stable-ID intersection.
5. Use structural precursor endpoints as the primary outcome target.
6. Report solvents, reagents, catalysts, salts, and spectators separately.
7. Never treat an executable alternative endpoint as a negative solely because it differs from dataset gold.
8. Remap product, imports, proof actions, and endpoints together in atom-map controls.

The detailed overlap policy is maintained in [`DATA_LEAKAGE_AND_ICLR_PLAN.md`](DATA_LEAKAGE_AND_ICLR_PLAN.md).

---

# 3. Pipeline A — source data, audit, and proof curriculum

## A0. Build state-annotated cold-start rows

**Input**

```text
FlowER mechanism trajectories
```

**Script**

```bash
python scripts/build_mechet_sft.py \
  --flower-root /path/to/flower_new_dataset \
  --out-dir data/mechet_sft \
  --splits train valid test
```

**Output**

```text
data/mechet_sft/{train,valid,test}.jsonl
```

**Acceptance gate**

- stable source IDs are preserved;
- product, states, edges, and original initial species parse;
- skipped rows and reasons are reported.

## A1. Compile action-only gold proofs

**Script**

```bash
python scripts/build_mechet_proof_sft.py \
  --input-dir data/mechet_sft \
  --output-dir data/mechet_proof_sft \
  --splits train valid test
```

**Output**

```text
data/mechet_proof_sft/{train,valid,test}.jsonl
```

**Acceptance gate**

- every accepted proof executes;
- executor-derived endpoint matches the source trajectory;
- no proof contains an independent `<answer>` channel;
- compilation failure reasons are counted.

## A2. Freeze benchmark lineage and audit overlap

**Inputs**

```text
FlowER-derived proof train
USPTO-50K train/valid/test
USPTO-MIT and USPTO-FULL
patent metadata when recoverable
```

**Script**

```bash
python scripts/audit_reaction_overlap.py \
  --train data/mechet_proof_sft/train.jsonl \
  --benchmark data/benchmarks/uspto50k/test.csv \
  --benchmark-format reaction_table \
  --reaction-field reaction_smiles \
  --out-dir outputs/data_audit/flower_vs_uspto50k_test
```

Repeat for every benchmark split.

**Required audit levels**

```text
exact_full
exact_structural
product
scaffold
reaction_center
proof_composition
patent
```

**Required outputs**

```text
input SHA-256 registry
normalization configuration and digest
overlap matrix
per-row conflicts
product-similarity distribution
patent/temporal coverage report
```

**Stopping gate**

No benchmark may be described as external until source-lineage independence is established. Standard USPTO results remain comparability results even after exact decontamination.

## A3. Build decontaminated training conditions

**Script**

```bash
python scripts/build_decontaminated_dataset.py \
  --train data/mechet_proof_sft/train.jsonl \
  --benchmark data/benchmarks/uspto50k/test.csv \
  --output data/mechet_proof_clean/train.jsonl \
  --manifest data/mechet_proof_clean/manifest.json \
  --policy exact_structural product
```

Build at least three conditions:

```text
exact-clean: exact_structural + product
scaffold-clean: exact_structural + product + scaffold
center-clean: exact_structural + product + scaffold + reaction_center
```

**Required result**

Report original rows, retained rows, removed rows, and removal reason distribution for each condition.

## A4. Build verified equivalence augmentation

**Purpose**

Prevent one arbitrary serialization from becoming the only supervised proof.

**Transformations**

- state-ID renaming;
- textual edge reordering;
- synchronized atom-map permutation;
- future synchronized random-SMILES traversal.

**Script**

```bash
python scripts/build_proof_equivalence_data.py \
  --input data/mechet_proof_clean/train.jsonl \
  --output data/proof_curriculum/equivalence_train.jsonl \
  --variants-per-row 4 \
  --seed 11
```

**Acceptance gate**

Every variant must:

```text
execute successfully
produce the same endpoint
match the reference partial-order proof class
```

## A5. Build controlled corruptions

**Script**

```bash
python scripts/build_proof_corruption_data.py \
  --input data/mechet_proof_clean/train.jsonl \
  --output data/proof_curriculum/corruptions.jsonl \
  --include-valid-controls \
  --seed 17
```

**Corruption families**

```text
LP_WRONG_DELTA
LP_DELETE
BOND_DELETE
BOND_WRONG_DELTA
CHARGE_WRONG_PRECONDITION
CHARGE_DELETE
IMPORT_DELETE
ATOM_MAP_REPLACE
UNREACHABLE_EDGE
PRECURSOR_NOT_DERIVED
EDGE_DEPENDENCY_REWIRE
```

**Acceptance gate**

- intended invalid samples must be rejected by the executor;
- valid controls must execute;
- failure code and first failing edge are stored;
- ambiguous transformations are skipped, not force-labelled.

## A6. Build verifier preferences and repair pairs

**Preference script**

```bash
python scripts/build_proof_preferences.py \
  --corruptions data/proof_curriculum/corruptions.jsonl \
  --output data/proof_curriculum/preferences.jsonl
```

Safe relation:

```text
executable proof > formally invalid proof
```

Do not automatically rank one executable endpoint below another.

**Repair script**

```bash
python scripts/build_proof_repair_data.py \
  --corruptions data/proof_curriculum/corruptions.jsonl \
  --output data/proof_curriculum/repairs.jsonl
```

Repair rows must include product, invalid proof, certificate, corrected proof, changed target lines, failure code, edge, and source ID.

---

# 4. Pipeline B — matched baselines and proof models

## B0. Build matched task variants

**Models requiring matched rows**

```text
Outcome-only
State-CoT
Net-edit
Proof-SFT
Proof-DPO
MechET-GFR
```

**Scripts**

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

**Gate**

All tasks must have identical stable IDs and structural endpoints.

## B1. Matched supervised baselines

Use the same Qwen3-8B base, tokenizer, LoRA capacity, seeds, and assistant-token budget wherever architecture permits.

**Configs**

```text
configs/iclr/outcome_only_sft.yaml
configs/iclr/state_cot_sft.yaml
configs/iclr/net_edit_sft.yaml
configs/proof/proof_actor_sft.yaml
```

**Loss**

```text
L_SFT = -sum_t m_t log p_theta(y_t | x, y_<t)
```

`m_t=1` only for assistant tokens.

**Required training report**

```text
seed
row count
assistant token count
optimizer steps
train loss
validation loss
GPU hours
peak memory
checkpoint hash
```

Train three seeds: `11`, `22`, and `33`.

## B2. Proof Actor SFT

**Data**

```text
data/proof_curriculum/equivalence_train.jsonl
```

**Command**

```bash
python scripts/train_mechet_sft.py \
  --config configs/proof/proof_actor_sft.yaml
```

**Scientific role**

Measures the contribution of proof representation plus verified equivalence augmentation before verifier preference or on-policy learning.

## B3. Verifier-DPO

**Data**

```text
data/proof_curriculum/preferences.jsonl
```

**Command**

```bash
python scripts/train_proof_dpo.py \
  --config configs/proof/proof_dpo.yaml --dry-run

python scripts/train_proof_dpo.py \
  --config configs/proof/proof_dpo.yaml
```

**Loss**

```text
L_DPO = -log sigmoid(beta * ((log pi(chosen)-log pi(rejected))
                              - reference_margin))
```

The frozen initial Proof-SFT adapter supplies the cached reference margin.

**Required ablation**

Compare Proof-SFT and Proof-DPO on:

```text
parse rate
execute rate
FAR/FRR
EndpointPass@K
effective RLVR group rate
```

The purpose of DPO is to reduce obvious formal failures before expensive on-policy sampling.

## B4. Proof-set RLVR

Train two separate policies or adapters.

### Accuracy reward mode

```text
parse reward
+ execute reward
+ structural endpoint exact reward
+ weak composition reward
```

### Hypothesis reward mode

```text
parse reward
+ execute reward
+ bonus for a new executable partial-order class within the group
- penalty for duplicate executable classes
```

Invalid candidates receive no novelty or diversity reward.

**Single-process smoke**

```bash
python scripts/train_iclr_proof_rlvr.py \
  --config configs/proof/proof_rlvr_accuracy.yaml --dry-run
```

**Staged training**

```bash
# rollout
python scripts/train_proof_rlvr_distributed.py \
  --config configs/proof/proof_rlvr_hypothesis.yaml \
  --mode rollout \
  --input data/mechet_proof_clean/train.jsonl \
  --output outputs/proof/rlvr_iter0/rollouts.jsonl \
  --adapter outputs/proof/actor_dpo/adapter

# deterministic scoring
python scripts/train_proof_rlvr_distributed.py \
  --config configs/proof/proof_rlvr_hypothesis.yaml \
  --mode score \
  --input outputs/proof/rlvr_iter0/rollouts.jsonl \
  --output outputs/proof/rlvr_iter0/scored.jsonl

# learner update
python scripts/train_proof_rlvr_distributed.py \
  --config configs/proof/proof_rlvr_hypothesis.yaml \
  --mode train \
  --input outputs/proof/rlvr_iter0/scored.jsonl \
  --output outputs/proof/rlvr_iter0/learner \
  --adapter outputs/proof/actor_dpo/adapter
```

**Objective**

```text
A_i = r_i - mean_{j != i}(r_j)
L_RLVR = -mean_i A_i * mean_t log p_theta(y_i,t | x, y_i,<t)
```

This is RLOO/group-relative REINFORCE, not clipped PPO or full GRPO.

**Stopping gates before full RLVR**

- Proof-DPO execute rate must be high enough to create within-group reward variation;
- at least 50% of rollout groups should have non-zero advantages;
- invalid candidates must never receive diversity bonuses;
- rollout and learner checkpoint hashes must match.

## B5. Certificate-conditioned Repair Actor

**Data**

```text
data/proof_curriculum/repairs.jsonl
```

**Command**

```bash
python scripts/train_proof_repair.py \
  --config configs/proof/proof_repair.yaml --dry-run

python scripts/train_proof_repair.py \
  --config configs/proof/proof_repair.yaml
```

**Loss**

```text
L_repair = -sum_t w_t log p_theta(y*_t | x, y*_<t)

w_t = 4 on tokens belonging to corrected target lines
w_t = 1 on other assistant proof tokens
```

The Repair Actor is a separate adapter so actor generation and local correction can be audited independently.

---

# 5. Pipeline C — inference modes

## C1. Single-proof inference

Use only for literature-compatible Top-1 and Top-k comparisons.

```bash
python scripts/infer_mechet_proof.py ...
python scripts/eval_mechet_proof_generations.py ...
```

## C2. Hypothesis-set inference

For each target, sample K complete proof programs from the same actor.

```bash
python scripts/infer_proof_hypotheses.py \
  --data data/mechet_proof_clean/test.jsonl \
  --adapter outputs/proof/actor_dpo/adapter \
  --samples-per-target 64 \
  --temperature 0.9 \
  --top-p 0.95 \
  --out outputs/proof/hypotheses.jsonl
```

**Pipeline**

```text
sample K raw proof strings
 -> parse
 -> execute
 -> derive structural precursor
 -> diagnose failures
 -> deduplicate executable candidates by partial-order proof class
 -> group by endpoint
 -> rank survivors
```

**Ranking order**

1. formal execution;
2. task constraint or endpoint compatibility;
3. external plausibility evidence;
4. model likelihood;
5. novelty among executable candidates.

Soft scores cannot rescue a formally invalid proof.

**Sampling study**

Run at least:

```text
K = 1, 4, 16, 64, 128
temperature = 0.7, 0.9, 1.1
```

Report performance versus sampling cost and unique executable proof classes. The preferred operational stopping criterion is either a maximum sample budget or a target number of unique executable proof classes.

## C3. Generate–Falsify–Repair inference

```bash
python scripts/infer_proof_gfr.py \
  --data data/mechet_proof_clean/test.jsonl \
  --actor-adapter outputs/proof/actor_dpo/adapter \
  --repair-adapter outputs/proof/repair/adapter \
  --samples-per-target 16 \
  --max-repairs 2 \
  --out outputs/proof/gfr.jsonl
```

Each candidate stores the proof, execution verdict, certificate, repair history, final endpoint, and checkpoint lineage.

Repair is bounded to two rounds in the primary experiment. Unrepaired candidates are discarded or resampled.

## C4. External plausibility scoring

```bash
python scripts/score_proof_plausibility.py \
  --predictions outputs/proof/hypotheses.jsonl \
  --oracle my_project.oracle:score \
  --out outputs/proof/hypotheses_plausibility.jsonl
```

The oracle may return typed evidence:

```text
precedent
conditions
energy
kinetics
expert
experiment
```

Every value requires source metadata. Missing evidence remains missing.

## C5. Proof-carrying route search

The first implementation uses an offline candidate pool and best-first search:

```bash
python scripts/search_proof_routes.py \
  --targets data/routes/targets.jsonl \
  --candidate-pool outputs/routes/candidate_pool.jsonl \
  --building-blocks data/routes/building_blocks.smi \
  --out outputs/routes/routes.jsonl
```

A route edge enters the frontier only when its proof executes. Online vLLM generation may later replace the candidate pool while retaining the same route verifier.

---

# 6. Pipeline D — validation experiments

## D0. Data integrity and leakage table

**Required table**

Rows:

```text
FlowER train -> USPTO-50K train
FlowER train -> USPTO-50K valid
FlowER train -> USPTO-50K test
FlowER train -> USPTO-MIT test
FlowER train -> USPTO-FULL test
```

Columns:

```text
exact_full
exact_structural
product
scaffold
reaction_center
proof_composition
patent
```

Also report removal counts and final train sizes for each clean condition.

## D1. Matched endpoint and proof-model comparison

**Models**

```text
Outcome-only
State-CoT
Net-edit
Proof-SFT
Proof-DPO
Proof-RLVR-accuracy
Proof-RLVR-hypothesis
MechET-GFR
```

**Datasets**

```text
FlowER internal test
USPTO-50K standard
USPTO exact-clean training condition
USPTO scaffold-clean training condition
USPTO center-clean training condition
MechComp-OOD
```

**Required metrics**

```text
structural precursor Top-1/Top-5/Top-10
format rate
execute rate
ExecutePass@K
EndpointPass@K
unique executable proof classes@K
unique endpoints@K
```

Report mean, standard deviation, and every seed.

## D2. Formal falsification benchmark

```bash
python scripts/eval_proof_falsification.py \
  --data data/proof_curriculum/corruptions.jsonl \
  --out outputs/eval/falsification.json
```

**Required metrics**

```text
false acceptance rate (FAR)
false rejection rate (FRR)
failure-code accuracy
first-failing-edge accuracy
per-corruption rejection
valid-control retention
```

A verifier that rejects every candidate is not useful; FAR and FRR must be reported together.

## D3. Faithfulness and causal intervention

For answer-bearing baselines, measure:

```text
answer correct / reasoning invalid
answer invalid / reasoning valid
answer-state disagreement
```

For MechET, perform controlled proof interventions:

```text
delete BOND
change BOND delta
change LP delta
delete IMPORT
change atom map
change charge precondition
rewire dependency
create DAG join conflict
```

Required result: changing a causally necessary proof operation should change the executor endpoint or cause rejection. Commuting nuisance transformations should preserve the proof class.

## D4. Representation invariance

Evaluate:

```text
atom-map permutation
state-ID renaming
textual edge reordering
commuting independent events
synchronized random product SMILES and proof remapping
fragment ordering
```

**Metrics**

```text
execution invariance
endpoint invariance
proof-class invariance
model log-probability variance
accuracy drop under perturbation
```

Dependent-edge perturbations serve as negative controls and should not be treated as invariances.

## D5. Hypothesis-set quality

```bash
python scripts/eval_proof_hypotheses.py \
  --predictions outputs/proof/hypotheses.jsonl \
  --k 1 4 16 64 128 \
  --out outputs/eval/hypotheses.json
```

**Required metrics**

```text
parse survival
execution survival
endpoint-compatible survival
ExecutePass@K
EndpointPass@K
unique executable proof classes@K
unique compositions@K
unique structural endpoints@K
valid hypotheses per 1,000 generated tokens
```

**Required figure**

A hypothesis survival curve:

```text
generated -> parseable -> executable -> endpoint-compatible
          -> plausibility-supported -> energy/experiment-supported
```

ICLR requires the first four layers. NMI requires external layers.

## D6. Repair quality

```bash
python scripts/eval_proof_repair.py \
  --predictions outputs/proof/gfr.jsonl \
  --out outputs/eval/repair.json
```

**Required metrics**

```text
repair success@1
repair success@2
same-endpoint retention
over-edit rate
new-error introduction rate
mean changed lines
failure-code transition matrix
```

Compare deterministic LP-only repair, learned Repair Actor, full regeneration, and no repair.

## D7. Compositional generalization

Build MechComp-OOD:

```bash
python scripts/build_mechcomp_ood.py \
  --input data/mechet_proof_clean/train.jsonl \
  --output-dir data/mechet_proof_mechcomp \
  --test-fraction 0.10 \
  --valid-fraction 0.10 \
  --min-train-primitive-count 5
```

Primary condition:

```text
all held-out primitives seen in train
complete held-out composition absent from train
```

Report by:

```text
proof length
number of elementary edges
chain/tree/DAG topology
number of imported species
reaction class where available
```

A primitive-unseen split is a harder secondary analysis and should not be conflated with composition OOD.

## D8. Data efficiency and compute

Train with:

```text
1%, 5%, 10%, 25%, 100% of clean training rows
```

Report:

```text
endpoint metrics
execute rate
MechComp-OOD
FAR/FRR
assistant tokens
GPU hours
peak memory
inference latency
executor overhead
repair overhead
valid proof classes per compute budget
```

The proof representation should show the largest advantage in low-data, OOD, or constrained-validity regimes; a gain only from longer compute is insufficient.

## D9. Multistep route pilot

```bash
python scripts/eval_proof_routes.py \
  --routes outputs/routes/routes.jsonl \
  --out outputs/eval/routes.json
```

**Primary metric**

```text
fully verified route rate under matched search budget
```

**Supporting metrics**

```text
solved target rate
invalid expansion rate
search nodes
model calls
executor calls
route length
route diversity
building-block success
wall time
```

Compare:

```text
endpoint-only expansion
proof-generated expansion without formal gate
proof-generated expansion with executor gate
proof-generated expansion with GFR
```

---

# 7. Required paper result package

## 7.1 Main tables

### Table 1 — Data lineage and clean splits

Must include overlap rates, train removals, retained sizes, and whether patent/temporal disjointness was verifiable.

### Table 2 — Matched single-step models

Rows: Outcome-only, State-CoT, Net-edit, Proof-SFT, Proof-DPO, Proof-RLVR, MechET-GFR.

Columns:

```text
Top-1/5/10 structural precursor
execute rate
ExecutePass@16/64
EndpointPass@16/64
proof classes@64
GPU hours
```

### Table 3 — Falsification and repair

Columns:

```text
FAR
FRR
failure-code accuracy
edge localization
repair@1
repair@2
over-edit
```

### Table 4 — OOD and invariance

Conditions:

```text
standard
exact-clean
scaffold-clean
center-clean
MechComp-OOD
map permutation
random SMILES
```

### Table 5 — Multistep pilot

Matched search budget, verified route rate, solved rate, invalid expansions, nodes, and runtime.

## 7.2 Main figures

### Figure 1 — Method

Show:

```text
product -> K proof programs -> executor -> certificates/repair
        -> equivalence classes -> endpoints/routes/network
```

The figure must show that `BOND`, `LP`, and `CHARGE` are local executable primitives, not whole-reaction templates.

### Figure 2 — Proof quality funnel

Generated, parseable, executable, endpoint-compatible, and unique proof classes as K increases.

### Figure 3 — Falsification map

Per-corruption rejection, FAR/FRR, failure localization, and repair flow.

### Figure 4 — Equivalence and invariance

Nuisance transformations should collapse to the same proof class; dependent interventions should alter or invalidate the proof.

### Figure 5 — Compositional OOD

Performance versus proof length/topology on seen-primitives/unseen-composition splits.

### Figure 6 — Proof-carrying route pilot

Matched-budget route search showing fewer invalid expansions and more fully verified routes.

## 7.3 Required qualitative cases

At least:

- one case with multiple serializations of the same proof class;
- one case with the same endpoint but two distinct executable proof classes;
- one case with different executable precursor endpoints;
- one invalid proof correctly localized and repaired;
- one MechComp-OOD composition;
- one multistep route where formal gating removes a hallucinated branch;
- for NMI, one externally supported alternative mechanism or path.

---

# 8. Result interpretation and stopping rules

## 8.1 Minimum success criteria for the ICLR story

The method is not ready for the central claim unless:

1. proof generation reaches a usable execute rate after SFT/DPO;
2. FAR is low without an excessive FRR;
3. atom-map and serialization perturbations preserve execution and proof class;
4. MechET improves either compositional OOD, formal validity, or hypothesis coverage under a matched compute budget;
5. GFR improves valid proof yield without large over-edit or endpoint corruption;
6. standard USPTO results are not presented as independent external validation.

## 8.2 Failure interpretations

- **High Top-1, low execute rate:** model is exploiting endpoint priors; proof learning has not succeeded.
- **High execute rate, low endpoint coverage:** proof grammar is learned, but proposal policy is weak.
- **High FAR:** verifier or corruption benchmark is insufficient.
- **High FRR:** formal language or equivalence rules are over-restrictive.
- **No MechComp-OOD advantage:** the model may be memorizing complete mechanism patterns.
- **Diversity only at high temperature with low execution:** apparent novelty is invalid-string noise.
- **Repair rewrites whole proofs:** certificate-conditioned local correction has failed.
- **Route solved rate rises but verified route rate does not:** planner is adding unverified hallucinated expansions.

## 8.3 NMI advancement gate

Proceed to large external mechanism and network studies only after the ICLR formal layer is stable. External discovery claims require:

```text
training-neighbor audit
formal execution
independent plausibility evidence
energy or kinetic validation where relevant
expert or experimental review
```

---

# 9. Reaction-network and catalytic-cycle extension

## 9.1 Executable reaction hypergraph

```bash
python scripts/explore_reaction_network.py \
  --predictions outputs/proof/hypotheses.jsonl \
  --out outputs/network/network.json
```

Nodes are molecular species. Hyperedges are executable proof programs. Optional edge attributes include energy, uncertainty, rate, conditions, and evidence.

## 9.2 Frontier selection

```bash
python scripts/score_network_frontier.py \
  --network outputs/network/network.json \
  --candidates outputs/network/candidates.jsonl \
  --out outputs/network/frontier.json
```

Frontier scores prioritize external calculations or experiments. They do not prove feasibility.

## 9.3 Catalytic-cycle formal checks

```bash
python scripts/eval_catalytic_cycles.py \
  --cycles data/cycles/cycles.jsonl \
  --out outputs/eval/cycles.json
```

Current checks:

```text
proof execution
catalyst continuity
catalyst regeneration
optional oxidation-state closure
declared net-reaction ledger
```

Still required externally:

```text
coordination chemistry
spin states
ligand exchange
transition states
barriers
microkinetics
deactivation pathways
experimental evidence
```

---

# 10. Reproducibility and artifact contract

Every official run must save:

```text
command
configuration file
git commit
base-model identifier and hash
adapter input and output hashes
dataset paths and SHA-256 hashes
normalization digest
random seed
hardware
software versions
training-token count
runtime and GPU hours
raw predictions
verifier outputs
summary metrics
```

Recommended layout:

```text
outputs/
  data_audit/<run>/
  proof/<stage>/<seed>/
  eval/<benchmark>/<model>/<seed>/
  routes/<run>/
  network/<run>/
```

No result enters a paper table without a frozen manifest and a rerunnable command.

---

# 11. Ordered execution schedule

1. Freeze source data and benchmark hashes.
2. Run FlowER–USPTO overlap audits.
3. Build exact-, scaffold-, and center-clean training sets.
4. Compile and verify gold proofs.
5. Build equivalence, corruption, preference, and repair corpora.
6. Train matched Outcome-only, State-CoT, Net-edit, and Proof-SFT baselines.
7. Train Verifier-DPO and evaluate execute rate/FAR/FRR.
8. Train the Repair Actor and run bounded GFR evaluation.
9. Run accuracy-mode RLVR only when reward groups are informative.
10. Run hypothesis-mode RLVR only when invalid candidates no longer dominate sampling.
11. Generate K-set predictions and complete survival, Pass@K, and diversity analysis.
12. Run invariance, causal-intervention, clean-split, and MechComp-OOD experiments.
13. Run the matched-budget multistep pilot.
14. Freeze the ICLR result package.
15. Only then add external mechanisms, plausibility evidence, network exploration, catalytic cycles, and prospective cases for NMI.

The machine-readable stage and checkpoint lineage is maintained in `configs/proof/proof_pipeline.yaml`.
