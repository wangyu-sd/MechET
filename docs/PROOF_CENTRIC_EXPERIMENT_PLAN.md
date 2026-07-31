# MechET-GFR: proof-centric experiment and implementation plan

This is the authoritative execution plan for MechET after the transition from
single-answer retrosynthesis to falsifiable mechanism-hypothesis search.

```text
Generate proof hypotheses
  -> execute and falsify
  -> diagnose failures
  -> repair locally or resample
  -> deduplicate by partial-order equivalence
  -> return surviving endpoints, routes, or network edges
```

The formal executor is deterministic and is never trained. Learned models may
generate, repair, rank, or search proofs, but they cannot override executor
failure.

---

## 1. Scientific claims by stage

### ICLR claim

Executable inverse electron-flow proofs make retrosynthetic reasoning:

- faithful by construction, because there is no independent answer channel;
- falsifiable, because invalid transitions yield deterministic counterexamples;
- invariant to arbitrary state names, atom-map labels, and commuting order;
- compositionally generalizable over reusable mechanism primitives;
- searchable as sets of proof hypotheses rather than only Top-1 strings.

### NMI extension

Surviving formally executable hypotheses can be connected to external evidence:
precedent, conditions, energy calculations, kinetic models, expert judgement, and
experiments. Formal execution remains separate from chemical plausibility.

---

## 2. Frozen data contract

Official runs use data that have passed the FlowER--USPTO overlap audit and
training-set quarantine pipeline.

```text
data/
  mechet_proof_clean/
    train.jsonl
    valid.jsonl
    test.jsonl
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
    external_mechanisms/
```

Every proof row should contain:

```text
stable id
mapped product
MECH_PROOF v1 assistant output
full executor-derived precursor
atom-contributing structural precursor
partial-order equivalence digest
mechanism composition digest
source/split metadata
leakage keys
```

Hard rules:

1. benchmark hashes are frozen before training;
2. conflicts are removed from training, not from test;
3. all removed training rows are retained in quarantine files;
4. executable alternative endpoints are not automatically negative;
5. atom-map perturbations remap product, imports, actions, and endpoints together.

---

# Pipeline A -- proof curriculum construction

## A0. Gold proof compilation

**Input**

```text
data/mechet_sft/{train,valid,test}.jsonl
```

**Script**

```bash
python scripts/build_mechet_proof_sft.py \
  --input-dir data/mechet_sft \
  --output-dir data/mechet_proof_sft \
  --splits train valid test
```

**Gate**

- every accepted proof executes;
- executor endpoint matches the state trajectory;
- failure reasons are reported in the manifest.

## A1. Verified equivalence augmentation

**Purpose**

Prevent one arbitrary serialization from becoming the only supervised target.
Variants may rename states, reorder serialized edges, and permute every atom map.
A variant is written only when it executes and is partial-order equivalent to the
reference proof.

**Script**

```bash
python scripts/build_proof_equivalence_data.py \
  --input data/mechet_proof_clean/train.jsonl \
  --output data/proof_curriculum/equivalence_train.jsonl \
  --variants-per-row 4 \
  --seed 11
```

**Implementation**

```text
src/mechet/proof_variants.py
scripts/build_proof_equivalence_data.py
```

**Output**

Each row keeps the source reaction ID and equivalence digest. The actor SFT file
contains multiple verified representatives of the same proof class.

## A2. Controlled corruption generation

**Purpose**

Build a falsification benchmark and verifier-grounded negatives. Each corruption
changes one controlled aspect of a valid proof and is executed before labelling.

Supported corruption families include:

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

Valid controls can include state renaming and textual edge reordering.

**Script**

```bash
python scripts/build_proof_corruption_data.py \
  --input data/mechet_proof_clean/train.jsonl \
  --output data/proof_curriculum/corruptions.jsonl \
  --include-valid-controls \
  --seed 17
```

**Gate**

- invalid corruption: executor must reject it;
- valid control: executor must accept it;
- failure code and first failing edge are stored;
- ambiguous transformations are skipped rather than mislabelled.

## A3. Verifier preference pairs

**Policy**

Safe pairs are:

```text
executable proof > formally invalid proof
```

The following pair is not created automatically:

```text
gold endpoint > different executable endpoint
```

because the second endpoint may be a useful alternative synthesis hypothesis.

**Script**

```bash
python scripts/build_proof_preferences.py \
  --corruptions data/proof_curriculum/corruptions.jsonl \
  --output data/proof_curriculum/preferences.jsonl
```

## A4. Certificate-conditioned repair rows

**Task**

```text
product + invalid proof + failure certificate -> corrected proof
```

**Script**

```bash
python scripts/build_proof_repair_data.py \
  --corruptions data/proof_curriculum/corruptions.jsonl \
  --output data/proof_curriculum/repairs.jsonl
```

The metadata record changed lines, failure code, failing edge, and source proof.

---

# Pipeline B -- model training

All headline comparisons use Qwen3-8B, matched tokenizer, matched LoRA capacity,
three seeds, and auditable assistant-token budgets.

## B1. Proof actor SFT

**Data**

```text
data/proof_curriculum/equivalence_train.jsonl
```

**Config**

```text
configs/proof/proof_actor_sft.yaml
```

**Trainer**

```bash
python scripts/train_mechet_sft.py \
  --config configs/proof/proof_actor_sft.yaml
```

**Loss**

```text
L_SFT = - sum_t m_t log p_theta(y_t | x, y_<t)
```

`m_t` is one only on assistant proof tokens. Equivalence variants are sampled or
flattened from the same source class. Main reports should match assistant-token
budgets, not only epoch counts.

**Output**

```text
outputs/proof/actor_sft/adapter
```

## B2. Verifier-DPO

**Data**

```text
data/proof_curriculum/preferences.jsonl
```

**Config**

```text
configs/proof/proof_dpo.yaml
```

**Trainer**

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

The reference margin is cached from the frozen initial SFT adapter before any
updates, avoiding a second full model copy.

**Output**

```text
outputs/proof/actor_dpo/adapter
```

## B3. Proof-set RLVR

Two reward modes are trained separately.

### Accuracy mode

```text
parse + execute + structural endpoint exact + weak composition match
```

### Hypothesis mode

```text
parse + execute + new executable equivalence class bonus
        - duplicate executable class penalty
```

Invalid proofs receive no diversity or novelty bonus.

**Configs**

```text
configs/proof/proof_rlvr_accuracy.yaml
configs/proof/proof_rlvr_hypothesis.yaml
```

**Single-process smoke trainer**

```bash
python scripts/train_iclr_proof_rlvr.py \
  --config configs/proof/proof_rlvr_accuracy.yaml --dry-run
```

**Staged distributed trainer**

```bash
# GPU rollout workers
python scripts/train_proof_rlvr_distributed.py \
  --config configs/proof/proof_rlvr_hypothesis.yaml \
  --mode rollout \
  --input data/mechet_proof_clean/train.jsonl \
  --output outputs/proof/rlvr_iter0/rollouts.jsonl \
  --adapter outputs/proof/actor_dpo/adapter

# CPU verifier pool
python scripts/train_proof_rlvr_distributed.py \
  --config configs/proof/proof_rlvr_hypothesis.yaml \
  --mode score \
  --input outputs/proof/rlvr_iter0/rollouts.jsonl \
  --output outputs/proof/rlvr_iter0/scored.jsonl

# learner GPU
python scripts/train_proof_rlvr_distributed.py \
  --config configs/proof/proof_rlvr_hypothesis.yaml \
  --mode train \
  --input outputs/proof/rlvr_iter0/scored.jsonl \
  --output outputs/proof/rlvr_iter0/learner \
  --adapter outputs/proof/actor_dpo/adapter
```

The rollout file records the policy adapter. The learner refuses groups produced
by a different checkpoint, preventing silent off-policy mixing.

**Objective**

```text
A_i = r_i - mean_{j != i}(r_j)
L_RLVR = -mean_i A_i * mean_t log p_theta(y_i,t | x, y_i,<t)
```

This is RLOO/group-relative REINFORCE. It is not described as clipped PPO or a
full reference-KL GRPO implementation.

## B4. Repair adapter

**Data**

```text
data/proof_curriculum/repairs.jsonl
```

**Config**

```text
configs/proof/proof_repair.yaml
```

**Trainer**

```bash
python scripts/train_proof_repair.py \
  --config configs/proof/proof_repair.yaml --dry-run

python scripts/train_proof_repair.py \
  --config configs/proof/proof_repair.yaml
```

**Loss**

```text
L_repair = - sum_t w_t log p_theta(y*_t | x, y*_<t)

w_t = 4 for tokens on changed lines
w_t = 1 for other assistant proof tokens
```

**Output**

```text
outputs/proof/repair/adapter
```

---

# Pipeline C -- inference

## C1. Single-proof inference

Used only for literature-compatible Top-1/Top-k comparisons.

```bash
python scripts/infer_mechet_proof.py ...
python scripts/eval_mechet_proof_generations.py ...
```

## C2. Hypothesis-set inference

```bash
python scripts/infer_proof_hypotheses.py \
  --data data/mechet_proof_clean/test.jsonl \
  --adapter outputs/proof/rlvr_hypothesis/adapter \
  --samples-per-target 64 \
  --temperature 0.9 \
  --out outputs/proof/hypotheses.jsonl
```

Pipeline:

```text
sample K proofs
-> execute all
-> compute structural endpoints
-> partial-order equivalence deduplication
-> endpoint grouping
-> lexicographic ranking
```

Ranking order is:

1. formal execution;
2. endpoint/constraint compatibility;
3. external plausibility evidence;
4. model likelihood;
5. novelty within executable candidates.

## C3. Generate--Falsify--Repair inference

```bash
python scripts/infer_proof_gfr.py \
  --data data/mechet_proof_clean/test.jsonl \
  --actor-adapter outputs/proof/rlvr_hypothesis/adapter \
  --repair-adapter outputs/proof/repair/adapter \
  --samples-per-target 16 \
  --max-repairs 2 \
  --out outputs/proof/gfr.jsonl
```

Each failed proof receives a deterministic certificate. The repair model may edit
it for at most two rounds. Remaining failures are discarded or resampled rather
than repaired indefinitely.

## C4. External plausibility evidence

```bash
python scripts/score_proof_plausibility.py \
  --predictions outputs/proof/hypotheses.jsonl \
  --oracle my_project.oracle:score \
  --out outputs/proof/hypotheses_plausibility.jsonl
```

The oracle returns typed evidence for precedent, conditions, energy, or expert
support. Missing evidence remains missing. Formal failure is always a hard gate.

---

# Pipeline D -- validation

## D1. Falsification benchmark

```bash
python scripts/eval_proof_falsification.py \
  --data data/proof_curriculum/corruptions.jsonl \
  --out outputs/eval/falsification.json
```

Metrics:

```text
false acceptance rate
false rejection rate
failure-code accuracy
first-failing-edge accuracy
per-corruption execution accuracy
```

Both FAR and FRR are required: a verifier that rejects everything is not useful.

## D2. Equivalence and invariance

Evaluate:

```text
atom-map permutation
state renaming
random edge serialization
commuting-order changes
random product SMILES with synchronized remapping
dependent-edge perturbations
```

Expected behaviour:

```text
nuisance transformations: same executable proof class
mechanistically dependent perturbations: changed result or rejection
```

## D3. Hypothesis-set quality

```bash
python scripts/eval_proof_hypotheses.py \
  --predictions outputs/proof/hypotheses.jsonl \
  --k 1 4 16 64 \
  --out outputs/eval/hypotheses.json
```

Metrics:

```text
parse/execute/endpoint survival
ExecPass@K
EndpointPass@K
unique executable proof classes@K
unique mechanism compositions@K
unique structural endpoints@K
valid hypotheses per sampling budget
```

## D4. Repair quality

```bash
python scripts/eval_proof_repair.py \
  --predictions outputs/proof/gfr.jsonl \
  --out outputs/eval/repair.json
```

Metrics:

```text
repair success@1
repair success@2
over-edit rate
new-error introduction rate
mean changed lines
failure transition matrix
```

## D5. Compositional OOD and leakage-clean evaluation

Retain the existing frozen benchmarks:

```text
USPTO standard
exact-clean
scaffold-clean
reaction-center-clean
MechComp-OOD primitive-seen/composition-unseen
```

Top-1 is reported for comparability, but the proof-centric headline metrics are
execution, Pass@K, proof-class coverage, falsification, and map robustness.

---

# Pipeline E -- proof-carrying multistep search

The initial implementation uses an offline candidate pool and best-first search.
It cleanly separates route verification from online model serving.

```bash
python scripts/search_proof_routes.py \
  --targets data/routes/targets.jsonl \
  --candidate-pool outputs/routes/candidate_pool.jsonl \
  --building-blocks data/routes/building_blocks.smi \
  --out outputs/routes/routes.jsonl

python scripts/eval_proof_routes.py \
  --routes outputs/routes/routes.jsonl \
  --out outputs/eval/routes.json
```

A route edge enters the frontier only when its proof executes. Route verification
checks edge execution, product/precursor connectivity, cycle rejection, and
building-block leaves.

Headline metric:

```text
fully verified route rate under matched search budget
```

Supporting metrics include solved target rate, invalid expansions, search nodes,
model calls, route length, route diversity, and wall time.

---

# Pipeline F -- reaction networks and catalytic cycles

## F1. Executable reaction hypergraph

```bash
python scripts/explore_reaction_network.py \
  --predictions outputs/proof/hypotheses.jsonl \
  --out outputs/network/network.json
```

Nodes are molecular species. Hyperedges are executable proof programs. Energy,
rate, conditions, uncertainty, and evidence are optional external attributes.

## F2. Frontier selection

```bash
python scripts/score_network_frontier.py \
  --network outputs/network/network.json \
  --candidates outputs/network/candidates.jsonl \
  --out outputs/network/frontier.json
```

The frontier score prioritizes candidates for external evaluation using novelty,
uncertainty, plausibility, and optional supplied energy. It is not a feasibility
proof.

## F3. External benchmark normalization

```bash
python scripts/build_external_mechanism_benchmark.py \
  --input data/raw_external/reactions.csv \
  --input-format csv \
  --source-name post_cutoff_literature \
  --reaction-field reaction_smiles \
  --cutoff-date 2026-01-01 \
  --require-post-cutoff \
  --output data/benchmarks/external_mechanisms/test.jsonl
```

Rows without verified publication dates cannot support temporal-disjoint claims.

## F4. Formal catalytic-cycle validation

Cycle proposals use a JSON schema with executable proof steps and explicit
catalyst states.

```bash
python scripts/eval_catalytic_cycles.py \
  --cycles outputs/cycles/proposals.jsonl \
  --out outputs/eval/cycles.json
```

The formal verifier checks:

```text
proof execution for every step
catalyst-state continuity
catalyst regeneration
optional oxidation-state closure
declared net-reaction ledger
```

It does not validate barriers, kinetics, spin crossings, solvent/ligand exchange,
or experimental feasibility. Those require external quantum-chemistry,
microkinetic, and experimental oracles.

---

## 3. Model matrix

### Matched representation baselines

```text
Outcome-only
State-CoT
Net-edit
Proof-SFT
```

### Proof-learning ablations

```text
Proof-SFT
Proof-SFT + verified equivalence augmentation
Proof-DPO
Proof-RLVR accuracy
Proof-RLVR hypothesis
MechET-GFR = actor + formal falsifier + repair adapter
```

All models use the same frozen row IDs for matched comparisons.

---

## 4. ICLR result structure

1. **Faithfulness by construction** -- proof is the only path to the precursor.
2. **Formal falsification** -- controlled errors, FAR/FRR, and localization.
3. **Equivalence and nuisance invariance** -- map/state/order controls.
4. **Hypothesis-set generation** -- survival curves, Pass@K, and proof classes.
5. **Mechanism compositional generalization** -- MechComp-OOD.
6. **Leakage-clean evaluation and multistep pilot** -- standard versus clean data
   and fully verified routes.

The ICLR paper does not require DFT, wet-lab validation, or full catalytic-cycle
discovery.

---

## 5. NMI extension gates

NMI-stage claims require at least:

- a genuinely external mechanism benchmark;
- precedent/condition evidence with source attribution;
- energy or kinetic validation for proposed novel pathways;
- proof-guided reaction-network exploration;
- expert review or experimental validation of selected hypotheses;
- clear separation of formal validity from chemical plausibility.

A pathway is called a discovery only when it is novel relative to the frozen
training corpus and survives formal, chemical, energetic, and expert/experimental
checks. Formal executability alone is not a discovery claim.

---

## 6. Recommended execution order

```text
0. complete leakage audit and clean-data freeze
1. build equivalence and corruption corpora
2. train Proof-SFT
3. train Verifier-DPO
4. measure rollout execute rate
5. train repair adapter
6. run GFR and falsification benchmarks
7. train accuracy and hypothesis RLVR
8. run hypothesis-set and MechComp-OOD evaluation
9. run multistep route pilot
10. begin external evidence/network/cycle work
```

Do not start large-scale RLVR until Proof-DPO has a non-trivial execute rate and
at least half of sampled groups contain reward variation. Otherwise group-relative
advantages collapse to zero and GPU time is wasted.
