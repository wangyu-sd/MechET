# `MECH_PROOF v1`: executable bond–electron programs

`MECH_PROOF v1` is the primary MechET representation. A model emits an action-only program; a deterministic executor reconstructs every state and derives the structural precursor. The assistant output contains no independent precursor answer.

```text
mapped product
  -> root imports + sparse proof edges
  -> deterministic execution and falsification
  -> executor-derived precursor
```

## 1. Program form

```text
<proof>
MECH_PROOF v1
TARGET_SMILES "<mapped product>"
ROOT s0
  IMPORT "<mapped species already present at the root>"
PRECURSOR_STATE sk
EDGE s0 s1
  IMPORT "<mapped species introduced for this transition>"
  BOND i j +1
  LP i -2
  CHARGE i -1 0
...
</proof>
```

The proof may be a chain, tree, or DAG. `EDGE` order in the text is not required to be topological; the executor resolves dependencies from state identifiers.

## 2. Electron-flow primitives

The proof grammar uses local operations rather than a library of complete reaction templates.

### `IMPORT`

Adds an atom-mapped molecular fragment to a root or elementary transition. Every imported atom must have a positive, globally unique atom map.

### `BOND i j delta`

Changes the bond order between mapped atoms `i` and `j`:

```text
new bond order = current bond order + delta
```

Examples:

```text
BOND 1 2 -1    # remove a single bond or reduce a double bond to single
BOND 1 3 +1    # form a single bond or increase bond order by one
```

### `LP i delta`

Declares the change in the diagonal bond–electron entry associated with atom `i`. In the current executor, the molecular graph is changed by `BOND` and `CHARGE`; the lone-pair delta is independently recomputed from the executed source and destination states and checked against the written declaration.

### `CHARGE i q0 q1`

Applies a formal-charge transition with a precondition. The executor rejects the proof when the source atom does not have charge `q0`.

## 3. Example

For the mapped product:

```text
[CH3:1][OH:2]
```

one executable reverse proof is:

```text
<proof>
MECH_PROOF v1
TARGET_SMILES "[CH3:1][OH:2]"
ROOT s0
  IMPORT "[Br-:3]"
PRECURSOR_STATE s1
EDGE s0 s1
  BOND 1 2 -1
  BOND 1 3 +1
  LP 2 +2
  LP 3 -2
  CHARGE 2 0 -1
  CHARGE 3 -1 0
</proof>
```

The edge encodes:

- loss of the C–O bond;
- formation of the C–Br bond;
- the corresponding non-bonding-electron redistribution;
- the checked formal-charge changes.

The executor applies the graph changes, sanitizes the resulting state, recomputes the bond–electron delta, and requires:

```text
written BOND changes  == executed BOND changes
written LP changes    == derived LP changes
written CHARGE changes == executed CHARGE changes
sum(LP delta) + 2 * sum(BOND delta) == 0
```

The precursor is the reconstructed `PRECURSOR_STATE`, not model-authored text.

## 4. Local primitives are not reaction templates

MechET does not emit a token such as `SN2_TEMPLATE_17` and then instantiate a stored whole-reaction rule. It autoregressively chooses:

- which mapped atoms participate;
- which bonds change and by how much;
- which lone-pair and charge changes accompany the transition;
- how multiple elementary transitions form a chain, tree, or DAG.

The representation is therefore template-free at inference in the usual sense of not retrieving a fixed transformation rule. However, the current cold-start corpus is compiled from FlowER trajectories and inherits the coverage and mechanistic priors of that data-construction procedure. Template-free generation does not imply template-free supervision.

The required tests for this distinction are:

- reaction-center-clean evaluation;
- MechComp-OOD with seen primitives and unseen full compositions;
- primitive-unseen analysis where feasible;
- post-cutoff or non-USPTO external mechanisms.

## 5. How K proof hypotheses are generated

For one product \(P\), the same autoregressive actor is sampled repeatedly:

```text
pi_k ~ p_theta(MECH_PROOF | P),  k = 1,...,K
```

The standard hypothesis-set inference uses stochastic decoding with a temperature and nucleus-sampling threshold. `K` is a compute budget, not a set of K stored templates.

Each raw sample is then:

1. parsed;
2. executed;
3. assigned an executor-derived structural precursor when valid;
4. diagnosed with a failure certificate when invalid;
5. optionally repaired for a bounded number of rounds;
6. deduplicated by partial-order proof equivalence;
7. grouped by structural precursor endpoint.

A useful hypothesis report distinguishes raw sequences from chemical proof classes:

```text
n_generated
n_parseable
n_executable
n_unique_executable_proof_classes
n_unique_mechanism_compositions
n_unique_structural_endpoints
```

The inference implementation is `scripts/infer_proof_hypotheses.py`; bounded Generate–Falsify–Repair inference is `scripts/infer_proof_gfr.py`.

## 6. Executor contract

For every proof, the executor:

1. constructs each root from `TARGET_SMILES` and declared imports;
2. applies edges only when their source state has been derived;
3. checks that all atoms carry unique positive maps;
4. applies bond-order and formal-charge transitions;
5. sanitizes every reconstructed molecular state;
6. recomputes bond, lone-pair, and charge deltas;
7. enforces electron conservation;
8. requires identical reconstructed states at DAG joins;
9. rejects unreachable edges;
10. returns the declared precursor state only when all checks pass.

The executor is deterministic and is not trained. Learned models may generate, repair, or rank proofs but cannot override an execution failure.

## 7. Formal failure certificates

`diagnose_proof` maps the first failure to a stable code, stage, and edge. Examples include:

```text
PROOF_PARSE_FAILED
ATOM_MAP_ERROR
BOND_EXECUTION_MISMATCH
LP_EXECUTION_MISMATCH
CHARGE_PRECONDITION_FAILED
CHARGE_EXECUTION_MISMATCH
ELECTRON_NOT_CONSERVED
CHEMICAL_STATE_INVALID
UNREACHABLE_EDGE
PRECURSOR_NOT_DERIVED
DAG_JOIN_MISMATCH
```

A certificate can be used as input to a repair actor:

```text
TARGET
INVALID_PROOF
FAILURE_CERTIFICATE
  -> corrected MECH_PROOF
```

Only semantics-preserving lone-pair declaration corrections are repaired deterministically. Bond, charge, import, and dependency changes alter the proposed chemistry and therefore require a learned repair proposal followed by complete re-execution.

## 8. What the current representation does not claim

`MECH_PROOF v1` represents sparse bond–electron state deltas. It does **not** yet uniquely pair every electron source with a specific electron sink. For an edge containing several donors and acceptors, the net redistribution may be executable even when multiple curved-arrow pairings are possible.

The following are outside the current formal language:

- explicit `SOURCE -> TARGET` electron moves;
- one-electron radical moves;
- orbital identity;
- metal oxidation and spin-state dynamics inside the molecular executor;
- coordination and ligand-exchange primitives;
- activation barriers, rate constants, solvent effects, or condition compatibility.

These limitations matter most for radical chemistry, organometallic mechanisms, reaction-network discovery, and catalytic cycles. The current catalytic-cycle module checks proof execution and global ledgers but does not provide quantum-chemical validation.

## 9. Cold-start compilation

```bash
python scripts/build_mechet_proof_sft.py \
  --input-dir data/mechet_sft \
  --output-dir data/mechet_proof_sft \
  --splits train valid test
```

The compiler reads state-annotated trajectories, removes model-authored intermediate-state text and the independent answer channel, constructs sparse proof actions, and accepts a row only when execution reconstructs the original endpoint.

## 10. Training and inference entry points

```text
Proof Actor SFT       scripts/train_mechet_sft.py
Verifier-DPO          scripts/train_proof_dpo.py
Proof-set RLVR        scripts/train_proof_rlvr_distributed.py
Repair Actor          scripts/train_proof_repair.py
Single proof          scripts/infer_mechet_proof.py
Hypothesis set        scripts/infer_proof_hypotheses.py
GFR                   scripts/infer_proof_gfr.py
```

The complete data, loss, checkpoint-lineage, inference, and validation contract is specified in [`PROOF_CENTRIC_EXPERIMENT_PLAN.md`](PROOF_CENTRIC_EXPERIMENT_PLAN.md).
