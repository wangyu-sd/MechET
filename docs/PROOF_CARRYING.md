# `MECH_PROOF v1`: executable bond–electron program format

`MECH_PROOF v1` is the deterministic executable format used by MechET. It has two roles:

1. the trace-owned main environment compiles its committed electron-flow actions into `MECH_PROOF v1`;
2. independent complete-proof generation remains a required compatibility baseline.

In the main method, the language model does **not** independently submit a proof. It completes an environment-owned trace with `finish_trace`; the environment compiles the trace and the executor derives the structural precursor.

```text
main method:
explicit source-to-sink actions
  -> environment-owned trace
  -> MECH_PROOF v1 compilation
  -> deterministic execution
  -> precursor

complete-proof baseline:
product
  -> independently generated MECH_PROOF v1
  -> deterministic execution
  -> precursor
```

Neither path contains an independent precursor answer channel.

## 1. Program form

```text
<proof>
MECH_PROOF v1
TARGET_SMILES "<mapped product>"
ROOT s0
  IMPORT "<mapped species>"
PRECURSOR_STATE sk
EDGE s0 s1
  IMPORT "<mapped species introduced for this transition>"
  BOND i j +1
  LP i -2
  CHARGE i -1 0
...
</proof>
```

A proof may be a chain, tree or DAG. The executor resolves dependencies from state identifiers rather than relying on textual edge order.

## 2. Proof operations

The proof grammar uses local operations rather than a library of complete reaction templates.

### `IMPORT`

Adds an atom-mapped fragment. Every imported atom requires a positive globally unique map.

### `BOND i j delta`

Changes the bond order between mapped atoms `i` and `j`:

```text
new bond order = current bond order + delta
```

### `LP i delta`

Declares the non-bonding-electron change for atom `i`. The executor recomputes the value from the executed source and destination states and rejects disagreement.

### `CHARGE i q0 q1`

Applies a formal-charge transition with a source-state precondition.

## 3. Example

For:

```text
[CH3:1][OH:2]
```

one executable reverse program is:

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

The precursor is the reconstructed `PRECURSOR_STATE`, not model-authored answer text.

## 4. Execution primitives versus proof deltas

The agent-facing causal vocabulary consists of explicit source-to-sink **electron-flow execution primitives**, such as:

```text
LP -> BOND
BOND -> ATOM
BOND -> BOND
```

`MECH_PROOF v1` stores the compiled bond, lone-pair and charge deltas produced by those actions. A proof edge may not uniquely identify every source/sink pairing when several pairings yield the same net delta; the conservative proof-to-trace converter accepts only uniquely recoverable cases and quarantines ambiguity.

Mechanistic knowledge-anchor IDs are separate retrieval metadata and are not part of the proof grammar.

## 5. Local operations are not whole-reaction templates

MechET does not emit a stored transformation token and instantiate an entire reaction rule. It specifies:

- mapped atoms and imported components;
- local bond, lone-pair and charge changes;
- dependencies among elementary transitions.

The current cold-start data is nevertheless compiled from mechanistic trajectories and inherits their coverage. Template-free execution does not imply template-free supervision.

## 6. Deterministic executor contract

For every proof, the executor:

1. parses the schema;
2. constructs roots and imports;
3. checks unique positive atom maps;
4. applies bond and formal-charge transitions;
5. sanitizes every state;
6. recomputes bond, lone-pair and charge deltas;
7. enforces electron conservation;
8. resolves dependencies and rejects unreachable edges;
9. requires identical states at DAG joins;
10. derives the structural precursor only after all checks pass.

The executor is deterministic and is not trained. Learned models, retrieval and forward evidence cannot override an execution failure.

## 7. Failure certificates

Stable failure families include:

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

Certificates may condition a learned repair proposal followed by complete re-execution. Only semantics-preserving declaration fixes may be deterministic; bond, charge, import and dependency edits change the proposed chemistry.

## 8. Partial-order equivalence

Exact text equality is not the semantic criterion. Valid transformations include:

```text
state-ID renaming
synchronized atom-map permutation
serialization changes
reordering of commuting independent events
```

Proof-class and composition signatures are used for equivalence, hypothesis deduplication and MechComp-OOD analysis.

## 9. K complete-proof hypotheses

The same complete-proof baseline actor may be sampled repeatedly:

```text
pi_k ~ p_theta(MECH_PROOF | product),  k = 1,...,K
```

`K` is a test-time compute budget, not a set of stored templates. Each sample is parsed, executed, diagnosed, optionally repaired, deduplicated by partial-order equivalence and grouped by structural endpoint.

Report:

```text
n_generated
n_parseable
n_executable
n_unique_executable_proof_classes
n_unique_execution-primitive_compositions
n_unique_structural_endpoints
ExecutePass@K
EndpointPass@K
```

K-hypothesis complete-proof generation is a baseline/extension; it does not replace the trace-owned causal study.

## 10. Current boundaries

`MECH_PROOF v1` and the current executor do not establish:

- a unique physical mechanism;
- activation barriers, kinetics, yield or experimental success;
- one-electron radical moves;
- orbital identity;
- transition-metal coordination, oxidation-state and spin dynamics;
- universal solvent or condition compatibility.

The main paper scope remains mapped, closed-shell, two-electron polar chemistry unless the representation and verifier are explicitly extended.

## 11. Required comparisons

The representation study includes:

```text
outcome-only
free-form CoT plus answer
state-CoT plus answer
net edit
independent complete proof
legacy loose trace plus submitted proof
trace-owned source-to-sink actions compiled into proof
```

The main causal claim depends on the last comparison and tool-observation interventions, not simply on complete-proof execution accuracy.
