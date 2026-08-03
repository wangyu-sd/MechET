# Mechanistic knowledge anchors

The filename is retained for compatibility, but the scientific term used in the paper and public documentation is **mechanistic knowledge anchor**.

## Critical distinction

MechET contains two different objects that must not be conflated.

### Electron-flow execution primitive

A local executable source-to-sink action such as:

```text
LP -> BOND
BOND -> ATOM
BOND -> BOND
```

Execution primitives define:

- the causal action vocabulary;
- deterministic state transitions;
- proof composition signatures;
- primitive-seen/composition-unseen splits.

### Mechanistic knowledge anchor

A provenance-aware structured record containing:

```text
molecular-role patterns
candidate execution primitives
preconditions
warnings
competing pathways
follow-up suggestions
source and license metadata
```

Knowledge anchors are retrieved soft evidence. They do not define the causal trace, formal validity or composition-OOD vocabulary.

## Scientific role

The anchor library tests whether external structured mechanistic evidence improves induction of a causal electron-flow program.

```text
reviewed external evidence
  -> mechanistic knowledge anchor
  -> role binding on the current mapped state
  -> candidate actions, warnings and competitors
  -> actor choice
  -> deterministic execution
```

A match does not establish kinetics, selectivity, yield, experimental feasibility or a unique physical mechanism. Lack of a match does not imply impossibility.

## Source inventory

The registry is `knowledge/source_registry.yaml`.

| Source | Role | Boundary |
|---|---|---|
| IUPAC Gold Book | stable terminology and definitions | per-term provenance; no assumption of unrestricted bulk redistribution |
| RXNO | reaction-family taxonomy | official ontology and attribution |
| Wikibooks Organic Chemistry | open explanatory text | revision-pinned text; media excluded |
| selected LibreTexts pages | open mechanism explanations | page-level license checks |
| MIT OpenCourseWare | non-commercial course evidence | explicit non-commercial acceptance; third-party exclusions |
| PMechDB/PMechRP | mechanism data and benchmark assets | manual/request-gated; upstream terms preserved |

Commercial textbook editions are not mirrored. Released descriptions and structured records must be independently authored.

## Acquisition and provenance

```bash
python scripts/download_mechanistic_sources.py \
  --registry knowledge/source_registry.yaml list

python scripts/download_mechanistic_sources.py \
  --registry knowledge/source_registry.yaml \
  download \
  --source iupac_goldbook_terms \
  --source rxno \
  --source wikibooks_organic_chemistry \
  --output knowledge/raw

python scripts/download_mechanistic_sources.py \
  --registry knowledge/source_registry.yaml \
  verify --output knowledge/raw
```

Every asset retains source ID, URL, revision information, license, acquisition method and SHA-256 hash.

## Anchor schema

The current compatibility path stores anchors in `knowledge/primitives/core_polar_primitives.yaml`.

```yaml
primitive_id: nucleophilic_substitution_sp3
version: "1.0"
status: executor_verified
patterns:
  - smarts: "[C:1]-[Cl,Br,I:2]"
    roles: {electrophile: 1, leaving_group: 2}
  - smarts: "[O-,N-,S-:1]"
    roles: {nucleophile: 1}
moves:
  - source: {kind: LP, roles: [nucleophile]}
    sink: {kind: BOND, roles: [electrophile, nucleophile]}
  - source: {kind: BOND, roles: [electrophile, leaving_group]}
    sink: {kind: ATOM, roles: [leaving_group]}
preconditions: [...]
warnings: [...]
competing_primitives: [...]
sources: [...]
```

The field name `primitive_id` remains for API compatibility. In scientific writing it is an **anchor ID**, not an execution-primitive ID.

## Extraction and review workflow

Downloaded text does not become executable knowledge automatically.

1. Register source, revision method, license and redistribution policy.
2. Download evidence and freeze hashes.
3. Build bounded evidence-linked extraction candidates.
4. Use `UNKNOWN` for unsupported extracted fields.
5. Independently encode role SMARTS, candidate actions, warnings and competitors.
6. Instantiate mapped examples and replay candidate actions.
7. Review scope, proton transfers, resonance, stereochemistry, conditions and competitors.
8. Assign a status such as `draft`, `text_supported`, `executor_verified`, `chemist_reviewed`, `released` or `deprecated`.

```bash
python scripts/build_primitive_extraction_queue.py \
  --download-root knowledge/raw \
  --output knowledge/candidates/extraction_queue.jsonl
```

LLM extraction is never automatically promoted to a released anchor.

## Online retrieval

`KnowledgeAugmentedAgentEnv` may expose:

```text
retrieve_primitives(query, top_k)
```

The compatibility class `PrimitiveAugmentedAgentEnv` remains available, but the main scientific condition uses trace ownership and `finish_trace`.

Retrieval returns anchor IDs, role bindings, candidate actions, warnings, competitors, follow-ups and provenance. It does not return a precursor or complete answer.

## Reward boundary

Knowledge-anchor retrieval receives no direct reward in the primary evidence experiment.

A separate bounded anchor-support reward may be studied only as an ablation:

- unmatched formally valid actions remain allowed;
- retrieval-only records cannot provide formal reward;
- anchor support is logged separately from execution and endpoint rewards;
- no anchor score can offset formal failure.

## Matched experiment

The required conditions are generated by `build_knowledge_ablation_suite.py`:

```text
trace_no_knowledge
trace_length_matched_irrelevant
trace_textbook_rag
trace_structured_anchors
trace_text_plus_anchors
direct_textbook_rag
```

Anchors-only rows are derived from combined trace rows by removing textbook retrieval. They must not be prepared from a different dataset.

Report:

```text
endpoint and execution metrics
anchor retrieval coverage
role-binding accuracy
exact candidate-action support
intervention effects
context and tool budgets
family/scaffold/composition OOD
latency
```

A gain cannot be attributed to structured anchors if it is explained by extra context, different IDs, different optimization or direct reward.

## Relationship to compositional generalization

H2 is tested over **execution-primitive compositions**. Anchor IDs may be used for secondary analyses but cannot define the headline composition holdout, because anchors contain higher-level curated knowledge and may encode reaction-family information.

## Boundaries

- Knowledge anchors summarize structured evidence, not experimental truth.
- A citation does not prove a candidate electron-flow action.
- A complete anchor match does not prove the full mechanism or selectivity.
- Lack of an anchor match does not imply impossibility.
- License constraints follow every source and derivative artifact.
- Report the source-registry hash, anchor-library hash and retrieval configuration with every evidence-conditioned checkpoint.
