<div align="center">

# MechET

**Bidirectional electron-flow reasoning for reliable retrosynthesis**

[![Proof tests](https://github.com/wangyu-sd/MechET/actions/workflows/proof-tests.yml/badge.svg)](https://github.com/wangyu-sd/MechET/actions/workflows/proof-tests.yml)
[![Forward expert tests](https://github.com/wangyu-sd/MechET/actions/workflows/forward-expert-tests.yml/badge.svg)](https://github.com/wangyu-sd/MechET/actions/workflows/forward-expert-tests.yml)
[![Agent framework tests](https://github.com/wangyu-sd/MechET/actions/workflows/agent-framework-tests.yml/badge.svg)](https://github.com/wangyu-sd/MechET/actions/workflows/agent-framework-tests.yml)
[![Primitive library tests](https://github.com/wangyu-sd/MechET/actions/workflows/primitive-library-tests.yml/badge.svg)](https://github.com/wangyu-sd/MechET/actions/workflows/primitive-library-tests.yml)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![RDKit](https://img.shields.io/badge/RDKit-required-2E7D32?style=flat-square)](https://www.rdkit.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

[Scientific story](#scientific-story) · [Method](#method) · [Primitive library](#mechanistic-primitive-reference-library) · [Quickstart](#quickstart) · [Training](#training-and-inference) · [Planning](#multistep-planning) · [Evaluation](#evaluation) · [Documentation](#documentation)

</div>

---

## One-sentence contribution

MechET trains a **small inverse actor** to reason backward through **local executable primitives** of electron flow, derives precursors only through a **deterministic executor**, and uses an architecturally independent **compact forward electron-flow expert** to falsify unsupported targets and competing outcomes.

```text
reference mechanism knowledge
        ↓ soft retrieval and role binding
reason backward with explicit electron moves
        ↓
execute every chemical claim
        ↓
derive the precursor
        ↓
falsify forward against target and competitors
        ↓
retain, repair, abstain, or search further
```

## Scientific story

### The problem

Retrosynthesis systems face a persistent trade-off:

- templates and named-reaction rules are dependable within their coverage but fragment chemistry into isolated transformation classes;
- unconstrained graph or language generation is flexible but can hallucinate atom maps, reaction centres, mechanisms, reagents and routes;
- endpoint-only evaluation cannot distinguish a grounded prediction from an answer that bypasses its explanation;
- ordinary forward round-trip scoring provides weak process supervision and often ignores competing products;
- classical organic-chemistry knowledge exists in textbooks and ontologies, but it is rarely represented as provenance-aware executable model actions.

### The hypothesis

Many apparently isolated reactions can be decomposed into a smaller vocabulary of reusable electron-flow operations: lone-pair donation, bond cleavage, pi-bond shift, leaving-group departure, addition, elimination, migration and re-formation of unsaturation.

If these primitives are:

1. represented as executable source-to-sink actions;
2. linked to reviewed terminology, mechanism references and explicit provenance;
3. retrieved and role-bound on the current molecular state rather than instantiated as whole-reaction templates;
4. composed by a tool-using inverse actor;
5. checked after every step by a deterministic environment;
6. independently evaluated in the forward direction;

then smaller models may achieve better compositional generalization and lower hallucination rates than larger direct-answer models.

### The ICLR claim under test

The paper does not claim that electron flow, chain-of-thought, retrieval, forward models or synthesis planning are individually new. The contribution is their integration around one auditable object:

> electron-flow primitives are simultaneously the reasoning vocabulary, tool actions, formal verification units, process-reward units, reference-library records and route-search edges.

The central experimental questions are:

1. Does executable electron-flow CoT improve faithfulness and validity over outcome-only, free-form CoT, state-CoT and net-edit baselines?
2. Does a provenance-aware primitive library improve data efficiency and primitive-seen/composition-unseen generalization?
3. Can a compact forward expert provide useful process and selectivity evidence beyond endpoint round trip?
4. Can a small tool-using actor match or outperform a larger direct generator under matched data and compute?
5. Do formally verified and forward-supported edges improve multistep route reliability under matched search budgets?

The full collaborator-facing experiment contract is [`docs/PROOF_CENTRIC_EXPERIMENT_PLAN.md`](docs/PROOF_CENTRIC_EXPERIMENT_PLAN.md).

## Method

```text
atom-mapped target
       |
       v
small inverse actor
  - inspect electron containers
  - retrieve reviewed mechanism primitives
  - bind generic roles to atom maps
  - propose source-to-sink moves
  - read deterministic feedback
  - submit an executable proof or abstain
       |
       v
deterministic executor
  - atom-map, bond, charge and electron checks
  - sanitizable state transitions
  - chain/tree/DAG consistency
  - executor-derived precursor
       |
       v
compact forward expert
  - next source/sink compatibility
  - precursor-to-target compatibility
  - target-versus-competitor margin
  - uncertainty
       |
       v
ranking, process reward, revision or route-search cost
```

### 1. Inverse actor

The inverse actor receives an atom-mapped product. The compatibility path generates `MECH_PROOF v1`; the agent path can inspect molecular state, retrieve reviewed primitives, test explicit source-to-sink moves and read execution feedback.

A proof contains `IMPORT`, `BOND`, `LP`, `CHARGE` and `EDGE` operations. These are local executable primitives rather than a library of complete reaction templates. The precursor is generated only by execution; there is no independent answer channel.

The reference small-model path uses Qwen-family tool-calling models and TRL agentic GRPO. Initial scale experiments compare approximately 0.6B, 1–2B and 8B actors.

### 2. Deterministic executor

The executor is never trained. For each proof edge it constructs the mapped state, applies checked bond and charge operations, recomputes bond-electron and lone-pair changes, enforces exact electron conservation, sanitizes the state, resolves chain/tree/DAG dependencies and derives the precursor.

A learned score or textbook reference can never override an execution failure.

### 3. Stateful agent environment

`MechETAgentEnv` exposes:

```text
inspect_state
apply_electron_move
apply_coupled_electron_moves
submit_proof
abstain
get_reward
```

`PrimitiveAugmentedAgentEnv` adds:

```text
retrieve_primitives
```

The environment owns molecular state, tool budget, visited states, failure history, proof execution, process reward, terminal reward and rollout trace. Agent frameworks wrap this contract rather than duplicate chemistry rules.

### 4. Compact forward electron-flow expert

The independent forward expert is a small graph model with atom/bond message passing, electron-container embeddings, source and source-conditioned sink heads, precursor-product compatibility, condition context, contrastive competitor training, uncertainty and route-cost outputs.

It supports mapped, closed-shell, two-electron polar chemistry in v1. Radicals, transition-metal orbitals, spin changes, coordination chemistry and photochemical one-electron processes are out of scope rather than guessed.

The learned expert is soft evidence, not an experimental oracle. Selectivity is only meaningful when explicit competing sites, products, mechanisms or stereochemical outcomes are included.

### 5. Generate–Falsify–Repair

Generate–Falsify–Repair remains the complete-proof compatibility path:

```text
generate
  -> deterministic falsification
  -> structured failure certificate
  -> local repair, agent revision or resampling
```

The same actor can revise after a failed tool result; a separate Repair Actor is retained as a controlled baseline.

### 6. K proof hypotheses

For one product, the same actor may be sampled repeatedly:

```text
pi_1, ..., pi_K ~ p_theta(proof | product)
```

K proof hypotheses are a test-time-compute budget, not K stored templates. Candidates are executed, deduplicated by partial-order equivalence, grouped by structural endpoint and optionally reranked with forward evidence.

Set-valued metrics include `ExecutePass@K`, `EndpointPass@K`, unique executable proof classes, mechanism compositions and precursor endpoints.

### 7. Alternating two-small-model learning

The inverse actor and forward expert are not trained simultaneously as a GAN:

1. pretrain both models independently;
2. freeze the forward expert and improve the actor with formal process and soft terminal rewards;
3. freeze the actor and mine high-scoring disagreements;
4. place disagreements in an audit queue rather than automatically labelling them negative;
5. update and recalibrate the forward expert only on independently verified negatives;
6. repeat for a small audited number of rounds.

An alternative endpoint is not chemically wrong merely because it differs from one patent record.

## Mechanistic primitive reference library

### What it is

`knowledge/primitives/core_polar_primitives.yaml` is a small, versioned seed ontology of executable and retrieval-only mechanism motifs. Each record contains:

- stable primitive ID and version;
- independently written description and aliases;
- atom-role SMARTS patterns;
- generic source-to-sink `E_MOVE` templates;
- preconditions, warnings, competitors and possible follow-ups;
- provenance references, license labels and evidence status.

The library is **soft guidance**, not a hard reaction-template database:

```text
reviewed generic primitive
  -> match structural motifs in current mapped state
  -> bind roles to concrete atom maps
  -> instantiate candidate E_MOVE set
  -> actor selects or ignores it
  -> executor decides formal validity
```

Lack of a match does not imply impossibility, and a match does not prove feasibility or selectivity.

### Web sources and licenses

The registry `knowledge/source_registry.yaml` currently covers:

| Source | Role | Acquisition policy |
|---|---|---|
| IUPAC Gold Book individual terms | standard terminology | per-term JSON with version and attribution |
| RXNO | named-reaction taxonomy | official CC BY OWL |
| Wikibooks Organic Chemistry | open textbook mechanism text | revision-pinned MediaWiki text; media excluded |
| Organic Synthesis (Shea), LibreTexts | selected open textbook pages | explicit page list and license-marker check |
| MIT OpenCourseWare Organic Chemistry | non-commercial course evidence | explicit non-commercial acknowledgement; third-party markers excluded |
| PMechDB/PMechRP | elementary mechanisms and textbook-pathway benchmark | upstream manual request; no automated derivative redistribution |

Commercial textbooks are not downloaded or mirrored. Human reviewers may consult them, but released descriptions and structured rules must be independently authored.

### Download and provenance

Install knowledge dependencies:

```bash
pip install -e ".[knowledge]"
```

List sources:

```bash
python scripts/download_mechanistic_sources.py \
  --registry knowledge/source_registry.yaml list
```

Preview the public download plan:

```bash
python scripts/download_mechanistic_sources.py \
  --registry knowledge/source_registry.yaml \
  download \
  --source iupac_goldbook_terms \
  --source rxno \
  --source wikibooks_organic_chemistry \
  --source libretexts_organic_synthesis_shea \
  --output knowledge/raw \
  --dry-run
```

Download and write revision/SHA-256 manifests:

```bash
python scripts/download_mechanistic_sources.py \
  --registry knowledge/source_registry.yaml \
  download \
  --source iupac_goldbook_terms \
  --source rxno \
  --source wikibooks_organic_chemistry \
  --source libretexts_organic_synthesis_shea \
  --output knowledge/raw

python scripts/download_mechanistic_sources.py \
  --registry knowledge/source_registry.yaml \
  verify --output knowledge/raw
```

Non-commercial or restricted sources require explicit acknowledgement. The PMechDB command only writes manual download instructions and never bypasses the upstream request flow.

### Build an evidence-linked extraction queue

Downloaded text is not automatically chemistry truth. Build bounded evidence spans and strict candidate-extraction tasks:

```bash
python scripts/build_primitive_extraction_queue.py \
  --download-root knowledge/raw \
  --output knowledge/candidates/extraction_queue.jsonl
```

Each task retains source ID, URL, license, revision, artifact hash, exact evidence span, extraction schema and prompt. A downstream LLM must use `UNKNOWN` for unsupported fields. Candidates require independent SMARTS/`E_MOVE` encoding, executor replay and chemistry review before release.

### Annotate reaction data

Attach primitive candidates, role bindings, reaction bond-delta support and compact primitive context:

```bash
python scripts/annotate_primitive_context.py \
  --input data/forward_expert/steps/train.jsonl \
  --output data/forward_expert/steps_primitive/train.jsonl \
  --library knowledge/primitives/core_polar_primitives.yaml \
  --source-registry knowledge/source_registry.yaml \
  --render-context
```

The script writes primitive IDs into the existing condition/context channel, enabling matched forward-expert ablations without changing the graph architecture.

### Train a primitive-augmented inverse actor

Baseline:

```bash
python scripts/train_inverse_agent_trl.py \
  --config configs/agent/inverse_trl_grpo.yaml
```

Primitive-augmented matched variant:

```bash
python scripts/train_inverse_agent_primitives.py \
  --config configs/knowledge/inverse_trl_grpo_primitives.yaml \
  --dry-run --limit 8

python scripts/train_inverse_agent_primitives.py \
  --config configs/knowledge/inverse_trl_grpo_primitives.yaml
```

The environment can supply a small bounded primitive-support reward, but unmatched formally valid actions remain allowed. Report primitive reward separately from executor and endpoint rewards.

### Required knowledge ablation

| Variant | Retrieval tool | Structured primitive IDs | Primitive reward |
|---|---:|---:|---:|
| baseline tool actor | no | no | no |
| retrieval only | yes | no | no |
| IDs only | no | yes | no |
| retrieval + IDs | yes | yes | no |
| retrieval + IDs + bounded reward | yes | yes | yes |

Match model scale, examples, assistant-token budget, tool-call budget, sampling budget, executor version and forward checkpoint. Include a length-matched generic-context control so gains cannot be explained by extra tokens alone.

## What is implemented

- `MECH_PROOF v1` compiler, executor, equivalence, diagnostics and failure certificates;
- proof-SFT, Verifier-DPO, proof-set RLVR, bounded repair, K-hypothesis inference and Generate–Falsify–Repair;
- conservative data normalization and leakage audit;
- source/sink electron containers and coupled-arrow execution;
- compact forward expert training, inference, generation, calibration and selectivity scoring;
- framework-neutral agent environment and TRL reference training;
- native and Syntheseus route search;
- provenance-aware source registry and Web downloader;
- evidence-linked extraction queue builder;
- reviewed mechanistic seed library with SMARTS role binding and executable `E_MOVE` templates;
- online primitive retrieval, optional bounded process reward and offline dataset annotation;
- dedicated proof, forward, agent, primitive-library and documentation CI.

## Current status

MechET is a research preview. Infrastructure is available; paper-scale checkpoints and frozen result tables are not yet released.

| Component | Status |
|---|---|
| Deterministic executor and formal verifier | available |
| Proof SFT/DPO/RLVR and K-hypothesis inference | available |
| Source-to-sink tool environment | available |
| Compact forward expert | available |
| TRL small-actor adapter | available |
| Web source registry/downloader/manifests | available |
| Primitive extraction queue | available |
| Seed polar primitive library and retrieval | available |
| Primitive-augmented actor and data annotation | available |
| Syntheseus offline planning adapter | available |
| Paper-scale inverse/forward checkpoints | not released |
| Primitive-library performance results | not released |
| Matched multistep results | not released |
| Kinetic, transition-state or experimental validation | external evidence required |

## Quickstart

### Install

```bash
git clone https://github.com/wangyu-sd/MechET.git
cd MechET

pip install -e ".[dev]"             # core, proof and tests
pip install -e ".[forward]"         # compact forward expert
pip install -e ".[data]"            # download/Arrow/Parquet data
pip install -e ".[mapping,ord]"     # optional mapping and ORD
pip install -e ".[agent]"           # small inverse actor with TRL
pip install -e ".[knowledge]"       # mechanism Web sources and primitive library
pip install -e ".[planning]"        # Syntheseus
```

### Execute a proof

```python
from mechet.proof_program import ChargeAction, ProofEdge, ProofProgram
from mechet.proof_program import format_proof_output, verify_proof

program = ProofProgram(
    target_smiles="[CH3:1][OH:2]",
    roots={"s0": ["[Br-:3]"]},
    precursor_state_id="s1",
    edges=[ProofEdge(
        "s0", "s1",
        bonds=[(1, 2, -1), (1, 3, +1)],
        lone_pairs=[(2, +2), (3, -2)],
        charges=[ChargeAction(2, 0, -1), ChargeAction(3, -1, 0)],
    )],
)
result = verify_proof(
    format_proof_output(program),
    expected_precursor="[CH3:1][Br:3].[OH-:2]",
)
print(result["execute_ok"], result["endpoint_exact"])
```

### Retrieve and bind primitives

```python
from mechet.primitive_library import PrimitiveLibrary

library = PrimitiveLibrary.load(
    "knowledge/primitives/core_polar_primitives.yaml",
    source_registry="knowledge/source_registry.yaml",
)
for match in library.retrieve("[CH3:1][Br:2].[OH-:3]", top_k=4):
    print(match.primitive_id, match.role_bindings, match.moves)
```

### Inspect the primitive-augmented agent environment

```python
import json
from mechet.primitive_agent_env import PrimitiveAugmentedAgentEnv

env = PrimitiveAugmentedAgentEnv()
print(env.reset(target_smiles="[CH3:1][Br:2].[OH-:3]"))
print(json.loads(env.retrieve_primitives())["matches"])
```

## Training and inference

### Forward expert

```bash
python scripts/train_forward_expert.py \
  --config configs/forward/forward_expert_small.yaml

python scripts/run_forward_expert.py infer \
  --checkpoint outputs/forward_expert/small/best \
  --input data/forward_expert/steps/test.jsonl \
  --output outputs/forward_expert/test_predictions.jsonl \
  --auto-competitors 8
```

### Inverse proof baselines

```bash
python scripts/train_mechet_sft.py \
  --config configs/proof/proof_actor_sft.yaml

python scripts/train_proof_dpo.py \
  --config configs/proof/proof_dpo.yaml
```

### Small inverse tool actor

```bash
python scripts/train_inverse_agent_trl.py \
  --config configs/agent/inverse_trl_grpo.yaml \
  --dry-run --limit 8
```

### K-hypothesis inference and forward reranking

```bash
python scripts/infer_proof_hypotheses.py \
  --data data/mechet_proof_clean/test.jsonl \
  --adapter outputs/proof/actor/adapter \
  --samples-per-target 64 \
  --out outputs/proof/hypotheses.jsonl

python scripts/rerank_proof_hypotheses_forward.py \
  --predictions outputs/proof/hypotheses.jsonl \
  --checkpoint outputs/forward_expert/small/best \
  --output outputs/proof/hypotheses_forward_ranked.jsonl
```

## Multistep planning

```bash
python scripts/run_syntheseus_search.py \
  --candidate-pool outputs/proof/hypotheses_forward_ranked.jsonl \
  --targets data/benchmarks/paroutes/targets.smi \
  --inventory data/benchmarks/paroutes/inventory.smi \
  --output-dir outputs/planning/syntheseus_retrostar \
  --algorithm retro_star
```

Formal invalidity is a hard prune. Primitive support, forward score, selectivity, precedent and uncertainty are soft ranking terms.

## Evaluation

The ICLR result package spans five evidence layers:

| Layer | Primary metrics |
|---|---|
| Endpoint | structural precursor Top-1/5/10, reaction-centre and synthon accuracy |
| Process | source/sink and complete-move accuracy, proof equivalence, execution rate |
| Reliability | false acceptance/rejection, calibration, risk–coverage, abstention |
| Generalization | family, scaffold, temporal and primitive-composition OOD |
| Planning | solved rate, fully verified route rate, invalid edges, route length/diversity and search cost |

Primitive-library experiments additionally report retrieval recall/precision against reviewed labels, role-binding accuracy, exact `E_MOVE` support, context tokens, tool latency and improvement per extra unit of compute.

No single Top-1 number is sufficient for the paper claim.

## Documentation

Start with [`docs/README.md`](docs/README.md).

- [`docs/PROOF_CENTRIC_EXPERIMENT_PLAN.md`](docs/PROOF_CENTRIC_EXPERIMENT_PLAN.md) — authoritative ICLR scientific and execution contract
- [`docs/MECHANISTIC_PRIMITIVE_LIBRARY.md`](docs/MECHANISTIC_PRIMITIVE_LIBRARY.md) — Web sources, extraction, schema, model integration and ablations
- [`knowledge/README.md`](knowledge/README.md) — source/download and local asset policy
- [`docs/PROOF_CARRYING.md`](docs/PROOF_CARRYING.md) — proof language and executor
- [`docs/FORWARD_ELECTRON_EXPERT.md`](docs/FORWARD_ELECTRON_EXPERT.md) — forward expert
- [`docs/FRAMEWORK_MIGRATION.md`](docs/FRAMEWORK_MIGRATION.md) — TRL, verl and Syntheseus strategy
- [`docs/PROOF_EQUIVALENCE.md`](docs/PROOF_EQUIVALENCE.md) — equivalence and compositional OOD
- [`docs/DATA_LEAKAGE_AND_ICLR_PLAN.md`](docs/DATA_LEAKAGE_AND_ICLR_PLAN.md) — lineage and leakage controls

## Boundaries

- Formal executability is not evidence of a low barrier, favorable kinetics, high yield or experimental success.
- Forward and primitive scores are learned or curated soft evidence, not infallible oracles.
- Selectivity requires explicit competitors.
- A source citation does not prove an inferred curved arrow.
- Lack of a primitive match does not imply chemical impossibility.
- The current explicit source/sink environment supports closed-shell two-electron chemistry, not all mechanisms.
- Tool-call or JSON validity does not imply chemical validity.
- Third-party datasets, models, Web pages and frameworks retain their upstream licenses.
