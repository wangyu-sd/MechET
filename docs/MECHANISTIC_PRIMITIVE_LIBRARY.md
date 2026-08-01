# Mechanistic primitive reference library

## Purpose

MechET uses a compact, provenance-aware mechanism library to connect classical organic-chemistry knowledge with executable electron-flow reasoning. The library is optional **soft guidance**, not a whole-reaction template engine and not a replacement for the deterministic executor.

```text
open reference evidence
  -> reviewed mechanism primitive
  -> role binding on the current mapped molecular state
  -> concrete candidate E_MOVE set and warnings
  -> actor choice and deterministic execution
```

A primitive match does not establish kinetics, selectivity, yield or experimental feasibility. Lack of a primitive match does not imply impossibility.

## Web source inventory

The source registry is `knowledge/source_registry.yaml`.

| Source | Information extracted | License and use boundary |
|---|---|---|
| IUPAC Gold Book | stable terminology IDs and definitions | download individual term JSON; record version and attribution; do not assume unrestricted bulk redistribution |
| RXNO | reaction-family taxonomy and stable ontology IDs | CC BY 4.0 official OWL |
| Wikibooks Organic Chemistry | open textbook explanations and mechanism sections | CC BY-SA/GFDL unless stated otherwise; text only and revision pinned |
| Organic Synthesis (Shea), LibreTexts | selected open textbook pages | registered CC BY pages only; page-level license markers checked |
| MIT OCW Organic Chemistry | non-commercial course explanations and exercises | CC BY-NC-SA except third-party material; explicit non-commercial acceptance and excluded markers |
| PMechDB/PMechRP | elementary polar source/sink data and textbook-pathway benchmark | CC BY-NC-ND and request-gated; local research use unless further permission is obtained |

Commercial textbook editions are not downloaded or redistributed. Human reviewers may consult them to identify consensus chemistry, but released records must use independent descriptions and legally shareable evidence.

## Downloader

`scripts/download_mechanistic_sources.py` supports:

- per-term IUPAC JSON;
- official ontology files;
- revision-pinned MediaWiki pages;
- text-only HTML with license-marker and excluded-marker checks;
- explicit non-commercial/restricted gates;
- manual instructions for form-gated datasets;
- SHA-256 manifests and verification.

```bash
python scripts/download_mechanistic_sources.py \
  --registry knowledge/source_registry.yaml list

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

Raw downloads and restricted datasets normally remain outside Git. Release artifacts should contain the source registry, manifests, extraction code, independently written primitive records and only permitted evidence excerpts.

## Primitive schema

The seed library is `knowledge/primitives/core_polar_primitives.yaml`.

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

The library validates stable IDs, registered provenance sources, SMARTS role maps, supported electron-container kinds, two-electron v1 scope and explicit evidence status.

## Extraction and review workflow

Web retrieval does not directly create executable knowledge.

1. Register URL, revision method, license and redistribution policy.
2. Download evidence and preserve revision identifiers and hashes.
3. Build evidence-linked extraction candidates. LLM outputs use a strict schema and `UNKNOWN` for missing fields.
4. Independently write SMARTS roles, `E_MOVE` templates, warnings and competitors without copying protected prose or figures.
5. Compile concrete mapped examples and replay them through MechET.
6. Review omitted proton transfers, resonance/reaction confusion, scope, stereochemistry, conditions and competitors.
7. Release with status `draft`, `text_supported`, `executor_verified`, `chemist_reviewed`, `released` or `deprecated`.

Build the extraction queue:

```bash
python scripts/build_primitive_extraction_queue.py \
  --download-root knowledge/raw \
  --output knowledge/candidates/extraction_queue.jsonl
```

Each task retains its source span, license, revision and artifact hash. Candidate extraction is never automatically promoted to a released primitive.

## Model integration

### Online retrieval for the inverse actor

`PrimitiveAugmentedAgentEnv` extends `MechETAgentEnv` with:

```text
retrieve_primitives(query, top_k)
```

Results contain primitive IDs, atom-map role bindings, instantiated candidate `E_MOVE` sets, preconditions, warnings, competitors, follow-ups and provenance.

The actor still chooses actions and the deterministic environment executes them. Retrieval never directly returns a precursor or a complete named-reaction template.

```bash
python scripts/train_inverse_agent_primitives.py \
  --config configs/knowledge/inverse_trl_grpo_primitives.yaml \
  --dry-run --limit 8

python scripts/train_inverse_agent_primitives.py \
  --config configs/knowledge/inverse_trl_grpo_primitives.yaml
```

The matched baseline is `scripts/train_inverse_agent_trl.py` with the same backbone, data, LoRA capacity, rollout budget and reward components.

### Optional soft process reward

For a successful explicit move, the environment compares the exact source/sink set with reviewed executable primitive instances. The optional support reward is bounded and small by default.

- unmatched actions are not hard failures;
- retrieval-only/text-supported entries cannot provide formal action reward;
- primitive reward is reported separately from executor and endpoint rewards;
- ablations include retrieval without reward and reward without natural-language context.

### Offline context for forward and supervised models

`scripts/annotate_primitive_context.py` attaches candidate IDs and role bindings, mapped reaction bond-delta support, per-step exact move support, optional rendered context and provenance hashes.

It also writes compact primitive fields into `conditions`, which the current forward expert already encodes through its condition channel. This enables a matched comparison without changing the graph architecture.

```bash
python scripts/annotate_primitive_context.py \
  --input data/forward_expert/steps/train.jsonl \
  --output data/forward_expert/steps_primitive/train.jsonl \
  --library knowledge/primitives/core_polar_primitives.yaml \
  --source-registry knowledge/source_registry.yaml \
  --render-context
```

Later extensions may add explicit primitive embeddings, a primitive-prediction auxiliary head or hierarchy-aware contrastive losses. They are not required for the first experiment.

## Performance hypotheses and ablations

The reference library can help through:

1. reduced action search space through state-specific role bindings;
2. compositional inductive bias shared across reaction families;
3. sparse action-level evidence for process learning;
4. explicit competing pathways and warnings for selectivity analysis.

Required first experiment:

| Variant | Retrieval context | Structured primitive IDs | Primitive soft reward |
|---|---:|---:|---:|
| baseline tool actor | no | no | no |
| retrieval only | yes | no | no |
| structured IDs only | no | yes | no |
| retrieval + IDs | yes | yes | no |
| retrieval + IDs + soft reward | yes | yes | yes |

Match model scale, examples, assistant-token budget, tool-call budget, sampling budget, executor version and forward checkpoint. Include a length-matched generic-context control.

Report endpoint Top-k, synthon/reaction-centre accuracy, execution and invalid-action rate, tool recovery, abstention, primitive and complete-move accuracy, family/scaffold/temporal/MechComp-OOD, calibration, context tokens and latency.

A gain cannot be attributed to the library if it is explained only by extra tokens or compute.

## Scientific boundaries

- The library records textbook consensus and structured evidence, not experimental truth.
- A source citation does not make an inferred arrow correct.
- A complete primitive match does not prove the full mechanism or selectivity.
- Lack of a primitive match does not imply impossibility.
- License constraints follow every source and derivative artifact.
- Report the library version, source-registry hash, primitive-file hash and retrieval/reward configuration with every checkpoint.
