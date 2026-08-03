# MechET mechanistic evidence assets

This directory contains provenance-tracked natural-language evidence and reviewed **mechanistic knowledge anchors** used as optional soft guidance. It is not a mirror of commercial textbooks, not a whole-reaction template database and not the electron-flow execution-primitive vocabulary used for causal traces or MechComp-OOD.

## Scientific distinction

```text
electron-flow execution primitive
  local source-to-sink action used by the environment and composition split

mechanistic knowledge anchor
  curated structured evidence with role patterns, candidate actions, warnings,
  competitors and provenance
```

The compatibility directory and schema retain the historical name `primitives`, but papers, README text and experiment interpretation must call these records knowledge anchors.

## Layout

```text
knowledge/
  source_registry.yaml                 # source URLs, licenses and download gates
  primitives/
    core_polar_primitives.yaml         # reviewed knowledge-anchor records
  raw/                                 # downloaded locally; not committed by default
  corpus/                              # bounded textbook passages and frozen index
  candidates/                          # evidence-linked extraction tasks
  manifests/                           # SHA-256 and revision manifests
```

## Source policy

Every source is registered before acquisition. The registry records source type, URL, license, redistribution constraints and explicit acceptance requirements.

| Source | Evidence role | Automated use |
|---|---|---|
| IUPAC Gold Book individual terms | standard terminology | per-term JSON with canonical-code validation |
| RXNO | reaction-family taxonomy | official ontology download |
| Wikibooks Organic Chemistry | open textbook explanations | revision-aware MediaWiki fallbacks |
| selected LibreTexts pages | open mechanism explanations | text-only download after license-marker checks |
| MIT OpenCourseWare | non-commercial course evidence | explicit non-commercial acknowledgement |
| PMechDB/PMechRP | mechanism data and benchmark assets | manual upstream request; no bypass or automatic derivative redistribution |

A Web page, LLM extraction or database row does not automatically become accepted chemistry or a released anchor.

## Download and verification

Install:

```bash
pip install -e ".[knowledge]"
```

Inspect and preview:

```bash
python scripts/download_mechanistic_sources.py \
  --registry knowledge/source_registry.yaml list

python scripts/download_mechanistic_sources.py \
  --registry knowledge/source_registry.yaml \
  download \
  --source iupac_goldbook_terms \
  --source rxno \
  --source wikibooks_organic_chemistry \
  --output knowledge/raw \
  --dry-run
```

Remove `--dry-run` for permitted public sources. Verify hashes:

```bash
python scripts/download_mechanistic_sources.py \
  --registry knowledge/source_registry.yaml \
  verify --output knowledge/raw
```

## Network and identifier handling

### Gold Book canonical codes

Historical term codes may redirect. The registry may preserve aliases, while manifests distinguish:

```text
configured_term_id
requested_term_id
canonical_term_id
```

Artifacts are stored under the canonical code.

### MediaWiki fallbacks

The default order is:

```text
REST source
-> Action API
-> Special:Export XML
-> action=raw
```

Each artifact records the successful backend, revision information when available, content hash and earlier backend errors.

Proxy example:

```bash
python scripts/download_mechanistic_sources.py \
  --registry knowledge/source_registry.yaml \
  download --source wikibooks_organic_chemistry \
  --output knowledge/raw \
  --proxy http://127.0.0.1:7890
```

Offline MediaWiki import accepts `.xml`, `.txt` or compatible `.json` files:

```bash
python scripts/download_mechanistic_sources.py \
  --registry knowledge/source_registry.yaml \
  download --source wikibooks_organic_chemistry \
  --output knowledge/raw \
  --mediawiki-import-dir knowledge/manual/wikibooks
```

### Restricted sources

Non-commercial and restricted sources require explicit acknowledgement. PMechDB/PMechRP remain manual-gated; the downloader writes instructions and never bypasses the upstream request flow.

## Build the natural-language passage corpus

```bash
python scripts/build_textbook_corpus.py \
  --download-root knowledge/raw \
  --output knowledge/corpus/passages.jsonl

python scripts/index_textbook_corpus.py \
  --corpus knowledge/corpus/passages.jsonl \
  --output knowledge/corpus/bm25_index.json
```

Every passage retains source, locator, revision, license, exact evidence hash and artifact provenance. The final corpus and index are frozen before test evaluation.

## Evidence extraction queue

```bash
python scripts/build_primitive_extraction_queue.py \
  --download-root knowledge/raw \
  --output knowledge/candidates/extraction_queue.jsonl
```

The script name is retained for compatibility. Each task preserves source ID, URL, revision, license, artifact hash, exact evidence span and a strict candidate schema. LLM output remains unreviewed and must use `UNKNOWN` for unsupported fields.

## Knowledge-anchor records

A record contains:

```text
stable anchor ID and version
independently written description
atom-role SMARTS patterns
candidate source-to-sink actions
preconditions and warnings
competing pathways and follow-ups
provenance and license metadata
review status
```

`executor_verified` means the encoded candidate actions replay for reviewed examples. It does not make the record a formal validity oracle. `text_supported` records remain retrieval-only evidence.

## Release workflow

```text
registered source
  -> revision/hash-preserving evidence acquisition
  -> bounded evidence-linked candidate
  -> independent role and action encoding
  -> deterministic replay on mapped examples
  -> chemistry review
  -> released versioned knowledge anchor
```

Never copy protected textbook prose or figures into released records. Commercial sources may inform human review only when released descriptions and schemas are independently authored.

## Experimental use

Knowledge anchors are evaluated in the six-condition evidence suite. Anchors-only rows are derived from combined trace rows, ensuring the same stable IDs and chemistry trajectories.

Anchor IDs must not define the headline execution-primitive composition split. Lack of an anchor match does not imply impossibility, and a match does not prove a full mechanism, selectivity, kinetics, yield or experimental success.
