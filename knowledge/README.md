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
  source_registry.yaml                 # URLs, licenses, quality and health policy
  primitives/
    core_polar_primitives.yaml         # reviewed knowledge-anchor records
  raw/                                 # downloaded locally; not committed by default
  corpus/                              # bounded textbook passages and frozen index
  candidates/                          # evidence-linked extraction tasks
  manifests/                           # SHA-256, revision and health manifests
```

## Source policy

Every source is registered before acquisition. The registry records source type, URL, license, redistribution constraints, explicit acceptance requirements, evidence quality, permitted uses and prohibited interpretations.

| Source | Evidence role | Automated use |
|---|---|---|
| IUPAC Gold Book individual terms | standard terminology | per-term JSON with canonical-code validation |
| RXNO | reaction-family taxonomy | official ontology download |
| Wikibooks Organic Chemistry | open textbook explanations | revision-aware MediaWiki fallbacks and page-level quality warnings |
| selected LibreTexts pages | open mechanism explanations | text-only download after license-marker checks |
| MIT OpenCourseWare | non-commercial course evidence | explicit non-commercial acknowledgement |
| PMechDB/PMechRP | mechanism data and benchmark assets | manual upstream request; no bypass or automatic derivative redistribution |

A Web page, LLM extraction or database row does not automatically become accepted chemistry or a released anchor.

## Quality metadata

Each source defines a `quality` block; individual pages may override it through `page_quality`.

```yaml
quality_status: reviewed | usable_with_caution | low_priority
retrieval_weight: 0.0-1.0
review_warning: true | false
scientific_scope: [...]
allowed_uses: [...]
disallowed_uses: [...]
quality_notes: ...
```

Quality metadata is an evidence-governance signal, not a learned probability and not a formal verifier. In particular, introductory community-authored pages may be used for broad context or candidate evidence while remaining prohibited as mechanism ground truth, selectivity evidence or feasibility evidence.

The downloader writes the resolved quality fields into each artifact manifest. `build_textbook_corpus.py` propagates them into `TextbookPassage.metadata`, enabling reviewed-source-only and low-quality-source-removed H3 analyses.

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

## Source health monitoring

Run a live source audit without modifying the corpus:

```bash
python scripts/check_source_health.py \
  --registry knowledge/source_registry.yaml \
  --output outputs/source_health.json
```

The audit checks:

```text
configured and resolved titles
page and revision identifiers
non-empty content and minimum content length
soft 404 responses and unresolved redirects
final URL, response bytes and SHA-256
source/page quality warnings
backend failures and fallbacks
```

`.github/workflows/source-health-check.yml` runs weekly and can be triggered manually. It uploads the complete report and opens or updates a source-health issue when a registered source fails. The scheduled job does not block ordinary pull requests because external availability is not deterministic.

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

Every backend result passes one common validator before it can enter the manifest. REST or Action API success without content or a revision is rejected; Export XML must contain a revision and text; raw/local imports retain an explicit revision limitation. Soft missing-page text and unresolved `#REDIRECT` content are rejected rather than accepted as evidence.

Each artifact records the successful backend, configured and resolved titles, revision information when available, content length, content hash, quality metadata and earlier backend errors.

Proxy example:

```bash
python scripts/download_mechanistic_sources.py \
  --registry knowledge/source_registry.yaml \
  download --source wikibooks_organic_chemistry \
  --output knowledge/raw \
  --proxy http://127.0.0.1:7890
```

Offline MediaWiki import accepts `.xml`, `.txt` or compatible `.json` files and passes the same content validator:

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

Every passage retains source, locator, revision, license, exact evidence hash, artifact provenance and source-quality metadata. The final corpus and index are frozen before test evaluation.

### Broad organic/gas-phase textbook corpus v2

The larger development build and independent OpenStax question queue are
documented in
[`docs/TEXTBOOK_CORPUS_V2.md`](../docs/TEXTBOOK_CORPUS_V2.md). It keeps prose
and evaluator-only answers in separate assets; the original 32-passage pilot
corpus is not overwritten. Experimental spectra are outside this corpus
contract.

## Evidence extraction queue

For the broad v2 passage corpus, prepare deterministic, provenance-bound
knowledge-extraction tasks with:

```bash
python scripts/extract_textbook_knowledge.py prepare \
  --corpus knowledge/organic_textbook_corpus_v2/passages.jsonl \
  --corpus-manifest knowledge/organic_textbook_corpus_v2/passages.manifest.json \
  --spec knowledge/corpus_v2_spec.yaml \
  --accept-noncommercial \
  --output knowledge/candidates/textbook_knowledge_v1/tasks.jsonl
```

The non-commercial flag is an explicit license-layer acknowledgement; omit it
for the redistributable-only layer. Model responses are still unreviewed
candidates and can be checked with the script's `validate` subcommand. Neither
preparation nor validation releases an anchor or establishes chemical truth.

The legacy raw-artifact queue remains available for compatibility:

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
  -> source health and license audit
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
