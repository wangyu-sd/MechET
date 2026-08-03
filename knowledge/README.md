# MechET mechanistic knowledge assets

This directory contains provenance records and reviewed mechanism primitives used as optional **soft guidance**. It is not a mirror of commercial textbooks and it is not a whole-reaction template database.

## Layout

```text
knowledge/
  source_registry.yaml                 # source URLs, licenses and download gates
  primitives/
    core_polar_primitives.yaml         # reviewed primitive records
  raw/                                 # downloaded locally; not committed by default
  candidates/                          # evidence-linked extraction tasks
  manifests/                           # SHA-256 and revision manifests
```

## Source policy

Every source is registered before download. The registry records source type, URL, license, redistribution constraints and whether explicit non-commercial or restricted acceptance is required.

| Source | Role | Automated use |
|---|---|---|
| IUPAC Gold Book individual terms | standard terminology | per-term JSON download with canonical-code validation |
| RXNO | named-reaction taxonomy | official OWL download |
| Wikibooks Organic Chemistry | open textbook explanations | revision-aware MediaWiki download with REST/API/export/raw fallback |
| LibreTexts Organic Synthesis (Shea) | selected CC BY textbook pages | page-list, text-only download after license marker check |
| MIT OpenCourseWare Organic Chemistry | non-commercial course evidence | explicit `--accept-noncommercial`; third-party markers excluded |
| PMechDB/PMechRP | elementary steps and textbook-pathway benchmark | manual upstream request only; no automatic derivative redistribution |

Downloaded material is evidence for candidate extraction. A Web page, LLM extraction or database row does not automatically become a released primitive.

## Download

Install the optional knowledge dependencies first:

```bash
pip install -e ".[knowledge]"
```

Inspect the registry and preview a download plan:

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
  --output knowledge/raw \
  --dry-run
```

Remove `--dry-run` to download public sources and write the hash manifest.

### Gold Book canonical codes

The Gold Book may retire or redirect historical term codes. The registry stores current canonical codes and may retain historical aliases under `term_aliases`. The downloader records all three identifiers when they differ:

```text
configured_term_id
requested_term_id
canonical_term_id
```

For example, historical `R05171` is resolved to the current `I03096` entry for **intermediate**. Artifacts are saved under the canonical code, so old redirects do not create duplicate or misleading filenames.

### Wikibooks network fallbacks

The default Wikibooks order is:

```text
MediaWiki REST source
→ Action API
→ Special:Export XML
→ action=raw
```

Each successful artifact records `retrieval_backend`, revision metadata when available, and all earlier backend errors. Wikimedia requires an identifiable User-Agent; the default includes the MechET repository URL.

If a local proxy is required, either configure the standard environment variables:

```bash
export HTTPS_PROXY=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
```

or pass it explicitly:

```bash
python scripts/download_mechanistic_sources.py \
  --registry knowledge/source_registry.yaml \
  download --source wikibooks_organic_chemistry \
  --output knowledge/raw \
  --proxy http://127.0.0.1:7890
```

You can force or reorder backends during diagnosis:

```bash
python scripts/download_mechanistic_sources.py \
  --registry knowledge/source_registry.yaml \
  download --source wikibooks_organic_chemistry \
  --output knowledge/raw \
  --mediawiki-backend export \
  --mediawiki-backend raw \
  --retries 5 --timeout 90
```

If all Wikimedia endpoints are unreachable from the current network, download a page through `Special:Export` on another machine and save it locally as:

```text
<slug(page title)>.xml
```

For example:

```text
Organic_Chemistry_Carbonyls.xml
```

Then import it without network access:

```bash
python scripts/download_mechanistic_sources.py \
  --registry knowledge/source_registry.yaml \
  download --source wikibooks_organic_chemistry \
  --output knowledge/raw \
  --mediawiki-import-dir knowledge/manual/wikibooks
```

The offline directory may contain `.xml` Special:Export files, `.txt` wikitext files, or `.json` files previously produced by the REST/downloader schema.

### Restricted sources

Non-commercial material requires explicit acknowledgement:

```bash
python scripts/download_mechanistic_sources.py \
  --registry knowledge/source_registry.yaml \
  download --source mit_ocw_organic_chemistry \
  --output knowledge/raw --accept-noncommercial
```

PMechDB/PMechRP remain manual-gated:

```bash
python scripts/download_mechanistic_sources.py \
  --registry knowledge/source_registry.yaml \
  download --source pmechdb_pmechrp \
  --output knowledge/raw \
  --accept-noncommercial --accept-restricted
```

The command writes instructions; it does not bypass the upstream request form.

Verify local hashes:

```bash
python scripts/download_mechanistic_sources.py \
  --registry knowledge/source_registry.yaml \
  verify --output knowledge/raw
```

## Evidence extraction queue

```bash
python scripts/build_primitive_extraction_queue.py \
  --download-root knowledge/raw \
  --output knowledge/candidates/extraction_queue.jsonl
```

Each task retains source ID, URL, revision, license, artifact hash, exact evidence span and a strict extraction schema. LLM output is an unreviewed candidate, never chemical truth.

## Primitive records

A primitive record contains a stable ID/version, independently written description, atom-role SMARTS, source-to-sink `E_MOVE` templates, preconditions, warnings, competitors, provenance and review status.

The first release contains a compact polar-chemistry seed library. `executor_verified` entries can support executable move matching; `text_supported` entries are retrieval-only context.

## Release workflow

```text
registered source
  -> downloaded evidence with revision/hash
  -> evidence-linked extraction candidate
  -> independent SMARTS and E_MOVE encoding
  -> deterministic executor replay
  -> chemistry review
  -> released versioned primitive
```

Never copy protected textbook prose or figures into released records. Commercial textbooks may be consulted by human reviewers, but released descriptions and schemas must be independently authored.
