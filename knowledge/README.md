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
| IUPAC Gold Book individual terms | standard terminology | per-term JSON download |
| RXNO | named-reaction taxonomy | official OWL download |
| Wikibooks Organic Chemistry | open textbook explanations | revision-pinned MediaWiki text |
| LibreTexts Organic Synthesis (Shea) | selected CC BY textbook pages | page-list, text-only download after license marker check |
| MIT OpenCourseWare Organic Chemistry | non-commercial course evidence | explicit `--accept-noncommercial`; third-party markers excluded |
| PMechDB/PMechRP | elementary steps and textbook-pathway benchmark | manual upstream request only; no automatic derivative redistribution |

Downloaded material is evidence for candidate extraction. A Web page, LLM extraction or database row does not automatically become a released primitive.

## Download

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
