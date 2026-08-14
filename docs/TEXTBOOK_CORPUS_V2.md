# Organic textbook corpus v2

## Scope

This pipeline builds provenance-preserving explanatory text for organic
chemistry, including solution-phase mechanisms and limited gas-phase ion
chemistry. It is retrieval evidence, not reaction supervision, mechanism
ground truth, or an executor validity oracle. Experimental spectra are not part
of this corpus.

The original 32-passage pilot remains immutable. The v2 build writes a separate
snapshot and keeps redistributable and non-commercial sources in distinct
layers.

## Sources and licenses

The allowlisted sources are declared in `knowledge/source_registry.yaml` and
the release policy is frozen in `knowledge/corpus_v2_spec.yaml`.

| Layer | Initial sources | Policy |
|---|---|---|
| Redistributable | Wikibooks Organic Chemistry and selected related pages | Preserve attribution, revision and share-alike metadata |
| Non-commercial research | OpenStax Organic Chemistry | Separate CC BY-NC-SA snapshot; explicit acknowledgement required |
| External reference only | Sources without bulk-redistribution permission | Store locators only; do not mirror into the corpus |

## Build

Install the knowledge dependencies:

```bash
pip install -e '.[knowledge]'
```

Extract revisioned Wikibooks pages and the separately licensed OpenStax book:

```bash
python scripts/download_wikibooks_dump.py \
  --output knowledge/raw_corpus_v2

python scripts/download_wikibooks_collections.py \
  --output knowledge/raw_corpus_v2

python scripts/download_openstax_book.py \
  --accept-noncommercial \
  --output knowledge/raw_corpus_v2
```

Build section-bounded passages, audit them, index them, and split license
layers:

```bash
python scripts/build_textbook_corpus.py \
  --download-root knowledge/raw_corpus_v2 \
  --output knowledge/organic_textbook_corpus_v2/passages.jsonl \
  --spec knowledge/corpus_v2_spec.yaml \
  --strict-artifacts \
  --min-chars 160 --max-chars 1200 --overlap-chars 120

python scripts/audit_textbook_corpus.py \
  --corpus knowledge/organic_textbook_corpus_v2/passages.jsonl \
  --spec knowledge/corpus_v2_spec.yaml \
  --output outputs/corpus_v2/audit.json

python scripts/index_textbook_corpus.py \
  --corpus knowledge/organic_textbook_corpus_v2/passages.jsonl \
  --output knowledge/organic_textbook_corpus_v2/bm25_index.json

python scripts/split_textbook_corpus_layers.py \
  --corpus knowledge/organic_textbook_corpus_v2/passages.jsonl \
  --output-dir knowledge/organic_textbook_corpus_v2
```

The cleaner parses source structure before chunking. It excludes exercises,
answer keys, navigation, references, figure-only content and page furniture.
Every passage retains source, revision, license, locator, evidence hash,
artifact hash and automatic quality flags. Topic, phase and modality labels are
candidate metadata rather than chemical truth.

## Knowledge extraction

Prepare deterministic extraction tasks from the frozen corpus:

```bash
python scripts/extract_textbook_knowledge.py prepare \
  --corpus knowledge/organic_textbook_corpus_v2/passages.jsonl \
  --corpus-manifest knowledge/organic_textbook_corpus_v2/passages.manifest.json \
  --spec knowledge/corpus_v2_spec.yaml \
  --accept-noncommercial \
  --output knowledge/candidates/textbook_knowledge_v1/tasks.jsonl
```

Model responses remain unreviewed candidates. Validate their schema and
provenance with:

```bash
python scripts/extract_textbook_knowledge.py validate \
  --tasks knowledge/candidates/textbook_knowledge_v1/tasks.jsonl \
  --responses outputs/textbook_knowledge_v1/model_responses.jsonl \
  --output knowledge/candidates/textbook_knowledge_v1/validated_candidates.jsonl
```

Validation does not establish chemical correctness. Promotion to a knowledge
anchor still requires independent wording, chemistry review and deterministic
replay for any proposed executable move.

## Evaluation isolation

OpenStax exercises and answer keys are extracted as evaluation-only artifacts
and excluded from the retrieval corpus. Model-facing questions and evaluator-
only answers are stored separately. Any final test must freeze source hashes,
passage hashes, split membership, retrieval parameters, context budget and
model checkpoint before evaluation.
