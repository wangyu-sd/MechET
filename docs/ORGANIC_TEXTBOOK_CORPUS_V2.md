# Organic textbook and gas-phase evidence corpus v2

## Purpose and separation

The v2 evidence build deliberately produces two assets:

```text
organic_textbook_corpus_v2
  explanatory prose for solution-phase and gas-phase organic chemistry

spectral_evidence_v1
  structured experimental mass spectra and acquisition metadata
```

A spectrum is not converted into pseudo-text and a textbook paragraph is not
treated as an experimental spectrum. Both are optional evidence; neither can
override the MechET executor.

The existing 32-passage corpus remains a pilot snapshot. Building v2 never
overwrites it or changes an already-running training job.

## Source layers

| Layer | Initial sources | Release policy |
|---|---|---|
| Redistributable textbook | Wikibooks Organic Chemistry, individual IUPAC Gold Book terms | Preserve attribution, revision and share-alike metadata |
| Non-commercial textbook | OpenStax Organic Chemistry, 10th edition | Separate NC snapshot; do not merge into a commercial-use release |
| Analytical/gas-phase supplements | Wikibooks analytical methods, gas-ion chemistry, solvation and electrochemistry | Candidate context with low/caution weights; review before headline use |
| Structured spectra | MassBank-data 2026.03 | CC BY 4.0, separate JSONL records |
| External reference only | NIST Chemistry WebBook SRD 69 | Do not bulk mirror or redistribute |

The registry is an allowlist. A crawl may follow only pages whose effective
license and source policy are known. Page furniture, figures without compatible
licenses, answer keys, and exercises are not silently mixed into training text.

## Extraction

Install the knowledge dependencies:

```bash
pip install -e '.[knowledge]'
```

Extract the complete Wikibooks Organic Chemistry subtree from the official
revisioned XML dump:

```bash
python scripts/download_wikibooks_dump.py \
  --output knowledge/raw_corpus_v2

python scripts/download_wikibooks_collections.py \
  --output knowledge/raw_corpus_v2
```

Extract OpenStax page structure. This is a separately marked non-commercial
research layer:

```bash
python scripts/download_openstax_book.py \
  --accept-noncommercial \
  --output knowledge/raw_corpus_v2
```

Build clean, section-bounded passages and a frozen BM25 index:

```bash
python scripts/build_textbook_corpus.py \
  --download-root knowledge/raw_corpus_v2 \
  --output knowledge/organic_textbook_corpus_v2/passages.jsonl \
  --spec knowledge/corpus_v2_spec.yaml --strict-artifacts \
  --min-chars 160 --max-chars 1200 --overlap-chars 120

python scripts/index_textbook_corpus.py \
  --corpus knowledge/organic_textbook_corpus_v2/passages.jsonl \
  --output knowledge/organic_textbook_corpus_v2/bm25_index.json

python scripts/audit_textbook_corpus.py \
  --corpus knowledge/organic_textbook_corpus_v2/passages.jsonl \
  --spec knowledge/corpus_v2_spec.yaml \
  --output outputs/corpus_v2/audit.json

python scripts/split_textbook_corpus_layers.py \
  --corpus knowledge/organic_textbook_corpus_v2/passages.jsonl \
  --output-dir knowledge/organic_textbook_corpus_v2
```

The cleaner parses MediaWiki headings before removing templates, links,
references and navigation. OpenStax pages are extracted from `#main-content`
and retain chapter/section provenance. Chunking never crosses a parsed section.
Exercise nodes, answer keys, glossaries, bibliographies, URL-heavy reference
lists and figure-only content are excluded. Topic, phase and modality labels are
automatic candidate tags, not chemical ground truth.

Build provenance-bound knowledge-extraction tasks from every eligible passage:

```bash
python scripts/extract_textbook_knowledge.py prepare \
  --corpus knowledge/organic_textbook_corpus_v2/passages.jsonl \
  --corpus-manifest knowledge/organic_textbook_corpus_v2/passages.manifest.json \
  --spec knowledge/corpus_v2_spec.yaml \
  --accept-noncommercial \
  --output knowledge/candidates/textbook_knowledge_v1/tasks.jsonl
```

`--accept-noncommercial` is required to include the separately licensed
OpenStax layer. Omit it for a redistributable-only queue. Each task contains the
exact passage and hash, source/revision/license, candidate topic/phase tags, a
strict JSON extraction schema and prompts requiring `UNKNOWN` for unsupported
fields. The deterministic queue remains `unreviewed_extraction_task`; it is
never promoted automatically to an executable knowledge anchor.

After running any model backend, schema- and provenance-check its JSONL output:

```bash
python scripts/extract_textbook_knowledge.py validate \
  --tasks knowledge/candidates/textbook_knowledge_v1/tasks.jsonl \
  --responses outputs/textbook_knowledge_v1/model_responses.jsonl \
  --output knowledge/candidates/textbook_knowledge_v1/validated_candidates.jsonl
```

Validation does not establish chemical correctness. Promotion still requires
independent wording, chemistry review and deterministic replay for any proposed
executable electron moves.

## Gas-phase and mass-spectral asset

MassBank is downloaded from the versioned Zenodo release and parsed without
discarding acquisition metadata:

```bash
python scripts/build_massbank_spectral_evidence.py \
  --output-dir knowledge/spectral_evidence_v1
```

Every record retains the accession, structure identifiers, formula, precursor,
ion mode, ionization method, collision energy, instrument and peak list. The
default split groups cyclic molecules by Bemis–Murcko scaffold and uses compound
identity for acyclic molecules. Identity and scaffold nodes are merged into
connected components before deterministic balanced 80/10/10 assignment. A
secondary exact-connectivity split is also stored. Both prevent replicate
spectra of one compound from appearing across train and test.

EI, positive ESI-MS/MS, negative ESI-MS/MS and other ionization regimes must be
reported separately. Their ion chemistry and experimental biases are not
interchangeable.

```bash
python scripts/audit_spectral_evidence.py \
  --records knowledge/spectral_evidence_v1/records.jsonl \
  --manifest knowledge/spectral_evidence_v1/manifest.json \
  --output outputs/corpus_v2/spectral_audit.json
```

## Evaluation

The frozen protocol is declared in
`configs/experiments/organic_textbook_eval_v1.yaml`.

### Retrieval

Use human-reviewed queries stratified across molecular-state retrieval,
reaction mechanisms, physical-organic concepts, and gas-phase mass
spectrometry. Report Recall@K, MRR, nDCG, source diversity and citation
precision. Empty, irrelevant and same-topic-wrong queries are required controls.

### Organic chemistry questions

OpenStax chapter exercises and answer keys are downloaded as evaluation-only
artifacts and excluded from the retrieval corpus. The builder physically
separates `text_only_questions.jsonl` (model-facing) from `answer_key.jsonl`
(evaluator-only). Only independently reviewed, text-only questions enter the
scored set. Compare closed-book, irrelevant-text, retrieved-textbook and
oracle-section conditions. Hold chapters 28–31 frozen for the final test.

```bash
python scripts/build_openstax_question_eval.py \
  --raw-root knowledge/raw_corpus_v2/openstax_organic_chemistry_10e \
  --output-dir data/openstax_organic_eval_v1

python scripts/audit_openstax_eval_leakage.py \
  --questions data/openstax_organic_eval_v1/text_only_questions.jsonl \
  --corpus knowledge/organic_textbook_corpus_v2/passages.jsonl \
  --output outputs/corpus_v2/openstax_leakage_audit.json
```

### MechET H3

Rebuild the six matched H3 conditions after the corpus is frozen. The primary
claim remains improvement in executor-derived endpoints, not fluent textbook
recitation. Textbook evidence must beat both trace-only and length-matched
irrelevant text and must be sensitive to passage shuffling.

### Mass spectrometry

Use scaffold-held-out MassBank records for library retrieval, formula
prediction, structure ranking and fragmentation explanation. In addition to
top-k identity, report structural similarity, explained peak intensity, peak
recall, mass balance and unsupported claims. Instrument-, laboratory- and
collision-energy-held-out evaluations are secondary stress tests.

The current solution-phase MechET environment is not a formal verifier for
radical-cation fragmentation. A gas-phase fragment executor would be a new
environment with charge/radical bookkeeping and its own replay tests.

## Freeze rule

Before a frozen test is examined, freeze source revisions, licenses, raw hashes,
passage and spectrum hashes, splits, questions, retrieval parameters, context
budget, checkpoint and decoding settings. Corpus changes prompted by final-test
errors invalidate the corresponding headline result.

## Current development snapshot (2026-08-12)

| Asset | Current size | Automatic status |
|---|---:|---|
| Textbook prose | 2,385 passages, 7 sources | Coverage/markup/duplicate gates pass; stratified human review remains open |
| Redistributable / NC layers | 801 / 1,584 passages | Physically separated; combined file is research-only |
| Solution/gas tags | 196 solution-phase, 39 gas-phase passages | Candidate tags only |
| MassBank spectra | 139,240 records; 18,523 identities | Exact 111,392 / 13,924 / 13,924 split; zero identity/scaffold leakage |
| OpenStax paired exercises | 561 paired; 220 text-only | Zero exact question-to-corpus matches; all remain unreviewed |

The textbook corpus digest is
`d68378fff7af2ea089e59ae17eb2db547bf3dc1264ddef27fce041caccd2cad6`.
It remains `development`, not `frozen`, until the 80-passage review sheet and
question rubrics have been independently checked.
