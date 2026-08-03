# Natural-language mechanistic evidence retrieval

## Scientific role

Textbook retrieval is an intervention for H3, not part of the causal endpoint computation.

```text
bounded external passage
  -> optional guidance for action selection
  -> explicit electron-flow actions
  -> trace-owned execution
  -> finish_trace
  -> executor-derived precursor
```

The passage does not define reaction validity, does not return the precursor, receives no direct reward and cannot override a deterministic failure.

## Why natural-language evidence

Fully compiling broad mechanism literature into executable structured records is costly and brittle. MechET therefore separates:

```text
electron-flow execution primitives        causal action vocabulary
natural-language textbook passages        broad soft evidence
mechanistic knowledge anchors             optional structured soft evidence
deterministic executor                    hard formal authority
```

These objects have different scientific roles and must not be conflated.

## Corpus

Build from provenance-tracked downloads:

```bash
python scripts/build_textbook_corpus.py \
  --download-root knowledge/raw \
  --output knowledge/corpus/passages.jsonl
```

Each passage retains:

```text
passage ID
source ID and URL
locator and revision
license
exact evidence hash
artifact path and hash
optional topics and functional-group tags
```

Downloaded text remains evidence, not automatically accepted chemistry.

## Frozen deterministic index

```bash
python scripts/index_textbook_corpus.py \
  --corpus knowledge/corpus/passages.jsonl \
  --output knowledge/corpus/bm25_index.json
```

`TextbookRetriever` combines:

```text
BM25 lexical relevance
functional-group and molecular-state term overlap
source-diversity limits
stable ordering
```

The first experiment uses this deterministic retriever so the corpus, query contract and ranking can be frozen. Dense retrieval is a separate later ablation.

Molecular-state terms improve retrieval only. They are not source/sink labels, reaction-family truth or formal actions.

## Bounded evidence cards

`compile_evidence_context` emits:

```text
title
retrieval score
matched query and state terms
sanitized evidence text
source, locator, revision, license and passage ID
explicit soft-evidence boundary
```

Controls:

- chat-role markers are neutralized;
- the exact context is character bounded;
- only the sanitized bounded card is model visible;
- the context is hashed;
- raw unbounded passage text is not duplicated in retrieval metadata.

## Matched conditions

The textbook-only source rows and textbook-plus-anchor source rows feed the six-condition suite:

```text
trace_no_knowledge
trace_length_matched_irrelevant
trace_textbook_rag
trace_structured_anchors
trace_text_plus_anchors
direct_textbook_rag
```

```bash
python scripts/build_knowledge_ablation_suite.py \
  --config configs/experiments/textbook_ablation.yaml
```

The direct model receives the same bounded evidence card as the trace-owned model. The irrelevant condition rotates only evidence text while preserving the target, query and chemistry trajectory.

## Evaluation

Retrieval metrics:

```text
Recall@K and Precision@K
passage rank
gold-passage rank when labels exist
source diversity
context characters and tokenizer-specific tokens
latency
citation correctness
```

Model metrics:

```text
endpoint and execution performance
trace–endpoint consistency
knowledge-call rate
tool-failure recovery
abstention and risk–coverage
```

## Causal evidence interventions

```text
length-matched irrelevant passage
passage shuffle
same-topic wrong passage
warnings removed
competing-pathway text removed
```

A textbook evidence claim requires improvement over both trace-only and irrelevant-text controls, plus sensitivity to passage content. Improvement caused only by longer context, different IDs, extra optimization or direct reward does not support the claim.

## Freeze and stopping rules

Before final-test evaluation freeze:

```text
source registry and licenses
raw artifact and corpus hashes
chunking config
retrieval config and index hash
query-generation contract
evidence-card size
matched stable-ID manifest
```

Stop or narrow the claim when passage shuffling has no material effect, irrelevant text explains the gain, retrieval degrades trace execution, or the corpus is changed after observing final-test failures.

## Boundaries

A passage may summarize prior chemical knowledge, but it does not prove a curved arrow, full mechanism, selectivity, condition compatibility, kinetics, yield or experimental feasibility.
