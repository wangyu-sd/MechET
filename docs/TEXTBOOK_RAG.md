# Natural-language textbook evidence retrieval

## Motivation

Fully compiling every textbook mechanism into SMARTS and executable `E_MOVE` templates is expensive and brittle. MechET therefore separates:

```text
small core executable action grammar
+ broad natural-language textbook evidence
+ model grounding to concrete atom maps
+ deterministic execution
```

Textbook passages are soft evidence. They do not define reaction validity and do not directly produce rewards.

## Corpus

Build a corpus from provenance-tracked downloads:

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
exact text hash
artifact path/hash
optional topics and functional-group tags
```

Build and validate the deterministic index:

```bash
python scripts/index_textbook_corpus.py \
  --corpus knowledge/corpus/passages.jsonl \
  --output knowledge/corpus/bm25_index.json
```

## Retrieval

`TextbookRetriever` combines:

```text
BM25 lexical relevance
+ functional-group and molecular-state term overlap
+ source-diversity limits
```

The initial implementation is deliberately deterministic and lightweight so matched experiments can freeze the exact retrieval contract. Dense encoders may be added later as a separate ablation.

The current state analyzer detects common motifs such as carbonyls, alkenes, alkyl/aryl halides, alcohols, anions, rings and aromatic systems. These terms improve retrieval only; they are not formal reaction labels.

## Evidence cards

`compile_evidence_context` converts retrieved passages into bounded cards containing:

```text
title
retrieval score
matched query/state terms
evidence text
source, locator, revision, license and passage ID
explicit soft-evidence boundary
```

Chat-role markers and role-like tags in downloaded text are neutralized before prompt construction. The context is size bounded and hashed for exact experiment reproduction.

## Scientific use

Required comparisons:

```text
no textbook evidence
length-matched irrelevant text
actual retrieved textbook text
frozen gold passage
structured executable anchors
text + executable anchors
```

The corpus and retriever must be frozen before final test evaluation. A performance gain cannot be attributed to chemical knowledge unless it survives the irrelevant-context control and causal passage interventions.
