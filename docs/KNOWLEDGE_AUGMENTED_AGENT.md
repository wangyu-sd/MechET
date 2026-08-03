# Knowledge-augmented trace-owned inverse agent

## Contract

`KnowledgeAugmentedAgentEnv` combines:

```text
natural-language textbook retrieval
optional structured executable anchors
trace-owned electron-flow actions
finish_trace compilation
formal executor
```

Knowledge is external soft evidence. It does not contribute direct reward, does not return a precursor and cannot override an executor failure.

## Tools

The environment inherits the trace-owned tools and adds:

```text
retrieve_textbook_guidance(query, top_k, max_characters)
retrieve_primitives(query, top_k)  # optional anchor condition
```

`retrieve_textbook_guidance` returns a bounded evidence context, passage metadata, retrieval scores, matched state/query terms and source citations. All retrieval events and context hashes are retained in the rollout trace.

`retrieve_primitives` is disabled by default in the natural-language condition. It can be enabled for the structured-anchor and combined ablations.

## Training

Build the corpus and index first:

```bash
python scripts/build_textbook_corpus.py \
  --download-root knowledge/raw \
  --output knowledge/corpus/passages.jsonl

python scripts/index_textbook_corpus.py \
  --corpus knowledge/corpus/passages.jsonl \
  --output knowledge/corpus/bm25_index.json
```

Dry-run:

```bash
python scripts/train_inverse_agent_knowledge.py \
  --config configs/knowledge/inverse_textbook_trace_grpo.yaml \
  --dry-run --limit 8
```

Training:

```bash
python scripts/train_inverse_agent_knowledge.py \
  --config configs/knowledge/inverse_textbook_trace_grpo.yaml
```

## Ablation switches

```text
auto_textbook_on_reset
textbook_top_k
textbook_max_characters
textbook_max_per_source
enable_structured_primitives
primitive_top_k
```

Textbook retrieval itself has no reward switch because natural-language matching must never be rewarded directly.

## Required comparisons

```text
trace-owned tool actor without knowledge
length-matched irrelevant context
textbook RAG
structured anchors only
textbook RAG + structured anchors
gold passage upper bound
```

All variants must share the same model, training rows, assistant-token budget, tool budget, executor, forward checkpoint and optimization schedule.
