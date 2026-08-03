# Matched evidence-layer experiments

## Scientific question

> Does external mechanistic evidence improve induction of a causally grounded electron-flow program beyond trace ownership and extra context alone?

Evidence is an H3 intervention, not a parallel endpoint generator or hard verifier.

## Frozen conditions

| Condition | Endpoint path | Evidence |
|---|---|---|
| `trace_no_knowledge` | trace-owned | none |
| `trace_length_matched_irrelevant` | trace-owned | irrelevant text, exact character budget |
| `trace_textbook_rag` | trace-owned | frozen textbook evidence |
| `trace_structured_anchors` | trace-owned | frozen structured anchors |
| `trace_text_plus_anchors` | trace-owned | both evidence types |
| `direct_textbook_rag` | direct answer | the same bounded textbook evidence |

Only two replay-verified source datasets are manually built; the remaining four conditions are derived automatically by `build_knowledge_ablation_suite.py`.

## Fairness contract

Headline comparisons require:

```text
same frozen stable-ID universe
same targets and endpoint references
same base-model family and revision
same examples and optimizer update schedule
reported tokenizer input and assistant-mask tokens
same generation temperature, top-p, K, token, and iteration budgets
evidence direct reward = zero
condition-specific adapter hashes reported
```

Direct and tool syntax lengths differ. Use supervised-token-normalized compute rather than claiming raw token equality.

## Evidence retrieval contract

The main textbook query uses target/current-state terms available at inference. Gold reaction-family queries are permitted only in a condition explicitly marked `label_oracle`; the contract validator rejects them elsewhere.

During H3 prediction, evidence modes replay the exact row-specific bounded evidence result. This prevents retrieval randomness or corpus changes from becoming a hidden condition difference.

## Build and validate

```bash
python scripts/build_knowledge_ablation_suite.py \
  --config configs/experiments/textbook_ablation.yaml

python scripts/validate_experiment_contract.py \
  --model-name Qwen/Qwen3-0.6B \
  --condition trace_no_knowledge=data/knowledge_ablation/v2/trace_no_knowledge.jsonl \
  --condition trace_length_matched_irrelevant=data/knowledge_ablation/v2/trace_length_matched_irrelevant.jsonl \
  --condition trace_textbook_rag=data/knowledge_ablation/v2/trace_textbook_rag.jsonl \
  --condition trace_structured_anchors=data/knowledge_ablation/v2/trace_structured_anchors.jsonl \
  --condition trace_text_plus_anchors=data/knowledge_ablation/v2/trace_text_plus_anchors.jsonl \
  --condition direct_textbook_rag=data/knowledge_ablation/v2/direct_textbook_rag.jsonl \
  --output outputs/contracts/evidence_conditions.json
```

## Training

Six SFT configs are provided. The primary comparison should first be completed at the Tool-SFT stage. Optional GRPO configs exist for trace-only, textbook, anchors, and combined trace conditions; each loads its corresponding Tool-SFT adapter manifest.

## Prediction artifacts

Use `scripts/infer_mechet.py` with modes:

```text
trace
irrelevant
textbook
anchors
combined
direct
```

Every output row must have `artifact_type=prediction`. The evaluator rejects supervision rows, duplicate IDs, and IDs absent from the reference set. Missing predictions are retained as failures.

## Evidence-content interventions

```bash
python scripts/build_evidence_interventions.py \
  --input data/knowledge_ablation/v2/trace_text_plus_anchors.jsonl \
  --output-dir data/evidence_interventions/v2 \
  --intervention passage_shuffle \
  --intervention same_topic_wrong \
  --intervention remove_warnings \
  --intervention remove_competing_pathways
```

The intervention builder preserves:

```text
stable IDs
targets and endpoints
chemistry trace
tool budget
context character budget
zero direct reward
```

`same_topic_wrong` requires reviewed/shared retrieval terms and fails rather than silently selecting an unrelated donor.

## Evaluation

```bash
python scripts/evaluate_knowledge_ablation.py \
  --reference data/knowledge_ablation/v2/trace_textbook_rag.jsonl \
  --condition trace_no_knowledge=outputs/h3/trace.jsonl \
  --condition trace_length_matched_irrelevant=outputs/h3/irrelevant.jsonl \
  --condition trace_textbook_rag=outputs/h3/textbook.jsonl \
  --condition trace_structured_anchors=outputs/h3/anchors.jsonl \
  --condition trace_text_plus_anchors=outputs/h3/combined.jsonl \
  --condition direct_textbook_rag=outputs/h3/direct.jsonl \
  --output outputs/h3/summary.json
```

The evaluator recomputes trace and proof results, uses structural exact match as the primary endpoint, reports mapped exact separately, and checks a matched base/revision/generation contract across conditions.

Implemented metrics:

```text
structural and mapped Top-1/5/10
ExecutePass@K and TraceBoundPass@K
coverage, selective risk, abstention
tool-failure recovery
retrieval Recall@K and Precision@K when gold passage IDs exist
retrieval latency
missing prediction and re-execution error rates
```

Reaction-center and synthon metrics remain null until frozen labels exist.

## Claim gates

Textbook evidence:

```text
trace_textbook_rag > trace_no_knowledge
and
trace_textbook_rag > trace_length_matched_irrelevant
```

Combined evidence:

```text
trace_text_plus_anchors > trace_textbook_rag
and
trace_text_plus_anchors > trace_structured_anchors
```

Additional requirements:

```text
all frozen IDs evaluated
trace paths fully re-executable
zero evidence reward violations
matched runtime contract
passage/tool interventions reported
```

A gain explained by context presence, query leakage, missing predictions, runtime mismatch, or post-test evidence editing does not support H3.
