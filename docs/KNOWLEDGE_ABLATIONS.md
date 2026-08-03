# Fair textbook-knowledge ablations

## Goal

Textbook knowledge is an explicit experimental variable. MechET must not compare an open-book knowledge condition with closed-book baselines and attribute the entire gain to the agent architecture.

The primary study therefore separates:

```text
trace-owned execution benefit
textbook text benefit
structured-anchor benefit
extra-context/token benefit
combined knowledge benefit
```

## Frozen conditions

The reference suite is `configs/experiments/textbook_ablation.yaml`.

| Condition | Trace tools | Textbook text | Structured anchors | Purpose |
|---|---:|---:|---:|---|
| `trace_no_knowledge` | yes | no | no | closed-book trace baseline |
| `trace_length_matched_irrelevant` | yes | irrelevant, same character budget | no | extra-context control |
| `trace_textbook_rag` | yes | retrieved | no | natural-language knowledge effect |
| `trace_structured_anchors` | yes | no | yes | executable-anchor effect |
| `trace_text_plus_anchors` | yes | retrieved | yes | combined condition |
| `direct_textbook_rag` | no | the same retrieved evidence | no | fair open-book direct baseline |

A frozen gold-passage condition is reported as an upper bound when gold passage labels are available, not as the default method.

## Build matched datasets

```bash
python scripts/build_knowledge_ablation_suite.py \
  --config configs/experiments/textbook_ablation.yaml
```

The builder:

- intersects stable IDs across all source conditions;
- validates identical targets and expected structural endpoints;
- derives the no-knowledge condition by removing only knowledge calls/results;
- derives length-matched irrelevant context by rotating evidence across different targets;
- preserves the chemistry tool trajectory;
- writes all condition files, hashes and a suite manifest.

The irrelevant-context control matches the exact evidence-card character count. It is not described as chemically neutral truth; it is only a control for context length and generic additional prose.

## Supervised training first

Reference Tool-SFT command:

```bash
python scripts/train_tool_sft.py \
  --config configs/knowledge/tool_sft_textbook.yaml \
  --dry-run

python scripts/train_tool_sft.py \
  --config configs/knowledge/tool_sft_textbook.yaml
```

All matched conditions must share:

```text
base-model and tokenizer revision
stable training IDs
LoRA rank and target modules
optimizer and schedule
number of updates
assistant-token budget
context-character budget where applicable
tool-call budget for tool conditions
random seeds
```

On-policy training begins only after Tool-SFT shows a credible executable-learning signal.

## Evaluation

```bash
python scripts/evaluate_knowledge_ablation.py \
  --condition trace_none=outputs/trace_none/predictions.jsonl \
  --condition irrelevant=outputs/irrelevant/predictions.jsonl \
  --condition textbook=outputs/textbook/predictions.jsonl \
  --output outputs/knowledge_ablation/summary.json
```

Required endpoint and process metrics:

```text
structural precursor Top-1/5
reaction-center accuracy
synthon exact match
execute rate
trace–endpoint consistency
tool-failure recovery
abstention and risk–coverage
```

Required knowledge metrics:

```text
retrieval Recall@K and Precision@K
passage rank
gold-passage rank when available
citation correctness
context length and retrieval latency
textbook/anchor tool-call rate
knowledge direct-reward violations
```

## Causal controls

```text
shuffle passages
length-matched irrelevant context
same-topic but wrong passage
remove warnings
remove competing-pathway text
remove tool observations
```

The textbook claim is supported only if the actual retrieval condition exceeds both the trace-only and length-matched irrelevant-context controls under matched data and compute.

## Stopping rules

Do not attribute gains to textbook knowledge when:

- the matched stable-ID intersection differs;
- context or optimization budgets differ without disclosure;
- the irrelevant-context control explains the gain;
- passage shuffling has no material effect;
- knowledge retrieval receives direct reward;
- trace binding or execution reliability degrades;
- the primitive/textbook assets were changed after viewing final test failures.
