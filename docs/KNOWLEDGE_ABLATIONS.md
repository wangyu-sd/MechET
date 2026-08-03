# Matched evidence-layer experiments

## Scientific question

> Does external mechanistic evidence improve induction of a causally grounded electron-flow program beyond trace ownership and extra context alone?

Knowledge is not a parallel headline innovation. It is an intervention used to test H3: formal executability and empirical chemical support are distinct evidence layers.

The primary study separates:

```text
trace ownership benefit
extra-context benefit
natural-language textbook evidence
structured mechanistic knowledge-anchor evidence
combined evidence
direct open-book generation
```

## Terminology

- **electron-flow execution primitive:** local executable action used by the trace and composition-OOD split;
- **mechanistic knowledge anchor:** structured retrieval record with role bindings, candidate moves, warnings, competitors and provenance;
- **soft evidence:** text, anchor matches or learned scores that cannot override execution.

## Frozen conditions

The reference suite is `configs/experiments/textbook_ablation.yaml`.

| Condition | Trace-owned endpoint | Textbook evidence | Knowledge anchors | Purpose |
|---|---:|---:|---:|---|
| `trace_no_knowledge` | yes | no | no | causal trace baseline |
| `trace_length_matched_irrelevant` | yes | irrelevant, exact character budget | no | extra-context control |
| `trace_textbook_rag` | yes | retrieved | no | natural-language evidence effect |
| `trace_structured_anchors` | yes | no | yes | structured-anchor effect |
| `trace_text_plus_anchors` | yes | retrieved | yes | combined evidence |
| `direct_textbook_rag` | no | the same bounded evidence | no | fair open-book direct baseline |

A frozen gold-passage condition is an upper bound when labels exist, not the default method.

## Automatic derivation

Only two replay-verified source datasets are required:

```text
textbook-only trace rows
textbook-plus-anchor trace rows
```

The suite automatically derives:

- no-knowledge by removing evidence tools only;
- irrelevant text by rotating evidence text across different targets while preserving the original query, target and chemistry trajectory;
- anchors-only by removing textbook retrieval from combined rows;
- direct open-book by exposing the exact same bounded evidence card to a direct-answer model without chemistry tools.

```bash
python scripts/build_knowledge_ablation_suite.py \
  --config configs/experiments/textbook_ablation.yaml
```

Manually assembled anchors-only or direct-open-book datasets are not permitted for headline comparisons.

## Matched-data contract

The builder validates identical:

```text
stable IDs
targets
expected structural precursors
row order after intersection
```

The suite manifest records:

```text
stable-ID digest
input and output hashes
assistant characters
user characters
context characters
total, chemistry and evidence tool calls
```

Validate before training:

```bash
python scripts/validate_experiment_contract.py \
  --condition none=data/knowledge_ablation/v2/trace_no_knowledge.jsonl \
  --condition irrelevant=data/knowledge_ablation/v2/trace_length_matched_irrelevant.jsonl \
  --condition textbook=data/knowledge_ablation/v2/trace_textbook_rag.jsonl \
  --condition anchors=data/knowledge_ablation/v2/trace_structured_anchors.jsonl \
  --condition combined=data/knowledge_ablation/v2/trace_text_plus_anchors.jsonl \
  --condition direct=data/knowledge_ablation/v2/direct_textbook_rag.jsonl \
  --output outputs/contracts/evidence_conditions.json
```

Final runs additionally report tokenizer-specific input and supervised token counts, optimizer steps, effective batch size and GPU hours.

## Supervised training first

```bash
python scripts/train_tool_sft.py \
  --config configs/knowledge/tool_sft_textbook.yaml \
  --dry-run

python scripts/train_tool_sft.py \
  --config configs/knowledge/tool_sft_textbook.yaml
```

On-policy training begins only after Tool-SFT shows a credible executable-learning signal on a frozen validation set.

## Evaluation

Endpoint and trace metrics:

```text
structural precursor Top-1/5
reaction-center accuracy
synthon exact match
execute rate
trace–proof consistency
trace–endpoint consistency
tool-failure recovery
abstention and risk–coverage
```

Evidence metrics:

```text
retrieval Recall@K and Precision@K
passage rank and gold-passage rank
citation correctness
context characters and tokenizer-specific tokens
retrieval latency
textbook and anchor tool-call rate
knowledge direct-reward violations
```

## Causal interventions

Required evidence interventions:

```text
length-matched irrelevant context
passage shuffling
same-topic wrong passage
remove warnings
remove competing-pathway text
```

Required trace interventions remain separate:

```text
remove tool observations
shuffle tool observations
replace observations with stale molecular states
```

The first group tests evidence use; the second tests causal dependence on the execution process.

## Claim gates

A natural-language evidence claim requires:

```text
trace_textbook_rag > trace_no_knowledge
and
trace_textbook_rag > trace_length_matched_irrelevant
```

A structured-anchor claim requires improvement over both trace-only and an equally budgeted evidence control.

A combined-evidence claim requires improvement beyond the best individual evidence condition, not merely the no-knowledge baseline.

A fair architecture claim compares trace-owned and direct-answer models under the same bounded textbook evidence and disclosed compute.

## Stopping rules

Do not attribute gains to mechanistic evidence when:

- stable IDs or endpoints differ;
- context or optimization budgets differ without disclosure;
- irrelevant text explains the gain;
- passage shuffling has no material effect;
- evidence retrieval receives direct reward;
- trace binding or execution reliability degrades;
- evidence assets are modified after final-test inspection;
- the direct baseline receives different or less evidence;
- the model is insensitive to the evidence content but sensitive only to its presence.

## Boundaries

A retrieved passage or structured anchor does not prove an electron-flow action, full mechanism, condition compatibility, selectivity, kinetics or experimental feasibility. Evidence remains subordinate to deterministic execution and external validation.
