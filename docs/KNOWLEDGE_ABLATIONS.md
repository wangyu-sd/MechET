# H3 — matched evidence-layer experiments

> **Question:** does external mechanistic evidence improve induction of a causally grounded electron-flow program beyond trace ownership and additional context alone?  
> **Constraint:** evidence is an intervention, not a parallel endpoint generator or hard verifier

## Identification strategy

H3 is not established by comparing “with retrieval” against “without retrieval” in isolation. A credible evidence claim must separate:

```text
information value
from
context length, retrieval drift, label leakage, direct answer access, and runtime mismatch
```

The experiment therefore uses six matched conditions **inside each frozen split** and reserves the held-out `test/` universe for final evidence evaluation.

## Frozen condition matrix

| Condition | Endpoint path | Evidence | Scientific role |
|---|---|---|---|
| `trace_no_knowledge` | Trace-owned | None | Causal-program baseline |
| `trace_length_matched_irrelevant` | Trace-owned | Irrelevant text with the same character budget | Context-presence control |
| `trace_textbook_rag` | Trace-owned | Frozen textbook evidence | Natural-language evidence condition |
| `trace_structured_anchors` | Trace-owned | Frozen structured anchors | Structured evidence condition |
| `trace_text_plus_anchors` | Trace-owned | Both evidence types | Combined evidence condition |
| `direct_textbook_rag` | Direct answer | The same bounded textbook evidence | Endpoint-path control |

Only two replay-verified source datasets are built manually per split. The remaining four conditions are derived automatically by `build_knowledge_ablation_suite.py` from the same stable-ID intersection within that split.

## Split isolation

`configs/experiments/textbook_ablation.yaml` declares independent source inputs for:

```text
train
valid
test
```

The builder writes:

```text
data/knowledge_ablation/v2/train/*.jsonl
data/knowledge_ablation/v2/valid/*.jsonl
data/knowledge_ablation/v2/test/*.jsonl
```

and fails if any stable ID appears in more than one split. Tool-SFT reads only `train/`; model selection uses `valid/`; final H3 uses `test/`. A train-derived evidence suite is not an evaluation set.

## Fairness contract

Headline comparisons require:

```text
same frozen stable-ID universe within the compared split
same targets and endpoint references
same base-model family and immutable revision
same examples and optimizer-update schedule
frozen model/tokenizer and adapter lineage
reported tokenizer input and assistant-mask tokens
same global seed policy and generation budget
same temperature, top-p, candidate, token, and iteration budgets
evidence direct reward = zero
condition-specific adapter hashes reported
```

Direct and tool syntax lengths differ. Use supervised-token-normalized compute rather than claiming raw token equality.

## Retrieval contract

The headline textbook query uses only target/current-state terms available at inference. Gold reaction-family queries are permitted only in a condition explicitly marked `label_oracle`; the validator rejects them elsewhere.

During H3 prediction, evidence modes replay the exact row-specific bounded evidence result. This prevents retrieval randomness, corpus updates, or index changes from becoming hidden condition differences.

Frozen evidence records include:

```text
passage or anchor identifiers
content hash
bounded text or structured fields
provenance and source revision
retrieval score and latency
direct_reward = false
```

## Build and validate

```bash
python scripts/build_knowledge_ablation_suite.py \
  --config configs/experiments/textbook_ablation.yaml
```

Validate each split separately. For example, training:

```bash
python scripts/validate_experiment_contract.py \
  --model-name Qwen/Qwen3-0.6B \
  --condition trace_no_knowledge=data/knowledge_ablation/v2/train/trace_no_knowledge.jsonl \
  --condition trace_length_matched_irrelevant=data/knowledge_ablation/v2/train/trace_length_matched_irrelevant.jsonl \
  --condition trace_textbook_rag=data/knowledge_ablation/v2/train/trace_textbook_rag.jsonl \
  --condition trace_structured_anchors=data/knowledge_ablation/v2/train/trace_structured_anchors.jsonl \
  --condition trace_text_plus_anchors=data/knowledge_ablation/v2/train/trace_text_plus_anchors.jsonl \
  --condition direct_textbook_rag=data/knowledge_ablation/v2/train/direct_textbook_rag.jsonl \
  --output outputs/contracts/evidence_conditions_train.json
```

Repeat with `valid/` and `test/`. The validator must report stable-ID alignment, endpoint alignment, schema validity, evidence budgets, tokenizer input tokens, supervised tokens, truncation, and query leakage.

## Training contract

Six Tool-SFT configurations are provided and all read from `data/knowledge_ablation/v2/train/`.

Qwen3 Tool-SFT uses the final-sequence assistant-mask contract `final_chatml_token_scan_v1` and a frozen `max_length=12288` with zero truncation. A mutable request such as `model_revision: main` must resolve to the actual immutable 40-hex model/tokenizer commit before the adapter is accepted.

Optional GRPO configurations exist for trace-only, textbook, anchors, and combined trace conditions. Each loads its corresponding Tool-SFT adapter and validates:

```text
adapter SHA-256
immutable base-model and tokenizer revisions
data contract
environment and executor revisions
seed and data seed
training split lineage
```

From-base RL is a separately named ablation rather than a hidden initialization difference.

## Prediction artifact contract

Use `scripts/infer_mechet.py` with modes:

```text
trace
irrelevant
textbook
anchors
combined
direct
```

Every row must use `artifact_type=prediction` and record a complete runtime contract. The evaluator rejects supervision rows, duplicate IDs, and IDs absent from the frozen reference. Missing predictions remain in the denominator as failures.

Trace conditions receive credit only after an explicit successful `finish_trace`; they never fall back to parsing a free-form direct answer.

## Run the six held-out conditions

```bash
python scripts/run_h3_suite.py \
  --suite-root data/knowledge_ablation/v2/test \
  --out-dir outputs/h3 \
  --samples-per-target 4 \
  --seed 17
```

Before inference, the runner verifies that every condition-specific adapter was trained on the corresponding file under `data/knowledge_ablation/v2/train/` by comparing the adapter manifest training SHA-256 with the frozen train file. It then runs all six conditions and calls `scripts/evaluate_knowledge_ablation.py` against the held-out `test/trace_textbook_rag.jsonl` reference.

## Evidence-content interventions

```bash
python scripts/build_evidence_interventions.py \
  --input data/knowledge_ablation/v2/test/trace_text_plus_anchors.jsonl \
  --output-dir data/evidence_interventions/v2 \
  --intervention passage_shuffle \
  --intervention same_topic_wrong \
  --intervention remove_warnings \
  --intervention remove_competing_pathways
```

| Intervention | Purpose | Integrity requirement |
|---|---|---|
| `passage_shuffle` | Test dependence on sample-specific text | Different donor, same evidence format and budget |
| `same_topic_wrong` | Distinguish relevant principles from topic-matched language | Shared inference-available terms and disjoint passage IDs |
| `remove_warnings` | Test contribution of caveats and contraindications | Competitor fields remain unchanged |
| `remove_competing_pathways` | Test contribution of alternative-mechanism information | Warning fields remain unchanged |

### `same_topic_wrong` donor eligibility

A same-topic wrong passage does not necessarily exist for every row. Absence of a donor is a valid **intervention eligibility** failure, not a chemistry failure.

The builder therefore writes, for every intervention:

```text
<intervention>.jsonl
<intervention>.reference.jsonl
<intervention>.eligible_ids.json
<intervention>.quarantine.jsonl   # only when needed
```

For `same_topic_wrong`, a no-donor row is excluded only from that intervention. The paired reference is the original baseline restricted to the exact same eligible IDs. The manifest reports eligible fraction, quarantine reasons, transformed stable-ID hash, and paired-reference stable-ID hash.

The following statement is **not** valid when quarantine occurs:

```text
all interventions preserve the full input stable-ID universe
```

The valid statement is:

```text
all paired comparisons preserve identical IDs within their declared eligible universe
```

This prevents selection bias from being hidden behind a global “same IDs” flag.

## Evaluation

Headline six-condition evaluation is performed by `run_h3_suite.py`. The underlying evaluator remains:

```bash
python scripts/evaluate_knowledge_ablation.py \
  --reference data/knowledge_ablation/v2/test/trace_textbook_rag.jsonl \
  --condition trace_no_knowledge=outputs/h3/trace_no_knowledge.jsonl \
  --condition trace_length_matched_irrelevant=outputs/h3/trace_length_matched_irrelevant.jsonl \
  --condition trace_textbook_rag=outputs/h3/trace_textbook_rag.jsonl \
  --condition trace_structured_anchors=outputs/h3/trace_structured_anchors.jsonl \
  --condition trace_text_plus_anchors=outputs/h3/trace_text_plus_anchors.jsonl \
  --condition direct_textbook_rag=outputs/h3/direct_textbook_rag.jsonl \
  --output outputs/h3/summary.json
```

For a reduced evidence-content intervention, generate both the baseline and intervention prediction artifacts over the builder-generated paired reference/eligible IDs. Do not compare a reduced intervention file to a full-universe baseline artifact.

The evaluator:

1. aligns all predictions to the frozen reference universe;
2. retains missing predictions as failures;
3. checks complete runtime metadata;
4. requires explicit trace completion for trace conditions;
5. recompiles and re-executes the trace-owned proof;
6. reports structural exact match as primary and mapped exact separately;
7. reports condition-specific adapter lineage.

## Metric semantics

Candidate rollouts are independent generations. Without a frozen ranking score, report **Pass@K**, not Top-K.

Implemented metrics include:

```text
StructuralEndpointPass@1/5/10
MappedEndpointPass@1/5/10
ExecutePass@1/5/10
TraceBoundPass@1/5/10
coverage and selective risk
abstention rate
tool-failure recovery
retrieval Recall@K and Precision@K when frozen passage labels exist
retrieval latency
missing-prediction and re-execution error rates
```

Reaction-center and synthon metrics remain unavailable until frozen labels exist.

## Claim gates

### Textbook evidence

```text
trace_textbook_rag > trace_no_knowledge
and
trace_textbook_rag > trace_length_matched_irrelevant
```

### Combined evidence

```text
trace_text_plus_anchors > trace_textbook_rag
and
trace_text_plus_anchors > trace_structured_anchors
```

### Global integrity

```text
all frozen held-out test IDs evaluated for headline conditions
runtime metadata complete and matched
trace paths explicitly finish and fully re-execute
zero evidence reward violations
query leakage absent
passage/tool interventions reported
paired evidence-content subsets use their generated reference IDs
paired uncertainty reported for primary contrasts
```

## Statistical identification and multi-seed aggregation

Each primary H3 contrast is evaluated on the same frozen IDs with paired bootstrap confidence intervals and exact McNemar tests. The four primary evidence contrasts are corrected with Holm's method. Final evidence claims require independent-seed aggregation via `scripts/aggregate_evaluation_seeds.py`, a confidence-interval lower bound above the declared minimum effect, and consistent effect direction across seeds.

## Interpretation boundary

A gain explained by context presence, query leakage, train-derived evaluation, missing predictions, runtime mismatch, post-test evidence editing, unmatched intervention IDs, or adapter-compute imbalance does not support H3. Evidence benefit is a claim about information improving program induction—not a claim that retrieved text establishes chemical truth.
