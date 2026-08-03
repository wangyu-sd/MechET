# Evidence-augmented trace-owned agent

`KnowledgeAugmentedAgentEnv` is an evidence-conditioned version of the causal trace-owned main method. It is not a separate endpoint-generation architecture.

## Scientific role

The environment tests H3:

> Can external mechanistic evidence improve induction of an executable electron-flow program beyond trace ownership and extra context alone?

The causal endpoint path remains:

```text
explicit electron-flow actions
  -> environment-owned trace
  -> finish_trace
  -> deterministic proof compilation
  -> executor-derived precursor
```

Evidence tools may affect action selection, but they are not part of the endpoint computation.

## Evidence tools

```text
retrieve_textbook_guidance(query, top_k, max_characters)
retrieve_primitives(query, top_k)
```

The second compatibility name retrieves structured **mechanistic knowledge anchors**, not the execution-primitive vocabulary used for MechComp-OOD.

Both tools return bounded provenance-aware evidence and satisfy:

```text
direct_reward = false
soft_evidence_only = true
no precursor return
no executor override
```

All evidence calls, passage or anchor IDs and context hashes are retained in the rollout trace.

## Inherited causal tools

```text
inspect_state
import_fragment
apply_electron_move
apply_coupled_electron_moves
finish_trace
abstain
```

Free-form `submit_proof` remains disabled.

## Corpus preparation

```bash
python scripts/build_textbook_corpus.py \
  --download-root knowledge/raw \
  --output knowledge/corpus/passages.jsonl

python scripts/index_textbook_corpus.py \
  --corpus knowledge/corpus/passages.jsonl \
  --output knowledge/corpus/bm25_index.json
```

The exact corpus and index manifests are frozen before final-test evaluation.

## Supervised initialization

Build replay-verified trace rows before on-policy training:

```bash
python scripts/build_textbook_tool_sft.py \
  --input data/mechet_proof_clean/train.jsonl \
  --corpus knowledge/corpus/passages.jsonl \
  --output data/textbook_tool_sft/train.jsonl
```

Then train Tool-SFT and verify executable learning:

```bash
python scripts/train_tool_sft.py \
  --config configs/knowledge/tool_sft_textbook.yaml
```

Paper-scale GRPO should initialize from this frozen adapter and record its hash.

## On-policy condition

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

Required checkpoint lineage:

```text
base-model and tokenizer revision
Tool-SFT adapter path and hash
Tool-SFT data-manifest hash
executor and environment revision
evidence corpus/index hash
reward config and seed
```

## Matched conditions

```text
trace_no_knowledge
trace_length_matched_irrelevant
trace_textbook_rag
trace_structured_anchors
trace_text_plus_anchors
direct_textbook_rag
```

Build them from the same source rows:

```bash
python scripts/build_knowledge_ablation_suite.py \
  --config configs/experiments/textbook_ablation.yaml
```

Anchors-only and direct open-book rows are derived automatically.

## Ablation switches

```text
auto_textbook_on_reset
textbook_top_k
textbook_max_characters
textbook_max_per_source
enable_structured_primitives
primitive_top_k
```

The compatibility field `enable_structured_primitives` means structured knowledge-anchor retrieval. It does not alter the execution-primitive grammar.

## Causal controls

Evidence-use controls:

```text
length-matched irrelevant evidence
passage shuffle
same-topic wrong passage
warnings removed
competing-pathway text removed
```

Trace-use controls:

```text
tool observations removed
tool observations shuffled
stale molecular states substituted
```

The two groups answer different questions and must be reported separately.

## Claim boundary

An evidence gain is supported only when it exceeds trace-only and irrelevant-context controls under matched IDs, endpoints, training budgets and compute. Evidence does not establish formal validity, a unique physical mechanism, selectivity, kinetics, yield or experimental success.
