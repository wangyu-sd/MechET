# MechET execution plan

This document is the operational source of truth for running the causal and compositional MechET study. Each phase has executable entrypoints, required artifacts, and a stopping gate. RL, forward evidence, and planning are postponed until the preceding scientific gate passes.

## Phase 0 — freeze contracts

Freeze and record:

```text
repository commit
source revisions and licenses
benchmark SHA-256 files
executor revision = MECH_PROOF_v1_move_bound
environment revision
base model and tokenizer revision
random seeds
```

Primary scope: mapped, closed-shell, two-electron polar organic chemistry.

Gate: all CI workflows pass and `docs/SCIENTIFIC_THESIS.md` matches the public README.

## Phase 1 — executable proof data

```bash
python scripts/build_mechet_sft.py \
  --flower-root /path/to/flower_new_dataset \
  --out-dir data/mechet_sft \
  --splits train valid test

python scripts/build_mechet_proof_sft.py \
  --input-dir data/mechet_sft \
  --output-dir data/mechet_proof_sft \
  --splits train valid test
```

Every accepted row must execute and store:

```text
full_precursor_state
structural_precursor
auxiliary_fragments
```

Then audit overlap and build a frozen clean set:

```bash
python scripts/audit_reaction_overlap.py \
  --train data/mechet_proof_sft/train.jsonl \
  --benchmark data/benchmarks/uspto50k/test.csv \
  --benchmark-format reaction_table \
  --reaction-field reaction_smiles \
  --out-dir outputs/data_audit/flower_vs_uspto50k_test

python scripts/build_decontaminated_dataset.py \
  --train data/mechet_proof_sft/train.jsonl \
  --benchmark data/benchmarks/uspto50k/test.csv \
  --output data/mechet_proof_clean/train.jsonl \
  --manifest data/mechet_proof_clean/manifest.json \
  --policy exact_structural product
```

Gate: source/benchmark manifests, quarantine counts, and overlap matrices are frozen before training.

## Phase 2 — evidence assets and proof-to-trace coverage

Build the provenance-aware corpus:

```bash
python scripts/download_mechanistic_sources.py \
  --registry knowledge/source_registry.yaml \
  download --source iupac_goldbook_terms --source rxno \
  --source wikibooks_organic_chemistry \
  --output knowledge/raw

python scripts/build_textbook_corpus.py \
  --download-root knowledge/raw \
  --output knowledge/corpus/passages.jsonl

python scripts/index_textbook_corpus.py \
  --corpus knowledge/corpus/passages.jsonl \
  --output knowledge/corpus/bm25_index.json
```

Build two replay-verified source conditions:

```bash
python scripts/build_textbook_tool_sft.py \
  --input data/mechet_proof_clean/train.jsonl \
  --corpus knowledge/corpus/passages.jsonl \
  --output data/textbook_tool_sft/train.jsonl \
  --query-mode state

python scripts/build_textbook_tool_sft.py \
  --input data/mechet_proof_clean/train.jsonl \
  --corpus knowledge/corpus/passages.jsonl \
  --output data/textbook_tool_sft/train_text_and_anchors.jsonl \
  --enable-structured-primitives \
  --query-mode state
```

The main condition must use `query-mode=state`. `label_oracle` is an upper bound and cannot enter headline results.

Required conversion report:

```text
root imports preserved
proof rows read/written/quarantined
stable quarantine reasons
conversion rate by source family
trace steps, source-to-sink moves, and imports
endpoint replay rate
```

Gate: the retained family and complexity distribution supports the declared scope. Otherwise narrow the scope or extend the converter.

## Phase 3 — matched six-condition data

```bash
python scripts/build_knowledge_ablation_suite.py \
  --config configs/experiments/textbook_ablation.yaml
```

Validate the actual tool schema and tokenizer contract:

```bash
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

Gate:

```text
same stable ID universe
same target and reference endpoints
no gold reaction-label retrieval query
valid tool-call/result pairing
non-empty assistant masks
zero truncation
frozen evidence character budgets
reported tokenizer input/supervised tokens
```

Raw direct and tool syntax lengths need not be equal. Match examples and optimizer updates, report supervised-token-normalized compute, and use the validator's multiplier only when exact cumulative supervision matching is required.

## Phase 4 — real Tool-SFT

Start with a 32-example overfit:

```bash
python scripts/train_tool_sft.py \
  --config configs/knowledge/tool_sft_trace_no_knowledge.yaml \
  --limit 32 --max-steps 100
```

Run the six matched SFT configs only after the overfit test succeeds:

```text
tool_sft_trace_no_knowledge.yaml
tool_sft_irrelevant.yaml
tool_sft_textbook.yaml
tool_sft_anchors.yaml
tool_sft_combined.yaml
tool_sft_direct_textbook.yaml
```

Required artifacts:

```text
data_contract.json
adapter_manifest.json
adapter SHA-256
base model revision
assistant-mask/token audit
loss and tool syntax curves
```

Gate: validation execution and `finish_trace` rates improve. Do not start GRPO from an untrained tool policy.

## Phase 5 — H1 causal faithfulness

Optional on-policy trace training:

```bash
python scripts/train_inverse_agent_trace.py \
  --config configs/agent/inverse_trace_grpo.yaml
```

Generate the normal artifact:

```bash
python scripts/infer_mechet.py \
  --config configs/agent/inverse_trace_grpo.yaml \
  --data data/benchmarks/h1/test.jsonl \
  --output outputs/h1/normal.jsonl \
  --mode trace --condition-name trace_no_knowledge \
  --intervention none --samples-per-target 4
```

Repeat with the same model, adapter, model revision, K, temperature, top-p, token limit, and iteration limit:

```text
remove_tool_observations
stale_tool_observations
shuffle_tool_observations
disable_inspect_state
disable_intermediate_execution
```

For shuffle, pass the normal artifact through `--intervention-source`.

Evaluate:

```bash
python scripts/evaluate_faithfulness.py \
  --reference data/benchmarks/h1/test.jsonl \
  --normal outputs/h1/normal.jsonl \
  --intervention remove_tool_observations=outputs/h1/remove.jsonl \
  --intervention stale_tool_observations=outputs/h1/stale.jsonl \
  --intervention shuffle_tool_observations=outputs/h1/shuffle.jsonl \
  --output outputs/h1/summary.json
```

Gate:

```text
all frozen IDs evaluated
normal path 100% trace-bound among completed traces
trace/proof metrics recompute without error
identical runtime contract across interventions
paired causal sensitivity is reported
```

If observation interventions have no material paired effect, do not claim tool-grounded causal reasoning.

## Phase 6 — H2 compositional generalization

```bash
python scripts/build_mechcomp_ood.py \
  --input data/knowledge_ablation/v2/trace_no_knowledge.jsonl \
  --output-dir data/ood/mechcomp_source_sink \
  --test-fraction 0.10 \
  --valid-fraction 0.10 \
  --min-train-primitive-count 5 \
  --seed 42
```

The split basis must be `source_to_sink_execution_moves_v1`, not net proof deltas or knowledge-anchor IDs.

Gate:

```text
non-empty held-out test
zero train/test complete-composition overlap
all held-out source-to-sink primitives seen in train
achieved split fractions disclosed
```

Train/evaluate direct, CoT, net-edit, complete-proof, and trace-owned representations on the same frozen split.

## Phase 7 — H3 evidence separation

Build evidence-content interventions when needed:

```bash
python scripts/build_evidence_interventions.py \
  --input data/knowledge_ablation/v2/trace_text_plus_anchors.jsonl \
  --output-dir data/evidence_interventions/v2 \
  --intervention passage_shuffle \
  --intervention same_topic_wrong \
  --intervention remove_warnings \
  --intervention remove_competing_pathways
```

Generate six prediction artifacts with `scripts/infer_mechet.py` using modes:

```text
trace
irrelevant
textbook
anchors
combined
direct
```

Evidence modes replay row-specific frozen evidence, so direct and trace conditions receive the same bounded evidence content.

Evaluate:

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

Gate:

```text
all frozen IDs evaluated; missing predictions count as failures
no supervision rows accepted as predictions
trace outputs recompile/re-execute
same base/revision/generation budget across conditions
condition-specific adapter hashes reported
textbook > trace-only and textbook > irrelevant for a text-evidence claim
combined > each individual condition for a combined-evidence claim
```

## Phase 8 — scale, forward evidence, and planning

Only after H1–H3 pilots pass:

```text
0.6B / 1.7B / 8B scale study
formal-process RL
calibrated forward closure and explicit competitors
K={1,4,16,64}
multistep planning under frozen candidate pools
```

The forward expert remains a cached, frozen soft-evidence model. Planning is a downstream extension and cannot rescue failed H1 or H2 results.

## Implemented prediction metrics

```text
structural precursor Top-1/5/10, ignoring atom maps
mapped structural Top-1/5/10
ExecutePass@1/5/10
TraceBoundPass@1/5/10
coverage, selective risk, abstention
tool-failure recovery
retrieval recall/precision when gold passage IDs exist
retrieval latency
missing-prediction and re-execution error rates
```

Reaction-center and synthon metrics remain unavailable until frozen labels exist.

## Global stopping rules

Stop or narrow a claim when:

- conversion coverage is too narrow;
- gold reaction labels enter a main retrieval query;
- root imports or declared moves cannot replay;
- tokenizer masks are empty or examples truncate;
- required adapters/manifests/hashes do not match;
- missing predictions are silently removed;
- runtime budgets differ across a claimed intervention or ablation;
- H1 is insensitive to tool observations;
- H2 contains unseen primitives rather than unseen compositions;
- irrelevant text explains the evidence gain;
- any learned score overrides deterministic execution.
