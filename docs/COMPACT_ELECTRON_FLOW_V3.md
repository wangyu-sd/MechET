# Compact electron-flow v3 (experimental / opt-in)

**No v3 model is trained or benchmarked by this PR.** The code implements an
independent representation, SFT conversion, product-only inference and audit
interface without changing the default v2 prompt, adapter or executor.

## Representation and chemical scope

The v2 event source/destination fields are the executor's chemically
significant signal. The model was also generating an English `instruction`
that repeats them. V3 preserves ordered coupled electron transfers and
removes only this repeated prose and verbose JSON structure:

```text
{"name":"apply_electron_flow",
 "arguments":{"flow":"B(A03,A04)>A04;LP(A02)>NB(A02,A03)"}}
```

- `LP(Axx)`: lone-pair electron source; `Axx`: atom sink
- `B(Axx,Ayy)`: bond electron container for an **existing** bond
- `NB(Axx,Ayy)`: new/prospective bond sink; disallowed as a source
- `RP(Axx,Ayy)`: radical-pair container within current frozen support
- `;`: move separator within one coupled event
- `DELTA|B(Axx,Ayy):+1;Q(Axx):0>+1`: explicit existing BE_DELTA
  representation; not silently recast as LP/BOND electron arrows

The same `compile_event_arguments` and `verify_electron_step` are used for
v2 and v3. Temporary atom labels are mapped only by the executor. A new
model may still make chemically wrong choices; this codec does not certify
synthetic feasibility or resolve the existing private-map/Kekulé parity issue.

## Generate supervised data without changing v2

Use the original *complete, frozen and training-allowed* State-SFT v2 data:

```bash
python scripts/build_compact_electron_flow_sft.py \
  --source-dir data/flower_natural_language_event_sft_v2 \
  --output-dir data/flower_compact_electron_flow_sft_v3
```

Every v2 decision becomes exactly one v3 decision. The builder preserves
source reaction IDs, user product/current-state observations, import and
finish actions, tool results, target labels, decision order and historical
executor outcome. It verifies source SHA-256 against the frozen v2 manifest
and checks full reaction/decision denominators. A partial
`--limit-reactions` data build receives `training_allowed=false` and
cannot be used by the training script.

Prepare the new, separate token cache:

```bash
torchrun --standalone --nproc_per_node=8 \
  scripts/prepare_tool_sft_arrow.py \
  --config configs/agent/compact_electron_flow_v3_qwen3_0_6b.yaml
```

The new config freezes 2,007,421 train and 22,341 validation decision rows.
No v2 token cache may be reused because the target byte strings changed.

## Separate one-epoch and three-epoch training hypotheses

```bash
# Representation comparison versus the existing frozen 1ep v2.
torchrun --standalone --nproc_per_node=8 scripts/train_tool_sft.py \
  --config configs/agent/compact_electron_flow_v3_qwen3_0_6b.yaml

# Longer-optimization comparator; different output directory.
torchrun --standalone --nproc_per_node=8 scripts/train_tool_sft.py \
  --config configs/agent/compact_electron_flow_v3_qwen3_0_6b_3ep.yaml
```

Both configurations retain the pinned Qwen3-0.6B revision and r16 q/k/v/o
LoRA. They isolate the epoch count, without adding model capacity or
changing fragmentation labels. Validation loss is available by epoch but
does not measure chemical feasibility.

## Product-only inference and replay scoring

To evaluate a *completed* Stage-I v3 adapter, pass `--compact-flow-v3` to
`scripts/eval_reliable_independent_episodes.py` in addition to the existing
frozen-source and adapter-SHA arguments. The underlying pure-policy sampling
budget is unchanged; the v3 prompt/tool schema and adapter lineage are
different and checked explicitly.

```bash
python scripts/eval_reliable_independent_episodes.py run \
  --data "$FROZEN_STRICT_SOURCE" \
  --expected-source-sha256 "$FROZEN_SOURCE_SHA256" \
  --adapter "$V3_ADAPTER_DIR" \
  --expected-adapter-sha256 "$V3_ADAPTER_SHA256" \
  --stage state --episodes 1 --sample-reactions 16 \
  --benchmark-view diagnostic --compact-flow-v3 \
  --output outputs/eval/compact_v3_diagnostic

# Repeat the same options with 'aggregate' instead of 'run'.
```

The evaluator records `action_representation` and hashes v3-specific runtime
dependencies. Use separate evaluation outputs and checkpoint SHA lineage;
never label a diagnostic as the full 28,967 strict-test or 28,971 endpoint
test. The standalone policy script accepts
`--compact-flow-v3 --matched-v2` to reuse *decoding parity* without reusing
the v2 prompt or adapter.

## Local regression checks

```bash
PYTHONPATH=src:. pytest -q tests/test_compact*_v3*.py
```

Tests cover natural/compact source-sink round-trip, BE_DELTA retention,
existing/new-bond distinction, malformed action rejection, unchanged v2
observation and tool-result bytes, dataset/trainer contract, strict adapter
lineage, and identical executor successor states.

**Out of scope:** v3 EARHO training, successor-equivalent multi-target
learning, reaction-centre head, automatic removal of endpoint context,
graph-edit replacement and any retraining of the already-frozen v2 model.
