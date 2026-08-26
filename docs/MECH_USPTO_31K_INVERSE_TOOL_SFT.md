# mech-USPTO-31k inverse Tool-SFT

> [!CAUTION]
> **INCOMPLETE TRACE-VIEW SUBSET.** The current compiler produces `1,253` test
> traces. They are only the
> replay-compatible inverse-trace subset of the complete `3,120`-reaction
> mech-USPTO-31k test split. They are valid for program supervision and
> diagnostics, but must not be reported as the full benchmark denominator or
> a headline endpoint result.

> **Role:** replay-verified inverse-trace supervision for model initialization
>
> **Contract:** `mech_uspto_inverse_trace_owned_v2`
> **Knowledge condition:** `trace_no_knowledge` (`corpus_used=false`)

## Dataset identity

This pipeline uses `SchwallerGroup/mech_uspto_31k`, a mechanistic
USPTO-derived collection with source-to-sink electron arrows for elementary
steps. It is not the standard USPTO-50K single-step retrosynthesis benchmark
and it is not extracted from FlowER.

| Asset | Role in MechET |
|---|---|
| mech-USPTO-31k | Electron-move supervision used to build inverse Tool-SFT |
| FlowER | Separate mechanistic trajectory source |
| USPTO-50K | Separate single-step retrosynthesis benchmark/source |
| Textbook corpus | Optional retrieval evidence for H3; not used here |

The source registry records mech-USPTO-31k as CC-BY-4.0. Frozen runs retain
the upstream revision and local file hashes. The current snapshot is upstream
commit `d708ff68be35fd02d2c1e183ee3d437b0b647f53`.

## What one inverse row means

An accepted row starts from the mapped structural product selected by matching
the raw `rxn_prod_min` field into the final globally mapped mechanism state.
Mapped fragments outside that structural target are imported explicitly. The
forward steps and their coupled electron moves are reversed, executed inside a
fresh `TraceOwnedAgentEnv`, and terminated by `finish_trace`.

```text
mapped product target
  -> import mapped auxiliary fragments, when required
  -> reverse the last forward step first
  -> reverse the move order inside each coupled step
  -> execute inverse source-to-sink moves
  -> compare every state with the corresponding forward precursor state
  -> finish_trace
  -> environment-owned full precursor and endpoint views
```

The supported two-electron inversions are:

```text
forward LP(a) -> BOND(a,b)     => inverse BOND(a,b) -> ATOM(a)
forward BOND(a,b) -> ATOM(c)   => inverse LP(c) -> BOND(a,b)
forward BOND(a,b) -> BOND(c,d) => inverse BOND(c,d) -> BOND(a,b)
```

Each JSONL row contains canonical `messages` and `tools`, tool-call/result
pairs, exactly one successful `finish_trace`, endpoint views, the compiled
proof, trace and move-sequence digests, source lineage, and replay metadata.
The precursor is never copied into a free-form assistant answer.

## Stereo policy in contract v2

Electron-pair moves encode connectivity and formal electronic state, but do
not encode attack-face geometry. A forward cleavage may turn a specified
tetrahedral precursor center into an achiral planar intermediate. Inverting
the moves restores the bond but cannot choose `@` versus `@@` when the final
product and trace contain no such information.

Contract v2 applies a narrow normalization:

1. the atom had specified tetrahedral chirality in the precursor;
2. the final state no longer specifies chirality for that atom; and
3. the atom participates in a declared electron move.

Only tags satisfying all three conditions are cleared from inverse endpoint
comparisons. Unreacted stereocenters and stereochemistry still specified by
the final state remain exact. Every row records the affected atom maps under
`metadata.stereo_normalization`.

This rule normalized one reaction-center tag in each of 19 rows. It does not
invent a stereoisomer and does not make general endpoint matching
stereo-insensitive.

## Versioned local coverage

The 2026-08-11 artifact is retained only as a deprecated historical pilot:

| 2026-08-11 stage | train | valid | test | total |
|---|---:|---:|---:|---:|
| Raw reactions | 24,959 | 3,120 | 3,120 | 31,199 |
| Raw elementary-step rows | 91,805 | 11,396 | 11,625 | 114,826 |
| Executable standardized step rows | 72,963 | 9,087 | 9,326 | 91,376 |
| Reactions with every step executable | 15,117 | 1,903 | 1,895 | 18,915 |
| Globally mapped stitched forward traces *(incomplete trace view)* | 9,118 | 1,187 | 1,124 | 11,429 |
| Accepted inverse Tool-SFT v2 rows *(incomplete trace view)* | **9,118** | **1,187** | **1,124** | **11,429** |

The inverse conversion is 11,429/11,429 with empty inverse quarantine files.
This is 100% of the globally stitched forward traces, not 100% of the original
31,199 reactions. Earlier filters remain conservative: every elementary move
must execute, every reaction must retain all steps, and adjacent independently
mapped states must be stitchable into one global atom-map scope.

The old paper-facing action-only derivative is retained under
`data/mech_uspto_31k_inverse_tool_sft_action_delta_v1/`. It has the identical
9,118 / 1,187 / 1,124 stable-ID universe, zero conversion quarantines, zero
split overlap, and zero model-visible intermediate-state leaks. Its manifest
requires `observation_mode=action_delta_v1`; the environment still owns the
private molecular state and terminal compiled proof.

It is labelled `deprecated_pilot` and is forbidden for new training.

The current compiler was rerun from the frozen raw parquet on 2026-08-24:

| 2026-08-24 current compiler stage | train | valid | test | total |
|---|---:|---:|---:|---:|
| Raw reactions | 24,959 | 3,120 | 3,120 | 31,199 |
| Raw elementary-step rows | 91,805 | 11,396 | 11,625 | 114,826 |
| Executable standardized step rows | 74,170 | 9,234 | 9,478 | 92,882 |
| Reactions with every step executable | 16,207 | 2,042 | 2,027 | 20,276 |
| Globally mapped stitched traces | 10,152 | 1,319 | 1,253 | 12,724 |
| Action-delta inverse rows | **10,152** | **1,319** | **1,253** | **12,724** |

The versioned outputs are
`data/forward_expert/mech_uspto_31k_recompiled_20260824/` and
`data/mech_uspto_31k_inverse_tool_sft_action_delta_v2_compiler_20260824/`.
The inverse conversion is 12,724/12,724; this is still a strict executable
trace subset, not 100% of the 31,199-reaction endpoint universe. Consult
`docs/MECH_USPTO_31K_ARTIFACT_REGISTRY.md` before selecting an artifact.

## Reproduce from download

Download the frozen upstream snapshot:

```bash
python scripts/forward_expert_data.py download \
  --dataset mech_uspto_31k \
  --revision d708ff68be35fd02d2c1e183ee3d437b0b647f53 \
  --output data/raw/mech_uspto_31k
```

Standardize and replay elementary rows, select reactions whose every raw step
survived, and stitch local step maps into global reaction traces:

```bash
for spec in train:train valid:val test:test; do
  split=${spec%%:*}
  raw_split=${spec##*:}

  python scripts/forward_expert_data.py standardize \
    --input data/raw/mech_uspto_31k/data/${raw_split}-00000-of-00001.parquet \
    --output data/forward_expert/mech_uspto_31k_recompiled_20260824/standardized/${split}.jsonl \
    --quarantine data/forward_expert/mech_uspto_31k_recompiled_20260824/standardized/${split}.quarantine.jsonl \
    --source mech_uspto_31k

  python scripts/select_complete_mech_uspto_reactions.py \
    --raw-parquet data/raw/mech_uspto_31k/data/${raw_split}-00000-of-00001.parquet \
    --standardized data/forward_expert/mech_uspto_31k_recompiled_20260824/standardized/${split}.jsonl \
    --output data/forward_expert/mech_uspto_31k_recompiled_20260824/complete_reaction_ids/${split}.jsonl

  python scripts/stitch_mech_uspto_traces.py \
    --input data/forward_expert/mech_uspto_31k_recompiled_20260824/standardized/${split}.jsonl \
    --complete-ids data/forward_expert/mech_uspto_31k_recompiled_20260824/complete_reaction_ids/${split}.jsonl \
    --output data/forward_expert/mech_uspto_31k_recompiled_20260824/traces/${split}.jsonl \
    --quarantine data/forward_expert/mech_uspto_31k_recompiled_20260824/traces/${split}.quarantine.jsonl
done
```

Build inverse rows using `val` as the raw filename corresponding to MechET's
normalized `valid` split:

```bash
for spec in train:train valid:val test:test; do
  split=${spec%%:*}
  raw_split=${spec##*:}

  python scripts/build_mech_uspto_inverse_tool_sft.py \
    --input data/forward_expert/mech_uspto_31k_recompiled_20260824/traces/${split}.jsonl \
    --raw-parquet data/raw/mech_uspto_31k/data/${raw_split}-00000-of-00001.parquet \
    --output data/mech_uspto_31k_inverse_tool_sft_action_delta_v2_compiler_20260824/${split}.jsonl \
    --quarantine data/mech_uspto_31k_inverse_tool_sft_action_delta_v2_compiler_20260824/${split}.quarantine.jsonl \
    --observation-mode action_delta

  python scripts/validate_mech_uspto_inverse_tool_sft.py \
    --input data/mech_uspto_31k_inverse_tool_sft_action_delta_v2_compiler_20260824/${split}.jsonl \
    --output data/mech_uspto_31k_inverse_tool_sft_action_delta_v2_compiler_20260824/${split}.validation.json

  python scripts/audit_tool_sft_token_lengths.py \
    --config configs/agent/tool_sft_mech_uspto_31k_action_delta_v2_qwen3_8b_a100.yaml \
    --input data/mech_uspto_31k_inverse_tool_sft_action_delta_v2_compiler_20260824/${split}.jsonl \
    --output data/mech_uspto_31k_inverse_tool_sft_action_delta_v2_compiler_20260824/${split}.tokenizer_audit.json
done

python scripts/finalize_mech_uspto_inverse_tool_sft.py \
  --data-dir data/mech_uspto_31k_inverse_tool_sft_action_delta_v2_compiler_20260824 \
  --training-config configs/agent/tool_sft_mech_uspto_31k_action_delta_v2_qwen3_8b_a100.yaml
```

The finalizer refuses stale replay reports, stale tokenizer audits, failed
validation, duplicate IDs, or split overlap. It writes aggregate validation,
tokenizer-audit, and manifest JSON files.

## Train

The current Qwen3-8B action-only screen uses the immutable revision
`b968826d9c46dd6066d109eabc6255188de91218`, QLoRA, assistant-only loss, three
epochs, no max-step cap, and the fully audited action-only files:

```bash
torchrun --standalone --nproc_per_node=8 scripts/train_tool_sft.py \
  --config configs/agent/tool_sft_mech_uspto_31k_action_delta_v2_qwen3_8b_a100.yaml
```

Its tokenizer audit covers all 12,724 rows with zero truncation; the largest
conversation is 3,049 tokens under the pinned Qwen3-8B tokenizer. A real run
must finish with `adapter_manifest.json` and `data_contract.json` whose train
SHA-256 equals the action-only manifest before inference is allowed.

After that adapter exists, the K=10 executable rollout runner is:

```bash
bash scripts/run_taiji_mech_uspto31k_action_delta_infer_k10.sh
```

For the current v2 artifact, the runner reuses the exact frozen training
system/user preamble and tool schema. It requires the prompt, environment and
outer rollout loop to share the same 12-call budget, and it uses one generation
worker per 40-GiB A100. Any prompt/schema/budget mismatch fails before model
loading. Each of the 1,253 rows must retain exactly ten candidates; generation
errors and OOMs remain explicit failed candidates rather than disappearing
from the denominator.

The output bundle reports generation-order Pass@1/5/10 and gold-independent
formal-execution-gated assistant-mean-NLL selection. NLL records retain both
cumulative and mean assistant NLL, plus candidate completion/error status.
First-candidate and NLL-selected results are each reported as mapped exact,
structural map-free exact and neutralized exact. The run manifest freezes the
adapter, dataset, compiler, executor, prompt contract and inference-config
lineage. Every artifact is stamped `program_view_subset_not_full_endpoint`:
1,253 is never a substitute for the official 3,120-reaction endpoint test
denominator.

The earlier approximately 1.60% Pass@10 result used a 40-call runtime against
12-call training prompts, regenerated the prompt, and ran two workers per
40-GiB A100. It is a protocol-mismatched diagnostic and must not be used as an
architecture result. The matched rerun is tracked in GitHub issue #36.

### Legacy small-model configuration

The frozen configuration uses Qwen3-0.6B at immutable model/tokenizer commit
`c1899de289a04d12100db370d81485cdf75e47ca`, assistant-only loss, LoRA, and
`max_length=12288`:

```bash
# Schema and data-contract check only
python scripts/train_tool_sft.py \
  --config configs/agent/tool_sft_mech_uspto_31k_inverse.yaml \
  --dry-run

# Real 32-row learnability smoke test
python scripts/train_tool_sft.py \
  --config configs/agent/tool_sft_mech_uspto_31k_inverse.yaml \
  --limit 32 \
  --max-steps 100

# Full train split
python scripts/train_tool_sft.py \
  --config configs/agent/tool_sft_mech_uspto_31k_inverse.yaml
```

The completed tokenizer audit covers all 11,429 rows with zero truncation and
zero rows lacking supervised assistant tokens. The maximum observed length is
12,262 tokens under the pinned tokenizer.

## Benchmark boundary

These files are supervision artifacts. Training on `train.jsonl` and
monitoring `valid.jsonl` does not establish H1, H2, or H3. If `test.jsonl` is
used for evaluation, it remains frozen and absent from training. External
FlowER and USPTO-50K benchmarks require explicit overlap/decontamination
audits; an official split name is not evidence of cross-dataset independence.
