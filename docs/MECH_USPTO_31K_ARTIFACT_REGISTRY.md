# mech-USPTO-31k artifact registry

This file is the human-readable source of truth for mech-USPTO artifact
lineage. The machine-readable mirror is
`configs/datasets/mech_uspto_31k_artifacts.json`. Every local artifact also has
an `ARTIFACT_STATUS.json` sidecar. Do not infer validity from a directory name
or from a conversion rate alone.

## Denominators

- Full reaction benchmark: 24,959 train / 3,120 valid / 3,120 test.
- 2026-08-11 executable pilot: 9,118 / 1,187 / 1,124.
- A claim of 100% inverse conversion for the pilot uses 11,429 stitched traces
  as its denominator, not the 31,199 raw reactions.

## Registered versions

| ID | Path | Status | Counts | Allowed use |
|---|---|---|---|---|
| `raw_hf_d708ff6` | `data/raw/mech_uspto_31k` | frozen source | 24,959 / 3,120 / 3,120 reactions | source only |
| `forward_20260811_v1` | `data/forward_expert/mech_uspto_31k` | deprecated pilot | 9,118 / 1,187 / 1,124 stitched traces | historical diagnostics only |
| `inverse_20260811_v2` | `data/mech_uspto_31k_inverse_tool_sft` | deprecated pilot | 9,118 / 1,187 / 1,124 | historical diagnostics only |
| `mixed_20260811_v1` | `data/mixed_inverse_tool_sft` | deprecated mixed pilot | mech-USPTO train contribution 9,118 | historical model lineage only |
| `full_endpoint_pseudomap_invalid` | `data/mech_uspto_31k_full_endpoint_sft` | invalid | no valid export | never use; requires Figshare v2 mapped reaction table |
| `full_endpoint_hf_rxnmapper_20260824` | `data/mech_uspto_31k_full_endpoint_rxnmapper` | building | 24,959 / 3,120 / 3,120 expected | active full endpoint and external-baseline handoff after manifest validation |
| `action_delta_20260823_v1` | `data/mech_uspto_31k_inverse_tool_sft_action_delta_v1` | deprecated pilot | 9,118 / 1,187 / 1,124 | reproduce the completed pilot only; no new training |
| `compiler_20260824_v2` | `data/forward_expert/mech_uspto_31k_recompiled_20260824` | validated trace source | 10,152 / 1,319 / 1,253 stitched traces | source for current inverse build |
| `action_delta_20260824_v2` | `data/mech_uspto_31k_inverse_tool_sft_action_delta_v2_compiler_20260824` | validated | 10,152 / 1,319 / 1,253 | current train-ready executable trace view |

The completed Qwen3-8B action-delta checkpoint under
`outputs/agent/tool_sft_mech_uspto_31k_action_delta_qwen3_8b_a100_20260823`
therefore remains a valid historical pilot checkpoint, but it is not a model
trained on the current compiler output.

## Status semantics

- `source`: immutable upstream input; not a train-ready artifact.
- `deprecated_pilot`: reproducible historical subset, forbidden for new jobs.
- `invalid`: scientifically invalid and forbidden for every job.
- `rebuilding`: incomplete; forbidden for every job.
- `validated_trace_source`: the forward trace build passed its declared checks;
  it is an upstream source rather than a Tool-SFT train file.
- `validated`: all declared build, replay, split, hash, and coverage checks pass.

Only `validated` artifacts with `training_allowed: true` may back a new job.
The full endpoint benchmark and executable-trace supervision remain different
views: compiler coverage never changes the full 3,120-row test denominator.
