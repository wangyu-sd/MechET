# PR #69 smoke campaign runbook

The scientific contract is in
`docs/superpowers/specs/2026-09-29-mechanistic-verified-retro-autoresearch-design.md`.
This runbook covers the first implementation only. It does not make a
performance claim or authorize full-scale 8B training.

State-SFT sampling counts **decision rows**, not FlowER reaction rows. The
underlying strict-executable FlowER universe is 257,167/2,890/28,967; the
unqualified FlowER full reaction split is 257,171/2,890/28,971. The current
mech-USPTO State-SFT source is the validated current-compiler trace view of
10,152/1,319/1,253 reactions, not the complete 31k endpoint benchmark.

## Read-only dependency audit

```bash
python scripts/autoresearch/run_campaign.py plan \
  --config configs/autoresearch/mechanistic_verified_retro_smoke.yaml \
  --data-root /absolute/path/to/MechET \
  --output /absolute/path/to/outputs/autoresearch/mechanistic_verified_retro_smoke
```

The plan always enumerates R1–R5, including missing sources. The PMechDB
request/download terms are not bypassed. The public PMechDB download page
requires agreement and registration for the full curated dataset; the small
Hugging Face `pmechdb_elem` mirror is a separate provenance/coverage question,
not automatically the licensed complete challenging test.

## Freeze and prepare engineering smoke

```bash
python scripts/autoresearch/run_campaign.py freeze-engineering \
  --config configs/autoresearch/mechanistic_verified_retro_smoke.yaml \
  --data-root /absolute/path/to/MechET \
  --output /absolute/path/to/outputs/autoresearch/mechanistic_verified_retro_smoke
python scripts/autoresearch/run_campaign.py prepare-engineering \
  --config configs/autoresearch/mechanistic_verified_retro_smoke.yaml \
  --data-root /absolute/path/to/MechET \
  --output /absolute/path/to/outputs/autoresearch/mechanistic_verified_retro_smoke
```

The second command runs the existing trainer's contract-only dry-run and
requires exactly 32 validated rows. An existing frozen directory is never
overwritten. The 32-row smoke is an optimizer/runtime check only.

To render a one-A100 task, supply a reviewed, successful single-GPU template:

```bash
python scripts/autoresearch/run_campaign.py render-engineering \
  --config configs/autoresearch/mechanistic_verified_retro_smoke.yaml \
  --data-root /absolute/path/to/MechET \
  --output /absolute/path/to/outputs/autoresearch/mechanistic_verified_retro_smoke \
  --template configs/taiji/mechet_uspto31k_full_rxnmapper_build_1a100_20260824.json \
  --gpu A100 --task-flag meteor_mechet_pr69_engineering_smoke_1a100_qy_YYYYMMDD_01
```

The rendered config contains only a private-init placeholder. Submission uses
`scripts/submit_taiji_with_donor_init.py`; a successful historical donor task
must be provided. Check actual POD state, Ceph mount, live stdout heartbeat,
GPU process and a completed checkpoint. A `start` response is not a run result.
If the Taiji image cannot reach Hugging Face, download the **same pinned
revision** to a Ceph HF cache, verify the safetensors SHA256, and render an
infrastructure-only retry with `--model-cache /absolute/shared/cache` and a
new `meteor` task flag. The backend sets offline mode but keeps the configured
Hub model ID/revision and the frozen training data unchanged. Use
`poll-engineering` to record the previous terminal failure before passing
`--retry-reason model_cache_unavailable`; the controller caps retries at two.

## Scientific freeze

Set all seven `evaluation_sources` to product-bearing JSONL manifests before
`freeze-scientific`. Their IDs and canonical products are excluded from
State-SFT mixtures. The command fails closed if any source is missing, if an
input sidecar hash disagrees or if an accepted decision lacks executor replay.
Curated rows must additionally carry source/license provenance and prove
replay compatibility before being configured; a file that merely exists is
insufficient. If zero curated rows survive, backfill is recorded but no
Mech-vs-Base scientific contrast is identifiable.

All R1–R5 packages are independent. Each evaluator writes `rN/result.json`
with `package: rN`, `status: complete|failed`, an immutable manifest hash,
denominators, and metrics. Missing/negative packages remain visible in the
scorecard; no test-driven resampling or silent exclusion is permitted.

The initial R1 source builder uses only the official FlowER **test** endpoint
records. A product enters the cohort only when distinct held-out records
support at least two distinct canonical structural precursor sets. It stores
the supporting record IDs rather than assuming a model-generated alternative
is known-valid. The predeclared overlap stratum is the minimum, over reference
sets, of the maximum radius-2/2048-bit Morgan Tanimoto similarity between the
product and any precursor component; `high` is >=0.6. Disconnection agreement
is computed from mapped product bonds absent from each precursor set. These
are descriptive strata, not physical feasibility labels.

## R3 controlled-corruption source

The R3 builder uses the frozen strict-executable FlowER test trace and its
Stage-II compressed-history decisions. It independently replays each selected
reference prefix, changes **one electron-flow decision**, and verifies that
the executor either rejects that action or accepts a different successor.
Only explicit closed-shell two-electron source/sink trajectories enter this
cohort; radical-pair and aggregate BE-delta trajectories and any recorded
state with RDKit radical electrons are excluded. The
coordination stratum counts moves in the **corrupted event**, not the maximum
over a whole reaction.

```bash
PYTHONPATH=src:. python scripts/autoresearch/build_r3_corruptions.py \
  --trace-source /absolute/path/to/MechET/data/flower_inverse_tool_sft_action_delta_v1/test.jsonl \
  --trace-manifest /absolute/path/to/MechET/data/flower_inverse_tool_sft_action_delta_v1/training_manifest.json \
  --history-source /absolute/path/to/MechET/data/flower_natural_language_event_history_v2/test.jsonl \
  --history-manifest /absolute/path/to/MechET/data/flower_natural_language_event_history_v2/manifest.json \
  --output /absolute/path/to/outputs/autoresearch/prepared_eval/r3_flower_polar_event
```

The frozen JSONL separates `model_visible` execution feedback from
`private_reference` localization/repair labels. An accepted wrong successor
appears to the policy only as `PASS` plus the actual successor, not as a gold-
relative error flag. The 288 target rows are 32 per early/middle/late ×
1/2/3+-move cell; reaction IDs may recur between cells, so uncertainty must
be clustered by reaction. This builder produces an evaluation **source**, not
R3 localization or repair scores. Older R3 diagnostic cohorts marked
`evaluation_allowed: false` must not be used.

## R5 external-model query cohort

Before querying an external retrosynthesis model, freeze 200 unique products
from the official FlowER full-endpoint test split: 100 sampled from the R1
multi-recorded-alternative products and 100 from other official test products.
Each half is evenly sampled across empirical product heavy-atom-size quartiles
using the fixed campaign seed. The external model receives only the canonical
unmapped product; all recorded precursor sets remain under `private_reference`.

```bash
PYTHONPATH=src:. python scripts/autoresearch/build_r5_products.py \
  --source /absolute/path/to/MechET/data/flower_full_endpoint_sft/test.jsonl \
  --official-manifest /absolute/path/to/MechET/data/flower_full_endpoint_sft/manifest.json \
  --r1-cohort /absolute/path/to/outputs/autoresearch/prepared_eval/r1_flower_official_test_20260929/r1_multi_reference.jsonl \
  --r1-manifest /absolute/path/to/outputs/autoresearch/prepared_eval/r1_flower_official_test_20260929/manifest.json \
  --output /absolute/path/to/outputs/autoresearch/prepared_eval/r5_flower_products
```

This is a **query cohort only**, not the `r5_external_predictions` evaluation
source. The latter remains missing until an external checkpoint produces
ranked Top-5 candidates for all frozen products, with missing ranks retained
in the denominator. RetroChimera is the initial candidate source; its
[official repository](https://github.com/microsoft/retrochimera) offers
Pistachio-, USPTO-50K-, and USPTO-FULL-trained checkpoints. Any overlap of
those training universes with the FlowER test chemistry must be reported;
without that audit, the result is diagnostic rather than leakage-clean.
