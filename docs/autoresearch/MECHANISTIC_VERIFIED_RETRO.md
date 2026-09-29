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

Once the seven R1–R5 evaluation sources and a replay-audited curated training
source are configured, freeze once, then prepare both scientific conditions:

```bash
python scripts/autoresearch/run_campaign.py freeze-scientific \
  --config configs/autoresearch/mechanistic_verified_retro_smoke.yaml \
  --data-root /absolute/path/to/MechET \
  --output /absolute/path/to/outputs/autoresearch/mechanistic_verified_retro_smoke
python scripts/autoresearch/run_campaign.py prepare-scientific \
  --config configs/autoresearch/mechanistic_verified_retro_smoke.yaml \
  --data-root /absolute/path/to/MechET \
  --output /absolute/path/to/outputs/autoresearch/mechanistic_verified_retro_smoke \
  --model-cache /absolute/path/to/pinned_hf_cache
```

`prepare-scientific` refuses the 32-row engineering manifest, missing/changed
evaluation inputs, zero compatible curated rows, changed source files, unequal
12,000-decision quotas, changed selected-row hashes and an unpinned backbone.
It writes separate frozen Base/Mech one-epoch Qwen3-0.6B LoRA configs only
after verifying their common 256-decision validation source. The trainer
dry-run validates all selected rows and both configurations undergo exact
assistant-mask/token-length audits; preparation is recorded only if both pass.
The optional model-cache argument runs the audit against the same pinned
offline snapshot later used by Taiji; in the network-isolated deployment it
should be supplied.
The two conditions share the optimizer configuration and update budget. Their
measured input/supervised token totals are recorded separately; equal decision
row counts alone must not be reported as equal token budgets. Preparation does
**not** submit a GPU task or establish a scientific result. The current
campaign has no scientific freeze, so this command intentionally fails before
creating any scientific training config.

After successful preparation, render each one-GPU job separately using the
same pinned offline model snapshot and an independently checked Taiji template:

```bash
python scripts/autoresearch/run_campaign.py render-scientific \
  --config configs/autoresearch/mechanistic_verified_retro_smoke.yaml \
  --data-root /absolute/path/to/MechET \
  --output /absolute/path/to/outputs/autoresearch/mechanistic_verified_retro_smoke \
  --condition base --template /absolute/path/to/validated_1gpu_template.json \
  --gpu A100 --model-cache /absolute/path/to/pinned_hf_cache \
  --task-flag meteor_mechet_pr69_base_scientific_1a100_YYYYMMDD_01
```

Repeat with `--condition mech` and a distinct `meteor` task flag. Rendering
rechecks the scientific freeze, both token audits, the selected condition's
training config and the pinned offline model snapshot. It labels the job as
12,000-row scientific State-SFT, not the 32-row/100-step engineering smoke;
the resulting JSON still has a private-init placeholder. A rendered JSON is
not a submitted or running Taiji task. Submit only after inspecting the rendered
config and selecting a proven successful Ceph donor:

```bash
python scripts/autoresearch/run_campaign.py submit-scientific \
  --config configs/autoresearch/mechanistic_verified_retro_smoke.yaml \
  --data-root /absolute/path/to/MechET \
  --output /absolute/path/to/outputs/autoresearch/mechanistic_verified_retro_smoke \
  --condition base --donor-task meteor_proven_successful_ceph_task \
  --client /absolute/path/to/taiji_client
python scripts/autoresearch/run_campaign.py poll-scientific \
  --config configs/autoresearch/mechanistic_verified_retro_smoke.yaml \
  --data-root /absolute/path/to/MechET \
  --output /absolute/path/to/outputs/autoresearch/mechanistic_verified_retro_smoke \
  --condition base --client /absolute/path/to/taiji_client
```

Submission requires the exact audited render event in the campaign ledger;
the helper fills private Ceph init at runtime without committing it. A repeat
submission requires a *new rendered job path and meteor task flag*, a recorded
terminal previous instance with no adapter, and a named infrastructure retry
reason. At most two such retries are allowed. Low accuracy is not a retry
reason. Poll records the actual Taiji instance state and default POD log tail;
creating/starting a task is not evidence that optimization ran. No scientific
job is currently rendered or submitted under this campaign.

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

Before new scientific-smoke models are trained, the existing full-test Direct
Qwen3-8B predictions can be rescored **diagnostically** against this frozen
multi-reference cohort using `scripts/autoresearch/score_r1_existing_direct.py`.
It verifies the R1, official test, and prediction hashes; selects one
deterministic official-test prediction row per product; and reports sample-0
exact match plus generation-order Pass@K against the single recorded reference
and all independently recorded precursor sets. It does not run the forward
model or MechET verifier, audit Direct train overlap, or constitute the R1
Base-vs-Mech scientific-smoke result.

The parallel diagnostic `scripts/autoresearch/score_r1_existing_trace.py`
reads the existing legacy compact-full-state MechET K=10 strict-test file and
scores the same 220 products. It counts a candidate only when the actual
`finish_trace` result is executor-owned, formally executed and trace-bound;
`expected_precursor` or free-form text is never substituted. Its executable
non-reference outputs are labeled *unrecorded*, not chemically invalid.
Neither diagnostic replaces the new paired R1 scientific-smoke evaluation.

## R2 recorded proposals and negative-evidence boundary

Freeze 400 positive proposals from the same official held-out endpoint source:
200 ordinary recorded precursor sets and 200 alternative precursor sets with
an independent second record for the product. Inputs contain the product and
proposed precursor set; the supporting record IDs and positive label are kept
in `private_label`. This is a record-backed positive source, not a statement
that the route is the unique physical mechanism.

```bash
PYTHONPATH=src:. python scripts/autoresearch/build_r2_positives.py \
  --source /absolute/path/to/MechET/data/flower_full_endpoint_sft/test.jsonl \
  --official-manifest /absolute/path/to/MechET/data/flower_full_endpoint_sft/manifest.json \
  --r1-cohort /absolute/path/to/outputs/autoresearch/prepared_eval/r1_flower_official_test_20260929/r1_multi_reference.jsonl \
  --r1-manifest /absolute/path/to/outputs/autoresearch/prepared_eval/r1_flower_official_test_20260929/manifest.json \
  --output /absolute/path/to/outputs/autoresearch/prepared_eval/r2_recorded_positives
```

One negative class can be generated with a formal witness: omit a precursor
component, then verify that the remaining **unmapped elemental inventory**
lacks at least one element required by the product. A missing map label alone
is insufficient because equivalent atoms could be remapped. The 50-row
source balances 25 original-record and 25 alternative-record positives.

```bash
PYTHONPATH=src:. python scripts/autoresearch/build_r2_missing_fragment_negatives.py \
  --source /absolute/path/to/MechET/data/flower_full_endpoint_sft/test.jsonl \
  --official-manifest /absolute/path/to/MechET/data/flower_full_endpoint_sft/manifest.json \
  --positives /absolute/path/to/outputs/autoresearch/prepared_eval/r2_recorded_positives/r2_positives.jsonl \
  --positive-manifest /absolute/path/to/outputs/autoresearch/prepared_eval/r2_recorded_positives/manifest.json \
  --output /absolute/path/to/outputs/autoresearch/prepared_eval/r2_missing_fragment_negatives
```

This proves only that the **stated closed precursor inventory** is
element-insufficient under the frozen executor. It does not rule out an unlisted
external reagent. The other negative classes need separate auditable support;
neither a GT-different endpoint nor an R3 executor-accepted non-reference
successor is automatically chemically impossible. In particular, the
executor-valid chemistry-inconsistent class must have independent evidence.
Both prepared R2 directories remain `evaluation_allowed: false` until a
complete, audited 400-positive/400-negative source is assembled. Do not
compute AUROC/AUPRC on these partial sources as if they were the R2 smoke.

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

When real RetroChimera or collaborator-owned R-SMILES/ReactSeq predictions
become available, normalize them with the frozen intake script. The raw
prediction JSONL has exactly one row per frozen query product, with
`product_smiles`, `inference_status` (`completed` or `failed`), and a `candidates`
list of `{rank, precursors}` entries for ranks 1–5. Missing ranks are allowed
but are explicitly retained as missing in the five-rank denominator. A
separate provenance JSON records `model_name`, `checkpoint_identifier`,
`checkpoint_source`, `training_corpus`, `license_or_terms`,
`input_fields: ["product_smiles"]`, `inference_status: "completed"`, and a
nonempty `inference_config` object.

```bash
PYTHONPATH=src:. python scripts/autoresearch/freeze_r5_external_predictions.py \
  --query /absolute/path/to/outputs/autoresearch/prepared_eval/r5_flower_products_20260929/r5_products.jsonl \
  --query-manifest /absolute/path/to/outputs/autoresearch/prepared_eval/r5_flower_products_20260929/manifest.json \
  --predictions /absolute/path/to/external_model_raw_predictions.jsonl \
  --provenance /absolute/path/to/external_model_provenance.json \
  --output /absolute/path/to/outputs/autoresearch/prepared_eval/r5_external_model_frozen
```

The intake checks all 200 product IDs, source hashes, rank uniqueness, and
SMILES parseability; it never drops a failed product or invalid candidate.
Absence from the recorded-reference list is labeled
`not_recorded_not_proven_invalid`, not "unsupported chemistry". The frozen
source may enter R5 evaluation, but its `headline_allowed` remains false until
MechET verification, external-model training-overlap audit, and full scorecard
gates are completed. The script cannot substitute for actually obtaining and
running the external model.
