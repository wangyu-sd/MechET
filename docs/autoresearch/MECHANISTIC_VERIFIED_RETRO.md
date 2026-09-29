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

## Candidate curated training source: SynEPD

The independently released [SynEPD](https://github.com/TieuLongPhan/SynEPD)
v0.4.1 data declares CC BY 4.0 in its release manifest. It can be prepared
without changing the executor or the three public tools:

```bash
git clone https://github.com/TieuLongPhan/SynEPD.git /absolute/path/to/SynEPD
git -C /absolute/path/to/SynEPD checkout 4fefc016fd4d4e4305dec92586e593bafe3b3e5b
python scripts/autoresearch/build_synepd_curated.py \
  --source /absolute/path/to/SynEPD/data/polar.json \
  --release-manifest /absolute/path/to/SynEPD/data/release-manifest.json \
  --output /absolute/path/to/outputs/autoresearch/prepared_train/synepd_v041
```

The builder refuses unpinned input hashes and an existing output directory.
It records every rejected source ID, verifies every accepted inverse transition
with the frozen executor, then independently replays the emitted *public* tool
calls. The audited run accepted 1,674 reactions / 3,975 decision rows and
quarantined 252 reactions; see `docs/autoresearch/SMOKE_STATUS_20260929.md`.
Scientific selection of 3,000 curated decisions applies a predeclared
50%-per-family cap before the other strata, with deterministic seed 17.

This is a **training candidate only**. It neither replaces PMechDB challenging
or PMechRP pathways in R4 nor permits scientific sampling before all seven
evaluation sources are frozen and product-decontaminated. The default config
now names this hash-audited local artifact under `sources.curated`. On a new
machine the plan correctly reports it as unavailable until the exact release
has been rebuilt; an available curated source alone does not unlock either
scientific training condition.

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
Each evaluation JSONL must have a sibling `manifest.json` with its exact
`cohort_sha256`; an `ARTIFACT_STATUS.json`, when present, must explicitly set
`evaluation_allowed: true`. A path to a raw or partial JSONL alone is not a
frozen evaluation source. The controller also checks the predeclared smoke
denominators (R1 200–300, R2 800, R3 288, PMechRP 350, literature cycles
12–20, and R5 200 products); a one-row hash-matched placeholder cannot unlock
scientific sampling. PMechDB challenging must be nonempty, with its official
split-completeness established separately in the source manifest.
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
python scripts/autoresearch/run_campaign.py audit-overlap \
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
The overlap audit independently re-reads both selected train files and every
frozen evaluation cohort, records reaction-ID and canonical-product overlap
for each condition, and must report zero before preparation can continue.
`prepare-scientific` reruns this audit itself, so skipping the explicit command
cannot bypass the gate. The same report feeds the final scorecard.
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
with `package: rN`, `status: complete|failed`, exact
`scientific_freeze_sha256` and `evaluation_source_hashes`, positive integer
denominators, metrics, zero-or-explicit data-contract errors, and paired Base/
Mech checkpoint SHA-256s. The collector rejects a `complete` file whose
source hashes differ from the scientific freeze; a failed package needs a
reason. Missing/negative packages remain visible in the scorecard; no
test-driven resampling or silent exclusion is permitted. If training metrics
are not yet present, the promotion state is `INCOMPLETE`, not a negative result.
Early collection writes only an immutable `scorecards/<evidence-hash>.json`
snapshot. It does not create `scorecard.json` until all five packages and the
training/overlap evidence are sufficient for a final recommendation; thus an
early status check cannot lock out the later scientific result.

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

`scripts/autoresearch/analyze_r1_existing_diagnostics.py` joins those two
hash-bound diagnostic outputs to the original 220-product cohort and reports
the single-to-multiple-reference recovery by reference count, mapped
disconnection signature and structural-overlap stratum. It includes a fixed-
seed product bootstrap interval, which describes uncertainty *within this
selected multi-record cohort only*. The existing Direct and legacy trace
models are unmatched, so the script deliberately computes no Direct-vs-MechET
effect size. The output is separate from `r1/result.json` and cannot satisfy
the scientific Base-vs-Mech package gate.

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

The final R2 intake is `scripts/autoresearch/freeze_r2_plausibility.py`.
Supply the recorded-positive JSONL with `--positives`, and repeat
`--negative-class CLASS=/absolute/path/to/class.jsonl` for **each** of the
eight class identifiers: `wrong_nucleophile`, `wrong_electrophile`,
`wrong_leaving_group`, `formal_charge_corruption`,
`regioisomeric_disconnection`, `bond_order_corruption`,
`missing_necessary_fragment`, and `executor_valid_wrong_successor`.
The output path is passed with `--output`. The intake requires immutable
component manifests/statuses, exactly 400 positives (200/200 strata) and
50 evidence-audited negatives per class. It rejects hash drift, duplicate
proposals, noncanonical model inputs, and any negative product without a
matched recorded positive. A final `r2_plausibility.jsonl`, manifest and
`evaluation_allowed: true` status are written only after all gates pass.
The final cohort still keeps evidence and labels in private fields; only
`model_input` is supplied to a scoring model.

For seven non-inventory classes, each source row must carry an independently
reviewed evidence reference (`independent_negative_evidence`: kind, locator,
SHA-256 and reviewer ID). The executor-valid/wrong-successor class additionally
requires an accepted replay and its successor. This intake checks the evidence
contract and source attestation; it **does not itself adjudicate chemical
truth**. Source maintainers must retain the cited evidence for manual audit.
The existing 400-positive/50-missing-fragment inputs are intentionally
insufficient, so the final R2 file has not been generated.

Once the cohort and paired model scores exist, run
`scripts/autoresearch/score_r2_plausibility.py --cohort .../r2_plausibility.jsonl
--base-scores .../base.jsonl --mech-scores .../mech.jsonl
--scientific-freeze .../scientific_freeze/manifests/freeze.json
--output .../r2`. Each score JSONL must contain exactly one row per frozen
`proposal_id`, with `support_probability` in `[0,1]`, `compile_status`
(`success`/`failed`) and `execute_status` (`success`/`failed`/`not_run`).
Its sibling `base.jsonl.manifest.json` or `mech.jsonl.manifest.json` must
record `scores_sha256`, the exact R2 `cohort_sha256`, `condition`,
`checkpoint_identifier`, `checkpoint_sha256`,
`input_fields: [product_smiles, proposed_precursors]`, and
`score_semantics: probability_known_valid_proposal`. The evaluator refuses
missing/duplicate IDs, drifted hashes and invalid probabilities; neither
compilation nor execution failure removes a proposal from the denominator.
It writes a `r2/result.json` package with paired AUROC difference and
product-cluster bootstrap interval, AUPRC, 10-bin reliability, execution
coverage and separate executor-valid-negative statistics. It **does not
generate model scores**, and no R2 metrics can be claimed before real Base
and Mech scoring artifacts exist.

## R3 controlled-corruption source

After freezing the 288-row source, run the private one-action oracle audit
before interpreting any repair metric. It replaces only the corrupted action
with the frozen correct action, then replays the unchanged reference suffix
through the same executor. The command deliberately requires RDKit 2026.03.4;
other RDKit versions can serialize mapped states differently and change the
meaning of temporary atom aliases.

```bash
PYTHONPATH=src:. python scripts/autoresearch/audit_r3_oracle_repair.py \
  --source /absolute/path/to/r3_corruptions.jsonl \
  --trace-source /absolute/path/to/flower_inverse_tool_sft_action_delta_v1/test.jsonl \
  --output /absolute/path/to/r3_oracle_repair_audit
```

The audit must show all 288 oracle repairs reaching the frozen precursor and
zero reference-successor divergence. This uses private labels and is an
upper-bound/data-integrity check, never a model result.

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

`scripts/autoresearch/export_r3_queries.py` exports the frozen 288-row
`r3_corruptions.jsonl` as model-facing queries. Run it with `--source` pointing
to that frozen evaluation source and `--output` pointing to a new directory.
Each query has a SHA-256 case ID and exactly the source's `model_visible`
fields: product, prefix actions, corrupted action and observed executor
result. The correct action, reference successor, suffix, expected precursor,
first-failure index and stratification labels remain in the private frozen
source, not in `model_input`. The export is immutable and inference-only;
its status explicitly forbids training and using the query file as a new
evaluation authority. On 2026-09-29 the actual export was frozen at
`outputs/autoresearch/prepared_eval/r3_flower_queries_v1_20260929/`, with 288
queries and SHA-256
`05c4d8db8cf7eb2ca6b08f1be418b8f3f93d06ea894a4fbf73d91f89953edc38`.
This export is suitable only for **repair at an exposed corruption**. In all
288 frozen cases, `private_reference.first_failure_index` equals the visible
`prefix_actions` length, and the next field is explicitly named
`corrupted_action`. A localization Top-1 score on this input would be trivially
100% by counting actions. Its status therefore sets
`localization_evaluation_allowed: false`. R3's localization objective needs a
separately frozen, unmarked candidate trajectory with no gold-action/suffix
leakage; it must not be scored on this repair query. Neither localization nor
repair has yet been measured.

Once a policy has generated one `apply_electron_flow` replacement per case,
score it with `scripts/autoresearch/score_r3_repair.py`. Predictions must cover
all 288 query IDs, including failed generations (with `repair_action: null`).
The prediction sidecar `<predictions.jsonl>.manifest.json` binds the exact
query SHA-256, prediction-file SHA-256, checkpoint identifier/SHA-256,
`input_fields: ["model_input"]`, and
`prediction_semantics: "one_replacement_action_at_exposed_failure_v1"`.
The scorer rejects other tool names, replays the unchanged private reference
suffix, and counts every missing, malformed, or rejected prediction as a
failure. It must use the frozen RDKit 2026.03.4 and mapped source trace.

```bash
PYTHONPATH=src:. python scripts/autoresearch/score_r3_repair.py \
  --source /absolute/path/to/r3_corruptions.jsonl \
  --queries /absolute/path/to/r3_queries.jsonl \
  --trace-source /absolute/path/to/flower_inverse_tool_sft_action_delta_v1/test.jsonl \
  --predictions /absolute/path/to/predictions.jsonl \
  --output /absolute/path/to/r3_model_repair_score
```

This yields an **oracle-suffix-assisted one-action repair** score, not an
autonomous trajectory success rate or first-failure localization result. The
private correct action is never model input. No prediction artifact exists yet.

For a nontrivial **synthetic first-reference-divergence localization** input,
`scripts/autoresearch/export_r3_unmarked_localization.py` derives a second
model-facing view from the same 288 frozen source rows. It concatenates the
known prefix, mutated action and reference suffix into one `candidate_actions`
list, then strips every action result. The prompt has only `target_smiles` and
the flat actions—no named prefix, corrupted-action field, feedback, reference
state or per-row failure index. The original frozen source remains the private
label authority. This is not equivalent to proving chemical impossibility:
245 mutations were executor-accepted wrong *relative to the reference*, and
the reference suffix may not execute after a changed step.

```bash
PYTHONPATH=src:. python scripts/autoresearch/export_r3_unmarked_localization.py \
  --source /absolute/path/to/outputs/autoresearch/prepared_eval/r3_flower_closed_shell_event_test_v5_20260929/r3_corruptions.jsonl \
  --output /absolute/path/to/outputs/autoresearch/prepared_eval/r3_flower_unmarked_localization_v1_20260929
```

The frozen query SHA-256 is
`2002816ab898061defe6401dbad828538a8e9fa17639663f234d833c29c8e0eb`.
There are still statistical shortcuts: the most frequent private index is
1 (93/288), and an **in-sample, label-informed** per-action-count majority
rule would reach 122/288. The latter is an optimistic audit, not a legitimate
held-out baseline. Localization results must report a predeclared shortcut
baseline and separately stratify executor-rejected versus reference-relative
accepted mutations. No localization model has been run.

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

An additional **diagnostic** precursor-set source is the authors'
[RetroBridge implementation](https://github.com/igashov/retrobridge), pinned at
commit `5442b4f45edc2e956f1d1c1763bb94cefd4a3a13`, with the
[official Zenodo checkpoint](https://zenodo.org/records/10688201) pinned by
SHA-256 `a78b9e251770e72a53b280fe0f19924e70f52098b2e7b1178538bf8d29bc70cc`.
Its standard inference uses 500 bridge steps and ten independent samples per
product. The repository's raw `score` field is always zero; its evaluation
instead ranks distinct predictions by empirical sample frequency. Our wrapper
uses that frequency, breaking ties by first sample occurrence, and never uses
the recorded precursors to rank. The fixed 200-product cohort has eight exact
product overlaps with RetroBridge's official USPTO-50K training CSV and four
products with Ru/Pd, outside the released checkpoint's atom vocabulary. These
remain in the 200-product denominator as marked overlap or failed inference;
this source cannot support a leakage-clean 200-product headline claim.

```bash
PYTHONPATH=src:. python scripts/autoresearch/run_retrobridge_r5.py \
  --query /absolute/path/to/r5_flower_products_20260929/r5_products.jsonl \
  --query-manifest /absolute/path/to/r5_flower_products_20260929/manifest.json \
  --source /absolute/path/to/pinned/official/retrobridge \
  --checkpoint /absolute/path/to/retrobridge.ckpt \
  --output /absolute/path/to/r5_retrobridge_official_diagnostic \
  --samples 10 --steps 500 --seed 17 --device cuda:0
```

The wrapper checks the official source commit, checkpoint hash, frozen query
hash and source-train overlap before inference. It writes every product row
immediately, can continue an interrupted identical run with `--resume`, and
emits raw predictions, provenance, a run configuration and an overlap audit.
`--limit` and reduced bridge steps are engineering tests only. Once all 200
rows exist, the raw prediction/provenance files can enter the same frozen R5
intake as any other external source, still labeled diagnostic.

When real RetroChimera or collaborator-owned R-SMILES/ReactSeq predictions
become available, normalize them with the frozen intake script. The raw
prediction JSONL has exactly one row per frozen query product, with
`product_smiles`, `inference_status` (`completed` or `failed`), and a `candidates`
list of `{rank, precursors}` entries for ranks 1–5. Missing ranks are allowed
but are explicitly retained as missing in the five-rank denominator. A
separate provenance JSON records `model_name`, `checkpoint_identifier`,
`checkpoint_source`, `training_corpus`, `license_or_terms`,
`input_fields: ["product_smiles"]`,
`target_semantics: "retrosynthetic_precursor_set"`,
`inference_status: "completed"`, and a nonempty `inference_config` object.
Predictions of a complete reaction world, including environment components,
fail this target contract even when every SMILES parses.

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

Once both scientific-smoke checkpoints exist, each valid external candidate
needs candidate-conditioned MechET verification. The model-visible input is
exactly `{product_smiles, proposed_precursors}`; recorded references and
chemical-validity labels remain private. Keep **one row for every product and
rank slot**, including missing/invalid/failed candidates. A verification row
contains `product_smiles`, `rank`, `model_input`, `verification_status`, and
`attempts`. Each attempt records its `termination_reason` and the actual
executor `final_result` if it reached one. The sidecar
`<condition>.jsonl.manifest.json` binds its file SHA, frozen R5 cohort SHA,
`condition` (`base` or `mech`), checkpoint identifier/SHA, `max_attempts`,
`input_fields: ["product_smiles", "proposed_precursors"]`, and
`verification_semantics: "candidate_conditioned_executor_trace_v1"`.

`scripts/autoresearch/score_r5_external.py` scores only an executor-owned,
formally executed, trace-bound terminal result whose *structural precursor*
equals the proposed candidate. It moves trace-supported candidates ahead of
others while preserving original rank within each group. Its outputs include
support by original external rank and known-recorded-reference Top-1 before
and after reranking. A candidate without a matching sampled trace is
**unverified**, not chemically impossible; therefore `unsupported@1` is
explicitly unavailable without independent negative evidence. The scorer
refuses a diagnostic/forbidden source, incomplete product×rank coverage,
source/checkpoint hash drift, and absent scientific freeze. It does not run
candidate-conditioned inference and currently has no real Base/Mech inputs;
synthetic scorer tests are not an R5 result.
