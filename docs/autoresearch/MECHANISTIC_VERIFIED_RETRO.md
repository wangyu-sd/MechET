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
