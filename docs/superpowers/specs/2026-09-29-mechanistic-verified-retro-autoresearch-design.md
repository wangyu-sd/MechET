# Mechanistically Verified Retrosynthesis AutoResearch Design

**Date:** 2026-09-29  
**Repository:** `wangyu-sd/MechET`  
**Status:** design frozen; initial campaign machinery and engineering smoke implemented (see `docs/autoresearch/SMOKE_STATUS_20260929.md`); scientific R1–R5 campaign pending source gates

## 1. Goal

Build a smoke-first AutoResearch campaign that evaluates whether the existing MechET system can support a Nature Machine Intelligence-scale story around **mechanistically verified retrosynthesis** without redesigning the model.

The campaign must run **R1–R5 end-to-end before any expensive Qwen3-8B full-data retraining**. Its purpose is to identify which scientific results have real signal, which data contracts are viable, and which experiments deserve full-scale compute.

The controller is not allowed to optimize headline Top-1 by freely modifying the architecture, executor, prompt, or training objective.

## 2. Frozen scientific boundary

The following remain unchanged during the smoke campaign:

- backbone family and tool-calling interface;
- the three public actions: `import_fragments`, `apply_electron_flow`, `finish_trace`;
- deterministic MechET executor semantics;
- State-SFT / compressed-history Trajectory-SFT / EARHO scientific definitions;
- endpoint canonicalization and complete-denominator benchmark evaluators;
- current closed-shell, two-electron polar chemistry scope.

The only permitted training-side scientific intervention is:

> add replay-compatible curated mechanism rows to State-SFT.

Trajectory-SFT and EARHO remain the current implementation. If a new mechanism source cannot be represented losslessly under the current executor, the row is quarantined rather than changing the executor.

## 3. Smoke-first philosophy

There are two smoke levels.

### 3.1 Engineering smoke

Purpose: prove that data conversion, executor replay, optimization, inference and scoring run without changing the scientific contract.

Default:
- Qwen3-0.6B;
- 32 training rows;
- `--max-steps 100`;
- one small held-out batch;
- no scientific claim.

The repository already supports this pattern for Tool-SFT and should be reused.

### 3.2 Scientific smoke

Purpose: produce directional R1–R5 results before 8B/full-data training.

Default model:
- Qwen3-0.6B at the already pinned immutable revision used by the small-model Tool-SFT path;
- LoRA / assistant-only loss;
- same tool schema and executor as the paper-scale model.

Target training budget:
- 12,000 accepted trace-owned training rows per condition;
- deterministic stratified sampling;
- one fixed campaign seed for dataset construction;
- Base-smoke and Mech-smoke must be compute matched.

Conditions:

**Base-smoke**
- 6,000 FlowER rows;
- 6,000 mech-USPTO-31k rows.

**Mech-smoke**
- 4,500 FlowER rows;
- 4,500 mech-USPTO-31k rows;
- 3,000 accepted curated-mechanism rows (initial target: PMechDB-compatible training portion).

If fewer than 3,000 curated rows survive executor replay, use all compatible curated rows and backfill the remaining quota equally from FlowER and mech-USPTO. The controller records the resolved counts and does not silently oversample.

No R1–R5 final evaluation row may enter the scientific-smoke training mixture.

## 4. Stratified sampling contract

Random sampling alone is forbidden for scientific smoke.

Every selected training/evaluation row must carry a machine-readable stratum record with:

- stable ID;
- source dataset;
- split;
- mechanism / reaction class when available;
- trajectory length bucket;
- coupled electron-move coordination bucket;
- structural novelty bucket when computable;
- reaction-frequency bucket when computable;
- inclusion reason;
- manifest hash.

### 4.1 Shared strata

Where the source supports the required metadata, sample across:

1. **Source**
   - FlowER
   - mech-USPTO-31k
   - curated mechanism source

2. **Trajectory length**
   - Q1
   - Q2
   - Q3
   - Q4

3. **Coordination**
   - one electron-pair move
   - two coupled moves
   - three or more coupled moves

4. **Structural novelty**
   - high overlap
   - medium overlap
   - structure-clean / low overlap

5. **Reaction frequency**
   - common
   - mid-frequency
   - rare

6. **Mechanism class**
   - preserve source-provided classes when available;
   - no single class may dominate a curated-mechanism smoke split merely because it is frequent.

When exact joint stratification is impossible because a cell is sparse, the sampler uses deterministic minimum-per-cell allocation followed by proportional fill and records every underfilled cell.

### 4.2 Manifest freeze

All scientific-smoke manifests are frozen before any model inference on R1–R5.

A failed scientific result does not authorize resampling.

Infrastructure failures may be retried against the same manifest only.

## 5. R1–R5 smoke result packages

All five result packages must execute even if one produces a negative scientific result. Only an engineering failure that prevents a required upstream artifact may block a dependent computation.

### R1 — Multi-reference retrosynthesis evaluation

**Problem:** single-reference exact-match evaluation can mark a documented alternative precursor set as wrong.

**Smoke objective:** determine whether the phenomenon is large enough to justify a main NMI result and whether mechanistic verification recovers known-valid alternatives.

**Target smoke cohort:** 200–300 products after decontamination.

**Required strata:**
- exactly two documented precursor sets vs three or more;
- similar disconnection vs substantially different disconnection;
- high vs low product/precursor structural overlap;
- common vs rare reaction family when labels are available.

**Primary outputs:**
- exact-match false-negative rate on known-valid alternatives;
- forward round-trip recovery when available;
- MechET executable/mechanistic support rate;
- support broken down by alternative-route stratum.

**Data rule:** a candidate counts as known-valid only if the alternative precursor set is supported by an independent held-out reaction record or literature source. Products/precursor pairs overlapping the smoke training mixture must be excluded from the headline smoke cohort.

### R2 — Electron-level plausibility / hallucination discrimination

**Problem:** endpoint models provide ranked precursors but do not identify whether a proposal is electronically supportable.

**Smoke objective:** distinguish known-valid proposals from hard negative proposals at the electron-transfer level.

**Target smoke cohort:** 800 proposals:
- 400 positives;
- 400 hard negatives.

**Positive strata:**
- recorded precursor;
- documented alternative precursor where available.

**Negative strata:** 50 examples each where feasible:
1. wrong nucleophile;
2. wrong electrophile;
3. wrong leaving group;
4. formal-charge corruption;
5. regioisomeric disconnection;
6. bond-order corruption;
7. missing necessary fragment;
8. executor-valid but wrong-successor / chemistry-inconsistent proposal.

The eighth class is mandatory and must be reported separately because trivial formal invalidity is not sufficient evidence of a useful mechanistic verifier.

**Primary outputs:**
- AUROC;
- AUPRC;
- calibration / reliability curve;
- executor coverage;
- performance on executor-valid hard negatives;
- Base-smoke vs Mech-smoke paired delta.

No proposal that fails compilation is silently dropped; compilation status is part of the outcome.

### R3 — First mechanistic failure localization and repair

**Problem:** existing retrosynthesis evaluation says a proposal is wrong but not where it first becomes chemically inconsistent.

**Smoke objective:** locate a controlled trajectory corruption and test whether the existing correction machinery can repair it.

**Target smoke cohort:** 288 corrupted trajectories where data permit, arranged as:

- failure depth: early / middle / late;
- coordination: 1 / 2 / >=3 moves;
- 32 examples per depth × coordination cell.

**Primary outputs:**
- first-failure localization Top-1;
- first-failure localization distance;
- executable repair rate;
- exact precursor recovery after one localized repair;
- number of altered electron events;
- performance by failure depth and coordination.

EARHO may be reused as the repair policy. The smoke campaign does not redesign the EARHO objective.

**Localization input audit (2026-09-29):** the currently frozen R3 source
exposes a known-good prefix followed by a separately named corrupted action.
In all 288 rows, the private first-failure index equals the visible prefix
length. Therefore that source supports repair-at-known-failure but **cannot**
support a nontrivial first-failure localization Top-1 result. Before reporting
localization, freeze a separate unmarked candidate-trajectory cohort where the
failure position is not recoverable from input structure; do not expose the
correct action or reference suffix as a model-visible hint. Preserve the
existing 288-row source and its repair scores as a distinct diagnostic.

### R4 — Independent mechanism and pathway validation

**Problem:** a model trained on mechanistically derived reaction corpora may still be learning source-specific conventions rather than transferable chemistry.

**Smoke objective:** test transfer to independent curated mechanisms and longer textbook pathways.

**Evaluation sources:**

1. **PMechDB challenging test**
   - use the complete official challenging test when licensing/access permits;
   - no rows enter training.

2. **PMechRP textbook pathways**
   - use the complete public 350-pathway benchmark when accessible;
   - no rows enter training.

3. **Literature catalytic-cycle mini-set**
   - target 12–20 high-quality closed-shell polar cycles for smoke;
   - mechanism classes should include, where compatible with the current executor:
     - nucleophilic / acyl-transfer catalysis (for example DMAP-like);
     - enamine / proline catalysis;
     - iminium catalysis;
     - Brønsted acid/base catalysis;
     - simple Lewis-acid catalysis.
   - transition-metal redox, photoredox, radical and SET cycles are out of scope for this smoke campaign.

**Primary outputs:**
- decision / electron-event agreement;
- map-invariant successor agreement;
- formal execution;
- H=1/2/3/full-path successor-chain recovery;
- first-error depth;
- success versus pathway length;
- catalytic-cycle closure when the literature cycle is represented;
- catalyst identity regeneration and net-reaction consistency for the catalytic mini-set.

The paper may claim recovery of literature-supported elementary transformations/cycles, not discovery of the unique physical mechanism.

### R5 — Mechanistic verification of external retrosynthesis models

**Problem:** strong endpoint models may rank unsupported candidates highly, while exact-match evaluation cannot separate valid alternatives from hallucinations.

**Smoke objective:** use MechET as a mechanistic verifier/reranker for candidates produced by an external retrosynthesis system.

**Initial external source:** RetroChimera public predictions/checkpoint when available; existing stored predictions from R-SMILES/ReactSeq may be added without requiring retraining.

**Target smoke cohort:**
- 200 products;
- Top-5 candidates per external model;
- approximately 1,000 proposals per model.

**Required strata:**
- candidate rank: 1, 2–3, 4–5;
- model agreement: consensus vs model-unique candidate when multiple sources exist;
- exact recorded precursor vs known alternative vs unsupported;
- common vs rare reaction family where available.

**Primary outputs:**
- executable fraction by external-model rank;
- high-ranked unsupported fraction;
- known-valid recall before and after MechET reranking;
- unsupported@1 before and after reranking;
- effect on rare / OOD strata.

Public RetroChimera checkpoints trained on broader proprietary/historical corpora are diagnostic only on overlapping USPTO-derived chemistry. They cannot support a leakage-clean headline improvement unless training-universe overlap is controlled.

Distillation is explicitly not part of the first smoke campaign. It is promoted only if R5 produces a substantial pool of distinct, executable, non-reference alternatives.

## 6. AutoResearch controller

The controller is a finite-state campaign runner, not an unconstrained research agent.

### 6.1 State machine

```text
FREEZE_MANIFESTS
      |
      v
ENGINEERING_SMOKE
      |
      v
MECHANISM_COMPATIBILITY_AUDIT
      |
      v
TRAIN_BASE_SMOKE --------+
      |                  |
      v                  |
TRAIN_MECH_SMOKE         |
      |                  |
      +------------------+
      |
      v
RUN_R1
RUN_R2
RUN_R3
RUN_R4
RUN_R5
      |
      v
COLLECT_SCORECARD
      |
      v
RECOMMEND_SCALE / RECOMMEND_REVISE_DATA / NO_GO
```

R1–R5 should run independently once their prerequisites exist. Scientific failure in one result package must not cancel the others.

### 6.2 Allowed automated decisions

The controller may:

- freeze deterministic stratified manifests;
- run deterministic converter/replay audits;
- submit approved smoke configs;
- monitor Taiji task/instance state;
- retry infrastructure failures at most twice;
- switch among pre-approved resource profiles without changing the scientific condition;
- reduce per-device batch size after OOM only if gradient accumulation restores the same effective batch;
- collect machine-readable metrics;
- compute paired deltas and bootstrap intervals;
- produce a campaign scorecard and scale recommendation.

### 6.3 Forbidden automated decisions

The controller may not:

- alter the executor;
- invent a new action vocabulary;
- modify prompts based on test outcomes;
- change the model architecture;
- change R1–R5 manifests after observing results;
- silently remove failed examples;
- add training epochs/seeds because a result is unfavorable;
- select a seed for headline reporting;
- search arbitrary loss/reward weights;
- promote a broader checkpoint with unresolved leakage into a clean headline comparison.

## 7. Taiji execution policy

Reuse existing repository infrastructure where possible:

- `scripts/submit_taiji_with_donor_init.py`;
- `scripts/watch_taiji_evaluations.py`;
- `scripts/taiji_run_with_heartbeat.sh`;
- existing training/inference launchers.

### 7.1 Smoke resource order

For Qwen3-0.6B smoke:
1. one H20 if available;
2. one A100;
3. one V100 if the current dependency/runtime supports the config.

Do not reserve 8 GPUs for the initial scientific smoke unless a specific existing launcher requires distributed execution and no single-device path is viable.

### 7.2 Retry policy

Infrastructure retry maximum: two attempts per stage.

Retryable:
- platform failed to start;
- Ceph mount/bootstrap failure;
- transient NCCL/runtime failure;
- OOM resolvable by scientifically equivalent batch/accumulation settings;
- incomplete output caused by infrastructure termination.

Not retryable as infrastructure:
- low accuracy;
- negative scientific delta;
- poor calibration;
- weak R1–R5 signal.

Every retry appends to the campaign ledger.

## 8. Campaign ledger and artifacts

Planned implementation namespace:

```text
configs/autoresearch/
  mechanistic_verified_retro_smoke.yaml

scripts/autoresearch/
  run_campaign.py
  stratified_manifest.py
  evaluate_gate.py
  taiji_backend.py
  collect_results.py

docs/autoresearch/
  MECHANISTIC_VERIFIED_RETRO.md
  DECISION_GATES.md
```

Runtime artifacts are written outside committed source by default:

```text
outputs/autoresearch/mechanistic_verified_retro_smoke/
  campaign_state.json
  manifests/
  jobs/
  r1/
  r2/
  r3/
  r4/
  r5/
  scorecard.json
  scorecard.md
```

Each stage record must include:

- campaign ID;
- git commit;
- executor/runtime revision;
- input manifest hashes;
- model/checkpoint revision;
- exact command/config;
- Taiji task and instance IDs when used;
- attempt number and infrastructure diagnosis;
- output paths and hashes;
- metrics;
- gate/recommendation;
- timestamp.

## 9. Smoke-to-full promotion rule

The smoke campaign is exploratory and must not be presented as the final statistical evidence.

Promotion to expensive Qwen3-8B/full-data experiments requires:

1. all R1–R5 pipelines complete without unresolved data-contract errors;
2. Mech-smoke does not degrade the frozen small held-out endpoint Top-1 by more than 3 percentage points relative to Base-smoke;
3. at least one electron-level external result (R2 or R4) shows a practically meaningful positive directional signal from curated mechanism augmentation;
4. at least two of R1/R3/R5 show a non-trivial usable signal or identify a concrete failure mode worth a final experiment;
5. all test/train overlap and source-provenance audits pass.

The controller reports the evidence; it does not automatically launch an 8B full-data campaign. Expensive promotion remains a human decision.

## 10. Success criterion for this PR

The first implementation PR is successful when:

- the design above is encoded in one campaign config;
- deterministic stratified manifest generation exists;
- a dry-run can enumerate all R1–R5 stages and their prerequisites;
- engineering smoke can be run without touching existing production configs;
- campaign state is resumable and auditable;
- Taiji submission/monitoring is delegated to existing validated helpers;
- no current ICLR/NMI checkpoint or result artifact is overwritten.

This first PR does **not** need to contain final R1–R5 numbers. It establishes the reproducible smoke-first AutoResearch machinery that will generate them.
