# PR #69 engineering-smoke status (2026-09-29)

This is runtime evidence for the 0.6B engineering gate, **not** a retrosynthesis
benchmark or a Mech-vs-Base scientific comparison.

## Frozen inputs

- State-SFT decisions: 32 total, 16 strict-executable FlowER plus 16
  current-compiler mech-USPTO-31k; frozen train SHA256
  `03885210854d01c12b4dbef3b8ae06cd664ec7f4f6bdd5e4177351d43308be8a`.
- Qwen3-0.6B: immutable revision
  `c1899de289a04d12100db370d81485cdf75e47ca`; downloaded model weight
  SHA256 `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`.
- Frozen manifest and every selected row's stratum record are in
  `outputs/autoresearch/mechanistic_verified_retro_smoke/engineering_freeze/`.

## Verified execution

- First 1xA100 attempt `_01` reached a real Pod and produced heartbeat logs,
  but stopped before optimization because the node could not access Hugging
  Face and lacked this pinned model snapshot. It is not a failed scientific
  result.
- Infrastructure-equivalent retry `_02`, instance
  `8b1d819aa0d2977801a0ebd0b3aa2481`, used the same model revision and
  data from a verified offline Ceph HF cache. Taiji reported successful `END`,
  the wrapper exited zero, and a final adapter was written after 100 optimizer
  steps. The trainer's frozen contract reports 32 rows, the matching train
  SHA256 and immutable model revision. Training runtime was about 253 seconds.
- Held-out local one-decision evaluation used four frozen **validation**
  reactions (28 decisions), not test. First NF4 attempt lacked `bitsandbytes`
  in the selected image. Infrastructure-equivalent unquantized FP16 retry
  `_02`, instance `8b1d8922a0d2976301a0ebdcbb2023cc`, completed 28/28
  decisions with zero missing/extra rows and wrote `evaluation.json`.
  Tool choice was 6/28 and exact decisions 0/28. Fourteen generations had no
  parseable tool call. These numbers only establish that the independent
  inference/scoring path runs and that 32-row overfit training does not
  generalize; they do not estimate the scientific-smoke or 8B model's quality.

## Evaluation-source preparation

The official FlowER full-endpoint **test** split of 28,971 records yields 220
products with at least two distinct recorded structural precursor sets:
191 have exactly two and 29 have three or more. The frozen provisional R1
cohort is at `outputs/autoresearch/prepared_eval/r1_flower_official_test_20260929/`;
75 products have different mapped disconnection signatures, 10 have the same,
and 135 cannot be classified as a product-bond cut by that rule. This is an
independent-record alternative-reference cohort, not proof that every route is
physically feasible. The evaluator still needs to run.

An **existing-model R1 diagnostic** now rescored the already completed
FlowER-full-test Direct Qwen3-8B K=10 predictions on these 220 products. With
one deterministic official-test reaction row per product, sample-0 exact
matches 52/220 single recorded references versus 78/220 when *all independently
recorded* precursor sets for that product are accepted: 26 additional
record-supported matches. Generation-order Pass@10 is 95/220 versus 125/220,
respectively (30 additional matches). This is **not** the PR #69 scientific
R1 Base-vs-Mech result: the checkpoint is an existing FlowER-trained Direct
baseline, product-level train overlap has not been audited for this diagnostic,
and no forward or executor-based chemical support is asserted. The diagnostic
result is at `outputs/autoresearch/prepared_eval/r1_existing_direct_diagnostic_20260929/`;
its selected prediction-row SHA256 is
`27c3d583d99c20e92061b096eea8d52e24d14c3a38d53157bd3ec6e8005748da`.

The same 220 R1 products were also rescored against the **existing legacy
compact-full-state MechET K=10** strict-test predictions, requiring an
executor-owned `finish_trace` result with `ok`, `formal_execute`, and
`trace_bound` all true. All 220 products had ten candidate slots. The first
candidate was formally executable for 7/220 and matched a recorded precursor
for 0/220; across ten candidates, 81/220 products had at least one formally
executable proposal and 6/220 had at least one recorded-precursor match.
There were 127 formally executed candidate slots outside the recorded
reference sets; **unrecorded is not chemically invalid**. The frozen
diagnostic rows SHA256 is
`d4029fc14cd778d1905987b10ef2bf10e02c5eba4e34cc474c969b1a0e8c524d`.
This old A7 run is neither the new Base/Mech scientific smoke nor a matched
head-to-head comparison with Direct; these numbers do not establish an R1
mechanistic advantage and must not be used as its headline result.

The R3 controlled-corruption **evaluation source** is now frozen at
`outputs/autoresearch/prepared_eval/r3_flower_closed_shell_event_test_v5_20260929/`.
It contains 288 replay-audited perturbations: 32 in each early/middle/late ×
1/2/3+-move event cell, from 286 distinct FlowER strict-executable test
reactions. The selected event is classified by its own move count, not the
reaction maximum. All selected trajectories use explicit two-electron
source/sink actions and all recorded states have zero RDKit radical electrons;
radical-pair and BE-delta trajectories are excluded. Of 288 corruptions,
245 execute to a non-reference successor and 43 are rejected by the executor.
The accepted 245 are **reference-relative wrong successors**, not evidence
that each is chemically impossible. Model-visible feedback contains only the
real execution result, never the reference-relative wrongness label. The
cohort SHA256 is `521920c7d8d6dec2a52330bcb51f0131b60b91ad772bf0a4f00988d3cad67116`.
Earlier R3 diagnostic versions are explicitly marked evaluation-forbidden.
No model localization or repair score has been measured yet.
The answer-free R3 inference export is frozen separately at
`outputs/autoresearch/prepared_eval/r3_flower_queries_v1_20260929/`:
288 queries, SHA-256
`05c4d8db8cf7eb2ca6b08f1be418b8f3f93d06ea894a4fbf73d91f89953edc38`.
It retains the corrupted action, reference prefix and real executor feedback
but removes every `private_reference` field and expected endpoint. This is a
**repair-at-exposed-failure** input, not a localization test or measured R3
result. A protocol audit found that the frozen first-failure index is exactly
the visible prefix length in **288/288** cases; counting prefix actions would
give a vacuous 100% localization Top-1. The query status now forbids localization
scoring. A separate unmarked-candidate-trajectory cohort is required to assess
first-error localization without revealing its position.

The R5 **product query** cohort is frozen at
`outputs/autoresearch/prepared_eval/r5_flower_products_20260929/`: 200 unique
official-test products, comprising 100 with independently recorded alternatives
and 100 other products, with 50 products in each empirical heavy-atom-size
quartile. SHA256:
`2eb2b3083ddf9819b145d54be36ae68d41662df6d68676ce6cc5a62c5c1f57e4`.
No external Top-5 predictions or MechET reranking results exist yet; this
query list must not be configured as `r5_external_predictions`.
The official RetroChimera checkpoint links currently return HTTP 403 from
this workspace (read-only HEAD check on 2026-09-29). No model weights have
been downloaded or substituted from an unverified mirror.
The R5 external-prediction intake is now implemented and tested in
`scripts/autoresearch/freeze_r5_external_predictions.py`. It requires actual
model provenance and exactly the frozen 200 product rows, preserves all five
rank positions including invalid/missing predictions, and marks non-reference
proposals as **unrecorded, not proven invalid**. It has not been run on a real
external prediction file; `evaluation_sources.r5_external_predictions` remains
unset. No collaborator-owned R-SMILES/ReactSeq prediction artifact was found
in this workspace during the current audit.

The R2 **recorded-positive proposal source** is frozen at
`outputs/autoresearch/prepared_eval/r2_flower_recorded_positive_test_20260929/`:
400 product/precursor proposals from the official FlowER test records, split
into 200 recorded precursors and 200 alternatives with a second independent
held-out record for the same product. Each positive stratum contributes 50
products to each empirical product-size quartile. The cohort SHA256 is
`4fef676bd9c7b12edde7a5507f1630e1e6f5a927a27202f0f60941edd1cb380a`.
Record IDs and evidence are private labels, not model-visible inputs. A
recorded route is not a claim of unique physical mechanism.

One R2 **evidence-audited negative class** is frozen separately at
`outputs/autoresearch/prepared_eval/r2_flower_missing_fragment_negative_elemental_v2_20260929/`:
50 proposals (25 from each positive source stratum) created by omitting a
precursor component so the remaining **unmapped elemental inventory** lacks
at least one product element. Every row stores the missing element counts and
the supporting atom maps; the elemental deficit, not map-label difference,
makes the stated precursor inventory insufficient under atom conservation.
The cohort SHA256 is
`de0fc5bfbd385f620929bd342cf4dbff88aa429a79494926b6ede499ab74fe87`.
This is a closed-inventory contradiction, **not** a claim that an unlisted
external reagent could never supply the atom. Both R2 source directories have
`evaluation_allowed: false`: the other 350 hard negatives, including the
mandatory executor-valid chemistry-inconsistent class, have not been
evidence-audited. A non-reference but executor-accepted successor from R3
does not automatically qualify as an R2 chemical negative.
The earlier map-label-only R2 negative diagnostic is marked
`evidence_audited: false` and superseded; it must never be counted.
The R2 final-cohort intake (`scripts/autoresearch/freeze_r2_plausibility.py`)
now enforces eight 50-row audited negative sources against the frozen 400-row
positive source. Its tests cover a complete synthetic 800-row intake,
missing-class/hash-drift rejection, and rejection of an executor-valid negative
without independent evidence. **No scientific R2 800-row cohort exists yet**;
the seven missing negative classes have not been created or audited.
The paired R2 scorer (`scripts/autoresearch/score_r2_plausibility.py`) is
implemented and tested against synthetic complete scores. It preserves all
800 frozen proposal IDs, including failed executions, and computes AUROC,
AUPRC, calibration, executor coverage, mandatory class-8 performance and a
paired product-cluster bootstrap interval. Real Base/Mech R2 scores do not
exist, so this is an evaluator readiness result, not a chemical result.

## Gates not yet satisfied

No replay-compatible curated mechanism State-SFT rows are configured. The
official [PMechDB download](https://deeprxn.ics.uci.edu/pmechdb/download)
requires a user-side license/registration step, and the public
[elementary-step mirror](https://huggingface.co/datasets/SchwallerGroup/pmechdb_elem)
has no machine-readable license field.
The complete R2/R4/R5 evaluation cohorts are not frozen, so scientific Base/Mech
sampling and all R1–R5 result claims remain blocked by their stated
prerequisites. The controller has not launched 8B/full-data retraining.
The scorecard now distinguishes missing science from a measured negative
result: absent training/metrics yields `INCOMPLETE`. A package merely claiming
`status: complete` is rejected unless its frozen evaluation hashes, scientific
manifest hash, denominators, metrics and paired checkpoint fingerprints are
present and consistent. This is a reporting gate, not evidence that R1–R5
have been run.
An actual collection against the current campaign wrote immutable preview
`outputs/autoresearch/mechanistic_verified_retro_smoke/scorecards/18894a8b079aa5360a4041367bea18b084cfb46321bbe4abb96a6b5ad1234562.json`
with all R1–R5 packages `missing` and recommendation `INCOMPLETE`.
It deliberately did **not** create `scorecard.json`, so later scientific
results can be added without overwriting this progress record.
