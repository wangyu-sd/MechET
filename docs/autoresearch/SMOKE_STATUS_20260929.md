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

The hash-checked **stratified R1 diagnostic** is frozen at
`outputs/autoresearch/prepared_eval/r1_existing_stratified_diagnostic_20260929/result.json`
(SHA-256 `3b5ee8f93727691e5303466d6a6ee0f4d037a9fd486fe5223e116b4f5f883e00`).
Among these deliberately multi-recorded 220 products, accepting all recorded
references recovers **26/220 (11.8 percentage points)** additional first
candidates and **30/220 (13.6 points)** additional Pass@10 products relative
to the selected single reference. The fixed-seed product-bootstrap intervals
for those cohort fractions are 7.7–16.4% and 9.5–18.2%, respectively; they do
not generalize to the complete test population. Of the 26 first-candidate
recoveries, only **4/75** occur in the stratum with a classified *different*
mapped disconnection; **20/135** are in the `disconnection=unavailable` stratum.
That unavailable category is not evidence of a different chemical route.
The legacy trace run has a recorded-reference hit on only **6/220** products
across ten samples, so it cannot yet demonstrate the proposed verification
benefit. These old checkpoints were not compute/protocol matched and no
forward round-trip or independent chemical-validity adjudication was run.

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
That second, unmarked **query view** is now frozen at
`outputs/autoresearch/prepared_eval/r3_flower_unmarked_localization_v1_20260929/`:
288 flat candidate trajectories, query SHA-256
`2002816ab898061defe6401dbad828538a8e9fa17639663f234d833c29c8e0eb`.
It strips per-action feedback and does not expose which action was mutated;
the original R3 source remains the private label/evaluation authority.
An audit found a 93/288 global majority-position shortcut and a 122/288
in-sample, label-informed action-count-majority shortcut. Neither is a model
result, and the latter is not a held-out baseline. This cohort tests first
divergence from the recorded path, not chemical impossibility; model
localization and repair scores are still missing. The new unmarked-query
localization scorer requires all 288 checkpoint- and query-hashed predictions;
its fixed input-only "first electron event" shortcut scores **96/288** on the
frozen cohort. This is a heuristic baseline, not a model result. A private
oracle-only integration check of the scorer reached 288/288 and was deleted
afterward; it must not be quoted as localization performance.
The pinned executor-only replay audit at
`outputs/autoresearch/prepared_eval/r3_executor_rejection_audit_pinned_20260929/`
finds the mutated action as the first rejected step in **43/288** cases
(14.9%). In the remaining 245 executor-accepted mutations, first rejection
occurs **later in 127** and **never in 118**. This is a deterministic shortcut,
not a model result or a judgment that all accepted successors are chemically
plausible. The replay-details SHA-256 is
`9e61f077df1f39f9fa43861f95b439a367aba94241e13127bd8eec0a732c8a22`.

The private R3 one-action oracle was independently replayed through the
unchanged executor with the frozen mapped test source. Under the source
runtime's **RDKit 2026.03.4**, all **288/288** rows reach their frozen precursor,
with zero prefix/correct-step successor divergence. The audit report is
`outputs/autoresearch/prepared_eval/r3_oracle_repair_audit_pinned_20260929/report.json`
(replay-details SHA-256
`1533f51cb541fcd14ed801def078f2c0714b7e5cc877ee0bcac84636f928dbc6`).
Using RDKit 2024.09.6 instead produced 30 false replay failures due to
representation/atom-alias drift; the new auditor now rejects that runtime
before scoring. **288/288 is an oracle data-quality ceiling, not a model repair
rate or localization result.**
The separate model-prediction scorer is implemented at
`scripts/autoresearch/score_r3_repair.py`: it requires all 288 frozen query IDs,
a checkpoint- and query-hashed prediction sidecar, and replays one predicted
electron event before the private reference suffix. No model predictions have
been generated or scored. Its future endpoint rate must be labelled
**oracle-suffix-assisted repair at an exposed failure**, not autonomous
trajectory recovery or failure localization.

The R5 **product query** cohort is frozen at
`outputs/autoresearch/prepared_eval/r5_flower_products_20260929/`: 200 unique
official-test products, comprising 100 with independently recorded alternatives
and 100 other products, with 50 products in each empirical heavy-atom-size
quartile. SHA256:
`2eb2b3083ddf9819b145d54be36ae68d41662df6d68676ce6cc5a62c5c1f57e4`.
An official RetroBridge checkpoint was downloaded and hash-checked from the
authors' [Zenodo release](https://zenodo.org/records/10688201). The pinned
wrapper `scripts/autoresearch/run_retrobridge_r5.py` completed the fixed
200-product cohort with 500 bridge steps and ten samples per product; its
raw-prediction SHA-256 is
`5cbb7fe905aa54dc9fd05ea3f0870ab6280f3711bd18606776019470762b045e`.
The authors' ranking uses sample frequency with first-occurrence tie-breaking,
not their raw `score` as a model confidence. Inference completed for **196/200**
products; **2 Pd and 2 Ru** products were retained as explicit failures because
those atom types are outside the official checkpoint vocabulary. The frozen
precursor-set Top-5 source is
`outputs/autoresearch/prepared_eval/r5_retrobridge_official_frozen_20260929/`
(cohort SHA-256
`cfaf303992fbed8b096949c405c26d51b46325be30894ed10b34d4cadbbd74a9`).
It preserves all **1,000** rank slots: **536 valid SMILES, 181 invalid SMILES,
283 missing slots**. The separate hash-bound source diagnostic at
`outputs/autoresearch/prepared_eval/r5_retrobridge_source_diagnostic_20260929/report.json`
(SHA-256 `dc9c86b783ae31c6752b432b9a4b75a6e1e0bf6b287f95fc351c34e0a14dc1c8`)
finds **52/200** Top-1 and **79/200** Top-5 hits against any independently
recorded precursor set for each frozen product. Among the 192 products with no
*exact product* in the official RetroBridge train CSV, the counts are **50/192**
and **75/192**; exact-product disjointness is not proof of broader molecular or
reaction-level decontamination. Eight products have exact-product train overlap.
This is an external-model source diagnostic, **not** MechET verification,
reranking, chemical-validity adjudication, or a leakage-clean R5 headline result.
The campaign now points `r5_external_predictions` at this frozen source solely
for a **diagnostic** R5 run; its artifact status forbids headline promotion.
No Base/Mech candidate-conditioned trace-support result has been measured.
The official RetroChimera checkpoint links currently return HTTP 403 from
this workspace (read-only HEAD check on 2026-09-29). No model weights have
been downloaded or substituted from an unverified mirror.
The R5 external-prediction intake is implemented in
`scripts/autoresearch/freeze_r5_external_predictions.py`. A real collaborator
Graph2SMILES archive was found under `reflow/outputs/paper_results/baseline_benchmark/g2s/test/`.
`scripts/autoresearch/import_r5_orbit_g2s.py` imports its hashed predictions
without using recorded precursors to choose among duplicate product rows. The
archived file has 28,970 rows (its provenance describes 28,971); all **200/200**
frozen R5 products are covered. Of these, 102 match more than one archived row,
but their first five predictions agree across duplicates. The frozen cohort at
`outputs/autoresearch/prepared_eval/r5_orbit_g2s_frozen_20260929/` contains
**1,000/1,000 syntactically valid Top-5 SMILES**; cohort SHA256
`0d9bf45fb29ac2d0ae36c74e2be3c985a198fb23e1a3f3df78cd018b7b396505`.
This archive is **not** RetroChimera: G2S was trained for only 5,000 updates,
and its original checkpoint is not locally available for independent
verification. More importantly, its target is the **full reaction world**
(including environment molecules), not the retrosynthetic precursor set that
R5 and the MechET endpoint compare. Syntactically valid SMILES do not repair
this target mismatch. The frozen archive is now explicitly
`evaluation_allowed: false`, the campaign's `r5_external_predictions` input is
unset again, and the R5 intake requires a provenance-declared precursor-set
target. Keep this archive for separate completion-task diagnostics only.
An exact-product audit against the archived G2S `flower_completion/train.txt`
(257,171 rows; SHA256 `a258b8137eb38dfca552c223ae73c908d6e3bff0199384469629085c899621df`)
found **47/200 R5 query products present in the G2S training input**. The other
153 are exact-product-disjoint, not necessarily chemically novel. The audit is
at `outputs/autoresearch/prepared_eval/r5_orbit_g2s_frozen_20260929/external_train_product_overlap.json`.
The archived 200-product source is also not leakage-clean headline evidence.
If analyzed as a separate completion-task diagnostic, report all 200 and the
overlap strata separately. No MechET verification/reranking score exists yet.
The paired R5 scoring interface is implemented in
`scripts/autoresearch/score_r5_external.py` and covered by synthetic tests.
It requires Base/Mech candidate-conditioned executor traces for every frozen
product/rank slot, preserves missing/failed slots, and ranks only candidates
with a matching formally executed endpoint ahead of unverified candidates.
It deliberately reports `unsupported@1` as unavailable without independent
chemical-negative evidence. This is evaluator readiness, not a measured R5
model result; the diagnostic G2S archive remains forbidden as R5 input.
No collaborator-owned R-SMILES/ReactSeq prediction artifact was found.

An additional archive audit of the sibling `reflow/outputs/paper_results/`
found no replacement R5 Top-5 precursor-set source **in that archive**; the
separately run official RetroBridge diagnostic above now fills that source role.
On the fixed 200 products,
archived LocalRetro covers all query strings but has a nonempty precursor on
only **4 distinct products** (two are Pd-containing, outside this campaign's
closed-shell polar scope); its file also has duplicate product rows and at
most two saved candidate slots. Archived NeuralSym has **0/200** nonempty
products. These files are hashed respectively
`de98eeb05309bff92a3a3e0288e5116309aad7e769327d350fbcbc081b6e67a4`
and `348a778d0b93b2df95bee39afd9fd3291e4bda56910889159f126af9b4eab20f`.
Retroformer predicts the **full reaction world** and covers only its 9,153-row
length-filtered test subset. `rxngraphormer_retro` instead sees the full
mechanistic mixture and predicts a previous state; neither is product-only
precursor-set retrosynthesis. This audit does not promote a sparse Top-1
archive to R5's ~1,000-proposal Top-5 condition. The official PMechDB portal
still presents a user-side CC-BY-NC-ND agreement; no licensed PMechDB or
PMechRP files have been imported into this campaign.

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

The pinned SynEPD training candidate is configured locally, but scientific
sampling remains unavailable until every R1–R5 evaluation source is frozen.
The official [PMechDB download](https://deeprxn.ics.uci.edu/pmechdb/download)
requires a user-side license/registration step, and the public
[elementary-step mirror](https://huggingface.co/datasets/SchwallerGroup/pmechdb_elem)
has no machine-readable license field.

An independently released candidate, [SynEPD](https://github.com/TieuLongPhan/SynEPD),
was source-audited at Git commit `4fefc016fd4d4e4305dec92586e593bafe3b3e5b`
(`data/polar.json` SHA-256
`84b3d907cc1595269e34ba34be0163c4863f4314f646cd768b44967648210a29`).
Its release manifest declares CC BY 4.0 and 1,926 mapped, closed-shell polar
records with ordered two-electron arrows. All 1,926 preserve the complete
mapped-atom set when explicit hydrogens are retained. Reversing each arrow and
applying the whole coupled event with the **unchanged** MechET executor recovers
the source reactant mixture exactly in 1,754/1,926; 121 fail the executor's
lone-pair precondition, 2 have invalid bond order, and 49 execute to a different
successor. Those 172 are excluded at this preliminary complete-mixture gate,
not repaired by changing the executor.
The 1,754 figure is only a complete-mixture replay diagnostic, **not** a count
of product-only, training-ready State-SFT reactions. Canonical unmapped
product overlap with the official FlowER train/valid/test products is 2/0/0;
the frozen R1 and R5 product cohorts have zero overlap.

The new pinned converter `scripts/autoresearch/build_synepd_curated.py` applies
the existing largest-organic-principal-product policy but rejects tied
principals, imports the other final-mixture components before their first
electron use, converts reversed arrows through the **existing** natural-
language v2 builder, and independently replays every emitted public tool call
without reading the private source moves. It accepted **1,674 reactions / 3,975
State-SFT decisions** and quarantined all remaining **252** records: 170
complete-mixture replay mismatches, 66 public actions that refer to an absent
temporary bond, and 16 ambiguous principal products. The local frozen artifact
is `outputs/autoresearch/prepared_train/synepd_curated_v041_public_replay_20260929/`
with train SHA-256
`a798b6086cfe4157895bd39ac69eaf56afe10ffe56d408f645507608189dfaf2`.
This artifact is not committed as raw data; its source, license, manifest and
quarantine hashes are recorded alongside it.

The largest POLAR family supplies 2,137/3,975 candidate decisions, so naive
proportional selection would violate the PR's no-majority-class rule. The
scientific sampler now allocates by family before joint strata: a 3,000-row
dry selection yields 1,500 POLAR.03 and 1,500 across the seven other families,
with no underfill. This is a **training-source qualification**, not an
independent R4 result or a Mech-vs-Base outcome. The official PMechDB
challenging and PMechRP pathway R4 sources remain separately required; all
R1–R5 evaluation inputs must freeze before either scientific training mix is
sampled. The read-only campaign plan now recognizes the curated source as
available; it does not interpret that as permission to start training.

The complete R2 and R4 evaluation cohorts are not frozen. R5 now has an official,
target-compatible RetroBridge **diagnostic** precursor-set source but no MechET
scores or leakage-clean 200-product headline source; the archived G2S source
still predicts a different full-reaction-world target. Scientific Base/Mech sampling and all paired scientific R1–R5 result claims
remain blocked by their stated prerequisites. The controller has not launched
8B/full-data retraining.
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
