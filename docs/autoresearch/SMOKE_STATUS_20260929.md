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

## Gates not yet satisfied

No replay-compatible curated mechanism State-SFT rows are configured. The
official [PMechDB download](https://deeprxn.ics.uci.edu/pmechdb/download)
requires a user-side license/registration step, and the public
[elementary-step mirror](https://huggingface.co/datasets/SchwallerGroup/pmechdb_elem)
has no machine-readable license field.
R2/R4/R5 evaluation cohorts are not frozen, so scientific Base/Mech
sampling and all R1–R5 result claims remain blocked by their stated
prerequisites. The controller has not launched 8B/full-data retraining.
