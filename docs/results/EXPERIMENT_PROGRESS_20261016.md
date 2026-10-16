# Experiment progress: R1/R2, mech-USPTO-31k MCTS and forward SFT

Live platform readback: **2026-10-16 16:35 CST**. Per-target result snapshots
were read immediately afterward. All three lines are included; this is not
only a forward-training update. No task was submitted, stopped or changed by
this documentation upload.

| Experiment | Requested hardware | Current state | Completed work | Remaining |
|---|---|---|---|---|
| R1/R2 shared multi-step MCTS | 8×A100 | Running | 170/460 targets finished (36.96%); 169 nonzero-step stock-connected, one already in stock | 290 targets still under search |
| mech-USPTO-31k single-reaction MCTS | 8×A100 | Replacement waiting for resources | 1,711/3,120 prior reports preserved (54.84%) | 1,409 targets to resume; replacement has not started new inference |
| Independent forward Qwen3-0.6B SFT | 4×H20 | Running full data conversion | valid 2,890 reactions/21,723 records and test 28,967/216,568 complete, zero replay errors | train conversion → tokenization/audit → two epochs; no optimizer checkpoint yet |

## R1/R2: actual multi-step search

Task: `meteor_mechet_r1_r2_full460_shared_mcts_8a100_qy_20261014_07`.
Instance: `8b1d89c4a13ad33601a13aee1c0b0052`.
Runtime: `e4cb617e8538edadf7b645e0c0c6cda2f8017e51`.
Running since October14 23:00:56 CST (about41.6hours at this snapshot).

| Cohort | Complete denominator | Preserved full greedy: nonzero stock-connected | MCTS: nonzero stock-connected observed so far | Already in stock |
|---|---:|---:|---:|---:|
| URSA Expert | 100 | 1 | 18 | 0 |
| URSA Drugs/Clinicals | 100 | 15 | 41 | 0 |
| RetroCast mkt-cnv | 160 | 17 | 90 | 1 |
| PaRoutes-derived n5 | 100 | 3 | 20 | 0 |
| Total | 460 | 36 | 169 | 1 |

The observed MCTS nonzero stock-closure lower bound is **169/460=36.74%**;
the completed greedy baseline is36/460=7.83%. Search is unfinished and has
used more work plus shared-cache reuse. These are not equal-compute gains or
independent chemical-feasibility rates. The already-stock target is counted
separately, not as a generated synthesis route.

This is actual sampled PUCT MCTS: selection, electronic-event expansion,
executor transitions and path-value/visit backup. A model `finish_trace`
completes one reaction; unresolved structural precursors are recursively
expanded until **all** obligations reach the frozen stock. R1/R2 has no fixed
chemical-depth or total-breadth cutoff; physical/context guards remain.
Shared execution/cache reuse and planning statistics have distinct contexts;
original accepted trees, journals and raw policy outputs remain saved.

The actor is the frozen FlowER Stage-II Qwen3-0.6B with adapter SHA
`40204c7a059d16976fc4b24b315c3bdde7bf61c705d24bbde515071757e9de75`.
No additional model training or learned value model was introduced for this
route search. R2 mechanism-composition C1/C2 analysis remains unfinished;
reference-depth descriptive strata are not substituted for that analysis.

## mech-USPTO-31k: saved single-reaction MCTS continuation

Replacement task: `meteor_mechet_uspto31k_mcts_ray_resume_8a100_qy_20261016_04`.
Instance: `8b1d89f5a13f013001a143290e8a0a56`.
Runtime: `337cba2fb0a445a747cce0d2f4713769d17f7324`; config `7040406`.
The platform reports resource-waiting/transition with
`底层资源不足，请稍侯重试或排队等待。` No replacement output directory exists
at the snapshot. This is **not** new inference progress.

The preceding `_03` instance ended after a CUDA illegal-memory-access report
in the vLLM0.8.5 LoRA expand kernel. Its 1,711 finished reports and all pending
controllers/trees are preserved; 1,409 targets remain. The replacement uses a
tested PyTorch-GPU expand bypass and the same persistent Ray WAL/native tree
cache. It does not restart from products or discard earlier search work.
Production A100 BF16 recovery with this bypass is not yet verified.

Preserved `_03` reports, not the queued replacement's results:

| Post-search diagnostic | Endpoint hits among 1,711 finished targets | Descriptive rate on finished targets |
|---|---:|---:|
| NLL-ranked Top1 | 32 | 1.87% |
| NLL-ranked Top5 | 72 | 4.21% |
| NLL-ranked Top10 / any saved candidate | 79 | 4.62% |

These are incomplete results on the frozen `rxn_prod_min`-proxy **diagnostic**,
not a corrected principal-product benchmark or full-test endpoint accuracy.
The known source-field audit changes799 selected targets when applying the
same selection rule to `rxn_prod_equ`; no source switch is hidden in this run.

This task searches **one reaction**, not a recursive purchase-stock route.
It still uses the declared old stopping budget of **ten total trajectory
attempts**, including invalid/censored attempts—not ten nodes or ten successful
endpoints. TopK is endpoint ranking after search, not the expansion rule.
The budget/scoring limitations discussed with the user have not been silently
fixed by the kernel recovery patch. No GT guides search and no independent
forward chemistry confirmation is claimed.

## Forward next-electron-event training

Task: `meteor_mechet_forward_event_06b_2ep_4h20_zjk_20261016_01`.
Instance: `8b1d8921a13fde2a01a143c120e408c3`.
Runtime: `887da22e3882c263ca1283085f9c5e4925112114`; config `b81aa88`.
Actual four H20s, Ceph and 64 CPU conversion workers were verified.

Original forward FlowER transitions, frozen strict reaction IDs
257,167/2,890/28,967, expected decision records
1,902,076/21,723/216,568. These are not unqualified official FlowER full
denominators (257,171/2,890/28,971). No new filtering. The intended product is
not supplied to the forward policy. A new original-base0.6B LoRA is trained,
not the inverse adapter or the local numerical-smoke adapter. The inverse
weights remain hash-protected. BF16, two epochs, global batch64=4×4×4.

Before submission, repairs covered interleaved reaction grouping, graph-path
history and stable aliases after equivalent mixture reserialization. The
explicitly requested real-data smoke passed933/933 GT closed-loop decisions
and prompt-parity checks. That is software/data verification, not model
endpoint accuracy. See [repair evidence](FORWARD_DATA_REPAIR_SMOKE_20261016.md)
and [forward task detail](FORWARD_4H20_PROGRESS_20261016.md).

The old October13 eight-H20 task failed before optimizer training and is not
the current run. The complete new valid/test conversions have zero replay
errors. Full train conversion is active; training has not reached an optimizer
checkpoint in this snapshot. After full build validation, four-rank
tokenization and intact-conditioning audits run automatically before SFT.

## Uploaded evidence and retained raw artifacts

The companion [snapshot directory](progress_20261016/) contains sanitized
machine-readable task summaries and **all 170 R1/R2 / all 1,711 preserved31k
case-result entries** with original per-case artifact paths. These are result
indexes, not a replacement for raw policy outputs or full trees. Live case files
may advance after this timestamp; the committed snapshot itself is frozen.

Paths below are relative to `/aaa/fionafyang/buddy1/whaleywang/MechET/`:

- R1/R2 trees/raw outputs: `outputs/eval/r1_r2_frozen_06b_shared_mcts_20261014_07/tree/`
- Preserved31k: `outputs/eval/mech_uspto31k_full_mcts_ray_20261016_03/tree/`
- Queued31k future output: `outputs/eval/mech_uspto31k_full_mcts_ray_20261016_04/`
- Shared search cache: `outputs/eval/shared_electron_search_v1/`
- Forward data: `data/flower_forward_qwen_event_v2_20261016/`
- Forward future weights: `outputs/agent/forward_event_flowER_qwen3_0_6b_4h20_20261016_seed17/`
- Source-bound upload snapshots: `outputs/analysis/progress_upload_20261016/`

Code/configuration and earlier MCTS recovery/throughput reports are uploaded
with the existing working branch. Uncommitted unrelated illustrations,
experimental configs and ledger changes remain local. No raw private Ceph
init, credentials, model weights or multi-GB dataset/tree files enter Git.
