# PR96 execution — Qwen3.5-0.8B main-product-only supervision

## Current update — 2026-10-23 01:43 CST

The original scope is still active, not completed. PR96 remains OPEN and
unmerged at requirement commit `001b761e1a2b0643340d8d24fc1b10586b1baec9`.
The eight-A100 mech31k pipeline is actually running: latest readback3380/6600
State updates (51.21%, epoch5.1212). Latest and validation-best checkpoint3300
has eval loss0.00217462. All optimizer/scheduler, eight RNG and eight counter
files were independently verified against the frozen deployed731af13 code;
adapter SHA256 `04f6be2cdeb1575a60c8a58f308d4abb56332fe7b698d5b09644fed0f88754a3`.
These are training and loss observations, not endpoint-accuracy results.

FlowER `_03` stopped on shared-storage `OSError(EIO)` during output flushing
at65350/257167 training reactions. It is NOT a live builder. Its completed
VALID2776/2890 and TEST27595/28967 reference reports and all partial TRAIN files
are retained and training-forbidden. All114 VALID and1372 TEST failures were
independently reproduced. Two missing-source-bond cases retain both the
actual before-action graph and original reference arrows; none was relabeled
as a passing reaction. Full TRAIN review remains incomplete.

Recovery adds new-version-only, rollback-safe batched appends and a bounded
transient-I/O retry, source/hash/decision/counter checks for the last committed
prefix, and byte/hash-verified reuse of completed splits. Original artifacts
are read-only. Actual new build `_04` is live (PID1992817, session75487):
completed VALID/TEST copies are hash-identical, and10200/65350 committed TRAIN
prefix reactions have been verified/copied. This is storage-recovery progress,
not full training conversion or model accuracy. No source IDs are removed.
All accepted recovered rows still require independent full
history/source replay before authorization. No FlowER H20 job is submitted yet.
The full local suite is554passed/1skipped; this includes actual partial-write/
EIO and end-to-end recovery fixtures, not new model experiments. A real PEFT
interrupted-versus-continuous CPU unit test also verifies exact adapter,
optimizer, scheduler, RNG and token-counter recovery.
Full quarantine review also supports ordered CPU parallelism while preserving
per-case raw states and refusing a final approval after any interrupted review.

The sections below preserve the earlier **00:38 capture-time snapshot**; where
they say FlowER `_03` is active or preflight is pending, the current update above
supersedes them. Live outputs remain authoritative.

Status snapshot: 2026-10-23 00:38 CST. This is execution work on
`exec/pr96-qwen35-dual-mainonly-20261022`, based on the merged PR95 plus PR96
requirements. The first Taiji pipeline failed preflight; its replacement has
passed real eight-GPU compatibility and entered formal State-SFT. No model
accuracy below is inferred from GT replay or unit tests.

## Actual allocation and formal training

Replacement task
`meteor_mechet_pr96_mech31k_qwen35_08b_2stage_8a100_qy_20261023_02`, instance
`8b1d810aa162a6bb01a16470ef310393`, ordinary Qingyuan8A100, Tencenttaiji6.
Actual Ceph mount and8A100-SXM4-40GB verified. Every rank passed finite nonzero
gradients, synchronized DDP gradients and scratch parameter updates. Scratch
weights were discarded. Raw base-model generations are saved; all eight were
truncated and not parseable, so this is **not** a base-model chemistry/format
accuracy pass. The native-template GT parse/mask/prefix round-trip passed.

Formal training reloaded the official base and has reached at least130/6600
optimizer steps (1.97% of10epochs). Step80 loss0.04933/finite gradient norm;
later steps are archived in the machine-readable snapshot below. Stable
throughput is about55–58decisions/sec after startup, suggesting roughly2–3h
for State training including overhead, not a promised completion time. No
endpoint accuracy or Stage-II GO is claimed. Stage-II still requires the same
fixed paired64 gate and own best State checkpoint.

Durable evidence:
`docs/results/artifacts/pr96_execution_20261023/mech31k_launch_snapshot.json`,
and shared `outputs/agent/pr96_mech31k_qwen35_08b_mainonly_20261022_01/`.
All eight training processes' stdout matches the default POD PID1 pipe. The
CLI log endpoint currently shows only bootstrap lines while structured shared
progress updates continuously; do not infer absent training from that CLI view.
The HF telemetry's `memory_rank=-1` is a TrainingArguments label, not an actual
missing GPU; device identities come from the eight real preflight records.

## Retained failed attempt and runtime correction

Update 00:26 CST: first eight-A100 allocation reached the real deployed
preflight, then failed before any formal optimizer update. Triton3.3.1 is
incompatible with Torch2.6's Inductor `AttrsDescriptor` import. The failure and
default POD logs are retained, not reported as successful training. The
corrected isolated runtime uses the base Torch-declared Triton3.2.0; direct FLA
operation imports and actual CPU language-model load (752393024 parameters,
zero missing/unexpected keys) pass. Replacement submission remains required.
Runtime archive:
`artifacts/pr96_runtime_tf550_peft0181_baseTr320_py311_20261023_02.tar.gz`,
SHA256 `a3b6ab71e016cc0cef60b2b273dfb6ccbfcdddc8b2973c6cdd75f317b8d8c8d0`.

FlowER `_02` builder was stopped explicitly after reproducing the native
high-symmetry metal VF2 stall. All partial artifacts remain training-forbidden.
Exact canonical **isomeric** positive checks now avoid that enumeration;
unequal spellings still use the chirality-aware fallback. A real stalled-state
regression also rejects a separate enantiomer mismatch. Full source rebuild
`data/pr96_flower_mainonly_source_arrows_v2_20261023_03` is active with16workers;
it does not reuse incomplete rows or change the original split denominator.

## Fixed scope

- Official `Qwen/Qwen3.5-0.8B`, revision
  `2fc06364715b967f1860aea9cf38778875588b17`; independent actors and caches.
- mech31k: 8 A100, State-SFT >=10 full epochs, then own validation-best parent
  and >=5 history epochs only after the fixed Stage-I gate.
- FlowER strict trace universe: 8 H20, >=5 State-SFT epochs, then >=3 history
  epochs after its own gate. No RL, MCTS, old 0.6B adapter reuse or auto merge.
- Original denominators remain 10152/1319/1253 for the mech31k executable
  trace view, and 257167/2890/28967 for FlowER strict trajectories. These are
  not the raw full 31k endpoint benchmark or the official unfiltered FlowER split.

## Completed data evidence

All paths below are under `/aaa/fionafyang/buddy1/whaleywang/MechET/`.

| Artifact | Current evidence |
|---|---|
| `data/pr96_mech31k_mainonly_source_arrows_v2_20261022_01` | Full original denominators scanned; 10137/1317/1252 accepted. Same 18 unresolved IDs as PR95. Original source steps and full endpoints retained; no replacement of source arrows by graph-delta repairs. |
| Same directory, `quarantine_review.json` | All 18 failures reproduced with original and actual executed states: 15 stereo endpoint mismatches, 3 root stereo round-trip mismatches. None relabeled as a passing reaction. |
| `data/pr96_mech31k_mainonly_history_v2_20261022_01` | All 12706 accepted reactions independently replayed from public roots; every history prefix reconstructed from actual accepted actions; same decision counts as State-SFT. |
| `data/pr96_mech31k_reference_approval_20261022_01/reference_approval.json` | Source hashes, original step payloads/endpoints, source IDs, split separation, quarantine and history verified. Reference subset authorized by code; actual GPU compatibility and complete tokenizer audits still required. |
| Source directory, `frozen_valid_cohorts.json` | Two disjoint 64-reaction VALID sets, each containing 60 root IMPORT and 4 root FLOW cases, stratified by mechanism length and import role. Frozen before training. |

Full State and History tokenizer caches now both contain all 42239 training
decisions and 330 paired-VALID decisions, with zero truncation/windowing. The
State input maximum is 1774 tokens and maximum supervised action is 196 tokens.
The new data remain training-disabled pending actual eight-device preflight. The old
PR95 data and its `training_allowed=false` status were not overwritten.

FlowER build is active at
`data/pr96_flower_mainonly_source_arrows_v2_20261023_03/progress.jsonl`.
The stopped `_02` attempt had VALID2776/2890 and TEST27595/28967 replay-passed;
these are historical partial-build figures, not `_03` completion evidence.
Every failure is retained with its source row. These are reference
conversion checks, not test-set model predictions. Failure review, history,
authorization and tokenizer audit remain pending for this track.

The incomplete FlowER `_01` attempt is explicitly training-forbidden. It was
stopped to fix factorial disconnected-graph matching over identical spectators.
The replacement compares constituent molecule multisets and retains exact
chirality/charge/isotope/H/stoichiometry checks. No executor was changed.

## Model/software evidence

`artifacts/pr96_qwen35_base_manifest_20261022.json` binds all downloaded official
files to HF revision and official LFS/Git fingerprints. In particular the weight
SHA256 is `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`.

The actual official language class `Qwen3_5ForCausalLM` loaded 752393024 text
parameters with zero missing/unexpected keys and no visual parameters. The
native XML tool template and tokenizer prefix were checked; all function names,
parameters and assistant EOS remain model-owned. Empty thinking scaffolding and
future tool responses are not supervised. Selective vocabulary projection has
the same loss and gradients as dense causal CE in regression tests.

An isolated Python3.11 overlay uses Transformers5.5.0, PEFT0.18.1,
HF-Hub1.5.0, tokenizers0.22.2, FLA0.5.2 and base Triton3.2.0 over the proven
Torch2.6/CUDA11.8 meteor base. It is packaged as
`artifacts/pr96_runtime_tf550_peft0181_baseTr320_py311_20261023_02.tar.gz`.
No global meteor packages were changed. Actual eight-device forward/backward,
LoRA updates, kernel behavior and generation still need deployed preflight.

The trainer additions are opt-in: full epoch guards, matched Qwen3.5 text loader,
all attention/DeltaNet/FFN LoRA targets, exact supervised-token normalization,
epoch validation/best checkpoints, optimizer/scheduler/RNG persistence and
structured default-stdout progress. They are not evidence of a launched job.

## Remaining before and during formal training

1. Complete FlowER reconstruction; classify/fix unresolved cases where reliable,
   then build and audit its history and freeze its own two VALID cohorts.
2. Run actual eight-device NCCL/Trainer gradient-accumulation preflight and
   discard scratch updates; only then authorize formal training.
3. The frozen teacher-forced/public-root K1 runners, raw-output archival,
   paired/independent reports, automatic fixed Stage-I gate and own-parent
   Stage-II handoff are implemented. Execute them on each resulting model.
4. Fresh whole-branch review identified five deployment defects; all were
   addressed in the review fix pass: duplicate accumulation scaling, unchecked
   resume lineage/files, interrupted validation recovery, actual VALID source
   bindings, and resumed cumulative token counters. Resource/donor checks precede
   submission of exactly one 8 A100 track
   and one 8 H20 track with meteor naming, proven Ceph init and <=60-second
   default-POD heartbeat. Verify real GPUs and optimizer updates after allocation.
5. Run all requested epochs and validations. Stage-II must fail closed if its own
   fixed gate is not met. No TEST accuracy is claimed until actual inference.

Latest complete local suite: 543 passed, 1 skipped. Pinned Transformers5.5
protocol/evaluation/trainer checks: 26 passed (two upstream SWIG deprecation
warnings); the real two-rank CPU DDP accumulation/update tests also pass using
spawn, including unequal token counts and partial final accumulation windows.
Tests are code verification, not chemical-model experimental results.

The successful Taiji donor actually used Tencent `metabo:taiji6`, not the
`taiji7` in older example configs. A read-only probe in the active MechET
R1/R2 node confirms Python3.11.13, Torch2.6.0+cu118, Accelerate1.9.0 and
Datasets4.0.0. New tasks must use that successful image/Ceph init and a frozen
45-second default-stdout heartbeat. Actual PR96 eight-GPU readiness and model
accuracy remain unproven until allocation and preflight.
