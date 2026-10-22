# PR96 execution — Qwen3.5-0.8B main-product-only supervision

Status snapshot: 2026-10-22 23:10 CST. This is execution work on
`exec/pr96-qwen35-dual-mainonly-20261022`, based on the merged PR95 plus PR96
requirements. No new Taiji training job has been submitted yet. No model
accuracy below is inferred from GT replay or unit tests.

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

The new data remain training-disabled pending the remaining gates. The old
PR95 data and its `training_allowed=false` status were not overwritten.

FlowER build is active at
`data/pr96_flower_mainonly_source_arrows_v2_20261022_02/progress.jsonl`:
VALID 2776/2890 replay-passed; TEST 27595/28967 replay-passed; TRAIN is still
processing. Every failure is retained with its source row. These are reference
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
HF-Hub1.5.0, tokenizers0.22.2, FLA0.5.2 and Triton3.3.1 over the proven
Torch2.6/CUDA11.8 meteor base. It is packaged as
`artifacts/pr96_runtime_tf550_peft0181_triton331_py311_20261022_01.tar.gz`.
No global meteor packages were changed. Actual eight-device forward/backward,
LoRA updates, kernel behavior and generation still need deployed preflight.

The trainer additions are opt-in: full epoch guards, matched Qwen3.5 text loader,
all attention/DeltaNet/FFN LoRA targets, exact supervised-token normalization,
epoch validation/best checkpoints, optimizer/scheduler/RNG persistence and
structured default-stdout progress. They are not evidence of a launched job.

## Remaining before and during formal training

1. Complete FlowER reconstruction; classify/fix unresolved cases where reliable,
   then build and audit its history and freeze its own two VALID cohorts.
2. Bind training inputs to reference authorization, pinned token caches and actual
   deployed preflight; audit complete assistant-only tokenization without state
   cropping or supervised token truncation.
3. Finish frozen teacher-forced and true public-root greedy K1 evaluation runners,
   raw-output archival, paired/independent reports, automatic Stage-I gate and
   Stage-II handoff; freeze both launch pipelines.
4. Fresh code review, resource/donor checks, then submit exactly one 8 A100 track
   and one 8 H20 track with meteor naming, proven Ceph init and <=60-second
   default-POD heartbeat. Verify real GPUs and optimizer updates after allocation.
5. Run all requested epochs and validations. Stage-II must fail closed if its own
   fixed gate is not met. No TEST accuracy is claimed until actual inference.

Latest complete local suite at this snapshot: 521 passed, 1 skipped. The isolated
Transformers5.5 protocol/Trainer/token-cache checks also passed (8 tests; two
upstream SWIG deprecation warnings). GPU deployment readiness remains unproven.
