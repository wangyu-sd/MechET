# PR96 execution — Qwen3.5-0.8B main-product-only supervision

## Current update — 2026-10-23 07:01 CST

PR96 is active and unmerged. The mech31k track completed all10State and5History
epochs, its paired and independent VALID confirmation, and final artifact checks.
Taiji reports IsSuccess=true/END; its8A100 are released. The FlowER
reference rebuild, independent full-history/quarantine and original-source
consistency audits completed. Its new reference subset is authorized; complete
History token cache and actual GPU preflight remain required. The complete
State token cache passed its manifest checks; eight CPU ranks have automatically
started History tokenization. No H20 task has been submitted yet.
No complete TEST model evaluation has been executed on either track.

### mech31k: real training and model predictions

Task `meteor_mechet_pr96_mech31k_qwen35_08b_2stage_8a100_qy_20261023_02`,
instance `8b1d810aa162a6bb01a16470ef310393`, uses8A100 in Qingyuan.
The independently initialized official Qwen3.5-0.8B actor completed6600State
updates at02:49CST. Its validation-loss best is checkpoint3300 (epoch5), not
the last checkpoint. Best adapter SHA256:
`04f6be2cdeb1575a60c8a58f308d4abb56332fe7b698d5b09644fed0f88754a3`.
All10epoch losses and full resume files are saved. The42239 unique training
decisions were consumed over10epochs;422400 distributed consumed examples
include one padding example per epoch, not422400 unique examples.

| Fixed paired64VALID metric | State best | History ep1 | History ep2 | History ep3 | History ep4 | History ep5 |
|---|---:|---:|---:|---:|---:|---:|
| Main-product-only greedyK1 structural endpoint |52/64|51/64|53/64|51/64|50/64|49/64|
| Complete precursor-component endpoint |52/64|51/64|53/64|51/64|50/64|49/64|
| TF root IMPORT fragment exact |54/60|49/60|52/60|50/60|49/60|48/60|
| TF FLOW chemical successor exact |157/166|161/166|159/166|162/166|163/166|163/166|
| TF FINISH |64/64|64/64|64/64|64/64|64/64|64/64|
| Truncated generations |0|0|0|0|0|0|

These are frozen VALID predictions, not TEST accuracy. TF conditions on the
reference prefix; greedyK1 starts from a single public product and generates
its own IMPORT/FLOW/FINISH actions. State52/64=81.25%. The64processed episodes
are not64successful executions:62actually reached FINISH; the other2were
rejected. Small paired changes are descriptive, not established improvement.

The State gate was GO, and History automatically started at02:52CST from
this track's best State adapter. History completed3300updates/all5epochs;
the final pipeline finished04:16:40CST and Taiji ended successfully04:16:43.
Per-epoch generation reports and raw inputs/outputs exist for every epoch.
Its predeclared validation-loss best is History checkpoint660(epoch1),
SHA256 `cd5f2f9dbdc883674be2f87781452366a3fb2c01c7c930482e0b5e4b3dab49ca`,
with eval loss0.00205156. The final epoch is not retrospectively selected;
neither is epoch2 picked just for its53endpoint hits. Root model files match
the chosen best adapters, while latest full-training checkpoints remain saved.

| Final greedyK1 VALID result | Structural endpoint | Complete precursor endpoint | TF FLOW successor |
|---|---:|---:|---:|
| State best, paired64 |52/64(81.25%)|52/64|157/166|
| History best, same paired64 |51/64(79.69%)|51/64|161/166|
| History best, independent64 |52/64(81.25%)|52/64|162/166|

The independent64 are a disjoint cohort, not a paired improvement over the
first64. All three reports have zero truncated calls. History improves
conditional FLOW successor matching, but these observations do not establish
a closed-loop endpoint improvement over State. All complete raw reports were
independently reaggregated from their8rank outputs and checked against cohort,
model, data and code hashes. No extra model sampling was run for that audit.

All following paths are relative to `/aaa/fionafyang/buddy1/whaleywang/MechET/`:

- Run/configs/checkpoints/progress:
  `outputs/agent/pr96_mech31k_qwen35_08b_mainonly_20261022_01/`.
- State report and all8rank raw call journals:
  `validation/state_best_paired/` below that run.
- History epoch reports and all raw calls:
  `models/history/epoch_validation/step-{660,1320,1980,2640,3300}/` below that run.
- Final selected-model paired/independent results and all raw journals:
  `validation/history_best_{paired,independent}/` below that run.
- Full training/checkpoint/resume/raw-report verification:
  `outputs/audits/pr96_mech31k_final_track_verification_20261023_01.json`.
- Audited, frozen TEST plan (not submitted or executed):
  `outputs/audits/pr96_mech31k_audited_test_plan_20261023_01.json` and
  `outputs/audits/pr96_mech31k_frozen_test_public_roots_20261023_01.jsonl`.
  All1253original trace-view TEST IDs have verified single-product public roots;
  1252have strict references, while24875retains its endpoint-stereo disagreement
  flag. Future endpoint evaluation must account for all1253, report reference
  availability separately and preserve original GT; TF coverage is1252only.
  This is not the raw3120TEST benchmark. No TEST model generation occurred.
- Independent replay of every captured State model request and action:
  `outputs/audits/pr96_mech31k_state64_raw_replay_20261023_01.json`.
  All64single-component private/public roots,642actual model requests
  (330TF+312closed-loop), exact messages/token IDs, parsed model outputs and
  executor successors reproduce the saved reports. All52endpoint hits reproduce.
  This audit used captured outputs only; it did not run a new model experiment.
- Independent full captured History I/O replay (both final64VALID cohorts):
  `outputs/audits/pr96_mech31k_history128_raw_replay_20261023_01.json`.
  All128single-product public/private roots and1321actual model requests
  reproduce: exact native history prompts/token IDs, raw XML outputs/EOS,
  parsed actions, all recorded successor states, full per-rank scores and journals.
  All532non-root TF prompts are distinct from State-only prompts, confirming
  that actual Stage-II requests include compressed accepted-action history.
  Paired51/64and independent52/64full endpoints reproduce unchanged; this
  performs no new model generation or TEST evaluation.
- Full64case State-vs-History-epoch4 comparison:
  `outputs/audits/pr96_mech31k_stage2_epoch4_pair_changes_20261023_01.json`.
  Same products/GT endpoints and raw result hashes verified:47retained hits,
  3gained hits,5lost hits,9both miss. All5lost hits first diverge at root IMPORT
  (leaving-group identity), not a later invalid FLOW. Two gains correct premature
  FINISH and one fixes a later electron event. A recorded-answer mismatch alone
  does not establish that an alternative reaction is chemically impossible.

### FlowER: complete initial product inventory, not the legacy selected component

The legacy source `target_smiles` was already a preselected component. Merely
rechecking it missed equal-size organic ties and two recoverable cases where
the legacy selector chose an inorganic byproduct. The corrected selector uses
the original target plus the original compiled initial product imports. Later
source imports, including explicit hydrogens, remain available for first-use
IMPORT but cannot win initial product selection. Original electronic arrows,
full endpoints, stereochemistry, charge and explicit H are retained.
The PR96 rule is the unique largest carbon-containing component by heavy-atom
count; it does not add an unrequested metal exclusion or a future-GT heuristic.
It is a declared product proxy, not an independently curated desired-product label.

Full root-construction/inventory precheck examined all289024source reactions:
0unexpected errors. Root-defined counts were255170/257167TRAIN,
2865/2890VALID and28768/28967TEST; remaining root ambiguities are explicit.
These counts are NOT strict electron-replay acceptance or model accuracy.
Evidence: `outputs/audits/pr96_all_product_sides_20261023_06/report.json`.

Fresh full public reference replay completed at04:06:22CST in
`data/pr96_flower_mainonly_source_arrows_v2_20261023_06/`:

| Split | Original denominator | Reference pass | Explicit unresolved |
|---|---:|---:|---:|
| VALID |2890|2751|139|
| TEST |28967|27400|1567|
| TRAIN |257167|243753|13414|

The builder terminated with exit0; all289024original IDs are accounted for,
with273904accepted and15120explicit unresolved reactions across all splits.
Manifest SHA256:
`2a203ee08df9a6099f66d554797a72b93750b88cdfe760669431f2b665ff083c`.
These are reference data checks, not model predictions. All source IDs stay
in accepted or quarantine ledgers. Old `_04` outputs have the wrong root scope
and `_05` was interrupted before the late-H inventory fix; both are historical,
training-forbidden artifacts. They cannot substitute for `_06` approval.

The dependent workflow completed history with16workers and quarantine review
with4workers. All243753TRAIN/2751VALID/27400TEST accepted reactions passed
independent public-prefix replay, including every rendered State and compressed
history decision. Full History manifest SHA256:
`edd299a832e7f15c53d134b74015097865737837d587c723db4527d2b9290f06`.
All online prefixes match, original source-arrow decisions are preserved, and
no failed or future action enters history. The completed history builder is
not a running job; its old PID2142031 has ended.
Quarantine review completed04:13:20: all15120failures reproduced and classified,
with11764public event-successor disagreements,2221main-product ambiguities,
1129endpoint disagreements and6source bonds absent after public-prefix replay.
No quarantined row was relabeled as accepted. The FlowER
paired64+independent64VALID cohorts are frozen and disjoint (each60rootIMPORT,
4rootFLOW; frozen cohort SHA256
`0b2615103ac87146c228da429da060cf2adaa977cee06d74c8cd6bd4b748822f`).
Full source integrity/new-version authorization completed for every original
TRAIN/VALID/TEST ID and file hash, unchanged source steps/endpoints, corrected
main-root/import partition, split separation and independent full history.
The old finalizer2218389 and after-build parent2063994 ended normally; workflow
session96928 returned exit0. New subset authorization:
`data/pr96_flower_reference_approval_20261023_06/reference_approval.json`, SHA256
`eb92320d9711e85bfcd8281cce8bc90cde496b4b47688405372809280d618ada`.
It sets reference_training_authorized=true but training_allowed=false until
complete pinned-token audits and actual deployed-GPU preflight pass. Original
PR95/legacy source statuses were not overwritten.

Independent FlowER run prepared at
`outputs/agent/pr96_flower_qwen35_08b_mainonly_20261023_06/`: State5epochs,
History3epochs only after own State gate/best-parent selection. Both configs
initially have no adapter parent; they cannot reuse mech31k or old0.6B weights.
Each stage has1908035TRAIN decisions and409paired-VALID decisions from64full
reactions. CPU token workflow2067621 completed all State shards and their source/
Arrow hash pass. The State manifest was observed complete at06:59:39CST;
SHA256 `9a986f702b60a5bac43f155f38ceb7e17d04cf9c214fa63d32bdf43eeff4befc`.
TRAIN1908035/VALID409 rows have one supervised decision per row, correct owned
XML assistant masks, no fallback or windowed rows and zero supervised-token
truncation. TRAIN contains2493771031input tokens and251982800supervised tokens;
the longest input is3419tokens and longest supervised span600tokens. The
immutable deployed pipeline derives a768-token generation budget from these
training labels, instead of retaining an insufficient512-token limit.
History tokenization automatically started at06:59:39CST under torchrun2319814,
with all eight CPU ranks active. Its complete manifest remains pending. The
sole submission operator still requires both complete caches and independently
checks all Arrow hashes before creating the H20 task. No GPU model training or
TEST inference is running here.
Their logs are `outputs/audits/pr96_flower_after_build_20261023_06.log` and
`outputs/audits/pr96_flower_token_after_approval_20261023_06.log`.
No duplicate builders or premature training jobs are needed.

The8H20/ZJK config is prepared only. Submission requires complete reference,
history, quarantine and token gates, fresh resource/donor checks, and an actual
eight-H20 compatibility preflight. It uses independent FlowER initialization
and the already deployed731af13 GPU trainer, Tencenttaiji6 image, pinned
runtime and45sdefault-POD heartbeat. Local corrected data code is ee9901e;
this is intentionally separate from the immutable running GPU code.

Latest full local code verification:564passed/1skipped, including real FlowER
root and late-H regression fixtures. Tests are not new experimental results.
Full accepted FlowER budget audit verified all273904reaction IDs/audit hashes:
maximum23decisions and25imported fragments, none beyond64/32public budgets;
all public roots are single components. Evidence:
`outputs/audits/pr96_flower_full_reference_budget_20261023_06.json`.

Additional full TEST input audit completed04:50:59CST (no model calls):
`outputs/audits/pr96_flower_test_public_input_contract_20261023_06.json` and
`outputs/audits/pr96_flower_frozen_test_public_roots_20261023_06.jsonl`.
All28967original strict-trace TEST IDs/source hashes are accounted for.
28768have single-component, stereo-equivalent public/private product roots;
199have non-equivalent largest-carbon-component ties and remain ROOT_UNDEFINED.
The27400strict reference passes are a different count:1368further reactions
have valid roots but unresolved later reference replay. Future evaluation must
keep the full28967denominator, report both availability limits and retain
the199undefined IDs. Do not substitute27400or28768for the full strict TEST;
this still is not the official unfiltered28971TEST. Root selection did not use
future moves or GT precursors. Final model SHA and generation budget will be
bound only after the FlowER model is actually trained and selected. This is
input construction/audit only, not TEST inference or test-set parameter tuning.
Full TRAIN history replay completed243753/243753; the actual post-token
submission operator2166044 waits on token-workflow2067621 and will
verify all completed source/history/token hashes and fresh resources before
submitting the exact8H20draft. It must not be duplicated by a manual submission.
Static full-epoch boundary check of the pinned Transformers5.5 implementation:
`outputs/audits/pr96_flower_epoch_budget_static_check_20261023_06.json`.
For1908035training decisions,8ranks,microbatch4andaccumulation2, each epoch
has59627microbatches/rank and29814optimizer updates. Its final one-microbatch
accumulation window is included. Expected State5epochs=149070updates;
History3epochs=89442updates. These are checked scheduling expectations, NOT
executed training steps. Actual startup/runtime counters and sampler padding
must still be verified after allocation; consumed examples are not inferred
from nominal global batch64.
Remaining: completebothFlowERtoken gates, submit8H20, runState5epochs, apply its own fixed
GO/NO-GO and runHistory3epochs ifGO, then archive final evidence. No auto merge.

## Historical update — 2026-10-23 01:56 CST (superseded)

The original scope is still active, not completed. PR96 remains OPEN and
unmerged at requirement commit `001b761e1a2b0643340d8d24fc1b10586b1baec9`.
The eight-A100 mech31k pipeline is actually running: latest readback3960/6600
State updates (60%, epoch6). Validation-best checkpoint3300
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
completed VALID/TEST copies are hash-identical, and all65350 committed TRAIN
prefix reactions have been verified/copied with unchanged62394accepted/2956
unresolved counts. Sixteen actual CPU workers now process NEW remaining source
rows:76100/257167(29.592%) complete,72655passed/3445unresolved. The original I/O
failure point has been passed. This is reference data preparation, not model
accuracy. No source IDs are removed.
All accepted recovered rows still require independent full
history/source replay before authorization. No FlowER H20 job is submitted yet.
The full local suite is554passed/1skipped; this includes actual partial-write/
EIO and end-to-end recovery fixtures, not new model experiments. A real PEFT
interrupted-versus-continuous CPU unit test also verifies exact adapter,
optimizer, scheduler, RNG and token-counter recovery.
Full quarantine review also supports ordered CPU parallelism while preserving
per-case raw states and refusing a final approval after any interrupted review.

An actual dependent local workflow is waiting (PID2000031, session79776,
`outputs/audits/pr96_flower_after_build_20261023_04.log`). After the complete
manifest exists, it runs all accepted history with16CPUworkers, all unresolved
review with4workers, fixed VALID cohort freezing, complete source authorization,
and training-view preparation. It fails closed if the builder exits without a
manifest or either full audit fails. Do not launch duplicate downstream jobs.

The ordinary8H20/ZJK configuration is prepared but NOT submitted:
`configs/taiji/meteor_mechet_pr96_flower_qwen35_08b_2stage_8h20_zjk_20261023_01.json`.
It uses the same frozen731af13 GPU trainer and proven Tencenttaiji6/runtime
archives as the running A100 track, but an independent FlowER model/data plan.
Naming, default-log wrapper, nonelastic allocation, GPU count and ZJK group/loc
policy checks passed. Actual submission still needs completed data approval,
both full token audits and fresh resource/donor checks; preflight must then
verify eight real H20 GPUs before formal training.

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
