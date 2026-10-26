# PR96 execution — Qwen3.5-0.8B main-product-only supervision

## Latest — 2026-10-26 EARHO result and production implementation

The dated sections below are historical execution snapshots, not current state.
See [the current result and complete evidence](results/PR96_EARHO_20261026.md).
mech31k State10/History5 and FlowER State5/History3 are complete. The four-H20
mech31k EARHO job also completed all10,137acceptedTRAIN reactions/80rounds/
160updates: same1319VALID endpoint hits939→971 (71.19%→73.62%,+2.43pp).
There are74corrections and42regressions; every update included both policy
gradients and explicit fallback replay. This is not new TEST accuracy or
isolated pure-RL attribution. FlowER four-H20 EARHO is still running; its
frozen progress is linked from the current result. No posttraining TEST yet.

This implementation and artifact publication follows the design in PR96;
the original PR96 itself contained only the goal, not these production changes.

## Performance handoff — 2026-10-23 13:22 CST

User requested additional acceleration, resubmission and checkpoint recovery.
The earlier persistent goal remains user-paused; this explicit performance
request authorizes the bounded implementation and single-task replacement.

Frozen implementation: `c42c94bd23c425f7c0f900e2cef195a52a6ed95e`.
584 unit tests passed, one missing-large-artifact test skipped. Independent
read-only review found two handoff races, both repaired and regression-tested:
recheck replacement existence before stop/submit, and pin the authenticated
source checkpoint instead of independently selecting a newer partial directory.

| Performance-only change | Preserved contract / verification |
|---|---|
| Read sequence lengths from Arrow offsets | Exact selected row order, duplicates, slices and native length-grouped sampler ordering retained. The earlier full-cache read measured0.618s; this is metadata loading, not whole-model speedup. |
| Compatible official causal-conv1d1.7.0 CUDA11/Torch2.6 ABI-false wheel | Actual H20 forward/backward tests passed, including6144channels and3419tokens; native Qwen3.5 fast-path enabled in isolated test. BF16 numerical closeness, not bitwise equality or measured throughput. |
| microbatch4/GA2 → microbatch8/GA1, activation checkpointing off | Global batch64, original data/masks/loss/LR/LoRA/seed and State5+History3epochs unchanged. Deployment requires eight-rank finite-backward probe on longest real rows across BOTH stage caches. |
| Recovery checkpoint every1000updates in addition to epoch saves | Full optimizer/scheduler/eight-rank RNG/counters retained. Epoch validation and minVALIDloss selection unchanged; recovery saves do not trigger additional mid-epoch History generation. |
| Explicit hashed performance-resume authorization | Same scientific identity, narrowly whitelisted performance/code differences only. Source/latest and existing validation-best checkpoint hashes bound; actual HF/PEFT tests verify optimizer, scheduler, counters and cursor recovery, including changed microbatch and partial final groups. Numerical matching tests disable dropout; live stochastic updates are not claimed bitwise identical. |

Current source remains `_03`, instance`8b1d8923a162a6a801a1661b13c6047a`.
At13:22CST it was actually training at11500/149070updates, with NO complete
checkpoint. The original configuration saves only after each29814update epoch;
interrupting now would lose the current partial epoch. The existing workers,
configs and code were NOT hot-patched. At the measured old cadence, the first
resumable checkpoint is about6hours away; this is an estimate, not an SLA.

CPU-only operator PID2798887 is active and emits its actual status every45s:
`outputs/agent/pr96_flower_qwen35_08b_mainonly_20261023_06/performance_v1/handoff_status.json`.
It reserves no extra GPUs. Only after a complete source checkpoint matches the
original deployed731identity and ALL component hashes will it stop the exact
source task, verify END/pipeline-lock release, and submit one ordinary8H20`_04`.
All other tasks, including unrelated CRN runs, remain untouched.
At this snapshot `_04` is prepared and render-validated, NOT submitted/queued.
After submission it separately checks actual8H20/PID1 default log bindings,
the new eight-rank native-kernel/max-batch preflight and fresh optimizer updates
beyond the resume anchor before reporting a successful resumed launch.

Same proven group`AI4LSLab_KEMO_ZJK_H20`; BOTH location and quota type empty,
nonelastic/Tencent`metabo:taiji6`/successful31k privateCeph donor, inherited
unbuffered defaultPOD stdout/stderr plus45sheartbeat. Host-only operator logs
are separate diagnostics; model stdout is never redirected to a file.

Frozen deployment artifacts:

- `artifacts/pr96_code_c42c94b_performance_20261023_01.tar.gz`, SHA256
  `1f9f52078e5fadc83e40794dcdf235e0fb58ae717c782584f3b9bfb6b7e8a09f`.
- `artifacts/pr96_runtime_tf550_causalconv170_Tr320_py311_20261023_01.tar.gz`, SHA256
  `cf0276843d01faf90fd2c13529c578d464673ab4d756e8d6a289ca295c26d911`.
- Shared`performance_v1/{plan,control,source_training_identity}.json`, separate
  new`state.yaml`/`history.yaml`, source snapshots and frozen payload;38input
  file hashes checked before waiting and again before stopping.
- `outputs/audits/pr96_causalconv_h20_unit_20261023.{py,json}` preserves native
  operator reproduction code and results. No model experiment/accuracy claim.

No new throughput or endpoint result is claimed before deployment. Data and
token-cache denominators remain exactly as documented below; no unrequested
smoke/subset training/evaluation, TEST inference, RL, or new model condition.

## Current update — 2026-10-23 09:45 CST

PR96 is active and unmerged. The mech31k track completed all10State and5History
epochs, its paired and independent VALID confirmation, and final artifact checks.
Taiji reports IsSuccess=true/END; its8A100 are released. The FlowER
reference rebuild, independent full-history/quarantine and original-source
consistency audits completed. Its new reference subset is authorized; both
complete token caches passed their manifest checks and independent source/Arrow
rehashing. The first H20 submission immediately ended before POD allocation:
Taiji rejected the unsupported explicit location `zjk` (-1100408). A replacement
payload `_02` removed that alias but also ended before allocation: its copied
A100 `quota_type=public` requested a zero-public-quota H20 pool. The replacement
`_03` leaves both `location` and `quota_type` empty, matching successful H20
requests in the same application group. Scientific/deployment fields are retained.
The `_03` job is now allocated and its POD is Running: eight actual NVIDIA H20,
Ceph and PID1/default-stdout binding are verified. Both complete local caches and
the actual eight-GPU model preflight passed. The one-time CPU length scan has
finished, and formal State-SFT is now making sustained optimizer updates.
First formal step1 was recorded09:27:52CST; step10 has a positive learning rate.
The independently checked09:44:36CST record is800/149070updates (0.5367% of
the full5epochs), loss0.0627604, finite gradient norm2.0662, global consumed
examples51200 and zero reported exceptions. This is training progress, not
endpoint accuracy. No epoch checkpoint or validation result exists yet.
No complete TEST model evaluation has been executed on either track.

Current FlowER task:
`meteor_mechet_pr96_flower_qwen35_08b_2stage_8h20_zjk_20261023_03`, instance
`8b1d8923a162a6a801a1661b13c6047a`, submitted08:13:33CST, platform
TRAINING_RUNNING. Formal model training is separately proven by its sustained
step journal and nonzero-learning-rate updates, not by this platform label.
Use `submission_receipt_03.json` in the FlowER run and live instance/pipeline
status, not the initial `_01` submission receipt or preparation-time plan flag.
Actual device/mount/log-chain evidence:
`outputs/audits/pr96_flower_deployment_03_h20_pod_readback_20261023_0817.json`.
Node cache `/tmp/mechet-pr96.JVB7sd` has3.3Treported free space; official base
files10/10have been copied/verified. Pipeline and heartbeat processes write to
the same stdout pipe as PID1. The logs API can lag; this does not claim its
bootstrap-only tail already displays current heartbeat or training updates.
State and History local cache audits passed08:29:19/08:43:52CST, with all16Arrow
files and complete manifests per stage, no state cropping and768generation tokens.
The eight-H20 preflight manifest is
`preflight_20261023T004353/manifest.json` under the FlowER run, SHA256
`e15ff0e3de3eb3c97acda70cef6770e3733b643d490420e0c5ddbb168505ea1c`.
All ranks passed finite nonzero gradients, actual LoRA updates, DDP synchronization,
assistant masks/tool prefix parity, language-only targets and generation. These
were discarded scratch updates, not formal SFT progress or model accuracy.
Actual runtime is torch2.6.0+cu118/TF5.5/PEFT0.18.1/Triton3.2/RDKit2026.3.4.
Independent deployed-validator readback:
`outputs/audits/pr96_flower_h20_preflight_verified_20261023_0853.json`.
The formal State launcher started08:48:59CST, preserving5epochs/max_steps=-1,
own-gate History3epochs, independent base initialization and all frozen data.
The eight workers6847–6854completed their startup hash checks and loaded the
model. Nonblocking read-only profiler samples at09:11/09:14identified the earlier
pre-first-batch delay as `Dataset.__iter__` inside `LengthGroupedSampler`, not
CUDA/NCCL. That scan subsequently finished; it is not the current phase.
The Arrow cache has no length column, so each rank formats all1908035decision
rows to infer lengths before its first batch. This was startup scanning, not
eightfold extra training data. Earlier memory-map observations established
forward traversal only; formal progress now comes from actual optimizer logs.
Raw stack: `outputs/audits/pr96_flower_state_startup_stack_20261023011410Z.txt`;
diagnosis: `outputs/audits/pr96_flower_state_startup_length_scan_20261023_0916.json`.
The current job is not restarted/hot-patched; model/data/epochs/sampling remain
unchanged. A future implementation can derive the identical lengths directly
from Arrow offsets before Trainer construction. Do not substitute the preflight's
one scratch update for a formal step.

Formal-start verification:
`outputs/audits/pr96_flower_formal_training_positive_steps_20261023_09.json`.
All81logged positive-step records through800are strictly increasing, retain
149070total steps/5epochs, have finite losses/gradients and zero exceptions;
global example counts match64per completed update in this early full-batch
portion. This does not extrapolate the final partial accumulation/padding count.
The audit binds the official base, source approval, State token cache and actual
eight-H20 preflight hashes. At09:45:43CST, the POD readback reconfirmed Ceph,
all eight live workers and their shared PID1 stdout/stderr pipes. Seven GPUs
sampled100%utilization and one13%; this instantaneous snapshot is not a sustained
utilization average. Heartbeat cumulative writes advanced to26568bytes.
The logs API tail remains bootstrap-only and is not used
as evidence of fresh training output. No scientific configuration, source,
sampling order, code/runtime package, epochs or checkpoint lineage was changed.

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

Additional required fine-grained metrics from these same selected checkpoints:

| VALID metric | State paired64 | History paired64 | History independent64 |
|---|---:|---:|---:|
| TF root IMPORT component precision / recall |90.00% /90.00%|81.67% /81.67%|88.33% /88.33%|
| TF electronic event exact |156/166|160/166|162/166|
| TF electron-source site exact |157/166|161/166|162/166|
| TF electron-destination site exact |156/166|160/166|162/166|
| TF formally executable decisions |330/330|329/330|330/330|
| Closed-loop actual FINISH |62/64|62/64|64/64|

The denominator166is FLOW decisions conditioned on the reference prefix, not
166independent reactions or autonomous endpoint hits. Component precision/recall
are micro-aggregated matched/predicted/gold fragments on the60root-IMPORT cases;
each happens to have one component here. Formal execution is distinct from
reference chemical-successor matching and from a correct complete endpoint.

Existing-output episode analysis accounts for all64IDs in each report. Among
the12State endpoint misses,8first diverge from the reference at depth0;
History paired has11of13misses first diverging at depth0, independent7of12.
Depth is zero-based. Two State episodes first diverge at depth1but subsequently
reach the exact complete endpoint: a reference divergence is not automatically
a chemistry error or an endpoint failure. The independent History cohort has
12terminal endpoint mismatches and no executor rejection; its64FINISH outcomes
must not be presented as64correct endpoints. Saved per-case classifications,
all24rank-result hashes, report/checkpoint bindings and full metric dictionaries:
`outputs/audits/pr96_mech31k_selected_fine_grained_existing_outputs_20261023_01.json`.
This reuses captured VALID outputs only; no new generation, TEST evaluation,
checkpoint selection or label change was performed.

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
History tokenization automatically started at06:59:39CST and completed with
exit0 at07:42:36CST. Its complete manifest SHA256 is
`6795858154bc84cdac2bdcdadc4d86b3da8a8731ecca61667f03863ae93c9042`.
History also has1908035TRAIN/409VALID decisions, zero truncation/windowing/
fallback and correct owned XML masks. Its TRAIN input contains2646532471tokens,
with the same251982800supervised tokens; longest input3529/supervised span600.
Both actual immutable731 token validators passed, deriving768-token generation
budgets from each TRAIN cache. The completed token workflow is not running now.
The sole submission operator2166044 completed independent hashing of every
authorized source/history file and every Arrow shard: State passed07:50:30,
History07:55:18. It submitted `_01` at07:55:21 and then ended normally. Its
successful create/start response was not a deployment pass: instance
`8b1d8923a162a6a801a1660a6a380474` is END/IsSuccess=false, and the logs endpoint
confirms no host ever ran. No GPU model training or TEST inference ran there.
Failure evidence is preserved in
`outputs/audits/pr96_flower_deployment_01_region_failure_20261023.json`.
The `_02` instance `8b1d810aa162a6bb01a16612ed32046f` also ended before any host
ran (1100400: public Beijing H20 quota0), despite the group's own ordinary
quota snapshot136/86used/0waiting. This was a quota-routing override, not proof
that the entitled business group had no capacity. Evidence:
`outputs/audits/pr96_flower_deployment_02_public_quota_failure_20261023.json`.
Neither failed instance created pipeline, preflight or model outputs.
Their logs are `outputs/audits/pr96_flower_after_build_20261023_06.log` and
`outputs/audits/pr96_flower_token_after_approval_20261023_06.log`.
No duplicate builders or premature training jobs are needed.

The replacement config is
`configs/taiji/meteor_mechet_pr96_flower_qwen35_08b_2stage_8h20_zjk_20261023_03.json`.
Only the task flag, description, region and quota-routing overrides change from
`_01`; both failed tasks/configs/receipts remain historical evidence. The working
same-group forward4H20 and currently running CRN H20 tasks all have empty
`location` AND empty `quota_type`; matching just the region field was insufficient.
Three regression tests reproduced the old request defects and passed on the replacement;
they also ensure all scientific/deployment fields are unchanged. Submission
requires complete reference, history, quarantine and token gates plus fresh
resource/donor checks. Formal
training additionally requires the actual eight-H20 compatibility preflight
after allocation; a successful submission alone is not a training pass.
It uses independent FlowER initialization
and the already deployed731af13 GPU trainer, Tencenttaiji6 image, pinned
runtime and45sdefault-POD heartbeat. Local corrected data code is ee9901e;
this is intentionally separate from the immutable running GPU code.

Latest full local verification with the pinned official tokenizer:567passed/
1skipped (the existing optional large-artifact RL test), including the new
region-routing regressions and real FlowER root/late-H fixtures. Tests are not
new experimental results.
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
Full TRAIN history replay completed243753/243753. The token workflow and sole
post-token submission operator completed normally; the original full source,
History and token/Arrow hash gates passed before submission. The live task is
the routing-corrected `_03` described above. Do not start a duplicate operator
or job from the obsolete preparation-time flags/receipts.
Static full-epoch boundary check of the pinned Transformers5.5 implementation:
`outputs/audits/pr96_flower_epoch_budget_static_check_20261023_06.json`.
For1908035training decisions,8ranks,microbatch4andaccumulation2, each epoch
has59627microbatches/rank and29814optimizer updates. Its final one-microbatch
accumulation window is included. Expected State5epochs=149070updates;
History3epochs=89442updates. The actual State trainer now confirms149070total
planned updates;800completed updates are verified separately above. No complete
epoch or History training is claimed. Actual per-rank runtime counters and
sampler padding must be audited at checkpoints rather than inferring final
consumption from nominal global batch64.
Remaining: finishState5epochs, apply its own fixed GO/NO-GO, runHistory3epochs
ifGO, then perform final paired+independent VALID verification and archive the
selected-checkpoint-bound audited TEST plan. No actual TEST inference or auto
merge is included in this handoff.

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
