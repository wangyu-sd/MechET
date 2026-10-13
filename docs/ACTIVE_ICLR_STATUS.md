# Active ICLR status

> **2026-10-14 07:51 CST complete-dependency retries supersede _03:**
> Both _03 jobs are verified END/false before model loading (cachetools absent).
> Original failure logs preserved. Runtime5121e5c restores existing CUDA11
> layering: gzip Cu118 vLLM/xFormers overlay + separately checksummed non-CUDA
> fallback appended after base packages; no CUDA12 Torch/driver overrides.
> Actual gzip extraction + automatic complete imports passed locally with
> Torch2.6.0+cu118/vLLM0.8.5/RDKit2026.03.4.778 unit tests passed/11 skipped.
> New ordinary QY8A100 tasks: full3120
> `meteor_mechet_uspto31k_full_dynamic_tree_8a100_qy_20261014_04`, instance
>8b1d8155a134b10601a137ada4f70708; full460
> `meteor_mechet_r1_r2_full460_dynamic_routes_8a100_qy_20261014_04`, instance
>8b1d891ba134b07e01a137ada9c206f5. First readback TRAINING_INIT for both;
> actual GPU generation not yet verified. Same40204... actor/full datasets/
> stocks/seeds/no-hard-cap search; Tencenttaiji6/Ceph/default45s stdout retained.
> Receipts: outputs/submissions/dynamic_electron_tree_20261014_04 and
> outputs/submissions/r1_r2_full460_routes_20261014_04. Others/forward untouched.


> **2026-10-14 07:39 CST bootstrap retries supersede _02 queue notes:**
> Both previous full3120/full460 _02 instances are verified END/false before
> model initialization: missing external zstd command. Default POD logs saved
> under their original submission directories; no model predictions produced.
> Fixed runtime522c048 uses authenticated gzip repack (same uncompressed SHA),
> actual no-zstd full archive extraction passed;777 unit tests passed/11 skipped.
> New full3120 `meteor_mechet_uspto31k_full_dynamic_tree_8a100_qy_20261014_03`,
> instance8b1d8155a134b10601a137a1e8b70704: TRAINING_RUNNING at first readback,
> startup only, GPU inference/default progress not yet verified. New full460
> `meteor_mechet_r1_r2_full460_dynamic_routes_8a100_qy_20261014_03`, instance
>8b1d822aa134b09001a137a1ebb306c1: PENDING at first readback. Same actor40204...,
> full3120/460 sources/stocks/renewable work/no chemical/token/time caps, ordinary
>8A100 each/QY/Tencenttaiji6/proven Ceph/default45s logs. Receipts in
> outputs/submissions/dynamic_electron_tree_20261014_03 and
> outputs/submissions/r1_r2_full460_routes_20261014_03. Others/forward untouched.


## 2026-10-14: full460 observed-work replacement queued

`meteor_mechet_r1_r2_full460_dynamic_routes_8a100_qy_20261014_02`, instance
`8b1d8145a134ab5e01a136e9abba05b2`, runtime4817754d:
TRAINING_RESOURCE_WAITING, not inference. Old full460 instance was stopped
while queued and verified END with no output. Same460 IDs/stocks/frozen0.6B
SHA40204...7e9de75/sampling, but now records whole-wave unique model calls,
all token cost and first stock solutions. No hard chemical/token/time limits.
Ordinary8A100/QY, successful Tencenttaiji6/Ceph donor/default45s logs.
31k task/unrelated jobs untouched. Main receipt
`outputs/submissions/r1_r2_full460_routes_20261014_02/submission.json`.
771 unit tests passed/11 skipped; no new model results. Independent matched-
work sampling and complete Figure2b comparison remain pending. Earlier
e973e1d queued task headings below are historical and superseded.

## 2026-10-14: full3120 uses renewable work windows too

Stopped only the old capped3120 instance8b1d8145a134ab5e01a135c0d60b028a;
verified END and zero output directory. Replacement ordinary8A100/QY task
`meteor_mechet_uspto31k_full_dynamic_tree_8a100_qy_20261014_02`, instance
`8b1d8145a134ab5e01a136cd41690531`, has immutable runtimea23cf576 and the same
frozen Stage-II0.6B/source3120 hashes. Initial state READY, not real inference.
No fixed chemical or token/time stopping cap; only actual context/RAM guards.
Tencenttaiji6/Ceph successful donor/default45s logs preserved. Full460 task and
all unrelated/forward tasks untouched. Receipt in main
`outputs/submissions/dynamic_electron_tree_20261014_02/submission.json`.
The earlier "old3120 still queued" operational notes below are historical.

## 2026-10-14 01:14 CST: full460 route task submitted, dynamic work windows

Task `meteor_mechet_r1_r2_full460_dynamic_routes_8a100_qy_20261013_01`, instance
`8b1d8038a134b09101a136418a43039c`: **TRAINING_RESOURCE_WAITING**, not inference.
One ordinary8A100/QY, successful Tencenttaiji6/Ceph donor/default45s log heartbeat.
Immutable runtimee973e1d, authenticated FlowER Stage-II0.6B SHA40204...7e9de75;
complete URSA100+100/RetroCast160/PaRoutes-derived100 with cohort stocks.
No fixed chemical caps and no token/time stopping ceilings: work checkpoints
renew while frontiers remain, physical memory/context safeguards retained.
Remote CPU64 stock prep precedes A/B greedy/tree inference; ALL nonstock leaves
must close (AND). Frozen R1 outputs feed R2, no RSI/RL, no other task stopped.
734 unit tests passed/11 skipped; zero new experimental results. Exact corpus
indexing/actual-SFT program parity and complete paper figure panels are pending.
The older3120 task remains queued at immutable0796e2d with its old task caps;
the new dynamic-window behavior must not be attributed to that historical runtime.

## 2026-10-13 22:58 CST: dynamic full3120 search, no chemical caps

Submitted ONE ordinary8xA100 QY task
`meteor_mechet_uspto31k_full_dynamic_tree_8a100_qy_20261013_01`, instance
`8b1d8145a134ab5e01a135c0d60b028a`; **TRAINING_RESOURCE_WAITING**, not running.
Runtime `0796e2d30eaf57a5f27a6bdccf1ff0be6be7b493`; actual frozen FlowER
Stage-II Qwen3-0.6B SHA40204c7a...7e9de75, full endpoint RXNMapper test3120.
No fixed depth/beam/import count; adaptive novelty and retained native-state/
history contexts, task-only work and physical-memory limits. No RSI/RL.
Default POD output/45s heartbeat, successful QY donor Ceph/Tencent taiji6.
696 unit tests passed/11 skipped; NO new inference accuracy measured yet.
See `docs/DYNAMIC_ELECTRON_TREE_PROTOCOL.md`, main checkout receipt
`outputs/submissions/dynamic_electron_tree_20261013/submission.json`.
After submission the paper PR29 R1/R2 full-route protocol was reread at
ee96f346ccfc2c40038dd1c4f9b6e7f955915659. Public raw460 route targets and three
stocks are present; full-route execution remains PENDING. Independent forward
8H20 task left untouched and still resource-waiting.

## 2026-10-13: separate forward next-electron-event SFT submitted

One8xH20/ZJK task `meteor_mechet_forward_event_06b_2ep_8h20_zjk_20261013_01`,
instance `8b1d89caa134b07c01a13555c25400d2`. Actual20:56CST readback:
TRAINING_RESOURCE_WAITING; no POD, conversion or optimizer yet. Original
FlowER forward transitions, complete257167/2890/28967 strict IDs, two epochs,
global64, BF16. Independent forward LoRA on original pinned Qwen3-0.6B;
reverse weights hash-protected and unchanged, no other tasks stopped.
Runtime `2ebb892cac0142705bb74e051f17bf4f4eeabce4`; successful Stage-II H20
Tencent taiji7/Ceph donor/default log45s heartbeat. See
`docs/FORWARD_QWEN_EVENT_SFT.md`. Submission is not evidence of running or gain.

## 2026-10-13: joint import–FLOW formal task running; first update verified

Submitted `meteor_mechet_joint_import_flow_full_1a100_qy_20261013_01`, instance
`8b1d8249a1250fb701a133bc40f81482`, ordinary main-QY one A100 using the proven
Tencent `:taiji6` donor/private Ceph init. Live readback is TRAINING_RUNNING;
one A100-SXM4-40GB and Ceph mount are confirmed. First committed checkpoint
has processed1/update1 (replay-only), both policies, two optimizer groups and
all224 Qwen LoRA tensors changed against the SHA-pinned parent. Live journals
continue to grow; no endpoint-contrast update or heldout gain is established
yet. Covers257,167 strict train roots. Model memory observed5055MiB.
Immutable code+CRN runtime commit `1b2bb7972b34118cfc85f87a611a727b0bad9c47`;
all four GitHub CI checks passed. No unrelated task stopped, no new smoke run.
Default worker/heartbeat stdout is connected to PID1's log pipe; the CLI
logs endpoint still shows initialization-only text. Do not hide this API
visibility limitation or mistake it for an absent optimizer/checkpoint.
Output: `outputs/agent/pr84_joint_import_flow_stage1_train257167_20261013_01`.

The user approved the joint-import direction and requested both a code PR and
a Taiji training task. Written design/plan:
`docs/superpowers/specs/2026-10-13-joint-import-flow-endpoint-design.md` and
`docs/superpowers/plans/2026-10-13-joint-import-flow-endpoint-plan.md`.
The user approved direct implementation. The new joint trainer is
`scripts/train_crn_qwen_joint_earho.py`: Qwen policy LoRA and graph heads receive
shared binary terminal credit, with separate replay and original conditional KL.
Its complete-support IID objective, score-space microbackward, paired export/
resume and matched inference have local unit checks. Three Important review
findings were fixed with regressions; local suite673 passed/8 skipped. Formal
submission/host allocation is verified above; no joint empirical gain is claimed.
Do not submit the old graph-only trainer under a joint-policy task name.

The authorized local FP16/T4 diagnostic is finished: validation8 baseline
1/8 structural endpoints, 6/8 terminals, 22/22 imports accepted and 34/36 FLOW
events accepted. Train32 processed all 32 roots but made **zero updates**:
30 controller-blocked roots and two explicit-H alignment KeyErrors. Its
autonomous train episodes had 6/32 endpoint hits before any update; this is
not an RL gain or heldout result. No trained head/post-RL evaluation exists.

Two prerequisites are now repaired with RED-to-GREEN regressions: retain
explicit hydrogen maps during gold/actor alignment, and remove the additional
unchanged-target reward veto that contradicted accepted benchmark GT terminals.
Primary endpoint scoring/denominator and executor legality remain unchanged.
Full local suite after these fixes: **653 passed, 8 skipped**. See
`docs/results/PR83_JOINT_IMPORT_DIAGNOSTIC_20261013.md` for original artifacts
and case-level findings; none establishes performance of a joint model.

Latest ordinary application quota readback (2026-10-13, not host allocation):
main QY A100 112 quota / 59 using / 4 applying / 24 waiting; ZJK H20
136 / 127 / 0 / 10; AILab QY A100 50 / 0 / 0 / 0; KEMO V100
146 / 0 / 0 / 0. Positive quota headroom is not confirmed GPU assignment.
The main-QY Tencent `:taiji6` donor's instance
`8b1d81dca1250f4401a12e1df01d0b75` is END/success, freshly verified.
Do not revive abandoned `:taiji` image or silently stop other projects' tasks.

## 2026-10-12: approved conditional-endpoint Graph-EARHO implementation

The latest user request is to repair code and assess RL's theoretical validity,
not to submit a job. PR83 upstream `584bee6` designates
`scripts/train_crn_graph_earho.py` as main; the fixed-GT-last-three trainer is
an ablation and must not be used by the formal launcher. The local stacked
branch integrates that trainer without losing the single-pass greedy router.
Main defaults are endpoint-only actor support and the same K1 greedy follower
as evaluation; local GT successor matches supply supervised replay only.
Horizon promotion requires real endpoint groups, not just local matches.
New immutable prepared output roots contain `pr83_graph_earho_conditional` /
`pr83_formal_graph_earho_conditional`; legacy root/head reports cannot seal the
new `graph_earho_conditional_endpoint_v4` stage.

The approved v4 implementation replaces that surrogate with complete-support
normalization, IID replacement draws and a leave-one-out score-function loss.
Native pointer exploration expands the support before it is frozen. Only
independently scheduled full-budget groups supply binary terminal RL returns;
short-H groups supply separate replay. Execution pooling preserves every
draw's multiplicity. Autonomous product-start episodes are saved in a journal
with authenticated checkpoint cursors, not embedded in growing checkpoints.
The estimator is unbiased for its declared **conditional fixed-support
frontier objective**, NOT a full-product-start policy gradient or full-policy
KL guarantee. See `docs/GRAPH_EARHO_METHOD.md`. Independent review identified
an omitted precision guard; Qwen dtype/4-bit mode now bind resume and exported
controller contracts. Final local suite: **651 passed, 8 skipped** under pinned
RDKit; details are in the execution ledger. No GPU job was submitted, and these
changes have not yet been pushed/merged. None establishes measured endpoint gain.

Previous formal attempts have not produced inference outputs or RL updates.
Latest inspected `_05` instance `8b1d89c6a1250f2401a12ef0da120bec` is failed/END,
not running. The old `:taiji` image recovery route is abandoned per the user;
do not call the historical attempt-2 PENDING note below a current status.
No replacement job was submitted by this audit/repair.

## 2026-10-12: PR83 formal campaign

The user's full-scale/no-smoke requirement overrides PR83's original 128-valid
and 64-train plan. Use its pinned Qwen3-0.6B Stage-I State-SFT adapter, without
compact history, and frozen CRN-JEPA PR18. Evaluate all 2,890 valid and 28,967
strict test reactions, retaining the 28,971 official test denominator; traverse
257,167 strict train reactions for graph-head-only endpoint optimization; then
compare the trained head with the frozen graph on the same full splits.

One formal task runs at a time. Eight-GPU evaluation uses eight independent
workers; graph-head training uses one A100. Current supervision:
`outputs/submissions/pr83_formal_campaign_20261012/status-ailab-attempt2.json`.
The initial main-QY queue attempt was cancelled before running and replaced by
`meteor_mechet_pr83_formal_pre_8a100_ailab_20261012_02`, instance
`8b1d822ca1250f4c01a12eb27a1d0ca8`. At the last check it was PENDING with an
MPIJob created, not confirmed model inference. Read live state and raw outputs
before reporting progress. Full RL and post-training evaluation are sequenced
by `scripts/run_pr83_formal_campaign.py` and not yet submitted.

The old 64-root RL attempt remains stopped. Existing Stage-II 128-row results
are historical: single-pass hybrid 46/128, pure Qwen 47/128, all first calls
identical. They are not the new formal Stage-I condition. See
`docs/results/PR83_EXECUTION_LEDGER.md` for exact artifacts and rulings.

## 2026-10-07: current reliable-MechET PR #82 handoff

PR #82 is the active lower-cost journal extension of the mechanism-first
three-stage lineage. Its current implementation authority is
`docs/RELIABLE_MECHET_THREE_STAGE.md`; the dated ICLR notes below are historical
and must not be treated as a live Taiji task list. This does not rename the
older A7 conditions or turn the executable trace view into unqualified
"FlowER full".

- Stage I is Qwen3-0.6B State-SFT on the **257,167 / 2,890 / 28,967**
  strict-executable reaction split. The Zhangjiakou 8xH20 candidate
  `meteor_mechet_reliable_state_06b_1ep_8h20_zjk_20261006_01` was cancelled
  while waiting for resources; it had no training POD or result. Its replacement
  `meteor_mechet_reliable_state_06b_1ep_8a100_qy_20261006_01` on Qingyuan
  8xA100 saved a final adapter after 31,366/31,366 optimizer steps and one
  epoch. Final weight SHA-256 is
  `fbd8db06094fd8029d4cb0cac38cbb88280a442d76f686258194723019b1a1bd`;
  it matches `checkpoint-31366` and passed the Stage-II parent lineage gate.
  Taiji nevertheless marks the overall task failed: the wrapper exited with a
  shell quote/EOF error after training and artifact save. Do not relabel the
  platform task successful.
- Stage II compressed-history Trajectory-SFT must warm-start from the **final**
  Stage-I adapter after its SHA-256 gate. Stage III EARHO must likewise wait
  for the final Stage-II SHA-256. Neither stage has been submitted. No formal
  28,971-row endpoint or 28,967-row process test result exists for this new
  three-stage 0.6B run yet.
- The checkpoint-14000 validation pilot is **provisional**, not a final-model
  result: on 16 identical reaction IDs, singleton gold-state decoding matched
  43/63 electron events, while independent product-only K=1 replay reached
  3/16 recorded structural endpoints. The first tool call matched between
  these two singleton evaluations on 16/16 reactions. The local evaluator now
  binds decode batch size; old batched local numbers are not silently pooled.
  The saved 123-decision and 16-episode artifacts, case-level reports, paired
  model-visible inputs/outputs, selection and training snapshot are under
  `docs/results/reliable_mechet_stage1_provisional_valid16_20261007/`.
  The original product-only evaluator saved parsed calls and feedback but not
  raw generation text; the paired export marks those completions unavailable.
  RDKit overview, per-case GT/model trajectory PNGs and a linked
  side-by-side HTML are in the same results directory; these depict the saved
  provisional smoke only and do not change any metric.

Check the current Taiji instance, adapter files and PR checks afresh before
reporting completion or submitting the next stage.

## 2026-09-18: natural-language protocol v2 repair

Keep the agreed mechanism-first method.  The current implementation authority
is `docs/NATURAL_LANGUAGE_PROTOCOL_V2.md`: one identical inventory-bearing
observation for all three tool choices, inference-time private-map mirroring in
the builder, executor-owned successor states, legal repeated imports, and
40-decision/32-import coverage.  Full validation gold replay is 2,890/2,890 and
prompt parity is 22,341/22,341.  Existing v1 history training/smoke remains
historical diagnostic evidence and must not be reported as v2.

The incorrect running v1 history task has now been stopped.  Full remote v2
conversion first found 15 train / 0 valid / 7 test Kekule-drift rows and
correctly prevented training.  Commit `408e132` repairs the fallback delta and
all 22 rows now replay exactly.  Replacement task
`meteor_mechet_nl_protocol_v2_full_sft_2stage_8h20_zjk_20260918_02` (instance
`8b1d8014a0af10a701a0b3501c720673`) is running on 8xH20.  It performs the full
zero-unresolved data gate, then clean State-SFT from the pinned Qwen3-8B base,
then compact-history Trajectory-SFT; submission/running state is not evidence
that either training stage has started or completed.

> This file is the short-lived execution authority for the current ICLR work.
> `PROJECT_MEMORY.md` remains authoritative for historical dataset/model lineage.
> If an older operational note in `PROJECT_MEMORY.md` conflicts with this file,
> this file wins for **current experiment priority and A7 observation choice**.

Historical ICLR section last updated: 2026-09-17.

## Current three-stage method decision

The active algorithm is now described as:

1. **State-SFT** for executable local inverse transitions;
2. **compressed-history Trajectory-SFT** for causal trajectory-prefix
   conditioning without repeated full-state transcripts;
3. **Execution-Anchored Receding-Horizon Optimization (EARHO)** for adaptive
   frontier-local policy improvement under real executor transitions.

EARHO replaces the informal label “hard-case RL.”  It identifies the first
recoverable divergence frontier, pools alternative actions by executed chemical
successor, assigns no positive advantage to all-negative groups, uses verified
transition replay when policy support is absent, and expands the optimization
horizon only as competence advances.  Reference successors and endpoints are
training-private and never enter product-only inference prompts.

The full algorithm contract and evidence gates are in
`docs/EXECUTION_ANCHORED_RECEDING_HORIZON_OPTIMIZATION.md`.  Historical
`anchor_branch` and `successor_horizon` identifiers remain unchanged for run
lineage.

## Natural-language anchor-branch post-training (v4 correction gate)

The bounded productive task ended during its pre-update validation after
228/512 planned candidates (114/256 reactions), so it produced no RL adapter
and must not be reported as a completed experiment. The preserved partial
baseline had 172/228 formal terminals, 45/228 productive terminals, 127/228
unchanged-target terminals, and 1/228 exact endpoints. The exact hit was a
genuine full product-only trajectory; the dominant failure was instead the
action/import branch appending context and finishing without transforming the
target.

PR #59 now carries the v4 implementation correction. The scientific method is
unchanged: exact executor reset, same-state branching, executor verification,
local first-action credit, value-ranked gold-free continuation, and curriculum
toward full product-only episodes. The correction rejects unchanged-target
finish, preserves explicit-hydrogen import participants, adds private
reference-successor credit,
materializes one verified replay record per train reaction, and preserves the
evaluation denominator across isolated collector exceptions. The next gate is
`configs/agent/natural_language_anchor_branch_rl_verified_replay_smoke_a100.yaml`
(64 train reactions, K=8, 32 validation reactions); do not restart the larger
2,560-reaction campaign until this gate completes. It was submitted to ordinary
Qingyuan 8xA100 as
`meteor_mechet_nl_anchor_branch_rl_verified_replay_smoke_8a100_qy_20260917_01`
(instance `8b1d813ea0a4c8ab01a0ad78ebbc0db3`). At the first post-submit check it was
still `PENDING`; submission success is not counted as execution success.

The endpoint-shaped v2 smoke completed successfully.  Its train rollouts had
315/512 formal terminals and one exact endpoint, but that exact candidate was
an H=1 suffix after 7/8 reference events.  Fixed full-episode validation stayed
24/32 formally terminal and 0/32 exact before and after the update.  Of the 24
post-update wrong terminals, 16 retained the unchanged target and appended
fragments.  This is an executable no-op, not a retrosynthetic solution.

Commit `f6bff25` adds a no-transform penalty, a separate productive-execution
metric and persisted per-decision continuation traces.  The full bounded task
`meteor_mechet_nl_anchor_branch_rl_productive_full_8a100_qy_20260916_01`
(instance `8b1d813ea0a4c8ab01a0aaabe7a30977`) uses 10 x 256 distinct train
reactions, K=8, 50% full episodes and a disjoint 256-reaction validation
monitor.  This is not a 257,167-row epoch.  At the initial live check the
ordinary Qingyuan 8xA100 Pod and Ceph mount were present, while the pinned vLLM
runtime was still staging and GPUs were idle; check collectors and GPU memory
before claiming rollout progress.

The current post-training actor is the completed natural-language electron-
event SFT, not the historical Python-template checkpoint. PR #59 implements
executor reset states, K=8 same-state first-tool branches, successor-state
equivalence pooling, endpoint-primary reward and first-tool-only local credit.
The bounded 8xA100 task was
`meteor_mechet_nl_anchor_branch_rl_smoke_8a100_qy_20260916`, instance
`8b1d813ea0a4c8ab01a0a9893c1f075d`. It ended successfully after 16 optimizer
updates. Validation execution improved from 16/32 to 20/32 candidates, but exact
endpoint success remained 0/32; the original SFT therefore remains the selected
adapter. Test was unused.

The failure is traced to sparse reward: only 1/512 training candidates was an
exact endpoint, while 256 wrong terminals all received zero and 255 invalid or
incomplete candidates received -0.1. The prepared v2 repair retains exactness as
the only positive outcome and adds map-invariant, heavy-atom-weighted endpoint
similarity plus first-successor progress to rank non-exact branches. All wrong
terminal and invalid outcomes remain negative. Twenty-five relevant tests pass, and
offline rescoring yields 218 reward values with contrast in 96/128 prompt-mode
groups. Continuation ranking now uses the frozen state-value-v2 adapter over
executor-valid unique successors instead of comparing raw NLL across the action
and event prompt formats. The v2 8xA100 smoke was submitted as
`meteor_mechet_nl_anchor_branch_rl_shaped_smoke_8a100_qy_20260916`, instance
`8b1d81eea0a4bdf601a0aa2e28e108b0`. Live inspection confirmed
`TRAINING_RUNNING`, the private Ceph mount, the wrapper/heartbeat processes and
local vLLM-runtime staging. GPU utilization is expected to remain zero until
that one-time copy finishes; do not report optimizer activity before collector
processes and GPU allocation are observed. See
`docs/NATURAL_LANGUAGE_ANCHOR_BRANCH_RL.md`.

## Natural-language electron-event candidate (active)

- Branch/implementation: `feature/natural-language-electron-sft-20260913`
  (`34074c7`).
- Source denominator: the complete strict-executable FlowER universe,
  `257,167 / 2,890 / 28,967`; this is not the 32k proof subset and must not be
  called unqualified “FlowER full”.
- Observation: target plus the executor's current unmapped SMILES. The
  environment annotates each current atom with a temporary per-step `Axx`
  alias; private atom maps and the next state are not model-visible.
- Prediction: one import, one natural-language retrosynthetic electron-flow
  event, or finish decision. The executor resolves aliases, applies the event,
  and returns the authoritative next state.
- Reproducibility: conversion and replay are pinned to RDKit `2026.03.4`.
  The 256-reaction smoke and complete validation split replayed with zero
  unresolved reactions. Taiji additionally completed the full test split at
  `28,967 / 28,967` with zero unresolved reactions before starting train data
  conversion.
- Active Taiji task:
  `meteor_mechet_natural_language_event_sft_1ep_8a100_qy_20260913_03`, instance
  `8b1d8228a08afcb601a096e8cdb913e6`, normal non-elastic `8 x A100` in Qingyuan.
  Full conversion and distributed tokenization are complete; eight training
  ranks are running the one-epoch Qwen3-8B QLoRA SFT. The `_02` instance ended
  before optimization because duplicating both the 20.4-GB token cache and the
  model cache exhausted node-local temporary storage; `_03` keeps only tokens
  local and loads the pinned model from the shared read-only cache.

This is a candidate representation experiment. It does not supersede the
paper's current A7 choice until its frozen validation evaluation is available.

### Frozen checkpoint-14000 local diagnostic

- Frozen adapter SHA256:
  `3adfa321f1cc06c257dd96cc5dde1bd2e7b0c9608dc7cd11b54b3749e5b79f10`.
- Fixed seed-17 stratified validation sample: 256 reactions, 1,979 decisions
  (704 import, 1,019 electron event, 256 finish).
- Completed normal non-elastic task:
  `meteor_mechet_nl_event_valid256_ckpt14000_k1_8v100_cq_20260913_01`, instance
  `8b1d81f5a08afd1e01a09aa1993619a8`, `8 x V100` in Chongqing.
- This is greedy K=1 at authoritative current states. It reports tool choice,
  import matching, electron source/destination, strict execution, and mapped
  plus map-invariant chemical successor accuracy. It is explicitly not a
  product-only closed-loop or test-set endpoint result.
- All 1,979 planned decisions completed with zero missing or extra rows. Tool
  selection is 99.34%, argument compilation 98.23%, electron-event formal
  execution 96.57%, strict event match 68.01%, and map-invariant chemical
  successor match 73.31%. Reactive-fragment exact match is 41.21%, versus
  8.02% for endpoint-only context. Event exact match is 86.8%, 71.2%, 31.0%
  and 0/23 for one through four coupled flows; six four-flow predictions reach
  the same chemical successor via a shorter representation. These are fixed
  local diagnostics, not autonomous endpoint results.
- Scientific framing and evidence boundaries are consolidated in
  `docs/EXECUTABLE_INVERSE_DYNAMICS.md`.

## What is already on `main`

- PR #35 (`b7e26027`) merged the full-data protocol/runtime update.
- PR #40 (`4b410e52`) merged the code/protocol repair for Issue #36:
  matched mech-USPTO training/inference prompt + tool schema + 12-call budget,
  one generation worker per 40-GiB A100, explicit failed/OOM candidate accounting,
  NLL/first-candidate metrics, and frozen lineage hashes.

The old protocol-mismatched ~1.60% mech-USPTO action-delta result remains
**diagnostic only**.

## Active ICLR issue

### #36 — matched mech-USPTO action-delta rerun (closed)

Implementation/protocol repair is complete on `main` via PR #40.
Issue #36 was closed on 2026-08-26. Do not create, retry or expand experiments
under #36. The one already-running K=10 + NLL task may finish and be archived;
any newly discovered defect must be scoped as a separate issue.

### #37 — compact-full-state A7 main run

This is the only active A7 implementation/training issue.

Current decision:

- `compact_full_state`: **main A7 candidate**;
- `action_delta`: ablation / fallback;
- legacy `full_state`: historical reference;
- reaction-site-marked SMILES: outside the current ICLR run.

The paper method is unchanged: after each accepted action, the next decision
must receive the actual executor-produced current chemical state. The compact
format removes duplicate serialization, not state information.

Fast execution gate:

1. implement one `compact_full_state` observation mode;
2. on a 1k–5k frozen slice verify identical stable IDs, byte-identical assistant
   actions, chemically identical replayed states/endpoints, no dropped rows and
   no truncation;
3. require a substantial input-token reduction (target <=50–60% of legacy
   full-state);
4. immediately submit Qwen3-8B A7 for **1 epoch**;
5. run the cheapest valid frozen validation evaluation after epoch 1;
6. if performance clearly recovers from action-delta and approaches the
   full-state reference, freeze compact-full-state as A7 main and launch the
   full K=10 + NLL-ranked test; continue epoch 2 only if validation still shows
   clear headroom.

Do not wait for three new matched full trainings before obtaining the A7 result.

Gate status on 2026-08-26:

- 2,048 frozen training rows rebuilt with zero quarantine;
- stable IDs, assistant actions, every authoritative state and the executed
  endpoint are identical to the matching legacy full-state rows;
- no model-visible proof/digest or duplicated before/after state fields;
- Qwen3-8B chat-template input tokens are 11,951,517 vs 23,721,228, or
  **50.38%** of legacy full-state;
- full strict-universe train/valid/test construction is in progress; submit
  exactly one resumable 1-epoch A7 job after the full artifact audit passes.

## Consolidated / deferred issues

- #38 is superseded by #37. Its useful content (lossless removal of redundant
  state/proof serialization) is now part of the compact-full-state A7 plan.
- #39 reaction-site-marked SMILES is deferred beyond the current ICLR paper and
  must not block A7.

## Priority order

1. implement + launch #37 compact-full-state A7;
2. allow the already-running #36 matched rerun artifact to finish; do not
   create a replacement;
3. external baselines / paper analyses continue independently;
4. do not reopen marked-SMILES representation work before the main A7 result.
