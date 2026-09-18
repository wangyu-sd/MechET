# MechET project memory: authoritative data and model lineage

> **2026-09-18 protocol-v2 full remote rebuild/training:** the running v1
> compact-history task was stopped because its action-conditioned observation
> and private-map replay contract are diagnostic only.  The first clean v2
> H20 task correctly failed closed during full conversion: valid was complete,
> while 15 train and 7 test reactions exposed a Kekule bond-order mismatch in
> fallback graph-delta extraction; no optimizer training started.  Commit
> `408e132` aligns that extraction with executor Kekule semantics, and all 22
> source rows replay exactly after the repair.  Replacement task
> `meteor_mechet_nl_protocol_v2_full_sft_2stage_8h20_zjk_20260918_02`, instance
> `8b1d8014a0af10a701a0b3501c720673`, is genuinely running on ordinary 8xH20
> with Ceph mounted and pinned runtime `408e132`.  It must rebuild and audit
> 257,167 / 2,890 / 28,967 reactions with zero unresolved rows before clean
> base-model State-SFT and then compact-history Trajectory-SFT can begin.

> **2026-09-18 natural-language protocol v2 implementation repair:** the
> mechanism-first scheme is unchanged, but v1 is diagnostic because its
> molecular inventory visibility was conditioned on the gold action type and
> its builder/runtime used different private-map assignment after imports.  v2
> uses one inventory-bearing prompt for import/event/finish, mirrors runtime
> remapping in the builder, advances only with executor outputs, allows repeated
> imports, and uses 40 decision / 32 import defaults.  The repaired full valid
> protocol achieved byte-exact prompt parity and executable gold replay for
> 2,890/2,890 reactions (22,341/22,341 decisions).  Primary endpoint scoring is
> structural exact; full-mixture exact is secondary.  See
> `docs/NATURAL_LANGUAGE_PROTOCOL_V2.md`.  Do not treat the running v1 history
> checkpoint or its dual-prompt smoke as the repaired v2 result.

> **2026-09-16 productive full anchor-branch run:** the endpoint-shaped smoke
> ended successfully but did not improve its fixed 16-reaction validation set:
> execution stayed 24/32 and exact endpoint stayed 0/32.  Inspection showed
> that 16/24 formally terminal wrong candidates retained the unchanged product
> and merely appended fragments; the sole exact train candidate was an H=1
> reset after 7/8 reference events, not a full episode.  Commit `f6bff25`
> therefore adds a distinct no-transform outcome/penalty, productive-execution
> reporting and complete continuation-action logging.  It does not relabel a
> formal terminal as chemically correct.  A full bounded post-training task was
> submitted as
> `meteor_mechet_nl_anchor_branch_rl_productive_full_8a100_qy_20260916_01`,
> instance `8b1d813ea0a4c8ab01a0aaabe7a30977`: 10 rounds x 256 distinct train
> reactions, K=8, 50% full episodes and a disjoint 256-reaction validation
> monitor.  This is 2,560 sampled reactions from the 257,167-row strict-
> executable universe, not a full-data epoch and not unqualified FlowER full.
> Test is never loaded.  Initial verification found an allocated ordinary
> Qingyuan 8xA100 Pod and mounted Ceph; it was still copying the pinned vLLM
> runtime with zero GPU memory, so rollout/model execution was not yet claimed.

> **2026-09-16 natural-language anchor-branch post-training:** the active
> lineage-correct parent is
> `outputs/agent/natural_language_event_sft_qwen3_8b_a100_seed17_20260913`
> (adapter weights SHA-256
> `16648e587e084c273c35faee0adcd2486fbdb4f71985d007648421ea5990f3fb`).
> PR #59 supersedes the Python-template parent in PR #58 for this run. One
> ordinary Qingyuan 8xA100 integration task was submitted:
> `meteor_mechet_nl_anchor_branch_rl_smoke_8a100_qy_20260916`, instance
> `8b1d813ea0a4c8ab01a0a9893c1f075d`. It uses 64 named train reactions,
> K=8 local first-tool branches, one actual policy-update round and 16 disjoint
> validation reactions; test is never loaded. Live verification found the Pod,
> Ceph mount, eight collector processes and all eight A100s at roughly 33.8 GiB
> with nonzero utilization. The 16-reaction baseline completed and the first
> K=8 training group was persisted with nonzero local advantages. This verifies
> real rollout collection, not yet a completed optimizer update. The CLI log
> endpoint still returned only launcher lines even though worker/heartbeat
> stdout was connected to PID 1's default pipe; use artifacts and Pod inspection
> together until Taiji exposes the worker stream.

> **Permanent denominator guard:** FlowER `3,080`, old mech-USPTO-31k `1,124`,
> and current-compiler mech-USPTO-31k `1,253` are incomplete
> replay-compatible test subsets. Never call any of them the
> full test set or use either as a headline benchmark denominator. Full test
> sizes are FlowER `28,971` and mech-USPTO-31k `3,120`.

> **Permanent mech-USPTO mapping guard:** HF `rxn_prod_min` has no atom maps.
> Never label a copy of it `product_mapped` and never feed that artifact to
> LocalRetro. The active full endpoint protocol uses HF step-0 `elem_reac_spe`
> as the complete initial mixture, selects the deterministic principal organic
> `rxn_prod_min` fragment, maps the pair once with RXNMapper 0.4.2 under
> Transformers 4.57.1, and then performs synchronized product-only canonical
> reindexing. Every external method shares this mapping. The build must fail
> rather than filter a mapping error.

> **Permanent mech-USPTO compiler-lineage guard:** the artifacts
> `data/forward_expert/mech_uspto_31k/`,
> `data/mech_uspto_31k_inverse_tool_sft/`, and
> `data/mech_uspto_31k_inverse_tool_sft_action_delta_v1/` descend from the
> 2026-08-11 compiler output. Their 9,118 / 1,187 / 1,124 rows are a deprecated
> pilot trace subset, not output from the current compiler and not the full
> 24,959 / 3,120 / 3,120 benchmark. "100% conversion" in their manifests means
> 11,429/11,429 already-stitched traces were converted; it never means all
> 31,199 raw reactions compiled. Before training, read
> `docs/MECH_USPTO_31K_ARTIFACT_REGISTRY.md` and the artifact's
> `ARTIFACT_STATUS.json`. A status with `training_allowed: false` is a hard stop.

> **Permanent A7 observation guard:** the frozen
> `flower_inverse_tool_sft_full_v4` rows and checkpoints trained from them use
> legacy full intermediate-state tool results. They are not the action-only A7
> main condition. The current main contract is `action_delta_v1`: no
> intermediate molecular-state SMILES or local-state SMILES are model-visible;
> the executor retains the complete private state and derives the endpoint via
> `finish_trace`. Rebuild and retrain before reporting an action-only A7 result.

> Last verified: 2026-08-24 14:35 (Asia/Shanghai). Read this file before
> describing any FlowER dataset as "full", submitting training, or comparing
> checkpoints.

## Current mech-USPTO compiler artifact

The current compiler was rerun from all frozen raw parquet rows on 2026-08-24.
It produced 74,170 / 9,234 / 9,478 executable elementary steps, then
16,207 / 2,042 / 2,027 reactions with every step executable, and finally
10,152 / 1,319 / 1,253 globally stitched traces. The action-delta inverse
conversion accepted all 12,724 stitched traces, independent replay validation
has zero failures, tokenizer audit has zero truncation, split overlap is zero,
and no intermediate states are model-visible.

The current train-ready path is
`data/mech_uspto_31k_inverse_tool_sft_action_delta_v2_compiler_20260824/` with
status `validated` and `training_allowed: true`. Its train/valid/test counts are
10,152 / 1,319 / 1,253. These are still executable trace-view counts, not the
full endpoint denominators 24,959 / 3,120 / 3,120. The old 9,118 / 1,187 /
1,124 artifacts are permanently labelled `deprecated_pilot` and forbidden for
new training. Compiler source hashes and stage counts are frozen in
`data/forward_expert/mech_uspto_31k_recompiled_20260824/COMPILER_LINEAGE.json`.

## 2026-08-24 pragmatic main-experiment policy

The earlier decision to stop A0/A2/A3/independent-proof inference as
"screening" was explicitly reversed by the user on 2026-08-24 at 19:43 CST.
Their dataset difference is limited to the four train and four test corrupt
FlowER endpoint rows, so run them as normal full-test evaluations and compare
on the shared 28,967 test IDs. Preserve both generation-order Pass@K and
gold-independent assistant mean-NLL ranked Top@K; proof/trace ranking uses a
formal-execution gate before mean NLL. A7 action-delta remains the main method
on 257,167 / 2,890 / 28,967 and receives its unified K=10 evaluation after
training. Exclude only the named corrupt upstream endpoint rows. Match
backbone, epoch count, decoder and evaluator, but do not block experiments on
exact supervised-token equality. The current-compiler mech-USPTO 10,152 /
1,319 / 1,253 run is a secondary program-view transfer experiment, not the
full 3,120-test endpoint benchmark.

### 2026-08-24 A1--A6 submission clarification

The working program-ID dataset is frozen. Do not rerun leakage, coverage,
decontamination, ID-intersection, or screening gates before these jobs. A2 and
A3 already have completed three-epoch checkpoints. A1, A4, A5, and A6 were
constructed only by changing the representation of the same 257,167 / 2,890 /
28,967 rows; no row was skipped. At 22:34 CST all four ordinary Qingyuan
8xA100 tasks were created, started, and verified in the resource-waiting queue:

- `mechet_flower_a1_free_cot_qwen3_8b_3ep_8a100_20260824_01`;
- `mechet_flower_a4_open_flow_qwen3_8b_3ep_8a100_20260824_01`;
- `mechet_flower_a5_loose_trace_answer_qwen3_8b_3ep_8a100_20260824_01`;
- `mechet_flower_a6_mechsmiles_qwen3_8b_3ep_8a100_20260824_01`.

Their shared manifest is `data/iclr_program_controls_v1/manifest.json`. They
use Qwen3-8B QLoRA for three epochs with no max-step cap. Queue admission is
not training success; confirm a real POD and logs after resource allocation.

The current-compiler mech-USPTO three-epoch Qwen3-8B training task is
`mechet_uspto31k_action_delta_v2_qwen3_8b_3ep_8a100_20260824_01` on ordinary
Qingyuan 8xA100. It uses 10,152 / 1,319 / 1,253 and must never be evaluated or
described as the full 3,120-test endpoint benchmark.

The external-baseline FlowER handoff is frozen at
`data/external_baselines/flower_full/`: 257,171 / 2,890 / 28,971 rows, zero
split-ID overlap, no executor/replay filtering, with mapped and unmapped
product/precursor/reaction fields. Its manifest is
`data/external_baselines/flower_full/manifest.json`. All published external
baseline repositories must preserve `stable_id` and derive only their native
labels from this handoff. It is independently generated with
`python scripts/export_full_baseline_pairs.py --datasets flower_full`; it does
not wait for the mech-USPTO full export.

The mech-USPTO external-baseline replacement source is the frozen public HF
snapshot already stored in `data/raw/mech_uspto_31k/data/`. The full 31,199-row
RXNMapper build uses `scripts/build_mech_uspto31k_rxnmapper_baseline.py`; a
30-row train/valid/test smoke passed without exclusions. Full task
`mechet_uspto31k_full_rxnmapper_build_1a100_20260824_01` was submitted on one
Qingyuan A100. Its target paths are
`data/mech_uspto_31k_full_endpoint_rxnmapper/` and
`data/baselines/localretro_mech_uspto_31k_rxnmapper/`. Do not use the invalid
legacy `data/mech_uspto_31k_full_endpoint_sft/` pseudomap artifact.

## Paper experiment authority

The ICLR experiment source of truth is the private `wangyu-sd/MechET-paper`
repository, with `EXPERIMENT_MATRIX.md` controlling matched conditions and run
priority. The implementation snapshot is
`configs/experiments/paper_experiment_matrix_v1.yaml` at paper revision
`47b0166174f0ab921c01d11f83cbf66db8659177`. Use paper IDs A0--A7, B1--B5 and
R1--R7 in new configs and reports. Textbook/RAG/H3 is outside the current ICLR
protocol; RL and model-size scaling are optional only after matched SFT, R3 and
C2 are frozen.

The current A0/A2/A3/A7 jobs are P0 screens. They are not yet headline-matched:
A0/A2 use 257,171 train reactions, while A3 and executable action-only A7 use
257,167 strict-proof reactions; supervised assistant-token budgets also differ
by representation. Final Tier-A training must use one common eligible-ID
universe and a declared matched supervised-token/update budget, as required by
the paper matrix.

The A7 observation-granularity protocol is frozen in
`configs/experiments/observation_granularity_v1.yaml`. `action_delta` is the
main/default builder mode; `reaction_center_delta` and `full_state` are matched
representation ablations. The three conditions must share stable IDs, action
targets, executor, model revision and candidate budget. Report both natural
length/throughput and a length-matched redaction control before attributing a
gain to state information.

## Non-negotiable FlowER terminology

`FlowER full` means the complete reaction-level official split, with no
mechanism compiler, replay, overlap, length, or deduplication filtering:

| split | rows |
|---|---:|
| train | 257,171 |
| valid | 2,890 |
| test | 28,971 |

The frozen artifact is `data/flower_full_endpoint_sft/`; its manifest declares
100% reaction-level coverage. Never call 32,050, 28,076, 27,640, 333, or 3,080
the FlowER full split.

The raw trajectory files contain 1,933,750 / 21,723 / 218,997 state-transition
lines. Those are transition rows, not reaction denominators.

## Derived subsets are separate experimental conditions

| artifact | train | valid | test | meaning |
|---|---:|---:|---:|---|
| `flower_full_endpoint_sft` | 257,171 | 2,890 | 28,971 | full official reaction-level endpoint data |
| `flower_full_endpoint_sft_decontaminated` | 237,440 | 2,675 | 28,971 | endpoint data after train/heldout exact-reaction removal; test preserved |
| `mechet_proof_sft` | 32,050 rows (32,047 unique IDs) | 374 | 3,562 | legacy MECH_PROOF-compilable subset |
| `textbook_tool_sft` | 28,076 | 333 | 3,080 | executable proof-to-trace subset |
| `flower_inverse_tool_sft` | 27,640 | 333 | 3,080 | decontaminated and A100-length-filtered trace subset |
| `flower_inverse_tool_sft_full_v4` | 257,171 | 2,890 | 28,971 | unfiltered reaction coverage: 289,024 strict executable traces plus 8 explicitly labelled corrupt-upstream endpoint fallbacks |
| `flower_inverse_tool_sft_action_delta_v1` | 257,167 | 2,890 | 28,967 | A7 `action_delta_v1` strict executable universe; zero conversion failures and zero model-visible intermediate-state leaks; explicitly excludes the named 4 / 0 / 4 corrupt upstream endpoints |
| `iclr_full_v4/outcome_only` | 257,171 | 2,890 | 28,971 | full reaction-level product-only direct baseline |
| `iclr_full_v4/state_cot` | 257,171 | 2,890 | 28,971 | full reaction-level textual MECH_ET State-CoT baseline |
| `iclr_full_v4/net_edit` | 257,167 | 2,890 | 28,967 | single-step net-edit baseline over every strict-proof row |
| `iclr_full_v4/proof` | 257,167 | 2,890 | 28,967 | independent complete-proof baseline over every strict-proof row |

The legacy `data/mechet_proof_sft/manifest.json` still records 226,183 train
rows skipped with `ProofProgramError`. The repaired v4 lineage is separate:
`data/mechet_proof_sft_flower_full_v4/` contains 257,167 / 2,890 / 28,967
strict executable proofs. The remaining 4 / 0 / 4 frozen FlowER rows (`RC`,
`PC`, `PM`, `RS` in train and test) have upstream endpoints that are not
atom-conserving. They are retained, never filtered, as explicitly labelled
direct endpoint rows in `data/flower_inverse_tool_sft_full_v4/`. Consequently
the v4 artifact has 100% reaction-row coverage but 289,024/289,032 strict
proof coverage; do not call the eight fallback rows executable mechanisms.

## Models already trained

| model/output | actual train input | rows | training |
|---|---|---:|---|
| `outputs/agent/tool_sft_mixed_inverse_qwen3_8b` | mixed FlowER + mech-USPTO-31K trace, decontaminated | 36,756 (27,640 + 9,116) | 1 epoch, 1,149 steps |
| `outputs/agent/tool_sft_flower_inverse_qwen3_8b_run_20260812` | `flower_inverse_tool_sft/train.jsonl` | 27,640 | 3 epochs, 2,592 steps |
| `outputs/agent/tool_sft_flower_inverse_qwen3_8b_h20_run_20260812` | same trace subset | 27,640 | 3 epochs, 2,592 steps |
| `outputs/agent/sft_flower_full_endpoint_qwen3_8b_h20_run_20260814` | decontaminated full-endpoint condition | 237,440 | 3 epochs, 11,130 steps |
| `outputs/iclr/net_edit_seed11` | legacy proof subset | 32,047 | 1,000 steps |
| `outputs/iclr/proof_sft_seed11` | legacy proof subset | 32,047 | 1,000 steps |
| `outputs/iclr/replayable_*_seed11` | matched replayable trace subset | 27,640 | 1,000-step jobs submitted 2026-08-19 |

The 237,440-row endpoint model is broad reaction-level pretraining, but it is
not an executable-mechanism model and it is not the unfiltered 257,171-row
official train split. The 27,640-row models are trace-owned mechanism models,
but they are not FlowER-full models.

The full-reaction v4 Qwen3-8B QLoRA job was submitted on 2026-08-19 as
`mechet_flower_full_trace_qwen3_8b_3ep_8v100_20260819_01` (instance
`8b1d8992a0144fe501a01972691b0b11`). It uses all 257,171 train rows for three
epochs with no max-step cap. At submission time it was `PENDING` because the
only visible V100 pool (`AI4LSLab_KEMO_V100`, Shenzhen) reported zero available
V100 hosts; do not describe it as training until instance state and POD logs
confirm execution.

The additional H20 replica
`mechet_flower_full_trace_qwen3_8b_3ep_8h20_20260819_02` was cancelled at the
user's request while waiting for resources. It was replaced by the 8xA100 task
`mechet_flower_full_trace_qwen3_8b_3ep_8a100_20260819_01` (instance
`8b1d80e1a0144ffc01a01995132f0b52`). The A100 task uses the same 257,171 rows,
seed, three-epoch schedule, and shared frozen token cache, with independent
output `outputs/agent/tool_sft_flower_full_qwen3_8b_a100_run_20260819`.

On 2026-08-20 the four representation baselines were rebuilt with
task-specific product-only prompts under `data/iclr_full_v4/`. No ID
intersection, decontamination, token-length filtering, or max-step cap is used.
Outcome-only and State-CoT retain the complete reaction split. Net-edit and
complete-proof contain every available strict proof; their 4 / 0 / 4 row
difference is solely the frozen upstream-corrupt set described above. Four
3-epoch Qwen3-8B QLoRA tasks were submitted to the ordinary Qingyuan A100
group with flags beginning `mechet_iclr_full_{outcome,state_cot,net_edit,proof}`.

The original A2 State-CoT task ended during pretokenization because the
answer-bearing rows inherited a trace-runtime fallback marker. The converter
now preserves that fact only as `source_upstream_endpoint_fallback`; all
257,171 / 2,890 / 28,971 A2 rows validate as a no-tool representation
baseline. The fixed replacement task
`mechet_iclr_full_state_cot_qwen3_8b_3ep_8a100_20260820_02` (instance
`8b1d8079a0144ff501a01e5bff7b1581`) entered `TRAINING_RUNNING` on 8xA100 and
started distributed pretokenization on 2026-08-20.

## 2026-08-23 action-only and evaluation runs

The action-only A7 artifact was rebuilt and frozen at
`data/flower_inverse_tool_sft_action_delta_v1/training_manifest.json` with
257,167 / 2,890 / 28,967 unique IDs, zero split overlap, zero quarantines, and
zero intermediate-state leaks. It is complete over the strict executable
universe but is not the unfiltered 257,171 / 2,890 / 28,971 reaction split.

At 00:17 CST on 2026-08-24, the following ordinary (non-elastic) Taiji work was
active or queued after checking all 31 visible application groups:

- mech-USPTO replayable action-only training
  `mechet_uspto31k_action_delta_qwen3_8b_3ep_8a100_20260823_02` was genuinely
  training on 8x A100-40GB; checkpoint 300 / 429 existed, epoch was 2.10, and
  all GPUs had previously been verified active. Its 9,118 / 1,187 / 1,124 data is a program
  subset, never the 24,959 / 3,120 / 3,120 full endpoint benchmark.
- A0 and the clean A2 full FlowER K=10 screens were resource-waiting in the
  ordinary Zhangjiakou H20 group. The clean complete-proof and EOS-fixed A3
  screens were resource-waiting in the ordinary Qingyuan A100 group.
- Action-only A7 training used independent A100 and H20 resource candidates,
  separate outputs and separate token caches. The A100 candidate
  `mechet_flower_action_delta_qwen3_8b_3ep_8a100_20260823_01` received a real
  8x A100-40GB POD at 00:19 CST; the still queued H20 duplicate was stopped
  immediately. At 00:29 the frozen cache reported 257,167 train rows/windows,
  zero windowed rows, zero truncation and max input length 8,784; all eight
  A100s were then at 100% utilization with 13.7--15.4 GiB allocated. A7 real
  gradient training is therefore active.
- The ordinary Qingyuan A100 quota was 112 / 39 using / 24 waiting; the
  ordinary Zhangjiakou H20 quota was 176 / 152 using / 28 waiting before the
  A7 H20 candidate was added. Global Shenzhen V100 still had zero available
  hosts. `TaiJi_HYAide_hy_exp_SH_A100H` displayed unused A100PRO quota but is
  a Hunyuan-platform application group: the normal non-elastic task API
  rejected the required scientific metadata, so it is not a usable ordinary
  fallback. AILab A100 candidates had already ended without a usable POD/log.

The first A3/proof inference attempts incorrectly loaded three BF16 Qwen3-8B
copies on each 40GB A100 and OOMed. A subsequent one-copy A3 diagnostic exposed
a second throughput/artifact bug: candidates that finished early were decoded
with thousands of batch-alignment `<|im_end|>` pad tokens whenever a sibling
candidate ran to the 4,096-token limit. That partial run was stopped and is not
a result. The launcher now trims only tokens after the first EOS, uses two model
workers per GPU, caps the 40GB candidate microbatch at two, and freezes decoding
limits from validation maxima with headroom (A0 1,024; A2 14,336; A3/proof
2,048). OOM candidate microbatches are still bisected. The associated inference
tests pass. Full-coverage A0/A2/A3/proof evaluations remain P0 screens rather
than matched Tier-A evidence.

After mech-USPTO training released its A100s, the clean complete-proof task
`mechet_iclr_full_proof_infer_k10_batched_8a100_20260823_03` entered real
`TRAINING_RUNNING` at 00:33 CST. Sixteen generation processes (two per GPU)
loaded successfully; observed memory was approximately 29--33 GiB per 40-GiB
A100 with no OOM/traceback in shard logs. By 00:40 it had written five complete
K=10 rows (50 candidates): batch size was two, generation errors and repeated
EOS padding were both zero. Four candidates legitimately reached the frozen
token cap without EOS. This validates the repaired runtime path but is far too
early for an accuracy or steady-state throughput claim.

The earlier Figshare-dependent full endpoint plan was replaced on 2026-08-24
because the current network receives HTTP 403 from Figshare. The active source
is the public HF snapshot with a single shared pinned RXNMapper remap. This is
explicitly a reconstructed HF endpoint protocol, not a claim that the original
Figshare numeric atom-map labels were recovered.

The action-only mech-USPTO training task finished successfully at 00:32 CST
with final adapter hash
`86592069fd603078e4d8023743ee3bbc65e86ded4b828c27d144a38e62c40c97`,
9,118 rows/windows, three epochs and zero truncation. Its train SHA matches the
frozen manifest. The K=10 rollout launcher is at
`scripts/run_taiji_mech_uspto31k_action_delta_infer_k10.sh`, with its ordinary
A100 task config at
`configs/taiji/mechet_uspto31k_action_delta_qwen3_8b_infer_k10_8a100_20260824.json`.
The finalized output passed the adapter/data/revision gate, so task
`mechet_uspto31k_action_delta_qwen3_8b_infer_k10_8a100_20260824_01` was
submitted to the ordinary Qingyuan A100 queue. The runner verifies the pinned
train SHA again inside the POD and labels 1,124/3,120 as a program-view subset.

After the 2026-08-24 inference/EOS changes, the complete local test suite passes:
255 tests, with `git diff --check` clean.

## Submission rule

Before launching a FlowER job, record all of the following in its config and
data contract: exact input path, row count, SHA-256, whether proof/replay
filtering occurred, whether overlap filtering occurred, whether token-length
filtering occurred, and whether the condition is endpoint-only or executable
trace. If the requested condition is "FlowER full", the launcher must require
257,171 / 2,890 / 28,971 and must reject any other denominator.
