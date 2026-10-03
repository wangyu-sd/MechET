# Issue #79: NMI compositional-generalization experiment record

## Step 0 audit (2026-10-02, before experiment-code changes)

| Item | Code / submission / completion / scientific evidence |
|---|---|
| PRs #70–#78 | All merged on `main` (`a4f01fc`); code merge alone is not an experimental result. |
| Protocol-v2 parity | `docs/NATURAL_LANGUAGE_PROTOCOL_V2.md` records 2,890/2,890 validation gold replays and 22,341/22,341 byte-exact decision prompts; the machine-readable audit is `docs/audits/natural_language_protocol_v2_valid_20260918.json`. |
| State-SFT / Trajectory-SFT | Both full strict-universe adapter manifests exist under `outputs/agent/natural_language_event_v2_qwen3_8b_h20_seed17_20260918` and `outputs/agent/natural_language_event_history_v2_qwen3_8b_h20_seed17_20260918`; train SHA-256 values are `22729aff36db3691b3db544aa83cb93c7383e9fbacda6934e6a52a3d690f6bfa` and `bc0a859a3c25b6146a8f6cac730610a0d66347baff1f9648b67329b17d785b97`. These full-data checkpoints cannot serve as headline H2 controls after a new split. The PR #70 matched-v2 valid256 task flag has no Taiji instance, and no corresponding completed local report was found; code merge is not a validation result. A completed **State-SFT-only** H=1/2/3/full suffix diagnostic is now documented below; the requested matched State-SFT **versus Trajectory-SFT** horizon comparison remains unverified. |
| State-SFT v2 local diagnostic | The completed fixed 256-reaction validation report at `outputs/eval/natural_language_event_v2_ckpt14000_finegrained_20260919/local_valid256/evaluation.json` records 1,979/1,979 decisions. Among 1,019 electron-event decisions, 961 (94.3%) formally execute and 785 (77.0%) reach a map-invariant chemically matching successor; only 176/704 (25.0%) fragment-import decisions match the reference. Its frozen adapter-model SHA-256 is `2cd25b892b812eef0d93b99c29dd96bf73bc61098f6f54077fbb6d5439c26a09`. This is an oracle-current-state **single-decision** validation diagnostic, not autonomous endpoint recovery. |
| State-SFT v2 suffix-horizon diagnostic | The same adapter was evaluated on a fixed 32-reaction validation cohort at `outputs/eval/natural_language_event_v2_ckpt14000_finegrained_20260919/suffix_valid32/suffix_evaluation.json` (report SHA-256 `275f9252415a74488e1a254a40e2bcfb9242f7288a8847628198c47910de`). All 128 planned H=1/2/3/full episodes are present. The evaluator supplies a trusted reference prefix and oracle-scheduled exogenous fragments, then gives **no reference state feedback after branching**; K=1 exact suffix endpoints are 28/32, 21/32, 18/32, and 13/32. These are suffix diagnostics, not product-only autonomous endpoint accuracy, a state-feedback intervention, a matched Trajectory-SFT comparison, or H2 evidence. |
| Trajectory-SFT v2 development smoke | `outputs/eval/nl_history_v2_ckpt27000_stratified_valid26_20260922/evaluation.json` records checkpoint 27,000 and adapter SHA-256 `093f12378db17fac898b17d331feb1f1a5ea00768c548cd778c1df820679ac14`: product-only K=1 closed-loop rollout reached a structural endpoint on 6/26, while its separate gold-state event diagnostic matched 71/92 chemical successors. The raw prediction files and `PROVENANCE.md` remain, but the checkpoint-27,000 weights have since been pruned; this is a recorded historical smoke, not a presently reproducible adapter. The selected 26 reactions are development-only; this is not a matched comparison against the 256-reaction State-SFT sample or a held-out H2 result. |
| Historical State-vs-Trajectory smoke | `outputs/eval/natural_language_history_ckpt10000_valid32_20260918/evaluation.json` compares the same 32 development reactions under protocol v1: both State-SFT and Trajectory-SFT had 0/32 exact endpoints, while formal terminals were 16/32 and 21/32. This pre-v2 diagnostic is not the requested H=1/2/3/full experiment and cannot be pooled with the repaired-v2 reports. |
| State-feedback intervention | The paper-matrix `B3: stale_feedback` and baseline-assignment notes specify a planned control, but no frozen stale/no-feedback task config, adapter or evaluation report was found in the accessible `configs/taiji`, `outputs/eval`, `outputs/agent` or `outputs/iclr` directories. The suffix diagnostic's **absence of reference-state feedback** is different: it still returns the executor's actual successor state to the model after each predicted event. It cannot be relabelled as a stale/no-feedback intervention or used to claim its effect. |
| EARHO historical outcomes | The accessible 128-reaction mech-USPTO-31k component smoke reports record baseline `group_pass_at_k=0/128` and round-one actor-only `20/128` versus actor+critic `12/128` (two candidates per reaction). Their adjacent `HISTORICAL_STATUS.json` files explicitly mark them `legacy_pre_v3` and forbid reporting them as the later prefix-v3 condition. They are small, differently sourced diagnostics, not frozen H2 evidence or a valid matched FlowER frontier-recovery result. |
| EARHO FlowER prefix-v3 | A separate completed five-round FlowER strict-executable prefix-v3 run exists at `outputs/agent/earho_paper_flower_strict_prefixv3_portfix_seed17_a100/`. Its fixed 128-reaction validation monitor has SHA-256 `83063564444a7a9401109f79ce374c1773c8fc504ee2f9d0b45acbe0526ef410`; K=1 full-episode endpoint hits were 10/128 for the initial Trajectory-SFT adapter and 10, 5, 5, 6, 7/128 after rounds 0–4. All six collections report 128/128 groups and zero collector errors. `completed.json` selects the unchanged SFT adapter as best. The plan records no test use. This is a valid **small validation diagnostic of that post-training run**, not an H2 split result, a matched closed-vs-open comparison, or an H=full horizon ablation. |
| Pointer | 16 product-start cases completed on Taiji; baseline and coupled pointer each have Top-1=0/16 and Pass@beam=0/16. Predeclared smoke gate stopped 256 and 1,319 stages. The 44 invalid handles count rejected actor proposals, not executed invalid actions. |
| Runtime | Five-mode oracle-state benchmark is not complete. The PR #78 recovery reached Taiji twice but its ledger ended `INFRA_FAILED`. POD traceback identifies a **user-program compatibility error**, `Unsupported option for the xgrammar backend: no-fallback`, in pinned vLLM 0.8.5 V1 engine initialization. The controller's `INFRA_FAILED` label is not a scientific result or proof of platform failure. This runtime arm is not necessary for E1/E2 CPU audits and must not be silently retried with changed benchmark gates. |
| Packing | Completed report: loss absolute difference 0.00146447, throughput speedup 0.86350, peak-memory ratio 1.28073. The speed gate failed: `SCIENTIFIC_STOP`, no further packing optimization for #79. |
| MechComp C1/C2/C3 and H2 | Split builder, overlap audit and H2 runner exist, but `data/ood/mechcomp_source_sink` and `outputs/h2` were absent at this audit. No valid matched split/checkpoint/result is established. |
| PMechDB / PMechRP | No authorized source assets were found in the workspace. E5 is access-blocked, not a reason to stall E1–E4; no mirror substitution or new download is authorized. |

The three cited EARHO smoke reports are under
`outputs/eval/earho_v2_stageii_aligned_prefix_smoke_20260923/`,
`outputs/eval/earho_v2_component_smoke_round01_actor_no_critic_20260923/`,
and `outputs/eval/earho_v2_component_smoke_round01_beam4_20260923/`.
Their separate prompt-prefix/actor/critic conditions preclude interpreting
the raw 0/20/12 counts as a clean causal comparison.
For the FlowER prefix-v3 run, `plan.json` pins the initial adapter, model
revision, validation-monitor ID file and K=1 metric; `completed.json` and the
baseline/round `collection_done.json` files prove completion and the reported
counts. Its best-checkpoint selection by the same validation monitor is a
historical within-run choice, not permission to select any #79 H2 checkpoint
using the held-out composition test.

The E1/E2 source candidate is the frozen strict-executable FlowER **training pool** `data/flower_inverse_tool_sft_action_delta_v1/train.jsonl`, 257,167 reaction rows with SHA-256 `edc80c5c5eb13d50753c5566c5d6ac1b90b955b1a8dfece915241b2da4a40a75` in its training manifest. This is not the unqualified FlowER-full official split and does not make an old full-data-trained adapter eligible for the new held-out condition. The older documented `data/knowledge_ablation/v2/train/trace_no_knowledge.jsonl` input is absent locally. No final held-out model test will be inspected for model selection.

## E1 — local-operator basis audit

The full-source streaming audit is at
`outputs/issue79/nmi_operator_basis_train_20261002.json` (large artifacts stay
outside Git). It verified the 257,167-row source SHA-256 above and independently
recomputed the first 32 stored signatures. There are **604 distinct local
execution primitives** and **6,553 distinct ordered move-composition
signatures**, across 1,043,352 execution steps and 2,111,100 electron moves.
Of the primitives, 53 occur in one reaction, while 435 occur in at least ten;
2,728 composition classes occur in one reaction. At random hash samples of
1%, 10%, 50%, and 100% of reactions, primitive vocabulary size was 321, 458,
562, and 604; composition vocabulary size was 640, 2,231, 4,836, and 6,553.

A **non-frozen exploratory** 10% reaction hash probe held out 25,738 reactions:
only 317 had an unseen complete move composition, and 316 of these used only
seen primitives. This is 316/25,738 = **1.23%** of the random held-out
reactions. The remaining 1/25,738 = **0.0039%** required at least one unseen
primitive; these two strata exhaust the 317 unseen-composition probe cases.
Thus a random split would mostly retest familiar programs; E2 must hold out composition
groups. The v1 composition digest does not separately normalize the schedule
of fragment imports; it must be described as an ordered **move composition**,
not a unique physical mechanism or reagent program.

## E2 — candidate composition-group split

`outputs/issue79/nmi_mechcomp_split_20261002/manifest.json` records seed 42,
test/valid target fractions 0.10/0.10 and minimum train primitive frequency 5.
Rows with the same v1 move-composition signature or the same canonical unmapped
product + structural-precursor reaction are joined into one component before
selection. The resulting **candidate** ID sets contain train 223,863,
validation 6,200, and held-out test 27,104 reactions; SHA-256 of their ordered
ID files is respectively `a9cd229116809179be7a5e3708e52a3a0f115e4ea8fb34418d44f8d5608eddde`,
`53f8acec4a41929a702741f7f25babe90ac814c2002e0980173a843d30df8747`,
and `ec76eb0e972ab449e1d86d9e76a692682e561cbc2202217c73db55481879fc6d`.
There is zero train/test composition overlap and zero exact structural reaction
overlap; all held-out local primitives remain in train at frequency ≥5.

The largest composition/reaction connected component has 208,178 reactions;
only 6,200 validation reactions could be selected from the remaining feasible
components. This is a property of the grouping constraint, **not** the official
FlowER validation size and not a reason to silently change the split. The
6,200 held-out validation reactions all have three recorded execution steps;
the held-out test spans one to fifteen steps. Thus validation is a narrow
monitor, not a representative proxy for H2 test performance. The matched SFT
configs use the fixed terminal update count rather than selecting a best
checkpoint on this skewed validation set. The
revised local-center structural audit has completed. A separate
negative-control scan of the official 2,890-row strict
validation set found **zero** reactions containing primitives unseen in the
candidate H2 train split. A primitive-unseen control therefore cannot be
formed from that set without changing the training split; it is marked
unavailable rather than relabeling familiar-primitive reactions.
The separately frozen official-validation support audit at
`outputs/issue79/nmi_official_valid_program_support_20261003/manifest.json`
verified the source and H2 train-ID SHA contracts. Of 2,890 official
validation reactions, 2,442 have a program composition seen in H2 train and
448 have an unseen composition using only seen primitives; none requires a
train-unseen primitive. These are **not** H2 headline test rows or a substitute
for the unavailable primitive-unseen negative control. They establish that a
program-seen diagnostic stratum exists without opening the final H2 test for
checkpoint selection.

The first structural smoke revealed an inherited audit implementation defect:
RDKit's default SMILES parser removes mapped explicit hydrogen, and the old
center extractor ignores `BE_DELTA` bond/charge atoms. The #79 audit uses a
separately versioned `step_state_plus_edge_imports_v3_explicit_h` center while
retaining the old v2 behavior for historical results. On the fixed 1,000-row
diagnostic prefix, previously undefined center keys fell from 92/113 held-out
rows to 0/113. The first complete audit (`nmi_structural_full_20261002.json`)
verified zero exact-reaction overlap and zero undefined centers. On 27,104
candidate test reactions, 17,704 (65.3%) share a Murcko scaffold with train
and 1,328 (4.90%) have a product Morgan/Tanimoto ≥0.90 train neighbor. On
6,200 validation reactions, the corresponding counts are 3,328 (53.7%) and
23 (0.37%). These are overlap diagnostics, **not model outcomes**.

That first audit also reported zero `reaction_center_seen` in both held-out
strata. This key includes the *whole ordered move-bearing context* and is
nearly tautological under a complete-composition-disjoint split. It must not
be used as an independent structural control. The corrected v2 audit adds
per-step, map-free radius-one local structural center contexts **without move
labels or step order**, measuring the fraction of a held-out reaction's local
contexts seen in training. The v2 full audit
(`outputs/issue79/nmi_structural_full_v2_20261002.json`, SHA-256
`1e75ceb8b766280212cd32e46c7f112275fa4cdc632dd955c4711d508efc1192`)
passed both zero-exact-reaction gates and has zero undefined centers. On the
27,104 H2 test reactions, 24,213 (89.3%) have at least one local structural
center seen in train; 9,994 (36.9%) have all such centers seen. On H2 valid,
5,573/6,200 (89.9%) have at least one; none has all. The old ordered-center
key remains zero by construction and is never used as a structural control.

## E3–E5 results

E3 matched-data preparation found that filtering the historical reaction-level
Direct dataset by the H2 IDs is **not** a valid matched control. On the
223,863-row H2 train split, 36,119 mapped products and 175,403 structural
precursors disagreed byte-for-byte with the strict-trace Open-Flow records;
examples include explicit mapped hydrogen and different precursor-world
fragments. That Direct materialization is quarantined at
`data/issue79/nmi_matched_v1/direct_old_endpoint_mismatch_QUARANTINED_20261003`
and is not eligible for H2 training or reporting. The corrected Direct
materializer makes an outcome-only target from each **same frozen strict-trace
row**, removing tool and proof supervision from the model-visible conversation.
The exact product/structural-endpoint parity gate has now passed on **all**
223,863 train, 6,200 validation, and 27,104 held-out test IDs across all
three conditions. The frozen verification is
`data/issue79/nmi_matched_v1/verification.json` (SHA-256
`6da760b689b9d054e61944455b7cdf981301e9d3f9ab07722dd48a7e69550da1`).
The three equal-update Qwen3-8B configs are frozen at
`data/issue79/nmi_matched_v1/configs/`: seed 17, global batch 64, 10,494
optimizer updates each. This does **not** equalize token/computation budgets;
token counts and wall time must be reported separately. Three-row training
schema checks passed for each representation.
As a preflight of the post-generation Open-Flow scorer, the first 200 frozen
H2 **validation** gold programs replayed 200/200 and reproduced the structural
endpoint 200/200 using the frozen 40-call execution budget. This checks the
scorer path, not model accuracy, and does not inspect the H2 test labels for
model selection.

Three ordinary, non-elastic Taiji SFT tasks were submitted on 2026-10-03:

| Condition | Task / instance | Requested resource | First observed state |
|---|---|---|---|
| Direct | `meteor_mechet_nmi_h2_direct_8a100_qy_20261003_01` / `8b1d8047a0d27a4401a0fd745edb4230` | 8×A100, Qingyuan | `TRAINING_RESOURCE_WAITING` |
| Open-Flow | `meteor_mechet_nmi_h2_open_flow_8a100_qy_20261003_01` / `8b1d80eea0d297ec01a0fd748a9c41b5` | 8×A100, Qingyuan | `TRAINING_RESOURCE_WAITING` |
| Closed-Loop | `meteor_mechet_nmi_h2_closed_loop_8h20_zjk_20261003_01` / `8b1d89f7a0d297d901a0fd74a67941de` | 8×H20, Zhangjiakou | `TRAINING_INIT` |

Submission/start acknowledgement is **not** a training result. Resource
allocation, Ceph mount, terminal heartbeat, tokenization, optimizer progress,
and checkpoint lineage must each be checked after admission. First resource
snapshot showed ordinary Qingyuan A100 quota 112/using 75/waiting 27 after the
two new 8-GPU requests; ordinary Zhangjiakou H20 quota 136/using 107/applying
8. The queue may reflect whole-node scheduling despite numerical quota room.

At the next live check, Closed-Loop reached `TRAINING_RUNNING` on a real 8×H20
Pod. The Ceph-mounted code and frozen config were both readable inside the
Pod; eight `python3.11` workers and a `pt_elastic` parent were active in the
pretokenization stage. GPU memory was still 0 MiB, so **no optimizer training
or checkpoint is claimed yet**. Direct and Open-Flow remained in
`TRAINING_RESOURCE_WAITING`. The Taiji CLI log endpoint still showed only
startup/precheck lines despite the launcher heartbeat and shared PID-1 stdout
pipe; keep checking both Pod processes and platform logs rather than treating
the platform state alone as evidence of progress.

Resource-only replacement on 2026-10-03: the Direct Qingyuan A100 instance
`8b1d8047a0d27a4401a0fd745edb4230` remained in
`TRAINING_RESOURCE_WAITING` for roughly 19 minutes and was explicitly stopped;
it reached `END` without creating a training output. The same frozen Direct
config/data/seed/output contract was resubmitted on ordinary Zhangjiakou
8×H20 as task `meteor_mechet_nmi_h2_direct_8h20_zjk_20261003_01`, instance
`8b1d89f7a0d297d901a0fd8659de41ea`, initially `PENDING`. The Closed-Loop
H20 and Open-Flow A100 instances were not stopped. This is not a change to
the split, model, optimizer or scientific comparison; GPU type and wall time
must be reported separately.

The Closed-Loop distributed tokenizer then completed its frozen cache
manifest: **223,863** train reactions became **231,270** lossless training
windows (7,286 rows windowed); validation was 6,200 reactions/windows.
Recorded truncation is zero, and independently checked train/valid source
SHA-256 values match the representation manifest. Natural train exposure is
1,443,677,982 input tokens and 170,411,040 assistant-supervised tokens;
these are not the final three-epoch presented-token totals. The Pod next
launched eight GPU processes with about 7.6 GiB reserved per H20, but no
optimizer step was yet verified at this observation.

The replacement Direct H20 task subsequently reached `TRAINING_RUNNING` on
a real 8×H20 Pod with Ceph config visible. Its tokenizer cache also completed:
223,863/223,863 train rows/windows, 6,200/6,200 validation rows/windows,
zero truncation, 91,654,571 natural train input tokens and 45,230,044
assistant-supervised tokens. Thus Closed-Loop has about 15.8× Direct's natural
input-token count before repetition across optimizer steps. E3 is matched on
IDs, backbone, seed and optimizer updates, **not** on token or FLOP budget;
any method comparison must state this limitation explicitly. At this check
the two H20 Pods had model processes and GPU memory allocations, but neither
had a verified optimizer step or saved checkpoint; Open-Flow A100 was still
waiting for resources.

At 01:19 CST the Closed-Loop Pod showed 100% utilization on all eight H20s
(about 25–26 GiB per GPU); Direct showed 76–96% utilization (about 14–16 GiB
per GPU). Neither output directory yet contained `trainer_state.json` or
adapter weights. This confirms sustained GPU work but does not by itself
prove a completed update, epoch, or usable checkpoint. The Taiji CLI's log
endpoint still displayed only launcher/precheck text, so any loss/step claim
must come from a saved trainer state or another direct Pod observation.

Frozen **K=10 evaluation task configs are prepared but not submitted** under
`configs/taiji/meteor_mechet_nmi_h2_{direct,open_flow,closed_loop}_k10_*`.
They retain all 27,104 H2 held-out reactions, the same stochastic candidate
budget, strict adapter/train-test SHA guards, and default terminal heartbeat.
Direct/Open-Flow use ordinary Qingyuan A100; Closed-Loop uses ordinary
Zhangjiakou H20. The Direct completion cap 2,560 exceeds the maximum 2,330
total train-sequence tokens, and the Open-Flow cap 4,096 exceeds the maximum
3,989 gold assistant characters in H2 train; these caps were chosen from
train-only evidence, before model test. Submission must wait for completed,
lineage-verified adapters and resource reinspection. No eval task or held-out
model result is claimed here.
The three evaluation configs were independently rendered into private
mode-0600 Taiji configs using successful A100 and H20 Ceph-init donor tasks;
each rendered config has the requested eight-GPU type, a Ceph mount command
and the default-log heartbeat wrapper. Rendering is **not** task creation or
submission. The private init commands and rendered files are not committed.

The Closed-Loop test launcher received a full **reference-prompt contract
preflight** over all 27,104 H2 test rows (no model generation): one system
prompt, one frozen tool schema, one 40-call budget, one
`compact_full_state` observation mode and `finish_trace` terminal tool;
27,104/27,104 user prompts began with their own frozen product target and
zero had malformed initial-observation JSON. This validates launch-format
uniformity only, not prediction quality or complete gold replay.
The Direct and Open-Flow H2 test files were also streamed in full: each has
27,104 unique IDs, one system prompt, exactly system/user/assistant message
roles, no tool schema, and zero rows whose user text differs from
`TARGET: <that row's product SMILES>`. The direct inference path's
`_direct_messages` retains only system/user roles and discards the stored
assistant gold continuation. These checks exclude an accidental answer-bearing
test prompt; they do not address information potentially encoded in mapped
product atom labels or predict model accuracy.
An additional prompt-isolation regression poisons the stored reference
assistant/tool messages, structural endpoint, compiled proof and trace plan;
neither the Direct/Open-Flow direct prompt nor the Closed-Loop runtime-generated
prompt includes that poison. This checks the generation-side code path, while
the reference answers remain available only to the later evaluator.
An independent full 27,104-row scan of each condition's H2 test JSONL found
zero non-null top-level or metadata `conditions` and `competitor_products`
fields. Therefore the environment reset cannot introduce these two auxiliary
inputs on this test; this scan does not certify the model's chemistry.

Resource-only Open-Flow replacement on 2026-10-03: the original Qingyuan A100
instance `8b1d80eea0d297ec01a0fd748a9c41b5` remained
`TRAINING_RESOURCE_WAITING` for about 35 minutes, then was stopped and verified
`END` without training output. The same frozen Open-Flow data/config/seed/output
was submitted to ordinary Zhangjiakou 8×H20 as task
`meteor_mechet_nmi_h2_open_flow_8h20_zjk_20261003_01`, instance
`8b1d8064a0d2977501a0fd957208409d`; its first observed state was
`TRAINING_RESOURCE_WAITING`. The H20 group then reported quota 136, using 123,
waiting 8. No duplicate Open-Flow instance remains active. The H20 task must
still obtain a real Pod before training is claimed.

At 01:29 CST, the same ordinary H20 group still reported quota 136, using
123, waiting 8, with exactly the Open-Flow task in its waiting list; the
cluster-wide resource display showed no currently available Zhangjiakou H20
host. This is evidence of an allocation wait despite nominal quota headroom,
not evidence that the task's application group or location is mistyped. The
earlier Qingyuan A100 Open-Flow instance is terminal, so there is no active
duplicate to consume capacity.

At 01:33 CST, Direct had a complete, nonzero `checkpoint-250` containing
`trainer_state.json`, LoRA adapter weights, optimizer and scheduler states.
The trainer state records global step **250/10,494** (2.38% of the fixed update
budget), epoch 0.07147, 1,147.9 seconds of trainer runtime and recent
teacher-forced loss 0.0673 at step 250. This is the first verified optimizer
progress, **not** endpoint accuracy or a finished adapter. The three task
handles were rechecked: Direct and Closed-Loop remained `TRAINING_RUNNING`,
Open-Flow remained `TRAINING_RESOURCE_WAITING`. No Closed-Loop trainer state or
weights were present yet, so its actual step count remains unverified.
The saved Direct adapter has 288 tensors / 15,335,424 parameters, all finite
and nonzero in a read-only safetensors audit. Logged training loss fell from
0.6376 at the first recorded step to 0.0673 at step 250; this only checks
training health and cannot establish held-out chemistry performance.

At 01:43 CST the Direct task had advanced to a complete `checkpoint-500`,
global step **500/10,494** (4.76%), epoch 0.14294, trainer runtime 2,307.7
seconds and recent teacher-forced loss 0.0423. The earlier checkpoint was at
01:21, so progress is real rather than a static active-state label. The
Closed-Loop task still had no saved checkpoint, but its eight actual
`train_tool_sft.py` workers had been alive about 55 minutes and all eight H20s
were at 99–100% utilization with roughly 25–26 GiB allocated per card.
This supports ongoing compute, not a verified Closed-Loop step count. The
Open-Flow handle remained `TRAINING_RESOURCE_WAITING`.

At 02:23 CST, Direct had reached complete `checkpoint-1000` (9.53% of
10,494 updates; epoch 0.28589), with trainer runtime 4,644.4 seconds,
26,183,079 input tokens seen and recent training loss 0.0319. There was no
validation loss or endpoint evaluation yet. Closed-Loop remained
`TRAINING_RUNNING` but still had no first checkpoint, so its step count cannot
be inferred from GPU utilization. Open-Flow remained in H20 resource waiting
after about 76 minutes. The Qingyuan A100 ordinary group simultaneously had
33 waiting GPUs, so moving the same task back to its earlier A100 queue had
no evidence-backed scheduling advantage at this check.

At 02:35 CST the Open-Flow H20 instance detail gave the explicit scheduler
message `底层资源不足，请稍侯重试或排队等待。` (insufficient underlying resources; retry
or queue). Its transition history repeatedly showed
`TRAINING_RESOURCE_WAITING → RESOURCE_WAIT_TRANSITION → TRAINING_INIT →
TRAINING_RESOURCE_WAITING`, with no allocated Pod. Thus the short
`RESOURCE_WAIT_TRANSITION` states are scheduler retry cycles, not training
startup or a user-program failure. Do not duplicate or rewrite the task in
response to those transient states.

At 04:31 CST, Direct had advanced to complete `checkpoint-2500` (23.82% of
the fixed updates). Closed-Loop still had no `trainer_state.json`, but all
eight `train_tool_sft.py` processes had run for about 3 h 42 min, each H20
remained at 100% utilization with roughly 25–26 GiB allocated, and no
terminal task state was observed. This is sustained computation, not evidence
of a particular step count or convergence. Open-Flow remained queued for
underlying H20 resources; the ordinary Qingyuan A100 group had 41 GPUs
waiting at this check, so the original A100 queue was not a clear faster
replacement.

At 05:03 CST, Closed-Loop finally saved complete `checkpoint-250`: verified
global step **250/10,494** (2.38%), epoch 0.06918, 99,751,766 input tokens
seen, and 14,570.8 seconds of trainer runtime. Loss fell from 0.8627 at the
first logged step to 0.1215 at step 250; no validation result exists yet.
The 288 LoRA tensors (15,335,424 parameters) were independently read and are
all finite and nonzero. The 50-step runtime intervals were approximately
2,880–2,990 seconds, or about 58 seconds/update. Naively extrapolating the
remaining 10,244 updates gives roughly seven more days **if throughput stays
constant**, excluding validation, saving and scheduling effects. This is an
operational estimate, not an accuracy result or a commitment to alter the
frozen update budget. Direct had reached complete `checkpoint-3000` at 04:59;
Open-Flow remained resource-waiting.

At 06:05 CST Direct had reached `checkpoint-3750` (epoch 1.0720). The first
scheduled teacher-forced validation at step 3,498/epoch 1 completed with
`eval_loss=0.0345774` on all 6,200 **narrow, all-three-step H2 validation**
rows; this is not autonomous endpoint recovery and must not be used as a
stand-in for the held-out H2 test. The most recent training loss was 0.0208.
Closed-Loop's latest complete state remained `checkpoint-250`; the Open-Flow
H20 scheduler handle remained in `TRAINING_RESOURCE_WAITING` after about five
hours, with no Pod and no model output.

At 07:11 CST Direct and Closed-Loop were still `TRAINING_RUNNING`; their latest
complete checkpoints were steps **4,500/10,494** and **250/10,494**, respectively.
Open-Flow was still `TRAINING_RESOURCE_WAITING` with no allocated Pod after
roughly six hours. A seemingly shorter AILab A100 queue is not a verified
drop-in alternative: the historical AILab task
`meteor_mechet_a4_open_flow_infer_k1_8a100_ailab_20260829_01` ended before
the user program because the Tencent `taiji7` image required `cuda>=12.2` and
the host failed the NVIDIA prestart check. An older CUDA 11.8 image probe on
AILab also ended unsuccessfully; it did not establish a compatible eight-GPU
training runtime. Moving Open-Flow to that group with the known-failing image
would trade a resource wait for a predictable container failure. The H20 task
therefore remains queued while a compatible alternative is not yet proven.

At 09:05 CST Closed-Loop saved a second complete checkpoint, step
**500/10,494**, epoch 0.13837, with 199,212,576 input tokens seen and recent
teacher-forced loss 0.0771. The interval from step 250 to 500 was about
4 h 2 min, confirming sustained optimizer progress at roughly 58 s/update;
there is still no validation or endpoint result. Direct had reached complete
`checkpoint-6000` at 08:59 (57.2% of its fixed updates). Open-Flow remained
in H20 resource waiting. These are training and scheduler observations, not
H2 model-effect estimates.

At 10:25 CST Direct was still `TRAINING_RUNNING` and had saved complete
`checkpoint-7000` (epoch 2.0011). Its second scheduled teacher-forced H2
validation was `eval_loss=0.0338372` at step 6,996 versus `0.0345774` at
step 3,498/epoch 1. The small decrease indicates continued fit on the narrow
6,200-row validation monitor, **not** endpoint recovery or H2 test accuracy.
Closed-Loop remained `TRAINING_RUNNING`; Open-Flow remained in resource wait.

At 13:19 CST Direct had saved complete `checkpoint-9250` (88.1% of the fixed
10,494 updates; epoch 2.6444, 242,411,945 cumulative input tokens). The
step-9,250 training loss was 0.0191, not a model-evaluation metric. Closed-Loop
had saved complete `checkpoint-750` (7.15%; epoch 0.2075, 299,670,544 input
tokens), after roughly 4 h 4 min since its step-500 checkpoint. Its recent
training loss was 0.0614. The Open-Flow H20 scheduler handle remained in
resource wait. These task/checkpoint observations confirm continuing
optimization but do not establish a held-out endpoint result.

At 14:58 CST Direct's H20 training instance
`8b1d89f7a0d297d901a0fd8659de41ea` reached successful `END`. Its final
checkpoint records exactly **10,494/10,494 updates** and epoch 3.0. The
root adapter has 288 finite, nonzero LoRA tensors (15,335,424 parameters);
its recomputed directory digest matches `adapter_manifest.json`:
`7eac4182629a14283eae7cf9071ca32b321c893c597211de3cb229564666ec47`.
The adapter and data contract both bind training SHA-256
`009d624447d8f36441a3885cfd61c357be7687e1edc17e2d9ffa18eaca311df9`
to the frozen Direct train manifest, with the pinned Qwen3-8B revision. The
last logged teacher-forced loss remains a training diagnostic, not H2 accuracy.

After this lineage gate and an A100 resource check (ordinary Qingyuan quota
112, using 72, waiting 23), one full **K=10 / 27,104-row Direct evaluation**
was submitted as `meteor_mechet_nmi_h2_direct_k10_8a100_qy_20261003_01`,
instance `8b1d8064a0d2977501a10092fffa4356`. The initial state was
`PENDING`; there is no generated prediction or evaluation result yet. Its
rendered config uses the successful A100 private Ceph donor, default terminal
heartbeat, and the immutable final adapter path. Submission is not a result.

The waiting Open-Flow H20 task obtained the H20 allocation released by
Direct. At 15:03 CST instance `8b1d8064a0d2977501a0fd957208409d` was
`TRAINING_RUNNING`: the frozen data were visible on the Pod's Ceph mount and
eight `train_tool_sft.py` workers were present. GPU memory was about 7.6 GiB
per card with zero instantaneous utilization; this is initialization, not yet
verified optimizer progress or a checkpoint. Closed-Loop continues on its own
eight H20s.

The first Direct K=10 evaluation allocated eight Qingyuan A100s but ended
unsuccessfully at 15:06 CST **before any prediction row**: all eight generation
workers raised `ModuleNotFoundError: No module named 'vllm'` in the base image.
The default Taiji log captured the worker stack traces, progress 0/27,104,
and the terminal heartbeat. The failure directory contains only eight shard
logs; no evaluation or candidate artifact exists. This is a runtime dependency
failure, not a scientific negative result. The failed instance and logs remain
untouched at `outputs/issue79/eval_h2_direct_k10_seed17/`.

A resource-only/runtime repair uses the already installed, versioned Ceph
runtime `artifacts/taiji_vllm_runtime/vllm_0_8_5_torch_2_6_cu124_py311`,
which imported locally as vLLM 0.8.5 / PyTorch 2.6.0+cu124 and was used by
the successfully completed historical eight-A100 vLLM task
`meteor_mechet_a7_infer_k10_vllm_8a100_qy_20260831_03`. The retry has a
new task flag, a separate output directory, and a bootstrap preflight that
fails before generation if this frozen runtime is missing or CUDA-incompatible.
Its model adapter, H2 reference, K=10, seed, temperature, top-p, token cap and
NLL ranking settings are byte-identical to the first config. The repair does
not reinterpret the first failure as an endpoint result.

The repaired Direct evaluation was submitted after checking ordinary Qingyuan
A100 resources (quota 112, using 71, waiting 22): task
`meteor_mechet_nmi_h2_direct_k10_vllm_runtime_8a100_qy_20261003_02`, instance
`8b1d89c4a0d297dc01a1009d39184465`. Its Pod mounted the Ceph adapter and
versioned runtime, started eight vLLM 0.8.5 engines on eight 40-GiB A100s,
completed model loading/compilation and began writing prediction shards by
15:42 CST. A format-only sample across all eight shards has exactly ten
independent candidates per row, `backend=vllm`, and one shared adapter hash;
no test answers or endpoint scores were examined. This is ongoing generation,
not a completed 27,104-row evaluation. The first failed task's output remains
separate from the retry.
The primary metric's candidate-order contract was checked without reading
test answers: installed vLLM 0.8.5 documents that offline `LLM.generate`
returns requests in input-prompt order and sorts completed requests by numeric
request ID. In the first 40 written rows from each of eight running shards
(320 partial rows), all candidate lists had indices 0–9 in order, ten distinct
candidate seeds, `selected_candidate_index=0`, and `backend=vllm`; zero
generation errors were recorded. This is a partial runtime integrity check,
not the final candidate-coverage gate or a model-accuracy result.

Open-Flow also advanced beyond initialization: by 15:51 CST it had saved a
complete `checkpoint-250` (250/10,494 updates, epoch 0.07147). Its eight H20
GPUs showed active compute before the checkpoint. The recent teacher-forced
loss of 0.1849 is a training-health observation, not H2 endpoint recovery.
The completed **superseded Open-Flow v1** token-cache manifest verifies 223,863/223,863
train reactions/windows, zero truncation, and source SHA-256
`67ee365386b16a7c9896c2ba63f1c770acc6284ce2eb4d6e71c04c2c5e9c3996`
against its frozen representation manifest. Natural train exposure is
150,871,377 input tokens and 102,432,083 assistant-supervised tokens.
The corrected v2 manifest below supersedes this v1 exposure for the headline
comparison. For the **active** Direct / Open-Flow v2 / Closed-Loop conditions,
single-pass natural train input tokens are 91,654,571 / 150,873,337 /
1,443,677,982, and assistant-supervised tokens are 45,230,044 / 102,434,043 /
170,411,040. All three target 10,494 optimizer updates, but these counts are
not measured three-epoch FLOPs or a claim of equal token budgets.
At global batch 64, that cap nominally presents 671,616 training windows.
Direct and corrected Open-Flow each have 223,863 windows, or about 3.00
window-epochs; Closed-Loop has 231,270 lossless windows from the same 223,863
reactions, or about 2.90 window-epochs. Actual sampler repetition/padding can
change exact per-window exposure. Thus matched update count is not an
identical per-window or per-token exposure claim.
The not-yet-submitted Open-Flow and Closed-Loop K=10 configs were updated to
activate the same frozen vLLM runtime at launch, preventing the Direct image's
missing-module failure from recurring by construction. Their dataset, model,
candidate and ranking contracts remain unchanged; neither is submitted before
its own final adapter exists.

### Open-Flow import-supervision repair (2026-10-03; before H2 model outcomes)

An additional full-data **program-target parity** audit compared Open-Flow
`<flow>` imports and ordered electron steps with the Closed-Loop tool calls on
every frozen H2 ID, checking both source-file SHA-256 values. Electron-step
sequences matched on **223,863/223,863 train, 6,200/6,200 valid and
27,104/27,104 test**. The older Open-Flow builder, however, serialized only
`trace_plan.initial_imports`, omitting `steps[*].imports`. Its import lists
therefore differed on **77 train, zero valid and 603 test** rows; these are
exactly the rows where Closed-Loop imports fragments after a previous step.
The audit is `outputs/issue79/nmi_h2_program_parity_20261003.json`. This is
a real representation bug, not a different electron-move target or a model
negative result. The 603 affected test references are 2.22% of the frozen
27,104-row denominator; no row is removed.

The OPEN_FLOW v1 grammar requires all imports before STEP 0. A deterministic
repair now serializes initial imports plus each later step's imports, in
recorded order, before the same electron steps. On all **77 train and 603
test** affected gold programs, this fixed representation strictly executed
and reproduced the frozen structural endpoint (680/680). Unaffected rows are
copied byte-for-byte. The independent v2 artifact is
`data/issue79/nmi_open_flow_all_step_imports_v2_20261003/`, still
223,863/6,200/27,104 rows, with train/valid/test SHA-256 values
`ef1a03b880e8cf4a5f43d0df175c7d6ae56a35eb4ca4db51e275aa02653e3f3c`,
`73d776d30c4506beb76c982b6e0f171f2aa288ed42631bc364f29e50e988d967`,
and `77b70a2df2b6c455acd801b0c790c8078986ce8d6a578533d1c3f05d5e82f3d8`.
Its config preserves the same Qwen3-8B revision, seed 17 and 10,494 update
budget. The first packaging-only preflight directory was archived with
`training_allowed: false`; no job used it. After the repaired data and runner
contracts passed, replacement task
`meteor_mechet_nmi_h2_open_flow_imports_v2_8h20_zjk_20261003_01`
(instance `8b1d89c4a0d297dc01a1010c9c4e45b4`) was submitted. Only the old
Open-Flow task `meteor_mechet_nmi_h2_open_flow_8h20_zjk_20261003_01` was
stopped; Taiji reports `END`, and its partial checkpoint-1000 remains archived.
That old adapter is superseded and ineligible for the headline H2 comparison.
The replacement initially reached `TRAINING_RUNNING` with a real eight-H20
POD and passed host prechecks. Its distributed token-cache manifest records
223,863/6,200 train/valid rows, no dropped/windowed/truncated source rows,
150,873,337 train input tokens and 102,434,043 supervised tokens, with the
v2 train and valid SHA-256 values. Eight H20s executed real optimizer work:
`outputs/issue79/h2_open_flow_imports_v2_seed17/checkpoint-250/` contains a
complete adapter and trainer state at **250/10,494 updates** (epoch 0.07147,
latest teacher-forced loss 0.1842, finite gradients). This proves training
started, not an H2 endpoint result. A sidecar now labels
the old Open-Flow data `training_allowed: false` and
`headline_evaluation_allowed: false`; both Taiji launchers enforce such
sidecars without changing frozen source manifests. Direct and Closed-Loop
artifacts/tasks remain unchanged.

An independent full-reference replay audit (not model evaluation) now checks
each v2 Open-Flow program against its frozen structural endpoint while
rechecking the split-file SHA-256 and denominator. The **entire 6,200-row
validation split passed 6,200/6,200 strict execution plus structural endpoint
match**, audit SHA-256
`3a88a3e8cf75e549715907dae81446b67a7430ee0575dd78290ab0df532a55a6`
at `outputs/issue79/nmi_open_flow_v2_reference_replay_valid_20261003.json`.
The **entire 27,104-row H2 test reference split also passed 27,104/27,104**
strict execution plus structural endpoint match, with zero failures. Its
source-file SHA-256 is
`77b70a2df2b6c455acd801b0c790c8078986ce8d6a578533d1c3f05d5e82f3d8`
and audit-result SHA-256 is
`accaeee0dfd3f7f9a5ed8ff5d28c31df915e0bf8dee5f7daa859505049834dec`
at `outputs/issue79/nmi_open_flow_v2_reference_replay_test_20261003.json`.
This is a complete **gold-reference data audit**, not a model prediction or a
checkpoint-selection signal.

The three frozen inference launchers all specify the same Qwen3-8B revision,
vLLM backend, seed 17, temperature 0.7, top-p 0.95 and **K=10** candidates
per reaction. Their representation-specific output ceilings are intentionally
different: Direct 2,560 generated tokens, Open-Flow 4,096, and Closed-Loop at
most 40 tool iterations with 512 tokens per iteration. Direct/Open-Flow use
eight A100s while Closed-Loop is configured for eight H20s; endpoint accuracy
is comparable under the matched IDs/candidate budget, but raw latency or GPU
throughput is **not** a matched-hardware contrast. This runtime distinction
must accompany any later efficiency claim.

No H2 model performance result is available yet. Every result must be tied to frozen source/split hashes and its own
model checkpoint lineage; historical full-data results remain supporting
evidence only.

The frozen E4 paired-analysis code reports generation-order first-candidate
accuracy and Pass@10 as the matched primary comparison. It also reports a
**separate, supplementary selected Top-1**: Direct uses assistant mean-NLL,
Open-Flow first gates on formal execution and then uses assistant mean-NLL,
and Closed-Loop uses its own executor selector. These three selectors are
different and their selected Top-1 numbers must not be presented as an
isolated effect of closed-loop feedback. The analyzer requires complete K=10
row-level candidate results and valid ranking permutations before producing
any model result; it additionally checks that candidate indices preserve
generation order and that the Closed-Loop parent-selected outcome agrees with
the indexed candidate. This prevents an index/selection bookkeeping mismatch
from silently contaminating the supplementary selected Top-1. All 25 NMI
regression tests passed after the gate was added. No H2 model outcome is
available yet.
For the method-specific process metrics, Open-Flow and Closed-Loop report
formal execution, while Closed-Loop additionally reports generation-order,
Pass@10 and selected-candidate **trace-bound** rates. The latter requires a
finished, digest-bound electron chain that independently replays, but it is
**not** exact agreement with the recorded reference's intermediate successor
states. The analyzer rejects Closed-Loop row artifacts missing per-candidate
trace-bound fields rather than silently treating them as failures. This field
and its semantic label were frozen before any H2 model outcomes; six focused
analysis/row-sink regression tests pass.
The Direct and Open-Flow evaluators now embed the same inference runtime
contract already recorded for Closed-Loop. Before paired analysis, the H2
analyzer requires complete, single-adapter runtime lineage for all three,
the pinned Qwen3-8B model/tokenizer revision, and the same seed, temperature,
top-p and K. It records each generated adapter hash with the result instead
of relying only on a task name. This is an analysis provenance gate, not a
change to generation or endpoint scoring; 37 NMI/evaluator regression tests
pass after the addition.
Before inspecting any model H2 outcome, E4 additionally froze a
reaction-paired adjusted slope: does the **Closed-Loop minus Open-Flow**
generation-order Top-1 gap change with the log frequency of the least-common
known primitive, holding scaffold, local-center, near-duplicate, trajectory
length and import count covariates fixed? It uses reaction-level bootstrap
intervals and is descriptive, not a causal estimate. This supplements—not
replaces—the primary paired accuracy contrast.
The same pre-outcome analysis now reports reaction-paired Closed-Loop minus
Open-Flow Top-1 and Pass@10 differences with bootstrap intervals separately
for scaffold-seen/unseen, local-center-any/none, and near-duplicate/non-near-
duplicate reactions. These fixed strata test whether any overall advantage
persists outside the obvious structural-overlap subsets; their intervals are
descriptive and should not be selected post hoc as a new primary endpoint.
The label-free 27,104-row H2 test covariates give full rank 7/7 for the
intercept-plus-six-feature design matrix (condition number 69.6), 185 distinct
log-frequency values, and full rank in 30/30 seeded reaction bootstrap
resamples. This is a feasibility check of the predeclared model, **not** an
estimate of any model's accuracy or primitive-frequency effect.
An analysis-runtime preflight used these same covariates with **synthetic**
binary outcomes, never model predictions. Twenty bootstrap logistic fits took
49.2 seconds with the host's default BLAS threading and 2.0 seconds with
BLAS limited to one thread. The analyzer now applies that thread limit around
the unchanged predeclared fits; this is only a CPU scheduling optimization,
not a different regression, draw count, seed or scientific result.
For an interpretable, pre-outcome display, the E4 analyzer also freezes five
minimum-primitive-train-frequency strata: 5–9 (1,572 reactions), 10–24
(2,334), 25–99 (1,932), 100–999 (7,342), and ≥1,000 (13,924). It will report
the three matched generation-order endpoint rates and the reaction-bootstrap
Closed-Loop minus Open-Flow difference within each stratum. These boundaries
were set from the frozen covariates before any H2 prediction result and do not
replace the adjusted regression or the primary all-test contrast.

The frozen final-analysis invocation below must run **only after all three
27,104-row K=10 evaluation reports and row files exist**. The analyzer refuses
an existing output and rechecks reference hashes, stable IDs, candidate
counts/order and runtime contracts before writing the paired result:

```bash
PYTHONPATH=src:. python scripts/analyze_nmi_h2_results.py \
  --covariates /aaa/fionafyang/buddy1/whaleywang/MechET/outputs/issue79/nmi_h2_covariates_20261003.jsonl \
  --matched-dir /aaa/fionafyang/buddy1/whaleywang/MechET/data/issue79/nmi_matched_v1 \
  --open-flow-dir /aaa/fionafyang/buddy1/whaleywang/MechET/data/issue79/nmi_open_flow_all_step_imports_v2_20261003 \
  --direct-evaluation /aaa/fionafyang/buddy1/whaleywang/MechET/outputs/issue79/eval_h2_direct_k10_seed17_vllm02/evaluation.json \
  --open-flow-evaluation /aaa/fionafyang/buddy1/whaleywang/MechET/outputs/issue79/eval_h2_open_flow_imports_v2_k10_seed17/evaluation.json \
  --closed-loop-evaluation /aaa/fionafyang/buddy1/whaleywang/MechET/outputs/issue79/eval_h2_closed_loop_k10_seed17/evaluation.json \
  --output /aaa/fionafyang/buddy1/whaleywang/MechET/outputs/issue79/nmi_h2_matched_analysis_seed42.json
```

### 2026-10-03 evening: Open-Flow replay and Direct postprocessing recovery

The corrected Open-Flow v2 reference programs passed strict execution and
structural-endpoint replay on **every** H2 train/valid/test row:
223,863/223,863, 6,200/6,200, and 27,104/27,104, with zero failures. The
three output JSON files are
`outputs/issue79/nmi_open_flow_v2_reference_replay_{train,valid,test}_20261003.json`.
This is a reference-data audit, not model accuracy. The v2 Open-Flow task is
training; its latest observed complete checkpoint was 2,750/10,494. The
Closed-Loop task was also training, with latest complete checkpoint
1,250/10,494 at this inspection.

The Direct 8×A100 vLLM retry completed all 8 generation shards: 27,104
targets × 10 candidates. The Taiji instance
`8b1d89c4a0d297dc01a1009d39184465` nevertheless ended unsuccessfully
after generation because the long-running Bash launcher reported a syntax
error in its inline merge block; **no NLL ranking or endpoint evaluation was
performed by that task**. Do not treat the task's `END/IsSuccess=false` as a
model failure. A separate strict streaming merge verified the frozen
reference ID order, eight shard manifests, 10 distinct sample indices per
reaction, the test hash, adapter hash, and pinned base-model revision. The
lossless merged file is
`outputs/issue79/eval_h2_direct_k10_seed17_vllm02/predictions.jsonl`, SHA-256
`f881ca50b317249f474dc55e8e895d708c218d1c9070ccea9ffd7d3407686c53`.
The recovery launcher runs only frozen assistant mean-NLL scoring and the
already-defined endpoint evaluator; it never regenerates or edits candidates.
The recovery task
`meteor_mechet_nmi_h2_direct_nll_recovery_8a100_qy_20261003_01`
(instance `8b1d8922a0d2976301a1023460fd4617`) was submitted to ordinary
Qingyuan 8×A100 at 22:39 CST, with a successful Ceph donor and a configured
default-log heartbeat. It reached Taiji `TRAINING_RUNNING`; eight NLL shard
files began accumulating actual score rows (95/27,104 at the first post-start
inspection), confirming scoring rather than merely resource allocation. The
CLI log endpoint had not yet exposed the runner's heartbeat, so continuous
platform-log capture remains to be checked. Until final evaluation finishes,
there is no Direct Top-1/Top-10 result to report.
The future shared inference launcher now invokes the separately tested
streaming shard merger instead of a long inline Bash heredoc. It validates
reference order, resumed-shard counts, candidate indices and per-row model
lineage before NLL scoring. This is a postprocessing reliability change;
sampling, ranking formula and scientific denominators are unchanged.
The same merger is now used by the planned Closed-Loop K=10 launcher. Its
previous inline block would have loaded all 27,104 × 10 complete interaction
trajectories into one Python list after generation. The stream merger instead
interleaves the eight prepartitioned shard files against the frozen selected
reference order, verifies the original full-test SHA supplied by the parent
launcher and preserves the old manifest semantics. This removes a late
memory-failure risk without changing any trajectory or metric.
The new prepartitioned trace path was exercised in read-only mode on an
existing completed 28,967-row A7 K=1 artifact: all 8 shard manifests and
28,967 row/model-lineage/order contracts passed; no prediction or evaluation
file was written. This is a runtime integration check, not an H2 model result.
An **optional, not-yet-submitted** cross-task merge is now available in
`scripts/merge_nmi_h2_task_shards.py` if resource and throughput checks justify
splitting the later Closed-Loop evaluation over multiple ordinary 8-GPU tasks.
The inference worker derives each candidate seed from the stable reaction ID
and sample index, independent of task partitioning. The merger checks one
complete modulo task shard per index, the full and selected-reference hashes,
all task prediction hashes, every reaction ID in original order, K distinct
candidate indices and identical adapter/runtime metadata. It copies candidate
rows without selecting or editing them; the existing full-reference evaluator
must still be run on its output. Six synthetic corruption/order tests pass.
This does **not** change the frozen 27,104-reaction denominator or establish a
model result, and no additional GPU task was submitted by this preparation.
The frozen endpoint evaluator now constructs each candidate's read-only view
without deep-copying the entire K-candidate parent. The prior implementation
duplicated all sibling histories for every K=10 candidate, a quadratic
evaluation-time cost unrelated to chemistry. Candidate fields, endpoint
execution and metric definitions are unchanged; 32 relevant regression tests
passed after this change. This does not alter any submitted training or
sampling job.

## Provisional paper Results structure (no paper edit yet)

1. **Operator support and the testable regime.** Report the observed 604 local
   primitives, 6,553 ordered move compositions, vocabulary-growth curve and
   the 1.23% random-holdout program-unseen/primitive-seen rate. Describe these
   as properties of the strict-executable training pool, not all official
   FlowER reactions or a claim about physical mechanism uniqueness.
2. **Frozen composition holdout and structural audit.** State the 223,863 /
   6,200 / 27,104 candidate H2 split, zero exact reaction/composition overlap,
   full primitive support, scaffold/local-center/near-duplicate overlap, and
   the all-three-step validation limitation. Separate this new split from the
   official FlowER 257,171 / 2,890 / 28,971 reaction-level split.
3. **Matched generation and execution outcomes — pending.** Once all three
   adapters finish, show generation-order first-candidate endpoint recovery and
   Pass@10 with reaction-paired intervals, emphasizing Closed-Loop versus
   Open-Flow. Put condition-specific NLL/executor selected Top-1 in a clearly
   separate supplement; report executable and trace-bound rates with their
   non-GT-successor semantics. Include token/FLOP and wall-time differences.
4. **Familiarity analysis and decision — pending.** Report the predeclared
   primitive-frequency strata, adjusted structural-overlap-controlled
   association and paired Closed-Loop-minus-Open-Flow slope; then issue the
   specified GO/NO-GO judgment. Do not write a positive Results claim or edit
   `MechET-paper` before those model outcomes exist.
