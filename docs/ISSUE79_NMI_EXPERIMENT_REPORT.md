# Issue #79: NMI compositional-generalization experiment record

## Step 0 audit (2026-10-02, before experiment-code changes)

| Item | Code / submission / completion / scientific evidence |
|---|---|
| PRs #70–#78 | All merged on `main` (`a4f01fc`); code merge alone is not an experimental result. |
| Protocol-v2 parity | `docs/NATURAL_LANGUAGE_PROTOCOL_V2.md` records 2,890/2,890 validation gold replays and 22,341/22,341 byte-exact decision prompts; the machine-readable audit is `docs/audits/natural_language_protocol_v2_valid_20260918.json`. |
| State-SFT / Trajectory-SFT | Both full strict-universe adapter manifests exist under `outputs/agent/natural_language_event_v2_qwen3_8b_h20_seed17_20260918` and `outputs/agent/natural_language_event_history_v2_qwen3_8b_h20_seed17_20260918`; train SHA-256 values are `22729aff36db3691b3db544aa83cb93c7383e9fbacda6934e6a52a3d690f6bfa` and `bc0a859a3c25b6146a8f6cac730610a0d66347baff1f9648b67329b17d785b97`. These full-data checkpoints cannot serve as headline H2 controls after a new split. Frozen H=1/2/3/full and EARHO outcome reports still require provenance-level inspection before reuse. |
| Pointer | 16 product-start cases completed on Taiji; baseline and coupled pointer each have Top-1=0/16 and Pass@beam=0/16. Predeclared smoke gate stopped 256 and 1,319 stages. The 44 invalid handles count rejected actor proposals, not executed invalid actions. |
| Runtime | Five-mode oracle-state benchmark is not complete. The PR #78 recovery reached Taiji twice but its ledger ended `INFRA_FAILED`. POD traceback identifies a **user-program compatibility error**, `Unsupported option for the xgrammar backend: no-fallback`, in pinned vLLM 0.8.5 V1 engine initialization. The controller's `INFRA_FAILED` label is not a scientific result or proof of platform failure. This runtime arm is not necessary for E1/E2 CPU audits and must not be silently retried with changed benchmark gates. |
| Packing | Completed report: loss absolute difference 0.00146447, throughput speedup 0.86350, peak-memory ratio 1.28073. The speed gate failed: `SCIENTIFIC_STOP`, no further packing optimization for #79. |
| MechComp C1/C2/C3 and H2 | Split builder, overlap audit and H2 runner exist, but `data/ood/mechcomp_source_sink` and `outputs/h2` were absent at this audit. No valid matched split/checkpoint/result is established. |
| PMechDB / PMechRP | No authorized source assets were found in the workspace. E5 is access-blocked, not a reason to stall E1–E4; no mirror substitution or new download is authorized. |

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
seen primitives. This is 1.23% of the random held-out reactions. Thus a random
split would mostly retest familiar programs; E2 must hold out composition
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
Before inspecting any model H2 outcome, E4 additionally froze a
reaction-paired adjusted slope: does the **Closed-Loop minus Open-Flow**
generation-order Top-1 gap change with the log frequency of the least-common
known primitive, holding scaffold, local-center, near-duplicate, trajectory
length and import count covariates fixed? It uses reaction-level bootstrap
intervals and is descriptive, not a causal estimate. This supplements—not
replaces—the primary paired accuracy contrast.
The label-free 27,104-row H2 test covariates give full rank 7/7 for the
intercept-plus-six-feature design matrix (condition number 69.6), 185 distinct
log-frequency values, and full rank in 30/30 seeded reaction bootstrap
resamples. This is a feasibility check of the predeclared model, **not** an
estimate of any model's accuracy or primitive-frequency effect.
