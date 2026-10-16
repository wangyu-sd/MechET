# Forward electron-event model: four-H20 progress

Snapshot: **2026-10-16 16:28 CST**. This records the independent forward
Qwen3-0.6B task, not a new inverse-model or RL result.

## Live task

| Item | Verified value |
|---|---|
| Task | `meteor_mechet_forward_event_06b_2ep_4h20_zjk_20261016_01` |
| Instance | `8b1d8921a13fde2a01a143c120e408c3` |
| Platform state | `TRAINING_RUNNING` |
| Actual resources | Four NVIDIA H20s; Ceph mounted |
| Application group | `AI4LSLab_KEMO_ZJK_H20`, ordinary/non-elastic |
| Image | `mirrors.tencent.com/whaleywang/metabo:taiji7` |
| Runtime | `887da22e3882c263ca1283085f9c5e4925112114` |
| Task configuration | `b81aa88` |
| Current phase | Full original-forward data conversion; train split in progress |
| Optimizer/checkpoint | Not started / no checkpoint at this snapshot |

[Taiji instance](http://taiji.woa.com/#/project-list/jizhi/task-inst-detail/8b1d8921a13fde2a01a143c120e408c3).
No unrelated task was stopped, resubmitted or modified.

## Data conversion

| Split | Frozen eligible reactions | Expected original decision records | Snapshot status |
|---|---:|---:|---|
| train | 257,167 | 1,902,076 | Building; partial output is not a sealed dataset |
| valid | 2,890 | 21,723 | Complete, all rows preserved, zero replay errors |
| test | 28,967 | 216,568 | Complete, all rows preserved, zero replay errors |

These are the named strict-executable reaction IDs, not unqualified FlowER
full. Official reaction denominators remain 257,171 / 2,890 / 28,971; only
the already-frozen corrupt `RC/PC/PM/RS` IDs are excluded in train/test. No
additional chemistry, overlap, deduplication or length filter was introduced.
The source is **original forward FlowER transitions**, not predicted inverse
paths. Sixty-four server-side CPU workers perform the build.

Completed output SHA-256:

- valid: `712206cab3a3e89a50d1e269867d8bdbacd6cd531abd8036b500e475241f8b2b`
- test: `5d54e73818fd076bb1f4eabaeaff893f676a5d106e3f39905ecf386545030b3a`

The validation SHA exactly matches the locally verified canonical-state repair.
Training remains disabled until all splits, counts, replay checks, source
hashes and token-cache checks pass. The remaining automatic pipeline is full
train conversion → four-rank tokenization → intact-conditioning audit → SFT.

## Training configuration

Pinned **original Qwen3-0.6B** base revision
`c1899de289a04d12100db370d81485cdf75e47ca`; fresh forward LoRA r16/alpha32,
BF16, two complete epochs, learning rate 2e-5, seed17. Global batch64 is
**4 GPUs × per-device batch4 × gradient accumulation4**. No max-step or
example-subset cap is applied. The inverse Stage-II adapter is neither a
trainable parent nor an output destination; its weights are hash-protected.

Current mixture and recorded predecessor actions are inputs; the next forward
electron event or `finish_forward` is supervised. The final intended product
does not enter the policy prompt. Shared inverse behavior is unchanged.

## Repairs and checks completed before submission

- Read every occurrence of a trajectory ID, including interleaved source
  blocks, without dropping repeated source rows.
- Derive history from actual graph predecessors rather than export order;
  retain one recorded path at branch joins and original decision indices.
- Normalize the current private mapped graph identically for forward prompts,
  action rendering and execution. This fixes alias swapping between identical
  reagent copies after SMILES reserialization while retaining explicit H.
- User-requested real-data smoke: 57 reactions, 933 decisions; all GT history
  plus current-step mapped replays and runtime/training prompt checks passed,
  with no token truncation. This is **software/data replay, not prediction accuracy**.
- Local isolated numerical LoRA check: three updates, loss
  0.968644 → 0.930237 → 0.876893; nonzero finite gradients/updates, protected
  inverse weights unchanged. The validation-fixture smoke adapter is forbidden
  for formal scientific use.
- Submission-time repository tests: **948 passed, 13 skipped**, with one
  pre-existing PEFT warning. Four-card launch parameters and shell syntax passed.

See [the full repair record](FORWARD_DATA_REPAIR_SMOKE_20261016.md) and
[forward protocol](../FORWARD_QWEN_EVENT_SFT.md). No forward predictive endpoint
or round-trip accuracy is claimed yet.

## Logs and retained artifacts

The launcher uses unbuffered default stdout/stderr and a 45-second heartbeat,
with no training-log file redirection. Worker stdout was verified to share
PID1's default pipe. At startup, the CLI log endpoint still exposed bootstrap
output rather than fresh worker progress: correct stream wiring does not by
itself prove UI/API freshness.

All paths below are under `/aaa/fionafyang/buddy1/whaleywang/MechET/`:

- Dataset: `data/flower_forward_qwen_event_v2_20261016/`
- Future model: `outputs/agent/forward_event_flowER_qwen3_0_6b_4h20_20261016_seed17/`
- Receipt/live checks: `outputs/submissions/forward_qwen_sft_4h20_20261016/`
- Full smoke inputs/outputs: `outputs/analysis/forward_repair_smoke_20261016/canonical_state_v2/`
- Raw source-case strings: `outputs/analysis/forward_repair_smoke_20261016/raw_cases.jsonl`

The failed October13 eight-H20 task and its partial v1 data remain historical;
they are not the replacement's outputs. Large datasets, weights, private Ceph
init commands and raw process argument dumps are not uploaded with this report.
