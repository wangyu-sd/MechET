# Forward data repair and explicitly requested smoke — October 16, 2026

The old task `meteor_mechet_forward_event_06b_2ep_8h20_zjk_20261013_01`
(instance `8b1d89caa134b07c01a13555c25400d2`) obtained eight H20s but
failed before optimizer training. Default logs show
`ValueError: non-contiguous source trajectory 282`; the platform retried the
same failure and ended the task on October 14 at 16:21 CST. Ceph and GPU
preflight passed. It must not be described as still queued or trained.

## Repairs

1. Group all raw occurrences of each reaction ID using byte-range indexing.
   Preserve repeated rows and original within-reaction order. No new filtering.
2. Construct history from graph predecessors, not raw line order. Initial
   real-data audit found 127 missing histories across 28/57 selected reactions.
   Keep original decision indices and one recorded branch at joins.
3. Normalize the current private mapped graph identically before forward
   prompt generation, action rendering and action execution. A genuine
   multi-step replay of validation ID925 exposed alias swapping between
   identical reagent copies after executor reserialization. Mapless comparison
   hid that error; mapped comparison caught it. Regression tests reproduce it.

Only forward behavior changed. The existing inverse serializer, executor
semantics, frozen reaction eligibility and protected inverse weights remain
unchanged. Explicit hydrogen nodes are retained.

## Verified data/software results

| Check | Result |
|---|---:|
| Complete validation reactions rebuilt | 2,890 / 2,890 |
| Original validation records retained and single-step replayed | 21,723 / 21,723 |
| Rebuild errors / missing reactions | 0 / 0 |
| Recorded terminal self-loop rows retained | 4,009 |
| Selected real reactions for closed-loop smoke | 57 |
| History followed by current-action mapped replay | 933 / 933 |
| Runtime versus training prompt equality | 933 / 933 |
| Qwen SFT versus inference prefix equality | 933 / 933 |
| Token truncation / zero-supervision rows in smoke | 0 / 0 |
| Maximum smoke input length / configured length | 4,555 / 16,384 |

Selection: first 32 validation IDs in first-occurrence order, all 17 IDs
with non-contiguous source blocks, and the eight largest validation groups
(overlaps counted once). It is a structural regression fixture, not a random
predictive evaluation. All original source lines and complete converted
conversations are saved. The final build+replay+token audit took 117.87 seconds
locally with 20 conversion workers and 16 replay workers.

Complete raw grouping was additionally checked for **all three splits**:

| Split | Official reaction IDs | Frozen eligible reaction IDs | Eligible original transition rows |
|---|---:|---:|---:|
| train | 257,171 | 257,167 | 1,902,076 |
| valid | 2,890 | 2,890 | 21,723 |
| test | 28,971 | 28,967 | 216,568 |

Only the existing named `RC/PC/PM/RS` corrupt source IDs are excluded in
train/test. Grouping checks are **not** complete train/test replay conversion.
Full forward train/test conversion and a replacement formal training job have
not been completed/submitted in this repair turn.

These are GT-program data/software checks, **not model endpoint accuracy**.
The additional GPU check uses an isolated original-base Qwen3-0.6B LoRA for
three numerical updates on validation fixtures. It must never be promoted to
a scientific checkpoint. Local T4 uses FP16; the approved formal H20 protocol
remains BF16, two epochs, global batch64.

The final GPU check passed on the local Tesla T4: finite losses
`0.968644 → 0.930237 → 0.876893`, finite nonzero gradient norms and nonzero
LoRA parameter updates at every step; peak allocated GPU memory 5,854,637,568
bytes. Protected inverse weights matched their pinned SHA before and after.
This same-fixture numerical loss decrease is not a generalization result.

Fresh repository suite: **945 passed, 13 skipped**, with one pre-existing PEFT
config warning in `test_paired_export_reloads_real_peft_adapter_and_rejects_corruption`.
The focused forward suite has **16 passing tests**. Independent read-only code
review found no blocking issues, including a 100-random-serialization
explicit-H symmetry check. No unrelated Taiji task was stopped or changed.

## Persistent artifacts and reproduction

Absolute root:
`/aaa/fionafyang/buddy1/whaleywang/MechET/outputs/analysis/forward_repair_smoke_20261016/`

- `raw_cases.jsonl`: complete original source strings for 57 reactions.
- `initial_history_audit.json`: failures of raw-order history construction.
- `closed_loop_initial_failure_925.json`: saved alias-swap failure details.
- `all_split_grouping_audit.json`: full-source split counts, exclusions, hashes.
- `canonical_state_v2/valid.jsonl`: final complete rebuilt validation view.
- `canonical_state_v2/smoke_rows.jsonl`: all 933 complete conversations.
- `canonical_state_v2/closed_loop_gt_history_audit.jsonl`: all per-row replay results.
- `canonical_state_v2/summary.json`: final data/software smoke summary.
- `canonical_state_v2/gpu_training_smoke.json`: numerical training evidence.
- `run_data_smoke.py`, `run_gpu_numerical_smoke.py`: reproducible saved drivers.

The earlier `fixed_graph_history/` directory predates the canonical-alias fix
and is historical diagnostic output only; use `canonical_state_v2/`.

From the corrected code checkout, run the saved drivers with:

```bash
env PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=/tmp/mechet_ray_runtime.K1QW8z:/tmp/pr83_rdkit_glibc235_20261012:.:src \
  /root/miniconda3/envs/meteor/bin/python \
  /aaa/fionafyang/buddy1/whaleywang/MechET/outputs/analysis/forward_repair_smoke_20261016/run_data_smoke.py
```

The temporary RDKit overlay path belongs to this local environment. Remote
formal jobs must continue installing the checksummed pinned RDKit wheel using
the established launcher, rather than assuming this `/tmp` path exists.
