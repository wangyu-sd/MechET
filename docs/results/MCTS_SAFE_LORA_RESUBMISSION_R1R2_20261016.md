# Safe-expand continuation and live R1/R2 snapshot

## 31k replacement submission

At2026-10-16 13:22:07 CST exactly one ordinary8xA100 task was submitted:

- Task:`meteor_mechet_uspto31k_mcts_ray_resume_8a100_qy_20261016_04`.
- Instance:`8b1d89f5a13f013001a143290e8a0a56`.
- Runtime:`337cba2fb0a445a747cce0d2f4713769d17f7324`; configuration:`7040406`.
- Group:`AI4LSLab_KEMO_QY_A100`,location`qy`,non-elastic.
- Image:`mirrors.tencent.com/whaleywang/metabo:taiji6`.
- Successful donor:`meteor_mechet_pr83_flower_valid128_singlepass_8a100_qy_20261012_01`.
- Proven private Ceph init,unbuffered default POD stdout/stderr,45-second heartbeat.

Initial readback is **TRAINING_RESOURCE_WAITING**, with scheduler message
`底层资源不足，请稍侯重试或排队等待。` No POD/model execution or production BF16
stability has been verified yet. Correct group/location and quota headroom do
not establish a full eligible eight-GPU host. Submission is not recovery success.

The new output is `outputs/eval/mech_uspto31k_full_mcts_ray_20261016_04`.
It resumes immutable failed `_03` plus compatible ancestors and reopens the
same migration03 Ray WAL. Full3120 saved-inventory verification passed:
1711 reports and1409 pending checkpoints,all original IDs and eight390-ID
assignments preserved. This inventory audit does not claim full native replay
or recovery of uncommitted GPU work. Source SHA,adapter SHA,cache namespace,
min-proxy DIAGNOSTIC condition,PUCT and ten-total-attempts stopping are unchanged.
No model smoke,new training,source-field change or unrelated cancellation.

Fresh whole-project verification before submission:939passed,13skipped,
one existing PEFT warning,84.08seconds. For the expand bypass limitations and
numerical verification,see `MCTS_LORA_EXPAND_FAILURE_20261016.md`.

Durable receipts under main checkout:
`outputs/submissions/uspto31k_mcts_ray_resume_20261016_04/`.
Private rendered configuration remains outside Git,mode0600.

## R1/R2 observed results at13:22 CST

The existing `_07` R1/R2 task remains TRAINING_RUNNING and is not modified.
Snapshot reads the actual per-target evaluation JSON files,not old aggregate
reports (those still contain only the completed greedy baseline).

| Cohort | Full denominator | Completed greedy stock closure | MCTS nonzero stock closure so far |
|---|---:|---:|---:|
|URSA Expert|100|1(1.0%)|18(18.0%)|
|URSA Drugs/Clinicals|100|15(15.0%)|39(39.0%)|
|RetroCast mkt-cnv|160|17(10.625%)|88(55.0%)|
|PaRoutes-derived n5|100|3(3.0%)|19(19.0%)|
|Total|460|36(7.83%)|164(35.65%)|

One separate RetroCast target is already in stock; exclude it from nonzero
route-success numerators.165/460 roots have finished (35.87% completion),
295 remain under search. All460 were retained and initialized; these MCTS
rates are current observed lower bounds,not final success rates. The larger
search work and shared-cache reuse prevent an equal-compute improvement claim.
Stock closure plus executable programs is not independent chemical verification.

R2 descriptive strata use the frozen **source reference depth**,not generated
route length,with40 targets per bin:

|Reference depth|Greedy stock closure|MCTS stock closure so far|
|---|---:|---:|
|2|6/40(15.0%)|27/40(67.5%)|
|3|5/40(12.5%)|26/40(65.0%)|
|4|3/40(7.5%)|18/40(45.0%)|
|5|3/40(7.5%)|17/40(42.5%)|

These are not final R2 mechanism-composition C1/C2 results: source mechanism
annotations remain incomplete. No unsupported composition-generalization claim.

Full source-bound snapshot:
`outputs/analysis/r1_r2_mcts_snapshot_20261016/snapshot_1792128167.json`.
