# 31k MCTS stop receipt, 2026-10-15

At the user's explicit request, stopped ONLY
`meteor_mechet_uspto31k_mcts_rolling_8a100_qy_20261015_01`, instance
`8b1d803aa13ad2de01a13d92657a055f`. CLI returned `stop success`; subsequent
`instance_detail` returned `END`.

The pre-stop POD read at 22:09:16 CST found 390 checkpoint files in each of
eight ranks, plus each rank's manifest and native journal directory. This is
evidence of saved artifacts, NOT proof that the old launcher supports resume.

| Rank | Finished / 390 | Completed rollouts | Fresh output tokens |
|---|---:|---:|---:|
| 0 | 30 | 1325 | 6902105 |
| 1 | 22 | 1317 | 6241899 |
| 2 | 16 | 1144 | 6032225 |
| 3 | 15 | 1062 | 5649836 |
| 4 | 9 | 890 | 4944691 |
| 5 | 13 | 1116 | 5438453 |
| 6 | 5 | 922 | 4722252 |
| 7 | 15 | 1092 | 5673212 |

Finished population: 125/3120 = 4.0064%. These rank snapshots are close in
time, not an atomic final stop barrier. No endpoint accuracy is inferred.
All old output under
`outputs/eval/mech_uspto31k_full_mcts_rolling_20261015_01` is retained.
Source remains frozen min-proxy DIAGNOSTIC; no product-field change was made.

Readback also confirmed both unrelated instances remain `TRAINING_RUNNING`:

- R1/R2: `8b1d89c4a13ad33601a13aee1c0b0052`.
- Joint-import-flow training: `8b1d8249a1250fb701a133bc40f81482`.

No replacement has been submitted at this receipt's creation. Code inspection
found `infer_single_reaction_mcts.py` rejects existing output and lacks a resume
entry; inherited route-controller restoration also does not match the single-
reaction constructor/checkpoint schema. Recovery must be implemented and tested
before a continuation is represented as resumable.
