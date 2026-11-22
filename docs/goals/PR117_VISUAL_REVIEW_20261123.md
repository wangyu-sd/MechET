# PR117 visual review

[Open the ten-page PDF](figures/pr117_visual_review.pdf).

| Pages | Figure | Evidence |
| --- | --- | --- |
| 1–2 | ReFIT electronic-state search and reaction-level AND/OR search | Both searched the same 37-heavy-atom development target with the same actor and stock, at 30 model calls each. The returned routes retain four and one reactions, respectively. Both frontiers remain open. [Trace audit](https://github.com/wangyu-sd/MechET/pull/117#issuecomment-7078015398) |
| 3–5 | Recovery of a six-reaction route | On a 33-heavy-atom training target, search resumes from the saved FLOW 31 successor after the initial branch reaches a cycle. The figure shows the original mapped action pair, all 99 model calls, and six stock leaves. [Trace audit](https://github.com/wangyu-sd/MechET/pull/117#issuecomment-7084200495) |
| 6–8 | Three complex stock-closed training routes | A 43-heavy-atom polycycle and two stereodefined targets with 39 and 34 heavy atoms. [Replayed routes](https://github.com/wangyu-sd/MechET/pull/117#issuecomment-7078437853) |
| 9 | Inventory closure across 100 training updates | 421 of 2,457 full-route targets requiring synthesis reach stock. The ten disjoint windows range from 12.9% to 21.2%. [Counts and diagnoses](https://github.com/wangyu-sd/MechET/pull/117#issuecomment-7084572384) |
| 10 | Electron-action credit from observed outcomes | Of 30 distinct mapped FLOW pairs, 20 local margins favor the eventually failed action. The observed route-history term reverses all 20 margins before PPO clipping. [Learning-record audit](https://github.com/wangyu-sd/MechET/pull/117#issuecomment-7084826839) |

The prospective Reference and CbyD evaluations use their registered protocols. These figures document archived development and training evidence.
