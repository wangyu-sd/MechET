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

## Eleven-reaction route stopped by an ancestor cycle

The first page shows all 11 committed reactions. The second identifies the
two structures involved and lists all 95 physical model calls. Call 89
commits the precursor with a methyl ester. Call 94 attempts to return to
the immediate parent carboxylic acid and the executor rejects it. The final
slate's other output repeats the same FINISH. Five leaf occurrences match
stock and one precursor remains open. [Open the two-page replay](figures/pr117_eleven_reaction_ancestor_cycle.pdf).

## One native electronic action changes the fourth reaction

On a 32-heavy-atom target, the controller first selects a fragment import
from a four-proposal group. The import lineage makes three commits, then
runs into a missing source bond or an electronic state cycle. Restoring the
saved FLOW moves a hydrogen from carbon to oxygen in the original electronic
state. The resulting fourth reaction yields two exact-stock precursors.
Both diagrams preserve the original call numbers and the second page lists
all 70 physical calls. [Open the chemical fork and trace](figures/pr117_update107_import_or_flow.pdf).

## Training closure through update 110

[Open the updated inventory closure curve](figures/pr117_training_closure_through110.pdf).
In updates 101–110, 44 of 250 full-route targets requiring synthesis reached
inventory, compared with 45 of 247 in updates 91–100. Different targets enter
each window. Across all 110 updates, 465 of 2,707 such targets reached
inventory.

The 101–110 window contains 195 executor rejections and 11 budget stops.
The most common terminal rejection is a return to a route ancestor (75),
followed by a missing source or destination bond (64) and a finish that
produces no usable precursor change (36). The final candidate group contains
a duplicate in 175 of the 195 rejected routes. This statistic covers the
last group of rejected routes, with original candidate identities preserved.

## Three fragment imports from the same model input

[Open the three-page replay and full call timeline](figures/pr117_update108_three_same_input_samples.pdf).
On the 32-heavy-atom training target in update 108, calls 19, 41 and 57
start at the same atom-mapped state and receive identical model input. All
three generate usable imports and later commit the same precursor for
reaction 3. The first two continuations stop on absent bonds during
reaction 4. The continuation from call 57 supplies the fourth reaction
and returns four exact inventory matches, including hydroxide listed
in the pinned inventory. The second page shows the retained reaction
structures and the third lists every original call.
[Source trace](https://github.com/wangyu-sd/MechET/pull/117#issuecomment-7087055615)

[Download the 74-call decision table](figures/pr117_update108_decisions.tsv).
It records each action's original arguments, duplicate or executor outcome,
mapped states where executed, and submitted precursor sets.

## Observed credit across four electronic actions

[Open the same-state electron-flow comparison](figures/pr117_update111_electron_flow_credit.pdf).
At update 111, four actions produced different executable structures from
the same mapped input. One continuation was unexplored. Both failed
continuations had higher local coefficients than the action that reached
the two inventory precursors. The history term gives the closing action
+0.333, gives each observed failure −0.167, and gives the unexplored
continuation zero. The figure shows the exact organic component of each
successor. This is a replayed one-reaction training example.
