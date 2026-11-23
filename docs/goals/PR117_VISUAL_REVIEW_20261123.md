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

## Electron-action ranking across 114 training updates

[Open the paired margin plot](figures/pr117_electron_history_credit_through114.pdf)
and [download the 34 paired measurements](figures/pr117_electron_history_credit_through114.csv).

Each line compares two executed FLOW actions from the same mapped input.
One branch failed, and the other closed in inventory. The 34 pairs
produce different mapped successors. The empty marker shows the centered
local coefficient of the closing action minus that of the failed action.
The filled marker includes the weighted outcome history, before PPO clipping.

The local term favors the eventual failed branch in 23 of 34 pairs,
including 16 of 25 full-route pairs. The history term reverses all
23 preferences. The complete training census contains 45 comparison
groups among 223,256 candidate groups. The registered independent evaluation
measures the route-level effect. [Complete original-receipt audit](https://github.com/wangyu-sd/MechET/pull/117#issuecomment-7088124445).

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

## Nine committed reactions and three untouched precursor obligations

[Open the full nine-reaction figure](figures/pr117_update111_nine_reaction_cutoff.pdf).
The final target of update 111 has 34 heavy atoms. It reaches the decision
limit after 112 model calls and nine reaction commits. Five terminal
occurrences match the frozen inventory. The three unresolved precursors
arise from reactions 7, 8 and 9 and receive zero model calls. In reaction
7, call 89 disconnects the carbon and chlorine that were already part of
the substrate. The carbon acquires formal charge −2 and the executor
records the fragment as a structural precursor at call 97.

## Ten discarded commits followed by a root-state cycle

[Open the branch ancestry and exact proton-transfer reversal](figures/pr117_update112_discarded_reaction_branches.pdf).
The final target of update 112 has 35 heavy atoms. Of its 102 physical
model calls, ten committed a reaction during exploration. Calls 11 and
40 independently produced the same first precursor. Four backtracks left
the returned route with zero retained reaction commits. At the root,
call 101 moved proton H37 from N25 to N10. Call 102 was rejected because
it would restore the exact mapped state reached by call 5.

## All six inventory closures from update 113

[Open the six complete route drawings and 123 original model calls](figures/pr117_update113_six_stock_closures.pdf).
The six training examples contain 15 committed reactions and 18 exact
inventory leaf occurrences. Three targets have at least 30 heavy atoms.
On the 30-heavy-atom reference-verified case, the generated route matches
the first two reactions of the stored reference exactly. Its second
reaction reaches a precursor already in inventory. Applying the same
inventory stopping rule to the stored three-reaction reference yields the
same two-reaction tree and terminal multiset. One other route reproduces
its reference's first disconnection. The 17-heavy-atom example retains
the R configuration at the mapped carbon in the alcohol precursor.

## Twelve committed reactions with four unresolved precursors

[Open all 12 reaction drawings and the 103-call decision timeline](figures/pr117_update115_twelve_reaction_frontier.pdf).
This 30-heavy-atom, five-ring training target reaches the decision limit.
The returned route has 12 committed reactions along one dependency path.
Seven leaf occurrences match exact inventory and four remain open.
The earlier open precursors appear at reactions 8 and 10, and the model
makes zero calls on either before the end of reaction 12. The final call
imports a fragment on the 12-heavy-atom precursor created at reaction 11.
Atom maps show that every open component shares heavy atoms with its
producing substrate. The 21- and 12-heavy-atom precursors are entirely
composed of those substrate heavy atoms.
