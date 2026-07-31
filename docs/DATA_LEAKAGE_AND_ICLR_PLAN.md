# Data leakage audit and ICLR training plan

## Required source data

1. FlowER-derived MechET JSONL with stable row ids, mapped products, proof/state targets, and source split metadata.
2. USPTO-50K standard train/valid/test with mapped reaction SMILES and stable ids.
3. USPTO-MIT / USPTO-FULL reaction tables for secondary overlap audits.
4. Patent id, publication date, and patent-family metadata when recoverable. If absent, patent-family and temporal disjointness must be reported as unverifiable.
5. Optional external data: ORD or post-cutoff literature reactions for NMI-stage validation.

## Frozen audit levels

- `exact_full`: all reactant/reagent/product fragments after canonicalization.
- `exact_structural`: atom-contributing precursor fragments only.
- `product`: exact product identity.
- `scaffold`: product Bemis--Murcko scaffold.
- `reaction_center`: changed-bond/local-atom-role signature.
- `proof_composition`: map-label- and serialization-invariant proof composition.
- `patent`: source patent identifier when retained in both corpora.

The benchmark is frozen first. Conflicting training rows are removed and written to a quarantine JSONL; the test set is never filtered post hoc.

## Models and objectives

| Model | Output | Supervised objective | Post-training |
|---|---|---|---|
| Outcome-only | core precursor | assistant-token CE | none |
| State-CoT | state trajectory + core precursor | assistant-token CE | optional verifier RL |
| Net-edit | net graph edit + core precursor | assistant-token CE | none |
| MECH_PROOF-SFT | executable proof only | assistant-token CE | none |
| MECH_PROOF-RLVR | executable proof only | initialized from proof SFT | group-relative REINFORCE / RLOO-style RLVR |

The causal language-model loss for SFT is

`L_SFT = - sum_t 1[t is assistant] log p_theta(y_t | x, y_<t)`.

Proof RLVR samples a group of proofs from the current policy, executes each proof, computes deterministic rewards, normalizes rewards within the prompt group, and minimizes

`L_RLVR = - mean_i A_i * mean_t log p_theta(y_i,t | x, y_i,<t)`.

The current implementation is group-relative REINFORCE/RLOO-style, not clipped PPO. It should only be called GRPO-style unless old-policy ratios and clipping are added.

## Core endpoint

The primary endpoint is the atom-contributing structural precursor multiset. A reactant fragment is structural when it shares at least one atom map with the product; free solvents, catalysts, salts, and spectators are reported separately. Unmapped rows cannot support this split and must be flagged.

## Recommended order

1. Run exhaustive exact/product/scaffold/reaction-center/proof-composition audit.
2. Freeze benchmark hashes and normalization digest.
3. Quarantine conflicting FlowER training rows.
4. Match proof-only and state-annotated corpora by stable row id, then build all task variants from the intersection.
5. Train three SFT seeds per task.
6. Initialize proof RLVR from each corresponding proof-SFT seed.
7. Evaluate standard and clean benchmarks, MechComp-OOD, atom-map permutations, causal proof perturbations, data efficiency, and compute cost.
