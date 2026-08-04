# MechET status matrix

This page separates implemented infrastructure from scientific results that still require real-model experiments.

| Layer | Status | Evidence currently available | Next gate |
|---|---|---|---|
| Deterministic proof executor | Implemented | Unit and workflow coverage | Full-data chemistry coverage audit |
| Trace-owned environment | Implemented | Scripted end-to-end CI | Real non-scripted model rollout |
| Proof-to-trace conversion | Implemented conservatively | Replay and quarantine tests | Full-data conversion and selection-bias report |
| Tool-SFT data contract | Implemented | Schema, mask, budget, and lineage checks | 32–128-example real Qwen overfit |
| Tool-SFT checkpoints | Pilot pending | No paper-scale checkpoint is claimed | Multi-seed validation runs |
| GRPO refinement | Smoke pending | Configuration and lineage paths exist | Real optimizer-step smoke and stability audit |
| H1 causal faithfulness result | Not established | Evaluator and interventions exist | Paired CI, McNemar test, and multi-seed consistency |
| H2 compositional generalization result | Not established | Composition-disjoint split builder exists | Scaffold, reaction-center, family, and near-duplicate audit |
| H3 evidence benefit result | Not established | Matched evidence conditions exist | Paired CI, corrected tests, and multi-seed consistency |
| Downstream planning utility | Extension only | Planning adapter exists | Evaluate only after H1/H2 gates pass |
| Experimental feasibility | Out of current scope | None | Independent experimental or trusted external evidence |

The authoritative claim ladder remains in [`SCIENTIFIC_THESIS.md`](SCIENTIFIC_THESIS.md). A green infrastructure row does not imply a positive scientific result.
