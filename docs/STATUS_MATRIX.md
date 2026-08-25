# MechET status matrix

This page separates implemented infrastructure from scientific results under
the paper-authoritative A0--A7, B1--B5 and R1--R7 protocol.

| Layer | Status | Evidence currently available | Next gate |
|---|---|---|---|
| Deterministic proof executor | Implemented | Unit and workflow coverage | Full-data chemistry coverage audit |
| Trace-owned environment | Implemented | Scripted end-to-end CI | Real non-scripted model rollout |
| Proof-to-trace conversion | Implemented conservatively | Replay and quarantine tests | Full-data conversion and selection-bias report |
| Tool-SFT data contract | Implemented | Schema, mask, budget, and lineage checks | 32–128-example real Qwen overfit |
| Tool-SFT checkpoints | Pilot pending | P0 representation screens exist, but the final common-ID/token-matched set does not | Freeze and train the Tier-A matrix |
| A0 Direct | Trained; old inference stopped | Checkpoint retained | Reuse only if selected for the compact unified comparison |
| A1 Free-CoT | Missing | None | Build matched answer-bearing rationale rows |
| A2 State-CoT | Trained; old inference stopped | Checkpoint retained | No more compute unless selected for the compact comparison |
| A3 NetEdit | Trained; old inference stopped | Checkpoint retained | No more compute unless selected for the compact comparison |
| Independent complete-proof audit | Trained; old inference stopped | Checkpoint retained | Diagnostic only; do not relabel it A4 OpenFlow |
| A4 OpenFlow | Missing, P0 | Complete-proof is not an equivalent substitute | Implement execute-at-end electron-flow control |
| A5 Loose trace + answer | Missing | Legacy components exist but no paper-frozen run | Build matched bypass condition for R2/R3 |
| A6 MechSMILES-format | Missing | None | Build matched one-shot serialization condition |
| A7 MechET | Main action-only training active | 257,167 / 2,890 / 28,967 clean `action_delta_v1` rows; checkpoint 3,750/12,057 at epoch 0.93/3, loss 0.0408 | Finish training, then unified K=10 evaluation |
| B1--B5 ablations | Not run | Runtime components partially exist | Freeze matched implementations before R3/C2 |
| R3 state adaptation | Not established | Intervention foundations exist | Accurate-vs-stale counterfactual paired test |
| R4 compositional generalization | Not established | Split builder exists | Fresh C1/C2/C3 checkpoints and overlap audits |
| R5/R6/R7 | Not run | Supporting metrics partially exist | Data scaling, OOD/transfer and recovery analyses |
| H1 causal faithfulness result | Not established | Runtime and intervention infrastructure only | Complete paired non-scripted interventions |
| H2 compositional generalization result | Not established | Split and overlap infrastructure only | Complete non-empty C1/C2/C3 evaluation |
| H3 evidence benefit result | Not established | Future-study matched evidence infrastructure only | Excluded from current ICLR; run separately if resumed |
| Textbook/RAG/H3 | Future study | Matched evidence infrastructure exists | Excluded from current ICLR result matrix |
| RL and downstream planning | Optional extension | Configuration/adapter paths exist | Only after matched SFT, R3 and C2 are frozen |
| Experimental feasibility | Out of current scope | None | Independent experimental or trusted external evidence |

The authoritative run mapping is
[`PAPER_EXPERIMENT_PROTOCOL.md`](PAPER_EXPERIMENT_PROTOCOL.md). A green infrastructure row does not imply a positive scientific result.
