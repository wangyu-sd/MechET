# Issue #79: NMI compositional-generalization experiment record

## Step 0 audit (2026-10-02, before experiment-code changes)

| Item | Code / submission / completion / scientific evidence |
|---|---|
| PRs #70–#78 | All merged on `main` (`a4f01fc`); code merge alone is not an experimental result. |
| Protocol-v2 parity | `docs/NATURAL_LANGUAGE_PROTOCOL_V2.md` records 2,890/2,890 validation gold replays and 22,341/22,341 byte-exact decision prompts; the machine-readable audit is `docs/audits/natural_language_protocol_v2_valid_20260918.json`. |
| State-SFT / Trajectory-SFT | Both full strict-universe adapter manifests exist under `outputs/agent/natural_language_event_v2_qwen3_8b_h20_seed17_20260918` and `outputs/agent/natural_language_event_history_v2_qwen3_8b_h20_seed17_20260918`; train SHA-256 values are `22729aff36db3691b3db544aa83cb93c7383e9fbacda6934e6a52a3d690f6bfa` and `bc0a859a3c25b6146a8f6cac730610a0d66347baff1f9648b67329b17d785b97`. These full-data checkpoints cannot serve as headline H2 controls after a new split. Frozen H=1/2/3/full and EARHO outcome reports still require provenance-level inspection before reuse. |
| Pointer | 16 product-start cases completed on Taiji; baseline and coupled pointer each have Top-1=0/16 and Pass@beam=0/16. Predeclared smoke gate stopped 256 and 1,319 stages. The 44 invalid handles count rejected actor proposals, not executed invalid actions. |
| Runtime | Five-mode oracle-state benchmark is not complete. The PR #78 recovery reached Taiji twice but its ledger ended `INFRA_FAILED`. POD traceback identifies a **user-program compatibility error**, `Unsupported option for the xgrammar backend: no-fallback`, in pinned vLLM 0.8.5 V1 engine initialization. The controller's `INFRA_FAILED` label is not a scientific result or proof of platform failure. This runtime arm is not necessary for E1/E2 CPU audits and must not be silently retried with changed benchmark gates. |
| Packing | Completed report: loss absolute difference 0.00146447, throughput speedup 0.86350, peak-memory ratio 1.28073. The speed gate failed: `SCIENTIFIC_STOP`, no further packing optimization for #79. |
| MechComp C1/C2/C3 and H2 | Split builder, overlap audit and H2 runner exist, but `data/ood/mechcomp_source_sink` and `outputs/h2` were absent at this audit. No valid matched split/checkpoint/result is established. |
| PMechDB / PMechRP | No authorized source assets were found in the workspace. E5 is access-blocked, not a reason to stall E1–E4; no mirror substitution or new download is authorized. |

The E1/E2 source candidate is the frozen strict-executable FlowER **training pool** `data/flower_inverse_tool_sft_action_delta_v1/train.jsonl`, 257,167 reaction rows with SHA-256 `edc80c5c5eb13d50753c5566c5d6ac1b90b955b1a8dfece915241b2da4a40a75` in its training manifest. This is not the unqualified FlowER-full official split and does not make an old full-data-trained adapter eligible for the new held-out condition. The older documented `data/knowledge_ablation/v2/train/trace_no_knowledge.jsonl` input is absent locally. No final held-out model test will be inspected for model selection.

## E1 — local-operator basis audit

The full-source streaming audit is at
`outputs/issue79/nmi_operator_basis_train_20261002.json` (large artifacts stay
outside Git). It verified the 257,167-row source SHA-256 above and independently
recomputed the first 32 stored signatures. There are **604 distinct local
execution primitives** and **6,553 distinct ordered move-composition
signatures**, across 1,043,352 execution steps and 2,111,100 electron moves.
Of the primitives, 53 occur in one reaction, while 435 occur in at least ten;
2,728 composition classes occur in one reaction. At random hash samples of
1%, 10%, 50%, and 100% of reactions, primitive vocabulary size was 321, 458,
562, and 604; composition vocabulary size was 640, 2,231, 4,836, and 6,553.

A **non-frozen exploratory** 10% reaction hash probe held out 25,738 reactions:
only 317 had an unseen complete move composition, and 316 of these used only
seen primitives. This is 1.23% of the random held-out reactions. Thus a random
split would mostly retest familiar programs; E2 must hold out composition
groups. The v1 composition digest does not separately normalize the schedule
of fragment imports; it must be described as an ordered **move composition**,
not a unique physical mechanism or reagent program.

## E2 — candidate composition-group split

`outputs/issue79/nmi_mechcomp_split_20261002/manifest.json` records seed 42,
test/valid target fractions 0.10/0.10 and minimum train primitive frequency 5.
Rows with the same v1 move-composition signature or the same canonical unmapped
product + structural-precursor reaction are joined into one component before
selection. The resulting **candidate** ID sets contain train 223,863,
validation 6,200, and held-out test 27,104 reactions; SHA-256 of their ordered
ID files is respectively `a9cd229116809179be7a5e3708e52a3a0f115e4ea8fb34418d44f8d5608eddde`,
`53f8acec4a41929a702741f7f25babe90ac814c2002e0980173a843d30df8747`,
and `ec76eb0e972ab449e1d86d9e76a692682e561cbc2202217c73db55481879fc6d`.
There is zero train/test composition overlap and zero exact structural reaction
overlap; all held-out local primitives remain in train at frequency ≥5.

The largest composition/reaction connected component has 208,178 reactions;
only 6,200 validation reactions could be selected from the remaining feasible
components. This is a property of the grouping constraint, **not** the official
FlowER validation size and not a reason to silently change the split. The
revised local-center structural audit has completed. A separate
negative-control scan of the official 2,890-row strict
validation set found **zero** reactions containing primitives unseen in the
candidate H2 train split. A primitive-unseen control therefore cannot be
formed from that set without changing the training split; it is marked
unavailable rather than relabeling familiar-primitive reactions.

The first structural smoke revealed an inherited audit implementation defect:
RDKit's default SMILES parser removes mapped explicit hydrogen, and the old
center extractor ignores `BE_DELTA` bond/charge atoms. The #79 audit uses a
separately versioned `step_state_plus_edge_imports_v3_explicit_h` center while
retaining the old v2 behavior for historical results. On the fixed 1,000-row
diagnostic prefix, previously undefined center keys fell from 92/113 held-out
rows to 0/113. The first complete audit (`nmi_structural_full_20261002.json`)
verified zero exact-reaction overlap and zero undefined centers. On 27,104
candidate test reactions, 17,704 (65.3%) share a Murcko scaffold with train
and 1,328 (4.90%) have a product Morgan/Tanimoto ≥0.90 train neighbor. On
6,200 validation reactions, the corresponding counts are 3,328 (53.7%) and
23 (0.37%). These are overlap diagnostics, **not model outcomes**.

That first audit also reported zero `reaction_center_seen` in both held-out
strata. This key includes the *whole ordered move-bearing context* and is
nearly tautological under a complete-composition-disjoint split. It must not
be used as an independent structural control. The corrected v2 audit adds
per-step, map-free radius-one local structural center contexts **without move
labels or step order**, measuring the fraction of a held-out reaction's local
contexts seen in training. The v2 full audit
(`outputs/issue79/nmi_structural_full_v2_20261002.json`, SHA-256
`1e75ceb8b766280212cd32e46c7f112275fa4cdc632dd955c4711d508efc1192`)
passed both zero-exact-reaction gates and has zero undefined centers. On the
27,104 H2 test reactions, 24,213 (89.3%) have at least one local structural
center seen in train; 9,994 (36.9%) have all such centers seen. On H2 valid,
5,573/6,200 (89.9%) have at least one; none has all. The old ordered-center
key remains zero by construction and is never used as a structural control.

## E3–E5 results

E3 matched-data preparation found that filtering the historical reaction-level
Direct dataset by the H2 IDs is **not** a valid matched control. On the
223,863-row H2 train split, 36,119 mapped products and 175,403 structural
precursors disagreed byte-for-byte with the strict-trace Open-Flow records;
examples include explicit mapped hydrogen and different precursor-world
fragments. That Direct materialization is quarantined at
`data/issue79/nmi_matched_v1/direct_old_endpoint_mismatch_QUARANTINED_20261003`
and is not eligible for H2 training or reporting. The corrected Direct
materializer makes an outcome-only target from each **same frozen strict-trace
row**, removing tool and proof supervision from the model-visible conversation.
The exact product/structural-endpoint parity gate has now passed on **all**
223,863 train, 6,200 validation, and 27,104 held-out test IDs across all
three conditions. The frozen verification is
`data/issue79/nmi_matched_v1/verification.json` (SHA-256
`6da760b689b9d054e61944455b7cdf981301e9d3f9ab07722dd48a7e69550da1`).
The three equal-update Qwen3-8B configs are frozen at
`data/issue79/nmi_matched_v1/configs/`: seed 17, global batch 64, 10,494
optimizer updates each. This does **not** equalize token/computation budgets;
token counts and wall time must be reported separately. Three-row training
schema checks passed for each representation.

No H2 model performance result is available yet. Every result must be tied to frozen source/split hashes and its own
model checkpoint lineage; historical full-data results remain supporting
evidence only.
