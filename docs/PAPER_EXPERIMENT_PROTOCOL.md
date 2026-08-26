# Paper-authoritative experiment protocol

> [!CAUTION]
> FlowER `3,080`, deprecated mech-USPTO-31k `1,124`, and current-compiler
> mech-USPTO-31k `1,253` denote **INCOMPLETE TRACE-VIEW SUBSETS**, never full
> benchmark tests. Headline denominators are respectively
> `28,971` and `3,120`.

> **Authority:** [`wangyu-sd/MechET-paper`](https://github.com/wangyu-sd/MechET-paper),
> revision `47b0166174f0ab921c01d11f83cbf66db8659177` (2026-08-20).
> Within that repository, `EXPERIMENT_MATRIX.md` controls matched experiments
> and run priority; manuscript tables control presentation of external methods.

This document maps the ICLR paper protocol onto this implementation repository.
If an older MechET document conflicts with this page, the paper protocol wins.

## Scientific evidence chain

| ID | Experiment | Required conclusion |
|---|---|---|
| R1 | Overall inverse performance on FlowER-derived and mech-USPTO corpora | Endpoint competitiveness plus formal execution |
| R2 | Reasoning provenance and endpoint consistency | Correct endpoints are not paired with invalid or incompatible programs |
| R3 | Counterfactual execution-feedback adaptation | The policy responds to environment acceptance/failure; molecular-state visibility is measured separately |
| R4 | MechComp-OOD C1/C2/C3 | Familiar primitives and local motifs compose into unseen complete programs |
| R5 | Data efficiency | Local operators improve low-data and rare-program generalization |
| R6 | Structural OOD and cross-corpus transfer | Gains persist beyond IID products and source-corpus patterns |
| R7 | Theory-linked analysis and recovery | Sparsity, branching, novelty and recovery explain where MechET helps |

C2 and the accurate-vs-stale counterfactual comparison are the central
compositional and closed-loop tests. IID Top-k alone cannot establish either
claim.

## Mandatory matched Qwen3-8B controls

The final Tier-A table requires the identical frozen stable-ID universe,
immutable base revision, optimizer family, matched supervised-token budget,
candidate count and inference budget. Same epoch count is not token matching:
Direct and trace/program targets have very different output lengths.

| ID | Paper name | Endpoint ownership | Current implementation status |
|---|---|---|---|
| A0 | Direct | independent answer | Full FlowER screen trained; full-test inference running |
| A1 | Free-CoT | independent answer after free rationale | Missing |
| A2 | State-CoT | independent answer after explicit states | Full FlowER screen trained; batched full-test inference running |
| A3 | NetEdit | independent answer after sparse edits | Full strict-proof screen trained; EOS-fixed batched full-test inference running; pre-fix partial output is diagnostic-only |
| A4 | OpenFlow | complete electron-flow program executed only at end | Missing; P0 |
| A5 | Loose trace + answer | environment interaction plus bypass answer | Missing; required for R2/R3 |
| A6 | MechSMILES-format | one-shot inverse-arrow serialization | Missing |
| A7 | MechET | endpoint derived only from committed online trace; no intermediate state in the main representation | Action-only strict data frozen; real 8×A100 gradient training active after a zero-truncation 257,167-window cache audit; unused H20 candidate stopped; legacy full-state run is not the main result |

The currently active coverage jobs are P0 screens, not yet the final matched
Tier-A evidence. A0/A2 contain all 257,171 FlowER train reactions, whereas A3,
complete-proof and executable A7 contain the 257,167 strict-proof reactions;
their assistant-token totals also differ. Before the headline comparison, freeze one
common strict eligible-ID universe (257,167 / 2,890 / 28,967 at the current v4
lineage), rebuild every eligible A0--A7 condition on those IDs, and normalize
or explicitly match the supervised-token/update budget. The separate full
28,971 test report remains valuable as a coverage result.

### Frozen P0 decoding budget

The current A0/A2/A3/complete-proof screens use K=10 stochastic decoding at
temperature 0.7 and top-p 0.95. Maximum generation lengths were selected from
the validation split only, with headroom above the longest reference: A0 1,024,
A2 14,336, A3 2,048 and complete-proof 2,048 tokens. Batched generation keeps
the first EOS token and removes only right-padding after it; those padding
tokens are never treated as model output or passed into NLL ranking. On 40GB
A100s the frozen launcher uses two model workers per GPU and candidate
microbatch two. These are throughput settings for the P0 screen and must be
held fixed across any rerun being compared in one table.

Independent complete-proof generation remains a useful R2 audit condition,
but it is not a substitute for A4 OpenFlow: A4 must use the same electron-flow
action coordinates as MechET and differ primarily in online execution feedback.

### A7 observation-granularity screen

The A7 main condition is `action_delta_v1`. The model sees the product once,
its own explicit actions and environment success/failure codes. Complete mapped
intermediate states and trace digests remain private executor audit data until
the terminal audit result.
Two matched representation screens reuse the exact same IDs and action targets:
`reaction_center_delta_v1` exposes only the changed one-hop mapped neighbourhood,
and `full_state_v1` exposes the legacy complete state. These are ablations, not
extra requirements imposed on the main method.

Compare the three variants using Structural Pass@K, ExecutePass, TraceBoundPass,
invalid-action rate, post-failure recovery, input/supervised tokens, examples per
second and rollout latency. A second, length-matched redaction control is required
before attributing a gain to state information rather than extra context length.
Natural-length efficiency and length-matched causal results must be reported
separately.

## Mandatory component ablations

- B1: remove source/sink enumeration while retaining action-outcome feedback;
- B2: execute the same electron-flow program only at the end;
- B3: execute the true transition but expose stale action-outcome feedback;
- B4: restore an independent answer channel;
- B5: give an answer model the same legal-action summary.

The ablation table reports IID and C2 Structural Pass@1, ExecutePass, SAE and
CEIR. This separates sparse representation, electron-flow semantics,
deterministic execution, online feedback and endpoint ownership.

## External comparisons

The matched-priority implementations are Inverse-FlowER, ReactSeq, EditRetro
and RetroBridge. LocalRetro, R-SMILES, Retro-MTGR, RETRO SYNFLOW, RetroDFM-R
and RetroReasoner provide wider field context. Results from original recipes
or external checkpoints must be shown separately from identical-row retraining
and cannot identify causal effects of MechET components.

## Dataset and result tracks

- The current full FlowER reaction split is 257,171 / 2,890 / 28,971.
- The completed 3,080-target result is a frozen historical same-source,
  executable-subset result, not the full benchmark denominator.
- mech-USPTO-31k contains 31,199 raw reactions and 12,724 current-compiler
  replay-verified inverse traces (10,152 / 1,319 / 1,253); retained-versus-rejected chemistry must be audited. The old 11,429-row artifact is deprecated.
- All final comparisons retain missing predictions as failures and freeze test
  IDs before model selection.

### mech-USPTO reaction-mapping source

The Hugging Face `rxn_prod_min` field is an unmapped mechanism-final mixture and
must never simply be copied into `product_mapped`. The active reproducible
protocol reconstructs the complete precursor from step-0 `elem_reac_spe`,
selects the deterministic principal organic product from `rxn_prod_min`, maps
the pair once with pinned RXNMapper, and then product-only canonically
reindexes both sides. All external methods share this frozen mapping. The
builder validates complete cross-side product mapping and preserves all
24,959 / 3,120 / 3,120 reaction IDs without executor filtering.

## Metrics and compute matching

Report Structural and Full-precursor Pass@1/5/10, mapped exact, auxiliary
precision/recall, ExecutePass, TraceBoundPass, endpoint-program consistency,
CEIR, SAE and frozen round-trip plausibility where applicable. Also record
supervised tokens, optimizer steps, GPU-hours, candidates per target, generated
tokens, environment transitions, wall time and peak memory.

### Frozen interactive-inference contract

An interactive A7 result is valid only when inference reuses the frozen
training system prompt and tool schema and matches the observation mode,
tool-call budget and outer iteration budget exactly. Candidate count is part of
the denominator: missing, OOM and generation-error candidates are retained as
explicit failures. Each result bundle records first-candidate and
gold-independent NLL-selected mapped, structural map-free and neutralized
metrics, together with per-candidate completion status and cumulative/mean
assistant NLL. It also freezes adapter, dataset, compiler, executor, prompt
contract and inference-config hashes/revisions.

The pre-issue-36 mech-USPTO current-compiler rollout violated this contract
(12-call training prompt versus 40-call runtime, regenerated prompt, and two
workers per 40-GiB A100). Its approximately 1.60% Pass@10 is diagnostic-only
and is not a paper comparison.

## Explicitly outside this ICLR protocol

Textbook/RAG/H3 and multistep route planning are separate future studies.
On-policy RL and model-size scaling are optional only after the matched SFT,
counterfactual and C2 results are frozen.
