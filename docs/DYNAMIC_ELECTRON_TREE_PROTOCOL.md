# Frozen Qwen dynamic electronic tree, October13 2026

User-approved scope: first evaluate all3120 mech-USPTO-31k test products,
submit exactly one8xA100 task (queued is a successful submission), then use
the same frozen FlowER actor for the R1/R2 complete-route scientific scenarios.
**RSI, RL and self-distillation are paused/not authorized.** Actual actor is
Qwen3-0.6B, not the conversational typo0.8B; user confirmed this checkpoint.

## Frozen model and full population

FlowER Stage-II compressed-history LoRA weights SHA256
`40204c7a059d16976fc4b24b315c3bdde7bf61c705d24bbde515071757e9de75`,
original Qwen revision `c1899de289a04d12100db370d81485cdf75e47ca`.
This is transfer to mech-USPTO, not the separate31k eight-epoch State-SFT actor.
Test source is `data/mech_uspto_31k_full_endpoint_rxnmapper/test.jsonl`,
SHA256 `819215c5d40720afd998aaf77b7b938045d8171a5487b26bf929897f65a957c4`.
All3120 unique IDs remain denominator. Only unmapped product enters the policy;
private atom addresses come from that product, not reaction-pair atom mapping.
No reference proof requirement,1253-row filter, new exclusion or test training.

## Adaptive search, not fixed truncation

No fixed chemical depth, beam width or max-import count. Every accepted
successor is retained; parents can be resampled and deferred contexts remain
available. Chemical nodes are pooled for graph display/accounting, while
private state, import inventory, accepted-action history and visited-state set
identify independent future policy contexts. Reaching one FINISH does not
stop other contexts. Duplicate executable payloads reuse transitions but every
original raw sample, likelihood and sampling cost is saved.
The context/cache key preserves the exact authoritative native Kekule string;
canonical equivalent molecules are merged only for display. All terminal
contexts survive until product-atom ancestry is used to extract structural
precursors, after which structural candidates are deduplicated for ranking.

A Good--Turing-style singleton mass plus3/N sampling margin controls sampling
effort. This is a **novelty heuristic**, not a proof of exhaustive chemistry.
Lower-mean-NLL contexts and unsaturated support get more inference effort.
The controller shrinks/expands request microbatches rather than imposing a
chemical branch cutoff. Incomplete tool calls get more per-call token room;
they never turn into fabricated valid actions.

The October14 current runner treats80million output tokens per condition and
24hours per worker/condition as INITIAL ACCOUNTING CHECKPOINTS, not stopping
caps. Crossing a checkpoint with active frontiers automatically extends the
next work window and logs the observed cost/decision. Only observed policy
support exhaustion, actual model context8192, and memory/allocation protection
close exploration. This is not a completeness guarantee: unbounded compute is
not promised and the sampling novelty rule remains a heuristic. A fair root queue prevents
one target consuming all inference before other test products are attempted.
Resource-limited/invalid/nonterminal roots stay in3120. They are not forced to
emit FINISH or dropped from evaluation.
Each worker reserves40% of its initial share of available host/container RAM
for persistence and other processes; driver growth uses at most60%/8 of that
available RAM. Cgroup limits are included. RAM closure saves the whole pending
cohort with `TASK_HOST_MEMORY_BUDGET`; it is not a chemical state-count cap.
JSON is streamed to avoid allocating a second complete serialized tree.

## Throughput and likelihoods

Eight independent TP1 vLLM0.8.5 engines; cross-molecule prompt microbatches,
prefix caching and64 CPU threads total for node execution. Do not reuse the
old serial selected-case demonstration as a full runner.
vLLM0.8.5 logprobs are post-warp. Use sampling temperature1/top_p1/top_k-1,
no penalties, for genuine unwarped action likelihoods; greedy uses temperature0.
Rank unique structural terminals by accumulated action-token mean NLL;
GT matching occurs only AFTER search. Preserve all terminal mixtures and
zero-change results, rather than hiding them with a reference-dependent gate.

Per worker save `policy_contexts.jsonl`, `raw_generations.jsonl`, every full
chemical/context tree and per-ID ranked candidates/errors/work. No raw prompt
or output is discarded. Aggregation rejects lost/extra/duplicate source IDs.
Greedy and search outputs are separate conditions; higher raw search accuracy
is not called a compute-matched gain without accounting for their work.
The single-trajectory greedy control reports Top1 only; Top5/Top10 are null.
Partial shards are preserved and require a new attempt directory, not an
overwrite or an unverified resume. Completed shards may be reused only within
the same frozen run.

## Complete-route follow-on, not self-improvement

Use the user's cited paper R1/R2 protocol: URSA Expert100+Drugs/Clinicals100,
publication-frozen RetroCast mkt-cnv160/buyables-stock, explicitly named
source-fixed PaRoutes-derived100/n5-stock. Preserve dataset/stock versions,
overlap, zero-step in-stock targets, full AND leaf closure, source reference
route complexity, and missing electronic annotations as Unknown.
Single-reaction31k outcomes alone do not establish stock-connected route success.
No URSA/ChemCensor proprietary scorer is downloaded/integrated. Source maps
are not invented electron ground truth. No new model/reward/critic is introduced.

## R1/R2 runnable implementation and review fixes, October14

Source declarations: `configs/eval/frozen_route_benchmarks_20261013.json`.
All460 IDs (100/100/160/100) enter ONE shared scheduler per condition, not
four independent work-window resets. The80M output-token initial checkpoint
is shared across all460, and each worker starts with one-eighth of that token
checkpoint;24h is an initial elapsed-work checkpoint. Neither truncates active
frontiers. Stocks differ by cohort; observations remain product-only. Remote
CPU stock canonicalization precedes all model generations. Seven raw source
checksums and all original target/reference populations have been verified.
The mkt-cnv primary references have2/3/4/5 depths×40; its407 accepted reference
variants contain1627 edges. PaRoutes-derived100 has387 edges; three targets
overlap mkt-cnv. These are not independent cohorts.

`reverse_route_network.RouteNetwork` recursively starts the same ElectronTree
on each nonstock structural precursor. Full AND closure is a least fixed point;
cycles cannot create stock solutions. Already-stock targets are separate
zero-step cases, not nontrivial generated routes. All inner graphs and terminal
programs remain saved. Whole purchased salt entries are preserved; otherwise
connected components are obligations, with no invented ion availability,
charge-neutralization or tautomer fallback.

Combined-shard source/stock/model/revision/runtime/seed/budget identities are
checked before aggregation or completed-result reuse. V0 TP1 scheduled model
executor calls are measured separately from API batches. Per-target participated
batches must NOT be summed as unique global calls; complete-shard counters
provide the latter. JSON stays in Ceph; default POD stdout remains live.

The next instrumented runtime also writes `work_waves.jsonl` after each FULL
generation/executor wave: unique engine-global scheduled forward calls, all
generated tokens (including failures), logical prompt tokens, executor CPU
thread time, and newly stock-connected target IDs. The complete shard seals
the journal hash. `r1_observed_solved_vs_work.json` uses balanced per-worker
observed call checkpoints; credit comes only from complete wave boundaries.
The x coordinate reports actual summed forward calls, not nominal budgets,
per-request double counting, uncached prefill tokens, GPU FLOPs or invented
global wall-clock ordering. Work above a finished run is never extrapolated.
This instrumentation does not change generation or impose a search cutoff.
At this edit the submitted460 runtime is still e973e1d and lacks these new
journals; a local edit cannot retrofit them. Independent matched-work sampling
and the complete Figure2b comparison are still pending.

R2 reuses exactly the same160 R1 rows and precomputed source complexities.
Published structural route recovery tests whether the FULL reference edge set
is present in the accepted graph after search, not just in a best continuation.
The online unique generated route counts are LOWER BOUNDS. Offline
`scripts/enumerate_frozen_route_diversity.py` enumerates all acyclic,
stock-connected production-choice DAGs in each saved graph, with no fixed
depth, width or count cutoff. One distinct nonstock molecule uses one production
choice; precursor multiplicity is retained, and alternative electron programs
for the same reaction do not multiply outer route counts. This can take
exponential time. Only a completed full460 condition and fully consumed
enumerations may seal an exhaustive-count manifest; interrupted counts remain
partial. No actual model-route counts have been produced yet.
To include completed counts in the R1/R2 report, run the enumerator once per
completed condition, then pass each sealed `manifest.json` using repeatable
`--route-diversity-manifest` arguments to `analyze_frozen_route_search.py`.
The analysis binds the model/assets, full denominator, exact shard hashes,
saved graph hashes and counts hash; inconsistent or partial counts are rejected.
Original inference rows remain unchanged. Without a matching sealed manifest,
the report continues to expose only the online lower bound, not an exact count.
The actual-SFT exact-match reference census is complete; unmatched or unsupported
reference edges remain unannotated. All160 primary routes remain Unknown, and
no C1 novelty claim is permitted. Source
maps alone never become mechanisms. Disjoint maps are missing correspondence,
not a zero center. Paired2000-bootstrap summaries and observed PDF panels are
written only from actual complete outputs. These are provisional panels, not
the complete locked paper Figures2/3. No new training/RSI, unrequested pilot,
or unrelated task cancellation.

The original full3120 task at immutable runtime0796e2d had task-level caps.
On October14 it was stopped while queued and verified END before replacement.
`meteor_mechet_uspto31k_full_dynamic_tree_8a100_qy_20261014_02` instead pins
runtimea23cf576, with renewable windows and the SAME3120 source and actor.
The initial READY readback establishes submission, not actual inference.
The full460 route task remains unchanged at runtimee973e1d. No local edit
silently changes a submitted immutable runtime; the replacement is explicit.

## Historical Figure3b panel, separate exposure basis

`scripts/plot_frozen_operator_growth.py` consumes the frozen historical
`outputs/issue79/nmi_operator_basis_train_20261002.json` in main (report SHA
`a7e2c029d53a51eeeab89befc44440d5c91d0ef689199349beb04084c889d6d1`).
The full original source SHA is checked before rendering PDF/SVG. Six recorded
nested seed42 identifier-hash coverage points end at604 local primitives and
6553 ordered program signatures across257167 strict train reactions. The
horizontal axis is training-pool coverage, NOT optimizer steps or a new model
learning curve. V1 omits import identities, is coupled-order sensitive and
includes some nonordinary BE_DELTA signatures. It is neither current Stage-II
actual-label exposure (603/6491) nor evidence of route-level model performance.
The panel and its provenance manifest do not constitute complete Figure3;
observed full-route outcome and real saved-route panels remain pending.
