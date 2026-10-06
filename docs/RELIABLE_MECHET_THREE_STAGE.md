# Reliable MechET: three-stage lightweight campaign

## Fixed scientific position

This branch does not replace MechET with an endpoint-only model or a standalone
verifier. The main object remains **reverse electron-flow prediction from the
product**, with the precursor derived from the executed terminal state.

The journal extension asks a reliability question:

> Can the same reverse-electron-flow prediction contract reduce hallucinated
> retrosynthetic reactions while remaining useful for one-step prediction and
> multi-step planning?

The training paradigm remains the established three-stage MechET curriculum:

1. **State-SFT** -- learn one inverse decision from an executor-authoritative
   molecular state.
2. **Compressed-history Trajectory-SFT** -- continue the same policy with
   accepted-prefix history so errors can be studied and controlled across a
   complete trajectory.
3. **EARHO** -- perform bounded execution-anchored improvement at the first
   recoverable divergence frontier.

No stage introduces an independent precursor answer channel.

## Model-size policy

New journal results use a lower-cost model family so that all primary results
can be reproduced from one frozen protocol.

- **Qwen3-0.6B**: primary model for all three stages and the full reliability
  study.
- **Qwen3-1.7B**: confirmation scale for the main final comparison after the
  0.6B protocol is frozen.
- **Qwen3-8B**: historical MechET evidence and failure diagnosis. It is not a
  requirement for systematic re-running of the new journal matrix.

The primary 0.6B revision is pinned to
`c1899de289a04d12100db370d81485cdf75e47ca`.

## Stage I -- State-SFT

Configuration:

    configs/agent/natural_language_event_v2_qwen3_0_6b.yaml

The policy sees the target product, executor-owned current molecular state and
the current decision inventory. It predicts exactly one of the existing MechET
decision families:

- introduce a missing precursor-side fragment;
- apply one coupled reverse electron-flow event;
- finish the trace.

The model-facing fragment interface is the grounded first-use interface:
ordinary unmapped fragment SMILES are proposed by the policy, and the executor
assigns private atom maps. The environment does not retrieve the reference
precursor or complete missing fragments for the model.

Stage-I evaluation separates:

- fragment proposal;
- source/sink grounding;
- formal execution;
- successor-state agreement;
- termination.

This is the local-competence stage. A high local score is not interpreted as
proof of reliable product-start retrosynthesis.

## Stage II -- compressed-history Trajectory-SFT

Configuration:

    configs/agent/natural_language_history_v2_qwen3_0_6b.yaml

Stage II initializes from the Stage-I adapter and keeps the same action space
and chemistry executor. The only intended change is the accepted-prefix
history available to the policy.

The main question is whether trajectory conditioning reduces the error growth
seen when fragment and electron-flow decisions are composed from the product.

Required measurements include:

- product-start endpoint recovery;
- first consequential error depth;
- fragment error rate;
- source/sink error rate;
- formal execution failure rate;
- executable-but-wrong-successor rate;
- termination failure rate;
- error rate versus trajectory length;
- risk--coverage behaviour.

The endpoint is still read only from the terminal executed state.

## Stage III -- EARHO

Template:

    configs/agent/earho_reliable_mechet_qwen3_0_6b.yaml

The Stage-III configuration is deliberately non-runnable until the exact
Stage-II adapter SHA-256 is inserted. This is a provenance gate, not a
placeholder to bypass.

The Stage-III driver now binds the frozen FlowER strict source, Stage-II
compressed-history manifest and exact adapter hash before preparing any
rollout. It streams the selected reactions rather than loading the full
257,167-row trace source, uses the natural-language actor-update path and
the 0.6B base for its successor critic. The Taiji launcher is
`scripts/run_taiji_reliable_mechet_earho_a100.sh`; it is **not** a submitted
experiment and must not run until the Stage-II parent and product-start
representation audit are accepted.

EARHO keeps the existing design:

- start from real product-start rollouts;
- locate the earliest recoverable divergence frontier;
- branch over alternative first decisions;
- execute every proposal before assigning credit;
- pool successor-equivalent alternatives;
- use bounded continuation;
- retain the exact endpoint as the dominant terminal success criterion.

The new journal reporting adds the first assigned failure cause:

1. fragment proposal;
2. source/sink grounding;
3. formal execution;
4. executable but wrong successor;
5. termination.

EARHO is successful only if reliability improves at the reaction edge and the
gain survives product-start evaluation.

## Hallucinated retrosynthetic reactions

The journal paper should avoid defining every non-reference precursor as a
hallucination. The operational target is narrower.

A **mechanistically unsupported retrosynthetic prediction** is a proposed
reaction edge whose precursor/product pair is structurally parseable but whose
predicted reverse-electron-flow process fails the frozen execution contract or
reaches an inconsistent successor/endpoint.

The main measurable failure classes are therefore executor-observable:

- invalid or unnecessary fragment introduction;
- impossible or mis-grounded source/sink assignment;
- transition execution failure;
- formally executable transition that reaches a wrong successor;
- inconsistent or premature termination.

These are reported separately from ordinary reference exact match.

The primary reliability quantities are:

- structural endpoint Pass@K / ranked Top-K where ranking is frozen;
- process-reliable endpoint recovery;
- mechanistic hallucination rate;
- high-confidence hallucination rate;
- risk--coverage curve;
- first-divergence depth;
- failure-cause distribution.

Formal execution does not certify reaction conditions, kinetics, selectivity,
yield or laboratory success.

Product-only private-map replay is a separate executor-parity prerequisite:
the mapping labels are hidden from the policy, but different private product
atom orders can change the executor's aromatic/Kekule successor. Audit with
`scripts/audit_reliable_product_mapping_parity.py` before interpreting a
first-divergence or wrong-successor label as a model error. A reference replay
under its original private mapping is not evidence of map-invariant execution.
On the frozen 2,890-reaction validation split, the audit found 2,890/2,890
original-map reference replays but only 2,778/2,890 product-only-remapped
replays; all 112 failures are successor mismatches after the identical visible
root prompt. The machine-readable report is
`outputs/eval/reliable_mechet_product_mapping_parity_valid2890_20261006.json`
in the shared artifact repository, not a model score. Do not silently remove
those 112 reactions from a benchmark denominator. Repair the representation
parity or explicitly report this evaluator limitation before Stage-III
product-start first-divergence credit is interpreted as chemically meaningful.

## Historical Qwen3-8B position

The previous Qwen3-8B MechET remains an important baseline rather than a result
that must be recreated under every new setting.

The earlier short-horizon replay-compatible experiment demonstrates that the
trace-owned inverse-electron-flow contract can recover precursors with high
execution success. Broader autonomous runs reveal a different fact: fragment
selection, electron-flow decisions and termination errors can accumulate over
longer trajectories.

The journal extension uses this contrast as motivation for the three-stage
reliability study. It does not claim that the old 8B numbers and the new 0.6B
numbers belong to one matched leaderboard.

## Controls

Controls remain necessary, but they are not the main method.

- endpoint-only prediction measures the unconstrained precursor task;
- NetEdit measures whether endpoint changes alone explain a reliability gain;
- one-shot/open-loop electron flow measures whether explicit electron-flow
  notation without executed-state feedback is sufficient;
- the historical sequential 8B model is a large-model lineage reference.

Matched causal comparisons use the same reaction IDs, endpoint evaluator and
decoding budget wherever the relevant contracts permit.

## One-step evaluation

The complete FlowER endpoint test remains 28,971 reactions. The strict
executable program view is 28,967 reactions.

- headline precursor recovery uses the complete 28,971-reaction endpoint
  denominator;
- process and failure-cause analyses use the 28,967 strict program view;
- missing predictions remain failures;
- Pass@K is not relabelled as ranked Top-K unless a frozen ranking score exists.

A smaller model is not required to reproduce historical 8B absolute accuracy
before it can support the reliability claim. The critical evidence is the
matched progression across State-SFT, Trajectory-SFT and EARHO under one frozen
0.6B protocol.

## Multi-step planning

Multi-step planning is evaluated after the one-step three-stage policy is
frozen.

Each synthesis edge is proposed by the same one-step MechET policy. Reverse
electron-flow state is not propagated from one reaction edge to the next; each
retrosynthetic edge starts from its current target molecule.

Use the same Syntheseus/Retro* planner, stock, candidate count and search budget
across compared policies.

In addition to solved rate and route recovery, report:

- hallucinated edge rate;
- all-edge executable route rate;
- certified route rate under the frozen MechET execution contract;
- wasted expansions below failed reaction edges;
- calls and wall time to first solution.

The purpose is to test whether reaction-level reliability changes search
efficiency and route reliability, not merely whether a larger search budget can
hide one-step errors.

## Run order

Do not re-run the historical 8B matrix first.

1. Freeze and audit Stage-I/II data manifests.
2. Train Qwen3-0.6B State-SFT for one epoch.
3. Evaluate local chemistry and product-start reliability.
4. Continue the same adapter with one epoch of Trajectory-SFT.
5. Re-evaluate with the identical product-start protocol.
6. Freeze the Stage-II SHA.
7. Run bounded EARHO Stage III.
8. Evaluate the final 0.6B policy on one-step reliability.
9. Run the multistep planning study.
10. Only after the result is stable, repeat the key final comparison at 1.7B.

This run order is the journal reproduction path.

## Frozen 0.6B training preparation

The existing State-SFT and compressed-history datasets each cover all 257,167
strict-executable train reactions, expanded into 2,007,421 one-decision rows;
their validation and test reaction counts are 2,890 and 28,967. The two
manifests record zero unresolved reactions and permit training. These are
decision-row counts, not a new FlowER reaction denominator.

The pinned Qwen3-0.6B and historical Qwen3-8B snapshots contain byte-identical
`tokenizer.json` and `tokenizer_config.json`. For this frozen pair only,
`scripts/rebind_identical_tokenizer_cache.py` checks both tokenizer assets,
complete source-file hashes, 4,096-token budgets, zero truncation and every
Arrow shard before writing a model-specific manifest that links the existing
tokens. The old cache is unchanged; a different tokenizer must be tokenized
again. The trainer rejects a cache with the wrong model name, immutable
revision or maximum length even during dry-run.

`scripts/run_taiji_reliable_mechet_sft_8gpu.sh` runs one stage per task. It
rechecks the frozen data and cache contracts on the allocated GPU host, stages
the Arrow shards and model locally, and leaves normal training progress on the
default POD log. Stage II additionally requires the exact Stage-I adapter
SHA-256; Stage III remains disabled until the Stage-II SHA-256 is frozen.
The trainer supplies length-grouped sampling with exact Arrow-vectorized token
lengths; it does not ask every DDP rank to Python-format all 2,007,421 rows
merely to build the sampler. This changes no example, order policy or loss.

The matched 128-reaction validation diagnostic uses
`scripts/run_taiji_reliable_mechet_local_eval_a100.sh`. Stage I and Stage II
read their own frozen model-visible decision rows; in particular, Stage II is
evaluated with its actual compact-history prompt, not a Stage-I or ad-hoc
transcript prompt. The evaluator checks adapter/model revision, generates with
the SFT-aligned Qwen tool prefix, and reports import, event, successor and
finish errors separately. Because each decision receives a reference current
state, this is a **local diagnostic**, not product-start endpoint recovery.

The matched product-start pilot uses
`scripts/run_taiji_reliable_mechet_product_start_a100.sh` with
`MECHET_RELIABLE_STAGE=state` or `trajectory`. Both stages use the same frozen
SHA256-selected 128 validation reactions, greedy K=1, 40 accepted decisions,
32 maximum imported fragment copies, rejection of a target-retaining finish
without any electron-flow transformation, no value critic, and the SFT-aligned Qwen
tool prefix. At the root, the evaluator strips **all original source atom maps**
and deterministically assigns fresh private maps from the unmapped product;
only the product/current executor state and its temporary atom/bond inventory
enter the policy prompt. Stage II reconstructs its compact history exclusively
from accepted runtime actions and tool results. The reference precursor is
read only for scoring after the rollout; it is never used to propose, filter or
rank actions. Adapter base model, pinned revision and Stage-I/II observation
contract are checked before loading. The runner verifies all 128 IDs in its
output. This is an autonomous validation pilot, **not** the 28,971-reaction
headline endpoint test or the 28,967-reaction strict-process test.
The runner also writes `failure_analysis.json` and per-case first-divergence
records. A mismatch with the single recorded reference trajectory is labeled
*reference-relative*: an exact endpoint reached through another path is not
counted as a chemical failure. Rejected imports, ungrounded source/sink phrases,
formal execution errors, executable but reference-different successors, and
termination errors are kept separate. The risk--coverage table uses the
policy's own sequence score for completed traces, with incomplete traces placed
after completed ones; it never uses the reference endpoint to rank predictions.
This sequence score is an uncalibrated pilot confidence proxy, not a probability
of chemical validity.
