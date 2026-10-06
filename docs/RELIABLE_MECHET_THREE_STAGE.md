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

An explicitly provisional smoke at Stage-I `checkpoint-2000` (only 2,000 of
31,366 optimizer steps; 4 fixed validation reactions, 28 gold-state decisions;
one T4, FP16) confirmed that model generations enter the SFT tool-call format.
Among 13 reference electron-event decisions, 12 chose the event tool, 10
executed, and 1 matched the reference successor. Fragment exactness was 0/11;
finish exactness was 1/4. This is a tiny incomplete-checkpoint diagnostic, **not**
final local competence, product-start accuracy, or a reason to select a model.
The report is
`outputs/eval/reliable_mechet_state_ckpt2000_valid4_t4_probe_20261007/evaluation.json`
in shared artifacts, with checkpoint hash and explicit provisional lineage.

A second explicitly provisional diagnostic used **the identical four validation
reaction IDs, 28 gold-state decisions, T4/FP16, greedy decoding and SFT-aligned
tool prefix** at `checkpoint-7000` (7,000/31,366 updates). Compared with
`checkpoint-2000`, correct tool selection rose 19/28 to 24/28, exact local
decisions 2/28 to 8/28, executable reference electron events 10/13 to 13/13,
and reference successor/event matches 1/13 to 5/13. Import-fragment exactness
rose only 0/11 to 1/11; finish exactness rose 1/4 to 2/4. The model still
proposes wrong electron-participant and endpoint-context fragments on these
cases. This is evidence of local learning during training, **not** a stable
validation estimate, product-start endpoint result or proof that later stages
will succeed. The complete report is in shared artifacts at
`outputs/eval/reliable_mechet_state_ckpt7000_valid4_t4_probe_20261007/evaluation.json`.

The exact-import metric needs a role breakdown. A SHA-bound audit of the
frozen State-SFT rows (`scripts/audit_reliable_import_supervision.py`) found:

| Split | Reactions | Import decisions | Electron-participant copies | Endpoint-context copies | Reactions with endpoint context |
|---|---:|---:|---:|---:|---:|
| train | 257,167 | 706,902 | 600,403 | 743,579 | 239,775 (93.2%) |
| valid | 2,890 | 7,919 | 6,577 | 8,626 | 2,712 (93.8%) |

In the 4-reaction `checkpoint-7000` diagnostic, the gold import roles were
seven electron-participant-only, two mixed, and two endpoint-context-only.
The model exactly matched one of the seven participant-only imports, and none
of the four context-containing imports. The concrete wrong predictions include
chloride as bicarbonate and water as a different organic reagent, so the
participant problem is genuine; it cannot all be explained away by
underdetermined solvents/spectators. Conversely, endpoint-context labels often
contain solvents, salts or other mixture components that a product-only input
cannot uniquely specify. The current SFT supervises these roles together in
most reactions; future quality reports must separate them and must not treat
context-copy exactness as a direct measure of reverse-electron-flow ability.
This audit does **not** change the running Stage-I data or claim a corrected
Stage-II training result. Reports are in shared artifacts at
`outputs/eval/reliable_mechet_state_import_supervision_{train257167,valid2890_ckpt7000}_20261007.json`.

A separate reference-only counterfactual asks whether endpoint-context copies
actually affect the electron-flow chemistry. It first resolves each Axx alias
against the **original** mapped state, then removes only disconnected context
components without renumbering retained atoms, executes the same private-map
electron move and checks the projected successor and structural endpoint. On
the frozen 2,890-reaction validation split, 2,873 reference trajectories retain
the exact structural endpoint after this projection; those successful cases
contain 11,444 verified projected electron events and 8,530 removed context
copies. The other 17 are **not** safe to project: a nominally
`endpoint_context` atom later joins an electron-participant component, including
H–Cl formation and Pd–Cl coordination. Thus the import-purpose annotation is
not a universal spectator/participant separator, and no context deletion is
applied to the live training or evaluator. Deleting those imports while keeping
literal Axx aliases is also invalid because the address inventory shifts.
This audit tests reference chemistry, **not** product-start model accuracy.
Reproduction: `scripts/audit_reliable_context_projection.py`; report in shared
artifacts at
`outputs/eval/reliable_mechet_state_context_projection_v2_valid2890_20261007.json`.
The frozen local evaluator now retains its original all-fragment exact metric
and additionally reports nominal electron-participant and endpoint-context
import exactness, each on its explicit role-present denominator (including
mixed imports). The reliable local launcher opts into this breakdown; legacy
evaluator invocations are unchanged. These are exact agreement with recorded
fragments and role labels, not judgments of chemical feasibility or model
product-start performance. A read-only re-score of the same unfinished
`checkpoint-7000` four-reaction shard gives participant-present exactness
**1/9** and context-present exactness **0/4**; the two mixed-role imports are
counted in both relevant denominators. This remains a tiny provisional local
diagnostic, not a final validation estimate.

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

The Stage-II warm-start path has an explicitly provisional single-update smoke:
`scripts/smoke_reliable_stage2_parent.py` loaded Stage-I `checkpoint-3000`
(3,000/31,366 steps), encoded an actual compressed-history electron-event
decision with one accepted prior import (1,274 input / 166 supervised tokens),
and ran one FP16/T4 optimizer update. The loss was finite (0.0160), the LoRA
gradient norm was nonzero, and a LoRA tensor changed. No weights were saved.
This proves that the Stage-II data, assistant mask, pinned 0.6B base, parent
adapter and training step are compatible; it does **not** establish Stage-II
quality or authorize continuation before the final Stage-I adapter SHA is frozen.

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
For the reliable protocol, a resumed collection checks the exact source,
actor and (when present) successor-critic weight hashes, model revision,
frontier, round and evaluation mode before accepting cached shards. A changed
critic cannot silently reuse old rollout scores.

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

For matched product-start K=1 rollouts, each parsed proposal now stores its
own generated-token count, summed log-probability and mean token
log-probability. `scripts/analyze_reliable_product_start.py` ranks these
**actions** by that frozen score and reports executor-rejection risk versus
coverage, including the top-scored decile. Unparseable proposals and nonfinite
scores remain separate counts; legacy shards without per-action scores report
this analysis as unavailable rather than inventing confidence. This is a
directly observed high-confidence **executor rejection** proxy, not a claim
that every accepted non-reference reaction is chemically valid or that every
rejected reaction is experimentally impossible. Endpoint risk--coverage remains
separate and uses terminal whole-path mean policy score.

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
The matched product-start diagnostic now consumes this SHA-checked audit: it
retains all selected reactions in endpoint accuracy, records the unstable IDs,
and reports reference-relative failure categories both overall and only for
parity-stable reactions. Neither category count is an independent verdict on
chemical plausibility. A trial that only canonicalized atom order left all six
unstable reactions in the fixed 128-case sample unstable; stripping private
maps before Kekulization aligned the two map views but changed nine original
gold successors in that sample. Neither trial is a validated drop-in executor
repair, so the frozen training runtime is unchanged.
State-SFT and Trajectory-SFT have different reference-decision files and
prompt contracts, so their parity reports must be built separately with the
same 2,890 source reactions. The State-SFT audit uses `--state-only`; the
Trajectory-SFT audit uses the default compressed-history mode. The product-
start analysis checks the exact source and decision SHA-256 of the selected
report and fails rather than silently applying one stage's audit to the other.
The independently run State-SFT audit also found 2,778/2,890 product-only
remapped gold replays, with the **same 112 reaction IDs** as the history-stage
audit. Its artifact is
`outputs/eval/reliable_mechet_state_product_mapping_parity_valid2890_20261006.json`.
An additional read-only classifier replayed all 112 State-SFT failures under
RDKit 2026.03.4, checking the SHA-256-bound source and decision files. **All
112 first divergent actions are `apply_electron_flow`; 94 occur at decision 1,
17 at decision 3 and one at decision 9.** In each case, the original-map and
product-only-map versions assign a *different Kekulé bond order to at least
one aromatic electron-source bond*. Every divergent successor also has a
different heavy-atom connectivity graph, not merely a different SMILES spelling
of the same connectivity. The classifier and machine-readable per-case report
are `scripts/classify_reliable_mapping_failures.py` and
`outputs/eval/reliable_mechet_state_mapping_failure_connectivity_valid2890_20261007.json`.
This identifies a representation-sensitive executor failure mode; it does not
establish which of the two successors is chemically correct. We retain all 112
in endpoint denominators and exclude them only from claims that assign a
reference-relative *model* error at the first divergence. The frozen executor
and active SFT job are unchanged pending a separately replay-audited repair.
An additional read-only pass through all 2,890 original-map validation traces
found 487 aromatic source-bond occurrences whose Kekulé order is double and
12 whose order is single (from six reactions). This full pass is reproducible
with `scripts/classify_reliable_mapping_failures.py --all-source-orders` using
the SHA-bound State-SFT parity audit above. Therefore a blanket "treat every
aromatic source as a π bond" rule would change some already-verified reference
transitions; it is **not** a safe repair. The current natural-language action
names an aromatic bond but does not state which electron pair of an ambiguous
Kekulé assignment is moved. A future representation/executor revision must resolve that
ambiguity explicitly and pass full frozen-reference replay before replacing
the current protocol.
An audit-only bounded branch probe enumerated alternate Kekulé assignments
without changing atoms, formal charges, explicit hydrogens, connectivity or
nonaromatic bond orders. The recorded reference successor was among the
executable branches for **112/112** remap failures (at most 50 assignments
examined for any one case; mean 4.5). The artifact is
`outputs/eval/reliable_mechet_state_kekule_branch_probe_valid2890_20261007.json`.
This proves that the frozen electron action can reach the reference under
another aromatic bond assignment; it does **not** supply a gold-independent
rule for choosing that branch at inference, and it is not a recovered model
accuracy result. The audit-only executor entry point defaults to the unchanged
single-assignment behavior in all training and deployed inference paths.
The stronger gold-free check enumerated the *bounded successor set*
from the original-map and product-only-map state before each of the 112 first
divergences. The two sets matched **112/112**, with 2--3 distinct executable
successors per case; neither enumeration hit its 128-structure limit. The
reference successor belonged to both sets in all 112 cases, scored only after
enumeration. This establishes map-invariant **set-valued** execution on these
cases, not a map-invariant single next state or an inference-time selector.
The report is
`outputs/eval/reliable_mechet_state_kekule_successor_sets_valid2890_20261007.json`.
After adding that private audit entry point, we reran the ordinary default
executor over all 2,890 State-SFT validation traces. Its counts and all 112
failure records are byte-equivalent after canonical JSON sorting to the
pre-change parity report (SHA-256 of `{counts,failures}`:
`580657b53fb8eb1f47ba58b2fad55bc9c71ae7a2c0301da7c005bf2d1d224226`).
The regression report is
`outputs/eval/reliable_mechet_state_mapping_default_regression_valid2890_20261007.json`.

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

The Stage-I/II full-test launcher is
`scripts/run_taiji_reliable_mechet_full_endpoint_a100.sh`. It evaluates
product-only greedy K=1 on the frozen, **unfiltered 28,971-row**
`flower_full_endpoint_sft/test.jsonl` and refuses to overwrite prior results.
`scripts/audit_reliable_full_endpoint_eval.py` binds the source to its manifest,
checks all source IDs and output shards, rejects duplicate/foreign predictions,
and counts missing predictions as failures against 28,971. The reported main
metric is structural precursor exact match; complete-state exactness is a
separate secondary metric. This launcher is ready for a **finished** Stage-I or
Stage-II adapter, but no full-test job is submitted while Stage I is training.
The preflight executes the actual product-only private-remapping path on every
test product before launching generation; a 2026-10-07 read-only audit passed
all **28,971/28,971** test products with zero input-mapping failures.
It does not report mechanism/process accuracy for the four upstream-corrupt
endpoint rows; that analysis remains on the separate 28,967 strict view.

A SHA-bound crosswalk between the frozen full-endpoint and strict-proof test
files (`scripts/audit_reliable_flower_test_crosswalk.py`) shows why their
results cannot be merged by reaction ID alone. All 28,967 strict reaction IDs
occur in the 28,971-row full test; the full-only IDs are `PC`, `PM`, `RC`, `RS`.
But **only 24,067/28,967** pairs have the exact same normalized product-only
model input. Another 4,899 products are chemically equivalent after hydrogen
normalization but differ in explicit-H representation seen by the policy, and
one reaction (`7007`) selects a genuinely different product molecule. The
structural precursor labels agree for 24,743 pairs; in 4,179 the strict view
includes additional fragments, and in 45 they differ in other ways. Thus the
official 28,971-row endpoint score must use its own input and labels; a
28,967-row process/endpoint view must use its strict input and labels, or be
explicitly presented as a matched-input subset. No result from one view can be
silently rescored or called the other. The reproducible audit report is in
shared artifacts at
`outputs/eval/reliable_mechet_flower_full_strict_test_crosswalk_20261007.json`.

The separate strict-view launcher is
`scripts/run_taiji_reliable_mechet_strict_test_a100.sh`. It freezes the
28,967-row `flower_inverse_tool_sft_action_delta_v1/test.jsonl` inputs and
scores against that view's own structural precursors. Its independent audit
(`scripts/audit_reliable_strict_test_eval.py`) checks the source hash, all
reaction IDs, missing/duplicate predictions, structural exactness, terminal
episodes, and accepted/rejected decision counts. Missing predictions remain
endpoint failures. Decision acceptance is reported only over observed episodes
and is **not** a chemical-validity or hallucination estimate. The launcher
requires a finished adapter and is not yet submitted. A read-only preflight
under RDKit 2026.03.4 successfully ran product-only private mapping for all
**28,967/28,967** strict-test inputs on 2026-10-07.

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

`scripts/run_syntheseus_search.py` now records the frozen candidate-pool,
target and stock hashes, search budgets, reaction-model calls, first-solution
calls/time, admitted-edge proof replay and route certification. It audits the
offline candidate file *before* the search adapter filters source-reported
execution failures. The source-reported proposal failure rate and the
independently replayed **admitted-edge** failure rate have different
denominators and must not be conflated. If an admitted edge lacks a proof, its
status is `unverified`; the admitted-edge hallucination, route certification
and wasted-expansion rates are `null`, never an artificial zero. The observed
certified-route count remains a lower bound when other routes are unverified.
Certification means proof replay
to the declared precursor under this executor, not chemical or laboratory
truth. This instrumentation has passed a real Syntheseus 0.7.2 toy Retro*
route; it is not yet a MechET planning result.

The online path is `scripts/run_syntheseus_online.py`. It takes a frozen final
Stage-I or Stage-II adapter, targets and stock; **every planner query** starts a
fresh product-only reverse-electron-flow rollout from the queried molecule.
It never reads reference precursors. K=1 is the frozen greedy policy; K>1
means K independently seeded stochastic episodes. A terminal episode is
admitted as a reaction edge only if the executor reaches a structural
precursor after at least one electron event. The edge carries the accepted
actions as a replay certificate, and the reliability evaluator independently
rebuilds its private product state and replays every action. The offline and
online providers share `run_planner`, so stock and Retro*/breadth-first budgets
are identical. Online reports include all attempted episodes, rejected
decisions, nonterminal episodes, unadmitted episodes and the pre-dedup episode
admission rate. The admitted-edge failure rate is conditional on admission;
it must not be presented as the failure rate over all model proposals.

The CLI refuses an adapter whose manifest does not match the requested stage,
Qwen3-0.6B base revision or executor lineage. `--dry-run` checks the artifact
and records hashes without loading the model. The actual final-policy planning
run is pending its frozen adapter; passing toy integration tests is not a
multi-step model result. Executor replay does not establish wet-lab chemistry.

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
weights SHA-256. The Stage-II launcher now also requires the final Stage-I
adapter manifest and data contract, and verifies the pinned base revision,
State-SFT condition, original train-file SHA, one-epoch setting and lack of a
prior adapter. A `checkpoint-*` directory or a hash-matching adapter from a
different condition cannot pass this gate. A non-submittable A100 task template
is at `configs/taiji/meteor_mechet_reliable_trajectory_06b_1ep_8a100_qy_TEMPLATE.json`;
its run ID, parent SHA and private Ceph init command must be replaced only
after Stage I finishes. Stage III remains disabled until the Stage-II SHA-256
is frozen.
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
