# CRN-JEPA FLOW routing with the existing Qwen Tool-SFT actor (opt-in v1)

## Decision

Only replace the **source/sink electron-flow argument generator**. Do not replace
Qwen's imported-fragment or finish behavior, retrain a new Qwen, fine-tune
CRN-JEPA from scratch, or modify the frozen executor. PR82's State-SFT,
Trajectory-SFT and EARHO code and checkpoints remain unchanged.

The existing trained Qwen in MechET PR82 is **Qwen3-0.6B**, pinned to
`c1899de289a04d12100db370d81485cdf75e47ca`.
No confirmed Qwen3-0.8B checkpoint was found in that protocol. An alternative
Qwen checkpoint is supported only if its base/revision and adapter-manifest
lineage match the CLI arguments exactly.

CRN-JEPA PR18 pretrained weights: 
`outputs/auto_research/pr18/full_corpus_pretrain_v1/train_full_8a100_v2/latest.pt`
under the separate CRN-JEPA checkout. SHA-256:
`bb2f7fe53e87fc0e2438c27065ceb588dcd38e604959c9d69fb3984885a376ee`.
This is the **selected train/development** checkpoint, not a new MechET
product-start result. CRN-JEPA native FlowER inverse Event@1=82.16% on
reference-provided current states; no product-start endpoint evidence follows
from this number.

## Inference

1. Render the exact Qwen3 State-SFT (or explicitly selected history-SFT)
   product/current-state/inventory prompt with SFT tool-call suffix.
2. Score the three tool-name continuations with frozen Qwen. Unlike the original
   full-call decoder, this is a **new, explicitly labeled routing protocol**.
3. If IMPORT, force only Qwen's import tool prefix and generate fragment JSON;
   if FINISH, use existing executor-owned final-state projection.
4. If FLOW, call CRN-JEPA PR18 `DirectionalCRNPretrainFull.policy`
   with `build_grounded_graph(current, None)` and
   `MechETGroundedInventory(current)`. Translate its existing
   source/sink/commit predictions into the **same** `apply_electron_flow`
   argument schema and verify them with the unchanged MechET executor.
5. No independent reactant answer, next-state SMILES, reference trajectory,
   reward or future label enters the inference actor.

No Qwen token generation is performed for FLOW arguments. The graph model
still invokes CPU RDKit and its existing Python/torch top-K loops; **no GPU-native
executor or proven wall-time speedup is claimed**. The original Qwen tool
routing differs from forced-prefix likelihood routing; this change must be
controlled when comparing endpoint results.

### Known coverage and chemistry limits

The reused CRN-JEPA FLOW head predicts coupled two-electron source/sink
events. Its v1 decoder does **not** produce MechET's `BE_DELTA` class.
The hybrid does not conceal such a gap by allowing Qwen to generate FLOW or
by silently reconstructing a gold move. Unknown/unsupported predictions and
nonterminal episodes count as failures.

CRN-JEPA PR18 was scored in a pinned RDKit-2024.09.6 and earlier MechET
executor environment; PR82 runs RDKit-2026.03.4. A matched executor-parity and
Kekulé-remapping audit is REQUIRED before any scientific comparison.
PR82 identified 112/2,890 product-remapping reference successor mismatches.
Do not exclude these from endpoint denominators or relabel them model errors.

### Sample launch on an authorized CUDA host

Paths below are relative placeholders resolved under real local checkouts;
the checkpoint files are not contained in this GitHub PR.

```bash
export PYTHONPATH=/path/to/MechET/src:/path/to/CRN-JEPA/src:/path/to/MechET
python scripts/run_crn_qwen_hybrid.py \
  --data data/flower_inverse_tool_sft_action_delta_v1/valid.jsonl \
  --output outputs/eval/crn_qwen_hybrid_valid128 \
  --sample-reactions 128 \
  --policy-adapter outputs/agent/natural_language_event_v2_qwen3_0_6b_seed17 \
  --crn-root /path/to/CRN-JEPA \
  --graph-checkpoint /path/to/CRN-JEPA/outputs/auto_research/pr18/full_corpus_pretrain_v1/train_full_8a100_v2/latest.pt \
  --graph-flow-topk 1
```

## Endpoint RL (train-only, graph policy exclusively)

Use `scripts/train_crn_graph_endpoint_rl.py`; frozen Qwen remains the
IMPORT/FINISH policy and **receives no gradients**.

For each strictly replayable train reaction, sample a frontier among its last
three reference electron-flow decisions, propose a stochastic distinct group
from CRN's model-ranked Top-32 events (K=8 by default), execute the candidates
and continue with Qwen+Graph for a bounded number of accepted decisions.
Rewards are:
- exact structural precursor endpoint after continuation: +1;
- exact reference successor under the verified train-only replay: +0.2;
- executor-accepted but unsupported: 0 (not a claim of impossibility);
- executor rejected: -0.2.

Only groups with **verified positive and competing negative** candidates
receive a signed group policy-gradient update. All-negative groups have zero
relative advantage; verified reference replay then supplies a supervised
fallback. The trained parameters are only CRN policy/decoder and direction
heads, with its 12-layer Graph Transformer, future head, target encoder and
all Qwen parameters frozen. Policy-gradient scores reuse existing differentiable
source/sink/commit teacher-forced event NLL. Because candidate sampling is from
a truncated Top-32 slate, this is a **biased policy-gradient surrogate, not an
unbiased GRPO/PPO implementation**.

Run on authenticated training rows only; the source-byte SHA must be explicitly
supplied and checked. The trainer saves a small `graph_endpoint_rl_head.pt`
delta plus per-group audit and counts effective endpoint-RL groups separately
from reference-only updates. If no effective groups were found, it honestly
reports a **reference-replay SFT-only** result. No claims of endpoint gain
before paired product-start validation.

```bash
python scripts/train_crn_graph_endpoint_rl.py \
  --data data/flower_inverse_tool_sft_action_delta_v1/train.jsonl \
  --source-sha256 "$(sha256sum data/flower_inverse_tool_sft_action_delta_v1/train.jsonl | cut -d' ' -f1)" \
  --output outputs/agent/crn_graph_endpoint_rl_smoke \
  --sample-reactions 64 --rl-updates 64 --slate 8 \
  --continuation-horizon 8 \
  --policy-adapter outputs/agent/natural_language_event_v2_qwen3_0_6b_seed17 \
  --crn-root /path/to/CRN-JEPA \
  --graph-checkpoint /path/to/CRN-JEPA/outputs/auto_research/pr18/full_corpus_pretrain_v1/train_full_8a100_v2/latest.pt
```

Then run the **same** 128-row product-start validation with the above
evaluation command plus `--graph-rl-head outputs/agent/crn_graph_endpoint_rl_smoke/graph_endpoint_rl_head.pt`.
Report Top-1 exact, formal acceptance, first-error depth, runtime, Qwen
import calls, no-op finish and training group support. Never infer endpoint
improvement from oracle-state Event@K.

## Minimal promotion gates

- Before RL: the 128-row hybrid K=1 product-start pilot must execute and score
  all 128 reaction IDs, log runtime and preserve the original endpoint denominator.
- After RL: compare identical frozen 128-row IDs and exact same controller,
  executor, and graph Top-K budget; only the graph head may differ.
- A negative 128-reaction result or zero effective RL groups must trigger
  an explicit cause audit and at most **two bounded corrective cycles** on
  train/validation; these are **not** reasons to abandon the PR or to claim
  success. After those cycles, publish the actual paired negative result if
  appropriate. Never expand to 2,890 validation or locked test solely to
  obscure a negative pilot. Preserve the historical Qwen path untouched.
- Full benchmark: 28,971 FlowER endpoint rows vs 28,967 strict-process rows;
  these denominators are never interchangeable.

## 2026-10-11 product-start pilot status

The original prefix-NLL router in this PR was not equivalent to the Qwen
greedy tool-name decision: on the same frozen 128 validation IDs it yielded
0/128 structural endpoints, whereas pure Qwen yielded 47/128. In that run,
125/128 first calls selected FLOW even though the reference first call was
IMPORT on 126/128. This is a routing-protocol failure, not evidence that the
graph FLOW model or endpoint RL cannot work.

The first opt-in `greedy_name` router generated the quoted tool name, then
started a second Qwen generation for IMPORT arguments. With the same Stage-II
history adapter and 128 IDs, this pilot hybrid reached
46/128 structural endpoints and 114/128 terminal episodes; pure Qwen remained
47/128. Paired counts were 38 both-correct, 8 hybrid-only and 9 Qwen-only.
However, the first IMPORT arguments differed from the pure-Qwen run on 17/128
cases. Thus the 46-vs-47 comparison is **not** a graph-only intervention and
does not establish an endpoint gain or loss attributable to the graph head.
The runtime now generates IMPORT arguments to EOS in the same greedy Qwen call
that selected IMPORT, while stopping FLOW immediately after its name. A real
Qwen3-0.6B + adapter token-level local check found identical full-call and
single-call IMPORT tokens on one divergent pilot ID (`val_201`); this is not
yet a rerun of the 128 cases. The new run contract uses
`single_generate_to_eos_v2` and cannot resume the older two-pass artifacts.
The historical prefix-NLL route remains only a labeled diagnostic ablation.

The train-only 64-anchor graph-head RL task was stopped at user request before
it wrote a training report or policy-head checkpoint. Consequently, there is
**no post-RL validation or full-test result** for this hybrid. Do not report
the stopped job as a completed RL run or submit the prepared post-RL evaluator
without a new verified head and positive effective RL groups. The full-test
runner is prepared but unexecuted; it audits all 28,967 strict test rows and
explicitly records the four unsupported official endpoint rows in the 28,971
official denominator.

The full strict-test comparison is prepared, not run. The pure-Qwen control
uses the unchanged matched-v2 greedy runtime through a separate resumable
eight-shard wrapper (`scripts/run_crn_qwen_pure_baseline.py`). It shares the
same 28,967 strict test IDs, product-only remapping, Stage-II adapter, 40/32/4096
budgets and 384-token full-call limit. The aggregator verifies complete shards,
identical selected IDs, source hash, model revision and adapter SHA before a
paired report. The four nonconserving official rows are explicit unsupported
failures for the graph hybrid; they must not be silently called evaluated graph
predictions or used to assert a pure-Qwen result on those four rows. No prepared
Taiji config is a submitted or completed experiment.

## Limitations of this PR

This PR is a separately importable, opt-in code path. The large private
checkpoints and Taiji CUDA/RDKit-2026.03.4 execution environment are not
mounted into this GitHub connector session. CPU-only unit-contract checks
and source compilation are possible; real GPU throughput, Qwen routing parity,
and endpoint results remain explicitly **unmeasured**.

## Codex continuation and resumable inference (PR83)

The user has authorized an aggressive **execution and debugging handoff**, not
an open-ended theoretical redesign. Follow
[the PR83 Codex Goal](CODEX_GOAL_PR83.md) and keep the
[execution ledger](results/PR83_EXECUTION_LEDGER.md) up to date.
The machine-readable bounded-stage contract is
`configs/auto_research/pr83_execution_contract.json`.
Do not stop at first failed GPU smoke, empty endpoint count, green CI or one
unsuccessful RL iteration. Locate/fix a supported cause and rerun an identical
cohort, preserving all prior artifacts. No silent changes to split, reference
availability or metric definitions.

The product-start evaluator now supports **SHA-locked resume**. It writes
`run_manifest.json` before generation (input data, frozen IDs,
model/adapter/graph weights and code hashes) and appends per-row wall-time,
Qwen/graph call counts and timing. Interrupted runs may be resumed with the
**identical command plus** `--resume`; only a corrupted incomplete
final JSONL tail is quarantined and truncated, while completed rows are
independently verified and never rerun.

For example, after a recoverable interruption to an existing 128-ID evaluation,
rerun the *same* original invocation, adding `--resume` before
its other flags. Never reuse an output directory for a new checkpoint, altered
code or sample.

Graph endpoint RL now also writes `latest_train_state.pt` with the frozen
source/model/code manifest, optimizer, graph policy-head weights, RNG state and
next unprocessed root, every four sampled roots by default. The identical
command plus `--resume` may continue that exact attempt; a code or experiment
change requires a new output directory. No claim of RL completion follows
from this checkpoint alone.

Graph-only endpoint RL still requires actual optimizer and GPU execution:
the current PR adds a train-only implementation, not a measured gain. Codex
must collect true endpoint-positive counts separately from reference replay
and repair experiment execution instead of claiming that loss descent alone
demonstrates endpoint alignment.

**Stopping rule:** final measured paired outcome (positive or honestly
negative after two diagnostic corrections), or an explicit integrity/security/
resource hard blocker with the attempted recoveries and precise unblock path.
