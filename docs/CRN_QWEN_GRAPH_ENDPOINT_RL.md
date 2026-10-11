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
- If progress is negative or no positive RL groups exist, do not extend to
  2,890 validation or full-test. Preserve the historical Qwen path untouched.
- Full benchmark: 28,971 FlowER endpoint rows vs 28,967 strict-process rows;
  these denominators are never interchangeable.

## 2026-10-11 product-start pilot status

The original prefix-NLL router in this PR was not equivalent to the Qwen
greedy tool-name decision: on the same frozen 128 validation IDs it yielded
0/128 structural endpoints, whereas pure Qwen yielded 47/128. In that run,
125/128 first calls selected FLOW even though the reference first call was
IMPORT on 126/128. This is a routing-protocol failure, not evidence that the
graph FLOW model or endpoint RL cannot work.

The opt-in `greedy_name` router instead lets Qwen generate just the quoted
tool name greedily, then delegates FLOW arguments to the graph model. With
the same Stage-II history adapter and 128 IDs, this corrected hybrid reached
46/128 structural endpoints and 114/128 terminal episodes; pure Qwen remained
47/128. Paired counts were 38 both-correct, 8 hybrid-only and 9 Qwen-only.
This establishes comparable pilot performance, not a gain. The historical
prefix-NLL route remains available only as a labeled diagnostic ablation.

The train-only 64-anchor graph-head RL task was stopped at user request before
it wrote a training report or policy-head checkpoint. Consequently, there is
**no post-RL validation or full-test result** for this hybrid. Do not report
the stopped job as a completed RL run or submit the prepared post-RL evaluator
without a new verified head and positive effective RL groups. The full-test
runner is prepared but unexecuted; it audits all 28,967 strict test rows and
explicitly records the four unsupported official endpoint rows in the 28,971
official denominator.

## Limitations of this PR

This PR is a separately importable, opt-in code path. The large private
checkpoints and Taiji CUDA/RDKit-2026.03.4 execution environment are not
mounted into this GitHub connector session. CPU-only unit-contract checks
and source compilation are possible; real GPU throughput, Qwen routing parity,
and endpoint results remain explicitly **unmeasured**.
