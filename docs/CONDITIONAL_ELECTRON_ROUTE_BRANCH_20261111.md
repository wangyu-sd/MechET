# Conditional Electron-State Route Branching (opt-in code PR)

**Status (2026-11-11):** runnable standalone collector and differentiable
conditional Dr.GRPO objective with CPU contract tests. **Not a GPU-trained
multistep RL result**, not integrated into an upstream target-only route trainer,
and not a replacement for the frozen G=4 root Route-Dr.GRPO experiment.

## Motivation and scientific hypothesis

In the established target-only route objective, a single terminal return is
assigned to all assistant actions in an independently sampled route. With G=4
independent routes from a target, identical returns imply zero centered
policy-gradient advantage; a failure near the end can also obscure which early
FLOW actions were productive. A paired conditional experiment probes whether
comparing different **executed FLOW successors from the same authentic
pre-decision chemical state** gives more informative local credit per unit of
sampled-token cost than branching only after a complete reaction boundary.

The value of the experiment depends on **actual matched-budget improvements**;
the existence of branching does not itself prove finer chemistry is superior.

## Existing upstream reality

- The MechET `main` branch includes `src/mechet/trace_agent_env.py`
  (single-reaction trace-owned executable environment),
  `src/mechet/rlvr.py` (historical RLVR methods), and
  `src/mechet/endpoint_process_rollout.py` (interactive per-event rollouts).
- `docs/target-only-route-rl-design-20261105` describes the **separate**
  multi-reaction Route-RL design, including `TargetOnlyRouteEpisode`, native
  route sampling and `optimize_routes`, but these implementation files are **not
  present in the checked Git branches**. They cannot be silently imported or
  claimed as an integrated backend. If their actual code/checkpoints live only
  on shared storage, add the adapter there when available.
- The initial electronic frontier search uses a frozen one-step actor and is an
  **inference** procedure; its selected candidates are not independent
  on-policy rollout samples and must not be used as such.

## Interface supplied here

`src/mechet/route_branch_credit.py` exposes:

1. `ForkAnchor`: a checkpoint from an **on-policy root rollout before hidden
   evaluation**, retaining target, root rollout ID, decision index, behavior
   checkpoint identity, full private *executor snapshot* and a SHA-256 hash of
   its public observation. Only accepted pre-FINISH FLOW decision sites are
   eligible. The snapshot must include current chemical state, accepted action
   history, pending precursor **occurrences** (not just unique molecules),
   frozen inventory, scheduler and route graph.
2. `choose_anchor`: deterministic hash-based selection independent of any
   hidden reward, reference route or future outcomes. Do not mine reference-
   successful prefixes, then treat branches as unbiased on-policy samples.
3. `collect_conditional_group`: re-initializes a fresh route environment per
   sibling, restores exactly the same snapshot, verifies the *public*
   observation fingerprint, samples independent complete suffixes from the
   **same frozen behavior policy**, and invokes the hidden reward callback only
   after **observed termination**. A resource/protection censor excludes the
   *entire* sibling group from the learner loss. It never assigns R=0 to a
   truncated unfinished episode. Invalid/rejected sampled actions still count
   toward the generated policy-token budget.
4. `conditional_advantages`: within-state, group-centered terminal returns,
   with **no per-group variance normalization**.
5. `conditional_drgrpo_loss`: epsilon-clipped token-level importance ratios
   using stored real behavior logprobs, the frozen SFT reference model for KL,
   and a fixed Dr.GRPO scale `C=1024`. All assistant generated tokens for
   **every suffix tool call** contribute, including sampled termination tokens
   and executor-rejected generated calls; no environment/prompt tokens do.
   Crucially, **prefix tokens do not appear in this conditional objective**.
6. `summarize_groups`: group reward variation, censoring and total model-call /
   generated-token accounting, including discarded groups.

A real route adapter must implement:

```python
class RouteEpisodeAdapter:
    def restore(self, snapshot): ...            # private, full route state
    def public_observation(self): ...            # no reference route/GT
    def step(self, action): ...                  # executes native tool call
    def outcome_status(self): ...               # pending, observed_terminal,
                                                # resource_censored, protected_censored
```

The behavior sampling callback receives **only** `public_observation()` plus an
independent seed. It must return a `PolicyAction` containing original
`prompt_token_ids`, `generated_token_ids`, true frozen-behavior per-token
`behavior_logprobs`, tool name and arguments; the source policy version is
checked against the anchor. The sampler should preserve the already-frozen
native sampling contract (`temperature=1`, `top_p=1`, `top_k=0`); no hidden
reference is allowed to influence source-state selection, stopping or tool
arguments. A malformed tool call is still a sampled action and is passed to
`step()` for typed environment rejection.

The reward callback receives the private terminal environment **only after**
`outcome_status()==observed_terminal`, and should call the original frozen
`HiddenRouteReward.score`, preserving its root-connected partial progress,
full AND/stock/reference-supported completion, and 0 invalid classes.

The loss API expects tensors indexed by `(group_index, sibling_index,
step_index)` corresponding exactly to the recorded *original prompt* of each
native tool call; it must not recompute all steps as one concatenated chat
completion. Behavior and frozen-reference logprobs are kept distinct. The
backend may compute per-call gradients or use memory-efficient chunks, but
must reproduce the exact returned scalar objective/gradients at each fixed
learner parameter value.

## Experiment contract (one optional study, not a new mainline)

Run three conditions under matched sampling/compute budgets:

| Condition | Decision to branch | Primary purpose |
| --- | --- | --- |
| A. Frozen baseline | Original G=4 independent complete target-only routes | Published-style mainline control; remains unchanged |
| B. Boundary conditional | At the same route checkpoint **after** an accepted FINISH | Test whether reaction-granularity conditional credit is sufficient |
| C. Electronic conditional | At a frozen authentic partial state **before** FINISH | Test incremental value of electronic decision granularity |

**This PR implements the generic same-state collector and suffix loss only.**
The production boundary-conditional arm, root sampler and large-scale trainer
integration require the multi-reaction episode implementation and its checkpoint
contract. Do not present B/C as complete GPU experiments.

To compare fairly, report total assistant tokens including root prefixes,
branch suffixes, rejected actions and FINISH; all model calls; executor calls;
explicitly censored groups; group reward variance; wall time; complete
reference-supported AND/stock closure on the frozen 2,986 VALID target
population and the predeclared 460 search targets. Never promote a cherry-
picked pilot success or feed search-selected branches into the on-policy loss.

Selection and branch grouping change the conditional state distribution. The
objective optimizes suffix behavior **conditional on visited on-policy states**,
not an unbiased estimator of the original complete root-trajectory objective.
Do not imply that root GRPO and conditional GRPO losses can be interchanged or
mixed without explicit weighting and a separate ablation.

## Reference papers / code repositories (accurately scoped)

| Role | Paper and/or repository | What is actually borrowed |
| --- | --- | --- |
| **Primary open code reference (read)** | [TreeHCA](https://github.com/TheBlackCat22/TreeHCA) (repo main at `0a140f517b627e86f2b1f3339d92e4c43cfbb9f6`); [paper](https://arxiv.org/abs/2610.12828) | Multi-turn tree rollout/credit engineering motifs, masking, sample-accounting and sampling-bias precautions. Its hindsight-guided sample weights and tree normalization are **not** implemented or copied here. |
| **Loss convention (read)** | [Hugging Face TRL](https://github.com/huggingface/trl) (main at `d115286bc5aef8d346e8510bfe312803c7881c4a`); [GRPO trainer docs](https://huggingface.co/docs/trl/en/grpo_trainer) | Clipped GRPO/Dr.GRPO scaling, frozen behavior/reference separation, token-level loss conventions. This module computes its own small explicit formula for multi-tool suffixes rather than copying TRL trainer code. |
| **Related scientific design, no verified public implementation reuse** | [TriGRPO: Hindsight Credit Assignment over Trees for Agentic RL](https://arxiv.org/abs/2611.02706) | Matched-state conditional branches and sibling outcome comparisons. We do **not** implement its hindsight selection, root/node/cross groups or claim equivalent results. Public official code was not established in this review. |
| **Related paper only** | [CoCA](https://arxiv.org/abs/2610.17733); [repository](https://github.com/gczhu/CoCA) (as checked, it contained only a README) | Outcome-grounded fine-grained credit idea, **not** its criterion-induced distribution-shift machinery, which would require extra model passes. |
| **Prior internal code** | [MechET TraceOwnedAgentEnv](https://github.com/wangyu-sd/MechET/blob/main/src/mechet/trace_agent_env.py), [RLVR utilities](https://github.com/wangyu-sd/MechET/blob/main/src/mechet/rlvr.py) | Native electron-flow executor and token-loss conventions. No action-space or existing experiment changes. |

The source in this PR is **original code**: no third-party files, weights or
large framework snippets are vendor-copied. Naming is descriptive, not a claim
of having reproduced TreeHCA / TriGRPO or another paper's implementation.

## Tests and promotion gates

```bash
python -m pytest -q tests/test_route_branch_credit.py
```

CPU contract tests verify on-policy checkpoint fingerprint, sibling replays,
reward timing, grouped censoring, behavior-version matching, zero-variance
rewards, token logprob shape, actual importance-ratio gradient signs, KL and
fixed normalization. They **do not** validate a real Qwen multi-reaction GPU
integration, actual route chemistry or improvement in NMI benchmark metrics.

Promote into a full training run only after: (i) the actual target-only
multi-reaction adapter is supplied and exercised; (ii) tokenizer/prompt/
logprob equivalence is checked against frozen sampling; (iii) resource-censor
and stock/route replay contracts pass; and (iv) fixed-budget A/B/C
validation produces positive route results. Keep original Route-Dr.GRPO,
search policy and reference isolation unchanged in the meantime.