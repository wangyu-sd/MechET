# Natural-language verified anchor-branch RL

This document records the implementation lineage of Stage III in the current
method.  The paper-facing name is **Execution-Anchored Receding-Horizon
Optimization (EARHO)**.  `anchor_branch` and `successor_horizon` remain stable
code/artifact names; they implement frontier-local branching, verified
successor credit and bounded receding-horizon continuation.  The complete
State-SFT -> compressed-history Trajectory-SFT -> EARHO flow is specified in
`docs/EXECUTION_ANCHORED_RECEDING_HORIZON_OPTIMIZATION.md`.

This is the lineage-correct post-training condition for the completed
natural-language electron-event actor:

- parent: `outputs/agent/natural_language_event_sft_qwen3_8b_a100_seed17_20260913`;
- parent weight SHA-256: `16648e587e084c273c35faee0adcd2486fbdb4f71985d007648421ea5990f3fb`;
- reaction source: `flower_inverse_tool_sft_action_delta_v1`, with
  257,167 / 2,890 / 28,967 strict-proof reactions;
- the post-training run samples train reactions only; test is never loaded.

For a reference trajectory `s0 -> ... -> sT`, the training environment may
reset to an executor-replayed state near the endpoint. The prompt contains only
the original product and this current state. It does not contain the hidden
suffix or expected precursor.

At one shared state, the actor samples multiple first tool calls. Each branch
then follows a gold-free greedy executor loop to a terminal precursor or a
decision budget. Surface-different actions are pooled when execution gives the
same chemical successor. The historical v1 smoke used the sparse reward:

- exact terminal precursor: `+1`;
- executable but wrong terminal precursor: `0`;
- invalid, truncated, or nonterminal rollout: `-0.1`.

The clipped policy update applies only to tokens of the first tool call. Later
rollout actions estimate that action's outcome and receive no copy of its
advantage. Import/finish prompts and electron-event prompts have separate local
baselines because the frozen SFT protocol exposes the molecular inventory only
for electron events.

The initial integration run uses 64 named train reactions, K=8 first branches,
16 disjoint validation reactions, and one update round. It is a real parameter
update gate, not another suffix diagnostic and not a headline test result.

## Endpoint-shaped repair contract

The completed v1 smoke exposed a reward-identifiability failure: among 512
training candidates there was one exact endpoint, 256 wrong terminal endpoints,
and 255 invalid/incomplete trajectories. The update therefore mostly separated
terminal from invalid behavior. Validation execution rose from 50.0% to 62.5%,
while exact endpoint success remained 0/32 candidates.

The v2 repair keeps strict endpoint equality as the only positive outcome. It
adds a private, training-only potential over the frozen precursor:

- atom maps are removed before scoring;
- disconnected components are matched by Morgan similarity, weighted by heavy
  atoms so the synthetic substrate dominates spectator ions;
- exact component overlap contributes a second term;
- hallucinated extra components enlarge the denominator;
- progress of the first executed successor contributes local credit;
- every wrong terminal and every invalid/incomplete trajectory remains below
  zero, while an exact endpoint remains exactly `+1`.

The frozen endpoint is used by the reward only and is never placed in the actor
prompt. The strict benchmark metric is unchanged. Historical v1 configuration
and output paths remain distinct from the v2 repair configuration:
`configs/agent/natural_language_anchor_branch_rl_shaped_smoke_a100.yaml`.

Offline rescoring of the frozen v1 rollouts produces 218 distinct reward values
instead of three. Wrong terminal endpoints span `[-0.473, -0.0602]`, and 96/128
prompt-mode groups contain usable within-group reward contrast. This is a reward
audit, not evidence of improved model accuracy; a new optimizer/evaluation run
is required.

The repair also removes cross-prompt raw-NLL ranking during continuation. Each
action and event prompt now proposes two candidates, executor-invalid and
duplicate-successor candidates are removed, and the frozen
`natural_language_state_value_v2` adapter ranks the surviving states as
productive (`A`), endpoint (`B`), or off-reference (`C`). Actor likelihood is a
small tie-breaker only. The critic receives the product and public current state,
never the expected precursor. Disabling the value adapter and setting one
continuation candidate exactly restores the historical v1 search contract.

## v4 implementation correction: productive gates and verified replay

The first bounded productive run exposed an implementation gap rather than a
change of algorithm. Its interrupted baseline produced 228 candidates over 114
validation reactions: 172/228 were formally terminal, but only 45/228 were
productive and 127/228 retained the unchanged target. One full product-only
trajectory did reach the exact endpoint, confirming that the protocol can
express a complete solution; most failures instead exploited an unproductive
finish/import path.

The v4 correction keeps the same executor-reset anchor-branch algorithm and
adds four safeguards:

- `finish_trace` is rejected while the unchanged target component remains;
- explicit-hydrogen import participants are preserved by the executor instead
  of being silently collapsed by SMILES normalization;
- the private reference first successor supplies bounded local credit, while
  all non-exact rewards remain strictly negative;
- every train reaction contributes one verified first-decision replay record
  after PPO, so the documented replay phase now exists in the training file.

The expected precursor and reference successor remain training-private and are
never rendered into the actor prompt. Per-reaction collector exceptions are
written to sidecars and represented in the evaluation denominator; a single
bad reaction no longer terminates all eight workers, while an aggregate error
rate above 5% still fails the run after preserving diagnostics.

## Successor-anchored receding-horizon repair

The v4 smoke showed that relative normalization is unsafe when every sampled
successor is wrong: it assigns positive advantage to the least-negative member
of an all-negative group.  The repaired contract therefore separates local
transition learning from long-horizon endpoint evaluation.

For each reset state, an actor proposal is locally positive only when execution
reaches the private reference first successor (up to the existing public-state
canonicalization) or when the complete rollout reaches the exact endpoint.
Prompt modes with no such proposal receive zero policy advantage.  They are
recovered by the verified reference replay record rather than by promoting a
wrong action.  Generated executable non-reference successors are persisted as
hard negatives in `successor_value.jsonl`; the corresponding binary critic sees
only target, current state, candidate successor, and terminal status.  It never
sees the expected precursor.

Continuation is no longer a single greedy chain.  After every executed action,
the policy proposes action-mode and event-mode alternatives, chemical-equivalent
successors are pooled, and a bounded beam is reranked.  Multiple states remain
available until the next executor transition, so failure of the current best
branch falls back to another state.  This is a small receding-horizon search,
not full MCTS and not one-shot proof sampling.

The bounded integration configuration is
`configs/agent/natural_language_successor_horizon_smoke_a100.yaml`.  It retains
the frozen A/B/C critic only for the first gate while simultaneously producing
the hard-negative data needed to train the P/N successor critic.  No benchmark
claim is attached until that gate and an unchanged product-only evaluation have
completed.
