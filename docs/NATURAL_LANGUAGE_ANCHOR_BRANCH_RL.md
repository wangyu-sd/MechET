# Natural-language verified anchor-branch RL

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
same chemical successor. Endpoint exactness supplies the task reward:

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
