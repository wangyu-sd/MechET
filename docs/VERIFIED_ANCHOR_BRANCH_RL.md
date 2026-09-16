# Verified anchor-branch RL

## Purpose

Whole-trajectory GRPO assigns one delayed endpoint score to every generated
electron step. A late error can therefore penalize an earlier correct action.
This condition instead compares alternative first actions sampled from the same
executor state and updates only the tokens selecting that first action.

It starts from the completed formal Python SFT checkpoint:

- `outputs/agent/python_template_slots_qwen3_8b_a100_seed17`
- 257,167 FlowER strict-proof training reactions;
- three SFT epochs and 12,057 optimizer steps.

This post-training sample is not another full-data epoch.

## Algorithm

For a frozen reference trajectory

```text
s0 -> s1 -> ... -> sT
```

the training environment may reset an episode to `s(T-h)`. The policy receives
the original product and the exact executor-produced current state, but never
the reference suffix or expected precursor. It samples `K` complete suffixes
from this shared state. Every suffix is executed and scored against the frozen
endpoint. Candidates with the same canonical first step are pooled before the
local action advantage is computed:

```text
Q(s,a) = mean endpoint reward of candidates whose first action is a
A(s,a) = normalized(Q(s,a) - mean_a Q(s,a))
```

Only generated tokens through the first complete `step(...)` receive this RL
advantage. Later suffix tokens provide a Monte Carlo outcome for the first
action but are not updated by that local advantage.

Reward is endpoint-primary:

- exact structural precursor endpoint: `+1`;
- executable but incorrect endpoint: `0`;
- invalid, truncated or no-op program: `-0.1` constraint penalty.

Formal execution alone never earns positive task reward.
For references whose product-connected scaffold is intentionally unchanged,
the scorer requires full endpoint equality rather than accepting structural
identity; this keeps protonation/auxiliary transformations learnable without
rewarding a bare no-op.

## Adaptive horizon

The initial competence frontier is one remaining step. Training samples from a
small band around the frontier and reserves 20% of groups for full product-only
episodes. The frontier advances only when there are enough informative groups
and group Pass@K exceeds the configured threshold. This avoids a fixed hand-set
`1 -> 2 -> 3` epoch schedule.

Validation is always the original product-only prompt. Intermediate reference
states are a train-only reset mechanism and never change validation or test
denominators.

## Leakage and equivalence boundaries

- Reference steps are used only to construct reset states and optional labelled
  rehearsal examples.
- Reference suffix actions and expected precursor SMILES are absent from RL
  prompts and executor diagnostics.
- Symmetry-equivalent or alternative first actions are not forced to match the
  reference action. They receive credit if their executed suffix reaches the
  correct endpoint.
- Evaluation uses no reset state, reference action or gold-dependent stopping.

## Implementation

- `src/mechet/anchor_branch_rl.py`: reset replay, prompt construction, local
  action pooling, endpoint reward, token mask and adaptive frontier.
- `scripts/anchor_branch_stage.py`: eight independent vLLM collectors and the
  existing memory-efficient clipped policy learner.
- `scripts/run_anchor_branch_rl.py`: restartable round driver, product-only
  validation and curriculum state.
- `configs/agent/python_anchor_branch_rl_a100.yaml`: bounded default run.

Local preparation only:

```bash
python scripts/run_anchor_branch_rl.py \
  --config configs/agent/python_anchor_branch_rl_a100.yaml \
  --prepare-only
```

The Taiji entrypoint streams unbuffered output to the default POD log:

```bash
bash scripts/run_taiji_anchor_branch_rl.sh
```

No Taiji job is implied by adding this entrypoint.

## Method lineage

The implementation combines recent results rather than reproducing an older
reverse-curriculum algorithm:

- [VinePPO (ICML 2025)](https://arxiv.org/abs/2410.01679): Monte Carlo value
  estimation from resettable intermediate states;
- [GiGPO (NeurIPS 2025)](https://arxiv.org/abs/2505.10978): same-state local
  action comparison for agent credit assignment;
- [Horizon-length study (ICML 2026)](https://arxiv.org/abs/2605.02572): horizon
  reduction as a training-stability principle;
- [BEACON (2026 preprint)](https://arxiv.org/abs/2605.06078): segment-local
  credit that prevents distant failure from corrupting earlier decisions;
- [PATR (2026 preprint)](https://arxiv.org/abs/2607.15610): selective branching
  and early termination of uninformative long-horizon rollouts.
