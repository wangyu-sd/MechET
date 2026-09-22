# EARHO v2 implementation gate

This worktree implements the Stage-III contract in the current MechET paper,
starting from a **compressed-history Trajectory-SFT** adapter rather than the
historical v1 State-SFT adapter.

## Training flow

1. `scripts/run_earho_v2.py` validates the pinned current-compiler executable
   31k trace view and the completed Stage-II adapter. The executable training
   denominator is 10,152 reactions; the complete endpoint denominator is
   24,959/3,120/3,120 and is not relabelled as program coverage. No test rows
   are loaded for post-training.
2. Each collector performs a product-start v2 policy rollout with the real
   executor. `scripts/earho_v2_protocol.py` independently replays the private
   reference decisions, compares *executed* canonical successors, and returns
   the first consequential divergence. An exact terminal precursor creates no
   correction frontier.
3. The branch collector resets to the independently replayed reference state
   and its accepted-action history. The actor sees only the product, current
   public state, v2 molecular inventory and compact history. One candidate
   action is executed before bounded continuation. Equivalent chemical
   successors are pooled regardless of action spelling.
4. A branch gets positive local credit only for a reference-equivalent first
   successor or an exact terminal endpoint. An all-negative group receives
   zero policy advantage and a verified replay transition. A fraction of
   training examples still use full product-start rehearsal. The continuation
   horizon expands only after sufficient verified productive support.
5. `scripts/build_earho_v2_successor_value.py` builds private train-only P/N
   labels from reference successors and distinct executable actor successors.
   Each round updates a separate P/N successor-value adapter, then the actor;
   subsequent bounded search scores `(product, current state, candidate
   successor, terminal)` without a reference endpoint or gold horizon.

The operational config is
`configs/agent/earho_v2_mech_uspto31k_8a100.yaml`. Its parent adapter SHA
contains an explicit placeholder and must be pinned to the completed Stage-II
weights before any Taiji submission. The config is a bounded 640-reaction RL
campaign, **not** a full 10,152-reaction RL epoch. A real GPU integration
smoke (vLLM sampling, PPO update, critic update, resume, product-only
validation) is still required before a full campaign is scientifically usable.

Local checks completed: 27 protocol/legacy regression tests, 25 reference
replays from each 31k split, 4-train/3-valid preparation dry run, and a
successor-critic `train_tool_sft.py --dry-run`. These are not evidence of
improved endpoint accuracy.
