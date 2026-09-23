# EARHO v2 implementation gate

This worktree implements the Stage-III contract in the current MechET paper,
starting from a **compressed-history Trajectory-SFT** adapter rather than the
historical v1 State-SFT adapter.

## Training flow

1. `scripts/run_earho_v2.py` validates the selected strict-executable trace
   view and its completed Stage-II adapter. The mech-USPTO-31k executable
   denominator is 10,152/1,319/1,253; its complete endpoint denominator is
   24,959/3,120/3,120. The FlowER strict-executable denominator is
   257,167/2,890/28,967; the official unfiltered reaction split is
   257,171/2,890/28,971. Neither trace view is relabelled as the full
   reaction benchmark. No test rows are loaded for post-training.
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

The operational configs are
`configs/agent/earho_v2_mech_uspto31k_8a100.yaml` and
`configs/agent/earho_v2_flower_strict_8a100.yaml`. Both parent adapter SHA-256
values are pinned to the completed Stage-II weights. Each config selects 640
distinct RL-train reactions in five 128-reaction rounds with K=8, plus a
separate 128-reaction validation monitor. These are **bounded campaigns**, not
full RL epochs or benchmark test runs. The source is sampled by line number
without materializing the multi-GB FlowER JSONL in memory. A real GPU
integration smoke (vLLM sampling, PPO update, critic update, resume,
product-only validation) is still required before endpoint improvement can be
claimed.

Local checks completed: 24 current protocol/regression tests, strict manifest
and adapter SHA checks, and full 640-train/128-validation reference-replay
preparation on both datasets. These are not evidence of improved endpoint
accuracy. The 2026-09-23 Taiji submissions are recorded in the respective
`configs/taiji/meteor_mechet_earho_v2_*` task configs; platform `start success`
alone is not proof of an allocated GPU or an optimizer update.

Operational retry on 2026-09-23: both ordinary A100 candidates remained in
resource waiting. The two ordinary H20 candidates allocated GPUs, but their
initial instances were stopped before any policy update because copying the
entire vLLM runtime directory from Ceph stalled on many small files. The
replacement launcher extracts the same pinned runtime from a single pruned
archive (only tests, RLlib and Python bytecode caches are omitted):
`artifacts/taiji_vllm_runtime/vllm_0_8_5_torch_2_6_cu124_py311_pruned.tar.zst`,
SHA-256 `60fcd6f2f2454e55cd17c70fb483514a1e043ad9da97621ccd8d717220783a65`.
Its zstd stream and completion marker were verified before the replacement
instances were started. All H20 preparation plans were replay-verified locally;
check the new instances' POD processes and GPUs before reporting RL progress.
