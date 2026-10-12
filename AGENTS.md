# Repository instructions for agents

Before answering questions about FlowER dataset size, MECH_PROOF coverage,
existing checkpoints, or before creating/submitting any training or evaluation
job, read `PROJECT_MEMORY.md` completely and verify the referenced manifests
and data contracts.

For **current ICLR experiment priority, active issue state, and the current A7
observation choice**, also read `docs/ACTIVE_ICLR_STATUS.md`. That short status
file supersedes older operational notes in `PROJECT_MEMORY.md` when they
conflict; `PROJECT_MEMORY.md` remains authoritative for historical dataset and
model lineage.

Never use "FlowER full" for the 32k proof subset or the 28k/27k executable
trace subsets. In this repository, unqualified "FlowER full" means the frozen
reaction-level official split with 257,171 train, 2,890 valid, and 28,971 test
rows. Any filtering must be explicitly named and must not silently replace the
requested full condition.

For the opt-in **PR #83 CRN-JEPA + Qwen hybrid** work, **always** read
`docs/CODEX_GOAL_PR83.md` and keep its execution ledger current. The PR's
scientific target is a real, matched product-start endpoint experiment;
passing CPU CI, a first failure, a zero-endpoint pilot, a submitted Taiji task
or a finite training loss must not be treated as task completion. Continue
with bounded, evidence-driven fixes and retries. Preserve benchmark splits,
checkpoint SHAs, executor semantics, resource safety, and raw failed attempts.
Only a measured final result or documented exhausted hard blocker can close
this PR83 execution objective. Do not merge without an explicit user request.

**PR83 RL authority (2026-10-12):** Read `docs/GRAPH_EARHO_METHOD.md` and use `scripts/train_crn_graph_earho.py` for the active graph policy RL experiment. The earlier `train_crn_graph_endpoint_rl.py` samples the last three gold FLOW states and is an **ablation only**. Do not terminate after implementing code/CI; collect authentic first-divergence, successor-class, receding-horizon and paired product-start outcome evidence. Do not claim graph controller fixes Qwen import/finish errors, or treat reference-only replay as RL gain.
