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

Unless the user explicitly requests a smoke test, do not run or submit smoke,
pilot, or small-subset training/evaluation jobs. Use the requested full protocol
and denominator. Unit tests and static checks remain normal code verification,
but must not be presented as experimental results.

For the opt-in **PR #83 CRN-JEPA + Qwen hybrid** work, **always** read
`docs/CODEX_GOAL_PR83.md` and keep its execution ledger current. The PR's
scientific target is a real, matched product-start endpoint experiment;
passing CPU CI, a first failure, a zero-endpoint pilot, a submitted Taiji task
or a finite training loss must not be treated as task completion. Continue
with bounded, evidence-driven fixes and retries. Preserve benchmark splits,
checkpoint SHAs, executor semantics, resource safety, and raw failed attempts.
Only a measured final result or documented exhausted hard blocker can close
this PR83 execution objective. Do not merge without an explicit user request.
