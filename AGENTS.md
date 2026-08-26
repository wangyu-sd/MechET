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
