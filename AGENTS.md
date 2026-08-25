# Repository instructions for agents

Before answering questions about FlowER dataset size, MECH_PROOF coverage,
existing checkpoints, or before creating/submitting any training or evaluation
job, read `PROJECT_MEMORY.md` completely and verify the referenced manifests
and data contracts.

Never use "FlowER full" for the 32k proof subset or the 28k/27k executable
trace subsets. In this repository, unqualified "FlowER full" means the frozen
reaction-level official split with 257,171 train, 2,890 valid, and 28,971 test
rows. Any filtering must be explicitly named and must not silently replace the
requested full condition.
