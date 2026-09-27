# EARHO historical task labels (2026-09-26)

The machine-readable [registry](EARHO_HISTORICAL_RUN_REGISTRY.json) enumerates
**all 21 existing EARHO Taiji task specifications** and **all 17 local EARHO
agent/evaluation output roots** found through 2026-09-25. Each existing output
root also has a `HISTORICAL_STATUS.json` sidecar. These labels do not stop a
job, delete an artifact, or assert its current Taiji state.

There are two distinct classifications:

- `confirmed_actor_prefix_mismatch`: the 2026-09-25 FlowER K=2 task
  `meteor_mechet_earho_flower_k2_prefixv2_8a100_qy_20260925_01`, with output
  `outputs/agent/earho_paper_flower_strict_k2_prefixv2_seed17_a100_20260925`.
  Its actor inference inserted an empty Qwen thinking block before a tool call,
  unlike the completed Stage-I/II tool-call SFT rows. Treat its results as
  diagnostic only; neither its rollout nor its adapter is clean Stage-III
  evidence.
- `legacy_pre_v3`: the other 20 task specifications and 16 output roots.
  They belong to earlier protocol/smoke lineages and must not be renamed or
  silently reused as `prefixv3`. This label **does not assert** that each had
  the confirmed 2026-09-25 prefix bug, nor that each task completed.

The five old `earho_paper_*` agent configs retain historical output identities
and explicit pre-v3 prompt markers. The current driver rejects those markers,
so replaying an old Taiji JSON cannot silently start a v3 experiment under an
old task name. New v3 paper configs have separate `*prefixv3*.yaml` filenames
and output paths; no v3 Taiji job has been submitted by this labeling change.
The version-3 prefix and verified-replay code has not yet been merged or
runtime-smoke-tested on GPU. Its tests establish prompt/replay contracts, not
new Stage-III efficacy results.
