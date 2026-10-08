# Superseded design note

This document previously explored endpoint-only and auxiliary-supervision
variants for a JevRetro spin-off. That direction is no longer the active
journal protocol.

The active design keeps **MechET** and the established three-stage curriculum:

1. State-SFT;
2. compressed-history Trajectory-SFT;
3. EARHO.

New journal experiments use Qwen3-0.6B as the primary reproducible model,
Qwen3-1.7B as a later confirmation scale, and Qwen3-8B only as historical
lineage evidence.

See:

- `docs/RELIABLE_MECHET_THREE_STAGE.md`
- `configs/experiments/reliable_mechet_three_stage_v1.yaml`
- `configs/agent/natural_language_event_v2_qwen3_0_6b.yaml`
- `configs/agent/natural_language_history_v2_qwen3_0_6b.yaml`
- `configs/agent/earho_reliable_mechet_qwen3_0_6b.yaml`

The endpoint-program utilities introduced on this branch remain available as
controls/audits, but they do not define the main method.
