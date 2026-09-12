# Endpoint-process RLVR full-pool status at update 422

This note freezes the first full-pool training telemetry snapshot for the
endpoint-grounded process RLVR implementation in PR #55. It is an optimization
diagnostic, not a held-out benchmark result and not evidence from a completed
checkpoint.

## Run contract

The run uses the complete **FlowER strict-executable train universe** of
257,167 reactions. The official reaction-level FlowER train split contains
257,171 rows; four corrupt, non-executable endpoint rows are named exclusions.
There is no further reaction filtering. Eight disjoint rank shards cover each
of the 257,167 usable reactions exactly once over 32,146 planned updates.

| Field | Value |
|---|---:|
| Task | `meteor_mechet_endpoint_process_rlvr_full_strict_8a100_qy_20260911_01` |
| Instance | `8b1d81f5a08afd1e01a0910cbba90c89` |
| Hardware | 8 x A100-SXM4-40GB |
| Parent | in-place grounded-flow SFT `checkpoint-8037` |
| Candidates per reaction | 8 |
| Global episodes per update | 64 |
| Checkpoint interval | 512 updates |

## Status at 2026-09-12 14:48 CST

The job was still running at capture time and had completed 422 / 32,146
updates (1.31%). It had sampled 27,008 interactive episodes in 15.40 hours,
equivalent to 27.4 updates/hour or 131 seconds/update. At that rate, one exact
coverage would take approximately 48.9 days. No new checkpoint existed because
the run had not reached update 512.

The cumulative strict endpoint rate was 39 / 27,008 = **0.144%**. More
importantly, the online trend did not show improvement:

| Window | Endpoint exact | Formal terminal | Retry exhausted | Committed events | Effective groups | Mean KL |
|---|---:|---:|---:|---:|---:|---:|
| First 50 updates | 0.375% | 64.9% | 31.7% | 1.231 | 97.8% | 0.092 |
| Last 100 updates | 0.016% | 18.7% | 81.3% | 0.204 | 42.5% | 0.201 |
| Last 50 updates | 0.031% | 14.9% | 85.1% | 0.173 | 40.5% | 0.246 |

At update 422, 95.3% of episodes exhausted the retry limit, only 4.7% reached
a formal terminal action, and 125 / 131 event terms were rejected actions.

## Interpretation

The current configuration is not merely slow; its optimization signal is
degenerating. Endpoint successes are too sparse to provide a reliable
group-relative contrast. As successful candidates disappear from most groups,
the optimizer mainly compares different failure modes. Formal termination and
committed-event rates fall while retry exhaustion and policy KL rise.

The runtime cost follows directly from the online protocol. Every optimizer
update samples 64 multi-turn episodes, alternates autoregressive Qwen generation
with synchronous RDKit execution, scores generated event spans under the frozen
reference adapter, and then runs actor backward passes with
`event_microbatch_size=1`. Variable trajectory lengths also force every DDP
rank to wait for the slowest rank. Observed GPU utilization around 55--60% is
consistent with these synchronization and tool-execution bubbles.

The update-422 evidence does not support treating the forthcoming
`checkpoint-512` as an improved model. The next experiment should first change
the rollout batching and credit-assignment boundary, then demonstrate a stable
short-horizon gain before another exact full-pool pass.

The machine-readable snapshot, including the captured log-prefix hash, is
[`results/endpoint_process_rlvr_full_status_update422_20260912.json`](results/endpoint_process_rlvr_full_status_update422_20260912.json).
