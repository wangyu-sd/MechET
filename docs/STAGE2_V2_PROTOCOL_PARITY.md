# Stage-II protocol-v2 parity repair

## Why this repair is required

Historical MechET experiments repeatedly showed that apparent model failures
can be caused by a train/inference contract mismatch rather than the inverse
electron-flow task itself.

For Stage-II specifically, the previous closed-loop smoke is not valid evidence
for protocol-v2 error accumulation because it combined multiple historical
contracts:

- the evaluated history `checkpoint-10000` belonged to the protocol-v1
  compact-history lineage rather than the clean v2 Stage-II lineage;
- the launcher explicitly enabled `--legacy-dual-prompt`, while protocol-v2
  trains one unified inventory-bearing prompt over import, electron-flow and
  finish actions;
- the historical smoke used `max_decisions=12` and `max_imports=8`, while
  the frozen v2 runtime contract is 40 decisions and 32 imported copies;
- later EARHO provenance work found that ordinary Qwen
  `add_generation_prompt=True` could insert an empty thinking block before a
  tool call that was absent from completed Tool-SFT rows.

These diagnostics remain useful for historical debugging but must not be used
to conclude that protocol-v2 Trajectory-SFT intrinsically suffers severe
long-horizon error accumulation.

## Matched-v2 contract

The new `--matched-v2` mode in
`scripts/run_natural_language_value_search.py` fails closed unless:

- one unified inventory-bearing observation is used;
- legacy dual-prompt inference is disabled;
- decision/import budgets are at least 40/32;
- branching is 1 and both beam widths are 1;
- no value adapter or value weight is present;
- the policy adapter manifest declares the expected v2 environment revision;
- the frozen executor revision and immutable Qwen3-8B base revision are present.

Generation is greedy K=1 and uses the exact ChatML assistant boundary preceding
a completed SFT tool call, not the ordinary Qwen thinking generation prefix.

## Token-level parity audit

`scripts/audit_stage2_v2_protocol_parity.py` rebuilds protocol-v2 State-SFT
and compact-history Stage-II rows from frozen validation reactions. For every
decision it checks:

1. State-SFT and Trajectory-SFT tool schemas and decision contracts;
2. runtime reconstruction of the compact accepted-action history;
3. exact equality of the runtime Stage-II user prompt and training user prompt;
4. token-prefix equality between matched inference and the completed SFT
   tool-call conversation.

The audit uses RDKit 2026.03.4 and refuses v1 adapter manifests.

## Immediate validation experiment

Run:

```bash
python scripts/submit_taiji_with_donor_init.py submit \
  --config configs/taiji/meteor_mechet_stage2_v2_matched_valid256_8a100_qy_20260929.json \
  --donor-task <successful-taiji-task-with-private-ceph-init>
```

The Taiji job first runs the parity audit on 256 deterministic FlowER validation
reactions. Only if that passes does it evaluate:

- **State-SFT v2:** product-start, no history;
- **Trajectory-SFT v2:** product-start, runtime-reconstructed compact history.

Both conditions use the same 256 reaction IDs, Qwen revision, executor,
40/32 budget, greedy K=1 generation and no search/value critic.

The result is written to:

`outputs/eval/stage2_v2_matched_valid256_20260929/matched_v2_evaluation.json`

This is a validation diagnostic, not a test-set headline result.

## Interpretation

The first question is not whether a new RL method improves MechET. It is:

> under a truly matched protocol-v2 runtime, how large is the autonomous
> product-start gap and does compact history help?

If the matched State-SFT policy is already strong, Stage-II history may be
unnecessary. If Trajectory-SFT gives a reproducible gain, history is useful.
Only if both local/oracle-state competence is high and matched autonomous
rollouts still collapse with horizon is an on-policy recovery method such as
EARHO justified by the evidence.
