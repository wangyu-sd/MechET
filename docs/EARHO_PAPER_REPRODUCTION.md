# EARHO paper reproduction contract

This document freezes the Stage-III code paths needed to reproduce the EARHO
conditions reported in the MechET paper. It is intentionally narrower than the
historical EARHO registry.

## Clean parent contract

All paper reproductions in this matrix use the prefix-v3 actor boundary and the
pinned Trajectory-SFT parent inherited by
`configs/agent/earho_v2_flower_strict_nf4_matched_8h20.yaml`. Historical
pre-v3 artifacts must not be resumed into these output directories.

The full FlowER Stage-III condition is:

`configs/agent/earho_paper_flower_strict_prefixv3_8h20.yaml`

It fixes the paper bounded-horizon reward, verified-positive gating, successor
pooling, fallback supervision, adaptive horizon control, and successor-value
guidance.

For new clean prefix-v3 runs, checkpoint selection uses product-start greedy K=1 validation by default
(`evaluation.candidates_per_reaction: 1`,
`evaluation.checkpoint_metric: group_pass_at_k`). This monitor uses one prompt
mode, one deterministic continuation sample, and beam width one throughout
the episode. Final paper Top-k evaluation is a separate benchmark step and must not be
substituted by this selection monitor. The manuscript does not currently link
the historical 74.1% checkpoint to a frozen selection record, so this clean K=1
rule is a forward reproduction contract rather than a claim about how the
unlinked historical number was selected.

## Matched component configurations

| Paper condition | Reproduction config |
|---|---|
| Full EARHO | `earho_paper_flower_strict_prefixv3_8h20.yaml` |
| Fixed H=2 | `earho_paper_flower_strict_prefixv3_fixed_h2_8h20.yaml` |
| Without successor pooling | `earho_paper_flower_strict_prefixv3_no_pooling_8h20.yaml` |
| Without value guidance | `earho_paper_flower_strict_prefixv3_no_value_8h20.yaml` |
| Without success gate | `earho_paper_flower_strict_prefixv3_no_success_gate_8h20.yaml` |
| Without fallback supervision | `earho_paper_flower_strict_prefixv3_no_fallback_8h20.yaml` |
| Without success gate and fallback | `earho_paper_flower_strict_prefixv3_no_gate_no_fallback_8h20.yaml` |

The fixed-horizon condition freezes the inherited initial frontier at H=2.
The manuscript currently says only "fixed H" and does not state the integer H,
so exact historical reproduction still requires confirming that the reported
row used H=2; the code does not silently relabel another historical run.
The no-gate condition retains the same bounded paper return but uses ordinary
within-group relative advantages, so unsupported actions can compete by return.
The no-pooling condition keeps distinct action realizations during first-action
credit assignment and bounded continuation. The no-fallback condition disables
verified reference replay only when the actor group lacks positive support.

For the full method, fallback supervision replays the **remaining verified
reference suffix from the correction anchor**. Each replay row has its own
public executor state and compact accepted history; future reference actions
are labels, not actor inputs.

## Running one frozen condition

```bash
python scripts/run_earho_v2.py \
  --config configs/agent/earho_paper_flower_strict_prefixv3_8h20.yaml
```

A repeated run must use a fresh output directory. The runner accepts an
explicit seed override so the recorded paper seeds can be replayed without
editing the frozen config:

```bash
python scripts/run_earho_v2.py \
  --config configs/agent/earho_paper_flower_strict_prefixv3_8h20.yaml \
  --seed <RECORDED_SEED> \
  --output-dir <FRESH_OUTPUT_DIR>
```

The current paper labels its three repeats as Seed A/B/C but does not expose
their integer seed IDs in the manuscript repository. This code does **not**
invent replacements. The final reproduction record must attach the actual
integer seeds, resolved config, parent-adapter SHA-256, actor/critic checkpoint
SHA-256 values, and evaluator revision for each reported repeat.

## Full executable-view validation

The full validation helper resolves the same `extends:` inheritance as the
training runner:

```bash
python scripts/eval_earho_v2_validation.py \
  --config configs/agent/earho_paper_flower_strict_prefixv3_8h20.yaml \
  --actor <ACTOR_DIR> \
  --critic <CRITIC_DIR> \
  --frontier <FROZEN_FRONTIER> \
  --output <FRESH_VALIDATION_OUTPUT>
```

The validation plan records source/history hashes, actor and critic hashes,
frontier, beam width, and candidate count. Rollout collection additionally
binds the resolved vLLM runtime marker and actor/critic weights into its lineage
hash, preventing stale collections from being reused after a runtime or
checkpoint change. The identity uses the pinned Ceph runtime source and marker
hash, not the random local extraction directory, so a restarted task can reuse
an otherwise identical completed collection.

## Remaining paper-provenance blockers

The current manuscript includes an `Outcome-only updating` Stage-III control
starting from Trajectory-SFT, but neither the paper nor the inspected code
history freezes its complete operational contract (frontier reset policy,
success gate, fallback replay, successor critic use, and horizon schedule).
PR #59 documents an earlier endpoint-primary anchor-branch algorithm, but its
parent lineage predates the current Trajectory-SFT condition. This reproduction
PR therefore does not manufacture an "outcome-only" config by guessing those
choices.

Before calling the paper table exactly reproduced, the experiment record must
supply:

- the outcome-only Stage-III resolved config and checkpoint lineage;
- the integer H used by the reported fixed-H row;
- the integer seeds corresponding to manuscript Seed A/B/C;
- the checkpoint-selection record used for the currently typeset EARHO result;
- final candidate/evaluator artifacts linking each reported number to a clean
  prefix-v3 run.

## Claim boundary

These configs reproduce the Stage-III training and validation contracts. They
do not by themselves establish the numerical values currently typeset in the
paper. A paper number is reproducible only after its completed run artifacts,
checkpoint hashes, candidate outputs, and final evaluator record are linked to
the corresponding condition above.
