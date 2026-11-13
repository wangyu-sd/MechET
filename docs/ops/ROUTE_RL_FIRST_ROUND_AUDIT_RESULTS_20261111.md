# Original Route-RL first-round reward replay and finer scoring records

The original frozen scorer reproduced all256 saved trajectory reward scalars and
complete reward-evidence dictionaries exactly. This is the first saved-report
audit deliverable of PR104; independent electron-executor/participant-role audits,
real4H20 phase profiling and formal training restoration remain incomplete.

## Source and protocol

Actual submitted trainer source:
`/aaa/fionafyang/buddy1/whaleywang/MechET/artifacts/external_route_stage2_dr_grpo_20261106_03/code`.
The source contains `TargetOnlyRouteEpisode`, `HiddenRouteReward`,
`route_policy_sampling.py` and `optimize_routes`; the older per-event RLVR trainer
was not used. The replay verifies all frozen source hashes plus the stock,
reference database and manifest bindings before importing that original scorer.

Original contract:
`/aaa/fionafyang/buddy1/whaleywang/MechET/outputs/submissions/external_route_stage2_dr_grpo_20261106_03/contract.json`,
SHA256 `2433c2d477dd97ba35db3960472df8b1f76809c365943d5e37aafffb75784f11`.
Actor: corrected Stage-II History89442, adapter SHA256
`2c98a9bff3afeaffbf2ee2733885cccc68e936f9310fd9f46ba4ee2211e5f453`.
The intended training condition remains23730 TRAIN/2986 VALID, independentG4,
seed17, one epoch. Auditing an existing first complete round is offline artifact
analysis; no smaller training or chemical-evaluation job was submitted.

## Measured saved-report results

| Quantity | Reverified value |
| --- | ---: |
| Target groups / trajectories | 64 /256 |
| Decision records / completed reaction occurrences | 5030 /622 |
| Structural stock closure / reference-supported closure | 74 /9 |
| Informative centered-advantage groups | 12 /64 =18.75% |
| Censored groups | 0 |
| Assistant tokens in saved policy receipts | 659577 |
| Policy tokens with nonzero centered advantage | 144884 =21.9662% |
| Reward/evidence reconstruction mismatches | 0 |

Nonzero advantage is not proof of a nonzero effective clipped gradient. Zero-PG
advantage groups still contribute KL computations. Saved policy-receipt tokens
are not a measurement of billing or unpublished/retried backend generations.
The historical2888.139s sampling wall is cold first-round evidence, not an
observed steady-state distribution or a new throughput measurement.

## Finer scoring format

Each trajectory now has diagnostic fields for execution receipts, completed
reaction occurrences, trusted reference progress, structural closure, reference
support, remaining precursor occurrences and the unchanged scalar reward. The
group ledger preserves rewards, centered advantages, whole-group censoring,
generated work, policy/KL token denominators and nonzero-advantage token counts.

The220 zero-reward trajectories separate as follows:

| Diagnostic category | Count |
| --- | ---: |
| Invalid route graph according to the frozen scorer | 79 |
| Completed reactions without trusted reference progress | 70 |
| Structural closure without full reference support | 56 |
| No completed reaction | 15 |

Across all256 trajectories there are65 structural closures without full reference
support,18 trusted partial but unclosed routes and9 reference-supported complete
routes. Some unsupported closures retain partial credit, so65 is not the count
of zero-reward unsupported closures. Unknown reference coverage is kept separate
from chemical invalidity. Recorded execution acceptance and state changes are
observations; they do not establish useful route progress or feasibility.

This change does not create a denser training reward. A future reward condition
should separately evaluate verified reaction/route progress and its effect on
G4 advantages. It must not automatically reward every accepted FLOW, expose
hidden references to the actor, or replace the frozen baseline in PR104.

## Reproduction on the shared host

Both scripts write new output paths and reject overwriting an existing result.
Private scorecards, molecule structures, trajectory IDs and hidden reward indices
stay on Ceph. Only this aggregate summary and code are committed.

```bash
source /aaa/fionafyang/buddy1/whaleywang/MechET/configs/runtime/shared_host_storage.env
REPO=/aaa/fionafyang/buddy1/whaleywang/MechET-earho-publication-20261026
EVIDENCE=/aaa/fionafyang/buddy1/whaleywang/datasets/mechet_route_rl_efficiency_audit_20261111
PIN=/aaa/fionafyang/buddy1/whaleywang/MechET/outputs/shared_host_storage/tmp/meteor-pr96-mcts.83nJdu
export PYTHONPATH="$PIN/runtime:$PIN/rdkit:/root/miniconda3/envs/meteor/lib/python3.11/site-packages"
# Pick unused output filenames for a fresh rerun.
/root/miniconda3/envs/meteor/bin/python -S -B "$REPO/scripts/replay_route_first_round_rewards.py" \
  --output "$EVIDENCE/reward_replay_new.json"
/root/miniconda3/envs/meteor/bin/python -S -B "$REPO/scripts/audit_route_reward_scorecards.py" \
  --audit "$EVIDENCE/reward_replay_new.json" --output "$EVIDENCE/scorecards_new.json"
# CPU software checks, using an unused Ceph basetemp directory.
PYTHONPATH="$REPO:/aaa/fionafyang/buddy1/whaleywang/MechET/outputs/shared_host_storage/test_tools/pytest_portable311.zip:/root/miniconda3/envs/meteor/lib/python3.11/site-packages" \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /root/miniconda3/envs/meteor/bin/python -S -B -m pytest -s -q \
  --basetemp "$TMPDIR/pr104_scorecards_new" "$REPO/tests/test_route_reward_scorecards.py"
```

Software checks:14 unit/contract cases passed, including zero-advantage KL-token
accounting, whole-G4 censoring, unsupported-versus-invalid classification, finite
reward/behavior receipts, original identities, changed input hashes and rejection
of a relabelled smaller first-round population, atomic no-clobber publication and
owner-only permissions for complete private reports. The original witness is
SHA-bound alongside the saved reports. RED failures were retained before
the implementation. Both scripts were also executed against the complete original
64 groups/256 trajectories; their aggregate counts match the original witness.
Full private artifact/code hashes are listed in
`configs/ops/route_rl_first_round_reverified_20261111.json`.

## Separate software acceleration evidence

The earlier complete saved64+8 H20 check measured26.15267s legacy versus22.60101s
HF BF16,1.1571x for that paired saved workload only. This is one software timing
comparison, not the requested4H20 steady-state Route-RL throughput result. BF16
learnerB1 passed; B2/B4 were excluded for gradient differences above the unchanged
0.02 threshold and slower execution. No new optimizer update is claimed here.
