# PR83 baseline/train32 diagnostic and joint-import motivation

This is the user-authorized small local test, not a formal benchmark result.
It ran on one Tesla T4 with FP16, RDKit 2026.03.4, seed 17 and the exact PR83
State-SFT Qwen3-0.6B/CRN PR18 parent hashes in the joint design. Tool routing
was `greedy_name`; histories were absent; max decisions/imports/context/import
tokens were 40/32/4096/384. No test split was loaded.

## Observed baseline

Validation: 8 frozen seed-17 rows from the strict validation source; 1/8 exact
structural endpoints, 6/8 accepted terminals. Accepted actions: 22 IMPORT,
34 FLOW, 6 FINISH. Two FLOW actions ended in STATE_CYCLE. Total recorded
reaction time was 160.614 seconds, mean 20.077 seconds/reaction: Qwen route
36.334s, Qwen import 85.681s, graph FLOW 37.551s. These are T4 timings, not
formal A100/H20 speed measurements.

| Source suffix | Terminal | Exact recorded endpoint | Specific observation |
|---|---|---|---|
| val_2533 | yes | no | Predicted benzoic acid route; recorded endpoint contains benzoyl chloride. Different precursor sets, not merely SMILES ordering. |
| val_1256 | yes | no | Principal organic precursor matches. Target-origin H enters HCl instead of the recorded methanol, so current atom-origin multiset scoring rejects it. Heavy-atom-only agreement is a diagnostic, not substituted accuracy. |
| val_2349 | yes | no | Recorded endpoint is a charged addition intermediate; prediction proceeds to the separated amine/chloro-heteroarene. The full endpoint source also records that intermediate. |
| val_1397 | yes | yes | Different initial proton donor/environment but the same terminal structural endpoint. Full-state GT-prefix matching would still block this root. |
| val_2565 | yes | no | Recorded selected target is unchanged. Prediction instead cleaves its mesylate; the prior RL reward additionally rejected even the verified GT terminal for retaining target. |
| val_2667 | yes | no | Methoxide rather than recorded ethoxide is imported at the first decision; the resulting methyl versus ethyl ester remains different through successful execution. |
| val_244 | no | no | Imported context leads to other reactions and then revisits a prior state; executor reports STATE_CYCLE. |
| val_2624 | no | no | Accepted action index9 (zero-based) reaches the recorded Pd+ligand structural projection, but Qwen continues instead of finishing; later cycle rejection prevents a terminal hit. This is not an oracle-stop metric. |

Seven of eight first visible-state divergences are IMPORT/context differences;
one is FLOW. This is not evidence that all seven chemistry choices are wrong:
val_1397 reaches the exact endpoint despite its different environment. The
current graph-only gate uses equality to the entire recorded successor mixture,
so it cannot train the import policy and discards such roots.

## Actual graph-only train32 outcome

`latest_train_state.pt` readback: selected/processed 32/32, autonomous episodes
32, autonomous exact endpoints 6, controller-blocked30, graph-correctable2,
exception-skips2, sampled actions0, endpoint-reward draws0, optimizer updates0,
effective RL groups0. No final trained graph head exists. The script's
zero-update completion guard prevents reporting this as successful RL.

The two eligible rows failed during atom alignment:
`flower_mech_proof_train_249661` with `KeyError:33`, and
`flower_mech_proof_train_199427` with `KeyError:28`. Identity alignment for the
actual first row reproduces the error after successful reference replay:
default RDKit parsing removed an explicit hydrogen used by the event.

## Implemented prerequisites versus proposed work

- **Implemented:** alignment now preserves explicit H; accepted GT-matching
  terminals are no longer vetoed solely for containing the target. Both bugs
  were observed as failing regressions, then fixed. Full suite653pass/8skip.
- **Unchanged:** primary structural multiset scorer, atom-origin definition,
  sample denominator, fresh product-only mapping and executor cycle rules.
- **Proposed, not implemented:** trainable Qwen controller/import LoRA with
  graph heads; joint delegated-action probabilities and shared terminal credit.
  This requires written architecture review and implementation tests before a
  new Taiji training task. No new model gain is claimed.

## Original evidence locations

All original files remain outside Git, unchanged:

```text
/aaa/fionafyang/buddy1/whaleywang/MechET/outputs/eval/pr83_conditional_endpoint_user_smoke_t4_20261012/pre/contract.shard-00-of-01.json
/aaa/fionafyang/buddy1/whaleywang/MechET/outputs/eval/pr83_conditional_endpoint_user_smoke_t4_20261012/pre/results.shard-00-of-01.jsonl
/aaa/fionafyang/buddy1/whaleywang/MechET/outputs/eval/pr83_conditional_endpoint_user_smoke_t4_20261012/pre/summary.shard-00-of-01.json
/aaa/fionafyang/buddy1/whaleywang/MechET/outputs/eval/pr83_conditional_endpoint_user_smoke_t4_20261012/train/train_manifest.json
/aaa/fionafyang/buddy1/whaleywang/MechET/outputs/eval/pr83_conditional_endpoint_user_smoke_t4_20261012/train/training_groups.jsonl
/aaa/fionafyang/buddy1/whaleywang/MechET/outputs/eval/pr83_conditional_endpoint_user_smoke_t4_20261012/train/autonomous_train_episodes.jsonl
/aaa/fionafyang/buddy1/whaleywang/MechET/outputs/eval/pr83_conditional_endpoint_user_smoke_t4_20261012/train/latest_train_state.pt
```

The validation JSONL includes captured model-visible prompts, raw completions,
graph observations, accepted/rejected attempts and state transitions. The
autonomous train journal preserves parsed actions/states, not the full raw
Qwen generation capture; do not claim those missing train completions exist.
No weights, private Ceph init command or credential is committed here.
