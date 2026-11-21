# CbyD evaluation for PR117

Run the nine original preregistered targets first. The expansion contains 18 additional targets. Combine the two disjoint sets for a 27-target result if the expansion is admitted. Select by frozen source ID rather than row position so the nine primary targets retain exactly their original identity and order. This preserves the released complex cases spanning 30–134 heavy atoms and published route depths of 6–38. [CbyD release](https://github.com/chemistry-by-design/cbyd)

The policy input file contains the canonical target structure, an opaque ID and a fixed seed. The literature mapping, publications and reference precursor signatures are available to the evaluator alone. The prepared condition uses the corrected History-89442 actor, the method selected at the Phase A freeze, and the larger official PaRoutes inventory `n1-n5-stock`. The same per-target ceiling of 64 native decision calls gives ceilings of 576 for the initial run and 1,152 for the additional targets. A complete expansion requires a total of at most 1,728 calls across both runs. The scheduled window is Nov 29 to Dec 03.

## Outcomes

| Outcome | Required evidence | Denominator |
| --- | --- | --- |
| Productive first reaction | Native executor replay commits a reaction at the target root. | Every target in the admitted cohort |
| Exact first disconnection | The first committed root precursor multiset agrees exactly with the frozen literature structure identities, including stereo and multiplicity. | Every target in the admitted cohort |
| Exact disconnection found | An executed root commit agrees with the same reference within the common native call prefix. Record the first matching call index. | Every target in the admitted cohort |
| Full inventory closure | Native replay independently discharges every precursor occurrence against the official inventory after one or more committed reactions. | Every target in the admitted cohort |

Retain alternative committed structures, incomplete outcomes and execution rejections in the denominator. The root-level literature endpoint measures a known disconnection. The inventory endpoint measures full routes, including valid alternatives. Report structural coverage with the exact source audit, including the nine constituent source steps that lack complete mapped target-heavy-atom coverage. The published literature tree is reference evidence outside the policy. [CbyD Methods and Appendix S1.7](https://arxiv.org/pdf/2611.24996)

Use the optional scorer argument `--source-trees` to report a stereochemistry diagnostic from the pinned literature tree. It distinguishes an exact first commitment, matching precursor connectivity with different stereochemistry, different connectivity, and absence of a productive root commitment. The comparison preserves precursor multiplicity, formal charge and isotopes. Record the first connectivity match at the same physical call prefix and retain the exact stereochemical match as the main literature metric.

## Reproduction inputs

- `python -m scripts.pr117_cbyd_package` builds the two scheduler bundles from the frozen Reference launch package and the prepared CbyD split. Use `--parent-launch`, `--cbyd-plan`, `--cbyd-plan-sha256` and `--output`. Each bundle has its own task flag, target-only payload and one-A100 execution command. `bash worker.sh --validate` verifies the isolated policy admission. During the registered window use the existing `scripts.pr117_admit_allocation` command for the assigned worker.
- Run `python -m scripts.pr117_cbyd --help` in the verified native runtime. `--validate` binds the existing Reference registration, the CbyD target-only file and its SHA-256. Live execution requires a separate, legitimate GPU allocation receipt.
- Run `python -m scripts.pr117_cbyd_score --help` using the same benchmark binding and the two evaluator-only digests. The scorer independently replays the accepted worker's full physical journal and refuses partial or mismatched results. Supply the existing `audit_reference_receipts.py` and `cbyd_first_disconnection.py` modules on `PYTHONPATH` from the archived PR117 run.
- The shared run's `cbyd_native_evaluation_prepared_20261122/prepared.json` records all three source views, policy admission and separate evaluator identifiers. The cumulative view verifies the full denominator. Use the initial and additional views to avoid repeated model calls.
- To publish the cumulative score, supply the original initial policy and its results to the scorer. Add `--merge-prepared` and `--merge-prepared-sha256` for the recorded split, `--additional-results` for the second run, and `--cumulative-policy` for `cbyd_expanded_targets_only_20261121.json`. The scorer verifies both native runs separately and publishes the two cohort totals plus the 27-row cumulative result in its frozen order. The cumulative policy file supplies identities and seeds to the scorer.
