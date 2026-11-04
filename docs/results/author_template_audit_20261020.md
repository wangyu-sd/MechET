# FlowER author-template audit: representation and radical/coordination repairs

This records an offline re-analysis of the same 189,233 frozen events. No new
model inference, training, executor changes, dataset filtering or platform job
was performed. Related experiment context: PR #89.

## Model lineage

All four observed cases below use **FlowER-trained Qwen3-0.6B Stage-II compressed
history SFT**, not a Stage-III RL checkpoint and not a 31k-specific adapter.

- Base: `Qwen/Qwen3-0.6B`, revision `c1899de289a04d12100db370d81485cdf75e47ca`.
- Adapter: `/aaa/fionafyang/buddy1/whaleywang/MechET/outputs/agent/natural_language_event_history_v2_qwen3_0_6b_seed17`.
- Adapter SHA: `40204c7a059d16976fc4b24b315c3bdde7bf61c705d24bbde515071757e9de75`.
- History contract: `natural_language_electron_event_history_v2`.
- Original R1/R2 and both 31k rolling/Ray manifests agree on the adapter SHA.
- The 31k runs are the frozen `rxn_prod_min`-proxy **diagnostic** condition, not
  a corrected principal-product benchmark. The input for test:1491 is isobutene.

## Repairs and acceptance rule

1. v2 resolves aromatic/Kekule and implicit/explicit-H representation differences
   using the original author's H expansion and complete mapped-state replay.
2. v3 removes the blanket radical-count-change rejection. A transition must still
   replay an author rule to the exact complete target; radical information,
   original mapped atoms, isotope, charge and real bond changes are not erased.
3. Identifiable mapped proton transfers plus a residual core may have a
   multi-rule proof. Every component and every chain boundary is checked against
   full states, using one catalog version. Composition is reported separately.
4. Historical positive matches are retained. Unmatched and unsupported cases are
   not automatically labelled chemically wrong.

Pinned author source: [jfjoung/Mechanistic_dataset](https://github.com/jfjoung/Mechanistic_dataset),
commit `8bd5b38e9602c4d45485bf50076d03c292dcc486`. The exact generator revision for
the frozen local FlowER archive is not established. Current and dated template
catalogs remain separate; the headline below does not use their union.

## Current author catalog results

| Population | Event denominator | v2 single | v3 single | v3 single or composed |
|---|---:|---:|---:|---:|
| FlowER strict-test GT | 117,365 | 96.352% | 99.520% | 99.621% |
| Saved model events, all six conditions | 71,868 | 80.272% | 80.667% | 80.713% |
| 31k MCTS diagnostic | 27,886 | 78.950% | 79.212% | 79.255% |
| R1/R2 MCTS | 27,402 | 77.082% | 77.841% | 77.888% |

GT single matches increase by 3,719, from 113,083 to 116,802. A further 118 GT
events have composition-only proofs. Remaining 445 unsupported-current-catalog
GT events: 72 recorded free-H import cases, one other radical transition, and
372 closed-shell transitions. These are unresolved diagnostic categories, not
chemical-negative ground truth.

| Saved single-reaction electronic programs | Programs | All events single-or-composed compatible |
|---|---:|---:|
| FlowER strict-test GT | 28,967 | 28,645 / 28,967 = 98.888% |
| 31k MCTS diagnostic | 19,194 | 12,840 / 19,194 = 66.896% |
| R1/R2 MCTS | 11,480 | 6,879 / 11,480 = 59.922% |

The official FlowER reaction test has 28,971 rows; the strict trace universe
explicitly excludes four named upstream-corrupt endpoints. Model programs are
the frozen complete saved program pool, not all rejected/incomplete proposals
and not a test-reaction denominator. Program counts must not be confused with
complete stock-connected multistep routes.

These are **local author-transition compatibility** measurements, not endpoint
accuracy, reaction-condition feasibility, literature support, or independent
chemical truth. GT sensitivity does not establish chemical precision on
out-of-distribution model branches.

## Four observed unsupported branches

[Raw cases JSON](author_template_audit_20261020_cases.json) contains complete
before/after states, inverse arrows, full native action contexts (imports through
FINISH), audit records, native manifests, source paths and verified tree hashes.
The downloadable artifact also includes all four original full tree files.

| Case / native context | Observed behavior | Interpretation, not a binary ground-truth label |
|---|---|---|
| 31k test:1491 / 19 | Imports DMF, cleaves a C-C bond of isobutene and transfers a methyl group to DMF N; subsequent water addition creates a carbon carrying -2 charge before later proton transfer | High-risk charge/speciation path. A locally ordinary last proton transfer does not validate the earlier branch. Old min-proxy diagnostic input is explicitly retained |
| RetroCast n5-07586 / 197 | Fluoride involvement breaks a ring C-N bond of a protonated pyridine, producing an open-chain imine-containing structure | Ring opening, not merely conventional aromatic F substitution. Needs specific ring-opening chemistry/conditions; not declared impossible |
| PaRoutes-derived n5-02804 / 42 | Bromoazetidine + HBr is transformed into dehalogenated ring + free Br+ + Br-; the endpoint retains Br+ | Free electrophilic bromine source/speciation is unexplained. Formal charge conservation is not a solution-phase synthesis certificate |
| AGN-190205 / 236 | `Ar-C(O-)(OH2+)-Cl -> Ar-COCl + H2O` restores the carbonyl and loses water | The local forward counterpart resembles familiar acid-chloride hydrolysis addition. Unmatched does not imply chemically wrong. The full endpoint also retains n-BuLi and water, requiring condition/addition-order clarification |

Chemical context for the last case: [OpenStax acid-halide chemistry](https://openstax.org/books/organic-chemistry/pages/21-4-chemistry-of-acid-halides)
describes acid-chloride hydrolysis via a tetrahedral intermediate. The
[manufacturer SDS](https://www.sigmaaldrich.com/US/en/sds/aldrich/20164) documents
n-BuLi's water reactivity. Thus the whole recorded endpoint should not be
interpreted as an automatically compatible simultaneous reagent mixture; the
record does not supply a validated staged-addition/workup schedule.

## Code, complete results and validation

- [v3 code and instructions](../../outputs/paper/flower_author_template_audit_20261020_v3/README.txt).
- [Final summary](../../outputs/paper/flower_author_template_audit_20261020_v3/results_final/summary.json).
- [Final verification](../../outputs/paper/flower_author_template_audit_20261020_v3/results_final/verification.json).
- [Complete artifact release](https://github.com/wangyu-sd/MechET/releases/tag/analysis-author-template-audit-20261020-v3).
- Asset: `mechet-author-template-audit-20261020-v3.tar.gz`, with adjacent SHA256.
  Contains the full frozen event inputs, historical and corrected event outputs,
  author catalogs/source references, program-event links, diagnostics, analysis
  code, original native case trees and inference manifests. No model weights.

New-entry tests: 28 passed, including 16 representation regressions inherited at
the new entry point. Historical v2 tests: 20 passed (48 executions, overlapping
behaviors). All 4,775 new fixed-binding witnesses replay-verified; processing
errors and safety-guard exhaustions are zero. Independent read-only review found
no important acceptance defect; provenance/status-label suggestions were fixed
in final delivery. Artifact hashes and gzip integrity were checked.

Whole-repository `python -m pytest -q` has two existing collection errors:
`tests/test_mech_uspto31k_causal_import.py` cannot import `mechet.a7_rescue`, and
`tests/test_nl_history_beam.py` cannot import `mechet.in_place_grounded_flow`.
This upload does not claim the repository-wide suite passes or merge to main.
