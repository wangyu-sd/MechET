# Review, verification and delivery decisions

## Independent source review

A fresh read-only whole-change reviewer examined source commit
`c511e86550a4e4f16a0d4701ef25517737dbf668`, actual manifests and selected final
previews. One consequential correction pass followed; this was not repeated
reviewer shopping or a change to the scientific goal.

| Finding | Decision and verification |
|---|---|
| Recorded timing/token work not yet summarized | Fixed. `tree_costs.py` and `analyze_nmi_work_evidence.py` export source-bound per-target distributions, true zero/missing counts, per-target cache ratios, and separate run scopes. A real-record Extended Data PDF was rendered. Missing timing and scope regressions were observed RED then GREEN. No global GPU work or matched-compute improvement is inferred. |
| Delivery seal did not bind every required figure/enrichment artifact | Fixed. Finalization checks file existence, inventory/enrichment bindings, table and SVG/PDF/PNG hashes, exact QA record sets and passing machine checks. Captions, case registry and typed schema are hash-bound. Missing/stale QA regressions were observed RED then GREEN. |
| Shard-only literature pointers lacked their row prefix | Promoted from minor to delivery-contract issue and fixed. `/rows/{index}/complete_reaction_edges/{index}` now resolves the original stored report. RED→GREEN regression. |
| Nonnegative visit counts had a negative displayed tick | Promoted to scientific-presentation issue and fixed. Axis lower bound is zero; the zero-inclusive ECDF regression passed and regenerated PDF was reopened. |
| Typed schema declaration missing | Promoted to interface issue and fixed. `schema_manifest.json` declares types/nullability/keys/units and shared-child scope. Numeric-node-handle regression rejects the malformed type. Schema is not mislabeled independent chemical validation. |

The reviewer independently confirmed source-preserving canonical-unique
ranking, all-leaf stock checks, explicit exposure/Unknown limits, the 7/13/5
action galleries, current-state address compilation and the limited
Cabozantinib structural-pair precedent.

## Explicit rulings on review limits

- Final supplemental reconciliation and seal were in progress during review.
  They subsequently completed: 16,534/16,534 original storage records accounted
  for, 9,374 source-scoped graph snapshots, zero unhandled original records,
  plus the separate historical parent-linked case. The final seal passed.
- The reviewer did not inspect every PDF. The implementing agent subsequently
  opened every final PDF-derived preview; details are in `VISUAL_REVIEW.md`.
- The reviewer did not execute the post-correction full suite; the implementer
  did, with the exact command below. Automated tests are software verification,
  not experimental model-quality results.
- Remote unexported controller state is not treated as locally available tree
  evidence. The scope is the accessible 12-root frozen export collection and
  explicitly linked legacy/flat adapters. Shared WALs are backing stores, not
  extra independent observations.
- Base/supplement/enrichment tables and stocks were hashed. Training ancestry
  uses the authenticated exposure manifest; multi-million-row training text
  was not independently rehashed during rendering. That limit is explicit.
- Neither reviewer nor implementer claims all-route laboratory feasibility,
  physical-intermediate observation, independent training-source novelty,
  complete H2/C1 predictive performance, full endpoint benchmarks or equal-work
  search gains. Those need the missing assets recorded in availability.
- Source/asset publication is a new code PR stacked on PR #84's exact
  `d77cf1584520dc2bb26dc409c12a59266eeb8a02` base. It does not merge PR #84/86,
  modify the manuscript or include unrelated dirty operational notes.

## Reproducible software verification

Focused analysis contracts: **56 passed** in the analysis environment.

Full repository command:

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=/tmp/mechet_ray_runtime.K1QW8z:/tmp/pr83_rdkit_glibc235_20261012:.:src \
  /root/miniconda3/envs/meteor/bin/python -m pytest -q
```

The runtime prefixes are the existing local pinned Ray/RDKit compatibility
environment, not additional model weights or a new training environment.
Final post-correction terminal result: **1,004 passed, 13 skipped, 1 warning in
87.44 seconds**, exit 0. The one
PEFT warning concerns an existing adapter fixture's unspecified base config;
it is not a new failed check. No experimental smoke/pilot job was submitted.

`delivery_manifest.json` binds all required produced artifacts. A compressed
copy of each source manifest preserves its original bytes; hashes refer to the
decompressed JSON, not the gzip envelope. Original data are retained unchanged
outside Git. Preview filenames, source IDs and absolute raw paths remain in the
published manifests so publication does not depend on an unsaved conversation.
