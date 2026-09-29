# Smoke-to-full decision gates

The campaign is exploratory. A full 8B retraining needs human approval and
all five R1–R5 pipelines, overlap/provenance audits and the frozen held-out
endpoint comparison. The controller never submits an 8B job automatically.

Hard machine-readable checks:

- every R1–R5 package completed without unresolved data-contract errors;
- zero reaction/product overlap between scientific-smoke training mixtures and
  frozen evaluation cohorts;
- curated mechanism augmentation is real (`curated_accepted_rows > 0`);
- Mech-smoke endpoint Top-1 is no more than 3 percentage points below Base.

Human scientific review then judges whether R2 or R4 has a meaningful positive
electron-level directional signal and at least two of R1/R3/R5 are usable or
reveal a concrete failure mode. The exact thresholds cannot be selected after
seeing smoke outcomes. The scorecard reports these as `human_review_required`
rather than inventing a binary result.

Infrastructure retries are limited to two per stage and may not change seed,
test manifest, architecture, executor, prompt, reward, epochs or candidate
budget. A failed science metric is not an infrastructure failure.
