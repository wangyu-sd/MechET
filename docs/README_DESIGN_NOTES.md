# README design notes

The project README was reorganized around the current proof-carrying retrosynthesis storyline.

## Design principles

The front page follows recurring patterns in widely adopted paper repositories:

1. state one contribution before implementation history;
2. provide a runnable example before the full reproduction pipeline;
3. separate available artifacts from planned or unreleased results;
4. keep the primary path visible and move legacy compatibility paths into a collapsible section;
5. place detailed method and reproducibility material in focused documents;
6. show CI, installation, data, model, citation, license, and limitations explicitly.

## MechET storyline

The primary path is:

```text
mapped product -> executable MECH_PROOF v1 -> deterministic executor -> precursor
```

`MECH_ET v3` remains available for state-annotated trajectory auditing, cold-start compilation, and comparison experiments, but it is not the headline method.

## Front-page result policy

The README includes only released and verified artifacts. Pending checkpoints, paper-scale results, and wrong-split experimental runs are described by status rather than reported as performance numbers. Detailed audit records remain in `docs/BENCHMARK_RESULTS.md`.
