# Mechanistically verified retrosynthesis: implementation plan

This plan implements the frozen design in the adjacent `specs/` document. It is
an execution checklist, not permission to change the scientific contract after
seeing a result.

## Phase 0: source and contract audit

1. Resolve every training and R1–R5 source to an immutable file hash and a
   license/provenance record. Require `executor_replayed=true` for any curated
   State-SFT row. Keep R4 independent tests and all R1–R5 evaluation records
   outside the reaction-level training universe.
2. Resolve what a "row" means: the State-SFT budget counts **decision rows**;
   reaction IDs are also tracked so no reaction crosses train/evaluation. A
   selected decision retains its original causal observation and tool target.
3. Record missing strata explicitly. In particular, do not relabel scaffold
   frequency as structural novelty or claim a Mech-vs-Base comparison if zero
   curated rows survive replay.

## Phase 1: deterministic data freeze

1. Pin the Qwen3-0.6B immutable revision and the existing State-SFT tool schema.
2. Generate 32-row engineering and 12,000-row Base/Mech manifests. Use stable
   IDs, seed, per-source quotas, deterministic minimum-per-cell allocation and
   proportional fill. Record underfilled cells, missing metadata and SHA256s.
3. Write selected JSONL files to a new campaign output directory. Rerunning
   against the same frozen output must verify hashes, not silently regenerate.

## Phase 2: campaign machinery

1. Add a finite-state controller with `plan`, `freeze`, `status`, `run` and
   `collect` entry points. Stage records include config/code/input hashes,
   exact command, output hashes, timestamp and attempts.
2. `plan` lists all R1–R5 prerequisites even when a source is unavailable;
   missing data is visible and must never become a fabricated score.
3. `run` delegates Taiji submission and monitoring to the existing validated
   helpers. Use one-GPU smoke profiles in H20/A100/V100 order, `meteor` task
   names, unredirected stdout/stderr and a <=60s heartbeat. Retry only
   infrastructure-equivalent failures, at most twice.

## Phase 3: smoke and evaluation

1. Run 32-row/100-step engineering smoke. Check optimizer updates, executor
   replay, one held-out inference and metric serialization.
2. Run paired Base/Mech scientific smoke on the frozen 12k decision rows.
   A zero-curated Mech condition remains a documented data-gate failure rather
   than being described as a mechanism-augmentation experiment.
3. Run R1–R5 independently once their own prerequisites exist. Negative
   scientific results never cancel another package. Preserve compilation
   failures in denominators and record train/evaluation overlap checks.
4. Produce a scorecard with paired deltas, bootstrap intervals and explicit
   promotion-gate status. The controller never auto-launches 8B/full-data work.

## Verification

- Unit tests cover deterministic selection, quotas, underfill, collision
  rejection, freeze immutability, stage transitions and infrastructure-only
  retries.
- A dry-run shows all five packages with concrete missing prerequisites.
- Engineering smoke is verified by a real completed checkpoint and held-out
  output, not merely a successful Taiji submission.
