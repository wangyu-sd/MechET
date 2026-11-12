# Reaction boundaries and complete event history v3

User-approved scope: fix the observation after a committed reaction and replace
repeated local snapshots with a complete, referenced history. The user then
explicitly requested a full training replacement after verification. No action
masking, new reward or sampling-distribution transform is introduced; the new
model-visible observation is a separately named experimental condition.

The new contract is `route_event_history_feedback_v3`. Keep v2 operational for
old receipts and checkpoints. New preparation defaults to v3; old contracts are
never relabelled. At each new non-stock precursor, record an explicit reaction
start. Historical action results are scoped to their original reaction and refer
to their resulting state; they do not instruct the actor to continue from an old
state. Render the entire event ledger first, then the authoritative active
reaction, then the native current observation/inventory/Stage-II history capsule.
Only the current reaction contributes to its electron-flow count and latest
feedback. All historical tool outputs, observations, errors and duplicate AND
obligations remain recoverable. Intern repeated text/state/observation/output
values; keep every decision event and its original saved policy receipt.

Implementation sequence:

1. Add failing boundary/history tests using the real RouteEpisode state machine.
2. Add a standard-library event ledger with lossless observation reconstruction,
   safe JSON rendering and validated JSON snapshot/restore.
3. Integrate the versioned ledger, active boundary rendering, preparation and
   native certificate replay; preserve v2 and all existing chemistry guards.
4. Run new tests plus affected environment, training, replay and snapshot tests.
   Replay saved complete-round artifacts where feasible as software evidence;
   old sampled actions under a new prompt are not a new policy rollout.
5. Publish the reusable module, tests and an exact integration patch against the
   frozen deployed source to PR104, whose main base lacks the submitted trainer.

Review focus: FINISH followed by a new non-stock precursor; repeated same-state
failures; duplicate precursor occurrences; snapshot reference corruption; chat
delimiter injection; native replay using the report's actual observation version.
Acceptance requires exact state/action/reward semantics on replay, unchanged
saved policy tokens/logprobs, current inventory at the end of the prompt, and no
dropped old events. No chemical or learning improvement is implied by CPU tests.

## Implemented behavior

`route_event_history.py` is a standard-library ledger. Shared text/state/output
tables retain every decision event and reconstruct each historical local
observation exactly, including its original inventory and Stage-II control
capsule. A new reaction has an explicit `reaction_started` event. Historical
feedback uses state references and `instruction_at_that_time`; it does not expose
the old endpoint as the active `current_state`. The active block contains the
new target, current state, occurrence, per-reaction accepted-flow count and only
that reaction's latest error. Native current inventory/capsule appears last.

There is no truncation, sibling merging, hidden-label feedback, automatic
success, FINISH masking or repeated-action removal. Every rejected output remains
in the original policy receipts. Recovery ceilings and chemistry guards remain
unchanged. Snapshot restoration validates both reference structure and one-to-one
correspondence with receipts, outputs, observations, states and reaction starts.
Native route certificate replay now uses the report's actual observation version.
V2 remains supported for old frozen contracts; new preparation defaults to v3.

Independent review found a missing restore-completeness check. Five regression
cases first reproduced omitted events, empty feedback, changed outputs, missing
reaction starts and changed observations; all were fixed before freezing v3.

## Verified evidence

- Final affected suite: **74 passed**, 705.69 s. The expensive preparation tests
  create and fsync their unit fixtures on the user-mandated Ceph storage; this
  duration is not model compute. Negative malformed-SMILES cases emit expected
  RDKit diagnostics.
- Portable standard-library module checks: **4 passed**, 17.02 s. These repeat
  some integration coverage and are not four additional experimental results.
- Full saved-round software replay: **256/256 episodes, 5,054/5,054 decision
  records, zero execution differences or restore failures**. It uses the frozen
  native executor with only explicit v3 integration changes. Saved action IDs
  and logprobs are retained; no model sampling or optimizer step is performed.
- On those same saved traces, message-content characters change from
  **176,034,404 to 109,048,750**, approximately **38.05% less**. This is neither
  a token-length measurement nor actual GPU speedup. The replay retains recorded
  censor flags; it does not estimate new policy success or counterfactual closure.
- The integration patch was dry-run, applied to a fresh shared-storage copy of
  the frozen base, and all 11 resulting file hashes matched the declared changes.

Numeric evidence: [verification](../../configs/ops/route_history_v3_verification_20261112.json).
Integration provenance: [before/after hashes](../../configs/ops/route_history_v3_integration_20261112.json).

## Integration and reproduction

PR104's main base lacks the deployed target-only trainer. Therefore the PR
contains the reusable module and portable tests directly, plus an
[exact integration patch](../../patches/route_history_v3_20261112.patch) for
`artifacts/route_phase_runtime_20261111_15/code`. The patch includes regression
fixtures needed by its new integration tests. It is applied to a fresh copy of
that frozen base, not to main or a live deployment. Existing files in the base
must not be modified through hardlinks.

For the portable checks, from this PR checkout:

```bash
source /aaa/fionafyang/buddy1/whaleywang/MechET/configs/runtime/shared_host_storage.env
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH=/aaa/fionafyang/buddy1/whaleywang/MechET/outputs/shared_host_storage/test_tools/pytest_portable311.zip:/root/miniconda3/envs/meteor/lib/python3.11/site-packages \
/root/miniconda3/envs/meteor/bin/python -S -B -m pytest -s -q \
  tests/test_route_event_history_core.py --basetemp "$TMPDIR/history_v3_portable_fresh"
```

For integration, copy the frozen base to an unused Ceph directory, verify the
before hashes against the integration manifest, and run
`patch --batch --forward -p1 -d "$COPY" -i "$PR_CHECKOUT/patches/route_history_v3_20261112.patch"`.
Verify every resulting file against the manifest's after hashes. With the pinned
runtime/RDKit paths and the synthetic `mechet` namespace bootstrap used by the
existing runtime, run `tests/test_route_event_history.py` in that copy with
`--rootdir "$COPY" --confcutdir "$COPY"` to avoid ancestor-directory collection
on shared storage. The base
contains the native chemistry dependencies; the copied test fixtures exercise
real native receipt replay as well as deterministic state-machine cases.

## Formal replacement

The user-authorized replacement was created **2026-11-12 14:57:35 CST** and the
platform reported `TRAINING_RUNNING` at **14:58:12 CST**:

`meteor_mechet_stage2_external_route_drgrpo_4h20_zjk_20261112_history_v3_01`

Instance: `8b1d8912a1b06b6f01a1ce8c2bfd3669`. Full 23,730 TRAIN / 2,986 VALID,
independent G4, four H20, seed17, one epoch. It starts directly from corrected
History-89442, SHA256
`2c98a9bff3afeaffbf2ee2733885cccc68e936f9310fd9f46ba4ee2211e5f453`,
with the same fixed reference. It does not resume the old v2-trained weights.
Contract SHA256:
`3c90c5bdaed9921fe1bfc4230b1bdb182cdc44b275fe3a4533cc4e6f98465035`.
The immutable source snapshot has 396 files; the deployed trainer has 377 Python
sources and 434 verified input bindings. HF BF16/TF32, learner B1, original scalar
reward, full validation protocol and phase profiling are preserved.

The owned v2 task was stopped after verification and preparation; all its
artifacts remain, including the completed round-13 checkpoint. Platform running
does not prove an optimizer update or improved route quality. Neither is claimed
for this new task by this submission record. PR104 remains Draft.
