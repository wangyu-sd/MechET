# Natural-language electron-event protocol v2

This is an implementation repair of the existing mechanism-first method.  It
does not change the scientific task: the model still receives only the product,
the executor-owned current state, and accepted past history; it chooses one of
`import_fragments`, `apply_electron_flow`, or `finish_trace`; the executor still
applies the electron event and owns the successor state.

## Why v1 is diagnostic only

The v1 builder displayed the molecular inventory only when the gold next action
was `apply_electron_flow`.  Import and finish rows omitted it.  Consequently the
observation leaked the teacher-forced action class.  The historical evaluator
worked around this by querying two different prompts at every state and forcing
one prompt to emit import/finish and the other to emit an event.  Likelihoods
from those prompts are not comparable, and neither prompt represents a genuine
single policy decision.

The v1 runtime also rejected paths that are present in its own supervision:

- a visible fragment could not be imported in two different batches;
- evaluation defaults allowed only 8 imported copies and 10 decisions;
- finish was rejected whenever the target survived as any endpoint component,
  even after a real electron event.

There was also a private-map replay mismatch.  The SFT builder appended the
source artifact's mapped fragments, while inference necessarily assigns fresh
private maps to model-authored unmapped imports.  On symmetric/aromatic graphs,
the same public Axx action could therefore select a different Kekule bond after
the first step.  The v2 builder now performs exactly the inference-time remap,
translates every source move into that private frame, and advances only with the
real executor output.  It never restores a hidden authoritative state between
decisions.

On the frozen 2,890-reaction validation split, 321 gold traces repeat a fragment
in separate import batches, 304 use more than 8 imported copies, 231 use more
than 12 decisions, and 39 gold full endpoints retain a target-equivalent
component.  These are protocol/runtime mismatches, not model errors.

## Frozen v2 contract

1. Every decision receives the same target, current state, complete temporary
   atom/bond inventory, and (for trajectory SFT) compact accepted-action history.
2. One policy distribution chooses all three tool types.  Search never creates
   a gold-action-conditioned alternate prompt.
3. Repeated imports are legal and counted against a total import budget.  The
   default budgets are 40 decisions and 32 imported copies; the frozen SFT
   maxima are 23 and 24, respectively.
4. The no-op finish guard applies only before any electron event.  It does not
   reject a transformed trajectory merely because a target-equivalent component
   remains in the complete mixture.
5. Primary retrosynthesis accuracy is structural-precursor exact match using
   the repository endpoint contract.  Full-mixture exact match is retained as a
   stricter secondary diagnostic.

The complete 2,890-reaction validation split was rebuilt and replayed through
the independent runtime after these repairs: all 22,341 decisions had byte-exact
prompt parity, all actions executed, and all 2,890 trajectories reached finish.
The machine-readable record is
`docs/audits/natural_language_protocol_v2_valid_20260918.json`.

The historical dual-prompt behavior remains available only through
`--legacy-dual-prompt` so old checkpoints and completed artifacts can be
reproduced without being confused with v2.

## Data lineage

v2 must be rebuilt from the same frozen strict-executable FlowER universe:
257,167 train, 2,890 valid, and 28,967 test reactions.  No reaction filtering,
new compiler, changed electron event, or changed endpoint is permitted.  Only
the model-visible observation contract changes from conditional to unified.
