# Historical handle scope repair and explicitly authorized smoke

The user explicitly requested a repair plus a smoke on 2026-11-12. This is a
diagnostic exception to the repository's default prohibition on subset jobs,
not a change to the full training/evaluation denominator. The existing v3 full
run is not modified or replaced by this check.

Hypothesis: exposing past inventories and raw tool calls with bare Axx/Bxx
aliases interferes with the current native inventory. Historical occurrence of
an absent alias is evidence of ambiguity, not causal proof of copying. The v4
opt-in observer namespaces historical handles by observation, renders parsed
past tool calls as structured historical actions, and scopes error references.
The final native current observation is unchanged. Every original action, raw
output, state and error remains in the archived ledger/receipts. No FINISH mask,
resampling, reward change or hidden reference feedback is introduced. V3 remains
the preparation default pending smoke results; v2/v3 replay stays operational.

Five new regression cases were observed RED, then passed with the affected
suite (57 cases total, including overlapping earlier coverage). The scopes are
per observation, not just per reaction: the same alias can denote different
atoms after a state change within one reaction.

## Fixed smoke protocol, declared before GPU execution

- One ordinary H20, exact original corrected History-89442 actor and reference,
  BF16/TF32, existing pinned Torch2.6cu118 / Transformers5.5 / PEFT0.18.1 runtime.
- Four fixed contexts from two saved first-round training trajectories: a root
  control, the first new small-target reaction, recovery after a missing-atom
  error, and a second new-target reaction. Replay stored predicted prefixes to
  construct identical states; never insert reference mechanism labels.
- Three input arms: checkpoint-native local messages; unmodified v3 history;
  opt-in scoped/structured v4 history. Seeds17 and18, 24 diagnostic continuations,
  at most eight new decisions each. Stop when the initial current reaction
  commits, an ordinary environment terminal is reached, or the explicit smoke
  cap is reached. The cap is diagnostic censoring, not a failed route label.
- Main generation uses physical B1 to isolate prompt effects from padding. A
  separate first-decision-only B4 probe runs all four contexts in each arm at
  seed17 (12 additional model requests), paired against the saved B1 first
  outputs. No reaction rollout is performed from these extra outputs.
- Transport request IDs include case/draw to prevent cross-context collisions.
  They exclude arm for paired RNG and never alter model-visible messages. Both
  transport and original environment IDs are recorded; no sampled action or
  probability is replaced after generation.
- Audit all372 actor/reference FP32 CUDA adapter masters against the original
  file, all186 active actor LoRA modules, and a real enabled-versus-disabled
  adapter forward. Repeat master equality after the test. No optimizer exists.
- Recompute every sampled output's teacher-forced logprobs individually, keeping
  the existing0.25 maximum absolute tolerance; record padding-probe differences
  without assuming all stochastic outputs must be byte-identical.

## Interpretation

Software status PASS requires the pinned provenance/runtime/actual CUDA-weight
and adapter-activation checks, finite receipts, and the declared probability
tolerance. A software FAIL is retained, never fixed by loosening the tolerance.
It also requires journal evidence of one physical four-request batch with four
nonempty outputs in every padding-probe arm. A configured maximum of four, split
batches, or equal empty outputs cannot pass this coverage requirement. Sampled
model requests are counted separately from logical attempts and verifier forwards.

Behavioral evidence for the candidate requires zero missing-atom and rejected
historical-handle actions, at least one accepted target-changing electron flow
in both fixed new-target cases across the two draws, and at least as many
target-changing continuations as the native arm. These small diagnostic counts
are not a benchmark score. Overall PASS requires both software and behavioral
criteria; otherwise report INCONCLUSIVE or FAIL explicitly. Preserve every case,
arm, seed, raw response and unsuccessful attempt. No automatic formal promotion.

The first attempt failed before loading any model: the entry wrapper used a
function-local `os` before its import. Attempt01 is preserved and its owned task
was stopped. Attempt02 repairs the wrapper with a RED/GREEN behavioral startup
test; it retains the same cases, seeds and budgets. Independent review also led
to scoped malformed historical call names/keys and a regression ensuring real
boron/ring SMILES are not mistaken for bond aliases. Final affected tests:
60 passed,43.08s. These are overlapping software cases, not chemical evaluations.
