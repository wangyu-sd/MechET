# Native current turn with complete route memory: authorized follow-up smoke

The user's 2026-11-12 request explicitly authorizes repair and smoke diagnostics.
Attempt02 compares native, v3 and v4 inputs using the original corrected
History-89442 checkpoint. Its first v4 draw still contains missing-atom errors,
so namespacing historical handles alone has already failed the declared behavioral
criterion. Preserve all of attempt02, including its final probability/probe gates.

The v5 hypothesis changes one boundary: preserve the checkpoint-native current
user message byte for byte, and append complete scoped historical memory plus
active reaction/last rejection feedback to the existing system message. The
underlying ledger, archive, executor, actions, probabilities, reward and FINISH
guards do not change. Model-visible memory retains every decision and error,
all observations/states, route obligations and completed reactions. No fake
assistant turn, substituted action, FINISH mask or reference mechanism is added.
V3 remains the preparation default; v5 is explicitly selected for this check.

## Fixed protocol before attempt03 execution

- Scope `explicit_user_authorized_native_history_smoke_v1`, arms exactly
  `native`, `v5`, in that order. Freshly generate both arms; do not reuse earlier
  native continuations as if they were new samples.
- Same exact four saved predicted-prefix contexts as attempt02, same seeds17/18,
  original parent2c98a9bff3afeaffbf2ee2733885cccc68e936f9310fd9f46ba4ee2211e5f453,
  same dataset/stock/executor, one H20 and pinned BF16/TF32 runtime.
- Sixteen continuations, at most eight NEW decisions each, physical generation
  B1. Stop at the initial current reaction's commit or ordinary environment
  termination. The diagnostic cap is censoring, not full-route failure.
- Then eight extra first-decision requests, one actual physical B4 per arm;
  require four nonempty outputs in the journal, not merely a configured B4.
- Retain actual before/after equality of all372 CUDA FP32 actor/reference adapter
  masters,186 active LoRA modules and an enabled/disabled forward-effect check.
- Individually teacher-force every output with the same actor; maximum absolute
  logprob difference must remain <=0.25. No tolerance relaxation.
- Transport correlation IDs pair case/seed across arms; original environment IDs
  are restored explicitly after sampling, with both IDs archived. No optimizer.

Software PASS requires all existing provenance/runtime/weight/probability and
actual B4 coverage gates. Behavioral evidence requires zero missing-atom or
historical-handle-rejected actions, target-changing accepted electron flow in
both fixed new-target cases, and at least as many target-changing draws as the
fresh native arm. Both criteria are required for overall PASS; software PASS
alone yields INCONCLUSIVE. Keep all unsuccessful cases and seeds.

These are four diagnostic contexts from two saved predictions, with fixed
historical prefixes and a short continuation cap. Counts establish neither
multistep benchmark performance nor RL improvement. Attempt03 must not replace
or restart the existing formal full v3 task automatically. Attempt01 bootstrap
failure and attempt02 evidence remain available at their original paths.
