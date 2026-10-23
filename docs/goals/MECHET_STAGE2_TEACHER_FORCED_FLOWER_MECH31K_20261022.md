# MechET Stage-II teacher-forced diagnosis — FlowER + mech-USPTO-31k

## Objective and scope
Measure whether the **frozen Stage-II policy** can predict the next IMPORT, FLOW or FINISH decision when the current executor state and preceding accepted-action history come from the reference trace. These are gold-state/oracle-prefix **next-action accuracy** measurements, not autonomous end-to-end retrosynthesis results.

**Model:** Qwen/Qwen3-0.6B (base revision `c1899de289a04d12100db370d81485cdf75e47ca`), original Stage-II R0 adapter `outputs/agent/natural_language_event_history_v2_qwen3_0_6b_seed17`; weight SHA256 `40204c7a059d16976fc4b24b315c3bdde7bf61c705d24bbde515071757e9de75`. No new SFT, DPO, RL, critic, search or adapter updates.

**Data:** FlowER strict executable validation `data/flower_inverse_tool_sft_action_delta_v1/valid.jsonl`, denominator 2,890 reactions. mech-USPTO-31k current-compiler executable-trace validation `data/mech_uspto_31k_inverse_tool_sft_action_delta_v2_compiler_20260824/valid.jsonl`, denominator **1,319** reactions (a subset of the full 3,120 reaction validation corpus). These are separate corpora, not interchangeable with full endpoint tests. The 0.6B actor was trained on FlowER; mech-31k is an OOD local diagnostic.

Initial fixed smoke: **64 FlowER + 32 mech-31k reactions**, seed17 and stratification by reaction mechanism length. After valid results, optional 256 per corpus then full validation.

## Frozen evaluation protocol
1. Read the **source-level inverse tool trace** and replay its gold actions using the established natural-language event data converter. Capture *real* private mapped states with an optional audit-only sink; do not reassign atomic maps or invent Kekule/H states from model-visible SMILES.
2. Use the existing Stage-II compressed accepted-action history converter. The model sees only target product, current gold-state molecular inventory and previously accepted reference actions, plus the frozen tools schema. It **never sees** target action, future state, expected precursor, future imports or remaining horizon.
3. Generate **one greedy tool call per gold decision** using the original 0.6B adapter and exact Qwen Tool-SFT assistant prefix. Parse the tool call, run predicted electron flow in the real executor, then restore the next gold state for the following decision.
4. Fail closed on source denominator mismatch, wrong split, private/public state drift, gold replay mismatch, checkpoint SHA mismatch, changed prepared tasks, or missing output rows.

## Metrics to report
- Overall IMPORT/FLOW/FINISH tool accuracy plus a three-class confusion matrix and premature FINISH rate.
- IMPORT exact SMILES multiset and quantity; exact purpose/scheduling separately; electron participant vs endpoint-only context.
- FLOW source and sink alignment, strict electron-event accuracy, formal executor success and *map-free successor exact*, stratified by event depth, coupled arrow count and explicit arrows versus BE_DELTA.
- Reaction-level all-next-decisions exact and all-local-successor/imports exact. These remain gold-state diagnostics, **not** free-running precursor Top-1.
- Raw per-decision errors, full denominator, model and source SHAs, source IDs, generated token counts.

## Existing-data quick run
On a TaiJi node with a CUDA GPU, existing model and validation files, and a checkout of this **new PR branch**:

```bash
export MECHET_SHARED_REPO=/aaa/fionafyang/buddy1/whaleywang/MechET
export MECHET_STAGE2_TF_CODE_DIR=/path/to/this/pr-branch
export MECHET_STAGE2_TF_OUTPUT=/path/to/fresh/output
bash "$MECHET_STAGE2_TF_CODE_DIR/scripts/run_taiji_stage2_teacher_forced_flower_mech31k.sh"
```

The launcher checks that both validation artifacts and the original 0.6B adapter exist, prepares each source with authentic executor-state parity, runs greedy evaluation on one GPU, then aggregates. It writes results under separate FlowER/mech31k output paths and refuses to overwrite an existing prepared manifest. `MECHET_FLOWER_SMOKE_N` and `MECHET_MECH31K_SMOKE_N` control reaction counts.

CPU-only regression tests, **committed but not yet run on the private execution environment**:

```bash
PYTHONPATH=src:. python -m pytest -q tests/test_mechet_stage2_teacher_forced.py
```

## Completion gate
- [x] Prepared independent evaluator and per-type diagnostics.
- [x] Authenticated private-state sink and original converter reuse.
- [x] Single-GPU quick launcher and mech-USPTO-31k smoke contract.
- [x] CPU regression test cases committed.
- [ ] CPU tests and source-level preflight executed on target environment.
- [ ] Real GPU outputs for FlowER valid64 and mech-31k valid32 produced, checked and archived.
- [ ] Results interpreted against existing frozen product-start R0 evidence.

**Do not claim a measured accuracy, run completion, or new end-to-end improvement until those real logs exist.** Leave PR91/92 closed without merge and do not touch MechET-paper.
