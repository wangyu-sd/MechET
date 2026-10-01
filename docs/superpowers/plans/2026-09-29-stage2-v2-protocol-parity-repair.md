# Stage-II v2 Protocol Parity Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make MechET Stage-II evaluation fail closed unless inference is token-/contract-aligned with protocol-v2 training, then provide a pure-policy K=1 validation smoke that can run immediately without retraining.

**Architecture:** Reuse the existing natural-language executor/runtime. Add the SFT-aligned Qwen tool-call generation prefix from the later EARHO audit, add explicit v2 policy/evaluator contract validation, and provide a dedicated matched Stage-I/Stage-II launcher. Historical legacy-dual-prompt/search paths remain available only for reproduction and are never silently selected by the matched evaluator.

**Tech Stack:** Python, PyTorch/Transformers/PEFT runtime, RDKit 2026.03.4, pytest, Taiji shell/JSON launchers.

**Spec:** Approved in conversation on 2026-09-29: repair Stage-II contract before further RL; unified prompt, SFT-aligned prefix, 40 decision / 32 import budget, pure policy K=1, no value/search, v2 checkpoint lineage gates.

## Global Constraints

- Do not retrain or alter State-SFT/Trajectory-SFT in this PR.
- Protocol-v2 matched inference uses one unified inventory-bearing prompt for all three actions.
- Tool-call generation prefix must be token-prefix-equivalent to completed SFT tool-call rows; do not use Qwen's thinking generation prefix.
- Matched autonomous evaluation uses max_decisions=40 and max_imports=32.
- Matched diagnostic uses greedy K=1, branching=1, beam width=1, value critic disabled.
- State-SFT and Trajectory-SFT matched conditions differ only by compact accepted-action history and their corresponding adapter.
- Refuse v1 adapter manifests for v2 matched evaluation.
- RDKit runtime must be 2026.03.4.
- Existing legacy evaluators/configs remain reproducible but must be labeled diagnostic-only.

## Review Focus

1. Qwen tool-call prefix inserts an empty thinking block — matched v2 mode must reject/avoid it.
2. Legacy dual prompt accidentally enabled — matched v2 mode must fail closed.
3. 12-decision / 8-import historical budget reused — matched v2 mode must reject it.
4. v1 State/History adapter passed to v2 evaluator — lineage gate must reject it.
5. Value critic/search silently changes the measured policy — pure-policy matched mode must not load/use a critic.

---

### Task 1: Add regression tests for v2 inference contract

**Files:**
- Create: `tests/test_stage2_v2_protocol_parity.py`

**Interfaces:**
- Consumes: existing `render_chat`, `policy_prompt`, argparse/runtime config surfaces.
- Produces: failing contract tests for Task 2.

- [ ] Add tests asserting SFT-aligned tool prefix is a prefix of a completed tool-call conversation and excludes Qwen empty thinking.
- [ ] Add tests asserting matched-v2 config rejects legacy dual prompt, budgets below 40/32, branching/beam != 1, sampling/value guidance, and v1 adapter lineage.
- [ ] Run focused tests and confirm failures are due to missing parity helpers/contracts.

### Task 2: Implement SFT-aligned prefix and matched-v2 pure-policy runtime

**Files:**
- Modify: `src/mechet/assistant_masking.py`
- Modify: `scripts/run_natural_language_value_search.py`
- Test: `tests/test_stage2_v2_protocol_parity.py`

**Interfaces:**
- Produces: `render_qwen_sft_tool_prefix(...)`, `validate_matched_v2_args(...)`, v2 adapter-manifest validation, optional critic-free Runtime.

- [ ] Port the audited SFT-aligned Qwen tool-call prefix helper.
- [ ] Add matched-v2 argument validation and adapter lineage validation.
- [ ] Make value adapter optional; when absent, values are zero and no critic is loaded.
- [ ] In matched-v2 mode force greedy K=1 generation and use the SFT-aligned prefix.
- [ ] Run focused tests green.

### Task 3: Add token-level parity audit and immediately runnable validation launcher

**Files:**
- Create: `scripts/audit_stage2_v2_protocol_parity.py`
- Create: `scripts/run_taiji_stage2_v2_matched_valid256.sh`
- Create: `configs/taiji/meteor_mechet_stage2_v2_matched_valid256_8a100_qy_20260929.json`
- Create: `docs/STAGE2_V2_PROTOCOL_PARITY.md`
- Test: `tests/test_stage2_v2_protocol_parity.py`

**Interfaces:**
- Audit consumes protocol-v2 State-SFT/Trajectory-SFT decision rows and adapter manifests; emits a fail-closed JSON report.
- Launcher consumes existing v2 adapters and validation data; runs 256 stratified validation reactions for State-SFT and Trajectory-SFT under identical pure-policy K=1 runtime.

- [ ] Audit exact system/tool/user/history contract and token-prefix parity on frozen validation rows.
- [ ] Launcher checks adapter hashes/manifests, RDKit, budgets, no legacy flags, then runs both conditions.
- [ ] Document historical invalid evidence: v1 checkpoint, legacy-dual-prompt, 12/8 budget, Qwen generation-prefix mismatch.
- [ ] Run focused tests, shell syntax, JSON parse, then full repository CI.

