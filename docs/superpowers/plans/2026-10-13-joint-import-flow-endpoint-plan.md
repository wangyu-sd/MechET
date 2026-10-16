# Joint Import–FLOW Endpoint Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train Qwen import/controller LoRA and graph FLOW heads on shared true
terminal endpoint returns, then submit a provenance-bound formal Taiji task.

**Architecture:** Preserve the existing typed tools and product-start executor.
Use complete delegated-action probabilities on a fixed proposal support, IID
replacement samples and leave-one-out endpoint credit. The follower remains
fixed within an update; current-state imports are interleaved, not preloaded.

**Tech Stack:** PyTorch, Transformers/PEFT, CRN-JEPA PR18, MechET executor,
RDKit 2026.03.4, Tencent Taiji CLI.

**Spec:** `docs/superpowers/specs/2026-10-13-joint-import-flow-endpoint-design.md`

**Execution status (2026-10-13):** User approved direct inline implementation.
Tasks 1–5 source/submission are implemented and verified. Formal ordinary1xA100
task is running with a first committed paired checkpoint/update. This first
update is replay-only, not evidence of endpoint RL gain; heldout results pending.

## Global Constraints

- Strict train/valid universe: 257,167 / 2,890 reactions with the exact source
  hashes in the spec. No new filtering, no silent replacement by a subset.
- Qwen parent revision/adapter SHA and CRN parent SHA exactly as in the spec.
- Formal BF16 without 4-bit loading; 40 decisions, 32 imports, 4,096 context
  tokens, 384 generation tokens. Product-only fresh maps and frozen primary
  structural scoring. Never inject GT into model inputs or on-policy draws.
- Graph encoder/Qwen base frozen; graph policy heads/Qwen policy LoRA trainable.
- K=8 IID replacement training draws; K1 fixed continuation and evaluation.
- Short-H actor gradients zero; full-budget schedule independent of returns.
- All task flags/readable names start with `meteor`, private Ceph init from a
  successful Tencent-image donor, default POD progress/heartbeat <=60 seconds.
- Unit/static checks are not experiments. New small GPU jobs require explicit
  user authorization; do not recycle the earlier smoke into formal results.

## Review Focus

- Explicit-H maps participate in proton transfers and must survive alignment.
- Complete IMPORT output lacking EOS or containing invalid JSON still has an
  auditable probability and failure return; it must not silently disappear.
- Equivalent environment choices must not create controller-blocked skips or
  be relabelled as chemically wrong solely for differing from GT.
- Exact accepted terminals containing unchanged targets retain benchmark
  return 1; no additional reward-only veto may contradict evaluation.
- Interrupted exports/resume must never pair a new Qwen adapter with an old
  graph head, nor silently restore different follower precision.

---

### Task 1: Verified prerequisite repairs

**Files:** Modify `scripts/train_crn_graph_earho.py`;
test `tests/test_graph_earho_training_integration.py`.

**Interfaces:** Retain `align_reference_moves(reference_mapped_state,
actor_mapped_state, moves)`; retain primary `structural_exact` and executor
legality. Correct the existing endpoint-return expression, not its denominator.

- [x] Add regressions for mapped H:33 in an actual proton-transfer event and
  for an accepted GT-matching terminal that retains its selected target.
- [x] Run the two tests and observe `KeyError:33` and a false endpoint reward.
- [x] Parse aligned molecules with `removeHs=False`; remove the additional
  unchanged-target reward veto, preserving executor checks.
- [x] Run targeted tests and the complete pytest suite (653 passed, 8 skipped).
- [x] Commit these fixes with the reviewed prerequisite/code handoff
  separately from the proposed joint model.

### Task 2: Delegated Qwen/graph action probabilities

**Files:** Create `src/mechet/joint_endpoint_policy.py` and
`tests/test_joint_endpoint_policy.py`; modify `src/mechet/crn_graph_bridge.py`.

**Interfaces:** Produce immutable `JointProposal` with tool name, parsed
arguments/error, original Qwen continuation token IDs, graph moves and raw
generation; `qwen_continuation_log_probability(model, input_ids,
continuation_ids) -> Tensor`; `joint_action_log_probability(proposal,
qwen_logp, graph_logp=None) -> Tensor`.

- [ ] Test exact summed token probabilities against a literal two-step
  categorical fixture, including EOS and unequal completion lengths; assert
  no mean-NLL normalization. Verify the FLOW score is route plus graph-event
  log probability and IMPORT is its full delegated Qwen completion score.
- [ ] Run the tests and observe missing-interface failures.
- [ ] Implement teacher-forced causal suffix scoring without generation-time
  autograd/KV reuse. Bind original token IDs rather than decode/re-encode
  guessed text. Malformed/truncated sampled completions retain scoreable raw
  continuations and a failure classification.
- [ ] Test real decoder/teacher score parity on the authorized model fixture;
  do not claim chemical performance from that check.
- [ ] Run the full suite and commit.

### Task 3: Joint frontiers, model-owned imports and endpoint returns

**Files:** Create `scripts/train_crn_qwen_joint_earho.py` and
`tests/test_joint_earho_training.py`; consume the probability interfaces from
Task 2 and existing `conditional_endpoint_policy_loss`/`AuditJournal`.

**Interfaces:** `select_joint_frontier(episode, reference=None)` returns an
actual autonomous pre-decision Node and attribution, not a gold-reset state;
`collect_joint_support(runtime, node, args, rng) -> tuple[JointProposal, ...]`;
`score_joint_support(...) -> list[Tensor]`.

- [ ] Add failing tests showing an IMPORT-first mismatch remains eligible and
  an environment difference does not discard the root. Reject GT reset/action
  insertion. Pin independent scheduling before any branch return is read.
- [ ] Implement bounded model-generated IMPORT support, existing graph
  beam/native FLOW support and a scoreable legal FINISH proposal. Freeze and
  deduplicate the complete ordered action support before IID draws; preserve
  invalid draws and duplicate multiplicity through execution pooling.
- [ ] Add exhaustive finite expected-gradient tests with route, import and
  graph components. Verify import successes differentiate Qwen LoRA, FLOW
  successes differentiate the graph head, and short-H losses are zero.
- [ ] Make existing Qwen policy LoRA trainable while freezing its base and
  the graph encoder. Create optimizer groups for Qwen (1e-6) and graph heads
  (5e-6); clip combined gradients to 1.0. Freeze current weights until one
  endpoint/replay/conditional-KL update is complete.
- [ ] Reuse the exact heldout terminal structural scorer. Verified replay is
  separate (weight 0.1), complete-support original-policy KL weight 0.02;
  reference labels are used only at genuinely label-compatible states.
- [ ] Persist raw candidate inputs/outputs, action probabilities, original
  draws, full branch attempts, errors and endpoint returns. Report actual
  endpoint-contrast updates separately from replay-only updates.
- [ ] Run the full suite and commit.

### Task 4: Import-role audit, joint artifacts and matched inference

**Files:** Modify `scripts/run_crn_qwen_hybrid.py` and
`scripts/pr83_resume.py`; create `tests/test_joint_earho_artifacts.py`.

**Interfaces:** Reuse existing `purpose` values and unmapped-SMILES import
schema. Track imported map sets and later executed participation as audit
metadata; export a paired `qwen_policy/` adapter, graph policy head and one
`joint_controller_contract.json` with both output SHA-256s.

- [ ] Test that advisory participant/context declarations neither mask valid
  moves nor change primary endpoint projections. Test catalytic participation
  and repeated imports with explicit counts; do not build a fixed fragment
  library or silently remove all solvents/catalysts from active state.
- [ ] Bind both parent/output SHAs, model/tokenizer revision, dtype, budgets,
  source hashes and `joint_import_flow_endpoint_v1` algorithm in exports.
  Joint inference consumes both exported models, not the original Qwen adapter.
- [ ] Test save/reload and resume with both optimizers, LoRA/head tensors,
  RNG and authenticated journal cursors. Reject mixed-generation artifacts,
  dtype changes and modified audit prefixes without overwriting evidence.
- [ ] Verify same-controller K1 input/action/executor contracts; run the full
  suite and commit.

### Task 5: Review, code PR and formal Taiji training submission

**Files:** Create `scripts/run_taiji_joint_import_flow_a100.sh` and
`configs/taiji/meteor_mechet_joint_import_flow_full_1a100_qy_20261013_01.json`;
update `docs/ACTIVE_ICLR_STATUS.md` and the execution ledger.

**Interfaces:** Use `scripts/submit_taiji_with_donor_init.py` with the proven
`meteor_mechet_pr83_flower_valid128_singlepass_8a100_qy_20261012_01` donor only
after rechecking its successful instance, group, location and Tencent image.

- [ ] Run the complete pytest suite, CLI help, Bash/JSON validation, artifact
  secret/path checks and independent read-only implementation review. Fix
  blocking findings with RED-to-GREEN regressions before deployment.
- [ ] Push the tested code and clearly separate measured historical smoke,
  prerequisite repairs, proposed interfaces and actual joint implementation.
  Do not merge without a separate request.
- [ ] Requery ordinary application quotas and live donor facts. Prefer the
  successful main Qingyuan A100 environment; do not retry abandoned `:taiji`
  images or infer no GPUs from physical-host summaries alone.
- [ ] Submit one ordinary 1xA100 task traversing 257,167 strict train roots;
  bind executable commit/model/source SHAs, use default-POD unbuffered
  progress plus a 60-second heartbeat, and retain private Ceph credentials
  outside Git. This is not an 8-GPU distributed trainer or a smoke task.
- [ ] Read instance/POD logs, model GPU activity and first persisted update;
  report pending/running/failed truthfully. Do not stop unrelated tasks.
  Schedule paired full validation only with the new paired model artifacts.

## Handoff

The requested end state is both a code PR and a tested training submission.
The user approved the written design and direct inline execution. The final
source review and full-protocol submission are next; no new smoke is authorized.
No job should be launched with the graph-only script while calling it joint RL.
