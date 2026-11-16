# PR109 mandatory final self-search distillation: preparation primitives

**Implemented: independent TRAIN-only target/loss primitives, a data-preparation CLI, and ReFIT export/update integration functions. Not executed: real-model TRAIN self-search collection, student distillation training, a distilled checkpoint or distillation evaluation.** Preparing this artifact does not complete PR109's mandatory final training phase. All model/teacher/search execution remains outside this module; existing tasks and live dependencies are unchanged.

The normative final-stage authority is `artifacts/pr109_handoff_20261116/RETRO_E_PROGRESSIVE_RL_BENCHMARK_SEARCH_DISTILLATION_20261116.md`, Stage5, together with the standardized-data addendum. The addenda's Standard exclusion of RetroChimera and additional forward FlowER overrides the older optional-teacher wording in `RETRO_E_REFIT_PO_CODEX_GOAL_20261116.md`. This first implementation supports **Standard only**. Optional +F needs its own separate conditioning/evidence contract before extension.

## Independent API

`src/mechet/route_self_distill.py` exposes:

- `content_sha256(value)`: canonical JSON SHA256, sorted keys, compact separators, nonfinite JSON values refused.
- `prepare_state(row, receipts, expected_actor_sha=..., training_target_ids=..., eos_token_id=..., temperature=1.0)`: same-state semantic targets with auditable eligibility/skips and exact source bindings.
- `prepare_anchor(row, receipts, expected_actor_sha=..., expected_source_sha=..., eos_token_id=...)`: explicitly sourced, replay-verified original-SFT or actor-state anchors.
- `mix_examples(search_states, anchors, sft_weight=0.0, actor_weight=0.0)`: declared source masses, equal state weight within each source, duplicate-state rejection, no silent substitution for unavailable anchor sources.
- `native_assistant_ce(logits, prompt_token_ids, generated_token_ids, weights, eos_token_id)`: lazy-Torch weighted summed native-action negative log likelihood, including action EOS. It performs no model forward, optimizer step or parameter update.
- `prepare_dataset(...)`: complete-input preparation and immutable shared-storage output. There is no target-count/subset limit.

Module import and CLI preparation use the Python standard library only. Only the CE primitive imports Torch. The CLI loads this file directly to avoid unrelated eager Torch imports in the repository's existing `mechet.__init__`.

## State-candidate contract

Each input JSONL row contains:

```text
split: "training"
node_id: exact ReFIT decision occurrence ID
source_actor_sha: frozen trained Standard RL actor SHA256
behavior_policy_version: nonempty recorded policy version
refit_config_sha, budget_sha, tokenizer_sha, tool_schema_sha,
executor_sha, stock_contract_sha: frozen SHA256 strings
external_teacher_calls: 0
prompt_token_ids: exact nonempty native conditioning IDs
conditioning:
  target_id: member of the frozen TRAIN target allowlist
  reaction_occurrence_id: nonempty reaction occurrence ID
  observation_sha, history_sha, route_context_sha, inventory_sha: exact SHA256
candidates: ordered observed candidate records
```

A state's `conditioning_sha` is canonical SHA256 of the dictionary containing **all** `HEADER_FIELDS`: node ID, actor, behavior version, ReFIT/budget/tokenizer/tool/Executor/stock hashes, exact prompt IDs and the complete conditioning dictionary. It is not just the molecule SMILES or a node label. Candidate and receipt bindings must match it exactly.

Each candidate includes:

```text
semantic_key: canonical native action meaning, supplied by the verified producer
 generated_role: "assistant"
 generated_token_ids: original native assistant action tokens, action EOS included
 Q_search: finite verified suffix value under the frozen continuation budget
 receipt_id, receipt_sha256
 conditioning_sha, source_actor_sha, behavior_policy_version,
 refit_config_sha, budget_sha
 proposal_kind: independent_policy_sample | puct_search |
                deduplicated_search | rejection_sampled_search
```

The leading spaces above are for readability; the keys are ordinary JSON members. No generated probabilities are fabricated. Known PUCT/deduplicated/rejection-selected proposals may supply **offline supervised** evidence despite not having an on-policy denominator. Unknown proposal provenance is rejected. Every prepared row is explicitly labelled `offline_search_selected_native_ce_not_grpo`; these records are not GRPO/PPO receipts and must not be sent to that objective.

The receipt manifest is JSON `{ "verifier_sha": "<SHA256>", "receipts": { "receipt-id": {...} } }`. Each receipt repeats every candidate field except `receipt_sha256` and adds `split`, `verified`, `completed`, `censored`, `executor_accepted`, `trust_tier`, and `external_teacher_calls`. Its canonical content hash must equal the candidate's `receipt_sha256`; action tokens, semantics, Q, policy version, state and budget must match. Duplicate receipt IDs in a state are rejected.

**Trust boundary:** this CLI consumes a manifest issued by an independently trusted Executor/route verifier. It validates supplied content hashes and bindings. It does not independently replay a native action, recompute route rewards, validate chemical references, authenticate a manifest signature, verify model weight bytes, decode prompts to audit hidden-reference leakage or establish the TRAIN allowlist's provenance. The upstream verifier and frozen source manifests remain mandatory production gates. Hand-written flags alone are not chemical evidence.

## Eligibility, informativeness and q_T

Positive targets currently require a trusted completed, noncensored, Executor-accepted receipt with `trust_tier="reference_supported_closed"`. Unsupported stock closure, partial outcomes, rejected actions, unverified receipts and resource censoring never enter positive CE targets. This does not label an unsupported route chemically wrong. Adding independently supported tier2 positives requires a separately declared evidence policy; it is not enabled implicitly.

**Informativeness is checked over all trusted completed noncensored semantic outcomes, including excluded failed actions.** It is not checked only between eligible positive candidates. This is required because ordinary successful complete routes can all have Q=1:

- One eligible successful action Q=1 and a different, trusted rejected action Q=0 produce a useful **unit-probability positive target**, q(success)=1. The failure remains excluded from positive CE.
- Two eligible successful actions both Q=1 and a third trusted failed action Q=0 remain useful; q=.5/.5 on the two eligible successes.
- A censored failure cannot supply the distinguishing outcome.
- All-ineligible, only one trusted observed semantic action, or equal trusted semantic mean Q values skip with explicit reason. No arbitrary labels are generated.

Repeated observations of the same semantic action use the explicitly declared **arithmetic mean Q**. The first eligible actually observed native token sequence is its representative; all contributing eligible receipt IDs and proposal kinds remain attached. No maximum-Q cherry-picking, visit-count probability, synthetic action or unobserved tokenization is introduced.

On eligible semantic actions:

`q_T(a|s) = softmax(mean eligible Q_search(a) / temperature)`.

The implementation subtracts the largest Q before exponentiation. Temperature must be positive and finite. There is no implicit value clipping or length normalization. This distribution is an offline training target conditioned on verified search evidence, not calibrated chemistry confidence or an unbiased policy proposal distribution. The informativeness ledger includes trusted semantic count and mean-Q range separately from the positive q_T target set.

## Native CE and anchors

`logits[b,t]` predicts the next token in concatenated prompt+generated IDs. CE selects exactly the positions predicting the original assistant-generated action tokens, **including the final EOS**. Prompt/tool targets and padding do not contribute. Zero-weight candidate logits are not read. Selected nonfinite response logits, invalid IDs/shapes, missing/early EOS and invalid source weights are rejected.

The loss is the weighted sum of **sequence-summed** response NLL, divided by the sum of candidate/source weights. It is not a mean over action tokens: different-length native actions retain their full sequence likelihood. The caller must supply a real model forward and map per-state q_T and mixture masses into those weights. This primitive itself performs no training.

Anchors must have `anchor_kind="original_sft"` or `"actor_state"`, split training, explicit `source_artifact_sha`, `source_example_id`, exact prompt/generated IDs, assistant role, no external calls and a separately trusted replay receipt with matching content SHA and source bindings. Actor-state anchors must additionally match the frozen RL actor SHA. Original SFT anchors declare the original SFT artifact rather than pretending gold labels were sampled from the RL actor.

With explicit source masses w_SFT and w_actor, search mass is `1-w_SFT-w_actor` and must remain positive. No anchor is enabled by default. A positive requested anchor mass with no matching records fails; duplicate prepared state keys fail. Per-state `mixture_weight` divides that source's mass by its record count. A future trainer must preserve the declared state/source objective rather than accidentally combining weighted sampling and these weights twice. All-skipped input with zero anchor weights writes an empty diagnostic artifact with `trainable=false`, never invented targets.

## Data-preparation CLI

Use previously collected **TRAIN-only** search records, independently verified receipt manifest and frozen audited TRAIN IDs. This command reads every supplied input record; it does not collect any new model rollouts:

```bash
source /aaa/fionafyang/buddy1/whaleywang/MechET/configs/runtime/shared_host_storage.env
python scripts/prepare_route_self_distill.py \
  --input "$TRAIN_REFIT_CANDIDATES_JSONL" \
  --receipt-manifest "$VERIFIED_RECEIPT_MANIFEST_JSON" \
  --training-targets "$FROZEN_TRAIN_IDS_JSONL" \
  --source-actor-sha "$FROZEN_STANDARD_RL_ACTOR_SHA256" \
  --eos-token-id "$NATIVE_ACTION_EOS_TOKEN_ID" \
  --temperature 1.0 \
  --output-dir "$NEW_CEPH_DISTILL_PREPARATION_DIR"
```

The training-ID JSONL accepts `target_id`, `task_id` or `id` keys, requires unique nonempty IDs and rejects an empty allowlist. All paths must resolve under `/aaa/fionafyang/buddy1/`; existing output directories are refused. Validation/TEST split records, non-allowlisted targets, mixed actors/runtime contracts and duplicate exact decision states are rejected before output publication.

Optional anchors use `--anchors`, `--sft-anchor-source-sha`, `--actor-anchor-source-sha`, `--sft-weight`, and `--actor-weight`. Source options must correspond to the recorded anchor kinds. The supplied receipt manifest must contain the anchor receipts too.

Output contains `targets.jsonl`, `skipped.jsonl` and `manifest.json`. The manifest pins all input hashes, output hashes, preparing-code hash, frozen runtime contract, actor-init lineage, verifier hash, temperature/EOS, source mixture/anchor lineage and prepared/skipped counts. Its `trained_checkpoint` is always null and status is `prepared_not_trained`. It is published last; interrupted writes lacking a manifest cannot be treated as complete. No existing directory is overwritten.

The preparer currently retains normalized states/anchors in memory. It does not claim a measured full-corpus memory profile. If production collection requires sharding, freeze the full source/shard counts and normalize source masses over the complete dataset rather than silently treating each shard as a new full corpus.

## Verification and remaining mandatory work

TDD software checks cover initial missing implementation RED, semantic deduplication/softmax, actor/prompt/node/budget/SHA contamination, TRAIN allowlists, untrusted/unsupported/censored/rejected eligibility, the successful-vs-failed informativeness ruling, EOS/role masks, prompt/padding/zero-weight gradients, anchors, duplicate-state weights, immutable CLI artifacts, zero-coverage reporting and CLI operation with neural imports blocked. CPU CE tests use synthetic logits and backward only; they are not optimizer updates, model predictions or training experiments. The host Torch emits its pre-existing CUDA-driver initialization warning during CPU backward; all test tensors are CPU and no accelerator job or model was created.

Focused verification:37 self-distillation tests pass; combined with27 baseline-adapter and11 adjacent route-data tests,75 pass. The full project suite was not run for this independent focused task. Evidence is saved under `outputs/diagnostics/pr109_baseline_readiness_20261116/distill_*.log`.

Final mandatory work still requires: freeze the **trained** Standard ReFIT-PO actor; collect full-protocol TRAIN-only self-search and verifier receipts; validate the collected evidence's real lineage/coverage; run supervised distillation initialized from that actor; save a separately identified distilled checkpoint; and evaluate distilled versus its own parent under zero-search and matched small/large search budgets on independent frozen targets. Optional +F must be collected/distilled/evaluated separately. Neither lower loss nor this prepared CLI is evidence that search gains have been amortized.


## ReFIT export and existing-forward update integration

`src/mechet/route_self_distill_runtime.py` supplies the integration layer without a launcher or a new training framework:

```python
exported = export_search_states(
    scored_groups,
    actual_actor_sha=authenticated_collection_actor_sha,
    training_task_ids=frozen_train_task_id_to_root_mapping,
    behavior_version=frozen_collection_behavior_version,
    tokenizer_sha=tokenizer_sha,
    tool_schema_sha=tool_schema_sha,
    executor_sha=executor_sha,
    stock_contract_sha=stock_contract_sha,
    budget_sha=continuation_budget_sha,
    reward_contract_sha=frozen_train_reward_contract_sha,
    reward_oracle=frozen_training_process_reward_oracle,
)
```

Input is the scored `score_groups` result: each original report contains `refit.conditioning`, `refit_path`, suffix reports, and `reward_evidence.process_credit`. The exporter requires an explicit TRAIN task-ID allowlist (optionally ID→root mapping). It uses each scored report's **task_id**; it never substitutes the molecule SMILES stored in the original ReFIT identity for a dataset ID. The original `decision_identity` preimage is preserved verbatim under `conditioning.refit_preimage`; its mapped state, prompt IDs, accepted-history prefix, occurrence, behavior version, source tree and action semantic key are checked against the actual journal record. Existing budget/config hashes are retained.

Every noncensored process-credit digest, request-ID sequence, return-to-go vector and reward conservation is revalidated through the existing `process_sequences` verifier. The supplied frozen **training** `ProcessRouteReward` oracle is then called again and its complete result must equal the stored reward evidence. Real terminal status, resource/truncation flags, and hidden-reference-free generation marker must agree. The caller remains responsible for authenticating that `actual_actor_sha` belongs to the collector's loaded snapshot and that the provided reward contract hash identifies this exact oracle/database/stock configuration.

All reused prefix records must be identical. A real sampled request enters export once, with its **original sampled continuation outcome**; a successful descendant suffix cannot relabel the shared prefix as another independent successful draw. A resource/protected censor on a newly sampled action, or `node_censored_receipts`, masks the **entire matched decision node**, including otherwise successful siblings. Survivors cannot stand in for a complete matched comparison. This is reported as `matched_group_censored` and remains ineligible to the target preparer.

The result contains `states`, `receipt_manifest`, summary counts and `status=prepared_not_trained`. Receipt scope is explicitly `executed_journal_and_frozen_training_reward_oracle_not_independent_replay`. `independent_executor_replay_verified` is always false. `verified` here means that already executed journal outcomes were checked against the frozen TRAIN reward oracle; it does **not** claim a fresh replay of native electron actions, independent chemical feasibility or authenticated model weight bytes. The exporter emits no positive target itself; pass its states and receipt manifest to `prepare_state`/the existing data-preparation path.

`distillation_update(actor, optimizer, prepared_examples, cfg)` consumes already prepared state/source masses and q_T targets. Required config includes `actual_actor_sha`, `source_actor_sha` (which must match), `eos_token_id`, `pad_token_id`, `loss_batch_size`, `training_max_padded_tokens`, and `max_grad_norm`; optional existing `policy_precision` and activation-checkpoint settings retain their current semantics.

The update uses existing `token_logps_batch`, `precision_mode`, `training_batches`, `checkpointed_policy_forward` and `synchronize_gradients`. Identical prompt/action forwards are compacted by summing their declared source-mass×q_T weights. The objective sums **all generated native action token log probabilities, including EOS**, with no sequence-length normalization, behavior ratio, GRPO advantage, reference KL, critic, or external teacher. Prompt/tool tokens remain conditioning only. Physical batch partitioning does not change the objective. Existing gradient reduction divides by total state/source mass, then finite gradient clipping and one caller-supplied optimizer step occur. Validation errors precede parameter updates; actor training/eval mode is restored. Distributed reduction follows the existing helpers but has not been exercised on a real multi-GPU job in this turn.

Runtime TDD:17 missing-integration tests failed first, then17 passed. Export tests used actual **CPU** ReFIT collection, executed chemistry journals, a SQLite TRAIN reference database and the real ProcessRouteReward oracle; static sampler responses were software fixtures, not a policy inference experiment. Tiny CPU model tests compared an independent dense sequence-summed gradient against the updater and checked identical parameter updates across physical batch partitions. Corrupt credit/status/conditioning/task IDs/proposals, supplied verified flags, matched-node censoring and invalid training/EOS/weight inputs are covered. Evidence: `outputs/diagnostics/pr109_baseline_readiness_20261116/distill_runtime_{red,green}.log` and `self_distill_runtime_verification.json`.

These functions prepare the final-stage handoff. No new launcher was added. Real TRAIN search collection and distilled-policy optimization remain gated on completion/freezing of the trained Standard actor, followed by independent benchmark evaluation. The CPU Tiny optimizer checks are software verification only and do not produce a trained research checkpoint.
