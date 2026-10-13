# Joint import–FLOW endpoint optimization: design for review

Status: proposed implementation; no joint-policy checkpoint or Taiji run yet.
User direction: submit both a code PR and a tested Taiji training task.
This document fixes the proposed architectural scope before implementation.

## Goal and evidence

Preserve product-only inverse-electron inference, with causally interleaved
imports and executor-owned states. Train the Qwen import/controller adapter
as well as the graph FLOW heads against the same bounded terminal endpoint.
Do not report graph-only optimization as this joint condition.

The authorized 2026-10-12 T4 diagnostic used the SHA-pinned Qwen3-0.6B
Stage-I adapter and CRN-JEPA PR18, FP16, seed 17, K1, 40 decisions/32 imports:
8 validation reactions yielded 1 exact endpoint, 6 accepted terminals,
22/22 accepted imports and 34/36 accepted electron events. These are baseline
results, not a post-RL gain. The graph-only train32 diagnostic produced no
update: its first-divergence gate rejects controller/import differences.
One real training row also reproduces a lost explicit-H map during alignment.

## Fixed contracts

- Train source: `data/flower_inverse_tool_sft_action_delta_v1/train.jsonl`,
  257,167 strict executable reactions, SHA-256
  `edc80c5c5eb13d50753c5566c5d6ac1b90b955b1a8dfece915241b2da4a40a75`.
- Valid source: the same artifact's `valid.jsonl`, 2,890 reactions, SHA-256
  `7303a6018850db61594af5854df936778c21a6668e61151f95e2c74c32d22d2a`.
- These are not unqualified FlowER full (257,171/2,890/28,971). Preserve the
  frozen named 4/0/4 upstream-corrupt exclusions; no new row filtering.
- Qwen3-0.6B revision: `c1899de289a04d12100db370d81485cdf75e47ca`.
  Parent adapter SHA: `fbd8db06094fd8029d4cb0cac38cbb88280a442d76f686258194723019b1a1bd`.
  This is the PR83 State-SFT condition, not a Stage-II/history condition.
- CRN PR18 parent SHA:
  `bb2f7fe53e87fc0e2438c27065ceb588dcd38e604959c9d69fb3984885a376ee`.
- Preserve the existing observation/tool schema and fresh product-only maps;
  GT reactants, map correspondence and next states never enter inference.
- Keep K1 evaluation, 40 decisions, 32 imports, 4,096 context tokens and
  384 import-generation tokens. BF16 on a compatible formal host; FP16
  diagnostics are separately labelled and cannot resume BF16 runs.
- No new smoke/pilot/subset task unless explicitly authorized. The earlier
  diagnostic is historical; a formal training submission traverses every
  requested strict train reaction, not 32 or 64 roots.

## Architecture

### 1. Joint ownership, unchanged causal execution

Qwen retains IMPORT/FLOW/FINISH routing and unmapped fragment SMILES generation.
Its existing policy LoRA becomes trainable; its base weights remain frozen.
The graph encoder remains frozen, while its source/sink/CONTINUE/COMMIT heads
are trainable. Tools assign fragment maps, validate events and own transitions.
No complete reaction-world import stage is added at the beginning.

Use the existing `purpose` declarations to track intended electron participants
versus background context. Track actual participation by imported map sets
intersecting subsequently executed moves. Declared purpose is advisory, not a
trusted legal-action mask or endpoint criterion. A solvent or catalyst that
participates in electron flow belongs in active state; a background-only label
does not justify dropping chemically needed atoms. Initially preserve the
existing complete state representation. Separating background-only metadata
from the model-visible graph is a separately versioned observation migration,
not silently activated on the existing SFT checkpoint.

### 2. Frontier and proposal support

Controller/import/finish frontiers are eligible for joint training, not
`controller_blocked` skips. Choose frontiers from actual product-start states;
differences from a recorded environment do not automatically invalidate them.
No late-gold state replaces an autonomous error state.

Construct a finite, deduplicated complete-action support before observing
returns: model-generated IMPORT calls, graph-generated FLOW events and an
explicit legal FINISH proposal. Never add a GT action to on-policy draws.
Invalid sampled actions retain return zero rather than being resampled away.
Store raw proposals, unfiltered draws, executor feedback and successor states.

### 3. Probability and shared endpoint credit

Define a typed hierarchical policy: Qwen chooses the tool route; Qwen generates
IMPORT arguments, the graph heads generate FLOW arguments, and FINISH has no
arguments. Score the complete delegated action with **summed** log probabilities
under that generative policy. Do not add mean token-NLL to mean graph-move-NLL
and call the result a common action probability.

Normalize complete scores over the entire fixed proposal support. Draw K=8
IID with replacement and use a leave-one-out baseline; preserve draw counts
when sharing execution of reward-equivalent successors. The endpoint gradient
must reach Qwen LoRA for import/route decisions and graph heads for FLOW.
Imports receive the endpoint return of their actual mapped state followed by
the same fixed K1 continuation, so correct imports are not rewarded merely
for being syntactically executable. No heuristic first-successor match replaces
terminal endpoint return.

Fix the policy/follower snapshot until one update. Schedule full-budget groups
before observing returns; short-H groups remain replay-only. This defines
conditional fixed-support local improvement, not unbiased whole-trajectory
policy gradient or guaranteed global convergence.

### 4. Verified replay and endpoint parity

Use train-private replay only when its label is valid at the actual state; an
ambiguous alignment must not throw or invent a label. Preserve explicit H atoms
through RDKit alignment. Independently labelled gold-state SFT remains an
auxiliary objective and must not be counted as autonomous RL success.

Use the exact same terminal structural scorer for training returns and heldout
evaluation. An accepted GT endpoint must not receive zero because it retains
an unchanged target molecule. Preserve executor legality and cycle/budget
checks; do not repair scores by adding GT stops at inference.

Keep the existing primary endpoint metric, including its current atom-origin
definition. Heavy-atom-only projection is a labelled diagnostic, not a quiet
replacement for the benchmark. Report catalytic/spectator targets and recorded
intermediate endpoints separately without dropping them from the denominator.

### 5. Artifacts and formal submission

Export the Qwen adapter, graph head and a controller contract together, binding
both parent SHAs, dtype, tokenizer/model revision, source hashes, objective
version and budgets. Resume must restore both optimizers, model weights,
Python/Torch/CUDA RNG and authenticated journal cursors atomically.

Every formal task name starts with `meteor`; use a successful Tencent-image
donor with its exact application group/location and private Ceph init. Worker
progress and a <=60-second heartbeat go to default POD stdout/stderr, never
only to a redirected file. Inspect POD logs/model activity after submission;
queue admission is not training success. Do not replace unrelated tasks.

## Acceptance gates before training submission

1. Explicit-H identity/remapping and GT-unchanged-target reward regressions pass.
2. Import candidate probabilities and scores match the real delegated decoder;
   stop/EOS and truncated output semantics are pinned.
3. Finite categorical expected gradients agree with the joint conditional
   objective; detached follower rewards yield nonzero Qwen/graph gradients
   on appropriate fixtures and zero short-H actor gradients.
4. Participant/context metadata neither masks valid moves nor changes the
   frozen primary endpoint denominator or introduces GT into prompts.
5. Both model exports reload, paired inference uses the exact same controller,
   and resume rejects either model, dtype, source or journal mismatch.
6. The full unit suite passes and an independent implementation review has no
   unresolved blocking findings. No deployment is claimed from unit tests.

## Scope and alternatives

Selected: train the existing Qwen/graph roles jointly against endpoint returns.
Only adding fragment role text leaves the frozen import-selection bottleneck;
moving fragment generation into a new graph generative model requires a new
model/data contract and is outside this change. Any stronger molecular-
background abstraction or new prompt schema requires matched conversion and
training rather than being slipped into checkpoint inference.
