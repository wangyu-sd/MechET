# Endpoint-Grounded Process RLVR for inverse electron flow

## Decision

The next experiment is **not another search, representation, verifier, or inference redesign**.

The next experiment is one closed-loop post-training algorithm:

> **Endpoint-Grounded Process RLVR** — the current Qwen electron-flow policy is trained inside the executor environment with event-level process credit, training-only endpoint progress shaping, same-state retry after rejected actions, and a mixed-horizon curriculum that makes exact endpoint reward reachable early in training.

The purpose is to solve the failure exposed by the completed 64-case PR #55 gate:

- independent interactive rollouts produced 91 formally executable terminals but **0/64 exact endpoints**;
- hard-prune search also produced **0/64 exact endpoints** and killed 62/64 frontiers;
- therefore inference-time filtering cannot rescue a policy whose correct-endpoint probability mass is effectively absent under the tested budget.

Search infrastructure remains available for later inference, but it is **not** the next optimization target.

---

## 1. Scientific hypothesis

The current failure is a closed-loop credit-assignment problem rather than a syntax problem.

Supervised training teaches expert next actions on expert states. At inference, one accepted wrong event moves the executor to a state outside the expert trajectory; later actions are then conditioned on the policy's own state distribution. Outcome-only trajectory reward is too sparse because exact precursor recovery is currently near zero from the true product start.

The proposed training objective therefore combines three signals while preserving one causal policy:

1. **event-level executor feedback** for invalid/cyclic actions;
2. **training-only endpoint progress** computed from the authoritative state and frozen gold precursor;
3. **dominant exact endpoint reward** issued only after explicit successful `finish_trace`.

No gold precursor, distance, gold next event, gold reaction centre, or gold remaining-step count is placed in the model observation.

---

## 2. Environment contract

Keep one model and one executor.

For state `S_t` and sampled event `e_t`:

```text
Qwen(current public state/history)
  -> one event or finish_trace
  -> executor
      invalid: state unchanged, typed rejection returned, retry budget consumed
      valid:   commit S_{t+1}
  -> continue
```

Rejected actions **must not kill the state immediately**. The model receives the same public executor error and may retry from the unchanged state.

Initial pilot:

```text
same_state_retry_limit = 2
max_committed_events   = 8
```

After two consecutive rejected proposals at the same state, terminate that rollout with a negative terminal penalty.

A state cycle terminates the rollout immediately.

`finish_trace` remains the only action that may produce a precursor endpoint. The evaluator must never convert a promising intermediate state into an answer.

---

## 3. Training-only endpoint distance

Define a deterministic distance between current authoritative state `S` and frozen gold precursor `R*`.

Do **not** use SMILES edit distance or Tanimoto similarity.

Use a mapped, component-aware bond-electron state distance over atoms shared with the target plus atom-contributing precursor fragments:

```text
D(S, R*) =
    w_bond   * bond-order difference
  + w_charge * formal-charge difference
  + w_conn   * connectivity/component difference
  + w_frag   * missing/extra atom-contributing precursor-fragment difference
```

Requirements:

- invariant to SMILES traversal and disconnected-component order;
- invariant to atom-map labels after matching by frozen mapped identity;
- spectators/free solvents must not dominate the distance;
- exact structural precursor match implies `D = 0`;
- every term must be deterministic and unit tested;
- the distance is **reward-side only** and never serialized into model-visible messages.

The first implementation may reuse existing endpoint/component utilities and mapped bond/charge signatures. Do not add a learned value model.

---

## 4. Potential-based progress shaping

For a committed valid event:

\[
r_t^{progress}
= clip\left(
\frac{D(S_t,R^*) - D(S_{t+1},R^*)}{D(S_{start},R^*) + \epsilon},
-c, c
\right).
\]

Use a small per-event cap, initially:

```text
c = 0.25
```

The cumulative absolute progress contribution per rollout is capped:

```text
|sum progress rewards| <= 1.0
```

Interpretation:

- moves toward the correct precursor receive a small positive reward;
- neutral moves receive approximately zero;
- moves away receive a small negative reward.

Progress reward must never outweigh exact endpoint reward.

---

## 5. Reward contract

Start with the following pilot magnitudes:

```text
exact endpoint after explicit finish_trace : +4.0
wrong explicit finish                      : -1.0
valid committed event                      :  0.0   # legality is not a bonus
invalid proposal                           : -0.25
state cycle                                : -0.50 and terminate
retry exhaustion                           : -0.75 and terminate
unfinished at event budget                 : -1.00
cumulative progress shaping                : clipped to [-1.0, +1.0]
```

The exact scalar values may be exposed in config, but the ordering is fixed:

```text
exact endpoint >> progress shaping > formal legality
```

Do not restore `successful_step_bonus`, `tool_use_bonus`, or a large `formal_execute` reward. PR #55 already showed that formally executable wrong endpoints are abundant.

---

## 6. Event-level credit assignment

Do not train every token in a complete trajectory with one shared scalar advantage.

Each sampled action/event span receives its own return.

For event `t`:

\[
G_t = r_t^{process} + \sum_{j=t}^{T} \gamma^{j-t} r_j,
\]

with exact endpoint reward included at the terminal step and discounted backward through preceding committed events.

Initial pilot:

```text
gamma = 0.95
```

The policy loss should be computed over the generated assistant/tool-call span for each event:

\[
L_{RL} = - \frac{1}{N}\sum_t A_t\,\bar{\log p_\theta(e_t|h_t)}
\]

where the event log-probability is length-normalized over generated tokens.

Use group-relative baselines **within the same start-state group**. RLOO is preferred for the first pilot because it does not require clipping or an old-policy ratio. Do not call the algorithm PPO unless PPO ratios/clipping are actually implemented.

Rejected proposals also contribute a policy term with their negative process reward; they are not silently dropped from training.

---

## 7. KL stabilization

The RL actor starts from one frozen SFT parent checkpoint and should remain close to that policy during the small pilot.

Use either:

- an explicit frozen-reference KL term on sampled event tokens, or
- a lightweight equivalent already supported by the current RL stack.

Initial target:

```text
beta_KL = small, configurable
```

Do not duplicate a full second trainable model. If memory makes a frozen full reference impractical, cache parent-policy token log-probabilities for the sampled events before each update or use an adapter-off reference pass.

Report mean KL per update.

---

## 8. Mixed-horizon curriculum

This is mandatory. Pure product-start endpoint RL is not acceptable for the first run because PR #55 observed 0 exact endpoints under a large inference budget.

For a gold trajectory

\[
S_0^* \rightarrow S_1^* \rightarrow ... \rightarrow S_T^*=R^*,
\]

sample the RL episode start state from the same frozen training trace.

### Phase A — make endpoint reward reachable

Start distribution:

```text
20% product start S0
30% random middle state
50% final 1-2 event suffix state
```

Promotion condition inside the same training run:

```text
near-end exact endpoint success >= 60%
```

### Phase B — shift toward the real task

Then use:

```text
50% product start
25% random middle
25% near-end
```

### Phase C — product-dominant

Once product-start endpoint success is stably non-zero on the frozen monitor set:

```text
80% product start
10% random middle
10% near-end
```

Do not reset optimizer or create new checkpoints between phases. This is one continuous run with a changing episode-start distribution.

All reported validation/test endpoint metrics start from the true product only.

---

## 9. Parent checkpoint

The algorithm must be representation-agnostic.

For the immediate pilot, use one reproducible existing event-action SFT checkpoint that satisfies all of the following:

1. produces the current executor-consumable elementary event action;
2. has a frozen base-model revision and adapter hash;
3. can execute at least some events from product start;
4. exposes no gold precursor or gold remaining-event count at inference.

Do not claim that adopting this RL algorithm adopts PR #53 as the final representation. If checkpoint-8037 is used only because it is the currently available trained event actor, record it explicitly as a **warm-start implementation vehicle**, not as a paper-method decision.

Do not change representation and RL algorithm in the same pilot.

---

## 10. Small pilot dataset

Do not launch full-data RL first.

Build a frozen training pilot with:

```text
512 training reactions
selection seed = 17
```

Stratify the sample by gold event length so it contains short, medium, and long trajectories.

Use a separate frozen product-start monitor set:

```text
128 validation reactions
selection seed = 17
```

No monitor ID may appear in RL updates.

Store exact ID manifests and hashes.

---

## 11. Rollout/update budget

Initial pilot configuration:

```text
group size                    = 8 rollouts per sampled start state
max committed events          = 8
same-state retry limit        = 2
gamma                         = 0.95
endpoint exact reward         = +4.0
wrong finish                  = -1.0
invalid action                = -0.25
cycle                         = -0.50
retry exhausted               = -0.75
unfinished                    = -1.00
progress total cap            = +/-1.0
optimizer                     = RLOO/group-relative REINFORCE
```

Keep inference/search disabled during the pilot. Evaluation is K=1 product-start unless a separately named oracle diagnostic is requested.

The purpose is to test whether the policy itself acquires endpoint probability mass.

---

## 12. Required metrics during training

Log at minimum:

### By start horizon

- exact endpoint rate;
- explicit finish rate;
- mean committed events;
- mean rejected proposals;
- retry-exhaustion rate;
- cycle rate;
- mean endpoint distance at start and terminal state;
- mean cumulative progress shaping;
- effective group rate / nonzero-advantage group rate.

### Product-start monitor

Every fixed evaluation interval, run the frozen 128-case monitor from the true product with no gold-visible information and report:

- EndpointPass@1;
- formal terminal rate;
- invalid-action rate;
- explicit-finish rate;
- mean generated tokens;
- mean trajectory length.

**EndpointPass@1 is the promotion metric.**

ExecutePass alone is not a success criterion.

---

## 13. Promotion and stop gates

### Immediate stop

Stop the pilot if any of the following holds after a meaningful update window:

- near-end start cannot exceed 20% exact endpoint despite direct reward access;
- endpoint distance decreases but exact endpoint remains zero and wrong-finish rate rises;
- KL diverges while endpoint metrics do not improve;
- invalid-action rate increases materially without endpoint gain;
- reward implementation leaks any gold information into model-visible messages.

### Continue to larger run

Promote only if:

```text
near-end endpoint success >= 60%
AND
product-start EndpointPass@1 > 5% on the frozen 128-case monitor
AND
product-start EndpointPass improves over the parent SFT checkpoint
```

The `>5%` value is a pilot promotion threshold, not a paper target.

If the pilot passes, expand to the full training universe while preserving the exact reward/environment contract.

---

## 14. File-level implementation tasks for Codex

Implement this plan on the current PR branch without deleting the completed PR #55 search audit.

### A. Deterministic endpoint distance

Add a module, suggested path:

```text
src/mechet/endpoint_progress.py
```

Responsibilities:

- canonical mapped endpoint/component decomposition;
- deterministic bond/charge/connectivity/fragment distance;
- exact-match zero distance;
- potential-difference reward;
- cumulative shaping cap helper.

Add focused tests including component reordering, atom-map relabeling where valid, charge differences, bond-order differences, missing precursor fragments, and exact match.

### B. RL environment

Add or extend a trace-owned environment, suggested path:

```text
src/mechet/endpoint_process_rl_env.py
```

Responsibilities:

- reset from product/middle/near-end gold states;
- keep gold endpoint private on the reward side;
- expose only public executor observations;
- same-state retry after rejected event;
- retry exhaustion and cycle termination;
- explicit `finish_trace` endpoint reward;
- event-level reward ledger.

The environment must prove by test that gold endpoint/distance never appears in model-visible messages.

### C. Event-level RL objective

Extend the existing RL code rather than creating an unrelated framework.

Likely files:

```text
src/mechet/rlvr.py
scripts/train_mechet_rlvr.py
```

or add a clearly named specialized trainer if the existing abstraction is too coupled.

Required changes:

- retain each generated event span and its token log-probability;
- compute per-event returns/advantages;
- include rejected proposal spans;
- group-relative RLOO baseline per start-state group;
- optional KL/reference term;
- detailed reward/advantage logging.

Do not collapse back to one trajectory-wide scalar advantage.

### D. Curriculum sampler

Add deterministic start-state sampling from frozen gold traces.

Suggested helper:

```text
src/mechet/mixed_horizon.py
```

It must implement the three curriculum distributions above and log the sampled start horizon.

### E. Pilot builders/config

Add:

```text
scripts/build_endpoint_rlvr_pilot.py
configs/agent/endpoint_process_rlvr_pilot.yaml
```

The builder must freeze:

- 512 train IDs;
- 128 disjoint monitor IDs;
- selection hashes;
- parent checkpoint hash/revision;
- reward contract version.

### F. Pilot evaluator

Add a product-start-only evaluator that never reads reward-side gold information during generation.

Suggested path:

```text
scripts/eval_endpoint_process_rlvr.py
```

It should compare parent SFT vs current RL checkpoint on the same 128 IDs.

### G. Tests/CI

Add focused CI for:

- endpoint distance invariants;
- no reward leakage;
- retry semantics;
- event-level return assignment;
- exact finish reward dominance;
- mixed-horizon sampler reproducibility;
- parent-vs-RL evaluation contract.

Do not submit a Taiji training task until all deterministic tests and one tiny optimizer smoke pass.

---

## 15. Tiny optimizer smoke before Taiji

Before the 512-reaction pilot, run a tiny local/GPU smoke:

```text
8-16 reactions
2-4 groups
1 optimizer step
```

Required checks:

- finite loss;
- nonzero event-level advantages when rewards differ;
- rejected events receive negative credit;
- exact endpoint event produces the dominant return;
- KL term finite if enabled;
- no gold text in prompt/transcript;
- parameters actually update.

Only after this passes should Codex prepare the Taiji launch script.

---

## 16. What not to do

Do not add in this iteration:

- a new molecular representation;
- another atom-address scheme;
- beam search in the RL rollout;
- forward critic/value model;
- self-review model;
- MCTS;
- reaction-template labels;
- extra endpoint reranker;
- automatic best-prefix answer extraction;
- new benchmark filtering.

The experiment must isolate one question:

> Can closed-loop, process-verified, endpoint-grounded RL move probability mass from formally executable wrong trajectories toward exact precursor endpoints?

---

## 17. Success criterion

The smallest meaningful success is not better ExecutePass.

It is:

\[
\boxed{\text{product-start EndpointPass@1 rises from approximately zero to reproducibly non-zero.}}
\]

The pilot promotion threshold is >5% on the frozen 128-case product-start monitor, together with strong near-end success and no reward leakage.

Only after that result should constrained search be reintroduced as a test-time amplifier.