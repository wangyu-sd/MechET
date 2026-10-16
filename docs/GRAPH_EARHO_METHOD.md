# Graph-EARHO: Execution-Anchored Receding-Horizon Optimization for the Graph Policy

**Candidate main post-training method for MechET code PR #83.**
The user forbids new smoke/pilot/subset experiments unless explicitly requested.
The current prepared protocol traverses all 257,167 strict train roots and
evaluates all 2,890 validation / 28,967 strict test reactions, retaining the
four named unsupported test rows in the official 28,971 denominator.
Prepared commands are not submitted jobs or evidence of endpoint improvement.
The previous script "scripts/train_crn_graph_endpoint_rl.py" is a **fixed
last-three-reference-FLOW diagnostic ablation**, NOT the manuscript method.
The main implementation is "scripts/train_crn_graph_earho.py", with
"src/mechet/graph_earho.py" and "src/mechet/graph_endpoint_rl.py".

## Model roles

- Frozen Qwen3-0.6B State-SFT policy: IMPORT / FLOW / FINISH routing,
  missing precursor fragment SMILES and finish decisions.
- Pretrained CRN-JEPA PR18 Graph Transformer and electron FLOW policy:
  source-pointer, compatible sink-pointer, CONTINUE / COMMIT.
- Unchanged MechET executor: electron validation, authoritative state
  transition and terminal precursor extraction.
- ONLY non-encoder graph policy heads receive Graph-EARHO gradients.

## Core algorithm

1. Replay the train-only reference through the verified executor; separately
   roll out the frozen Qwen+Graph actor from a **fresh product-only mapping**.
   Compare **executed chemical successor states** to local references; find
   the first divergence, NOT the first text string difference.
2. Attribute it to Graph FLOW only if the reference and generated tool
   families BOTH equal apply_electron_flow. If Qwen first misroutes,
   hallucinates an import or finishes incorrectly, it is a frozen Qwen
   controller error. The current main skips that root for graph updates and
   records the error. It does not substitute a later gold FLOW state and call
   it an on-policy correctable graph frontier.
3. **Optimize the actor-owned first FLOW error, not a gold suffix.**
   Retain the actual product-start state, imports, action history and private
   maps. H bounds FORWARD continuation from this error; increasing H never
   means providing more correct reference-prefix states to the actor.
   Reference FLOW supervision requires unambiguous model/reference atom-map
   alignment; otherwise no supervised fallback is applied to that state.
   Qwen-first faults are recorded as controller-limited, not graph RL.

4. For each anchor, construct a Top-32 support plus four native full-head
   source/sink/CONTINUE/COMMIT exploration proposals. Freeze the unique ordered
   event set, normalize its COMPLETE scores and sample K=8 **independently with
   replacement**. Execute unique draws but retain every draw's multiplicity.
   Pool accepted source/sink programs only when their private mapped chemical
   state, bond form and stereochemistry agree; preserve product atom origins
   because structural endpoint reward depends on them. Keep rejected events
   as distinct failures. Only one continuation per accepted successor class
   is required.
5. For every distinct valid successor, perform **gold-free bounded
   lookahead** using frozen Qwen routing / imports and graph FLOW actions.
   Selection uses the model and executed state only. In the corrected v4 main
   condition, the follower is exactly the held-out
   K=1 greedy controller, not a multi-branch any-hit search. Only its completed
   exact structural endpoint supplies positive actor evidence (reward 1).
   Local reference-successor matches supply verified supervised replay, not
   positive endpoint actor advantages. Short-H groups remain replay-only even
   when a draw happens to terminate. Full-budget groups are scheduled before
   observing returns; under the declared full budget, wrong/invalid/unfinished
   follower episodes have binary return 0, not a physical-impossibility label.
   Old mixed rewards/class loss require explicit `--policy-objective legacy_class`.
6. Expand each reward-equivalent successor's observed return onto every original
   IID draw. Compute `A_i = R_i - mean(R_j for j != i)` without sign/std gates
   or equal weighting of unique classes. The baseline excludes the own return.
7. The differentiable event score is the sum of source, sink and CONTINUE/COMMIT
   log-probabilities. The COMPLETE pre-return support scores are used for both
   sampling and the loss's normalized `log q`. Optimize `-mean(A_i log q(a_i))`,
   plus separately declared verified replay and frozen-original conditional KL
   over the whole support. Weights/follower stay fixed until one optimizer step.
   The endpoint term is unbiased for the **conditional fixed-support frontier
   objective**, not full product-start policy gradient or PPO/GRPO.
8. Every 16 train roots, promote H by one if at least four **on-policy graph-correctable** groups show
   nonzero policy contrast and actual endpoint support covers at least 10%
   of these eligible on-policy groups; conditional reference resets after
   Qwen-first mistakes NEVER promote H. Maximum H=8 by default. All checks are train-only.
   Continue monitoring product-start behavior, not just oracle-state
   rewards. Save graph weights, optimizer, Python/Torch/CUDA RNG, horizon
   and per-group audits for SHA-locked resume.

The method is **first graph-correctable divergence + reward-equivalent state
reuse + matched-follower terminal credit + normalized IID frontier updates**.
H controls short replay/diagnostic continuation; independently scheduled full
budget groups supply endpoint RL. Full autonomous product-start train episodes
are retained on that schedule, including Qwen-blocked reactions.

## Scientific/implementation boundaries

- Graph-EARHO's current code uses **bounded executable lookahead**, not
  the old EARHO's *separately trained successor critic*. Do not insert a
  nonexistent v_psi into the current manuscript's implementation description.
  A learned value function is a potential controlled extension, not a
  completed main-model component.
- Main routing is the repaired single-generation `greedy_name` decoder.
  The historical `prefix_nll` route is a separately named diagnostic, not a
  silent substitute for the main Qwen decision rule.
- CRN graph policy predicts FLOW only, not MechET BE_DELTA; unsupported
  chemistry is counted. Aromatic/Kekulé, hydrogen, stereochemical and map
  permutation/executor-version parity must be audited.
- Before claims of Graph-EARHO performance, run a real CUDA update, verify
  source/sink/commit event log-probability against original graph decoder,
  compare the same full validation/test protocol before and after training.
  Do not submit the historical valid128/64-root suggestions. There are NO authenticated endpoint
  results from this trainer yet.

## Training command (only on a host with actual checkpoint files)

~~~bash
python scripts/train_crn_graph_earho.py \
  --data data/flower_inverse_tool_sft_action_delta_v1/train.jsonl \
  --source-sha256 "$(sha256sum data/flower_inverse_tool_sft_action_delta_v1/train.jsonl | cut -d' ' -f1)" \
  --output outputs/agent/crn_graph_earho_train257167 \
  --sample-reactions 257167 --rl-updates 257167 --slate 8 \
  --graph-flow-topk 1 --rollout-beam-width 1 --actor-reward endpoint \
  --policy-objective conditional_endpoint --exploration-proposals 4 \
  --continuation-horizon 8 --curriculum-round 16 \
  --policy-adapter outputs/agent/natural_language_event_v2_qwen3_0_6b_seed17 \
  --crn-root /path/to/CRN-JEPA \
  --graph-checkpoint /path/to/CRN-JEPA/outputs/auto_research/pr18/full_corpus_pretrain_v1/train_full_8a100_v2/latest.pt
~~~

Use the resulting "graph_earho_policy_head.pt" under the existing hybrid
evaluator's --graph-rl-head option. The example is NOT a submitted job.
Retain original fixed last-three-reference code for a named ablation;
never attribute its results to the Graph-EARHO method.

## Theoretical audit: what is and is not justified (2026-10-12)

### The objective

The desired product-start objective is

`J(theta) = E[P~D, tau~Pi(theta)] 1{terminal(tau) and D(s_T)=R_GT}`.

`Pi(theta)` is the frozen Qwen controller/import policy composed with the
trainable graph FLOW policy and the unchanged executor. The full state must
include the product, atom-origin bookkeeping, current molecules, accepted
history, visited states and remaining budgets; molecules alone are not a
Markov state when imports, cycle rejection and history condition decisions.
Freezing Qwen restricts the attainable policy family: graph-only optimization
cannot repair a controller/import error before a graph intervention. Traversing
every training reaction does not mean every reaction receives an actor update.

### Why frontier optimization can help

Given a frozen follower, the quantity needed at a frontier is the probability
of recovering the correct endpoint after a particular executed action.
If that action-value is estimated accurately, choosing a higher-value action
can improve a *local conditional* decision without giving the actor a GT suffix.
This is a policy-improvement motivation, not a proof of global convergence.
Frontier resets change the state distribution; greedy deployment, truncated
search, approximation error and shared neural parameters prevent importing a
general monotonic-improvement theorem without its additional assumptions.

### Three conditions the old objective did not satisfy (fixed in v4)

1. **Return semantics:** a first successor matching GT is not an endpoint
   return. Arbitrary +0.2 local shaping is not a potential difference
   `gamma*Phi(x_next)-Phi(x)` with correct terminal boundary conditions, so
   policy-invariance results do not apply. The v3 main actor uses terminal
   endpoint support only; local GT remains an explicit imitation objective.
2. **Follower consistency:** an any-hit Top-2 search estimates a search-assisted
   recovery value, not the value of the deployed K=1 follower. The v3 default
   uses K=1 in both; search-assisted rewards remain a declared ablation.
3. **Sampling/gradient consistency:** Gumbel Top-K without replacement from a
   beam-truncated Top-32 support is not IID sampling from the unconstrained
   graph policy. Equal successor-class weighting, support gating and partially
   sampled class logsumexp add bias. The legacy signed loss remains a surrogate;
   v4's default instead uses normalized IID draws and their individual log q.

For a fixed support S, v4 defines the normalized restricted policy

`q_theta(a|x,S) = exp(log p_theta(a|x)/T) / sum[b in S] exp(log p_theta(b|x)/T)`.

Let `Q_k(s,a)` be binary endpoint return after a executes and the declared K1
follower, frozen at update k, continues under the full remaining budget. The
precise per-update endpoint objective is `J_k(theta)=sum[a in S] q_theta(a|s,S)
Q_k(s,a)`. Support construction/state selection are fixed conditioning variables
for this objective, not terms differentiated through argmax/Top32.

For IID draws `a_1..a_K`, let `b_i=mean[j != i] R_j`. Conditional on s,S,
the other returns are independent of a_i and `E[grad log q(a_i)]=0`, hence

`E[(R_i-b_i) grad log q(a_i)] = grad J_k(theta)`.

The implementation preserves duplicate draw counts, uses the same complete
pre-return scores to sample and train, keeps weights fixed until one update,
and does not drop individual nonterminating full-budget draws. Short-H groups
are excluded as a whole by an independent pre-return schedule; even incidental
short-H hits cannot selectively activate the actor loss. Verified replay and
conditional KL are explicit auxiliary objectives, not proof that pure endpoint
J is unchanged. In particular, the combined gradient is not the gradient of
the endpoint objective alone.

`tests/test_conditional_endpoint_pg.py` exhausts all two-draw outcomes of a
three-action policy and verifies the estimator's expected autograd gradient
against the exact conditional objective. Native pointer proposal scores are
also checked against the source/sink/commit heads on a CPU component fixture.
These prove implementation-level identities, not learned chemical competence.

The result is **conditional fixed-support endpoint REINFORCE**, not unbiased
full product-start policy gradient, PPO, GRPO or a global convergence guarantee.
Frontier occupancy and candidate coverage remain constraints. Native proposals
can expand support outside the beam but cannot guarantee GT inclusion.

### Conservative regularization is limited

In v4 the frozen-original CRN KL uses the whole fixed proposal support, not
only sampled events. It is still a conditional regularizer, not a full-policy
trust region. Moving mass
from `[.4,.4]` to `[.1,.1]` on two observed events leaves their normalized KL
zero while unsampled probability changes `.2 -> .8`. No global monotonic
improvement or no-forgetting guarantee follows from this term. Likewise,
zero explicit advantage for a censored action does not imply its probability
is unchanged: shared parameters and softmax normalization couple actions.

### Reward-equivalent state pooling

Map-free visible chemistry alone is not a sufficient pooling key. With product
origin `[CH3:1][CH2:2][OH:3]`, the two states below have the same visible mixture
`C=O.CBr.CO.[Br-]` but structural projections `C=O.CO` and `C=O.CBr`:

```text
[CH3:1][OH:6].[Br-:4].[CH3:5][Br:7].[CH2:2]=[O:3]
[CH3:5][OH:6].[Br-:7].[CH3:1][Br:4].[CH2:2]=[O:3]
```

Assigning one representative reward to both is therefore incorrect even with
perfect execution. The repaired key conservatively retains private mapped
identity, integer bond form and mapped isomeric serialization. This may split
some genuinely equivalent states but does not silently transfer rewards across
different atom-origin assignments. Continuation dedup also retains import/
mapping counters, cycle history and applicable compressed-history control facts.
The private maps are an executor bookkeeping key, not new model-visible inputs.

### Verified mathematical regression, not a model experiment

`tests/test_graph_earho_objective_boundaries.py` constructs a three-action
categorical policy with probabilities `[.4,.5,.1]`. A local-only positive,
a censored action that is the actual endpoint winner, and an invalid action
produce an old surrogate update that **reduces** the endpoint-winner probability.
The corrected endpoint-only gate does not fabricate a positive actor update
on that group. Another test demonstrates the conditional-KL limitation above.
These are CPU tensor/autograd checks, not a smoke benchmark or measured gain.

### Sources and conclusion

- Sutton et al., [Policy Gradient Methods for Reinforcement Learning with
  Function Approximation](https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf), NeurIPS 1999:
  the estimator must match the policy, state distribution and return.
- Ng, Harada and Russell, [Policy Invariance Under Reward
  Transformations](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf), ICML 1999:
  policy-preserving shaping requires the stated potential-based conditions.
- Kakade and Langford, [Approximately Optimal Approximate Reinforcement
  Learning](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/KakadeLangford-icml2002.pdf), ICML 2002:
  approximate policy improvement depends on restart-distribution coverage,
  action-value accuracy and conservative update assumptions.

The executor-anchored frontier idea is defensible. V4 implements a consistent
estimator for its declared conditional endpoint objective; the legacy surrogate
is a separate ablation. This does not establish global endpoint convergence,
correctness of the frozen Qwen imports, coverage of all chemistry, or held-out
effectiveness. Those remain experimental questions, not unit-test claims.

## Follow-on causal and conservative repairs — 2026-10-12

- Model-owned first graph-error prestate, mapped product origin and accepted
  import/action history are retained as the RL anchor. No late-gold reset
  substitutes for a failed autonomous trajectory.
- Cross-parent Qwen routing and graph FLOW probabilities are not a common
  comparable NLL. Main v4 uses the same K1 greedy continuation as evaluation,
  not an any-hit breadth-first follower.
- Short-H groups contribute replay only, irrespective of incidental success.
  Independently scheduled full-budget groups return 1 for an exact terminal
  endpoint and 0 otherwise. Nontermination at the declared budget is a failure
  of that bounded benchmark policy, not proof of chemical impossibility.
- Gold FLOW replay requires unambiguous map correspondence. A frozen-original
  CRN KL regularizer is computed on the identical *complete fixed support*
  (default coefficient 0.02) and is not a full-policy KL guarantee.
- Resume and exported-controller contracts bind Qwen dtype and 4-bit loading
  mode, so changing follower precision cannot silently resume the same run.
- The pretrained graph encoder is shared across original/adapted policy
  scorer forwards. Train-only endpoints and model outputs are still unverified
  until CUDA checkpoint and product-start experiments complete.
