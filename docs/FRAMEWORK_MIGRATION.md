# Framework migration strategy

This document defines which community frameworks MechET adopts, which remain
optional adapters, and which chemistry responsibilities must remain inside this
repository. The objective is to reuse mature infrastructure without making the
scientific method depend on one rapidly changing agent framework.

## Decision summary

| Layer | Default | Alternative | Decision |
|---|---|---|---|
| Chemistry state and rewards | `MechETAgentEnv` | none | owned by MechET; framework-neutral |
| Small-scale tool-use RL | Hugging Face TRL | Prime Verifiers nano trainer | implemented reference path |
| Distributed agentic RL | verl | OpenRLHF / PRIME-RL | migrate after the TRL experiment is stable |
| Agent tracing and credit assignment | Agent Lightning | native MLflow/OpenTelemetry traces | optional observability adapter |
| Multi-step planning and benchmarking | Syntheseus | AiZynthFinder | Syntheseus is the default benchmark adapter |
| Formal chemistry | RDKit + MechET executor | none | never delegated to an LLM framework |
| Learned forward evidence | compact Forward Electron-Flow Expert | external forward ensembles | frozen soft evidence, never a hard gate |

## Why this split

The chemistry environment changes much more slowly than training frameworks.
`MechETAgentEnv` therefore owns:

- atom-mapped target and current molecular state;
- electron-container enumeration;
- explicit source-to-sink move execution;
- tool budget and state-cycle checks;
- complete `MECH_PROOF v1` execution;
- optional reference endpoint supervision;
- optional frozen forward-expert reward;
- abstention and structured rollout traces.

Training backends only receive prompts, tool schemas, observations and scalar
rewards. They must not duplicate or reinterpret chemistry rules.

## 1. TRL: reference agentic-RL implementation

TRL is the recommended first implementation because `GRPOTrainer` exposes plain
Python tools and a per-rollout `environment_factory`. This matches MechET's
stateful chemistry environment directly and keeps the prototype close to the
Hugging Face model/data ecosystem.

Install:

```bash
pip install -e ".[agent]"
```

Validate the environment and dataset without loading TRL:

```bash
python scripts/train_inverse_agent_trl.py \
  --config configs/agent/inverse_trl_grpo.yaml \
  --dry-run --limit 8
```

Train:

```bash
python scripts/train_inverse_agent_trl.py \
  --config configs/agent/inverse_trl_grpo.yaml
```

The initial run should use Qwen3-0.6B or another small tool-calling model. The
forward checkpoint remains frozen during actor RL.

### TRL limitations

- the agent environment API is still evolving;
- synchronous rollout is not the final high-throughput solution;
- group-relative advantages are noisy if group members start from different
  products;
- model chat templates must preserve earlier tool-call prefixes.

For controlled experiments, all members of a GRPO group should share the same
initial product and differ only in sampled actions.

## 2. verl: scale-out backend

verl is the preferred distributed backend after the environment and rewards have
been validated. It already supports asynchronous server-based rollouts,
multi-turn tool calls and custom agent loops with SGLang or vLLM.

The migration boundary is:

```text
MechETAgentEnv.reset
MechETAgentEnv tool methods
MechETAgentEnv.get_reward
             |
             v
verl tool_agent_loop / custom agent loop
```

Do not rewrite the executor or forward expert inside a verl worker. Wrap the same
environment methods and keep rollout traces in the same JSON schema.

Recommended trigger for migration:

- TRL learning curves and reward decomposition are stable;
- reward hacking audits pass;
- environment unit tests cover the supported reaction families;
- synchronous tool latency becomes the dominant training bottleneck.

## 3. Agent Lightning: optional execution/training decoupling

Agent Lightning is useful when the agent workflow becomes more complicated than
a single trainer loop. It captures agent execution as transitions and separates
runtime, tracing and optimization. This is attractive for alternating inverse
actor and forward-expert experiments and for later multi-step agents.

It is not the default dependency because MechET currently has one principal
learned actor and a deterministic environment. Introduce it only when one of the
following becomes necessary:

- training the same actor through multiple orchestration frameworks;
- hierarchical credit assignment across route-level and reaction-level actions;
- distributed trace collection independent of the trainer;
- optimizing only selected LLM calls in a longer workflow.

The environment trace emitted by `MechETAgentEnv` is intentionally simple so it
can be converted to Agent Lightning or OpenTelemetry spans without changing
chemistry code.

## 4. Prime Verifiers: packaging and evaluation option

Prime Verifiers packages a dataset, harness and rubric as one environment used
for both evaluation and RL. It is a good target for publishing a reproducible
MechET benchmark environment containing:

- frozen product tasks;
- tool definitions;
- formal and forward reward components;
- family/composition OOD splits;
- risk-coverage and abstention metrics.

Because its API and environment hub are developing quickly, it should be an
adapter around `MechETAgentEnv`, not the source of chemistry truth. PRIME-RL is a
possible scale backend when its infrastructure is desired; verl remains the
first self-hosted scale target.

## 5. OpenRLHF: distributed alternative

OpenRLHF supports Ray/vLLM-based multi-turn agent training and common online RL
algorithms. It is a viable alternative when the existing cluster already uses
Ray/DeepSpeed or when REINFORCE++ experiments are desired. MechET should not
maintain simultaneous first-class implementations for both verl and OpenRLHF;
choose one per compute environment and compare algorithms only after matching
rollout and reward contracts.

## 6. Syntheseus: default planning benchmark

Syntheseus provides standardized reaction-model interfaces, Retro*, MCTS and
route-analysis utilities. MechET now includes an offline candidate-pool adapter:

```bash
pip install -e ".[planning]"

python scripts/run_syntheseus_search.py \
  --candidate-pool outputs/proof/hypotheses_forward_ranked.jsonl \
  --targets data/benchmarks/paroutes/targets.smi \
  --inventory data/benchmarks/paroutes/inventory.smi \
  --output-dir outputs/planning/syntheseus_retrostar \
  --algorithm retro_star
```

Use offline pools first so planner comparisons are deterministic and matched in
single-step model-call budget. After this benchmark is frozen, replace the pool
with an online provider implementing the same target-to-candidate contract.

AiZynthFinder remains a useful external baseline, especially for template-based
MCTS, but Syntheseus is preferred for method development because it is designed
to plug custom models into multiple search algorithms and evaluate them under a
shared interface.

## Alternating two-small-model training

The recommended learning schedule is alternating, not simultaneous GAN-style
updates:

1. pretrain the small inverse tool-using actor;
2. pretrain and calibrate the compact forward expert;
3. freeze the forward expert and improve the actor with process and terminal
   rewards;
4. freeze the actor and mine verifier hard negatives;
5. update and recalibrate the forward expert;
6. repeat for a small fixed number of rounds.

The fixed deterministic executor remains the hard gate in every round. Forward
scores, selectivity margins and route costs are soft evidence. A learned verifier
must not permanently prune a formally valid branch unless its false-rejection
rate has been calibrated for that reaction family.

## Migration stages

### Stage A — implemented now

- framework-neutral `MechETAgentEnv`;
- TRL GRPO adapter and dry-run contract;
- Syntheseus offline candidate-pool adapter;
- shared tests for environment rewards and planning-pool normalization.

### Stage B — next engineering milestone

- convert proof-SFT records to multi-turn tool traces;
- train Qwen3-0.6B and 1.7B matched actors;
- add state-level rollout grouping and reward decomposition;
- export rollout traces with stable transition IDs.

### Stage C — scale and planning

- implement the same environment in verl's tool agent loop;
- run Retro* and MCTS in Syntheseus under matched budgets;
- connect online actor expansion only after offline benchmark reproducibility;
- add route-level value learning without modifying single-step chemistry rules.

## Scientific guardrails

- A tool-call framework cannot make a chemical rule correct.
- JSON validity is not chemical validity.
- Forward round-trip recovery is not proof of experimental feasibility.
- Selectivity requires explicit competing pathways or products.
- Actor and verifier checkpoints must have independent lineage and frozen audit
  sets.
- Framework comparisons must use identical products, tool budgets, executor
  version, forward checkpoint and search budget.
