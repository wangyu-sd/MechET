# Framework and systems strategy

This document covers implementation backends. It does not define the scientific method; see `SCIENTIFIC_THESIS.md` and `TRACE_FAITHFULNESS.md`.

## Decision summary

| Layer | Default | Alternative | Boundary |
|---|---|---|---|
| Main chemistry environment | `TraceOwnedAgentEnv` | `KnowledgeAugmentedAgentEnv` | owns causal state, trace and `finish_trace` |
| Legacy baseline environment | `MechETAgentEnv` | none | independent submitted-proof baseline only |
| Tool-SFT | TRL `SFTTrainer` | custom trainer | must preserve conversational tool structure and assistant mask |
| Small-scale on-policy training | Hugging Face TRL | Prime Verifiers trainer | only after Tool-SFT signal |
| Distributed rollout backend | verl | OpenRLHF | wrap the same environment contract |
| Formal chemistry | RDKit + MechET executor | none | never delegated to an LLM framework |
| Empirical forward evidence | compact forward expert | external ensembles | calibrated soft evidence only |
| Planning extension | Syntheseus | AiZynthFinder | downstream matched-budget evaluation |

## Chemistry ownership

The trace-owned environment owns:

```text
atom-mapped target and current state
electron-container enumeration
imports
explicit source-to-sink action execution
coupled-action semantics
tool budget and state-cycle checks
authoritative committed trace
finish_trace
trace-to-proof compilation
formal execution and endpoint derivation
abstention and rollout trace
```

Training frameworks receive prompts, tool schemas, observations and scalar rewards. They must not duplicate or reinterpret chemical state.

## Tool-SFT first

The preferred initialization is replay-verified Tool-SFT:

```bash
python scripts/train_tool_sft.py \
  --config configs/knowledge/tool_sft_textbook.yaml
```

Before paper-scale training, run a small overfit test and verify:

```text
assistant supervision mask is non-empty
loss decreases
valid tool calls increase
finish_trace is learned
trace-bound execution increases
```

The Tool-SFT adapter hash and data-manifest hash become part of all later checkpoint lineage.

## TRL reference path

Trace-owned dry-run:

```bash
python scripts/train_inverse_agent_trace.py \
  --config configs/agent/inverse_trace_grpo.yaml \
  --dry-run --limit 8
```

Knowledge condition:

```bash
python scripts/train_inverse_agent_knowledge.py \
  --config configs/knowledge/inverse_textbook_trace_grpo.yaml \
  --dry-run --limit 8
```

Legacy baseline:

```bash
python scripts/train_inverse_agent_trl.py \
  --config configs/agent/inverse_trl_grpo.yaml
```

The legacy command must not appear as the default main-method entrypoint.

## On-policy training gate

Begin GRPO or related optimization only when:

- Tool-SFT shows executable learning on frozen validation data;
- trace and reward decomposition are reproducible;
- group members share the same initial task when group-relative advantages are used;
- reward-hacking checks pass;
- executor and environment revisions are frozen;
- the initial adapter lineage is recorded.

Formal failure cannot be offset by forward, evidence, novelty or length rewards.

## Distributed backend migration

verl is the preferred scale backend only after the small TRL pilot is stable.

The adapter boundary is:

```text
TraceOwnedAgentEnv.reset
tool methods
finish_trace
get_reward
state_dict and rollout trace
          |
          v
verl or OpenRLHF agent loop
```

Do not reimplement the executor or maintain a second chemistry semantics inside rollout workers.

Migration trigger:

```text
credible Tool-SFT and small-model learning signal
stable causal intervention results
stable reward decomposition
synchronous rollout latency dominates training
```

## Observability

Agent Lightning or OpenTelemetry may be used to collect transitions and assign credit across larger workflows. They remain optional adapters around the same trace schema.

Every rollout artifact should retain:

```text
task ID
target and expected endpoint when supervised
all tool calls and results
committed versus failed actions
trace digest
compiled proof
terminal reward decomposition
model, adapter, data, environment and executor revisions
```

## Forward evidence integration

The forward expert is trained and calibrated independently before actor integration.

Allowed uses:

```text
post-execution reranking
soft terminal evidence
uncertainty-aware abstention
soft route-edge cost
```

Disallowed use:

```text
rescuing a formal execution failure
hard-pruning without calibrated false-rejection analysis
automatically labelling alternative executable endpoints negative
```

Actor–forward disagreements enter an audit queue. Only independently supported labels may update the forward model.

## Planning

Use frozen offline candidate pools first:

```bash
python scripts/run_syntheseus_search.py \
  --candidate-pool outputs/proof/hypotheses_forward_ranked.jsonl \
  --targets data/benchmarks/planning/targets.smi \
  --inventory data/benchmarks/planning/inventory.smi \
  --output-dir outputs/planning/syntheseus_retrostar \
  --algorithm retro_star
```

Compare planners under identical candidate pools, reaction-model calls, iterations, wall-clock limits and stock. Online actor expansion is a later extension.

Planning cannot rescue failed causal-faithfulness or compositional-generalization claims.

## Recommended stages

### Stage A — causal supervised pilot

```text
build replay-verified rows
measure conversion coverage
run tiny overfit
train matched Tool-SFT conditions
run H1 interventions
```

### Stage B — compositional and evidence study

```text
freeze execution-primitive composition splits
run H2 baselines
run six-condition H3 evidence suite
validate and calibrate forward evidence independently
```

### Stage C — scale and optimization

```text
0.6B / 1.7B / 8B study
formal process RL
optional calibrated forward reward
K-hypothesis inference
```

### Stage D — downstream extension

```text
offline planning
optional online planning
reaction-network analyses only with sufficient evidence
```

## Systems guardrails

- Framework adoption is not a scientific contribution.
- Tool-call syntax is not chemical validity.
- The trace-owned environment is the source of causal state and endpoint semantics.
- `MechETAgentEnv.submit_proof` remains a baseline compatibility path.
- No framework may bypass source licenses, benchmark freezing or checkpoint lineage.
