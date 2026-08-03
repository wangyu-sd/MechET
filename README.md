<div align="center">

# MechET

**Causal and compositional electron-flow reasoning for retrosynthesis**

[![Proof tests](https://github.com/wangyu-sd/MechET/actions/workflows/proof-tests.yml/badge.svg)](https://github.com/wangyu-sd/MechET/actions/workflows/proof-tests.yml)
[![Agent framework tests](https://github.com/wangyu-sd/MechET/actions/workflows/agent-framework-tests.yml/badge.svg)](https://github.com/wangyu-sd/MechET/actions/workflows/agent-framework-tests.yml)
[![Knowledge ablation tests](https://github.com/wangyu-sd/MechET/actions/workflows/knowledge-ablation-tests.yml/badge.svg)](https://github.com/wangyu-sd/MechET/actions/workflows/knowledge-ablation-tests.yml)
[![Forward expert tests](https://github.com/wangyu-sd/MechET/actions/workflows/forward-expert-tests.yml/badge.svg)](https://github.com/wangyu-sd/MechET/actions/workflows/forward-expert-tests.yml)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

[Scientific question](#scientific-question) · [Main method](#main-method) · [Experiments](#experiments) · [Quickstart](#quickstart) · [Current status](#current-status) · [Documentation](#documentation)

</div>

---

## One-sentence contribution

MechET formulates retrosynthesis as **causal program induction over executable electron-flow actions**: an environment-owned action trace is the sole computational source of the proof and precursor, enabling controlled tests of causal faithfulness and primitive-seen/composition-unseen generalization.

## Scientific question

> Can mechanistic reasoning in retrosynthesis be made causal and compositional, rather than merely plausible in language?

A language model can generate a correct precursor and a plausible-looking explanation without using that explanation to obtain the answer. MechET replaces the independent rationale/answer pair with an executable chemical program:

```text
atom-mapped product
  -> explicit source-to-sink electron-flow actions
  -> environment-owned molecular-state transitions
  -> committed trace
  -> deterministic trace-to-proof compilation
  -> executor-derived structural precursors
```

The main study tests three hypotheses:

1. **Causal faithfulness:** the generated trace, not an independent answer channel, determines the endpoint.
2. **Compositional basis:** local electron-flow execution primitives can be recombined into mechanism compositions not seen during training.
3. **Evidence separation:** formal executability and empirical chemical support are different evidence layers.

The scientific definitions and permitted claims are frozen in [`docs/SCIENTIFIC_THESIS.md`](docs/SCIENTIFIC_THESIS.md).

## Main method

### Trace-owned electron-flow program

The main environment is `TraceOwnedAgentEnv` or its knowledge-augmented subclass. The actor may:

```text
inspect_state
retrieve_textbook_guidance      optional soft evidence
retrieve_primitives             optional structured knowledge anchors
import_fragment
apply_electron_move
apply_coupled_electron_moves
finish_trace
abstain
```

`submit_proof` is disabled in the main method. `finish_trace` deterministically compiles the committed environment trace into `MECH_PROOF v1`, replays the proof and derives the precursor.

```text
model actions
  -> authoritative trace
  -> compiled proof
  -> precursor
```

A correct endpoint cannot be credited independently of an incompatible trace.

### Deterministic executor

The executor is not trained. It checks atom maps, imports, bond and charge preconditions, electron accounting, sanitization, reachability and proof topology. A formal failure is a hard rejection.

### Evidence layers

Natural-language textbook passages and structured mechanistic knowledge anchors may guide the actor. An independent compact forward expert may provide process, target-recovery, competitor and uncertainty evidence. These are **soft evidence**:

- they do not define the endpoint;
- they do not directly establish formal validity;
- they cannot override execution failure;
- they do not establish kinetics, yield or experimental success.

### Main method and baselines

| Condition | Role |
|---|---|
| Trace-owned Tool-CoT with `finish_trace` | main method |
| Knowledge-augmented trace-owned Tool-CoT | main evidence condition |
| Independent complete `MECH_PROOF v1` generation | required baseline |
| Loose tool trace followed by submitted proof | faithfulness baseline |
| Outcome-only, free-form CoT, state-CoT and net-edit | representation baselines |
| Forward expert and multistep planning | evidence/downstream extensions |

## Terminology

### Electron-flow execution primitive

A local executable action such as `LP -> BOND`, `BOND -> ATOM` or `BOND -> BOND`. Execution primitives define the causal action space and compositional split.

### Mechanistic knowledge anchor

A structured provenance-aware record with molecular-role patterns, candidate moves, preconditions, warnings and competitors. Knowledge anchors guide retrieval; they are not the primitive basis used to define MechComp-OOD.

## Experiments

The main result sequence is intentionally narrow:

### H1 — causal reasoning

Compare direct, answer-bearing CoT, complete-proof, loose-trace and trace-owned models. Intervene on the tool process by removing, shuffling or replacing observations and by disabling state inspection or intermediate execution.

Primary metrics:

```text
structural precursor Top-k
ExecutePass
trace–proof agreement
trace–endpoint agreement
answer–reasoning disagreement
intervention effect size
tool-failure recovery and abstention
```

### H2 — compositional reasoning

Hold out complete execution-primitive compositions while ensuring each constituent primitive appears in training.

Compare:

```text
direct answer
free-form CoT
state-CoT
net edit
complete proof
trace-owned Tool-CoT
```

Report performance versus composition novelty, proof length, topology, reaction family and scaffold.

### H3 — evidence separation

Build matched conditions from the same stable-ID intersection:

```text
trace_no_knowledge
trace_length_matched_irrelevant
trace_textbook_rag
trace_structured_anchors
trace_text_plus_anchors
direct_textbook_rag
```

A textbook claim requires `textbook > trace-only` and `textbook > length-matched irrelevant text`. A forward-evidence claim requires independent calibration and explicit competitors where selectivity is discussed.

The complete experiment contract is [`docs/PROOF_CENTRIC_EXPERIMENT_PLAN.md`](docs/PROOF_CENTRIC_EXPERIMENT_PLAN.md). The ordered commands and stopping gates are in [`docs/EXECUTION_PLAN.md`](docs/EXECUTION_PLAN.md).

## Quickstart

### Install

```bash
git clone https://github.com/wangyu-sd/MechET.git
cd MechET

pip install -e ".[dev]"
pip install -e ".[agent]"       # Tool-SFT and trace-owned actor training
pip install -e ".[knowledge]"   # textbook corpus and knowledge anchors
pip install -e ".[forward]"     # optional independent forward evidence
pip install -e ".[planning]"    # optional planning extension
```

### Execute a complete-proof baseline

```python
from mechet.proof_program import ChargeAction, ProofEdge, ProofProgram
from mechet.proof_program import format_proof_output, verify_proof

program = ProofProgram(
    target_smiles="[CH3:1][OH:2]",
    roots={"s0": ["[Br-:3]"]},
    precursor_state_id="s1",
    edges=[ProofEdge(
        "s0", "s1",
        bonds=[(1, 2, -1), (1, 3, +1)],
        lone_pairs=[(2, +2), (3, -2)],
        charges=[ChargeAction(2, 0, -1), ChargeAction(3, -1, 0)],
    )],
)
result = verify_proof(
    format_proof_output(program),
    expected_precursor="[CH3:1][Br:3].[OH-:2]",
)
print(result["execute_ok"], result["endpoint_exact"])
```

### Inspect the trace-owned main environment

```python
import json
from mechet.trace_agent_env import TraceOwnedAgentEnv

env = TraceOwnedAgentEnv()
print(env.reset(target_smiles="[CH3:1][OH:2]"))
print(json.loads(env.submit_proof("invented"))["code"])
# FREE_FORM_PROOF_DISABLED
```

The successful terminal operation is `finish_trace` after one or more committed electron-flow transitions.

### Build textbook evidence

```bash
python scripts/download_mechanistic_sources.py \
  --registry knowledge/source_registry.yaml \
  download \
  --source iupac_goldbook_terms \
  --source rxno \
  --source wikibooks_organic_chemistry \
  --output knowledge/raw

python scripts/build_textbook_corpus.py \
  --download-root knowledge/raw \
  --output knowledge/corpus/passages.jsonl

python scripts/index_textbook_corpus.py \
  --corpus knowledge/corpus/passages.jsonl \
  --output knowledge/corpus/bm25_index.json
```

### Build replay-verified Tool-SFT rows

```bash
python scripts/build_textbook_tool_sft.py \
  --input data/mechet_proof_clean/train.jsonl \
  --corpus knowledge/corpus/passages.jsonl \
  --output data/textbook_tool_sft/train.jsonl \
  --quarantine data/textbook_tool_sft/quarantine.jsonl

python scripts/build_textbook_tool_sft.py \
  --input data/mechet_proof_clean/train.jsonl \
  --corpus knowledge/corpus/passages.jsonl \
  --output data/textbook_tool_sft/train_text_and_anchors.jsonl \
  --enable-structured-primitives
```

### Build all matched evidence conditions

```bash
python scripts/build_knowledge_ablation_suite.py \
  --config configs/experiments/textbook_ablation.yaml
```

The anchors-only and direct open-book conditions are derived automatically; no separately prepared datasets are required.

Validate alignment and deterministic budgets:

```bash
python scripts/validate_experiment_contract.py \
  --condition none=data/knowledge_ablation/v2/trace_no_knowledge.jsonl \
  --condition irrelevant=data/knowledge_ablation/v2/trace_length_matched_irrelevant.jsonl \
  --condition textbook=data/knowledge_ablation/v2/trace_textbook_rag.jsonl \
  --condition anchors=data/knowledge_ablation/v2/trace_structured_anchors.jsonl \
  --condition combined=data/knowledge_ablation/v2/trace_text_plus_anchors.jsonl \
  --condition direct=data/knowledge_ablation/v2/direct_textbook_rag.jsonl \
  --output outputs/contracts/evidence_conditions.json
```

### Train Tool-SFT

```bash
python scripts/train_tool_sft.py \
  --config configs/knowledge/tool_sft_textbook.yaml \
  --dry-run

python scripts/train_tool_sft.py \
  --config configs/knowledge/tool_sft_textbook.yaml
```

Run a 32–128-example overfit test before paper-scale training. On-policy training begins only after Tool-SFT demonstrates a credible executable-learning signal.

### Train the trace-owned main actor

```bash
python scripts/train_inverse_agent_trace.py \
  --config configs/agent/inverse_trace_grpo.yaml \
  --dry-run --limit 8

python scripts/train_inverse_agent_trace.py \
  --config configs/agent/inverse_trace_grpo.yaml
```

Knowledge-augmented condition:

```bash
python scripts/train_inverse_agent_knowledge.py \
  --config configs/knowledge/inverse_textbook_trace_grpo.yaml \
  --dry-run --limit 8
```

Legacy complete-proof/loose-trace baseline:

```bash
python scripts/train_inverse_agent_trl.py \
  --config configs/agent/inverse_trl_grpo.yaml
```

## Current status

MechET is a research preview. The causal execution infrastructure is implemented; the scientific claims remain to be established by frozen experiments.

| Component | Status |
|---|---|
| Deterministic proof executor | available |
| Trace-owned environment and `finish_trace` | available |
| Free-form proof rejection in the main method | available |
| Proof-to-trace conservative conversion | available; full-data coverage not yet reported |
| Textbook corpus and bounded retrieval | available |
| Structured mechanistic knowledge anchors | available as soft evidence |
| Replay-verified Tool-SFT builder | available |
| Automatically derived six-condition evidence suite | available |
| Matched experiment contract validator | available; tokenizer-specific token audit still required for final runs |
| Paper-scale Tool-SFT and trace-owned checkpoints | not released |
| Causal intervention benchmark results | not released |
| Primitive-seen/composition-unseen results | not released |
| Calibrated forward-evidence results | not released |
| Multistep planning results | optional extension; not released |
| Kinetic, transition-state or experimental validation | external evidence required |

## Scientific boundaries

- Formal executability is not evidence of a low barrier, favorable kinetics, high yield or experimental success.
- A retrieved passage or knowledge-anchor match does not prove an inferred electron-flow action.
- A learned forward score is soft evidence and cannot override formal execution.
- Lack of a knowledge-anchor match does not imply chemical impossibility.
- The current main scope is mapped, closed-shell, two-electron polar chemistry.
- The main method does not infer a unique physical mechanism from the product alone.

## Documentation

Start with:

1. [`docs/SCIENTIFIC_THESIS.md`](docs/SCIENTIFIC_THESIS.md) — scientific question, hypotheses, terminology and permitted claims.
2. [`docs/TRACE_FAITHFULNESS.md`](docs/TRACE_FAITHFULNESS.md) — main causal inference contract.
3. [`docs/PROOF_CENTRIC_EXPERIMENT_PLAN.md`](docs/PROOF_CENTRIC_EXPERIMENT_PLAN.md) — authoritative experiment definitions.
4. [`docs/EXECUTION_PLAN.md`](docs/EXECUTION_PLAN.md) — ordered commands, gates and stop conditions.
5. [`docs/TOOL_SFT.md`](docs/TOOL_SFT.md) — replay-verified supervised trajectories.
6. [`docs/KNOWLEDGE_ABLATIONS.md`](docs/KNOWLEDGE_ABLATIONS.md) — matched evidence conditions and causal controls.
7. [`docs/MECHANISTIC_PRIMITIVE_LIBRARY.md`](docs/MECHANISTIC_PRIMITIVE_LIBRARY.md) — structured mechanistic knowledge anchors and provenance.
8. [`docs/FORWARD_ELECTRON_EXPERT.md`](docs/FORWARD_ELECTRON_EXPERT.md) — optional independent empirical evidence.
9. [`docs/README.md`](docs/README.md) — complete documentation map and deprecations.
