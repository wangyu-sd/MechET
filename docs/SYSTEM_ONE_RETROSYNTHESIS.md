# System-One retrosynthesis: Jev-style prefill-only chemical decisions

## Status

Spin-off research path. This does **not** replace the frozen MechET paper protocol
or Issue #79 experiments.

## Motivation

The expensive part of the current closed-loop MechET policy is autoregressive
language generation.  Most local chemistry decisions are not open-ended text:
they select action types and graph locations from the current executor-owned
state.  A decision model is therefore a better architectural match.

This PR follows the public *Jev-like* design pattern implemented by projects
such as Kev and any2jev, without claiming access to TypeSafe AI's private Jev
architecture.  The reusable idea is:

1. encode the state once;
2. represent legal options explicitly;
3. score all options with a small pointer/readout head;
4. execute the selected chemical action;
5. encode the successor state for the next decision.

Reference implementations:
- https://github.com/jaredpalmer/kev
- https://github.com/hwfengcs/any2jev

## Phase-0 architecture

The first implementation targets electron-flow localization because this is
already represented as a finite, executor-grounded option set in MechET.

```text
authoritative state S_t
        |
        v
small Qwen backbone + LoRA
(one prefill, no generation)
        |
        +--> atom/bond option states
        |
        v
Jev-style pointer readout
        |
        v
coupled source/sink distribution
        |
        v
deterministic MechET executor
        |
        v
S_{t+1}
```

The model is deliberately hierarchical rather than flattening every complete
reaction program into one classification problem.  Source candidates are atoms
and present bonds; sink candidates are atoms and unordered atom pairs.  The
primary training loss requires each reference source/sink pair in a multi-flow
event. One high-scoring flow is not treated as the whole event. For causal
atom-option readout, the current observation appends a gold-independent list
of atom handles *after* the full executor state. This is a derived input
contract, not byte-identical to the Stage-II text prefix.

## Why this is deployable on limited compute

- default backbone: `Qwen/Qwen3-0.6B`;
- LoRA instead of full fine-tuning;
- one causal-LM prefill per decision;
- no vLLM sampling loop and no JSON generation;
- pointer head is sub-million-scale relative to the backbone;
- first experiment can run on one H20/A100.

## Included entry points

```bash
python scripts/export_system_one_electron_flow.py \
  --input data/.../train.jsonl \
  --output outputs/system_one/train.decisions.jsonl

python scripts/train_system_one_electron_flow.py \
  --train data/.../train.jsonl \
  --valid data/.../valid.jsonl \
  --model Qwen/Qwen3-0.6B \
  --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --epochs 1 \
  --output outputs/system_one/qwen3_0p6b
```

The exporter is an audit artifact; the trainer reads the original Stage-II rows,
verifies frozen split hashes and training permission, and appends only the
gold-independent option handles. Its `preflight.json` records source hashes,
decision denominators, token lengths and the exact input contract. The default
experiment is the mech-USPTO-31k **strict executable trace view** (10,152 /
1,319 / 1,253 reactions; 19,199 / 2,543 / 2,371 electron-event decisions),
not the full 24,959 / 3,120 / 3,120 endpoint benchmark.

## Promotion metrics

Phase 0 should report only metrics that directly test the decision architecture:

- coupled source/sink Recall@1/@4/@8;
- successor-state agreement after executor application;
- latency per decision;
- tokens processed per decision;
- peak GPU memory.

Current Phase-0 code measures paired localization, latency, tokens and memory.
The training report records successor agreement as unavailable rather than
mistaking site recall for chemical execution. A separate frozen-checkpoint
evaluator reconstructs temporary executor-only atom maps from the visible
annotated SMILES, checks that the regenerated public inventory preserves atom
addresses and canonical isomeric chemistry (allowing equivalent RDKit stereo
text normalization), and replays the prediction against the reference successor:

```bash
python scripts/eval_system_one_successor.py \
  --checkpoint outputs/agent/system_one_pr81_phase0_31k_20261005_v2/full \
  --valid data/mech_uspto_31k_natural_language_history_v2/valid.jsonl \
  --output outputs/agent/system_one_pr81_phase0_successor_valid
```

The evaluator reports fixed one-flow and two-flow policies that do not use the
answer, plus an explicitly labelled **oracle move-count diagnostic**. All are
local evaluations at reference current states; none is product-start endpoint
accuracy. Executor failure, successor chemistry and move-count strata are kept
separate, and test is not loaded. Promotion still requires competitive local
quality before a closed-loop rollout.

The evaluator's atom-address reconstruction has been independently audited on
the frozen train and valid event splits: GT pair indices replay to exactly the
same executor result as the original action for **19,199/19,199 train** and
**2,543/2,543 valid** events. Four train states acquire a different textual
`@`/`@@` serialization when RDKit removes temporary maps; their alias-indexed
graphs and canonical isomeric structures are unchanged. The replay check
allows only this chemically equivalent normalization, not atom-address or
stereochemical changes. These four rows remain in training.

Only after local decision quality is competitive should product-start endpoint
rollout be attempted.  No new RL algorithm is part of Phase 0.

For scale, the existing PR71 Qwen3-8B conditional pointer reports paired
Recall@1 = 0.9178 and all-flow Recall@8 = 0.9241 on the same frozen 2,543-event
validation source (`valid_sha256=e4b68bd9f52ed5b0c24b2453d1ee6197a3686c354db9aa3f4c3689647d74a995`).
Its recorded validation wall time is 412.1 s. This is a **quality reference**,
not a matched-compute causal comparison: its 8B checkpoint, option-state
format, and eight-GPU execution differ from Phase 0. No successor metric was
reported for that reference, so it must not be used as a successor baseline.

## Measured Phase-0 validation (2026-10-05)

The one-A100, one-epoch Qwen3-0.6B run finished successfully. Its checkpoint
manifest pins all 19,199 train and 2,543 validation electron events, source
hashes, model revision, adapter hash and pointer-head hash. The full validation
report gives paired Recall@1 **82.23%**, paired Recall@8 **92.65%**, and all-flow
Recall@8 **85.88%**. Backbone-plus-head validation took 148.15 s for 2,543
decisions (58.26 ms/decision) on one A100. The PR71 8B quality reference is
9.56 percentage points higher at paired Recall@1 and 6.53 points higher at
all-flow Recall@8; its different architecture and eight-GPU validation runtime
are not a matched latency comparison.

The separate executor-grounded validation replay succeeded for every GT event
(2,543/2,543). Results below are **local next-state agreement at reference
current states**, not product-start retrosynthesis or reaction endpoint accuracy:

| Gold-independent action selection | Strictly executes | Exact next state |
|---|---:|---:|
| fixed one flow | 1,941/2,543 (76.33%) | 308/2,543 (12.11%) |
| fixed two flows | 1,846/2,543 (72.59%) | 1,617/2,543 (63.59%) |
| two flows, back off to one only if execution fails | 2,306/2,543 (90.68%) | 1,877/2,543 (73.81%) |

The backoff rule was chosen **after** examining the fixed-count validation
results. It uses only executor validity, not reference flow count, but its
73.81% validation number is exploratory and must not be presented as a
pre-registered primary outcome. The oracle-count diagnostic, which reads the
reference move count, reaches 1,925/2,543 (75.70%) exact successors and is
not an inference policy. The frozen backoff rule is next evaluated once on the
2,371-event current-compiler strict trace-view **test** split. That split has
1,253 reactions and is not the full 3,120-reaction endpoint benchmark. No
product-start endpoint or end-to-end rollout claim follows from this phase.

## Next extensions

If Phase 0 works, merge IMPORT and FINISH into the same System-One interface via
an action-family head, then add calibrated abstention/branching.  The long-term
scientific question is whether retrosynthesis is better represented as
calibrated state-conditioned chemical decisions than as autoregressive text
generation.
