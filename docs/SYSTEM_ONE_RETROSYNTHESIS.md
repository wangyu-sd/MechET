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

## Current architecture: typed Jev-style v2

The canonical implementation in this PR is now a **typed Jev-style decision
model**, not the earlier post-state marker pointer.  It follows the public
design pattern used by Kev: one shared state, independent typed question
branches, explicit option spans, a `<decide>` readout, and option-isolated
block-causal attention.

```text
authoritative state S_t
        |
        v
      <state>
        |
        +-------------------------------+
        |                               |
        v                               v
 <question: SOURCE>              <question: SINK>
 <option> atom/bond ...          <option> atom ...
 <option> ...                    <option> ...
 <decide>                        <decide>
        |                               |
        |                     symmetric atom-pair composition
        +-------------+-----------------+
                      v
             coupled move score
                      |
                      v
          deterministic executor
                      |
                      v
                    S_{t+1}
```

Question branches share the chemical state but cannot attend to one another.
Within a question, one option span cannot attend to sibling options; only the
question's `<decide>` token aggregates all of its options. This removes the
ordering leakage of a plain causal option list and makes the readout much closer
to the Jev/Kev typed-decision abstraction.

For Qwen3, the implementation reuses existing tokenizer control tokens
(`fim_prefix`, `fim_middle`, `box_start`, `box_end`, `fim_suffix`)
as state/question/option/decide delimiters, so it does not add new embedding
rows. The trainer verifies that every delimiter is a distinct single tokenizer
token before training.

The electron-flow task remains hierarchical. Source candidates are atoms and
present bonds. The sink question exposes each atom once; a symmetric readout
composes every unordered atom pair from the two atom representations. Thus the
numeric sink score space still includes **all** atoms and unordered pairs,
while prompt length grows linearly rather than quadratically in atom count.
Separate typed SOURCE and SINK distributions are combined by a coupled move
head, and the loss requires every reference move in a multi-flow event rather
than rewarding only the easiest move. This factorization was required by the
full-data preflight: explicitly listing all pairs produced 38,622 tokens for
a real 85-atom training state against an 8,192-token cap, with no scientifically
valid way to drop that row or silently truncate options. The factorized full
train/valid audit retains 19,199 / 2,543 electron events, scores up to 3,655
sink candidates, and reduces the maximum actual tokenizer length to 2,927;
zero records exceed the cap.

The previous 0.6B marker-pointer implementation is retained only as a v1
baseline because its measured results are already frozen. Its input contract
(`full_executor_state_plus_gold_independent_post_state_atom_anchors_v1`) is
not relabelled as Jev-style v2.

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

python scripts/train_jev_style_electron_flow.py \
  --train data/.../train.jsonl \
  --valid data/.../valid.jsonl \
  --model Qwen/Qwen3-0.6B \
  --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --epochs 1 \
  --output outputs/system_one/jev_typed_v2_qwen3_0p6b
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
Its recorded `elapsed_s=412.1` starts before that script's training loop and
therefore includes training plus validation; it is **not** validation latency.
This is a quality reference, not a matched-compute speed comparison: its 8B
checkpoint, option-state format, and eight-GPU execution differ from Phase 0.
The frozen PR71 pointer head is evaluated separately on one A100 under the
same executor and action-count rules to obtain a comparable local successor
baseline. This still does not isolate the effects of model size, Stage-II
pretraining, and input representation.

## Jev-style v2 evaluation contract

The v2 trainer writes `system_one_jev_typed_v2_factorized_decision_policy` manifests and
is evaluated separately with:

```bash
python scripts/eval_jev_style_successor.py \
  --checkpoint outputs/system_one/jev_typed_v2_qwen3_0p6b \
  --data data/mech_uspto_31k_natural_language_history_v2/valid.jsonl \
  --split valid \
  --output outputs/system_one/jev_typed_v2_successor_valid
```

Primary local metrics remain paired Recall@1/@4/@8, strict execution and exact
next-state agreement. The typed-v2 model must be retrained before any comparison;
none of the v1 numbers below are attributed to the new architecture.
After a hash-verified typed-v2 checkpoint exists, the prepared one-A100
`scripts/run_taiji_jev_style_typed_v2_successor_1a100.sh` evaluates the full
2,543-event validation and 2,371-event test trace views separately, retaining
per-event cases. It must not be launched from a merely pending training task.

## Frozen v1 marker-pointer baseline (2026-10-05)

The one-A100, one-epoch Qwen3-0.6B run finished successfully. Its checkpoint
manifest pins all 19,199 train and 2,543 validation electron events, source
hashes, model revision, adapter hash and pointer-head hash. The full validation
report gives paired Recall@1 **82.23%**, paired Recall@8 **92.65%**, and all-flow
Recall@8 **85.88%**. Backbone-plus-head validation took 148.15 s for 2,543
decisions (58.26 ms/decision) on one A100. The PR71 8B quality reference is
9.56 percentage points higher at paired Recall@1 and 6.53 points higher at
all-flow Recall@8. PR71's 412.1 s report includes its preceding training loop,
so it cannot be compared with the 148.15 s PR81 validation-only wall time.

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
not an inference policy.

The rule was then frozen and evaluated once on the 2,371-event
current-compiler strict trace-view **test** split (1,253 reactions). The frozen
PR71 Qwen3-8B conditional pointer was also replayed through the same executor
and action-count rules on both splits. Its re-evaluated validation pair metrics
match the originally published PR71 pair metrics to floating-point precision.
This checks the comparator's model-input and head reconstruction.

| Local metric | Validation: 0.6B / PR71 8B | Test: 0.6B / PR71 8B |
|---|---:|---:|
| Paired Recall@1 | 82.23% / 91.78% | 81.74% / 91.44% |
| All-flow Recall@8 | 85.88% / 92.41% | 85.24% / 92.49% |
| Frozen executor-validity backoff: strictly executable | 90.68% / 78.10% | 90.38% / 79.59% |
| Frozen executor-validity backoff: exact next state | 73.81% / 66.38% | **73.85% / 67.44%** |
| Same one-A100 local evaluation-loop wall time | 186.9 s / 385.3 s | 175.5 s / 360.6 s |

The held-out test next-state difference is +6.41 percentage points in favor of
the 0.6B policy (reaction-cluster bootstrap 95% interval +4.22 to +8.61 points;
5,000 resamples, seed 17). On two-flow validation events, the gold pair **set**
occupies the top two positions in 1,614/2,215 for 0.6B versus 1,422/2,215
for PR71 8B; on test the counts are 1,514/2,079 versus 1,358/2,079. Thus a
better *individual* pair Recall@1 need not yield a better executable multi-flow
action. This is a ranking/set-coherence observation, not causal proof that the
smaller backbone is inherently better: the two checkpoints also differ in
Stage-II adaptation, option-state format and training objective. The wall-time
comparison uses the same A100 and frozen data/executor, but the two evaluator
implementations have small differences; it is an operational, not controlled
architecture-only, comparison.

The matched-ID audit, case/report SHA-256 values, paired discordances and
cluster-bootstrap results are frozen in
`outputs/agent/pr81_pr71_matched_successor_20261005/comparison.json`; the
reproducible calculation is `scripts/compare_system_one_pr71_successor.py`.

This evidence supports a **local Phase-0 viability pass** for the spin-off:
lower compute with stable held-out successor agreement. It does not establish
product-start retrosynthesis, reaction endpoint accuracy, or superiority over
the complete MechET system. The 2,371-event test is not the full 3,120-reaction
endpoint benchmark. The next experiment should add IMPORT/FINISH decisions
under the same compact interface and then measure autonomous product-start
rollouts, rather than claiming the local test as a complete task result.

## Next extensions

If Phase 0 works, merge IMPORT and FINISH into the same System-One interface via
an action-family head, then add calibrated abstention/branching.  The long-term
scientific question is whether retrosynthesis is better represented as
calibrated state-conditioned chemical decisions than as autoregressive text
generation.

### Phase-1a action-family gate

This is an exploratory **v1 marker-pointer extension**, run concurrently with
typed-v2 development; it is not a result for the canonical typed-v2 architecture.
It freezes the completed v1 Phase-0 Qwen3-0.6B adapter and
learns a three-way `apply_electron_flow` / `import_fragments` / `finish_trace`
readout from its current-state prefill. It uses **all** current-compiler
mech-USPTO-31k strict trace-view decisions: 32,401 train, 4,288 valid and
4,006 test (10,152 / 1,319 / 1,253 reactions). The frozen input hash, Phase-0
adapter hash and zero-overlength tokenizer preflight are checked before GPU
work. The completed preflight found maximum lengths 1,946 / 1,787 / 1,743,
below the 4,096-token cap. A history-count majority baseline accompanies
accuracy, macro-F1, IMPORT recall and premature-FINISH error.
The history-count majority baseline has 70.57% valid and 70.79% test
accuracy, but zero IMPORT recall on both splits; overall accuracy alone is
therefore insufficient to promote the router.

This is **routing only** at reference executor states. It does not generate
fragment SMILES, use a closed list of observed fragments, or demonstrate an
autonomous product-start rollout. The observed eight IMPORT batches in this
trace view must not be treated as a general fragment vocabulary. A later
open-vocabulary IMPORT interface and closed-loop endpoint evaluation remain
separate promotion gates.

The reproducible entry point is `scripts/train_system_one_action_family.py`;
the one-A100 launcher is `scripts/run_taiji_system_one_phase1a_1a100.sh`.
Outputs include the source/checkpoint/code hashes, frozen prefill features,
selected action-family head, and valid/test confusion matrices.
`scripts/analyze_system_one_action_family.py` checks those hashes again and
reproduces the confusion matrix before extracting confidence, trajectory-depth
strata, IMPORT recall, premature-FINISH rates and high-confidence error cases.

The one-A100 Phase-1a job completed successfully on 2026-10-05. Its valid/test
results are **97.64% / 97.73%** three-way accuracy and **95.95% / 95.95%**
macro-F1. IMPORT recall is **391/426 = 91.78%** valid and **350/382 = 91.62%**
test. On test, 49 of 2,753 non-FINISH decisions are prematurely classified as
FINISH (1.78%). The head selects epoch 30 by validation macro-F1. Frozen
features, head hash and confusion matrices are in
`outputs/agent/system_one_pr81_phase1a_route_31k_20261005/`.

Two source-only controls constrain interpretation. A count-only history
majority classifier achieves 70.79% test accuracy, while a stronger classifier
using the full *past-action summary* achieves 80.33% accuracy and **zero**
IMPORT recall. The learned head's 97.73% is therefore not explained by those
history fields alone. Nevertheless, all 1,253 first decisions on test are
electron events; their perfect routing is a source-distribution feature, not
proof of autonomous chemical reasoning. The remaining 2,753 test decisions
are 96.69% correct. Several confident IMPORT→FINISH errors involve importing
counterions such as `[Cl-]` after the reactive structures have already
separated; the executable reference label is unambiguous for this local task,
but this alone does not establish whether the alternate terminal mixture is
chemically unacceptable. The next gate still requires open-vocabulary IMPORT
arguments and product-start executor rollout.
