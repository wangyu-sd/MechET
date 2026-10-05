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
next-state agreement. The typed-v2 model was trained from the pinned Qwen3-0.6B
revision for one epoch on all 19,199 training electron events; the completed
adapter and typed head are under
`outputs/agent/system_one_pr81_jev_typed_v2_31k_20261005/full/`.
None of the v1 numbers below are attributed to this new architecture.
The hash-verified checkpoint was evaluated on one A100 by
`scripts/run_taiji_jev_style_typed_v2_successor_1a100.sh` on the full
2,543-event validation and 2,371-event test trace views separately, retaining
per-event cases. The training task had ended successfully before evaluation
was submitted.
Before loading weights, the evaluator checks the completed manifest against the
preflight's source counts, model revision, input contract, zero-overlength audit,
and exact trainer/encoder SHA-256 values. This prevents a later source-code
change from silently redefining the typed-v2 checkpoint. The evaluator records
the same frozen source and gold-replay fields as the v1
and PR71 reports. `scripts/compare_jev_style_successor.py` then checks identical
event IDs and reference chemistry, independently reconstructs the frozen
executor-validity backoff from each model's `fixed1`/`fixed2` cases, checks the
aggregate report totals, and reports typed-v2 minus v1 and typed-v2 minus PR71
paired successor differences with reaction-cluster bootstrap intervals. It also
retains one- versus two-flow strata and the two-flow top-two **set** hit rate,
which distinguishes per-pair recall from coherent multi-flow actions. The
completed reports and paired comparison are under
`outputs/agent/system_one_pr81_jev_typed_v2_successor_20261005/` and
`outputs/agent/pr81_typed_v2_three_model_comparison_20261005.json`.
The comparison JSON has SHA-256
`6ecc1c26b146d6f0c45fe2eb5311679713cb07f4668bf3ccd5b3338056438d3c`.

An independent held-out test tokenizer audit, using the pinned Qwen3-0.6B
revision and the same typed encoder, covered all **2,371** test electron events:
maximum input length **2,508**, mean **1,158.45**, and **zero** inputs over the
trained 8,192-token cap. The evaluator also checks this cap on every event at
runtime and records its maximum observed length; no test row is silently
truncated.

```bash
shared=/aaa/fionafyang/buddy1/whaleywang/MechET
PYTHONPATH=.:src python scripts/compare_jev_style_successor.py \
  --typed-v2-valid "$shared/outputs/agent/system_one_pr81_jev_typed_v2_successor_20261005/valid" \
  --typed-v2-test "$shared/outputs/agent/system_one_pr81_jev_typed_v2_successor_20261005/test" \
  --marker-v1-valid "$shared/outputs/agent/system_one_pr81_phase0_successor_valid_20261005" \
  --marker-v1-test "$shared/outputs/agent/system_one_pr81_phase0_successor_test_20261005" \
  --pr71-8b-valid "$shared/outputs/agent/pr81_pr71_matched_successor_20261005/valid" \
  --pr71-8b-test "$shared/outputs/agent/pr81_pr71_matched_successor_20261005/test" \
  --output "$shared/outputs/agent/pr81_typed_v2_three_model_comparison_20261005.json"
```

### Completed typed-v2 local successor results (2026-10-05)

The Taiji training and full valid/test successor-replay tasks both ended
successfully. Every reference event replayed through the executor, with no
held-out input exceeding the trained 8,192-token cap. The comparison script
verified identical event IDs, source hashes and reference successors before
computing the following **executor-validity-backoff** results. This policy
tries the top two flows and uses the top one only when that action fails;
it does not use the reference move count.

| Split | Events | Typed-v2 0.6B exact next state | Marker-v1 0.6B | PR71 8B |
|---|---:|---:|---:|---:|
| Valid | 2,543 | 2,182 (85.80%) | 1,877 (73.81%) | 1,688 (66.38%) |
| Test | 2,371 | 2,075 (87.52%) | 1,751 (73.85%) | 1,599 (67.44%) |

On test, typed-v2 executes strictly in 2,246/2,371 events (94.73%), has
paired source/sink Recall@1 of 92.91%, and puts both reference flows in its
top two on 1,810/2,079 two-flow events (87.06%). Exact next-state rates are
264/292 (90.41%) for one-flow and 1,811/2,079 (87.11%) for two-flow events.
Against marker-v1, the paired test improvement is **+13.67 percentage points**
(reaction-cluster bootstrap 95% CI +11.90 to +15.37); against PR71 8B it is
**+20.08 points** (95% CI +18.29 to +21.93). These are same-event quality
comparisons, **not** an architecture-only causal ablation: model size,
pretraining and representations differ. Evaluator wall times likewise do not
establish a matched-compute speedup.

This establishes stronger *local decisions at reference current states*, not
autonomous product-start retrosynthesis, generated IMPORT arguments, or a
3,120-reaction full-endpoint result. Promotion to the latter still requires a
typed-v2 action-family/IMPORT/FINISH policy and closed-loop executor rollout.

Joining the frozen v1 Phase-1a action-family router with typed-v2 electron
decisions along every *reference* trajectory gives **927/1,319 (70.28%)**
validation and **937/1,253 (74.78%)** test reactions with all recorded local
decisions agreeing. The same router with v1 electrons gave 689/1,319 (52.24%)
and 666/1,253 (53.15%). On test, the first mismatch is an electron successor
in 274 reactions versus an action-family route in 42; 35/49 reactions with
five electron events pass the typed-v2 local checks, compared with 2/49 under
v1. These gains diagnose reduced reference-path error accumulation, but
recorded IMPORT fragment arguments and reference current states are still
assumed. They are **not** autonomous endpoint hit rates. The hash-checked
valid/test reports are under
`outputs/agent/system_one_pr81_typed_v2_reference_path_20261005/`.

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

The v1 held-out cases further localize the failure: among 2,079 two-flow
events, the gold pair **set** occupies the top two slots in 1,514, and all
1,514 execute to the reference next state. Only one additional two-flow case
is exact with a different top-two set. For the 292 one-flow events, fixed-one
recovers 275 exact successors, whereas executor-validity backoff recovers 236:
39 correct single-flow actions are displaced by a different but executable
two-flow action. Action-count choice therefore explains only 39/2,371 =
1.64 percentage points of the current policy's gap to its oracle-count
diagnostic; improving coherent two-flow pair-set ranking is the larger target
for typed-v2.

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

IMPORT recall is not uniform across the seven test batches: the dominant
`[H][H]` plus `[Na+]` batch is recognized in 277/292 cases (94.86%), while
`[Cl-]` is recognized in 8/13 (61.54%) and `[Br-]` in 10/13 (76.92%). The
rare-batch denominators are small, so these are failure-mode indicators rather
than stable per-chemistry estimates. The hash-checked error-analysis script
now records every batch's support and full three-way confusion.

### IMPORT-space generalization boundary

`scripts/audit_system_one_import_space.py` measures observed IMPORT actions
without turning them into a model vocabulary. The current-compiler 31k trace
view has only **8 distinct train fragment batches** across 3,050 IMPORT
decisions; all 7 batches in each held-out split already occur in train. In
contrast, the broader strict-executable FlowER validation trace view alone
has **2,064 distinct batches and 790 distinct fragment SMILES** across 7,919
IMPORT decisions. The reports preserve the source hashes under
`outputs/agent/system_one_pr81_import_space_20261005/`.

The complete strict-executable FlowER import audit counts **706,902** train
IMPORT decisions, **63,108** distinct train batches and **24,654** distinct
train fragment SMILES. The held-out test view has 79,281 IMPORT decisions:
5,836 of its 12,373 distinct batches are absent from train, affecting
**6,548 / 79,281 = 8.26%** of test IMPORT decisions. The corresponding
validation figure is **620 / 7,919 = 7.83%**. These are exact batch-string
novelty rates, not chemical-equivalence or reaction-success rates.

A closed eight-way fragment classifier could look strong on the 31k pilot but
fail the intended open-chemistry task. Phase-1a only routes to IMPORT; it does
not claim fragment-generation competence. Any subsequent IMPORT argument
model must be open-vocabulary and evaluated on the broader held-out fragment
distribution before promoting a full System-One rollout.

As a deliberately limited **oracle-free proposal baseline**, the train-only
Morgan-fingerprint nearest-state retriever in
`scripts/eval_system_one_import_retrieval.py` uses the strict final-mixture target and
current state to retrieve distinct training IMPORT batches. It verifies that
simple disjoint-fragment composition reproduces every authoritative IMPORT
successor in this trace view: 3,050/3,050 train, 426/426 valid and 382/382
test. On reference current states, its exact-chemical-batch Top-1 recall is
392/426 (92.02%) valid and 345/382 (90.31%) test; test Top-2 is 366/382
(95.81%). A train-majority batch gives 292/382 (76.44%) on test. No held-out
IMPORT target mixture is an exact train mixture, but **all seven held-out
batches occur in the eight-batch training support**. The Top-8 recall of 100%
is consequently trivial and is not evidence of open-vocabulary generalization.
Results and per-case proposals are under
`outputs/agent/system_one_pr81_import_retrieval_v2_20261005/`. This
reference-state candidate recall does not establish off-trajectory robustness
or product-start endpoint accuracy; it only supplies a transparent pilot
proposal source for the subsequent closed-loop diagnostic.

### Product-start hybrid pilot gate

**Input-contract correction (2026-10-05):** In this strict natural-language
trace view, `target_smiles` is the executor's **final molecular mixture**,
which often contains salts or byproducts. It is not the selected principal
product used as input by the separate 3,120-reaction mech-USPTO full-endpoint
benchmark. A hash-verified, reaction-ID-matched audit against that benchmark
found exact equality of the two input SMILES in only **45/1,319 validation**
and **37/1,253 test** reactions. The strict input has more molecular
components in **1,270/1,319 validation** and **1,212/1,253 test** reactions.
The benchmark principal product is an exact component of the strict mixture
in only 1,098/1,319 validation and 1,074/1,253 test reactions; merely
dismissing extra components would therefore still leave unmatched inputs.
After ignoring stereochemical annotation, however, the principal product is
a component in **all 10,152/1,319/1,253 train/valid/test** strict-trace
reactions. Thus the remaining component mismatches are stereochemical, not
connectivity mismatches under this audit; the principal-product benchmark may
omit stereo information visible in the strict mixture. The residual
final-mixture context has only 14 distinct stereo-agnostic component batches
and 13 distinct fragments in the strict training view; every validation/test
residual batch occurs in train. This makes a train-only context-completion
pilot feasible **within this trace view**, but does not establish coverage of
the 1,867 remaining full-endpoint test reactions without executable traces.
For example, strict test reaction 4 includes a bromide byproduct in its
input, whereas the principal-product benchmark input does not. All rollout
results below must therefore be read as **final-mixture-start, strict-trace-
view diagnostics**, despite the historical `product_start` path and
`product_only` report-scope strings. Neither the 75.26% result nor any other
number in this section is a standard principal-product-only retrosynthesis
accuracy. The audits are
`outputs/agent/system_one_pr81_input_gap_{valid,test}_20261005.json`, made by
`scripts/audit_system_one_full_endpoint_input_gap.py`, and bind every rollout
case target to its strict source row. Their SHA-256 values are
`3ca787e171f5d4d2dda245043f924d4866bb5cf7cf9f5cb457197d5837cbe879`
and `10110365ad5a4c5d1f1343529abf624de2d586e4063d4529b55f79b8d1d67c51`.
The expanded stereo-agnostic context audits are preserved separately as
`outputs/agent/system_one_pr81_input_gap_{train,valid,test}_v2_20261005.json`.
Their respective SHA-256 values are
`9ee126c51f4e9b412b3818cd2416f879ef32346a92770d132f26dd1f8706751a`,
`e67168ce633970e0ae700b1490f167bb779a3639abe38b5610c765786efceecd`,
and `45e76fc2ff74fc3c31f0cba09abb04001a10d0d924fb60c141ab9e8dec748843`.
Each frozen rollout directory also contains an `INPUT_CONTRACT_CORRECTION.json`
sidecar; report and case hashes were not modified. A genuine full-endpoint experiment needs
a matched principal-product input/training contract and the full 3,120-row
denominator; simply renaming these artifacts cannot provide one.

As a first train-only context-completion diagnostic,
`scripts/eval_system_one_context_retrieval.py` retrieves one of the 14
stereo-agnostic residual component batches from the 10,152 strict-trace
training reactions using only the full-endpoint principal product's Morgan
fingerprint (radius 2, 2,048 bits). No held-out mixture or context label is
used to propose a batch. On the 1,319 validation reactions, exact batch
Top-1 is **837/1,319 (63.46%)**, Top-3 coverage is **1,198/1,319 (90.83%)**,
and the train-majority batch gives 634/1,319 (48.07%). With the rule unchanged,
the 1,253 test reactions give **800/1,253 (63.85%)** Top-1,
**1,138/1,253 (90.82%)** Top-3, and 583/1,253 (46.53%) train-majority.
Only three held-out principal products in each split are exact train-product
matches. Reports/cases are under
`outputs/agent/system_one_pr81_context_retrieval_{valid,test}_20261005/`;
report SHA-256 values are
`b1fdae248be5d4cd42563682031ca97acbda9594b853c0878fde112a1700bf39`
and `9507dafa77f8381c2cde4a79f391256c38cb521c417441f4c4fa327add22852e`.
This tests only the context proposal, not stereo restoration, electron-policy
rollout, endpoint accuracy, or coverage of all 3,120 reactions.

The principal-product context bridge in
`scripts/eval_system_one_context_bridge.py` now runs the frozen hybrid policy
from the full-endpoint **principal product plus a train-only predicted context
batch**. Before constructing policy input, it projects the retrieval artifact
onto product and predicted batch only; held-out reference context and
precursor are scorer-only. It retains the strict-trace-view **full precursor**
as its terminal scorer, so it is an input-contract diagnostic on the 1,319-row
overlap, **not** the 3,120-row structural endpoint benchmark. The 64-reaction
validation pilot ended successfully at **26/64 (40.63%)** exact versus
**48/64 (75.00%)** on the same IDs supplied with the original complete final
mixture. All 33 byte-exact reconstructed inputs reproduced the frozen
mixture-start rollout, including 26 successes; the other 31 inputs gave no
strict endpoint hits. The pilot report/cases are under
`outputs/agent/system_one_pr81_context_bridge_valid64_20261005/`.

The complete strict validation bridge task then ended successfully on one
A100. Exact full-precursor endpoints are **511/1,319 (38.74%)**, versus
**944/1,319 (71.57%)** for the same frozen policy on the original complete
mixtures. The bridge reconstructs the exact original input in 700 reactions;
all 700 reproduce the frozen rollout and 509 hit the reference endpoint. Of
the 619 other inputs, two happen to reach that endpoint. The retrieved
stereo-agnostic context batch is correct in 837 reactions, but 137 of these
still do not reproduce the original mixture byte-for-byte, consistent with
stereochemical information absent from the principal-product input. In fact,
all 137 are chemically equal after removing stereochemistry and none is
isomeric-SMILES equal. The
retrieval Top-3 contains the reference context in 1,198/1,319 reactions; this
is **candidate coverage**, not a deployable reranked endpoint result. Full
bridge report/cases are under
`outputs/agent/system_one_pr81_context_bridge_validfull_20261005/` and the
paired, hash-checked audit is
`outputs/agent/system_one_pr81_context_bridge_validfull_comparison_20261005.json`.
Their SHA-256 values are respectively
`1780e5b69b6dffeefbc308d58286d1e51385e4741f5c694ea550b7a8bdb9447a`,
`4b7a99a80572bb80d790fb3ac18ef23efb44d36e77f7489f75efddb5cf82ab9d`,
and `4a35985dbba9ee2560f9024dfa1b4affda425fd4e725c9c0c458ad0de915aa47`.

The bridge/ranking rule was then frozen and evaluated once on **all 1,253
held-out test reactions in the strict trace view**. The one-A100 task ended
successfully. Strict full-precursor exact is **553/1,253 (44.13%)**, versus
**943/1,253 (75.26%)** for the same policy given the reference final mixture.
It reconstructs 696 original inputs exactly; all 696 reproduce the complete
frozen action traces, and 551 reach the endpoint. Only two of the 557 other
inputs reach it. Context-batch Top-1 is 800/1,253, while 104 of these 800
still lack stereochemical annotation required for an isomeric match to the
strict input; all 104 are otherwise chemically equal without stereo. The
test report, cases and paired audit are respectively under
`outputs/agent/system_one_pr81_context_bridge_testfull_20261005/` and
`outputs/agent/system_one_pr81_context_bridge_testfull_comparison_20261005.json`;
their SHA-256 values are
`7604839b22145e199e70e54a8948e5fd62be2c4c58ec46db632433d4c78220a5`,
`4f821db6b85ea6e5cb97a353ff9de6e7a8751ef0fd4579cadb393f37936fa44a`,
and `7e25ccbf531ae7f553a6ae62b54b4e6cc2784e8c591d34213967b3c17b9b6510`.

### Frozen weighted context proposal follow-up

The input-gap diagnostic also tested whether a slightly stronger **train-only
proposal**, without changing any System-One weights or electron decisions,
could raise principal-product bridge success. On validation, weighted voting
over the 11 nearest training principal products, using squared Morgan
Tanimoto similarity, was selected from a small k/p grid; that single rule
was then frozen for test. The procedure never reads held-out reference
context until scoring. It is still a retrieval component, not a learned
principal-product-compatible trajectory policy.

| Strict trace-view split | Nearest-context Top-1 | Weighted-context Top-1 | Nearest bridge endpoint | Weighted bridge endpoint |
|---|---:|---:|---:|---:|
| Valid, 1,319 reactions | 837 (63.46%) | 922 (69.90%) | 511 (38.74%) | **578 (43.82%)** |
| Test, 1,253 reactions | 800 (63.85%) | 893 (71.27%) | 553 (44.13%) | **615 (49.08%)** |

Both one-A100 weighted-bridge tasks ended successfully and used the same
frozen route adapter, typed electron adapter, IMPORT retriever, executor
rule, reaction IDs and strict full-precursor scorer as the nearest-context
bridge. On test, endpoint predictions change in both directions: 108
previously wrong reactions become exact, while 46 previously exact reactions
become wrong, for a net gain of 62/1,253 (+4.95 points). Validation has
107 gains and 40 regressions, net +67/1,319 (+5.08 points). The weighted
bridge reconstructs the original strict input byte-for-byte in 779 valid
and 780 test reactions. The paired auditor verifies that all such reconstructed
inputs reproduce the **complete frozen action trace**; 575 valid and 613 test
reactions in that group reach the exact endpoint. This supports an input
completion effect, not an improvement to the electron-flow policy itself.

Retrieval reports/cases are under
`outputs/agent/system_one_pr81_context_knn_{valid,test}_20261005/` and
weighted bridge reports/cases under
`outputs/agent/system_one_pr81_context_knn_bridge_{valid,test}full_20261005/`.
The retrieval report hashes are
`4278b197aaa44c8957922a5af9a012a74a4ae3275e1ce6f71e3e18871bfa898a`
and `734afe62e0aa3abf3c37f0366a1f0b00aa70081fd3d161ef2a8d2636b1c43471`.
The paired bridge audit hashes are
`14f2ad8e1716e5949ed76c69febe4f58ca3e1680f02604b36edd3b85bb075379`
and `a67742e3ec3904fc57648cb22b1d243a7224caedec2a1c6413d128a6df6f5e02`.
These remain strict-trace-view, full-precursor diagnostics, **not** a full
3,120-row principal-product structural endpoint benchmark.

This large observed gap means that the current final-mixture-trained policy
must not be promoted as a principal-product-only system. In this trace view,
extra final-mixture components and sometimes stereochemistry enter its first
observation, and product-only inference cannot be evaluated by silently
supplying them. A matched product-only training/evaluation contract or a
gold-independent multi-context proposal/ranking policy is required before a
full endpoint claim. The **output** contract also differs: the frozen
reaction-ID audit found the strict replay's full precursor equal to the
full-endpoint dataset's `reactants_unmapped` in only 812/1,319 validation and
816/1,253 test overlaps. The standard benchmark instead scores structural
precursors. Consequently, simply extending this strict full-precursor scorer
to all 3,120 products would still not be the standard endpoint evaluation.

`scripts/eval_system_one_product_start_pilot.py` now composes the frozen v1
three-way router, the frozen typed-v2 electron policy, and **train-only**
IMPORT retrieval into a final-mixture-start executor loop. Every next prompt is
built from that mixture, the policy's own accepted state and compact action
history; neither action type, fragment choice, electron pair nor stopping uses
the held-out answer. A deterministic SHA-256 reaction-ID sample selects 64
validation mixtures for the first GPU pilot. The typed and route inputs were
checked with the pinned Qwen tokenizer against their teacher-forced training
encodings; the first decision is byte-identical in each format. Rebuilt IMPORT
states match all 3,858 authoritative IMPORT successors across the three splits,
and the frozen valid/test terminal answers match the reference executor outputs
under explicit-H-preserving structural canonicalization.

The pilot is **hybrid**, not a unified typed-v2 action-family model. It checks
strict execution of each electron event and exact component-preserving
precursor endpoints, but does not yet compile the whole predicted trajectory
to MECH_PROOF. A completed pilot must report all 64 reactions, failure modes
and latency before any larger run; an allocated Taiji task alone is not a
result. Its configured one-A100 runner is
`scripts/run_taiji_system_one_product_start_valid64_1a100.sh`.
The first launch failed before producing any case because the new wrapper
passed one extra hidden vector to the four-argument typed head. The wrapper
now matches the frozen evaluator's call signature, with a regression test that
executes the actual typed-head forward path. The failed task/output remain
archived; the replacement uses a distinct task flag and output directory.
That replacement reached actual closed-loop decisions, then exposed a second
runtime-only mismatch: the inventory preserves an explicit `[H]` atom while
the position parser had used RDKit's default hydrogen-removal setting. The
parser now retains explicit hydrogens. The corrected parser still accepts
**all** frozen 19,199 / 2,543 / 2,371 train/valid/test electron events with
zero unsupported records and zero changed reference candidate inventories
relative to the old parser. The partial second-attempt cases are preserved;
neither failed launch is reported as an endpoint result.

The corrected third launch ended successfully and wrote all 64 selected
validation cases to
`outputs/agent/system_one_pr81_product_start_valid64_v3_20261005/`.
Starting from the final mixture with no reference actions in the policy input, the
hybrid policy reaches an exact, component-preserving precursor endpoint in
**48/64 (75.00%)** cases. It finishes with the wrong endpoint in 8 cases,
encounters a strict electron-event executor failure in 7, and finishes early
with pending work in 1. Five of the seven electron failures occur before any
electron event is accepted. Six of the eight completed-but-wrong cases make
no IMPORT decision; these include both missing-context and alternative
disconnection outcomes and are not all attributable to one error type. The
evaluator's measured rollout time after model loading is 16.27 seconds for
64 cases; this excludes task startup and checkpoint loading and is not an
end-to-end production latency measurement. `report.json` SHA-256 is
`75b4aa6dd9d71bbe515713ef93aeea030f583c20fccbff87859d43d571f6f97f`;
`cases.jsonl` SHA-256 is
`e5b8e536a5e020eb13b46152e167a3b07f981cdddcdbfa0f8e4401f550395189`.
This is a 64-reaction **validation pilot** from the 1,319-reaction strict
trace view, not the full 3,120-reaction endpoint benchmark or a test-set
result. It is also not formal whole-trajectory MECH_PROOF success: only each
electron event is checked by the executor.

The same frozen hybrid policy then completed **all 1,319 validation reactions**
in this strict trace view, with no subsampling and no test-set loading. Exact
precursor endpoints were **915/1,319 (69.37%)**. Of 1,119 FINISHED episodes,
204 reached a wrong endpoint; 184 stopped on an invalid electron event, 14 on
premature/pending FINISH, and 2 at the 12-action budget. The one-A100 task
`meteor_mechet_pr81_system_one_product_start_validfull_1a100_qy_20261005_01`
ended successfully; measured rollout time after model loading was 334.40 s.
The full report and all 1,319 cases are under
`outputs/agent/system_one_pr81_product_start_validfull_20261005/`. Their
SHA-256 values are `8fe6138ea8d49ca4feca85bb4a467ec1458061cb7dc68dfe7b70641914982c6d`
and `293eea192d2de2659f5b2aa06b008170b317a0916676b897d82466bfeba9670b`.

An offline analysis independently recomputed every endpoint and terminal
count from the case file. Of the 184 failed electron events, 131 occurred
before any electron event had been accepted. Replaying each frozen Top-8
ranking against the **predicted** current state found a legal singleton in
183/184 failures, without using the reference action. This establishes only
local executability: no alternative was rolled out to an endpoint, and a
legal electronic action need not be the recorded or chemically preferred
reaction. Of the 204 wrong FINISHED endpoints, 119 made no IMPORT decision;
10 reactions repeated an IMPORT batch. The analysis is saved as
`failure_analysis.json` beside the full report. These outcomes motivate a
separately labelled executor-constrained fallback experiment on the same
validation IDs, reported below; they do not revise the frozen baseline result.

A retrospective GT-IMPORT stratification (held-out actions read only after
rollout, never by the policy) locates a second bottleneck. Of 426 reactions
whose reference trajectory imports a fragment batch, the policy imports at
least once in 325, chooses the exact reference chemistry batch first in 296,
and reaches the exact endpoint in 220. Thus 101 reference-IMPORT reactions
miss IMPORT altogether and 29 import an incorrect first batch. On the 893
reference trajectories without IMPORT, endpoint exact is 695. This breakdown
is in `failure_analysis_v2.json`; it is not an oracle-assisted policy result.

The separately frozen validation-only follow-up enables
`--legality-backoff`: only if the original top-two and top-one electron
actions both fail the executor, it tries the remaining predicted Top-8
singletons in rank order and accepts the first executable one. It does not
use a reference action or endpoint score to choose among alternatives. On
the same 1,319 validation IDs and identical checkpoint hashes, endpoint exact
rose from **915/1,319 (69.37%)** to **944/1,319 (71.57%)**, a paired gain of
**29 reactions / +2.20 percentage points** (reaction bootstrap 95% CI
[+1.44, +3.03] points; 5,000 resamples, seed 17). The paired auditor verified
that every episode not stopped by a baseline electron failure was unchanged,
and that the predicted Top-8 ranking was identical at each failed decision:
29 reactions improved, none worsened. Of 184 original electron-failure
episodes, 183 were locally rescued; 173 eventually FINISHED, while only 29
reached the exact endpoint. Final electron failures fell to 7, with 15
premature/pending FINISH and 5 action-budget outcomes. Measured rollout time
after model loading was 381.51 s, versus 334.40 s without the fallback.
The paired audit is
`outputs/agent/system_one_pr81_product_start_top8legal_comparison_20261005.json`;
its SHA-256 is `53e0109660776023f32a960df61c373887d94634b8717df5714cc8ae2c2a3e1b`.
The candidate report and cases are under
`outputs/agent/system_one_pr81_product_start_top8legal_validfull_20261005/`.
This is executor-constrained decoding on the strict trace-view validation set,
not a learned policy improvement, full 3,120-reaction benchmark, test result,
or whole-trajectory MECH_PROOF verification.

With the rule frozen after validation, both policies were run on **all 1,253
held-out test reactions of the same strict executable trace view** (source
SHA-256 `7aa93a98361fb6fbe4f09465706deab6111a15a2167bc721a9e67b386dc78131`).
Both one-A100 tasks ended successfully. The original top-two/top-one policy
reached **923/1,253 (73.66%)** exact precursor endpoints; the Top-8 legality
backoff reached **943/1,253 (75.26%)**. The paired gain is **20 reactions /
+1.60 percentage points** (reaction bootstrap 95% CI [+0.96, +2.31] points;
5,000 resamples, seed 17). The auditor confirmed identical source and weight
hashes, identical reaction IDs and reference endpoints, unchanged trajectories
outside baseline electron failures, and identical frozen rankings at each
failed decision: 20 improved, none worsened. Electron-execution failures fell
from 157 to 3; of 157 original failed episodes, 156 were locally rescued,
145 eventually FINISHED, but only 20 reached the exact endpoint. Thus
executor legality is useful but clearly insufficient for chemical endpoint
selection. The full test artifacts are
`outputs/agent/system_one_pr81_product_start_{testfull,top8legal_testfull}_20261005/`;
the paired comparison is
`outputs/agent/system_one_pr81_product_start_top8legal_test_comparison_20261005.json`
(SHA-256 `9688593637e1a60fb07477cf9f9344edda7773ddcdaa5f7e9aee9ec4f4e87f1d`).
These are **final-mixture-start strict-trace-view** endpoint results, not the full
3,120-reaction mech-USPTO-31k endpoint benchmark or formal proof success.

### Runtime-observation parity gate for closed-loop evaluation

`scripts/audit_system_one_observation_parity.py` reconstructs each decision's
model-visible prompt from the strict final-mixture target, the **previous accepted tool
result**, and the accumulated compact history. It never uses the current or a
future reference action to construct that decision's input. On the frozen
mech-USPTO-31k trace view, 32,397/32,401 train prompts, all 4,288 validation
prompts, and all 4,006 test prompts are byte-identical to SFT. The remaining
four train prompts differ only in RDKit's equivalent `@`/`@@` serialization of
the same alias-indexed stereochemical state; history and all other prompt text
are byte-identical. A chemical or atom-address mismatch fails the audit.

This parity audit establishes teacher-forced observation construction, not
autonomous product-start rollout. Fragment-argument selection and
off-reference-state behavior are measured separately by the hybrid rollout
above, not by this parity audit. Reproduce the parity audit with:

```bash
PYTHONPATH=.:src python scripts/audit_system_one_observation_parity.py \
  --data data/mech_uspto_31k_natural_language_history_v2/test.jsonl
```

### Reference-path long-horizon diagnostic (v1, not typed-v2)

`scripts/analyze_system_one_reference_path.py` joins the frozen Phase-1a route
predictions with the frozen v1 local electron-successor cases by decision ID and
reaction ID. On held-out strict trace-view test, **1,173/1,253 (93.62%)**
reactions route every action type correctly, **686/1,253 (54.75%)** have every
electron event produce the reference next state, and **666/1,253 (53.15%)**
satisfy both conditions at every recorded decision. Validation gives
**689/1,319 (52.24%)** for the joint condition. The first recorded mismatch on
test is an electron successor in 555 reactions, versus 32 action-family routing
mismatches. Among the 49 test reactions with five electron events, only 2 pass
all local checks, versus 437/772 with two events.

These are **teacher-forced reference-state** checks. Recorded IMPORT fragments
are assumed, not generated; a different valid trajectory is counted as a
mismatch. Therefore 53.15% is neither autonomous endpoint accuracy nor a
mathematical upper bound on it. It diagnoses long-horizon error accumulation
and makes improvement in coherent electron-event selection a more immediate
gate than further fitting the three-way route classifier. Hash-bound reports
are under `outputs/agent/system_one_pr81_reference_path_20261005/`.
