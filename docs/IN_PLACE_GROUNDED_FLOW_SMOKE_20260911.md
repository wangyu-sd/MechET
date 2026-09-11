# In-place grounded FLOW: audit and inference smoke results

Date: 2026-09-11

This note records the current model-free audit, optimizer smoke, two-case
product-only rollout, and the representation-tolerant transactional filter
added during PR #53. The rollout contains only two fixed validation reactions
and an incomplete training checkpoint. It is a diagnostic smoke test, **not**
a benchmark result.

The corresponding machine-readable snapshot is
[`results/in_place_grounded_flow_smoke_20260911.json`](results/in_place_grounded_flow_smoke_20260911.json).

## 1. Tested contract

The model receives only the unmapped product and the executor-owned current
state. It predicts first-use unmapped imports, inserts event-local role markers
in a SMILES rendering of that state, and emits compact FLOW clauses. The
executor, rather than the model, commits the next state and derives the final
precursor.

The rollout does not expose the gold precursor, atom maps, reaction centre,
gold intermediate states, or gold `remaining_events`.

## 2. Full artifact audit

The representation build used the complete strict executable universe already
defined by the source manifest. It did not apply new reaction, length, or
overlap filtering.

| Split | Reactions | Events | Electron moves | BE-delta events | Round-trip failures | Successor failures | Endpoint failures | Map leaks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 257,167 | 1,043,352 | 2,111,100 | 1,165 | 0 | 0 | 0 | 0 |
| valid | 2,890 | 11,532 | 23,039 | 17 | 0 | 0 | 0 | 0 |
| test | 28,967 | 117,365 | 236,310 | 113 | 0 | 0 | 0 | 0 |

Additional audit results:

- train/valid/test stable-ID overlap is zero;
- compact FLOW text is 6.18% of the legacy move-JSON character count;
- all 257,167 training reactions remain one lossless training example each;
- Qwen3-8B token audit found a maximum of 6,339 input tokens under the 8,192
  limit and zero truncation;
- every audited row contains supervised assistant tokens and `finish_trace`.

Frozen local artifacts:

- `data/flower_in_place_grounded_flow_v1/manifest.json`, SHA-256
  `4612373158f391507d2ee9007b118921d6304ae1127be98ad2b1e5949efcf75c`;
- `data/flower_in_place_grounded_flow_v1/qwen3_8b_tokens_8192/manifest.json`,
  SHA-256
  `148b67f5ceb9aced1ddd208574f686752ef5e8a914e13c539ef4a5ed5717099e`.

The earlier local one-step Qwen3-8B QLoRA optimizer smoke also completed with
finite loss (`1.6141`).

## 3. Product-only inference smoke

Both runs use greedy K=1 decoding, a four-bit-loaded Qwen3-8B checkpoint, a
12-event budget, and the same two fixed validation reactions. The first run
used checkpoint 5,500 and exact-string insertion matching. The second used
checkpoint 6,000 and the flexible executor described below.

| Protocol | Parsed first tool | Accepted events | Rejected events | `finish_trace` | Exact endpoint |
|---|---:|---:|---:|---:|---:|
| exact-string, checkpoint 5,500 | 2/2 | 4 | 17 | 1/2 | 0/2 |
| graph-aligned + transactional, checkpoint 6,000 | 2/2 | 17 | 5 | 1/2 | 0/2 |

Per-case results under graph-aligned transactional execution:

| Stable ID | Accepted | Rejected | Finish | Exact endpoint | Stop reason |
|---|---:|---:|---:|---:|---|
| `textbook-tool-sft:flower_mech_proof_val_1256` | 9 | 3 | no | no | 12-turn limit |
| `textbook-tool-sft:flower_mech_proof_val_2533` | 8 | 2 | yes | no | `finish_trace` |

The reduction from 17 to 5 rejected events is evidence that representation
normalization and rollback remove a large class of avoidable cascading
failures. It is not a controlled effect size because the later run also uses a
newer checkpoint.

Raw local artifacts:

- exact-string predictions:
  `outputs/eval/in_place_grounded_valid8_ckpt5500_k1_t4_20260911/predictions.jsonl`,
  SHA-256
  `ce7a200f490e10ec34d09591e5f0acb5a060ae533fea9259551bd150d4c12a86`;
- flexible predictions:
  `outputs/eval/in_place_grounded_flexible_cases2_ckpt6000_k1_retry2_t4_20260911/predictions.jsonl`,
  SHA-256
  `7c292e856cabc9e734b1d839f5935390e5e81417c11edd232f00990d0acf36e0`;
- flexible summary:
  `outputs/eval/in_place_grounded_flexible_cases2_ckpt6000_k1_retry2_t4_20260911/summary.json`,
  SHA-256
  `bbe3904cf2619c0aa4beb8dd1b601cb59122c4bfe3ac498defed2813627a8c16`.

## 4. Flexible filtering contract

“Flexible” means representation-tolerant, not endpoint-tolerant. It changes
how an event is grounded and rejected; it does not accept an incorrect final
precursor as correct.

The inference adapter now:

1. aligns the marked molecule to the authoritative state by full molecular
   graph isomorphism, so valid SMILES traversal order and disconnected-component
   order do not cause false rejection;
2. allows any symmetry-equivalent atom binding and preserves explicit
   hydrogens;
3. assigns executor-private maps only after parsing unmapped imported fragments;
4. executes the whole event transactionally and rolls back both imports and
   the private-map counter on rejection;
5. always returns the authoritative current state after success or failure;
6. rejects malformed graphs/FLOW, failed electron replay, repeated states,
   more than four imports per event, more than 64 imported atoms per event, or
   any imported fragment with more than 24 heavy atoms.

The graph match still requires molecular identity, including bond, charge,
aromatic, and stereochemical constraints. Endpoint accuracy remains strict.

## 5. What the two cases show

### `flower_mech_proof_val_2533`

The gold first import is hydroxide (`[H]O`). The model instead imports an
HOBt-like fragment (`[H]On1nnc2ccccc21`). Its first proton transfer and the
following O-acyl addition are formally executable, but they enter a different
retrosynthetic branch. The flexible executor prevents later serialization and
loop failures from corrupting state, yet the model eventually finishes at the
wrong precursor mixture.

### `flower_mech_proof_val_1256`

The gold first import is `COB(Br)Br`. The model starts with ethoxide and follows
with lithium, acid/base, acylation, solvent, and salt fragments. Several local
electron moves execute, but they do not implement the required O-demethylation
route. The trace reaches the event limit without a correct endpoint.

These examples separate three notions that must be reported independently:

- **syntactic validity**: the output parses as one tool call;
- **formal executability**: the executor can apply an event to its current
  molecular state;
- **reaction-level correctness**: the executed trajectory reaches an accepted
  precursor endpoint.

A formally executable local move is not, by itself, evidence that the selected
fragment or complete route is chemically appropriate for the target.

## 6. Current conclusions and next measurement

The checkpoint has learned the basic interaction grammar: both cases produced
a parsed first tool call and at least one executable electron-flow event. The
flexible adapter fixes representation-only rejection and prevents rejected
imports from contaminating later turns. The remaining smoke failures are
dominated by wrong-but-locally-executable fragment/path selection and by trace
termination, not by atom-map visibility or compiler coverage.

The next result should therefore be a frozen product-only validation evaluation
with multiple candidates and model-likelihood ranking, reporting parse rate,
event executability, finish rate, strict endpoint accuracy, and failure counts
over the same denominator. The present two-case smoke must not be promoted to a
paper accuracy number.
