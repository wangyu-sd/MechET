# PR93 frozen Stage-II next-action diagnosis

Status: mech-USPTO-31k completed; FlowER inference stopped at the user's request,
with316/491 decision outputs preserved. No completed FlowER aggregate is
reported, no automatic continuation or new evaluation is authorized. This is the
user-authorized fixed64/32 diagnosis, not a full-test or product-start result.

## Frozen inputs and actual execution

- Original Qwen3-0.6B Stage-II R0 adapter SHA256:
  `40204c7a059d16976fc4b24b315c3bdde7bf61c705d24bbde515071757e9de75`.
  This is NOT the newly completed cumulative-three-epoch continuation.
- Base/tokenizer revision:`c1899de289a04d12100db370d81485cdf75e47ca`.
- Local idle Tesla T4; Torch2.6.0+cu118; RDKit2026.03.4 overlay;
  FP16, greedy, batch4,512 new-token limit,16384 context limit. FP16 is a
  supported CLI option and is explicitly distinguished from the BF16 launcher
  default; no precision-matched BF16 improvement claim is made.
- No new training, model downloads, Taiji tasks, paper changes or PR merge.
- FlowER source:2890 validation reactions; stratified64 selected,491 decisions.
  All491 public prompts/tools match the saved Stage-II validation SFT rows
  exactly. All selected reference FLOWs replay and match their authentic source
  successor under the stricter audit.
  This491/491 input-parity audit is distinct from the cancelled model generation;
  it does not mean491 predictions finished. The partial316 outputs are not a
  completed64-reaction accuracy result.
- mech31k source:1319 current-compiler trace-view validation reactions;
  stratified32 selected,157 decisions; all selected reference FLOWs pass replay.
  This actor is FlowER-trained, so this is a cross-corpus local diagnostic,
  NOT the full3120 endpoint benchmark or a31k-trained checkpoint evaluation.

## Completed mech31k results

| Metric | Count | Rate |
| --- | ---: | ---: |
| All planned local decisions scored |157/157|100%|
| IMPORT tool choice |21/23|91.30%|
| IMPORT complete fragment multiset + quantities |7/23|30.43%|
| IMPORT fragment multiset + purpose |7/23|30.43%|
| FLOW tool choice |49/102|48.04%|
| FLOW exact electron event |7/102|6.86%|
| FLOW formal execution |42/102|41.18%|
| FLOW map-free chemical successor exact |23/102|22.55%|
| FINISH exact |25/32|78.13%|
| Reactions with all reference decisions exact |0/32|0%|
| Reactions with all local successors/imports exact |0/32|0%|

Of102 gold FLOW decisions,53 are predicted as IMPORT. All157 responses parse;
these are not missing-output failures. Purpose-specific matching is16/23 for
electron participants and1/13 for endpoint context. These purpose-conditioned
subsets can overlap and do not replace whole-import exact matching.

Every decision receives the reference state/history, so previous model errors
cannot explain these local mistakes. This supports investigating next-action
selection as well as free-running error accumulation. It does not establish
that every non-reference action is chemically unreasonable, or explain
in-corpus FlowER behavior before that run finishes.

## Execution fixes and verification

1. Original PR's source-successor mismatch test failed (six other tests passed):
   the evaluation compared recomputed converter output against itself. The
   audit-only converter path now verifies the authentic translated source
   successor. Default SFT conversion behavior is unchanged. Existing failing
   test passes after the fix; both real selected corpora pass stricter replay.
2. Frozen FlowER legacy rows have no per-row corpus/split fields. The evaluator
   now authenticates the complete file against its frozen training manifest's
   valid split, checking artifact type, corpus, row count and SHA. Two added
   regressions were observed failing before the fix. No source row was altered.
3. Independent read-only review found that old resume/aggregation trusted keys
   without validating run provenance. New runs write immutable model/task/code/
   decoding contracts before generation; mismatches refuse resume. Aggregation
   checks completed shard provenance and freezes original adapter by default.
   Two regressions were observed failing before this fix.

Full suite after fixes:482 passed,1 skipped. Shell syntax and diff whitespace
checks passed. No scoring change was made to `score_prediction`.

The actual fresh GPU run began before the resume-contract hardening. Its
original completion provenance is preserved and verified, and the new report
marks `immutable_resume_contract:false`; do not retrospectively claim that a
contract existed before this run. No prior prediction files were reused.
Generation used the exact PR head243e955 plus the
[preserved execution patch](pr93_generation_code_20261022.patch), not the later
resume-only hardening. No running shell was edited.

## Raw records

Absolute shared output root:
`/aaa/fionafyang/buddy1/whaleywang/MechET/outputs/eval/stage2_teacher_forced_flower_mech31k_pr93_20261022_01`

For each of `flower` and `mech31k`:

- `prepared/tasks.jsonl`: every complete model-visible system/user prompt and
  tool schema, gold action, private authentic state and reference successor.
- `prepared/selection.json`: exact selection/source/task hashes and denominators.
- `predictions/decisions.shard-00-of-01.jsonl`: every raw generated response,
  parsed arguments, generated token count and per-decision score/error.
- `predictions/provenance.rank-00.json`: actual frozen model/task/precision.
- `predictions/evaluation.json`: full aggregate, only when all outputs exist.

Gold/private fields are for offline scoring and never enter actor inputs.
