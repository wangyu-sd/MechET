#!/usr/bin/env bash
set -Eeuo pipefail

BASE=/home/estar/pxy/mechet/MechET/baselines/RetroBridge-main
REPO=/home/estar/pxy/mechet/MechET
PY="$BASE/.venv-retrobridge/bin/python"
DATA=/data/pxy/data/RetroBridge/mech_uspto_31k_full
CKPT=/data/pxy/models/RetroBridge/mech_uspto_31k_full/checkpoints/mechet_retrobridge_mech_uspto_31k_full_31_08_12_40_20/last.ckpt
REFERENCE="$REPO/data/external_baselines/mech_uspto_31k_full/test.jsonl"
OUT="$REPO/outputs/external_baselines/retrobridge/mech_uspto_31k_full_last_epoch51"
PARTS="$OUT/parts"

GPUS=(0 1 2 3 5 6 7)
STARTS=(0 448 896 1344 1792 2240 2688)
ENDS=(448 896 1344 1792 2240 2688 3120)

mkdir -p "$PARTS"
cd "$BASE"

pids=()
cleanup() {
  for pid in "${pids[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup INT TERM

for index in "${!GPUS[@]}"; do
  gpu=${GPUS[$index]}
  start=${STARTS[$index]}
  end=${ENDS[$index]}
  name=$(printf 'part-%02d' "$index")
  prediction_part="$PARTS/$name.predictions.jsonl"
  trace_part="$PARTS/$name.inference_traces.jsonl"
  log_part="$PARTS/$name.log"
  expected_rows=$((end - start))

  if [[ -f "$prediction_part" && -f "$trace_part" ]] \
      && [[ $(wc -l < "$prediction_part") -eq $expected_rows ]] \
      && [[ $(wc -l < "$trace_part") -eq $expected_rows ]]; then
    printf 'Reusing completed %s: rows [%d, %d)\n' "$name" "$start" "$end"
    continue
  fi

  env \
    CUDA_VISIBLE_DEVICES="$gpu" \
    MPLCONFIGDIR="/tmp/retrobridge-mpl-gpu$gpu" \
    TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
    TQDM_DISABLE=1 \
    "$PY" -u sample_mechet_retrobridge.py \
      --data-root "$DATA" \
      --checkpoint "$CKPT" \
      --output "$prediction_part" \
      --trace-output "$trace_part" \
      --mode test \
      --batch-size 16 \
      --num-workers 0 \
      --n-samples 10 \
      --n-steps 500 \
      --sampling-seed 42 \
      --start-index "$start" \
      --end-index "$end" \
      --device cuda:0 \
      >"$log_part" 2>&1 &
  pids+=("$!")
  printf 'Started %s on GPU %d: PID %d, rows [%d, %d)\n' \
    "$name" "$gpu" "$!" "$start" "$end"
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    printf 'Sampling process %d failed. Inspect %s/part-*.log\n' "$pid" "$PARTS" >&2
    failed=1
  fi
done
if [[ $failed -ne 0 ]]; then
  exit 1
fi
trap - INT TERM

"$PY" - "$REFERENCE" "$PARTS" "$OUT" "$CKPT" <<'PY'
import json
import sys
from pathlib import Path

reference_path, parts_path, output_path, checkpoint = map(Path, sys.argv[1:])
references = [json.loads(line) for line in reference_path.open() if line.strip()]
expected_ids = [str(row.get("id") or row.get("stable_id")) for row in references]

prediction_rows = []
trace_rows = []
for path in sorted(parts_path.glob("part-*.predictions.jsonl")):
    prediction_rows.extend(json.loads(line) for line in path.open() if line.strip())
for path in sorted(parts_path.glob("part-*.inference_traces.jsonl")):
    trace_rows.extend(json.loads(line) for line in path.open() if line.strip())

actual_ids = [row["stable_id"] for row in prediction_rows]
trace_ids = [row["stable_id"] for row in trace_rows]
if actual_ids != expected_ids:
    raise RuntimeError("Merged prediction IDs do not exactly match the reference order")
if trace_ids != expected_ids:
    raise RuntimeError("Merged trace IDs do not exactly match the reference order")
if [row["source_index"] for row in prediction_rows] != list(range(len(references))):
    raise RuntimeError("Merged source_index values are not contiguous")

valid_candidates = 0
for row in prediction_rows:
    candidates = row.get("candidates") or []
    if len(candidates) != 10:
        raise RuntimeError(f"{row['stable_id']} has {len(candidates)} candidates")
    if Path(row["checkpoint"]).resolve() != checkpoint.resolve():
        raise RuntimeError(f"{row['stable_id']} records an unexpected checkpoint")
    for sample_index, candidate in enumerate(candidates):
        if candidate.get("sample_index") != sample_index:
            raise RuntimeError(f"{row['stable_id']} has a bad sample_index")
        if candidate.get("prediction") != candidate.get("precursors"):
            raise RuntimeError(f"{row['stable_id']} has inconsistent prediction fields")
    valid_candidates += int(row["valid_candidate_count"])

output_path.mkdir(parents=True, exist_ok=True)
with (output_path / "predictions.jsonl").open("w") as handle:
    for row in prediction_rows:
        handle.write(json.dumps(row) + "\n")
with (output_path / "inference_traces.jsonl").open("w") as handle:
    for row in trace_rows:
        handle.write(json.dumps(row) + "\n")

report = {
    "rows": len(prediction_rows),
    "candidates_per_row": 10,
    "total_candidates": len(prediction_rows) * 10,
    "valid_candidates_from_sampler": valid_candidates,
    "checkpoint": str(checkpoint.resolve()),
    "checkpoint_epoch": 51,
    "checkpoint_global_step": 20280,
    "sampling_steps": 500,
    "sampling_seed": 42,
    "batch_size_per_gpu": 16,
    "gpus": [0, 1, 2, 3, 5, 6, 7],
    "id_order_matches_reference": True,
}
(output_path / "merge_report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
PY

"$PY" "$REPO/scripts/evaluate_endpoint_candidates.py" \
  --reference "$REFERENCE" \
  --predictions "$OUT/predictions.jsonl" \
  --output "$OUT/evaluation.json" \
  --expected-rows 3120 \
  --expected-candidates 10 \
  >"$OUT/evaluation.log" 2>&1

printf 'Completed predictions: %s/predictions.jsonl\n' "$OUT"
printf 'Completed evaluation:  %s/evaluation.json\n' "$OUT"
