#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/home/estar/pxy/mechet/baselines/RetroBridge-main
TRAINING_ROOT=/home/estar/pxy/mechet/MechET/baselines/RetroBridge-main
MECHET=/home/estar/pxy/mechet/MechET
PY="$TRAINING_ROOT/.venv-retrobridge/bin/python"
DATASET_KEY=${1:?Usage: run_retrobridge_mechet_eval.sh uspto31k|flower}
GPUS_TEXT=${RETROBRIDGE_GPUS:-"4 5 7"}
read -r -a GPUS <<<"$GPUS_TEXT"

case "$DATASET_KEY" in
  uspto31k)
    DATASET_NAME=mech_uspto_31k_full
    DATA_ROOT=/data/pxy/data/RetroBridge/mech_uspto_31k_full
    REFERENCE="$MECHET/data/external_baselines/mech_uspto_31k_full/test.jsonl"
    CHECKPOINT=/data/pxy/models/RetroBridge/mech_uspto_31k_full/checkpoints/mechet_retrobridge_mech_uspto_31k_full_single_gpu_03_09_06_59_06/last.ckpt
    OUT="$MECHET/outputs/external_baselines/retrobridge/mech_uspto_31k_full_single_gpu_epoch43"
    EXPECTED_REFERENCE_ROWS=3120
    EXPECTED_NATIVE_ROWS=3120
    ;;
  flower)
    DATASET_NAME=flower_full
    DATA_ROOT=/data/pxy/data/RetroBridge/flower_full
    REFERENCE="$MECHET/data/external_baselines/flower_full/test.jsonl"
    CHECKPOINT=/data/pxy/models/RetroBridge/flower_full/eval_snapshots/best-epoch=079-global_step=321520-sha256=e9d33a50.ckpt
    OUT="$MECHET/outputs/external_baselines/retrobridge/flower_full_best_epoch79"
    EXPECTED_REFERENCE_ROWS=28971
    EXPECTED_NATIVE_ROWS=28966
    ;;
  *)
    printf 'Unknown dataset key: %s\n' "$DATASET_KEY" >&2
    exit 2
    ;;
esac

mkdir -p "$OUT/parts"
cd "$ROOT"

"$PY" prepare_retrobridge_inference_plan.py \
  --dataset-name "$DATASET_NAME" \
  --data-root "$DATA_ROOT" \
  --reference "$REFERENCE" \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUT" \
  --gpus "${GPUS[@]}" \
  --batch-size 16 \
  --n-samples 10 \
  --n-steps 500 \
  --sampling-seed 42 \
  >"$OUT/plan.log" 2>&1

pids=()
cleanup() {
  if ((${#pids[@]})); then
    kill "${pids[@]}" 2>/dev/null || true
  fi
}
trap cleanup INT TERM

for worker_index in "${!GPUS[@]}"; do
  gpu=${GPUS[$worker_index]}
  worker=$(printf '%02d' "$worker_index")
  indices="$OUT/parts/worker-$worker.indices.json"
  predictions="$OUT/parts/worker-$worker.predictions.jsonl"
  traces="$OUT/parts/worker-$worker.traces.jsonl"
  log="$OUT/parts/worker-$worker.log"
  expected_rows=$("$PY" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$indices")
  completed_rows=0
  if [[ -f "$predictions" ]]; then
    completed_rows=$(wc -l <"$predictions")
  fi
  printf 'Starting worker %s on GPU %s: %s/%s rows already complete\n' \
    "$worker" "$gpu" "$completed_rows" "$expected_rows"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export MPLCONFIGDIR="/tmp/retrobridge-mpl-gpu$gpu"
    export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
    export TQDM_DISABLE=1
    export OMP_NUM_THREADS=4
    export MKL_NUM_THREADS=4
    export OPENBLAS_NUM_THREADS=4
    export NUMEXPR_NUM_THREADS=4
    exec "$PY" -u sample_mechet_retrobridge_indexed.py \
      --data-root "$DATA_ROOT" \
      --checkpoint "$CHECKPOINT" \
      --indices-json "$indices" \
      --output "$predictions" \
      --trace-output "$traces" \
      --mode test \
      --batch-size 16 \
      --num-workers 0 \
      --n-samples 10 \
      --n-steps 500 \
      --sampling-seed 42 \
      --torch-threads 4 \
      --device cuda:0 \
      --resume
  ) >"$log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    printf 'Sampling process %s failed; inspect %s/parts/worker-*.log\n' \
      "$pid" "$OUT" >&2
    failed=1
  fi
done
trap - INT TERM
if ((failed)); then
  exit 1
fi

"$PY" merge_retrobridge_inference.py \
  --plan "$OUT/inference_plan.json" \
  --output-dir "$OUT" \
  >"$OUT/merge.log" 2>&1

PYTHONPATH="$MECHET/src" "$PY" "$MECHET/scripts/evaluate_endpoint_candidates.py" \
  --reference "$REFERENCE" \
  --predictions "$OUT/predictions.jsonl" \
  --output "$OUT/evaluation.json" \
  --expected-rows "$EXPECTED_REFERENCE_ROWS" \
  --expected-candidates 10 \
  >"$OUT/evaluation.log" 2>&1

PYTHONPATH="$MECHET/src" "$PY" "$MECHET/scripts/evaluate_endpoint_candidates.py" \
  --reference "$OUT/reference.native_compatible.jsonl" \
  --predictions "$OUT/predictions.native_compatible.jsonl" \
  --output "$OUT/evaluation.native_compatible.json" \
  --expected-rows "$EXPECTED_NATIVE_ROWS" \
  --expected-candidates 10 \
  >"$OUT/evaluation.native_compatible.log" 2>&1

printf 'Completed predictions: %s/predictions.jsonl\n' "$OUT"
printf 'Completed full-denominator evaluation: %s/evaluation.json\n' "$OUT"
printf 'Completed native-compatible evaluation: %s/evaluation.native_compatible.json\n' "$OUT"
