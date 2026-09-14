#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/estar/pxy/mechet/baselines/RetroBridge-main
MECHET=/home/estar/pxy/mechet/MechET
DATA=/data/pxy/data/RetroBridge/mech_uspto_31k_full
REFERENCE="$MECHET/data/external_baselines/mech_uspto_31k_full/test.jsonl"
CHECKPOINT=/data/pxy/models/RetroBridge/mech_uspto_31k_full/checkpoints/mechet_retrobridge_mech_uspto_31k_full_31_08_12_40_20/best-epoch=021.ckpt
OUT="$MECHET/outputs/external_baselines/retrobridge/mech_uspto_31k_full"
PARTS="$OUT/parts"

GPUS=(0 1 2 3 5 6 7)
STARTS=(0 448 896 1344 1792 2240 2688)
LIMITS=(448 448 448 448 448 448 432)

mkdir -p "$PARTS"
cd "$ROOT"

pids=()
terminate_children() {
    if ((${#pids[@]})); then
        kill "${pids[@]}" 2>/dev/null || true
    fi
}
trap terminate_children INT TERM

for index in "${!GPUS[@]}"; do
    gpu=${GPUS[$index]}
    start=${STARTS[$index]}
    limit=${LIMITS[$index]}
    part=$(printf '%02d' "$index")
    log="$PARTS/part-$part.log"

    (
        export CUDA_VISIBLE_DEVICES="$gpu"
        export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
        export TQDM_DISABLE=1
        export MPLCONFIGDIR=/tmp/retrobridge-mpl
        export OMP_NUM_THREADS=4
        export MKL_NUM_THREADS=4
        export OPENBLAS_NUM_THREADS=4
        export NUMEXPR_NUM_THREADS=4
        exec .venv-retrobridge/bin/python -u sample_mechet_retrobridge.py \
            --data-root "$DATA" \
            --checkpoint "$CHECKPOINT" \
            --output "$PARTS/part-$part.predictions.jsonl" \
            --trace-output "$PARTS/part-$part.inference_traces.jsonl" \
            --mode test \
            --batch-size 4 \
            --num-workers 0 \
            --n-samples 10 \
            --n-steps 500 \
            --sampling-seed 42 \
            --torch-threads 4 \
            --start "$start" \
            --limit "$limit" \
            --device cuda:0 \
            --resume
    ) >"$log" 2>&1 &
    pid=$!
    pids+=("$pid")
    printf 'Started part %s on GPU %s: PID %s, rows [%s, %s)\n' \
        "$part" "$gpu" "$pid" "$start" "$((start + limit))"
done

failed=0
for ((remaining = ${#pids[@]}; remaining > 0; remaining--)); do
    if ! wait -n; then
        printf 'A part failed; inspect %s/part-*.log\n' "$PARTS" >&2
        failed=1
        terminate_children
        wait || true
        break
    fi
done
trap - INT TERM
if ((failed)); then
    exit 1
fi

predictions_tmp="$OUT/.predictions.jsonl.tmp"
traces_tmp="$OUT/.inference_traces.jsonl.tmp"
: >"$predictions_tmp"
: >"$traces_tmp"
for index in "${!GPUS[@]}"; do
    part=$(printf '%02d' "$index")
    cat "$PARTS/part-$part.predictions.jsonl" >>"$predictions_tmp"
    cat "$PARTS/part-$part.inference_traces.jsonl" >>"$traces_tmp"
done
mv "$predictions_tmp" "$OUT/predictions.jsonl"
mv "$traces_tmp" "$OUT/inference_traces.jsonl"

PYTHONPATH="$MECHET/src" .venv-retrobridge/bin/python \
    "$MECHET/scripts/evaluate_endpoint_candidates.py" \
    --reference "$REFERENCE" \
    --predictions "$OUT/predictions.jsonl" \
    --output "$OUT/evaluation.json" \
    --expected-rows 3120 \
    --expected-candidates 10 \
    >"$OUT/evaluation.log" 2>&1

printf 'Completed: %s\n' "$OUT/predictions.jsonl"
printf 'Traces:    %s\n' "$OUT/inference_traces.jsonl"
printf 'Metrics:   %s\n' "$OUT/evaluation.json"
