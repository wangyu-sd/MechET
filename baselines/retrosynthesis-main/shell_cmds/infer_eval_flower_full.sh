#!/usr/bin/env bash
set -Eeuo pipefail

repo_dir=/home/estar/pxy/mechet/baselines/retrosynthesis-main
mechet_dir=/home/estar/pxy/mechet/MechET
python_bin=/home/estar/anaconda3/envs/opennmt3/bin/python
chem_python_bin=/home/estar/anaconda3/envs/mechet/bin/python
onmt_bin=/home/estar/anaconda3/envs/opennmt3/bin
train_config=train-from-scratch/PtoR/flower-full-aug5-config.yml
translate_config=train-from-scratch/PtoR/flower-full-aug5-translate.yml
checkpoint_prefix=exp/flower_full_PtoR_aug5/model.product-reactants_step_
run_dir=/data/pxy/models/R-SMILES/flower_full/full/official_full_seed3435
checkpoint_dir="$run_dir/checkpoints"
inference_dir="$run_dir/inference"
evaluation_dir="$run_dir/evaluation"
average_model="$checkpoint_dir/average_model_final5.pt"
completion_marker="$run_dir/infer_eval.complete"
raw_predictions="$inference_dir/test_predictions_beam10.txt"
ranked_predictions="$inference_dir/test_predictions_ranked_top10.txt"
runtime_file="$inference_dir/runtime_seconds.txt"
shared_predictions="$mechet_dir/outputs/external_baselines/r_smiles/flower_full/predictions.jsonl"
reference="$mechet_dir/data/external_baselines/flower_full/test.jsonl"

cd "$repo_dir"

configured_step="$($python_bin - "$train_config" <<'PY'
import sys
from pathlib import Path

for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    key, separator, value = line.partition(":")
    if separator and key.strip() == "train_steps":
        print(int(value.strip()))
        break
else:
    raise SystemExit(f"train_steps not found in {sys.argv[1]}")
PY
)"
final_step="${1:-$configured_step}"
if [[ ! "$final_step" =~ ^[0-9]+$ ]] || (( final_step <= 0 )); then
    echo "Invalid final step: $final_step" >&2
    exit 2
fi
if (( final_step != configured_step )); then
    echo "Refusing checkpoint/config mismatch: requested=$final_step configured=$configured_step" >&2
    exit 2
fi

checkpoint_steps=()
for offset in 40000 30000 20000 10000 0; do
    step=$((final_step - offset))
    checkpoint="${checkpoint_prefix}${step}.pt"
    if (( step <= 0 )) || [[ ! -s "$checkpoint" ]]; then
        echo "Required final-five checkpoint is missing: $checkpoint" >&2
        exit 1
    fi
    checkpoint_steps+=("$checkpoint")
done

mkdir -p "$checkpoint_dir" "$inference_dir" "$evaluation_dir" "$(dirname "$shared_predictions")"
exec 9> "$run_dir/.infer_eval.lock"
if ! flock -n 9; then
    echo "Another FlowerFull infer/eval pipeline already holds $run_dir/.infer_eval.lock; exiting."
    exit 0
fi
if [[ -s "$completion_marker" ]] && [[ "$(<"$completion_marker")" == "$final_step" ]]; then
    echo "FlowerFull infer/eval is already complete for step $final_step; exiting."
    exit 0
fi

{
    printf '[pipeline] started_at=%s\n' "$(date --iso-8601=seconds)"
    printf '[pipeline] configured_final_step=%s gpu=6\n' "$final_step"
    printf '[pipeline] averaging_checkpoints=%s\n' "${checkpoint_steps[*]}"
} | tee "$run_dir/infer_eval_status.log"

"$onmt_bin/onmt_average_models" \
    -output "$average_model" \
    -m "${checkpoint_steps[@]}" \
    2>&1 | tee "$checkpoint_dir/average_models.log"

CUDA_VISIBLE_DEVICES=6 /usr/bin/time -f '%e' -o "$runtime_file" \
    "$onmt_bin/onmt_translate" \
    -config "$translate_config" \
    2>&1 | tee "$inference_dir/inference.log"

expected_raw_predictions=$((28971 * 5 * 10))
actual_raw_predictions=$(wc -l < "$raw_predictions")
if (( actual_raw_predictions != expected_raw_predictions )); then
    echo "Raw prediction count mismatch: expected=$expected_raw_predictions actual=$actual_raw_predictions" >&2
    exit 1
fi

"$chem_python_bin" score.py \
    -beam_size 10 \
    -n_best 10 \
    -augmentation 5 \
    -targets dataset/flower_full_PtoR_aug5/test/tgt-test.txt \
    -predictions "$raw_predictions" \
    -process_number 32 \
    -score_alpha 1 \
    -save_file "$ranked_predictions" \
    2>&1 | tee "$evaluation_dir/native_score.log"

"$python_bin" scripts/export_mechet_ranked_predictions.py \
    --ranked-predictions "$ranked_predictions" \
    --reference "$reference" \
    --output "$shared_predictions" \
    --checkpoint "$average_model" \
    --runtime-seconds "$runtime_file" \
    --candidates-per-target 10

"$chem_python_bin" "$mechet_dir/scripts/evaluate_endpoint_candidates.py" \
    --reference "$reference" \
    --predictions "$shared_predictions" \
    --output "$evaluation_dir/predictions.evaluation.json" \
    --expected-rows 28971 \
    --expected-candidates 10 \
    --candidate-semantics native_ranked \
    2>&1 | tee "$evaluation_dir/mechet_evaluation.log"

{
    printf '[pipeline] completed_at=%s\n' "$(date --iso-8601=seconds)"
    printf '[pipeline] evaluation=%s\n' "$evaluation_dir/predictions.evaluation.json"
} | tee -a "$run_dir/infer_eval_status.log"
printf '%s\n' "$final_step" > "$completion_marker"

echo "R-SMILES FlowerFull inference and evaluation complete: $evaluation_dir"
