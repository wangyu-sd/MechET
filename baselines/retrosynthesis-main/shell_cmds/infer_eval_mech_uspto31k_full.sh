#!/usr/bin/env bash
set -Eeuo pipefail

repo_dir=/home/estar/pxy/mechet/baselines/retrosynthesis-main
mechet_dir=/home/estar/pxy/mechet/MechET
python_bin=/home/estar/anaconda3/envs/opennmt3/bin/python
chem_python_bin=/home/estar/anaconda3/envs/mechet/bin/python
onmt_bin=/home/estar/anaconda3/envs/opennmt3/bin
run_dir=/data/pxy/models/R-SMILES/mech_uspto_31k_full/full/official_full_seed3435
checkpoint_dir="$run_dir/checkpoints"
inference_dir="$run_dir/inference"
evaluation_dir="$run_dir/evaluation"
average_model="$checkpoint_dir/average_model_560000-600000.pt"
raw_predictions="$inference_dir/test_predictions_beam10.txt"
ranked_predictions="$inference_dir/test_predictions_ranked_top10.txt"
runtime_file="$inference_dir/runtime_seconds.txt"
shared_predictions="$mechet_dir/outputs/external_baselines/r_smiles/mech_uspto_31k_full/predictions.jsonl"
reference="$mechet_dir/data/external_baselines/mech_uspto_31k_full/test.jsonl"

cd "$repo_dir"
mkdir -p "$checkpoint_dir" "$inference_dir" "$evaluation_dir" "$(dirname "$shared_predictions")"

"$onmt_bin/onmt_average_models" \
    -output "$average_model" \
    -m exp/mech_uspto_31k_full_PtoR_aug20/model.product-reactants_step_560000.pt \
       exp/mech_uspto_31k_full_PtoR_aug20/model.product-reactants_step_570000.pt \
       exp/mech_uspto_31k_full_PtoR_aug20/model.product-reactants_step_580000.pt \
       exp/mech_uspto_31k_full_PtoR_aug20/model.product-reactants_step_590000.pt \
       "$checkpoint_dir/model.product-reactants_step_600000.pt"

CUDA_VISIBLE_DEVICES=4 /usr/bin/time -f '%e' -o "$runtime_file" \
    "$onmt_bin/onmt_translate" \
    -config train-from-scratch/PtoR/mech-uspto31k-aug20-translate.yml \
    2>&1 | tee "$inference_dir/inference.log"

"$chem_python_bin" score.py \
    -beam_size 10 \
    -n_best 10 \
    -augmentation 20 \
    -targets dataset/mech_uspto_31k_full_PtoR_aug20/test/tgt-test.txt \
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
    --expected-rows 3120 \
    --expected-candidates 10 \
    --candidate-semantics native_ranked \
    2>&1 | tee "$evaluation_dir/mechet_evaluation.log"

echo "R-SMILES USPTO31k inference and evaluation complete: $evaluation_dir"
