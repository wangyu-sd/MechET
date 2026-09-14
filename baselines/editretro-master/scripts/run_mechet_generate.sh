#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
    echo "Usage: $0 {mech_uspto_31k_full|flower_full} CHECKPOINT [full|audit100] [train|valid|test]" >&2
    exit 2
fi

dataset=$1
checkpoint=$(realpath "$2")
profile=${3:-full}
logical_split=${4:-test}
case "$dataset" in
    mech_uspto_31k_full|flower_full) ;;
    *) echo "Unsupported dataset: $dataset" >&2; exit 2 ;;
esac
case "$profile" in
    full|audit100) ;;
    *) echo "Unsupported profile: $profile" >&2; exit 2 ;;
esac
case "$logical_split" in
    train|valid|test) ;;
    *) echo "Unsupported split: $logical_split" >&2; exit 2 ;;
esac
if [[ ! -s "$checkpoint" ]]; then
    echo "Missing checkpoint: $checkpoint" >&2
    exit 1
fi

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mechet_dir=${MECHET_DIR:-/home/estar/pxy/mechet/MechET}
python_bin=${PYTHON_BIN:-python}
chem_python_bin=${CHEM_PYTHON_BIN:-$python_bin}
cpu=${CPU:-0}
if command -v fairseq-generate >/dev/null 2>&1; then
    fairseq_generate=(fairseq-generate)
else
    export PYTHONPATH="$repo_dir/fairseq:$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
    fairseq_generate=("$python_bin" "$repo_dir/fairseq/fairseq_cli/generate.py")
fi
augmentation=${AUGMENTATION:-10}
native_dataset=$dataset
if [[ "$profile" == "audit100" ]]; then
    native_dataset="${dataset}_audit100"
fi
artifact_dir="$repo_dir/datasets/$native_dataset/aug$augmentation"
"$python_bin" "$repo_dir/utils/require_validated_artifact.py" "$artifact_dir"

native_split=$logical_split
if [[ "$logical_split" == "valid" ]]; then
    native_split=valid
    line_map_split=val
else
    line_map_split=$logical_split
fi

gpus=${GPUS:-0}
beam_size=${BEAM_SIZE:-20}
repos_beam=${REPOS_BEAM:-5}
mask_beam=${MASK_BEAM:-1}
token_beam=${TOKEN_BEAM:-4}
top_n=${TOP_N:-10}
max_tokens=${MAX_TOKENS:-1000}
max_iter=${MAX_ITER:-10}
score_alpha=${SCORE_ALPHA:-0.1}
num_workers=${NUM_WORKERS:-1}
runtime_args=()
precision_args=(--fp16)
launcher=(env CUDA_VISIBLE_DEVICES="$gpus" CUDA_LAUNCH_BLOCKING=1)
if [[ "$cpu" == "1" ]]; then
    runtime_args+=(--cpu)
    precision_args=()
    launcher=(env)
fi
if (( repos_beam * mask_beam * token_beam != beam_size )); then
    echo "REPOS_BEAM*MASK_BEAM*TOKEN_BEAM must equal BEAM_SIZE" >&2
    exit 2
fi

checkpoint_name=$(basename "$checkpoint" .pt)
output_dir=${OUTPUT_DIR:-$repo_dir/results/mechet/$dataset/$profile/generation/$checkpoint_name/$logical_split}
mkdir -p "$output_dir"
generation_log="$output_dir/generation.txt"
runtime_file="$output_dir/runtime_seconds.txt"

"${launcher[@]}" /usr/bin/time \
    -f '%e' -o "$runtime_file" \
    "${fairseq_generate[@]}" \
    --user-dir "$repo_dir/editretro" \
    "$artifact_dir/data-bin" \
    -s src -t tgt \
    --gen-subset "$native_split" \
    --task translation_retro \
    --path "$checkpoint" \
    --iter-decode-max-iter "$max_iter" \
    --iter-decode-eos-penalty 0 \
    --beam 1 --remove-bpe \
    --init-src \
    --TOPK "$beam_size" \
    --max-tokens "$max_tokens" \
    --num-workers "$num_workers" \
    --repos-beam "$repos_beam" \
    --mask-beam "$mask_beam" \
    --token-beam "$token_beam" \
    "${precision_args[@]}" \
    "${runtime_args[@]}" \
    --print-step --retain-iter-history \
    > "$generation_log"

reference="$mechet_dir/data/external_baselines/$dataset/$logical_split.jsonl"
predictions="$output_dir/predictions.jsonl"
if [[ "$profile" == "full" && "$logical_split" == "test" ]]; then
    predictions="$mechet_dir/outputs/external_baselines/editretro/$dataset/predictions.jsonl"
fi

expected_rows=0
if [[ "$profile" == "audit100" ]]; then
    expected_rows=100
elif [[ "$dataset" == "mech_uspto_31k_full" ]]; then
    case "$logical_split" in
        train) expected_rows=24959 ;;
        valid|test) expected_rows=3120 ;;
    esac
elif [[ "$dataset" == "flower_full" ]]; then
    case "$logical_split" in
        train) expected_rows=257171 ;;
        valid) expected_rows=2890 ;;
        test) expected_rows=28971 ;;
    esac
fi

"$chem_python_bin" "$repo_dir/utils/export_mechet_predictions.py" \
    --reference "$reference" \
    --line-map "$artifact_dir/$line_map_split.line_map.jsonl" \
    --fairseq-log "$generation_log" \
    --output "$predictions" \
    --checkpoint "$checkpoint" \
    --profile "$profile" \
    --augmentation "$augmentation" \
    --beam-size "$beam_size" \
    --top-n "$top_n" \
    --score-alpha "$score_alpha" \
    --runtime-seconds-file "$runtime_file" \
    --expected-rows "$expected_rows"

evaluation=${predictions%.jsonl}.evaluation.json
evaluation_profile_args=()
if [[ "$profile" == "audit100" ]]; then
    evaluation_profile_args+=(--reference-limit "$expected_rows")
fi
"$chem_python_bin" "$mechet_dir/scripts/evaluate_endpoint_candidates.py" \
    --reference "$reference" \
    --predictions "$predictions" \
    --output "$evaluation" \
    --expected-rows "$expected_rows" \
    --expected-candidates "$top_n" \
    --candidate-semantics native_ranked \
    "${evaluation_profile_args[@]}"

echo "$predictions"
