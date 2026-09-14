#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 {mech_uspto_31k_full|flower_full} [full|audit100]" >&2
    exit 2
fi

dataset=$1
profile=${2:-audit100}
case "$dataset" in
    mech_uspto_31k_full|flower_full) ;;
    *) echo "Unsupported dataset: $dataset" >&2; exit 2 ;;
esac
case "$profile" in
    full|audit100) ;;
    *) echo "Unsupported profile: $profile" >&2; exit 2 ;;
esac

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mechet_dir=${MECHET_DIR:-/home/estar/pxy/mechet/MechET}
python_bin=${PYTHON_BIN:-python}
chem_python_bin=${CHEM_PYTHON_BIN:-$python_bin}
augmentation=${AUGMENTATION:-10}
processes=${PROCESSES:--1}
binarize_workers=${BINARIZE_WORKERS:-40}
overwrite=${OVERWRITE:-0}
reuse_preprocessed=${REUSE_PREPROCESSED:-0}

native_dataset=$dataset
if [[ "$profile" == "audit100" ]]; then
    native_dataset="${dataset}_audit100"
fi
raw_dir="$repo_dir/datasets/$native_dataset/raw"
output_dir="$repo_dir/datasets/$native_dataset/aug$augmentation"
reference_dir="$mechet_dir/data/external_baselines/$dataset"

overwrite_args=()
if [[ "$overwrite" == "1" ]]; then
    overwrite_args+=(--overwrite)
fi

if [[ "$reuse_preprocessed" == "1" ]]; then
    for split in train val test; do
        for suffix in src tgt line_map.jsonl preprocess_report.json failures.jsonl; do
            if [[ ! -e "$output_dir/$split.$suffix" ]]; then
                echo "Cannot reuse incomplete preprocessing: $output_dir/$split.$suffix" >&2
                exit 1
            fi
        done
    done
    if [[ ! -s "$output_dir/preprocess_manifest.json" ]]; then
        echo "Cannot reuse preprocessing without $output_dir/preprocess_manifest.json" >&2
        exit 1
    fi
    echo "Reusing audited EditRetro text preprocessing: $output_dir"
else
    "$chem_python_bin" "$repo_dir/preprocess/prepare_mechet_data.py" \
        --dataset "$native_dataset" \
        --data-dir "$raw_dir" \
        --output-dir "$output_dir" \
        --augmentation "$augmentation" \
        --processes "$processes" \
        --tokenization spe \
        "${overwrite_args[@]}"
fi

dictionary="$output_dir/dict.mechet.txt"
if [[ "$reuse_preprocessed" == "1" && -s "$dictionary" ]]; then
    echo "Reusing audited EditRetro dictionary: $dictionary"
else
    "$chem_python_bin" "$repo_dir/preprocess/build_mechet_dictionary.py" \
        --base-dictionary "$repo_dir/preprocess/dict.txt" \
        --train-src "$output_dir/train.src" \
        --train-tgt "$output_dir/train.tgt" \
        --raw-train-csv "$repo_dir/datasets/$dataset/raw/raw_train.csv" \
        --output "$dictionary" \
        "${overwrite_args[@]}"
fi

"$repo_dir/preprocess/binarize_mechet.sh" \
    "$output_dir" "$dictionary" "$binarize_workers"

"$chem_python_bin" "$repo_dir/preprocess/audit_mechet_preprocessing.py" \
    --dataset "$dataset" \
    --profile "$profile" \
    --reference-dir "$reference_dir" \
    --preprocessed-dir "$output_dir" \
    --dictionary "$dictionary" \
    --augmentation "$augmentation"

echo "Validated EditRetro data: $output_dir"
