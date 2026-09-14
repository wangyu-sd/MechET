#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
    echo "Usage: $0 PREPROCESSED_DIR [DICT] [WORKERS]" >&2
    exit 2
fi

input_dir=$(realpath "$1")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd "$script_dir/.." && pwd)
python_bin=${PYTHON_BIN:-python}
dictionary=$(realpath "${2:-$script_dir/dict.txt}")
workers=${3:-40}
output_dir="$input_dir/data-bin"

for split in train val test; do
    for lang in src tgt; do
        if [[ ! -s "$input_dir/$split.$lang" ]]; then
            echo "Missing or empty input: $input_dir/$split.$lang" >&2
            exit 1
        fi
    done
done

mkdir -p "$output_dir"
if command -v fairseq-preprocess >/dev/null 2>&1; then
    fairseq_preprocess=(fairseq-preprocess)
else
    export PYTHONPATH="$repo_dir/fairseq:$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
    fairseq_preprocess=("$python_bin" "$repo_dir/fairseq/fairseq_cli/preprocess.py")
fi
"${fairseq_preprocess[@]}" \
    --source-lang src \
    --target-lang tgt \
    --trainpref "$input_dir/train" \
    --validpref "$input_dir/val" \
    --testpref "$input_dir/test" \
    --destdir "$output_dir" \
    --workers "$workers" \
    --tgtdict "$dictionary" \
    --srcdict "$dictionary"
