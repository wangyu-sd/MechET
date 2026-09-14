#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source_root=${1:-"$(cd "$repo_dir/.." && pwd)/baselines"}
destination_root="$repo_dir/baselines"

baseline_names=(
    LocalRetro-main
    ReactSeq-main
    Retro-MTGR-main
    RetroBridge-main
    RetroDFM-R-main
    RetroSynFlow-main
    RxnNano-main
    editretro-master
    retrosynthesis-main
)

if ! command -v rsync >/dev/null 2>&1; then
    echo "rsync is required" >&2
    exit 1
fi
if [[ ! -d "$source_root" ]]; then
    echo "baseline source root does not exist: $source_root" >&2
    exit 1
fi

mkdir -p "$destination_root"

common_excludes=(
    --exclude='/.git/'
    --exclude='/.venv*/'
    --exclude='/venv/'
    --exclude='/.conda-envs/'
    --exclude='/.cache/'
    --exclude='/.runtime/'
    --exclude='/.deps/'
    --exclude='/.eggs/'
    --exclude='/audits/'
    --exclude='/checkpoints/'
    --exclude='/models/'
    --exclude='/trained_models/'
    --exclude='/outputs/'
    --exclude='/output/'
    --exclude='/results/'
    --exclude='/runs/'
    --exclude='/logs/'
    --exclude='/exp/'
    --exclude='/runtime/'
    --exclude='/data/'
    --exclude='/dataset/'
    --exclude='/datasets/'
    --exclude='/unsloth_compiled_cache/'
    --exclude='__pycache__/'
    --exclude='.pytest_cache/'
    --exclude='.mypy_cache/'
    --exclude='.ruff_cache/'
    --exclude='build/'
    --exclude='dist/'
    --exclude='*.egg-info/'
    --exclude='*.csv'
    --exclude='*.jsonl'
    --exclude='*.parquet'
    --exclude='*.arrow'
    --exclude='*.feather'
    --exclude='*.pt'
    --exclude='*.pth'
    --exclude='*.ckpt'
    --exclude='*.safetensors'
    --exclude='*.bin'
    --exclude='*.idx'
    --exclude='*.pkl'
    --exclude='*.pickle'
    --exclude='*.npy'
    --exclude='*.npz'
    --exclude='*.src'
    --exclude='*.tgt'
    --exclude='*.so'
    --exclude='*.o'
    --exclude='*.log'
    --exclude='*.pid'
    --exclude='errorlog.txt'
    --exclude='/original Data/output-50k.txt'
)

for name in "${baseline_names[@]}"; do
    source_dir="$source_root/$name"
    destination_dir="$destination_root/$name"
    if [[ ! -d "$source_dir" ]]; then
        echo "missing expected baseline source: $source_dir" >&2
        exit 1
    fi
    mkdir -p "$destination_dir"
    # Never overwrite an existing destination file: local MechET adaptations
    # and uncommitted work take precedence over the external source snapshot.
    rsync -a --ignore-existing "${common_excludes[@]}" \
        "$source_dir/" "$destination_dir/"
done

# Small non-reaction support files required by the published code paths.
mkdir -p "$destination_root/LocalRetro-main/data/configs"
rsync -a --ignore-existing \
    "$source_root/LocalRetro-main/data/README.md" \
    "$destination_root/LocalRetro-main/data/"
rsync -a --ignore-existing \
    "$source_root/LocalRetro-main/data/configs/default_config.json" \
    "$destination_root/LocalRetro-main/data/configs/"

mkdir -p "$destination_root/ReactSeq-main/datasets/vocabs"
rsync -a --ignore-existing \
    "$source_root/ReactSeq-main/datasets/vocabs/whole_vocabs.src" \
    "$destination_root/ReactSeq-main/datasets/vocabs/"

mkdir -p "$destination_root/Retro-MTGR-main/data/USPT-50K/other_data"
rsync -a --ignore-existing \
    "$source_root/Retro-MTGR-main/data/USPT-50K/other_data/" \
    "$destination_root/Retro-MTGR-main/data/USPT-50K/other_data/"

echo "Synchronized ${#baseline_names[@]} source-only baselines into $destination_root"
echo "Existing destination files were preserved; datasets and model artifacts were skipped."
