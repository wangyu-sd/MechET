#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 DATASET_NAME GPU_ID FLOW_BATCH_SIZE" >&2
  exit 2
fi

dataset_name="$1"
gpu_id="$2"
flow_batch_size="$3"
project_dir="/home/estar/pxy/mechet/baselines/RetroSynFlow-main"
dataset_root="${project_dir}/data/${dataset_name}"
run_dir="${project_dir}/runs/retro_synflow/${dataset_name}"
python_bin="/home/estar/anaconda3/envs/molgen/bin/python"

mkdir -p "${run_dir}/logs" "${run_dir}/center" "${run_dir}/flow"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${project_dir}/.deps:${project_dir}/src"
export PATH="${project_dir}/.deps/bin:${PATH}"
export TORCH_EXTENSIONS_DIR="/tmp/retrosynflow_torch_extensions"
export MPLCONFIGDIR="/tmp/retrosynflow_matplotlib"
export RETRO_WANDB_ENABLE="false"
export RETRO_WORKSPACE="${project_dir}/runtime"

cd "${project_dir}"

"${python_bin}" scripts/preprocess_mechet.py \
  --dataset-root "${dataset_root}" \
  --builders synthon \
  --splits train val test \
  --allow-failures \
  2>&1 | tee -a "${run_dir}/logs/preprocess.log"

"${python_bin}" scripts/wait_for_gpu.py --gpu "${gpu_id}" \
  2>&1 | tee -a "${run_dir}/logs/gpu_wait.log"

export CUDA_VISIBLE_DEVICES="${gpu_id}"

"${python_bin}" scripts/train_reaction_center_mechet.py \
  --dataset-root "${dataset_root}" \
  --output-dir "${run_dir}/center" \
  --epochs 100 \
  --batch-size 128 \
  2>&1 | tee -a "${run_dir}/logs/center_train.log"

"${python_bin}" scripts/train_synthon_flow_mechet.py \
  --dataset-root "${dataset_root}" \
  --output-dir "${run_dir}/flow" \
  --epochs 500 \
  --batch-size "${flow_batch_size}" \
  2>&1 | tee -a "${run_dir}/logs/flow_train.log"
