#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 DATASET_NAME GPU_ID" >&2
  exit 2
fi

dataset_name="$1"
gpu_id="$2"
project_dir="/home/estar/pxy/mechet/baselines/RetroSynFlow-main"
mechet_dir="/home/estar/pxy/mechet/MechET"
dataset_root="${project_dir}/data/${dataset_name}"
run_dir="${project_dir}/runs/retro_synflow/${dataset_name}"
reference="${mechet_dir}/data/external_baselines/${dataset_name}/test.jsonl"
output_dir="${mechet_dir}/outputs/external_baselines/retrosynflow/${dataset_name}"
predictions="${output_dir}/predictions.jsonl"
python_bin="/home/estar/anaconda3/envs/molgen/bin/python"
evaluation_python_bin="/home/estar/anaconda3/envs/mechet/bin/python"
flow_checkpoint="${run_dir}/flow/final_model.pt"
center_checkpoint="${run_dir}/center/g2g_center.pth"
log_path="${run_dir}/logs/infer_eval.log"
lock_path="${run_dir}/evaluation/infer_eval.lock"

case "${dataset_name}" in
  mech_uspto_31k_full)
    expected_rows=3120
    ;;
  flower_full)
    expected_rows=28971
    ;;
  *)
    echo "unsupported frozen evaluation dataset: ${dataset_name}" >&2
    exit 2
    ;;
esac

for required in \
  "${python_bin}" \
  "${evaluation_python_bin}" \
  "${dataset_root}" \
  "${reference}" \
  "${flow_checkpoint}" \
  "${center_checkpoint}"; do
  if [[ ! -e "${required}" ]]; then
    echo "missing required path: ${required}" >&2
    exit 1
  fi
done

mkdir -p "${run_dir}/logs" "${run_dir}/evaluation" "${output_dir}"
exec 9>"${lock_path}"
if ! flock -n 9; then
  echo "another inference/evaluation job already holds ${lock_path}" >&2
  exit 1
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${project_dir}/.deps:${project_dir}/src"
export PATH="${project_dir}/.deps/bin:${PATH}"
export TORCH_EXTENSIONS_DIR="/tmp/retrosynflow_eval_torch_extensions"
export MPLCONFIGDIR="/tmp/retrosynflow_eval_matplotlib"
export RETRO_WANDB_ENABLE="false"
export RETRO_WORKSPACE="${project_dir}/runtime"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

cd "${project_dir}"
{
  echo "$(date --iso-8601=seconds) queued dataset=${dataset_name} gpu=${gpu_id}"
  "${python_bin}" scripts/wait_for_gpu.py \
    --gpu "${gpu_id}" \
    --ignore-skip-marker \
    --max-memory-mib 1024 \
    --max-utilization 10 \
    --consecutive 3 \
    --interval 60

  echo "$(date --iso-8601=seconds) inference_start dataset=${dataset_name} gpu=${gpu_id}"
  CUDA_VISIBLE_DEVICES="${gpu_id}" "${python_bin}" -u scripts/infer_eval_mechet.py \
    --dataset-root "${dataset_root}" \
    --reference "${reference}" \
    --flow-checkpoint "${flow_checkpoint}" \
    --center-checkpoint "${center_checkpoint}" \
    --work-dir "${run_dir}/evaluation" \
    --predictions "${predictions}" \
    --steps 100 \
    --samples 100 \
    --synthon-topk 2 \
    --samples-per-synthon 70 30 \
    --top-candidates 10 \
    --seed 42

  echo "$(date --iso-8601=seconds) evaluation_start dataset=${dataset_name}"
  "${evaluation_python_bin}" "${mechet_dir}/scripts/evaluate_endpoint_candidates.py" \
    --reference "${reference}" \
    --predictions "${predictions}" \
    --output "${output_dir}/predictions.evaluation.json" \
    --expected-rows "${expected_rows}" \
    --expected-candidates 10 \
    --candidate-semantics native_ranked
  echo "$(date --iso-8601=seconds) evaluation_complete dataset=${dataset_name}"
} 2>&1 | tee -a "${log_path}"
