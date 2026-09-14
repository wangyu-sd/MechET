#!/usr/bin/env bash
set -uo pipefail

project_dir="/home/estar/pxy/mechet/baselines/RetroSynFlow-main"
python_bin="/home/estar/anaconda3/envs/molgen/bin/python"
run_root="${project_dir}/runs/retro_synflow"
coordinator_log="${run_root}/wait_and_resume_flow.log"

# Override with, for example: RETROSYNFLOW_GPUS="1 6 7" ./scripts/wait_for_gpu_and_resume_flow.sh
read -r -a candidate_gpus <<< "${RETROSYNFLOW_GPUS:-0 1 4 5 6 7}"
max_memory_mib="${RETROSYNFLOW_MAX_MEMORY_MIB:-1024}"
max_utilization="${RETROSYNFLOW_MAX_UTILIZATION:-10}"
required_consecutive="${RETROSYNFLOW_CONSECUTIVE_CHECKS:-3}"
poll_seconds="${RETROSYNFLOW_POLL_SECONDS:-60}"

datasets=("mech_uspto_31k_full" "flower_full")
declare -A target_epochs=(
  [mech_uspto_31k_full]=500
  [flower_full]=50
)
declare -A batch_sizes=(
  [mech_uspto_31k_full]=32
  [flower_full]=16
)
declare -A accumulation_steps=(
  [mech_uspto_31k_full]=3
  [flower_full]=6
)
declare -A ready_counts=()
declare -A reserved_gpus=()
declare -A job_pids=()
declare -A job_gpus=()
declare -A job_finished=()
declare -A pending=()
overall_status=0

mkdir -p "${run_root}" "${project_dir}/runtime"
touch "${coordinator_log}"

timestamp() {
  date -u '+%Y-%m-%dT%H:%M:%SZ'
}

log() {
  printf '[%s] %s\n' "$(timestamp)" "$*" | tee -a "${coordinator_log}"
}

die() {
  log "ERROR: $*"
  exit 1
}

[[ ${#candidate_gpus[@]} -gt 0 ]] || die "RETROSYNFLOW_GPUS is empty"
[[ "${max_memory_mib}" =~ ^[0-9]+$ ]] || die "invalid memory threshold"
[[ "${max_utilization}" =~ ^[0-9]+$ ]] || die "invalid utilization threshold"
[[ "${required_consecutive}" =~ ^[1-9][0-9]*$ ]] || die "invalid consecutive-check count"
[[ "${poll_seconds}" =~ ^[1-9][0-9]*$ ]] || die "invalid poll interval"
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is unavailable"
[[ -x "${python_bin}" ]] || die "Python is not executable: ${python_bin}"

exec 9>"${project_dir}/runtime/wait_and_resume_flow.lock"
if ! flock -n 9; then
  die "another RetroSynFlow wait/resume coordinator is already running"
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${project_dir}/.deps:${project_dir}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PATH="${project_dir}/.deps/bin:${PATH}"
export TORCH_EXTENSIONS_DIR="/tmp/retrosynflow_torch_extensions"
export MPLCONFIGDIR="/tmp/retrosynflow_matplotlib"
export RETRO_WANDB_ENABLE="false"
export RETRO_WORKSPACE="${project_dir}/runtime"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
mkdir -p "${TORCH_EXTENSIONS_DIR}" "${MPLCONFIGDIR}"

for gpu in "${candidate_gpus[@]}"; do
  [[ "${gpu}" =~ ^[0-9]+$ ]] || die "invalid GPU id: ${gpu}"
  if ! nvidia-smi -i "${gpu}" --query-gpu=index --format=csv,noheader,nounits >/dev/null 2>&1; then
    die "GPU ${gpu} does not exist or is not visible"
  fi
  ready_counts["${gpu}"]=0
done

for dataset in "${datasets[@]}"; do
  final_model="${run_root}/${dataset}/flow/final_model.pt"
  if [[ -f "${final_model}" ]]; then
    log "${dataset}: final_model.pt already exists; skipping"
  else
    pending["${dataset}"]=1
  fi
done

launch_training() {
  local dataset="$1"
  local gpu="$2"
  local dataset_root="${project_dir}/data/${dataset}"
  local output_dir="${run_root}/${dataset}/flow"
  local resume_log="${run_root}/${dataset}/logs/flow_resume.log"

  mkdir -p "${output_dir}" "$(dirname "${resume_log}")"
  log "${dataset}: assigning GPU ${gpu}; output=${output_dir}; log=${resume_log}"

  (
    cd "${project_dir}" || exit 1
    printf '[%s] starting on physical GPU %s\n' "$(timestamp)" "${gpu}" >> "${resume_log}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u scripts/train_synthon_flow_mechet.py \
      --dataset-root "${dataset_root}" \
      --output-dir "${output_dir}" \
      --epochs "${target_epochs[${dataset}]}" \
      --batch-size "${batch_sizes[${dataset}]}" \
      --gradient-accumulation "${accumulation_steps[${dataset}]}" \
      >> "${resume_log}" 2>&1
  ) &

  job_pids["${dataset}"]=$!
  job_gpus["${dataset}"]="${gpu}"
  reserved_gpus["${gpu}"]=1
  unset 'pending['"${dataset}"']'
}

reap_finished_jobs() {
  local dataset pid gpu exit_code
  for dataset in "${datasets[@]}"; do
    pid="${job_pids[${dataset}]:-}"
    [[ -n "${pid}" && -z "${job_finished[${dataset}]:-}" ]] || continue
    if kill -0 "${pid}" 2>/dev/null; then
      continue
    fi

    gpu="${job_gpus[${dataset}]}"
    if wait "${pid}"; then
      log "${dataset}: training completed successfully on GPU ${gpu}"
    else
      exit_code=$?
      log "${dataset}: training failed with exit code ${exit_code} on GPU ${gpu}"
      overall_status=1
    fi
    job_finished["${dataset}"]=1
    unset 'reserved_gpus['"${gpu}"']'
    ready_counts["${gpu}"]=0
  done
}

while [[ ${#pending[@]} -gt 0 ]]; do
  reap_finished_jobs

  for gpu in "${candidate_gpus[@]}"; do
    if [[ -n "${reserved_gpus[${gpu}]:-}" ]]; then
      continue
    fi

    gpu_state="$(nvidia-smi -i "${gpu}" \
      --query-gpu=memory.used,utilization.gpu \
      --format=csv,noheader,nounits 2>/dev/null || true)"
    if [[ -z "${gpu_state}" ]]; then
      ready_counts["${gpu}"]=0
      log "GPU ${gpu}: query failed"
      continue
    fi

    IFS=',' read -r memory_mib utilization <<< "${gpu_state}"
    memory_mib="${memory_mib//[[:space:]]/}"
    utilization="${utilization//[[:space:]]/}"
    if (( memory_mib <= max_memory_mib && utilization <= max_utilization )); then
      ready_counts["${gpu}"]=$((ready_counts["${gpu}"] + 1))
    else
      ready_counts["${gpu}"]=0
    fi
    log "GPU ${gpu}: memory=${memory_mib} MiB, utilization=${utilization}%, ready=${ready_counts[${gpu}]}/${required_consecutive}"
  done

  for dataset in "${datasets[@]}"; do
    [[ -n "${pending[${dataset}]:-}" ]] || continue
    selected_gpu=""
    for gpu in "${candidate_gpus[@]}"; do
      if [[ -z "${reserved_gpus[${gpu}]:-}" ]] \
        && (( ready_counts["${gpu}"] >= required_consecutive )); then
        selected_gpu="${gpu}"
        break
      fi
    done
    if [[ -n "${selected_gpu}" ]]; then
      launch_training "${dataset}" "${selected_gpu}"
    fi
  done

  if [[ ${#pending[@]} -gt 0 ]]; then
    sleep "${poll_seconds}"
  fi
done

for dataset in "${datasets[@]}"; do
  pid="${job_pids[${dataset}]:-}"
  [[ -n "${pid}" && -z "${job_finished[${dataset}]:-}" ]] || continue
  if wait "${pid}"; then
    log "${dataset}: training completed successfully on GPU ${job_gpus[${dataset}]}"
  else
    exit_code=$?
    log "${dataset}: training failed with exit code ${exit_code} on GPU ${job_gpus[${dataset}]}"
    overall_status=1
  fi
done

exit "${overall_status}"
