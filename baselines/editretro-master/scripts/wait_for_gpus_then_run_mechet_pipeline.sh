#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: $0 DATASET {GPU[,GPU...]|any:COUNT} [RUN_NAME]" >&2
    exit 2
fi

dataset=$1
gpu_request=$2
run_name=${3:-official_full_seed1}
gpu_mode=fixed
required_gpu_count=0
if [[ "$gpu_request" =~ ^any:([1-9][0-9]*)$ ]]; then
    gpu_mode=any
    required_gpu_count=${BASH_REMATCH[1]}
elif [[ ! "$gpu_request" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "GPUs must be comma-separated CUDA indices or any:COUNT: $gpu_request" >&2
    exit 2
fi

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
output_root=${PIPELINE_OUTPUT_ROOT:-/data/pxy/models/EditRetro}
run_dir="$output_root/$dataset/full/$run_name"
poll_seconds=${GPU_WAIT_POLL_SECONDS:-60}
if [[ ! "$poll_seconds" =~ ^[1-9][0-9]*$ ]]; then
    echo "GPU_WAIT_POLL_SECONDS must be a positive integer: $poll_seconds" >&2
    exit 2
fi

mkdir -p "$run_dir"
exec 9>"$run_dir/gpu_wait.lock"
if ! flock -n 9; then
    echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] another GPU waiter is already active for $run_dir" >&2
    exit 1
fi

if [[ "$gpu_mode" == "fixed" ]]; then
    IFS=',' read -r -a gpu_ids <<< "$gpu_request"
    echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] waiting for GPUs $gpu_request to have no compute processes"
else
    echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] waiting for any $required_gpu_count GPUs to have no compute processes"
fi

while true; do
    if [[ "$gpu_mode" == "any" ]]; then
        if ! gpu_index_output=$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null); then
            echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] failed to enumerate GPUs; retrying"
            sleep "$poll_seconds"
            continue
        fi
        gpu_ids=()
        while IFS= read -r gpu; do
            gpu=${gpu//[[:space:]]/}
            [[ -n "$gpu" ]] && gpu_ids+=("$gpu")
        done <<< "$gpu_index_output"
        if (( ${#gpu_ids[@]} < required_gpu_count )); then
            echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] requested $required_gpu_count GPUs, but only ${#gpu_ids[@]} are visible" >&2
            exit 1
        fi
    fi

    busy=()
    idle=()
    for gpu in "${gpu_ids[@]}"; do
        if ! pids=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null); then
            busy+=("$gpu(query-failed)")
            continue
        fi
        compact=${pids//$'\n'/,}
        compact=${compact//[[:space:]]/}
        compact=${compact%,}
        if [[ -n "$compact" ]]; then
            busy+=("$gpu(pid=$compact)")
        else
            idle+=("$gpu")
        fi
    done

    if [[ "$gpu_mode" == "fixed" && ${#busy[@]} -eq 0 ]]; then
        selected_gpus=$gpu_request
    elif [[ "$gpu_mode" == "any" && ${#idle[@]} -ge required_gpu_count ]]; then
        selected=("${idle[@]:0:required_gpu_count}")
        selected_gpus=$(IFS=,; echo "${selected[*]}")
    else
        selected_gpus=
    fi

    if [[ -n "$selected_gpus" ]]; then
        echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] selected idle GPUs $selected_gpus; continuing $dataset pipeline"
        exec "$repo_dir/scripts/run_mechet_full_pipeline.sh" "$dataset" "$selected_gpus" "$run_name"
    fi

    busy_text=$(IFS=' '; echo "${busy[*]}")
    if [[ "$gpu_mode" == "any" ]]; then
        echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] still waiting: need $required_gpu_count idle GPUs, found ${#idle[@]}; busy: $busy_text"
    else
        echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] still waiting: $busy_text"
    fi
    sleep "$poll_seconds"
done
