#!/usr/bin/env bash
set -Eeuo pipefail

repo=/home/estar/pxy/mechet/baselines/RetroDFM-R-main
mechet_new=/home/estar/pxy/MechET_new
common_data=/home/estar/pxy/mechet/MechET/data/external_baselines
output_root=/home/estar/pxy/mechet/MechET/outputs/external_baselines/retrodfm_r
model_revision=d0a1d37956ba6ba6c2af6ad36a6f57fa8497b142
model_dir=/data/pxy/models/RetroDFM-R-8B/${model_revision}
infer_python=/home/estar/anaconda3/envs/verl080/bin/python
hf_cli=/home/estar/anaconda3/envs/verl080/bin/hf
vllm=/home/estar/anaconda3/envs/verl080/bin/vllm
eval_python=/home/estar/anaconda3/envs/mechet/bin/python
candidate_count=10
temperature=1.0
top_p=1.0
max_tokens=4096
server_seed=42
shard_rows=128
requested_gpu_count=3
max_idle_memory_mib=512
port=30080

mkdir -p "$output_root"
run_log="$output_root/run.log"
exec > >(tee -a "$run_log") 2>&1

timestamp() {
    date -u '+%Y-%m-%dT%H:%M:%SZ'
}

log() {
    printf '[%s] %s\n' "$(timestamp)" "$*"
}

fail() {
    log "ERROR: $*"
    exit 1
}

require_rows() {
    local path=$1
    local expected=$2
    local actual
    actual=$(wc -l < "$path")
    [[ "$actual" -eq "$expected" ]] || fail "$path has $actual rows; expected $expected"
}

download_model() {
    if [[ -s "$model_dir/config.json" && -s "$model_dir/model.safetensors.index.json" ]]; then
        log "Pinned checkpoint already present: $model_dir"
        return
    fi
    log "Downloading OpenDFM/RetroDFM-R-8B at pinned revision $model_revision"
    mkdir -p "$model_dir"
    # The host's local proxy avoids a broken direct IPv6 route to the Hub.
    HF_HUB_DISABLE_XET=0 HF_HUB_ETAG_TIMEOUT=60 HF_HUB_DOWNLOAD_TIMEOUT=3600 \
        "$hf_cli" download OpenDFM/RetroDFM-R-8B \
        --revision "$model_revision" --local-dir "$model_dir"
    [[ -s "$model_dir/config.json" && -s "$model_dir/model.safetensors.index.json" ]] \
        || fail "checkpoint download did not produce a complete index/config"
}

select_idle_gpus() {
    local -a idle=()
    local index used
    while true; do
        idle=()
        while IFS=, read -r index used; do
            index=${index//[[:space:]]/}
            used=${used//[[:space:]]/}
            if [[ "$used" =~ ^[0-9]+$ ]] && (( used <= max_idle_memory_mib )); then
                idle+=("$index")
            fi
        done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
        if (( ${#idle[@]} >= requested_gpu_count )); then
            local joined
            joined=$(IFS=,; printf '%s' "${idle[*]:0:requested_gpu_count}")
            printf '%s' "$joined"
            return
        fi
        # This function is used in command substitution; keep status messages
        # out of stdout so gpu_list contains only the comma-separated IDs.
        log "Waiting for $requested_gpu_count idle GPUs (<=${max_idle_memory_mib} MiB); found ${#idle[@]}" >&2
        sleep 60
    done
}

server_pid=''
cleanup_server() {
    if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
        log "Stopping vLLM server PID $server_pid"
        kill "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
}
trap cleanup_server EXIT
trap 'exit 130' INT TERM

start_server() {
    local gpu_list=$1
    local gpu_count=$2
    while ss -ltn | awk '{print $4}' | grep -Eq ":${port}$"; do
        port=$((port + 1))
    done
    log "Starting pinned checkpoint on GPUs $gpu_list with data_parallel_size=$gpu_count, port=$port"
    CUDA_VISIBLE_DEVICES="$gpu_list" \
        VLLM_WORKER_MULTIPROC_METHOD=spawn \
        "$vllm" serve "$model_dir" \
        --served-model-name OpenDFM/RetroDFM-R-8B \
        --host 127.0.0.1 \
        --port "$port" \
        --data-parallel-size "$gpu_count" \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.88 \
        --max-model-len 8192 \
        --seed "$server_seed" \
        > "$output_root/server.log" 2>&1 &
    server_pid=$!
    for _ in $(seq 1 180); do
        if curl -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
            log "vLLM server is ready"
            return
        fi
        kill -0 "$server_pid" 2>/dev/null || fail "vLLM server exited; see $output_root/server.log"
        sleep 10
    done
    fail "vLLM server did not become ready within 30 minutes"
}

prepare_shards() {
    local input=$1
    local shard_dir=$2
    local expected=$3
    mkdir -p "$shard_dir"
    if [[ ! -f "$shard_dir/.complete" ]]; then
        log "Creating resumable ${shard_rows}-row inference shards under $shard_dir"
        split -d -a 5 -l "$shard_rows" --additional-suffix=.jsonl \
            "$input" "$shard_dir/input-"
        local shard_total
        shard_total=$(wc -l "$shard_dir"/input-*.jsonl | tail -1 | awk '{print $1}')
        [[ "$shard_total" -eq "$expected" ]] \
            || fail "shard total $shard_total does not match expected $expected"
        touch "$shard_dir/.complete"
    fi
}

run_dataset() {
    local dataset=$1
    local expected_rows=$2
    local generation_input="$repo/data/eval/$dataset/test.jsonl"
    local frozen_reference="$common_data/$dataset/test.jsonl"
    local output_dir="$output_root/$dataset"
    local shard_dir="$output_dir/input_shards"
    local raw_dir="$output_dir/raw_shards"
    local raw_generations="$output_dir/raw_generations.jsonl"
    local predictions="$output_dir/predictions.jsonl"
    local evaluation_reference="$output_dir/evaluation_reference.jsonl"
    local manifest="$output_dir/run_manifest.json"
    local evaluation="$output_dir/evaluation.json"

    mkdir -p "$output_dir" "$raw_dir"
    require_rows "$generation_input" "$expected_rows"
    require_rows "$frozen_reference" "$expected_rows"
    prepare_shards "$generation_input" "$shard_dir" "$expected_rows"

    log "Starting $dataset inference: $expected_rows targets x $candidate_count samples"
    local input_shard stem raw_shard input_rows raw_rows
    for input_shard in "$shard_dir"/input-*.jsonl; do
        stem=$(basename "$input_shard" .jsonl)
        raw_shard="$raw_dir/$stem.jsonl"
        input_rows=$(wc -l < "$input_shard")
        if [[ -f "$raw_shard" ]]; then
            raw_rows=$(wc -l < "$raw_shard")
            if [[ "$raw_rows" -eq "$input_rows" ]]; then
                log "$dataset $stem already complete ($raw_rows rows); skipping"
                continue
            fi
            mv "$raw_shard" "$raw_shard.incomplete.$(date -u +%Y%m%dT%H%M%SZ)"
        fi
        log "$dataset $stem: generating $input_rows rows"
        "$infer_python" "$repo/eval/generate.py" \
            --data_path "$input_shard" \
            --output_dir "$raw_dir" \
            --task "$stem" \
            --base_url "http://127.0.0.1:${port}/v1" \
            --model OpenDFM/RetroDFM-R-8B \
            --n "$candidate_count" \
            --temperature "$temperature" \
            --max_tokens "$max_tokens" \
            --concurrency "$shard_rows" \
            --timeout 3600
    done

    log "Merging $dataset raw shards"
    awk 'FNR == 1 && NR != 1 { } { print }' "$raw_dir"/input-*.jsonl > "$raw_generations"
    require_rows "$raw_generations" "$expected_rows"

    log "Exporting $dataset predictions to the shared MechET contract"
    "$eval_python" "$repo/eval/export_mechet_predictions.py" \
        --generation-input "$generation_input" \
        --generations "$raw_generations" \
        --reference "$frozen_reference" \
        --predictions "$predictions" \
        --evaluation-reference "$evaluation_reference" \
        --manifest "$manifest" \
        --dataset "$dataset" \
        --checkpoint "$model_dir" \
        --model-revision "$model_revision" \
        --expected-rows "$expected_rows" \
        --expected-candidates "$candidate_count" \
        --temperature "$temperature" \
        --top-p "$top_p" \
        --max-new-tokens "$max_tokens" \
        --seed "$server_seed" \
        --inference-server "vLLM 0.20.2 OpenAI-compatible server; official eval/generate.py unchanged"

    log "Evaluating $dataset with MechET_new/scripts/evaluate_endpoint_candidates.py"
    (
        cd "$mechet_new"
        PYTHONPATH="$mechet_new/src" "$eval_python" \
            scripts/evaluate_endpoint_candidates.py \
            --reference "$evaluation_reference" \
            --predictions "$predictions" \
            --output "$evaluation" \
            --expected-rows "$expected_rows" \
            --expected-candidates "$candidate_count"
    ) | tee "$output_dir/evaluation.log"
    log "$dataset evaluation complete: $evaluation"
}

log "RetroDFM-R frozen-checkpoint evaluation started"
log "Revision: $model_revision"
log "Output root: $output_root"
download_model

require_rows "$repo/data/eval/mech_uspto_31k_full/test.jsonl" 3120
require_rows "$repo/data/eval/flower_full/test.jsonl" 28971
require_rows "$common_data/mech_uspto_31k_full/test.jsonl" 3120
require_rows "$common_data/flower_full/test.jsonl" 28971

gpu_list=$(select_idle_gpus)
gpu_count=$(awk -F, '{print NF}' <<< "$gpu_list")
start_server "$gpu_list" "$gpu_count"

run_dataset mech_uspto_31k_full 3120
run_dataset flower_full 28971

cleanup_server
server_pid=''
touch "$output_root/COMPLETED"
log "All RetroDFM-R frozen-checkpoint inference and evaluation jobs completed"
