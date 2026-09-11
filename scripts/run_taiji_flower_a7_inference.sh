#!/usr/bin/env bash
set -Eeuo pipefail

repo_dir=/aaa/fionafyang/buddy1/whaleywang/MechET
shared_hf_cache=/aaa/fionafyang/buddy1/whaleywang/OpenEvolveChem/data/hf_cache
shared_model_snapshot=$shared_hf_cache/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
config=configs/agent/tool_sft_flower_compact_full_state_qwen3_8b_a100.yaml
reference=data/flower_inverse_tool_sft_compact_full_state_v1/test.jsonl
dataset_manifest=data/flower_inverse_tool_sft_compact_full_state_v1/training_manifest.json
adapter=${MECHET_TRACE_ADAPTER:-${MECHET_A7_ADAPTER:-outputs/agent/tool_sft_flower_compact_full_state_qwen3_8b_a100_20260826}}
expected_rows=28967
expected_gpu=${MECHET_EXPECTED_GPU:-A100}
samples_per_target=${SAMPLES_PER_TARGET:-1}
inference_backend=${MECHET_INFERENCE_BACKEND:-vllm}
inference_script=${MECHET_INFERENCE_SCRIPT:-scripts/infer_mechet.py}
paper_condition=${MECHET_PAPER_CONDITION:-A7}
evaluation_condition=${MECHET_EVALUATION_CONDITION:-flower_a7_compact_full_state_seed17_k${samples_per_target}}
inference_protocol=${MECHET_INFERENCE_PROTOCOL:-trace_owned_compact_full_state_v1}
expected_adapter_condition=${MECHET_EXPECTED_ADAPTER_CONDITION:-}
expected_adapter_sha256=${MECHET_EXPECTED_ADAPTER_SHA256:-}
expected_adapter_train_sha256=${MECHET_EXPECTED_ADAPTER_TRAIN_SHA256:-}
headline_eligible=${MECHET_HEADLINE_ELIGIBLE:-1}
max_iterations=40
gpu_count=8
generation_workers_per_gpu=${MECHET_GENERATION_WORKERS_PER_GPU:-1}
trace_sample_batch_size=${MECHET_TRACE_SAMPLE_BATCH_SIZE:-$samples_per_target}
task_shard_count=${MECHET_TASK_SHARD_COUNT:-1}
task_shard_index=${MECHET_TASK_SHARD_INDEX:-0}
revision=b968826d9c46dd6066d109eabc6255188de91218
output_dir=${MECHET_INFERENCE_OUTPUT:-outputs/eval/iclr_full/a7_compact_full_state_seed17_k${samples_per_target}}

if [[ "${MECHET_SKIP_CONDA_ACTIVATE:-0}" != "1" ]]; then
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate meteor
fi
cd "$repo_dir"

export HF_HUB_CACHE="$shared_hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$repo_dir/src:$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ ! "$samples_per_target" =~ ^[1-9][0-9]*$ ]]; then
  echo >&2 "SAMPLES_PER_TARGET must be a positive integer"
  exit 2
fi
if [[ ! "$generation_workers_per_gpu" =~ ^[1-9][0-9]*$ ]]; then
  echo >&2 "MECHET_GENERATION_WORKERS_PER_GPU must be a positive integer"
  exit 2
fi
if [[ ! "$trace_sample_batch_size" =~ ^[1-9][0-9]*$ ]]; then
  echo >&2 "MECHET_TRACE_SAMPLE_BATCH_SIZE must be a positive integer"
  exit 2
fi
if [[ "$inference_backend" != "vllm" && "$inference_backend" != "transformers" ]]; then
  echo >&2 "MECHET_INFERENCE_BACKEND must be vllm or transformers"
  exit 2
fi
if [[ "$inference_backend" == "vllm" && "$generation_workers_per_gpu" -ne 1 ]]; then
  echo >&2 "vLLM requires one engine per GPU; set MECHET_GENERATION_WORKERS_PER_GPU=1"
  exit 2
fi
if [[ ! "$task_shard_count" =~ ^[1-9][0-9]*$ ]] || \
   [[ ! "$task_shard_index" =~ ^[0-9]+$ ]] || \
   (( task_shard_index >= task_shard_count )); then
  echo >&2 "require MECHET_TASK_SHARD_COUNT >= 1 and 0 <= MECHET_TASK_SHARD_INDEX < count"
  exit 2
fi
generation_shards=$((gpu_count * generation_workers_per_gpu))
task_expected_rows=$(( (expected_rows + task_shard_count - 1 - task_shard_index) / task_shard_count ))

python - "$config" "$reference" "$dataset_manifest" "$adapter" "$expected_gpu" "$revision" "$expected_rows" "$max_iterations" "$paper_condition" "$inference_protocol" "$expected_adapter_condition" "$expected_adapter_sha256" "$expected_adapter_train_sha256" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

import torch
import yaml

(
    config_path,
    reference_path,
    manifest_path,
    adapter_path,
    expected_gpu,
    revision,
    expected_rows,
    max_iterations,
    paper_condition,
    inference_protocol,
    expected_adapter_condition,
    expected_adapter_sha256,
    expected_adapter_train_sha256,
) = sys.argv[1:]
config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
adapter = Path(adapter_path)
reference = Path(reference_path)
required = [
    reference,
    Path(manifest_path),
    adapter / "adapter_config.json",
    adapter / "adapter_model.safetensors",
    adapter / "adapter_manifest.json",
    adapter / "data_contract.json",
]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise SystemExit(f"missing frozen A7 inference artifacts: {missing}")
split = dict(manifest["splits"]["test"])
if int(split["rows"]) != int(expected_rows):
    raise SystemExit(f"A7 test rows {split['rows']} != {expected_rows}")
if hashlib.sha256(reference.read_bytes()).hexdigest() != str(split["sha256"]):
    raise SystemExit("A7 test-file hash mismatch")
if str(manifest.get("observation_mode")) != "compact_full_state_v1":
    raise SystemExit("A7 dataset is not compact_full_state_v1")
adapter_manifest = json.loads((adapter / "adapter_manifest.json").read_text())
contract = json.loads((adapter / "data_contract.json").read_text())
if str(adapter_manifest.get("base_model_revision") or "") != revision:
    raise SystemExit("adapter/base revision mismatch")
expected_train_sha256 = expected_adapter_train_sha256 or str(manifest["splits"]["train"]["sha256"])
if str(adapter_manifest.get("train_file_sha256") or "") != expected_train_sha256:
    raise SystemExit("adapter train-data lineage mismatch")
if expected_adapter_condition and str(adapter_manifest.get("condition_name") or "") != expected_adapter_condition:
    raise SystemExit("adapter condition mismatch")
if expected_adapter_sha256 and str(adapter_manifest.get("adapter_sha256") or "") != expected_adapter_sha256:
    raise SystemExit("adapter artifact hash mismatch")
if str(contract.get("environment_revision") or "") != "trace_owned_compact_full_state_v1":
    raise SystemExit("adapter is not the frozen compact-full-state A7 artifact")
if int(config["environment"]["max_tool_calls"]) != int(max_iterations):
    raise SystemExit("runtime max_iterations differs from the training tool budget")
if torch.cuda.device_count() != 8:
    raise SystemExit(f"expected 8 GPUs, got {torch.cuda.device_count()}")
names = [torch.cuda.get_device_name(index) for index in range(8)]
if not all(expected_gpu.upper() in name.upper() for name in names):
    raise SystemExit(f"expected {expected_gpu}, got {names}")
print({
    "paper_condition": paper_condition,
    "protocol": inference_protocol,
    "test_rows": int(split["rows"]),
    "adapter": str(adapter.resolve()),
    "adapter_condition": str(adapter_manifest.get("condition_name") or ""),
    "adapter_sha256": str(adapter_manifest.get("adapter_sha256") or ""),
    "max_iterations": int(max_iterations),
    "gpus": names,
}, flush=True)
PY

mkdir -p "$output_dir/generation"
reference_sha256=$(python - "$dataset_manifest" <<'PY'
import json
from pathlib import Path
import sys
manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(manifest["splits"]["test"]["sha256"])
PY
)

# A 24-worker H20 launch must not make every process parse the 727-MiB test
# JSONL and reread the 16-GiB base checkpoint from Ceph. Materialize each
# immutable artifact once on node-local storage; inference semantics and the
# recorded full-dataset digest remain unchanged.
local_model_dir=${MECHET_LOCAL_MODEL_DIR:-/tmp/mechet_qwen3_8b_${revision}}
if [[ ! -f "$local_model_dir/.mechet_stage_complete" ]]; then
  model_stage=$(mktemp -d /tmp/mechet_qwen3_8b_stage.XXXXXX)
  echo "[meteor-stage] artifact=base-model source=$shared_model_snapshot destination=$local_model_dir time=$(date --iso-8601=seconds)"
  cp -aL "$shared_model_snapshot/." "$model_stage/"
  printf '%s\n' "$revision" > "$model_stage/.mechet_stage_complete"
  mv "$model_stage" "$local_model_dir"
fi

local_reference_dir=${MECHET_LOCAL_REFERENCE_DIR:-/tmp/mechet_a7_test_${reference_sha256:0:16}_${generation_shards}_task${task_shard_index}of${task_shard_count}}
if [[ ! -f "$local_reference_dir/.mechet_stage_complete" ]]; then
  reference_stage=$(mktemp -d /tmp/mechet_a7_test_stage.XXXXXX)
  echo "[meteor-stage] artifact=test-shards source=$reference destination=$local_reference_dir local_shards=$generation_shards task_shard=${task_shard_index}/${task_shard_count} time=$(date --iso-8601=seconds)"
  python - "$reference" "$reference_stage" "$generation_shards" "$expected_rows" "$task_shard_count" "$task_shard_index" "$task_expected_rows" <<'PY'
from contextlib import ExitStack
from pathlib import Path
import sys

source = Path(sys.argv[1])
destination = Path(sys.argv[2])
local_shards = int(sys.argv[3])
expected_source = int(sys.argv[4])
task_shards = int(sys.argv[5])
task_index = int(sys.argv[6])
expected_selected = int(sys.argv[7])
source_count = 0
selected_count = 0
with ExitStack() as stack:
    handles = [
        stack.enter_context((destination / f"test.shard-{index:03d}.jsonl").open("w", encoding="utf-8"))
        for index in range(local_shards)
    ]
    selected_handle = stack.enter_context(
        (destination / "test.selected.jsonl").open("w", encoding="utf-8")
    )
    with source.open("r", encoding="utf-8") as input_handle:
        for line in input_handle:
            if not line.strip():
                continue
            if source_count % task_shards == task_index:
                handles[selected_count % local_shards].write(line)
                selected_handle.write(line)
                selected_count += 1
            source_count += 1
if source_count != expected_source:
    raise SystemExit(f"expected {expected_source} source rows, read {source_count}")
if selected_count != expected_selected:
    raise SystemExit(f"expected {expected_selected} selected rows, staged {selected_count}")
(destination / ".mechet_stage_complete").write_text(
    f"source_rows={source_count}\nselected_rows={selected_count}\n"
    f"local_shards={local_shards}\ntask_shard={task_index}/{task_shards}\n"
)
PY
  mv "$reference_stage" "$local_reference_dir"
fi
selected_reference="$local_reference_dir/test.selected.jsonl"
echo "inference_concurrency backend=$inference_backend generation_workers_per_gpu=$generation_workers_per_gpu generation_shards=$generation_shards samples_per_target=$samples_per_target trace_sample_batch_size=$trace_sample_batch_size task_shard=${task_shard_index}/${task_shard_count} task_expected_rows=$task_expected_rows"
pids=()
for worker in $(seq 0 $((generation_shards - 1))); do
  gpu=$((worker % gpu_count))
  shard=$(printf '%03d' "$worker")
  worker_reference="$local_reference_dir/test.shard-${shard}.jsonl"
  # Every data-parallel vLLM process sees its selected GPU as local rank zero.
  # Without separate caches they all compile into rank_0_0 concurrently and
  # can livelock during LoRA/Triton kernel generation.
  worker_cache_root="/tmp/mechet_vllm_cache_worker_${worker}"
  mkdir -p "$worker_cache_root"
  CUDA_VISIBLE_DEVICES="$gpu" \
  VLLM_CACHE_ROOT="$worker_cache_root/vllm" \
  TORCHINDUCTOR_CACHE_DIR="$worker_cache_root/torchinductor" \
  TRITON_CACHE_DIR="$worker_cache_root/triton" \
  TORCHINDUCTOR_COMPILE_THREADS="${MECHET_TORCHINDUCTOR_COMPILE_THREADS:-2}" \
  OMP_NUM_THREADS="${MECHET_OMP_NUM_THREADS_PER_WORKER:-4}" \
  MKL_NUM_THREADS="${MECHET_OMP_NUM_THREADS_PER_WORKER:-4}" \
  python "$inference_script" \
    --config "$config" \
    --data "$worker_reference" \
    --output "$output_dir/generation/predictions.shard-${shard}.jsonl" \
    --mode trace \
    --observation-mode compact_full_state \
    --condition-name "$evaluation_condition" \
    --model-name "$local_model_dir" \
    --adapter "$adapter" \
    --backend "$inference_backend" \
    --prompt-source reference \
    --shard-count 1 \
    --shard-index 0 \
    --prevalidated-data-sha256 "$reference_sha256" \
    --progress-file "$output_dir/generation/progress.shard-${shard}.json" \
    --progress-every 5 \
    --samples-per-target "$samples_per_target" \
    --trace-sample-batch-size "$trace_sample_batch_size" \
    --max-iterations "$max_iterations" \
    --max-new-tokens 512 \
    --vllm-max-model-len "${MECHET_VLLM_MAX_MODEL_LEN:-16384}" \
    --vllm-max-num-seqs "${MECHET_VLLM_MAX_NUM_SEQS:-64}" \
    --vllm-gpu-memory-utilization "${MECHET_VLLM_GPU_MEMORY_UTILIZATION:-0.9}" \
    --temperature 0.7 \
    --top-p 0.95 \
    --seed 17 \
    --resume \
    >"$output_dir/generation/shard-${shard}.log" 2>&1 &
  pids+=("$!")
done

while :; do
  live=0
  for pid in "${pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      live=$((live + 1))
    fi
  done
  completed=$(python - "$output_dir/generation" <<'PY'
import json
from pathlib import Path
import sys
total = 0
for path in Path(sys.argv[1]).glob("progress.shard-*.json"):
    try:
        total += int(json.loads(path.read_text()).get("completed_targets") or 0)
    except (OSError, ValueError, json.JSONDecodeError):
        pass
print(total)
PY
)
  echo "[meteor-progress] stage=${paper_condition}-generation completed_targets=${completed}/${task_expected_rows} live_shards=${live}/${generation_shards} k=${samples_per_target} task_shard=${task_shard_index}/${task_shard_count} time=$(date --iso-8601=seconds)"
  if (( live == 0 )); then
    break
  fi
  sleep 60
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
if [[ "$status" -ne 0 ]]; then
  tail -n 80 "$output_dir"/generation/shard-*.log
  exit "$status"
fi

python - "$samples_per_target" "$task_expected_rows" "$generation_shards" "$output_dir" "$task_shard_count" "$task_shard_index" "$paper_condition" "$headline_eligible" "$evaluation_condition" "$inference_protocol" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

k, expected, expected_shards = map(int, sys.argv[1:4])
root = Path(sys.argv[4])
task_shards, task_index = map(int, sys.argv[5:7])
paper_condition, headline_eligible, condition_name, protocol = sys.argv[7:11]
shards = sorted((root / "generation").glob("predictions.shard-*.jsonl"))
if len(shards) != expected_shards:
    raise SystemExit(f"expected {expected_shards} shards, got {len(shards)}")
rows = []
for path in shards:
    rows.extend(json.loads(line) for line in path.open() if line.strip())
rows.sort(key=lambda row: str(row.get("id") or ""))
if len(rows) != expected or len({row["id"] for row in rows}) != expected:
    raise SystemExit(f"expected {expected} unique predictions, got {len(rows)}")
bad = [row["id"] for row in rows if len(row.get("candidates") or []) != k]
if bad:
    raise SystemExit(f"incomplete A7 candidate sets: {bad[:10]}")
output = root / "predictions.jsonl"
with output.open("w", encoding="utf-8") as handle:
    for row in rows:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
manifest = {
    "artifact_type": "flower_trace_owned_compact_full_state_sampled_test_manifest",
    "paper_condition": paper_condition,
    "headline_eligible": headline_eligible == "1",
    "condition_name": condition_name,
    "protocol": protocol,
    "n_targets": len(rows),
    "samples_per_target": k,
    "n_candidates": len(rows) * k,
    "task_shard_count": task_shards,
    "task_shard_index": task_index,
    "candidate_selection": "formal-execution/reward rank; no ground truth used",
    "predictions": str(output.resolve()),
    "predictions_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
}
(root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2), flush=True)
PY

exec python scripts/evaluate_prediction_set.py \
  --reference "$selected_reference" \
  --predictions "$output_dir/predictions.jsonl" \
  --output "$output_dir/evaluation.json" \
  --condition-name "$evaluation_condition" \
  --expected-rows "$task_expected_rows" \
  --expected-candidates "$samples_per_target"
