#!/usr/bin/env bash
set -Eeuo pipefail

repo_dir=${MECHET_REPO_DIR:-/aaa/fionafyang/buddy1/whaleywang/MechET-controls}
artifact_root=${MECHET_ARTIFACT_ROOT:-/aaa/fionafyang/buddy1/whaleywang/MechET}
shared_hf_cache=/aaa/fionafyang/buddy1/whaleywang/OpenEvolveChem/data/hf_cache
revision=b968826d9c46dd6066d109eabc6255188de91218
shared_model_snapshot=$shared_hf_cache/models--Qwen--Qwen3-8B/snapshots/$revision
control=${MECHET_TRACE_CONTROL:?set MECHET_TRACE_CONTROL}
expected_gpu=${MECHET_EXPECTED_GPU:-A100}
samples_per_target=${SAMPLES_PER_TARGET:-1}
expected_rows=28967
gpu_count=8

case "$control" in
  a5_independent_answer)
    config=configs/iclr/a5_loose_trace_answer_inference.yaml
    reference=$artifact_root/data/iclr_program_controls_v1/loose_trace_answer/test.jsonl
    dataset_manifest=$artifact_root/data/iclr_program_controls_v1/manifest.json
    manifest_task=loose_trace_answer
    adapter=$artifact_root/outputs/iclr/a5_loose_trace_answer_seed17
    intervention=none
    observation_mode=action_delta
    allow_answer=1
    output_dir=${MECHET_INFERENCE_OUTPUT:-outputs/eval/iclr_full/a5_independent_answer_seed17_k${samples_per_target}}
    ;;
  b1_no_enumeration)
    config=configs/iclr/b1_no_enumeration_sft.yaml
    reference=data/iclr_feedback_controls_v1/b1_no_enumeration/test.jsonl
    dataset_manifest=data/iclr_feedback_controls_v1/b1_no_enumeration/manifest.json
    manifest_task=b1_no_enumeration
    adapter=outputs/iclr/b1_no_enumeration_seed17
    intervention=disable_inspect_state
    observation_mode=compact_full_state
    allow_answer=0
    output_dir=${MECHET_INFERENCE_OUTPUT:-outputs/eval/iclr_full/b1_no_enumeration_seed17_k${samples_per_target}}
    ;;
  b3_stale_feedback)
    config=configs/iclr/b3_stale_feedback_sft.yaml
    reference=data/iclr_feedback_controls_v1/b3_stale_feedback/test.jsonl
    dataset_manifest=data/iclr_feedback_controls_v1/b3_stale_feedback/manifest.json
    manifest_task=b3_stale_feedback
    adapter=outputs/iclr/b3_stale_feedback_seed17
    intervention=stale_tool_observations
    observation_mode=compact_full_state
    allow_answer=0
    output_dir=${MECHET_INFERENCE_OUTPUT:-outputs/eval/iclr_full/b3_stale_feedback_seed17_k${samples_per_target}}
    ;;
  *) echo >&2 "unsupported MECHET_TRACE_CONTROL=$control"; exit 2 ;;
esac

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

python - "$dataset_manifest" "$manifest_task" "$reference" "$adapter" "$expected_rows" "$expected_gpu" "$revision" <<'PY'
import hashlib, json, sys
from pathlib import Path
import torch

manifest_path, task, reference_path, adapter_path, expected_rows, expected_gpu, revision = sys.argv[1:]
manifest = json.loads(Path(manifest_path).read_text())
split = manifest["tasks"][task]["test"]
reference = Path(reference_path)
adapter = Path(adapter_path)
required = [reference, adapter / "adapter_config.json", adapter / "adapter_model.safetensors", adapter / "adapter_manifest.json"]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise SystemExit(f"missing trace-control artifacts: {missing}")
if int(split["rows"]) != int(expected_rows):
    raise SystemExit(f"test rows {split['rows']} != {expected_rows}")
if hashlib.sha256(reference.read_bytes()).hexdigest() != str(split["sha256"]):
    raise SystemExit("trace-control test hash mismatch")
adapter_manifest = json.loads((adapter / "adapter_manifest.json").read_text())
if str(adapter_manifest.get("base_model_revision") or "") != revision:
    raise SystemExit("adapter/base revision mismatch")
if torch.cuda.device_count() != 8:
    raise SystemExit(f"expected 8 GPUs, got {torch.cuda.device_count()}")
names = [torch.cuda.get_device_name(i) for i in range(8)]
if not all(expected_gpu.upper() in name.upper() for name in names):
    raise SystemExit(f"expected {expected_gpu}, got {names}")
print({"control": task, "test_rows": int(split["rows"]), "adapter": str(adapter.resolve()), "gpus": names}, flush=True)
PY

reference_sha256=$(sha256sum "$reference" | awk '{print $1}')
local_model_dir=/tmp/mechet_qwen3_8b_${revision}
if [[ ! -f "$local_model_dir/.mechet_stage_complete" ]]; then
  model_stage=$(mktemp -d /tmp/mechet_qwen3_8b_stage.XXXXXX)
  echo "[meteor-stage] artifact=base-model destination=$local_model_dir time=$(date --iso-8601=seconds)"
  cp -aL "$shared_model_snapshot/." "$model_stage/"
  printf '%s\n' "$revision" > "$model_stage/.mechet_stage_complete"
  mv "$model_stage" "$local_model_dir"
fi
local_reference=/tmp/mechet_${control}_${reference_sha256:0:16}.jsonl
if [[ ! -f "$local_reference" ]]; then
  cp "$reference" "$local_reference.tmp.$$"
  mv "$local_reference.tmp.$$" "$local_reference"
fi

mkdir -p "$output_dir/generation"
extra_args=()
if (( allow_answer )); then extra_args+=(--allow-independent-answer); fi
pids=()
for worker in $(seq 0 7); do
  shard=$(printf '%03d' "$worker")
  cache=/tmp/mechet_${control}_vllm_cache_${worker}
  mkdir -p "$cache"
  CUDA_VISIBLE_DEVICES="$worker" \
  VLLM_CACHE_ROOT="$cache/vllm" TORCHINDUCTOR_CACHE_DIR="$cache/torchinductor" TRITON_CACHE_DIR="$cache/triton" \
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  python scripts/infer_mechet.py \
    --config "$config" --data "$local_reference" \
    --output "$output_dir/generation/predictions.shard-${shard}.jsonl" \
    --mode trace --observation-mode "$observation_mode" \
    --condition-name "flower_${control}_seed17_k${samples_per_target}" \
    --model-name "$local_model_dir" --adapter "$adapter" --backend vllm \
    --prompt-source reference --intervention "$intervention" \
    --shard-count 8 --shard-index "$worker" \
    --prevalidated-data-sha256 "$reference_sha256" \
    --progress-file "$output_dir/generation/progress.shard-${shard}.json" --progress-every 5 \
    --samples-per-target "$samples_per_target" --trace-sample-batch-size "$samples_per_target" \
    --max-iterations 40 --max-new-tokens 512 --vllm-max-model-len 16384 \
    --vllm-max-num-seqs 64 --vllm-gpu-memory-utilization 0.9 \
    --temperature 0.7 --top-p 0.95 --seed 17 --resume "${extra_args[@]}" \
    >"$output_dir/generation/shard-${shard}.log" 2>&1 &
  pids+=("$!")
done

while :; do
  live=0
  for pid in "${pids[@]}"; do kill -0 "$pid" 2>/dev/null && live=$((live + 1)) || true; done
  completed=$(python - "$output_dir/generation" <<'PY'
import json, sys
from pathlib import Path
total = 0
for path in Path(sys.argv[1]).glob("progress.shard-*.json"):
    try: total += int(json.loads(path.read_text()).get("completed_targets") or 0)
    except Exception: pass
print(total)
PY
)
  echo "[meteor-progress] stage=${control}-generation completed_targets=${completed}/${expected_rows} live_shards=${live}/8 k=${samples_per_target} time=$(date --iso-8601=seconds)"
  (( live == 0 )) && break
  sleep 60
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
if (( status )); then tail -n 100 "$output_dir"/generation/shard-*.log; exit "$status"; fi

python - "$output_dir" "$expected_rows" "$samples_per_target" "$control" <<'PY'
import hashlib, json, sys
from pathlib import Path
root, expected, k, control = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
shards = sorted((root / "generation").glob("predictions.shard-*.jsonl"))
if len(shards) != 8: raise SystemExit(f"expected 8 shards, got {len(shards)}")
rows = [json.loads(line) for path in shards for line in path.open() if line.strip()]
rows.sort(key=lambda row: str(row.get("id") or ""))
if len(rows) != expected or len({row["id"] for row in rows}) != expected:
    raise SystemExit(f"expected {expected} unique predictions, got {len(rows)}")
if any(len(row.get("candidates") or []) != k for row in rows):
    raise SystemExit("incomplete candidate set")
output = root / "predictions.jsonl"
with output.open("w") as handle:
    for row in rows: handle.write(json.dumps(row, ensure_ascii=False) + "\n")
manifest = {"artifact_type": "flower_trace_control_evaluation_v1", "control": control, "n_targets": expected, "samples_per_target": k, "predictions": str(output.resolve()), "predictions_sha256": hashlib.sha256(output.read_bytes()).hexdigest()}
(root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2), flush=True)
PY

if [[ "$control" == a5_independent_answer ]]; then
  exec python scripts/evaluate_endpoint_candidates.py --reference "$reference" --predictions "$output_dir/predictions.jsonl" --output "$output_dir/evaluation.json" --expected-rows "$expected_rows" --expected-candidates "$samples_per_target"
fi
exec python scripts/evaluate_prediction_set.py --reference "$reference" --predictions "$output_dir/predictions.jsonl" --output "$output_dir/evaluation.json" --condition-name "flower_${control}_seed17_k${samples_per_target}" --expected-rows "$expected_rows" --expected-candidates "$samples_per_target"
