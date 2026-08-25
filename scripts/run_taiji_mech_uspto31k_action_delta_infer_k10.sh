#!/usr/bin/env bash
set -Eeuo pipefail

repo_dir=/aaa/fionafyang/buddy1/whaleywang/MechET
shared_hf_cache=/aaa/fionafyang/buddy1/whaleywang/OpenEvolveChem/data/hf_cache
data_version=${MECHET_ACTION_DELTA_VERSION:-v1}
case "$data_version" in
  v1)
    config=configs/agent/tool_sft_mech_uspto_31k_action_delta_qwen3_8b_a100.yaml
    reference=data/mech_uspto_31k_inverse_tool_sft_action_delta_v1/test.jsonl
    manifest=data/mech_uspto_31k_inverse_tool_sft_action_delta_v1/manifest.json
    adapter=outputs/agent/tool_sft_mech_uspto_31k_action_delta_qwen3_8b_a100_20260823
    expected_rows=1124
    default_output=outputs/eval/mech_uspto31k_action_delta_v1_qwen3_8b_k10
    ;;
  v2)
    config=configs/agent/tool_sft_mech_uspto_31k_action_delta_v2_qwen3_8b_a100.yaml
    reference=data/mech_uspto_31k_inverse_tool_sft_action_delta_v2_compiler_20260824/test.jsonl
    manifest=data/mech_uspto_31k_inverse_tool_sft_action_delta_v2_compiler_20260824/manifest.json
    adapter=outputs/agent/tool_sft_mech_uspto_31k_action_delta_v2_qwen3_8b
    expected_rows=1253
    default_output=outputs/eval/mech_uspto31k_action_delta_v2_qwen3_8b_k10
    ;;
  *)
    echo >&2 "unsupported MECHET_ACTION_DELTA_VERSION=$data_version"
    exit 2
    ;;
esac
echo >&2 "WARNING: PROGRAM-VIEW SUBSET: this evaluates ${expected_rows}/3,120 mech-USPTO-31k test reactions and is not the full endpoint benchmark."

output_dir=${MECHET_INFERENCE_OUTPUT:-$default_output}
expected_gpu=${MECHET_EXPECTED_GPU:-A100}
samples_per_target=${SAMPLES_PER_TARGET:-10}
workers_per_gpu=${MECHET_GENERATION_WORKERS_PER_GPU:-2}
gpu_count=8
full_endpoint_rows=3120
revision=b968826d9c46dd6066d109eabc6255188de91218

source /root/miniconda3/etc/profile.d/conda.sh
conda activate meteor
cd "$repo_dir"

export HF_HUB_CACHE="$shared_hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$repo_dir/src:$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ ! "$samples_per_target" =~ ^[1-9][0-9]*$ ]] || \
   [[ ! "$workers_per_gpu" =~ ^[1-9][0-9]*$ ]]; then
  echo >&2 "samples and worker counts must be positive integers"
  exit 2
fi
generation_shards=$((gpu_count * workers_per_gpu))

python - "$config" "$reference" "$manifest" "$adapter" "$expected_gpu" "$revision" "$expected_rows" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

import torch
import yaml

config_path, reference_path, manifest_path, adapter_path, expected_gpu, revision, expected_rows = sys.argv[1:]
config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
reference = Path(reference_path)
manifest_file = Path(manifest_path)
adapter = Path(adapter_path)
required = [
    reference,
    manifest_file,
    adapter / "adapter_config.json",
    adapter / "adapter_model.safetensors",
    adapter / "adapter_manifest.json",
    adapter / "data_contract.json",
]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise SystemExit(f"missing frozen action-delta inference artifacts: {missing}")
manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
if manifest.get("observation_mode") != "action_delta_v1":
    raise SystemExit("manifest is not action_delta_v1")
if manifest.get("intermediate_state_model_visible") is not False:
    raise SystemExit("manifest permits model-visible intermediate state")
split = dict(manifest["splits"]["test"])
if int(split["rows"]) != int(expected_rows):
    raise SystemExit(f"test rows {split['rows']} != {expected_rows}")
if hashlib.sha256(reference.read_bytes()).hexdigest() != str(split["sha256"]):
    raise SystemExit("test-file hash mismatch")
adapter_manifest = json.loads((adapter / "adapter_manifest.json").read_text())
if str(adapter_manifest.get("base_model_revision") or "") != revision:
    raise SystemExit("adapter/base revision mismatch")
if str(adapter_manifest.get("train_file_sha256") or "") != str(
    manifest["splits"]["train"]["sha256"]
):
    raise SystemExit("adapter train-data lineage mismatch")
contract = json.loads((adapter / "data_contract.json").read_text())
if str(contract.get("observation_mode") or config["environment"]["observation_mode"]) != "action_delta":
    raise SystemExit("adapter data contract is not action_delta")
if torch.cuda.device_count() != 8:
    raise SystemExit(f"expected 8 GPUs, got {torch.cuda.device_count()}")
names = [torch.cuda.get_device_name(index) for index in range(8)]
if not all(expected_gpu.upper() in name.upper() for name in names):
    raise SystemExit(f"expected {expected_gpu}, got {names}")
print(
    {
        "benchmark_scope": "program_view_subset_not_full_endpoint",
        "test_rows": int(split["rows"]),
        "observation_mode": manifest["observation_mode"],
        "adapter": str(adapter.resolve()),
        "gpus": names,
    }
)
PY

mkdir -p "$output_dir/generation" "$output_dir/nll_ranking"
pids=()
for worker in $(seq 0 $((generation_shards - 1))); do
  gpu=$((worker % gpu_count))
  shard=$(printf '%03d' "$worker")
  CUDA_VISIBLE_DEVICES="$gpu" python scripts/infer_mechet.py \
    --config "$config" \
    --data "$reference" \
    --output "$output_dir/generation/predictions.shard-${shard}.jsonl" \
    --mode trace \
    --observation-mode action_delta \
    --condition-name "mech_uspto31k_action_delta_${data_version}_k${samples_per_target}" \
    --adapter "$adapter" \
    --shard-count "$generation_shards" \
    --shard-index "$worker" \
    --samples-per-target "$samples_per_target" \
    --max-iterations 12 \
    --max-new-tokens 512 \
    --temperature 0.7 \
    --top-p 0.95 \
    --seed 17 \
    --resume \
    >"$output_dir/generation/shard-${shard}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
if [[ "$status" -ne 0 ]]; then
  tail -n 80 "$output_dir"/generation/shard-*.log
  exit "$status"
fi

python - "$output_dir" "$expected_rows" "$full_endpoint_rows" "$samples_per_target" "$generation_shards" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
expected_rows = int(sys.argv[2])
full_endpoint_rows = int(sys.argv[3])
k = int(sys.argv[4])
expected_shards = int(sys.argv[5])
shards = sorted((root / "generation").glob("predictions.shard-*.jsonl"))
if len(shards) != expected_shards:
    raise SystemExit(f"expected {expected_shards} shards, got {len(shards)}")
rows = []
for path in shards:
    rows.extend(json.loads(line) for line in path.open() if line.strip())
rows.sort(key=lambda row: str(row.get("id") or ""))
if len(rows) != expected_rows or len({row["id"] for row in rows}) != expected_rows:
    raise SystemExit(f"expected {expected_rows} unique predictions, got {len(rows)}")
bad = [row["id"] for row in rows if len(row.get("candidates") or []) != k]
if bad:
    raise SystemExit(f"incomplete candidate sets: {bad[:10]}")
predictions = root / "predictions.jsonl"
with predictions.open("w", encoding="utf-8") as handle:
    for row in rows:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
manifest = {
    "artifact_type": "mech_uspto31k_action_delta_sampled_test_manifest",
    "benchmark_scope": "program_view_subset_not_full_endpoint",
    "headline_eligible_as_full_endpoint_benchmark": False,
    "n_targets": len(rows),
    "full_endpoint_test_denominator": full_endpoint_rows,
    "samples_per_target": k,
    "candidate_metric_semantics": "generation_order_pass_at_k; environment-ranked selected candidate reported separately",
    "predictions": str(predictions.resolve()),
    "predictions_sha256": hashlib.sha256(predictions.read_bytes()).hexdigest(),
}
(root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2))
PY

score_pids=()
for worker in $(seq 0 $((gpu_count - 1))); do
  shard=$(printf '%03d' "$worker")
  CUDA_VISIBLE_DEVICES="$worker" python scripts/score_and_rank_predictions.py score \
    --predictions "$output_dir/predictions.jsonl" \
    --output "$output_dir/nll_ranking/nll_scores.shard-${shard}.jsonl" \
    --model Qwen/Qwen3-8B \
    --revision "$revision" \
    --adapter "$adapter" \
    --shard-count "$gpu_count" \
    --shard-index "$worker" \
    --max-length 24576 \
    >"$output_dir/nll_ranking/shard-${shard}.log" 2>&1 &
  score_pids+=("$!")
done
status=0
for pid in "${score_pids[@]}"; do
  wait "$pid" || status=1
done
if [[ "$status" -ne 0 ]]; then
  tail -n 80 "$output_dir"/nll_ranking/shard-*.log
  exit "$status"
fi

python scripts/score_and_rank_predictions.py aggregate \
  --reference "$reference" \
  --predictions "$output_dir/predictions.jsonl" \
  --ranking-dir "$output_dir/nll_ranking" \
  --output "$output_dir/nll_evaluation.json"

exec python scripts/evaluate_prediction_set.py \
  --reference "$reference" \
  --predictions "$output_dir/predictions.jsonl" \
  --output "$output_dir/evaluation.json" \
  --condition-name "mech_uspto31k_action_delta_${data_version}_k${samples_per_target}" \
  --expected-rows "$expected_rows" \
  --expected-candidates "$samples_per_target"
