#!/usr/bin/env bash
set -Eeuo pipefail

repo_dir=/aaa/fionafyang/buddy1/whaleywang/MechET
shared_hf_cache=/aaa/fionafyang/buddy1/whaleywang/OpenEvolveChem/data/hf_cache
adapter="$repo_dir/outputs/agent/tool_sft_mixed_inverse_qwen3_8b"
test_file="$repo_dir/data/benchmarks/uspto50k/test.inverse_eval.jsonl"
samples_per_target=${SAMPLES_PER_TARGET:-10}
gpu_count=${EVAL_GPU_COUNT:-8}
if ! [[ "$gpu_count" =~ ^[1-8]$ ]]; then
  echo "EVAL_GPU_COUNT must be an integer from 1 to 8, got: $gpu_count" >&2
  exit 2
fi
output_dir="$repo_dir/outputs/eval/uspto50k_standard_mixed_qwen3_8b_k${samples_per_target}"
ranking_dir="$output_dir/nll_ranking_v1"
revision=b968826d9c46dd6066d109eabc6255188de91218

source /root/miniconda3/etc/profile.d/conda.sh
conda activate meteor
cd "$repo_dir"

export HF_HUB_CACHE="$shared_hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$repo_dir/src:$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
export EVAL_GPU_COUNT="$gpu_count"

python - <<'PY'
from pathlib import Path
import hashlib
import json
import os
import torch

test_file = Path("data/benchmarks/uspto50k/test.inverse_eval.jsonl")
manifest_file = Path("data/benchmarks/uspto50k/manifest.json")
adapter_manifest = Path("outputs/agent/tool_sft_mixed_inverse_qwen3_8b/adapter_manifest.json")
required = [test_file, manifest_file, adapter_manifest]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise SystemExit(f"missing frozen evaluation artifacts: {missing}")
manifest = json.loads(manifest_file.read_text())
test = next(item for item in manifest["splits"] if item["split"] == "test")
digest = hashlib.sha256(test_file.read_bytes()).hexdigest()
if digest != test["inverse_eval_sha256"]:
    raise SystemExit(f"USPTO-50K test hash mismatch: {digest}")
if test["rows"] != 5007 or not test["mapped_role_rows"] == 5007:
    raise SystemExit(f"unexpected frozen test manifest: {test}")
expected_devices = int(os.environ["EVAL_GPU_COUNT"])
if not torch.cuda.is_available() or torch.cuda.device_count() != expected_devices:
    raise SystemExit(
        f"expected exactly {expected_devices} CUDA devices, "
        f"got {torch.cuda.device_count()}"
    )
print({"cuda_devices": expected_devices, "torch": torch.__version__, "test_rows": test["rows"]})
PY

mkdir -p "$output_dir"
pids=()
for gpu in $(seq 0 $((gpu_count - 1))); do
  (
    shard_index="$gpu"
    while [[ "$shard_index" -lt 8 ]]; do
      shard=$(printf '%02d' "$shard_index")
      CUDA_VISIBLE_DEVICES="$gpu" python scripts/infer_mechet.py \
        --config configs/agent/tool_sft_mixed_inverse_qwen3_8b.yaml \
        --data "$test_file" \
        --output "$output_dir/predictions.shard-${shard}.jsonl" \
        --mode trace \
        --condition-name "uspto50k_standard_product_only_trace_k${samples_per_target}" \
        --adapter "$adapter" \
        --shard-count 8 \
        --shard-index "$shard_index" \
        --samples-per-target "$samples_per_target" \
        --max-iterations 12 \
        --max-new-tokens 512 \
        --temperature 0.7 \
        --top-p 0.95 \
        --seed 17 \
        --resume \
        >"$output_dir/shard-${shard}.log" 2>&1
      shard_index=$((shard_index + gpu_count))
    done
  ) &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
if [[ "$status" -ne 0 ]]; then
  tail -n 80 "$output_dir"/shard-*.log
  exit "$status"
fi

python - "$samples_per_target" "$output_dir" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

samples_per_target = int(sys.argv[1])
root = Path(sys.argv[2])
shards = sorted(root.glob("predictions.shard-*.jsonl"))
if len(shards) != 8:
    raise SystemExit(f"expected 8 prediction shards, got {len(shards)}")
rows = []
for path in shards:
    rows.extend(json.loads(line) for line in path.open() if line.strip())
rows.sort(key=lambda row: str(row.get("id") or ""))
if len(rows) != 5007 or len({row["id"] for row in rows}) != 5007:
    raise SystemExit(f"expected 5007 unique predictions, got {len(rows)}")
bad = [row["id"] for row in rows if len(row.get("candidates") or []) != samples_per_target]
if bad:
    raise SystemExit(f"rows with incomplete candidate sets: {bad[:10]}")
output = root / "predictions.jsonl"
with output.open("w", encoding="utf-8") as handle:
    for row in rows:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
manifest = {
    "artifact_type": "uspto50k_standard_sampled_test_manifest",
    "protocol": "product-only, class-unknown, environment-owned trace",
    "n_targets": len(rows),
    "samples_per_target": samples_per_target,
    "n_candidates": len(rows) * samples_per_target,
    "temperature": 0.7,
    "top_p": 0.95,
    "max_iterations": 12,
    "max_new_tokens": 512,
    "predictions": str(output.resolve()),
    "predictions_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    "shards": [str(path.resolve()) for path in shards],
}
(root / "manifest.json").write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)
print(json.dumps(manifest, indent=2, ensure_ascii=False))
PY

python scripts/evaluate_prediction_set.py \
  --reference "$test_file" \
  --predictions "$output_dir/predictions.jsonl" \
  --output "$output_dir/evaluation.json" \
  --condition-name "uspto50k_standard_product_only_trace_k${samples_per_target}" \
  --expected-rows 5007 \
  --expected-candidates "$samples_per_target"

mkdir -p "$ranking_dir"
pids=()
for gpu in $(seq 0 $((gpu_count - 1))); do
  (
    shard_index="$gpu"
    while [[ "$shard_index" -lt 8 ]]; do
      shard=$(printf '%02d' "$shard_index")
      CUDA_VISIBLE_DEVICES="$gpu" python scripts/score_and_rank_predictions.py score \
        --predictions "$output_dir/predictions.jsonl" \
        --output "$ranking_dir/nll_scores.shard-${shard}.jsonl" \
        --model Qwen/Qwen3-8B \
        --revision "$revision" \
        --adapter "$adapter" \
        --shard-count 8 \
        --shard-index "$shard_index" \
        --max-length 24576 \
        >"$ranking_dir/shard-${shard}.log" 2>&1
      shard_index=$((shard_index + gpu_count))
    done
  ) &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
if [[ "$status" -ne 0 ]]; then
  tail -n 100 "$ranking_dir"/shard-*.log
  exit "$status"
fi

exec python scripts/score_and_rank_predictions.py aggregate \
  --reference "$test_file" \
  --predictions "$output_dir/predictions.jsonl" \
  --ranking-dir "$ranking_dir" \
  --output "$ranking_dir/evaluation.json"
