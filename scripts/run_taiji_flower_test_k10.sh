#!/usr/bin/env bash
set -Eeuo pipefail

echo >&2 "WARNING: INCOMPLETE TRACE-VIEW SUBSET: this legacy job evaluates 3,080/28,971 FlowER test reactions; never report it as the full benchmark."

repo_dir=/aaa/fionafyang/buddy1/whaleywang/MechET
shared_hf_cache=/aaa/fionafyang/buddy1/whaleywang/OpenEvolveChem/data/hf_cache
adapter="$repo_dir/outputs/agent/tool_sft_flower_inverse_qwen3_8b_h20_run_20260812"
test_file="$repo_dir/data/flower_inverse_tool_sft/test.jsonl"
samples_per_target=${SAMPLES_PER_TARGET:-10}
output_dir="$repo_dir/outputs/eval/flower_qwen3_8b_3epoch_k${samples_per_target}"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate meteor
cd "$repo_dir"

export HF_HUB_CACHE="$shared_hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$repo_dir/src:$repo_dir${PYTHONPATH:+:$PYTHONPATH}"

python - <<'PY'
from pathlib import Path
import json
import torch

required = [
    Path("data/flower_inverse_tool_sft/test.jsonl"),
    Path("outputs/agent/tool_sft_flower_inverse_qwen3_8b_h20_run_20260812/adapter_manifest.json"),
    Path("outputs/agent/tool_sft_flower_inverse_qwen3_8b_h20_run_20260812/checkpoint-2592/trainer_state.json"),
]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise SystemExit(f"missing frozen FlowER evaluation artifacts: {missing}")
state = json.loads(required[-1].read_text())
if float(state.get("epoch", 0)) != 3.0:
    raise SystemExit(f"expected completed epoch 3 checkpoint, got {state.get('epoch')}")
if not torch.cuda.is_available() or torch.cuda.device_count() != 8:
    raise SystemExit(f"expected exactly 8 CUDA devices, got {torch.cuda.device_count()}")
print({"cuda_devices": torch.cuda.device_count(), "torch": torch.__version__, "epoch": state["epoch"]})
PY

mkdir -p "$output_dir"
pids=()
for gpu in $(seq 0 7); do
  shard=$(printf '%02d' "$gpu")
  CUDA_VISIBLE_DEVICES="$gpu" python scripts/infer_mechet.py \
    --config configs/agent/tool_sft_flower_inverse_qwen3_8b_h20.yaml \
    --data "$test_file" \
    --output "$output_dir/predictions.shard-${shard}.jsonl" \
    --mode trace \
    --condition-name "flower_trace_3epoch_k${samples_per_target}" \
    --adapter "$adapter" \
    --shard-count 8 \
    --shard-index "$gpu" \
    --samples-per-target "$samples_per_target" \
    --max-iterations 24 \
    --max-new-tokens 512 \
    --temperature 0.7 \
    --top-p 0.95 \
    --seed 17 \
    --resume \
    >"$output_dir/shard-${shard}.log" 2>&1 &
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
if len(rows) != 3080 or len({row["id"] for row in rows}) != 3080:
    raise SystemExit(f"expected 3080 unique predictions, got {len(rows)}")
bad = [row["id"] for row in rows if len(row.get("candidates") or []) != samples_per_target]
if bad:
    raise SystemExit(f"rows with incomplete candidate sets: {bad[:10]}")
output = root / "predictions.jsonl"
with output.open("w", encoding="utf-8") as handle:
    for row in rows:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
digest = hashlib.sha256(output.read_bytes()).hexdigest()
manifest = {
    "artifact_type": "flower_3epoch_sampled_test_manifest",
    "benchmark_scope": "incomplete_trace_view_subset_not_headline",
    "full_test_denominator": 28971,
    "headline_eligible": False,
    "n_targets": len(rows),
    "samples_per_target": samples_per_target,
    "n_candidates": len(rows) * samples_per_target,
    "temperature": 0.7,
    "top_p": 0.95,
    "max_iterations": 24,
    "max_new_tokens": 512,
    "predictions": str(output.resolve()),
    "predictions_sha256": digest,
    "shards": [str(path.resolve()) for path in shards],
}
(root / "manifest.json").write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)
print(json.dumps(manifest, indent=2, ensure_ascii=False))
PY

exec python scripts/evaluate_prediction_set.py \
  --reference "$test_file" \
  --predictions "$output_dir/predictions.jsonl" \
  --output "$output_dir/evaluation.json" \
  --condition-name "flower_trace_3epoch_k${samples_per_target}" \
  --expected-rows 3080 \
  --expected-candidates "$samples_per_target"
