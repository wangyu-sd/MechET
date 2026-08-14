#!/usr/bin/env bash
set -Eeuo pipefail

repo_dir=/aaa/fionafyang/buddy1/whaleywang/MechET
shared_hf_cache=/aaa/fionafyang/buddy1/whaleywang/OpenEvolveChem/data/hf_cache
adapter="$repo_dir/outputs/agent/tool_sft_mixed_inverse_qwen3_8b"
samples_per_target=${SAMPLES_PER_TARGET:-4}
output_dir="$repo_dir/outputs/eval/mech_uspto31k_qwen3_8b_k${samples_per_target}"

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
import torch

required = [
    Path("data/mixed_inverse_tool_sft/test.jsonl"),
    Path("outputs/agent/tool_sft_mixed_inverse_qwen3_8b/adapter_manifest.json"),
]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise SystemExit(f"missing frozen evaluation artifacts: {missing}")
if not torch.cuda.is_available() or torch.cuda.device_count() != 8:
    raise SystemExit(f"expected exactly 8 CUDA devices, got {torch.cuda.device_count()}")
print({"cuda_devices": torch.cuda.device_count(), "torch": torch.__version__})
PY

mkdir -p "$output_dir"
pids=()
for gpu in $(seq 0 7); do
  shard=$(printf '%02d' "$gpu")
  CUDA_VISIBLE_DEVICES="$gpu" python scripts/infer_mechet.py \
    --config configs/agent/tool_sft_mixed_inverse_qwen3_8b.yaml \
    --data data/mixed_inverse_tool_sft/test.jsonl \
    --output "$output_dir/predictions.shard-${shard}.jsonl" \
    --mode trace \
    --condition-name "mech_uspto31k_trace_k${samples_per_target}" \
    --adapter "$adapter" \
    --source mech_uspto_31k \
    --shard-count 8 \
    --shard-index "$gpu" \
    --samples-per-target "$samples_per_target" \
    --max-iterations 12 \
    --max-new-tokens 512 \
    --temperature 0.7 \
    --top-p 0.95 \
    --seed 17 \
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

python - <<'PY'
import hashlib
import json
from pathlib import Path

import os

samples_per_target = int(os.environ.get("SAMPLES_PER_TARGET", "4"))
root = Path(f"outputs/eval/mech_uspto31k_qwen3_8b_k{samples_per_target}")
shards = sorted(root.glob("predictions.shard-*.jsonl"))
rows = []
for path in shards:
    rows.extend(json.loads(line) for line in path.open() if line.strip())
rows.sort(key=lambda row: str(row.get("id") or ""))
if len(rows) != 1124 or len({row["id"] for row in rows}) != 1124:
    raise SystemExit(f"expected 1124 unique predictions, got {len(rows)}")
output = root / "predictions.jsonl"
with output.open("w", encoding="utf-8") as handle:
    for row in rows:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
digest = hashlib.sha256(output.read_bytes()).hexdigest()
manifest = {
    "artifact_type": "mech_uspto31k_sampled_test_manifest",
    "n_targets": len(rows),
    "samples_per_target": samples_per_target,
    "n_candidates": len(rows) * samples_per_target,
    "temperature": 0.7,
    "top_p": 0.95,
    "max_iterations": 12,
    "max_new_tokens": 512,
    "predictions": str(output),
    "predictions_sha256": digest,
    "shards": [str(path) for path in shards],
}
(root / "manifest.json").write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)
print(json.dumps(manifest, indent=2, ensure_ascii=False))
PY
