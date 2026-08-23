#!/usr/bin/env bash
set -Eeuo pipefail

echo >&2 "NOTICE: the primary evaluation is full FlowER 28,971; every matched 3,080-row output is an INCOMPLETE TRACE-VIEW diagnostic subset only."

repo_dir=/aaa/fionafyang/buddy1/whaleywang/MechET
shared_hf_cache=/aaa/fionafyang/buddy1/whaleywang/OpenEvolveChem/data/hf_cache
adapter=${ENDPOINT_ADAPTER:-$repo_dir/outputs/agent/sft_flower_full_endpoint_qwen3_8b_h20_run_20260814}
test_file=$repo_dir/data/flower_full_endpoint_sft_decontaminated/test.jsonl
trace_test=$repo_dir/data/textbook_tool_sft/test.jsonl
matched_file=$repo_dir/data/benchmarks/flower_full_endpoint/executable_trace_matched_test.jsonl
samples_per_target=${SAMPLES_PER_TARGET:-10}
output_dir=${ENDPOINT_EVAL_OUTPUT:-$repo_dir/outputs/eval/flower_full_endpoint_qwen3_8b_k${samples_per_target}}
revision=b968826d9c46dd6066d109eabc6255188de91218

source /root/miniconda3/etc/profile.d/conda.sh
conda activate meteor
cd "$repo_dir"

export HF_HUB_CACHE="$shared_hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$repo_dir/src:$repo_dir${PYTHONPATH:+:$PYTHONPATH}"

python - "$adapter" <<'PY'
from pathlib import Path
import json
import sys
import torch

adapter = Path(sys.argv[1])
required = [
    Path("data/flower_full_endpoint_sft_decontaminated/test.jsonl"),
    Path("data/flower_full_endpoint_sft_decontaminated/manifest.json"),
    Path("data/textbook_tool_sft/test.jsonl"),
    adapter / "adapter_config.json",
    adapter / "adapter_model.safetensors",
    adapter / "adapter_manifest.json",
]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise SystemExit(f"missing frozen endpoint evaluation artifacts: {missing}")
manifest = json.loads(Path("data/flower_full_endpoint_sft_decontaminated/manifest.json").read_text())
if int((manifest.get("test") or {}).get("rows") or 0) != 28_971:
    raise SystemExit("full FlowER test denominator is not 28,971")
if not torch.cuda.is_available() or torch.cuda.device_count() != 8:
    raise SystemExit(f"expected exactly 8 CUDA devices, got {torch.cuda.device_count()}")
print({"cuda_devices": 8, "adapter": str(adapter), "test_rows": 28_971})
PY

python scripts/build_flower_endpoint_matched_subset.py \
  --full-reference "$test_file" \
  --trace-reference "$trace_test" \
  --output "$matched_file" \
  --expected-rows 3080

echo >&2 "NOTICE: matched subset created: 3,080/28,971, incomplete trace-view diagnostic only."

mkdir -p "$output_dir/generation" "$output_dir/nll_ranking"
pids=()
for gpu in $(seq 0 7); do
  shard=$(printf '%02d' "$gpu")
  CUDA_VISIBLE_DEVICES="$gpu" python scripts/infer_mechet.py \
    --config configs/agent/sft_flower_full_endpoint_qwen3_8b_h20.yaml \
    --data "$test_file" \
    --output "$output_dir/generation/predictions.shard-${shard}.jsonl" \
    --mode direct \
    --condition-name "flower_full_endpoint_product_only_k${samples_per_target}" \
    --adapter "$adapter" \
    --shard-count 8 \
    --shard-index "$gpu" \
    --samples-per-target "$samples_per_target" \
    --max-new-tokens 256 \
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
  tail -n 100 "$output_dir"/generation/shard-*.log
  exit "$status"
fi

python - "$samples_per_target" "$output_dir" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

k = int(sys.argv[1])
root = Path(sys.argv[2])
shards = sorted((root / "generation").glob("predictions.shard-*.jsonl"))
if len(shards) != 8:
    raise SystemExit(f"expected 8 prediction shards, got {len(shards)}")
rows = []
for path in shards:
    rows.extend(json.loads(line) for line in path.open() if line.strip())
rows.sort(key=lambda row: str(row.get("id") or ""))
if len(rows) != 28_971 or len({row["id"] for row in rows}) != 28_971:
    raise SystemExit(f"expected 28,971 unique predictions, got {len(rows)}")
bad = [row["id"] for row in rows if len(row.get("candidates") or []) != k]
if bad:
    raise SystemExit(f"incomplete candidate sets: {bad[:10]}")
output = root / "predictions.jsonl"
with output.open("w", encoding="utf-8") as handle:
    for row in rows:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
manifest = {
    "artifact_type": "flower_full_endpoint_sampled_test_manifest",
    "protocol": "product-only direct endpoint generation",
    "n_targets": len(rows),
    "samples_per_target": k,
    "n_candidates": len(rows) * k,
    "candidate_semantics": "independent samples; ranked separately by frozen NLL",
    "predictions": str(output.resolve()),
    "predictions_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
}
(root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2))
PY

pids=()
for gpu in $(seq 0 7); do
  shard=$(printf '%02d' "$gpu")
  CUDA_VISIBLE_DEVICES="$gpu" python scripts/score_and_rank_predictions.py score \
    --predictions "$output_dir/predictions.jsonl" \
    --output "$output_dir/nll_ranking/nll_scores.shard-${shard}.jsonl" \
    --model Qwen/Qwen3-8B \
    --revision "$revision" \
    --adapter "$adapter" \
    --shard-count 8 \
    --shard-index "$gpu" \
    --max-length 4096 \
    >"$output_dir/nll_ranking/shard-${shard}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
if [[ "$status" -ne 0 ]]; then
  tail -n 100 "$output_dir"/nll_ranking/shard-*.log
  exit "$status"
fi

python scripts/evaluate_endpoint_candidates.py \
  --reference "$test_file" \
  --predictions "$output_dir/predictions.jsonl" \
  --ranking-dir "$output_dir/nll_ranking" \
  --output "$output_dir/evaluation.full_28971.json" \
  --expected-rows 28971 \
  --expected-candidates "$samples_per_target"

exec python scripts/evaluate_endpoint_candidates.py \
  --reference "$matched_file" \
  --predictions "$output_dir/predictions.jsonl" \
  --ranking-dir "$output_dir/nll_ranking" \
  --output "$output_dir/evaluation.matched_3080.json" \
  --expected-rows 3080 \
  --expected-candidates "$samples_per_target" \
  --allow-prediction-superset
