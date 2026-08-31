#!/usr/bin/env bash
set -Eeuo pipefail

repo_dir=/aaa/fionafyang/buddy1/whaleywang/MechET
shared_hf_cache=/aaa/fionafyang/buddy1/whaleywang/OpenEvolveChem/data/hf_cache
baseline=${MECHET_BASELINE:?set MECHET_BASELINE to outcome_only, free_cot, state_cot, net_edit, proof, or open_flow}
expected_gpu=${MECHET_EXPECTED_GPU:-A100}
samples_per_target=${SAMPLES_PER_TARGET:-10}
inference_backend=${MECHET_INFERENCE_BACKEND:-vllm}
revision=b968826d9c46dd6066d109eabc6255188de91218

case "$baseline" in
  outcome_only)
    config=configs/iclr/full_outcome_only_sft.yaml
    adapter=outputs/iclr/full_outcome_only_seed17
    test_file=data/iclr_full_v4/outcome_only/test.jsonl
    expected_rows=28971
    # Frozen on the validation split: max reference answer is 830 tokens.
    max_new_tokens=1024
    nll_max_length=4096
    direct_sample_batch_size=${MECHET_DIRECT_SAMPLE_BATCH_SIZE:-10}
    preferred_generation_workers_per_gpu=2
    ;;
  free_cot)
    config=configs/iclr/a1_free_cot_sft.yaml
    adapter=outputs/iclr/a1_free_cot_seed17
    test_file=data/iclr_program_controls_v1/free_cot/test.jsonl
    dataset_manifest=data/iclr_program_controls_v1/manifest.json
    manifest_task=free_cot
    expected_rows=28967
    # The frozen validation audit has a 2,526-token maximum complete chat
    # sequence.  A 4,096-token completion cap leaves headroom for stochastic
    # reasoning without inheriting the prohibitively long State-CoT budget.
    max_new_tokens=4096
    nll_max_length=6144
    direct_sample_batch_size=${MECHET_DIRECT_SAMPLE_BATCH_SIZE:-2}
    preferred_generation_workers_per_gpu=1
    ;;
  state_cot)
    config=configs/iclr/full_state_cot_sft.yaml
    adapter=outputs/iclr/full_state_cot_seed17
    test_file=data/iclr_full_v4/state_cot/test.jsonl
    expected_rows=28971
    # Frozen on the validation split: max reference answer is 13,322 tokens.
    max_new_tokens=14336
    nll_max_length=20480
    direct_sample_batch_size=${MECHET_DIRECT_SAMPLE_BATCH_SIZE:-2}
    preferred_generation_workers_per_gpu=2
    ;;
  net_edit)
    config=configs/iclr/full_net_edit_sft.yaml
    adapter=outputs/iclr/full_net_edit_seed17
    test_file=data/iclr_full_v4/net_edit/test.jsonl
    expected_rows=28967
    # Frozen on the validation split: max reference answer is 1,609 tokens.
    max_new_tokens=2048
    nll_max_length=6144
    direct_sample_batch_size=${MECHET_DIRECT_SAMPLE_BATCH_SIZE:-8}
    preferred_generation_workers_per_gpu=2
    ;;
  proof)
    config=configs/iclr/full_proof_sft.yaml
    adapter=outputs/iclr/full_proof_seed17
    test_file=data/iclr_full_v4/proof/test.jsonl
    expected_rows=28967
    # Frozen on the validation split: max reference answer is 1,935 tokens.
    max_new_tokens=2048
    nll_max_length=6144
    direct_sample_batch_size=${MECHET_DIRECT_SAMPLE_BATCH_SIZE:-8}
    preferred_generation_workers_per_gpu=2
    ;;
  open_flow)
    config=configs/iclr/a4_open_flow_sft.yaml
    adapter=outputs/iclr/a4_open_flow_seed17
    test_file=data/iclr_program_controls_v1/open_flow/test.jsonl
    dataset_manifest=data/iclr_program_controls_v1/manifest.json
    manifest_task=open_flow
    expected_rows=28967
    # Frozen validation maximum is 2,032 total tokens; this leaves a
    # conservative completion margin for the short inference preamble.
    max_new_tokens=2304
    nll_max_length=4096
    direct_sample_batch_size=${MECHET_DIRECT_SAMPLE_BATCH_SIZE:-8}
    preferred_generation_workers_per_gpu=2
    ;;
  *)
    echo >&2 "unsupported MECHET_BASELINE=$baseline"
    exit 2
    ;;
esac

dataset_manifest=${dataset_manifest:-data/iclr_full_v4/manifest.json}
manifest_task=${manifest_task:-$baseline}

output_dir=${MECHET_INFERENCE_OUTPUT:-outputs/eval/iclr_full/${baseline}_seed17_k${samples_per_target}}
gpu_count=8

source /root/miniconda3/etc/profile.d/conda.sh
conda activate meteor
cd "$repo_dir"

# Loading one BF16 Qwen3-8B copy consumes roughly 16 GiB before KV-cache
# growth.  The Qingyuan A100 pool contains 40-GiB cards, where the historical
# three-process default deterministically OOMs while loading the third copy.
# Candidate-level batching supplies most throughput. Two model copies fit on
# the observed 40-GiB A100s only with a small candidate microbatch; size two
# also limits head-of-line blocking when one stochastic candidate degenerates
# into a repetitive sequence. The historical three-copy launch OOMed.
gpu_memory_mib=$(nvidia-smi --id=0 --query-gpu=memory.total --format=csv,noheader,nounits | tr -d '[:space:]')
if [[ ! "$gpu_memory_mib" =~ ^[0-9]+$ ]]; then
  echo >&2 "could not determine GPU memory from nvidia-smi: $gpu_memory_mib"
  exit 2
fi
generation_workers_per_gpu=${MECHET_GENERATION_WORKERS_PER_GPU:-$preferred_generation_workers_per_gpu}
if [[ "$inference_backend" == "vllm" ]]; then
  # vLLM owns continuous batching and reserves a KV cache. Run one engine per
  # GPU instead of competing replicas on the same device.
  generation_workers_per_gpu=${MECHET_GENERATION_WORKERS_PER_GPU:-1}
fi
if (( gpu_memory_mib < 70000 && generation_workers_per_gpu > 1 && direct_sample_batch_size > 2 )) && \
   [[ -z ${MECHET_DIRECT_SAMPLE_BATCH_SIZE:-} ]]; then
  direct_sample_batch_size=2
fi
if [[ -n ${MECHET_NLL_WORKERS_PER_GPU:-} ]]; then
  nll_workers_per_gpu=$MECHET_NLL_WORKERS_PER_GPU
elif (( gpu_memory_mib < 70000 )); then
  nll_workers_per_gpu=1
else
  nll_workers_per_gpu=2
fi
if [[ ! "$generation_workers_per_gpu" =~ ^[1-9][0-9]*$ ]] || \
   [[ ! "$direct_sample_batch_size" =~ ^[1-9][0-9]*$ ]] || \
   [[ ! "$nll_workers_per_gpu" =~ ^[1-9][0-9]*$ ]]; then
  echo >&2 "worker counts must be positive integers"
  exit 2
fi
if [[ "$inference_backend" != "vllm" && "$inference_backend" != "transformers" ]]; then
  echo >&2 "MECHET_INFERENCE_BACKEND must be vllm or transformers"
  exit 2
fi
generation_shards=$((gpu_count * generation_workers_per_gpu))
nll_shards=$((gpu_count * nll_workers_per_gpu))
echo "inference_concurrency backend=$inference_backend gpu_memory_mib=$gpu_memory_mib generation_workers_per_gpu=$generation_workers_per_gpu direct_sample_batch_size=$direct_sample_batch_size nll_workers_per_gpu=$nll_workers_per_gpu"

export HF_HUB_CACHE="$shared_hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$repo_dir/src:$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python - "$baseline" "$manifest_task" "$dataset_manifest" "$config" "$adapter" "$test_file" "$expected_rows" "$expected_gpu" "$revision" "$samples_per_target" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

import torch
import yaml

baseline, manifest_task, manifest_path, config_path, adapter_path, test_path, expected_rows, expected_gpu, revision, samples_per_target = sys.argv[1:]
config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
adapter = Path(adapter_path)
test_file = Path(test_path)
required = [
    test_file,
    Path(manifest_path),
    adapter / "adapter_config.json",
    adapter / "adapter_model.safetensors",
    adapter / "adapter_manifest.json",
    adapter / "data_contract.json",
]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise SystemExit(f"missing frozen inference artifacts: {missing}")
manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
actual_rows = int(manifest["tasks"][manifest_task]["test"]["rows"])
if actual_rows != int(expected_rows):
    raise SystemExit(f"{baseline} test rows {actual_rows} != {expected_rows}")
expected_hash = str(manifest["tasks"][manifest_task]["test"]["sha256"])
actual_hash = hashlib.sha256(test_file.read_bytes()).hexdigest()
if actual_hash != expected_hash:
    raise SystemExit(f"{baseline} test hash mismatch")
adapter_manifest = json.loads((adapter / "adapter_manifest.json").read_text())
if str(adapter_manifest.get("base_model_revision") or "") != revision:
    raise SystemExit("adapter/base revision mismatch")
contract = json.loads((adapter / "data_contract.json").read_text())
if str(contract.get("train_file_sha256") or "") != str(
    manifest["tasks"][manifest_task]["train"]["sha256"]
):
    raise SystemExit("adapter train-data lineage mismatch")
if torch.cuda.device_count() != 8:
    raise SystemExit(f"expected 8 GPUs, got {torch.cuda.device_count()}")
names = [torch.cuda.get_device_name(index) for index in range(8)]
if not all(expected_gpu.upper() in name.upper() for name in names):
    raise SystemExit(f"expected {expected_gpu}, got {names}")
print({
    "baseline": baseline,
    "test_rows": actual_rows,
    "samples_per_target": int(samples_per_target),
    "adapter": str(adapter.resolve()),
    "gpus": names,
})
PY

mkdir -p "$output_dir/generation" "$output_dir/nll_ranking"

pids=()
for worker in $(seq 0 $((generation_shards - 1))); do
  gpu=$((worker % gpu_count))
  shard=$(printf '%03d' "$worker")
  CUDA_VISIBLE_DEVICES="$gpu" python scripts/infer_mechet.py \
    --config "$config" \
    --data "$test_file" \
    --output "$output_dir/generation/predictions.shard-${shard}.jsonl" \
    --mode direct \
    --condition-name "iclr_full_${baseline}_seed17_k${samples_per_target}" \
    --adapter "$adapter" \
    --backend "$inference_backend" \
    --shard-count "$generation_shards" \
    --shard-index "$worker" \
    --samples-per-target "$samples_per_target" \
    --direct-sample-batch-size "$direct_sample_batch_size" \
    --vllm-max-model-len "$nll_max_length" \
    --vllm-max-num-seqs "${MECHET_VLLM_MAX_NUM_SEQS:-64}" \
    --vllm-gpu-memory-utilization "${MECHET_VLLM_GPU_MEMORY_UTILIZATION:-0.9}" \
    --max-new-tokens "$max_new_tokens" \
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
  completed=$(find "$output_dir/generation" -name 'predictions.shard-*.jsonl' -type f -print0 2>/dev/null | xargs -0 -r wc -l | awk '/ total$/{print $1; found=1} END{if(!found) print 0}')
  echo "[meteor-progress] stage=${baseline}-generation completed_targets=${completed}/${expected_rows} live_shards=${live}/${generation_shards} k=${samples_per_target} time=$(date --iso-8601=seconds)"
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

python - "$baseline" "$samples_per_target" "$expected_rows" "$generation_shards" "$direct_sample_batch_size" "$output_dir" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

baseline = sys.argv[1]
k = int(sys.argv[2])
expected = int(sys.argv[3])
expected_shards = int(sys.argv[4])
direct_sample_batch_size = int(sys.argv[5])
root = Path(sys.argv[6])
shards = sorted((root / "generation").glob("predictions.shard-*.jsonl"))
if len(shards) != expected_shards:
    raise SystemExit(f"expected {expected_shards} prediction shards, got {len(shards)}")
rows = []
for path in shards:
    rows.extend(json.loads(line) for line in path.open() if line.strip())
rows.sort(key=lambda row: str(row.get("id") or ""))
if len(rows) != expected or len({row["id"] for row in rows}) != expected:
    raise SystemExit(f"expected {expected} unique predictions, got {len(rows)}")
bad = [row["id"] for row in rows if len(row.get("candidates") or []) != k]
if bad:
    raise SystemExit(f"incomplete candidate sets: {bad[:10]}")
output = root / "predictions.jsonl"
with output.open("w", encoding="utf-8") as handle:
    for row in rows:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
manifest = {
    "artifact_type": "iclr_full_baseline_sampled_test_manifest",
    "paper_status": "full-test evaluation; compare methods on the shared 28,967-ID universe",
    "baseline": baseline,
    "n_targets": len(rows),
    "samples_per_target": k,
    "n_candidates": len(rows) * k,
    "candidate_semantics": "stochastic samples; generation-order Success@K and separately frozen ranking",
    "direct_sample_batch_size": direct_sample_batch_size,
    "predictions": str(output.resolve()),
    "predictions_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
}
(root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2))
PY

if [[ "$samples_per_target" -eq 1 ]]; then
  echo "[meteor-progress] stage=${baseline}-evaluation reason=k1-needs-no-ranking time=$(date --iso-8601=seconds)"
  if [[ "$baseline" == proof ]]; then
    exec python scripts/evaluate_proof_candidates.py \
      --reference "$test_file" \
      --predictions "$output_dir/predictions.jsonl" \
      --output "$output_dir/evaluation.json" \
      --expected-rows "$expected_rows" \
      --expected-candidates 1
  fi
  if [[ "$baseline" == open_flow ]]; then
    exec python scripts/evaluate_open_flow_candidates.py \
      --reference "$test_file" \
      --predictions "$output_dir/predictions.jsonl" \
      --output "$output_dir/evaluation.json" \
      --expected-rows "$expected_rows" \
      --expected-candidates 1
  fi
  exec python scripts/evaluate_endpoint_candidates.py \
    --reference "$test_file" \
    --predictions "$output_dir/predictions.jsonl" \
    --output "$output_dir/evaluation.json" \
    --expected-rows "$expected_rows" \
    --expected-candidates 1
fi

pids=()
for worker in $(seq 0 $((nll_shards - 1))); do
  gpu=$((worker % gpu_count))
  shard=$(printf '%03d' "$worker")
  CUDA_VISIBLE_DEVICES="$gpu" python scripts/score_and_rank_predictions.py score \
    --predictions "$output_dir/predictions.jsonl" \
    --output "$output_dir/nll_ranking/nll_scores.shard-${shard}.jsonl" \
    --model Qwen/Qwen3-8B \
    --revision "$revision" \
    --adapter "$adapter" \
    --shard-count "$nll_shards" \
    --shard-index "$worker" \
    --max-length "$nll_max_length" \
    >"$output_dir/nll_ranking/shard-${shard}.log" 2>&1 &
  pids+=("$!")
done
while :; do
  live=0
  for pid in "${pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      live=$((live + 1))
    fi
  done
  completed=$(find "$output_dir/nll_ranking" -name 'nll_scores.shard-*.jsonl' -type f -print0 2>/dev/null | xargs -0 -r wc -l | awk '/ total$/{print $1; found=1} END{if(!found) print 0}')
  echo "[meteor-progress] stage=${baseline}-nll completed_targets=${completed}/${expected_rows} live_shards=${live}/${nll_shards} time=$(date --iso-8601=seconds)"
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
  tail -n 80 "$output_dir"/nll_ranking/shard-*.log
  exit "$status"
fi

if [[ "$baseline" == proof ]]; then
  exec python scripts/evaluate_proof_candidates.py \
    --reference "$test_file" \
    --predictions "$output_dir/predictions.jsonl" \
    --ranking-dir "$output_dir/nll_ranking" \
    --output "$output_dir/evaluation.json" \
    --expected-rows "$expected_rows" \
    --expected-candidates "$samples_per_target"
fi

if [[ "$baseline" == open_flow ]]; then
  exec python scripts/evaluate_open_flow_candidates.py \
    --reference "$test_file" \
    --predictions "$output_dir/predictions.jsonl" \
    --ranking-dir "$output_dir/nll_ranking" \
    --output "$output_dir/evaluation.json" \
    --expected-rows "$expected_rows" \
    --expected-candidates "$samples_per_target"
fi

exec python scripts/evaluate_endpoint_candidates.py \
  --reference "$test_file" \
  --predictions "$output_dir/predictions.jsonl" \
  --ranking-dir "$output_dir/nll_ranking" \
  --output "$output_dir/evaluation.json" \
  --expected-rows "$expected_rows" \
  --expected-candidates "$samples_per_target"
