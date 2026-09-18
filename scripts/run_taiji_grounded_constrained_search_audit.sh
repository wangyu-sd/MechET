#!/usr/bin/env bash
set -Eeuo pipefail

shared_repo=/aaa/fionafyang/buddy1/whaleywang/MechET
runtime_repo=${MECHET_SEARCH_RUNTIME_DIR:?set MECHET_SEARCH_RUNTIME_DIR}
output_dir=$shared_repo/outputs/eval/grounded_constrained_search_valid64_ckpt8037_20260911
data=$shared_repo/data/flower_in_place_grounded_flow_v1/valid.jsonl
gold_data=$shared_repo/data/flower_inverse_tool_sft_action_delta_v1/valid.jsonl
adapter=$shared_repo/outputs/agent/in_place_grounded_flow_qwen3_8b_a100_seed17_20260911/checkpoint-8037
base_model=/aaa/fionafyang/buddy1/whaleywang/models/orbit_qwen_base
bitsandbytes_wheel=$shared_repo/artifacts/wheels/bitsandbytes-0.49.2-py3-none-manylinux_2_24_x86_64.whl
rdkit_wheel=$shared_repo/artifacts/wheels/rdkit-2026.3.4-cp311-cp311-manylinux_2_28_x86_64.whl

source /root/miniconda3/etc/profile.d/conda.sh
conda activate meteor
cd "$runtime_repo"

echo "[search-audit] runtime=$runtime_repo output=$output_dir"
test -f "$runtime_repo/scripts/eval_grounded_constrained_search.py"
test -f "$data"
test -f "$gold_data"
test -f "$adapter/adapter_model.safetensors"
echo "54b771f06e1a3c73af5c7f16ccf0fc23a846052813d4b008d10cb6e017dd1c8c  $bitsandbytes_wheel" | sha256sum --check --strict
echo "a41dde42ecb24e7d93d62b89e8e5d267b02bbac764f965f3968296c99234c685  $rdkit_wheel" | sha256sum --check --strict
echo "8446e15d8c9933c715f17bebc1055925ff379b0ca1119d9c01614f6fa3ac3f21  $adapter/adapter_model.safetensors" | sha256sum --check --strict
echo "b4b800858b9cb5d6da893401dd13b36080624455e154a8494d66a8f27bc44324  $data" | sha256sum --check --strict
echo "7303a6018850db61594af5854df936778c21a6668e61151f95e2c74c32d22d2a  $gold_data" | sha256sum --check --strict

runtime_target=$(mktemp -d /tmp/mechet_search_runtime.XXXXXX)
python -m pip install --quiet --no-deps --target "$runtime_target" \
  "$bitsandbytes_wheel" "$rdkit_wheel"
export PYTHONPATH=$runtime_target:$runtime_repo/src:$runtime_repo
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

python - <<'PY'
import rdkit, torch
if torch.cuda.device_count() != 8:
    raise SystemExit(f"expected 8 GPUs, got {torch.cuda.device_count()}")
names = [torch.cuda.get_device_name(i) for i in range(8)]
if not all("A100" in name.upper() for name in names):
    raise SystemExit(f"expected A100 GPUs, got {names}")
print({"gpus": names, "rdkit": rdkit.__version__}, flush=True)
PY

mkdir -p "$output_dir"
pids=()
for rank in $(seq 0 7); do
  echo "[search-audit] launching shard=$rank gpu=$rank"
  CUDA_VISIBLE_DEVICES=$rank python -u scripts/eval_grounded_constrained_search.py \
    --data "$data" \
    --gold-source-data "$gold_data" \
    --adapter "$adapter" \
    --base-model "$base_model" \
    --output "$output_dir/predictions.shard-$(printf '%03d' "$rank").jsonl" \
    --summary "$output_dir/summary.shard-$(printf '%03d' "$rank").json" \
    --limit 64 \
    --shard-count 8 \
    --shard-index "$rank" \
    --selection-seed 17 \
    --seed 17 \
    --independent-candidates 4 \
    --beam-width 2 \
    --branch-factor 2 \
    --max-depth 12 \
    --max-responses 48 \
    --max-new-tokens 384 \
    --temperature 0.7 \
    --top-p 0.95 \
    --microbatch-size 2 &
  pids+=("$!")
done

failed=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "[search-audit] shard=$index finished"
  else
    echo "[search-audit] shard=$index failed" >&2
    failed=1
  fi
done
if [[ $failed -ne 0 ]]; then
  exit 1
fi

python -u scripts/aggregate_grounded_search_audit.py \
  --directory "$output_dir" \
  --output "$output_dir/summary.json"
sha256sum "$output_dir"/predictions.shard-*.jsonl "$output_dir/summary.json"
echo "[search-audit] completed output=$output_dir"
