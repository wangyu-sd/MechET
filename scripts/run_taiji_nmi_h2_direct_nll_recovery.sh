#!/usr/bin/env bash
set -Eeuo pipefail

# Recovery only: all eight frozen Direct K=10 generation shards were completed
# by the prior task. Never launch generation or change candidate content here.
repo_dir=${MECHET_REPO_DIR:-/aaa/fionafyang/buddy1/whaleywang/MechET-pr73-execute-20261001}
data_root=/aaa/fionafyang/buddy1/whaleywang/MechET
output_dir="$data_root/outputs/issue79/eval_h2_direct_k10_seed17_vllm02"
test_file="$data_root/data/issue79/nmi_matched_v1/direct/test.jsonl"
adapter="$data_root/outputs/issue79/h2_direct_seed17"
revision=b968826d9c46dd6066d109eabc6255188de91218

source /root/miniconda3/etc/profile.d/conda.sh
conda activate meteor
cd "$repo_dir"
export HF_HUB_CACHE=/aaa/fionafyang/buddy1/whaleywang/OpenEvolveChem/data/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$repo_dir/src:$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

python - "$output_dir" "$test_file" "$adapter" "$revision" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
import torch

root, reference, adapter = map(Path, sys.argv[1:4])
revision = sys.argv[4]
manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
adapter_manifest = json.loads((adapter / "adapter_manifest.json").read_text(encoding="utf-8"))
contract = json.loads((adapter / "data_contract.json").read_text(encoding="utf-8"))
data_manifest = json.loads((reference.parent / "manifest.json").read_text(encoding="utf-8"))
predictions = root / "predictions.jsonl"
if manifest["n_targets"] != 27104 or manifest["n_candidates"] != 271040:
    raise SystemExit("merged candidate count does not match frozen H2 Direct K=10")
if manifest["reference_sha256"] != data_manifest["tasks"]["direct"]["test"]["sha256"]:
    raise SystemExit("reference data lineage mismatch")
if manifest["shard_adapter_sha256"] != adapter_manifest["adapter_sha256"]:
    raise SystemExit("generation/scoring adapter mismatch")
if manifest["shard_model_revision"] != revision or adapter_manifest["base_model_revision"] != revision:
    raise SystemExit("generation/scoring base-model revision mismatch")
if contract["train_file_sha256"] != data_manifest["tasks"]["direct"]["train"]["sha256"]:
    raise SystemExit("adapter training lineage mismatch")
with predictions.open("rb") as source:
    digest = hashlib.sha256()
    for block in iter(lambda: source.read(1024 * 1024), b""):
        digest.update(block)
if digest.hexdigest() != manifest["predictions_sha256"]:
    raise SystemExit("merged predictions changed after generation")
names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
if len(names) != 8 or not all("A100" in name.upper() for name in names):
    raise SystemExit(f"expected ordinary eight-A100 host, got {names}")
print(f"[meteor-recovery] frozen Direct K=10 lineage verified; GPUs={names}", flush=True)
PY

mkdir -p "$output_dir/nll_ranking"
pids=()
for worker in $(seq 0 7); do
  shard=$(printf '%03d' "$worker")
  CUDA_VISIBLE_DEVICES="$worker" python scripts/score_and_rank_predictions.py score \
    --predictions "$output_dir/predictions.jsonl" \
    --output "$output_dir/nll_ranking/nll_scores.shard-${shard}.jsonl" \
    --model Qwen/Qwen3-8B --revision "$revision" --adapter "$adapter" \
    --shard-count 8 --shard-index "$worker" --max-length 4096 --resume &
  pids+=("$!")
done

while :; do
  live=0
  for pid in "${pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then live=$((live + 1)); fi
  done
  complete=$(find "$output_dir/nll_ranking" -name 'nll_scores.shard-*.jsonl' -type f -print0 | xargs -0 -r wc -l | awk '$2 != "total" {sum += $1} END{print sum+0}')
  echo "[meteor-progress] stage=nmi-h2-direct-nll completed_targets=${complete}/27104 live_shards=${live}/8 time=$(date --iso-8601=seconds)"
  if (( live == 0 )); then break; fi
  sleep 60
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
if (( status != 0 )); then exit "$status"; fi

echo "[meteor-progress] stage=nmi-h2-direct-evaluation time=$(date --iso-8601=seconds)"
exec python scripts/evaluate_endpoint_candidates.py \
  --reference "$test_file" \
  --predictions "$output_dir/predictions.jsonl" \
  --ranking-dir "$output_dir/nll_ranking" \
  --output "$output_dir/evaluation.json" \
  --expected-rows 27104 --expected-candidates 10
