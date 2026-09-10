#!/usr/bin/env bash
set -Eeuo pipefail

repo_dir=/aaa/fionafyang/buddy1/whaleywang/MechET
runtime_dir=${MECHET_RESCUE_RUNTIME_DIR:-$repo_dir/artifacts/rapid_a7_rescue_runtime_20260910}
model_source=/aaa/fionafyang/buddy1/whaleywang/OpenEvolveChem/data/hf_cache/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
adapter_source=$repo_dir/outputs/agent/tool_sft_flower_compact_full_state_qwen3_8b_a100_20260826
data=$repo_dir/outputs/gates/rapid_a7_rescue_20260910/selected_rows.jsonl
output=$repo_dir/outputs/eval/rapid_a7_rescue_gold_state_k1_ckpt4140_20260910
expected_data_sha=d16405259dd60bc6ef6f73fd5a3551827ef2b2ec672380887a09b0e6957d9f8a
expected_adapter_sha=403825d4d02029e1d49c13c8912baf7ecfac1ac15e726c90a0dd14a775a878fc

source /root/miniconda3/etc/profile.d/conda.sh
conda activate meteor
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$runtime_dir/src:$runtime_dir${PYTHONPATH:+:$PYTHONPATH}"

python - "$runtime_dir" "$data" "$adapter_source" "$expected_data_sha" "$expected_adapter_sha" <<'PY'
import hashlib
from pathlib import Path
import sys
import torch

runtime, data, adapter = map(Path, sys.argv[1:4])
data_sha, adapter_sha = sys.argv[4:]
required = [
    runtime / "scripts/eval_a7_gold_state_local.py",
    data,
    adapter / "adapter_model.safetensors",
    adapter / "adapter_config.json",
]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise SystemExit(f"missing A7 rescue artifacts: {missing}")
def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
if sha(data) != data_sha:
    raise SystemExit("selected validation data SHA mismatch")
if sha(adapter / "adapter_model.safetensors") != adapter_sha:
    raise SystemExit("A7 adapter SHA mismatch")
if torch.cuda.device_count() != 8:
    raise SystemExit(f"expected 8 GPUs, got {torch.cuda.device_count()}")
names = [torch.cuda.get_device_name(index) for index in range(8)]
if not all("A100" in name.upper() for name in names):
    raise SystemExit(f"expected eight A100 GPUs, got {names}")
print({"stage": "validated", "data_sha": data_sha, "adapter_sha": adapter_sha, "gpus": names}, flush=True)
PY

local_model=/tmp/mechet_a7_rescue_qwen3_8b_b968826d
if [[ ! -f "$local_model/.complete" ]]; then
  stage=$(mktemp -d /tmp/mechet_a7_rescue_model.XXXXXX)
  echo "[meteor-stage] copying base model to node-local storage $(date --iso-8601=seconds)"
  cp -aL "$model_source/." "$stage/"
  touch "$stage/.complete"
  mv "$stage" "$local_model"
fi

local_adapter=/tmp/mechet_a7_rescue_adapter_403825d4
if [[ ! -f "$local_adapter/.complete" ]]; then
  stage=$(mktemp -d /tmp/mechet_a7_rescue_adapter.XXXXXX)
  echo "[meteor-stage] copying adapter to node-local storage $(date --iso-8601=seconds)"
  cp -aL "$adapter_source/." "$stage/"
  touch "$stage/.complete"
  mv "$stage" "$local_adapter"
fi

mkdir -p "$output"
cd "$runtime_dir"
echo "[meteor-stage] starting F-oracle gold-state K=1 evaluation $(date --iso-8601=seconds)"
torchrun --standalone --nproc_per_node=8 scripts/eval_a7_gold_state_local.py run \
  --data "$data" \
  --output "$output" \
  --model "$local_model" \
  --adapter "$local_adapter" \
  --max-new-tokens 512 \
  --max-context 16384

python -u scripts/eval_a7_gold_state_local.py aggregate \
  --data "$data" \
  --output "$output" \
  --model "$local_model" \
  --adapter "$local_adapter"
echo "[meteor-stage] A7 local-policy smoke complete $(date --iso-8601=seconds)"
