#!/usr/bin/env bash
set -Eeuo pipefail

repo_dir=/aaa/fionafyang/buddy1/whaleywang/MechET
shared_hf_cache=/aaa/fionafyang/buddy1/whaleywang/OpenEvolveChem/data/hf_cache
liger_wheel="$repo_dir/artifacts/wheels/liger_kernel-0.6.2-py3-none-any.whl"
liger_wheel_sha256=303b9bbf5c10f9289c3139afb41e4d989e8c809516624a106b89b064163d971d
training_config=${MECHET_TRAINING_CONFIG:?set MECHET_TRAINING_CONFIG}

source /root/miniconda3/etc/profile.d/conda.sh
conda activate meteor
cd "$repo_dir"

test -f "$training_config"
test -f "$liger_wheel"
echo "$liger_wheel_sha256  $liger_wheel" | sha256sum --check --strict
liger_target=$(mktemp -d /tmp/mechet_iclr_liger.XXXXXX)
python -m pip install --quiet --no-deps --target "$liger_target" "$liger_wheel"

export HF_HUB_CACHE="$shared_hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$liger_target:$repo_dir/src:$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python - "$training_config" <<'PY'
import json
from pathlib import Path
import sys

import torch
import yaml

config_path = Path(sys.argv[1])
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
required = [
    Path(config["train_file"]),
    Path(config["validation_file"]),
    Path(config["contract"]["validation_report"]),
]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise SystemExit(f"missing frozen ICLR artifacts: {missing}")
contract = json.loads(required[-1].read_text(encoding="utf-8"))
expected_rows = int(config["contract"].get("expected_train_rows") or 0)
if not contract.get("ok") or int(contract.get("n_reference") or 0) != expected_rows:
    raise SystemExit(f"invalid matched-data contract: {contract}")
if not torch.cuda.is_available() or torch.cuda.device_count() != 8:
    raise SystemExit(f"expected exactly 8 CUDA devices, got {torch.cuda.device_count()}")
print({
    "config": str(config_path),
    "condition": config["condition_name"],
    "cuda_devices": torch.cuda.device_count(),
    "gpus": [torch.cuda.get_device_name(i) for i in range(8)],
    "matched_train_rows": expected_rows,
    "seed": config["training"]["seed"],
    "max_steps": config["training"]["max_steps"],
})
PY

exec torchrun \
  --standalone \
  --nproc_per_node=8 \
  scripts/train_tool_sft.py \
  --config "$training_config"
