#!/usr/bin/env bash
set -Eeuo pipefail

repo_dir=/aaa/fionafyang/buddy1/whaleywang/MechET
shared_hf_cache=/aaa/fionafyang/buddy1/whaleywang/OpenEvolveChem/data/hf_cache
liger_wheel="$repo_dir/artifacts/wheels/liger_kernel-0.6.2-py3-none-any.whl"
liger_wheel_sha256=303b9bbf5c10f9289c3139afb41e4d989e8c809516624a106b89b064163d971d
num_train_epochs=${NUM_TRAIN_EPOCHS:-3.0}
training_config=${MECHET_TRAINING_CONFIG:-configs/agent/sft_flower_full_endpoint_qwen3_8b_h20.yaml}

source /root/miniconda3/etc/profile.d/conda.sh
conda activate meteor
cd "$repo_dir"

test -f "$liger_wheel"
echo "$liger_wheel_sha256  $liger_wheel" | sha256sum --check --strict
liger_target=$(mktemp -d /tmp/mechet_liger.XXXXXX)
python -m pip install --quiet --no-deps --target "$liger_target" "$liger_wheel"

export HF_HUB_CACHE="$shared_hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$liger_target:$repo_dir/src:$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python - "$num_train_epochs" <<'PY'
import json
from pathlib import Path
import sys
import torch
import liger_kernel

epochs = float(sys.argv[1])
if epochs <= 0:
    raise SystemExit(f"NUM_TRAIN_EPOCHS must be positive, got {epochs}")
root = Path("data/flower_full_endpoint_sft_decontaminated")
required = [root / name for name in ("train.jsonl", "valid.jsonl", "test.jsonl", "manifest.json")]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise SystemExit(f"missing frozen FlowER endpoint artifacts: {missing}")
manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
if int((manifest.get("test") or {}).get("rows") or 0) != 28_971:
    raise SystemExit("full FlowER test denominator is not 28,971")
if int((manifest.get("train") or {}).get("removed") or 0) <= 0:
    raise SystemExit("train held-out decontamination was not applied")
if not torch.cuda.is_available() or torch.cuda.device_count() != 8:
    raise SystemExit(f"expected exactly 8 CUDA devices, got {torch.cuda.device_count()}")
print({
    "cuda_devices": torch.cuda.device_count(),
    "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    "torch": torch.__version__,
    "num_train_epochs": epochs,
    "train_rows": (manifest.get("train") or {}).get("kept"),
    "valid_rows": (manifest.get("valid") or {}).get("kept"),
    "test_rows": (manifest.get("test") or {}).get("rows"),
})
PY

test -f "$training_config"

exec torchrun \
  --standalone \
  --nproc_per_node=8 \
  scripts/train_tool_sft.py \
  --config "$training_config" \
  --num-train-epochs "$num_train_epochs"
