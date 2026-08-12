#!/usr/bin/env bash
set -Eeuo pipefail

repo_dir=/aaa/fionafyang/buddy1/whaleywang/MechET
shared_hf_cache=/aaa/fionafyang/buddy1/whaleywang/OpenEvolveChem/data/hf_cache
liger_wheel="$repo_dir/artifacts/wheels/liger_kernel-0.6.2-py3-none-any.whl"
liger_wheel_sha256=303b9bbf5c10f9289c3139afb41e4d989e8c809516624a106b89b064163d971d

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

python - <<'PY'
from pathlib import Path
import torch
import liger_kernel

required = [
    Path("data/mixed_inverse_tool_sft/train.a100_20480.jsonl"),
    Path("data/mixed_inverse_tool_sft/valid.jsonl"),
    Path("data/mixed_inverse_tool_sft/training_manifest.json"),
]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise SystemExit(f"missing frozen training artifacts: {missing}")
if not torch.cuda.is_available() or torch.cuda.device_count() != 8:
    raise SystemExit(f"expected exactly 8 CUDA devices, got {torch.cuda.device_count()}")
print({"cuda_devices": torch.cuda.device_count(), "torch": torch.__version__})
PY

exec torchrun \
  --standalone \
  --nproc_per_node=8 \
  scripts/train_tool_sft.py \
  --config configs/agent/tool_sft_mixed_inverse_qwen3_8b.yaml
