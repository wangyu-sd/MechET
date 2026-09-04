#!/usr/bin/env bash
set -Eeuo pipefail

repo_dir=${MECHET_REPO_DIR:-/aaa/fionafyang/buddy1/whaleywang/MechET}
artifact_root=${MECHET_ARTIFACT_ROOT:-/aaa/fionafyang/buddy1/whaleywang/MechET}
shared_hf_cache=/aaa/fionafyang/buddy1/whaleywang/OpenEvolveChem/data/hf_cache
liger_wheel="$artifact_root/artifacts/wheels/liger_kernel-0.6.2-py3-none-any.whl"
xformers_wheel="$artifact_root/artifacts/wheels/xformers-0.0.29.post3-cp311-cp311-manylinux_2_28_x86_64.whl"
bitsandbytes_wheel="$artifact_root/artifacts/wheels/bitsandbytes-0.49.2-py3-none-manylinux_2_24_x86_64.whl"
training_config=${MECHET_TRAINING_CONFIG:?set MECHET_TRAINING_CONFIG}
expected_gpu=${MECHET_EXPECTED_GPU:-A100}

source /root/miniconda3/etc/profile.d/conda.sh
conda activate meteor
cd "$repo_dir"

test -f "$training_config"
echo "303b9bbf5c10f9289c3139afb41e4d989e8c809516624a106b89b064163d971d  $liger_wheel" | sha256sum --check --strict
echo "bbf2f500dfdbcf4649bf568cc2c9f434399f704dc4064fd1fbdbef2b524a8139  $xformers_wheel" | sha256sum --check --strict
echo "54b771f06e1a3c73af5c7f16ccf0fc23a846052813d4b008d10cb6e017dd1c8c  $bitsandbytes_wheel" | sha256sum --check --strict
runtime_target=$(mktemp -d /tmp/mechet_iclr_full_runtime.XXXXXX)
python -m pip install --quiet --no-deps --target "$runtime_target" \
  "$liger_wheel" "$xformers_wheel" "$bitsandbytes_wheel"

export HF_HUB_CACHE="$shared_hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$runtime_target:$repo_dir/src:$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_COMPILE_THREADS=1

python - "$training_config" "$expected_gpu" <<'PY'
import json
from pathlib import Path
import sys
import torch
import yaml

cfg = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
task = cfg["contract"]["baseline_task"]
manifest = json.loads(Path(cfg["contract"]["stable_id_manifest"]).read_text())
for split, key in (("train", "expected_train_rows"), ("valid", "expected_validation_rows"), ("test", "expected_test_rows")):
    actual = int(manifest["tasks"][task][split]["rows"])
    expected = int(cfg["contract"][key])
    if actual != expected:
        raise SystemExit(f"{task}/{split} rows {actual} != {expected}")
if torch.cuda.device_count() != 8:
    raise SystemExit(f"expected 8 GPUs, got {torch.cuda.device_count()}")
names = [torch.cuda.get_device_name(i) for i in range(8)]
if not all(sys.argv[2].upper() in name.upper() for name in names):
    raise SystemExit(f"expected {sys.argv[2]}, got {names}")
print({"task": task, "rows": manifest["tasks"][task]["train"]["rows"], "gpus": names})
PY

token_cache_manifest=$(python - "$training_config" <<'PY'
from pathlib import Path
import sys, yaml
cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
print(Path(cfg["pretokenized_cache_dir"]) / "manifest.json")
PY
)
if [[ ! -f "$token_cache_manifest" ]]; then
  torchrun --standalone --nproc_per_node=8 scripts/prepare_tool_sft_arrow.py --config "$training_config"
fi

exec torchrun --standalone --nproc_per_node=8 scripts/train_tool_sft.py --config "$training_config"
