#!/usr/bin/env bash
set -Eeuo pipefail

repo_dir=/aaa/fionafyang/buddy1/whaleywang/MechET
shared_hf_cache=/aaa/fionafyang/buddy1/whaleywang/OpenEvolveChem/data/hf_cache
liger_wheel="$repo_dir/artifacts/wheels/liger_kernel-0.6.2-py3-none-any.whl"
liger_wheel_sha256=303b9bbf5c10f9289c3139afb41e4d989e8c809516624a106b89b064163d971d
xformers_wheel="$repo_dir/artifacts/wheels/xformers-0.0.29.post3-cp311-cp311-manylinux_2_28_x86_64.whl"
xformers_wheel_sha256=bbf2f500dfdbcf4649bf568cc2c9f434399f704dc4064fd1fbdbef2b524a8139
bitsandbytes_wheel="$repo_dir/artifacts/wheels/bitsandbytes-0.49.2-py3-none-manylinux_2_24_x86_64.whl"
bitsandbytes_wheel_sha256=54b771f06e1a3c73af5c7f16ccf0fc23a846052813d4b008d10cb6e017dd1c8c
training_config=${MECHET_TRAINING_CONFIG:-configs/agent/tool_sft_flower_full_qwen3_8b_v100.yaml}
expected_gpu=${MECHET_EXPECTED_GPU:-V100}

source /root/miniconda3/etc/profile.d/conda.sh
conda activate meteor
cd "$repo_dir"

test -f "$training_config"
test -f "$liger_wheel"
test -f "$xformers_wheel"
test -f "$bitsandbytes_wheel"
echo "$liger_wheel_sha256  $liger_wheel" | sha256sum --check --strict
echo "$xformers_wheel_sha256  $xformers_wheel" | sha256sum --check --strict
echo "$bitsandbytes_wheel_sha256  $bitsandbytes_wheel" | sha256sum --check --strict
liger_target=$(mktemp -d /tmp/mechet_flower_full_liger.XXXXXX)
python -m pip install --quiet --no-deps --target "$liger_target" \
  "$liger_wheel" "$xformers_wheel" "$bitsandbytes_wheel"

export HF_HUB_CACHE="$shared_hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$liger_target:$repo_dir/src:$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_COMPILE_THREADS=1

python - <<'PY'
import importlib.metadata
import bitsandbytes
import liger_kernel
import xformers

print(
    {
        "bitsandbytes": importlib.metadata.version("bitsandbytes"),
        "liger_kernel": importlib.metadata.version("liger-kernel"),
        "xformers": importlib.metadata.version("xformers"),
    }
)
PY

python - "$training_config" <<'PY'
import json
import os
from pathlib import Path
import sys
import torch
import yaml

cfg = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
manifest = Path(cfg["contract"]["stable_id_manifest"])
required = [Path(cfg["train_file"]), Path(cfg["validation_file"]), manifest]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise SystemExit(f"missing full FlowER artifacts: {missing}")
contract = json.loads(manifest.read_text(encoding="utf-8"))
expected = int(cfg["contract"]["expected_train_rows"])
actual = int(contract["splits"]["train"]["rows"])
expected_valid = int(cfg["contract"]["expected_validation_rows"])
expected_test = int(cfg["contract"]["expected_test_rows"])
actual_valid = int(contract["splits"]["valid"]["rows"])
actual_test = int(contract["splits"]["test"]["rows"])
if (
    actual != expected
    or actual_valid != expected_valid
    or actual_test != expected_test
    or not contract.get("reaction_coverage_complete")
):
    raise SystemExit(f"invalid full-data contract: expected={expected} manifest={contract}")
if not torch.cuda.is_available() or torch.cuda.device_count() != 8:
    raise SystemExit(f"expected exactly 8 CUDA devices, got {torch.cuda.device_count()}")
names = [torch.cuda.get_device_name(i) for i in range(8)]
expected_gpu = os.environ.get("MECHET_EXPECTED_GPU", "V100").upper()
if not all(expected_gpu in name.upper() for name in names):
    raise SystemExit(f"expected 8 {expected_gpu} GPUs, got {names}")
print({"config": sys.argv[1], "train_rows": actual, "gpus": names})
PY

token_cache_manifest=$(python - "$training_config" <<'PY'
from pathlib import Path
import sys
import yaml
cfg = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(Path(cfg["pretokenized_cache_dir"]) / "manifest.json")
PY
)
token_cache_dir=$(dirname "$token_cache_manifest")
for _ in $(seq 1 480); do
  if [[ -f "$token_cache_manifest" || ! -d "$token_cache_dir/.preparing-local" ]]; then
    break
  fi
  sleep 30
done
if [[ -d "$token_cache_dir/.preparing-local" && ! -f "$token_cache_manifest" ]]; then
  echo "local pretokenization lock did not clear within four hours" >&2
  exit 1
fi
if [[ ! -f "$token_cache_manifest" ]]; then
  torchrun \
    --standalone \
    --nproc_per_node=8 \
    scripts/prepare_tool_sft_arrow.py \
    --config "$training_config"
fi

# Large Arrow shards and model blobs perform poorly when eight ranks fault the
# same Ceph-backed pages concurrently.  Opt-in staging reads each artifact once
# and lets all ranks mmap it from node-local storage.  The default remains
# unchanged for existing jobs.
if [[ "${MECHET_STAGE_TO_LOCAL:-0}" == "1" ]]; then
  local_stage_root=${MECHET_LOCAL_STAGE_ROOT:-/tmp/mechet_training_stage}
  local_token_cache="$local_stage_root/token_cache"
  local_hf_cache="$local_stage_root/hf_cache"
  mkdir -p "$local_stage_root"

  if [[ ! -f "$local_token_cache/manifest.json" ]]; then
    echo "staging token cache to node-local storage: $local_token_cache"
    [[ ! -e "$local_token_cache" ]] || {
      echo "incomplete local token cache already exists: $local_token_cache" >&2
      exit 2
    }
    token_stage_tmp=$(mktemp -d "$local_stage_root/.token_cache.XXXXXX")
    cp -a "$token_cache_dir/." "$token_stage_tmp/"
    mv "$token_stage_tmp" "$local_token_cache"
  fi
  if [[ ! -d "$local_hf_cache/models--Qwen--Qwen3-8B" ]]; then
    echo "staging Hugging Face cache to node-local storage: $local_hf_cache"
    [[ ! -e "$local_hf_cache" ]] || {
      echo "incomplete local Hugging Face cache already exists: $local_hf_cache" >&2
      exit 2
    }
    hf_stage_tmp=$(mktemp -d "$local_stage_root/.hf_cache.XXXXXX")
    cp -a "$shared_hf_cache/." "$hf_stage_tmp/"
    mv "$hf_stage_tmp" "$local_hf_cache"
  fi

  export MECHET_PRETOKENIZED_CACHE_DIR="$local_token_cache"
  export HF_HUB_CACHE="$local_hf_cache"
  echo "node-local staging complete"
fi

exec torchrun \
  --standalone \
  --nproc_per_node=8 \
  scripts/train_tool_sft.py \
  --config "$training_config"
