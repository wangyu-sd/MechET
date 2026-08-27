#!/usr/bin/env bash
set -Eeuo pipefail

repo_dir=/aaa/fionafyang/buddy1/whaleywang/MechET
shared_hf_cache=/aaa/fionafyang/buddy1/whaleywang/OpenEvolveChem/data/hf_cache
liger_wheel="$repo_dir/artifacts/wheels/liger_kernel-0.6.2-py3-none-any.whl"
xformers_wheel="$repo_dir/artifacts/wheels/xformers-0.0.29.post3-cp311-cp311-manylinux_2_28_x86_64.whl"
bitsandbytes_wheel="$repo_dir/artifacts/wheels/bitsandbytes-0.49.2-py3-none-manylinux_2_24_x86_64.whl"
training_config=${MECHET_TRAINING_CONFIG:?set MECHET_TRAINING_CONFIG}
expected_gpu=${MECHET_EXPECTED_GPU:-A100}

source /root/miniconda3/etc/profile.d/conda.sh
conda activate meteor
cd "$repo_dir"

test -f "$training_config"
echo "303b9bbf5c10f9289c3139afb41e4d989e8c809516624a106b89b064163d971d  $liger_wheel" | sha256sum --check --strict
echo "bbf2f500dfdbcf4649bf568cc2c9f434399f704dc4064fd1fbdbef2b524a8139  $xformers_wheel" | sha256sum --check --strict
echo "54b771f06e1a3c73af5c7f16ccf0fc23a846052813d4b008d10cb6e017dd1c8c  $bitsandbytes_wheel" | sha256sum --check --strict
runtime_target=$(mktemp -d /tmp/mechet_compact_full_state_runtime.XXXXXX)
python -m pip install --quiet --no-deps --target "$runtime_target" \
  "$liger_wheel" "$xformers_wheel" "$bitsandbytes_wheel"

export HF_HUB_CACHE="$shared_hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
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
contract_cfg = dict(cfg["contract"])
manifest_path = Path(contract_cfg["stable_id_manifest"])
gate_path = Path(contract_cfg["gate_report"])
required = [
    Path(cfg["train_file"]),
    Path(cfg["validation_file"]),
    Path(cfg["test_file"]),
    manifest_path,
    gate_path,
]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise SystemExit(f"missing compact-full-state artifacts: {missing}")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
gate = json.loads(gate_path.read_text(encoding="utf-8"))
if not gate.get("passed") or float(gate.get("token_ratio", 1.0)) > 0.60:
    raise SystemExit(f"compact equivalence/token gate did not pass: {gate_path}")
if manifest.get("observation_mode") != "compact_full_state_v1":
    raise SystemExit(f"manifest is not compact-full-state: {manifest_path}")
if manifest.get("intermediate_state_model_visible") is not True:
    raise SystemExit("manifest omits the authoritative intermediate state")
for split, key in (
    ("train", "expected_train_rows"),
    ("valid", "expected_validation_rows"),
    ("test", "expected_test_rows"),
):
    actual = int(manifest["splits"][split]["rows"])
    expected = int(contract_cfg[key])
    if actual != expected:
        raise SystemExit(f"{split} rows {actual} != {expected}")
if contract_cfg.get("require_strict_trace_universe_complete") and not manifest.get(
    "strict_trace_universe_complete"
):
    raise SystemExit("strict trace universe is not complete")
if torch.cuda.device_count() != 8:
    raise SystemExit(f"expected 8 GPUs, got {torch.cuda.device_count()}")
names = [torch.cuda.get_device_name(i) for i in range(8)]
if not all(sys.argv[2].upper() in name.upper() for name in names):
    raise SystemExit(f"expected {sys.argv[2]}, got {names}")
print(
    {
        "config": sys.argv[1],
        "condition": cfg["condition_name"],
        "observation_mode": manifest["observation_mode"],
        "token_gate_ratio": gate["token_ratio"],
        "rows": {
            name: manifest["splits"][name]["rows"]
            for name in ("train", "valid", "test")
        },
        "gpus": names,
    }
)
PY

if [[ ${MECHET_STAGE_INITIAL_ADAPTER:-0} == 1 ]]; then
  initial_adapter=$(python - "$training_config" <<'PY'
from pathlib import Path
import sys
import yaml

cfg = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(str(cfg.get("initial_adapter_path") or ""))
PY
  )
  if [[ -z "$initial_adapter" || ! -f "$initial_adapter/adapter_model.safetensors" ]]; then
    echo "MECHET_STAGE_INITIAL_ADAPTER=1 but no valid initial adapter is configured" >&2
    exit 2
  fi
  local_initial_adapter=${MECHET_LOCAL_INITIAL_ADAPTER_DIR:-/tmp/mechet_initial_adapter}
  mkdir -p "$local_initial_adapter"
  cp -a "$initial_adapter/." "$local_initial_adapter/"
  export MECHET_INITIAL_ADAPTER_PATH="$local_initial_adapter"
  echo "[MechET] staged initial adapter to node-local storage: $local_initial_adapter"
fi

token_cache_manifest=$(python - "$training_config" <<'PY'
from pathlib import Path
import sys
import yaml

cfg = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(Path(cfg["pretokenized_cache_dir"]) / "manifest.json")
PY
)
if [[ ! -f "$token_cache_manifest" ]]; then
  torchrun --standalone --nproc_per_node=8 \
    scripts/prepare_tool_sft_arrow.py --config "$training_config"
fi

if [[ ${MECHET_STAGE_PRETOKENIZED_CACHE:-0} == 1 ]]; then
  shared_token_cache=$(dirname "$token_cache_manifest")
  local_token_cache_parent=${MECHET_LOCAL_PRETOKENIZED_CACHE_PARENT:-/tmp}
  mkdir -p "$local_token_cache_parent"
  local_token_cache=$(mktemp -d \
    "$local_token_cache_parent/mechet_pretokenized_cache.XXXXXX")
  echo "[MechET] staging pretokenized cache to node-local storage: source=$shared_token_cache destination=$local_token_cache"
  cp -a "$shared_token_cache/." "$local_token_cache/"
  python - "$shared_token_cache" "$local_token_cache" <<'PY'
from pathlib import Path
import sys

source = Path(sys.argv[1])
destination = Path(sys.argv[2])
source_files = {
    path.name: path.stat().st_size
    for path in source.iterdir()
    if path.is_file()
}
destination_files = {
    path.name: path.stat().st_size
    for path in destination.iterdir()
    if path.is_file()
}
if source_files != destination_files:
    missing = sorted(set(source_files) - set(destination_files))
    extra = sorted(set(destination_files) - set(source_files))
    mismatched = sorted(
        name
        for name in set(source_files) & set(destination_files)
        if source_files[name] != destination_files[name]
    )
    raise SystemExit(
        "node-local pretokenized cache verification failed: "
        f"missing={missing} extra={extra} size_mismatches={mismatched}"
    )
if not (destination / "manifest.json").is_file():
    raise SystemExit("node-local pretokenized cache has no manifest.json")
print(
    "[MechET] node-local pretokenized cache verified: "
    f"files={len(destination_files)} bytes={sum(destination_files.values())}"
)
PY
  export MECHET_PRETOKENIZED_CACHE_DIR="$local_token_cache"
fi

resume_args=()
if find "$(python - "$training_config" <<'PY'
from pathlib import Path
import sys
import yaml
cfg = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(Path(cfg["output_dir"]))
PY
)" -maxdepth 2 -type f -name trainer_state.json -print -quit 2>/dev/null | grep -q .; then
  resume_args+=(--resume-from-checkpoint)
fi

exec torchrun --standalone --nproc_per_node=8 \
  scripts/train_tool_sft.py --config "$training_config" "${resume_args[@]}"
