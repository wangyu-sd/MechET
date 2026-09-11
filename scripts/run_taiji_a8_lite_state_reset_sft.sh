#!/usr/bin/env bash
set -Eeuo pipefail

repo_dir=${MECHET_REPO_DIR:-/aaa/fionafyang/buddy1/whaleywang/MechET-a8-lite}
artifact_root=/aaa/fionafyang/buddy1/whaleywang/MechET
shared_hf_cache=/aaa/fionafyang/buddy1/whaleywang/OpenEvolveChem/data/hf_cache
training_config=${MECHET_TRAINING_CONFIG:-configs/agent/tool_sft_flower_a8_lite_state_reset_qwen3_8b_a100.yaml}
expected_gpu=${MECHET_EXPECTED_GPU:-A100}

cd "$repo_dir"
if [[ "$training_config" != /* ]]; then
  training_config="$repo_dir/$training_config"
fi
source /root/miniconda3/etc/profile.d/conda.sh
conda activate meteor

runtime_target=$(mktemp -d /tmp/mechet_a8_lite_runtime.XXXXXX)
python -m pip install --quiet --no-deps --target "$runtime_target" \
  "$artifact_root/artifacts/wheels/liger_kernel-0.6.2-py3-none-any.whl" \
  "$artifact_root/artifacts/wheels/xformers-0.0.29.post3-cp311-cp311-manylinux_2_28_x86_64.whl" \
  "$artifact_root/artifacts/wheels/bitsandbytes-0.49.2-py3-none-manylinux_2_24_x86_64.whl"

export HF_HUB_CACHE="$shared_hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$runtime_target:$repo_dir/src:$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_COMPILE_THREADS=1

# The frozen A7 adapter manifest records its data contract relative to the main
# artifact checkout.  Keep that checkout as cwd while executing code from this
# isolated implementation worktree.
cd "$artifact_root"

python - "$training_config" "$expected_gpu" <<'PY'
import json
from pathlib import Path
import sys

import torch
import yaml

cfg = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
contract = dict(cfg["contract"])
manifest_path = Path(contract["stable_id_manifest"])
required = [
    Path(cfg["train_file"]),
    Path(cfg["validation_file"]),
    Path(cfg["test_file"]),
    Path(cfg["initial_adapter_path"]) / "adapter_model.safetensors",
    Path(cfg["initial_adapter_manifest"]),
    manifest_path,
]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise SystemExit(f"missing A8-Lite artifacts: {missing}")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if manifest.get("artifact_type") != "a8_lite_state_reset_continuation_sft_v1":
    raise SystemExit(f"unexpected A8-Lite manifest: {manifest_path}")
if manifest.get("observation_mode") != "compact_full_state_v1":
    raise SystemExit("A8-Lite must continue the compact-full-state A7 contract")
if manifest.get("semantics", {}).get("off_policy_recovery") is not False:
    raise SystemExit("A8-Lite v1 must not be mislabeled as off-policy recovery")
for split, key in (
    ("train", "expected_train_rows"),
    ("valid", "expected_validation_rows"),
    ("test", "expected_test_rows"),
):
    actual = int(manifest["splits"][split]["rows"])
    expected = int(contract[key])
    if actual != expected:
        raise SystemExit(f"{split} rows {actual} != {expected}")
if torch.cuda.device_count() != 8:
    raise SystemExit(f"expected 8 GPUs, got {torch.cuda.device_count()}")
names = [torch.cuda.get_device_name(index) for index in range(8)]
if not all(sys.argv[2].upper() in name.upper() for name in names):
    raise SystemExit(f"expected {sys.argv[2]}, got {names}")
print(
    "[A8-Lite][gate] "
    + json.dumps(
        {
            "condition": cfg["condition_name"],
            "rows": {name: manifest["splits"][name]["rows"] for name in ("train", "valid", "test")},
            "roles": manifest["selection"],
            "initial_adapter": cfg["initial_adapter_path"],
            "gpus": names,
        },
        ensure_ascii=False,
    ),
    flush=True,
)
PY

initial_adapter=$(python - "$training_config" <<'PY'
from pathlib import Path
import sys
import yaml
cfg = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(cfg["initial_adapter_path"])
PY
)
local_initial_adapter=$(mktemp -d /tmp/mechet_a8_initial_adapter.XXXXXX)
echo "[A8-Lite] staging A7 adapter to $local_initial_adapter"
cp -a "$initial_adapter/." "$local_initial_adapter/"
export MECHET_INITIAL_ADAPTER_PATH="$local_initial_adapter"

token_cache=$(python - "$training_config" <<'PY'
from pathlib import Path
import sys
import yaml
cfg = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(cfg["pretokenized_cache_dir"])
PY
)
if [[ ! -f "$token_cache/manifest.json" ]]; then
  echo "[A8-Lite] building distributed token cache"
  torchrun --standalone --nproc_per_node=8 \
    "$repo_dir/scripts/prepare_tool_sft_arrow.py" --config "$training_config"
fi

resume_args=()
output_dir=$(python - "$training_config" <<'PY'
from pathlib import Path
import sys
import yaml
cfg = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(cfg["output_dir"])
PY
)
if find "$output_dir" -maxdepth 2 -type f -name trainer_state.json -print -quit 2>/dev/null | grep -q .; then
  resume_args+=(--resume-from-checkpoint)
fi

echo "[A8-Lite] starting A7 continuation training"
exec torchrun --standalone --nproc_per_node=8 \
  "$repo_dir/scripts/train_tool_sft.py" --config "$training_config" "${resume_args[@]}"
