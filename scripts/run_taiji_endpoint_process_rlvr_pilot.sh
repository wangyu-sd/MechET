#!/usr/bin/env bash
set -Eeuo pipefail

shared_repo=/aaa/fionafyang/buddy1/whaleywang/MechET
runtime_repo=${MECHET_ENDPOINT_RLVR_RUNTIME_DIR:?set MECHET_ENDPOINT_RLVR_RUNTIME_DIR}
pilot_dir=$shared_repo/data/endpoint_process_rlvr_pilot_v1
parent_adapter=$shared_repo/outputs/agent/in_place_grounded_flow_qwen3_8b_a100_seed17_20260911/checkpoint-8037
base_model=/aaa/fionafyang/buddy1/whaleywang/models/orbit_qwen_base
output_dir=$shared_repo/outputs/agent/endpoint_process_rlvr_pilot_qwen3_8b_seed17_20260911
config=$runtime_repo/configs/agent/endpoint_process_rlvr_pilot.yaml
rdkit_wheel=$shared_repo/artifacts/wheels/rdkit-2026.3.4-cp311-cp311-manylinux_2_28_x86_64.whl

source /root/miniconda3/etc/profile.d/conda.sh
conda activate meteor
cd "$shared_repo"

echo "[endpoint-rlvr] runtime=$runtime_repo pilot=$pilot_dir parent=$parent_adapter"
test -f "$pilot_dir/manifest.json"
test -f "$parent_adapter/adapter_model.safetensors"
test -f "$config"
echo "a41dde42ecb24e7d93d62b89e8e5d267b02bbac764f965f3968296c99234c685  $rdkit_wheel" | sha256sum --check --strict

runtime_target=$(mktemp -d /tmp/mechet_endpoint_rlvr_runtime.XXXXXX)
python -m pip install --quiet --no-deps --target "$runtime_target" "$rdkit_wheel"

export PYTHONPATH=$runtime_target:$runtime_repo/src:$runtime_repo
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_COMPILE_THREADS=1

python - "$pilot_dir/manifest.json" <<'PY'
import json,sys
from pathlib import Path
import rdkit,torch
manifest=json.loads(Path(sys.argv[1]).read_text())
if rdkit.__version__ != '2026.03.4':
    raise SystemExit(f'wrong RDKit runtime: {rdkit.__version__}')
if torch.cuda.device_count() != 8:
    raise SystemExit(f'expected 8 GPUs, got {torch.cuda.device_count()}')
names=[torch.cuda.get_device_name(i) for i in range(8)]
if not all('A100' in name.upper() for name in names):
    raise SystemExit(f'expected 8 A100 GPUs, got {names}')
if manifest['train']['rows'] != 512 or manifest['monitor']['rows'] != 128:
    raise SystemExit('pilot denominator mismatch')
if manifest.get('gold_model_visible') is not False:
    raise SystemExit('reward-side gold visibility contract failed')
print({'runtime_gate':'passed','rdkit':rdkit.__version__,'gpus':names},flush=True)
PY

echo "[endpoint-rlvr] 512-reaction event-level RLOO pilot starts"
exec torchrun --standalone --nproc_per_node=8 \
  "$runtime_repo/scripts/train_endpoint_process_rlvr.py" \
  --config "$config" \
  --train-file "$pilot_dir/train.jsonl" \
  --monitor-file "$pilot_dir/monitor.jsonl" \
  --parent-adapter "$parent_adapter" \
  --base-model "$base_model" \
  --output-dir "$output_dir"
