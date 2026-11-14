#!/usr/bin/env bash
# Full corrected continuation only. Use a frozen copy of this source tree in a job.
set -Eeuo pipefail
shared_repo=/aaa/fionafyang/buddy1/whaleywang/MechET
source "$shared_repo/configs/runtime/shared_host_storage.env"
runtime_repo=${MECHET_QWEN3_CODE_ROOT:?set the frozen source tree for this job}
training_config=${MECHET_QWEN3_CONFIG:?set the prepared immutable JSON config}
export HF_HOME=/aaa/fionafyang/buddy1/whaleywang/OpenEvolveChem/data/hf_cache
export HF_HUB_CACHE="$HF_HOME"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTHONPATH="$runtime_repo/src:$runtime_repo${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_COMPILE_THREADS=1
python - "$training_config" <<'PY'
import json,os,sys,torch
from mechet.qwen3_continuation import validate_config,shared_path
cfg=json.load(open(sys.argv[1]))
validate_config(cfg,world_size=8,gpu_names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
for key in ('TMPDIR','HF_HOME','HF_HUB_CACHE','TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR'):
    shared_path(os.environ[key])
for key in ('MECHET_OUTPUT_DIR','MECHET_PRETOKENIZED_CACHE_DIR','MECHET_MAX_STEPS','MECHET_RESUME_FROM_CHECKPOINT'):
    if os.environ.get(key): raise ValueError('UNDECLARED_TRAINING_OVERRIDE:'+key)
print({'stage':'four-h20-hardware-and-storage-checked','gpu_names':[torch.cuda.get_device_name(i) for i in range(8)]},flush=True)
PY
# Require CPU preparation BEFORE GPU allocation. Unchanged files reuse the
# hash-bound verification receipt; do not occupy eight H20s tokenizing JSONL.
python "$runtime_repo/scripts/prepare_qwen3_8b_continuation.py" --output-config "$training_config" --world-size 8 --verify-cache
resume_args=()
if [[ -n "${MECHET_QWEN3_RESUME:-}" ]]; then
  resume_args=(--resume-from-checkpoint "$MECHET_QWEN3_RESUME")
fi
python -m torch.distributed.run --standalone --nproc_per_node=8 "$runtime_repo/scripts/train_tool_sft.py" --config "$training_config" "${resume_args[@]}"
