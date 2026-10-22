#!/usr/bin/env bash
set -Eeuo pipefail
if (( $# != 5 )); then
  echo 'usage: bootstrap CODE_ARCHIVE CODE_SHA RUNTIME_ARCHIVE RUNTIME_SHA PLAN' >&2
  exit 2
fi
code_archive=$1
code_sha=$2
runtime_archive=$3
runtime_sha=$4
plan=$5
source /root/miniconda3/etc/profile.d/conda.sh
conda activate meteor
pr96_node=$(mktemp -d /tmp/mechet-pr96.XXXXXX)
export PR96_NODE_CACHE="$pr96_node"
mkdir "$pr96_node/code" "$pr96_node/runtime" "$pr96_node/rdkit"
echo "[meteor][pr96] node-cache=$pr96_node validating frozen archives"
python - "$code_archive" "$code_sha" "$runtime_archive" "$runtime_sha" <<'PY'
import hashlib,sys
from pathlib import Path
for path,digest in ((sys.argv[1],sys.argv[2]),(sys.argv[3],sys.argv[4])):
    result=hashlib.sha256()
    with Path(path).open('rb') as reader:
        for chunk in iter(lambda:reader.read(8*1024*1024),b''):result.update(chunk)
    if result.hexdigest()!=digest:raise ValueError('PR96_ARCHIVE_HASH_MISMATCH:'+path)
    print({'phase':'archive-verified','path':path,'sha256':digest},flush=True)
PY
tar -xzf "$code_archive" -C "$pr96_node/code"
tar -xzf "$runtime_archive" -C "$pr96_node/runtime"
python -m pip install --no-deps --no-index --target "$pr96_node/rdkit" /aaa/fionafyang/buddy1/whaleywang/MechET/artifacts/wheels/rdkit-2026.3.4-cp311-cp311-manylinux_2_28_x86_64.whl
export PYTHONPATH="$pr96_node/runtime:$pr96_node/rdkit:$pr96_node/code/src:$pr96_node/code"
export HF_HUB_CACHE="$pr96_node/hf_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python - <<'PY'
import sys,torch,triton,transformers,peft
from mechet.qwen35_protocol import validate_runtime_versions
if sys.version_info[:2]!=(3,11):raise ValueError('PR96_REQUIRES_PYTHON311')
validate_runtime_versions(torch_version=torch.__version__,cuda_version=torch.version.cuda,triton_version=triton.__version__)
from transformers import Qwen3_5ForCausalLM
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from fla.modules import FusedRMSNormGated
print({'phase':'runtime-imported','python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,
       'transformers':transformers.__version__,'peft':peft.__version__,'triton':triton.__version__,
       'gpu_names':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]},flush=True)
PY
cd "$pr96_node/code"
exec python -u scripts/run_pr96_pipeline.py --plan "$plan" --node-cache "$pr96_node"
