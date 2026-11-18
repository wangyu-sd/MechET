#!/usr/bin/env bash
set -Eeuo pipefail
[[ $# == 1 ]] || { echo 'usage: offline-dpo-bootstrap CONTRACT' >&2; exit 2; }
mountpoint -q /aaa/fionafyang/buddy1
source /aaa/fionafyang/buddy1/whaleywang/MechET/configs/runtime/shared_host_storage.env
source /root/miniconda3/etc/profile.d/conda.sh
conda activate meteor
route_python=/root/miniconda3/envs/meteor/bin/python
route_code=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
launch_values=$("$route_python" -S -Bu "$route_code/scripts/pr116_offline_launch_contract.py" "$1")
mapfile -t launch <<< "$launch_values"
[[ ${#launch[@]} == 5 ]]
route_site=$("$route_python" -S - <<'PY'
from pathlib import Path
import os
import sysconfig
site = Path(sysconfig.get_path('purelib')).resolve()
print(os.pathsep.join(str(p) for p in (site, *sorted(site.glob('*.egg')))))
PY
)
export PYTHONPATH="${launch[1]}:$route_code/scripts/_nvml_runtime:${launch[0]}/runtime:${launch[0]}/rdkit:$route_code/src:$route_code:$route_site"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false USE_TF=0 USE_FLAX=0
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_THREADING_LAYER=GNU
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
mkdir -p -- "$CUDA_CACHE_PATH"
cd "$route_code"
"$route_python" -S -Bu "$route_code/scripts/prefetch_runtime_imports.py" --runtime "${launch[0]}/runtime" \
    --workers 8 --packages peft fla
exec "$route_python" -S -u -m torch.distributed.run --standalone --nproc_per_node=4 \
    --no-python "$route_python" -S -u -m scripts.run_pr116_offline_dpo train \
    --manifest "${launch[2]}" --manifest-sha256 "${launch[3]}" --output "${launch[4]}"
