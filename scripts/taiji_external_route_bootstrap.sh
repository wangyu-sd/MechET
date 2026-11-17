#!/usr/bin/env bash
set -Eeuo pipefail
[[ $# == 1 ]] || { echo 'usage: external-route-bootstrap CONTRACT' >&2; exit 2; }
route_contract=$1
mountpoint -q /aaa/fionafyang/buddy1
source /aaa/fionafyang/buddy1/whaleywang/MechET/configs/runtime/shared_host_storage.env
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
mkdir -p -- "$CUDA_CACHE_PATH"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate meteor
route_python=/root/miniconda3/envs/meteor/bin/python
route_code=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
route_runtime=$("$route_python" -S - "$route_contract" <<'PY'
import json,sys
from pathlib import Path
c=json.loads(Path(sys.argv[1]).read_text())
print(c['runtime_root'])
PY
)
route_world=$("$route_python" -S - "$route_contract" <<'PY'
import json,sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text())['world_size'])
PY
)
route_site=$("$route_python" -S - <<'PY'
import os,sysconfig
from pathlib import Path
root=Path(sysconfig.get_path('purelib')).resolve()
eggs=[str(p.resolve()) for p in sorted(root.glob('*.egg')) if p.resolve().is_relative_to(root)]
print(os.pathsep.join([str(root),*eggs]))
PY
)
# -S prevents unrelated editable-project .pth files from expanding worker paths.
# The pinned packages and ordinary fallback site are provided explicitly.
export PYTHONPATH="$route_code/scripts/_nvml_runtime:$route_runtime/runtime:$route_runtime/rdkit:$route_code/src:$route_code:$route_site"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false USE_TF=0 USE_FLAX=0
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_THREADING_LAYER=GNU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_COMPILE_THREADS=1 TQDM_DISABLE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
cd "$route_code"
# Reuse the existing pinned, read-only runtime. All writes/caches stay on Ceph.
# Warm source reads in parallel before four learners independently import it.
"$route_python" -S -u "$route_code/scripts/prefetch_external_route_inputs.py" "$route_contract"
"$route_python" -S -u "$route_code/scripts/prefetch_runtime_imports.py" --runtime "$route_runtime/runtime" --workers 16
# The engine's exact import/metadata indexes avoid discovery scans. Import its
# required modules on demand instead of pre-reading every optional model/package.
# Validate launch presets in isolation before four actor/reference replicas
# occupy the GPUs. This is a tensor/gradient unit check, not a model rollout.
route_profile=$("$route_python" -S - "$route_contract" <<'PY'
import json,sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text()).get('fla_kernel_profile') or '')
PY
)
if [[ -n "$route_profile" ]]; then
  route_unit_dir=$("$route_python" -S - "$route_contract" <<'PY'
import json,sys
from pathlib import Path
print(Path(json.loads(Path(sys.argv[1]).read_text())['output_dir'])/'runtime_validation')
PY
)
  mkdir -p -- "$route_unit_dir"
  "$route_python" -S -u "$route_code/scripts/verify_route_fla_kernel_profile.py" --contract "$route_contract" \
    2>&1 | tee "$route_unit_dir/l2norm_gpu_unit.log"
fi
exec "$route_python" -S -u -m torch.distributed.run --standalone --nproc_per_node="$route_world" \
  --no-python "$route_python" -S -u -m scripts.train_external_route_rl --contract "$route_contract"
