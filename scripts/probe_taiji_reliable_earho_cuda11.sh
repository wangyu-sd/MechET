#!/usr/bin/env bash
set -Eeuo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate meteor
export PYTHONUNBUFFERED=1

echo '[earho-cuda11-probe] GPU and driver'
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo '[earho-cuda11-probe] Python, PyTorch, CUDA toolkit'
python -u - <<'PY'
import sys
import torch
import transformers

assert torch.cuda.device_count() == 1, torch.cuda.device_count()
assert 'V100' in torch.cuda.get_device_name(0).upper()
assert torch.version.cuda.startswith('11.'), torch.version.cuda
torch.empty(1, device='cuda').add_(1).cpu()
print({'stage': 'cuda11-torch-passed', 'python': sys.version.split()[0],
       'torch': torch.__version__, 'torch_cuda': torch.version.cuda,
       'transformers': transformers.__version__,
       'gpu': torch.cuda.get_device_name(0)}, flush=True)
PY
if command -v nvcc >/dev/null 2>&1; then
  nvcc --version | tail -4
else
  echo '[earho-cuda11-probe] nvcc absent from PATH'
fi
for candidate in /usr/local/cuda/bin/nvcc /usr/local/cuda-11.8/bin/nvcc; do
  if test -x "$candidate"; then "$candidate" --version | tail -4; fi
done
echo '[earho-cuda11-probe] completed'
