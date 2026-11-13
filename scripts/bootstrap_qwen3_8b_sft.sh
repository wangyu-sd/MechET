#!/usr/bin/env bash
set -Eeuo pipefail
source /aaa/fionafyang/buddy1/whaleywang/MechET/configs/runtime/shared_host_storage.env
runtime_repo=${MECHET_QWEN3_CODE_ROOT:?frozen code required}
source /root/miniconda3/etc/profile.d/conda.sh
conda activate meteor
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 USE_TF=0 USE_FLAX=0
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# Bypass unrelated editable-project .pth traversal. This wrapper retains the
# image's installed dependency versions, rather than pip-installing at startup.
task_site=$(/root/miniconda3/envs/meteor/bin/python -S -c 'import sysconfig;print(sysconfig.get_path("purelib"))')
export PYTHONPATH="$runtime_repo/src:$runtime_repo:$task_site"
# -S is inherited by each worker via this narrow wrapper on shared storage.
export PATH="$runtime_repo/scripts/qwen3_runtime_bin:$PATH"
exec bash "$runtime_repo/scripts/run_qwen3_8b_corrected_8h20.sh"
