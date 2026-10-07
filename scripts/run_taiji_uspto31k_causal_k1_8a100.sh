#!/usr/bin/env bash
set -Eeuo pipefail

shared=/aaa/fionafyang/buddy1/whaleywang/MechET
runtime=/aaa/fionafyang/buddy1/whaleywang/MechET-pr82-reliable-20261006
nl_runtime=/aaa/fionafyang/buddy1/whaleywang/MechET-nl-v2-full-runtime-20260918-02
cache=/aaa/fionafyang/buddy1/whaleywang/OpenEvolveChem/data/hf_cache
runner=$runtime/scripts/eval_uspto31k_causal_k1.py

source /root/miniconda3/etc/profile.d/conda.sh
conda activate meteor
cd "$shared"
mountpoint -q /aaa/fionafyang/buddy1
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export HF_HUB_CACHE=$cache
export MECHET_NL_V2_RUNTIME_DIR=$nl_runtime

python - <<'PY'
import torch
names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
assert len(names) == 8 and all('A100' in name for name in names), names
print({'gate': 'eight_a100', 'gpus': names}, flush=True)
PY

wheels=$(mktemp -d /tmp/mechet_31k_causal_eval_wheels.XXXXXX)
for item in \
  artifacts/wheels/bitsandbytes-0.49.2-py3-none-manylinux_2_24_x86_64.whl \
  artifacts/wheels/rdkit-2026.3.4-cp311-cp311-manylinux_2_28_x86_64.whl; do
  test -s "$item"
done
python -m pip install --quiet --no-deps --target "$wheels" \
  artifacts/wheels/bitsandbytes-0.49.2-py3-none-manylinux_2_24_x86_64.whl \
  artifacts/wheels/rdkit-2026.3.4-cp311-cp311-manylinux_2_28_x86_64.whl
export PYTHONPATH=$wheels:$shared:$nl_runtime:$nl_runtime/src${PYTHONPATH:+:$PYTHONPATH}
python - <<'PY'
import rdkit
assert rdkit.__version__ == '2026.03.4', rdkit.__version__
print({'gate': 'rdkit_version', 'version': rdkit.__version__}, flush=True)
PY

python -u "$runner" --mode preflight
echo '[meteor-31k] eight product-only K=1 inference workers starting'
pids=()
for rank in {0..7}; do
  CUDA_VISIBLE_DEVICES=$rank python -u "$runner" --mode worker --rank "$rank" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    echo '[meteor-31k] inference worker failed; stopping remaining workers' >&2
    kill "${pids[@]}" 2>/dev/null || true
    wait || true
    exit 1
  fi
done
python -u "$runner" --mode aggregate
echo '[meteor-31k] full 1,253-reaction executable trace-view evaluation complete'
