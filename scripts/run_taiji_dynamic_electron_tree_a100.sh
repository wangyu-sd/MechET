#!/usr/bin/env bash
set -Eeuo pipefail
shared=/aaa/fionafyang/buddy1/whaleywang/MechET
source_repo=/aaa/fionafyang/buddy1/whaleywang/MechET-pr83-greedy
: "${MECHET_TREE_CODE_SHA:?pinned code SHA required}"
: "${MECHET_TREE_OUTPUT:?explicit NEW task output directory required}"
mountpoint -q /aaa/fionafyang/buddy1
source /root/miniconda3/etc/profile.d/conda.sh
conda activate meteor
runtime_code=$(mktemp -d /tmp/mechet_dynamic_tree_code.XXXXXX)
git -C "$source_repo" archive "$MECHET_TREE_CODE_SHA" | tar -x -C "$runtime_code"
vllm_local=$(mktemp -d /tmp/mechet_dynamic_tree_vllm.XXXXXX)
bundle=$shared/artifacts/taiji_vllm_runtime/vllm_0_8_5_torch_2_6_cu118_py311.tar.gz
echo '[dynamic-tree] staging existing pinned CUDA11.8 vLLM runtime'
bash "$runtime_code/scripts/unpack_taiji_runtime.sh" "$bundle" \
  b1185493fb316b2d4534d888df2a6cd871dbb74f6c95255235ba4947ae8486a5 "$vllm_local"
dependencies=$(mktemp -d /tmp/mechet_dynamic_tree_dependencies.XXXXXX)
bash "$runtime_code/scripts/unpack_taiji_runtime.sh" \
  "$shared/artifacts/taiji_vllm_runtime/vllm_0_8_5_noncuda_dependencies_from_archive_20261014.tar.gz" \
  817efd049ae42972a4be38b1b43c9f975fb068050c8f52923cfcc85178bd389d "$dependencies"
cp "$runtime_code/scripts/vllm_cuda11_sitecustomize.py" "$vllm_local/sitecustomize.py"
export MECHET_VLLM_PUREPY_FALLBACK=$dependencies
rdkit_local=$(mktemp -d /tmp/mechet_dynamic_tree_rdkit.XXXXXX)
wheel=$shared/artifacts/wheels/rdkit-2026.3.4-cp311-cp311-manylinux_2_28_x86_64.whl
echo "a41dde42ecb24e7d93d62b89e8e5d267b02bbac764f965f3968296c99234c685  $wheel" | sha256sum --check --strict
python -m pip install --no-deps --no-index --target "$rdkit_local" "$wheel"
export PYTHONPATH=$rdkit_local:$vllm_local:$runtime_code/src:$runtime_code
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_CACHE=/aaa/fionafyang/buddy1/whaleywang/OpenEvolveChem/data/hf_cache
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_V1=0
export NCCL_P2P_DISABLE=1
cd "$runtime_code"
python - <<'PY'
import os,rdkit,torch,vllm,vllm._C,xformers
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
assert not torch.__file__.startswith(os.environ['MECHET_VLLM_PUREPY_FALLBACK']+'/')
assert rdkit.__version__=='2026.03.4'
assert torch.version.cuda=='11.8',torch.version.cuda
assert vllm.__version__.split('+')[0]=='0.8.5'
assert torch.cuda.device_count()==8
assert all('A100' in torch.cuda.get_device_name(i) for i in range(8))
print({'phase':'dynamic-tree-preflight','rdkit':rdkit.__version__,'torch':torch.__version__,
       'vllm':vllm.__version__,'gpus':8},flush=True)
PY
source=$shared/data/mech_uspto_31k_full_endpoint_rxnmapper/test.jsonl
source_sha=819215c5d40720afd998aaf77b7b938045d8171a5487b26bf929897f65a957c4
adapter=$shared/outputs/agent/natural_language_event_history_v2_qwen3_0_6b_seed17
output=$MECHET_TREE_OUTPUT
shared_cache=${MECHET_SHARED_TREE_CACHE:-$shared/outputs/eval/shared_electron_search_v1}
echo "$source_sha  $source" | sha256sum --check --strict
echo "40204c7a059d16976fc4b24b315c3bdde7bf61c705d24bbde515071757e9de75  $adapter/adapter_model.safetensors" | sha256sum --check --strict
for mode in greedy tree; do
  pids=()
  for rank in $(seq 0 7); do
    CUDA_VISIBLE_DEVICES=$rank python -u scripts/infer_dynamic_electron_trees.py \
      --data "$source" --source-sha "$source_sha" --expected-rows 3120 \
      --adapter "$adapter" --output "$output" --mode "$mode" --rank "$rank" --world 8 \
      --shared-cache "$shared_cache" \
      --task-token-budget 80000000 --wall-hours 24 --cpu-workers 8 --batch-prompts 32 &
    pids+=("$!")
  done
  status=0
  for pid in "${pids[@]}"; do wait "$pid" || status=1; done
  (( status == 0 )) || { echo '[dynamic-tree] worker failed; preserve raw artifacts' >&2; exit 2; }
  python -u scripts/summarize_dynamic_electron_trees.py --data "$source" --output "$output" --mode "$mode"
done
echo '[dynamic-tree] full3120 fixed-policy greedy and dynamic-tree evaluation completed'
