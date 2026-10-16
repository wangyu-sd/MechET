#!/usr/bin/env bash
set -Eeuo pipefail
shared=/aaa/fionafyang/buddy1/whaleywang/MechET
source_repo=/aaa/fionafyang/buddy1/whaleywang/MechET-pr83-greedy
: "${MECHET_TREE_CODE_SHA:?pinned code SHA required}"
: "${MECHET_TREE_OUTPUT:?explicit NEW output directory required}"
: "${MECHET_PRESERVED_GREEDY_OUTPUT:?sealed full3120 baseline required}"
if [[ ${MECHET_USE_RAY:-0} == 1 && -z ${MECHET_TREE_RESUME_FROM:-} ]]; then
  echo '[Ray-MCTS] resume source required' >&2
  exit 2
fi
if [[ -e "$MECHET_TREE_OUTPUT" || -L "$MECHET_TREE_OUTPUT" ]]; then
  echo '[rolling-MCTS] output exists; do not overwrite evidence' >&2
  exit 2
fi
mountpoint -q /aaa/fionafyang/buddy1
source /root/miniconda3/etc/profile.d/conda.sh
conda activate meteor
runtime_code=$(mktemp -d /tmp/mechet_single_mcts_code.XXXXXX)
git -C "$source_repo" archive "$MECHET_TREE_CODE_SHA" | tar -x -C "$runtime_code"
vllm_local=$(mktemp -d /tmp/mechet_single_mcts_vllm.XXXXXX)
bash "$runtime_code/scripts/unpack_taiji_runtime.sh" \
  "$shared/artifacts/taiji_vllm_runtime/vllm_0_8_5_torch_2_6_cu118_py311.tar.gz" \
  b1185493fb316b2d4534d888df2a6cd871dbb74f6c95255235ba4947ae8486a5 "$vllm_local"
dependencies=$(mktemp -d /tmp/mechet_single_mcts_deps.XXXXXX)
bash "$runtime_code/scripts/unpack_taiji_runtime.sh" \
  "$shared/artifacts/taiji_vllm_runtime/vllm_0_8_5_noncuda_dependencies_from_archive_20261014.tar.gz" \
  817efd049ae42972a4be38b1b43c9f975fb068050c8f52923cfcc85178bd389d "$dependencies"
cp "$runtime_code/scripts/vllm_cuda11_sitecustomize.py" "$vllm_local/sitecustomize.py"
export MECHET_VLLM_PUREPY_FALLBACK=$dependencies
rdkit_local=$(mktemp -d /tmp/mechet_single_mcts_rdkit.XXXXXX)
wheel=$shared/artifacts/wheels/rdkit-2026.3.4-cp311-cp311-manylinux_2_28_x86_64.whl
echo "a41dde42ecb24e7d93d62b89e8e5d267b02bbac764f965f3968296c99234c685  $wheel" | sha256sum --check --strict
python -m pip install --no-deps --no-index --target "$rdkit_local" "$wheel"
export PYTHONPATH=$rdkit_local:$vllm_local:$runtime_code/src:$runtime_code
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_CACHE=/aaa/fionafyang/buddy1/whaleywang/OpenEvolveChem/data/hf_cache
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_V1=0 NCCL_P2P_DISABLE=1
cd "$runtime_code"
if [[ ${MECHET_USE_RAY:-0} == 1 ]]; then
  ray_local=$(mktemp -d /tmp/mechet_mcts_ray.XXXXXX)
  ray_wheel=$shared/artifacts/wheels/ray_mcts_20261015/ray-2.44.1-cp311-cp311-manylinux2014_x86_64.whl
  echo "6e6bd0430d2eb664ae632c96e74c01e4a1bf14ab2a15102e1809b05ea9e0c2c7  $ray_wheel" | sha256sum --check --strict
  python -m pip install --no-deps --no-index --target "$ray_local" "$ray_wheel"
  export PYTHONPATH=$ray_local:$PYTHONPATH
fi
python -u - <<'PY'
import os,json,torch,vllm,rdkit
from pathlib import Path
from scripts.infer_single_reaction_mcts import SOURCE_SHA,ADAPTER_SHA,load_source,sha256,policy_cache_contract
from mechet.shared_electron_search import SharedSearchStore
shared=Path('/aaa/fionafyang/buddy1/whaleywang/MechET')
data=shared/'data/mech_uspto_31k_full_endpoint_rxnmapper/test.jsonl'
assert sha256(data)==SOURCE_SHA
ids={r['id'] for r in load_source(data,3120)}
actual=[]
for rank in range(8):
    path=Path(os.environ['MECHET_PRESERVED_GREEDY_OUTPUT'])/'greedy'/f'rank{rank:02d}'/'complete.json'
    sealed=json.loads(path.read_text());m=sealed['manifest']
    assert m['source_sha']==SOURCE_SHA and m['adapter_sha']==ADAPTER_SHA and m['mode']=='greedy'
    assert m['code_sha']=='f4b35ea094087f75e001b5fda355088a95be19f6'
    actual.extend(r['id'] for r in sealed['rows'])
assert len(actual)==len(set(actual))==3120 and set(actual)==ids
assert torch.version.cuda=='11.8' and vllm.__version__.split('+')[0]=='0.8.5'
assert rdkit.__version__=='2026.03.4' and torch.cuda.device_count()==8
assert all('A100' in torch.cuda.get_device_name(i) for i in range(8))
cache=SharedSearchStore(shared/'outputs/eval/shared_electron_search_v1',policy_cache_contract())
assert cache.namespace=='5df137df20153114e967310ed660270c7c471ccd443ceea1ad3a4d77652e8011'
print({'phase':'rolling-MCTS-preflight','full_denominator':3120,'preserved_greedy':'verified_no_regeneration',
       'gpus':8,'torch':torch.__version__,'vllm':vllm.__version__,'shared_cache':cache.namespace},flush=True)
PY
pids=()
server_pid=''
cleanup() {
  for pid in "${pids[@]}"; do if kill -0 "$pid" 2>/dev/null; then kill -TERM "$pid"; fi; done
  # Workers drain raw results/checkpoints before the tree owner is stopped.
  for pid in "${pids[@]}"; do wait "$pid" || true; done
  if [[ -n "$server_pid" ]]; then
    if kill -0 "$server_pid" 2>/dev/null; then kill -TERM "$server_pid"; fi
    wait "$server_pid" || true
  fi
}
trap cleanup EXIT
trap 'exit 130' TERM INT
extra_args=()
if [[ -n ${MECHET_TREE_RESUME_FROM:-} ]]; then
  extra_args+=(--resume-from "$MECHET_TREE_RESUME_FROM")
fi
if [[ ${MECHET_USE_RAY:-0} == 1 ]]; then
  python -u scripts/start_ray_mcts_actor.py --source-output "$MECHET_TREE_RESUME_FROM" \
    --output "$MECHET_TREE_OUTPUT" --address-file "$runtime_code/ray_address.json" &
  server_pid=$!
  while [[ ! -f "$runtime_code/ray_address.json" ]]; do
    if ! kill -0 "$server_pid" 2>/dev/null; then
      wait "$server_pid" || true
      echo '[Ray-MCTS] tree owner failed before publishing readiness' >&2
      exit 2
    fi
    sleep 1
  done
  export RAY_ADDRESS
  RAY_ADDRESS=$(python -c "import json;print(json.load(open('ray_address.json'))['address'])")
  ray_namespace=$(python -c "import json;print(json.load(open('ray_address.json'))['namespace'])")
  extra_args+=(--ray-actor single-reaction-tree --ray-namespace "$ray_namespace")
fi
for rank in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$rank python -u scripts/infer_single_reaction_mcts.py \
    --data "$shared/data/mech_uspto_31k_full_endpoint_rxnmapper/test.jsonl" \
    --adapter "$shared/outputs/agent/natural_language_event_history_v2_qwen3_0_6b_seed17" \
    --output "$MECHET_TREE_OUTPUT" --rank "$rank" --world 8 --program-goal 10 \
    --inflight-requests 128 --cpu-workers 16 \
    --shared-cache "$shared/outputs/eval/shared_electron_search_v1" \
    --mcts-shared-root "$shared/outputs/eval/shared_single_reaction_mcts_v1" "${extra_args[@]}" &
  pids+=("$!")
done
while (( ${#pids[@]}>0 )); do
  live=()
  if [[ -n "$server_pid" ]] && ! kill -0 "$server_pid" 2>/dev/null; then
    echo '[Ray-MCTS] shared tree owner stopped; preserve outputs for recovery' >&2
    exit 2
  fi
  for pid in "${pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then live+=("$pid")
    elif ! wait "$pid"; then
      echo '[rolling-MCTS] worker failed; preserve all outputs; stop sibling workers' >&2
      exit 2
    fi
  done
  pids=("${live[@]}")
  if (( ${#pids[@]}>0 )); then sleep 1; fi
done
if [[ -n "$server_pid" ]]; then
  kill -TERM "$server_pid"
  wait "$server_pid"
  server_pid=''
fi
trap - EXIT
python -u scripts/summarize_dynamic_electron_trees.py \
  --data "$shared/data/mech_uspto_31k_full_endpoint_rxnmapper/test.jsonl" --output "$MECHET_TREE_OUTPUT" --mode tree
echo '[rolling-MCTS] full3120 endpoint evaluation complete; preserved greedy untouched'
