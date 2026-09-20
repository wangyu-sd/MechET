#!/usr/bin/env bash
set -Eeuo pipefail

RUNTIME_DIR="${MECHET_GRAPH_RUNTIME_DIR:-/aaa/fionafyang/buddy1/whaleywang/MechET-graph-electron-iql-20260919}"
SHARED_REPO="/aaa/fionafyang/buddy1/whaleywang/MechET"
SOURCE_ROOT="$SHARED_REPO/data/flower_inverse_tool_sft_action_delta_v1"
SOURCE_MANIFEST="$SOURCE_ROOT/training_manifest.json"
DATA_ROOT="$SHARED_REPO/data/graph_electron_two_track_v1"
TENSOR_ROOT="$SHARED_REPO/data/graph_electron_two_track_v1_tensor"
OUTPUT_ROOT="$SHARED_REPO/outputs/agent/graph_electron_state_bc_tensor_8a100_20260920"
RDKIT_WHEEL="$SHARED_REPO/artifacts/wheels/rdkit-2026.3.4-cp311-cp311-manylinux_2_28_x86_64.whl"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate meteor
echo "a41dde42ecb24e7d93d62b89e8e5d267b02bbac764f965f3968296c99234c685  $RDKIT_WHEEL" | sha256sum --check --strict
RUNTIME_TARGET=$(mktemp -d /tmp/mechet_graph_state_bc.XXXXXX)
python -m pip install --quiet --no-deps --target "$RUNTIME_TARGET" "$RDKIT_WHEEL"
cd "$RUNTIME_DIR"
export PYTHONPATH="$RUNTIME_TARGET:$RUNTIME_DIR/src:$RUNTIME_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

echo "[graph-state-bc] runtime=$RUNTIME_DIR commit=$(git rev-parse HEAD)"
echo "[graph-state-bc] gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ';')"

if [[ ! -f "$DATA_ROOT/manifest.json" ]]; then
  CPUS=$(getconf _NPROCESSORS_ONLN)
  if (( CPUS > 48 )); then CPUS=48; fi
  echo "[graph-state-bc] building schema-v3 canonical decisions workers=$CPUS"
  python scripts/build_graph_electron_full.py \
    --source-root "$SOURCE_ROOT" \
    --source-manifest "$SOURCE_MANIFEST" \
    --output "$DATA_ROOT" \
    --workers "$CPUS" \
    --shards 8
fi

if [[ ! -f "$TENSOR_ROOT/manifest.json" ]]; then
  echo "[graph-state-bc] compiling audit JSONL to graph tensor cache"
  python scripts/build_graph_electron_tensor_cache.py \
    --data "$DATA_ROOT" \
    --output "$TENSOR_ROOT" \
    --splits train \
    --workers 24 \
    --chunk-size 2048
fi

python - "$DATA_ROOT/manifest.json" "$TENSOR_ROOT/manifest.json" <<'PY'
import json,sys,rdkit,torch
decisions=json.load(open(sys.argv[1])); cache=json.load(open(sys.argv[2]))
expected={'train':257167,'valid':2890,'test':28967}
assert decisions['schema_version'] == 3
assert decisions['reaction_denominator'] == expected
assert decisions['splits']['train']['decisions'] == 2644501
assert decisions['action_contract']['flow'] == 'direct_conditional_node_pointers_no_candidate_inventory'
assert decisions['policy_protocol']['tracks'] == ['llm','graph']
assert cache['splits']['train']['decisions'] == 2644501
names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
assert len(names) == 8 and all('A100' in name.upper() for name in names), names
assert rdkit.__version__ == '2026.03.4'
print({'runtime_gate':'passed','stage':'state_bc','loader':'tensor_cache',
       'candidate_enumeration':False,'train_decisions':2644501,'gpus':names},flush=True)
PY

echo "[graph-state-bc] starting stage-1 state-only behavior cloning"
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  scripts/train_graph_electron_full_batched.py \
  --data "$DATA_ROOT" \
  --tensor-cache "$TENSOR_ROOT" \
  --output "$OUTPUT_ROOT" \
  --stage state_bc \
  --epochs 1 \
  --hidden-dim 192 \
  --layers 6 \
  --learning-rate 0.0003 \
  --batch-size 64 \
  --cpu-workers 2 \
  --prefetch 4 \
  --seed 17 \
  --log-updates 10 \
  --checkpoint-updates 1000
