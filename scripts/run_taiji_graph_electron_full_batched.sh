#!/usr/bin/env bash
set -Eeuo pipefail

RUNTIME_DIR="${MECHET_GRAPH_BATCHED_RUNTIME_DIR:-/aaa/fionafyang/buddy1/whaleywang/MechET-graph-electron-iql-20260919}"
SHARED_REPO="/aaa/fionafyang/buddy1/whaleywang/MechET"
SOURCE_ROOT="$SHARED_REPO/data/flower_inverse_tool_sft_action_delta_v1"
SOURCE_MANIFEST="$SOURCE_ROOT/training_manifest.json"
DATA_ROOT="$SHARED_REPO/data/graph_electron_direct_pointer_v1"
OUTPUT_ROOT="$SHARED_REPO/outputs/agent/graph_electron_direct_pointer_full_8a100_20260920"
RDKIT_WHEEL="$SHARED_REPO/artifacts/wheels/rdkit-2026.3.4-cp311-cp311-manylinux_2_28_x86_64.whl"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate meteor
echo "a41dde42ecb24e7d93d62b89e8e5d267b02bbac764f965f3968296c99234c685  $RDKIT_WHEEL" | sha256sum --check --strict
RUNTIME_TARGET=$(mktemp -d /tmp/mechet_graph_batched_runtime.XXXXXX)
python -m pip install --quiet --no-deps --target "$RUNTIME_TARGET" "$RDKIT_WHEEL"
cd "$RUNTIME_DIR"
export PYTHONPATH="$RUNTIME_TARGET:$RUNTIME_DIR/src:$RUNTIME_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

echo "[graph-batched] runtime=$RUNTIME_DIR"
echo "[graph-batched] commit=$(git rev-parse HEAD)"
echo "[graph-batched] gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ';')"
if [[ ! -f "$DATA_ROOT/manifest.json" ]]; then
  CPUS=$(getconf _NPROCESSORS_ONLN)
  if (( CPUS > 48 )); then CPUS=48; fi
  echo "[graph-direct] building direct-pointer artifact workers=$CPUS"
  python scripts/build_graph_electron_full.py \
    --source-root "$SOURCE_ROOT" \
    --source-manifest "$SOURCE_MANIFEST" \
    --output "$DATA_ROOT" \
    --workers "$CPUS" \
    --shards 8
fi

python - "$DATA_ROOT/manifest.json" <<'PY'
import json,sys,rdkit,torch
names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
if len(names) != 8 or any('A100' not in name.upper() for name in names):
    raise SystemExit(f'expected 8xA100, got {names}')
if rdkit.__version__ != '2026.03.4':
    raise SystemExit(f'expected RDKit 2026.03.4, got {rdkit.__version__}')
m=json.load(open(sys.argv[1])); expected={'train':257167,'valid':2890,'test':28967}
if m.get('reaction_denominator') != expected or m.get('shard_count') != 8:
    raise SystemExit('graph decision manifest does not satisfy the frozen full contract')
if m['splits']['train']['decisions'] != 2644501:
    raise SystemExit('unexpected train decision count')
if m.get('artifact_type') != 'graph_electron_direct_pointer_decisions':
    raise SystemExit('expected no-enumeration direct-pointer artifact')
contract=m.get('action_contract') or {}
if (contract.get('flow') != 'direct_conditional_node_pointers_no_candidate_inventory'
        or contract.get('imports') != 'open_graph_program_for_environment_and_reactive_fragments'):
    raise SystemExit('direct action contract mismatch')
print({'runtime_gate':'passed','gpus':names,'rdkit':rdkit.__version__,
       'train_decisions':m['splits']['train']['decisions'],
       'action_contract':contract},flush=True)
PY

echo "[graph-direct] starting no-enumeration direct-pointer 8-GPU training"
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  scripts/train_graph_electron_full_batched.py \
  --data "$DATA_ROOT" \
  --output "$OUTPUT_ROOT" \
  --epochs 3 \
  --hidden-dim 192 \
  --layers 6 \
  --learning-rate 0.0003 \
  --batch-size 32 \
  --cpu-workers 6 \
  --prefetch 12 \
  --seed 17 \
  --log-updates 10 \
  --checkpoint-updates 1000
