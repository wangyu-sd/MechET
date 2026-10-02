#!/usr/bin/env bash
set -Eeuo pipefail

RUNTIME_DIR="${MECHET_GRAPH_FULL_RUNTIME_DIR:-/aaa/fionafyang/buddy1/whaleywang/MechET-graph-electron-iql-20260919}"
SHARED_REPO="/aaa/fionafyang/buddy1/whaleywang/MechET"
SOURCE_ROOT="$SHARED_REPO/data/flower_inverse_tool_sft_action_delta_v1"
DATA_ROOT="$SHARED_REPO/data/graph_electron_dual_import_v1"
OUTPUT_ROOT="$SHARED_REPO/outputs/agent/graph_electron_dual_import_full_8a100_20260919"
RDKIT_WHEEL="$SHARED_REPO/artifacts/wheels/rdkit-2026.3.4-cp311-cp311-manylinux_2_28_x86_64.whl"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate meteor
echo "a41dde42ecb24e7d93d62b89e8e5d267b02bbac764f965f3968296c99234c685  $RDKIT_WHEEL" | sha256sum --check --strict
RUNTIME_TARGET=$(mktemp -d /tmp/mechet_graph_full_runtime.XXXXXX)
python -m pip install --quiet --no-deps --target "$RUNTIME_TARGET" "$RDKIT_WHEEL"
cd "$RUNTIME_DIR"
export PYTHONPATH="$RUNTIME_TARGET:$RUNTIME_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2

echo "[graph-full] runtime=$RUNTIME_DIR"
echo "[graph-full] commit=$(git rev-parse HEAD)"
echo "[graph-full] gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ';')"
python - <<'PY'
import rdkit, torch
names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
if len(names) != 8 or any('A100' not in name.upper() for name in names):
    raise SystemExit(f'expected 8xA100, got {names}')
if rdkit.__version__ != '2026.03.4':
    raise SystemExit(f'expected RDKit 2026.03.4, got {rdkit.__version__}')
print({'runtime_gate':'passed','gpus':names,'rdkit':rdkit.__version__,'torch':torch.__version__},flush=True)
PY

if [[ -f "$DATA_ROOT/manifest.json" ]]; then
  python - "$DATA_ROOT/manifest.json" <<'PY'
import json,sys
m=json.load(open(sys.argv[1]))
expected={'train':257167,'valid':2890,'test':28967}
if m.get('reaction_denominator') != expected or m.get('shard_count') != 8:
    raise SystemExit('existing graph decision manifest does not satisfy the frozen full contract')
for split,count in expected.items():
    if m['splits'][split]['reactions'] != count or len(m['splits'][split]['shards']) != 8:
        raise SystemExit(f'existing {split} decision artifact is incomplete')
print({'graph_dataset':'reused_verified','manifest':sys.argv[1]},flush=True)
PY
else
  CPUS=$(getconf _NPROCESSORS_ONLN)
  if (( CPUS > 48 )); then CPUS=48; fi
  echo "[graph-full] building complete decision artifact workers=$CPUS"
  python scripts/build_graph_electron_full.py \
    --source-root "$SOURCE_ROOT" \
    --source-manifest "$SOURCE_ROOT/training_manifest.json" \
    --output "$DATA_ROOT" \
    --workers "$CPUS" \
    --shards 8
fi

echo "[graph-full] starting synchronized 8-GPU training"
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  scripts/train_graph_electron_full.py \
  --data "$DATA_ROOT" \
  --output "$OUTPUT_ROOT" \
  --epochs 3 \
  --hidden-dim 192 \
  --layers 6 \
  --learning-rate 0.0003 \
  --accumulate 16 \
  --import-negatives 31 \
  --seed 17 \
  --log-updates 50
