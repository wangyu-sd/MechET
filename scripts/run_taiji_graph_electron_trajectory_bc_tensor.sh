#!/usr/bin/env bash
set -Eeuo pipefail

RUNTIME_DIR="${MECHET_GRAPH_RUNTIME_DIR:-/aaa/fionafyang/buddy1/whaleywang/MechET-graph-electron-iql-20260919}"
SHARED_REPO="/aaa/fionafyang/buddy1/whaleywang/MechET"
DATA_ROOT="$SHARED_REPO/data/graph_electron_two_track_v2_first_use"
TENSOR_ROOT="$SHARED_REPO/data/graph_electron_two_track_v2_first_use_tensor"
STAGE1_ROOT="$SHARED_REPO/outputs/agent/graph_electron_state_bc_v2_first_use_8a100_20260920"
STAGE1_CHECKPOINT="${GRAPH_STAGE1_CHECKPOINT:-$STAGE1_ROOT/checkpoint-epoch1-update5166.pt}"
OUTPUT_ROOT="$SHARED_REPO/outputs/agent/graph_electron_trajectory_bc_v2_first_use_8a100_20260920"
RDKIT_WHEEL="$SHARED_REPO/artifacts/wheels/rdkit-2026.3.4-cp311-cp311-manylinux_2_28_x86_64.whl"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate meteor
echo "a41dde42ecb24e7d93d62b89e8e5d267b02bbac764f965f3968296c99234c685  $RDKIT_WHEEL" | sha256sum --check --strict
RUNTIME_TARGET=$(mktemp -d /tmp/mechet_graph_trajectory_bc.XXXXXX)
python -m pip install --quiet --no-deps --target "$RUNTIME_TARGET" "$RDKIT_WHEEL"
cd "$RUNTIME_DIR"
export PYTHONPATH="$RUNTIME_TARGET:$RUNTIME_DIR/src:$RUNTIME_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

test -s "$DATA_ROOT/manifest.json"
test -s "$TENSOR_ROOT/manifest.json"
test -s "$STAGE1_CHECKPOINT"
echo "[graph-trajectory-bc] runtime=$RUNTIME_DIR commit=$(git rev-parse HEAD)"
echo "[graph-trajectory-bc] parent=$STAGE1_CHECKPOINT"
echo "[graph-trajectory-bc] starting compressed-history behavior cloning"

python -m torch.distributed.run --standalone --nproc_per_node=8 \
  scripts/train_graph_electron_full_batched.py \
  --data "$DATA_ROOT" \
  --tensor-cache "$TENSOR_ROOT" \
  --output "$OUTPUT_ROOT" \
  --stage trajectory_bc \
  --initialize-from "$STAGE1_CHECKPOINT" \
  --epochs 1 \
  --hidden-dim 192 \
  --layers 6 \
  --learning-rate 0.0001 \
  --batch-size 64 \
  --cpu-workers 2 \
  --prefetch 4 \
  --seed 17 \
  --log-updates 10 \
  --checkpoint-updates 1000
