#!/usr/bin/env bash
set -euo pipefail

RUNTIME_DIR="${MECHET_GRAPH_POLICY_RUNTIME_DIR:-/aaa/fionafyang/buddy1/whaleywang/MechET-graph-electron-iql-20260919}"
DATA_ROOT="/aaa/fionafyang/buddy1/whaleywang/MechET/data/flower_inverse_tool_sft_action_delta_v1"
OUTPUT_ROOT="/aaa/fionafyang/buddy1/whaleywang/MechET/outputs/agent/graph_electron_iql_bc_pilot_20260919"

cd "$RUNTIME_DIR"
export PYTHONPATH="$RUNTIME_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

echo "[graph-electron] runtime=$RUNTIME_DIR"
echo "[graph-electron] commit=$(git rev-parse HEAD)"
echo "[graph-electron] gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"

python scripts/train_graph_electron_pilot.py \
  --train "$DATA_ROOT/train.jsonl" \
  --manifest "$DATA_ROOT/training_manifest.json" \
  --output "$OUTPUT_ROOT" \
  --reaction-limit 512 \
  --decision-limit 4096 \
  --bank-rows 50000 \
  --bank-size 5000 \
  --import-negatives 31 \
  --hidden-dim 192 \
  --layers 6 \
  --epochs 3 \
  --learning-rate 0.0003 \
  --accumulate 16 \
  --eval-decisions 256 \
  --seed 17 \
  --heartbeat 45

