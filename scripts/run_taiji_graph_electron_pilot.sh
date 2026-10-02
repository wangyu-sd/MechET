#!/usr/bin/env bash
set -euo pipefail

RUNTIME_DIR="${MECHET_GRAPH_POLICY_RUNTIME_DIR:-/aaa/fionafyang/buddy1/whaleywang/MechET-graph-electron-iql-20260919}"
SHARED_REPO="/aaa/fionafyang/buddy1/whaleywang/MechET"
DATA_ROOT="/aaa/fionafyang/buddy1/whaleywang/MechET/data/flower_inverse_tool_sft_action_delta_v1"
OUTPUT_ROOT="/aaa/fionafyang/buddy1/whaleywang/MechET/outputs/agent/graph_electron_iql_bc_pilot_20260919"
RDKIT_WHEEL="$SHARED_REPO/artifacts/wheels/rdkit-2026.3.4-cp311-cp311-manylinux_2_28_x86_64.whl"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate meteor
echo "a41dde42ecb24e7d93d62b89e8e5d267b02bbac764f965f3968296c99234c685  $RDKIT_WHEEL" | sha256sum --check --strict
RUNTIME_TARGET=$(mktemp -d /tmp/mechet_graph_policy_runtime.XXXXXX)
python -m pip install --quiet --no-deps --target "$RUNTIME_TARGET" "$RDKIT_WHEEL"
cd "$RUNTIME_DIR"
export PYTHONPATH="$RUNTIME_TARGET:$RUNTIME_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

echo "[graph-electron] runtime=$RUNTIME_DIR"
echo "[graph-electron] commit=$(git rev-parse HEAD)"
echo "[graph-electron] gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
python - <<'PY'
import rdkit, torch
if torch.cuda.device_count() != 1:
    raise SystemExit(f"expected one GPU, got {torch.cuda.device_count()}")
name = torch.cuda.get_device_name(0)
if "A100" not in name.upper():
    raise SystemExit(f"expected A100, got {name}")
if rdkit.__version__ != "2026.03.4":
    raise SystemExit(f"expected RDKit 2026.03.4, got {rdkit.__version__}")
print({"gpu": name, "rdkit": rdkit.__version__, "torch": torch.__version__}, flush=True)
PY

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
