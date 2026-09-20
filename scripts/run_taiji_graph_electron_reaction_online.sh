#!/usr/bin/env bash
set -Eeuo pipefail

RUNTIME_DIR="${MECHET_GRAPH_RUNTIME_DIR:-/aaa/fionafyang/buddy1/whaleywang/MechET-graph-electron-iql-20260919}"
SHARED_REPO="/aaa/fionafyang/buddy1/whaleywang/MechET"
SOURCE_ROOT="$SHARED_REPO/data/flower_inverse_tool_sft_action_delta_v1"
SOURCE_MANIFEST="$SOURCE_ROOT/training_manifest.json"
OUTPUT_ROOT="$SHARED_REPO/outputs/agent/graph_electron_reaction_online_v1_8a100_20260920"
RDKIT_WHEEL="$SHARED_REPO/artifacts/wheels/rdkit-2026.3.4-cp311-cp311-manylinux_2_28_x86_64.whl"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate meteor
echo "a41dde42ecb24e7d93d62b89e8e5d267b02bbac764f965f3968296c99234c685  $RDKIT_WHEEL" | sha256sum --check --strict
RUNTIME_TARGET=$(mktemp -d /tmp/mechet_graph_online.XXXXXX)
python -m pip install --quiet --no-deps --target "$RUNTIME_TARGET" "$RDKIT_WHEEL"
cd "$RUNTIME_DIR"
export PYTHONPATH="$RUNTIME_TARGET:$RUNTIME_DIR/src:$RUNTIME_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export MKL_THREADING_LAYER=GNU
export OMP_NUM_THREADS=1

echo "[graph-online-launch] runtime=$RUNTIME_DIR commit=$(git rev-parse HEAD)"
echo "[graph-online-launch] gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ';')"
python - "$SOURCE_MANIFEST" "$OUTPUT_ROOT" <<'PY'
import json,sys,torch
from pathlib import Path
manifest=json.load(open(sys.argv[1])); output=Path(sys.argv[2])
expected={'train':257167,'valid':2890,'test':28967}
hashes={
 'train':'edc80c5c5eb13d50753c5566c5d6ac1b90b955b1a8dfece915241b2da4a40a75',
 'valid':'7303a6018850db61594af5854df936778c21a6668e61151f95e2c74c32d22d2a',
 'test':'70a06f01a08490d4b5da602056a8f8e2f978128d3da3224d1601186529aeee3f'}
assert manifest['strict_trace_universe_complete'] is True
for split,count in expected.items():
    assert manifest['splits'][split]['rows'] == count
    assert manifest['splits'][split]['sha256'] == hashes[split]
names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
assert len(names)==8 and all('A100' in name.upper() for name in names),names
if output.exists() and list(output.glob('checkpoint-*.pt')):
    raise SystemExit('fresh online output already contains checkpoints')
output.mkdir(parents=True,exist_ok=True)
print({'runtime_gate':'passed','data_contract':'reaction_level_online_expansion_v1',
       'reactions':expected,'expanded_decisions_on_disk':0,'gpus':names},flush=True)
PY

echo "[graph-online-launch] starting state-SFT directly from frozen reaction traces"
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  scripts/train_graph_electron_reaction_online.py \
  --source-root "$SOURCE_ROOT" \
  --source-manifest "$SOURCE_MANIFEST" \
  --output "$OUTPUT_ROOT" \
  --stage state_bc \
  --epochs 1 \
  --hidden-dim 192 \
  --layers 6 \
  --learning-rate 0.0003 \
  --reactions-per-batch 6 \
  --cpu-workers 4 \
  --prefetch 3 \
  --seed 17 \
  --log-updates 10 \
  --checkpoint-updates 500
