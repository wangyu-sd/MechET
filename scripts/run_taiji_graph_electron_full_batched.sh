#!/usr/bin/env bash
set -Eeuo pipefail

RUNTIME_DIR="${MECHET_GRAPH_BATCHED_RUNTIME_DIR:-/aaa/fionafyang/buddy1/whaleywang/MechET-graph-electron-iql-20260919}"
SHARED_REPO="/aaa/fionafyang/buddy1/whaleywang/MechET"
DATA_ROOT="$SHARED_REPO/data/graph_electron_dual_import_v1"
OUTPUT_ROOT="$SHARED_REPO/outputs/agent/graph_electron_dual_import_full_multiprocess_8a100_20260920"
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
python - "$DATA_ROOT/manifest.json" "$DATA_ROOT/environment_fragment_bank.json" <<'PY'
import hashlib,json,sys,rdkit,torch
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
env=m['environment_fragment_bank']; bank=json.load(open(sys.argv[2]))
digest=hashlib.sha256(open(sys.argv[2],'rb').read()).hexdigest()
if (env.get('occurrences'),env.get('unique'),len(bank),len(set(bank)),digest) != (
        743579,6964,6964,6964,env.get('sha256')):
    raise SystemExit('IMPORT_ENV train-only catalog is incomplete or corrupted')
print({'runtime_gate':'passed','gpus':names,'rdkit':rdkit.__version__,
       'train_decisions':m['splits']['train']['decisions'],
       'IMPORT_ENV_occurrences':env['occurrences'],
       'IMPORT_ENV_unique_train_only':env['unique']},flush=True)
PY

echo "[graph-batched] starting packed-graph 8-GPU training with 48 process workers"
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
  --import-negatives 31 \
  --seed 17 \
  --log-updates 10 \
  --checkpoint-updates 1000
