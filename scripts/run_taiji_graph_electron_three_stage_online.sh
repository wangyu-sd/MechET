#!/usr/bin/env bash
set -Eeuo pipefail

RUNTIME_DIR="${MECHET_GRAPH_RUNTIME_DIR:-/aaa/fionafyang/buddy1/whaleywang/MechET-graph-electron-iql-20260919}"
SHARED_REPO="/aaa/fionafyang/buddy1/whaleywang/MechET"
SOURCE_ROOT="$SHARED_REPO/data/flower_inverse_tool_sft_action_delta_v1"
SOURCE_MANIFEST="$SOURCE_ROOT/training_manifest.json"
RESUME_CHECKPOINT="$SHARED_REPO/outputs/agent/graph_electron_reaction_online_v1_8a100_20260920/checkpoint-epoch1-update500.pt"
STATE_ROOT="$SHARED_REPO/outputs/agent/graph_electron_three_stage_v1/state_bc"
TRAJECTORY_ROOT="$SHARED_REPO/outputs/agent/graph_electron_three_stage_v1/trajectory_bc"
RL_ROOT="$SHARED_REPO/outputs/agent/graph_electron_three_stage_v1/executor_rl"
RDKIT_WHEEL="$SHARED_REPO/artifacts/wheels/rdkit-2026.3.4-cp311-cp311-manylinux_2_28_x86_64.whl"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate meteor
echo "a41dde42ecb24e7d93d62b89e8e5d267b02bbac764f965f3968296c99234c685  $RDKIT_WHEEL" | sha256sum --check --strict
RUNTIME_TARGET=$(mktemp -d /tmp/mechet_graph_three_stage.XXXXXX)
python -m pip install --quiet --no-deps --target "$RUNTIME_TARGET" "$RDKIT_WHEEL"
cd "$RUNTIME_DIR"
export PYTHONPATH="$RUNTIME_TARGET:$RUNTIME_DIR/src:$RUNTIME_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export MKL_THREADING_LAYER=GNU
export OMP_NUM_THREADS=1

echo "[graph-three-stage] runtime=$RUNTIME_DIR commit=$(git rev-parse HEAD)"
echo "[graph-three-stage] gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ';')"
python - "$SOURCE_MANIFEST" "$RESUME_CHECKPOINT" "$STATE_ROOT" "$TRAJECTORY_ROOT" "$RL_ROOT" <<'PY'
import json,os,sys,torch
from pathlib import Path
manifest=json.load(open(sys.argv[1])); resume=Path(sys.argv[2]); outputs=[Path(x) for x in sys.argv[3:]]
assert manifest['strict_trace_universe_complete'] is True
assert {s:manifest['splits'][s]['rows'] for s in ('train','valid','test')} == {'train':257167,'valid':2890,'test':28967}
assert resume.is_file()
checkpoint=torch.load(resume,map_location='cpu',weights_only=False)
assert checkpoint['update']==500 and checkpoint['config']['stage']=='state_bc'
names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
expected_gpu=os.environ.get('MECHET_EXPECTED_GPU','A100').upper()
assert len(names)==8 and all(expected_gpu in name.upper() for name in names),(expected_gpu,names)
for output in outputs:
    if output.exists() and list(output.glob('checkpoint-*.pt')):
        raise SystemExit(f'fresh stage output already contains checkpoints: {output}')
    output.mkdir(parents=True,exist_ok=True)
print({'runtime_gate':'passed','pipeline':['state_bc_resume','trajectory_bc','online_executor_rl'],
       'resume_update':500,'expanded_decisions_on_disk':0,'expected_gpu':expected_gpu,'gpus':names},flush=True)
PY

echo "[graph-three-stage] stage=1 state-SFT resume update=500"
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  scripts/train_graph_electron_reaction_online.py \
  --source-root "$SOURCE_ROOT" --source-manifest "$SOURCE_MANIFEST" \
  --output "$STATE_ROOT" --stage state_bc --epochs 1 \
  --resume-from "$RESUME_CHECKPOINT" --resume-update 500 \
  --hidden-dim 192 --layers 6 --learning-rate 0.0003 \
  --reactions-per-batch 6 --cpu-workers 4 --prefetch 3 --seed 17 \
  --log-updates 10 --checkpoint-updates 500

STATE_FINAL="$STATE_ROOT/checkpoint-epoch1-update5358.pt"
test -f "$STATE_FINAL"
echo "[graph-three-stage] stage=2 trajectory-SFT initialization=$STATE_FINAL"
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  scripts/train_graph_electron_reaction_online.py \
  --source-root "$SOURCE_ROOT" --source-manifest "$SOURCE_MANIFEST" \
  --output "$TRAJECTORY_ROOT" --stage trajectory_bc --epochs 1 \
  --initialize-from "$STATE_FINAL" \
  --hidden-dim 192 --layers 6 --learning-rate 0.0001 \
  --reactions-per-batch 6 --cpu-workers 4 --prefetch 3 --seed 29 \
  --log-updates 10 --checkpoint-updates 500

TRAJECTORY_FINAL="$TRAJECTORY_ROOT/checkpoint-epoch1-update5358.pt"
test -f "$TRAJECTORY_FINAL"
echo "[graph-three-stage] stage=3 online executor RL initialization=$TRAJECTORY_FINAL"
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  scripts/train_graph_electron_executor_rl.py \
  --source-root "$SOURCE_ROOT" --source-manifest "$SOURCE_MANIFEST" \
  --initialize-from "$TRAJECTORY_FINAL" --output "$RL_ROOT" \
  --episodes 1024 --episodes-per-update 1 --max-steps 12 \
  --learning-rate 0.000005 --discount 0.97 --invalid-penalty 1.0 \
  --value-weight 0.5 --bc-weight 0.05 --temperature 1.0 --seed 41 \
  --log-updates 5 --checkpoint-updates 25

echo "[graph-three-stage] all stages completed"
