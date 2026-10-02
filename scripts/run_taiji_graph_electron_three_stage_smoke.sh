#!/usr/bin/env bash
set -Eeuo pipefail

RUNTIME_DIR="${MECHET_GRAPH_RUNTIME_DIR:-/aaa/fionafyang/buddy1/whaleywang/MechET-graph-electron-iql-20260919}"
SHARED_REPO="/aaa/fionafyang/buddy1/whaleywang/MechET"
SOURCE_ROOT="$SHARED_REPO/data/flower_inverse_tool_sft_action_delta_v1"
SOURCE_MANIFEST="$SOURCE_ROOT/training_manifest.json"
RESUME_CHECKPOINT="$SHARED_REPO/outputs/agent/graph_electron_reaction_online_v1_8a100_20260920/checkpoint-epoch1-update500.pt"
VERIFIED_INDEX="$SHARED_REPO/outputs/agent/graph_electron_reaction_online_v1_8a100_20260920/train.reaction_offsets.u64"
SMOKE_ROOT="$SHARED_REPO/outputs/agent/graph_electron_three_stage_smoke_v3"
STATE_ROOT="$SMOKE_ROOT/state_bc"
TRAJECTORY_ROOT="$SMOKE_ROOT/trajectory_bc"
RL_ROOT="$SMOKE_ROOT/executor_rl"
RDKIT_WHEEL="$SHARED_REPO/artifacts/wheels/rdkit-2026.3.4-cp311-cp311-manylinux_2_28_x86_64.whl"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate meteor
echo "a41dde42ecb24e7d93d62b89e8e5d267b02bbac764f965f3968296c99234c685  $RDKIT_WHEEL" | sha256sum --check --strict
RUNTIME_TARGET=$(mktemp -d /tmp/mechet_graph_three_stage_smoke.XXXXXX)
python -m pip install --quiet --no-deps --target "$RUNTIME_TARGET" "$RDKIT_WHEEL"
cd "$RUNTIME_DIR"
export PYTHONPATH="$RUNTIME_TARGET:$RUNTIME_DIR/src:$RUNTIME_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export MKL_THREADING_LAYER=GNU
export OMP_NUM_THREADS=1

mkdir -p "$STATE_ROOT" "$TRAJECTORY_ROOT" "$RL_ROOT"
existing_checkpoint=$(find "$SMOKE_ROOT" -mindepth 2 -type f -name 'checkpoint-*.pt' -print -quit)
if [[ -n "$existing_checkpoint" ]]; then
  echo "[graph-three-stage-smoke] refusing non-fresh output root: $SMOKE_ROOT" >&2
  exit 2
fi
# Keep terminal output live for Taiji and a persistent copy for post-mortem
# diagnosis if the pod disappears. This is teeing, not log redirection.
exec > >(tee -a "$SMOKE_ROOT/runtime.log") 2>&1
trap 'status=$?; echo "[graph-three-stage-smoke] exit_status=$status time=$(date --iso-8601=seconds)"; trap - EXIT; exit "$status"' EXIT
echo "[graph-three-stage-smoke] runtime=$RUNTIME_DIR commit=$(git rev-parse HEAD)"
echo "[graph-three-stage-smoke] gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ';')"
python - "$SOURCE_MANIFEST" "$RESUME_CHECKPOINT" "$VERIFIED_INDEX" <<'PY'
import json,os,sys,torch
from pathlib import Path
manifest=json.load(open(sys.argv[1])); checkpoint=torch.load(sys.argv[2],map_location='cpu',weights_only=False)
index=Path(sys.argv[3]); resume=Path(sys.argv[2])
assert manifest['strict_trace_universe_complete'] is True
assert {s:manifest['splits'][s]['rows'] for s in ('train','valid','test')} == {'train':257167,'valid':2890,'test':28967}
assert checkpoint['update']==500 and checkpoint['config']['stage']=='state_bc'
assert checkpoint['source_manifest']['splits']['train']['sha256'] == manifest['splits']['train']['sha256']
assert index.parent == resume.parent and index.is_file() and index.stat().st_size == 257167 * 8
names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
expected=os.environ.get('MECHET_EXPECTED_GPU','H20').upper()
assert len(names)==8 and all(expected in name.upper() for name in names),(expected,names)
print({'runtime_gate':'passed','mode':'three_stage_smoke','gpus':names},flush=True)
PY

echo "[graph-three-stage-smoke] stage=1 state-SFT updates=8 resume=500"
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  scripts/train_graph_electron_reaction_online.py \
  --source-root "$SOURCE_ROOT" --source-manifest "$SOURCE_MANIFEST" \
  --verified-offset-index "$VERIFIED_INDEX" \
  --output "$STATE_ROOT" --stage state_bc --epochs 1 \
  --resume-from "$RESUME_CHECKPOINT" --resume-update 500 --max-updates 8 \
  --hidden-dim 192 --layers 6 --learning-rate 0.0003 \
  --reactions-per-batch 6 --cpu-workers 2 --prefetch 2 --seed 17 \
  --log-updates 1 --checkpoint-updates 8

STATE_FINAL="$STATE_ROOT/checkpoint-epoch1-update508.pt"
test -f "$STATE_FINAL"
echo "[graph-three-stage-smoke] stage=2 trajectory-SFT updates=8 initialization=$STATE_FINAL"
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  scripts/train_graph_electron_reaction_online.py \
  --source-root "$SOURCE_ROOT" --source-manifest "$SOURCE_MANIFEST" \
  --verified-offset-index "$VERIFIED_INDEX" \
  --output "$TRAJECTORY_ROOT" --stage trajectory_bc --epochs 1 \
  --initialize-from "$STATE_FINAL" --max-updates 8 \
  --hidden-dim 192 --layers 6 --learning-rate 0.0001 \
  --reactions-per-batch 6 --cpu-workers 2 --prefetch 2 --seed 29 \
  --log-updates 1 --checkpoint-updates 8

TRAJECTORY_FINAL="$TRAJECTORY_ROOT/checkpoint-epoch1-update8.pt"
test -f "$TRAJECTORY_FINAL"
echo "[graph-three-stage-smoke] stage=3 executor-RL episodes=64 initialization=$TRAJECTORY_FINAL"
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  scripts/train_graph_electron_executor_rl.py \
  --source-root "$SOURCE_ROOT" --source-manifest "$SOURCE_MANIFEST" \
  --verified-offset-index "$VERIFIED_INDEX" \
  --initialize-from "$TRAJECTORY_FINAL" --output "$RL_ROOT" \
  --episodes 64 --episodes-per-update 1 --max-steps 12 \
  --learning-rate 0.000005 --discount 0.97 --invalid-penalty 1.0 \
  --value-weight 0.5 --bc-weight 0.05 --temperature 1.0 --seed 41 \
  --log-updates 1 --checkpoint-updates 8

python - "$STATE_FINAL" "$TRAJECTORY_FINAL" "$RL_ROOT/train_report.json" <<'PY'
import json,math,sys
from pathlib import Path
state,trajectory,report_path=map(Path,sys.argv[1:])
assert state.is_file() and trajectory.is_file()
report=json.load(report_path.open())
assert report['status']=='completed' and report['episodes']==64,report
assert report['steps'] >= 64,report
assert report['invalid'] > 0,report
assert report['penalized_invalid'] > 0,report
assert math.isclose(report['invalid_penalty_reward'], -float(report['penalized_invalid']), abs_tol=1e-6),report
print({'smoke_gate':'passed','state_checkpoint':str(state),'trajectory_checkpoint':str(trajectory),'rl':report},flush=True)
PY
echo "[graph-three-stage-smoke] all stages and reward gates passed"
