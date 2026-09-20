#!/usr/bin/env bash
set -Eeuo pipefail

RUNTIME_DIR="${MECHET_GRAPH_RUNTIME_DIR:-/aaa/fionafyang/buddy1/whaleywang/MechET-graph-electron-iql-20260919}"
SHARED_REPO="/aaa/fionafyang/buddy1/whaleywang/MechET"
SOURCE_ROOT="$SHARED_REPO/data/flower_inverse_tool_sft_action_delta_v1"
SOURCE_MANIFEST="$SOURCE_ROOT/training_manifest.json"
INITIAL_CHECKPOINT="$SHARED_REPO/outputs/agent/graph_electron_reaction_online_v1_8a100_20260920/checkpoint-epoch1-update500.pt"
VERIFIED_INDEX="$SHARED_REPO/outputs/agent/graph_electron_reaction_online_v1_8a100_20260920/train.reaction_offsets.u64"
RUN_ROOT="$SHARED_REPO/outputs/agent/graph_electron_overfit_smoke_1v100_v1"
STATE_ROOT="$RUN_ROOT/state_bc"
TRAJECTORY_ROOT="$RUN_ROOT/trajectory_bc"
RL_ROOT="$RUN_ROOT/executor_rl"
EVAL_ROOT="$RUN_ROOT/eval"
TRAIN_SUBSET="$RUN_ROOT/train_overfit32.jsonl"
RDKIT_WHEEL="$SHARED_REPO/artifacts/wheels/rdkit-2026.3.4-cp311-cp311-manylinux_2_28_x86_64.whl"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate meteor
echo "a41dde42ecb24e7d93d62b89e8e5d267b02bbac764f965f3968296c99234c685  $RDKIT_WHEEL" | sha256sum --check --strict
RUNTIME_TARGET=$(mktemp -d /tmp/mechet_graph_overfit_smoke.XXXXXX)
python -m pip install --quiet --no-deps --target "$RUNTIME_TARGET" "$RDKIT_WHEEL"
cd "$RUNTIME_DIR"
export PYTHONPATH="$RUNTIME_TARGET:$RUNTIME_DIR/src:$RUNTIME_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export MKL_THREADING_LAYER=GNU
export OMP_NUM_THREADS=2

mkdir -p "$STATE_ROOT" "$TRAJECTORY_ROOT" "$RL_ROOT" "$EVAL_ROOT"
existing_checkpoint=$(find "$RUN_ROOT" -mindepth 2 -type f -name 'checkpoint-*.pt' -print -quit)
if [[ -n "$existing_checkpoint" ]]; then
  echo "[graph-overfit-smoke] refusing non-fresh output root: $RUN_ROOT" >&2
  exit 2
fi
exec > >(tee -a "$RUN_ROOT/runtime.log") 2>&1
trap 'status=$?; echo "[graph-overfit-smoke] exit_status=$status time=$(date --iso-8601=seconds)"; trap - EXIT; exit "$status"' EXIT
echo "[graph-overfit-smoke] runtime=$RUNTIME_DIR commit=$(git rev-parse HEAD)"
echo "[graph-overfit-smoke] gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ';')"

python - "$SOURCE_MANIFEST" "$INITIAL_CHECKPOINT" "$VERIFIED_INDEX" <<'PY'
import json, os, sys, torch
from pathlib import Path
manifest=json.load(open(sys.argv[1])); checkpoint=torch.load(sys.argv[2],map_location='cpu',weights_only=False)
index=Path(sys.argv[3])
assert manifest['strict_trace_universe_complete'] is True
assert {s:manifest['splits'][s]['rows'] for s in ('train','valid','test')} == {'train':257167,'valid':2890,'test':28967}
assert checkpoint['update']==500 and checkpoint['config']['stage']=='state_bc'
assert checkpoint['source_manifest']['splits']['train']['sha256'] == manifest['splits']['train']['sha256']
assert index.is_file() and index.stat().st_size == 257167 * 8
names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
expected=os.environ.get('MECHET_EXPECTED_GPU','V100').upper()
assert len(names)==1 and expected in names[0].upper(),(expected,names)
print({'runtime_gate':'passed','mode':'fixed_train_overfit_then_holdout','gpus':names},flush=True)
PY

# Materialize exactly the same shuffled 32 train reactions selected by the
# single-rank trainers. This gives the in-sample denominator an auditable file.
python - "$SOURCE_ROOT/train.jsonl" "$VERIFIED_INDEX" "$TRAIN_SUBSET" <<'PY'
from array import array
import hashlib, json, random, sys
from pathlib import Path
source,index_path,output=map(Path,sys.argv[1:])
offsets=array('Q')
with index_path.open('rb') as handle:
    offsets.fromfile(handle,index_path.stat().st_size // offsets.itemsize)
indices=list(range(len(offsets))); random.Random(17).shuffle(indices); indices=indices[:32]
rows=[]
with source.open('rb') as handle:
    for index in indices:
        handle.seek(int(offsets[index])); rows.append(handle.readline())
with output.open('wb') as handle:
    handle.writelines(rows)
meta={'seed':17,'indices':indices,'rows':32,'sha256':hashlib.sha256(b''.join(rows)).hexdigest()}
output.with_suffix('.manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
print({'train_subset_materialized':str(output),**meta},flush=True)
PY

echo "[graph-overfit-smoke] stage=1 state-SFT fixed_train=32 epochs=25"
python -m torch.distributed.run --standalone --nproc_per_node=1 \
  scripts/train_graph_electron_reaction_online.py \
  --source-root "$SOURCE_ROOT" --source-manifest "$SOURCE_MANIFEST" \
  --verified-offset-index "$VERIFIED_INDEX" \
  --output "$STATE_ROOT" --stage state_bc --epochs 25 \
  --initialize-from "$INITIAL_CHECKPOINT" --expected-world-size 1 \
  --reaction-limit-per-rank 32 --amp-dtype auto \
  --hidden-dim 192 --layers 6 --learning-rate 0.0003 \
  --reactions-per-batch 8 --cpu-workers 2 --prefetch 2 --seed 17 \
  --log-updates 4 --checkpoint-updates 100

STATE_FINAL="$STATE_ROOT/checkpoint-epoch25-update100.pt"
test -f "$STATE_FINAL"
echo "[graph-overfit-smoke] stage=2 trajectory-SFT fixed_train=32 epochs=25"
python -m torch.distributed.run --standalone --nproc_per_node=1 \
  scripts/train_graph_electron_reaction_online.py \
  --source-root "$SOURCE_ROOT" --source-manifest "$SOURCE_MANIFEST" \
  --verified-offset-index "$VERIFIED_INDEX" \
  --output "$TRAJECTORY_ROOT" --stage trajectory_bc --epochs 25 \
  --initialize-from "$STATE_FINAL" --expected-world-size 1 \
  --reaction-limit-per-rank 32 --amp-dtype auto \
  --hidden-dim 192 --layers 6 --learning-rate 0.0001 \
  --reactions-per-batch 8 --cpu-workers 2 --prefetch 2 --seed 17 \
  --log-updates 4 --checkpoint-updates 100

TRAJECTORY_FINAL="$TRAJECTORY_ROOT/checkpoint-epoch25-update100.pt"
test -f "$TRAJECTORY_FINAL"
echo "[graph-overfit-smoke] stage=3 executor-RL fixed_train=32 episodes=256"
python -m torch.distributed.run --standalone --nproc_per_node=1 \
  scripts/train_graph_electron_executor_rl.py \
  --source-root "$SOURCE_ROOT" --source-manifest "$SOURCE_MANIFEST" \
  --verified-offset-index "$VERIFIED_INDEX" \
  --initialize-from "$TRAJECTORY_FINAL" --output "$RL_ROOT" \
  --expected-world-size 1 --reaction-limit-per-rank 32 \
  --episodes 256 --episodes-per-update 1 --max-steps 24 \
  --learning-rate 0.000005 --discount 0.97 --invalid-penalty 1.0 \
  --value-weight 0.5 --bc-weight 0.05 --temperature 1.0 --seed 17 \
  --log-updates 8 --checkpoint-updates 64

RL_FINAL="$RL_ROOT/checkpoint-update256.pt"
test -f "$RL_FINAL"

for stage in state trajectory rl; do
  case "$stage" in
    state) checkpoint="$STATE_FINAL" ;;
    trajectory) checkpoint="$TRAJECTORY_FINAL" ;;
    rl) checkpoint="$RL_FINAL" ;;
  esac
  for split in train valid test; do
    case "$split" in
      train) data="$TRAIN_SUBSET" ;;
      valid) data="$SOURCE_ROOT/valid.jsonl" ;;
      test) data="$SOURCE_ROOT/test.jsonl" ;;
    esac
    echo "[graph-overfit-smoke] eval stage=$stage split=$split reactions=32 greedy=true"
    python scripts/evaluate_graph_electron_policy.py \
      --data "$data" --checkpoint "$checkpoint" \
      --output "$EVAL_ROOT/${stage}_${split}32.json" \
      --limit 32 --max-steps 32 --seed 17 --log-every 8
  done
done

python - "$RUN_ROOT" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]); rows=[]
for stage in ('state','trajectory','rl'):
    for split in ('train','valid','test'):
        report=json.load((root/'eval'/f'{stage}_{split}32.json').open())
        rows.append({
            'stage':stage,'split':split,'reactions':report['reactions'],
            'endpoint_exact':report['endpoint_exact'],
            'endpoint_exact_rate':report['endpoint_exact_rate'],
            'executor_accept_rate':report['executor_accept_rate'],
            'finished':report['finished'],
            'generation_failures':report['generation_failures'],
        })
summary={'status':'completed','protocol':'fixed_train32_overfit_then_disjoint_valid32_test32','rows':rows}
(root/'overfit_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print('[graph-overfit-smoke] summary='+json.dumps(summary,sort_keys=True),flush=True)
PY
echo "[graph-overfit-smoke] completed output=$RUN_ROOT"
