#!/usr/bin/env bash
set -euo pipefail
set -o pipefail

ROOT="/home/estar/pxy/mechet/baselines/Retro-MTGR-main"
PY="/home/estar/anaconda3/envs/gnn/bin/python"
DATA="$ROOT/data/MechET/flower_full_retro_mtgr_processed"
RAW_TEST="$ROOT/data/MechET/flower_full/test.jsonl"
RUN="$ROOT/runs/flower_full"
LOG="$ROOT/logs/flower_full_train_and_test.log"

mkdir -p "$RUN" "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
cd "$ROOT"

RESUME_ARGS=()
if [[ -f "$RUN/best.pt" ]]; then
  RESUME_ARGS=(--resume "$RUN/best.pt")
fi

echo "[$(date -Is)] flower_full training started"
CUDA_VISIBLE_DEVICES=6 "$PY" -u train_mechet_retro_mtgr.py \
  --data-dir "$DATA" \
  --output-dir "$RUN" \
  --epochs 300 \
  --batch-size 16 \
  --threads 4 \
  --device cuda:0 \
  --log-every 1 \
  --patience 20 \
  --min-delta 1e-4 \
  "${RESUME_ARGS[@]}"

echo "[$(date -Is)] flower_full training finished; starting raw-test evaluation"
CUDA_VISIBLE_DEVICES=6 "$PY" -u evaluate_mechet_retro_mtgr_full.py \
  --data-dir "$DATA" \
  --raw-test "$RAW_TEST" \
  --checkpoint "$RUN/best.pt" \
  --output "$RUN/test_evaluation.json" \
  --device cuda:0 \
  --threads 4

echo "[$(date -Is)] flower_full training and evaluation completed"
