#!/usr/bin/env bash
set -euo pipefail
set -o pipefail

ROOT="/home/estar/pxy/mechet/baselines/Retro-MTGR-main"
PY="/home/estar/anaconda3/envs/gnn/bin/python"
DATA="$ROOT/data/MechET/mech_uspto_31k_full_retro_mtgr_processed"
RAW_TEST="$ROOT/data/MechET/mech_uspto_31k_full/test.jsonl"
RUN="$ROOT/runs/mech_uspto_31k_full"
LOG="$ROOT/logs/mech_uspto_31k_train_and_test.log"

mkdir -p "$RUN" "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
cd "$ROOT"

echo "[$(date -Is)] validation loss did not improve after epoch 40; selected best checkpoint and starting raw-test evaluation"
"$PY" -u evaluate_mechet_retro_mtgr_full.py \
  --data-dir "$DATA" \
  --raw-test "$RAW_TEST" \
  --checkpoint "$RUN/best.pt" \
  --output "$RUN/test_evaluation.json" \
  --device cpu \
  --threads 8

echo "[$(date -Is)] mech_uspto_31k_full evaluation completed"
