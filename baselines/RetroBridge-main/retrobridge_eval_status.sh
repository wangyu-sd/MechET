#!/usr/bin/env bash
set -u

OUTPUT_ROOT=/home/estar/pxy/mechet/MechET/outputs/external_baselines/retrobridge

if tmux has-session -t retrobridge_eval_20260904 2>/dev/null; then
  printf 'queue: running (tmux session retrobridge_eval_20260904)\n'
else
  printf 'queue: not running\n'
fi

for spec in \
  'USPTO31K:mech_uspto_31k_full_single_gpu_epoch43:3120' \
  'FLOWER:flower_full_best_epoch79:28966'; do
  label=${spec%%:*}
  remainder=${spec#*:}
  directory=${remainder%%:*}
  expected=${remainder##*:}
  output="$OUTPUT_ROOT/$directory"
  completed=0
  shopt -s nullglob
  parts=("$output"/parts/worker-*.predictions.jsonl)
  shopt -u nullglob
  for part in "${parts[@]}"; do
    rows=$(wc -l <"$part")
    completed=$((completed + rows))
  done
  printf '%s inference: %d/%d native-compatible rows\n' \
    "$label" "$completed" "$expected"
  if [[ -f "$output/evaluation.json" ]]; then
    printf '%s evaluation: %s\n' "$label" "$output/evaluation.json"
  fi
done
