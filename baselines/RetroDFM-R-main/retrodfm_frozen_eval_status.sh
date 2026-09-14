#!/usr/bin/env bash
set -u

output_root=/home/estar/pxy/mechet/MechET/outputs/external_baselines/retrodfm_r
if tmux has-session -t pxy1 2>/dev/null; then
    printf 'tmux pxy1: present\n'
else
    printf 'tmux pxy1: missing\n'
fi

if [[ -f "$output_root/COMPLETED" ]]; then
    printf 'suite: completed\n'
else
    printf 'suite: running, waiting, or not yet completed\n'
fi

for spec in 'mech_uspto_31k_full:3120' 'flower_full:28971'; do
    dataset=${spec%%:*}
    expected=${spec##*:}
    directory="$output_root/$dataset"
    completed=0
    shopt -s nullglob
    parts=("$directory"/raw_shards/input-*.jsonl)
    shopt -u nullglob
    for part in "${parts[@]}"; do
        rows=$(wc -l < "$part")
        completed=$((completed + rows))
    done
    printf '%s inference: %d/%d rows\n' "$dataset" "$completed" "$expected"
    if [[ -f "$directory/evaluation.json" ]]; then
        printf '%s evaluation: %s\n' "$dataset" "$directory/evaluation.json"
    fi
done

if [[ -f "$output_root/run.log" ]]; then
    printf '\nRecent log:\n'
    tail -20 "$output_root/run.log"
fi
