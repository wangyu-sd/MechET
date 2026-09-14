#!/usr/bin/env bash
set -Eeuo pipefail

repo_dir=/home/estar/pxy/mechet/baselines/retrosynthesis-main
python_bin=/home/estar/anaconda3/envs/opennmt3/bin/python
train_config="$repo_dir/train-from-scratch/PtoR/flower-full-aug5-config.yml"
checkpoint_prefix="$repo_dir/exp/flower_full_PtoR_aug5/model.product-reactants_step_"
poll_seconds="${RSMILES_WATCH_POLL_SECONDS:-60}"
train_pattern='[o]nmt_train.*flower-full-aug5-config.yml'

read_target_step() {
    "$python_bin" - "$train_config" <<'PY'
import sys
from pathlib import Path

for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    key, separator, value = line.partition(":")
    if separator and key.strip() == "train_steps":
        print(int(value.strip()))
        break
else:
    raise SystemExit(f"train_steps not found in {sys.argv[1]}")
PY
}

cd "$repo_dir"
printf '[watcher] started_at=%s poll_seconds=%s\n' "$(date --iso-8601=seconds)" "$poll_seconds"

while true; do
    if pgrep -f "$train_pattern" >/dev/null; then
        target_step=$(read_target_step)
        printf '[watcher] %s training_active target_step=%s\n' "$(date --iso-8601=seconds)" "$target_step"
        sleep "$poll_seconds"
        continue
    fi

    # Require two consecutive inactive observations. This avoids racing a
    # deliberate stop/resume between checkpoints.
    sleep "$poll_seconds"
    if pgrep -f "$train_pattern" >/dev/null; then
        continue
    fi

    target_step=$(read_target_step)
    final_checkpoint="${checkpoint_prefix}${target_step}.pt"
    if [[ ! -s "$final_checkpoint" ]]; then
        printf '[watcher] %s training_inactive_but_target_missing target_step=%s; waiting_for_resume\n' \
            "$(date --iso-8601=seconds)" "$target_step"
        sleep "$poll_seconds"
        continue
    fi

    printf '[watcher] %s training_complete checkpoint=%s; starting_infer_eval\n' \
        "$(date --iso-8601=seconds)" "$final_checkpoint"
    exec bash shell_cmds/infer_eval_flower_full.sh "$target_step"
done
