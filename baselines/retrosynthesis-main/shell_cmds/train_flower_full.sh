#!/usr/bin/env bash
set -Eeuo pipefail

repo_dir=/home/estar/pxy/mechet/baselines/retrosynthesis-main
log_file="$repo_dir/logs/r_smiles/flower_full.log"
config_file=train-from-scratch/PtoR/flower-full-aug5-config.yml

cd "$repo_dir"
mkdir -p logs/r_smiles exp/flower_full_PtoR_aug5

{
  printf '[launcher] started_at=%s\n' "$(date --iso-8601=seconds)"
  printf '[launcher] host=%s gpu=6 config=%s\n' "$(hostname)" "$config_file"
  /home/estar/anaconda3/envs/opennmt3/bin/python --version
  CUDA_VISIBLE_DEVICES=6 /home/estar/anaconda3/envs/opennmt3/bin/onmt_train \
    -config "$config_file"
} 2>&1 | tee -a "$log_file"

bash shell_cmds/infer_eval_flower_full.sh
