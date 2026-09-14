#!/usr/bin/env bash
set -Eeuo pipefail

repo_dir=/home/estar/pxy/mechet/baselines/retrosynthesis-main
log_file="$repo_dir/logs/r_smiles/mech_uspto31k_full.log"
config_file=train-from-scratch/PtoR/mech-uspto31k-aug20-config.yml

cd "$repo_dir"
mkdir -p logs/r_smiles exp/mech_uspto_31k_full_PtoR_aug20

{
  printf '[launcher] started_at=%s\n' "$(date --iso-8601=seconds)"
  printf '[launcher] host=%s gpu=4 config=%s\n' "$(hostname)" "$config_file"
  /home/estar/anaconda3/envs/opennmt3/bin/python --version
  CUDA_VISIBLE_DEVICES=4 /home/estar/anaconda3/envs/opennmt3/bin/onmt_train \
    -config "$config_file"
} 2>&1 | tee -a "$log_file"

