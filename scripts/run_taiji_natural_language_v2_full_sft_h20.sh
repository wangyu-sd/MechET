#!/usr/bin/env bash
set -Eeuo pipefail

shared_repo=/aaa/fionafyang/buddy1/whaleywang/MechET
runtime_repo=${MECHET_NL_V2_RUNTIME_DIR:?set MECHET_NL_V2_RUNTIME_DIR}
shared_hf_cache=/aaa/fionafyang/buddy1/whaleywang/OpenEvolveChem/data/hf_cache
source_dir=$shared_repo/data/flower_inverse_tool_sft_action_delta_v1
event_dir=$shared_repo/data/flower_natural_language_event_sft_v2
history_dir=$shared_repo/data/flower_natural_language_event_history_v2
event_base_config=$runtime_repo/configs/agent/natural_language_event_v2_qwen3_8b_h20.yaml
history_base_config=$runtime_repo/configs/agent/natural_language_history_v2_qwen3_8b_h20.yaml
event_output=$shared_repo/outputs/agent/natural_language_event_v2_qwen3_8b_h20_seed17_20260918
history_output=$shared_repo/outputs/agent/natural_language_event_history_v2_qwen3_8b_h20_seed17_20260918
liger_wheel=$shared_repo/artifacts/wheels/liger_kernel-0.6.2-py3-none-any.whl
xformers_wheel=$shared_repo/artifacts/wheels/xformers-0.0.29.post3-cp311-cp311-manylinux_2_28_x86_64.whl
bitsandbytes_wheel=$shared_repo/artifacts/wheels/bitsandbytes-0.49.2-py3-none-manylinux_2_24_x86_64.whl
rdkit_wheel=$shared_repo/artifacts/wheels/rdkit-2026.3.4-cp311-cp311-manylinux_2_28_x86_64.whl

source /root/miniconda3/etc/profile.d/conda.sh
conda activate meteor
cd "$shared_repo"
export PYTHONUNBUFFERED=1

echo "[nl-v2] runtime=$runtime_repo revision=$(git -C "$runtime_repo" rev-parse HEAD)"
echo "[nl-v2] source=$source_dir event=$event_dir history=$history_dir"
test -f "$event_base_config"
test -f "$history_base_config"
test -f "$source_dir/training_manifest.json"
echo "303b9bbf5c10f9289c3139afb41e4d989e8c809516624a106b89b064163d971d  $liger_wheel" | sha256sum --check --strict
echo "bbf2f500dfdbcf4649bf568cc2c9f434399f704dc4064fd1fbdbef2b524a8139  $xformers_wheel" | sha256sum --check --strict
echo "54b771f06e1a3c73af5c7f16ccf0fc23a846052813d4b008d10cb6e017dd1c8c  $bitsandbytes_wheel" | sha256sum --check --strict
echo "a41dde42ecb24e7d93d62b89e8e5d267b02bbac764f965f3968296c99234c685  $rdkit_wheel" | sha256sum --check --strict

runtime_target=$(mktemp -d /tmp/mechet_nl_v2_runtime.XXXXXX)
python -m pip install --quiet --no-deps --target "$runtime_target" \
  "$liger_wheel" "$xformers_wheel" "$bitsandbytes_wheel" "$rdkit_wheel"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$runtime_target:$runtime_repo/src:$runtime_repo
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_COMPILE_THREADS=1

python - <<'PY'
import os, rdkit, torch
if torch.cuda.device_count() != 8:
    raise SystemExit(f"expected 8 GPUs, got {torch.cuda.device_count()}")
names = [torch.cuda.get_device_name(index) for index in range(8)]
if not all("H20" in name.upper() for name in names):
    raise SystemExit(f"expected H20 GPUs, got {names}")
if rdkit.__version__ != "2026.03.4":
    raise SystemExit(f"expected RDKit 2026.03.4, got {rdkit.__version__}")
print({"gpus": names, "cpu_affinity": len(os.sched_getaffinity(0)),
       "rdkit_version": rdkit.__version__}, flush=True)
PY

workers=$(python - <<'PY'
import os
print(min(64, len(os.sched_getaffinity(0))))
PY
)

if ! python - "$event_dir/manifest.json" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.is_file(): raise SystemExit(1)
m = json.loads(p.read_text())
ok = m.get("training_allowed") is True and m.get("status") == "validated_complete"
ok &= m.get("reaction_denominator") == {"train":257167,"valid":2890,"test":28967}
ok &= m.get("reference_replay_verified") is True
ok &= m.get("decision_contract") == "unified_inventory_tool_decision_v2"
ok &= m.get("split_reaction_id_overlap") == {"test_train":0,"test_valid":0,"train_valid":0}
raise SystemExit(0 if ok else 1)
PY
then
  echo "[nl-v2] full event-v2 conversion starts workers=$workers"
  python -u "$runtime_repo/scripts/build_natural_language_event_sft.py" \
    --source-dir "$source_dir" --output-dir "$event_dir" --workers "$workers"
fi

if ! python - "$history_dir/manifest.json" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.is_file(): raise SystemExit(1)
m = json.loads(p.read_text())
ok = m.get("training_allowed") is True and m.get("status") == "validated_complete"
ok &= m.get("reaction_denominator") == {"train":257167,"valid":2890,"test":28967}
ok &= m.get("decision_contract") == "unified_inventory_compressed_history_tool_decision_v2"
ok &= m.get("failed_actions") == 0 and m.get("model_visible_gold_horizon") is False
raise SystemExit(0 if ok else 1)
PY
then
  echo "[nl-v2] full compact-history-v2 transform starts"
  python -u "$runtime_repo/scripts/build_natural_language_history_sft.py" \
    --source-dir "$event_dir" --output-dir "$history_dir"
fi

python - "$event_dir/manifest.json" "$history_dir/manifest.json" <<'PY'
import json, sys
from pathlib import Path
event = json.loads(Path(sys.argv[1]).read_text())
history = json.loads(Path(sys.argv[2]).read_text())
expected = {"train":257167,"valid":2890,"test":28967}
assert event["reaction_denominator"] == history["reaction_denominator"] == expected
assert event["decision_rows"] == history["decision_rows"]
assert event["training_allowed"] is history["training_allowed"] is True
assert sum(int(v.get("unresolved_reactions", 0)) for v in event["splits"].values()) == 0
print({"dataset_gate":"passed", "reactions":expected,
       "decisions":event["decision_rows"]}, flush=True)
PY

make_config() {
  local base_config=$1
  local manifest=$2
  local generated_config=$3
  python - "$base_config" "$manifest" "$generated_config" <<'PY'
import json, sys, yaml
from pathlib import Path
cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
m = json.loads(Path(sys.argv[2]).read_text())
assert m["training_allowed"] is True
assert m["reaction_denominator"] == {"train":257167,"valid":2890,"test":28967}
cfg["contract"]["expected_train_rows"] = m["decision_rows"]["train"]
cfg["contract"]["expected_validation_rows"] = m["decision_rows"]["valid"]
cfg["contract"]["expected_test_rows"] = m["decision_rows"]["test"]
Path(sys.argv[3]).write_text(yaml.safe_dump(cfg, sort_keys=False))
PY
}

event_config=$(mktemp /tmp/mechet_nl_v2_event.XXXXXX.yaml)
history_config=$(mktemp /tmp/mechet_nl_v2_history.XXXXXX.yaml)
make_config "$event_base_config" "$event_dir/manifest.json" "$event_config"
make_config "$history_base_config" "$history_dir/manifest.json" "$history_config"

local_hf_cache=$(mktemp -d /tmp/mechet_nl_v2_hf_cache.XXXXXX)
echo "[nl-v2] staging pinned Qwen3-8B cache to $local_hf_cache"
cp -a "$shared_hf_cache/models--Qwen--Qwen3-8B" "$local_hf_cache/"
export HF_HUB_CACHE=$local_hf_cache
echo "[nl-v2] local model cache ready bytes=$(du -sb "$local_hf_cache" | cut -f1)"

prepare_and_stage_tokens() {
  local config=$1
  local data_dir=$2
  local label=$3
  local cache_manifest=$data_dir/qwen3_8b_tokens_4096/manifest.json
  if [[ ! -f "$cache_manifest" ]]; then
    echo "[nl-v2] $label distributed tokenization starts"
    torchrun --standalone --nproc_per_node=8 \
      "$runtime_repo/scripts/prepare_tool_sft_arrow.py" --config "$config"
  fi
  python - "$cache_manifest" "$data_dir/manifest.json" <<'PY'
import json, sys
from pathlib import Path
c = json.loads(Path(sys.argv[1]).read_text())
m = json.loads(Path(sys.argv[2]).read_text())
assert c["splits"]["train"]["n_rows"] == m["decision_rows"]["train"]
assert c["splits"]["validation"]["n_rows"] == m["decision_rows"]["valid"]
assert c["splits"]["train"].get("truncation_count", 0) == 0
assert c["splits"]["validation"].get("truncation_count", 0) == 0
print({"token_gate":"passed", "train":c["splits"]["train"]}, flush=True)
PY
  local local_cache
  local_cache=$(mktemp -d "/tmp/mechet_${label}_tokens.XXXXXX")
  echo "[nl-v2] staging $label token cache to $local_cache"
  cp -a "$data_dir/qwen3_8b_tokens_4096/." "$local_cache/"
  STAGED_CACHE=$local_cache
}

run_training() {
  local config=$1
  local output=$2
  local label=$3
  local -a resume_args=()
  if compgen -G "$output/checkpoint-*" >/dev/null; then
    resume_args=(--resume-from-checkpoint)
    echo "[nl-v2] $label resumes from latest checkpoint"
  fi
  echo "[nl-v2] $label one-epoch training starts"
  torchrun --standalone --nproc_per_node=8 \
    "$runtime_repo/scripts/train_tool_sft.py" --config "$config" "${resume_args[@]}"
  test -s "$output/adapter_model.safetensors"
  test -s "$output/adapter_manifest.json"
  echo "[nl-v2] $label training completed"
}

if [[ ! -s "$event_output/adapter_manifest.json" ]]; then
  prepare_and_stage_tokens "$event_config" "$event_dir" event_v2
  event_local_cache=$STAGED_CACHE
  export MECHET_PRETOKENIZED_CACHE_DIR=$event_local_cache
  run_training "$event_config" "$event_output" state_sft_v2
  [[ $event_local_cache == /tmp/mechet_event_v2_tokens.* ]]
  rm -rf -- "$event_local_cache"
else
  echo "[nl-v2] completed state-v2 adapter already present; skip retraining"
fi

test -s "$event_output/adapter_model.safetensors"
if [[ ! -s "$history_output/adapter_manifest.json" ]]; then
  prepare_and_stage_tokens "$history_config" "$history_dir" history_v2
  history_local_cache=$STAGED_CACHE
  export MECHET_PRETOKENIZED_CACHE_DIR=$history_local_cache
  run_training "$history_config" "$history_output" trajectory_sft_v2
  [[ $history_local_cache == /tmp/mechet_history_v2_tokens.* ]]
  rm -rf -- "$history_local_cache"
else
  echo "[nl-v2] completed history-v2 adapter already present; skip retraining"
fi

echo "[nl-v2] full v2 data + State-SFT + Trajectory-SFT pipeline completed"
