#!/usr/bin/env bash
set -Eeuo pipefail

artifact_root=/aaa/fionafyang/buddy1/whaleywang/MechET
code_dir=${MECHET_STAGE2_V2_RUNTIME_DIR:-$artifact_root}
data=$artifact_root/data/flower_inverse_tool_sft_action_delta_v1/valid.jsonl
state_adapter=${MECHET_STAGE2_STATE_ADAPTER:-$artifact_root/outputs/agent/natural_language_event_v2_qwen3_8b_h20_seed17_20260918}
history_adapter=${MECHET_STAGE2_HISTORY_ADAPTER:-$artifact_root/outputs/agent/natural_language_event_history_v2_qwen3_8b_h20_seed17_20260918}
tokenizer_snapshot=$artifact_root/../OpenEvolveChem/data/hf_cache/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
output=${MECHET_STAGE2_V2_OUTPUT:-$artifact_root/outputs/eval/stage2_v2_matched_valid256_20260929}
rdkit_wheel=$artifact_root/artifacts/wheels/rdkit-2026.3.4-cp311-cp311-manylinux_2_28_x86_64.whl

source /root/miniconda3/etc/profile.d/conda.sh
conda activate meteor
cd "$code_dir"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for required in   "$data"   "$state_adapter/adapter_model.safetensors"   "$state_adapter/adapter_manifest.json"   "$history_adapter/adapter_model.safetensors"   "$history_adapter/adapter_manifest.json"   "$rdkit_wheel"; do
  if [[ ! -e "$required" ]]; then
    echo "[stage2-v2] required artifact missing: $required" >&2
    exit 2
  fi
done
if [[ ! -d "$tokenizer_snapshot" ]]; then
  echo "[stage2-v2] pinned tokenizer snapshot missing: $tokenizer_snapshot" >&2
  exit 2
fi
if [[ -e "$output" ]]; then
  echo "[stage2-v2] refusing to overwrite existing output: $output" >&2
  exit 2
fi
mkdir -p "$output"

runtime_target=$(mktemp -d /tmp/mechet_stage2_v2_runtime.XXXXXX)
echo "a41dde42ecb24e7d93d62b89e8e5d267b02bbac764f965f3968296c99234c685  $rdkit_wheel" | sha256sum --check --strict
python -m pip install --quiet --no-deps --target "$runtime_target" "$rdkit_wheel"
export PYTHONPATH=$runtime_target:$code_dir/src:$code_dir

python - <<'PY'
import rdkit, torch
names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
if len(names) != 8 or not all("A100" in name.upper() for name in names):
    raise SystemExit(f"expected 8 A100 GPUs, got {names}")
if rdkit.__version__ != "2026.03.4":
    raise SystemExit(f"expected RDKit 2026.03.4, got {rdkit.__version__}")
print({"gpus": names, "rdkit_version": rdkit.__version__}, flush=True)
PY

echo "[stage2-v2] phase=parity-audit"
python -u scripts/audit_stage2_v2_protocol_parity.py   --source "$data"   --state-adapter "$state_adapter"   --history-adapter "$history_adapter"   --tokenizer "$tokenizer_snapshot"   --sample-reactions 256   --seed 17   --output "$output/parity_audit.json"

common_args=(
  --data "$data"
  --sample-reactions 256
  --seed 17
  --branching 1
  --early-beam 1
  --late-beam 1
  --early-depth 2
  --max-decisions 40
  --max-imports 32
  --max-new-tokens 512
  --value-weight 0
  --no-4bit
  --matched-v2
)

echo "[stage2-v2] phase=matched-product-start state=GPUs0-3 trajectory=GPUs4-7"
mkdir -p "$output/state_sft" "$output/trajectory_sft"

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun   --standalone --master_port=29731 --nproc_per_node=4   scripts/run_natural_language_value_search.py   "${common_args[@]}"   --policy-adapter "$state_adapter"   --output "$output/state_sft" &
state_pid=$!

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun   --standalone --master_port=29732 --nproc_per_node=4   scripts/run_natural_language_value_search.py   "${common_args[@]}"   --policy-adapter "$history_adapter"   --compact-history   --output "$output/trajectory_sft" &
history_pid=$!

set +e
wait "$state_pid"
state_status=$?
wait "$history_pid"
history_status=$?
set -e
echo "[stage2-v2] state_status=$state_status history_status=$history_status"
if (( state_status != 0 || history_status != 0 )); then
  exit 1
fi

python -u scripts/summarize_natural_language_history_smoke.py   --state-sft "$output/state_sft"   --trajectory-sft "$output/trajectory_sft"   --output "$output/evaluation.json"

python - "$output/evaluation.json" "$output/parity_audit.json" "$output/matched_v2_evaluation.json" <<'PY'
import json, sys
from pathlib import Path
evaluation = json.loads(Path(sys.argv[1]).read_text())
parity = json.loads(Path(sys.argv[2]).read_text())
evaluation["artifact_type"] = "stage2_protocol_v2_matched_pure_policy_valid256"
evaluation["protocol"] = {
    **dict(evaluation.get("protocol") or {}),
    "protocol_v2_parity_audit": True,
    "unified_inventory_prompt": True,
    "qwen_sft_aligned_tool_prefix": True,
    "max_decisions": 40,
    "max_imports": 32,
    "greedy_k1": True,
    "branching": 1,
    "beam_width": 1,
    "value_critic": False,
    "search": False,
}
evaluation["parity_audit"] = {
    key: parity[key]
    for key in (
        "sample_reactions", "decision_rows", "rdkit_version",
        "runtime_history_exact", "sft_tool_prefix_token_exact",
        "ordinary_generation_template_matches",
    )
}
evaluation["claim_boundary"] = (
    "Matched protocol-v2 validation diagnostic. State-SFT and Trajectory-SFT "
    "use identical product-start pure-policy K=1 execution; the only policy "
    "observation difference is compact accepted-action history."
)
Path(sys.argv[3]).write_text(json.dumps(evaluation, indent=2) + "\n")
print(json.dumps({
    "state_sft": evaluation["state_sft"],
    "trajectory_sft": evaluation["trajectory_sft"],
    "delta": evaluation["trajectory_minus_state"],
}, ensure_ascii=False), flush=True)
PY

echo "[stage2-v2] complete result=$output/matched_v2_evaluation.json"
