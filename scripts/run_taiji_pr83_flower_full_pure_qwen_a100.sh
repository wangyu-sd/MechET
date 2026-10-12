#!/usr/bin/env bash
set -Eeuo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate meteor

MECHET_PR83=/aaa/fionafyang/buddy1/whaleywang/MechET-pr83-greedy
MECHET_DATA=/aaa/fionafyang/buddy1/whaleywang/MechET
SOURCE="$MECHET_DATA/data/flower_inverse_tool_sft_action_delta_v1/test.jsonl"
SOURCE_SHA=70a06f01a08490d4b5da602056a8f8e2f978128d3da3224d1601186529aeee3f
ADAPTER="$MECHET_DATA/outputs/agent/natural_language_event_history_v2_qwen3_0_6b_seed17"
OUTPUT="$MECHET_DATA/outputs/eval/pr83_pure_qwen_stage2_test28967_20261012"

cd "$MECHET_PR83"
mountpoint -q /aaa/fionafyang/buddy1
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export HF_HUB_CACHE=/aaa/fionafyang/buddy1/whaleywang/OpenEvolveChem/data/hf_cache
RUNTIME=$(mktemp -d /tmp/mechet_pr83_pure_full_runtime.XXXXXX)
RDKIT_WHEEL="$MECHET_DATA/artifacts/wheels/rdkit-2026.3.4-cp311-cp311-manylinux_2_28_x86_64.whl"
echo "a41dde42ecb24e7d93d62b89e8e5d267b02bbac764f965f3968296c99234c685  $RDKIT_WHEEL" | sha256sum --check --strict
python -m pip install --quiet --no-deps --target "$RUNTIME" "$RDKIT_WHEEL"
export PYTHONPATH="$RUNTIME:$MECHET_PR83:$MECHET_PR83/src${PYTHONPATH:+:$PYTHONPATH}"

echo "$SOURCE_SHA  $SOURCE" | sha256sum --check --strict
python - <<'PY'
import rdkit, torch
if rdkit.__version__ != "2026.03.4":
    raise SystemExit(f"RDKit mismatch: {rdkit.__version__}")
if torch.cuda.device_count() != 8 or any("A100" not in torch.cuda.get_device_name(i) for i in range(8)):
    raise SystemExit("expected one ordinary eight-A100 host")
print({"stage": "pr83-pure-qwen-full-preflight", "rdkit": rdkit.__version__,
       "gpus": torch.cuda.device_count()}, flush=True)
PY

python -m torch.distributed.run --standalone --nproc_per_node=8 \
  scripts/run_crn_qwen_pure_baseline.py \
  --data "$SOURCE" --source-sha256 "$SOURCE_SHA" \
  --output "$OUTPUT" --sample-reactions 28967 --seed 17 --resume \
  --model Qwen/Qwen3-0.6B \
  --model-revision c1899de289a04d12100db370d81485cdf75e47ca \
  --policy-adapter "$ADAPTER" --compact-history

python scripts/summarize_crn_qwen_hybrid.py \
  --source "$SOURCE" --source-sha256 "$SOURCE_SHA" \
  --output "$OUTPUT" --sample-reactions 28967 --seed 17 --world 8 \
  --policy-label pure_qwen_stage2_matched_v2
echo '[meteor-pr83] pure-Qwen matched strict-test K1 complete: 28967 rows'
