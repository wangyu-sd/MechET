#!/usr/bin/env bash
set -Eeuo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate meteor

MECHET_PR83=/aaa/fionafyang/buddy1/whaleywang/MechET-pr83-greedy
MECHET_DATA=/aaa/fionafyang/buddy1/whaleywang/MechET
CRN_ROOT=/aaa/fionafyang/buddy1/whaleywang/CRN-JEPA-pr18
OUTPUT="$MECHET_DATA/outputs/eval/pr83_crn_qwen_flow_valid128_stage2_greedy_20261011"
BASELINE="$MECHET_DATA/outputs/eval/pr83_crn_qwen_flow_valid128_stage2_20261011/pure_qwen"
ADAPTER="$MECHET_DATA/outputs/agent/natural_language_event_history_v2_qwen3_0_6b_seed17"
SOURCE="$MECHET_DATA/data/flower_inverse_tool_sft_action_delta_v1/valid.jsonl"
GRAPH="$CRN_ROOT/outputs/auto_research/pr18/full_corpus_pretrain_v1/train_full_8a100_v2/latest.pt"
SOURCE_SHA=7303a6018850db61594af5854df936778c21a6668e61151f95e2c74c32d22d2a
GRAPH_SHA=bb2f7fe53e87fc0e2438c27065ceb588dcd38e604959c9d69fb3984885a376ee

cd "$MECHET_PR83"
mountpoint -q /aaa/fionafyang/buddy1
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export HF_HUB_CACHE=/aaa/fionafyang/buddy1/whaleywang/OpenEvolveChem/data/hf_cache
RUNTIME=$(mktemp -d /tmp/mechet_pr83_greedy_runtime.XXXXXX)
RDKIT_WHEEL="$MECHET_DATA/artifacts/wheels/rdkit-2026.3.4-cp311-cp311-manylinux_2_28_x86_64.whl"
echo "a41dde42ecb24e7d93d62b89e8e5d267b02bbac764f965f3968296c99234c685  $RDKIT_WHEEL" | sha256sum --check --strict
python -m pip install --quiet --no-deps --target "$RUNTIME" "$RDKIT_WHEEL"
export PYTHONPATH="$RUNTIME:$MECHET_PR83:$MECHET_PR83/src:$CRN_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

echo "$SOURCE_SHA  $SOURCE" | sha256sum --check --strict
echo "$GRAPH_SHA  $GRAPH" | sha256sum --check --strict
python - <<'PY'
import rdkit, torch, transformers
from crn_jepa.models.pr18_full_corpus_world import DirectionalCRNPretrainFull
if rdkit.__version__ != "2026.03.4":
    raise SystemExit(f"RDKit mismatch: {rdkit.__version__}")
if torch.cuda.device_count() != 8 or any("A100" not in torch.cuda.get_device_name(i) for i in range(8)):
    raise SystemExit("expected one ordinary eight-A100 host")
print({"stage": "pr83-greedy-preflight", "rdkit": rdkit.__version__,
       "transformers": transformers.__version__, "gpus": torch.cuda.device_count()}, flush=True)
PY

python -m torch.distributed.run --standalone --nproc_per_node=8 \
  scripts/run_crn_qwen_hybrid.py \
  --data "$SOURCE" --source-sha256 "$SOURCE_SHA" \
  --output "$OUTPUT/hybrid_pre_rl" --sample-reactions 128 --seed 17 --resume \
  --model Qwen/Qwen3-0.6B \
  --model-revision c1899de289a04d12100db370d81485cdf75e47ca \
  --policy-adapter "$ADAPTER" --compact-history \
  --crn-root "$CRN_ROOT" --graph-checkpoint "$GRAPH" --graph-sha256 "$GRAPH_SHA" \
  --tool-route greedy_name --graph-initial-budget 4 \
  --graph-flow-topk 1 --max-import-tokens 384 \
  --max-decisions 40 --max-imports 32 --max-context 4096

python scripts/summarize_crn_qwen_hybrid.py \
  --source "$SOURCE" --source-sha256 "$SOURCE_SHA" \
  --output "$OUTPUT/hybrid_pre_rl" --sample-reactions 128 --seed 17 --world 8 \
  --baseline "$BASELINE"
echo '[meteor-pr83-greedy] paired valid128 complete'
