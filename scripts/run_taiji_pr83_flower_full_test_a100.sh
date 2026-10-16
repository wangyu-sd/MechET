#!/usr/bin/env bash
set -Eeuo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate meteor

MECHET_PR83=/aaa/fionafyang/buddy1/whaleywang/MechET-pr83-greedy
MECHET_DATA=/aaa/fionafyang/buddy1/whaleywang/MechET
CRN_ROOT=/aaa/fionafyang/buddy1/whaleywang/CRN-JEPA-pr18
SOURCE="$MECHET_DATA/data/flower_inverse_tool_sft_action_delta_v1/test.jsonl"
SOURCE_SHA=70a06f01a08490d4b5da602056a8f8e2f978128d3da3224d1601186529aeee3f
OFFICIAL="$MECHET_DATA/data/flower_full_endpoint_sft/test.jsonl"
OFFICIAL_SHA=a0700444e03cdc73ef041ac1b33251486529976246e469c27e15f2bf6f296048
GRAPH="$CRN_ROOT/outputs/auto_research/pr18/full_corpus_pretrain_v1/train_full_8a100_v2/latest.pt"
GRAPH_SHA=bb2f7fe53e87fc0e2438c27065ceb588dcd38e604959c9d69fb3984885a376ee
ADAPTER="$MECHET_DATA/outputs/agent/natural_language_event_history_v2_qwen3_0_6b_seed17"

MODE=${1:-pre}
case "$MODE" in
  pre)
    OUTPUT="$MECHET_DATA/outputs/eval/pr83_crn_qwen_flow_test28967_pre_rl_20261011"
    GRAPH_HEAD_ARGS=()
    ;;
  post)
    OUTPUT="$MECHET_DATA/outputs/eval/pr83_crn_qwen_flow_test28967_post_rl64_20261011"
    HEAD="$MECHET_DATA/outputs/agent/pr83_crn_graph_endpoint_rl_flower_train64_20261011/graph_endpoint_rl_head.pt"
    REPORT="$MECHET_DATA/outputs/agent/pr83_crn_graph_endpoint_rl_flower_train64_20261011/train_report.json"
    [[ -s "$HEAD" && -s "$REPORT" ]] || { echo 'verified RL head/report is missing' >&2; exit 2; }
    jq -e '.effective_rl_updates > 0' "$REPORT" >/dev/null || {
      echo 'no effective endpoint-RL group; full post-RL evaluation forbidden' >&2; exit 2;
    }
    jq -e '.controller == {"tool_route":"greedy_name","greedy_import_decode":"single_generate_to_eos_v2","compact_history":true,"graph_initial_budget":4,"graph_flow_topk":1,"max_import_tokens":384,"max_context":4096,"max_decisions":40,"max_imports":32}' "$REPORT" >/dev/null || {
      echo 'RL controller differs from the full-test inference protocol' >&2; exit 2;
    }
    echo "$(jq -r '.graph_rl_head_sha256' "$REPORT")  $HEAD" | sha256sum --check --strict
    GRAPH_HEAD_ARGS=(--graph-rl-head "$HEAD")
    ;;
  *) echo "usage: $0 [pre|post]" >&2; exit 2 ;;
esac

cd "$MECHET_PR83"
mountpoint -q /aaa/fionafyang/buddy1
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export HF_HUB_CACHE=/aaa/fionafyang/buddy1/whaleywang/OpenEvolveChem/data/hf_cache
RUNTIME=$(mktemp -d /tmp/mechet_pr83_fulltest_runtime.XXXXXX)
RDKIT_WHEEL="$MECHET_DATA/artifacts/wheels/rdkit-2026.3.4-cp311-cp311-manylinux_2_28_x86_64.whl"
echo "a41dde42ecb24e7d93d62b89e8e5d267b02bbac764f965f3968296c99234c685  $RDKIT_WHEEL" | sha256sum --check --strict
python -m pip install --quiet --no-deps --target "$RUNTIME" "$RDKIT_WHEEL"
export PYTHONPATH="$RUNTIME:$MECHET_PR83:$MECHET_PR83/src:$CRN_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

echo "$SOURCE_SHA  $SOURCE" | sha256sum --check --strict
echo "$OFFICIAL_SHA  $OFFICIAL" | sha256sum --check --strict
echo "$GRAPH_SHA  $GRAPH" | sha256sum --check --strict
python - <<'PY'
import rdkit, torch
from crn_jepa.models.pr18_full_corpus_world import DirectionalCRNPretrainFull
if rdkit.__version__ != "2026.03.4":
    raise SystemExit(f"RDKit mismatch: {rdkit.__version__}")
if torch.cuda.device_count() != 8 or any("A100" not in torch.cuda.get_device_name(i) for i in range(8)):
    raise SystemExit("expected one ordinary eight-A100 host")
print({"stage": "pr83-full-test-preflight", "rdkit": rdkit.__version__,
       "gpus": torch.cuda.device_count()}, flush=True)
PY

python -m torch.distributed.run --standalone --nproc_per_node=8 \
  scripts/run_crn_qwen_hybrid.py \
  --data "$SOURCE" --source-sha256 "$SOURCE_SHA" \
  --output "$OUTPUT" --sample-reactions 28967 --seed 17 --resume \
  --model Qwen/Qwen3-0.6B \
  --model-revision c1899de289a04d12100db370d81485cdf75e47ca \
  --policy-adapter "$ADAPTER" --compact-history \
  --crn-root "$CRN_ROOT" --graph-checkpoint "$GRAPH" --graph-sha256 "$GRAPH_SHA" \
  "${GRAPH_HEAD_ARGS[@]}" \
  --tool-route greedy_name --graph-initial-budget 4 \
  --graph-flow-topk 1 --max-import-tokens 384 \
  --max-decisions 40 --max-imports 32 --max-context 4096

python scripts/summarize_crn_qwen_hybrid.py \
  --source "$SOURCE" --source-sha256 "$SOURCE_SHA" \
  --output "$OUTPUT" --sample-reactions 28967 --seed 17 --world 8 \
  --official-denominator 28971 \
  --official-source "$OFFICIAL" --official-source-sha256 "$OFFICIAL_SHA"
echo "[meteor-pr83] full FlowER test mode=$MODE complete: 28967 strict + 4 explicit unsupported"
