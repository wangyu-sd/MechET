#!/usr/bin/env bash
set -Eeuo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate meteor
PR83=/aaa/fionafyang/buddy1/whaleywang/MechET-pr83-greedy
DATA_ROOT=/aaa/fionafyang/buddy1/whaleywang/MechET
CRN_ROOT=/aaa/fionafyang/buddy1/whaleywang/CRN-JEPA-pr18
MODE=${1:?usage: pre or post}
[[ "$MODE" == pre || "$MODE" == post ]]
ADAPTER="$DATA_ROOT/outputs/agent/natural_language_event_v2_qwen3_0_6b_seed17"
GRAPH="$CRN_ROOT/outputs/auto_research/pr18/full_corpus_pretrain_v1/train_full_8a100_v2/latest.pt"
RL_ROOT="$DATA_ROOT/outputs/agent/pr83_graph_endpoint_rl_stage1_train257167_20261012"
OUTPUT_ROOT="$DATA_ROOT/outputs/eval/pr83_formal_stage1_20261012"
BASE_REV=c1899de289a04d12100db370d81485cdf75e47ca
cd "$PR83"
mountpoint -q /aaa/fionafyang/buddy1
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export HF_HUB_CACHE=/aaa/fionafyang/buddy1/whaleywang/OpenEvolveChem/data/hf_cache
RUNTIME=$(mktemp -d /tmp/mechet_pr83_formal_eval.XXXXXX)
WHEEL="$DATA_ROOT/artifacts/wheels/rdkit-2026.3.4-cp311-cp311-manylinux_2_28_x86_64.whl"
echo "a41dde42ecb24e7d93d62b89e8e5d267b02bbac764f965f3968296c99234c685  $WHEEL" | sha256sum --check --strict
python -m pip install --quiet --no-deps --target "$RUNTIME" "$WHEEL"
export PYTHONPATH="$RUNTIME:$PR83:$PR83/src:$CRN_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
echo "fbd8db06094fd8029d4cb0cac38cbb88280a442d76f686258194723019b1a1bd  $ADAPTER/adapter_model.safetensors" | sha256sum --check --strict
echo "bb2f7fe53e87fc0e2438c27065ceb588dcd38e604959c9d69fb3984885a376ee  $GRAPH" | sha256sum --check --strict
python - <<'PY'
import rdkit, torch
if rdkit.__version__ != "2026.03.4":
    raise SystemExit("formal executor RDKit version mismatch")
if torch.cuda.device_count() != 8 or any("A100" not in torch.cuda.get_device_name(i) for i in range(8)):
    raise SystemExit("formal inference expects eight independent A100 workers")
print({"stage": "pr83-formal-preflight", "rdkit": rdkit.__version__, "gpus": 8}, flush=True)
PY
HEAD_ARGS=()
if [[ "$MODE" == post ]]; then
  [[ -s "$RL_ROOT/train_report.json" && -s "$RL_ROOT/graph_endpoint_rl_head.pt" ]]
  jq -e '.stats.selected == 257167 and .processed_roots == 257167' "$RL_ROOT/train_report.json" >/dev/null
  echo "$(jq -r '.graph_rl_head_sha256' "$RL_ROOT/train_report.json")  $RL_ROOT/graph_endpoint_rl_head.pt" | sha256sum --check --strict
  HEAD_ARGS=(--graph-rl-head "$RL_ROOT/graph_endpoint_rl_head.pt")
  jq -e '.controller == {"tool_route":"greedy_name","greedy_import_decode":"single_generate_to_eos_v2","compact_history":false,"graph_initial_budget":4,"graph_flow_topk":1,"max_import_tokens":384,"max_context":4096,"max_decisions":40,"max_imports":32}' "$RL_ROOT/train_report.json" >/dev/null
fi
for SPLIT in valid test; do
  if [[ "$SPLIT" == valid ]]; then
    COUNT=2890
    SHA=7303a6018850db61594af5854df936778c21a6668e61151f95e2c74c32d22d2a
    OFFICIAL_ARGS=()
  else
    COUNT=28967
    SHA=70a06f01a08490d4b5da602056a8f8e2f978128d3da3224d1601186529aeee3f
    OFFICIAL_ARGS=(--official-denominator 28971 --official-source "$DATA_ROOT/data/flower_full_endpoint_sft/test.jsonl" --official-source-sha256 a0700444e03cdc73ef041ac1b33251486529976246e469c27e15f2bf6f296048)
  fi
  SOURCE="$DATA_ROOT/data/flower_inverse_tool_sft_action_delta_v1/$SPLIT.jsonl"
  echo "$SHA  $SOURCE" | sha256sum --check --strict
  CONTROL="$OUTPUT_ROOT/$SPLIT/pure_qwen"
  HYBRID="$OUTPUT_ROOT/$SPLIT/graph_qwen_$MODE"
  PAIRED_ARGS=(--baseline "$CONTROL")
  [[ "$MODE" != post ]] || PAIRED_ARGS=(--baseline "$OUTPUT_ROOT/$SPLIT/graph_qwen_pre" --baseline-policy-kind frozen_graph)
  if [[ "$MODE" == pre ]]; then
    python -m torch.distributed.run --standalone --nproc_per_node=8 scripts/run_crn_qwen_pure_baseline.py \
      --data "$SOURCE" --source-sha256 "$SHA" --output "$CONTROL" \
      --sample-reactions "$COUNT" --seed 17 --resume --model Qwen/Qwen3-0.6B \
      --model-revision "$BASE_REV" --policy-adapter "$ADAPTER"
    python scripts/summarize_crn_qwen_hybrid.py --source "$SOURCE" --source-sha256 "$SHA" \
      --output "$CONTROL" --sample-reactions "$COUNT" --seed 17 --world 8 \
      --policy-label pure_qwen_stage1_matched_v2 "${OFFICIAL_ARGS[@]}"
  fi
  python -m torch.distributed.run --standalone --nproc_per_node=8 scripts/run_crn_qwen_hybrid.py \
    --data "$SOURCE" --source-sha256 "$SHA" --output "$HYBRID" \
    --sample-reactions "$COUNT" --seed 17 --resume --model Qwen/Qwen3-0.6B \
    --model-revision "$BASE_REV" --policy-adapter "$ADAPTER" \
    --crn-root "$CRN_ROOT" --graph-checkpoint "$GRAPH" "${HEAD_ARGS[@]}" \
    --tool-route greedy_name --graph-initial-budget 4 --graph-flow-topk 1 \
    --max-import-tokens 384 --max-decisions 40 --max-imports 32 --max-context 4096
  python scripts/summarize_crn_qwen_hybrid.py --source "$SOURCE" --source-sha256 "$SHA" \
    --output "$HYBRID" --sample-reactions "$COUNT" --seed 17 --world 8 \
    "${PAIRED_ARGS[@]}" "${OFFICIAL_ARGS[@]}"
  echo "[meteor-pr83-formal] mode=$MODE split=$SPLIT complete count=$COUNT"
done
