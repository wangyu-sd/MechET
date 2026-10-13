#!/usr/bin/env bash
set -Eeuo pipefail
CODE_ROOT=${MECHET_JOINT_CODE_ROOT:-/aaa/fionafyang/buddy1/whaleywang/MechET-pr83-greedy}
cd "$CODE_ROOT"
if [[ -f runtime_provenance.json ]]; then
  ACTUAL_COMMIT=$(python3 -c 'import json; print(json.load(open("runtime_provenance.json"))["commit"])')
else
  ACTUAL_COMMIT=$(git rev-parse HEAD)
fi
[[ -n ${MECHET_EXPECTED_COMMIT:-} && "$ACTUAL_COMMIT" == "$MECHET_EXPECTED_COMMIT" ]] || {
  echo 'joint runtime commit mismatch' >&2; exit 2;
}
source /root/miniconda3/etc/profile.d/conda.sh
conda activate meteor
DATA_ROOT=/aaa/fionafyang/buddy1/whaleywang/MechET
CRN_ROOT=/aaa/fionafyang/buddy1/whaleywang/CRN-JEPA-pr18
GRAPH_CHECKPOINT="$CRN_ROOT/outputs/auto_research/pr18/full_corpus_pretrain_v1/train_full_8a100_v2/latest.pt"
[[ ! -d "$CODE_ROOT/crn_snapshot/src" ]] || CRN_ROOT="$CODE_ROOT/crn_snapshot"
OUTPUT="$DATA_ROOT/outputs/agent/pr84_joint_import_flow_stage1_train257167_20261013_01"
mountpoint -q /aaa/fionafyang/buddy1
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_CACHE=/aaa/fionafyang/buddy1/whaleywang/OpenEvolveChem/data/hf_cache
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
RUNTIME=$(mktemp -d /tmp/mechet_joint_rdkit.XXXXXX)
WHEEL="$DATA_ROOT/artifacts/wheels/rdkit-2026.3.4-cp311-cp311-manylinux_2_28_x86_64.whl"
echo "a41dde42ecb24e7d93d62b89e8e5d267b02bbac764f965f3968296c99234c685  $WHEEL" | sha256sum --check --strict
python -m pip install --quiet --no-deps --target "$RUNTIME" "$WHEEL"
export PYTHONPATH="$RUNTIME:$CODE_ROOT:$CODE_ROOT/src:$CRN_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
python - <<'PY'
import json, hashlib, pathlib, rdkit, torch, transformers, peft
root = pathlib.Path('.')
if (root/'runtime_provenance.json').is_file():
    manifest = json.loads((root/'runtime_provenance.json').read_text())
    for name, expected in manifest['sha256'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != expected:
            raise SystemExit('runtime file hash mismatch: '+name)
if rdkit.__version__ != '2026.03.4' or torch.cuda.device_count() != 1 or 'A100' not in torch.cuda.get_device_name(0):
    raise SystemExit('joint training requires pinned RDKit and one ordinary A100')
if not torch.cuda.is_bf16_supported():
    raise SystemExit('BF16 unavailable on allocated host')
print(dict(stage='joint-training-preflight', rdkit=rdkit.__version__, torch=torch.__version__,
    cuda=torch.version.cuda, transformers=transformers.__version__, peft=peft.__version__,
    gpu=torch.cuda.get_device_name(0)), flush=True)
PY
ADAPTER="$DATA_ROOT/outputs/agent/natural_language_event_v2_qwen3_0_6b_seed17"
RESUME_ARGS=()
[[ ! -s "$OUTPUT/latest_train_state.pt" ]] || RESUME_ARGS=(--resume)
exec python -u scripts/train_crn_qwen_joint_earho.py \
  --data "$DATA_ROOT/data/flower_inverse_tool_sft_action_delta_v1/train.jsonl" \
  --source-sha256 edc80c5c5eb13d50753c5566c5d6ac1b90b955b1a8dfece915241b2da4a40a75 \
  --output "$OUTPUT" --sample-reactions 257167 --rl-updates 257167 --seed 17 \
  --slate 8 --import-proposals 4 --flow-proposals 4 --exploration-proposals 4 \
  --continuation-horizon 8 --full-episode-fraction 0.25 --reference-weight 0.1 \
  --learning-rate 5e-6 --qwen-learning-rate 1e-6 --slate-kl-weight 0.02 --save-every 32 \
  --model Qwen/Qwen3-0.6B --model-revision c1899de289a04d12100db370d81485cdf75e47ca \
  --policy-adapter "$ADAPTER" --crn-root "$CRN_ROOT" \
  --graph-checkpoint "$GRAPH_CHECKPOINT" \
  --tool-route greedy_name --graph-initial-budget 4 --graph-flow-topk 1 \
  --dtype bfloat16 --max-import-tokens 384 --max-decisions 40 --max-imports 32 --max-context 4096 \
  "${RESUME_ARGS[@]}"
