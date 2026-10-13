#!/usr/bin/env bash
set -Eeuo pipefail
source "${MECHET_CONDA_ROOT:-/root/miniconda3}/etc/profile.d/conda.sh"
conda activate "${MECHET_CONDA_ROOT:-/root/miniconda3}/envs/meteor"
if [[ -n ${MECHET_PYTHON_WRAPPER:-} ]]; then
  (cd "$MECHET_GLIBC_ROOT" && sha256sum --check --strict SHA256SUMS)
  python() { "$MECHET_PYTHON_WRAPPER" "$@"; }
  export PYTHON_EXEC="$MECHET_PYTHON_WRAPPER"
fi
python - <<'PY'
import os, sys
print({"stage": "pr83-python-abi", "executable": sys.executable,
       "python": sys.version, "glibc": os.confstr("CS_GNU_LIBC_VERSION")}, flush=True)
if sys.version_info[:2] != (3, 11):
    raise SystemExit("formal pinned RDKit wheel requires the configured Python 3.11 interpreter")
if tuple(map(int, os.confstr("CS_GNU_LIBC_VERSION").split()[1].split("."))) < (2, 28):
    raise SystemExit("formal pinned RDKit wheel requires process glibc >= 2.28")
PY
PR83=/aaa/fionafyang/buddy1/whaleywang/MechET-pr83-greedy
DATA_ROOT=/aaa/fionafyang/buddy1/whaleywang/MechET
CRN_ROOT=/aaa/fionafyang/buddy1/whaleywang/CRN-JEPA-pr18
OUTPUT="$DATA_ROOT/outputs/agent/pr83_graph_earho_conditional_stage1_train257167_20261012"
cd "$PR83"
mountpoint -q /aaa/fionafyang/buddy1
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_CACHE=/aaa/fionafyang/buddy1/whaleywang/OpenEvolveChem/data/hf_cache
RUNTIME=$(mktemp -d /tmp/mechet_pr83_formal_rl.XXXXXX)
WHEEL="$DATA_ROOT/artifacts/wheels/rdkit-2026.3.4-cp311-cp311-manylinux_2_28_x86_64.whl"
echo "a41dde42ecb24e7d93d62b89e8e5d267b02bbac764f965f3968296c99234c685  $WHEEL" | sha256sum --check --strict
python -m pip install --quiet --no-deps --target "$RUNTIME" "$WHEEL"
export PYTHONPATH="$RUNTIME:$PR83:$PR83/src:$CRN_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
ADAPTER="$DATA_ROOT/outputs/agent/natural_language_event_v2_qwen3_0_6b_seed17"
echo "fbd8db06094fd8029d4cb0cac38cbb88280a442d76f686258194723019b1a1bd  $ADAPTER/adapter_model.safetensors" | sha256sum --check --strict
python - <<'PY'
import os, rdkit, torch
expected_cuda = os.environ.get("MECHET_EXPECT_TORCH_CUDA")
if expected_cuda and torch.version.cuda != expected_cuda:
    raise SystemExit(f"configured CUDA runtime mismatch: {torch.version.cuda}")
if rdkit.__version__ != "2026.03.4" or torch.cuda.device_count() != 1 or "A100" not in torch.cuda.get_device_name(0):
    raise SystemExit("formal graph RL requires pinned RDKit and one A100")
print({"stage": "pr83-formal-rl-preflight", "rdkit": rdkit.__version__, "gpu": torch.cuda.get_device_name(0)}, flush=True)
PY
RESUME_ARGS=()
[[ ! -s "$OUTPUT/latest_train_state.pt" ]] || RESUME_ARGS=(--resume)
exec "${MECHET_PYTHON_WRAPPER:-$CONDA_PREFIX/bin/python}" -u scripts/train_crn_graph_earho.py \
  --data "$DATA_ROOT/data/flower_inverse_tool_sft_action_delta_v1/train.jsonl" \
  --source-sha256 edc80c5c5eb13d50753c5566c5d6ac1b90b955b1a8dfece915241b2da4a40a75 \
  --output "$OUTPUT" --sample-reactions 257167 --rl-updates 257167 --seed 17 \
  --slate 8 --continuation-horizon 8 --reference-weight 0.1 --save-every 128 \
  --actor-reward endpoint --policy-objective conditional_endpoint --exploration-proposals 4 \
  --rollout-beam-width 1 --slate-kl-weight 0.02 \
  --model Qwen/Qwen3-0.6B --model-revision c1899de289a04d12100db370d81485cdf75e47ca \
  --policy-adapter "$ADAPTER" --crn-root "$CRN_ROOT" \
  --graph-checkpoint "$CRN_ROOT/outputs/auto_research/pr18/full_corpus_pretrain_v1/train_full_8a100_v2/latest.pt" \
  --tool-route greedy_name --graph-initial-budget 4 --graph-flow-topk 1 \
  --max-import-tokens 384 --max-decisions 40 --max-imports 32 --max-context 4096 "${RESUME_ARGS[@]}"
