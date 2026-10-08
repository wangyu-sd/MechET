#!/usr/bin/env bash
# Frozen full-test K=1/5/10 reverse-electron-flow episodes on one 8-GPU host.
# Taiji submission must wrap this script in taiji_run_with_heartbeat.sh.
set -Eeuo pipefail

shared_repo=/aaa/fionafyang/buddy1/whaleywang/MechET
runtime_repo=/aaa/fionafyang/buddy1/whaleywang/MechET-pr82-reliable-20261006
shared_hf_cache=/aaa/fionafyang/buddy1/whaleywang/OpenEvolveChem/data/hf_cache
stage=${MECHET_RELIABLE_STAGE:?set state or trajectory}
view=${MECHET_RELIABLE_BENCHMARK_VIEW:?set strict_test or full_endpoint_test}
episodes=${MECHET_RELIABLE_K_EPISODES:?set 1, 5, or 10 explicitly}
expected_gpu=${MECHET_RELIABLE_EXPECTED_GPU:-A100}
case "$stage" in
  state) adapter=$shared_repo/outputs/agent/natural_language_event_v2_qwen3_0_6b_seed17 ;;
  trajectory) adapter=$shared_repo/outputs/agent/natural_language_event_history_v2_qwen3_0_6b_seed17 ;;
  *) echo "unsupported stage=$stage" >&2; exit 2 ;;
esac
case "$view" in
  strict_test)
    source_data=$shared_repo/data/flower_inverse_tool_sft_action_delta_v1/test.jsonl
    source_manifest=$shared_repo/data/flower_inverse_tool_sft_action_delta_v1/training_manifest.json
    denominator=28967
    ;;
  full_endpoint_test)
    source_data=$shared_repo/data/flower_full_endpoint_sft/test.jsonl
    source_manifest=$shared_repo/data/flower_full_endpoint_sft/manifest.json
    denominator=28971
    ;;
  *) echo "unsupported benchmark view=$view" >&2; exit 2 ;;
esac
case "$episodes" in 1|5|10) ;; *) echo "unsupported K=$episodes" >&2; exit 2 ;; esac
case "$expected_gpu" in A100|H20) ;; *) echo "unsupported GPU=$expected_gpu" >&2; exit 2 ;; esac
output=${MECHET_RELIABLE_EVAL_OUTPUT:-$shared_repo/outputs/eval/reliable_mechet_${stage}_${view}_k${episodes}_seed17}
if [[ $output != /* ]]; then
  echo "MECHET_RELIABLE_EVAL_OUTPUT must be an absolute path" >&2
  exit 2
fi

source /root/miniconda3/etc/profile.d/conda.sh
conda activate meteor
cd "$shared_repo"
export PYTHONUNBUFFERED=1
export HF_HUB_CACHE=$shared_hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$runtime_repo/src:$runtime_repo

rdkit_wheel=$shared_repo/artifacts/wheels/rdkit-2026.3.4-cp311-cp311-manylinux_2_28_x86_64.whl
echo "a41dde42ecb24e7d93d62b89e8e5d267b02bbac764f965f3968296c99234c685  $rdkit_wheel" | sha256sum --check --strict
runtime_target=$(mktemp -d /tmp/mechet_reliable_k_rdkit.XXXXXX)
python -m pip install --quiet --no-deps --target "$runtime_target" "$rdkit_wheel"
export PYTHONPATH=$runtime_target:$PYTHONPATH

test -s "$source_data"
test -s "$source_manifest"
test -s "$adapter/adapter_model.safetensors"
test -s "$adapter/adapter_manifest.json"
source_sha=$(sha256sum "$source_data" | cut -d' ' -f1)
adapter_sha=$(sha256sum "$adapter/adapter_model.safetensors" | cut -d' ' -f1)

python - "$source_data" "$source_manifest" "$source_sha" "$denominator" "$adapter" "$stage" "$view" "$expected_gpu" <<'PY'
import sys
from argparse import Namespace
from pathlib import Path
import rdkit
import torch
from scripts.eval_reliable_independent_episodes import validate_benchmark_source
from scripts.run_natural_language_value_search import validate_v2_adapter_manifest

source, manifest = map(Path, sys.argv[1:3])
source_sha, denominator = sys.argv[3], int(sys.argv[4])
adapter = Path(sys.argv[5])
stage, view, expected_gpu = sys.argv[6:9]
validate_benchmark_source(Namespace(
    data=source, source_manifest=manifest, benchmark_view=view,
    sample_reactions=denominator,
), source_sha)
validate_v2_adapter_manifest(
    adapter, compact_history=stage == 'trajectory',
    expected_model='Qwen/Qwen3-0.6B',
    expected_revision='c1899de289a04d12100db370d81485cdf75e47ca',
)
names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
if len(names) != 8 or not all(expected_gpu in name.upper() for name in names):
    raise RuntimeError(f'expected eight {expected_gpu} GPUs, observed {names}')
if not torch.cuda.is_bf16_supported() or rdkit.__version__ != '2026.03.4':
    raise RuntimeError('BF16/RDKit runtime differs from frozen evaluation contract')
print({'gate': 'frozen_k_evaluation', 'view': view,
       'denominator': denominator, 'stage': stage, 'gpus': names}, flush=True)
PY

common=(
  --data "$source_data" --expected-source-sha256 "$source_sha"
  --benchmark-view "$view" --source-manifest "$source_manifest"
  --adapter "$adapter" --expected-adapter-sha256 "$adapter_sha"
  --stage "$stage" --output "$output" --episodes "$episodes"
  --sample-reactions "$denominator" --seed 17
  --max-new-tokens 512 --max-context 4096 --no-4bit --dtype bfloat16
)
echo "[meteor-reliable-k] stage=$stage view=$view K=$episodes denominator=$denominator adapter_sha=$adapter_sha"
echo "[meteor-reliable-k] output=$output runtime=$(git -C "$runtime_repo" rev-parse HEAD)"
torchrun --standalone --nproc_per_node=8 \
  "$runtime_repo/scripts/eval_reliable_independent_episodes.py" run "${common[@]}"
python -u "$runtime_repo/scripts/eval_reliable_independent_episodes.py" aggregate "${common[@]}"
python - "$output/independent_episodes_audit.json" "$denominator" <<'PY'
import json, sys
from pathlib import Path
report = json.loads(Path(sys.argv[1]).read_text())
denominator = int(sys.argv[2])
if report['denominator'] != denominator or report['observed_reactions'] != denominator:
    raise RuntimeError('formal K evaluation did not cover every frozen test reaction')
if report['missing_reactions'] != 0:
    raise RuntimeError('formal K evaluation has missing predictions')
print({'gate': 'complete_formal_k_evaluation', 'denominator': denominator,
       'report': sys.argv[1]}, flush=True)
PY
