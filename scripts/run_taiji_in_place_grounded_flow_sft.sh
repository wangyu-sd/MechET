#!/usr/bin/env bash
set -Eeuo pipefail

shared_repo=/aaa/fionafyang/buddy1/whaleywang/MechET
runtime_repo=${MECHET_IN_PLACE_RUNTIME_DIR:?set MECHET_IN_PLACE_RUNTIME_DIR}
shared_hf_cache=/aaa/fionafyang/buddy1/whaleywang/OpenEvolveChem/data/hf_cache
training_config=$runtime_repo/configs/agent/in_place_grounded_flow_qwen3_8b_a100.yaml
source_dir=$shared_repo/data/flower_inverse_tool_sft_action_delta_v1
output_dir=$shared_repo/data/flower_in_place_grounded_flow_v1
liger_wheel=$shared_repo/artifacts/wheels/liger_kernel-0.6.2-py3-none-any.whl
xformers_wheel=$shared_repo/artifacts/wheels/xformers-0.0.29.post3-cp311-cp311-manylinux_2_28_x86_64.whl
bitsandbytes_wheel=$shared_repo/artifacts/wheels/bitsandbytes-0.49.2-py3-none-manylinux_2_24_x86_64.whl

source /root/miniconda3/etc/profile.d/conda.sh
conda activate meteor
cd "$shared_repo"

echo "[in-place-sft] runtime=$runtime_repo source=$source_dir output=$output_dir"
test -f "$training_config"
test -f "$source_dir/train.jsonl"
echo "303b9bbf5c10f9289c3139afb41e4d989e8c809516624a106b89b064163d971d  $liger_wheel" | sha256sum --check --strict
echo "bbf2f500dfdbcf4649bf568cc2c9f434399f704dc4064fd1fbdbef2b524a8139  $xformers_wheel" | sha256sum --check --strict
echo "54b771f06e1a3c73af5c7f16ccf0fc23a846052813d4b008d10cb6e017dd1c8c  $bitsandbytes_wheel" | sha256sum --check --strict

runtime_target=$(mktemp -d /tmp/mechet_in_place_runtime.XXXXXX)
python -m pip install --quiet --no-deps --target "$runtime_target" \
  "$liger_wheel" "$xformers_wheel" "$bitsandbytes_wheel"

export HF_HUB_CACHE=$shared_hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$runtime_target:$runtime_repo/src:$runtime_repo
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_COMPILE_THREADS=1

python - <<'PY'
import rdkit
import torch
if torch.cuda.device_count() != 8:
    raise SystemExit(f"expected 8 GPUs, got {torch.cuda.device_count()}")
names=[torch.cuda.get_device_name(i) for i in range(8)]
if not all('A100' in name.upper() for name in names):
    raise SystemExit(f"expected A100 GPUs, got {names}")
print({'gpus': names, 'rdkit_version': rdkit.__version__}, flush=True)
PY

python - "$source_dir" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
m=json.loads((root/'training_manifest.json').read_text())
s=json.loads((root/'ARTIFACT_STATUS.json').read_text())
expected={'train':257167,'valid':2890,'test':28967}
if m.get('observation_mode') != 'action_delta_v1':
    raise SystemExit('wrong source observation contract')
if not m.get('strict_trace_universe_complete') or s.get('training_allowed') is not True:
    raise SystemExit('source strict artifact is not train-ready')
for split,n in expected.items():
    if int(m['splits'][split]['rows']) != n:
        raise SystemExit(f'source {split} denominator mismatch')
print({'source_gate':'passed','rows':expected},flush=True)
PY

manifest=$output_dir/manifest.json
if ! python - "$manifest" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
if not p.is_file(): raise SystemExit(1)
m=json.loads(p.read_text())
expected={'train':257167,'valid':2890,'test':28967}
ok=bool(m.get('training_allowed') and m.get('gate_passed'))
ok &= all(int(m['splits'][s]['rows']) == n for s,n in expected.items())
ok &= not any((m.get('split_id_overlap') or {}).values())
ok &= m.get('model_visible_atom_maps') is False
raise SystemExit(0 if ok else 1)
PY
then
  workers=$(python - <<'PY'
import os
print(max(1, min(64, len(os.sched_getaffinity(0)))))
PY
)
  echo "[in-place-sft] building full strict artifact workers=$workers"
  python -u "$runtime_repo/scripts/build_in_place_grounded_flow_sft.py" \
    --source-dir "$source_dir" --output-dir "$output_dir" --workers "$workers"
fi

python - "$manifest" "$source_dir" "$output_dir" <<'PY'
import json,sys
from pathlib import Path
m=json.loads(Path(sys.argv[1]).read_text())
expected={'train':257167,'valid':2890,'test':28967}
if not m.get('training_allowed') or not m.get('gate_passed'):
    raise SystemExit('representation gate did not pass')
for split,n in expected.items():
    if int(m['splits'][split]['rows']) != n:
        raise SystemExit(f'{split} rows mismatch')
    if not (Path(sys.argv[3]) / f'{split}.jsonl').is_file():
        raise SystemExit(f'missing converted {split}')
if any(m['split_id_overlap'].values()):
    raise SystemExit('split ID overlap')
if m.get('model_visible_atom_maps') is not False:
    raise SystemExit('model-visible atom maps')
print({'gate_passed':True,'rows':expected,'flow_char_ratio':m['legacy_move_to_compact_flow_char_ratio']},flush=True)
PY

cache_manifest=$output_dir/qwen3_8b_tokens_8192/manifest.json
if [[ ! -f "$cache_manifest" ]]; then
  echo "[in-place-sft] distributed tokenization starts"
  torchrun --standalone --nproc_per_node=8 \
    "$runtime_repo/scripts/prepare_tool_sft_arrow.py" --config "$training_config"
fi

python - "$cache_manifest" <<'PY'
import json,sys
from pathlib import Path
m=json.loads(Path(sys.argv[1]).read_text())
if int(m['splits']['train']['n_rows']) != 257167:
    raise SystemExit('token cache train denominator mismatch')
if int(m['splits']['validation']['n_rows']) != 2890:
    raise SystemExit('token cache validation denominator mismatch')
if int(m['splits']['train'].get('truncation_count',0)):
    raise SystemExit('token truncation detected')
print({'token_gate':'passed','train':m['splits']['train']},flush=True)
PY

echo "[in-place-sft] one-epoch Qwen3-8B training starts"
exec torchrun --standalone --nproc_per_node=8 \
  "$runtime_repo/scripts/train_tool_sft.py" --config "$training_config"
