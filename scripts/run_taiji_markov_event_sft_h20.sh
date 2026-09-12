#!/usr/bin/env bash
set -Eeuo pipefail

shared_repo=/aaa/fionafyang/buddy1/whaleywang/MechET
runtime_repo=${MECHET_MARKOV_SFT_RUNTIME_DIR:?set MECHET_MARKOV_SFT_RUNTIME_DIR}
shared_hf_cache=/aaa/fionafyang/buddy1/whaleywang/OpenEvolveChem/data/hf_cache
source_dir=$shared_repo/data/flower_in_place_grounded_flow_v1
output_dir=$shared_repo/data/flower_markov_event_sft_v2
training_config=$runtime_repo/configs/agent/markov_event_sft_qwen3_8b_h20.yaml
liger_wheel=$shared_repo/artifacts/wheels/liger_kernel-0.6.2-py3-none-any.whl
xformers_wheel=$shared_repo/artifacts/wheels/xformers-0.0.29.post3-cp311-cp311-manylinux_2_28_x86_64.whl
bitsandbytes_wheel=$shared_repo/artifacts/wheels/bitsandbytes-0.49.2-py3-none-manylinux_2_24_x86_64.whl

source /root/miniconda3/etc/profile.d/conda.sh
conda activate meteor
cd "$shared_repo"

echo "[markov-sft] runtime=$runtime_repo source=$source_dir output=$output_dir"
test -f "$training_config"
test -f "$source_dir/train.jsonl"
echo "303b9bbf5c10f9289c3139afb41e4d989e8c809516624a106b89b064163d971d  $liger_wheel" | sha256sum --check --strict
echo "bbf2f500dfdbcf4649bf568cc2c9f434399f704dc4064fd1fbdbef2b524a8139  $xformers_wheel" | sha256sum --check --strict
echo "54b771f06e1a3c73af5c7f16ccf0fc23a846052813d4b008d10cb6e017dd1c8c  $bitsandbytes_wheel" | sha256sum --check --strict

runtime_target=$(mktemp -d /tmp/mechet_markov_runtime.XXXXXX)
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
names = [torch.cuda.get_device_name(i) for i in range(8)]
if not all("H20" in name.upper() for name in names):
    raise SystemExit(f"expected H20 GPUs, got {names}")
print({"gpus": names, "rdkit_version": rdkit.__version__}, flush=True)
PY

manifest=$output_dir/manifest.json
if ! python - "$manifest" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
if not p.is_file(): raise SystemExit(1)
m=json.loads(p.read_text())
expected_reactions={'train':257167,'valid':2890,'test':28967}
expected_decisions={'train':1300519,'valid':14422,'test':146332}
ok=m.get('observation_contract')=='target_plus_executor_current_state_v1'
ok &= m.get('gold_horizon_model_visible') is False
ok &= m.get('training_allowed') is True and m.get('gate_passed') is True
ok &= m.get('reaction_denominator')==expected_reactions
ok &= m.get('decision_rows')==expected_decisions
raise SystemExit(0 if ok else 1)
PY
then
  echo "[markov-sft] building audited one-step dataset"
  python -u "$runtime_repo/scripts/build_markov_event_sft.py" \
    --source-dir "$source_dir" --output-dir "$output_dir"
fi

python - "$manifest" <<'PY'
import json,sys
from pathlib import Path
m=json.loads(Path(sys.argv[1]).read_text())
expected_reactions={'train':257167,'valid':2890,'test':28967}
expected_decisions={'train':1300519,'valid':14422,'test':146332}
assert m['reaction_denominator']==expected_reactions
assert m['decision_rows']==expected_decisions
assert not any(m['split_reaction_id_overlap'].values())
assert m['gold_horizon_model_visible'] is False
assert m['model_visible_atom_maps'] is False
print({'dataset_gate':'passed','reactions':expected_reactions,'decisions':expected_decisions},flush=True)
PY

cache_manifest=$output_dir/qwen3_8b_tokens_3072/manifest.json
if [[ ! -f "$cache_manifest" ]]; then
  echo "[markov-sft] distributed tokenization starts"
  torchrun --standalone --nproc_per_node=8 \
    "$runtime_repo/scripts/prepare_tool_sft_arrow.py" --config "$training_config"
fi

python - "$cache_manifest" <<'PY'
import json,sys
from pathlib import Path
m=json.loads(Path(sys.argv[1]).read_text())
assert int(m['splits']['train']['n_rows']) == 1300519
assert int(m['splits']['validation']['n_rows']) == 14422
assert int(m['splits']['train'].get('truncation_count',0)) == 0
print({'token_gate':'passed','train':m['splits']['train']},flush=True)
PY

echo "[markov-sft] one-epoch Qwen3-8B H20 training starts"
exec torchrun --standalone --nproc_per_node=8 \
  "$runtime_repo/scripts/train_tool_sft.py" --config "$training_config"
