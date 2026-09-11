#!/usr/bin/env bash
set -Eeuo pipefail

repo_dir=/aaa/fionafyang/buddy1/whaleywang/MechET
python_bin=/aaa/fionafyang/buddy1/whaleywang/miniconda3/envs/meteor/bin/python
shared_hf_cache=/aaa/fionafyang/buddy1/whaleywang/OpenEvolveChem/data/hf_cache
revision=b968826d9c46dd6066d109eabc6255188de91218

echo "[meteor-probe] stage=container time=$(date --iso-8601=seconds)"
cat /etc/os-release | sed -n '1,6p'
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

test -x "$python_bin"
test -d "$repo_dir"
cd "$repo_dir"

export HF_HUB_CACHE="$shared_hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$repo_dir/src:$repo_dir${PYTHONPATH:+:$PYTHONPATH}"

echo "[meteor-probe] stage=python-cuda time=$(date --iso-8601=seconds)"
"$python_bin" -u - <<'PY'
import json

import torch

assert torch.version.cuda == "11.8", torch.version.cuda
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 1
name = torch.cuda.get_device_name(0)
assert "A100" in name.upper(), name
x = torch.randn((2048, 2048), device="cuda", dtype=torch.float16)
y = x @ x
torch.cuda.synchronize()
assert torch.isfinite(y).all().item()
print(json.dumps({
    "stage": "torch-cuda-ok",
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "gpu": name,
    "matmul_shape": list(y.shape),
}), flush=True)

import peft
from rdkit import Chem
import transformers

assert Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(=O)O") is not None
print(json.dumps({
    "stage": "python-dependencies-ok",
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "transformers": transformers.__version__,
    "peft": peft.__version__,
    "gpu": name,
    "matmul_shape": list(y.shape),
}), flush=True)
PY

echo "[meteor-probe] stage=qwen-load time=$(date --iso-8601=seconds)"
"$python_bin" -u - "$revision" <<'PY'
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

revision = sys.argv[1]
model_name = "Qwen/Qwen3-8B"
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    revision=revision,
    local_files_only=True,
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    revision=revision,
    local_files_only=True,
    torch_dtype=torch.bfloat16,
    device_map={"": 0},
)
inputs = tokenizer("Product: CCO", return_tensors="pt").to("cuda")
with torch.inference_mode():
    output = model.generate(**inputs, max_new_tokens=4, do_sample=False)
print({
    "loaded": model_name,
    "generated_tokens": int(output.shape[-1] - inputs.input_ids.shape[-1]),
    "gpu_memory_allocated_gib": round(torch.cuda.memory_allocated() / 2**30, 3),
}, flush=True)
PY

echo "[meteor-probe] stage=success time=$(date --iso-8601=seconds)"
