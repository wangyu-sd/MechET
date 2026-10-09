#!/usr/bin/env bash
set -Eeuo pipefail

shared_repo=/aaa/fionafyang/buddy1/whaleywang/MechET
runtime_dir=$shared_repo/artifacts/taiji_vllm_runtime/vllm_0_8_5_torch_2_6_cu124_py311
model_dir=/aaa/fionafyang/buddy1/whaleywang/OpenEvolveChem/data/hf_cache/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca

source /root/miniconda3/etc/profile.d/conda.sh
conda activate meteor
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_USE_V1=0 VLLM_ATTENTION_BACKEND=XFORMERS
export PYTHONPATH=$runtime_dir${PYTHONPATH:+:$PYTHONPATH}

test -f "$runtime_dir/.mechet_vllm_runtime_complete"
test -d "$model_dir"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python -u - "$model_dir" <<'PY'
import asyncio
import sys
import torch
import vllm
from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs

model = sys.argv[1]
name = torch.cuda.get_device_name(0)
assert torch.cuda.device_count() == 1 and 'V100' in name.upper(), name
assert vllm.__version__ == '0.8.5', vllm.__version__
print({'stage': 'v100-import', 'gpu': name, 'torch': torch.__version__,
       'torch_cuda': torch.version.cuda, 'vllm': vllm.__version__}, flush=True)

async def main():
    engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
        model=model, tokenizer=model, dtype='float16', tensor_parallel_size=1,
        gpu_memory_utilization=0.75, max_model_len=512, max_num_seqs=2,
        enable_prefix_caching=True, enable_lora=True, max_lora_rank=16,
        enforce_eager=True, trust_remote_code=True,
    ))
    chunks = []
    async for output in engine.generate('Hello', SamplingParams(max_tokens=4), request_id='v100-probe'):
        chunks = output.outputs
    assert chunks and chunks[0].token_ids, 'empty generation'
    print({'stage': 'v100-generation-passed', 'output_tokens': len(chunks[0].token_ids)}, flush=True)

asyncio.run(main())
PY
