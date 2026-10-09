#!/usr/bin/env bash
set -Eeuo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate meteor
export PYTHONUNBUFFERED=1
export PATH=$PATH:/root/miniconda3/bin
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_USE_V1=0 VLLM_ATTENTION_BACKEND=XFORMERS

shared_repo=/aaa/fionafyang/buddy1/whaleywang/MechET
runtime_repo=/aaa/fionafyang/buddy1/whaleywang/MechET-flower-stage3-full-20261008
runtime_archive=$shared_repo/artifacts/taiji_vllm_runtime/vllm_0_8_5_torch_2_6_cu118_py311.tar.zst
dependency_archive=$shared_repo/artifacts/taiji_vllm_runtime/vllm_0_8_5_torch_2_6_cu124_py311_pruned.tar.zst
model_snapshot=/aaa/fionafyang/buddy1/whaleywang/OpenEvolveChem/data/hf_cache/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca
actor_adapter=$shared_repo/outputs/agent/natural_language_event_history_v2_qwen3_0_6b_seed17

echo '[earho-vllm-cuda11-probe] allocated GPU'
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
df -h /tmp
python -u - <<'PY'
import torch
assert torch.version.cuda == "11.8", torch.version.cuda
assert torch.cuda.device_count() == 1
assert "V100" in torch.cuda.get_device_name(0).upper()
torch.empty(1, device="cuda").add_(1).cpu()
print({"gate": "cuda11-base-passed", "torch": torch.__version__,
       "cuda": torch.version.cuda}, flush=True)
PY

test -s "$runtime_archive"
test -s "$dependency_archive"
test -f "$model_snapshot/config.json"
command -v zstd
echo "23c44e907d6f50ed75f46f908063a3d6c37144d676dd444cc055df9a93b98f9c  $runtime_archive" | sha256sum --check --strict
runtime_dir=$(mktemp -d /tmp/mechet_earho_cuda11_vllm.XXXXXX)
mkdir -p "$runtime_dir/cu118" "$runtime_dir/fallback"
echo "[earho-vllm-cuda11-probe] staging runtime in $runtime_dir"
tar --zstd -xf "$runtime_archive" -C "$runtime_dir/cu118"
# Never put CUDA 12 GPU packages in the CUDA 11.8 Python search path.
tar --zstd --exclude='./torch*' --exclude='./functorch*' \
  --exclude='./nvidia*' --exclude='./triton*' --exclude='./vllm*' \
  --exclude='./xformers*' --exclude='./cupy*' --exclude='./cupyx*' \
  --exclude='./cuda*' -xf "$dependency_archive" -C "$runtime_dir/fallback"
cp "$runtime_repo/scripts/earho_cuda11_sitecustomize.py" "$runtime_dir/cu118/sitecustomize.py"
export PYTHONPATH="$runtime_dir/cu118"
export MECHET_VLLM_PUREPY_FALLBACK="$runtime_dir/fallback"

python -u - "$model_snapshot" "$actor_adapter" <<'PY'
import asyncio
import sys
import torch
import vllm
import vllm._C
import xformers
from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.lora.request import LoRARequest
from vllm.platforms import current_platform

assert torch.version.cuda == "11.8", torch.version.cuda
assert vllm.__version__ == "0.8.5", vllm.__version__
assert xformers.__version__ == "0.0.29.post2", xformers.__version__
print({"gate": "cuda11-vllm-import-passed", "torch_cuda": torch.version.cuda,
       "vllm": vllm.__version__, "xformers": xformers.__version__}, flush=True)
current_platform.get_punica_wrapper = lambda: (
    "vllm.lora.punica_wrapper.punica_cpu.PunicaWrapperCPU"
)
assert current_platform.get_punica_wrapper().endswith("PunicaWrapperCPU")

async def probe():
    engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
        model=sys.argv[1], tokenizer=sys.argv[1], dtype="float16",
        enforce_eager=True, max_model_len=4096, max_num_seqs=32,
        gpu_memory_utilization=0.7, enable_lora=True, max_lora_rank=16,
        max_loras=1, max_cpu_loras=1,
    ))
    try:
        lora = LoRARequest("nl_anchor_actor", 1, sys.argv[2])
        params = SamplingParams(max_tokens=16, temperature=0.0, logprobs=0)
        async def one(index):
            last = None
            async for result in engine.generate(
                "Predict the next electron-flow action for CCO.", params,
                request_id=f"v100-lora-probe-{index}", lora_request=lora,
            ):
                last = result
            assert last and last.outputs and last.outputs[0].token_ids
            return len(last.outputs[0].token_ids)
        counts = await asyncio.gather(*(one(index) for index in range(4)))
        print({"gate": "cuda11-vllm-qwen3-lora-async-generation-passed",
               "generated_tokens": counts}, flush=True)
    finally:
        if hasattr(engine, "shutdown"):
            engine.shutdown()
        else:
            engine.shutdown_background_loop()

asyncio.run(probe())
PY
echo '[earho-vllm-cuda11-probe] completed'
