# Unified vLLM sampling

`scripts/infer_mechet.py` supports the same vLLM backend for every inference
condition:

- `direct`: outcome-only, free/State-CoT, net-edit, complete-proof, and open-flow;
- `trace`: trace-owned MechET with environment feedback;
- `textbook`, `irrelevant`, `anchors`, and `combined`: knowledge conditions;
- `legacy`: the legacy proof environment.

vLLM is an execution backend, not a scientific condition. It does not change
the prompts, tool schemas, environment transitions, stopping conditions,
candidate seeds, or prediction artifact schema. Each candidate/turn is sent as
an independently seeded request. Interactive modes repeatedly batch the active
trajectories, execute their tool calls in their independent environments, and
return unfinished trajectories to the next scheduler round.

Install the optional backend in a CUDA-12 environment with:

```bash
pip install -e '.[agent,inference]'
```

Qwen3 requires vLLM 0.8.5 or newer. The existing CUDA-11/V100 environment does
not satisfy that deployment contract and must use
`MECHET_INFERENCE_BACKEND=transformers`; A100/H20 CUDA-12 jobs use vLLM.

Example direct sampling:

```bash
python scripts/infer_mechet.py \
  --config configs/iclr/full_outcome_only_sft.yaml \
  --data data/iclr_full_v4/outcome_only/test.jsonl \
  --output outputs/eval/outcome-vllm.jsonl \
  --mode direct \
  --adapter outputs/iclr/full_outcome_only_seed17 \
  --backend vllm \
  --samples-per-target 10 \
  --direct-sample-batch-size 10 \
  --vllm-max-model-len 4096
```

Example trace-owned sampling:

```bash
python scripts/infer_mechet.py \
  --config configs/agent/tool_sft_flower_compact_full_state_qwen3_8b_a100.yaml \
  --data data/flower_inverse_tool_sft_compact_full_state_v1/test.jsonl \
  --output outputs/eval/a7-vllm.jsonl \
  --mode trace \
  --observation-mode compact_full_state \
  --prompt-source reference \
  --adapter outputs/agent/tool_sft_flower_compact_full_state_qwen3_8b_a100_20260826 \
  --backend vllm \
  --samples-per-target 10 \
  --trace-sample-batch-size 10 \
  --max-iterations 40 \
  --vllm-max-model-len 16384
```

For an 8B model, prefer one vLLM process per GPU and data-parallel dataset
sharding. Tensor parallelism is available through
`--vllm-tensor-parallel-size`, but should only be used when one model replica
does not fit on a single GPU. The Taiji launchers default to vLLM, emit a
heartbeat to the platform log, and retain `MECHET_INFERENCE_BACKEND=transformers`
as an explicit compatibility fallback.
