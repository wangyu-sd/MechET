"""Checked single-adapter GPU expand for pinned eager vLLM 0.8.5.

Avoids Punica's raw-pointer Triton expand kernel; weights remain unmerged.
Shrink, attention, scheduling and sampling are unchanged. The same BF16/FP16
cast-before-GEMM and cast-before-add semantics are retained, but cuBLAS and
Triton reductions need not be bit-identical. Record this runtime in manifests.
Never use this specialization with multiple resident adapters or CUDA graphs.
"""
from functools import wraps

import torch


KERNEL = 'torch_single_adapter_expand_v1'


def validate_single_adapter_mapping(mapping):
    """Fail before GPU metadata preparation if a second adapter is admitted.

    One synchronization per metadata update, not per layer or expand call.
    No-LoRA (-1) profile/warm-up rows remain supported.
    """
    if mapping.ndim != 1 or (mapping.numel() and
            not bool(((mapping == -1) | (mapping == 0)).all().item())):
        raise ValueError('safe expand requires single-adapter slot 0 or no-adapter slot -1')


@torch.inference_mode()
def safe_lora_expand(inputs, lora_b_weights, output_tensor, token_lora_mapping,
                     token_indices_sorted_by_lora_ids, num_tokens_per_lora,
                     lora_token_start_loc, lora_ids, no_lora_flag_cpu,
                     offset_start=0, add_inputs=False):
    """vLLM expand ABI, without indirect row/weight pointers or grouped tiles.

    Mapping values are checked by the installed prepare_tensors guard once per
    forward. The equality mask also never indexes weights with these values.
    Scratch memory scales with tokens * slice width, not tokens * weights.
    """
    if no_lora_flag_cpu.numel() != 1:
        raise ValueError('expected one CPU no-LoRA flag')
    if no_lora_flag_cpu.item():
        return
    if (inputs.ndim != 3 or output_tensor.ndim != 2 or offset_start < 0
            or inputs.shape[0] != len(lora_b_weights)
            or inputs.shape[1] != output_tensor.shape[0]
            or token_lora_mapping.shape != (inputs.shape[1],)):
        raise ValueError('LoRA expand input/output/mapping shape mismatch')
    weights = [w.squeeze(1) if w.ndim == 4 and w.shape[1] == 1 else w
               for w in lora_b_weights]
    if any(w.ndim != 3 or w.shape[0] != 1 or w.shape[2] != inputs.shape[2]
           or w.dtype not in (torch.float16, torch.bfloat16)
           or w.device != inputs.device for w in weights):
        raise ValueError('safe expand requires single-adapter weights with matching rank/device')
    if output_tensor.device != inputs.device or token_lora_mapping.device != inputs.device:
        raise ValueError('LoRA expand device mismatch')
    if offset_start + sum(w.shape[1] for w in weights) > output_tensor.shape[1]:
        raise ValueError('LoRA output slice widths exceed output width')
    if not inputs.shape[1]:
        return
    active = (token_lora_mapping == 0).unsqueeze(1)
    offset = offset_start
    for index, weight in enumerate(weights):
        width = weight.shape[1]
        # Upstream CAST_TYPE converts the FP32 shrink buffer to B's dtype;
        # expand rounds its accumulator to B's dtype before adding the base.
        delta = inputs[index].to(weight.dtype) @ weight[0].transpose(0, 1)
        target = output_tensor[:, offset:offset + width]
        if add_inputs:
            target.add_(delta.masked_fill(~active, 0))
        else:
            target.copy_(torch.where(active, delta, target))
        offset += width


def install_safe_lora_expand(version, *, punica=None, metadata_type=None):
    """Install before LLM construction (including profiling); fail closed."""
    if version.split('+')[0] != '0.8.5':
        raise ValueError('safe LoRA expand is validated only for vLLM 0.8.5')
    if punica is None:
        from vllm.lora.punica_wrapper import punica_gpu as punica
    if metadata_type is None:
        from vllm.lora.ops.triton_ops.lora_kernel_metadata import LoRAKernelMeta
        metadata_type = LoRAKernelMeta
    original = metadata_type.prepare_tensors
    if not getattr(original, '_mechet_single_adapter_guard', False):
        @wraps(original)
        def prepare(self, mapping):
            validate_single_adapter_mapping(mapping)
            return original(self, mapping)
        prepare._mechet_single_adapter_guard = True
        metadata_type.prepare_tensors = prepare
    punica.lora_expand = safe_lora_expand
    return {'kernel': KERNEL, 'vllm': version, 'weight_merge': False,
            'bitwise_equivalence_claimed': False, 'multiple_adapters_supported': False}
