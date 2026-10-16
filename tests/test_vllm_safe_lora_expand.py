"""Numerical/runtime regression tests, not reaction-model evaluations."""
from types import SimpleNamespace

import pytest
import torch


def expand_function():
    from scripts.vllm_safe_lora_expand import safe_lora_expand
    return safe_lora_expand


@pytest.mark.parametrize('add_inputs', [False, True])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_expand_handles_gqa_slices_offsets_and_no_adapter_tokens(add_inputs, dtype):
    # Removing the token mask, using one shared slice width, or ignoring the
    # offset would each corrupt a different literal expected column below.
    expand = expand_function()
    inputs = torch.tensor([[[1., 2.], [3., 4.], [5., 6.]],
                           [[2., 1.], [4., 3.], [6., 5.]]], dtype=torch.float32)
    weights = [torch.tensor([[[1., 0.], [0., 1.]]], dtype=dtype),
               torch.tensor([[[1., 1.]]], dtype=dtype)]
    output = torch.full((3, 5), 10., dtype=dtype)
    expand(inputs, weights, output, torch.tensor([0, -1, 0]),
           torch.tensor([1, 0, 2]), torch.tensor([1, 2]),
           torch.tensor([0, 1, 3]), torch.tensor([-1, 0]),
           torch.tensor([False]), offset_start=1, add_inputs=add_inputs)
    wanted = ([[10., 11., 12., 13., 10.], [10., 10., 10., 10., 10.],
               [10., 15., 16., 21., 10.]] if add_inputs else
              [[10., 1., 2., 3., 10.], [10., 10., 10., 10., 10.],
               [10., 5., 6., 11., 10.]])
    torch.testing.assert_close(output, torch.tensor(wanted, dtype=dtype), rtol=0, atol=0)


def test_no_adapter_or_empty_batch_never_reads_weights():
    expand = expand_function()
    output = torch.full((2, 3), 7.)
    # Empty weights deliberately prove that the warm-up no-LoRA branch exits.
    expand(torch.empty(0), [], output, None, None, None, None, None,
           torch.tensor([True]))
    assert output.tolist() == [[7., 7., 7.], [7., 7., 7.]]
    expand(torch.empty(1, 0, 2), [torch.ones(1, 3, 2, dtype=torch.float16)], torch.empty(0, 3),
           torch.empty(0, dtype=torch.int32), None, None, None, None,
           torch.tensor([False]))


def test_bad_slice_cannot_partially_overwrite_the_output():
    expand = expand_function()
    output = torch.full((2, 3), 7.)
    with pytest.raises(ValueError, match='slice|width'):
        expand(torch.ones(2, 2, 2), [torch.ones(1, 2, 2, dtype=torch.float16),
                                   torch.ones(1, 2, 2, dtype=torch.float16)],
               output, torch.zeros(2, dtype=torch.int32), None, None, None, None,
               torch.tensor([False]))
    assert output.tolist() == [[7., 7., 7.], [7., 7., 7.]]


def test_second_adapter_is_rejected_not_silently_treated_as_first():
    from scripts.vllm_safe_lora_expand import validate_single_adapter_mapping
    with pytest.raises(ValueError, match='single.adapter'):
        validate_single_adapter_mapping(torch.tensor([0, 1]))
    with pytest.raises(ValueError, match='single.adapter'):
        validate_single_adapter_mapping(torch.tensor([-2, 0]))


def test_install_is_version_gated_idempotent_and_validates_before_metadata_copy():
    from scripts.vllm_safe_lora_expand import install_safe_lora_expand
    class Metadata:
        def prepare_tensors(self, mapping):
            self.copied = mapping.clone()
    original = lambda *a, **kw: None
    module = SimpleNamespace(lora_expand=original)
    with pytest.raises(ValueError, match='0.8.5'):
        install_safe_lora_expand('0.8.4', punica=module, metadata_type=Metadata)
    assert module.lora_expand is original
    receipt = install_safe_lora_expand('0.8.5+cu118', punica=module, metadata_type=Metadata)
    assert receipt['kernel'] == 'torch_single_adapter_expand_v1'
    assert install_safe_lora_expand('0.8.5+cu118', punica=module, metadata_type=Metadata) == receipt
    meta = Metadata()
    with pytest.raises(ValueError, match='single.adapter'):
        meta.prepare_tensors(torch.tensor([0, 1]))
    assert not hasattr(meta, 'copied')
    meta.prepare_tensors(torch.tensor([0, -1, 0]))
    assert meta.copied.tolist() == [0, -1, 0]
    assert module.lora_expand is expand_function()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU arithmetic verification needs CUDA')
def test_gpu_expand_real_model_widths_and_irregular_batch_tails():
    expand = expand_function()
    # Rank16/Qwen3-0.6B QKV widths, batch boundaries around the former tile64.
    for tokens in (1, 31, 63, 64, 65, 255, 256, 257):
        generator = torch.Generator(device='cuda').manual_seed(17 + tokens)
        inputs = torch.randn(3, tokens, 16, generator=generator, device='cuda')
        weights = [torch.randn(1, width, 16, generator=generator, device='cuda', dtype=torch.float16)
                   for width in (2048, 1024, 1024)]
        mapping = torch.zeros(tokens, device='cuda', dtype=torch.int32)
        mapping[::3] = -1
        output = torch.randn(tokens, 4096, generator=generator, device='cuda', dtype=torch.float16)
        expected = output.clone()
        offset = 0
        for index, weight in enumerate(weights):
            # Independent scalar-rank accumulation, not the production matmul.
            delta = torch.zeros(tokens, weight.shape[1], device='cuda')
            for rank in range(16):
                delta += inputs[index, :, rank:rank+1].half().float() * weight[0, :, rank].float()
            delta = delta.half()
            delta[mapping == -1] = 0
            expected[:, offset:offset+weight.shape[1]] += delta
            offset += weight.shape[1]
        expand(inputs, weights, output, mapping, None, None, None, None, torch.tensor([False]),
               add_inputs=True)
        torch.cuda.synchronize()
        torch.testing.assert_close(output, expected, rtol=0.002, atol=0.03125)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_actual_pinned_vllm_metadata_and_triton_numerical_comparison(dtype):
    """Opt-in through the staged vLLM PYTHONPATH; no model/data is loaded."""
    # is_bf16_supported() includes emulation on Turing in PyTorch2.6, but the
    # original Triton tensor-core BF16 comparison requires native Ampere SM80+.
    if dtype == torch.bfloat16 and torch.cuda.get_device_capability()[0] < 8:
        pytest.skip('production BF16 GPU check requires Ampere or newer')
    vllm = pytest.importorskip('vllm')
    if vllm.__version__.split('+')[0] != '0.8.5':
        pytest.skip('comparison requires the deployed vLLM 0.8.5')
    from vllm.lora.ops.triton_ops.lora_expand import _lora_expand
    from vllm.lora.ops.triton_ops.lora_kernel_metadata import LoRAKernelMeta
    from scripts.vllm_safe_lora_expand import install_safe_lora_expand, safe_lora_expand
    install_safe_lora_expand(vllm.__version__)
    metadata = LoRAKernelMeta.make(1, 1024, 'cuda')
    generator = torch.Generator(device='cuda').manual_seed(1700)
    # Compare the actual packaged Triton implementation, not a mocked kernel.
    # T4 runs FP16. On A100 this also exercises the production BF16 path.
    for widths in ((1024,), (3072, 3072), (2048, 1024, 1024)):
        weights = [torch.randn(1, width, 16, device='cuda', generator=generator,
                               dtype=dtype) for width in widths]
        for tokens in (1, 31, 63, 64, 65, 255, 256, 257):
            inputs = torch.randn(len(widths), tokens, 16, device='cuda', generator=generator)
            mapping = torch.zeros(tokens, device='cuda', dtype=torch.int32)
            mapping[::3] = -1
            metadata.prepare_tensors(mapping)
            reference = torch.randn(tokens, sum(widths), device='cuda', generator=generator,
                                    dtype=dtype)
            actual = reference.clone()
            _lora_expand(inputs, weights, reference, *metadata.meta_args(tokens), add_inputs=True)
            safe_lora_expand(inputs, weights, actual, *metadata.meta_args(tokens), add_inputs=True)
            torch.cuda.synchronize()
            tolerance = dict(rtol=0.002, atol=0.03125) if dtype == torch.float16 else dict(rtol=0.02, atol=0.125)
            torch.testing.assert_close(actual, reference, **tolerance)
