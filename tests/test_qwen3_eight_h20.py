import json

import pytest


def test_eight_h20_reuses_protocol_cache_and_keeps_global_batch():
    from mechet.qwen3_continuation import continuation_config, validate_config, validate_cache
    from test_qwen3_continuation import cache
    old = continuation_config()
    cfg = continuation_config(world_size=8)
    assert cfg['training']['gradient_accumulation_steps'] * 8 == 64
    assert cfg['pretokenized_cache_dir'] == old['pretokenized_cache_dir']
    assert cfg['pretokenization_world_size'] == 4
    assert cfg['output_dir'] != old['output_dir']
    validate_config(cfg, world_size=8, gpu_names=['NVIDIA H20'] * 8)
    validate_cache(cfg, cache())
    with pytest.raises(ValueError):
        validate_config(cfg, world_size=4, gpu_names=['NVIDIA H20'] * 4)


def test_eight_rank_resume_requires_all_eight_rng_states(tmp_path):
    from mechet.qwen3_continuation import continuation_config, save_checkpoint_binding, verify_resume
    cfg = continuation_config(world_size=8)
    for name in ['adapter_model.safetensors', 'adapter_config.json', 'optimizer.pt', 'scheduler.pt',
                 *[f'rng_state_{i}.pth' for i in range(8)]]:
        (tmp_path / name).write_bytes(b'checkpoint')
    (tmp_path / 'trainer_state.json').write_text(json.dumps({'global_step': 8}))
    saved = save_checkpoint_binding(cfg, tmp_path)
    assert 'rng_state_7.pth' in saved['files']
    verify_resume(cfg, tmp_path)
    (tmp_path / 'rng_state_7.pth').unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        verify_resume(cfg, tmp_path)


def test_other_gpu_allocations_are_not_new_conditions():
    from mechet.qwen3_continuation import continuation_config
    for n in (0, 1, 16):
        with pytest.raises(ValueError):
            continuation_config(world_size=n)
