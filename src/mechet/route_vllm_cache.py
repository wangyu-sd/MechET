"""Exact engine-prefix identity and quiescent adapter invalidation."""
from .external_route_data import signature


def prefix_namespace(cfg,version):
    return signature(dict(version=version,model=cfg.get('base_path'),
        tokenizer_manifest=cfg.get('base_manifest'),contract=cfg.get('_sha'),
        tools=cfg.get('public_observation_contract'),precision=cfg.get('policy_precision'),
        engine_dtype=cfg.get('vllm_runtime',{}).get('diagnostic_dtype','bfloat16')))


async def invalidate_prefix(engine,lease,enabled):
    lease.invalidate()  # Refuses active requests before touching the engine.
    if enabled and await engine.reset_prefix_cache() is not True:
        raise ValueError('VLLM_PREFIX_RESET_FAILED')
    return bool(enabled)
