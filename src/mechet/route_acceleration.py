"""Explicit new sampling condition; defaults never mutate frozen PR110 jobs."""
from copy import deepcopy

SKIPPED='duplicate_pruned_no_continuation'
DEFAULT_ACCELERATION=dict(version='pr111_compute_lanes_dynamic_v1',execution_cache_capacity=8192,
    duplicate_suffix_policy='skip_extra_duplicates_v1',scheduling='global_requests_v1',
    hf_length_bucketing=True,sampling_kernel='counter_categorical_v1')


def validate_acceleration(cfg):
    value=cfg.get('acceleration')
    if value is None:return
    if (not isinstance(value,dict) or set(value)!=set(DEFAULT_ACCELERATION) or
            any(value[k]!=v for k,v in DEFAULT_ACCELERATION.items() if k!='execution_cache_capacity') or
            type(value['execution_cache_capacity']) is not int or value['execution_cache_capacity']<1 or
            not cfg.get('refit') or cfg.get('candidates')!=4 or cfg.get('world_size')!=4 or
            cfg.get('inference_backend','hf')!='hf'):
        raise ValueError('ACCELERATION_EXPLICIT_NEW_CONDITION_REQUIRED')


def skipped_reward(report):
    if report.get('status')!=SKIPPED:raise ValueError('NOT_A_BUDGET_SKIPPED_REPORT')
    return dict(reward=None,process_credit=None,budget_skipped=True,censored=False,
        verified_stock_closed=False,structural_stock_closed=False,reward_status=SKIPPED,
        supported_reactions=0,unsupported_reactions=0,
        chemical_validation='not_evaluated_no_genuine_suffix_return')


def attach_execution_cache(episodes,cfg):
    if not cfg.get('acceleration'):return
    from .route_execution_cache import ExecutionCache,CachedChemistry
    cache=cfg.setdefault('_execution_cache',ExecutionCache(cfg['acceleration']['execution_cache_capacity']))
    from .external_route_data import signature
    contracts=dict(executor_sources_sha=signature(cfg.get('code_files_sha256',{})),
        endpoint=cfg['route_endpoint_contract'],material=cfg.get('stock_material_contract'),
        failure=cfg.get('route_failure_policy'))
    for ep in episodes:ep.chemistry=CachedChemistry(ep.chemistry,cache,contracts)
