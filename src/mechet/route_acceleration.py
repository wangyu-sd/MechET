"""Explicit new sampling condition; defaults never mutate frozen PR110 jobs."""
from copy import deepcopy

SKIPPED='duplicate_pruned_no_continuation'
DEFAULT_ACCELERATION=dict(version='pr111_compute_lanes_dynamic_v1',execution_cache_capacity=8192,
    duplicate_suffix_policy='skip_extra_duplicates_v1',scheduling='global_requests_v1',
    hf_length_bucketing=True,sampling_kernel='counter_categorical_v1')


def full_acceleration(*,refit):
    """Explicit native condition; the frozen HF version remains independently valid."""
    return dict(version='pr111_full_acceleration_v2',execution_cache_capacity=8192,
        duplicate_suffix_policy='skip_extra_duplicates_v2' if refit else 'disabled',
        scheduling='global_length_fair_v2',hf_length_bucketing=False,
        sampling_kernel='vllm_native_v1',prefix_cache=True,
        dispatch_max_prefill_tokens=8192,dispatch_max_estimated_tokens=131072,
        dispatch_max_pending=4096,reference_cache_capacity=512,
        refit_lanes=dict(credit_forks_per_tree=16,exploration_forks_per_tree=0) if refit else None)


def dispatch_settings(cfg):
    value=cfg.get('acceleration',{})
    if value.get('scheduling')!='global_length_fair_v2':return None
    return dict(max_prefill_tokens=value['dispatch_max_prefill_tokens'],
        max_estimated_tokens=value['dispatch_max_estimated_tokens'],
        max_pending=value['dispatch_max_pending'],gpu_count=cfg['world_size'])


def validate_acceleration(cfg):
    value=cfg.get('acceleration')
    if value is None:return
    if isinstance(value,dict) and value.get('version')=='pr111_full_acceleration_v2':
        expected=full_acceleration(refit=bool(cfg.get('refit')))
        variable={'execution_cache_capacity','dispatch_max_prefill_tokens',
                  'dispatch_max_estimated_tokens','dispatch_max_pending','reference_cache_capacity','refit_lanes'}
        if (set(value)!=set(expected) or any(value[k]!=v for k,v in expected.items() if k not in variable)
                or cfg.get('inference_backend','hf')!='vllm' or cfg.get('candidates')!=4 or cfg.get('world_size')!=4
                or any(type(value[k]) is not int or value[k]<(0 if k=='reference_cache_capacity' else 1)
                       for k in variable-{'refit_lanes'})):
            raise ValueError('FULL_ACCELERATION_EXPLICIT_NATIVE_CONDITION_REQUIRED')
        lanes=value['refit_lanes']
        if cfg.get('refit'):
            if (not isinstance(lanes,dict) or set(lanes)!=set(expected['refit_lanes'])
                    or any(type(v) is not int or v<0 for v in lanes.values())
                    or lanes['credit_forks_per_tree']<1
                    or sum(lanes.values())>cfg['refit']['max_forks_per_tree']):
                raise ValueError('FULL_ACCELERATION_REFIT_LANE_BUDGETS')
        elif lanes is not None:raise ValueError('FULL_ACCELERATION_D1_WITH_REFIT_LANES')
        return
    expected_duplicate='skip_extra_duplicates_v1' if cfg.get('refit') else 'disabled'
    if (not isinstance(value,dict) or set(value)!=set(DEFAULT_ACCELERATION) or
            any(value[k]!=v for k,v in DEFAULT_ACCELERATION.items() if k not in (
                'execution_cache_capacity','duplicate_suffix_policy','sampling_kernel')) or
            type(value['execution_cache_capacity']) is not int or value['execution_cache_capacity']<1 or
            value['duplicate_suffix_policy']!=expected_duplicate or
            value['sampling_kernel'] not in ('counter_categorical_v1','private_pinned_argmax_v1') or
            cfg.get('candidates')!=4 or cfg.get('world_size')!=4 or
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
