"""Versioned opt-in edge-terminal failures; legacy jobs keep their contract."""
from copy import deepcopy

FIRST_REJECTION = 'terminate_on_first_rejection_v1'
REWARD_VERSION = 'trusted_progress_first_rejection_cost_v2'


def check_failure_policy(value):
    if value not in (None, FIRST_REJECTION):
        raise ValueError('UNRECOGNIZED_ROUTE_FAILURE_POLICY')
    return value


def configure_first_rejection(cfg):
    from .route_process_reward import CONFIG
    cfg.update(route_failure_policy=FIRST_REJECTION, max_consecutive_recoveries=1,
               reward_contract=REWARD_VERSION,
               process_reward=dict(deepcopy(CONFIG), version=REWARD_VERSION))
    return cfg


def reward_kwargs(cfg,*,process=False):
    """Pass explicit opt-ins only, preserving legacy constructor bindings."""
    out={}
    for source,target in [('stock_material_contract','material_contract'),
                          ('route_failure_policy','failure_policy')]:
        if source in cfg:out[target]=cfg[source]
    if process and 'flow_credit_alpha' in cfg:out['flow_credit_alpha']=cfg['flow_credit_alpha']
    return out


def validate_failed_rollout(report, policy):
    """Bind scoring to collection and forbid a continuation after a rejection."""
    check_failure_policy(policy)
    if report.get('route_failure_policy') != policy:
        raise ValueError('REWARD_FAILURE_POLICY_MISMATCH')
    if policy != FIRST_REJECTION:
        return
    records = report.get('records', [])
    failed = [i for i, r in enumerate(records)
              if r.get('accepted') is False and r.get('error')
              and not r['error'].startswith('RESOURCE_CENSORED:')]
    censored = report.get('status') in ('resource_censored', 'protected_censored')
    if failed and (failed != [len(records)-1] or
                   (not censored and report.get('status') != 'rejected_action')):
        raise ValueError('FIRST_REJECTION_CONTINUATION_OR_STATUS')
    if report.get('status') == 'rejected_action' and not failed:
        raise ValueError('FIRST_REJECTION_RECEIPT_REQUIRED')
