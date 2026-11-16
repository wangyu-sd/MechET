"""Post-rollout verifiable process rewards and critic-free action-level credit.

No oracle fields enter collection. Legacy scalar scoring remains unchanged.
"""
from collections import Counter
import hashlib
import json
import math

from .external_route_rl import HiddenRouteReward, PROCESS_GRPO_ALGORITHM
from .route_action_feedback import semantic_action

ALGORITHM = PROCESS_GRPO_ALGORITHM
VERSION = 'trusted_progress_bounded_validator_cost_v1'
CONFIG = dict(version=VERSION, progress_weight=.2,
              validator_weight=.05, cost_scale=4., repeat_extra=1., cycle_cost=2.)


def validate_process_protocol(cfg):
    if cfg.get('algorithm') != ALGORITHM:
        if 'process_reward' in cfg:
            raise ValueError('PROCESS_REWARD_REQUIRES_EXPLICIT_ALGORITHM')
        return
    from .route_native_observation import NATIVE_SFT_ROUTE
    from .route_failure_policy import FIRST_REJECTION, REWARD_VERSION, check_failure_policy
    policy=check_failure_policy(cfg.get('route_failure_policy'))
    version=REWARD_VERSION if policy==FIRST_REJECTION else VERSION
    expected=dict(CONFIG,version=version)
    if (cfg.get('process_reward')!=expected or cfg.get('reward_contract')!=version or
            cfg.get('advantage_estimator')!='centered_action_return_to_go_no_std' or
            cfg.get('public_observation_contract')!=NATIVE_SFT_ROUTE or
            cfg.get('partial_reward_weight')!=.2):
        raise ValueError('PROCESS_PROTOCOL_CHANGED')
    if policy==FIRST_REJECTION and cfg.get('max_consecutive_recoveries')!=1:
        raise ValueError('PROCESS_PROTOCOL_FIRST_REJECTION_CEILING')
    if 'flow_credit_alpha' in cfg:
        from .route_flow_credit import validate_alpha
        alpha=validate_alpha(cfg['flow_credit_alpha'])
        if alpha and policy!=FIRST_REJECTION:
            raise ValueError('PROCESS_PROTOCOL_FLOW_CREDIT_FAILURE_POLICY')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def returns_to_go(rewards):
    values = [float(v) for v in rewards]
    if not all(math.isfinite(v) for v in values):
        raise ValueError('PROCESS_NONFINITE_REWARD')
    return [math.fsum(values[i:]) for i in range(len(values))]


def process_group_advantages(rewards):
    """Center future returns across independent siblings at each tool position.

    Absorbed siblings have zero future reward. This is not chemical-state
    grouping or leave-one-out. No standard-deviation/length scaling is used.
    """
    if len(rewards) < 2:
        raise ValueError('PROCESS_INDEPENDENT_GROUP_REQUIRED')
    if any(r is None for r in rewards):
        return None
    returns = [returns_to_go(r) for r in rewards]
    horizon = max(map(len, returns), default=0)
    means = [math.fsum(r[t] if t < len(r) else 0. for r in returns)/len(returns)
             for t in range(horizon)]
    return [[value-means[t] for t, value in enumerate(r)] for r in returns]


class ProcessRouteReward(HiddenRouteReward):
    def __init__(self,*args,flow_credit_alpha=0.,**kwargs):
        from .route_flow_credit import validate_alpha
        from .route_failure_policy import FIRST_REJECTION
        self.flow_credit_alpha=validate_alpha(flow_credit_alpha)
        if self.flow_credit_alpha and kwargs.get('failure_policy')!=FIRST_REJECTION:
            raise ValueError('FLOW_CREDIT_REQUIRES_FIRST_REJECTION_CONTRACT')
        super().__init__(*args,**kwargs)

    def score(self, task_id, report):
        from .route_failure_policy import FIRST_REJECTION, REWARD_VERSION
        version=REWARD_VERSION if self.failure_policy==FIRST_REJECTION else VERSION
        result = super().score(task_id, report, include_progress=True)
        if result['reward'] is None:
            return dict(result, process_reward_version=version, process_credit=None)
        records = report['records']
        if not records and report['status']!='already_stock':
            raise ValueError('PROCESS_OBSERVED_DECISIONS_REQUIRED')
        ids = [r['request_id'] for r in records]
        if len(set(ids)) != len(ids):
            raise ValueError('PROCESS_DUPLICATE_REQUEST_ID')
        actions = [semantic_action(r.get('raw', '')) for r in records]
        commits = [(i, r['occurrence']) for i, (r, action) in enumerate(zip(records, actions))
                   if r.get('accepted') is True and action and action[1]['name']=='finish_trace']
        if [o for _, o in commits] != [r['occurrence'] for r in report['reactions']]:
            raise ValueError('PROCESS_COMMIT_RECEIPT_MISMATCH')
        events = [0.] * len(records)
        increments=[]
        progress = 0.
        for item in result.pop('progress_events'):
            supported = item['supported_reactions']
            next_progress = .2 * min(supported/max(result['reference_steps'], 1), 1.)
            index = commits[item['reaction_index']][0]
            events[index] += next_progress-progress
            increments.append((index,next_progress-progress))
            progress = next_progress
        from .route_flow_credit import redistribute_progress
        events,attribution=redistribute_progress(records,events,increments,alpha=self.flow_credit_alpha)
        task_reward = (1. if result['verified_stock_closed'] else
                       0. if result['graph_error'] else progress)
        if self.failure_policy==FIRST_REJECTION and report['status']=='rejected_action':
            task_reward=0.
        error_weight = 0.
        cost = 0.
        penalty_scale = .05/max(result['reference_steps'], 1)
        seen = Counter()
        for i, (record, action) in enumerate(zip(records, actions)):
            error = record.get('error', '')
            if record.get('accepted') is not False or not error:
                continue
            key = (record['occurrence'], record['state_before'],
                   action[0] if action else record.get('raw', ''))
            error_weight += (2. if error.startswith('ROUTE_CYCLE') else 1.)
            if seen[key] or record.get('executor_skipped')=='repeated_rejected_action':
                error_weight += 1.
            seen[key] += 1
            next_cost = -math.expm1(-error_weight/4.)
            events[i] -= penalty_scale*(next_cost-cost)
            cost = next_cost
        if events:events[-1] += task_reward-progress
        final_reward = task_reward-penalty_scale*cost
        if not math.isclose(math.fsum(events), final_reward, rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError('PROCESS_REWARD_CONSERVATION_FAILED')
        credit = dict(version=version, request_ids=ids, event_rewards=events,
                      returns=returns_to_go(events), task_reward=task_reward,
                      validator_cost=cost, validator_error_weight=error_weight,
                      validator_penalty_scale=penalty_scale)
        if self.flow_credit_alpha:
            credit['flow_credit']=dict(alpha=self.flow_credit_alpha,evidence=attribution)
        credit['sha256'] = digest(credit)
        return dict(result, base_route_reward=result['reward'], reward=final_reward,
                    route_task_reward=task_reward, process_reward_version=version,
                    process_credit=credit)


def process_sequences(report, advantages):
    from .route_failure_policy import FIRST_REJECTION, REWARD_VERSION
    version=REWARD_VERSION if report.get('route_failure_policy')==FIRST_REJECTION else VERSION
    credit = report['reward_evidence'].get('process_credit')
    if not isinstance(credit, dict) or credit.get('version') != version:
        raise ValueError('PROCESS_CREDIT_REQUIRED')
    payload = {k:v for k,v in credit.items() if k!='sha256'}
    records = report['records']
    if (digest(payload) != credit.get('sha256') or
            credit.get('request_ids') != [r['request_id'] for r in records] or
            len(advantages) != len(records) or
            len(credit.get('event_rewards', [])) != len(records) or
            credit.get('returns') != returns_to_go(credit['event_rewards']) or
            not math.isclose(math.fsum(credit['event_rewards']), report['reward'],
                             rel_tol=1e-10, abs_tol=1e-12)):
        raise ValueError('PROCESS_CREDIT_BINDING_MISMATCH')
    for record, advantage in zip(records, advantages):
        if not math.isfinite(advantage):
            raise ValueError('PROCESS_NONFINITE_ADVANTAGE')
        p, r, b = (record.get(k) for k in
                   ('prompt_token_ids', 'generated_token_ids', 'behavior_logprobs'))
        if r:
            if not p or b is None or len(b)!=len(r):
                raise ValueError('PROCESS_POLICY_TOKEN_RECEIPT_MISMATCH')
            yield p, r, b, advantage
