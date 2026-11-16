"""Post-rollout hidden endpoint rewards and on-policy full-route optimization."""
from collections import Counter, OrderedDict
import json
import math
from pathlib import Path
import sqlite3
import traceback

from .external_route_data import edge_signature
from .route_training_contract import chemical_identity
from .route_policy_episode import RouteEpisode


DR_GRPO_ALGORITHM = 'external_target_only_route_dr_grpo_v2'
RLOO_ALGORITHM = 'external_target_only_route_rloo_v1'
PROCESS_GRPO_ALGORITHM = 'external_target_only_route_process_grpo_v1'


def route_group_advantages(rewards, algorithm=DR_GRPO_ALGORITHM):
    """Declared group baseline; censor a whole group, never an individual route.

    The primary Dr.GRPO condition uses mean centering without standard-deviation
    scaling. The historical RLOO baseline is retained under its explicit name.
    """
    if algorithm not in (DR_GRPO_ALGORITHM,RLOO_ALGORITHM,PROCESS_GRPO_ALGORITHM):
        raise ValueError('UNSUPPORTED_ROUTE_ALGORITHM:'+str(algorithm))
    if len(rewards)<2:
        raise ValueError('ROUTE_GROUP_REQUIRES_TWO_INDEPENDENT_EPISODES')
    if any(reward is None for reward in rewards):
        return None
    values=[float(reward) for reward in rewards]
    if not all(math.isfinite(value) for value in values):
        raise ValueError('NONFINITE_ROUTE_REWARD')
    mean=math.fsum(values)/len(values)
    scale=len(values)/(len(values)-1) if algorithm==RLOO_ALGORITHM else 1.
    return [(value-mean)*scale for value in values]


class HiddenRouteReward:
    """Reads only after rollout. Reference-supported closure is not a general oracle
    of chemical feasibility; unsupported alternatives remain explicitly unverified.
    """
    def __init__(self, database, stock, *, split='training', partial_weight=.2, material_contract=None):
        if not 0 <= partial_weight < 1:
            raise ValueError('PARTIAL_REWARD_MUST_BE_BELOW_TERMINAL')
        # Published reward databases are immutable. Avoid Ceph's costly SQLite
        # advisory locking and prevent sidecar-file writes from this reader.
        self.db = sqlite3.connect(Path(database).resolve().as_uri()+'?mode=ro&immutable=1',uri=True)
        self.stock = frozenset(stock)
        self.split, self.partial_weight = split, partial_weight
        from .route_materials import material_contract as checked, EXACT_MATERIAL
        self.material_contract=checked(dict(stock_material_contract=(EXACT_MATERIAL if material_contract is None else material_contract)))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.db.close()

    def score(self, task_id, report, *, include_progress=False):
        from .route_materials import material_contract, validate_material_edge
        if material_contract(report) != self.material_contract:
            raise ValueError('REWARD_STOCK_MATERIAL_CONTRACT_MISMATCH')
        row = self.db.execute('SELECT target,split,reference_steps FROM tasks WHERE id=?',(task_id,)).fetchone()
        if row is None or row[1] != self.split:
            raise ValueError('REWARD_TASK_SPLIT_MISMATCH')
        root, _, steps = row
        if chemical_identity(report['root']) != root:
            raise ValueError('REWARD_ROOT_MISMATCH')
        if report['status'] in ('resource_censored','protected_censored'):
            return dict(reward=None,verified_stock_closed=False,censored=True,
                        supported_reactions=0,unsupported_reactions=0,
                        untrusted_reference_matches=0,censor_reason=report['status'],
                        reward_status=report['status'])
        if report['status'] == 'budget_exhausted':
            raise ValueError('AGGREGATE_EPISODE_BUDGET_FORBIDDEN')
        references = {r[0] for r in self.db.execute('SELECT signature FROM edges WHERE task_id=?',(task_id,))}
        frontier = {'0':root}
        ancestors = {'0':set()}
        trusted_occurrences = {'0'}
        supported, unsupported, invalid = set(), 0, ''
        untrusted_reference_matches = 0
        progress_events = []
        for reaction_index, reaction in enumerate(report['reactions']):
            validate_material_edge(reaction,self.stock,self.material_contract)
            occurrence = reaction['occurrence']
            product = chemical_identity(reaction['product'])
            children = [chemical_identity(v) for v in reaction['precursors']]
            if frontier.get(occurrence) != product or not children:
                invalid = 'DISCONNECTED_OR_REUSED_REACTION'; break
            if product in self.stock or any(c in ancestors[occurrence] | {product} for c in children):
                invalid = 'NONPRODUCTIVE_OR_CYCLIC_REACTION'; break
            key = edge_signature(product,children)
            trusted_edge = occurrence in trusted_occurrences and key in references
            if trusted_edge:
                previous_count = len(supported)
                supported.add(key)
                if include_progress and len(supported) > previous_count:
                    progress_events.append(dict(reaction_index=reaction_index,
                                                supported_reactions=len(supported)))
            elif key not in references:
                unsupported += 1
            else:
                untrusted_reference_matches += 1
            del frontier[occurrence]
            for i, child in enumerate(children):
                child_occurrence = occurrence+'.'+str(i)
                frontier[child_occurrence] = child
                ancestors[child_occurrence] = ancestors[occurrence] | {product}
                if trusted_edge:
                    trusted_occurrences.add(child_occurrence)
        expected_leaves = Counter(frontier.values())
        claimed_leaves = Counter(chemical_identity(v) for v in report.get('stock_leaves',[]))
        stock_closed = (not invalid and bool(report['reactions']) and bool(frontier)
                        and all(v in self.stock for v in frontier.values())
                        and expected_leaves == claimed_leaves)
        verified = stock_closed and report['status']=='stock_solved' and not unsupported
        partial = self.partial_weight*min(len(supported)/max(steps,1),1.)
        reward = 1. if verified else 0. if invalid else partial
        reward_status = ('invalid_graph' if invalid else 'verified_closed' if verified
                         else 'unsupported_closed' if stock_closed and unsupported else 'observed_partial')
        namespace = {'training':'TRAIN','validation':'VALID'}.get(self.split,self.split.upper())
        result = dict(reward=reward, verified_stock_closed=bool(verified),censored=False,
                    supported_reactions=len(supported),unsupported_reactions=unsupported,
                    untrusted_reference_matches=untrusted_reference_matches,
                    censor_reason='',reward_status=reward_status,
                    graph_error=invalid,structural_stock_closed=bool(stock_closed),
                    chemical_validation='hidden_'+namespace+'_reference_support_only')
        if include_progress:
            result.update(progress_events=progress_events, reference_steps=steps)
        return result


class TargetOnlyRouteEpisode(RouteEpisode):
    """Native observations and stopping; no hidden reference object is accepted.

    Only the per-call physical output/context capacity is finite. A truncated
    action remains resource-censored in RouteEpisode, never a chemical failure.
    """
    def __init__(self, contract, public_root, stock, uid, seed, *, chemistry=None):
        cfg = dict(contract, episode_token_budget=float('inf'))
        super().__init__(cfg,public_root,stock,uid,seed,chemistry=chemistry)

    def next_decision(self):
        request = super().next_decision()
        if self.status=='resource_censored' and self.error=='PROTECTED_EVAL_IDENTITY':
            self.status='protected_censored'
        if request is not None:
            request.remaining_tokens = int(self.cfg.get('max_action_tokens',2048))
        return request


def optimize_routes(actor, reference, optimizer, groups, cfg):
    """One fresh fixed-scale clipped update; repeated records share a forward.

    Repetition multiplicity remains its original Monte Carlo weight. Opposite
    advantages are NOT merged: clipping is not linear across advantage signs.
    """
    import torch
    import torch.distributed as dist
    from .route_policy_objective import action_loss, policy_sequences
    from scripts.pr96_earho_contract import token_logps_batch, synchronize_gradients
    from .route_training_memory import checkpointed_policy_forward
    from .route_precision import precision_mode
    from .route_phase_profile import phase
    precision_kwargs=({'precision':precision_mode(cfg)} if 'policy_precision' in cfg else {})
    distributed = dist.is_available() and dist.is_initialized()
    def collective_check(error):
        if distributed:
            errors=[None]*dist.get_world_size()
            # Preserve the originating frames on every rank. Converting only
            # the message loses the failing kernel/forward/backward location.
            detail='' if error is None else ''.join(traceback.format_exception(error))
            with phase(cfg,'sync.collective'):
                dist.all_gather_object(errors,detail)
            if any(errors):
                failure=RuntimeError('COLLECTIVE_ROUTE_OPTIMIZATION_FAILED:'+json.dumps(errors))
                if error is not None:
                    raise failure from error
                raise failure
        elif error is not None:
            raise error
    usable,compact=[],OrderedDict()
    sampled_records=policy_tokens=0
    algorithm=cfg.get('algorithm',DR_GRPO_ALGORITHM)
    process = algorithm==PROCESS_GRPO_ALGORITHM
    error=None
    try:
        for group in groups:
            if 'candidates' in cfg and len(group)!=cfg['candidates']:
                raise ValueError('ROUTE_GROUP_SIZE_MISMATCH')
            if process:
                from .route_process_reward import process_group_advantages
                advantages=process_group_advantages([
                    None if r['reward'] is None else r['reward_evidence']['process_credit']['event_rewards']
                    for r in group])
            else:
                advantages=route_group_advantages([r['reward'] for r in group],algorithm)
            if advantages is not None:
                usable.append((group,advantages))
        for group,advantages in usable:
            for report,advantage in zip(group,advantages):
                if process:
                    from .route_process_reward import process_sequences
                    sequences=process_sequences(report,advantage)
                else:
                    sequences=((p,r,b,advantage) for p,r,b in policy_sequences(report))
                for p,r,b,action_advantage in sequences:
                    if any(not math.isfinite(v) or v>0 for v in b):
                        raise ValueError('INVALID_BEHAVIOR_TOKEN_LOGPROB')
                    key=(tuple(p),tuple(r),tuple(b),action_advantage,len(group))
                    compact[key]=compact.get(key,0)+1
                    sampled_records+=1;policy_tokens+=len(r)
    except Exception as exc:
        error=exc
    collective_check(error)
    device = next(actor.parameters()).device
    count = torch.tensor(float(len(usable)),dtype=torch.float64,device=device)
    if distributed:
        with phase(cfg,'sync.collective'):
            dist.all_reduce(count)
    denominator = count.item()
    stats = dict(algorithm=algorithm,
                 advantage_estimator=('centered_action_return_to_go_no_std' if process else
                     'leave_one_out' if algorithm==RLOO_ALGORITHM else 'group_mean_centered'),
                 loss_token_normalizer=cfg['loss_token_normalizer'],
                 policy_groups=len(usable),censored_groups=len(groups)-len(usable),
                 effective_groups=sum((any(a != 0 for row in advantages for a in row) if process else
                                       any(a != 0 for a in advantages)) for _,advantages in usable),
                 sampled_action_records=sampled_records,optimized_unique_records=len(compact),policy_tokens=policy_tokens,
                 global_policy_groups=denominator,optimizer_updated=False,
                 max_behavior_logprob_drift=0.,behavior_logprob_drift_warnings=0,
                 pg_loss=0.,reference_kl=0.)
    if process:
        stats['effective_episode_groups']=sum(any(a!=0 for a in
            route_group_advantages([r['reward'] for r in group],algorithm)) for group,_ in usable)
    optimizer.zero_grad(set_to_none=True)
    if denominator == 0:
        return stats
    actor.eval(); reference.eval()
    entries = list(compact.items())
    size = cfg['loss_batch_size']
    from .route_training_batches import training_batches
    padded_capacity=cfg.get('training_max_padded_tokens',
        (2**31-1)//cfg.get('generation_qkv_channels',6144))
    error=None
    try:
        with checkpointed_policy_forward(actor,cfg):
            for batch in training_batches(entries,size,padded_capacity):
                pairs = [(list(k[0]),list(k[1])) for k,_ in batch]
                with torch.no_grad(),phase(cfg,'learner.reference_forward',batch_size=len(pairs)):
                    refs = token_logps_batch(reference,pairs,cfg['pad_token_id'],**precision_kwargs)
                with phase(cfg,'learner.actor_forward',batch_size=len(pairs)):
                    news = token_logps_batch(actor,pairs,cfg['pad_token_id'],**precision_kwargs)
                loss = None
                for new,ref,(key,multiplicity) in zip(news,refs,batch):
                    _,_,old,advantage,candidates = key
                    behavior = torch.tensor(old,device=new.device,dtype=new.dtype)
                    drift = (new.detach()-behavior).abs().max().item()
                    stats['max_behavior_logprob_drift'] = max(stats['max_behavior_logprob_drift'],drift)
                    if drift > cfg['behavior_logprob_tolerance']:
                        # Cached sampling and full-forward BF16 arithmetic need not
                        # match tokenwise. Keep the real sampling denominator below:
                        # action_loss already uses current/sampler importance ratios.
                        # This threshold diagnoses mismatch; it is not a validity
                        # condition for a finite, provenance-bound sampling receipt.
                        stats['behavior_logprob_drift_warnings'] += multiplicity
                    values = action_loss(new,behavior,ref,advantage,clip=cfg['clip_epsilon'],beta=cfg['kl_beta'],
                                         normalizer=cfg['loss_token_normalizer']*denominator*candidates)
                    part = values['total']*multiplicity
                    loss = part if loss is None else loss+part
                    stats['pg_loss'] += values['pg'].detach().item()*multiplicity
                    stats['reference_kl'] += values['kl'].detach().item()*multiplicity
                if not torch.isfinite(loss):
                    raise ValueError('NONFINITE_ROUTE_POLICY_LOSS')
                with phase(cfg,'learner.backward',batch_size=len(pairs)):
                    loss.backward()
    except Exception as exc:
        error=exc
    collective_check(error)
    if distributed:
        with phase(cfg,'learner.gradient_reduce'):
            synchronize_gradients(actor,1.)
    parameters = [p for p in actor.parameters() if p.requires_grad]
    norm = torch.nn.utils.clip_grad_norm_(parameters,cfg['max_grad_norm'])
    if not torch.isfinite(norm):
        raise ValueError('NONFINITE_ROUTE_GRADIENT')
    # No sequence on any rank means no Adam moment/weight update is warranted.
    token_count = torch.tensor(stats['policy_tokens'],device=device,dtype=torch.int64)
    if distributed:
        with phase(cfg,'sync.collective'):
            dist.all_reduce(token_count)
    if token_count.item():
        with phase(cfg,'learner.optimizer_step'):
            optimizer.step()
        stats['optimizer_updated'] = True
    stats['grad_norm'] = norm.item()
    return stats
