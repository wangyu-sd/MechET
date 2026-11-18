"""Frozen, completed forward evidence shared by ReFIT process rewards and DPO."""
import math
from collections import defaultdict

UNKNOWN = {None, 'resource_censored', 'protected_censored', 'duplicate_pruned_no_continuation'}

def forward_events(n, edges, reference_steps, status, cap):
    if status in UNKNOWN:
        return None
    if not (0 <= cap <= 1) or reference_steps < 0:
        raise ValueError('invalid teacher reward bound')
    events = [0.] * n
    seen = set()
    total = 0.
    for index, identity, score in edges:
        if not 0 <= index < n:
            raise ValueError('finish index outside native action sequence')
        if score is None or identity in seen:
            continue
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('invalid forward recovery rate')
        seen.add(identity)
        updated = min(cap, total + cap * score / max(1,reference_steps))
        events[index] += updated-total
        total = updated
    if status=='rejected_action' and events:
        events[-1] -= total
    return events

def ranked_pairs(rows, margin):
    if not 0 < margin <= 1:
        raise ValueError('invalid DPO eligibility margin')
    nodes = defaultdict(dict)
    for row in rows:
        group_key = (row['node'],row['version'],row['protocol'],tuple(row['prompt']))
        if not row['tokens'] or row['tokens'][-1]!=row['eos']:
            continue
        # M1: the first real observed continuation owns the action, even when unavailable.
        nodes[group_key].setdefault(row['action'],row)
    result = []
    for choices in nodes.values():
        observed = [r for r in choices.values() if r['score'] is not None]
        if any(not math.isfinite(r['score']) or not 0<=r['score']<=1 for r in observed):
            raise ValueError('invalid pair evidence')
        result.extend((a['receipt'],b['receipt']) for a in observed for b in observed
                      if a['score']-b['score']>=margin and a['tokens']!=b['tokens'])
    return result

def dpo_loss(chosen,rejected,ref_chosen,ref_rejected,beta):
    import torch.nn.functional as F
    return -F.logsigmoid(beta*((chosen-ref_chosen.detach())-(rejected-ref_rejected.detach())))

def apply_scores(report, decisions, cap):
    """decisions holds one authenticated outcome for every eligible FINISH in this report."""
    from .route_process_reward import digest,returns_to_go
    ev=report['reward_evidence']
    credit=ev.get('process_credit')
    if credit is None:return
    if 'forward_joint' in ev:raise ValueError('teacher reward already applied')
    edges=[(x['index'],x['reaction_key'],x['score']) for x in decisions]
    if ev.get('graph_error'):
        edges=[]
    original=report['reward']
    extra=forward_events(len(report['records']),edges,ev['reference_steps'],report['status'],cap)
    if extra is None:return
    bonus=math.fsum(extra)
    root_sha=credit['sha256']
    ev['forward_joint']=dict(version='pr116_post_finish_v1',route_reward=original,
                             forward_reward=bonus,route_credit_sha=root_sha,decisions=decisions)
    if not any(extra):return
    joint=[math.fsum((x,y)) for x,y in zip(credit['event_rewards'],extra,strict=True)]
    credit['event_rewards']=joint;credit['returns']=returns_to_go(joint)
    credit['forward_joint']=dict(route_credit_sha=root_sha,forward_events=extra)
    credit['sha256']=digest({k:v for k,v in credit.items() if k!='sha256'})
    ev['reward']=report['reward']=math.fsum((original,bonus))

def group_preferences(group,margin):
    rows=[]
    for root in group:
        tree=root['refit']
        seen=set();first_actions=set()
        masked={n for n,receipts in tree['node_censored_receipts'].items() if receipts}
        for r in [root,*sorted(tree['suffix_reports'],key=lambda s:s.get('continuation_declaration_order',0))]:
            if r.get('refit_lane')=='exploration' or r['status']=='duplicate_pruned_no_continuation':continue
            ev=r['reward_evidence']
            by_occ={d['occurrence']:d for d in ev.get('forward_joint',{}).get('decisions',[])}
            for e in r.get('refit_path',[]):
                n=e['node_id'];uid=e['request_id']
                if n in masked or n not in tree['conditioning'] or e['name']!='apply_electron_flow' or uid in seen:continue
                seen.add(uid)
                rec=r['records'][e['index']];context=tree['conditioning'][n]
                if rec['request_id']!=uid or rec['prompt_token_ids']!=context['prompt_token_ids'] or rec['behavior_version']!=context['behavior_version']:
                    raise ValueError('joint DPO conditioning mismatch')
                if e['proposal']!='original_policy':continue
                identity=(n,e['action'])
                if identity in first_actions:continue
                first_actions.add(identity)
                item=by_occ.get(rec['occurrence'],{})
                usable=(r['status'] not in UNKNOWN and not ev.get('graph_error') and
                        rec.get('accepted') is True and not rec.get('truncated') and
                        e['proposal']=='original_policy' and item.get('index',-1)>e['index'])
                rows.append(dict(node=n,version=rec['behavior_version'],prompt=tuple(rec['prompt_token_ids']),
                                 tokens=tuple(rec['generated_token_ids']),eos=rec['eos_token_id'],receipt=uid,
                                 action=e['action'],protocol=item.get('protocol'),score=item.get('score') if usable else None))
    return ranked_pairs(rows,margin)

def likelihood_pairs(groups,refit_entries,margin):
    result=[]
    for group in groups:
        entries=refit_entries[id(group)][0]
        ids={r['request_id']:(tuple(r['prompt_token_ids']),tuple(r['generated_token_ids'])) for r,_ in entries}
        for yes,no in group_preferences(group,margin):
            if yes in ids and no in ids:
                result.append((ids[yes],ids[no]))
    return result

def joint_batches(compact,pairs,size,padded):
    """Compute both PO terms on the same logits used by DPO; charge each PO row once."""
    from collections import Counter
    from .route_training_batches import training_batches
    by_action=defaultdict(list)
    for key,multiplicity in compact.items():
        by_action[key[:2]].append((key,multiplicity))
    for (chosen,rejected),count in Counter(pairs).items():
        if chosen[0]!=rejected[0] or chosen==rejected:
            raise ValueError('DPO pair must have identical prompt and distinct outputs')
        actions=[chosen,rejected]
        if 2*(len(chosen[0])+max(len(chosen[1]),len(rejected[1])))>padded:
            raise ValueError('DPO pair exceeds physical padded capacity')
        yield actions,[by_action.pop(a,[]) for a in actions],count
    remaining=[item for items in by_action.values() for item in items]
    for batch in training_batches(remaining,size,padded):
        yield [key[:2] for key,_ in batch],[[item] for item in batch],0
