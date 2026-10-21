"""Conservative common-prompt preferences and within-target F0 allocation."""
from collections import defaultdict
from .rsi_contract import fingerprint
from .roundtrip_reward import components


def chemical_key(value):
    if value is None:
        return None
    return tuple(sorted(components(value).items()))


def strong_pairs(episodes, max_per_reaction=2):
    groups = defaultdict(list)
    for e in episodes:
        groups[e['id']].append(e)
    result = []
    for source, group in sorted(groups.items()):
        pairs = {}; successors = set()
        for chosen in sorted(group, key=lambda e: e['episode_id']):
            if not (chosen.get('terminal') and chosen.get('flow_count') and
                    chosen.get('outcome', {}).get('success') and
                    chosen.get('productivity', {}).get('task_compatible_net_progress') is True):
                continue
            for rejected in sorted(group, key=lambda e: e['episode_id']):
                if chosen['episode_id'] == rejected['episode_id']:
                    continue
                for a, b in zip(chosen['steps'], rejected['steps']):
                    if a['prompt_ids'] != b['prompt_ids']:
                        break
                    if a.get('accepted') == b.get('accepted') and chemical_key(a.get('successor')) == chemical_key(b.get('successor')):
                        continue
                    if not a.get('accepted'):
                        break
                    evidence = None
                    if not b.get('accepted') and b.get('error'):
                        evidence = 'executor_rejected'
                    elif rejected.get('terminal') and rejected.get('productivity', {}).get('task_compatible_net_progress') is False:
                        evidence = 'task_incompatible_terminal'
                    if not evidence or not all(s.get('complete') and s.get('completion_ids') for s in (a, b)):
                        break
                    if a['completion_ids'] == b['completion_ids']:
                        break
                    identity = [source, a['prompt_ids'], chemical_key(a.get('successor')),
                                chemical_key(b.get('successor')), b.get('error'), b['completion_ids'] if not b.get('accepted') else None]
                    key = fingerprint(identity)
                    pairs[key] = {'id': key, 'source_id': source, 'prompt_ids': a['prompt_ids'],
                        'prompt_text': a.get('prompt_text'), 'chosen_ids': a['completion_ids'],
                        'rejected_ids': b['completion_ids'], 'chosen_raw': a['raw'], 'rejected_raw': b['raw'],
                        'chosen_episode': chosen['episode_id'], 'rejected_episode': rejected['episode_id'],
                        'chosen_actor_sha': chosen['actor_sha'], 'rejected_actor_sha': rejected['actor_sha'],
                        'tool': a['name'], 'evidence': evidence}
                    break
        # A source is the independent statistical unit, not its many retries.
        result.extend(sorted(pairs.values(), key=lambda p: p['id'])[:max_per_reaction])
    return result


def paired_candidates(episodes, limit, *, known_keys=(), resolved_sources=()):
    """Select complementary terminal mixtures per product; no global score sort.

    Exact common-state potential is preferred. A singleton is never checked at
    the expense of an available pair. Unknown F0 outcomes are not negatives.
    """
    groups = defaultdict(list); known_keys = set(known_keys); resolved_sources = set(resolved_sources)
    for e in episodes:
        if e['id'] not in resolved_sources:
            groups[e['id']].append(e)
    selected = []; singles = []
    def common_divergence(a, b):
        for x, y in zip(a['steps'], b['steps']):
            if x['prompt_ids'] != y['prompt_ids']:return None
            if chemical_key(x.get('successor')) != chemical_key(y.get('successor')) or x.get('accepted') != y.get('accepted'):
                return x, y
        return None
    def key(e):return e.get('bridge_key',e['episode_id'])
    for source, group in sorted(groups.items(), key=lambda x: fingerprint(x[0])):
        unique = {}
        for e in sorted(group, key=lambda e: e['episode_id']):
            if not e.get('terminal') or not e.get('flow_count') or not e.get('forward_check_eligible',True):continue
            state = e.get('final_state') or e['steps'][-1].get('successor')
            if state:
                unique.setdefault(chemical_key(state), e)
        values = list(unique.values())
        options = [(a, b) for i, a in enumerate(values) for b in values[i+1:]]
        options.sort(key=lambda pair: (common_divergence(*pair) is None,
            -sum(key(e) not in known_keys for e in pair), fingerprint([e['episode_id'] for e in pair])))
        allocated = False
        for pair in options:
            fresh = [e for e in pair if key(e) not in known_keys]
            if fresh and len(selected)+len(fresh) <= limit:
                selected.extend(fresh);known_keys.update(key(e) for e in fresh);allocated=True;break
        if allocated:continue
        for e in values:
            if key(e) in known_keys:continue
            for other in group:
                divergence=common_divergence(e,other)
                if divergence is None:continue
                a,b=divergence
                if a.get('accepted') and b.get('complete') and (
                    (not b.get('accepted') and b.get('error')) or
                    (other.get('terminal') and other.get('productivity',{}).get('task_compatible_net_progress') is False)):
                    singles.append(e);break
    # One terminal + a truly rejected same-prompt action needs only one F0 call.
    for e in singles:
        if len(selected)>=limit:break
        if key(e) not in known_keys:selected.append(e);known_keys.add(key(e))
    return selected


def per_target_budget(episodes, target_ids, budget):
    """Whole-episode token censors, independently within every frozen target."""
    ids = set(target_ids); kept = []; spent = defaultdict(int); censored = set()
    observed = {e['id'] for e in episodes}
    for e in sorted(episodes, key=lambda e: (e['id'], e.get('candidate', 0))):
        if e['id'] not in ids:
            raise ValueError('unexpected validation target')
        cost = sum(len(s['completion_ids']) for s in e['steps'])
        cap = budget[e['id']] if isinstance(budget, dict) else budget
        if spent[e['id']]+cost > cap:
            censored.add(e['id']); continue
        kept.append(e); spent[e['id']] += cost
    return {'targets': len(ids), 'episodes': kept, 'spent_tokens': sum(spent.values()),
        'per_target_spent': dict(spent), 'censored_target_ids': sorted(censored),
        'missing_target_ids': sorted(ids-observed), 'budget': budget}
