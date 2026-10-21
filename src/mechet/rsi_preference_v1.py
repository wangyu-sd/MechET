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


def paired_candidates(episodes, limit):
    """Select complementary terminal mixtures per product; no global score sort.

    Exact common-state potential is preferred. A singleton is never checked at
    the expense of an available pair. Unknown F0 outcomes are not negatives.
    """
    groups = defaultdict(list)
    for e in episodes:
        if e.get('terminal') and e.get('flow_count'):
            groups[e['id']].append(e)
    selected = []
    for source, group in sorted(groups.items(), key=lambda x: fingerprint(x[0])):
        unique = {}
        for e in sorted(group, key=lambda e: e['episode_id']):
            state = e.get('final_state') or e['steps'][-1].get('successor')
            if state:
                unique.setdefault(chemical_key(state), e)
        values = list(unique.values())
        if len(values) < 2 or len(selected)+2 > limit:
            continue
        options = [(a, b) for i, a in enumerate(values) for b in values[i+1:]]
        options.sort(key=lambda pair: (not any(a['prompt_ids'] == b['prompt_ids'] and
            chemical_key(a.get('successor')) != chemical_key(b.get('successor'))
            for a, b in zip(pair[0]['steps'], pair[1]['steps'])), fingerprint([e['episode_id'] for e in pair])))
        selected.extend(options[0])
    return selected
