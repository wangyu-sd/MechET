#!/usr/bin/env python3
"""S0: re-label immutable real PR91 records without GPU generation."""
import argparse
from collections import Counter
import hashlib
import json
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from mechet.rsi_contract import sha256, write_json
from mechet.rsi_productivity import net_reaction, productivity
from mechet.rsi_preference_v1 import strong_pairs


def reference_classes(path, identifiers, expected_sha):
    identifiers = set(identifiers); result = {}; digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for line in f:
            digest.update(line)
            # ID is a top-level prefix field, verified again after decoding.
            match = re.search(rb'"id"\s*:\s*"([^"]+)"', line[:500])
            if not match or match[1].decode() not in identifiers:
                continue
            row = json.loads(line)
            if row['id'] not in identifiers:
                continue
            split = row['metadata']['source_split']
            split = 'valid' if split == 'val' else split
            if split not in {'train', 'valid'}:
                raise ValueError('TEST forbidden in RSI source gate')
            result[row['id']] = {**net_reaction(row['target_smiles'], row['structural_precursor']),
                'reference_id': row['id'], 'source_path': str(path), 'source_sha256': expected_sha,
                'mapped_target': row['target_smiles'],
                'source_split': split}
    if digest.hexdigest() != expected_sha or set(result) != identifiers:
        raise ValueError('reference source SHA / exact source-ID crosswalk failed')
    return result


def label_episodes(episodes, references):
    for e in episodes:
        e['productivity'] = productivity(e, references[e['id']])
    return episodes


def audit(source, proof, proof_sha, output):
    source, output = Path(source), Path(output)
    train = json.loads((source/'selected_train_ids.json').read_text())
    refs = reference_classes(proof, [r['id'] for r in train], proof_sha)
    report = {'references': refs, 'source_artifacts': {}, 'rounds': {}, 'test_used': False}
    for phase in ('D0', 'D1'):
        episodes = []
        for path in sorted(source.glob(phase+'_K[248].feedback.json')):
            report['source_artifacts'][str(path)] = sha256(path)
            episodes += json.loads(path.read_text())['episodes']
        if not episodes:
            report['rounds'][phase] = {'status': 'not_available'}; continue
        label_episodes(episodes, refs)
        pairs = strong_pairs(episodes)
        old_path = source/(phase.lower()+'_preferences.json')
        old = json.loads(old_path.read_text()) if old_path.exists() else []
        by_id = {e['episode_id']: e for e in episodes}
        old_audit = []
        for p in old:
            chosen = by_id[p['chosen_episode']]
            old_audit.append({'id': p['id'], 'source_id': chosen['id'], 'old_evidence': p['evidence'],
                'chosen_productivity': chosen['productivity'],
                'eligible_identical_pair': any(q['chosen_episode'] == p['chosen_episode'] and
                    q['rejected_episode'] == p['rejected_episode'] for q in pairs)})
        summary = {'episodes': len(episodes), 'independent_reactions': len({e['id'] for e in episodes}),
            'old_pairs': len(old), 'corrected_strong_pairs': len(pairs),
            'corrected_pair_sources': len({p['source_id'] for p in pairs}),
            'distinct_chosen_episodes': len({p['chosen_episode'] for p in pairs}),
            'tool_pairs': dict(Counter(p['tool'] for p in pairs)),
            'strong_evidence': dict(Counter(p['evidence'] for p in pairs)),
            'forward_status': dict(Counter(e.get('outcome', {}).get('status', 'unknown') for e in episodes)),
            'candidate_class': dict(Counter(e['productivity']['candidate']['kind'] for e in episodes)),
            'productive_forward_episodes': sum(e.get('outcome', {}).get('success', False) and
                e['productivity']['task_compatible_net_progress'] is True for e in episodes),
            'forward_success_task_incompatible': sum(e.get('outcome', {}).get('success', False) and
                e['productivity']['task_compatible_net_progress'] is False for e in episodes),
            'old_pair_audit': old_audit}
        report['rounds'][phase] = summary
        write_json(output/(phase+'.corrected_pairs.json'), pairs)
        # Full labeled records are separate from the small committed audit.
        write_json(output/(phase+'.labeled.json'), {'episodes': episodes})
        print({'phase': 'S0', 'round': phase, **{k:v for k,v in summary.items() if k != 'old_pair_audit'}}, flush=True)
    report['historical_validation'] = {}
    for name in ('R0', 'R1', 'R2', 'R2_static_D0'):
        path = source/'evaluation'/(name+'.json')
        if path.exists():
            report['historical_validation'][name] = json.loads(path.read_text())['summary']
            report['source_artifacts'][str(path)] = sha256(path)
    write_json(output/'audit.json', report)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True); p.add_argument('--proof', type=Path, required=True)
    p.add_argument('--proof-sha', required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); audit(a.source, a.proof, a.proof_sha, a.output)
