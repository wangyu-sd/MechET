from copy import deepcopy
import hashlib
import json

import pytest

from test_route_action_feedback import send, flow
from test_route_experience_memory import evidence


def empty_config():
    from mechet.route_memory_integration import memory_digest
    snapshot = {'schema': 'route_experience_memory_v2', 'entries': []}
    return dict(behavior_version='memory-test', public_observation_contract='route_action_feedback_v8',
                max_action_tokens=2048, max_consecutive_recoveries=8,
                route_memory=dict(snapshot=snapshot, sha256=memory_digest(snapshot),
                                  stock_version='s1', executor_version='e1'))


def episode():
    from mechet.route_memory_integration import MemoryTargetOnlyRouteEpisode
    return MemoryTargetOnlyRouteEpisode(empty_config(), 'CO', set(), 'memory', 17)


def test_live_memory_retains_native_state_and_complete_failed_history(tmp_path):
    from mechet.route_memory_integration import (execution_scope, memory_digest,
                                               MemoryTargetOnlyRouteEpisode)
    from mechet.route_experience_memory import ExperienceMemory
    ep = episode(); first = ep.next_decision()
    send(ep, 'apply_electron_flow', flow())
    scope = execution_scope(ep)
    content = {'strategy_id': 'inspect_current_source_bonds'}
    mem = ExperienceMemory()
    proof = evidence(tmp_path, 'proof', entry='repair', content=content, current=scope)
    mem.add('repair', scope, content, evidence=[proof])
    saved = ep.snapshot(); snap = mem.snapshot()
    saved['cfg']['route_memory'].update(snapshot=snap, sha256=memory_digest(snap))
    # Install this memory before replaying: the earlier request had no eligible memory.
    ep = MemoryTargetOnlyRouteEpisode(saved['cfg'], 'CO', set(), 'memory', 17)
    first2 = ep.next_decision()
    assert 'CURRENT STATE SMILES:' in first2.messages[-1]['content']
    send(ep, 'apply_electron_flow', flow())
    req = ep.next_decision()
    assert 'inspect_current_source_bonds' in req.messages[0]['content']
    assert ep.records[0]['raw'] in json.dumps(ep.event_history.snapshot()) or ep.event_history.data['output_texts'][0] == ep.records[0]['raw']
    assert req.messages[-1]['content'].endswith(first.messages[-1]['content'].split('\nNATIVE CURRENT-DECISION OBSERVATION\n')[-1])
    restored = MemoryTargetOnlyRouteEpisode.restore(json.loads(json.dumps(ep.snapshot())))
    assert restored.next_decision() == req
    forged = deepcopy(ep.snapshot()); forged['records'][0]['messages'][0]['content'] += 'forged'
    with pytest.raises(ValueError):
        MemoryTargetOnlyRouteEpisode.restore(forged)


def test_memory_snapshot_is_frozen_and_tampering_is_rejected():
    from mechet.route_memory_integration import MemoryTargetOnlyRouteEpisode
    cfg = empty_config(); ep = MemoryTargetOnlyRouteEpisode(cfg, 'CO', set(), 'x', 17)
    cfg['route_memory']['snapshot']['entries'].append({'forged': True})
    assert ep.next_decision() is not None
    bad = deepcopy(ep.snapshot()); bad['cfg']['route_memory']['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='MEMORY'):
        MemoryTargetOnlyRouteEpisode.restore(bad)


def test_unstarted_memory_episode_can_restore_without_starting_a_reaction():
    from mechet.route_memory_integration import MemoryTargetOnlyRouteEpisode
    ep = episode(); before = ep.snapshot()
    assert MemoryTargetOnlyRouteEpisode.restore(before).snapshot() == before


def test_frozen_baseline_cannot_change_memory_condition(tmp_path):
    from mechet.external_route_schedule import _compatible
    parent = dict(route_memory={'sha256': 'old'}, output_dir=str(tmp_path/'old'),
                  code_files_sha256={}, code_root=str(tmp_path))
    cfg = dict(route_memory={'sha256': 'new'}, output_dir=str(tmp_path/'new'), code_files_sha256={})
    with pytest.raises(ValueError, match='BASELINE_CONDITION_CHANGED:route_memory'):
        _compatible(cfg, parent, tmp_path)


def test_episode_factory_uses_memory_in_train_and_validation():
    from scripts.train_external_route_rl import make_episodes
    from mechet.route_memory_integration import MemoryTargetOnlyRouteEpisode, MemoryValidationRouteEpisode
    cfg = dict(empty_config(), world_size=8, executor_workers_per_rank=2, candidates=2, seed=17)
    rows = [{'task_id': 'train-1', 'target': 'CO'}]
    eps = make_episodes(rows, set(), set(), cfg, 'v')
    assert all(isinstance(e, MemoryTargetOnlyRouteEpisode) for e in eps)
    cfg['_validation_rng_namespace'] = 'valid'
    eps = make_episodes(rows, set(), set(), cfg, 'v')
    assert all(isinstance(e, MemoryValidationRouteEpisode) for e in eps)
    assert eps[0].next_decision().request_id.startswith('valid/train-1/')
