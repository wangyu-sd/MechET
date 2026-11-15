"""Replay completed TRAIN episodes before constructing memory or supervision."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .external_route_rl import TargetOnlyRouteEpisode
from .qwen3_continuation import shared_path
from .route_experience_memory import ExperienceMemory, scope_key
from .route_memory_integration import (MemoryTargetOnlyRouteEpisode, execution_scope,
                                       memory_digest, strategy_for_error)
from .route_training_contract import atomic_json, chemical_identity


def supervised_decision(record, tools, *, source_sha, memory_sha):
    from .agent_inference import parse_tool_calls
    if record.get('accepted') is not True:
        raise ValueError('POSITIVE_SUPERVISION_REQUIRES_ACCEPTED_ACTION')
    calls = parse_tool_calls(record['raw'])
    if len(calls) != 1:
        raise ValueError('REPLAY_REQUIRES_SINGLE_TOOL_ACTION')
    call = calls[0]
    call_id = memory_digest([source_sha, record['request_id']])
    return dict(id=call_id, tools=deepcopy(tools),
        messages=[*deepcopy(record['messages']),
            dict(role='assistant', content='', tool_calls=[dict(id=call_id, type='function',
                 function=dict(name=call.name, arguments=call.arguments))]),
            dict(role='tool', name=call.name, tool_call_id=call_id, content=json.dumps(record['feedback']))],
        metadata=dict(decision_contract='agentic_route_memory_decision_v1',
            executor_replayed=True, source_snapshot_sha256=source_sha,
            memory_snapshot_sha256=memory_sha, history_contains_failed_actions=True,
            history_model_visible_gold_horizon=False))


def curate_episode(snapshot, *, task_id, public_tasks, protected, output_dir,
                   split='training', oracle=None, binding=None, stock=None):
    if split != 'training':
        raise ValueError('MEMORY_CURATION_REQUIRES_TRAIN')
    if task_id not in public_tasks or chemical_identity(public_tasks[task_id]) != snapshot['root']:
        raise ValueError('MEMORY_PUBLIC_TRAIN_TASK_MISMATCH')
    if snapshot.get('status') is None:
        raise ValueError('MEMORY_COMPLETE_EPISODE_REQUIRED')
    if snapshot['status'] in ('resource_censored', 'protected_censored', 'budget_exhausted'):
        raise ValueError('MEMORY_CENSORED_EPISODE')
    identities = {snapshot['root']}
    for edge in snapshot['reactions']:
        identities.add(edge['product']); identities.update(edge['precursors'])
    if identities.intersection(protected):
        raise ValueError('MEMORY_PROTECTED_IDENTITY')
    directory = shared_path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    cls = MemoryTargetOnlyRouteEpisode if 'route_memory' in snapshot['cfg'] else TargetOnlyRouteEpisode
    ep = cls(snapshot['cfg'], snapshot['public_root'],
             set(snapshot['stock']) if stock is None else stock, snapshot['uid'], snapshot['seed'])
    binding = binding or snapshot['cfg'].get('route_memory')
    if not binding:
        raise ValueError('MEMORY_CURATION_VERSION_BINDING_REQUIRED')
    recoveries, record_tools = [], {}
    rejected = None
    for record in snapshot['records']:
        request = ep.next_decision()
        if request is None:
            raise ValueError('MEMORY_EXTRA_POST_TERMINAL_RECORD')
        from .route_action_feedback import tools_for_request
        record_tools[record['request_id']] = tools_for_request(request)
        scope = execution_scope(ep, binding=binding)
        ep.accept(deepcopy(record))
        actual = ep.records[-1]
        if (actual.get('accepted') and actual['feedback']['state_changed'] and rejected is not None
                and rejected['occurrence'] == actual['occurrence']
                and rejected['state_before'] == actual['state_before']):
            strategy = strategy_for_error(rejected['error'])
            if strategy:
                recoveries.append((scope, strategy, actual['request_id']))
        rejected = actual if actual.get('accepted') is False else None
    ep.next_decision()
    if ep.snapshot(include_stock='stock' in snapshot) != snapshot:
        raise ValueError('MEMORY_EXECUTION_REPLAY_MISMATCH')
    report = ep.finish_report()
    committed = {edge['occurrence'] for edge in report['reactions']}
    recoveries = [item for item in recoveries if item[0]['occurrence'] in committed]
    verified = bool(oracle.score(task_id, report)['verified_stock_closed']) if oracle is not None else False
    # Legacy target-only snapshots encode an unlimited episode budget as Infinity.
    # Bind their exact canonical Python JSON representation; memory content remains strict JSON.
    source_sha = hashlib.sha256(json.dumps(snapshot, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    memory = ExperienceMemory()
    for scope, strategy, request_id in recoveries:
        content = dict(strategy_id=strategy)
        uid = memory_digest(dict(source=source_sha, request=request_id, scope=scope_key(scope), content=content))
        proof = dict(receipt_id=uid, split='training', episode_complete=True, resource_censored=False,
                     success=True, signal='executor_recovery', verification='executor_replayed', memory_id=uid,
                     scope_key=scope_key(scope), content_sha256=memory_digest(content),
                     source_snapshot_sha256=source_sha, request_id=request_id,
                     task_id=task_id, route_reference_verified=verified)
        path = directory/(uid+'.json')
        if path.exists() and json.loads(path.read_text()) != proof:
            raise ValueError('MEMORY_EXISTING_RECEIPT_CONFLICT')
        atomic_json(path, proof)
        evidence = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        memory.add(uid, scope, content, evidence=[evidence])
        memory.feedback(uid, success=True, evidence=evidence)
    supervision = []
    if verified:
        # Every input is the original pre-action observation, including failures
        # as history. Rejected assistant outputs never become positive CE labels.
        for record in report['records']:
            if record.get('accepted') is True:
                supervision.append(supervised_decision(record, record_tools[record['request_id']],
                    source_sha=source_sha, memory_sha=(snapshot['cfg'].get('route_memory') or {}).get('sha256','')))
    return dict(schema='verified_route_recovery_curation_v1', task_id=task_id,
                source_snapshot_sha256=source_sha, reference_verified=verified,
                executor_recovery_count=len(recoveries), memory=memory.snapshot(), supervision=supervision)
