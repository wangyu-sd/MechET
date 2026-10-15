"""Compatible single-reaction checkpoint recovery without changing planner SHAs."""
from collections import deque
import fcntl
import json
from pathlib import Path

from scripts.single_reaction_mcts import SingleReactionMCTS
from scripts.shared_route_mcts import RolloutLease
from mechet.shared_electron_search import read_record


def recover_shard(rows, source_root, output_root, manifest, *, planning, cache):
    """Read immutable old evidence and publish a new continuation journal."""
    from mechet.shared_electron_search import SearchJournal, fingerprint
    from scripts.infer_dynamic_electron_trees import dump
    source_root, output_root = Path(source_root), Path(output_root)
    old = json.loads((source_root / 'manifest.json').read_text())
    keys = ('source_sha', 'adapter_sha', 'revision', 'rank', 'world', 'program_goal',
            'seed', 'max_model_len', 'history_contract', 'shared_cache_namespace',
            'evaluation_unit', 'source_protocol', 'stopping_contract')
    if any(old.get(k) != manifest.get(k) for k in keys):
        raise ValueError('resume manifest contract mismatch')
    controllers, journals, reports = {}, {}, []
    stats = {'reused_finished_reports': 0, 'restored_controllers': 0,
             'restored_pending_rollouts': 0, 'restored_completed_rollouts': 0}
    try:
        for row in rows:
            identifier = row['id']; filename = fingerprint(identifier) + '.json'
            report_path = source_root / 'evaluations' / filename
            if report_path.exists():
                report = json.loads(report_path.read_text())
                if report['id'] != identifier:
                    raise ValueError('resume report identity contract mismatch')
                if any(word in report.get('stop_reason', '') for word in ('MEMORY', 'RESOURCE', 'CONTEXT_BUDGET')):
                    raise ValueError('resource-limited report requires explicit recovery policy')
                if report.get('tree_artifact'):
                    report['tree_artifact'] = str((source_root / report['tree_artifact']).resolve())
                report['resumed_from'] = str(report_path)
                dump(output_root / 'evaluations' / filename, report)
                reports.append(report); stats['reused_finished_reports'] += 1
                continue
            j = SearchJournal(source_root / 'journals', identifier, old)
            c = restore_single_controller(j, checkpoint_path=source_root / 'mcts' / filename,
                mcts_store=planning, shared_store=cache, output_checkpoint=output_root / 'mcts' / filename)
            if c.case_id != identifier:
                c.close(); raise ValueError('resume controller identity contract mismatch')
            controllers[identifier] = c
            fresh = SearchJournal(output_root / 'journals', identifier, manifest)
            fresh.initialize(c); journals[identifier] = fresh
            stats['restored_controllers'] += 1
            stats['restored_pending_rollouts'] += len(c.rollouts)
            stats['restored_completed_rollouts'] += c.completed_rollouts
        return controllers, journals, reports, stats
    except BaseException:
        for c in controllers.values():
            c.close()
        raise


def backup_record(store, uid):
    if hasattr(store, 'backup_record'):
        return store.backup_record(uid)
    path = store.root / 'backups' / (uid + '.json')
    return read_record(path) if path.exists() else None


def resume_rollout(store, uid):
    if hasattr(store, 'resume_rollout'):
        return store.resume_rollout(uid)
    path = store.root / 'rollouts' / (uid + '.lock')
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open('a+')
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise ValueError('saved rollout still has a live owner') from None
    return RolloutLease(uid, handle)


def restore_single_controller(journal, *, checkpoint_path, mcts_store,
                              shared_store=None, output_checkpoint=None):
    saved = read_record(checkpoint_path)
    if saved.get('planning_contract') != mcts_store.contract:
        raise ValueError('MCTS checkpoint contract mismatch')
    if saved['program_goal'] != mcts_store.contract.get('program_goal', saved['program_goal']):
        raise ValueError('MCTS program goal contract mismatch')
    if any(mcts_store.alive(r['uid']) for r in saved['pending_rollouts']):
        raise ValueError('MCTS checkpoint still has a live rollout owner')
    base = journal.restore(stock=set(), shared_store=shared_store)
    c = SingleReactionMCTS(base.case_id, base.target, mcts_store=mcts_store,
                          shared_store=shared_store, checkpoint_path=output_checkpoint,
                          program_goal=saved['program_goal'])
    if saved['case_id'] != base.case_id or saved['root'] != c.mcts_root:
        raise ValueError('MCTS checkpoint identity contract mismatch')
    c.__dict__.update(base.__dict__)
    for name in ('completed_rollouts', 'positive_backups', 'censored_rollouts'):
        setattr(c, name, saved['metrics'][name])
    c.seen_states = set(saved['seen_states'])
    c.blocked = set(saved['blocked'])
    c.completed_programs = set(saved['completed_programs'])
    c.stop_reason = saved.get('stop_reason', '')
    # Checkpoint publication can precede the native expansion journal. Native
    # graph payloads and recorded proposals let us fill that gap without a LLM.
    states = {sid: mcts_store.snapshot(sid) for sid in c.seen_states}
    for value in states.values():
        c._hydrate(value['state'])
    for sid, value in states.items():
        for edge in value['edges'].values():
            child = edge.get('child')
            if not child:
                continue
            child_state = mcts_store.snapshot(child)['state']
            key = child_state.get('terminal_context') if child_state['solved'] else child_state['active']
            if key not in c.contexts:
                c._ensure_transition(sid, edge)
    if not c.completed_programs.issubset(c.contexts):
        raise ValueError('saved completed program lacks recoverable native context')
    try:
        for pending in saved['pending_rollouts']:
            uid = pending['uid']
            committed = backup_record(mcts_store, uid)
            if committed is not None:
                # A durable backup can be newer than the controller checkpoint.
                # Idempotently complete it, then advance the local counter once.
                mcts_store.backup(uid, committed['path'], committed['reward'])
                c.completed_rollouts += 1
                if committed['reward']:
                    sid, eid = committed['path'][-1]
                    child = mcts_store.snapshot(sid)['edges'][eid]['child']
                    state = mcts_store.snapshot(child)['state']
                    c.positive_backups += 1
                    c.completed_programs.add(state['terminal_context'])
                    c.blocked.add(child)
                continue
            c.rollouts.append({'lease': resume_rollout(mcts_store, uid),
                               'sid': pending['sid'],
                               'path': [tuple(pair) for pair in pending['path']]})
        count = c.completed_rollouts + c.censored_rollouts
        if count > c.program_goal or count + len(c.rollouts) > c.program_goal:
            raise ValueError('saved trajectory-attempt accounting exceeds contract')
        c.has_pending = count < c.program_goal
        if not c.has_pending:
            c.stop_reason = 'MCTS_TRAJECTORY_ATTEMPT_GOAL_REACHED'
        c.checkpoint()
        return c
    except BaseException:
        c.close()
        raise
