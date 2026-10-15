"""Recovery software tests use real executor transitions, never model sampling."""
from collections import deque
import pytest


def fixture_controller(tmp_path, goal=1):
    from scripts.single_reaction_mcts import SingleReactionMCTS, SingleReactionMCTSStore
    from mechet.shared_electron_search import SearchJournal
    store = SingleReactionMCTSStore(tmp_path / 'store', {'program_goal': goal})
    checkpoint = tmp_path / 'checkpoint.json'
    c = SingleReactionMCTS('case', 'CO', mcts_store=store,
                           checkpoint_path=checkpoint, program_goal=goal)
    journal = SearchJournal(tmp_path / 'journals', 'case', {'fixture': True})
    journal.initialize(c)
    return c, store, journal, checkpoint


def import_chloride(c, journal=None):
    from mechet.dynamic_electron_tree import Proposal
    ctx = c.next_context()
    edges = c.incorporate(ctx.key, [Proposal('import_fragments', {'fragments': [
        {'smiles': '[Cl-]', 'count': 1, 'purpose': 'electron_participant'}]}, 'import', -1., 4)])
    if journal is not None:
        journal.record(c, ctx.key, edges)


def test_pending_path_keeps_uid_and_is_not_charged_as_censored(tmp_path):
    from scripts.mcts_resume import restore_single_controller
    from mechet.shared_electron_search import read_record
    c, store, journal, checkpoint = fixture_controller(tmp_path)
    import_chloride(c, journal)
    ctx = c.next_context()
    saved = read_record(checkpoint)
    uid = saved['pending_rollouts'][0]['uid']
    state = ctx.node.state
    c.close()
    restored = restore_single_controller(journal, checkpoint_path=checkpoint, mcts_store=store)
    try:
        assert restored.completed_rollouts == restored.censored_rollouts == 0
        assert restored.rollouts[0]['lease'].uid == uid
        assert restored.rollouts[0]['path'] == [tuple(x) for x in saved['pending_rollouts'][0]['path']]
        assert restored.next_context().node.state == state
    finally:
        restored.close()


def test_committed_backup_with_stale_checkpoint_counts_exactly_once(tmp_path):
    from scripts.mcts_resume import restore_single_controller
    from mechet.shared_electron_search import read_record, durable_json
    from mechet.dynamic_electron_tree import Proposal
    c, store, journal, checkpoint = fixture_controller(tmp_path)
    ctx = c.next_context()
    edges = c.incorporate(ctx.key, [Proposal('finish_trace', {}, 'finish', -1., 3)])
    journal.record(c, ctx.key, edges)
    stale = read_record(checkpoint)
    while c.has_pending:
        assert c.next_context() is None
    assert store.snapshot(c.mcts_root)['visits'] == 1
    c.close()
    durable_json(checkpoint, stale)
    restored = restore_single_controller(journal, checkpoint_path=checkpoint, mcts_store=store)
    try:
        assert restored.completed_rollouts == 1
        assert restored.censored_rollouts == 0
        assert not restored.has_pending and not restored.rollouts
        assert store.snapshot(restored.mcts_root)['visits'] == 1
    finally:
        restored.close()


def test_complete_checkpoint_needs_no_more_sampling(tmp_path):
    from scripts.mcts_resume import restore_single_controller
    from mechet.dynamic_electron_tree import Proposal
    c, store, journal, checkpoint = fixture_controller(tmp_path)
    ctx = c.next_context()
    edges = c.incorporate(ctx.key, [Proposal('finish_trace', {}, 'finish', -1., 3)])
    journal.record(c, ctx.key, edges)
    while c.has_pending:
        c.next_context()
    c.close()
    restored = restore_single_controller(journal, checkpoint_path=checkpoint, mcts_store=store)
    try:
        assert not restored.has_pending
        assert restored.next_context() is None
        assert restored.completed_rollouts == 1
    finally:
        restored.close()


def test_graph_native_payload_recovers_context_ahead_of_journal(tmp_path):
    from scripts.mcts_resume import restore_single_controller
    c, store, journal, checkpoint = fixture_controller(tmp_path)
    import_chloride(c)  # crash between planner checkpoint and native journal commit
    c.close()
    restored = restore_single_controller(journal, checkpoint_path=checkpoint, mcts_store=store)
    try:
        ctx = restored.next_context()
        assert ctx.node.actions[-1]['name'] == 'import_fragments'
        assert 'Cl' in ctx.node.state
        assert restored.censored_rollouts == 0
    finally:
        restored.close()


def test_resume_rejects_changed_goal_or_live_owner(tmp_path):
    from scripts.mcts_resume import restore_single_controller
    from mechet.shared_electron_search import read_record, durable_json
    c, store, journal, checkpoint = fixture_controller(tmp_path)
    c.next_context()
    with pytest.raises(ValueError, match='live'):
        restore_single_controller(journal, checkpoint_path=checkpoint, mcts_store=store)
    c.close()
    value = read_record(checkpoint)
    value['planning_contract']['program_goal'] = 999
    durable_json(checkpoint, value)
    with pytest.raises(ValueError, match='contract'):
        restore_single_controller(journal, checkpoint_path=checkpoint, mcts_store=store)


def test_runner_reuses_finished_report_without_new_controller(tmp_path):
    import json
    from scripts.mcts_resume import recover_shard
    from mechet.shared_electron_search import fingerprint
    old = tmp_path / 'old'; old.mkdir()
    manifest = {'source_sha': 'source', 'adapter_sha': 'actor', 'rank': 0, 'world': 1,
                'program_goal': 1, 'seed': 17, 'max_model_len': 8192,
                'history_contract': 'history', 'shared_cache_namespace': 'cache'}
    (old / 'manifest.json').write_text(json.dumps(manifest))
    (old / 'evaluations').mkdir()
    (old / 'evaluations' / (fingerprint('done') + '.json')).write_text(json.dumps(
        {'id': 'done', 'stop_reason': 'MCTS_TRAJECTORY_ATTEMPT_GOAL_REACHED', 'top1': True}))
    controllers, journals, reports, stats = recover_shard(
        [{'id': 'done', 'product_unmapped': 'CO'}], old, tmp_path / 'new',
        manifest, planning=None, cache=None)
    assert controllers == journals == {}
    assert reports[0]['top1'] is True
    assert stats['reused_finished_reports'] == 1


def test_runner_refuses_actor_mismatch_before_restoring_any_target(tmp_path):
    import json
    from scripts.mcts_resume import recover_shard
    old = tmp_path / 'old'; old.mkdir()
    (old / 'manifest.json').write_text(json.dumps({'adapter_sha': 'wrong'}))
    with pytest.raises(ValueError, match='contract'):
        recover_shard([], old, tmp_path / 'new', {'adapter_sha': 'right'}, planning=None, cache=None)
