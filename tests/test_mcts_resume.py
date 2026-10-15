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


def test_committed_native_transition_is_not_counted_again_after_resume(tmp_path):
    from scripts.mcts_resume import restore_single_controller
    c, store, journal, checkpoint = fixture_controller(tmp_path)
    import_chloride(c, journal)
    c.close()
    restored = restore_single_controller(journal, checkpoint_path=checkpoint, mcts_store=store)
    try:
        restored.next_context()
        assert restored.work['sampled_tokens'] == 4
        assert restored.work['raw_candidates'] == 1
        assert len(restored.trees[restored.target].edges) == 1
    finally: restored.close()


def test_new_rollout_admission_is_checkpointed_before_cached_backup(tmp_path, monkeypatch):
    from scripts.mcts_resume import ResumableSingleReactionMCTS, restore_single_controller
    from mechet.shared_electron_search import SearchJournal
    from mechet.dynamic_electron_tree import Proposal
    c, store, _, _ = fixture_controller(tmp_path)
    ctx = c.next_context(); c.incorporate(ctx.key, [Proposal('finish_trace', {}, 'finish', -1., 3)])
    c.close()
    checkpoint = tmp_path / 'new.json'
    new = ResumableSingleReactionMCTS('new', 'CO', mcts_store=store,
        checkpoint_path=checkpoint, program_goal=1)
    journal = SearchJournal(tmp_path / 'journals2', 'new', {}); journal.initialize(new)
    new.checkpoint()
    original = new.checkpoint
    def crash_after_backup():
        if new.completed_rollouts: raise RuntimeError('simulated crash after backup')
        original()
    monkeypatch.setattr(new, 'checkpoint', crash_after_backup)
    with pytest.raises(RuntimeError, match='simulated crash'):
        new.next_context()
    new.close()
    restored = restore_single_controller(journal, checkpoint_path=checkpoint, mcts_store=store)
    try:
        assert restored.completed_rollouts == 1
        assert restored.next_context() is None
    finally: restored.close()


def test_unjournaled_batch_recovers_rejected_proposals_and_full_multiplicity(tmp_path):
    from dataclasses import asdict
    from scripts.mcts_resume import restore_single_controller
    from mechet.dynamic_electron_tree import Proposal
    c, store, journal, checkpoint = fixture_controller(tmp_path)
    ctx = c.next_context(); selected = dict(c.selected)
    proposals = [Proposal('import_fragments', {'fragments': [
        {'smiles': '[Cl-]', 'count': 1, 'purpose': 'electron_participant'}]}, 'import', -1., 4),
        Proposal('unknown', {}, 'invalid', -2., 3, 'bad tool')]
    c.incorporate(ctx.key, proposals); c.close()
    evidence = [{'context': ctx.key, 'generations': [asdict(p) for p in proposals],
                 'planning_expansion': {'sid': selected['sid'], 'batch': selected['batch']}}]
    restored = restore_single_controller(journal, checkpoint_path=checkpoint, mcts_store=store,
                                         policy_records=evidence)
    try:
        assert restored.work['sampled_tokens'] == 7
        assert restored.work['raw_candidates'] == 2
        assert len(restored.trees[restored.target].edges) == 2
        assert any(e['status'] == 'rejected' for e in restored.trees[restored.target].edges)
        restored.next_context()
        assert restored.work['sampled_tokens'] == 7
    finally: restored.close()


def test_legacy_raw_request_hash_recovers_exact_committed_batch(tmp_path):
    from dataclasses import asdict
    from scripts.mcts_resume import restore_single_controller
    from mechet.shared_electron_search import fingerprint, sampling_seed
    from mechet.dynamic_electron_tree import Proposal
    c, store, journal, checkpoint = fixture_controller(tmp_path)
    ctx = c.next_context()
    proposals = [Proposal('import_fragments', {'fragments': [
        {'smiles': '[Cl-]', 'count': 1, 'purpose': 'electron_participant'}]}, 'import', -1., 4),
        Proposal('unknown', {}, 'invalid', -2., 3, 'bad tool')]
    request = {'context': ctx.key, 'prompt_ids': [101, 102], 'n': 16, 'max_tokens': 384,
        'temperature': 1, 'top_p': 1, 'top_k': -1, 'repetition_penalty': 1,
        'presence_penalty': 0, 'frequency_penalty': 0, 'skip_special_tokens': False,
        'logprobs': 0, 'min_tokens': 0, 'seed': sampling_seed(17, ctx.key, 0)}
    c.incorporate(ctx.key, proposals); c.close()
    records = [{'context': ctx.key, 'generations': [asdict(p) for p in proposals],
        'shared_request_key': fingerprint(request), '_prompt_lookup': lambda key: [101, 102]}]
    restored = restore_single_controller(journal, checkpoint_path=checkpoint,
        mcts_store=store, policy_records=records)
    try:
        assert restored.work['sampled_tokens'] == 7
        assert restored.work['raw_candidates'] == 2
    finally: restored.close()


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


def test_interrupted_restore_reuses_ancestor_report_without_writing_source(tmp_path):
    import json
    from scripts.mcts_resume import recover_shard
    from mechet.shared_electron_search import fingerprint
    previous = tmp_path / 'previous'; old = previous / 'tree/rank00'
    old.mkdir(parents=True)
    manifest = {'source_sha': 'source', 'adapter_sha': 'actor', 'rank': 0, 'world': 1,
        'program_goal': 1, 'seed': 17, 'max_model_len': 8192,
        'history_contract': 'history', 'shared_cache_namespace': 'cache'}
    (old / 'manifest.json').write_text(json.dumps(manifest))
    report = old / 'evaluations' / (fingerprint('done') + '.json')
    report.parent.mkdir(); report.write_text(json.dumps({'id': 'done', 'top1': True,
        'stop_reason': 'MCTS_TRAJECTORY_ATTEMPT_GOAL_REACHED'}))
    partial = tmp_path / 'partial/tree/rank00'; partial.mkdir(parents=True)
    (partial / 'manifest.json').write_text(json.dumps({**manifest, 'resume_source': str(previous)}))
    before = {str(p.relative_to(partial)): p.read_bytes() for p in partial.rglob('*') if p.is_file()}
    controllers, journals, reports, stats = recover_shard(
        [{'id': 'done', 'product_unmapped': 'CO'}], partial, tmp_path / 'new',
        manifest, planning=None, cache=None)
    assert not controllers and not journals
    assert reports[0]['top1'] is True and reports[0]['resumed_from'] == str(report)
    assert stats['reused_finished_reports'] == 1
    assert {str(p.relative_to(partial)): p.read_bytes() for p in partial.rglob('*') if p.is_file()} == before


def test_interrupted_journal_initialization_keeps_latest_checkpoint_and_old_path(tmp_path):
    import json
    from scripts.mcts_resume import recover_shard
    from scripts.single_reaction_mcts import SingleReactionMCTS, SingleReactionMCTSStore
    from mechet.shared_electron_search import SearchJournal, fingerprint, read_record, durable_json
    previous = tmp_path / 'previous'; old = previous / 'tree/rank00'; old.mkdir(parents=True)
    manifest = {'source_sha': 'source', 'adapter_sha': 'actor', 'rank': 0, 'world': 1,
        'program_goal': 2, 'seed': 17, 'max_model_len': 8192,
        'history_contract': 'history', 'shared_cache_namespace': 'cache'}
    (old / 'manifest.json').write_text(json.dumps(manifest))
    store = SingleReactionMCTSStore(tmp_path / 'shared', {'program_goal': 2})
    filename = fingerprint('case') + '.json'
    c = SingleReactionMCTS('case', 'CO', mcts_store=store,
        checkpoint_path=old / 'mcts' / filename, program_goal=2)
    j = SearchJournal(old / 'journals', 'case', manifest); j.initialize(c)
    import_chloride(c, j); context = c.next_context()
    expected_key, expected_state = context.key, context.node.state
    saved = read_record(old / 'mcts' / filename); c.close()
    partial = tmp_path / 'partial/tree/rank00'; partial.mkdir(parents=True)
    current = {**manifest, 'resume_source': str(previous)}
    (partial / 'manifest.json').write_text(json.dumps(current))
    saved['stop_reason'] = 'latest-checkpoint-marker'
    durable_json(partial / 'mcts' / filename, saved)
    SearchJournal(partial / 'journals', 'case', current)  # crash before initialize()
    before = {str(p.relative_to(partial)): p.read_bytes() for p in partial.rglob('*') if p.is_file()}
    controllers, _, _, stats = recover_shard([{'id': 'case', 'product_unmapped': 'CO'}],
        partial, tmp_path / 'new', manifest, planning=store, cache=None)
    restored = controllers['case']
    try:
        assert restored.stop_reason == 'latest-checkpoint-marker'
        assert restored.rollouts[0]['lease'].uid == saved['pending_rollouts'][0]['uid']
        assert restored.rollouts[0]['path'] == [tuple(p) for p in saved['pending_rollouts'][0]['path']]
        assert restored.contexts[expected_key].node.state == expected_state
        assert {r['lease'].uid: {'sid': r['sid'], 'path': [list(p) for p in r['path']]}
                for r in restored.rollouts} == {
                    r['uid']: {'sid': r['sid'], 'path': r['path']} for r in saved['pending_rollouts']}
        assert stats['restored_controllers'] == 1
        assert {str(p.relative_to(partial)): p.read_bytes() for p in partial.rglob('*') if p.is_file()} == before
    finally:
        restored.close()
