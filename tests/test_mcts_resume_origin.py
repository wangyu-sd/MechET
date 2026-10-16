"""Startup-interruption verification, with no model generation or GPU job."""
import json
from pathlib import Path

import pytest


def parent_manifests(tmp_path):
    parent = tmp_path / 'parent'
    rows = []
    for rank in range(8):
        row = {'rank': rank, 'world': 8, 'shard_rows': 390, 'source_sha': 'source',
            'adapter_sha': 'actor', 'seed': 17, 'program_goal': 1, 'max_model_len': 8192,
            'history_contract': 'history', 'shared_cache_namespace': 'native',
            'mcts_shared_root': str(tmp_path / 'legacy')}
        root = parent / 'tree' / f'rank{rank:02d}'; root.mkdir(parents=True)
        (root / 'manifest.json').write_text(json.dumps(row)); rows.append(row)
    return parent, rows


def publish(tmp_path):
    from scripts.mcts_resume import write_resume_origin
    parent, rows = parent_manifests(tmp_path)
    output = tmp_path / 'interrupted'
    path = write_resume_origin(output, parent, rows, store_root=output / 'ray_tree_store',
        legacy_root=tmp_path / 'legacy', code_sha='unit-runtime')
    return output, parent, rows, path


def test_origin_does_not_create_fresh_output_and_recovers_all_missing_rank_manifests(tmp_path):
    from scripts.start_ray_mcts_actor import load_resume_manifests
    output, parent, _, origin = publish(tmp_path)
    assert origin.exists() and not output.exists()
    rows = load_resume_manifests(output)
    assert [r['rank'] for r in rows] == list(range(8))
    assert all(r['source_sha'] == 'source' and r['resume_source'] == str(parent) for r in rows)
    assert all(r['mcts_shared_root'] == str(output / 'ray_tree_store') for r in rows)
    assert all(r['tree_store_backend'] == 'ray_memory_grouped_WAL_v1' for r in rows)
    assert not output.exists()  # reader cannot defeat the launcher's overwrite guard


def test_origin_can_reuse_parent_report_before_any_worker_manifest_is_written(tmp_path):
    from scripts.mcts_resume import recover_shard
    from mechet.shared_electron_search import fingerprint
    output, parent, rows, _ = publish(tmp_path)
    report = parent / 'tree/rank00/evaluations' / (fingerprint('done') + '.json')
    report.parent.mkdir(); report.write_text(json.dumps({'id': 'done', 'top1': True,
        'stop_reason': 'MCTS_TRAJECTORY_ATTEMPT_GOAL_REACHED'}))
    controllers, journals, reports, stats = recover_shard(
        [{'id': 'done', 'product_unmapped': 'CO'}], output / 'tree/rank00',
        tmp_path / 'next/tree/rank00', rows[0], planning=None, cache=None)
    assert not controllers and not journals and reports[0]['top1'] is True
    assert stats['reused_finished_reports'] == 1 and not output.exists()


def test_origin_cannot_be_copied_to_an_unrelated_output(tmp_path):
    from scripts.start_ray_mcts_actor import load_resume_manifests
    _, _, _, origin = publish(tmp_path)
    wrong = tmp_path / 'unrelated'
    Path(str(wrong) + '.resume_origin.json').write_bytes(origin.read_bytes())
    with pytest.raises(ValueError, match='origin|binding'):
        load_resume_manifests(wrong)


def test_origin_does_not_mask_an_incompatible_real_rank_manifest(tmp_path):
    from scripts.start_ray_mcts_actor import load_resume_manifests
    output, parent, rows, _ = publish(tmp_path)
    root = output / 'tree/rank03'; root.mkdir(parents=True)
    row = {**rows[3], 'adapter_sha': 'different', 'resume_source': str(parent),
        'code_sha': 'unit-runtime', 'tree_store_backend': 'ray_memory_grouped_WAL_v1',
        'mcts_shared_root': str(output / 'ray_tree_store')}
    (root / 'manifest.json').write_text(json.dumps(row))
    with pytest.raises(ValueError, match='contract'):
        load_resume_manifests(output)


def test_origin_is_idempotent_but_cannot_overwrite_a_different_parent(tmp_path):
    from scripts.mcts_resume import write_resume_origin
    output, parent, rows, origin = publish(tmp_path); before = origin.read_bytes()
    write_resume_origin(output, parent, rows, store_root=output / 'ray_tree_store',
        legacy_root=tmp_path / 'legacy', code_sha='unit-runtime')
    assert origin.read_bytes() == before
    with pytest.raises(ValueError, match='origin|conflict'):
        write_resume_origin(output, tmp_path / 'different', rows,
            store_root=output / 'ray_tree_store', legacy_root=tmp_path / 'legacy', code_sha='unit-runtime')
    assert origin.read_bytes() == before


def test_origin_keeps_planned_store_identity_if_owner_never_started(tmp_path):
    from scripts.start_ray_mcts_actor import load_resume_manifests, storage_paths
    output, _, _, _ = publish(tmp_path)
    root, legacy = storage_paths(load_resume_manifests(output)[0], tmp_path / 'next')
    assert root == output / 'ray_tree_store' and legacy == tmp_path / 'legacy'


def test_publishing_origin_rejects_existing_conflicting_manifest_without_new_metadata(tmp_path):
    from scripts.mcts_resume import write_resume_origin, origin_path
    parent, rows = parent_manifests(tmp_path)
    output = tmp_path / 'interrupted'; root = output / 'tree/rank00'; root.mkdir(parents=True)
    (root / 'manifest.json').write_text(json.dumps({**rows[0], 'adapter_sha': 'wrong'}))
    with pytest.raises(ValueError, match='contract'):
        write_resume_origin(output, parent, rows, store_root=output / 'ray_tree_store',
            legacy_root=tmp_path / 'legacy', code_sha='unit-runtime')
    assert not origin_path(output).exists()


def test_concurrent_origin_publications_cannot_replace_unacknowledged_identity(tmp_path, monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor, TimeoutError
    from scripts import mcts_resume as module
    parent, rows = parent_manifests(tmp_path); output = tmp_path / 'interrupted'
    entered, release, second_started = threading.Event(), threading.Event(), threading.Event()
    original = module.durable_json
    def delayed(path, value):
        if value.get('source_output') == str(parent):
            entered.set(); assert release.wait(5)
        return original(path, value)
    monkeypatch.setattr(module, 'durable_json', delayed)
    def write(source):
        return module.write_resume_origin(output, source, rows,
            store_root=output / 'ray_tree_store', legacy_root=tmp_path / 'legacy', code_sha='unit-runtime')
    def second():
        second_started.set(); return write(tmp_path / 'other-parent')
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(write, parent); assert entered.wait(2)
        other = pool.submit(second); assert second_started.wait(2)
        try:
            with pytest.raises(TimeoutError): other.result(timeout=.2)
        finally:
            release.set(); first.result(timeout=5)
        with pytest.raises(ValueError, match='origin|conflict'): other.result(timeout=5)
    assert module.read_record(module.origin_path(output))['source_output'] == str(parent)


def test_missing_rank_manifest_does_not_discard_committed_current_journal(tmp_path):
    from scripts.mcts_resume import write_resume_origin, recover_shard
    from scripts.single_reaction_mcts import SingleReactionMCTS, SingleReactionMCTSStore
    from mechet.shared_electron_search import SearchJournal, fingerprint, read_record, durable_json
    from mechet.dynamic_electron_tree import Proposal
    parent, rows = parent_manifests(tmp_path)
    output = tmp_path / 'interrupted'; source = output / 'tree/rank00'
    old = parent / 'tree/rank00'
    store = SingleReactionMCTSStore(tmp_path / 'planning', {'program_goal': 1})
    filename = fingerprint('case') + '.json'
    c = SingleReactionMCTS('case', 'CO', mcts_store=store, program_goal=1,
        checkpoint_path=old / 'mcts' / filename)
    original_journal = SearchJournal(old / 'journals', 'case', rows[0]); original_journal.initialize(c)
    context = c.next_context()
    edges = c.incorporate(context.key, [Proposal('import_fragments', {'fragments': [
        {'smiles': '[Cl-]', 'count': 1, 'purpose': 'electron_participant'}]}, 'import', -1., 4)])
    original_journal.record(c, context.key, edges); context = c.next_context()
    expected_key, expected_state = context.key, context.node.state
    saved = read_record(old / 'mcts' / filename)
    write_resume_origin(output, parent, rows, store_root=store.root,
        legacy_root=tmp_path / 'legacy', code_sha='unit-runtime')
    current = {**rows[0], 'resume_source': str(parent), 'mcts_shared_root': str(store.root),
        'tree_store_backend': 'ray_memory_grouped_WAL_v1', 'code_sha': 'unit-runtime',
        'worker_only_metadata': 'not inherited from parent'}
    current_journal = SearchJournal(source / 'journals', 'case', current); current_journal.initialize(c)
    saved['stop_reason'] = 'latest-current-checkpoint'; durable_json(source / 'mcts' / filename, saved)
    c.close()
    before = {str(p.relative_to(output)): p.read_bytes() for p in output.rglob('*') if p.is_file()}
    controllers, _, _, _ = recover_shard([{'id': 'case', 'product_unmapped': 'CO'}], source,
        tmp_path / 'next/tree/rank00', rows[0], planning=store, cache=None)
    restored = controllers['case']
    try:
        assert restored.stop_reason == 'latest-current-checkpoint'
        assert restored.contexts[expected_key].node.state == expected_state
        assert restored.work['sampled_tokens'] == 4
        assert not (source / 'manifest.json').exists()
        assert {str(p.relative_to(output)): p.read_bytes() for p in output.rglob('*') if p.is_file()} == before
    finally:
        restored.close()


def test_partial_startup_reads_nearest_saved_progress_without_new_work(tmp_path):
    from scripts.mcts_resume import read_prior_progress
    output, parent, _, _ = publish(tmp_path)
    path = parent / 'tree/rank00/progress.json'
    path.write_text(json.dumps({'output_tokens': 101, 'model_forward_calls': 7, 'elapsed_seconds': 12.5}))
    value = read_prior_progress(output / 'tree/rank00')
    assert (value['output_tokens'], value['model_forward_calls'], value['elapsed_seconds']) == (101, 7, 12.5)
    assert value['progress_source'] == str(path) and value['historical_costs_available'] is True


def test_repeated_resume_counts_carried_prior_costs_once_not_every_ancestor(tmp_path):
    from scripts.mcts_resume import read_prior_progress
    output, parent, _, _ = publish(tmp_path)
    (parent / 'tree/rank00/progress.json').write_text(json.dumps(
        {'output_tokens': 101, 'model_forward_calls': 7, 'elapsed_seconds': 12.5}))
    root = output / 'tree/rank00'; root.mkdir(parents=True)
    (root / 'progress.json').write_text(json.dumps({'output_tokens': 23, 'prior_output_tokens': 101,
        'model_forward_calls': 2, 'prior_model_forward_calls': 7,
        'elapsed_seconds': 4, 'prior_elapsed_seconds': 12.5}))
    value = read_prior_progress(root)
    assert (value['output_tokens'], value['model_forward_calls'], value['elapsed_seconds']) == (124, 9, 16.5)
    assert value['progress_source'] == str(root / 'progress.json')
    (root / 'progress.json').write_text(json.dumps(value))
    repeated = read_prior_progress(root)
    assert (repeated['output_tokens'], repeated['model_forward_calls'], repeated['elapsed_seconds']) == (124, 9, 16.5)


def test_missing_progress_is_explicitly_unavailable_not_a_missing_checkpoint_error(tmp_path):
    from scripts.mcts_resume import read_prior_progress
    output, _, _, _ = publish(tmp_path)
    value = read_prior_progress(output / 'tree/rank00')
    assert value['progress_source'] is None and value['historical_costs_available'] is False
