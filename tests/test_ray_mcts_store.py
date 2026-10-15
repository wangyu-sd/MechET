"""Durable search-store tests, independent of GPU/model generation."""
import copy
import json
import os
from concurrent.futures import ThreadPoolExecutor
import pytest


def make_core(tmp_path, legacy=None):
    from scripts.ray_mcts_store import MemoryMCTSCore
    return MemoryMCTSCore(tmp_path / 'memory', {'fixture': True}, legacy_root=legacy)


def seed(core):
    sid = core.intern({'active': 'root', 'solved': False})
    core.add_edges(sid, [{'id': 'a', 'child': None, 'count': 1}], observation='obs')
    return sid


def test_concurrent_backup_is_idempotent_and_recoverable(tmp_path):
    core = make_core(tmp_path)
    sid = seed(core)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: core.backup('same', [(sid, 'a')], 1.), range(16)))
    assert core.snapshot(sid)['visits'] == 1
    core.close()
    core = make_core(tmp_path)
    try:
        assert core.snapshot(sid)['value_sum'] == 1.
        core.backup('same', [(sid, 'a')], 1.)
        assert core.snapshot(sid)['edges']['a']['n'] == 1
        with pytest.raises(ValueError, match='conflict'):
            core.backup('same', [(sid, 'a')], 0.)
    finally:
        core.close()


def test_legacy_read_through_does_not_modify_original_files(tmp_path):
    from scripts.single_reaction_mcts import SingleReactionMCTSStore
    legacy = SingleReactionMCTSStore(tmp_path / 'legacy', {})
    sid = legacy.intern({'active': 'root', 'solved': False})
    legacy.add_edges(sid, [{'id': 'a', 'child': None, 'count': 1}], observation='obs')
    before = legacy.path(sid).read_bytes()
    from scripts.ray_mcts_store import MemoryMCTSCore
    core = MemoryMCTSCore(tmp_path / 'memory', legacy.contract, legacy_root=legacy.root)
    core.backup('new', [(sid, 'a')], 1.)
    assert core.snapshot(sid)['visits'] == 1
    assert legacy.path(sid).read_bytes() == before
    core.close()
    core = MemoryMCTSCore(tmp_path / 'memory', legacy.contract, legacy_root=legacy.root)
    try:
        assert core.snapshot(sid)['visits'] == 1
    finally:
        core.close()


def test_incomplete_wal_tail_is_recovered_but_corruption_is_rejected(tmp_path):
    core = make_core(tmp_path); sid = seed(core); core.close()
    wal = tmp_path / 'memory' / 'transactions.jsonl'
    with wal.open('ab') as f:
        f.write(b'{"incomplete":')
    core = make_core(tmp_path)
    core.backup('ok', [(sid, 'a')], 0.); core.close()
    core = make_core(tmp_path)
    assert core.snapshot(sid)['visits'] == 1
    core.close()
    lines = wal.read_bytes().splitlines(keepends=True)
    row = json.loads(lines[0]); row['sha256'] = 'bad'
    lines[0] = json.dumps(row).encode() + b'\n'; wal.write_bytes(b''.join(lines))
    with pytest.raises(ValueError, match='corrupt'):
        make_core(tmp_path)


def test_two_live_tree_owners_are_rejected(tmp_path):
    core = make_core(tmp_path)
    try:
        with pytest.raises((ValueError, BlockingIOError), match='live|temporarily'):
            make_core(tmp_path)
    finally:
        core.close()


def test_pending_leases_can_be_reclaimed_after_owner_restart(tmp_path):
    core = make_core(tmp_path)
    core.register_owner('worker', os.getpid())
    core.claim_rollout('uid', 'worker')
    assert core.alive('uid')
    core.release_rollout('uid', 'worker')
    assert not core.alive('uid')
    core.claim_rollout('uid', 'worker')
    core.close()
    core = make_core(tmp_path)
    try:
        assert not core.alive('uid')
        core.register_owner('replacement', os.getpid())
        core.claim_rollout('uid', 'replacement')
        assert core.alive('uid')
    finally:
        core.close()


def test_memory_and_legacy_selection_match_without_ceph_node_probes(tmp_path):
    from scripts.single_reaction_mcts import SingleReactionMCTSStore
    from scripts.ray_mcts_store import MemoryMCTSCore
    legacy = SingleReactionMCTSStore(tmp_path / 'legacy', {})
    root = legacy.intern({'active': 'root', 'solved': False})
    legacy.add_edges(root, [{'id': 'a', 'child': None, 'count': 2},
                           {'id': 'b', 'child': 'b', 'count': 1}], observation='obs')
    core = MemoryMCTSCore(tmp_path / 'memory', legacy.contract, legacy_root=legacy.root)
    try:
        for i in range(8):
            assert core.needs_expansion(root) == legacy.needs_expansion(root)
            assert core.select_edge(root) == legacy.select_edge(root)
            chosen = core.select_edge(root)['id']
            core.backup(str(i), [(root, chosen)], float(i % 2))
            legacy.backup(str(i), [(root, chosen)], float(i % 2))
    finally:
        core.close()


def test_client_identity_returns_exact_contract_and_persistence_root(tmp_path):
    core = make_core(tmp_path)
    try:
        path, contract = core.identity()
        assert path == str(tmp_path / 'memory') and contract == {'fixture': True}
    finally:
        core.close()


def test_invalid_backup_cannot_partially_mutate_a_valid_prefix(tmp_path):
    core = make_core(tmp_path); sid = seed(core)
    try:
        with pytest.raises(KeyError):
            core.backup('broken', [(sid, 'a'), ('absent', 'bad')], 1.)
        assert core.snapshot(sid)['visits'] == 0
        assert core.backup_record('broken') is None
    finally:
        core.close()


def test_failed_fsync_prevents_durable_ack_and_future_reads(tmp_path, monkeypatch):
    core = make_core(tmp_path)
    def disk_failed(*args): raise OSError('unit disk failure')
    monkeypatch.setattr(os, 'fsync', disk_failed)
    try:
        with pytest.raises(RuntimeError, match='writer failed'):
            core.intern({'active': 'x', 'solved': False})
        from mechet.shared_electron_search import fingerprint
        with pytest.raises(RuntimeError, match='writer failed'):
            core.snapshot(fingerprint({'active': 'x', 'solved': False}))
    finally:
        with pytest.raises(RuntimeError, match='writer failed'):
            core.close()


def _abrupt_owner(root):
    from pathlib import Path
    core = make_core(Path(root)); sid = seed(core)
    core.backup('acknowledged', [(sid, 'a')], 1.)
    os._exit(0)  # no close(), checkpoint(), Python finalizers, or surviving RAM


def test_abrupt_process_exit_recovers_acknowledged_transactions(tmp_path):
    import multiprocessing
    proc = multiprocessing.get_context('spawn').Process(target=_abrupt_owner, args=(str(tmp_path),))
    proc.start(); proc.join(20)
    if proc.is_alive():
        proc.terminate(); proc.join(); pytest.fail('owner failed to exit')
    assert proc.exitcode == 0
    core = make_core(tmp_path)
    try:
        from mechet.shared_electron_search import fingerprint
        assert core.snapshot(fingerprint({'active': 'root', 'solved': False}))['visits'] == 1
    finally:
        core.close()


def test_real_ray_client_and_actor_restart(tmp_path):
    ray = pytest.importorskip('ray')
    from scripts.ray_mcts_store import RayMCTSStore, start_actor
    ray.init(num_cpus=2, include_dashboard=False, object_store_memory=100 * 1024 * 1024,
             namespace='unit-mcts', _node_ip_address='127.0.0.1')
    try:
        actor = start_actor(tmp_path / 'memory', {'fixture': True}, None,
                            name='unit-tree', namespace='unit-mcts')
        store = RayMCTSStore('unit-tree', namespace='unit-mcts')
        sid = seed(store)
        lease = store.start_rollout(); store.reserve(lease.uid, sid, 'a')
        store.backup(lease.uid, [(sid, 'a')], 1.); lease.close()
        store._call('close')
        ray.kill(actor, no_restart=True)
        # Ray name registration can lag process termination briefly.
        import time
        for _ in range(100):
            try: ray.get_actor('unit-tree', namespace='unit-mcts')
            except ValueError: break
            time.sleep(.05)
        actor = start_actor(tmp_path / 'memory', {'fixture': True}, None,
                            name='unit-tree', namespace='unit-mcts')
        restored = RayMCTSStore('unit-tree', namespace='unit-mcts')
        assert restored.snapshot(sid)['visits'] == 1
        restored._call('close')
    finally:
        ray.shutdown()


def test_real_ray_restores_legacy_pending_controller(tmp_path):
    ray = pytest.importorskip('ray')
    from scripts.ray_mcts_store import RayMCTSStore, start_actor
    from scripts.mcts_resume import restore_single_controller
    from scripts.single_reaction_mcts import SingleReactionMCTS, SingleReactionMCTSStore
    from mechet.shared_electron_search import SearchJournal, read_record
    from mechet.dynamic_electron_tree import Proposal
    legacy = SingleReactionMCTSStore(tmp_path / 'legacy', {'program_goal': 1})
    checkpoint = tmp_path / 'controller.json'
    c = SingleReactionMCTS('case', 'CO', mcts_store=legacy, checkpoint_path=checkpoint, program_goal=1)
    journal = SearchJournal(tmp_path / 'journals', 'case', {'fixture': True}); journal.initialize(c)
    ctx = c.next_context()
    edges = c.incorporate(ctx.key, [Proposal('import_fragments', {'fragments': [
        {'smiles': '[Cl-]', 'count': 1, 'purpose': 'electron_participant'}]}, 'import', -1., 4)])
    journal.record(c, ctx.key, edges)
    ctx = c.next_context(); old_state = ctx.node.state
    uid = read_record(checkpoint)['pending_rollouts'][0]['uid']; c.close()
    ray.init(num_cpus=2, include_dashboard=False, object_store_memory=100 * 1024 * 1024,
             namespace='unit-resume', _node_ip_address='127.0.0.1')
    restored = None
    try:
        start_actor(tmp_path / 'memory', legacy.contract, legacy.root,
                    name='resume-tree', namespace='unit-resume')
        store = RayMCTSStore('resume-tree', namespace='unit-resume')
        restored = restore_single_controller(journal, checkpoint_path=checkpoint,
                                              mcts_store=store)
        assert restored.rollouts[0]['lease'].uid == uid
        assert restored.next_context().node.state == old_state
        assert restored.completed_rollouts == restored.censored_rollouts == 0
    finally:
        if restored is not None: restored.close()
        ray.shutdown()


def test_runtime_contract_matches_unchanged_legacy_planner(tmp_path):
    from scripts.ray_mcts_store import single_planning_contract
    from scripts.single_reaction_mcts import SingleReactionMCTSStore
    expected = SingleReactionMCTSStore(tmp_path / 'legacy', {
        'policy_executor_namespace': 'native', 'max_model_len': 8192,
        'sampling_seed': 17, 'program_goal': 10}).contract
    assert single_planning_contract('native', 8192, 17, 10) == expected
