"""CPU-only storage/scheduler regression tests, not model experiments."""
import json
import multiprocessing
import queue
import threading
import time
from pathlib import Path

import pytest


def test_owned_rollouts_need_no_filesystem_liveness_probe(tmp_path, monkeypatch):
    from scripts.mcts_runtime_io import FastSharedMCTSStore
    store = FastSharedMCTSStore(tmp_path, {})
    root = store.intern({'position': 'root'})
    store.add_edges(root, [{'id': 'a', 'child': None, 'count': 1}], observation='unit')
    leases = [store.start_rollout() for _ in range(8)]
    for lease in leases:
        store.reserve(lease.uid, root, 'a')
    original_open = Path.open

    def no_rollout_open(path, *args, **kwargs):
        if path.parent.name == 'rollouts':
            raise AssertionError('owned liveness must not open a Ceph rollout lock')
        return original_open(path, *args, **kwargs)

    try:
        monkeypatch.setattr(Path, 'open', no_rollout_open)
        assert len(store.snapshot(root)['reservations']) == 8
        assert store.needs_expansion(root)
        assert store.select_edge(root)['id'] == 'a'
    finally:
        for lease in leases:
            lease.close()
    assert store.liveness_metrics()['local_checks'] == 24
    assert store.liveness_metrics()['filesystem_checks'] == 0


@pytest.mark.parametrize('single', [False, True])
def test_fast_store_keeps_existing_shared_namespace_and_cross_store_locks(tmp_path, single):
    from scripts.mcts_runtime_io import FastSharedMCTSStore, FastSingleReactionMCTSStore
    from scripts.shared_route_mcts import SharedMCTSStore
    from scripts.single_reaction_mcts import SingleReactionMCTSStore
    old_type, fast_type = ((SingleReactionMCTSStore, FastSingleReactionMCTSStore) if single
                           else (SharedMCTSStore, FastSharedMCTSStore))
    old = old_type(tmp_path, {'actor': 'unit'})
    fast = fast_type(tmp_path, {'actor': 'unit'})
    assert fast.contract == old.contract and fast.root == old.root
    root = old.intern({'position': 'root'})
    old.add_edges(root, [{'id': 'a', 'child': None, 'count': 1}], observation='unit')
    lease = old.start_rollout()
    try:
        old.reserve(lease.uid, root, 'a')
        assert fast.snapshot(root)['reservations'] == {lease.uid: 'a'}
        assert fast.liveness_metrics()['filesystem_checks'] == 1
    finally:
        lease.close()
    value = fast.snapshot(root)
    assert value['reservations'] == {}
    assert value['visits'] == 0 and value['edges']['a']['n'] == 0


def test_same_process_store_instances_share_owned_lease_knowledge(tmp_path, monkeypatch):
    from scripts.mcts_runtime_io import FastSharedMCTSStore
    first, second = FastSharedMCTSStore(tmp_path, {}), FastSharedMCTSStore(tmp_path, {})
    lease = first.start_rollout()
    original_open = Path.open

    def no_rollout_open(path, *args, **kwargs):
        if path.parent.name == 'rollouts':
            raise AssertionError('same-process lease should be known locally')
        return original_open(path, *args, **kwargs)

    try:
        monkeypatch.setattr(Path, 'open', no_rollout_open)
        assert second.alive(lease.uid)
        lease.close()
        assert not second.alive(lease.uid)
    finally:
        lease.close()


def test_forked_process_does_not_trust_inherited_lease_registry(tmp_path):
    from scripts.mcts_runtime_io import FastSharedMCTSStore
    store = FastSharedMCTSStore(tmp_path, {})
    lease = store.start_rollout()
    parent, child = multiprocessing.get_context('fork').Pipe()

    def check_after_parent_closes():
        parent.close()
        child.send('ready')
        child.recv()
        child.send(store.alive(lease.uid))
        child.close()

    process = multiprocessing.get_context('fork').Process(target=check_after_parent_closes)
    process.start()
    child.close()
    try:
        assert parent.poll(5) and parent.recv() == 'ready'
        lease.close()
        parent.send('check')
        assert parent.poll(5) and parent.recv() is False
        process.join(5)
        assert process.exitcode == 0
    finally:
        lease.close()
        parent.close()
        if process.is_alive():
            process.terminate()
            process.join(5)


def record(raw='raw tool call'):
    return {'request_id': 'unit', 'cached_prefix_tokens': 2,
            'generations': [{'name': 'finish_trace', 'arguments': {}, 'raw': raw,
                             'logprob': -3.5, 'tokens': 2, 'error': '',
                             'truncated': False, 'token_ids': [11, 12]}],
            'engine_finish_reasons': ['stop']}


def poll_until(writer):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        result = writer.poll()
        if result is not None:
            return result
        time.sleep(.002)
    pytest.fail('policy persistence failed to finish')


def test_slow_commit_does_not_block_other_case_or_publish_early(tmp_path, monkeypatch):
    from scripts.mcts_runtime_io import DurablePolicyWriter
    from mechet import shared_electron_search as storage
    store = storage.SharedSearchStore(tmp_path / 'cache', {'actor': 'unit'})
    request_a, request_b = {'context': 'a', 'prompt_ids': [1]}, {'context': 'b', 'prompt_ids': [2]}
    lease_a, _ = store.acquire_request(request_a)
    lease_b, _ = store.acquire_request(request_b)
    entered, release = threading.Event(), threading.Event()
    original = storage.durable_json

    def slow_disk(path, value):
        if Path(path) == lease_a.path:
            entered.set()
            assert release.wait(5), 'test disk gate timed out'
        return original(path, value)

    monkeypatch.setattr(storage, 'durable_json', slow_disk)
    writer = DurablePolicyWriter(tmp_path / 'raw.jsonl', store.namespace, workers=2, max_pending=2)
    try:
        writer.submit('a', request_a, record('first'), lease=lease_a)
        assert entered.wait(5)
        assert writer.poll() is None
        assert store.acquire_request(request_a) == (None, None)
        writer.submit('b', request_b, record('second'), lease=lease_b)
        result = poll_until(writer)
        assert result.case_id == 'b' and not result.cache_hit
        assert result.record == record('second')
        assert store.acquire_request(request_b) == (None, record('second'))
        assert lease_b.handle is None
        raw = json.loads((tmp_path / 'raw.jsonl').read_text())
        assert raw['case_id'] == 'b' and raw['generations'][0]['token_ids'] == [11, 12]
        assert raw['generations'][0]['logprob'] == -3.5
        release.set()
        assert poll_until(writer).case_id == 'a'
    finally:
        release.set()
        writer.close()


def test_full_pending_queue_applies_backpressure_without_losing_a_lease(tmp_path, monkeypatch):
    from scripts.mcts_runtime_io import DurablePolicyWriter
    from mechet import shared_electron_search as storage
    store = storage.SharedSearchStore(tmp_path / 'cache', {})
    first, _ = store.acquire_request({'context': 'first'})
    second, _ = store.acquire_request({'context': 'second'})
    entered, release = threading.Event(), threading.Event()
    original = storage.durable_json

    def slow_disk(path, value):
        entered.set()
        assert release.wait(5)
        return original(path, value)

    monkeypatch.setattr(storage, 'durable_json', slow_disk)
    writer = DurablePolicyWriter(tmp_path / 'raw.jsonl', store.namespace, workers=1, max_pending=1)
    try:
        writer.submit('first', {'context': 'first'}, record(), lease=first)
        assert entered.wait(5) and not writer.available
        with pytest.raises(queue.Full):
            writer.submit('second', {'context': 'second'}, record(), lease=second)
        assert second.handle is not None
        release.set()
        assert poll_until(writer).case_id == 'first'
        assert writer.available
        writer.submit('second', {'context': 'second'}, record(), lease=second)
    finally:
        release.set()
        writer.close()
        second.close()
    assert len((tmp_path / 'raw.jsonl').read_text().splitlines()) == 2


@pytest.mark.parametrize('observe_with', ['poll', 'close'])
def test_disk_failure_propagates_and_releases_policy_claim(tmp_path, monkeypatch, observe_with):
    from scripts.mcts_runtime_io import DurablePolicyWriter
    from mechet import shared_electron_search as storage
    store = storage.SharedSearchStore(tmp_path / 'cache', {})
    request = {'context': 'unit'}
    lease, _ = store.acquire_request(request)

    def broken_disk(*args, **kwargs):
        raise OSError('unit disk full')

    monkeypatch.setattr(storage, 'durable_json', broken_disk)
    writer = DurablePolicyWriter(tmp_path / 'raw.jsonl', store.namespace)
    writer.submit('unit', request, record(), lease=lease)
    try:
        with pytest.raises(OSError, match='disk full'):
            poll_until(writer) if observe_with == 'poll' else writer.close()
    finally:
        writer.close()
    assert lease.handle is None
    retry, saved = store.acquire_request(request)
    assert retry is not None and saved is None
    retry.close()
    assert (tmp_path / 'raw.jsonl').read_text() == ''


def test_cached_results_and_shutdown_preserve_every_raw_candidate(tmp_path):
    from scripts.mcts_runtime_io import DurablePolicyWriter
    writer = DurablePolicyWriter(tmp_path / 'raw.jsonl', 'unit-namespace', workers=2, max_pending=16)
    for index in range(16):
        writer.submit(str(index), {'context': str(index), 'prompt_ids': [1, 2]},
                      record('raw\nwith unicode 氯'), cache_hit=True)
    writer.close()
    rows = [json.loads(line) for line in (tmp_path / 'raw.jsonl').read_text().splitlines()]
    assert {row['case_id'] for row in rows} == {str(i) for i in range(16)}
    assert all(row['shared_cache_hit'] and row['prompt_tokens'] == 2 for row in rows)
    assert all(row['shared_cache_namespace'] == 'unit-namespace' for row in rows)
    assert all(row['generations'][0]['raw'] == 'raw\nwith unicode 氯' for row in rows)
    assert writer.metrics()['persisted_batches'] == 16


def test_raw_fsync_failure_never_publishes_an_executable_result(tmp_path, monkeypatch):
    import os
    from scripts.mcts_runtime_io import DurablePolicyWriter
    from mechet.shared_electron_search import SharedSearchStore
    store = SharedSearchStore(tmp_path / 'cache', {})
    request = {'context': 'unit', 'prompt_ids': [1]}
    lease, _ = store.acquire_request(request)
    raw_path = tmp_path / 'raw.jsonl'
    writer = DurablePolicyWriter(raw_path, store.namespace)
    identity = raw_path.stat()
    original = os.fsync

    def broken_raw_disk(fd):
        current = os.fstat(fd)
        if (current.st_dev, current.st_ino) == (identity.st_dev, identity.st_ino):
            raise OSError('unit raw fsync failed')
        return original(fd)

    monkeypatch.setattr(os, 'fsync', broken_raw_disk)
    writer.submit('unit', request, record(), lease=lease)
    try:
        with pytest.raises(OSError, match='raw fsync failed'):
            poll_until(writer)
    finally:
        writer.close()
    assert lease.handle is None
    assert writer.metrics()['persisted_batches'] == 0
    assert writer.metrics()['failed_batches'] == 1
    assert store.acquire_request(request) == (None, record())  # Durable original remains recoverable.
