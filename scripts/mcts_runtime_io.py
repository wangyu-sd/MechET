"""Runtime-only MCTS optimizations; do not change planner/cache namespaces.

Only leases genuinely owned by this process bypass the filesystem probe.
Foreign leases, mutable node snapshots and exactly-once backup still use the
original shared store. Immutable policy receipts and raw outputs must finish
durable persistence before their result is returned to the chemistry executor.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import os
from pathlib import Path
import queue
import threading
import time
import weakref

from scripts.shared_route_mcts import SharedMCTSStore
from scripts.single_reaction_mcts import SingleReactionMCTSStore
from mechet.shared_electron_search import fingerprint


_OWNED_LEASES = weakref.WeakValueDictionary()
_LEASE_REGISTRY_LOCK = threading.RLock()


class _OwnedLeaseFastPath:
    def __init__(self, *args, **kwargs):
        self._local_liveness_checks = 0
        self._filesystem_liveness_checks = 0
        super().__init__(*args, **kwargs)
        self._lease_namespace = os.path.abspath(self.root)

    def start_rollout(self):
        lease = super().start_rollout()
        with _LEASE_REGISTRY_LOCK:
            _OWNED_LEASES[(os.getpid(), self._lease_namespace, lease.uid)] = lease
        return lease

    def alive(self, uid):
        # PID scoping is essential after fork: an inherited Python object is
        # not evidence that the parent process still holds its OS lock.
        with _LEASE_REGISTRY_LOCK:
            lease = _OWNED_LEASES.get((os.getpid(), self._lease_namespace, uid))
            if lease is not None:
                self._local_liveness_checks += 1
                return lease.handle is not None
            self._filesystem_liveness_checks += 1
        return super().alive(uid)

    def liveness_metrics(self):
        with _LEASE_REGISTRY_LOCK:
            return {'local_checks': self._local_liveness_checks,
                    'filesystem_checks': self._filesystem_liveness_checks}


class FastSharedMCTSStore(_OwnedLeaseFastPath, SharedMCTSStore):
    """Same multi-reaction planner and on-disk contract, cheaper local leases."""


class FastSingleReactionMCTSStore(_OwnedLeaseFastPath, SingleReactionMCTSStore):
    """Same single-reaction planner and on-disk contract, cheaper local leases."""


@dataclass(frozen=True)
class PersistedPolicyResult:
    case_id: str
    request: dict
    record: dict
    cache_hit: bool
    wall_seconds: float


class DurablePolicyWriter:
    """Bounded asynchronous persistence owned by one scheduler thread.

    submit/poll/close are scheduler-thread methods. Workers never touch a tree.
    An admitted lease transfers to the writer and is released on every worker
    exit, including disk errors. A rejected/full submission retains caller
    ownership. poll only exposes records after cache commit AND raw fsync.
    close drains every admitted write and raises unobserved write failures.
    """
    def __init__(self, raw_path, namespace, *, workers=2, max_pending=128):
        if workers < 1 or max_pending < 1:
            raise ValueError('positive I/O worker and pending capacities required')
        raw_path = Path(raw_path)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        self._raw = raw_path.open('a', encoding='utf-8')
        self._namespace = namespace
        self._capacity = max_pending
        self._raw_lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='policy-persistence')
        self._pending = {}
        self._closed = False
        self._persisted = 0
        self._failed = 0
        self._wall_seconds = 0.

    @property
    def available(self):
        return not self._closed and len(self._pending) < self._capacity

    @property
    def has_pending(self):
        return bool(self._pending)

    def submit(self, case_id, request, record, *, lease=None, cache_hit=False):
        if self._closed:
            raise RuntimeError('policy writer is closed')
        if not self.available:
            raise queue.Full('policy persistence backpressure')
        if (lease is None) != bool(cache_hit):
            raise ValueError('fresh results require a lease; cached results must not own one')
        # The scheduler retains no mutable use of request/record until poll.
        future = self._pool.submit(self._persist, case_id, request, record, lease, cache_hit)
        self._pending[future] = time.monotonic()

    def _persist(self, case_id, request, record, lease, cache_hit):
        started = time.monotonic()
        try:
            if lease is not None:
                lease.commit(record)
            row = {'case_id': case_id, 'context': request['context'],
                   'prompt_tokens': len(request.get('prompt_ids', [])),
                   'shared_cache_hit': cache_hit,
                   'shared_request_key': fingerprint(request),
                   'shared_cache_namespace': self._namespace, **record}
            encoded = json.dumps(row, ensure_ascii=False) + '\n'
            with self._raw_lock:
                self._raw.write(encoded)
                self._raw.flush()
                os.fsync(self._raw.fileno())
            return PersistedPolicyResult(case_id, request, record, cache_hit, time.monotonic() - started)
        finally:
            if lease is not None:
                lease.close()

    def _consume(self, future):
        self._pending.pop(future)
        try:
            result = future.result()
        except BaseException:
            self._failed += 1
            raise
        self._persisted += 1
        self._wall_seconds += result.wall_seconds
        return result

    def poll(self):
        for future in list(self._pending):
            if future.done():
                return self._consume(future)
        return None

    def metrics(self):
        now = time.monotonic()
        return {'pending_batches': len(self._pending), 'persisted_batches': self._persisted,
                'failed_batches': self._failed,
                'oldest_pending_seconds': max((now - start for start in self._pending.values()), default=0.),
                # Concurrent operations overlap: never divide this sum by wall
                # time and report it as an exclusive end-to-end time fraction.
                'worker_wall_seconds_sum': self._wall_seconds}

    def close(self):
        if self._closed:
            return
        self._closed = True
        error = None
        try:
            self._pool.shutdown(wait=True)
            for future in list(self._pending):
                try:
                    self._consume(future)
                except BaseException as exc:
                    if error is None:
                        error = exc
        finally:
            self._raw.close()
        if error is not None:
            raise error
