"""Durable shared policy/executor cache and incremental, recoverable graphs.

No labels, stocks, target IDs or endpoint rewards enter shared cache lookup.
Per-task stock closure and evaluation remain separate. Requires a filesystem
supporting POSIX rename, fsync and cross-host flock (including the Ceph mount).
"""
from __future__ import annotations

from dataclasses import asdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def durable_json(path, value):
    """Publish a checksum-protected complete record; never expose a torn JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', suffix='.partial', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            json.dump({'sha256': fingerprint(value), 'payload': value}, handle, ensure_ascii=False)
            handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_record(path):
    with Path(path).open(encoding='utf-8') as handle:
        record = json.load(handle)
    if fingerprint(record['payload']) != record['sha256']:
        raise ValueError(f'corrupt search record: {path}')
    return record['payload']


class RequestLease:
    def __init__(self, path, handle):
        self.path, self.handle = path, handle

    def commit(self, value):
        if self.handle is None:
            raise RuntimeError('closed request lease')
        if self.path.exists():
            if read_record(self.path) != value:
                raise ValueError('shared request already committed with different result')
            return
        durable_json(self.path, value)

    def close(self):
        if self.handle is not None:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()
            self.handle = None


class SharedSearchStore:
    """Immutable cache records; nonblocking claims avoid multi-key deadlocks."""
    def __init__(self, root, contract):
        self.contract = {'schema': 'shared_electron_search_v1', **contract}
        self.namespace = fingerprint(self.contract)
        self.root = Path(root) / self.namespace
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, kind, key):
        return self.root / kind / key[:2] / (key + '.json')

    def acquire_request(self, request):
        path = self._path('policy', fingerprint(request))
        if path.exists():
            return None, read_record(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.with_suffix('.lock').open('a+')
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            return None, None  # another worker owns this request; defer, don't resample
        if path.exists():
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN); handle.close()
            return None, read_record(path)
        return RequestLease(path, handle), None

    def execution(self, key):
        path = self._path('execution', key)
        return read_record(path) if path.exists() else None

    def save_execution(self, key, value):
        path = self._path('execution', key)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.with_suffix('.lock').open('a+') as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            if path.exists():
                if read_record(path) != value:
                    raise ValueError('nondeterministic executor receipt under same contract')
            else:
                durable_json(path, value)


def sampling_seed(seed, context, batch_index):
    # Case ID/rank/source are deliberately excluded so matching contexts share.
    return int(fingerprint(['shared_context_seed_v1', seed, context, batch_index])[:8], 16)


def resolve_policy_requests(store, requests, generate):
    """Return (record, cache_hit) or None/busy, generating only claimed misses.

    Leases are always released, including engine/serialization failures. Shared
    results are durably committed BEFORE returning them to the executor loop.
    """
    results = [None] * len(requests)
    missing = []; leases = []
    try:
        for i, request in enumerate(requests):
            lease, cached = store.acquire_request(request)
            if cached is not None:
                results[i] = (cached, True)
            elif lease is not None:
                missing.append(i); leases.append(lease)
        if missing:
            records = generate([requests[i] for i in missing])
            if len(records) != len(missing):
                raise ValueError('policy result/request cardinality mismatch')
            for i, lease, record in zip(missing, leases, records, strict=True):
                lease.commit(record)
                results[i] = (record, False)
        return results
    finally:
        for lease in leases:
            lease.close()


def context_record(context):
    node = asdict(context.node)
    node['visited'] = sorted(node['visited'])
    return {'id': context.key, 'chemical_id': context.chemical_id, 'node': node,
            'stats': {'draws': context.stats.draws, 'counts': dict(context.stats.counts),
                      'batches': context.stats.batches},
            'resource_closed': context.resource_closed, 'max_new_tokens': context.max_new_tokens,
            'greedy_done': context.greedy_done}


def tree_snapshot(tree):
    return {'case_id': tree.case_id, 'target': tree.target, 'root_key': tree.root_key,
            'contexts': {key: context_record(c) for key, c in tree.contexts.items()},
            'chemical_nodes': tree.chemical_nodes, 'edges': tree.edges,
            'work': dict(tree.work), 'stop_reason': tree.stop_reason}


def restore_tree(value, shared_store=None):
    from collections import Counter
    from .dynamic_electron_tree import ElectronTree, SearchContext, NoveltyStats
    from scripts.run_natural_language_value_search import Node
    tree = ElectronTree.__new__(ElectronTree)
    for key in ('case_id', 'target', 'root_key', 'chemical_nodes', 'edges', 'work', 'stop_reason'):
        setattr(tree, key, value[key])
    tree.contexts = {}
    for key, record in value['contexts'].items():
        node = dict(record['node']); node['visited'] = set(node['visited'])
        stats = record['stats']
        tree.contexts[key] = SearchContext(key, Node(**node), record['chemical_id'],
            NoveltyStats(stats['draws'], Counter(stats['counts']), stats['batches']),
            record['resource_closed'], record['max_new_tokens'], record['greedy_done'])
    tree.execution_cache = {}
    tree.shared_store = shared_store
    return tree


def route_state(controller):
    return {'case_id': controller.case_id, 'target': controller.target, 'mode': controller.mode,
            'already_in_stock': controller.already_in_stock,
            'forced': dict(controller.forced), 'reaction_edges': [
                controller._serial_edge(e) for e in controller.reaction_edges],
            'failed_obligations': list(controller.failed_obligations),
            'rejected_route_terminals': list(controller.rejected_route_terminals),
            'first_solution_work': controller.first_solution_work,
            'stop_reason': controller.stop_reason}


class SearchJournal:
    """One writer per target; atomic incremental commits, independent of finish.

    Every commit contains parent statistics, accepted AND rejected edges, exact
    changed native contexts and outer route state. Recovery needs no model or
    executor calls. Files named .partial are never treated as committed records.
    """
    def __init__(self, root, case_id, contract):
        self.root = Path(root) / fingerprint(case_id)
        self.root.mkdir(parents=True, exist_ok=True)
        self.contract = {'schema': 'electron_search_journal_v1', 'case_id': case_id, **contract}
        header = self.root / 'contract.json'
        with (self.root / '.writer.lock').open('a+') as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            if header.exists():
                if read_record(header) != self.contract:
                    raise ValueError('search journal contract mismatch')
            else:
                durable_json(header, self.contract)
        paths = sorted(self.root.glob('events/*.json'))
        self.sequence = len(paths)
        if [p.name for p in paths] != [f'{i:09d}.json' for i in range(self.sequence)]:
            raise ValueError('search journal has missing commits')
        self.known_trees = set()
        if paths:
            for path in paths:
                event = read_record(path)
                self.known_trees.update(event.get('new_trees', {}))
        self.last_route_key = None

    def _commit(self, event):
        with (self.root / '.writer.lock').open('a+') as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            path = self.root / 'events' / f'{self.sequence:09d}.json'
            if path.exists():
                raise ValueError('another journal writer already committed this sequence; restore first')
            durable_json(path, event)
            self.sequence += 1

    def initialize(self, controller):
        if self.sequence:
            raise ValueError('journal already initialized; restore rather than overwrite')
        route = hasattr(controller, 'trees')
        trees = controller.trees if route else {'single': controller}
        self._commit({'kind': 'initial', 'route': route,
            'new_trees': {k: tree_snapshot(v) for k, v in trees.items()},
            'route_state': route_state(controller) if route else None})
        self.known_trees.update(trees)

    def record(self, controller, parent_key, edges):
        route = hasattr(controller, 'trees')
        trees = controller.trees if route else {'single': controller}
        owner = controller.owners[parent_key] if route else 'single'
        tree = trees[owner]
        keys = {parent_key} | {edge['child_context'] for edge in edges if edge.get('child_context')}
        contexts = {k: context_record(tree.contexts[k]) for k in keys}
        new = {k: tree_snapshot(v) for k, v in trees.items() if k not in self.known_trees}
        self._commit({'kind': 'expansion', 'new_trees': new, 'owner': owner,
            'contexts': contexts,
            'chemical_nodes': {c['chemical_id']: tree.chemical_nodes[c['chemical_id']] for c in contexts.values()},
            'edges': edges, 'work': dict(tree.work), 'stop_reason': tree.stop_reason,
            'route_state': route_state(controller) if route else None})
        self.known_trees.update(new)
        if route and controller.first_solution_work is not None:
            certificate = controller._certificate()
            certificate_key = fingerprint(certificate) if certificate else None
            if certificate and certificate_key != self.last_route_key:
                durable_json(self.root / 'best_route.json', {'case_id': controller.case_id,
                    'journal_sequence': self.sequence - 1, 'certificate': certificate,
                    'chemical_plausibility': 'not_independently_verified'})
                self.last_route_key = certificate_key

    def restore(self, *, stock=None, shared_store=None):
        trees = {}; route = False; outer = None
        for i in range(self.sequence):
            event = read_record(self.root / 'events' / f'{i:09d}.json')
            route = event.get('route', route)
            trees.update(event['new_trees'])
            if event['kind'] == 'expansion':
                tree = trees[event['owner']]
                tree['contexts'].update(event['contexts'])
                tree['chemical_nodes'].update(event['chemical_nodes'])
                tree['edges'].extend(event['edges'])
                tree['work'] = event['work']; tree['stop_reason'] = event['stop_reason']
            if event.get('route_state') is not None:
                outer = event['route_state']
        if not trees and not route:
            raise ValueError('journal has no initialized tree')
        if not route:
            return restore_tree(trees['single'], shared_store)
        if stock is None:
            raise ValueError('route recovery requires the original contract-verified stock')
        import time
        from .reverse_route_network import RouteNetwork
        controller = RouteNetwork.__new__(RouteNetwork)
        for key, value in outer.items():
            setattr(controller, key, value)
        controller.stock = stock; controller.shared_store = shared_store
        controller.trees = {k: restore_tree(v, shared_store) for k, v in trees.items()}
        controller.contexts = {}; controller.owners = {}
        for key in controller.trees:
            controller._index_contexts(key)
        for edge in controller.reaction_edges:
            edge['terminal_contexts'] = set(edge['terminal_contexts'])
        controller.edge_index = {e['id']: e for e in controller.reaction_edges}
        controller._closure_cache = None; controller.started = time.monotonic()
        return controller
