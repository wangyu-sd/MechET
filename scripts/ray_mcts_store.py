"""Ray-owned MCTS state: RAM transactions, durable grouped WAL, legacy read-through.

The legacy planner modules are not edited: their SHA-bound scientific contract
and state/edge identities stay unchanged. Only persistence and coordination move.
"""
from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from mechet.shared_electron_search import durable_json, fingerprint, read_record


def single_planning_contract(native_namespace, max_model_len, seed, program_goal):
    from scripts import shared_route_mcts, single_reaction_mcts
    def sha(module): return hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    return {'schema': shared_route_mcts.ALGORITHM, 'c_puct': 1.5,
            'planner_sha': sha(shared_route_mcts), 'policy_executor_namespace': native_namespace,
            'max_model_len': max_model_len, 'sampling_seed': seed, 'program_goal': program_goal,
            'evaluation_unit': 'single_reaction_endpoint', 'reward': 'productive_strict_finish_v1',
            'single_planner_sha': sha(single_reaction_mcts)}


class GroupedWAL:
    def __init__(self, path, sequence):
        self.path = Path(path)
        existed = self.path.exists()
        self.file = self.path.open('ab')
        if not existed:
            os.fsync(self.file.fileno())
            fd = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try: os.fsync(fd)
            finally: os.close(fd)
        self.committed = sequence
        self.condition = threading.Condition()
        self.queue = queue.Queue()
        self.error = None
        self.fsyncs = 0
        self.thread = threading.Thread(target=self._run, name='mcts-durable-WAL', daemon=True)
        self.thread.start()

    def append(self, sequence, operation):
        self.queue.put((sequence, operation))

    def wait(self, sequence):
        with self.condition:
            while self.committed < sequence and self.error is None:
                self.condition.wait(.5)
            if self.error is not None:
                raise RuntimeError('MCTS durable writer failed') from self.error

    def _run(self):
        try:
            while True:
                first = self.queue.get()
                if first is None: break
                batch = [first]
                deadline = time.monotonic() + .005
                while len(batch) < 512:
                    try:
                        item = self.queue.get(timeout=max(0., deadline - time.monotonic()))
                    except queue.Empty: break
                    if item is None:
                        self.queue.put(None); break
                    batch.append(item)
                for seq, op in batch:
                    payload = {'sequence': seq, 'operation': op}
                    self.file.write((json.dumps({'sha256': fingerprint(payload), 'payload': payload},
                                               ensure_ascii=False) + '\n').encode())
                self.file.flush(); os.fsync(self.file.fileno()); self.fsyncs += 1
                with self.condition:
                    self.committed = batch[-1][0]
                    self.condition.notify_all()
        except BaseException as exc:
            with self.condition:
                self.error = exc; self.condition.notify_all()

    def close(self):
        self.queue.put(None); self.thread.join()
        self.file.close()
        self.wait(self.committed)


class MemoryMCTSCore:
    """Serialized mutations; acknowledgments fence their durable sequence.

    Ray uses threaded actor methods so independent callers can share a WAL
    group commit. The lock protects state, NOT the fsync wait. No GPU calls or
    chemistry execute in this owner. All public values are detached copies.
    """
    def __init__(self, root, contract, *, legacy_root=None):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        self.contract = dict(contract)
        self.legacy_root = Path(legacy_root) if legacy_root else None
        self._owner_lock = (self.root / '.owner.lock').open('a+')
        try: fcntl.flock(self._owner_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._owner_lock.close(); raise ValueError('another live MCTS tree owner') from None
        self._lock = threading.RLock(); self._checkpoint_lock = threading.Lock()
        self.nodes = {}; self.backups = {}; self.logged_backups = set()
        self._readers = ThreadPoolExecutor(max_workers=32, thread_name_prefix='legacy-prefetch')
        self._legacy_ids = {p.stem for p in (self.legacy_root / 'nodes').glob('*/*.json')} if self.legacy_root else set()
        self._legacy_backup_ids = {p.stem for p in (self.legacy_root / 'backups').glob('*.json')} if self.legacy_root else set()
        self._legacy_rollout_ids = {p.stem for p in (self.legacy_root / 'rollouts').glob('*.lock')} if self.legacy_root else set()
        self._legacy_claims = {}
        self._lease_locks = {}
        self.owners = {}; self.leases = {}; self.expansions = {}
        self.recovery_workers = set(); self.expected_workers = 8
        self.sequence = 0; self.closed = False; self.legacy_reads = 0
        try:
            header = self.root / 'contract.json'
            identity = {'contract': self.contract,
                        'legacy_root': str(self.legacy_root.resolve()) if self.legacy_root else None}
            if header.exists():
                if read_record(header) != identity: raise ValueError('tree store contract mismatch')
            else: durable_json(header, identity)
            self._replay()
            self.writer = GroupedWAL(self.root / 'transactions.jsonl', self.sequence)
        except BaseException:
            self._readers.shutdown(wait=True)
            self._owner_lock.close(); raise

    def _prefetch(self, sids):
        with self._lock:
            missing = set(sids).intersection(self._legacy_ids).difference(self.nodes)
        def load(sid):
            return sid, read_record(self.legacy_root / 'nodes' / sid[:2] / (sid + '.json'))
        for sid, value in self._readers.map(load, missing):
            with self._lock:
                if sid not in self.nodes:
                    self.nodes[sid] = value; self.legacy_reads += 1

    def _replay(self):
        path = self.root / 'transactions.jsonl'
        if not path.exists(): return
        good = 0
        with path.open('rb') as f:
            for line in f:
                if not line.endswith(b'\n'): break
                try:
                    record = json.loads(line); value = record['payload']
                    if record['sha256'] != fingerprint(value): raise ValueError('checksum')
                    if value['sequence'] != self.sequence + 1: raise ValueError('sequence')
                    op = value['operation']
                    if op['name'] == 'backup': self._prefetch(s for s, _ in op['data']['path'])
                    elif op['name'] in ('reserve', 'add_edges'): self._prefetch([op['data']['sid']])
                    self._apply(value['operation']); self.sequence = value['sequence']
                except (ValueError, KeyError, TypeError) as exc:
                    raise ValueError('corrupt MCTS transaction journal') from exc
                good += len(line)
        # A writer can die mid-record; never append after an uncommitted tail.
        if path.stat().st_size != good:
            with path.open('r+b') as f:
                f.truncate(good); f.flush(); os.fsync(f.fileno())

    def _node(self, sid):
        if sid not in self.nodes:
            if self.legacy_root is None: raise KeyError(sid)
            path = self.legacy_root / 'nodes' / sid[:2] / (sid + '.json')
            self.nodes[sid] = read_record(path); self.legacy_reads += 1
        return self.nodes[sid]

    def _apply(self, op):
        name, data = op['name'], op['data']
        if name == 'intern':
            self.nodes.setdefault(data['sid'], {'state': data['state'], 'visits': 0, 'value_sum': 0.,
                'selections': 0, 'edges': {}, 'observations': [], 'policy_batches': 0,
                'last_expansion_selection': -1, 'reservations': {}, 'backed_up': {}})
        elif name == 'reserve':
            value = self._node(data['sid'])
            if data['edge'] not in value['edges']: raise ValueError('unknown MCTS edge')
            if data['uid'] not in value['reservations']: value['selections'] += 1
            value['reservations'][data['uid']] = data['edge']
        elif name == 'add_edges':
            value = self._node(data['sid'])
            if data['observation'] in value['observations']: return
            for edge in data['edges']:
                old = value['edges'].get(edge['id'])
                if old is None: value['edges'][edge['id']] = {**edge, 'n': 0, 'w': 0.}
                else: old['count'] += edge['count']
            value['observations'].append(data['observation']); value['policy_batches'] += 1
            value['last_expansion_selection'] = value['selections']
        elif name == 'backup':
            uid = data['uid']
            existing = self._backup_record(uid)
            if existing is not None and existing != data: raise ValueError('conflicting duplicate backup')
            # Validate the complete transaction before changing any prefix.
            for sid, eid in data['path']:
                self._node(sid)['edges'][eid]
            for sid, eid in data['path']:
                value = self._node(sid)
                if uid not in value['backed_up']:
                    edge = value['edges'][eid]
                    edge['n'] += 1; edge['w'] += data['reward']
                    value['visits'] += 1; value['value_sum'] += data['reward']
                    value['backed_up'][uid] = [eid, data['reward']]
                value['reservations'].pop(uid, None)
            self.backups[uid] = data; self.logged_backups.add(uid)
        else: raise ValueError('unknown MCTS transaction')

    def _mutation(self, name, data):
        if name == 'backup':
            self._prefetch(s for s, _ in data['path'])
            self.backup_record(data['uid'])
        elif name in ('reserve', 'add_edges'):
            self._prefetch([data['sid']])
        with self._lock:
            if self.closed: raise RuntimeError('tree owner closed')
            self._apply({'name': name, 'data': data})
            self.sequence += 1; seq = self.sequence
            self.writer.append(seq, {'name': name, 'data': copy.deepcopy(data)})
        self.writer.wait(seq)

    def intern(self, state):
        sid = fingerprint({k: v for k, v in state.items() if k != 'native'})
        self._prefetch([sid])
        with self._lock:
            if sid in self.nodes:
                seq = self.sequence
            else: seq = None
        if seq is None: self._mutation('intern', {'sid': sid, 'state': state})
        else: self.writer.wait(seq)
        return sid

    def snapshot(self, sid):
        self._prefetch([sid])
        with self._lock:
            value = self._node(sid)
            eligible = [uid for uid in value['reservations'] if uid in self.leases
                        or len(self.recovery_workers) == self.expected_workers]
        for uid in eligible:
            # Per-UID ordering fences lease changes, without holding the global
            # tree lock across an old Ceph lock probe.
            with self._lease_lock(uid):
                if not self.alive(uid):
                    with self._lock: self._node(sid)['reservations'].pop(uid, None)
        with self._lock:
            value = self._node(sid)
            result = copy.deepcopy(value); seq = self.sequence
        self.writer.wait(seq)
        return result

    def snapshots(self, sids):
        sids = list(dict.fromkeys(sids))
        self._prefetch(sids)
        return {sid: self.snapshot(sid) for sid in sids}

    def reserve(self, uid, sid, edge):
        self._mutation('reserve', {'uid': uid, 'sid': sid, 'edge': edge})

    def add_edges(self, sid, edges, *, observation):
        self._mutation('add_edges', {'sid': sid, 'edges': edges, 'observation': observation})

    def select_edge(self, sid, *, excluded=()):
        value = self.snapshot(sid)
        width = 1 + math.isqrt(value['selections'])
        edges = sorted((e for e in value['edges'].values() if e.get('child') not in excluded),
                       key=lambda e: (-e['count'], e['id']))[:width]
        if not edges: return None
        total = sum(e['count'] for e in value['edges'].values())
        virtual = list(value['reservations'].values()); visits = value['visits'] + len(virtual)
        def score(e):
            return (e['w'] / e['n'] if e['n'] else 0.) + self.contract.get('c_puct', 1.5) * (
                e['count'] / max(1, total)) * math.sqrt(visits + 1) / (1 + e['n'] + virtual.count(e['id']))
        return max(edges, key=lambda e: (score(e), e['id']))

    def needs_expansion(self, sid):
        value = self.snapshot(sid)
        return not value['edges'] or (1 + math.isqrt(value['selections']) > len(value['edges'])
            and value['selections'] > value.get('last_expansion_selection', -1))

    def _backup_record(self, uid):
        if uid not in self.backups:
            path = self.legacy_root / 'backups' / (uid + '.json') if uid in self._legacy_backup_ids else None
            self.backups[uid] = read_record(path) if path and path.exists() else None
        return self.backups[uid]

    def backup_record(self, uid):
        with self._lock: known = uid in self.backups
        if not known:
            path = self.legacy_root / 'backups' / (uid + '.json') if uid in self._legacy_backup_ids else None
            value = read_record(path) if path else None
            with self._lock: self.backups.setdefault(uid, value)
        with self._lock:
            result = copy.deepcopy(self._backup_record(uid)); seq = self.sequence
        self.writer.wait(seq)
        return result

    def backup(self, uid, path, reward):
        path = [list(pair) for pair in path]
        if reward not in (0., 1.) or len({s for s, _ in path}) != len(path):
            raise ValueError('invalid completed-rollout backup')
        self._mutation('backup', {'uid': uid, 'path': path, 'reward': reward})

    def register_owner(self, owner, pid):
        import psutil
        with self._lock: self.owners[owner] = (int(pid), psutil.Process(pid).create_time())

    def alive(self, uid):
        import psutil
        with self._lease_lock(uid):
            with self._lock:
                if uid in self.leases:
                    owner = self.leases[uid]
                    if owner is None: return False
                    pid, created = self.owners[owner]
                else:
                    if uid not in self._legacy_rollout_ids: return False
                    if uid in self._legacy_claims: return True
                    pid = None
            if pid is None: return self._legacy_alive(uid)
            try: return psutil.Process(pid).create_time() == created and psutil.Process(pid).is_running()
            except psutil.Error: return False

    def _lease_lock(self, uid):
        with self._lock:
            return self._lease_locks.setdefault(uid, threading.RLock())

    def _legacy_alive(self, uid):
        with (self.legacy_root / 'rollouts' / (uid + '.lock')).open('a+') as handle:
            try: fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError: return True
            return False

    def claim_rollout(self, uid, owner):
        with self._lease_lock(uid):
            with self._lock:
                if owner not in self.owners: raise ValueError('unknown rollout owner')
            if self.alive(uid): raise ValueError('rollout still has a live owner')
            if uid in self._legacy_rollout_ids and uid not in self._legacy_claims:
                handle = (self.legacy_root / 'rollouts' / (uid + '.lock')).open('a+')
                try: fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    handle.close(); raise ValueError('legacy rollout still has a live owner') from None
                self._legacy_claims[uid] = handle
            with self._lock: self.leases[uid] = owner
        return uid

    def release_rollout(self, uid, owner):
        with self._lease_lock(uid):
            with self._lock:
                handle = None
                if self.leases.get(uid) == owner:
                    self.leases[uid] = None
                    handle = self._legacy_claims.pop(uid, None)
            if handle is not None: handle.close()

    def claim_expansion(self, sid, owner):
        with self._lock:
            old = self.expansions.get(sid)
            if old and self._owner_alive(old[0]): return None
            ticket = uuid.uuid4().hex; self.expansions[sid] = (owner, ticket)
            return ticket

    def _owner_alive(self, owner):
        import psutil
        pid, created = self.owners[owner]
        try: return psutil.Process(pid).create_time() == created
        except psutil.Error: return False

    def release_expansion(self, sid, owner, ticket):
        with self._lock:
            if self.expansions.get(sid) == (owner, ticket): self.expansions.pop(sid)

    def recovery_ready(self, owner):
        with self._lock: self.recovery_workers.add(owner)

    def metrics(self):
        with self._lock:
            return {'backend': 'ray_memory_grouped_WAL_v1', 'nodes_in_RAM': len(self.nodes),
                    'legacy_node_reads': self.legacy_reads, 'sequence': self.sequence,
                    'committed_sequence': self.writer.committed, 'WAL_fsyncs': self.writer.fsyncs,
                    'writer_queue': self.writer.queue.qsize(), 'registered_workers': len(self.owners)}

    def identity(self):
        return str(self.root), dict(self.contract)

    def close(self):
        with self._lock:
            if self.closed: return
            self.closed = True
        try: self.writer.close()
        finally:
            self._readers.shutdown(wait=True)
            for handle in self._legacy_claims.values(): handle.close()
            self._legacy_claims.clear()
            self._owner_lock.close()


class ActorHost:
    def __init__(self, root, contract, legacy_root):
        self.core = MemoryMCTSCore(root, contract, legacy_root=legacy_root)

    def call(self, method, args, kwargs):
        if method.startswith('_'): raise ValueError('private store method')
        return getattr(self.core, method)(*args, **kwargs)


class RayLease:
    def __init__(self, store, uid, *, expansion=None):
        self.store, self.uid, self.expansion = store, uid, expansion
        self.handle = True

    def close(self):
        if self.handle is None: return
        if self.expansion:
            self.store._call('release_expansion', self.uid, self.store.owner, self.expansion)
        else: self.store._call('release_rollout', self.uid, self.store.owner)
        self.handle = None


class RayMCTSStore:
    def __init__(self, actor_name, *, namespace='mechet-mcts', address='auto'):
        import ray
        if not ray.is_initialized(): ray.init(address=address, namespace=namespace, log_to_driver=False)
        self.ray = ray; self.actor = ray.get_actor(actor_name, namespace=namespace)
        self.owner = uuid.uuid4().hex
        self.root, self.contract = self._call('identity')
        self.root = Path(self.root)
        self._call('register_owner', self.owner, os.getpid())

    def _call(self, method, *args, **kwargs):
        return self.ray.get(self.actor.call.remote(method, args, kwargs))

    def __getattr__(self, method):
        if method.startswith('_'): raise AttributeError(method)
        return lambda *args, **kwargs: self._call(method, *args, **kwargs)

    def start_rollout(self): return self.resume_rollout(uuid.uuid4().hex)

    def resume_rollout(self, uid):
        self._call('claim_rollout', uid, self.owner)
        return RayLease(self, uid)

    def claim_expansion(self, sid):
        ticket = self._call('claim_expansion', sid, self.owner)
        return RayLease(self, sid, expansion=ticket) if ticket else None

    def recovery_ready(self): self._call('recovery_ready', self.owner)

    def liveness_metrics(self): return self._call('metrics')


def start_actor(root, contract, legacy_root, *, name='single-reaction-tree', namespace='mechet-mcts'):
    import ray
    return ray.remote(num_cpus=1, max_concurrency=64)(ActorHost).options(
        name=name, namespace=namespace, lifetime='detached').remote(str(root), contract,
                                                                  str(legacy_root) if legacy_root else None)
