"""Compatible single-reaction checkpoint recovery without changing planner SHAs."""
from collections import Counter
import fcntl
import json
from pathlib import Path

from scripts.single_reaction_mcts import SingleReactionMCTS
from scripts.shared_route_mcts import RolloutLease, MCTSRouteNetwork
from mechet.shared_electron_search import read_record, SearchJournal, fingerprint, durable_json


CONTRACT_KEYS = ('source_sha', 'adapter_sha', 'revision', 'rank', 'world', 'program_goal',
    'seed', 'max_model_len', 'history_contract', 'shared_cache_namespace',
    'evaluation_unit', 'source_protocol', 'stopping_contract')


def origin_path(output):
    return Path(str(Path(output).resolve()) + '.resume_origin.json')


def validate_origin(value, output):
    if value.get('schema') != 'mcts_resume_origin_v1' or value.get('output') != str(Path(output).resolve()):
        raise ValueError('resume origin output binding mismatch')
    if Path(value['source_output']).resolve() == Path(output).resolve():
        raise ValueError('cyclic resume origin')
    rows = value['source_manifests']
    if len(rows) != 8 or any(r.get('rank') != i or r.get('world') != 8
            or r.get('shard_rows') != 390 for i, r in enumerate(rows)):
        raise ValueError('resume origin full3120 shard contract mismatch')
    keys = [k for k in CONTRACT_KEYS if k != 'rank'] + ['mcts_shared_root', 'tree_store_backend']
    if any(any(r.get(k) != rows[0].get(k) for k in keys) for r in rows):
        raise ValueError('resume origin mixed source contract')


def origin_rank_manifest(origin, rank):
    return {**origin['source_manifests'][rank], 'resume_source': origin['source_output'],
        'mcts_shared_root': origin['store_root'], 'tree_store_backend': 'ray_memory_grouped_WAL_v1',
        'code_sha': origin['runtime_code_sha']}


def check_origin_manifest(actual, expected):
    keys = (*CONTRACT_KEYS, 'resume_source', 'mcts_shared_root', 'tree_store_backend', 'code_sha')
    if any(actual.get(k) != expected.get(k) for k in keys):
        raise ValueError('resume origin conflicts with real manifest contract')


def write_resume_origin(output, source_output, manifests, *, store_root, legacy_root, code_sha):
    """Durable sibling metadata; never create the fresh inference output itself."""
    value = {'schema': 'mcts_resume_origin_v1', 'output': str(Path(output).resolve()),
        'source_output': str(Path(source_output).resolve()), 'source_manifests': manifests,
        'store_root': str(Path(store_root).resolve()),
        'legacy_root': str(Path(legacy_root).resolve()) if legacy_root else None,
        'runtime_code_sha': code_sha}
    validate_origin(value, output)
    path = origin_path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(str(path) + '.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        for rank in range(8):
            manifest = Path(output) / 'tree' / f'rank{rank:02d}' / 'manifest.json'
            if manifest.exists():
                check_origin_manifest(json.loads(manifest.read_text()), origin_rank_manifest(value, rank))
        if path.exists():
            if read_record(path) != value: raise ValueError('conflicting resume origin')
        else:
            durable_json(path, value)
    return path


def rank_manifest(root):
    root = Path(root); path = root / 'manifest.json'
    actual = json.loads(path.read_text()) if path.exists() else None
    companion = origin_path(root.parent.parent) if root.parent.name == 'tree' else None
    if companion is None or not companion.exists():
        if actual is None: raise FileNotFoundError(path)
        return actual
    origin = read_record(companion); validate_origin(origin, root.parent.parent)
    rank = int(root.name.removeprefix('rank'))
    if not 0 <= rank < 8: raise ValueError('resume origin rank mismatch')
    expected = origin_rank_manifest(origin, rank)
    if actual is not None:
        check_origin_manifest(actual, expected)
        return actual
    return {**expected, '__origin_virtual__': True, '__origin_legacy_root': origin['legacy_root']}


def resume_lineage(root, expected=None):
    """Newest-to-oldest compatible evidence; never manufacture a new target."""
    result = []; visited = set()
    while root is not None:
        root = Path(root)
        if root.resolve() in visited: raise ValueError('cyclic resume provenance')
        visited.add(root.resolve())
        manifest = rank_manifest(root)
        if expected is None: expected = manifest
        if any(manifest.get(k) != expected.get(k) for k in CONTRACT_KEYS):
            raise ValueError('resume manifest contract mismatch')
        result.append((root, manifest))
        previous = manifest.get('resume_source')
        root = Path(previous) / 'tree' / f"rank{manifest['rank']:02d}" if previous else None
    return result


class ReadOnlySearchJournal(SearchJournal):
    """Restore source evidence without mkdir, file locks or new empty headers."""
    def __init__(self, root, case_id, manifest):
        self.root = Path(root) / fingerprint(case_id)
        self.contract = {'schema': 'electron_search_journal_v1', 'case_id': case_id, **manifest}
        actual = read_record(self.root / 'contract.json')
        keys = (*CONTRACT_KEYS, 'schema', 'case_id', 'code_sha')
        mismatch = (any(actual.get(k) != self.contract.get(k) for k in keys)
                    if manifest.get('__origin_virtual__') else actual != self.contract)
        if mismatch:
            raise ValueError('search journal contract mismatch')
        self.contract = actual
        paths = sorted(self.root.glob('events/*.json'))
        self.sequence = len(paths)
        if [p.name for p in paths] != [f'{i:09d}.json' for i in range(self.sequence)]:
            raise ValueError('search journal has missing commits')

    def _commit(self, event):
        raise PermissionError('source search journal is read-only')


def case_evidence(lineage, identifier):
    filename = fingerprint(identifier) + '.json'
    for index, (root, manifest) in enumerate(lineage):
        report = root / 'evaluations' / filename
        if report.exists(): return report, None, None
        checkpoint = root / 'mcts' / filename
        if not checkpoint.exists(): continue
        for journal_root, journal_manifest in lineage[index:]:
            directory = journal_root / 'journals' / fingerprint(identifier)
            if not (directory / 'contract.json').exists():
                if list(directory.glob('events/*.json')):
                    raise ValueError('source journal commits have no contract')
                continue
            journal = ReadOnlySearchJournal(journal_root / 'journals', identifier, journal_manifest)
            if journal.sequence: return None, checkpoint, journal
        raise ValueError('saved checkpoint has no initialized source journal')
    raise ValueError('target has no recoverable saved report or checkpoint')


class PolicyEvidence:
    """Offset index of preserved raw receipts; no model regeneration or GT."""
    def __init__(self, root, visited=None):
        self.root = Path(root); self.rows = {}; self.prompts = None
        self.previous = None
        visited = set() if visited is None else set(visited)
        if self.root.resolve() in visited: raise ValueError('cyclic raw evidence provenance')
        visited.add(self.root.resolve())
        manifest = rank_manifest(self.root)
        previous = manifest.get('resume_source')
        if previous:
            self.previous = PolicyEvidence(Path(previous) / 'tree' / f"rank{manifest['rank']:02d}", visited)
        path = self.root / 'raw_generations.jsonl'
        if path.exists():
            with path.open('rb') as f:
                while True:
                    offset = f.tell(); line = f.readline()
                    if not line: break
                    if not line.endswith(b'\n'): break
                    value = json.loads(line)
                    self.rows.setdefault(value['case_id'], []).append(offset)

    def records(self, case_id):
        if self.previous is not None: yield from self.previous.records(case_id)
        path = self.root / 'raw_generations.jsonl'
        if case_id not in self.rows: return
        with path.open('rb') as f:
            for offset in self.rows[case_id]:
                f.seek(offset); record = json.loads(f.readline())
                record['_prompt_lookup'] = lambda key, cid=case_id: self.prompt_ids(cid, key)
                yield record

    def prompt_ids(self, case_id, key):
        if self.prompts is None:
            self.prompts = {}
            path = self.root / 'policy_contexts.jsonl'
            if path.exists():
                with path.open() as f:
                    for line in f:
                        if not line.endswith('\n'): break
                        v = json.loads(line); self.prompts[(v['case_id'], v['context'])] = v['token_ids']
        return self.prompts[(case_id, key)]


def _semantic(proposal):
    from mechet.shared_electron_search import fingerprint
    def strip(x):
        if isinstance(x, dict): return {k: strip(v) for k, v in x.items() if k != 'instruction'}
        if isinstance(x, list): return [strip(v) for v in x]
        return x
    args = proposal['arguments']
    if proposal['name'] in ('import_fragments', 'apply_electron_flow'): args = strip(args)
    return fingerprint([proposal['name'], args])


def _record_batch(record, node, controller):
    from mechet.shared_electron_search import fingerprint, sampling_seed
    explicit = record.get('planning_expansion')
    if explicit is not None: return explicit['batch']
    ids = record['_prompt_lookup'](record['context'])
    room = controller.mcts_store.contract.get('max_model_len', 8192) - len(ids)
    if room < 2: return None
    limits = set(); n = 384
    while True:
        limits.add(min(n, room))
        if n >= room: break
        n *= 2
    for batch in range(node['policy_batches']):
        for limit in limits:
            request = {'context': record['context'], 'prompt_ids': ids, 'n': 16, 'max_tokens': limit,
                'temperature': 1, 'top_p': 1, 'top_k': -1, 'repetition_penalty': 1,
                'presence_penalty': 0, 'frequency_penalty': 0, 'skip_special_tokens': False,
                'logprobs': 0, 'min_tokens': 0,
                'seed': sampling_seed(controller.mcts_store.contract.get('sampling_seed', 17), record['context'], batch)}
            if fingerprint(request) == record['shared_request_key']: return batch
    return None  # generated but no committed planning observation before interruption


def _restore_native_evidence(c, states, records):
    from mechet.shared_electron_search import fingerprint
    from mechet.dynamic_electron_tree import Proposal
    from mechet.reverse_route_network import RouteNetwork
    native = c.trees[c.target]
    by_parent = {}
    for edge in native.edges:
        by_parent.setdefault(edge['parent_context'], Counter())[fingerprint(edge['action'])] += 1
    contexts = {v['state']['active']: (sid, v) for sid, v in states.items() if not v['state']['solved']}
    for record in (() if records is None else records):
        parent = record['context']
        if parent not in contexts: continue
        counts = Counter(fingerprint(p) for p in record['generations'])
        known = by_parent.setdefault(parent, Counter())
        if all(known[k] >= n for k, n in counts.items()):
            known.subtract(counts); continue
        sid, value = contexts[parent]
        batch = _record_batch(record, value, c)
        if batch is None or fingerprint([sid, batch]) not in value['observations']: continue
        c.contexts[parent].stats.batches = batch
        RouteNetwork.incorporate(c, parent, [Proposal(**p) for p in record['generations']])
    # Restore the per-controller applied set from its OWN native edges, not all
    # global alternatives. Otherwise selection duplicates previously counted
    # work, or eagerly imports other targets' unseen terminal candidates.
    own = {}
    for edge in native.edges:
        action = edge['action']
        if action.get('truncated') or any(word in str(edge.get('error', '')).upper()
                for word in ('RESOURCE', 'MEMORY', 'ALLOCATION', 'TIMEOUT', 'TEMPORARY')): continue
        own.setdefault(edge['parent_context'], set()).add(_semantic(action))
    for sid, value in states.items():
        if value['state']['solved']: continue
        parent = value['state']['active']
        for edge in value['edges'].values():
            if _semantic(edge['proposal']) in own.get(parent, set()):
                c.applied.add((sid, edge['id']))
            elif records is None and edge.get('child') in c.seen_states:
                # Standalone graph recovery can replay accepted path edges;
                # deployment recovery supplies the full raw proposal batches.
                c._ensure_transition(sid, edge)


class ResumableSingleReactionMCTS(SingleReactionMCTS):
    def next_context(self):
        count = self.completed_rollouts + self.censored_rollouts
        if count >= self.program_goal:
            self.has_pending = False; self.stop_reason = 'MCTS_TRAJECTORY_ATTEMPT_GOAL_REACHED'
            self.checkpoint(); return None
        if not self.has_pending: return None
        if self.selected is not None: return self.contexts[self.selected['context']]
        concurrency = min(8, self.program_goal - count)
        if len(self.rollouts) < concurrency:
            self.rollouts.append({'lease': self.mcts_store.start_rollout(),
                                  'sid': self.mcts_root, 'path': []})
            # UID admission must precede even a wholly cached traversal/backup.
            self.checkpoint()
        self.rollout_concurrency = len(self.rollouts)
        try: return MCTSRouteNetwork.next_context(self)
        finally: self.rollout_concurrency = concurrency


def recover_shard(rows, source_root, output_root, manifest, *, planning, cache):
    """Read immutable old evidence and publish a new continuation journal."""
    from mechet.shared_electron_search import SearchJournal, fingerprint
    from scripts.infer_dynamic_electron_trees import dump
    source_root, output_root = Path(source_root), Path(output_root)
    lineage = resume_lineage(source_root, manifest)
    controllers, journals, reports = {}, {}, []
    stats = {'reused_finished_reports': 0, 'restored_controllers': 0,
             'restored_pending_rollouts': 0, 'restored_completed_rollouts': 0}
    try:
        evidence = PolicyEvidence(source_root)
        for row in rows:
            identifier = row['id']; filename = fingerprint(identifier) + '.json'
            report_path, checkpoint_path, source_journal = case_evidence(lineage, identifier)
            if report_path is not None:
                report = json.loads(report_path.read_text())
                if report['id'] != identifier:
                    raise ValueError('resume report identity contract mismatch')
                if any(word in report.get('stop_reason', '') for word in ('MEMORY', 'RESOURCE', 'CONTEXT_BUDGET')):
                    raise ValueError('resource-limited report requires explicit recovery policy')
                if report.get('tree_artifact'):
                    report['tree_artifact'] = str((report_path.parent.parent / report['tree_artifact']).resolve())
                report['resumed_from'] = str(report_path)
                dump(output_root / 'evaluations' / filename, report)
                reports.append(report); stats['reused_finished_reports'] += 1
                continue
            c = restore_single_controller(source_journal, checkpoint_path=checkpoint_path,
                mcts_store=planning, shared_store=cache, output_checkpoint=output_root / 'mcts' / filename,
                policy_records=evidence.records(identifier))
            if c.case_id != identifier:
                c.close(); raise ValueError('resume controller identity contract mismatch')
            controllers[identifier] = c
            fresh = SearchJournal(output_root / 'journals', identifier, manifest)
            fresh.initialize(c); journals[identifier] = fresh
            stats['restored_controllers'] += 1
            stats['restored_pending_rollouts'] += len(c.rollouts)
            stats['restored_completed_rollouts'] += c.completed_rollouts
            if stats['restored_controllers'] % 20 == 0:
                print({'phase': 'restoring-MCTS', 'rank': manifest.get('rank'), **stats}, flush=True)
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
                              shared_store=None, output_checkpoint=None, policy_records=None):
    saved = read_record(checkpoint_path)
    if saved.get('planning_contract') != mcts_store.contract:
        raise ValueError('MCTS checkpoint contract mismatch')
    if saved['program_goal'] != mcts_store.contract.get('program_goal', saved['program_goal']):
        raise ValueError('MCTS program goal contract mismatch')
    if any(mcts_store.alive(r['uid']) for r in saved['pending_rollouts']):
        raise ValueError('MCTS checkpoint still has a live rollout owner')
    base = journal.restore(stock=set(), shared_store=shared_store)
    c = ResumableSingleReactionMCTS(base.case_id, base.target, mcts_store=mcts_store,
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
    batch_snapshot = (mcts_store.snapshots if hasattr(mcts_store, 'snapshots') else
                      lambda ids: {sid: mcts_store.snapshot(sid) for sid in ids})
    states = batch_snapshot(c.seen_states)
    for value in states.values():
        c._hydrate(value['state'])
    _restore_native_evidence(c, states, policy_records)
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
