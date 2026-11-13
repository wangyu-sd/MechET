"""Frozen TRAIN experience in real route observations, with replayable provenance.

Only typed advice is transferred across states. Stored atom handles and tool calls
are never rendered as current executable actions. Full route history is retained.
"""
from copy import deepcopy
import hashlib
import json

from .external_route_rl import TargetOnlyRouteEpisode
from .external_route_validation import ValidationRouteEpisode
from .route_event_history import ACTION_ROUTE_HISTORY, public_json
from .route_experience_memory import ExperienceMemory

MEMORY_HEADER = '\nVERIFIED TRAIN EXPERIENCE (ADVICE, NOT CURRENT ACTIONS)\n'
MEMORY_END = '\nEND VERIFIED TRAIN EXPERIENCE\n'
STRATEGIES = {
    'inspect_current_source_bonds': 'Inspect the current source-bond inventory before proposing electron flow.',
    'continue_before_finish': 'Check current productive-change eligibility before proposing FINISH.',
    'avoid_route_ancestors': 'Check current branch ancestors before committing a precursor transformation.',
}


def memory_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def strategy_for_error(error):
    if error.startswith('NONPRODUCTIVE_FINISH'):
        return 'continue_before_finish'
    if error.startswith('ROUTE_CYCLE'):
        return 'avoid_route_ancestors'
    if 'source bond' in error.lower() or ('KeyError' in error and 'bond' in error.lower()):
        return 'inspect_current_source_bonds'
    return None


def execution_scope(ep, *, binding=None):
    if ep.node is None or ep.current is None or ep.event_history is None:
        raise ValueError('MEMORY_ACTIVE_EVENT_EPISODE_REQUIRED')
    binding = binding or ep.cfg['route_memory']
    return dict(state=ep.node.state, target=ep.current['product'], route_root=ep.root,
        next_map=ep.node.next_map, imported=deepcopy(ep.node.imported),
        visited=sorted(ep.node.visited), accepted_action_types=[a['name'] for a in ep.node.actions],
        ancestors=list(ep.current['ancestors']), stock_version=binding['stock_version']+'/'+memory_digest(sorted(ep.stock)),
        executor_version=binding['executor_version'], occurrence=ep.current['occurrence'],
        event_history_sha256=memory_digest(ep.event_history.snapshot()))


def strip_memory(messages):
    result = deepcopy(messages)
    text = result[0]['content']
    if not text.startswith(MEMORY_HEADER) or MEMORY_END not in text:
        raise ValueError('MEMORY_OBSERVATION_MISSING')
    payload, remainder = text[len(MEMORY_HEADER):].split(MEMORY_END, 1)
    json.loads(payload)
    result[0]['content'] = remainder
    return result


class MemoryEpisodeMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = deepcopy(self.cfg)
        self._load_memory()

    def _load_memory(self):
        b = self.cfg.get('route_memory', {})
        if (self.cfg.get('public_observation_contract') != ACTION_ROUTE_HISTORY
                or not all(isinstance(b.get(k), str) and b[k] for k in ('sha256', 'stock_version', 'executor_version'))
                or memory_digest(b.get('snapshot')) != b['sha256']):
            raise ValueError('MEMORY_BINDING_OR_OBSERVATION_MISMATCH')
        self._memory_store = ExperienceMemory(b['snapshot'])

    def _messages(self):
        messages = super()._messages()
        scope = execution_scope(self)
        exact = self._memory_store.retrieve(scope)
        latest = self.event_history.latest_feedback(self.uid+'/r'+str(len(self.reactions)))
        wanted = strategy_for_error(latest['error']) if latest else None
        transferable = [e for e in self._memory_store.entries.values()
                        if wanted and e['content'].get('strategy_id') == wanted
                        and e['scope']['executor_version'] == scope['executor_version']
                        and e['scope']['stock_version'] == scope['stock_version']]
        selected = {e['id']: e for e in [*exact, *transferable] if e['content'].get('strategy_id') in STRATEGIES}
        advice = [dict(id=e['id'], strategy_id=e['content']['strategy_id'],
                       advice=STRATEGIES[e['content']['strategy_id']], utility=e['utility'], signal=e['signal'],
                       applicability='exact' if e in exact else 'generic_error_recovery_advice')
                  for e in sorted(selected.values(), key=lambda e: (-e['utility'], e['id']))[:5]]
        payload = dict(memory_sha256=self.cfg['route_memory']['sha256'],
                       scope_sha256=memory_digest(scope), advice=advice)
        messages[0]['content'] = MEMORY_HEADER+public_json(payload)+MEMORY_END+messages[0]['content']
        return messages

    def snapshot(self, *, include_stock=True):
        result = super().snapshot(include_stock=include_stock)
        result.pop('_memory_store', None)
        return result

    def _validate_restored_event_history(self):
        original = self.records
        try:
            self.records = [dict(r, messages=strip_memory(r['messages'])) for r in original]
            super()._validate_restored_event_history()
        finally:
            self.records = original

    @classmethod
    def restore(cls, snapshot, *, stock=None, chemistry=None):
        result = super().restore(snapshot, stock=stock, chemistry=chemistry)
        result._load_memory()
        # Re-render every saved decision against its true historical scope.
        # Matching only the final graph would miss a forged memory/prompt prefix.
        replay = cls(result.cfg, result.public_root, result.stock, result.uid, result.seed, chemistry=chemistry)
        for record in result.records:
            replay.accept(deepcopy(record))
        if snapshot.get('current') is not None:
            replay.next_decision()
        actual = replay.snapshot(include_stock='stock' in snapshot)
        if actual != snapshot:
            raise ValueError('MEMORY_REPLAY_SNAPSHOT_MISMATCH')
        return result


class MemoryTargetOnlyRouteEpisode(MemoryEpisodeMixin, TargetOnlyRouteEpisode):
    pass


class MemoryValidationRouteEpisode(MemoryEpisodeMixin, ValidationRouteEpisode):
    pass
