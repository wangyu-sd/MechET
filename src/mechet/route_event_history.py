"""Complete public route events with shared observations and scoped results.

This module uses only the standard library. It neither executes chemistry nor
changes sampled actions/probabilities. Tables preserve text verbatim; references
remove duplicate snapshots without deleting decision events.
"""
from copy import deepcopy
import json


EVENT_ROUTE_HISTORY = 'route_event_history_feedback_v3'
HISTORY_HEADER = '\nCOMPLETE ROUTE EVENT HISTORY (PAST EVENTS)\n'
ACTIVE_HEADER = '\nCURRENT REACTION (AUTHORITATIVE FOR NEXT ACTION)\n'
LOCAL_HEADER = '\nCURRENT LOCAL OBSERVATION\n'
_CAPSULE = 'TRAJECTORY HISTORY (executor-owned; past actions only)\n'
_TABLES = ('state_texts', 'observation_texts', 'observations', 'output_texts')


def public_json(value):
    """Keep raw chat markers in JSON values without injecting real chat turns."""
    return json.dumps(value, ensure_ascii=False, separators=(',', ':')).replace('<|', '\\u003c|')


class RouteEventHistory:
    def __init__(self, snapshot=None):
        self.data = (dict(schema_version=3, events=[], **{k: [] for k in _TABLES})
                     if snapshot is None else deepcopy(snapshot))
        self._validate()
        self.indexes = {name: {self._key(v): i for i, v in enumerate(self.data[name])}
                        for name in _TABLES}
        self.started = {e['reaction_id'] for e in self.data['events']
                        if e['type'] == 'reaction_started'}

    @staticmethod
    def _key(value):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))

    def _intern(self, name, value):
        key = self._key(value)
        if key not in self.indexes[name]:
            self.indexes[name][key] = len(self.data[name])
            self.data[name].append(deepcopy(value))
        return self.indexes[name][key]

    def intern_observation(self, messages):
        frames = []
        for message in messages:
            # Split the derived Stage-II capsule from the graph inventory so a
            # changed control-flow count need not duplicate the same inventory.
            content = message['content']
            if _CAPSULE in content:
                before, after = content.split(_CAPSULE, 1)
                pieces = [before, _CAPSULE + after]
            else:
                pieces = [content]
            frames.append(dict(role=message['role'], parts=[
                self._intern('observation_texts', text) for text in pieces]))
        return self._intern('observations', frames)

    def observation(self, ref):
        self._reference('observations', ref)
        return [dict(role=m['role'], content=''.join(
            self.data['observation_texts'][i] for i in m['parts']))
            for m in self.data['observations'][ref]]

    def start(self, reaction_id, obligation, state):
        if reaction_id in self.started:
            raise ValueError('DUPLICATE_REACTION_START')
        self.started.add(reaction_id)
        self.data['events'].append(dict(type='reaction_started', reaction_id=reaction_id,
            occurrence=obligation['occurrence'], target=obligation['public_product'],
            ancestors=list(obligation['ancestors']), level=obligation['level'],
            initial_state_ref=self._intern('state_texts', state)))

    def record(self, record, observation, before, after, *, reaction_committed):
        if record['reaction_id'] not in self.started:
            raise ValueError('HISTORY_REACTION_NOT_STARTED')
        feedback = self.scoped_feedback(record['feedback'])
        self.data['events'].append(dict(type='attempt', request_id=record['request_id'],
            reaction_id=record['reaction_id'], occurrence=record['occurrence'],
            observation_ref=self.intern_observation(observation),
            before_state_ref=self._intern('state_texts', before),
            after_state_ref=self._intern('state_texts', after),
            output_ref=self._intern('output_texts', record.get('raw', '')),
            feedback=feedback, reaction_committed=bool(reaction_committed)))

    @staticmethod
    def scoped_feedback(original):
        feedback = deepcopy(original)
        # These are historical results in the ORIGINAL reaction's scope. The
        # next active state is provided separately after the whole event log.
        feedback.pop('current_state', None)
        instruction = feedback.pop('instruction', '')
        if instruction:
            feedback['instruction_at_that_time'] = instruction
        return feedback

    def latest_feedback(self, reaction_id):
        for event in reversed(self.data['events']):
            if event['reaction_id'] == reaction_id and event['type'] == 'attempt':
                return dict(accepted=event['feedback']['accepted'],
                    error=event['feedback']['error'],
                    state_changed=event['feedback']['state_changed'],
                    resulting_state_ref=event['after_state_ref'])
        return None

    def snapshot(self):
        return deepcopy(self.data)

    def _reference(self, table, value):
        if type(value) is not int or not 0 <= value < len(self.data[table]):
            raise ValueError('HISTORY_REFERENCE_INVALID:' + table)

    def _validate(self):
        if (not isinstance(self.data, dict) or self.data.get('schema_version') != 3
                or set(self.data) != {'schema_version', 'events', *_TABLES}
                or any(not isinstance(self.data[k], list) for k in ('events', *_TABLES))):
            raise ValueError('HISTORY_SCHEMA_INVALID')
        for name in ('state_texts', 'observation_texts', 'output_texts'):
            if any(not isinstance(v, str) for v in self.data[name]):
                raise ValueError('HISTORY_SCHEMA_TEXT_REQUIRED')
        for observation in self.data['observations']:
            if not isinstance(observation, list):
                raise ValueError('HISTORY_SCHEMA_OBSERVATION_INVALID')
            for message in observation:
                if (not isinstance(message, dict) or set(message) != {'role', 'parts'}
                        or not isinstance(message['role'], str)
                        or not isinstance(message['parts'], list)):
                    raise ValueError('HISTORY_SCHEMA_MESSAGE_INVALID')
                for ref in message['parts']:
                    self._reference('observation_texts', ref)
        started = set()
        active = None
        committed = False
        for event in self.data['events']:
            if not isinstance(event, dict) or not isinstance(event.get('reaction_id'), str):
                raise ValueError('HISTORY_SCHEMA_EVENT_INVALID')
            if event.get('type') == 'reaction_started':
                if event['reaction_id'] in started or (active is not None and not committed):
                    raise ValueError('HISTORY_SCHEMA_DUPLICATE_REACTION')
                if (not all(isinstance(event.get(k), str) for k in ('target', 'occurrence'))
                        or type(event.get('level')) is not int or event['level'] < 0
                        or not isinstance(event.get('ancestors'), list)
                        or any(not isinstance(v, str) for v in event['ancestors'])):
                    raise ValueError('HISTORY_SCHEMA_REACTION_INVALID')
                started.add(event['reaction_id'])
                active, committed = event['reaction_id'], False
                self._reference('state_texts', event.get('initial_state_ref'))
            elif event.get('type') == 'attempt' and event['reaction_id'] == active and not committed:
                for field, table in (('observation_ref', 'observations'),
                                     ('before_state_ref', 'state_texts'),
                                     ('after_state_ref', 'state_texts'),
                                     ('output_ref', 'output_texts')):
                    self._reference(table, event.get(field))
                feedback = event.get('feedback')
                if (not isinstance(feedback, dict)
                        or type(feedback.get('accepted')) is not bool
                        or type(feedback.get('state_changed')) is not bool
                        or not isinstance(feedback.get('error'), str)
                        or not isinstance(event.get('request_id'), str)
                        or not isinstance(event.get('occurrence'), str)
                        or type(event.get('reaction_committed')) is not bool):
                    raise ValueError('HISTORY_SCHEMA_FEEDBACK_INVALID')
                committed = event['reaction_committed']
                if committed and not feedback['accepted']:
                    raise ValueError('HISTORY_SCHEMA_REJECTED_COMMIT')
            else:
                raise ValueError('HISTORY_SCHEMA_EVENT_ORDER_INVALID')
