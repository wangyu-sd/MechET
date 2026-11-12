"""Complete public route events with shared observations and scoped results.

This module uses only the standard library. It neither executes chemistry nor
changes sampled actions/probabilities. Tables preserve text verbatim; references
remove duplicate snapshots without deleting decision events.
"""
from copy import deepcopy
import json
import re


EVENT_ROUTE_HISTORY = 'route_event_history_feedback_v3'
SCOPED_ROUTE_HISTORY = 'route_scoped_history_feedback_v4'
NATIVE_ROUTE_HISTORY = 'route_native_history_feedback_v5'
NATIVE_LAST_ROUTE_HISTORY = 'route_native_last_history_feedback_v6'
PASSIVE_ROUTE_HISTORY = 'route_passive_history_feedback_v7'
NATIVE_SYSTEM_HEADER = '\nNATIVE CURRENT-DECISION SYSTEM INSTRUCTIONS\n'
HISTORY_HEADER = '\nCOMPLETE ROUTE EVENT HISTORY (PAST EVENTS)\n'
ACTIVE_HEADER = '\nCURRENT REACTION (AUTHORITATIVE FOR NEXT ACTION)\n'
LOCAL_HEADER = '\nCURRENT LOCAL OBSERVATION\n'
_CAPSULE = 'TRAJECTORY HISTORY (executor-owned; past actions only)\n'
_DERIVED_CAPSULE = re.compile(re.escape(_CAPSULE)+
    r'accepted_actions: \d+\naccepted_action_types: (?:START|(?:import_fragments|apply_electron_flow|finish_trace)(?:>(?:import_fragments|apply_electron_flow|finish_trace))*)\n'
    r'import_batches_committed: \d+\nimported_fragments_committed: \d+\nelectron_events_committed: \d+\n'
    r'last_action: (?:START|import_fragments|apply_electron_flow|finish_trace)\nlast_result: (?:START|PASS)'
    r'\nChoose the single next retrosynthetic action\.')
_TABLES = ('state_texts', 'observation_texts', 'observations', 'output_texts')
_HANDLE = re.compile(r'(?<![A-Za-z0-9_])([AB])(\d+)(?![A-Za-z0-9_])')
_OLD_SCOPED_HANDLE = re.compile(r'(?<![A-Za-z0-9_])past_(\d+)\.(atom|bond)(\d+)(?![A-Za-z0-9_])')


def _history_handle(kind, digits, observation, opaque):
    # Digit-wise encoding is injective even for invalid spellings A1/A01/A001.
    letters=lambda value: ''.join(chr(ord('a')+int(d)) for d in str(value))
    scope=letters(observation) if opaque else str(observation)
    label=('atom' if kind=='A' else 'bond')
    return 'past_'+scope+'.'+label+('_'+letters(digits) if opaque else digits)


def _past_text(text, observation, *, opaque=False):
    """An old local handle is never a handle in the final active inventory."""
    replace=lambda m: _history_handle(m[1],m[2],observation,opaque)
    if opaque:
        text=_OLD_SCOPED_HANDLE.sub(lambda m:_history_handle(
            'A' if m[2]=='atom' else 'B',m[3],m[1],True),text)
    lines=[]
    for line in text.splitlines(keepends=True):
        if line.startswith(('TARGET PRODUCT SMILES:','CURRENT STATE SMILES:','ROUTE_CYCLE:')):
            lines.append(line)
        elif line.startswith('ANNOTATED CURRENT STATE:'):
            # Only angle-bracket inventory labels are identifiers; B12 can be
            # a real boron atom with ring closures in the underlying SMILES.
            lines.append(re.sub(r'<(A)(\d+)>',lambda m:'<'+replace(m)+'>',line))
        else:
            lines.append(_HANDLE.sub(replace,line))
    scoped=''.join(lines)
    return scoped.replace('<', '\\u003c').replace('>', '\\u003e')


def _past_value(value, observation, *, opaque=False):
    if isinstance(value, str):
        return _past_text(value, observation,opaque=opaque)
    if isinstance(value, list):
        return [_past_value(v, observation,opaque=opaque) for v in value]
    if isinstance(value, dict):
        items=[]
        for k,v in value.items():
            key=_past_text(k,observation,opaque=opaque)
            if k in ('smiles','target_smiles','current_state','initial_state','final_state') and isinstance(v,str):
                # Axx is not an element/ring token; Bxx can be. Keep valid
                # chemical B/ring notation rather than treating it as a bond ID.
                text=re.sub(r'(?<![A-Za-z0-9_])(A)(\d+)(?![A-Za-z0-9_])',
                    lambda m:_history_handle('A',m[2],observation,opaque),v)
                if opaque:
                    text=_OLD_SCOPED_HANDLE.sub(lambda m:_history_handle(
                        'A' if m[2]=='atom' else 'B',m[3],m[1],True),text)
                rendered=text.replace('<','\\u003c').replace('>','\\u003e')
            else:
                rendered=_past_value(v,observation,opaque=opaque)
            items.append((key,rendered))
        if len({k for k,_ in items})!=len(items):
            return {'historical_mapping_items':[[k,v] for k,v in items]}
        return dict(items)
    return value


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

    def latest_feedback(self, reaction_id, *, scope_aliases=False,opaque_handles=False):
        for event in reversed(self.data['events']):
            if event['reaction_id'] == reaction_id and event['type'] == 'attempt':
                return dict(accepted=event['feedback']['accepted'],
                    error=(_past_text(event['feedback']['error'],event['observation_ref'],opaque=opaque_handles)
                           if scope_aliases else event['feedback']['error']),
                    state_changed=event['feedback']['state_changed'],
                    resulting_state_ref=event['after_state_ref'])
        return None

    def model_view(self, *, parse_calls=None,passive=False,native_system=None):
        """Scoped semantic display; the verbatim ledger and receipts stay intact.

        The same raw response can occur under different local inventories, so
        display actions are keyed by BOTH observation and raw-output reference.
        Raw tool delimiters never become apparent current assistant calls.
        """
        result = dict(schema_version=3, model_view_version=7 if passive else 4,
            handle_scope=('Alphabetic historical atom/bond labels belong to their past observations; '
                          'only the final native inventory supplies current Axx/Bxx handles' if passive else
                         'past_N.atomXX/bondXX belong only to historical observation N; '
                         'only the final native inventory provides current Axx/Bxx handles'),
            state_texts=list(self.data['state_texts']), observation_texts=[],
            observations=[], past_actions=[], events=[])
        text_index, action_index = {}, {}
        for ref, observation in enumerate(self.data['observations']):
            frames = []
            for message in observation:
                if (passive and message['role']=='system' and native_system is not None
                        and ''.join(self.data['observation_texts'][p] for p in message['parts'])==native_system):
                    continue  # The identical policy remains in the current native SYSTEM.
                parts = []
                for part in message['parts']:
                    original=self.data['observation_texts'][part]
                    if passive and _DERIVED_CAPSULE.fullmatch(original):
                        continue  # Derived counts/actions remain reconstructable from every event.
                    text = _past_text(original, ref,opaque=passive)
                    if text not in text_index:
                        text_index[text] = len(result['observation_texts'])
                        result['observation_texts'].append(text)
                    parts.append(text_index[text])
                frames.append(dict(role=message['role'],parts=parts))
            result['observations'].append(frames)
        for original in self.data['events']:
            event = deepcopy(original)
            if event['type']=='attempt':
                ref, output = event['observation_ref'],event.pop('output_ref')
                key = (ref,output)
                if key not in action_index:
                    raw = self.data['output_texts'][output]
                    try:
                        calls = parse_calls(raw) if parse_calls is not None else []
                    except Exception:
                        calls = []
                    if calls:
                        action = dict(calls=[dict(name=_past_text(c.name,ref,opaque=passive),
                            arguments=_past_value(c.arguments,ref,opaque=passive)) for c in calls])
                        other = re.sub(r'<tool_call>.*?</tool_call>', '', raw,
                                       flags=re.DOTALL|re.IGNORECASE)
                        other = other.replace('<|im_end|>','').strip()
                        if other:
                            action['other_response_text']=_past_text(other,ref,opaque=passive)
                    else:
                        action = dict(unparsed_response=_past_text(raw,ref,opaque=passive))
                    action_index[key] = len(result['past_actions'])
                    result['past_actions'].append(action)
                event['action_ref'] = action_index[key]
                event['feedback'] = _past_value(event['feedback'],ref,opaque=passive)
                if passive and event['feedback'].get('instruction_at_that_time') in (
                        'Continue from the executor-owned state.',
                        'State was not committed. Correct the error and choose another action.'):
                    event['feedback'].pop('instruction_at_that_time')
            result['events'].append(event)
        return result

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
