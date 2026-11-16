"""SFT-native current chemistry, with an explicit failure-only tool extension."""
from copy import deepcopy
import json

NATIVE_SFT_ROUTE = 'route_sft_aligned_feedback_v9'


def safe_feedback(value):
    """Quoted failure evidence must not become real chat/tool boundaries."""
    if isinstance(value, str):
        return value.replace('<', '\\u003c').replace('>', '\\u003e')
    if isinstance(value, list):
        return [safe_feedback(v) for v in value]
    if isinstance(value, dict):
        return {safe_feedback(k):safe_feedback(v) for k,v in value.items()}
    return value


def native_messages(episode, messages):
    if (len(messages)!=2 or messages[0]['role']!='system' or messages[1]['role']!='user'):
        raise ValueError('SFT_NATIVE_MESSAGE_SHAPE_REQUIRED')
    if not episode.records:
        return messages
    last = episode.records[-1]
    rid = episode.uid+'/r'+str(len(episode.reactions))
    if (last.get('accepted') is not False or last.get('reaction_id')!=rid or
            last.get('state_before')!=episode.node.state or not last.get('error')):
        return messages
    from .agent_inference import parse_tool_calls
    raw = last.get('raw', '')
    try:
        calls = parse_tool_calls(raw)
    except Exception:
        calls = []
    if len(calls)==1:
        call = calls[0]
        assistant = dict(role='assistant', content='', tool_calls=[dict(
            id=last['request_id'], type='function',
            function=dict(name=call.name, arguments=deepcopy(call.arguments)))])
    else:
        # A malformed output is evidence, not an invented executable tool call.
        assistant = dict(role='assistant', content=raw)
    error = last['error']
    result = dict(accepted=False, error_family=error.split(':',1)[0],
                  error=error[:256], error_detail_truncated=len(error)>256,
                  state_changed=False)
    tool = dict(role='tool', tool_call_id=last['request_id'],
                content=json.dumps(result, ensure_ascii=False, separators=(',', ':')))
    if len(calls)==1:
        tool['name']=calls[0].name
    return [messages[0], safe_feedback(assistant), safe_feedback(tool), messages[1]]


def restore_native_episode(cls, snapshot, *, stock=None, chemistry=None, error_prefix='SFT_HISTORY'):
    """Rerender all past requests; native projection must not weaken replay."""
    try:
        original = deepcopy(snapshot)
        ep = cls(original['cfg'], original['public_root'],
                 stock if stock is not None else set(original['stock']),
                 original['uid'], original['seed'], chemistry=chemistry)
        for record in original['records']:
            request = ep.next_decision()
            if request is None or request.messages!=record.get('messages'):
                raise ValueError(error_prefix+'_OBSERVATION_MISMATCH')
            ep.accept(deepcopy(record))
        if (original['records'] or original.get('node') is not None or
                original.get('status') is not None or
                original.get('event_history', {}).get('events')):
            ep.next_decision()
        if ep.snapshot(include_stock='stock' in original)!=original:
            raise ValueError(error_prefix+'_SNAPSHOT_MISMATCH')
        return ep
    except (ValueError, KeyError, TypeError) as error:
        raise ValueError(error_prefix+'_REPLAY_FAILED:'+type(error).__name__) from None
