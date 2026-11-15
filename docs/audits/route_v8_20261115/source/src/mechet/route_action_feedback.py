"""Current-state action constraints; raw sampling probabilities stay untouched."""
from copy import deepcopy
import json
import re

ACTION_HEADER='\nCURRENT EXECUTION CONSTRAINTS\n'
NATIVE_LOCAL_HEADER='\nNATIVE CURRENT-DECISION OBSERVATION\n'


def semantic_action(raw):
    """Ignore explanatory prose, never electron-flow ordering or chemical fields."""
    from .agent_inference import parse_tool_calls
    try: calls=parse_tool_calls(raw)
    except Exception:return None
    if len(calls)!=1:return None
    call=calls[0];args=deepcopy(call.arguments)
    if call.name=='apply_electron_flow':
        for field in ('electron_flow','bond_order_changes','charge_changes'):
            if not args.get(field):args[field]=[]
            items=args.get(field)
            for item in items if isinstance(items,list) else []:
                if isinstance(item,dict):item.pop('instruction',None)
                if (field=='bond_order_changes' and isinstance(item,dict)
                        and isinstance(item.get('atoms'),list)
                        and all(isinstance(a,str) for a in item['atoms'])):
                    item['atoms']=sorted(item['atoms'])
        moves=args.get('electron_flow')
        for move in moves if isinstance(moves,list) else []:
            if not isinstance(move,dict):continue
            for field in ('source','destination'):
                value=move.get(field)
                if not isinstance(value,str):continue
                match=re.fullmatch(r'(the bond (?:to form )?between atoms |a radical pair on atoms )([AB]\d+) and ([AB]\d+)',value)
                if match:
                    left,right=sorted(match.group(2,3))
                    move[field]=match[1]+left+' and '+right
    action=dict(name=call.name,arguments=args)
    return json.dumps(action,sort_keys=True,separators=(',',':')),action


def state_rejections(ep):
    """Derive memory from receipts: no second mutable cache to corrupt on restore."""
    rid=ep.uid+'/r'+str(len(ep.reactions));failures={}
    for record in ep.records:
        if (record.get('accepted') is not False or record.get('reaction_id')!=rid
                or record.get('state_before')!=ep.node.state or not record.get('error')):
            continue
        parsed=semantic_action(record.get('raw',''))
        if parsed is None:continue
        key,action=parsed
        previous=failures.get(key)
        failures[key]=dict(action=action,error=record['error'],count=1+(previous['count'] if previous else 0))
    return failures


def previous_rejection(ep,receipt):
    parsed=semantic_action(receipt.get('raw',''))
    return state_rejections(ep).get(parsed[0]) if parsed is not None else None


def decision_control(ep):
    from .natural_language_electron_flow import build_inventory
    inventory=build_inventory(ep.node.state)
    bonds=[]
    for pair in inventory.maps_to_bond:
        left,right=sorted(inventory.map_to_atom[a] for a in pair)
        bonds.append('the bond between atoms '+left+' and '+right)
    reason=ep._finish_guard_error(ep.node)
    return dict(phase='new_reaction' if not ep.node.actions else 'ongoing_reaction',
        finish_available=not bool(reason),finish_blocked_reason=reason,
        source_bond_candidates=sorted(bonds),
        rejected_actions_current_state=list(state_rejections(ep).values()),
        instruction=('Use only the current atom inventory. Historical names are not current names. '
            'An existing bond used as an electron source must be one of source_bond_candidates. '
            'A rejected action did not change this state; choose a different chemical action, '
            'not a new explanation of the same action. finish_trace is available only when '
            'finish_available is true; otherwise continue import/electron-flow reasoning.'))


def tools_for_episode(ep):
    from scripts.build_natural_language_event_sft import TOOLS
    allowed=not bool(ep._finish_guard_error(ep.node))
    return deepcopy([t for t in TOOLS if allowed or t['function']['name']!='finish_trace'])


def tools_for_request(request):
    from scripts.build_natural_language_event_sft import TOOLS
    tools=getattr(request,'tools',None)
    if tools is None:return TOOLS
    if not isinstance(tools,list) or not tools:raise ValueError('INVALID_REQUEST_TOOLS')
    names=[t.get('function',{}).get('name') for t in tools]
    if len(set(names))!=len(names) or any(n not in {'import_fragments','apply_electron_flow','finish_trace'} for n in names):
        raise ValueError('INVALID_REQUEST_TOOLS')
    return tools
