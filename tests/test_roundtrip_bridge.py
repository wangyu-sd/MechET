import copy
import pytest


def example():
    return {'target':'CO','initial_state':'[CH3:1][OH:2]','final_state':'[CH3:1][Cl:3].[OH-:2]',
      'terminal':True,'steps':[
       {'name':'import_fragments','accepted':True,'state_before':'[CH3:1][OH:2]',
        'state_after':'[CH3:1][OH:2].[Cl-:3]'},
       {'name':'apply_electron_flow','accepted':True,'state_before':'[CH3:1][OH:2].[Cl-:3]',
        'state_after':'[CH3:1][Cl:3].[OH-:2]'},
       {'name':'finish_trace','accepted':True,'state_before':'[CH3:1][Cl:3].[OH-:2]',
        'state_after':'[CH3:1][Cl:3].[OH-:2]'}]}


def test_bridge_inverts_real_states_and_lifts_later_imports():
    from mechet.roundtrip_bridge import bridge_reverse_episode
    from mechet.forward_qwen_events import execute_forward_call
    from mechet.roundtrip_reward import component_match
    b=bridge_reverse_episode(example())
    assert b['status']=='verified',b
    assert len(b['witness'])==2
    assert b['witness'][0]['name']=='apply_forward_electron_flow'
    state=b['initial_state']
    for step in b['witness']:
        result=execute_forward_call(state,step['name'],step['arguments'])
        assert result['ok'];state=result['state_smiles']
        assert 'CO' not in step['messages'][0]['content'] # target not in system
    assert component_match(state,'CO')
    assert not component_match(b['initial_state'],'CO')


def test_noop_and_unknown_cannot_become_success():
    from mechet.roundtrip_reward import forward_outcome
    assert forward_outcome({'initial_state':'CO.[Cl-]','final_state':'CO.[Cl-]','terminal':True,'flow_count':0},'CO')['status']=='nonproductive'
    assert forward_outcome({'initial_state':'CCl.[OH-]','final_state':'CO.[Cl-]','terminal':False,'flow_count':1},'CO')['status']=='censored'
    assert forward_outcome({'initial_state':'CCl.[OH-]','final_state':'CO.[Cl-]','terminal':True,'flow_count':1},'CO')['success'] is True
    assert forward_outcome({'initial_state':'CCl.[OH-]','final_state':'CCl.[OH-]','terminal':False,'flow_count':0},'CO')['status']=='censored'


def test_bridge_rejects_unexplained_missing_atom():
    from mechet.roundtrip_bridge import bridge_reverse_episode
    x=example();x['steps'][1]['state_after']='[CH3:1][Cl:3]';x['final_state']='[CH3:1][Cl:3]'
    assert bridge_reverse_episode(x)['status']=='bridge_failed'


def test_pairs_require_exact_common_prompt_and_chemical_divergence():
    from mechet.roundtrip_reward import preference_pairs
    def e(id,prompt,succ,out):
        return {'episode_id':id,'actor_sha':'R0','outcome':out,'steps':[{'prompt_ids':prompt,'raw':id,'completion_ids':[ord(id),2],
         'name':'apply_electron_flow','accepted':True,'successor':succ}]}
    a=e('a',[1,2],'x',{'success':True,'status':'roundtrip_hit'})
    b=e('b',[1,2],'y',{'success':False,'status':'forward_mismatch'})
    assert len(preference_pairs([a,b]))==1
    assert not preference_pairs([a,e('b',[2,3],'y',b['outcome'])])
    assert not preference_pairs([a,e('b',[1,2],'x',b['outcome'])])
    assert not preference_pairs([a,e('b',[1,2],'y',{'success':False,'status':'censored'})])
