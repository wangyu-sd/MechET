import pytest


def episode(identifier,success,status='roundtrip_hit',progress=True,accepted=True,prompt=(1,2),successor='CO'):
    return {'id':'reaction-a','episode_id':identifier,'actor_sha':'r0','terminal':True,'flow_count':1,
       'outcome':{'success':success,'status':status},'productivity':{'task_compatible_net_progress':progress},
       'steps':[{'prompt_ids':list(prompt),'completion_ids':[3 if success else 4,5],
         'complete':True,'accepted':accepted,'successor':successor,'name':'apply_electron_flow',
         'raw':identifier,'error':'' if accepted else 'CHEMICAL_STATE_INVALID','arguments':{}}]}


@pytest.mark.parametrize('status',['forward_mismatch','forward_unchecked','censored','inverse_incomplete'])
def test_single_forward_failure_or_missing_check_is_not_hard_chemistry_negative(status):
    from mechet.rsi_preference_v1 import strong_pairs
    good=episode('good',True);unknown=episode('unknown',False,status,successor='C=O')
    assert strong_pairs([good,unknown])==[]


def test_exact_prompt_and_productive_chosen_are_required_for_invalid_action_pair():
    from mechet.rsi_preference_v1 import strong_pairs
    good=episode('good',True);bad=episode('bad',False,'forward_unchecked',accepted=False,successor=None)
    pairs=strong_pairs([good,bad]);assert len(pairs)==1 and pairs[0]['source_id']=='reaction-a'
    assert pairs[0]['evidence']=='executor_rejected'
    good['productivity']['task_compatible_net_progress']=False
    assert strong_pairs([good,bad])==[]
    good['productivity']['task_compatible_net_progress']=True;bad['steps'][0]['prompt_ids']=[9]
    assert strong_pairs([good,bad])==[]


def test_task_incompatible_terminal_can_be_rejected_but_not_same_successor():
    from mechet.rsi_preference_v1 import strong_pairs
    good=episode('good',True);bad=episode('bad',False,'forward_unchecked',progress=False,successor='C=O')
    assert len(strong_pairs([good,bad]))==1
    bad['steps'][0]['successor']='CO'
    assert strong_pairs([good,bad])==[]


def test_source_cap_prevents_many_correlated_rejects_from_multiplying_one_reaction():
    from mechet.rsi_preference_v1 import strong_pairs
    es=[episode('good',True)]
    for i in range(8):
        e=episode(f'bad{i}',False,accepted=False,successor=None)
        e['steps'][0]['error']=f'invalid-{i}';e['steps'][0]['completion_ids']=[10+i,5];es.append(e)
    assert len(strong_pairs(es,max_per_reaction=2))==2


def test_forward_allocation_checks_complementary_candidates_within_a_target():
    from mechet.rsi_preference_v1 import paired_candidates
    es=[]
    for target in ['a','b','c']:
        for i in range(2):
            e=episode(target+str(i),False,'forward_unchecked',successor='CO' if i==0 else 'C=O')
            e['id']=target;es.append(e)
    selected=paired_candidates(es,limit=4)
    assert len(selected)==4
    from collections import Counter
    assert sorted(Counter(e['id'] for e in selected).values())==[2,2]
    assert len({e['episode_id'] for e in selected})==4


def test_cached_comparators_do_not_starve_fresh_targets_or_consume_new_budget():
    from mechet.rsi_preference_v1 import paired_candidates
    es=[]
    for i in range(20):
        for j in range(2):
            e=episode(f'{i}-{j}',False,'forward_unchecked',successor='CO' if j==0 else 'C=O')
            e['id']=str(i);e['bridge_key']=e['episode_id'];es.append(e)
    first=paired_candidates(es,limit=32)
    second=paired_candidates(es,limit=16,known_keys={e['bridge_key'] for e in first})
    assert len(second)==8 and not {e['bridge_key'] for e in first}&{e['bridge_key'] for e in second}


def test_single_terminal_with_common_prompt_invalid_alternative_gets_one_check():
    from mechet.rsi_preference_v1 import paired_candidates
    candidate=episode('candidate',False,'forward_unchecked')
    invalid=episode('invalid',False,'inverse_incomplete',accepted=False,successor=None)
    invalid['terminal']=False
    assert [e['episode_id'] for e in paired_candidates([candidate,invalid],limit=1)]==['candidate']
    invalid['steps'][0]['prompt_ids']=[999]
    assert paired_candidates([candidate,invalid],limit=1)==[]
