def test_common_decoding_seed_does_not_depend_on_actor_or_phase():
    from scripts.rsi_runtime import decode_seed
    args=('reaction',1,3,'reverse')
    assert decode_seed(*args,actor_sha='r0',phase='old',contract='v1')==decode_seed(*args,actor_sha='r1',phase='new',contract='v1')
    assert decode_seed(*args,actor_sha='r0',phase='old',contract='v0')!=decode_seed(*args,actor_sha='r1',phase='new',contract='v0')


def test_per_target_budget_never_drops_later_target_ids():
    from mechet.rsi_preference_v1 import per_target_budget
    es=[{'id':'a','candidate':0,'steps':[{'completion_ids':list(range(100))}]},
        {'id':'b','candidate':0,'steps':[{'completion_ids':[1,2]}]}]
    got=per_target_budget(es,['a','b','c'],10)
    assert got['targets']==3 and got['spent_tokens']==2
    assert got['censored_target_ids']==['a'] and got['missing_target_ids']==['c']
    assert [e['id'] for e in got['episodes']]==['b']


def test_rejected_action_guard_requires_same_state_and_same_call_twice():
    from scripts.rsi_runtime import repeat_rejection
    seen=set()
    assert not repeat_rejection(seen,'CO','bad',enabled=True)
    assert not repeat_rejection(seen,'C=O','bad',enabled=True)
    assert repeat_rejection(seen,'CO','bad',enabled=True)
    assert not repeat_rejection(seen,'CO','bad',enabled=False)
