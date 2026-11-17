from copy import deepcopy


def test_extra_accepted_duplicate_stops_unknown_but_original_keeps_running():
    from mechet.route_refit_runtime import RefitCollection,DEFAULTS
    from mechet.route_refit import RefitRouteEpisode
    from test_route_first_rejection import episode
    from test_route_native_observation import flow,send
    base=episode();ep=RefitRouteEpisode(base.cfg,'CO',set(),'root',17)
    c=RefitCollection([ep],DEFAULTS,duplicate_policy='skip_extra_duplicates_v1')
    original=c.initial[0];original.next_decision();args=flow(original)
    send(original,'apply_electron_flow',args);sibling=c.completed(original)[0]
    send(sibling,'apply_electron_flow',args)
    assert c.completed(sibling)==[]
    assert sibling.status=='duplicate_pruned_no_continuation' and sibling.reward is None
    assert sibling.next_decision() is None and original.next_decision() is not None
    assert sibling.records[-1]['accepted'] is True
    assert len(sibling.records[-1]['behavior_logprobs'])==1


def test_budget_skipped_duplicate_never_masks_valid_branch_or_enters_loss():
    from mechet.route_refit_credit import training_records
    from test_route_refit_credit import fixture
    group=fixture();expected,_=training_records(group)
    pruned=deepcopy(group[0]);pruned.pop('refit');pruned.update(status='duplicate_pruned_no_continuation',reward=None,
        reward_evidence=dict(process_credit=None,budget_skipped=True))
    pruned['records'][-1]['request_id']='pruned-real-draw'
    group[0]['refit']['suffix_reports'].append(pruned)
    actual,stats=training_records(group)
    assert actual==expected and stats['conditional_groups']==1


def test_skip_reward_is_unknown_without_querying_private_reference():
    from mechet.route_acceleration import skipped_reward
    result=skipped_reward({'status':'duplicate_pruned_no_continuation'})
    assert result['reward'] is None and result['process_credit'] is None
    assert result['budget_skipped'] is True and result['censored'] is False


def test_acceleration_contract_rejects_incompatible_or_partial_settings():
    import pytest
    from mechet.route_acceleration import validate_acceleration,DEFAULT_ACCELERATION
    from mechet.route_refit_runtime import DEFAULTS
    c=dict(acceleration=dict(DEFAULT_ACCELERATION),refit=DEFAULTS,candidates=4,world_size=4,
           inference_backend='hf')
    validate_acceleration(c)
    with pytest.raises(ValueError):validate_acceleration(dict(c,refit=None))
    with pytest.raises(ValueError):validate_acceleration(dict(c,acceleration={'version':'pretend'}))
