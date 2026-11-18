"""Real chemistry receipts exercise owner-completion and deterministic lane publication."""
from copy import deepcopy
import json

import pytest

POLICY='skip_extra_duplicates_v2'
LANES=dict(credit_forks_per_tree=16,exploration_forks_per_tree=0)


def collection(*,lanes=None,count=1,budget=None,max_nodes=8,recoveries=None):
    from mechet.route_refit_runtime import RefitCollection,DEFAULTS
    from mechet.route_refit import RefitRouteEpisode
    from test_route_first_rejection import episode
    cfg=episode().cfg
    if recoveries is not None:
        cfg=dict(cfg,max_consecutive_recoveries=recoveries);cfg.pop('route_failure_policy')
    episodes=[RefitRouteEpisode(cfg,'CO',set(),'root'+str(i),17) for i in range(count)]
    # Exercise RouteEpisode's legacy finite-budget terminal boundary directly;
    # target-only production intentionally has no aggregate episode ceiling.
    if budget is not None:
        for ep in episodes:ep.cfg['episode_token_budget']=budget
    return RefitCollection(episodes,dict(DEFAULTS,max_nodes_per_tree=max_nodes),
                           duplicate_policy=POLICY,lane_config=LANES if lanes is None else lanes)


def send(ep,name,args,**extra):
    request=ep.next_decision()
    ep.accept(dict(request_id=request.request_id,behavior_version=request.behavior_version,
        raw='<tool_call>'+json.dumps(dict(name=name,arguments=args))+'</tool_call>',
        prompt_token_ids=[1],generated_token_ids=[2],behavior_logprobs=[-.2],truncated=False,**extra))


def first(c):
    from test_route_native_observation import flow
    root=c.initial[0];root.next_decision();args=flow(root)
    send(root,'apply_electron_flow',args)
    return root,args,c.completed(root)[0]


@pytest.mark.parametrize('owner_state',['pending','resource','protected','terminal_unharvested'])
def test_only_serialized_completed_eligible_owner_covers_m1(owner_state):
    c=collection();root,args,fork=first(c)
    if owner_state!='pending':
        if owner_state=='protected':
            root.ep.protected.add('[CH3+]');send(root,'finish_trace',{})
            assert root.status=='protected_censored'
        else:
            send(root,'unknown',{},**({'resource_error':'RESOURCE_CENSORED:test'} if owner_state!='terminal_unharvested' else {}))
        if owner_state!='terminal_unharvested':c.completed(root)
    send(fork,'apply_electron_flow',args);c.completed(fork)
    assert fork.status is None
    assert c.coverage(root.path[0]['node_id'],root.path[0]['action'])==[]
    if owner_state=='terminal_unharvested':
        c.completed(root)
        assert [r['request_id'] for r in c.coverage(root.path[0]['node_id'],root.path[0]['action'])]==['root0/r0/a0']


def test_completed_failure_covers_m1_and_stopped_draw_keeps_actual_receipt():
    c=collection();root,args,fork=first(c)
    send(root,'unknown',{});c.completed(root)
    # Selection must never inspect a route reward or hidden reference evidence.
    root.ep.reward=object()
    send(fork,'apply_electron_flow',args);assert c.completed(fork)==[]
    assert fork.status=='duplicate_pruned_no_continuation' and fork.reward is None
    assert fork.records[-1]['accepted'] is True
    assert fork.records[-1]['generated_token_ids']==[2] and fork.records[-1]['behavior_logprobs']==[-.2]
    assert c.coverage(root.path[0]['node_id'],root.path[0]['action'])[0]['continuation_id']=='root0'


def test_completed_descendant_prefix_cannot_rescue_censored_owner():
    c=collection(recoveries=2);root,args,_=first(c)
    # A recovering invalid FLOW opens a descendant point whose saved prefix
    # belongs to the original continuation; then that original is censored.
    send(root,'apply_electron_flow',{});descendant=c.completed(root)[0]
    send(root,'unknown',{},resource_error='RESOURCE_CENSORED:test');c.completed(root)
    assert descendant.path[0]['receipt_owner']=='root0'
    for _ in range(2):
        send(descendant,'unknown',{});c.completed(descendant)
    assert descendant.status=='recovery_exhausted'
    event=root.path[0]
    assert c.coverage(event['node_id'],event['action'])==[]
    assert c.receipt_owner['root0/r0/a0']=='root0'


def test_already_terminal_accepted_duplicate_retains_genuine_budget_outcome():
    c=collection(budget=1);root,args,fork=first(c)
    assert root.status=='budget_exhausted'
    send(fork,'apply_electron_flow',args);c.completed(fork)
    assert fork.status=='budget_exhausted' and fork.reward==0.
    assert c.duplicate_pruned==0


def test_four_original_semantic_duplicates_keep_independent_full_outcomes():
    from test_route_native_observation import flow
    c=collection(count=4)
    for root in c.initial:
        root.next_decision();send(root,'apply_electron_flow',flow(root));c.completed(root)
        send(root,'unknown',{});c.completed(root)
    assert [e.status for e in c.initial]==['rejected_action']*4
    assert len({r['request_id'] for e in c.initial for r in e.records})==8
    assert all(e.lane=='original_g4' and len(e.records)==2 for e in c.initial)
    assert c.duplicate_pruned==0


def test_credit_exploration_budgets_and_owner_coverage_replay_exactly():
    from test_route_native_observation import flow
    c=collection(lanes=dict(credit_forks_per_tree=2,exploration_forks_per_tree=1))
    journal=[]
    # Deterministic harvester order: finish root, then each published fork.
    index=0
    while index<len(c.all):
        ep=c.all[index]
        while ep.next_decision() is not None:
            if not ep.records:send(ep,'apply_electron_flow',flow(ep))
            else:send(ep,'unknown',{})
            c.completed(ep);journal.append(deepcopy(ep.records[-1]))
        index+=1
    assert len(c.all)==4
    assert [e.lane for e in c.all]==['original_g4','credit','credit','exploration']
    assert len({e.continuation_id for e in c.all})==4
    assert all(e.uid=='root0' for e in c.all)
    assert [e.creation_order for e in c.all]==[0,1,2,3]
    assert c.all[1].status=='duplicate_pruned_no_continuation'
    packed=c.pack([e.finish_report() for e in c.all])
    counts=packed[0]['refit']['lane_counts']
    assert counts['credit']['forks']==2 and counts['exploration']['forks']==1
    assert counts['credit']['skipped']==1 and counts['exploration']['failed']==1
    assert counts['original_g4']['completed_eligible_coverage']==1
    assert 'actually_used_M1' not in packed[0]['refit']
    replay=collection(lanes=dict(credit_forks_per_tree=2,exploration_forks_per_tree=1))
    replay.replay(journal)
    assert replay.pack([e.finish_report() for e in replay.all])==packed
    assert replay.completed_eligible==c.completed_eligible and replay.receipt_owner==c.receipt_owner


def test_exploration_duplicate_continues_after_credit_coverage():
    from test_route_native_observation import flow
    c=collection(lanes=dict(credit_forks_per_tree=1,exploration_forks_per_tree=1))
    root,args,credit=first(c)
    # Pending credit repeats, publishes exploration before root's real failure.
    send(credit,'apply_electron_flow',args);explore=c.completed(credit)[0]
    send(root,'unknown',{});c.completed(root)
    assert explore.lane=='exploration'
    send(explore,'apply_electron_flow',args);c.completed(explore)
    assert explore.status is None
    assert c.lane_metrics()['exploration']['attempts']==1


@pytest.mark.parametrize('bad',[None,{}, {'credit_forks_per_tree':1},
    {'credit_forks_per_tree':True,'exploration_forks_per_tree':0},
    {'credit_forks_per_tree':0,'exploration_forks_per_tree':1},
    {'credit_forks_per_tree':16,'exploration_forks_per_tree':1},
    {'credit_forks_per_tree':1,'exploration_forks_per_tree':-1}])
def test_v2_requires_complete_finite_lane_config(bad):
    from mechet.route_refit_runtime import RefitCollection,DEFAULTS
    with pytest.raises(ValueError,match='LANE'):
        RefitCollection([],DEFAULTS,duplicate_policy=POLICY,lane_config=bad)


def test_exploration_large_return_cannot_change_conditional_q_or_advantages():
    from test_route_refit_credit import fixture
    from mechet.route_refit_credit import training_records
    from mechet.route_process_reward import digest,returns_to_go
    group=fixture();extra=deepcopy(group[0]['refit']['suffix_reports'][0])
    extra['refit_lane']='exploration'
    rec=extra['records'][-1];rec['request_id']='exploration-c'
    rec['raw']=rec['raw'].replace('"b"','"c"')
    from mechet.route_refit import grounded_action_key
    extra['refit_path'][-1].update(request_id='exploration-c',action=grounded_action_key(rec['state_before'],rec['raw']))
    credit=extra['reward_evidence']['process_credit'];credit.update(request_ids=['p','exploration-c'],event_rewards=[0.,100.],returns=returns_to_go([0.,100.]))
    credit.pop('sha256');credit['sha256']=digest(credit);extra['reward']=100.
    group[0]['refit']['suffix_reports'].append(extra)
    records,stats=training_records(group);values={r['request_id']:a for r,a in records}
    assert values==pytest.approx({'p':.105,'a':.11,'b':-.11,'other':-.105})
    assert stats['conditional_groups']==1


def test_genuine_duplicate_negative_return_adds_no_second_semantic_q_vote():
    from test_route_refit_credit import fixture
    from mechet.route_refit_credit import training_records
    from mechet.route_process_reward import digest
    group=fixture();extra=deepcopy(group[0]);extra.pop('refit')
    extra['records'][-1]['request_id']='duplicate-a'
    extra['refit_path'][-1]['request_id']='duplicate-a'
    credit=extra['reward_evidence']['process_credit'];credit.update(request_ids=['p','duplicate-a'],event_rewards=[0.,-9.],returns=[-9.,-9.])
    credit.pop('sha256');credit['sha256']=digest(credit);extra['reward']=-9.
    group[0]['refit']['suffix_reports'].append(extra)
    records,stats=training_records(group)
    assert {r['request_id']:a for r,a in records}==pytest.approx({'p':.105,'a':.11,'b':-.11,'other':-.105,'duplicate-a':.11})
    assert stats['conditional_groups']==1


def test_exploration_censor_is_diagnostic_and_cannot_mask_credit_node():
    c=collection(lanes=dict(credit_forks_per_tree=1,exploration_forks_per_tree=1))
    root,args,credit=first(c)
    send(credit,'apply_electron_flow',args);explore=c.completed(credit)[0]
    send(explore,'unknown',{},resource_error='RESOURCE_CENSORED:test');c.completed(explore)
    for ep in c.all:
        while ep.next_decision() is not None:
            send(ep,'unknown',{});c.completed(ep)
    packed=c.pack([e.finish_report() for e in c.all])[0]['refit']
    uid=explore.records[-1]['request_id'];node=root.path[0]['node_id']
    assert uid not in packed['node_censored_receipts'].get(node,[])
    assert packed['exploration_node_censored_receipts'][node]==[uid]
    assert packed['lane_counts']['exploration']['censored']==1


def test_original_and_credit_censor_still_mask_credit_node():
    c=collection();root,args,credit=first(c)
    send(credit,'unknown',{},resource_error='RESOURCE_CENSORED:test');c.completed(credit)
    for ep in c.all:
        while ep.next_decision() is not None:
            send(ep,'unknown',{});c.completed(ep)
    packed=c.pack([e.finish_report() for e in c.all])[0]['refit']
    assert packed['node_censored_receipts'][root.path[0]['node_id']]==[credit.records[-1]['request_id']]


def test_coverage_diagnostics_keep_owned_flow_after_retained_node_ceiling():
    c=collection(max_nodes=1);root,args,_=first(c)
    send(root,'apply_electron_flow',{});c.completed(root)
    assert root.status=='rejected_action' and len(c.nodes)==1
    assert c.lane_metrics('root0')['original_g4']['completed_eligible_coverage']==2
    for ep in c.all:
        while ep.next_decision() is not None:
            send(ep,'unknown',{});c.completed(ep)
    packed=c.pack([e.finish_report() for e in c.all])[0]['refit']
    assert len(packed['completed_eligible_coverage'])==2


def test_already_stock_original_has_complete_lane_count_without_fabricated_receipt():
    from mechet.route_refit import RefitRouteEpisode
    from mechet.route_refit_runtime import RefitCollection,DEFAULTS
    from test_route_first_rejection import episode
    ep=RefitRouteEpisode(episode().cfg,'CO',{'CO'},'stock-root',17)
    c=RefitCollection([ep],DEFAULTS,duplicate_policy=POLICY,lane_config=LANES)
    packed=c.pack([e.finish_report() for e in c.all])[0]
    assert packed['status']=='already_stock' and packed['records']==[]
    counts=packed['refit']['lane_counts']['original_g4']
    assert counts['complete']==1 and counts['attempts']==counts['completed_eligible_coverage']==0
