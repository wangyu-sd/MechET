from copy import deepcopy
import json
import math

import pytest

STOP='terminate_on_first_rejection_v1'


def episode(uid='stop'):
    from mechet.external_route_rl import TargetOnlyRouteEpisode
    return TargetOnlyRouteEpisode(dict(behavior_version='first-rejection',
        public_observation_contract='route_sft_aligned_feedback_v9',
        route_failure_policy=STOP,max_consecutive_recoveries=1,max_action_tokens=2048),
        'CO',set(),uid,17)


@pytest.mark.parametrize('name,args',[
    ('finish_trace',{}),
    ('apply_electron_flow',dict(direction='retrosynthetic',electron_flow=[],bond_order_changes=[],charge_changes=[])),
    ('unknown_action',{}),
])
def test_first_rejection_ends_route_and_never_requests_another_action(name,args):
    from test_route_native_observation import send
    ep=episode();ep.next_decision();before=deepcopy(vars(ep.node))
    send(ep,name,args)
    assert ep.status=='rejected_action'
    assert ep.next_decision() is None
    assert len(ep.records)==1 and ep.records[0]['accepted'] is False
    assert vars(ep.node)==before
    assert 'terminated' in ep.records[0]['feedback']['instruction'].lower()
    assert ep.finish_report()['route_failure_policy']==STOP
    with pytest.raises(ValueError,match='EPISODE_ALREADY_FINISHED'):
        ep.accept(ep.records[0])


def test_rejected_sibling_does_not_stop_other_independent_candidates():
    from test_route_native_observation import send
    siblings=[episode(str(i)) for i in range(4)]
    send(siblings[0],'finish_trace',{})
    assert siblings[0].next_decision() is None
    assert all(ep.next_decision() is not None and not ep.records for ep in siblings[1:])


def test_first_rejection_restore_and_contract_tampering():
    from mechet.external_route_rl import TargetOnlyRouteEpisode
    from test_route_native_observation import send
    ep=episode();send(ep,'finish_trace',{})
    snapshot=json.loads(json.dumps(ep.snapshot()))
    assert TargetOnlyRouteEpisode.restore(snapshot).finish_report()==ep.finish_report()
    snapshot['cfg'].pop('route_failure_policy')
    with pytest.raises(ValueError,match='SFT_HISTORY'):
        TargetOnlyRouteEpisode.restore(snapshot)


def failed_after_progress(tmp_path):
    from test_external_route_rl import oracle_fixture
    from test_route_process_rewards import decision
    out,task,raw=oracle_fixture(tmp_path)
    report=dict(raw,status='rejected_action',route_failure_policy=STOP,
        error='NONPRODUCTIVE_FINISH',max_consecutive_recoveries=1,
        reactions=raw['reactions'][:1],stock_leaves=['C'],records=[decision(0),
            decision(1,'0.0',accepted=False,error='NONPRODUCTIVE_FINISH',state='CCO')])
    return out,task,report


def test_failed_route_loses_all_partial_task_reward_and_keeps_failed_sample(tmp_path):
    from mechet.route_process_reward import ProcessRouteReward,process_sequences
    out,task,report=failed_after_progress(tmp_path)
    with ProcessRouteReward(out/'references.sqlite',{'C','CC','O'},failure_policy=STOP) as oracle:
        score=oracle.score(task['task_id'],report)
    cost=.025*(1-math.exp(-1/4))  # two reference steps; one ordinary rejection
    assert score['reward']==pytest.approx(-cost)
    assert score['route_task_reward']==0. and score['base_route_reward']==0.
    assert score['process_credit']['event_rewards']==pytest.approx([.1,-.1-cost])
    assert score['process_credit']['returns'][0]==pytest.approx(-cost)
    assert score['process_credit']['version']=='trusted_progress_first_rejection_cost_v2'
    report.update(reward=score['reward'],reward_evidence=score)
    assert len(list(process_sequences(report,[.1,-.2])))==2


def test_success_reward_is_unchanged_and_censoring_is_not_a_failed_route(tmp_path):
    from mechet.route_process_reward import ProcessRouteReward,process_group_advantages
    from test_external_route_rl import oracle_fixture
    from test_route_process_rewards import complete_report
    out,task,raw=oracle_fixture(tmp_path);report=complete_report(raw)
    report['route_failure_policy']=STOP
    with ProcessRouteReward(out/'references.sqlite',{'C','CC','O'},failure_policy=STOP) as oracle:
        result=oracle.score(task['task_id'],report)
        assert result['reward']==1.
        assert result['process_credit']['event_rewards']==pytest.approx([.1,.9])
        interrupted=oracle.score(task['task_id'],dict(report,status='resource_censored'))
        assert interrupted['reward'] is None
    assert process_group_advantages([[1.],None,[0.],[0.]]) is None


def test_first_rejection_reports_cannot_claim_continued_rollouts(tmp_path):
    from mechet.route_process_reward import ProcessRouteReward
    from test_route_process_rewards import decision
    out,task,report=failed_after_progress(tmp_path)
    report['records'].append(decision(2,'0.0',name='import_fragments'))
    with ProcessRouteReward(out/'references.sqlite',{'C','CC','O'},failure_policy=STOP) as oracle:
        with pytest.raises(ValueError,match='FIRST_REJECTION'):
            oracle.score(task['task_id'],report)


def test_first_rejection_configuration_binds_new_reward_and_preserves_legacy():
    from mechet.route_failure_policy import configure_first_rejection
    from mechet.route_process_reward import ALGORITHM,CONFIG,VERSION,validate_process_protocol
    legacy=dict(algorithm=ALGORITHM,process_reward=deepcopy(CONFIG),reward_contract=VERSION,
        advantage_estimator='centered_action_return_to_go_no_std',
        public_observation_contract='route_sft_aligned_feedback_v9',partial_reward_weight=.2)
    validate_process_protocol(legacy)
    cfg=deepcopy(legacy);configure_first_rejection(cfg)
    assert cfg['max_consecutive_recoveries']==1 and cfg['route_failure_policy']==STOP
    validate_process_protocol(cfg)
    cfg['reward_contract']=VERSION
    with pytest.raises(ValueError,match='PROCESS_PROTOCOL'):
        validate_process_protocol(cfg)


def test_reward_reader_rejects_mixed_failure_policies(tmp_path):
    from mechet.external_route_rl import HiddenRouteReward
    out,task,report=failed_after_progress(tmp_path)
    with HiddenRouteReward(out/'references.sqlite',{'C','CC','O'}) as oracle:
        with pytest.raises(ValueError,match='FAILURE_POLICY'):
            oracle.score(task['task_id'],report)


def test_finish_requires_real_state_changing_flow_in_current_occurrence():
    from test_route_native_observation import send,flow
    ep=episode();ep.next_decision();send(ep,'apply_electron_flow',flow(ep))
    assert not ep._finish_guard_error(ep.node)
    ep.records[-1]['state_after']=ep.records[-1]['state_before']
    assert ep._finish_guard_error(ep.node)=='NONPRODUCTIVE_FINISH'


def test_make_episodes_propagates_failure_and_material_contracts():
    from scripts.train_external_route_rl import make_episodes
    from mechet.route_materials import PROTIC_MATERIAL
    cfg=dict(candidates=2,max_action_tokens=2048,world_size=1,executor_workers_per_rank=2,seed=17,
             public_observation_contract='route_sft_aligned_feedback_v9',max_consecutive_recoveries=1,
             route_failure_policy=STOP,stock_material_contract=PROTIC_MATERIAL)
    episodes=make_episodes([dict(task_id='id',target='CO')],set(),[],cfg,'b0')
    assert all(e.cfg['stock_material_contract']==PROTIC_MATERIAL and e.cfg['route_failure_policy']==STOP for e in episodes)
