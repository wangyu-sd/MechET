from copy import deepcopy
import json
import math

import pytest

from test_external_route_rl import oracle_fixture


def decision(index, occurrence='0', *, accepted=True, name='finish_trace',
             arguments=None, error='', state='CCOC'):
    return dict(request_id=f'episode/a{index}', occurrence=occurrence,
                state_before=state, accepted=accepted, error=error,
                raw='<tool_call>'+json.dumps(dict(name=name, arguments=arguments or {}))+'</tool_call>',
                prompt_token_ids=[1, 2], generated_token_ids=[3], behavior_logprobs=[-.5])


def complete_report(report):
    return dict(report, records=[decision(0), decision(1, '0.0', state='CCO')])


def test_verified_reward_is_allocated_to_commits_and_conserves_final_return(tmp_path):
    from mechet.route_process_reward import ProcessRouteReward
    out, task, raw = oracle_fixture(tmp_path)
    report = complete_report(raw)
    with ProcessRouteReward(out/'references.sqlite', {'C', 'CC', 'O'}) as oracle:
        result = oracle.score(task['task_id'], report)
    assert result['reward'] == 1.
    assert result['process_credit']['event_rewards'] == pytest.approx([.1, .9])
    assert result['process_credit']['returns'] == pytest.approx([1., .9])
    assert sum(result['process_credit']['event_rewards']) == pytest.approx(result['reward'])
    assert result['process_credit']['request_ids'] == ['episode/a0', 'episode/a1']


def test_partial_progress_has_bounded_failure_cost_and_clean_recovery_wins(tmp_path):
    from mechet.route_process_reward import ProcessRouteReward
    out, task, raw = oracle_fixture(tmp_path)
    partial = dict(raw, status='recovery_exhausted', reactions=raw['reactions'][:1], stock_leaves=['C'])
    clean = dict(partial, records=[decision(0)])
    failure = decision(1, '0.0', accepted=False, name='apply_electron_flow',
                       arguments={'electron_flow': []}, error='ILLEGAL_EVENT', state='CCO')
    failed = dict(partial, records=[decision(0), failure])
    repeated = dict(partial, records=[decision(0), failure,
                    dict(failure, request_id='episode/a2')])
    with ProcessRouteReward(out/'references.sqlite', {'C', 'CC', 'O'}) as oracle:
        a, b, c = [oracle.score(task['task_id'], r) for r in (clean, failed, repeated)]
    assert a['reward'] == pytest.approx(.1)
    assert .075 < c['reward'] < b['reward'] < a['reward']
    assert b['process_credit']['event_rewards'][0] == pytest.approx(.1)
    assert b['process_credit']['event_rewards'][1] < 0
    assert c['process_credit']['validator_error_weight'] == 3.


def test_accepted_imports_do_not_create_progress_or_dilute_prior_errors(tmp_path):
    from mechet.route_process_reward import ProcessRouteReward
    out, task, raw = oracle_fixture(tmp_path)
    fail=decision(0, accepted=False, error='NONPRODUCTIVE_FINISH')
    reports=[dict(raw,status='recovery_exhausted',reactions=[],stock_leaves=[],records=[fail]),
             dict(raw,status='recovery_exhausted',reactions=[],stock_leaves=[],records=[fail,
                  decision(1,name='import_fragments',arguments={'fragments': []})])]
    with ProcessRouteReward(out/'references.sqlite', {'C', 'CC', 'O'}) as oracle:
        a,b=[oracle.score(task['task_id'],r) for r in reports]
    assert a['reward'] == pytest.approx(-.025*(1-math.exp(-1/4)))
    assert b['reward'] == a['reward']
    assert b['process_credit']['event_rewards'][1] == 0.


def test_unknown_resource_outcome_has_no_trainable_process_rewards(tmp_path):
    from mechet.route_process_reward import ProcessRouteReward
    out, task, raw = oracle_fixture(tmp_path)
    with ProcessRouteReward(out/'references.sqlite', {'C', 'CC', 'O'}) as oracle:
        result=oracle.score(task['task_id'],dict(complete_report(raw),status='resource_censored'))
    assert result['reward'] is None and result['process_credit'] is None


def test_untrusted_ancestor_does_not_award_descendant_progress(tmp_path):
    from mechet.route_process_reward import ProcessRouteReward
    out, task, raw = oracle_fixture(tmp_path)
    report=complete_report(deepcopy(raw))
    report['reactions'][0]['precursors']=['CCO','CC']
    report['stock_leaves']=['CC','CC','O']
    with ProcessRouteReward(out/'references.sqlite', {'C', 'CC', 'O'}) as oracle:
        result=oracle.score(task['task_id'],report)
    assert result['reward']==0.
    assert result['process_credit']['event_rewards']==[0.,0.]


def test_invalid_graph_terminal_correction_erases_earlier_progress(tmp_path):
    from mechet.route_process_reward import ProcessRouteReward
    out, task, raw = oracle_fixture(tmp_path)
    report=complete_report(deepcopy(raw));report['reactions'][1]['occurrence']='0.4'
    report['records'][1]['occurrence']='0.4'
    with ProcessRouteReward(out/'references.sqlite', {'C', 'CC', 'O'}) as oracle:
        result=oracle.score(task['task_id'],report)
    assert result['reward']==0. and result['graph_error']
    assert result['process_credit']['event_rewards']==pytest.approx([.1,-.1])


def test_step_baseline_is_centered_return_to_go_with_absorbing_zero_padding():
    from mechet.route_process_reward import process_group_advantages
    assert process_group_advantages([[0.,.6],[.2],[0.,0.,0.],[-.05]]) == [
        pytest.approx([.4125,.45]),pytest.approx([.0125]),
        pytest.approx([-.1875,-.15,0.]),pytest.approx([-.2375])]
    assert process_group_advantages([[.2,.1]]*4)==[[0.,0.]]*4
    assert process_group_advantages([[1.],None,[0.],[0.]]) is None


def test_credit_binds_original_request_order_and_reward_conservation(tmp_path):
    from mechet.route_process_reward import ProcessRouteReward, process_sequences
    out, task, raw = oracle_fixture(tmp_path);report=complete_report(raw)
    with ProcessRouteReward(out/'references.sqlite', {'C', 'CC', 'O'}) as oracle:
        result=oracle.score(task['task_id'],report)
    report.update(reward=result['reward'],reward_evidence=result)
    seq=list(process_sequences(report,[.2,.3]))
    assert seq==[([1,2],[3],[-.5],.2),([1,2],[3],[-.5],.3)]
    report['records'].reverse()
    with pytest.raises(ValueError,match='PROCESS_'):
        list(process_sequences(report,[.2,.3]))
