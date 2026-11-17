from copy import deepcopy
import pytest


def record(name, before, after=None, accepted=True):
    import json
    return dict(raw='<tool_call>'+json.dumps(dict(name=name,arguments={}))+'</tool_call>',
                occurrence='0',state_before=before,state_after=after or before,accepted=accepted)


def test_unambiguous_flow_gets_only_conserved_reference_progress():
    from mechet.route_flow_credit import redistribute_progress
    a='[CH3:1][OH:2]';b='[CH3+:1].[OH-:2]'
    records=[record('apply_electron_flow',a,b),record('finish_trace',b)]
    events=[0.,.1]
    shifted,evidence=redistribute_progress(records,events,[(1,.1)],alpha=.5)
    assert shifted==pytest.approx([.05,.05]) and sum(shifted)==pytest.approx(.1)
    assert evidence[0]['flow_indices']==[0]
    assert events==[0.,.1]


def test_cancelled_edits_or_import_inventory_change_fall_back_to_finish():
    from mechet.route_flow_credit import redistribute_progress
    a='[CH3:1][OH:2]';b='[CH3+:1].[OH-:2]'
    records=[record('apply_electron_flow',a,b),record('apply_electron_flow',b,a),
             record('apply_electron_flow',a,b),record('finish_trace',b)]
    events=[0.,0.,0.,.1]
    assert redistribute_progress(records,events,[(3,.1)],alpha=.5)[0]==events
    records=[record('apply_electron_flow',a,b),
             record('import_fragments',b,b+'.[Na+:3]'),record('finish_trace',b+'.[Na+:3]')]
    events=[0.,0.,.1]
    assert redistribute_progress(records,events,[(2,.1)],alpha=.5)[0]==events


def test_alpha_zero_is_exact_legacy_and_invalid_alpha_fails():
    from mechet.route_flow_credit import redistribute_progress
    values=[.1,-.12];records=[{},{}]
    assert redistribute_progress(records,values,[(0,.1)],alpha=0)==(values,[])
    for alpha in [-1,1.01,float('nan'),True]:
        with pytest.raises(ValueError):
            redistribute_progress(records,values,[],alpha=alpha)


def test_failed_terminal_rollback_still_conserves_total(tmp_path):
    from mechet.route_process_reward import ProcessRouteReward
    from test_route_first_rejection import failed_after_progress,STOP
    out,task,report=failed_after_progress(tmp_path)
    # Missing actual mapped FLOW evidence must fall back rather than invent credit.
    with ProcessRouteReward(out/'references.sqlite',{'C','CC','O'},failure_policy=STOP,
                            flow_credit_alpha=.5) as oracle:
        scored=oracle.score(task['task_id'],deepcopy(report))
    assert sum(scored['process_credit']['event_rewards'])==pytest.approx(scored['reward'])
    assert scored['process_credit']['flow_credit']['alpha']==.5
