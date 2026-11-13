import copy
import json
import importlib
import pytest


def group(rewards):
    return [dict(episode_id='target/'+str(i),task_id='target',behavior_version='actor',
                 reward=value,generated_tokens=i+1,records=[dict(prompt_token_ids=[7],
                 generated_token_ids=[8]*(i+1),behavior_logprobs=[-.5]*(i+1))],
                 reactions=[],status='rejected_action',pending=[],stock_leaves=[],
                 reward_evidence=dict(verified_stock_closed=False,structural_stock_closed=False,
                     supported_reactions=0,unsupported_reactions=0,untrusted_reference_matches=0,
                     graph_error='',censored=value is None))
            for i,value in enumerate(rewards)]


def test_zero_advantage_keeps_actual_learner_and_kl_token_denominators():
    from scripts.audit_route_reward_scorecards import group_scorecards
    rows=group([.2]*4);before=copy.deepcopy(rows)
    result=group_scorecards(rows)
    assert result['generated_tokens']==10
    assert result['learner_policy_tokens']==result['kl_tokens']==10
    assert result['nonzero_advantage_policy_tokens']==0
    assert result['informative'] is False
    assert rows==before


def test_group_fraction_and_nonzero_advantage_token_fraction_are_distinct():
    from scripts.audit_route_reward_scorecards import group_scorecards
    # Centered advantages [0, -.5, 0, .5]: only 2/4 episodes, 6/10 tokens.
    result=group_scorecards(group([.5,0.,.5,1.]))
    assert result['advantages']==[0.,-.5,0.,.5]
    assert result['informative'] is True
    assert result['nonzero_advantage_policy_tokens']==6
    assert result['learner_policy_tokens']==10


def test_censoring_excludes_whole_group_but_keeps_generated_work():
    from scripts.audit_route_reward_scorecards import group_scorecards
    result=group_scorecards(group([1.,0.,None,.2]))
    assert result['usable'] is False
    assert result['advantages'] is None
    assert result['generated_tokens']==10
    assert result['learner_policy_tokens']==result['kl_tokens']==0


def test_unsupported_stock_closure_is_separate_from_invalid_graph():
    from scripts.audit_route_reward_scorecards import group_scorecards
    rows=group([0.]*4)
    rows[0]['status']='stock_solved'
    rows[0]['reward_evidence'].update(structural_stock_closed=True,unsupported_reactions=1)
    rows[1]['reward_evidence']['graph_error']='DISCONNECTED_OR_REUSED_REACTION'
    result=group_scorecards(rows)
    assert result['scorecards'][0]['category']=='structural_closure_without_full_reference_support'
    assert result['scorecards'][1]['category']=='invalid_route_graph'
    assert all(r['training_reward_unchanged']==0. for r in result['scorecards'])
    assert result['scorecards'][0]['executor_replay_verified'] is False
    assert result['scorecards'][0]['auxiliary_role_audit_verified'] is False


@pytest.mark.parametrize('bad', [float('nan'),float('inf'),-float('inf')])
def test_nonfinite_reward_fails_instead_of_creating_a_signal_statistic(bad):
    from scripts.audit_route_reward_scorecards import group_scorecards
    with pytest.raises(ValueError,match='NONFINITE_REWARD'):
        group_scorecards(group([bad,0.,0.,0.]))


def test_original_receipt_and_g4_identity_fail_closed():
    from scripts.audit_route_reward_scorecards import group_scorecards
    rows=group([0.]*4)
    with pytest.raises(ValueError,match='G4'):group_scorecards(rows[:3])
    rows[3]['task_id']='different-root'
    with pytest.raises(ValueError,match='ROOT_OR_BEHAVIOR'):group_scorecards(rows)
    rows[3]['task_id']='target';rows[3]['records'][0]['behavior_logprobs']=[]
    with pytest.raises(ValueError,match='TOKEN_RECEIPT'):group_scorecards(rows)


def test_first_round_cli_rejects_relabelled_small_population(tmp_path):
    from scripts.audit_route_reward_scorecards import replay
    audit=tmp_path/'audit.json'
    audit.write_text(json.dumps(dict(scope='original_first_round_saved256_reward_replay_and_signal_ledger',
        status='PASS',mismatches=[],witness_agreement={'targets':True},counts={'groups':1,'episodes':4})))
    with pytest.raises(ValueError,match='FULL64_GROUPS_256_EPISODES'):
        replay(audit,tmp_path/'output.json')
    assert not (tmp_path/'output.json').exists()


def test_first_round_cli_rejects_changed_bound_input(tmp_path):
    from scripts.audit_route_reward_scorecards import replay
    audit=tmp_path/'audit.json';source=tmp_path/'changed.json';source.write_text('{}')
    audit.write_text(json.dumps(dict(scope='original_first_round_saved256_reward_replay_and_signal_ledger',
        status='PASS',mismatches=[],witness_agreement={'targets':True},counts={'groups':64,'episodes':256},
        source_files_sha256={str(source):'0'*64},raw_artifacts_sha256={},
        source_contract=str(source),contract_sha256='0'*64)))
    with pytest.raises(ValueError,match='AUDITED_INPUT_CHANGED'):
        replay(audit,tmp_path/'output.json')
    assert not (tmp_path/'output.json').exists()


@pytest.mark.parametrize('module_name',[
    'scripts.replay_route_first_round_rewards','scripts.audit_route_reward_scorecards'])
def test_completed_report_publication_never_overwrites_another_writer(tmp_path,module_name):
    module=importlib.import_module(module_name)
    output=tmp_path/'same.json';output.write_text('owned-by-another-writer')
    with pytest.raises(FileExistsError):module.atomic(output,{'status':'PASS'})
    assert output.read_text()=='owned-by-another-writer'
    assert not list(tmp_path.glob('*.writing'))


@pytest.mark.parametrize('module_name',[
    'scripts.replay_route_first_round_rewards','scripts.audit_route_reward_scorecards'])
def test_full_private_report_is_published_complete_with_owner_only_access(tmp_path,module_name):
    module=importlib.import_module(module_name)
    output=tmp_path/'private.json';module.atomic(output,{'complete':[1,2,3]})
    assert json.loads(output.read_text())=={'complete':[1,2,3]}
    assert output.stat().st_mode&0o777==0o600
    assert not list(tmp_path.glob('*.writing'))
