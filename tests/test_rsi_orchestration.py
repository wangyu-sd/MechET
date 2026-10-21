import pytest


def test_refresh_gate_rejects_r0_data_for_r1_round():
    from scripts.run_mechet_rsi_smoke import require_fresh_round
    rows=[{'actor_sha':'r0','round':'D0','episode_id':'old'}]
    with pytest.raises(ValueError):require_fresh_round(rows,'r1','D1',{'old'})
    require_fresh_round([{'actor_sha':'r1','round':'D1','episode_id':'new'}],'r1','D1',{'old'})


def test_fixed_validation_keeps_failed_and_unsampled_denominator():
    from scripts.run_mechet_rsi_smoke import summarize_evaluation
    summary=summarize_evaluation([{'id':'a','episode_id':'e','terminal':False,'steps':[],
      'outcome':{'success':False,'status':'censored'},'final_state':'C','initial_state':'C'}],['a','b'])
    assert summary['targets']==2
    assert summary['forward_verified_targets']==0
    assert summary['missing_target_ids']==['b']


def test_discovery_deduplicates_only_auxiliary_differences():
    from scripts.run_mechet_rsi_smoke import summarize_evaluation
    def e(identifier,end):
        return {'id':'a','episode_id':identifier,'terminal':True,'steps':[],
          'outcome':{'success':True},'initial_state':'[CH3:1][OH:2]',
          'final_state':end}
    a=e('x','[CH3:1][Cl:3].[OH-:2]')
    b=e('y','[CH3:1][Cl:3].[OH-:2].[Na+:4]')
    assert summarize_evaluation([a,b],['a'])['distinct_forward_verified_reactions']==1


def test_adaptive_expansion_keeps_uninformative_targets_only():
    from scripts.run_mechet_rsi_smoke import targets_needing_comparison
    a={'id':'a','episode_id':'hit','actor_sha':'r0','outcome':{'success':True},
       'steps':[{'prompt_ids':[1],'completion_ids':[2],'raw':'hit','accepted':True,'successor':'A','name':'finish_trace'}]}
    b={'id':'a','episode_id':'fail','actor_sha':'r0','outcome':{'success':False,'status':'forward_mismatch'},
       'steps':[{'prompt_ids':[1],'completion_ids':[3],'raw':'fail','accepted':True,'successor':'B','name':'apply_electron_flow'}]}
    c={'id':'b','episode_id':'unknown','actor_sha':'r0','outcome':{'success':False,'status':'censored'},'steps':[]}
    assert targets_needing_comparison([a,b,c],['a','b'])==['b']


def test_resume_requires_the_same_immutable_contract(tmp_path):
    from scripts.run_mechet_rsi_smoke import stage_file
    p=tmp_path/'stage.json'
    assert stage_file(p,{'actor':'r0'},lambda:{'value':3})=={'value':3}
    assert stage_file(p,{'actor':'r0'},lambda:1/0)=={'value':3}
    with pytest.raises(ValueError):stage_file(p,{'actor':'r1'},lambda:{})


def test_likelihood_gate_rejects_missing_or_disagreeing_model_logps():
    from scripts.run_mechet_rsi_smoke import check_likelihood_parity
    assert check_likelihood_parity([{'tokens':10,'sampled_sum_logp':-10.,'scored_sum_logp':-10.01}])['pass']
    assert not check_likelihood_parity([{'tokens':10,'sampled_sum_logp':-10.,'scored_sum_logp':-12.}])['pass']
    assert not check_likelihood_parity([])['pass']


def test_roundtrip_at_one_is_first_draw_not_best_of_k():
    from scripts.run_mechet_rsi_smoke import summarize_evaluation
    def episode(candidate,hit):
        return {'id':'a','candidate':candidate,'episode_id':str(candidate),'terminal':True,
                'steps':[],'initial_state':'[CH3:1][OH:2]',
                'final_state':'[CH3:1][Cl:3].[OH-:2]','outcome':{'success':hit}}
    result=summarize_evaluation([episode(0,False),episode(1,True)],['a','b'])
    assert result['roundtrip_at_1_targets']==0
    assert result['roundtrip_at_k_targets']==1
    assert result['roundtrip_at_k_rate']==.5


def test_training_pool_enters_all_adaptive_stages_without_log_field_collision(tmp_path,monkeypatch):
    import json
    from scripts.run_mechet_rsi_smoke import Experiment
    experiment=Experiment({},tmp_path);requests=[]
    def empty_gpu_collection(rows,actor,phase,indices):
        requests.append((actor,phase,list(indices)))
        return []
    monkeypatch.setattr(experiment,'collect',empty_gpu_collection)
    try:
        episodes,pairs=experiment.train_pool([{'id':'reaction-a'}],'R0','D0')
    finally:experiment.raw.close()
    assert episodes==[] and pairs==[]
    assert requests==[('R0','D0_K2',[0,1]),('R0','D0_K4',[2,3]),('R0','D0_K8',[4,5,6,7])]
    events=[json.loads(line) for line in (tmp_path/'phase_events.jsonl').read_text().splitlines()]
    assert len(events)==3
    assert all(e['phase']=='adaptive-stage' and e['round_id']=='D0' for e in events)
    assert json.loads((tmp_path/'D0.allocation.json').read_text())['forward_checks_used']==0
