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
