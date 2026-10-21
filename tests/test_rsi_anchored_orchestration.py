def test_productive_metrics_keep_old_roundtrip_and_full_target_denominator():
    from scripts.run_mechet_rsi_anchored import productive_summary
    es=[{'id':'a','candidate':0,'terminal':True,'initial_state':'[NH4+:1]',
         'final_state':'[NH3:1]','steps':[],'outcome':{'success':True,'status':'roundtrip_hit'},
         'productivity':{'task_compatible_net_progress':False}}]
    got=productive_summary(es,['a','b'])
    assert got['roundtrip_at_1_targets']==1 and got['productive_roundtrip_at_1_targets']==0
    assert got['targets']==2 and got['missing_target_ids']==['b']


def test_matched_budget_is_per_target_and_keeps_complete_denominator():
    from scripts.run_mechet_rsi_anchored import matched_views
    def e(target,n):
        return {'id':target,'candidate':0,'terminal':False,'initial_state':'CO','final_state':'CO',
                'steps':[{'completion_ids':[1]*n,'accepted':True}], 'outcome':{'status':'censored','success':False}}
    result=matched_views({'r0':[e('a',100),e('b',2)],'m3':[e('a',10),e('b',2)]},['a','b'])
    assert result['per_target_reverse_budget']=={'a':10,'b':2}
    assert result['models']['r0']['targets']==2 and result['models']['r0']['observed_targets']==1


def test_finalized_config_keeps_fixed_sources_and_declares_real_update():
    from pathlib import Path
    import yaml
    p=Path(__file__).resolve().parents[1]/'configs/experiments/mechet_rsi_real_anchored_resolved_20261021.yaml'
    cfg=yaml.safe_load(p.read_text())
    assert cfg['protocol']=='mechet_rsi_real_anchored_v1'
    assert cfg['reverse_sha']=='40204c7a059d16976fc4b24b315c3bdde7bf61c705d24bbde515071757e9de75'
    assert cfg['forward_sha']=='38e663a395a81c6901d078eb395026c4d062f17052730d54093c506d33aaa89e'
    assert cfg['max_optimizer_steps']==20 and cfg['max_pair_exposures']==4
    assert cfg['primary_actor']=='M3' and cfg['primary_rejection_guard'] is False
    assert cfg['test_used'] is False and 'real_anchored' in cfg['output']


def test_legacy_success_is_not_v1_confirmation_before_allocation(tmp_path,monkeypatch):
    import scripts.run_mechet_rsi_anchored as runner
    import mechet.roundtrip_bridge as bridge
    ex=runner.AnchoredExperiment.__new__(runner.AnchoredExperiment)
    ex.output=tmp_path;ex.actor_sha={'F0':'f0'};ex.roundtrips={};ex.log=lambda *a,**k:None
    e={'id':'a','episode_id':'a0','actor_sha':'r0','terminal':True,'flow_count':1,
       'initial_state':'[NH4+:1]','final_state':'[NH3:1]','target':'[NH4+]',
       'steps':[],'outcome':{'success':True,'status':'roundtrip_hit'}}
    monkeypatch.setattr(bridge,'bridge_reverse_episode',lambda e:{'initial_state':'[NH3:1]','status':'verified'})
    def allocate(es,*a,**k):
        assert not es[0]['outcome']['success']
        return []
    monkeypatch.setattr(runner,'paired_candidates',allocate)
    result,_=ex.feedback_v1([e],'new',{'a':{'kind':'acid_base'}},64)
    assert result[0]['outcome']['status']=='forward_unchecked'
