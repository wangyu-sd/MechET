"""Frozen-denominator aggregation and reference-only route analysis."""
import pytest


def test_missing_and_duplicate_predictions_cannot_change_full_denominator():
    from scripts.analyze_frozen_route_search import aggregate_cohort
    targets=[{'id':'a','already_in_stock':False,'reference_depth':2},
             {'id':'b','already_in_stock':False,'reference_depth':3}]
    with pytest.raises(ValueError,match='denominator'):
        aggregate_cohort(targets,{'greedy':[{'id':'a','stock_solved':True}]})
    with pytest.raises(ValueError,match='denominator'):
        aggregate_cohort(targets,{'greedy':[{'id':'a','stock_solved':False}]*2})


def test_all_target_solved_rate_retains_zero_step_targets_and_unknown_labels():
    from scripts.analyze_frozen_route_search import aggregate_cohort
    targets=[{'id':'a','already_in_stock':False,'reference_depth':2,'composition_stratum':'Unknown'},
             {'id':'b','already_in_stock':True,'reference_depth':3,'composition_stratum':'Unknown'}]
    rows=[{'id':'a','stock_solved':True,'already_in_stock':False,'work':{'sampled_tokens':100}},
          {'id':'b','stock_solved':False,'already_in_stock':True,'work':{'sampled_tokens':0}}]
    report=aggregate_cohort(targets,{'tree':rows})
    assert report['denominator'] == 2
    assert report['conditions']['tree']['solved'] == .5
    assert report['conditions']['tree']['zero_step_targets'] == 1
    assert report['strata']['Unknown']['n'] == 2
    assert report['strata']['C1']['n'] == 0
    assert report['strata']['C1']['conditions']['tree']['solved'] is None


def test_paired_bootstrap_uses_identical_target_ids_not_independent_step_samples():
    from scripts.analyze_frozen_route_search import paired_bootstrap
    result=paired_bootstrap([False,False],[True,True],resamples=2000,seed=17)
    assert result['n'] == 2
    assert result['delta'] == 1
    assert result['ci95'] == [1,1]
    with pytest.raises(ValueError):paired_bootstrap([True],[True,False])


def test_published_route_recovery_is_whole_edge_set_not_one_good_precursor():
    from scripts.analyze_frozen_route_search import published_route_recovered
    source=[{'reference_route_id':'ref','source_product_smiles':'CCO','source_precursor_smiles':['C','CO']},
            {'reference_route_id':'ref','source_product_smiles':'CO','source_precursor_smiles':['C','O']}]
    partial=[{'product':'CCO','precursors':['C','CO']}]
    complete=partial+[{'product':'CO','precursors':['O','C']}]
    assert not published_route_recovered(partial,source)
    assert published_route_recovered(complete,source)


def test_unfinished_stock_preparation_cannot_enter_model_inference(tmp_path):
    import json
    from scripts.infer_dynamic_electron_trees import load_route_stock
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'stock_preparation_complete':False}))
    with pytest.raises(ValueError,match='pending'):
        load_route_stock(manifest,'cohort','hash',100)


def test_forward_accounting_counts_real_scheduled_model_batches_once():
    from types import SimpleNamespace
    from scripts.infer_dynamic_electron_trees import ForwardCallCounter
    class Executor:
        def execute_model(self, execute_model_req):
            return 'completed'
    executor=Executor()
    engine=SimpleNamespace(llm_engine=SimpleNamespace(model_executor=executor))
    counter=ForwardCallCounter(engine)
    request=SimpleNamespace(seq_group_metadata_list=[SimpleNamespace(request_id='a'),SimpleNamespace(request_id='b')])
    assert executor.execute_model(execute_model_req=request)=='completed'
    assert counter.total_calls==1
    assert counter.by_request=={'a':1,'b':1}
    executor.execute_model(SimpleNamespace(seq_group_metadata_list=[]))
    assert counter.total_calls==1


def test_completed_result_cannot_be_reused_with_changed_runtime_or_mode():
    from scripts.infer_dynamic_electron_trees import verify_saved_run
    original={'mode':'greedy','code_sha':'old','source_sha':'source','rank':0,'world':8}
    with pytest.raises(ValueError,match='contract'):
        verify_saved_run(original,{**original,'mode':'tree'})
    with pytest.raises(ValueError,match='contract'):
        verify_saved_run(original,{**original,'code_sha':'new'})


def test_shard_cannot_masquerade_as_other_rank_condition_or_assets():
    from scripts.analyze_frozen_route_search import validate_shard_contract
    expected={'adapter_sha':'a','code_sha':'code','revision':'r','source_sha':'src','route_assets_sha':'actual-assets'}
    manifest={**expected,'rank':0,'world':8,'mode':'tree','evaluation_unit':'stock_connected_AND_route','route_cohort':'all'}
    validate_shard_contract(manifest,expected,rank=0,world=8,mode='tree')
    for field,value in [('rank',1),('mode','greedy'),('route_assets_sha','other-assets'),('code_sha','other-code')]:
        with pytest.raises(ValueError,match='contract'):
            validate_shard_contract({**manifest,field:value},expected,rank=0,world=8,mode='tree')


def test_reference_recovery_uses_retained_graph_not_only_best_continuation():
    from scripts.analyze_frozen_route_search import graph_contains_published_route
    source=[{'reference_route_id':'ref','source_product_smiles':'CCO','source_precursor_smiles':['C','CO']},
            {'reference_route_id':'ref','source_product_smiles':'CO','source_precursor_smiles':['C','O']}]
    graph=[{'product':'CCO','precursors':['C','CO']},
           {'product':'CO','precursors':['C','O']},
           {'product':'CO','precursors':['C=O']}]  # cheaper inner alternative does not erase reference
    assert graph_contains_published_route(graph,source)
    assert not graph_contains_published_route(graph[1:],source)
