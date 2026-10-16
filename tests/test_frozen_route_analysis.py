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


def route_analysis_cli_fixture(tmp_path):
    """Independent small CPU fixture; never runs an actor or an experiment."""
    import json
    from mechet.frozen_route_assets import file_sha256
    from scripts.infer_dynamic_electron_trees import ADAPTER_SHA, REVISION
    targets=[{'id':'a','already_in_stock':False}, {'id':'b','already_in_stock':False}]
    source=tmp_path/'targets.jsonl'
    source.write_text(''.join(json.dumps(row)+'\n' for row in targets))
    assets={'stock_preparation_complete':True,'composition_annotation_complete':False,
        'preparation_code_sha':'a'*40,'full_targets_sha256':file_sha256(source),
        'stocks':{'unit':{'canonical_sha256':'stock'}},
        'cohorts':{'unit':{'targets_path':str(source),'targets_sha256':file_sha256(source),
                           'target_ids':['a','b']}}}
    manifest=tmp_path/'assets.json';manifest.write_text(json.dumps(assets))
    output=tmp_path/'run'
    for mode in ['greedy','tree']:
        directory=output/mode/'rank00';directory.mkdir(parents=True)
        contract={'adapter_sha':ADAPTER_SHA,'revision':REVISION,'model':'Qwen/Qwen3-0.6B',
            'source_sha':file_sha256(source),'route_assets_sha':file_sha256(manifest),
            'code_sha':'b'*40,'rank':0,'world':1,'mode':mode,
            'evaluation_unit':'stock_connected_AND_route','route_cohort':'all',
            'task_budget_mode':'renewable_checkpoints_no_token_or_time_cap_v1',
            'stock_canonical_shas':{'unit':'stock'},'seed':17,'task_token_budget':80000000,
            'max_model_len':8192,'wall_hours':24.,'shared_cache_namespace':'same-policy'}
        rows=[{'id':'a','stock_solved':True,'already_in_stock':False,'work':{'sampled_tokens':1}},
              {'id':'b','stock_solved':False,'already_in_stock':False,'work':{'sampled_tokens':1}}]
        (directory/'complete.json').write_text(json.dumps({'manifest':contract,'rows':rows,
                                                          'tokens':2,'model_forward_calls':2}))
    return manifest,output


def test_analysis_accepts_frozen_assets_prepared_by_an_older_code_revision(tmp_path,monkeypatch):
    import json,sys
    from scripts.analyze_frozen_route_search import main
    assets,output=route_analysis_cli_fixture(tmp_path)
    monkeypatch.setattr(sys,'argv',['analyze','--assets',str(assets),'--output',str(output),'--world','1'])
    main()
    report=json.loads((output/'r1_route_results.json').read_text())
    assert report['unit']['denominator']==2
    assert report['unit']['conditions']['tree']['solved']==.5
    provenance=json.loads((output/'figure3_manifest.json').read_text())
    assert provenance['asset_preparation_code_sha']=='a'*40
    assert provenance['inference_code_shas']=={'greedy':'b'*40,'tree':'b'*40}


def test_explicit_pins_allow_preserving_greedy_without_rerunning_it(tmp_path,monkeypatch):
    import json,sys
    from scripts.analyze_frozen_route_search import main
    assets,old_output=route_analysis_cli_fixture(tmp_path)
    new_output=tmp_path/'new-run'
    (new_output/'tree').mkdir(parents=True)
    old_tree=old_output/'tree'/'rank00'/'complete.json'
    shard=json.loads(old_tree.read_text());shard['manifest']['code_sha']='c'*40
    new_tree=new_output/'tree'/'rank00';new_tree.mkdir()
    (new_tree/'complete.json').write_text(json.dumps(shard))
    old_greedy=old_output/'greedy'/'rank00'/'complete.json'
    original_bytes=old_greedy.read_bytes()
    monkeypatch.setattr(sys,'argv',['analyze','--assets',str(assets),'--output',str(new_output),'--world','1',
        '--greedy-output',str(old_output),'--greedy-code-sha','b'*40,'--tree-code-sha','c'*40])
    main()
    assert old_greedy.read_bytes()==original_bytes
    assert not (new_output/'greedy').exists()
    provenance=json.loads((new_output/'figure3_manifest.json').read_text())
    assert provenance['inference_code_shas']=={'greedy':'b'*40,'tree':'c'*40}
    assert provenance['explicit_cross_runtime_comparison'] is True
    assert any(item['path']==str(old_greedy) for item in provenance['source_shards'])
    # Declared runtime pins are verified, not just recorded.
    monkeypatch.setattr(sys,'argv',['analyze','--assets',str(assets),'--output',str(new_output),'--world','1',
        '--greedy-output',str(old_output),'--greedy-code-sha','d'*40,'--tree-code-sha','c'*40])
    with pytest.raises(ValueError,match='code_sha'):main()
    # Even explicit pins must not mix a different policy or executor.
    shard['manifest']['shared_cache_namespace']='different-policy'
    (new_tree/'complete.json').write_text(json.dumps(shard))
    monkeypatch.setattr(sys,'argv',['analyze','--assets',str(assets),'--output',str(new_output),'--world','1',
        '--greedy-output',str(old_output),'--greedy-code-sha','b'*40,'--tree-code-sha','c'*40])
    with pytest.raises(ValueError,match='shared_cache_namespace'):main()


def test_missing_external_greedy_is_not_silently_accepted_as_a_single_condition(tmp_path,monkeypatch):
    import sys
    from scripts.analyze_frozen_route_search import main
    assets,output=route_analysis_cli_fixture(tmp_path)
    monkeypatch.setattr(sys,'argv',['analyze','--assets',str(assets),'--output',str(output),'--world','1',
        '--greedy-output',str(tmp_path/'missing'),'--allow-single-condition'])
    with pytest.raises(ValueError,match='greedy'):main()


def test_analysis_still_rejects_undeclared_mixed_inference_revisions(tmp_path,monkeypatch):
    import json,sys
    from scripts.analyze_frozen_route_search import main
    assets,output=route_analysis_cli_fixture(tmp_path)
    path=output/'tree'/'rank00'/'complete.json'
    shard=json.loads(path.read_text());shard['manifest']['code_sha']='c'*40
    path.write_text(json.dumps(shard))
    monkeypatch.setattr(sys,'argv',['analyze','--assets',str(assets),'--output',str(output),'--world','1'])
    with pytest.raises(ValueError,match='mixed frozen route conditions: code_sha'):main()


def test_mcts_condition_cannot_mix_adaptive_shards_or_fake_an_mcts_label():
    from scripts.analyze_frozen_route_search import condition_search_algorithm
    mcts={'search_algorithm':'mcts','mcts_contract':'shared_sampled_puct_mcts_v1',
          'mcts_c_puct':1.5,'mcts_rollout_concurrency_per_target':8,'stop_after_first_stock_route':True}
    assert condition_search_algorithm([mcts,mcts])=='mcts'
    assert condition_search_algorithm([{},{}])=='adaptive'
    for invalid in [{}, {**mcts,'mcts_contract':'fake'}, {**mcts,'stop_after_first_stock_route':False}]:
        with pytest.raises(ValueError,match='algorithm'):
            condition_search_algorithm([mcts,invalid])


def test_reference_recovery_uses_retained_graph_not_only_best_continuation():
    from scripts.analyze_frozen_route_search import graph_contains_published_route
    source=[{'reference_route_id':'ref','source_product_smiles':'CCO','source_precursor_smiles':['C','CO']},
            {'reference_route_id':'ref','source_product_smiles':'CO','source_precursor_smiles':['C','O']}]
    graph=[{'product':'CCO','precursors':['C','CO']},
           {'product':'CO','precursors':['C','O']},
           {'product':'CO','precursors':['C=O']}]  # cheaper inner alternative does not erase reference
    assert graph_contains_published_route(graph,source)
    assert not graph_contains_published_route(graph[1:],source)


def test_reference_complexity_keeps_missing_mapping_distinct_from_observed_zero():
    from scripts.analyze_frozen_route_search import complexity_rows
    targets=[{'id':'partial','reference_depth':2,
        'reference_routes':[{'reference_route_id':'ref','reaction_edge_count':2}],
        'reference_reactions':[
            {'reference_route_id':'ref','mapping_valid':True,
             'changed_bonds_shared_mapped_atoms':0,'ring_count_change':0},
            {'reference_route_id':'ref','mapping_valid':False,
             'changed_bonds_shared_mapped_atoms':None,'ring_count_change':None}]}]
    row=complexity_rows(targets)[0]
    assert row['changed_bonds_shared_mapped_atoms'] is None
    assert row['changed_bonds_shared_mapped_atoms_available'] is False
    assert row['ring_count_change'] is None
    assert row['ring_count_change_available'] is False
    assert row['reference_depth_available'] is True
    assert row['mapped_reference_edges'] == 1
    assert row['reference_event_length_available'] is False


def test_ring_opening_and_closure_do_not_cancel_in_route_complexity():
    from scripts.analyze_frozen_route_search import complexity_rows
    target={'id':'rings','reference_depth':2,
        'reference_routes':[{'reference_route_id':'ref','reaction_edge_count':2}],
        'reference_reactions':[
            {'reference_route_id':'ref','mapping_valid':True,'ring_count_change':1},
            {'reference_route_id':'ref','mapping_valid':True,'ring_count_change':-1}]}
    row=complexity_rows([target])[0]
    assert row['ring_count_change'] == 0
    assert row['ring_count_change_available'] is True
    assert row['ring_closure_edge_count'] == 1
    assert row['ring_opening_edge_count'] == 1
    assert row['ring_closure_edge_count_available'] is True


def test_no_reference_target_has_missing_complexity_not_empty_route_zero():
    from scripts.analyze_frozen_route_search import complexity_rows
    row=complexity_rows([{'id':'no-reference','reference_depth':None}])[0]
    assert row['ring_closure_edge_count'] is None
    assert row['ring_closure_edge_count_available'] is False
    assert row['reference_depth_available'] is False
    assert row['mapped_reference_edges'] == 0


def test_offline_mechanism_index_must_bind_actor_exposure_and_reference_source(tmp_path):
    import json
    from scripts.analyze_frozen_route_search import load_mechanism_index
    assets={'config_sha256':'sources','cohorts':{'c':{'reference_reactions_sha256':'references'}}}
    manifest={'schema_version':'actual_sft_frozen_route_annotation_v1','model_sha256':'actor',
        'source_config_sha256':'sources','actual_train_exposure':{'source_sha256':'train','decision_rows':10,'reaction_rows':2},
        'cohorts':{'c':{'reference_reactions_sha256':'references'}}}
    path=tmp_path/'manifest.json';path.write_text(json.dumps(manifest))
    load_mechanism_index(path,assets,model_sha='actor',train_sha='train',decisions=10,reactions=2)
    for field,value in [('model_sha256','other'),('source_config_sha256','changed')]:
        bad={**manifest,field:value};path.write_text(json.dumps(bad))
        with pytest.raises(ValueError,match='annotation'):
            load_mechanism_index(path,assets,model_sha='actor',train_sha='train',decisions=10,reactions=2)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='exposure'):
        load_mechanism_index(path,assets,model_sha='actor',train_sha='subset',decisions=10,reactions=2)


def diversity_fixture(tmp_path):
    import json
    from mechet.frozen_route_assets import file_sha256
    root=tmp_path/'rank00';root.mkdir()
    graph=root/'tree.json';graph.write_text('{"saved_graph":true}')
    shard=root/'complete.json'
    rows=[{'id':'a','stock_solved':True,'already_in_stock':False,'tree_artifact':'tree.json',
           'unique_route_count_lower_bound':1}]
    shard.write_text(json.dumps({'rows':rows}))
    counts=tmp_path/'counts.json'
    counts.write_text(json.dumps({'complete':True,'denominator':1,'rows':[
        {'id':'a','unique_route_count':3,'route_count_is_exhaustive':True,'already_in_stock':False}]}))
    value={'mode':'tree','denominator':1,'model_sha256':'actor','assets_sha256':'assets',
        'counts_sha256':file_sha256(counts),'route_count_is_exhaustive':True,
        'route_definition':'one_production_choice_per_distinct_nonstock_molecule_DAG_v1',
        'electron_program_alternatives_counted_as_separate_routes':False,'model_called':False,
        'source_shards':[{'path':str(shard),'sha256':file_sha256(shard)}],
        'source_graphs':[{'id':'a','graph_sha256':file_sha256(graph)}]}
    path=tmp_path/'manifest.json';path.write_text(json.dumps(value))
    return path,shard,rows,value


def test_exact_diversity_is_attached_without_overwriting_raw_predictions(tmp_path):
    from scripts.analyze_frozen_route_search import attach_exact_diversity
    path,shard,rows,_=diversity_fixture(tmp_path)
    enriched=attach_exact_diversity(rows,path,mode='tree',assets_sha='assets',model_sha='actor',shard_paths=[shard])
    assert enriched[0]['unique_route_count']==3
    assert enriched[0]['route_count_is_exhaustive'] is True
    assert 'unique_route_count' not in rows[0]
    target=[{'id':'a','already_in_stock':False}]
    enriched[0]['work']={'sampled_tokens':20}
    from scripts.analyze_frozen_route_search import aggregate_cohort
    report=aggregate_cohort(target,{'tree':enriched})['conditions']['tree']
    assert report['unique_route_count']==3
    assert report['route_count_is_exhaustive'] is True


def test_diversity_from_another_or_modified_saved_graph_is_rejected(tmp_path):
    from scripts.analyze_frozen_route_search import attach_exact_diversity
    path,shard,rows,_=diversity_fixture(tmp_path)
    (shard.parent/'tree.json').write_text('{"different_graph":true}')
    with pytest.raises(ValueError,match='graph'):
        attach_exact_diversity(rows,path,mode='tree',assets_sha='assets',model_sha='actor',shard_paths=[shard])


def test_partial_or_contradictory_diversity_does_not_replace_online_lower_bound(tmp_path):
    import json
    from mechet.frozen_route_assets import file_sha256
    from scripts.analyze_frozen_route_search import attach_exact_diversity
    path,shard,rows,value=diversity_fixture(tmp_path)
    counts=path.parent/'counts.json'
    for complete,count in [(False,3),(True,0)]:
        counts.write_text(json.dumps({'complete':complete,'denominator':1,'rows':[
            {'id':'a','unique_route_count':count,'route_count_is_exhaustive':True,'already_in_stock':False}]}))
        value['counts_sha256']=file_sha256(counts);path.write_text(json.dumps(value))
        with pytest.raises(ValueError,match='diversity'):
            attach_exact_diversity(rows,path,mode='tree',assets_sha='assets',model_sha='actor',shard_paths=[shard])


def test_work_wave_uses_unique_forward_calls_and_preserves_failed_generation_cost():
    from scripts.infer_dynamic_electron_trees import record_work_wave
    from types import SimpleNamespace
    tree=SimpleNamespace(first_solution_work={'work':{'sampled_tokens':40}})
    costs={'solved':{'prompt_tokens':100,'policy_request_batches':2,'model_forward_batches_participated':15},
           'failed':{'prompt_tokens':60,'policy_request_batches':1,'model_forward_batches_participated':15}}
    seen=set()
    wave=record_work_wave(wave=1,elapsed=2.,tokens=80,forward_calls=15,runtime_work=costs,
                          trees={'solved':tree,'failed':SimpleNamespace(first_solution_work=None)},seen_solutions=seen)
    assert wave['model_forward_calls']==15  # NOT 30: batches shared by both requests
    assert wave['prompt_tokens']==160
    assert wave['output_tokens']==80  # includes failed proposals, not just accepted program
    assert wave['policy_request_batches']==3
    assert wave['new_stock_solved_ids']==['solved']
    assert seen=={'solved'}
    next_wave=record_work_wave(wave=2,elapsed=3.,tokens=90,forward_calls=20,runtime_work=costs,
                             trees={'solved':tree},seen_solutions=seen)
    assert next_wave['new_stock_solved_ids']==[]


def test_empty_work_wave_does_not_credit_finished_or_zero_step_targets():
    from scripts.infer_dynamic_electron_trees import record_work_wave
    wave=record_work_wave(wave=0,elapsed=0.,tokens=0,forward_calls=0,runtime_work={},trees={},seen_solutions=set())
    assert wave['new_stock_solved_ids']==[]
    assert wave['prompt_tokens']==0
    assert wave['model_forward_calls']==0


def test_budget_curve_uses_complete_observed_waves_not_future_solutions():
    from scripts.analyze_frozen_route_search import observed_work_curve
    rows=[{'id':'a','stock_solved':True,'already_in_stock':False},
          {'id':'b','stock_solved':False,'already_in_stock':False}]
    waves=[{'wave':0,'model_forward_calls':0,'output_tokens':0,'prompt_tokens':0,
            'new_stock_solved_ids':[],'credited_stock_solved_targets':0},
           {'wave':1,'model_forward_calls':20,'output_tokens':60,'prompt_tokens':100,
            'new_stock_solved_ids':['a'],'credited_stock_solved_targets':1},
           {'wave':2,'model_forward_calls':40,'output_tokens':120,'prompt_tokens':160,
            'new_stock_solved_ids':[],'credited_stock_solved_targets':1}]
    shard={'rows':rows,'tokens':120,'model_forward_calls':40}
    curve=observed_work_curve([(shard,waves)],per_worker_call_grid=[0,10,20,40,100])
    assert [point['solved_targets'] for point in curve['points']]==[0,0,1,1,1]
    assert [point['actual_model_forward_calls'] for point in curve['points']]==[0,0,20,40,40]
    assert curve['points'][2]['output_tokens']==60
    assert curve['points'][2]['prompt_tokens']==100
    assert curve['denominator']==2
    assert curve['equal_actual_compute_claim'] is False
    assert curve['points'][-1]['evaluated_beyond_observed_work'] is False


def test_work_curve_rejects_phantom_solution_and_missing_failure_denominator():
    from scripts.analyze_frozen_route_search import observed_work_curve
    shard={'rows':[{'id':'a','stock_solved':False,'already_in_stock':False}],
           'tokens':10,'model_forward_calls':5}
    waves=[{'wave':0,'model_forward_calls':0,'output_tokens':0,'prompt_tokens':0,
            'new_stock_solved_ids':[],'credited_stock_solved_targets':0},
           {'wave':1,'model_forward_calls':5,'output_tokens':10,'prompt_tokens':5,
            'new_stock_solved_ids':['unknown'],'credited_stock_solved_targets':1}]
    with pytest.raises(ValueError,match='work wave'):
        observed_work_curve([(shard,waves)],per_worker_call_grid=[0,5])
    waves[1]['new_stock_solved_ids']=[];waves[1]['credited_stock_solved_targets']=0
    shard['tokens']=999
    with pytest.raises(ValueError,match='work wave'):
        observed_work_curve([(shard,waves)],per_worker_call_grid=[0,5])


def test_generated_tokens_with_missing_forward_telemetry_are_not_zero_cost_work():
    from scripts.analyze_frozen_route_search import observed_work_curve
    shard={'rows':[{'id':'a','stock_solved':False,'already_in_stock':False}],
           'tokens':10,'model_forward_calls':0}
    waves=[{'wave':0,'model_forward_calls':0,'output_tokens':0,'prompt_tokens':0,
            'new_stock_solved_ids':[],'credited_stock_solved_targets':0},
           {'wave':1,'model_forward_calls':0,'output_tokens':10,'prompt_tokens':5,
            'new_stock_solved_ids':[],'credited_stock_solved_targets':0}]
    with pytest.raises(ValueError,match='work wave'):
        observed_work_curve([(shard,waves)],per_worker_call_grid=[0,5])


def test_reference_complexity_bins_keep_missing_separate_from_observed_zero_and_use_primary_route():
    from scripts.analyze_frozen_route_search import aggregate_cohort
    targets=[{'id':'zero','already_in_stock':False,'reference_routes':[
                    {'reference_route_id':'primary','reaction_edge_count':2,'convergent_branch_count':0},
                    {'reference_route_id':'alternative','reaction_edge_count':10,'convergent_branch_count':4}]},
             {'id':'complex','already_in_stock':False,'reference_routes':[
                    {'reference_route_id':'primary','reaction_edge_count':6,'convergent_branch_count':2}]},
             {'id':'missing','already_in_stock':False}]
    def predictions(solved):
        return [{'id':t['id'],'stock_solved':t['id'] in solved,'already_in_stock':False,
                 'work':{'sampled_tokens':1}} for t in targets]
    result=aggregate_cohort(targets,{'greedy':predictions(set()),'tree':predictions({'complex'})})
    convergence=result['reference_complexity']['convergent_branch_count']
    assert convergence['bins']['0']['target_ids']==['zero']
    assert convergence['bins']['2+']['target_ids']==['complex']
    assert convergence['bins']['Missing']['target_ids']==['missing']
    assert convergence['bins']['2+']['paired_tree_minus_greedy']['delta']==1
    assert convergence['bins']['0']['conditions']['tree']['solved']==0
    edges=result['reference_complexity']['reaction_edge_count']
    assert edges['bins']['2–3']['target_ids']==['zero']
    assert edges['bins']['6+']['target_ids']==['complex']
    assert sum(group['n'] for group in edges['bins'].values())==3


def test_reference_complexity_assignment_does_not_change_with_model_outcomes():
    from scripts.analyze_frozen_route_search import aggregate_cohort
    targets=[{'id':'a','already_in_stock':False,'reference_routes':[
        {'reference_route_id':'primary','reaction_edge_count':4,'convergent_branch_count':1}]}]
    def analysis(hit):
        return aggregate_cohort(targets,{'tree':[{'id':'a','stock_solved':hit,'already_in_stock':False,
                                                'work':{'sampled_tokens':1}}]})['reference_complexity']
    before,after=analysis(False),analysis(True)
    for dimension in before:
        assert before[dimension]['definition']==after[dimension]['definition']
        assert {name:group['target_ids'] for name,group in before[dimension]['bins'].items()}=={
            name:group['target_ids'] for name,group in after[dimension]['bins'].items()}
    missing=before['reference_event_length']['bins']['Missing']
    assert missing['n']==1
    assert before['reference_event_length']['bins']['0–4']['conditions']['tree']['solved'] is None


def test_reference_labels_can_be_frozen_without_any_prediction_or_accuracy_field():
    from scripts.analyze_frozen_route_search import reference_complexity_assignments
    source=[{'id':'a','reference_routes':[{'reference_route_id':'primary',
                                         'reaction_edge_count':3,'convergent_branch_count':0}]}]
    labels=reference_complexity_assignments(source)
    assert labels['reaction_edge_count']['bins']['2–3']==['a']
    assert labels['reference_event_length']['bins']['Missing']==['a']
    assert all('conditions' not in value and 'solved' not in value for value in labels.values())


def test_reference_only_cli_needs_no_inference_files_and_publishes_no_model_scores(tmp_path,monkeypatch):
    import json,sys
    from mechet.frozen_route_assets import file_sha256
    from scripts.analyze_frozen_route_search import main
    targets=[{'id':str(i),'reference_routes':[{'reference_route_id':f'r{i}',
               'reaction_edge_count':2,'convergent_branch_count':0}]} for i in range(460)]
    source=tmp_path/'targets.jsonl';source.write_text(''.join(json.dumps(t)+'\n' for t in targets))
    assets={'stock_preparation_complete':True,'full_target_rows':460,'config_sha256':'unit',
        'cohorts':{'unit':{'rows':460,'targets_path':str(source),'targets_sha256':file_sha256(source),
                         'target_ids':[t['id'] for t in targets]}}}
    manifest=tmp_path/'assets.json';manifest.write_text(json.dumps(assets))
    output=tmp_path/'reference-only'
    monkeypatch.setattr(sys,'argv',['analyze','--assets',str(manifest),'--output',str(output),'--reference-only'])
    main()
    result=json.loads((output/'reference_complexity_bins.json').read_text())
    assert result['denominator']==460
    assert result['new_model_result'] is False
    assert len(result['cohorts']['unit']['reference_complexity']['reaction_edge_count']['bins']['2–3'])==460
    assert not (output/'r1_route_results.json').exists()
