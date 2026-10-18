"""Same-target chemical discovery comparisons, never endpoint ranking."""
import importlib

import pytest


def implementation():
    try:
        return importlib.import_module('mechet.analysis.literature_comparison')
    except ModuleNotFoundError:
        pytest.fail('literature-supported disconnection comparison missing')


def support(tier='A'):
    return {'tier':tier,'supported':True,'verification_status':'source_passage_checked'}


def test_disconnection_comparison_is_paired_by_target_and_deduplicated():
    method=[{'target_id':'a','reaction_key':'x'}]*3+[
        {'target_id':'b','reaction_key':'y'}]
    baseline=[{'target_id':'a','reaction_key':'x'},
              {'target_id':'b','reaction_key':'z'},
              {'target_id':'a','reaction_key':'y'}]
    result=implementation().compare_disconnections(method,baseline,
        {'x':support(),'y':support('B'),'z':{'tier':'C','supported':False}},universe=['a','b','pending'])
    assert result['denominator']==3
    assert result['unique_supported_reactions']=={'both':2,'method_only':0,'baseline_only':0}
    assert result['target_reaction_pairs']=={'both':1,'method_only':1,'baseline_only':1}
    assert result['targets']['b']['method_only']==['y']
    assert result['targets']['a']['baseline_only']==['y']
    assert result['targets']['pending']['both']==[]


def test_single_step_target_source_outcomes_preserve_missingness_and_all_candidates():
    occurrences=[{'target_id':'a','reaction_key':'x'}]*4+[
        {'target_id':'a','reaction_key':'y'}, {'target_id':'b','reaction_key':'z'}]
    result=implementation().reaction_target_outcomes(occurrences,
        {'x':support('B'),'y':support(),'z':{'tier':'C','supported':False,
         'verification_status':'machine_structural_match'}},universe=['a','b','pending'])
    assert result['denominator']==3
    assert result['any_supported_targets']==['a']
    assert result['exact_supported_targets']==['a']
    assert result['targets']['a']['distinct_supported_disconnections']==2
    assert result['targets']['a']['distinct_generated_reactions']==2
    assert result['targets']['b']['source_checked_fraction']==0
    assert result['targets']['pending']['source_checked_fraction'] is None
    assert result['source_checked_unique_reactions']==2
    assert result['unique_generated_reactions']==3


def test_efficiency_exposes_cache_and_missing_costs_without_fictitious_gpu_time():
    work={'per_target':[{'values':{'sampled_tokens':1000,'reused_output_tokens':400,
            'executor_calls':10,'wall_seconds':2}},
                        {'values':{'sampled_tokens':500,'reused_output_tokens':100,
            'executor_calls':5,'wall_seconds':3}}]}
    result=implementation().discovery_efficiency(work,supported_disconnections=2,covered_targets=1)
    assert result['reported_work_sums']['sampled_tokens']==1500
    assert result['reported_work_sums']['reused_output_tokens']==500
    assert result['report_accounted_nonreused_output_tokens']==1000
    assert result['supported_disconnections_per_million_report_accounted_nonreused_tokens']==2000
    assert result['supported_disconnections_per_executor_call']==pytest.approx(2/15)
    assert result['actual_cluster_gpu_seconds'] is None
    assert result['summed_target_elapsed_seconds']==5
    assert result['elapsed_is_global_wall_time'] is False
    work['per_target'][1]['values']['reused_output_tokens']=None
    result=implementation().discovery_efficiency(work,supported_disconnections=2,covered_targets=1)
    assert result['report_accounted_nonreused_output_tokens'] is None
    assert result['supported_disconnections_per_million_report_accounted_nonreused_tokens'] is None
    assert result['missing_cost_target_counts']['reused_output_tokens']==1


def test_single_reaction_work_does_not_invent_a_zero_whole_route_rate():
    result=implementation().discovery_efficiency({'per_target':[{'values':{
        'sampled_tokens':100,'reused_output_tokens':0,'executor_calls':2}}]},
        supported_disconnections=1,covered_targets=None)
    assert result['precedent_covered_targets_per_million_report_accounted_nonreused_tokens'] is None


def test_comparison_rejects_outside_original_universe():
    with pytest.raises(ValueError,match='universe'):
        implementation().compare_disconnections([{'target_id':'extra','reaction_key':'r'}],[],{},universe=['a'])
