"""Extended panels use whole reactions, paired targets and measured work."""
import copy
import importlib
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest


def implementation():
    try:
        return importlib.import_module('mechet.analysis.literature_extended')
    except ModuleNotFoundError:
        pytest.fail('paired/complexity/work extended analysis missing')


def inputs():
    from test_nmi_literature_population_figures import fixture
    from mechet.analysis.literature_reaction_keys import make_reaction
    metrics, cohorts = fixture()
    queries = [make_reaction('CO', ['CCl', 'O']), make_reaction('CN', ['CCl', 'N']),
               make_reaction('C1CC1', ['CCC'])]
    for row, count in zip(queries, (3, None, 1)):
        row['changed_bond_count'] = count
    evidence = {q['reaction_key']: {'tier': 'C', 'supported': False,
                 'verification_status': 'unresolved', 'source_candidates': []} for q in queries}
    evidence[queries[0]['reaction_key']] = {'tier': 'A', 'supported': True,
        'verification_status': 'source_passage_checked', 'source_candidates': ['s']}
    return queries, evidence, metrics, cohorts


def test_complexity_retains_unknown_contexts_and_counts_unique_reactions_not_sources():
    q, e, m, c = inputs()
    data = implementation().prepare_extended(q, e, m, c)
    assert data['unique_reactions'] == 3
    assert data['complexity']['changed_bonds']['Unknown'] == {'reactions': 1, 'indexed_candidates': 0, 'source_supported': 0}
    assert data['complexity']['changed_bonds']['1-3'] == {'reactions': 2, 'indexed_candidates': 1, 'source_supported': 1}
    assert data['complexity']['ring_count_change']['increase']['reactions'] == 1
    assert data['complexity']['product_heavy_atoms']['1-15']['reactions'] == 3
    e[q[0]['reaction_key']]['source_candidates'] += ['s2', 's3']
    assert implementation().prepare_extended(q, e, m, c)['complexity'] == data['complexity']


def test_paired_lower_bound_does_not_hide_pending_or_confuse_common_observed_set():
    q, e, m, c = inputs()
    data = implementation().prepare_extended(q, e, m, c)
    stock = data['paired']['r1_greedy_dynamic']['StockSolved@B']
    assert (stock['both'], stock['method_only'], stock['baseline_only'], stock['neither']) == (0, 1, 1, 0)
    assert stock['method_pending_reports'] == 1 and stock['pending_is_failure'] is False
    assert stock['observed_both']['denominator'] == 1
    assert stock['observed_both']['baseline_only'] == 0
    assert stock['bootstrap_draws'] == 2000
    assert data['work'] is None and data['matched_compute_claim'] is False


@pytest.mark.parametrize('change', ['duplicate_query', 'missing_evidence', 'extra_evidence'])
def test_complexity_rejects_changed_reaction_pool(change):
    q, e, m, c = inputs()
    if change == 'duplicate_query': q.append(copy.deepcopy(q[0]))
    if change == 'missing_evidence': e.pop(q[0]['reaction_key'])
    if change == 'extra_evidence': e['foreign'] = {'tier': 'C'}
    with pytest.raises(ValueError, match='reaction pool'):
        implementation().prepare_extended(q, e, m, c)


def test_vector_panels_have_original_units_and_no_gpu_time_estimate():
    import yaml
    q, e, m, c = inputs()
    data = implementation().prepare_extended(q, e, m, c)
    style = yaml.safe_load((Path(__file__).parents[1]/'configs/figures/mechet_nmi_tree_style_v1.yaml').read_text())
    drawings = implementation().render_extended(data, style)
    assert set(drawings) == {'extended_data_paired_route', 'extended_data_source_retrieval_coverage'}
    for svg in drawings.values():
        root = ET.fromstring(svg)
        assert root.attrib['width'] == '183mm' and float(root.attrib['height'][:-2]) <= 170
        assert not list(root.iter('{http://www.w3.org/2000/svg}image'))
        text = ' '.join(root.itertext())
        assert 'unequal work' in text


def test_work_charges_generated_tokens_not_cache_returns_or_gpu_seconds():
    q, e, m, c = inputs()
    data = implementation().prepare_extended(q, e, m, c)
    work = {'original_target_denominator': 2, 'observed_reports': 1, 'pending_reports': 1,
        'nonzero_stock_connected_targets': 1, 'matched_compute_claim': False,
        'worker_elapsed_is_global_wall_time': False, 'actual_gpu_active_seconds': None,
        'fixed_observed_reports_only_cost_claim': False,
        'totals': {'actual_generated_output_tokens': 10000000, 'reused_output_tokens_not_charged': 3000000,
                   'unique_scheduled_model_forward_calls': 100, 'cpu_executor_thread_seconds': 4.0}}
    enriched = implementation().attach_work(data, work, supported_disconnections=1)
    assert enriched['work']['covered_targets_per_million_generated_tokens'] == .1
    assert enriched['work']['supported_disconnections_per_million_generated_tokens'] == .1
    assert enriched['work']['actual_generated_output_tokens'] == 10000000
    assert enriched['work']['cache_return_tokens_excluded'] == 3000000
    assert enriched['work']['actual_gpu_active_seconds'] is None
    assert data['work'] is None
    import yaml
    style = yaml.safe_load((Path(__file__).parents[1]/'configs/figures/mechet_nmi_tree_style_v1.yaml').read_text())
    svg = implementation().render_extended(enriched, style)['extended_data_work_attribution']
    root = ET.fromstring(svg)
    text = ' '.join(root.itertext())
    assert '10,000,000' in text and '3,000,000' in text and 'Unknown' in text
    bad = copy.deepcopy(work); bad['original_target_denominator'] = 1
    with pytest.raises(ValueError, match='work population'):
        implementation().attach_work(data, bad, supported_disconnections=1)
