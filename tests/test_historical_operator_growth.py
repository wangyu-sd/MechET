"""Frozen historical census rendering; fixtures are not model evaluations."""
import hashlib
import json
import pytest


def basis_fixture(tmp_path):
    source=tmp_path/'source.jsonl';source.write_text('unit fixture only\n')
    report={'artifact_type':'nmi_execution_operator_basis_audit_v1',
        'scope':'strict_executable_training_pool_only_no_final_heldout_test',
        'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'primitive_basis':'source_to_sink_execution_moves_v1',
        'composition_basis':'execution_composition_signature_ordered_moves_v1',
        'rows':257167,'unique_source_ids':257167,
        'unique_local_primitives':3,'unique_complete_move_compositions':7,
        'vocabulary_growth':[{'hash_sample_fraction':.1,'unique_primitives':2,'unique_compositions':4},
                             {'hash_sample_fraction':1.,'unique_primitives':3,'unique_compositions':7}]}
    path=tmp_path/'report.json';path.write_text(json.dumps(report))
    return path,hashlib.sha256(path.read_bytes()).hexdigest(),source


def test_historical_growth_exports_observed_points_and_self_contained_vectors(tmp_path):
    from scripts.plot_frozen_operator_growth import render_growth
    path,sha,_=basis_fixture(tmp_path)
    destination=tmp_path/'figure'
    provenance=render_growth(path,expected_sha=sha,destination=destination)
    assert provenance['observed_points'][0]['unique_compositions']==4
    assert provenance['observed_points'][-1]['unique_primitives']==3
    assert provenance['new_model_result'] is False
    assert (destination/'figure3b_historical_operator_growth.pdf').read_bytes().startswith(b'%PDF-')
    svg=(destination/'figure3b_historical_operator_growth.svg').read_text()
    assert '<svg' in svg and '<image' not in svg


def test_modified_report_or_source_cannot_generate_a_historical_figure(tmp_path):
    from scripts.plot_frozen_operator_growth import render_growth
    path,sha,source=basis_fixture(tmp_path)
    with pytest.raises(ValueError,match='SHA'):
        render_growth(path,expected_sha='0'*64,destination=tmp_path/'wrong-report')
    source.write_text('changed\n')
    with pytest.raises(ValueError,match='SHA'):
        render_growth(path,expected_sha=sha,destination=tmp_path/'wrong-source')


def test_nonmonotone_or_incomplete_growth_is_not_drawn_as_the_full_census(tmp_path):
    from scripts.plot_frozen_operator_growth import render_growth
    path,_,_=basis_fixture(tmp_path)
    report=json.loads(path.read_text());report['vocabulary_growth'][-1]['hash_sample_fraction']=.5
    path.write_text(json.dumps(report));sha=hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='growth'):
        render_growth(path,expected_sha=sha,destination=tmp_path/'incomplete')
